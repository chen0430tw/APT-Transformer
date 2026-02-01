"""
vb_sparse_attention.py
----------------------
A pragmatic "local (sliding-window) causal attention" wrapper intended for speed testing.

Why:
- Your current VB timing is dominated by attention+FFN math, not by scale/quant bookkeeping.
- A *real* sparse kernel (xFormers/FlashAttention-2 block-sparse) would be ideal, but keeping deps minimal,
  we implement a local-window attention that reduces attention complexity from O(L^2) -> O(L*W).

Notes:
- This wrapper assumes an attention module that exposes W_q/W_k/W_v/W_o (nn.Linear-like).
- It monkey-patches `attn.forward(x, attn_mask=None, ...)` style modules to a local-attn forward.
- If your attention implementation differs, adapt the "probe" in `looks_like_qkv_attention`.

This is meant as an optional accelerator to combine with VB pulse mode.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Callable, Any, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LocalAttnConfig:
    window: int = 64           # sliding window size
    causal: bool = True        # causal masking
    dropout_p: float = 0.0     # keep 0 for deterministic speed tests
    use_sdpa: bool = True      # use torch SDPA when possible


def _reshape_heads(t: torch.Tensor, n_heads: int) -> torch.Tensor:
    # t: [B, L, D] -> [B, H, L, Dh]
    b, l, d = t.shape
    assert d % n_heads == 0, f"hidden {d} not divisible by heads {n_heads}"
    dh = d // n_heads
    return t.view(b, l, n_heads, dh).transpose(1, 2).contiguous()


def _merge_heads(t: torch.Tensor) -> torch.Tensor:
    # t: [B, H, L, Dh] -> [B, L, D]
    b, h, l, dh = t.shape
    return t.transpose(1, 2).contiguous().view(b, l, h * dh)


def local_causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    window: int, causal: bool, dropout_p: float = 0.0,
) -> torch.Tensor:
    """
    Compute local sliding-window attention.

    Inputs:
      q,k,v: [B, H, L, Dh]
    Output:
      out:   [B, H, L, Dh]
    """
    b, h, l, dh = q.shape
    w = max(1, min(window, l))

    # Process in blocks to avoid materializing full LxL.
    # For each position i, attend to keys in [i-w+1, i] (causal) or centered window (non-causal).
    out = torch.empty_like(q)

    scale = 1.0 / math.sqrt(dh)

    # We do a simple loop over sequence blocks. For L=128 this is fine; for larger L you can increase block size.
    block = 128  # tune if you like
    for start in range(0, l, block):
        end = min(l, start + block)
        # query chunk
        q_chunk = q[:, :, start:end, :]  # [B,H,T,Dh]

        if causal:
            k_start = max(0, end - w)  # ensures every query in chunk sees at most last w keys
            k_end = end
        else:
            # centered window: keys from [start-w//2, end+w//2]
            half = w // 2
            k_start = max(0, start - half)
            k_end = min(l, end + half)

        k_chunk = k[:, :, k_start:k_end, :]
        v_chunk = v[:, :, k_start:k_end, :]

        # attn scores: [B,H,T,S]
        scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * scale

        if causal:
            # Need to mask out keys that are "future" relative to each query index.
            # Build a small mask for this chunk only.
            # Query indices: [start, end)
            qi = torch.arange(start, end, device=q.device).view(1, 1, -1, 1)
            ki = torch.arange(k_start, k_end, device=q.device).view(1, 1, 1, -1)
            scores = scores.masked_fill(ki > qi, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        if dropout_p and dropout_p > 0:
            attn = F.dropout(attn, p=dropout_p, training=True)

        out[:, :, start:end, :] = torch.matmul(attn, v_chunk)

    return out


def looks_like_qkv_attention(m: nn.Module) -> bool:
    return all(hasattr(m, name) for name in ("W_q", "W_k", "W_v", "W_o"))


def patch_attention_module(m: nn.Module, *, n_heads: int, cfg: LocalAttnConfig) -> None:
    """
    Monkey patch an attention module `m` that has W_q/W_k/W_v/W_o.
    """

    W_q: nn.Module = getattr(m, "W_q")
    W_k: nn.Module = getattr(m, "W_k")
    W_v: nn.Module = getattr(m, "W_v")
    W_o: nn.Module = getattr(m, "W_o")

    def _forward_local(x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        # x: [B, L, D]
        q = W_q(x)
        k = W_k(x)
        v = W_v(x)

        qh = _reshape_heads(q, n_heads)
        kh = _reshape_heads(k, n_heads)
        vh = _reshape_heads(v, n_heads)

        out = local_causal_attention(qh, kh, vh, window=cfg.window, causal=cfg.causal, dropout_p=cfg.dropout_p)
        out = _merge_heads(out)
        out = W_o(out)
        return out

    # Attach for debug
    setattr(m, "_vb_sparse_attn_patched", True)
    setattr(m, "_vb_sparse_attn_cfg", cfg)

    # Replace forward
    m.forward = _forward_local  # type: ignore[attr-defined]


def apply_sparse_attention(model: nn.Module, *, n_heads: int, cfg: Optional[LocalAttnConfig] = None) -> int:
    """
    Walk model and patch any modules that "look like" a QKV attention block.

    Returns number of modules patched.
    """
    cfg = cfg or LocalAttnConfig()
    patched = 0
    for name, mod in model.named_modules():
        if looks_like_qkv_attention(mod) and not getattr(mod, "_vb_sparse_attn_patched", False):
            patch_attention_module(mod, n_heads=n_heads, cfg=cfg)
            patched += 1
    return patched
