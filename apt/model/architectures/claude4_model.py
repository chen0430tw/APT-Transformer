"""
Claude4Model (refactored reflection)
-----------------------------------
This version replaces the previous graph-theoretic "reflection" stack with a
GPU-friendly "true reflection" module built from:

- Flash Attention via torch.nn.functional.scaled_dot_product_attention (SDPA)
  (PyTorch will choose FlashAttention / Mem-Efficient kernels when available)
- RMSNorm (fallback to LayerNorm if RMSNorm is unavailable)
- SwiGLU FFN

Why:
- The previous reflection stack introduced many small Linear ops + Python-side
  orchestration, which is slow and also defeats VB cache assumptions
  (lots of layers, lots of tiny "updates", low cache hit rate).
- SDPA + fused FFN is closer to what modern LLMs do for "reasoning/reflection"
  while being dramatically friendlier to CUDA scheduling.

This file keeps the public API compatible with the original claude4_model.py:
- class Claude4Model
- forward() returns logits and (optionally) reflection_info
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

from apt.core.fake_torch import get_torch

torch = get_torch()
nn = torch.nn
F = torch.nn.functional


# ------------------------------------------------------------------------------
# Small utilities
# ------------------------------------------------------------------------------

def _has_rmsnorm() -> bool:
    return hasattr(nn, "RMSNorm")


class RMSNorm(nn.Module):
    """RMSNorm wrapper with LayerNorm fallback."""
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        if _has_rmsnorm():
            self.norm = nn.RMSNorm(d, eps=eps)
        else:
            # Fallback: LayerNorm is not identical, but keeps shapes/behavior sane.
            self.norm = nn.LayerNorm(d, eps=eps)

    def forward(self, x):
        return self.norm(x)


class SwiGLU(nn.Module):
    """SwiGLU FFN: (x W1) * silu(x Wg) -> W2"""
    def __init__(self, d_model: int, d_ff: int, bias: bool = True):
        super().__init__()
        # project to 2*d_ff then split
        self.w12 = nn.Linear(d_model, 2 * d_ff, bias=bias)
        self.w3 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        u, g = self.w12(x).chunk(2, dim=-1)
        return self.w3(u * F.silu(g))


# ------------------------------------------------------------------------------
# Core blocks
# ------------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Standard MHA with SDPA (FlashAttention when available)."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model({d_model}) must be divisible by n_heads({n_heads})"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, attn_mask=None, is_causal: bool = False):
        # x: [B, S, D]
        B, S, D = x.shape
        qkv = self.w_qkv(x)  # [B, S, 3D]
        q, k, v = qkv.chunk(3, dim=-1)

        # [B, H, S, Hd]
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # SDPA expects [B, H, S, Hd]
        # If torch<2.0 or SDPA missing, fall back to manual attention.
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            # Manual attention: (qk^T)/sqrt(d) -> softmax -> v
            scale = (self.head_dim ** -0.5)
            attn = (q * scale) @ k.transpose(-2, -1)  # [B,H,S,S]
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    # bool mask: True = masked position
                    attn = attn.masked_fill(attn_mask, float("-inf"))
                else:
                    # float additive mask (e.g. -inf for masked positions)
                    attn = attn + attn_mask
            if is_causal:
                causal = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
                attn = attn.masked_fill(causal, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            if self.training and self.dropout > 0:
                attn = F.dropout(attn, p=self.dropout)
            out = attn @ v  # [B,H,S,Hd]

        out = out.transpose(1, 2).contiguous().view(B, S, D)  # [B,S,D]
        return self.w_o(out)


class ReflectionBlock(nn.Module):
    """
    "True reflection" module:
      RMSNorm -> FlashAttention (causal) -> residual
      RMSNorm -> SwiGLU -> residual
    Additionally outputs a lightweight "reflection score" for debugging
    (only computed when return_reflection=True to avoid GPU sync).
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.score_head = nn.Linear(d_model, 1, bias=True)

    def forward(self, x, return_reflection: bool = False) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        h = self.attn(self.norm1(x), is_causal=True)
        x = x + h
        h2 = self.ffn(self.norm2(x))
        x = x + h2

        info = None
        if return_reflection:
            # score: [B, S, 1] -> [B, 1] pooled
            score = torch.sigmoid(self.score_head(x)).mean(dim=1)  # [B,1]
            info = {
                "reflection_score_mean": float(score.mean().item()),
            }
        return x, info

class FeedForward(nn.Module):
    """Plain FFN (kept for baseline transformer path)."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = SwiGLU(d_model, d_ff)

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, enable_reflection: bool, dropout: float = 0.0):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout=dropout)

        self.ffn_norm = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

        self.enable_reflection = enable_reflection
        self.reflection = ReflectionBlock(d_model, n_heads, d_ff, dropout=dropout) if enable_reflection else None

    def forward(self, x, return_reflection: bool = False) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        # Attention
        x = x + self.attn(self.attn_norm(x), is_causal=True)
        # FFN
        x = x + self.ffn(self.ffn_norm(x))

        refl_info = None
        if self.reflection is not None:
            x, refl_info = self.reflection(x, return_reflection=return_reflection)

        return x, refl_info


# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------

class Claude4Model(nn.Module):
    """
    A compact GPT-style model with an optional "reflection" stage per layer.
    """
    def __init__(
        self,
        vocab_size: int = 50000,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        num_layers: int = 6,
        max_seq_len: int = 4096,
        enable_reflection: bool = True,
        reflection_layers: Optional[List[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        if reflection_layers is None:
            reflection_layers = list(range(num_layers)) if enable_reflection else []

        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                enable_reflection=(i in reflection_layers),
                dropout=dropout,
            )
            for i in range(num_layers)
        ])

        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids, return_reflection: bool = False):
        # input_ids: [B, S]
        B, S = input_ids.shape
        pos = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, S)

        x = self.tok_emb(input_ids) + self.pos_emb(pos)

        refl_all: List[Dict[str, Any]] = []
        for blk in self.blocks:
            x, info = blk(x, return_reflection=return_reflection)
            if return_reflection and info is not None:
                refl_all.append(info)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        if return_reflection:
            return logits, {"layers": refl_all}
        return logits


# ------------------------------------------------------------------------------
# Factory function for compatibility
# ------------------------------------------------------------------------------

def create_claude4_model(
    vocab_size: int = 50000,
    d_model: int = 256,
    num_layers: int = 6,
    num_heads: int = 8,
    ffn_hidden: int = 1024,
    max_seq_len: int = 4096,
    enable_reflection: bool = False,
    **kwargs
):
    """
    Factory function for backward compatibility.
    Maps old parameter names (num_heads, ffn_hidden) to new ones (n_heads, d_ff).
    """
    return Claude4Model(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        n_heads=num_heads,
        d_ff=ffn_hidden,
        max_seq_len=max_seq_len,
        enable_reflection=enable_reflection,
        **kwargs
    )