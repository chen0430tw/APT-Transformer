# vb_sparse_attention.py
# -*- coding: utf-8 -*-
"""Local-window ("sparse") attention patcher for claude4_model.py-style MHA.

This is a **single-file** implementation intended to be imported by your speed test:

    from vb_sparse_attention import apply_sparse_attention, LocalAttnConfig

What it does
- Detects attention modules that look like Claude4 test model MHA:
    - w_qkv: Linear(d_model, 3*d_model)
    - w_o:   Linear(d_model, d_model)
    - attributes: n_heads, head_dim, dropout
- Monkey-patches their forward() to apply a **local-window** additive mask.

Performance note (important)
- This implementation uses a *dense* [S,S] additive mask, which often disables
  FlashAttention / fast SDPA kernels. So it is mainly for **wiring validation**
  ("patched modules > 0" and correctness), not guaranteed speedups.
- Real speedups for long sequences require true block-sparse kernels.

API
- LocalAttnConfig matches what your test script expects: window/causal/dropout_p/use_sdpa.
  (dropout_p/use_sdpa are accepted but not used here, kept for compatibility.)
- apply_sparse_attention(model, n_heads=..., cfg=...) returns patched module count.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Public config (matches your test script)
# ---------------------------------------------------------------------------

@dataclass
class LocalAttnConfig:
    """Config expected by test_vb_training_speed_v6_4.py."""

    window: int = 32
    causal: bool = False
    dropout_p: float = 0.0  # accepted but unused (module.dropout is used)
    use_sdpa: bool = True   # accepted but unused
    verbose: bool = False


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

@dataclass
class PatchReport:
    patched_modules: int = 0
    skipped_modules: int = 0
    reasons: Optional[Dict[str, int]] = None

    def __post_init__(self) -> None:
        if self.reasons is None:
            self.reasons = {}


# Cache additive masks by (S, window, device_type, device_index, dtype, causal)
_MASK_CACHE: Dict[Tuple[int, int, str, int, torch.dtype, bool], torch.Tensor] = {}


def _is_claude4_mha(mod: nn.Module) -> bool:
    """Duck-typing detection for Claude4-style MHA."""
    if not hasattr(mod, "w_qkv") or not hasattr(mod, "w_o"):
        return False
    wqkv = getattr(mod, "w_qkv", None)
    wo = getattr(mod, "w_o", None)
    if not isinstance(wqkv, nn.Linear) or not isinstance(wo, nn.Linear):
        return False

    # Required attributes
    if not all(hasattr(mod, k) for k in ("n_heads", "head_dim", "dropout")):
        return False

    # Shape sanity
    if wqkv.out_features != 3 * wqkv.in_features:
        return False
    if wo.in_features != wo.out_features:
        return False

    # head_dim consistency (best-effort)
    n_heads = int(getattr(mod, "n_heads"))
    head_dim = int(getattr(mod, "head_dim"))
    if n_heads * head_dim != wqkv.in_features:
        # Not necessarily fatal, but usually indicates a different module
        return False

    return True


def _device_key(device: torch.device) -> Tuple[str, int]:
    if device.type == "cuda":
        return ("cuda", int(device.index or 0))
    return (device.type, -1)


def _get_local_mask(
    S: int,
    window: int,
    device: torch.device,
    dtype: torch.dtype,
    is_causal: bool,
) -> torch.Tensor:
    """Build additive attention mask [S,S]: 0 inside window, -inf outside."""
    dev_type, dev_idx = _device_key(device)
    key = (int(S), int(window), dev_type, dev_idx, dtype, bool(is_causal))
    cached = _MASK_CACHE.get(key)
    if cached is not None:
        return cached

    # Build on CPU (float32) then move; avoids GPU graph pollution.
    idx = torch.arange(S)
    i = idx[:, None]
    j = idx[None, :]

    if is_causal:
        allow = (j <= i) & ((i - j) <= window)
    else:
        allow = (i - j).abs() <= window

    mask = torch.zeros((S, S), dtype=torch.float32)
    mask[~allow] = float("-inf")
    mask = mask.to(device=device, dtype=dtype)

    _MASK_CACHE[key] = mask
    return mask


def patch_sparse_attention(
    model: nn.Module,
    local_window: int = 32,
    causal: bool = False,
    verbose: bool = False,
) -> int:
    """Monkey-patch Claude4-style MHA modules to use local-window attention.

    Returns number of patched modules.
    """
    report = PatchReport()

    for name, mod in model.named_modules():
        if not _is_claude4_mha(mod):
            continue

        # Skip if already patched
        if getattr(mod, "_vb_sparse_patched", False):
            report.skipped_modules += 1
            report.reasons["already_patched"] = report.reasons.get("already_patched", 0) + 1
            continue

        # Capture module wiring
        w_qkv: nn.Linear = mod.w_qkv
        w_o: nn.Linear = mod.w_o
        n_heads: int = int(mod.n_heads)
        head_dim: int = int(mod.head_dim)
        dropout: float = float(mod.dropout)

        # Some variants may not expose d_model; fall back to Linear.in_features.
        d_model = getattr(mod, "d_model", None)
        if d_model is None:
            d_model = int(getattr(w_qkv, "in_features", 0) or 0)

        def _forward_local(
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
            is_causal: bool = False,
            _mod: nn.Module = mod,
        ) -> torch.Tensor:
            # x: [B,S,D]
            B, S, D = x.shape
            if D != d_model:
                # Defensive: avoid silent wrong shapes
                raise RuntimeError(f"[vb_sparse_attention] unexpected d_model: got {D}, expected {d_model}")

            qkv = w_qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)

            q = q.view(B, S, n_heads, head_dim).transpose(1, 2)
            k = k.view(B, S, n_heads, head_dim).transpose(1, 2)
            v = v.view(B, S, n_heads, head_dim).transpose(1, 2)

            # Local additive mask
            local_mask = _get_local_mask(S, local_window, x.device, q.dtype, causal or is_causal)

            # Combine with upstream mask if present
            combined = local_mask if attn_mask is None else (attn_mask + local_mask)

            if hasattr(F, "scaled_dot_product_attention"):
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=combined,
                    dropout_p=dropout if _mod.training else 0.0,
                    is_causal=False,  # causality already encoded by mask
                )
            else:
                scale = (head_dim ** -0.5)
                attn = (q * scale) @ k.transpose(-2, -1)  # [B,H,S,S]
                attn = attn + combined
                attn = F.softmax(attn, dim=-1)
                if _mod.training and dropout > 0:
                    attn = F.dropout(attn, p=dropout)
                out = attn @ v

            out = out.transpose(1, 2).contiguous().view(B, S, D)
            return w_o(out)

        # Patch
        mod.forward = _forward_local  # type: ignore[assignment]
        setattr(mod, "_vb_sparse_patched", True)
        setattr(mod, "_vb_sparse_window", int(local_window))
        setattr(mod, "_vb_sparse_causal", bool(causal))

        report.patched_modules += 1
        if verbose:
            print(f"[SparsePatch] patched {name} (window={local_window}, causal={causal})")

    if verbose:
        print(
            f"[SparsePatch] patched_modules={report.patched_modules}, "
            f"skipped={report.skipped_modules}, reasons={report.reasons}"
        )

    return report.patched_modules


# ---------------------------------------------------------------------------
# Backward-compatible entrypoint expected by your test script
# ---------------------------------------------------------------------------

def apply_sparse_attention(
    model: nn.Module,
    n_heads: int = 0,
    cfg: Optional[LocalAttnConfig] = None,
    **_: Any,
) -> int:
    """Entry point used by test_vb_training_speed_v6_4.py.

    `n_heads` is ignored (we read it from each module), kept only for signature compat.
    """
    if cfg is None:
        cfg = LocalAttnConfig()

    return patch_sparse_attention(
        model,
        local_window=int(cfg.window),
        causal=bool(cfg.causal),
        verbose=bool(cfg.verbose),
    )
