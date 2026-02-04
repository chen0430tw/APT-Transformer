# vb_sparse_attention.py
# -*- coding: utf-8 -*-
"""Compatibility shim for sparse/local attention patching.

Your speed test script (`test_vb_training_speed_v6_4.py`) imports:

    from vb_sparse_attention import apply_sparse_attention, LocalAttnConfig

…but the actual implementation is versioned (`vb_sparse_attention_v5.py`).

This file keeps the *test-script-facing* API stable while delegating to v5.

Notes:
- v5 implements local attention via a dense [S,S] additive mask. This is mainly
  for wiring/correctness validation; it may disable FlashAttention and be slower.
- Extra config fields (dropout_p/use_sdpa) are accepted for forward-compat but
  are not used by v5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any

import torch.nn as nn

from vb_sparse_attention_v5 import patch_sparse_attention


@dataclass
class LocalAttnConfig:
    """Test-script-facing config.

    Fields match what `test_vb_training_speed_v6_4.py` expects.
    """

    window: int = 32
    causal: bool = False
    dropout_p: float = 0.0  # accepted but unused by v5
    use_sdpa: bool = True   # accepted but unused by v5
    verbose: bool = False


def apply_sparse_attention(
    model: nn.Module,
    n_heads: int = 0,
    cfg: Optional[LocalAttnConfig] = None,
    **_: Any,
) -> int:
    """Apply a local-window attention patch.

    Args:
        model: PyTorch model.
        n_heads: Kept for backward-compat; ignored (heads are read from modules).
        cfg: Local attention configuration.

    Returns:
        Number of patched modules.
    """
    if cfg is None:
        cfg = LocalAttnConfig()

    # Delegate to v5 patcher.
    return patch_sparse_attention(
        model,
        local_window=int(cfg.window),
        causal=bool(cfg.causal),
        verbose=bool(cfg.verbose),
    )
