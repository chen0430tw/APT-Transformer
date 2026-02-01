
"""
Virtual Blackwell Adapter v6.2
- Fixes pulse scheduling semantics (round-robin per layer, global step)
- Adds cheap, cache-friendly scale update (approx quantile on small, fixed sample)
- Avoids randperm() per call (precomputed indices)
- Keeps everything on GPU (no .cpu() or Python loops over elements)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class VBLayerStats:
    total_calls: int = 0          # forward calls of the VB wrapped layer
    pulse_calls: int = 0          # how many times we ran scale update + fake-int8
    fast_calls: int = 0           # how many times we ran pure fp path
    scale_updates: int = 0        # how many times scale was recomputed
    scale_reuses: int = 0         # how many times cached scale reused


class VirtualBlackwellAdapterV62:
    """
    Per-layer cached quant scale:
      - scale = quantile(|W|, q) estimated on a fixed-size sample of elements
      - scale updated only when cheap drift test triggers
    """

    def __init__(
        self,
        q: float = 0.999,
        sample_size: int = 8192,
        drift_threshold: float = 0.20,  # ±20%
        eps: float = 1e-8,
        enable_fp4_coarse: bool = True,
        int8_clip: int = 127,
    ):
        self.q = float(q)
        self.sample_size = int(sample_size)
        self.drift_threshold = float(drift_threshold)
        self.eps = float(eps)
        self.enable_fp4_coarse = bool(enable_fp4_coarse)
        self.int8_clip = int(int8_clip)

        # caches keyed by layer_name
        self._scale_cache: Dict[str, torch.Tensor] = {}
        self._maxabs_cache: Dict[str, torch.Tensor] = {}
        self._sample_idx_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}
        self.stats: Dict[str, VBLayerStats] = {}

    def _get_stats(self, name: str) -> VBLayerStats:
        st = self.stats.get(name)
        if st is None:
            st = VBLayerStats()
            self.stats[name] = st
        return st

    @torch.no_grad()
    def _get_sample_indices(self, numel: int, device: torch.device) -> torch.Tensor:
        """
        Fixed indices per (numel, device). Use deterministic strided sampling to avoid randperm O(N).
        """
        key = (numel, device)
        if key in self._sample_idx_cache:
            return self._sample_idx_cache[key]
        k = min(self.sample_size, numel)
        if k <= 0:
            idx = torch.zeros((0,), device=device, dtype=torch.long)
        else:
            # stride sampling: i * (numel/k) modulo numel
            stride = max(1, numel // k)
            idx = (torch.arange(k, device=device, dtype=torch.long) * stride) % numel
        self._sample_idx_cache[key] = idx
        return idx

    @torch.no_grad()
    def _approx_quantile_abs(self, w: torch.Tensor) -> torch.Tensor:
        """
        Approximate quantile(|w|, q) from a fixed sample, using kthvalue (O(k)).
        Returns scalar tensor on same device.
        """
        flat = w.reshape(-1)
        idx = self._get_sample_indices(flat.numel(), flat.device)
        if idx.numel() == 0:
            return torch.tensor(1.0, device=flat.device, dtype=flat.dtype)
        sample = flat.index_select(0, idx).abs()
        # kth index for quantile: ceil(q*k)-1
        k = sample.numel()
        kth = int(max(1, min(k, int(self.q * k + 0.9999))))  # 1..k
        # kthvalue returns the kth smallest
        val = torch.kthvalue(sample, kth).values
        return torch.clamp(val, min=self.eps)

    @torch.no_grad()
    def _drifted(self, name: str, w: torch.Tensor) -> bool:
        """
        Cheap drift test: compare max(|w|) to cached max(|w|).
        """
        maxabs = w.abs().amax()
        old = self._maxabs_cache.get(name)
        if old is None:
            self._maxabs_cache[name] = maxabs
            return True
        ratio = maxabs / (old + self.eps)
        # drift if outside [1-thr, 1+thr]
        thr = self.drift_threshold
        drift = (ratio > (1.0 + thr)) | (ratio < (1.0 - thr))
        if drift.item():
            self._maxabs_cache[name] = maxabs
        return bool(drift.item())

    @torch.no_grad()
    def get_scale(self, name: str, w: torch.Tensor) -> torch.Tensor:
        """
        Return cached scale; refresh only if drifted.
        """
        st = self._get_stats(name)
        if name in self._scale_cache and (not self._drifted(name, w)):
            st.scale_reuses += 1
            return self._scale_cache[name]

        scale = self._approx_quantile_abs(w)
        self._scale_cache[name] = scale
        st.scale_updates += 1
        return scale

    def linear_fast(self, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor], name: str) -> torch.Tensor:
        st = self._get_stats(name)
        st.total_calls += 1
        st.fast_calls += 1
        return F.linear(x, w, b)

    @torch.no_grad()
    def _fake_int8(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # symmetric fake quant
        s = scale.to(dtype=x.dtype)
        q = torch.round(x / s).clamp(-self.int8_clip, self.int8_clip)
        return q * s

    def linear_pulse(self, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor], name: str) -> torch.Tensor:
        st = self._get_stats(name)
        st.total_calls += 1
        st.pulse_calls += 1

        scale = self.get_scale(name, w)
        # Optional FP4 coarse stage (very cheap): quantize x to a coarse grid before int8
        if self.enable_fp4_coarse:
            # 16-level symmetric grid: [-8..7] * (scale/8)
            s4 = (scale / 8.0).to(dtype=x.dtype)
            xq4 = torch.round(x / s4).clamp(-8, 7) * s4
            xq = self._fake_int8(xq4, scale)
        else:
            xq = self._fake_int8(x, scale)

        wq = self._fake_int8(w, scale)
        return F.linear(xq, wq, b)
