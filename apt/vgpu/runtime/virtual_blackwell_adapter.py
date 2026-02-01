
"""
Virtual Blackwell Adapter v6.4
==============================

Goal of v6.4
------------
Fix the two practical issues seen in v6.2 results:

1) scale cache reuse stays at 0%
   - v6.2 effectively "updated" on every pulse because the reuse decision
     was tied to unstable signals (or was not wired tightly per-layer).

2) training-speed tests should not regress
   - Fake INT8 in PyTorch (quantize->dequantize->fp GEMM) is *slower*.
     v6.4 keeps fake INT8 optional and defaults it OFF for speed tests.
     You can still enable it to validate numerical behavior / statistics.

Key idea
--------
On each pulse we do:
  - cheap drift check (mean(|w|) on a fixed subsample)
  - only if drift > threshold OR no cached scale -> run quantile sampling
  - otherwise reuse cached scale

This produces real reuse once the weight distribution stabilizes.

This file is self-contained (no external deps beyond torch).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple
import math
import torch
import torch.nn.functional as F


@dataclass
class LayerQuantState:
    # cached weight scale (scalar tensor on same device as weights)
    w_scale: Optional[torch.Tensor] = None

    # drift stats (Python floats)
    metric_ema: Optional[float] = None          # EMA of quantile(|w|)
    cheap_ema: Optional[float] = None           # EMA of mean(|w|) on subsample
    last_q: Optional[float] = None              # last quantile(|w|)

    # fixed subsample indices for cheap drift check
    sample_idx: Optional[torch.Tensor] = None

    # counters
    total_calls: int = 0
    fast_calls: int = 0
    pulse_calls: int = 0
    scale_updates: int = 0
    scale_reuses: int = 0


def _rand_sample_abs(x: torch.Tensor, max_samples: int, *, seed: int) -> torch.Tensor:
    """Return 1D abs-sample tensor on the same device."""
    flat = x.detach().reshape(-1)
    n = flat.numel()
    if n == 0:
        return flat
    if n <= max_samples:
        return flat.abs()
    g = torch.Generator(device=flat.device)
    g.manual_seed(seed & 0xFFFFFFFF)
    idx = torch.randint(0, n, (max_samples,), device=flat.device, generator=g, dtype=torch.int64)
    return flat.index_select(0, idx).abs()


def _approx_quantile_abs(x: torch.Tensor, q: float, max_samples: int, *, seed: int) -> torch.Tensor:
    """
    Approximate quantile(|x|) via sampling + kthvalue.
    Returns a scalar tensor on the same device.
    """
    s = _rand_sample_abs(x, max_samples=max_samples, seed=seed)
    n = s.numel()
    if n == 0:
        return torch.zeros((), device=x.device, dtype=torch.float32)
    k = int(math.ceil(q * n))
    k = max(1, min(n, k))
    # kthvalue is 1-indexed in docs; torch.kthvalue takes k in [1..n]
    return torch.kthvalue(s, k).values


def _fake_int8_quant_dequant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Fake symmetric INT8 quantization (for behavior tests only).
    quantize to int8 then dequantize back to float, on the same device.
    """
    # clamp scale
    s = torch.clamp(scale.to(torch.float32), min=1e-12)
    q = torch.round(x.to(torch.float32) / s).clamp(-127, 127).to(torch.int8)
    return (q.to(torch.float32) * s).to(dtype=x.dtype)


class VirtualBlackwellAdapterV64:
    """
    Per-layer scale cache + pulse statistics.

    This class is purely a helper; the layer wrapper decides when to call
    linear_fast vs linear_pulse.
    """
    def __init__(
        self,
        *,
        q: float = 0.999,
        update_threshold: float = 0.20,
        ema_alpha: float = 0.10,
        quant_samples: int = 50000,
        cheap_samples: int = 2048,
        use_fake_int8: bool = False,  # OFF by default (speed tests)
    ):
        self.q = float(q)
        self.update_threshold = float(update_threshold)
        self.ema_alpha = float(ema_alpha)
        self.quant_samples = int(quant_samples)
        self.cheap_samples = int(cheap_samples)
        self.use_fake_int8 = bool(use_fake_int8)

        self._states: Dict[str, LayerQuantState] = {}

        # convenience counters for your printers
        self.total_wrapped_linear_calls = 0
        self.pulse_calls = 0
        self.fast_calls = 0
        self.scale_cache_reuse = 0
        self.scale_cache_updates = 0

    def _state(self, layer_id: str) -> LayerQuantState:
        st = self._states.get(layer_id)
        if st is None:
            st = LayerQuantState()
            self._states[layer_id] = st
        return st

    @torch.no_grad()
    def _cheap_absmean(self, w: torch.Tensor, st: LayerQuantState, *, seed: int) -> float:
        flat = w.detach().reshape(-1)
        n = flat.numel()
        if n == 0:
            return 0.0
        k = min(self.cheap_samples, n)

        if st.sample_idx is None or st.sample_idx.numel() != k or st.sample_idx.device != flat.device:
            g = torch.Generator(device=flat.device)
            g.manual_seed(seed & 0xFFFFFFFF)
            st.sample_idx = torch.randint(0, n, (k,), device=flat.device, generator=g, dtype=torch.int64)

        x = flat.index_select(0, st.sample_idx)
        return float(x.abs().mean().item())

    @torch.no_grad()
    def _ensure_scale(self, layer_id: str, w: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Returns (scale_tensor, did_update)
        """
        st = self._state(layer_id)
        seed = hash(layer_id)

        # If we have a scale, try cheap drift check first
        if st.w_scale is not None and st.cheap_ema is not None:
            cheap = self._cheap_absmean(w, st, seed=seed)
            rel = abs(cheap - st.cheap_ema) / max(st.cheap_ema, 1e-12)
            if rel <= self.update_threshold:
                # reuse
                st.scale_reuses += 1
                self.scale_cache_reuse += 1
                return st.w_scale, False

        # Need update (first time or drift too big): compute sampled quantile(|w|)
        qv = _approx_quantile_abs(w, q=self.q, max_samples=self.quant_samples, seed=seed)
        qf = float(qv.to(torch.float32).item())
        scale = max(qf / 127.0, 1e-12)
        st.w_scale = torch.tensor(scale, device=w.device, dtype=torch.float32)

        # update EMAs
        cheap_now = self._cheap_absmean(w, st, seed=seed ^ 0x9E3779B9)
        if st.metric_ema is None:
            st.metric_ema = qf
            st.cheap_ema = cheap_now
        else:
            a = self.ema_alpha
            st.metric_ema = (1.0 - a) * st.metric_ema + a * qf
            st.cheap_ema = (1.0 - a) * st.cheap_ema + a * cheap_now
        st.last_q = qf

        st.scale_updates += 1
        self.scale_cache_updates += 1
        return st.w_scale, True

    def linear_fast(self, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor]) -> torch.Tensor:
        return F.linear(x, w, b)

    def linear_pulse(self, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor], layer_id: str) -> torch.Tensor:
        """
        Pulse path:
          - update/reuse scale
          - optional fake INT8 (OFF by default)
        """
        scale, _ = self._ensure_scale(layer_id, w)
        if not self.use_fake_int8:
            return F.linear(x, w, b)

        # fake quant-dequant on activations and weights then fp GEMM
        xq = _fake_int8_quant_dequant(x, scale)
        wq = _fake_int8_quant_dequant(w, scale)
        return F.linear(xq, wq, b)

    def observe_call(self, layer_id: str, *, is_pulse: bool) -> None:
        st = self._state(layer_id)
        st.total_calls += 1
        self.total_wrapped_linear_calls += 1
        if is_pulse:
            st.pulse_calls += 1
            self.pulse_calls += 1
        else:
            st.fast_calls += 1
            self.fast_calls += 1

    def export_stats(self) -> Dict[str, Any]:
        """
        Returns a dict compatible with the speed-test printer plus detailed per-layer stats.
        """
        layers = {}
        for k, st in self._states.items():
            layers[k] = {
                "total": st.total_calls,
                "pulse": st.pulse_calls,
                "fast": st.fast_calls,
                "scale_updates": st.scale_updates,
                "scale_reuses": st.scale_reuses,
                "last_q": st.last_q,
                "metric_ema": st.metric_ema,
                "cheap_ema": st.cheap_ema,
            }

        return {
            "Total wrapped-linear calls": self.total_wrapped_linear_calls,
            "Pulse calls": self.pulse_calls,
            "Fast calls": self.fast_calls,
            "Scale cache reuse": self.scale_cache_reuse,
            "Scale cache updates": self.scale_cache_updates,
            "layers": layers,
        }
