
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

    # drift stats (tensor scalars for torch.compile compatibility)
    metric_ema: Optional[torch.Tensor] = None   # EMA of quantile(|w|)
    cheap_ema: Optional[torch.Tensor] = None    # EMA of mean(|w|) on subsample
    last_q: Optional[torch.Tensor] = None       # last quantile(|w|)

    # activation RMS stats (for threshold-based pulse triggering)
    act_rms_ema: Optional[torch.Tensor] = None  # EMA of input activation RMS
    last_act_rms: Optional[torch.Tensor] = None # last observed activation RMS
    act_rms_samples: int = 0                    # number of RMS observations

    # last step when scale was updated (for throttling)
    scale_last_step: Optional[int] = None

    # fixed subsample indices for cheap drift check
    sample_idx: Optional[torch.Tensor] = None

    # meta-plan (exact, permutation-based) for GEMM routing
    perm_out: Optional[torch.Tensor] = None
    inv_perm_out: Optional[torch.Tensor] = None
    plan_last_step: Optional[int] = None
    plan_updates: int = 0
    plan_reuses: int = 0

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
    Fake symmetric INT8 quantization with Straight-Through Estimator (STE).
    Forward: quantize to int8 then dequantize back to float
    Backward: gradients pass through directly (STE)
    """
    # clamp scale
    s = torch.clamp(scale.to(torch.float32), min=1e-12)

    # Forward path: fake quantization
    # Use a custom autograd function for STE
    class FakeQuantSTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, s):
            # Quantize to int8 and back
            x_float = x.to(torch.float32)
            q = torch.round(x_float / s).clamp(-127, 127)
            # Return dequantized result (int8 conversion is just for simulation)
            return (q * s).to(dtype=x.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            # STE: gradient passes through directly
            return grad_output, None

    return FakeQuantSTE.apply(x, s)

@torch.no_grad()
def _build_perm_out(w: torch.Tensor, metric: str = "row_norm") -> torch.Tensor:
    """
    Build a deterministic permutation of output channels (rows of W).
    This is an *exact* transformation: it does NOT change math if you later
    unpermute outputs. It is a lightweight "routing scaffold" for future
    block/cluster planners.

    metric:
      - "row_norm": sort rows by L2 norm (descending)
      - "row_absmean": sort rows by mean(|w|)
    """
    ww = w.detach()
    if metric == "row_absmean":
        score = ww.abs().mean(dim=1)
    else:
        # default row_norm
        score = torch.linalg.vector_norm(ww.to(torch.float32), ord=2, dim=1)
    perm = torch.argsort(score, descending=True)
    return perm.to(dtype=torch.int64, device=w.device)


@torch.no_grad()
def _invert_perm(perm: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=perm.device, dtype=perm.dtype)
    return inv



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
        enable_stats: bool = True,
        w3_act_rms_threshold: float = 0.02,
        w3_min_update_interval: int = 0,
        enable_meta_plan: bool = False,
        meta_plan_apply: bool = False,
        meta_plan_metric: str = "row_norm",
        meta_plan_interval: int = 200,  # steps between plan refresh
        use_fake_int8: bool = False,  # OFF by default (speed tests)
        enable_scale_cache: bool = True,
        act_rms_threshold: float = 0.02,  # 2% activation RMS change triggers update
        min_act_samples: int = 3,  # min samples before using RMS-based trigger
    ):
        self.enable_stats = bool(enable_stats)
        self.q = float(q)
        self.update_threshold = float(update_threshold)
        self.ema_alpha = float(ema_alpha)
        self.quant_samples = int(quant_samples)
        self.cheap_samples = int(cheap_samples)
        self.use_fake_int8 = bool(use_fake_int8)
        self.enable_scale_cache = bool(enable_scale_cache)
        self.act_rms_threshold = float(act_rms_threshold)
        self.w3_act_rms_threshold = float(w3_act_rms_threshold)
        self.w3_min_update_interval = int(w3_min_update_interval)
        self.min_act_samples = int(min_act_samples)

        # meta-plan routing (exact permutation only; OFF by default)
        self.enable_meta_plan = bool(enable_meta_plan)
        self.meta_plan_apply = bool(meta_plan_apply)
        self.meta_plan_metric = str(meta_plan_metric)
        self.meta_plan_interval = int(max(1, meta_plan_interval))

        self._states: Dict[str, LayerQuantState] = {}

        # convenience counters for your printers
        self.total_wrapped_linear_calls = 0
        self.pulse_calls = 0
        self.fast_calls = 0
        self.scale_cache_reuse = 0
        self.scale_cache_updates = 0
        self.meta_plan_reuse = 0
        self.meta_plan_updates = 0

    def _rms_threshold_for(self, layer_id: str) -> float:
        if layer_id.endswith("w3") or ".w3" in layer_id:
            return self.w3_act_rms_threshold
        return self.act_rms_threshold

    def _min_update_interval_for(self, layer_id: str) -> int:
        if layer_id.endswith("w3") or ".w3" in layer_id:
            return self.w3_min_update_interval
        return 0

    def _state(self, layer_id: str) -> LayerQuantState:
        st = self._states.get(layer_id)
        if st is None:
            st = LayerQuantState()
            self._states[layer_id] = st
        return st

    @torch.no_grad()
    def _cheap_absmean(self, w: torch.Tensor, st: LayerQuantState, *, seed: int) -> torch.Tensor:
        """Return sampled abs mean as tensor scalar (no .item() for torch.compile)"""
        flat = w.detach().reshape(-1)
        n = flat.numel()
        if n == 0:
            return torch.tensor(0.0, device=w.device, dtype=torch.float32)
        k = min(self.cheap_samples, n)

        if st.sample_idx is None or st.sample_idx.numel() != k or st.sample_idx.device != flat.device:
            # Create generator and sample indices only when needed
            g = torch.Generator(device=flat.device)
            g.manual_seed(seed & 0xFFFFFFFF)
            st.sample_idx = torch.randint(0, n, (k,), device=flat.device, generator=g, dtype=torch.int64)

        x = flat.index_select(0, st.sample_idx)
        return x.abs().mean()  # tensor scalar, no .item()

    @torch.no_grad()
    def _ensure_scale(self, layer_id: str, w: torch.Tensor, step: Optional[int] = None) -> Tuple[torch.Tensor, bool]:
        """
        Returns (scale_tensor, did_update)
        """
        st = self._state(layer_id)
        seed = hash(layer_id)

        # If we have a scale, try cheap drift check first
        if st.w_scale is not None and st.cheap_ema is not None:
            cheap = self._cheap_absmean(w, st, seed=seed)
            rel = torch.abs(cheap - st.cheap_ema) / torch.clamp(st.cheap_ema, min=1e-12)
            if bool((rel <= self.update_threshold).all()):  # tensor bool -> Python bool
                # reuse
                st.scale_reuses += 1
                self.scale_cache_reuse += 1
                return st.w_scale, False

        # Need update (first time or drift too big): compute sampled quantile(|w|)
        qv = _approx_quantile_abs(w, q=self.q, max_samples=self.quant_samples, seed=seed)
        qf = qv.to(torch.float32)  # keep as tensor
        scale = torch.clamp(qf / 127.0, min=1e-12)
        st.w_scale = scale

        # update EMAs (using tensor operations)
        cheap_now = self._cheap_absmean(w, st, seed=seed ^ 0x9E3779B9)
        if st.metric_ema is None:
            st.metric_ema = qf.detach().clone()
            st.cheap_ema = cheap_now.detach().clone()
        else:
            a = self.ema_alpha
            st.metric_ema = (1.0 - a) * st.metric_ema + a * qf
            st.cheap_ema = (1.0 - a) * st.cheap_ema + a * cheap_now
        st.last_q = qf

        st.scale_updates += 1
        if step is not None:
            st.scale_last_step = int(step)
        self.scale_cache_updates += 1
        # meta-plan bookkeeping is separate
        return st.w_scale, True

    @torch.no_grad()
    def _should_update_scale(self, layer_id: str, act_rms: torch.Tensor, step: Optional[int] = None) -> bool:
        """Heuristic for when to refresh the cached scale for a given layer."""
        st = self._state(layer_id)

        if st.w_scale is None:
            return True
        if st.act_rms_samples < self.min_act_samples:
            return True

        min_int = self._min_update_interval_for(layer_id)
        if min_int > 0 and step is not None and st.scale_last_step is not None:
            if (int(step) - int(st.scale_last_step)) < min_int:
                return False

        thr = self._rms_threshold_for(layer_id)
        if st.act_rms_ema is not None:
            # Use tensor operations for torch.compile compatibility
            rel_change = torch.abs(act_rms - st.act_rms_ema) / torch.clamp(st.act_rms_ema, min=1e-12)
            if bool((rel_change <= thr).all()):  # tensor bool -> Python bool
                return False

        return True

    @torch.no_grad()
    def _update_act_rms(self, layer_id: str, act_rms: torch.Tensor) -> None:
        """Update activation RMS EMA for this layer."""
        if not self.enable_stats:
            return
        st = self._state(layer_id)
        st.act_rms_samples += 1
        # Keep as tensor, only convert to float when needed for stats export
        st.last_act_rms = act_rms

        if st.act_rms_ema is None:
            st.act_rms_ema = act_rms.detach().clone()
        else:
            a = self.ema_alpha
            # Tensor EMA update for torch.compile compatibility
            st.act_rms_ema = (1.0 - a) * st.act_rms_ema + a * act_rms

    @torch.no_grad()
    def _ensure_meta_plan(self, layer_id: str, w: torch.Tensor, step: Optional[int]) -> None:
        """Update or reuse a cached output-channel permutation plan."""
        if not self.enable_meta_plan:
            return
        st = self._state(layer_id)
        # If no step is provided, treat every call as eligible to refresh.
        now = int(step) if step is not None else None
        if st.perm_out is not None and st.plan_last_step is not None and now is not None:
            if (now - st.plan_last_step) < self.meta_plan_interval:
                st.plan_reuses += 1
                self.meta_plan_reuse += 1
                return

        # (re)build permutation
        perm = _build_perm_out(w, metric=self.meta_plan_metric)
        st.perm_out = perm
        st.inv_perm_out = _invert_perm(perm)
        st.plan_last_step = now
        st.plan_updates += 1
        self.meta_plan_updates += 1



    def linear_fast(self, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor]) -> torch.Tensor:
        return F.linear(x, w, b)

    def linear_pulse(self, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor], layer_id: str, step: Optional[int] = None) -> torch.Tensor:
        """
        Pulse path with activation-RMS-based threshold triggering:
          - compute activation RMS
          - if RMS changed < threshold -> reuse scale, skip expensive quantile
          - else -> update scale via quantile sampling
          - optional fake INT8 (OFF by default)
        """
        # torch.compile 兼容：pulse 路径包含统计/采样/标量化（.item()）等控制面逻辑。
        # 编译时直接回退到普通线性，避免 graph break / 频繁重编译。
        if torch._dynamo.is_compiling():
            return torch.nn.functional.linear(x, w, b)

        st = self._state(layer_id)

        # Compute cheap activation RMS (keep as tensor for torch.compile compatibility)
        with torch.no_grad():
            act_rms_t = x.pow(2).mean().sqrt()  # tensor scalar, no .item()
            should_update = self._should_update_scale(layer_id, act_rms_t, step)

        if should_update:
            scale, _ = self._ensure_scale(layer_id, w, step)
        else:
            # Fast path: reuse cached scale
            scale = st.w_scale if st.w_scale is not None else self._ensure_scale(layer_id, w, step)[0]
            if self.enable_stats:
                st.scale_reuses += 1
                self.scale_cache_reuse += 1

        # Update activation RMS tracking
        self._update_act_rms(layer_id, act_rms_t)

        # Optional: maintain a cached permutation plan (exact; for structural experiments)
        self._ensure_meta_plan(layer_id, w, step)
        if self.enable_meta_plan and self.meta_plan_apply:
            # st 已经在上面获取了，不需要再次调用 _state
            if st.perm_out is not None and st.inv_perm_out is not None:
                wp = w.index_select(0, st.perm_out)
                bp = b.index_select(0, st.perm_out) if b is not None else None
                yp = F.linear(x, wp, bp)
                return yp.index_select(-1, st.inv_perm_out)

        if not self.use_fake_int8:
            return F.linear(x, w, b)

        # fake quant-dequant on activations and weights then fp GEMM
        xq = _fake_int8_quant_dequant(x, scale)
        wq = _fake_int8_quant_dequant(w, scale)
        return F.linear(xq, wq, b)

    @torch.compiler.disable  # Disable compilation for stats to avoid graph break
    def observe_call(self, layer_id: str, *, is_pulse: bool) -> None:
        if not self.enable_stats:
            return  # 吞吐模式：跳过所有统计
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
                "plan_updates": st.plan_updates,
                "plan_reuses": st.plan_reuses,
                "last_q": st.last_q,
                "metric_ema": st.metric_ema,
                "cheap_ema": st.cheap_ema,
                "act_rms_ema": st.act_rms_ema,
                "last_act_rms": st.last_act_rms,
                "act_rms_samples": st.act_rms_samples,
            }

        return {
            "Total wrapped-linear calls": self.total_wrapped_linear_calls,
            "Pulse calls": self.pulse_calls,
            "Fast calls": self.fast_calls,
            "Scale cache reuse": self.scale_cache_reuse,
            "Scale cache updates": self.scale_cache_updates,
            "Meta plan reuse": self.meta_plan_reuse,
            "Meta plan updates": self.meta_plan_updates,
            "layers": layers,
        }
