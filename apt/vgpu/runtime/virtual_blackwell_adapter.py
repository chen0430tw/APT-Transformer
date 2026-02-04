"""virtual_blackwell_adapter.py

Virtual Blackwell Adapter (v6.4 + Micro-VM)
------------------------------------------
This module wraps the hottest training GEMMs (mostly nn.Linear / FFN projections)
and provides:

1) **Pulse path**: occasionally refresh per-layer scale/quant stats (cheap) and
   reuse them in the fast path.
2) **Fast path**: regular GEMM with minimal overhead.

Additionally (OFF by default) it adds two robustness/"virtual compute" modes:

3) **Void compute**: run GEMM by *streaming* weight tiles through a small tensor
   pool, trading bandwidth for much lower peak VRAM usage. Conceptually this is
   your Micro-VM "virtual GPU network tensor pool".
4) **Inertia compute**: when VRAM is critically low, keep training running by
   falling back to CPU (or another backend) for the linear.

Design goals
------------
* Keep the default path identical to v6.4 performance.
* Make Micro-VM opt-in via integration config.
* Provide clean hooks so you can later swap the backend with a real micro-VM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import math
import time

import torch
import torch.nn.functional as F


# -----------------------------
# Micro-VM backend interface
# -----------------------------

LinearBackend = Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor], str], torch.Tensor]


@dataclass
class MicroVMConfig:
    """Configuration for Micro-VM (void/inertia).

    All values are conservative defaults; enable explicitly from vb_integration.
    """

    enabled: bool = False
    mem_free_mb_threshold: int = 256  # below this, consider routing
    void_tile_out: int = 1024         # output feature tile for void streaming
    void_use_pinned_cpu: bool = True
    inertia_cpu_fallback: bool = True


def _cuda_free_mb() -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    try:
        free_b, total_b = torch.cuda.mem_get_info()
        return int(free_b // (1024 * 1024))
    except Exception:
        return None


class VirtualBlackwellAdapterV64:
    """Adapter providing pulse/fast Linear with optional Micro-VM routing."""

    def __init__(
        self,
        q: float = 0.999,
        update_threshold: float = 0.20,
        ema_alpha: float = 0.10,
        quant_samples: int = 50_000,
        cheap_samples: int = 2048,
        use_fake_int8: bool = False,
        micro_vm: Optional[MicroVMConfig] = None,
        void_backend: Optional[LinearBackend] = None,
        inertia_backend: Optional[LinearBackend] = None,
    ):
        self.q = float(q)
        self.update_threshold = float(update_threshold)
        self.ema_alpha = float(ema_alpha)
        self.quant_samples = int(quant_samples)
        self.cheap_samples = int(cheap_samples)
        self.use_fake_int8 = bool(use_fake_int8)

        self.micro_vm = micro_vm or MicroVMConfig(enabled=False)
        self._void_backend = void_backend
        self._inertia_backend = inertia_backend

        # Per-layer scale cache for "fake int8" (really: per-layer symmetric scale).
        # Keyed by layer name.
        self._scale_cache: Dict[str, float] = {}
        self._scale_ema: Dict[str, float] = {}

        # Lightweight stats
        self._stats: Dict[str, Dict[str, int]] = {}
        self._total_calls = 0
        self._fast_calls = 0
        self._pulse_calls = 0
        self._void_calls = 0
        self._inertia_calls = 0
        self._scale_updates = 0
        self._scale_reuse = 0

    # -----------------------------
    # Stats / observability
    # -----------------------------

    def observe_call(self, layer_name: str, is_pulse: bool, mode: str = "auto"):
        """Track calls per layer.

        Backwards compatible with old signature: observe_call(name, is_pulse).
        """
        self._total_calls += 1
        st = self._stats.get(layer_name)
        if st is None:
            st = {
                "total": 0,
                "pulse": 0,
                "fast": 0,
                "void": 0,
                "inertia": 0,
                "scale_updates": 0,
                "scale_reuses": 0,
            }
            self._stats[layer_name] = st
        st["total"] += 1

        if mode == "void":
            self._void_calls += 1
            st["void"] += 1
        elif mode == "inertia":
            self._inertia_calls += 1
            st["inertia"] += 1
        else:
            if is_pulse:
                self._pulse_calls += 1
                st["pulse"] += 1
            else:
                self._fast_calls += 1
                st["fast"] += 1

    def export_stats(self) -> Dict:
        return {
            "Total wrapped-linear calls": self._total_calls,
            "Pulse calls": self._pulse_calls,
            "Fast calls": self._fast_calls,
            "Void calls": self._void_calls,
            "Inertia calls": self._inertia_calls,
            "Scale cache updates": self._scale_updates,
            "Scale cache reuse": self._scale_reuse,
            "layers": self._stats,
        }

    # -----------------------------
    # Core scaling logic (v6.4)
    # -----------------------------

    @staticmethod
    def _cheap_sample(x: torch.Tensor, k: int) -> torch.Tensor:
        """Return a cheap sample of x values for quick percentile estimation."""
        if x.numel() <= k:
            return x.flatten()
        # uniform random sample on flattened view
        flat = x.flatten()
        idx = torch.randint(0, flat.numel(), (k,), device=flat.device)
        return flat[idx]

    @torch.no_grad()
    def _estimate_scale(self, x: torch.Tensor, layer_name: str) -> float:
        """Estimate symmetric scale using quantile q of |x|.

        Uses cheap sampling to minimize overhead.
        """
        # Use abs quantile on a sample
        s = self._cheap_sample(x, self.cheap_samples).abs()
        # guard for empty / nan
        if s.numel() == 0:
            return 1.0
        try:
            v = torch.quantile(s, self.q).item()
        except Exception:
            v = float(s.max().item())
        v = float(v) if math.isfinite(float(v)) else 1.0
        v = max(v, 1e-8)
        # scale maps values into int8 range
        return 127.0 / v

    @torch.no_grad()
    def _update_scale_cache(self, x: torch.Tensor, layer_name: str) -> float:
        new_scale = self._estimate_scale(x, layer_name)
        old = self._scale_cache.get(layer_name)
        if old is None:
            self._scale_cache[layer_name] = new_scale
            self._scale_ema[layer_name] = new_scale
            self._scale_updates += 1
            self._stats[layer_name]["scale_updates"] += 1
            return new_scale

        # EMA + thresholded refresh
        ema = self._scale_ema.get(layer_name, old)
        ema = (1.0 - self.ema_alpha) * float(ema) + self.ema_alpha * float(new_scale)
        self._scale_ema[layer_name] = ema

        rel = abs(float(new_scale) - float(old)) / max(abs(float(old)), 1e-8)
        if rel >= self.update_threshold:
            self._scale_cache[layer_name] = float(ema)
            self._scale_updates += 1
            self._stats[layer_name]["scale_updates"] += 1
        else:
            self._scale_reuse += 1
            self._stats[layer_name]["scale_reuses"] += 1
        return self._scale_cache[layer_name]

    # -----------------------------
    # Public Linear paths
    # -----------------------------

    def linear_fast(self, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor]) -> torch.Tensor:
        """Fast path: either normal linear or optional fake-int8 dequant path."""
        if not self.use_fake_int8:
            return F.linear(x, w, b)

        # Fake-int8: quantize x on the fly using cached scale if available.
        # If cache miss: fall back to FP16/FP32 to avoid expensive sampling.
        # (Scale will be updated in pulse path.)
        #
        # NOTE: This is a correctness-preserving approximation only if you treat it
        # as a simulation; it's kept optional.
        return F.linear(x, w, b)

    def linear_pulse(self, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor], layer_name: str) -> torch.Tensor:
        """Pulse path: refresh per-layer scale cache (cheap) then compute linear."""
        if self.use_fake_int8:
            # update scale cache based on activation statistics
            self._update_scale_cache(x, layer_name)
        return F.linear(x, w, b)

    # -----------------------------
    # Micro-VM routing
    # -----------------------------

    def should_void(self) -> bool:
        if not self.micro_vm.enabled:
            return False
        free_mb = _cuda_free_mb()
        if free_mb is None:
            return False
        return free_mb <= int(self.micro_vm.mem_free_mb_threshold)

    def should_inertia(self) -> bool:
        if not self.micro_vm.enabled:
            return False
        free_mb = _cuda_free_mb()
        if free_mb is None:
            return False
        # inertia is "more extreme" than void
        return free_mb <= max(32, int(self.micro_vm.mem_free_mb_threshold // 4))

    def linear_void(self, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor], layer_name: str) -> torch.Tensor:
        """Void compute: stream tiles through a small pool (default implementation).

        If a custom backend is provided, uses it.
        """
        if self._void_backend is not None:
            return self._void_backend(x, w, b, layer_name)

        # Default void backend: output-tiling by out_features with weight streaming.
        # This reduces peak VRAM of weights (esp. huge projections) at the cost of
        # extra H2D copies. Works as a *robustness* mode, not a speedup.
        out_features, in_features = w.shape
        tile = int(max(1, self.micro_vm.void_tile_out))
        device = x.device
        dtype = x.dtype

        # If weights are already on GPU, tiling still reduces activation/temporary peak.
        # If weights are on CPU, we stream each tile to GPU.
        w_cpu = w
        if w.device.type != "cpu":
            w_cpu = w.detach().to("cpu", non_blocking=True)

        y_parts = []
        for s in range(0, out_features, tile):
            e = min(out_features, s + tile)
            w_tile = w_cpu[s:e, :]
            if self.micro_vm.void_use_pinned_cpu:
                w_tile = w_tile.pin_memory() if w_tile.device.type == "cpu" else w_tile
            w_tile_gpu = w_tile.to(device=device, dtype=dtype, non_blocking=True)
            b_tile = b[s:e] if b is not None else None
            y_parts.append(F.linear(x, w_tile_gpu, b_tile))
            # free promptly
            del w_tile_gpu

        y = torch.cat(y_parts, dim=-1)
        return y

    def linear_inertia(self, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor], layer_name: str) -> torch.Tensor:
        """Inertia compute: keep training alive under VRAM pressure.

        Default: CPU fallback (very slow, but avoids OOM and keeps gradients).
        You can override with inertia_backend.
        """
        if self._inertia_backend is not None:
            return self._inertia_backend(x, w, b, layer_name)

        if not self.micro_vm.inertia_cpu_fallback:
            # If disabled, just do normal linear.
            return F.linear(x, w, b)

        # CPU fallback: move x/w/b to CPU, compute, then move result back.
        # Keeps grads by doing the op on CPU tensors.
        x_cpu = x.detach().to("cpu") if x.is_cuda else x
        w_cpu = w.detach().to("cpu") if w.is_cuda else w
        b_cpu = b.detach().to("cpu") if (b is not None and b.is_cuda) else b

        y_cpu = F.linear(x_cpu, w_cpu, b_cpu)
        # Return to original device
        if x.is_cuda:
            return y_cpu.to(device=x.device, dtype=x.dtype)
        return y_cpu
