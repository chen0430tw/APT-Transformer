
"""
VB Integration v6.4
-------------------
- Uses VirtualBlackwellAdapterV64 (scale reuse fixed + optional fake INT8)
- Round-robin per-layer pulse phases to spread overhead (global-step schedule)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn

from apt.vgpu.runtime.virtual_blackwell_adapter import VirtualBlackwellAdapterV64




@dataclass
class VBConfigV64:
    pulse_interval: int = 20
    q: float = 0.999
    update_threshold: float = 0.20
    ema_alpha: float = 0.10
    quant_samples: int = 50000
    cheap_samples: int = 2048
    use_fake_int8: bool = False  # OFF by default for speed tests
    enable_scale_cache: bool = True  # still track/update per-layer activation scales on pulse
    enable_meta_plan: bool = False
    meta_plan_apply: bool = False
    meta_plan_metric: str = "row_norm"
    meta_plan_interval: int = 200
    act_rms_threshold: float = 0.02  # 2% activation RMS change triggers update
    min_act_samples: int = 3  # min samples before using RMS-based trigger
    # --- extensions for benchmark harness compatibility ---
    pulse_cooldown: int = 0  # if >0, suppress pulses within this many steps per-layer
    gate_projected_mode: bool = False  # if True, bypass VB wrapper on non-pulse calls
    # optional per-layer overrides for '.w3' layers
    w3_act_rms_threshold: float = 0.02
    w3_min_update_interval: int = 0
    enable_stats: bool = True  # False = throughput mode, disable detailed stats


class VBController(nn.Module):
    """A very light-weight step counter.

    IMPORTANT: keep this on CPU/Python to avoid implicit CUDA synchronizations.
    """

    def __init__(self):
        super().__init__()
        self._global_step: int = 0

    def bump(self) -> None:
        self._global_step += 1

    @property
    def step(self) -> int:
        return self._global_step

class VBOptimizedLinearV64(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        adapter: VirtualBlackwellAdapterV64,
        controller: VBController,
        name: str,
        layer_index: int,
        pulse_interval: int,
        gate_projected_mode: bool = False,
        pulse_cooldown: int = 0,
    ):
        super().__init__()
        self.base = base
        self.adapter = adapter
        self.controller = controller
        self.name = name
        self.layer_index = int(layer_index)
        self.pulse_interval = int(max(1, pulse_interval))
        self.phase = self.layer_index % self.pulse_interval
        # 预缓存这些值，避免热路径中的 getattr/字典查找开销
        self._track_stats = adapter.enable_stats
        self._gate_projected_mode = bool(gate_projected_mode)
        self._pulse_cooldown = int(pulse_cooldown)
        self._last_pulse_step = -999

        # 关键优化：如果 gate_projected_mode=True，使用优化的 forward
        if self._gate_projected_mode:
            # 门投影模式：非门点完全绕过 wrapper，直接返回原生线性层结果
            self.forward = self._gate_forward
        else:
            # 传统模式：每次都经过 wrapper
            self.forward = self._wrapper_forward

    def _gate_forward(self, x: torch.Tensor) -> torch.Tensor:
        """门投影模式 forward：非门点零开销"""
        # torch.compile 兼容：编译时不要把"门控/脉冲"控制面纳入图里。
        # 否则 step/phase 这种 Python int 静态属性会触发频繁重编译或回退，导致不稳定甚至变慢。
        if torch._dynamo.is_compiling():
            return self.base.forward(x)

        step = self.controller.step
        is_pulse = (step % self.pulse_interval) == self.phase

        if not is_pulse:
            # 非门点：直接调用原生 linear，完全绕过 VB 逻辑
            return self.base.forward(x)

        # 门点：触发 VB 逻辑
        if self._track_stats:
            self.adapter.observe_call(self.name, is_pulse=True)
        return self.adapter.linear_pulse(x, self.base.weight, self.base.bias, self.name, step=step)

    def _wrapper_forward(self, x: torch.Tensor) -> torch.Tensor:
        """传统 wrapper 模式 forward（gate_projected_mode=False 时使用）"""
        # torch.compile 兼容：wrapper 模式下也避免把 Python 侧统计/门控塞进编译图。
        if torch._dynamo.is_compiling():
            return self.base.forward(x)

        step = self.controller.step
        is_pulse = (step % self.pulse_interval) == self.phase

        # Optional: suppress pulses too close together (per-layer cooldown)
        if is_pulse and self._pulse_cooldown > 0:
            steps_since_last = step - self._last_pulse_step
            if steps_since_last < self._pulse_cooldown:
                is_pulse = False
            else:
                self._last_pulse_step = step

        # 只在需要统计时才调用 observe_call（bench mode 完全跳过）
        if self._track_stats:
            self.adapter.observe_call(self.name, is_pulse=is_pulse)

        if is_pulse:
            return self.adapter.linear_pulse(x, self.base.weight, self.base.bias, self.name, step=step)
        return self.adapter.linear_fast(x, self.base.weight, self.base.bias)

def apply_virtual_blackwell_v64(
    model: nn.Module,
    config: Optional[VBConfigV64] = None,
    skip_large_out_features: int = 20000,
) -> Tuple[nn.Module, VirtualBlackwellAdapterV64]:
    """
    Replace nn.Linear layers in-place (except huge output heads).
    Returns (model, adapter).
    """
    if config is None:
        config = VBConfigV64()

    controller = VBController()
    model._vb_controller = controller  # type: ignore[attr-defined]

    adapter = VirtualBlackwellAdapterV64(
        q=config.q,
        update_threshold=config.update_threshold,
        ema_alpha=config.ema_alpha,
        quant_samples=config.quant_samples,
        cheap_samples=config.cheap_samples,
        enable_meta_plan=config.enable_meta_plan,
        meta_plan_apply=config.meta_plan_apply,
        meta_plan_metric=config.meta_plan_metric,
        meta_plan_interval=config.meta_plan_interval,
        use_fake_int8=config.use_fake_int8,
        enable_scale_cache=config.enable_scale_cache,
        act_rms_threshold=config.act_rms_threshold,
        min_act_samples=config.min_act_samples,
        w3_act_rms_threshold=config.w3_act_rms_threshold,
        w3_min_update_interval=config.w3_min_update_interval,
        enable_stats=config.enable_stats,
    )

    def _pre_hook(_module, _inputs):
        controller.bump()

    if hasattr(model, "_vb_pre_hook_handle"):
        try:
            model._vb_pre_hook_handle.remove()  # type: ignore[attr-defined]
        except Exception:
            pass

    model._vb_pre_hook_handle = model.register_forward_pre_hook(_pre_hook)  # type: ignore[attr-defined]

    replaced = 0
    layer_index = 0

    def replace_in(parent: nn.Module, prefix: str = ""):
        nonlocal replaced, layer_index
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name

            if isinstance(child, nn.Linear):
                if child.out_features >= skip_large_out_features:
                    continue

                wrapped = VBOptimizedLinearV64(
                    base=child,
                    adapter=adapter,
                    controller=controller,
                    name=full_name,
                    layer_index=layer_index,
                    pulse_interval=config.pulse_interval,
                    gate_projected_mode=config.gate_projected_mode,
                    pulse_cooldown=config.pulse_cooldown,
                )
                setattr(parent, child_name, wrapped)
                replaced += 1
                layer_index += 1
            else:
                replace_in(child, full_name)

    replace_in(model)

    model._vb_replaced_linears = replaced  # type: ignore[attr-defined]
    model._vb_pulse_interval = config.pulse_interval  # type: ignore[attr-defined]
    return model, adapter


def vb_stats_summary(adapter: VirtualBlackwellAdapterV64, top_k: int = 12) -> str:
    d = adapter.export_stats()
    layers = d["layers"]
    items = sorted(layers.items(), key=lambda kv: kv[1].get("pulse", 0), reverse=True)

    lines = []
    lines.append(f"Total wrapped-linear calls: {d['Total wrapped-linear calls']}")
    lines.append(f"Pulse calls: {d['Pulse calls']} | Fast calls: {d['Fast calls']}")
    denom = d["Scale cache updates"] + d["Scale cache reuse"]
    if denom > 0:
        hit = 100.0 * d["Scale cache reuse"] / denom
        lines.append(f"Scale cache reuse: {d['Scale cache reuse']} / {denom} ({hit:.1f}%)")
    else:
        lines.append("Scale cache reuse: 0 / 0 (n/a)")
    lines.append("")
    lines.append(f"Top {top_k} layers by pulse calls:")
    for name, st in items[:top_k]:
        act_rms_info = f", act_rms={st['act_rms_ema']:.4f}" if st.get('act_rms_ema') is not None else ""
        lines.append(
            f"- {name}: total={st['total']}, pulse={st['pulse']}, fast={st['fast']}, "
            f"scale_updates={st['scale_updates']}, scale_reuses={st['scale_reuses']}{act_rms_info}"
        )
    return "\n".join(lines)
