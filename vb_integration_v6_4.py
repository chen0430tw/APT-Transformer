
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
import torch.nn as nn

from virtual_blackwell_adapter_v6_4 import VirtualBlackwellAdapterV64


@dataclass
class VBConfigV64:
    pulse_interval: int = 20
    q: float = 0.999
    update_threshold: float = 0.20
    ema_alpha: float = 0.10
    quant_samples: int = 50000
    cheap_samples: int = 2048
    use_fake_int8: bool = False  # OFF by default for speed tests


class VBController(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("_global_step", torch.zeros((), dtype=torch.long), persistent=False)

    def bump(self):
        self._global_step += 1

    @property
    def step(self) -> int:
        return int(self._global_step.item())


class VBOptimizedLinearV64(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        adapter: VirtualBlackwellAdapterV64,
        controller: VBController,
        name: str,
        layer_index: int,
        pulse_interval: int,
    ):
        super().__init__()
        self.base = base
        self.adapter = adapter
        self.controller = controller
        self.name = name
        self.layer_index = int(layer_index)
        self.pulse_interval = int(max(1, pulse_interval))
        self.phase = self.layer_index % self.pulse_interval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        step = self.controller.step
        is_pulse = (step % self.pulse_interval) == self.phase

        self.adapter.observe_call(self.name, is_pulse=is_pulse)

        if is_pulse:
            return self.adapter.linear_pulse(x, self.base.weight, self.base.bias, self.name)
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
        use_fake_int8=config.use_fake_int8,
    )

    def _pre_hook(_module, _inputs):
        controller.bump()

    if hasattr(model, "_vb_pre_hook_handle"):
        try:
            model._vb_pre_hook_handle.remove()
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
        lines.append(
            f"- {name}: total={st['total']}, pulse={st['pulse']}, fast={st['fast']}, "
            f"scale_updates={st['scale_updates']}, scale_reuses={st['scale_reuses']}"
        )
    return "\n".join(lines)
