
"""
VB Integration v6.2
- Uses VirtualBlackwellAdapterV62
- Global-step pulse scheduling with round-robin phases (smooth overhead)
- Optional sparse attention patching hook (if vb_sparse_attention is available)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from virtual_blackwell_adapter_v6_2 import VirtualBlackwellAdapterV62


@dataclass
class VBConfigV62:
    pulse_interval: int = 20
    q: float = 0.999
    sample_size: int = 8192
    drift_threshold: float = 0.20
    enable_fp4_coarse: bool = True


class VBController(nn.Module):
    """
    Keeps a global step counter (increments once per model forward).
    Attached to model as model._vb_controller.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer("_global_step", torch.zeros((), dtype=torch.long), persistent=False)

    def bump(self):
        self._global_step += 1

    @property
    def step(self) -> int:
        return int(self._global_step.item())


class VBOptimizedLinearV62(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        adapter: VirtualBlackwellAdapterV62,
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
        # round-robin pulse: only a subset of layers do pulse in a given step
        step = self.controller.step
        is_pulse = (step % self.pulse_interval) == self.phase
        if is_pulse:
            return self.adapter.linear_pulse(x, self.base.weight, self.base.bias, self.name)
        return self.adapter.linear_fast(x, self.base.weight, self.base.bias, self.name)


def _iter_named_linears(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, module


def apply_virtual_blackwell_v62(
    model: nn.Module,
    config: Optional[VBConfigV62] = None,
    skip_large_out_features: int = 20000,
) -> Tuple[nn.Module, VirtualBlackwellAdapterV62]:
    """
    Replace most nn.Linear layers with VBOptimizedLinearV62 in-place.
    Keeps ultra-large output heads as-is by default (e.g., vocab projection).
    Returns (model, adapter).
    """
    if config is None:
        config = VBConfigV62()

    # attach controller
    controller = VBController()
    model._vb_controller = controller  # type: ignore[attr-defined]

    adapter = VirtualBlackwellAdapterV62(
        q=config.q,
        sample_size=config.sample_size,
        drift_threshold=config.drift_threshold,
        enable_fp4_coarse=config.enable_fp4_coarse,
    )

    # bump global step once per forward of the root model
    def _pre_hook(_module, _inputs):
        controller.bump()

    # remove existing hook if any
    if hasattr(model, "_vb_pre_hook_handle"):
        try:
            model._vb_pre_hook_handle.remove()
        except Exception:
            pass
    model._vb_pre_hook_handle = model.register_forward_pre_hook(_pre_hook)  # type: ignore[attr-defined]

    # Do replacement by traversing immediate children recursively
    replaced = 0
    layer_index = 0

    def replace_in(parent: nn.Module, prefix: str = ""):
        nonlocal replaced, layer_index
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name

            if isinstance(child, nn.Linear):
                # Skip enormous vocab heads etc.
                if child.out_features >= skip_large_out_features:
                    continue

                wrapped = VBOptimizedLinearV62(
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


def vb_stats_summary(adapter: VirtualBlackwellAdapterV62, top_k: int = 10) -> str:
    items = []
    for name, st in adapter.stats.items():
        items.append((name, st))
    # sort by pulse calls desc
    items.sort(key=lambda t: t[1].pulse_calls, reverse=True)
    lines = []
    total_calls = sum(st.total_calls for _, st in items)
    pulse_calls = sum(st.pulse_calls for _, st in items)
    fast_calls = sum(st.fast_calls for _, st in items)
    scale_updates = sum(st.scale_updates for _, st in items)
    scale_reuses = sum(st.scale_reuses for _, st in items)

    lines.append(f"Total wrapped-linear calls: {total_calls}")
    lines.append(f"Pulse calls: {pulse_calls} | Fast calls: {fast_calls}")
    if scale_updates + scale_reuses > 0:
        hit = 100.0 * scale_reuses / (scale_updates + scale_reuses)
        lines.append(f"Scale cache reuse: {scale_reuses} / {scale_updates + scale_reuses} ({hit:.1f}%)")
    lines.append("")
    lines.append(f"Top {top_k} layers by pulse calls:")
    for name, st in items[:top_k]:
        lines.append(
            f"- {name}: total={st.total_calls}, pulse={st.pulse_calls}, fast={st.fast_calls}, "
            f"scale_updates={st.scale_updates}, scale_reuses={st.scale_reuses}"
        )
    return "\n".join(lines)
