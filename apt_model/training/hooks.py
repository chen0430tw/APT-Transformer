# -*- coding: utf-8 -*-
"""Training-time hooks (e.g., DBC-DAC gradient stabilizers)."""
import torch
def register_dbc_dac_hooks(model) -> list:
    try:
        from apt_model.modeling.apt_model import DBCDAC_Optimizer, create_gradient_stabilizer_hook
    except Exception:
        return []
    opt = DBCDAC_Optimizer()
    hooks = []
    for _, p in model.named_parameters():
        if p.requires_grad:
            hooks.append(p.register_hook(create_gradient_stabilizer_hook(opt)))
    return hooks
