# -*- coding: utf-8 -*-
"""Training-time hooks (e.g., DBC-DAC gradient stabilizers)."""
from apt.core.fake_torch import get_torch
torch = get_torch()


class TrainingHook:
    """
    Base class for training hooks.

    Training hooks can be used to modify or monitor the training process
    at various points (e.g., before/after forward pass, gradient computation).
    """

    def __init__(self):
        pass

    def before_forward(self, model, inputs):
        """Called before the forward pass"""
        pass

    def after_forward(self, model, outputs):
        """Called after the forward pass"""
        pass

    def before_backward(self, loss):
        """Called before the backward pass"""
        pass

    def after_backward(self, model):
        """Called after the backward pass"""
        pass

    def before_step(self, optimizer):
        """Called before optimizer step"""
        pass

    def after_step(self, optimizer):
        """Called after optimizer step"""
        pass


def register_dbc_dac_hooks(model) -> list:
    try:
        from apt.model.architectures.apt_model import DBCDAC_Optimizer, create_gradient_stabilizer_hook
    except Exception:
        return []
    opt = DBCDAC_Optimizer()
    hooks = []
    for _, p in model.named_parameters():
        if p.requires_grad:
            hooks.append(p.register_hook(create_gradient_stabilizer_hook(opt)))
    return hooks
