#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model checkpoint utilities for APT Model.
Re-exports from apt.trainops.checkpoints.checkpoint for backward compatibility.
"""

# Use lazy import to avoid circular dependency
load_model = None

def _ensure_load_model():
    """Lazy load the load_model function"""
    global load_model
    if load_model is None:
        # Import directly from the module, bypassing package __init__.py
        import apt.trainops.checkpoints.checkpoint as checkpoint_module
        load_model = checkpoint_module.load_model
    return load_model

# Make load_model available when imported
_ensure_load_model()

__all__ = ['load_model']
