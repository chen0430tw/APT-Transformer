#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Apps Training Package
"""

# Lazy import to avoid circular dependency
load_model = None

def __getattr__(name):
    """Lazy load on attribute access"""
    global load_model
    if name == 'load_model':
        if load_model is None:
            from .checkpoint import load_model as _load_model
            load_model = _load_model
        return load_model
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = ['load_model']
