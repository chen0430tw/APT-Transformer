#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Modeling Module

Model assembly and component management for APT architecture.
"""

try:
    from apt.modeling.compose import ModelBuilder
except ImportError:
    ModelBuilder = None

__all__ = [
    'ModelBuilder',
]
