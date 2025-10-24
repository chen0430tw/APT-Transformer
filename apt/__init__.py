#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT - Autopoietic Transformer

A microkernel-based deep learning framework with plugin architecture.
"""

__version__ = "0.1.0"
__author__ = "APT Team"

from apt.core.registry import registry, Provider, register_provider, get_provider

__all__ = [
    'registry',
    'Provider',
    'register_provider',
    'get_provider',
]
