#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model Runtime Module

Provides runtime components for inference and generation:
- Decoder strategies (greedy, beam search, sampling)
- Reasoning mechanisms (CoT, ToT, self-consistency)
- Adaptive computation (halting, budgeting)
"""

try:
    from apt.core.runtime import decoder
except ImportError:
    pass

__all__ = ['decoder']
