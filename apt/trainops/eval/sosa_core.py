#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SOSA Core - Re-export from apt.trainops.engine.sosa_core

This module provides backward compatibility for imports within apt.trainops.eval.
The actual implementation is in apt.trainops.engine.sosa_core.
"""

from apt.trainops.engine.sosa_core import (
    Event,
    BinaryTwin,
    SparseMarkov,
    SOSA,
    SOSACore,
)

__all__ = [
    'Event',
    'BinaryTwin',
    'SparseMarkov',
    'SOSA',
    'SOSACore',
]
