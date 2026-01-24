#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model checkpoint utilities for APT Model.
Re-exports from apt.trainops.checkpoints.checkpoint for backward compatibility.
"""

from apt.trainops.checkpoints.checkpoint import load_model

__all__ = ['load_model']
