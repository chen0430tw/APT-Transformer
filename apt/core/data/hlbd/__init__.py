#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HLBD (Hierarchical Language Bootstrapping Dataset) Module

This module provides comprehensive support for training APT models
on the Hierarchical Language Bootstrapping Dataset.

Components:
- hlbd_adapter: Data processing and evaluation utilities
- hlbd: Command-line training and evaluation interface
"""

# Export data processing components
from .hlbd_adapter import (
    HLBDDataProcessor,
    HLBDDataset,
    HLBDModelEvaluator,
    prepare_hlbd_tokenizer,
    create_hlbd_apt_config,
    prepare_hlbd_datasets
)

# Export main training function for programmatic use
try:
    from .hlbd import main as train_hlbd
except ImportError:
    # hlbd.py might import torch which may not be available
    train_hlbd = None

__all__ = [
    # Data processing
    'HLBDDataProcessor',
    'HLBDDataset',
    'HLBDModelEvaluator',
    'prepare_hlbd_tokenizer',
    'create_hlbd_apt_config',
    'prepare_hlbd_datasets',
    # Training
    'train_hlbd',
]
