#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model (自生成变换器) Data Processing Module
Provides functionality for loading, processing, and managing datasets
"""

# Export the main data loading functions
from .external_data import load_external_data, train_with_external_data
from .huggingface_loader import load_huggingface_dataset
from .data_processor import get_training_texts

# Export the unified data pipeline (new)
from .pipeline import DataLoader, DataProcessor, DataPipeline, quick_load

# Export multimodal dataset components
from .multimodal_dataset import (
    MultimodalDataset,
    MultimodalCollator,
    create_multimodal_dataloader,
    TextOnlyDataset,
    VisionOnlyDataset,
    AudioOnlyDataset
)

# Define module exports
__all__ = [
    # Legacy functions
    'load_external_data',
    'train_with_external_data',
    'load_huggingface_dataset',
    'get_training_texts',
    # Unified pipeline
    'DataLoader',
    'DataProcessor',
    'DataPipeline',
    'quick_load',
    # Multimodal
    'MultimodalDataset',
    'MultimodalCollator',
    'create_multimodal_dataloader',
    'TextOnlyDataset',
    'VisionOnlyDataset',
    'AudioOnlyDataset',
]