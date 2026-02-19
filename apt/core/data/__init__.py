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

# Export 多语言流式混合器
from .streaming_mixer import (
    MULTILINGUAL_BASE_MIX,
    make_multi_source_iterable,
    create_multilingual_base_iterable,
    make_mixed_hf_iterable,
    create_mixed_iterable,
    MixedStreamDataset,
    make_url_iterable_dataset,
    make_prolong_mds_iterable,
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
    # 多语言流式混合器
    'MULTILINGUAL_BASE_MIX',
    'make_multi_source_iterable',
    'create_multilingual_base_iterable',
    'make_mixed_hf_iterable',
    'create_mixed_iterable',
    'MixedStreamDataset',
    'make_url_iterable_dataset',
    'make_prolong_mds_iterable',
    # Multimodal
    'MultimodalDataset',
    'MultimodalCollator',
    'create_multimodal_dataloader',
    'TextOnlyDataset',
    'VisionOnlyDataset',
    'AudioOnlyDataset',
]