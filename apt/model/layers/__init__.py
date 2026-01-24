#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Layers

基础层组件：
- Attention mechanisms
- Feed-forward networks
- Normalization layers
- Embedding layers
- Custom blocks
"""

try:
    from apt.model.layers.embeddings import (
        PositionalEncoding,
        TokenEmbedding,
        ImageEmbedding,
    )
except ImportError:
    PositionalEncoding = None
    TokenEmbedding = None
    ImageEmbedding = None
try:
    from apt.model.layers.advanced_rope import AdvancedRoPE
except ImportError:
    AdvancedRoPE = None

# 尝试导入可选模块（如果导入失败则跳过）
try:
    try:
        from apt.model.layers.apt_control import APTControl
    except ImportError:
        APTControl = None
except ImportError:
    APTControl = None

try:
    try:
        from apt.model.layers.left_spin_smooth import LeftSpinSmooth
    except ImportError:
        LeftSpinSmooth = None
except ImportError:
    LeftSpinSmooth = None

try:
    try:
        from apt.model.layers.memory_augmented_smooth import MemoryAugmentedSmooth
    except ImportError:
        MemoryAugmentedSmooth = None
except ImportError:
    MemoryAugmentedSmooth = None

try:
    try:
        from apt.model.layers.moe_optimized import OptimizedMoE
    except ImportError:
        OptimizedMoE = None
except ImportError:
    OptimizedMoE = None

__all__ = [
    'PositionalEncoding',
    'TokenEmbedding',
    'ImageEmbedding',
    'AdvancedRoPE',
    'APTControl',
    'LeftSpinSmooth',
    'MemoryAugmentedSmooth',
    'OptimizedMoE',
]
