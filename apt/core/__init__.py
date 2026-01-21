#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Core Module (L0 Kernel Layer)

核心层 - 包含APT的三大创新算法：
- Autopoietic Transform (自生成变换)
- DBC-DAC 优化
- Left-Spin Smooth 平滑

以及基础的训练、生成、配置系统。

使用示例：
    >>> import apt
    >>> apt.enable('lite')  # 仅加载 L0 核心
    >>> from apt.core import APTModel, APTConfig
"""

# ═══════════════════════════════════════════════════════════
# Registry System (旧代码兼容)
# ═══════════════════════════════════════════════════════════
try:
    from apt.core.registry import (
        Provider,
        Registry,
        registry,
        register_provider,
        get_provider
    )
except ImportError:
    Provider = None
    Registry = None
    registry = None
    register_provider = None
    get_provider = None

# ═══════════════════════════════════════════════════════════
# Configuration System
# ═══════════════════════════════════════════════════════════
try:
    from apt.core.config import config as APTConfig
except ImportError:
    try:
        from apt.core.config import APTConfig
    except ImportError:
        APTConfig = None

try:
    from apt.core.config import multimodal_config as MultimodalConfig
except ImportError:
    try:
        from apt.core.config import MultimodalConfig
    except ImportError:
        MultimodalConfig = None

# ═══════════════════════════════════════════════════════════
# Core Models (从迁移的 modeling/ 导入)
# ═══════════════════════════════════════════════════════════
try:
    from apt.core.modeling.apt_model import APTLargeModel as APTModel
except ImportError:
    APTModel = None

try:
    from apt.core.modeling.multimodal_model import MultimodalAPTModel
except ImportError:
    MultimodalAPTModel = None

try:
    from apt.core.modeling.embeddings import APTEmbedding
except ImportError:
    APTEmbedding = None

# ═══════════════════════════════════════════════════════════
# Training System
# ═══════════════════════════════════════════════════════════
try:
    from apt.core.training.trainer import train_model
except ImportError:
    train_model = None

# ═══════════════════════════════════════════════════════════
# Generation System
# ═══════════════════════════════════════════════════════════
try:
    from apt.core.generation.generator import Generator
except ImportError:
    Generator = None

try:
    from apt.core.generation.evaluator import Evaluator
except ImportError:
    Evaluator = None

# ═══════════════════════════════════════════════════════════
# Exceptions
# ═══════════════════════════════════════════════════════════
try:
    from apt.core.exceptions import (
        APTError,
        ConfigError,
        ModelError,
        TrainingError
    )
except ImportError:
    APTError = None
    ConfigError = None
    ModelError = None
    TrainingError = None

# ═══════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════
__all__ = [
    # Registry (backward compatibility)
    'Provider',
    'Registry',
    'registry',
    'register_provider',
    'get_provider',

    # Configuration
    'APTConfig',
    'MultimodalConfig',

    # Core Models
    'APTModel',
    'MultimodalAPTModel',
    'APTEmbedding',

    # Training
    'train_model',

    # Generation
    'Generator',
    'Evaluator',

    # Exceptions
    'APTError',
    'ConfigError',
    'ModelError',
    'TrainingError',
]
