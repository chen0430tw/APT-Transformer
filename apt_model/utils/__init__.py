#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model (自生成变换器) Utils Module
提供各种辅助功能的工具模块

重构后：
- 核心功能迁移到 apt_model/core
- 基础设施迁移到 apt_model/infrastructure
- 保持向后兼容性
"""

# ============================================================================
# 从core模块导入（向后兼容）
# ============================================================================
from apt_model.core.system import (
    set_seed,
    get_device,
    memory_cleanup,
    device,
    SystemInitializer,
    _initialize_common,
)

# ============================================================================
# 从infrastructure模块导入（向后兼容）
# ============================================================================
from apt_model.infrastructure.logging import setup_logging
from apt_model.infrastructure.errors import ErrorHandler

# ============================================================================
# 保留的utils模块
# ============================================================================
from .resource_monitor import ResourceMonitor
from .cache_manager import CacheManager
from .language_manager import LanguageManager
from .hardware_check import check_hardware_compatibility
from .visualization import ModelVisualizer
from .time_estimator import TrainingTimeEstimator

# ============================================================================
# 版本信息
# ============================================================================
__version__ = '0.2.0'  # 版本更新以反映重构

# ============================================================================
# 公共API（向后兼容）
# ============================================================================
__all__ = [
    # 从core模块导入
    'set_seed',
    'get_device',
    'memory_cleanup',
    'device',
    'SystemInitializer',
    '_initialize_common',

    # 从infrastructure模块导入
    'setup_logging',
    'ErrorHandler',

    # 保留的utils模块
    'ResourceMonitor',
    'CacheManager',
    'LanguageManager',
    'check_hardware_compatibility',
    'ModelVisualizer',
    'TrainingTimeEstimator',
]

# ============================================================================
# 别名（向后兼容）
# ============================================================================
# 保持EnhancedErrorHandler别名
EnhancedErrorHandler = ErrorHandler
__all__.append('EnhancedErrorHandler')
