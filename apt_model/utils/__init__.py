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
from apt.core.system import (
    set_seed,
    get_device,
    memory_cleanup,
    device,
    SystemInitializer,
    _initialize_common,
)

from apt.core.hardware import (
    check_hardware_compatibility,
    get_hardware_profile,
    HardwareProfiler,
)

from apt.core.resources import (
    ResourceMonitor,
    CacheManager,
)

# ============================================================================
# 从infrastructure模块导入（向后兼容）
# ============================================================================
from apt_model.infrastructure.logging import setup_logging
from apt_model.infrastructure.errors import ErrorHandler

# ============================================================================
# 保留的utils模块
# ============================================================================
from .language_manager import LanguageManager
from .hardware_check import check_hardware_compatibility
# Optional utilities that rely on heavy visualization dependencies may not be
# available in lightweight environments. Import them lazily so the rest of the
# training stack can function without matplotlib/plotly.
try:
    from .visualization import ModelVisualizer
except Exception:  # pragma: no cover - best effort fallback for optional deps
    ModelVisualizer = None

try:
    from .time_estimator import TrainingTimeEstimator
except Exception:  # pragma: no cover - best effort fallback for optional deps
    TrainingTimeEstimator = None

# ============================================================================
# 版本信息
# ============================================================================
__version__ = '0.3.0'  # 版本更新以反映硬件和资源整合

# ============================================================================
# 公共API（向后兼容）
# ============================================================================
__all__ = [
    # 从core.system模块导入
    'set_seed',
    'get_device',
    'memory_cleanup',
    'device',
    'SystemInitializer',
    '_initialize_common',

    # 从core.hardware模块导入
    'check_hardware_compatibility',
    'get_hardware_profile',
    'HardwareProfiler',

    # 从core.resources模块导入
    'ResourceMonitor',
    'CacheManager',

    # 从infrastructure模块导入
    'setup_logging',
    'ErrorHandler',

    # 保留的utils模块
    'LanguageManager',
    'ModelVisualizer',
    'TrainingTimeEstimator',
]

# ============================================================================
# 别名（向后兼容）
# ============================================================================
# 保持EnhancedErrorHandler别名
EnhancedErrorHandler = ErrorHandler
__all__.append('EnhancedErrorHandler')
