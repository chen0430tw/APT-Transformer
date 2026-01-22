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
# 使用try-except避免在缺少torch等依赖时导入失败
# ============================================================================
try:
    from apt_model.core.system import (
        set_seed,
        get_device,
        memory_cleanup,
        device,
        SystemInitializer,
        _initialize_common,
    )
except ImportError:
    # 如果torch等依赖缺失，提供占位符
    set_seed = None
    get_device = None
    memory_cleanup = None
    device = None
    SystemInitializer = None
    _initialize_common = None

try:
    from apt_model.core.hardware import (
        check_hardware_compatibility,
        get_hardware_profile,
        HardwareProfiler,
    )
except ImportError:
    check_hardware_compatibility = None
    get_hardware_profile = None
    HardwareProfiler = None

try:
    from apt_model.core.resources import (
        ResourceMonitor,
        CacheManager,
    )
except ImportError:
    ResourceMonitor = None
    CacheManager = None

# ============================================================================
# 从infrastructure模块导入（向后兼容）
# ============================================================================
try:
    from apt_model.infrastructure.logging import setup_logging
except ImportError:
    setup_logging = None

try:
    from apt_model.infrastructure.errors import ErrorHandler
except ImportError:
    ErrorHandler = None

# ============================================================================
# 保留的utils模块
# ============================================================================
try:
    from .language_manager import LanguageManager
except ImportError:
    LanguageManager = None

# 注意：check_hardware_compatibility 已从 apt_model.core.hardware 导入
# 这里不需要重复从 hardware_check 导入
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
