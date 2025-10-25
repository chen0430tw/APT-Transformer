#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
核心模块

提供系统初始化、设备管理、硬件检测、资源管理等核心功能
"""

# ============================================================================
# 系统模块
# ============================================================================
from .system import (
    get_device,
    get_device_info,
    set_device,
    set_seed,
    memory_cleanup,
    get_memory_info,
    SystemInitializer,
    _initialize_common,
    device,
)

# ============================================================================
# 硬件模块
# ============================================================================
from .hardware import (
    estimate_gpu_performance,
    get_gpu_properties,
    get_all_gpu_properties,
    HardwareProfiler,
    get_hardware_profile,
    estimate_model_memory,
    check_hardware_compatibility,
    get_recommended_batch_size,
)

# ============================================================================
# 资源模块
# ============================================================================
from .resources import (
    ResourceMonitor,
    CacheManager,
)

__all__ = [
    # 系统
    'get_device',
    'get_device_info',
    'set_device',
    'device',
    'set_seed',
    'memory_cleanup',
    'get_memory_info',
    'SystemInitializer',
    '_initialize_common',

    # 硬件
    'estimate_gpu_performance',
    'get_gpu_properties',
    'get_all_gpu_properties',
    'HardwareProfiler',
    'get_hardware_profile',
    'estimate_model_memory',
    'check_hardware_compatibility',
    'get_recommended_batch_size',

    # 资源
    'ResourceMonitor',
    'CacheManager',
]
