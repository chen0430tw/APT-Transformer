#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
核心模块

提供系统初始化、设备管理、随机种子等核心功能
"""

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

__all__ = [
    # 设备管理
    'get_device',
    'get_device_info',
    'set_device',
    'device',
    # 随机种子
    'set_seed',
    # 内存管理
    'memory_cleanup',
    'get_memory_info',
    # 系统初始化
    'SystemInitializer',
    '_initialize_common',
]
