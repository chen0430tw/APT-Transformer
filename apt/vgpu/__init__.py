#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT vGPU Domain (Virtual Blackwell)

虚拟GPU栈：独立的GPU虚拟化和资源管理域

子模块：
- runtime: GPU运行时环境
- scheduler: GPU任务调度
- memory: GPU内存管理
- monitoring: GPU监控和性能分析

Virtual Blackwell特性：
- GPU虚拟化
- 资源隔离
- 动态调度
- 性能监控

使用示例：
    try:
        from apt.vgpu.runtime import VirtualGPU
    except (ImportError, OSError):
        VirtualGPU = None
    try:
        from apt.vgpu.scheduler import GPUScheduler
    except (ImportError, OSError):
        GPUScheduler = None
    try:
        from apt.vgpu.memory import GPUMemoryManager
    except (ImportError, OSError):
        GPUMemoryManager = None
"""

__version__ = '2.0.0-alpha'

# 主要模块导出
try:
    from apt.vgpu.runtime import (
        VirtualBlackwellAdapter,
        create_virtual_blackwell,
        VGPUStack,
        create_vgpu_stack,
        VBOptimizedLinear,
        enable_vb_optimization,
    )
except (ImportError, OSError):
    VirtualBlackwellAdapter = None
    create_virtual_blackwell = None
    VGPUStack = None
    create_vgpu_stack = None
    VBOptimizedLinear = None
    enable_vb_optimization = None
try:
    from apt.vgpu.scheduler import (
        VGPUResourceEstimator,
        quick_estimate,
    )
except (ImportError, OSError):
    VGPUResourceEstimator = None
    quick_estimate = None

# LECAC — 激活值量化存储（INT2 默认，显存降至 FP32 的 1/16）
try:
    from apt.vgpu.runtime.lecac import (
        LECACConfig,
        LECACLinear,
        replace_linear_with_lecac,
    )
except (ImportError, OSError):
    LECACConfig = None
    LECACLinear = None
    replace_linear_with_lecac = None

__all__ = [
    # Runtime
    'VirtualBlackwellAdapter',
    'create_virtual_blackwell',
    'VGPUStack',
    'create_vgpu_stack',
    'VBOptimizedLinear',
    'enable_vb_optimization',
    # Scheduler
    'VGPUResourceEstimator',
    'quick_estimate',
    # LECAC
    'LECACConfig',
    'LECACLinear',
    'replace_linear_with_lecac',
]
