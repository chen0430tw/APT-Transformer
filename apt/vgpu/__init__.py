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
    except ImportError:
        pass
    try:
        from apt.vgpu.scheduler import GPUScheduler
    except ImportError:
        pass
    try:
        from apt.vgpu.memory import GPUMemoryManager
    except ImportError:
        pass
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
except ImportError:
    pass
try:
    from apt.vgpu.scheduler import (
        VGPUResourceEstimator,
        quick_estimate,
    )
except ImportError:
    pass

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
]
