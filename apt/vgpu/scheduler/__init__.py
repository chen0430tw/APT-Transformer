#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU Scheduler

GPU任务调度：
- Task scheduling algorithms
- Priority management
- Load balancing
- Resource allocation
- VGPU resource estimation
"""

try:
    from apt.vgpu.scheduler.vgpu_estimator import (
        VGPUResourceEstimator,
        ModelConfig,
        MemoryEstimate,
        VGPUConfig,
        quick_estimate,
    )
except ImportError:
    VGPUResourceEstimator = None
    ModelConfig = None
    MemoryEstimate = None
    VGPUConfig = None
    quick_estimate = None

__all__ = [
    'VGPUResourceEstimator',
    'ModelConfig',
    'MemoryEstimate',
    'VGPUConfig',
    'quick_estimate',
]
