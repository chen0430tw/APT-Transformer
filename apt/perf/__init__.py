#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Performance Module (L1 Performance Layer)

性能层 - Virtual Blackwell 优化栈：
- GPU Flash (MXFP4 量化 + GPU-SSD 直通)
- VGPU Stack (多级内存层次)
- NPU 多厂商适配
- 分布式训练 (DDP, FSDP)

使用示例:
    >>> import apt
    >>> apt.enable('standard')  # 加载 L0 + L1
    >>> from apt.perf import enable_virtual_blackwell
    >>> enable_virtual_blackwell(mode='balanced')
"""

# ═══════════════════════════════════════════════════════════
# Virtual Blackwell Stack
# ═══════════════════════════════════════════════════════════
try:
    from apt.perf.optimization.virtual_blackwell_adapter import (
        VirtualBlackwellAdapter,
        enable_virtual_blackwell
    )
except ImportError:
    VirtualBlackwellAdapter = None
    enable_virtual_blackwell = None

try:
    from apt.perf.optimization.vgpu_stack import VGPUStack, VGPUConfig
except ImportError:
    VGPUStack = None
    VGPUConfig = None

try:
    from apt.perf.optimization.vgpu_estimator import VGPUEstimator
except ImportError:
    VGPUEstimator = None

try:
    from apt.perf.optimization.microvm_compression import MicroVMCompression
except ImportError:
    MicroVMCompression = None

try:
    from apt.perf.optimization.mxfp4_quantization import MXFP4Quantizer
except ImportError:
    MXFP4Quantizer = None

# ═══════════════════════════════════════════════════════════
# NPU Infrastructure
# ═══════════════════════════════════════════════════════════
try:
    from apt.perf.infrastructure.npu_manager import NPUManager
except ImportError:
    NPUManager = None

try:
    from apt.perf.infrastructure.cloud_npu import CloudNPUAdapter
except ImportError:
    CloudNPUAdapter = None

# ═══════════════════════════════════════════════════════════
# Training Optimizations
# ═══════════════════════════════════════════════════════════
try:
    from apt.perf.training.mixed_precision import MixedPrecisionTrainer
except ImportError:
    MixedPrecisionTrainer = None

try:
    from apt.perf.training.checkpoint import AtomicCheckpoint
except ImportError:
    AtomicCheckpoint = None

# ═══════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════
__all__ = [
    # Virtual Blackwell
    'VirtualBlackwellAdapter',
    'enable_virtual_blackwell',
    'VGPUStack',
    'VGPUConfig',
    'VGPUEstimator',
    'MicroVMCompression',
    'MXFP4Quantizer',

    # NPU
    'NPUManager',
    'CloudNPUAdapter',

    # Training
    'MixedPrecisionTrainer',
    'AtomicCheckpoint',
]