"""
APT Model Optimization Module

包含GPU Flash优化框架（FP4量化 + Triton Kernel融合 + Flash Attention）
"""

# GPU Flash优化（推荐）
from apt_model.optimization.gpu_flash_optimization import (
    FP4Codec,
    FusedFP4Linear,
    FlashAttention,
    OptimizedTransformerBlock,
    HAS_TRITON
)

# 旧版虚拟Blackwell（已弃用，仅兼容）
from apt_model.optimization.microvm_compression import (
    AutoCompressor,
    compress,
    CompressedLinear
)

from apt_model.optimization.virtual_blackwell_adapter import (
    VirtualBlackwellAdapter,
    create_virtual_blackwell
)

# PyTorch集成模块
from apt_model.optimization.vb_integration import (
    VBOptimizedLinear,
    VBModelWrapper,
    enable_vb_optimization,
    TORCH_AVAILABLE as VB_TORCH_AVAILABLE
)

# VGPU堆叠技术（最新）
from apt_model.optimization.vgpu_stack import (
    VGPUStack,
    VGPULevel,
    VGPUStackLinear,
    create_vgpu_stack
)

# VGPU资源评估器
from apt_model.optimization.vgpu_estimator import (
    VGPUResourceEstimator,
    ModelConfig,
    MemoryEstimate,
    VGPUConfig,
    quick_estimate
)

# VGPU全局启用器（最简单的方式）
from apt_model.optimization import vb_global
from apt_model.optimization import vb_autopatch

# NPU后端适配器（统一GPU/NPU/CPU接口）
from apt_model.optimization.npu_backend import (
    DeviceBackend,
    UnifiedDeviceManager,
    get_device_manager,
    get_unified_backend,
    is_npu_available,
    get_accelerator_type
)

__all__ = [
    # GPU Flash优化（推荐使用）
    'FP4Codec',
    'FusedFP4Linear',
    'FlashAttention',
    'OptimizedTransformerBlock',
    'HAS_TRITON',

    # VGPU堆叠（最新）
    'VGPUStack',
    'VGPULevel',
    'VGPUStackLinear',
    'create_vgpu_stack',

    # VGPU资源评估器
    'VGPUResourceEstimator',
    'ModelConfig',
    'MemoryEstimate',
    'VGPUConfig',
    'quick_estimate',

    # VGPU全局启用器（一行启用虚空算力）
    'vb_global',
    'vb_autopatch',

    # NPU后端适配器（GPU/NPU/CPU统一接口）
    'DeviceBackend',
    'UnifiedDeviceManager',
    'get_device_manager',
    'get_unified_backend',
    'is_npu_available',
    'get_accelerator_type',

    # 旧版（兼容）
    'AutoCompressor',
    'compress',
    'CompressedLinear',
    'VirtualBlackwellAdapter',
    'create_virtual_blackwell',
    'VBOptimizedLinear',
    'VBModelWrapper',
    'enable_vb_optimization',
    'VB_TORCH_AVAILABLE',
]
