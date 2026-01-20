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
