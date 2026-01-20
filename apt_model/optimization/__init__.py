"""
APT Model Optimization Module

包含虚拟Blackwell优化框架和MicroVM压缩技术。
"""

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

__all__ = [
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
