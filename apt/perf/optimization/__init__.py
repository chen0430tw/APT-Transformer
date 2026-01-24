"""
APT Model Optimization Module

包含GPU Flash优化框架（FP4量化 + Triton Kernel融合 + Flash Attention）
"""

# GPU Flash优化（推荐）
try:
    from apt.perf.optimization.gpu_flash_optimization import (
        FP4Codec,
        FusedFP4Linear,
        FlashAttention,
        OptimizedTransformerBlock,
        HAS_TRITON
    )
except ImportError:
    pass

# 旧版虚拟Blackwell（已弃用，仅兼容）
try:
    from apt.perf.optimization.microvm_compression import (
        AutoCompressor,
        compress,
        CompressedLinear
    )
except ImportError:
    pass

# ⚠️ Virtual Blackwell已迁移到apt.vgpu域
# 为保持向后兼容，从新位置重导出
import warnings

warnings.warn(
    "Importing Virtual Blackwell from apt.perf.optimization is deprecated. "
    "Please use apt.vgpu instead:\n"
    "  OLD: from apt.perf.optimization import VirtualBlackwellAdapter\n"
    "  NEW: from apt.vgpu.runtime import VirtualBlackwellAdapter",
    DeprecationWarning,
    stacklevel=2
)

try:
    from apt.vgpu.runtime import (
        VirtualBlackwellAdapter,
        create_virtual_blackwell,
        VBOptimizedLinear,
        VBModelWrapper,
        enable_vb_optimization,
        VGPUStack,
        VGPULevel,
        VGPUStackLinear,
        create_vgpu_stack,
    )
except ImportError:
    pass
try:
    from apt.vgpu.scheduler import (
        VGPUResourceEstimator,
        ModelConfig,
        MemoryEstimate,
        VGPUConfig,
        quick_estimate,
    )
except ImportError:
    pass

# 暂时保留旧导入方式（用于gradual migration）
try:
    from apt.vgpu.runtime.vb_integration import TORCH_AVAILABLE as VB_TORCH_AVAILABLE
except ImportError:
    VB_TORCH_AVAILABLE = False

# NPU/XPU/HPU后端适配器（统一多厂商加速器接口）
try:
    from apt.perf.optimization.npu_backend import (
        DeviceBackend,
        UnifiedDeviceManager,
        get_device_manager,
        get_unified_backend,
        # 设备检测
        is_cuda_available,
        is_npu_available,
        is_hpu_available,
        is_xpu_available,
        # 工具函数
        get_accelerator_type,
        get_all_accelerator_types,
    )
except ImportError:
    pass

# 云端NPU适配器（无需购买硬件，通过API调用远程NPU）
try:
    from apt.perf.optimization.cloud_npu_adapter import (
        CloudNPUBackend,
        HuaweiModelArtsNPU,
        CloudNPULinear,
        CloudNPUManager,
        get_cloud_npu_manager,
        enable_cloud_npu,
    )
except ImportError:
    pass

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

    # NPU/XPU/HPU后端适配器（多厂商加速器统一接口）
    'DeviceBackend',
    'UnifiedDeviceManager',
    'get_device_manager',
    'get_unified_backend',
    # 设备检测
    'is_cuda_available',
    'is_npu_available',
    'is_hpu_available',
    'is_xpu_available',
    # 工具函数
    'get_accelerator_type',
    'get_all_accelerator_types',

    # 云端NPU适配器（API调用远程NPU）
    'CloudNPUBackend',
    'HuaweiModelArtsNPU',
    'CloudNPULinear',
    'CloudNPUManager',
    'get_cloud_npu_manager',
    'enable_cloud_npu',

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
