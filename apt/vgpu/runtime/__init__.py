#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Virtual GPU Runtime (Virtual Blackwell)

虚拟GPU运行时环境：
- VirtualGPU: 虚拟GPU抽象
- GPU context management
- Device initialization
- Runtime configuration
- Virtual Blackwell adapter
- VGPU stack technology
"""

# Virtual Blackwell Core
try:
    from apt.vgpu.runtime.virtual_blackwell_adapter import (
        VirtualBlackwellAdapter,
        create_virtual_blackwell,
    )
except (ImportError, OSError):
    VirtualBlackwellAdapter = None
    create_virtual_blackwell = None
try:
    from apt.vgpu.runtime.vb_global import VBGlobalConfig
except (ImportError, OSError):
    VBGlobalConfig = None
try:
    from apt.vgpu.runtime.vb_integration import (
        VBOptimizedLinear,
        VBModelWrapper,
        enable_vb_optimization,
    )
except (ImportError, OSError):
    VBOptimizedLinear = None
    VBModelWrapper = None
    enable_vb_optimization = None
try:
    from apt.vgpu.runtime.vb_autopatch import VBAutoPatcher
except (ImportError, OSError):
    VBAutoPatcher = None

# VGPU Stack Technology
try:
    from apt.vgpu.runtime.vgpu_stack import (
        VGPUStack,
        VGPULevel,
        VGPUStackLinear,
        create_vgpu_stack,
    )
except (ImportError, OSError):
    VGPUStack = None
    VGPULevel = None
    VGPUStackLinear = None
    create_vgpu_stack = None

# Random Projection Kernel (CompAct Optimization)
try:
    from apt.vgpu.runtime.random_projection_kernel import (
        RandomProjectionKernel,
        ProjectionKernelConfig,
        CompActLinear,
        compact_act_forward,
        estimate_memory,
    )
except (ImportError, OSError):
    RandomProjectionKernel = None
    ProjectionKernelConfig = None
    CompActLinear = None
    compact_act_forward = None
    estimate_memory = None

try:
    from apt.vgpu.runtime.vb_compact_integration import (
        VBCompActManager,
        VBCompActConfig,
        VBCompActLinear,
        replace_linear_with_vb_compact,
    )
except (ImportError, OSError):
    VBCompActManager = None
    VBCompActConfig = None
    VBCompActLinear = None
    replace_linear_with_vb_compact = None

# Random Projection Kernel - Tiered Storage
try:
    from apt.vgpu.runtime.compact_tiered_storage import (
        TieredProjectionKernel,
        TieredStorageConfig,
        estimate_tiered_memory,
    )
except (ImportError, OSError):
    TieredProjectionKernel = None
    TieredStorageConfig = None
    estimate_tiered_memory = None

# LECAC — 激活值量化存储（INT2 默认）
try:
    from apt.vgpu.runtime.lecac import (
        quantize_int2_symmetric,
        dequantize_int2_symmetric,
        quantize_int4_symmetric,
        dequantize_int4_symmetric,
        LECACLinearFunction,
        OrthogonalLECACLinearFunction,
        LECACConfig,
        LECACLinear,
        replace_linear_with_lecac,
    )
except (ImportError, OSError):
    quantize_int2_symmetric = None
    dequantize_int2_symmetric = None
    quantize_int4_symmetric = None
    dequantize_int4_symmetric = None
    LECACLinearFunction = None
    OrthogonalLECACLinearFunction = None
    LECACConfig = None
    LECACLinear = None
    replace_linear_with_lecac = None

__all__ = [
    # Virtual Blackwell
    'VirtualBlackwellAdapter',
    'create_virtual_blackwell',
    'VBGlobalConfig',
    'VBOptimizedLinear',
    'VBModelWrapper',
    'enable_vb_optimization',
    'VBAutoPatcher',
    # VGPU Stack
    'VGPUStack',
    'VGPULevel',
    'VGPUStackLinear',
    'create_vgpu_stack',
    # Random Projection Kernel (CompAct)
    'RandomProjectionKernel',
    'ProjectionKernelConfig',
    'CompActLinear',
    'compact_act_forward',
    'estimate_memory',
    'VBCompActManager',
    'VBCompActConfig',
    'VBCompActLinear',
    'replace_linear_with_vb_compact',
    # Tiered Storage
    'TieredProjectionKernel',
    'TieredStorageConfig',
    'estimate_tiered_memory',
    # LECAC (INT2 默认激活值量化)
    'quantize_int2_symmetric',
    'dequantize_int2_symmetric',
    'quantize_int4_symmetric',
    'dequantize_int4_symmetric',
    'LECACLinearFunction',
    'OrthogonalLECACLinearFunction',
    'LECACConfig',
    'LECACLinear',
    'replace_linear_with_lecac',
]
