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
from apt.vgpu.runtime.virtual_blackwell_adapter import (
    VirtualBlackwellAdapter,
    create_virtual_blackwell,
)
from apt.vgpu.runtime.vb_global import VBGlobalConfig
from apt.vgpu.runtime.vb_integration import (
    VBOptimizedLinear,
    VBModelWrapper,
    enable_vb_optimization,
)
from apt.vgpu.runtime.vb_autopatch import VBAutoPatcher

# VGPU Stack Technology
from apt.vgpu.runtime.vgpu_stack import (
    VGPUStack,
    VGPULevel,
    VGPUStackLinear,
    create_vgpu_stack,
)

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
]
