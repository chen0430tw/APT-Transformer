#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT - Autopoietic Transformer

统一入口包，支持分层架构和按需加载：
- L0 (Kernel): 核心APT算法
- L1 (Performance): Virtual Blackwell性能优化栈
- L2 (Memory): AIM-Memory记忆系统
- L3 (Product): WebUI、API、插件、可观测性

使用示例：
    # 基础版（仅L0核心）
    import apt
    apt.enable('lite')

    # 标准版（L0 + L1性能）
    import apt
    apt.enable('standard')

    # 专业版（L0 + L2记忆）
    import apt
    apt.enable('pro')

    # 完整版（L0 + L1 + L2 + L3）
    import apt
    apt.enable('full')
"""

__version__ = "0.2.0"
__author__ = "APT Team"

import os
import warnings
from typing import Optional, List

# 后向兼容：保留旧的 registry 导入路径
try:
    try:
        from apt.core.registry import registry, Provider, register_provider, get_provider
    except ImportError:
        registry = None
        Provider = None
        register_provider = None
        get_provider = None
except ImportError:
    # 如果旧路径不存在，设置为 None，后续会在迁移时修复
    registry = None
    Provider = None
    register_provider = None
    get_provider = None

# 全局状态：已启用的层级
_enabled_layers: List[str] = []

# Profile 定义（与 profiles/*.yaml 对应）
PROFILES = {
    'lite': ['L0'],                    # 仅核心算法
    'standard': ['L0', 'L1'],          # 核心 + 性能
    'pro': ['L0', 'L2'],               # 核心 + 记忆
    'full': ['L0', 'L1', 'L2', 'L3'],  # 全功能
}

def enable(profile: str = 'lite', layers: Optional[List[str]] = None):
    """
    启用指定的发行版配置或自定义层级组合

    Args:
        profile: 发行版名称 ('lite'|'standard'|'pro'|'full')
        layers: 自定义层级列表，如 ['L0', 'L1']，会覆盖 profile

    Examples:
        >>> import apt
        >>> apt.enable('standard')  # 启用 L0 + L1
        >>> apt.enable(layers=['L0', 'L2'])  # 自定义启用 L0 + L2
    """
    global _enabled_layers

    # 确定要启用的层级
    if layers is not None:
        target_layers = layers
    elif profile in PROFILES:
        target_layers = PROFILES[profile]
    else:
        raise ValueError(
            f"未知的 profile '{profile}'。"
            f"可用选项: {list(PROFILES.keys())} 或使用 layers 参数自定义"
        )

    # 动态导入各层级模块
    for layer in target_layers:
        if layer in _enabled_layers:
            continue  # 已启用，跳过

        if layer == 'L0':
            # 导入核心算法模块
            try:
                try:
                    from apt.core import autopoietic, dbc_dac, left_spin_smooth
                except ImportError:
                    autopoietic = None
                    dbc_dac = None
                    left_spin_smooth = None
                _enabled_layers.append('L0')
                print(f"✓ L0 (Kernel) 已启用: Autopoietic Transform, DBC-DAC, Left-Spin Smooth")
            except ImportError as e:
                warnings.warn(f"L0 导入失败: {e}，某些核心功能可能不可用")

        elif layer == 'L1':
            # 导入性能优化栈
            try:
                try:
                    from apt.perf import virtual_blackwell, gpu_flash, vgpu_stack
                except ImportError:
                    virtual_blackwell = None
                    gpu_flash = None
                    vgpu_stack = None
                _enabled_layers.append('L1')
                print(f"✓ L1 (Performance) 已启用: Virtual Blackwell, GPU Flash, VGPU Stack")
            except ImportError as e:
                warnings.warn(f"L1 导入失败: {e}，性能优化功能不可用")

        elif layer == 'L2':
            # 导入记忆系统
            try:
                try:
                    from apt.memory import aim_memory, aim_nc, graph_rag
                except ImportError:
                    aim_memory = None
                    aim_nc = None
                    graph_rag = None
                _enabled_layers.append('L2')
                print(f"✓ L2 (Memory) 已启用: AIM-Memory, AIM-NC, GraphRAG")
            except ImportError as e:
                warnings.warn(f"L2 导入失败: {e}，记忆系统功能不可用")

        elif layer == 'L3':
            # 导入应用层
            try:
                try:
                    from apt.apps import webui, api, plugins, observability
                except ImportError:
                    webui = None
                    api = None
                    plugins = None
                    observability = None
                _enabled_layers.append('L3')
                print(f"✓ L3 (Product) 已启用: WebUI, API, 插件系统, 可观测性")
            except ImportError as e:
                warnings.warn(f"L3 导入失败: {e}，应用层功能不可用")

        else:
            warnings.warn(f"未知层级 '{layer}'，跳过")

    return _enabled_layers

def get_enabled_layers() -> List[str]:
    """返回当前已启用的层级列表"""
    return _enabled_layers.copy()

def is_layer_enabled(layer: str) -> bool:
    """检查指定层级是否已启用"""
    return layer in _enabled_layers

# 默认导出
__all__ = [
    'enable',
    'get_enabled_layers',
    'is_layer_enabled',
    'PROFILES',
    # 后向兼容导出
    'registry',
    'Provider',
    'register_provider',
    'get_provider',
]
