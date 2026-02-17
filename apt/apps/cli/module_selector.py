#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module Selector for APT Model

动态模块选择和管理
"""

import os
from pathlib import Path
from typing import Dict, List, Set, Optional
import importlib.util


class ModuleSelector:
    """模块选择器 - 管理 L0/L1/L2/L3 模块和插件的启用/禁用"""

    def __init__(self):
        """初始化模块选择器"""
        self.root_dir = Path(__file__).parent.parent.parent.parent
        self.apt_dir = self.root_dir / "apt"
        self.plugins_dir = self.root_dir / "apt" / "apps" / "plugins"

        # 定义所有可用模块
        self.available_modules = {
            # 核心层级
            'L0': {
                'name': 'L0 (Kernel)',
                'desc': '核心APT算法和基础架构',
                'path': 'apt/core',
                'essential': True,
            },
            'L1': {
                'name': 'L1 (Performance)',
                'desc': '性能优化和加速',
                'path': 'apt/perf',
                'essential': False,
            },
            'L2': {
                'name': 'L2 (Memory)',
                'desc': '记忆和知识系统',
                'path': 'apt/memory',
                'essential': False,
            },
            'L3': {
                'name': 'L3 (Product)',
                'desc': '产品和应用层',
                'path': 'apt/apps',
                'essential': False,
            },

            # 插件类别
            'monitoring': {
                'name': 'Monitoring Plugins',
                'desc': '监控和诊断插件',
                'path': 'apt/apps/plugins/monitoring',
                'essential': False,
            },
            'visualization': {
                'name': 'Visualization Plugins',
                'desc': '可视化插件',
                'path': 'apt/apps/plugins/visualization',
                'essential': False,
            },
            'evaluation': {
                'name': 'Evaluation Plugins',
                'desc': '评估和基准测试插件',
                'path': 'apt/apps/plugins/evaluation',
                'essential': False,
            },
            'infrastructure': {
                'name': 'Infrastructure Plugins',
                'desc': '基础设施插件',
                'path': 'apt/apps/plugins/infrastructure',
                'essential': False,
            },
            'optimization': {
                'name': 'Optimization Plugins',
                'desc': '性能优化插件',
                'path': 'apt/apps/plugins/optimization',
                'essential': False,
            },
            'rl': {
                'name': 'RL Plugins',
                'desc': '强化学习插件',
                'path': 'apt/apps/plugins/rl',
                'essential': False,
            },
            'protocol': {
                'name': 'Protocol Plugins',
                'desc': '协议集成插件',
                'path': 'apt/apps/plugins/protocol',
                'essential': False,
            },
            'retrieval': {
                'name': 'Retrieval Plugins',
                'desc': '检索增强插件',
                'path': 'apt/apps/plugins/retrieval',
                'essential': False,
            },
            'hardware': {
                'name': 'Hardware Plugins',
                'desc': '硬件模拟和适配插件',
                'path': 'apt/apps/plugins/hardware',
                'essential': False,
            },
            'deployment': {
                'name': 'Deployment Plugins',
                'desc': '部署和虚拟化插件',
                'path': 'apt/apps/plugins/deployment',
                'essential': False,
            },
            'memory': {
                'name': 'Memory Plugins',
                'desc': '高级记忆系统插件',
                'path': 'apt/apps/plugins/memory',
                'essential': False,
            },
            'experimental': {
                'name': 'Experimental Plugins',
                'desc': '实验性插件',
                'path': 'apt/apps/plugins/experimental',
                'essential': False,
            },
            'core_plugins': {
                'name': 'Core Plugins',
                'desc': '核心插件',
                'path': 'apt/apps/plugins/core',
                'essential': False,
            },
            'integration': {
                'name': 'Integration Plugins',
                'desc': '集成插件',
                'path': 'apt/apps/plugins/integration',
                'essential': False,
            },
            'distillation': {
                'name': 'Distillation Plugins',
                'desc': '蒸馏套件插件',
                'path': 'apt/apps/plugins/distillation',
                'essential': False,
            },
        }

        # 默认启用的模块
        self.default_enabled = {'L0', 'L1', 'L2', 'L3'}

    def list_all_modules(self) -> Dict[str, Dict]:
        """
        列出所有可用模块

        Returns:
            模块信息字典
        """
        result = {}
        for module_id, module_info in self.available_modules.items():
            exists = (self.root_dir / module_info['path']).exists()
            result[module_id] = {
                **module_info,
                'exists': exists,
                'enabled_by_default': module_id in self.default_enabled,
            }
        return result

    def parse_module_list(self, module_string: Optional[str]) -> Set[str]:
        """
        解析模块列表字符串

        Args:
            module_string: 逗号分隔的模块列表，如 "L0,L1,monitoring"

        Returns:
            模块ID集合
        """
        if not module_string:
            return set()

        modules = set()
        for module in module_string.split(','):
            module = module.strip()
            if module in self.available_modules:
                modules.add(module)
            else:
                print(f"Warning: Unknown module '{module}' (ignored)")

        return modules

    def get_enabled_modules(
        self,
        enable_modules: Optional[str] = None,
        disable_modules: Optional[str] = None
    ) -> Set[str]:
        """
        获取最终启用的模块列表

        Args:
            enable_modules: 要启用的模块（逗号分隔）
            disable_modules: 要禁用的模块（逗号分隔）

        Returns:
            启用的模块ID集合
        """
        # 从默认启用开始
        enabled = self.default_enabled.copy()

        # 添加显式启用的模块
        if enable_modules:
            to_enable = self.parse_module_list(enable_modules)
            enabled.update(to_enable)

        # 移除显式禁用的模块
        if disable_modules:
            to_disable = self.parse_module_list(disable_modules)
            enabled -= to_disable

        # 确保必需模块总是启用
        for module_id, module_info in self.available_modules.items():
            if module_info.get('essential', False):
                enabled.add(module_id)

        return enabled

    def print_module_status(
        self,
        enabled_modules: Optional[Set[str]] = None
    ):
        """
        打印模块状态

        Args:
            enabled_modules: 启用的模块集合（可选）
        """
        if enabled_modules is None:
            enabled_modules = self.default_enabled

        print("=" * 80)
        print("APT-Transformer Module Status")
        print("=" * 80)
        print()

        # 按类别分组显示
        categories = {
            'Core Layers (L0-L3)': ['L0', 'L1', 'L2', 'L3'],
            'Monitoring & Evaluation': ['monitoring', 'visualization', 'evaluation'],
            'Infrastructure': ['infrastructure', 'core_plugins'],
            'Training & Optimization': ['optimization', 'rl'],
            'Integration': ['integration', 'protocol', 'retrieval'],
            'Advanced Features': ['hardware', 'deployment', 'memory'],
            'Experimental': ['experimental', 'distillation'],
        }

        for category_name, module_ids in categories.items():
            print(f"\n{category_name}:")
            print("-" * 80)

            for module_id in module_ids:
                if module_id not in self.available_modules:
                    continue

                module_info = self.available_modules[module_id]
                exists = (self.root_dir / module_info['path']).exists()
                is_enabled = module_id in enabled_modules
                is_essential = module_info.get('essential', False)

                # 状态标记
                status = "✅" if is_enabled else "❌"
                essential_mark = " [ESSENTIAL]" if is_essential else ""
                exists_mark = "" if exists else " [NOT FOUND]"

                print(
                    f"  {status} {module_id:20s} - {module_info['name']}"
                    f"{essential_mark}{exists_mark}"
                )
                print(f"      {module_info['desc']}")

        print()
        print("=" * 80)
        print(f"Total Modules: {len(self.available_modules)}")
        print(f"Enabled: {len(enabled_modules)}")
        print(f"Disabled: {len(self.available_modules) - len(enabled_modules)}")
        print("=" * 80)

    def apply_module_selection(
        self,
        config: Dict,
        enabled_modules: Set[str]
    ) -> Dict:
        """
        应用模块选择到配置

        Args:
            config: 配置字典
            enabled_modules: 启用的模块集合

        Returns:
            更新后的配置
        """
        config['enabled_modules'] = list(enabled_modules)

        # 根据模块启用状态调整配置
        # 例如，如果禁用了 L1，则不使用性能优化
        if 'L1' not in enabled_modules:
            config['use_performance_optimization'] = False

        # 如果禁用了监控插件，则不启用监控
        if 'monitoring' not in enabled_modules:
            config['enable_monitoring'] = False

        return config


# 全局实例
module_selector = ModuleSelector()


if __name__ == "__main__":
    # 测试
    selector = ModuleSelector()

    print("1. List all modules:")
    all_modules = selector.list_all_modules()
    print(f"Total: {len(all_modules)} modules\n")

    print("2. Default module status:")
    selector.print_module_status()

    print("\n3. Custom module selection:")
    enabled = selector.get_enabled_modules(
        enable_modules="L0,L1,monitoring,rl",
        disable_modules="L3,experimental"
    )
    selector.print_module_status(enabled)
