#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module Management Commands (模块管理命令)

提供模块管理相关的命令：
- modules list - 列出所有模块
- modules load <name> - 加载指定模块
- modules enable <name> - 启用模块
- modules disable <name> - 禁用模块
- modules info <name> - 显示模块信息
- modules status - 显示所有模块状态
"""

import logging
from typing import Any
from apt.apps.cli.command_registry import register_command
from apt.apps.console.module_manager import ModuleManager, ModuleStatus

logger = logging.getLogger(__name__)


def register_module_commands(module_manager: ModuleManager):
    """
    注册模块管理命令

    Args:
        module_manager: 模块管理器实例
    """

    # ========================================================================
    # modules list - 列出所有模块
    # ========================================================================
    @register_command(
        name="modules-list",
        aliases=["modules", "mod-list", "lsmod"],
        category="console",
        help_text="列出所有注册的模块"
    )
    def cmd_modules_list(args) -> int:
        """列出所有模块"""
        category = getattr(args, 'category', None)
        status_filter = getattr(args, 'status', None)

        # 转换状态字符串为枚举
        if status_filter:
            try:
                status_filter = ModuleStatus(status_filter.lower())
            except ValueError:
                print(f"错误: 无效的状态值 '{status_filter}'")
                print(f"可用状态: {', '.join([s.value for s in ModuleStatus])}")
                return 1

        modules = module_manager.list_modules(category=category, status=status_filter)

        if not modules:
            print("没有找到符合条件的模块")
            return 0

        print("\n" + "="*100)
        print(f" APT Modules {f'(Category: {category})' if category else ''} {f'(Status: {status_filter.value})' if status_filter else ''}")
        print("="*100)
        print(f"{'Name':<20} {'Version':<10} {'Category':<15} {'Status':<15} {'Dependencies':<30}")
        print("-"*100)

        for metadata in sorted(modules, key=lambda m: m.name):
            status = module_manager.get_status(metadata.name)
            status_str = status.value if status else "unknown"

            # 添加状态符号
            if status == ModuleStatus.READY:
                status_str = f"✓ {status_str}"
            elif status == ModuleStatus.ERROR:
                status_str = f"✗ {status_str}"
            elif status == ModuleStatus.DISABLED:
                status_str = f"○ {status_str}"

            deps = ", ".join(metadata.dependencies) if metadata.dependencies else "-"
            if len(deps) > 28:
                deps = deps[:25] + "..."

            print(f"{metadata.name:<20} {metadata.version:<10} {metadata.category:<15} {status_str:<15} {deps:<30}")

        print("="*100)
        print(f"Total: {len(modules)} module(s)")
        print()
        return 0

    # ========================================================================
    # modules load - 加载模块
    # ========================================================================
    @register_command(
        name="modules-load",
        aliases=["mod-load", "loadmod"],
        category="console",
        help_text="加载指定模块"
    )
    def cmd_modules_load(args) -> int:
        """加载模块"""
        module_name = getattr(args, 'module_name', None)
        if not module_name:
            print("错误: 请指定模块名称")
            print("用法: modules-load <module_name>")
            return 1

        force = getattr(args, 'force', False)

        print(f"正在加载模块: {module_name}")
        if module_manager.load_module(module_name, force_reload=force):
            print(f"✓ 模块 '{module_name}' 加载成功")
            return 0
        else:
            print(f"✗ 模块 '{module_name}' 加载失败")
            return 1

    # ========================================================================
    # modules enable - 启用模块
    # ========================================================================
    @register_command(
        name="modules-enable",
        aliases=["mod-enable"],
        category="console",
        help_text="启用指定模块"
    )
    def cmd_modules_enable(args) -> int:
        """启用模块"""
        module_name = getattr(args, 'module_name', None)
        if not module_name:
            print("错误: 请指定模块名称")
            return 1

        try:
            module_manager.enable_module(module_name)
            print(f"✓ 模块 '{module_name}' 已启用")
            return 0
        except Exception as e:
            print(f"✗ 启用模块失败: {e}")
            return 1

    # ========================================================================
    # modules disable - 禁用模块
    # ========================================================================
    @register_command(
        name="modules-disable",
        aliases=["mod-disable"],
        category="console",
        help_text="禁用指定模块"
    )
    def cmd_modules_disable(args) -> int:
        """禁用模块"""
        module_name = getattr(args, 'module_name', None)
        if not module_name:
            print("错误: 请指定模块名称")
            return 1

        try:
            module_manager.disable_module(module_name)
            print(f"✓ 模块 '{module_name}' 已禁用")
            return 0
        except Exception as e:
            print(f"✗ 禁用模块失败: {e}")
            return 1

    # ========================================================================
    # modules info - 显示模块信息
    # ========================================================================
    @register_command(
        name="modules-info",
        aliases=["mod-info", "modinfo"],
        category="console",
        help_text="显示模块详细信息"
    )
    def cmd_modules_info(args) -> int:
        """显示模块信息"""
        module_name = getattr(args, 'module_name', None)
        if not module_name:
            print("错误: 请指定模块名称")
            return 1

        metadata = module_manager.get_module_info(module_name)
        if not metadata:
            print(f"错误: 模块 '{module_name}' 不存在")
            return 1

        status = module_manager.get_status(module_name)

        print("\n" + "="*70)
        print(f" Module Information: {module_name}")
        print("="*70)
        print(f"Name:               {metadata.name}")
        print(f"Version:            {metadata.version}")
        print(f"Category:           {metadata.category}")
        print(f"Status:             {status.value if status else 'unknown'}")
        print(f"Enabled:            {'Yes' if metadata.enabled else 'No'}")
        print(f"Auto-load:          {'Yes' if metadata.auto_load else 'No'}")
        print(f"Module Path:        {metadata.module_path}")
        print(f"Description:        {metadata.description}")
        print(f"Dependencies:       {', '.join(metadata.dependencies) if metadata.dependencies else 'None'}")
        print(f"Optional Deps:      {', '.join(metadata.optional_dependencies) if metadata.optional_dependencies else 'None'}")
        if metadata.init_function:
            print(f"Init Function:      {metadata.init_function}")
        if metadata.config_key:
            print(f"Config Key:         {metadata.config_key}")
        print("="*70 + "\n")
        return 0

    # ========================================================================
    # modules status - 显示所有模块状态
    # ========================================================================
    @register_command(
        name="modules-status",
        aliases=["mod-status", "modstat"],
        category="console",
        help_text="显示所有模块的状态统计"
    )
    def cmd_modules_status(args) -> int:
        """显示模块状态统计"""
        module_manager.print_status()
        return 0
