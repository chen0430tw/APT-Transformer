#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
System Commands (系统命令)

提供系统相关的命令：
- console status - 显示控制台状态
- console version - 显示版本信息
- console config - 配置管理
"""

import logging
import sys
from typing import TYPE_CHECKING

from apt.apps.cli.command_registry import register_command

if TYPE_CHECKING:
    from apt.apps.console.core import ConsoleCore

logger = logging.getLogger(__name__)


def register_system_commands(console_core: 'ConsoleCore'):
    """
    注册系统命令

    Args:
        console_core: 控制台核心实例
    """

    # ========================================================================
    # console status - 显示控制台状态
    # ========================================================================
    @register_command(
        name="console-status",
        aliases=["status", "stat"],
        category="console",
        help_text="显示控制台状态信息"
    )
    def cmd_console_status(args) -> int:
        """显示控制台状态"""
        console_core.print_banner()
        console_core.print_status()
        return 0

    # ========================================================================
    # console version - 显示版本信息
    # ========================================================================
    @register_command(
        name="console-version",
        aliases=["version", "ver"],
        category="console",
        help_text="显示版本信息"
    )
    def cmd_console_version(args) -> int:
        """显示版本信息"""
        import apt_model
        import apt.apps.console

        print("\n" + "="*70)
        print(" APT Console Version Information")
        print("="*70)
        print(f"APT Model:          {apt_model.__version__}")
        print(f"Console Core:       {apt_model.console.__version__}")
        print(f"Python:             {sys.version.split()[0]}")
        print(f"Platform:           {sys.platform}")
        print("="*70 + "\n")
        return 0

    # ========================================================================
    # console config get - 获取配置
    # ========================================================================
    @register_command(
        name="console-config-get",
        aliases=["config-get", "cfg-get"],
        category="console",
        help_text="获取配置值"
    )
    def cmd_config_get(args) -> int:
        """获取配置值"""
        key = getattr(args, 'config_key', None)
        if not key:
            print("错误: 请指定配置键")
            print("用法: console-config-get <key>")
            return 1

        value = console_core.get_config(key)
        if value is None:
            print(f"配置键 '{key}' 不存在或值为 None")
            return 1

        print(f"{key} = {value}")
        return 0

    # ========================================================================
    # console config set - 设置配置
    # ========================================================================
    @register_command(
        name="console-config-set",
        aliases=["config-set", "cfg-set"],
        category="console",
        help_text="设置配置值"
    )
    def cmd_config_set(args) -> int:
        """设置配置值"""
        key = getattr(args, 'config_key', None)
        value = getattr(args, 'config_value', None)

        if not key or value is None:
            print("错误: 请指定配置键和值")
            print("用法: console-config-set <key> <value>")
            return 1

        try:
            console_core.set_config(key, value)
            print(f"✓ 配置已更新: {key} = {value}")
            return 0
        except Exception as e:
            print(f"✗ 设置配置失败: {e}")
            return 1

    # ========================================================================
    # console config list - 列出所有配置
    # ========================================================================
    @register_command(
        name="console-config-list",
        aliases=["config-list", "cfg-list"],
        category="console",
        help_text="列出所有配置"
    )
    def cmd_config_list(args) -> int:
        """列出所有配置"""
        def print_dict(d, indent=0):
            """递归打印字典"""
            for key, value in sorted(d.items()):
                if isinstance(value, dict):
                    print(" " * indent + f"{key}:")
                    print_dict(value, indent + 2)
                else:
                    print(" " * indent + f"{key}: {value}")

        print("\n" + "="*70)
        print(" Console Configuration")
        print("="*70)
        if console_core.config:
            print_dict(console_core.config)
        else:
            print("(无配置)")
        print("="*70 + "\n")
        return 0

    # ========================================================================
    # console commands - 列出所有命令
    # ========================================================================
    @register_command(
        name="console-commands",
        aliases=["commands", "cmd-list"],
        category="console",
        help_text="列出所有可用命令"
    )
    def cmd_console_commands(args) -> int:
        """列出所有命令"""
        category = getattr(args, 'category', None)

        commands_by_category = console_core.command_registry.get_commands_by_category(
            include_placeholders=False
        )

        print("\n" + "="*100)
        print(" APT Console Commands")
        print("="*100)

        for cat in sorted(commands_by_category.keys()):
            if category and cat != category:
                continue

            print(f"\n【{cat.upper()}】")
            print("-" * 100)

            for metadata in sorted(commands_by_category[cat], key=lambda m: m.name):
                aliases_str = f" (aliases: {', '.join(metadata.aliases)})" if metadata.aliases else ""
                help_text = metadata.help_text or "无说明"

                print(f"  {metadata.name:<30}{aliases_str:<40}")
                print(f"    {help_text}")

        print("\n" + "="*100 + "\n")
        return 0

    # ========================================================================
    # console help - 显示帮助
    # ========================================================================
    @register_command(
        name="console-help",
        aliases=["help", "h", "?"],
        category="console",
        help_text="显示帮助信息"
    )
    def cmd_console_help(args) -> int:
        """显示帮助信息"""
        command_name = getattr(args, 'command_name', None)

        if command_name:
            # 显示特定命令的帮助
            metadata = console_core.command_registry.get_command(command_name)
            if not metadata:
                print(f"错误: 命令 '{command_name}' 不存在")
                return 1

            print("\n" + "="*70)
            print(f" 命令帮助: {metadata.name}")
            print("="*70)
            print(f"名称:     {metadata.name}")
            if metadata.aliases:
                print(f"别名:     {', '.join(metadata.aliases)}")
            print(f"类别:     {metadata.category}")
            print(f"说明:     {metadata.help_text or '无'}")
            print("="*70 + "\n")
        else:
            # 显示通用帮助
            console_core.print_banner()
            print("\nAPT Console 是一个统一的控制台系统，用于管理所有 APT 核心模块。\n")
            print("常用命令:")
            print("  console-status                - 显示控制台状态")
            print("  modules-list                  - 列出所有模块")
            print("  modules-load <name>           - 加载模块")
            print("  console-commands              - 列出所有命令")
            print("  console-help <command>        - 显示命令帮助")
            print("\n使用 'console-commands' 查看完整命令列表")
            print("使用 'console-help <命令名>' 查看特定命令的帮助\n")

        return 0
