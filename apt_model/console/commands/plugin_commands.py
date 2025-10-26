#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plugin Management Commands (插件管理命令)

提供插件管理相关的命令：
- plugins list - 列出所有插件
- plugins enable <name> - 启用插件
- plugins disable <name> - 禁用插件
- plugins info <name> - 显示插件信息
- plugins status - 显示所有插件状态
- plugins stats - 显示插件统计信息
"""

import logging
from typing import Any
from apt_model.cli.command_registry import register_command

logger = logging.getLogger(__name__)


def register_plugin_commands(console_core):
    """
    注册插件管理命令

    Args:
        console_core: 控制台核心实例
    """

    # ========================================================================
    # plugins list - 列出所有插件
    # ========================================================================
    @register_command(
        name="plugins-list",
        aliases=["plugins", "plugin-list", "lsplugin"],
        category="console",
        help_text="列出所有注册的插件"
    )
    def cmd_plugins_list(args) -> int:
        """列出所有插件"""
        console_core.print_plugin_status()
        return 0

    # ========================================================================
    # plugins enable - 启用插件
    # ========================================================================
    @register_command(
        name="plugins-enable",
        aliases=["plugin-enable"],
        category="console",
        help_text="启用指定插件"
    )
    def cmd_plugins_enable(args) -> int:
        """启用插件"""
        plugin_name = getattr(args, 'plugin_name', None)
        if not plugin_name:
            print("错误: 请指定插件名称")
            print("用法: plugins-enable <plugin_name>")
            return 1

        try:
            console_core.enable_plugin(plugin_name)
            print(f"✓ 插件 '{plugin_name}' 已启用")
            return 0
        except Exception as e:
            print(f"✗ 启用插件失败: {e}")
            return 1

    # ========================================================================
    # plugins disable - 禁用插件
    # ========================================================================
    @register_command(
        name="plugins-disable",
        aliases=["plugin-disable"],
        category="console",
        help_text="禁用指定插件"
    )
    def cmd_plugins_disable(args) -> int:
        """禁用插件"""
        plugin_name = getattr(args, 'plugin_name', None)
        if not plugin_name:
            print("错误: 请指定插件名称")
            print("用法: plugins-disable <plugin_name>")
            return 1

        reason = getattr(args, 'reason', 'manual')

        try:
            console_core.disable_plugin(plugin_name, reason=reason)
            print(f"✓ 插件 '{plugin_name}' 已禁用")
            return 0
        except Exception as e:
            print(f"✗ 禁用插件失败: {e}")
            return 1

    # ========================================================================
    # plugins info - 显示插件信息
    # ========================================================================
    @register_command(
        name="plugins-info",
        aliases=["plugin-info", "plugininfo"],
        category="console",
        help_text="显示插件详细信息"
    )
    def cmd_plugins_info(args) -> int:
        """显示插件信息"""
        plugin_name = getattr(args, 'plugin_name', None)
        if not plugin_name:
            print("错误: 请指定插件名称")
            print("用法: plugins-info <plugin_name>")
            return 1

        handle = console_core.plugin_bus.get_handle(plugin_name)
        if not handle:
            print(f"错误: 插件 '{plugin_name}' 不存在")
            return 1

        manifest = handle.manifest

        print("\n" + "="*80)
        print(f" Plugin Information: {plugin_name}")
        print("="*80)
        print(f"Name:               {manifest.name}")
        print(f"Version:            {manifest.version}")
        print(f"Priority:           {manifest.priority} ({manifest.get_priority_class()})")
        print(f"Status:             {'✓ ACTIVE' if handle.healthy else f'✗ {handle.disabled_reason or 'DISABLED'}'}")
        print(f"Blocking:           {'Yes' if manifest.blocking else 'No'}")
        print(f"Sandbox:            {'Yes' if manifest.sandbox else 'No'}")
        print(f"Description:        {manifest.description}")
        print(f"Author:             {manifest.author}")
        print(f"Events:             {', '.join(manifest.events)}")
        print(f"Capabilities:       {', '.join(manifest.capabilities) if manifest.capabilities else 'None'}")
        print(f"Requires:           {', '.join(manifest.requires) if manifest.requires else 'None'}")
        print(f"Conflicts:          {', '.join(manifest.conflicts) if manifest.conflicts else 'None'}")

        # 资源预算
        print(f"\nResource Budget:")
        print(f"  CPU:              {manifest.resources.get('cpu_ms', 0):.1f} ms")
        print(f"  GPU:              {manifest.resources.get('gpu_ms', 0):.1f} ms")
        print(f"  I/O:              {manifest.resources.get('io_mb', 0):.2f} MB")

        # 速率限制
        if manifest.rate_limit:
            print(f"\nRate Limit:")
            for key, value in manifest.rate_limit.items():
                print(f"  {key}:            {value}")

        # 运行时统计
        print(f"\nRuntime Statistics:")
        print(f"  Invocations:      {handle.total_invocations}")
        print(f"  Total Time:       {handle.total_time_ms:.2f} ms")
        avg_time = handle.total_time_ms / handle.total_invocations if handle.total_invocations > 0 else 0.0
        print(f"  Avg Time:         {avg_time:.2f} ms")
        print(f"  Fail Count:       {handle.fail_count}")
        print(f"  Last Step:        {handle.last_step_called if handle.last_step_called > -10**9 else 'Never'}")

        print("="*80 + "\n")
        return 0

    # ========================================================================
    # plugins status - 显示插件状态
    # ========================================================================
    @register_command(
        name="plugins-status",
        aliases=["plugin-status", "plugstat"],
        category="console",
        help_text="显示所有插件的状态"
    )
    def cmd_plugins_status(args) -> int:
        """显示插件状态"""
        console_core.print_plugin_status()
        return 0

    # ========================================================================
    # plugins stats - 显示插件统计信息
    # ========================================================================
    @register_command(
        name="plugins-stats",
        aliases=["plugin-stats"],
        category="console",
        help_text="显示插件统计信息"
    )
    def cmd_plugins_stats(args) -> int:
        """显示插件统计信息"""
        stats = console_core.get_plugin_statistics()

        print("\n" + "="*80)
        print(" Plugin Statistics")
        print("="*80)
        print(f"Total Plugins:      {stats['total_plugins']}")
        print(f"Active Plugins:     {stats['active_plugins']}")
        print(f"Disabled Plugins:   {stats['disabled_plugins']}")
        print(f"Total Invocations:  {stats['total_invocations']}")
        print(f"Total Time:         {stats['total_time_ms']:.2f} ms")
        print("="*80)

        if stats['plugins']:
            print(f"\n{'Plugin':<25} {'Invocations':<15} {'Total Time':<15} {'Avg Time':<15} {'Status':<15}")
            print("-"*80)
            for name, plugin_stats in sorted(stats['plugins'].items()):
                status = "✓ Active" if plugin_stats['healthy'] else f"✗ {plugin_stats['disabled_reason'] or 'Disabled'}"
                print(f"{name:<25} {plugin_stats['invocations']:<15} "
                      f"{plugin_stats['total_time_ms']:<15.2f} "
                      f"{plugin_stats['avg_time_ms']:<15.2f} "
                      f"{status:<15}")

        print()
        return 0

    # ========================================================================
    # plugins compile - 重新编译插件
    # ========================================================================
    @register_command(
        name="plugins-compile",
        aliases=["plugin-compile"],
        category="console",
        help_text="重新编译插件（执行静态冲突检查）"
    )
    def cmd_plugins_compile(args) -> int:
        """重新编译插件"""
        fail_fast = getattr(args, 'fail_fast', False)

        try:
            print("正在编译插件...")
            console_core.compile_plugins(fail_fast=fail_fast)
            print("✓ 插件编译完成")
            return 0
        except Exception as e:
            print(f"✗ 插件编译失败: {e}")
            import traceback
            traceback.print_exc()
            return 1
