#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model (自生成变换器) CLI Module
Command-line interface for APT model training and evaluation tool

重构后的CLI模块：
- 使用命令注册系统
- 支持插件动态添加命令
- 清晰的公共API
"""

try:
    from apt.apps.cli.parser import parse_arguments, get_available_commands
except ImportError:
    pass
try:
    from apt.apps.cli.command_registry import (
        CommandRegistry,
        CommandMetadata,
        command_registry,
        register_command,
        get_command,
        execute_command
    )
except ImportError:
    pass

# 导入命令（自动注册到命令注册中心）
try:
    import apt.apps.cli.commands  # noqa: F401
except ImportError:
    pass

# Import APX commands and register them
try:
    from apt.apps.cli.apx_commands import register_apx_commands
    register_apx_commands()
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"Failed to import APX commands: {e}")

__all__ = [
    # 参数解析
    'parse_arguments',
    'get_available_commands',

    # 命令注册系统
    'CommandRegistry',
    'CommandMetadata',
    'command_registry',
    'register_command',
    'get_command',
    'execute_command',

    # 插件支持
    'register_plugin_command',
]


def register_plugin_command(
    name: str,
    func,
    category: str = "plugin",
    help_text: str = "",
    aliases=None,
    is_placeholder: bool = False
):
    """
    插件命令注册便捷函数

    这是为插件开发者提供的简化API，用于注册新命令。

    参数:
        name: 命令名称
        func: 命令处理函数，接收args参数，返回int退出码
        category: 命令类别（默认: plugin）
        help_text: 命令帮助文本
        aliases: 命令别名列表
        is_placeholder: 是否为占位符命令

    示例:
        >>> def my_custom_command(args):
        ...     print("执行自定义命令")
        ...     return 0
        ...
        >>> register_plugin_command(
        ...     "my-cmd",
        ...     my_custom_command,
        ...     category="custom",
        ...     help_text="我的自定义命令",
        ...     aliases=["mycmd"]
        ... )
    """
    register_command(
        name=name,
        func=func,
        category=category,
        help_text=help_text,
        aliases=aliases or [],
        is_placeholder=is_placeholder
    )


# 示例：插件如何注册命令
# 插件开发者可以在他们的插件模块中这样做：
#
# from apt.apps.cli import register_plugin_command
#
# def my_plugin_command(args):
#     """插件命令实现"""
#     print("执行插件命令")
#     return 0
#
# # 注册命令
# register_plugin_command(
#     "my-plugin",
#     my_plugin_command,
#     category="plugins",
#     help_text="我的插件命令"
# )
