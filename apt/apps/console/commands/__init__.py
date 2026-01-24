#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Console Commands (控制台命令)

包含所有控制台相关的命令实现。
"""

try:
    from apt.apps.console.commands.module_commands import register_module_commands
except ImportError:
    register_module_commands = None
try:
    from apt.apps.console.commands.system_commands import register_system_commands
except ImportError:
    register_system_commands = None

__all__ = [
    'register_module_commands',
    'register_system_commands',
]
