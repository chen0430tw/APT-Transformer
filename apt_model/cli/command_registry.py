#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
命令注册系统 - 支持插件动态注册命令

该模块提供了一个集中的命令注册系统，允许：
1. 核心命令注册
2. 插件命令动态注册
3. 命令元数据管理（帮助文本、类别等）
4. 命令的自动发现和加载
"""

from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CommandMetadata:
    """命令元数据"""
    name: str
    func: Callable
    category: str = "general"
    help_text: str = ""
    args_help: Dict[str, str] = None
    aliases: List[str] = None
    is_placeholder: bool = False

    def __post_init__(self):
        if self.args_help is None:
            self.args_help = {}
        if self.aliases is None:
            self.aliases = []


class CommandRegistry:
    """
    命令注册中心

    管理所有CLI命令的注册、查找和执行。
    支持插件动态注册新命令。
    """

    def __init__(self):
        self._commands: Dict[str, CommandMetadata] = {}
        self._categories: Dict[str, List[str]] = {}
        self._aliases: Dict[str, str] = {}  # alias -> command_name

    def register(
        self,
        name: str,
        func: Callable,
        category: str = "general",
        help_text: str = "",
        args_help: Optional[Dict[str, str]] = None,
        aliases: Optional[List[str]] = None,
        is_placeholder: bool = False
    ) -> None:
        """
        注册一个命令

        参数:
            name: 命令名称
            func: 命令处理函数，接收args参数，返回int退出码
            category: 命令类别（用于帮助信息分组）
            help_text: 命令帮助文本
            args_help: 参数帮助字典
            aliases: 命令别名列表
            is_placeholder: 是否为占位符命令（尚未实现）
        """
        if name in self._commands:
            logger.warning(f"命令 '{name}' 已存在，将被覆盖")

        metadata = CommandMetadata(
            name=name,
            func=func,
            category=category,
            help_text=help_text,
            args_help=args_help or {},
            aliases=aliases or [],
            is_placeholder=is_placeholder
        )

        self._commands[name] = metadata

        # 更新类别索引
        if category not in self._categories:
            self._categories[category] = []
        if name not in self._categories[category]:
            self._categories[category].append(name)

        # 注册别名
        for alias in (aliases or []):
            self._aliases[alias] = name

        logger.debug(f"注册命令: {name} (类别: {category})")

    def unregister(self, name: str) -> bool:
        """
        注销一个命令

        参数:
            name: 命令名称

        返回:
            bool: 是否成功注销
        """
        if name not in self._commands:
            return False

        metadata = self._commands[name]

        # 移除别名
        for alias in metadata.aliases:
            if alias in self._aliases:
                del self._aliases[alias]

        # 从类别中移除
        if metadata.category in self._categories:
            if name in self._categories[metadata.category]:
                self._categories[metadata.category].remove(name)

        # 移除命令
        del self._commands[name]
        logger.debug(f"注销命令: {name}")
        return True

    def get_command(self, name: str) -> Optional[CommandMetadata]:
        """
        获取命令元数据

        参数:
            name: 命令名称或别名

        返回:
            CommandMetadata 或 None
        """
        # 检查别名
        if name in self._aliases:
            name = self._aliases[name]

        return self._commands.get(name)

    def execute_command(self, name: str, args: Any) -> int:
        """
        执行命令

        参数:
            name: 命令名称或别名
            args: 命令行参数

        返回:
            int: 退出码
        """
        metadata = self.get_command(name)

        if metadata is None:
            logger.error(f"未知命令: {name}")
            print(f"错误: 未知命令 '{name}'")
            print("使用 'help' 查看可用命令")
            return 1

        if metadata.is_placeholder:
            print(f"命令 '{name}' 尚未实现（占位符）")
            return 0

        try:
            return metadata.func(args)
        except Exception as e:
            logger.error(f"执行命令 '{name}' 时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            print(f"错误: 执行命令 '{name}' 失败: {e}")
            return 1

    def list_commands(
        self,
        category: Optional[str] = None,
        include_placeholders: bool = True
    ) -> List[str]:
        """
        列出命令

        参数:
            category: 仅列出指定类别的命令，None表示所有
            include_placeholders: 是否包含占位符命令

        返回:
            命令名称列表
        """
        commands = []

        for name, metadata in self._commands.items():
            if category and metadata.category != category:
                continue
            if not include_placeholders and metadata.is_placeholder:
                continue
            commands.append(name)

        return sorted(commands)

    def list_categories(self) -> List[str]:
        """
        列出所有命令类别

        返回:
            类别名称列表
        """
        return sorted(self._categories.keys())

    def get_commands_by_category(
        self,
        include_placeholders: bool = True
    ) -> Dict[str, List[CommandMetadata]]:
        """
        按类别获取所有命令

        参数:
            include_placeholders: 是否包含占位符命令

        返回:
            {category: [CommandMetadata, ...]}
        """
        result = {}

        for category, command_names in self._categories.items():
            result[category] = []
            for name in command_names:
                metadata = self._commands[name]
                if not include_placeholders and metadata.is_placeholder:
                    continue
                result[category].append(metadata)

        return result

    def has_command(self, name: str) -> bool:
        """
        检查命令是否存在

        参数:
            name: 命令名称或别名

        返回:
            bool: 命令是否存在
        """
        if name in self._aliases:
            name = self._aliases[name]
        return name in self._commands


# 全局命令注册中心单例
command_registry = CommandRegistry()


def register_command(*args, **kwargs) -> None:
    """
    快捷注册函数（使用全局注册中心）
    """
    command_registry.register(*args, **kwargs)


def get_command(name: str) -> Optional[CommandMetadata]:
    """
    快捷获取命令函数（使用全局注册中心）
    """
    return command_registry.get_command(name)


def execute_command(name: str, args: Any) -> int:
    """
    快捷执行命令函数（使用全局注册中心）
    """
    return command_registry.execute_command(name, args)
