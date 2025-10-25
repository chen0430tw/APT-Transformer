#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Console Core (控制台核心)

控制台核心负责整合模块管理、命令系统和启动器。
"""

import logging
import sys
from typing import Dict, Any, Optional, List
from pathlib import Path

from apt_model.console.module_manager import ModuleManager, ModuleStatus
from apt_model.cli.command_registry import command_registry

logger = logging.getLogger(__name__)


class ConsoleCore:
    """
    控制台核心类

    整合：
    1. 模块管理器 - 管理所有核心模块
    2. 命令系统 - 注册和执行命令
    3. 配置管理 - 全局配置
    4. 生命周期管理 - 启动、运行、关闭
    """

    _instance = None  # 单例

    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化控制台核心

        Args:
            config: 全局配置字典
        """
        # 避免重复初始化
        if hasattr(self, '_initialized'):
            return

        self.config = config or {}
        self.module_manager = ModuleManager(config=self.config)
        self.command_registry = command_registry
        self._initialized = True
        self._running = False

        logger.info("Console Core initialized")

    def start(self, auto_load_modules: bool = True) -> bool:
        """
        启动控制台

        Args:
            auto_load_modules: 是否自动加载模块

        Returns:
            是否启动成功
        """
        if self._running:
            logger.warning("Console Core is already running")
            return True

        try:
            logger.info("Starting Console Core...")

            # 加载模块
            if auto_load_modules:
                logger.info("Auto-loading modules...")
                results = self.module_manager.load_all(auto_only=True)

                # 检查加载结果
                failed = [name for name, success in results.items() if not success]
                if failed:
                    logger.warning(f"Failed to load modules: {', '.join(failed)}")

            # 注册控制台命令
            self._register_console_commands()

            self._running = True
            logger.info("Console Core started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start Console Core: {e}")
            logger.exception(e)
            return False

    def stop(self):
        """停止控制台"""
        if not self._running:
            return

        logger.info("Stopping Console Core...")
        self._running = False
        logger.info("Console Core stopped")

    def _register_console_commands(self):
        """注册控制台相关命令"""
        from apt_model.console.commands.module_commands import register_module_commands
        from apt_model.console.commands.system_commands import register_system_commands

        # 注册模块管理命令
        register_module_commands(self.module_manager)

        # 注册系统命令
        register_system_commands(self)

    def get_module(self, name: str) -> Optional[Any]:
        """
        获取模块实例

        Args:
            name: 模块名称

        Returns:
            模块实例，如果未加载则返回 None
        """
        return self.module_manager.get_instance(name)

    def load_module(self, name: str, force_reload: bool = False) -> bool:
        """
        加载模块

        Args:
            name: 模块名称
            force_reload: 是否强制重新加载

        Returns:
            是否加载成功
        """
        return self.module_manager.load_module(name, force_reload=force_reload)

    def enable_module(self, name: str) -> bool:
        """
        启用模块

        Args:
            name: 模块名称

        Returns:
            是否成功
        """
        try:
            self.module_manager.enable_module(name)
            return True
        except Exception as e:
            logger.error(f"Failed to enable module '{name}': {e}")
            return False

    def disable_module(self, name: str) -> bool:
        """
        禁用模块

        Args:
            name: 模块名称

        Returns:
            是否成功
        """
        try:
            self.module_manager.disable_module(name)
            return True
        except Exception as e:
            logger.error(f"Failed to disable module '{name}': {e}")
            return False

    def print_banner(self):
        """打印欢迎横幅"""
        banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║          APT Console - Autopoietic Transformer Console           ║
    ║                    自生成变换器控制台                              ║
    ║                                                                   ║
    ║  Version: 1.0.0                                                  ║
    ║  Unified module management and command system                     ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
        """
        print(banner)

    def print_status(self):
        """打印系统状态"""
        print("\n" + "="*80)
        print(" APT Console Status")
        print("="*80)
        print(f"Running: {'Yes' if self._running else 'No'}")
        print(f"Loaded Modules: {len([s for s in self.module_manager.status.values() if s == ModuleStatus.READY])}")
        print(f"Total Modules: {len(self.module_manager.modules)}")
        print(f"Registered Commands: {len(self.command_registry._commands)}")
        print("="*80)

        # 打印模块状态
        self.module_manager.print_status()

    def execute_command(self, command_name: str, *args, **kwargs) -> Any:
        """
        执行命令

        Args:
            command_name: 命令名称
            *args, **kwargs: 命令参数

        Returns:
            命令执行结果
        """
        from apt_model.cli.command_registry import execute_command
        return execute_command(command_name, *args, **kwargs)

    def list_commands(self, category: Optional[str] = None) -> List[str]:
        """
        列出可用命令

        Args:
            category: 按类别筛选

        Returns:
            命令名称列表
        """
        if category:
            return self.command_registry.get_commands_by_category().get(category, [])
        return list(self.command_registry._commands.keys())

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键（支持点号分隔的路径，如 "module.option"）
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default

    def set_config(self, key: str, value: Any):
        """
        设置配置值

        Args:
            key: 配置键（支持点号分隔的路径）
            value: 配置值
        """
        keys = key.split('.')
        config = self.config

        # 导航到最后一级之前的字典
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # 设置值
        config[keys[-1]] = value


# 全局控制台实例
_console = None


def get_console(config: Optional[Dict[str, Any]] = None) -> ConsoleCore:
    """
    获取全局控制台实例（单例）

    Args:
        config: 配置字典（仅首次调用时使用）

    Returns:
        控制台核心实例
    """
    global _console
    if _console is None:
        _console = ConsoleCore(config=config)
    return _console


def initialize_console(config: Optional[Dict[str, Any]] = None,
                      auto_start: bool = True) -> ConsoleCore:
    """
    初始化并启动控制台

    Args:
        config: 配置字典
        auto_start: 是否自动启动

    Returns:
        控制台核心实例
    """
    console = get_console(config=config)
    if auto_start:
        console.start()
    return console
