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
from apt_model.console.plugin_bus import PluginBus, EventContext
from apt_model.console.eqi_manager import EQIManager
from apt_model.console.plugin_standards import PluginBase, PluginManifest
from apt_model.console.auto_loader import AutoPluginLoader
from apt_model.console.apx_loader import APXLoader
from apt_model.cli.command_registry import command_registry

logger = logging.getLogger(__name__)


class ConsoleCore:
    """
    控制台核心类

    整合：
    1. 模块管理器 - 管理所有核心模块
    2. 插件总线 - 管理插件和事件调度
    3. EQI 管理器 - 插件决策和资源优化
    4. 命令系统 - 注册和执行命令
    5. 配置管理 - 全局配置
    6. 生命周期管理 - 启动、运行、关闭
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

        # 初始化插件系统
        enable_eqi = self.config.get('plugins', {}).get('enable_eqi', False)
        default_timeout_ms = self.config.get('plugins', {}).get('default_timeout_ms', 100.0)
        engine_version = self.config.get('engine_version', '1.0.0')

        self.plugin_bus = PluginBus(
            enable_eqi=enable_eqi,
            default_timeout_ms=default_timeout_ms,
            engine_version=engine_version
        )

        # 初始化 EQI Manager（可选）
        self.eqi_manager = None
        if enable_eqi:
            eqi_config = self.config.get('plugins', {}).get('eqi', {})
            self.eqi_manager = EQIManager(
                default_time_budget_ms=eqi_config.get('time_budget_ms', 20.0),
                phi_gate=eqi_config.get('phi_gate', (2.0, 2.0, 1.0, 0.7)),
                kappa_stability=eqi_config.get('kappa_stability', 0.1)
            )
            self.plugin_bus.eqi_manager = self.eqi_manager

        # 新增：插件注册表（用于自动加载）
        self.plugin_registry: Dict[str, type] = {}

        # 新增：自动插件加载器
        self.auto_loader = AutoPluginLoader(self.plugin_registry)

        # 新增：APX加载器
        self.apx_loader = APXLoader()

        self._initialized = True
        self._running = False

        logger.info(f"Console Core initialized (engine version: {engine_version})")

    def start(self, auto_load_modules: bool = True, auto_load_plugins: bool = True) -> bool:
        """
        启动控制台

        Args:
            auto_load_modules: 是否自动加载模块
            auto_load_plugins: 是否自动加载和编译插件

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

            # 编译插件（静态冲突检查）
            if auto_load_plugins:
                logger.info("Compiling plugins...")
                try:
                    self.plugin_bus.compile(fail_fast=False)
                    logger.info("Plugin compilation completed")
                except Exception as e:
                    logger.error(f"Plugin compilation failed: {e}")
                    logger.exception(e)

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

        # 注册插件管理命令
        try:
            from apt_model.console.commands.plugin_commands import register_plugin_commands
            register_plugin_commands(self)
        except ImportError:
            logger.debug("Plugin commands not yet implemented")

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

    # ========================================================================
    # 插件管理方法
    # ========================================================================

    def register_plugin(self, plugin: PluginBase, manifest: Optional[PluginManifest] = None):
        """
        注册插件

        Args:
            plugin: 插件实例
            manifest: 插件清单（如果为 None，则从 plugin.get_manifest() 获取）
        """
        self.plugin_bus.register(plugin, manifest)
        logger.info(f"Plugin registered: {manifest.name if manifest else plugin.get_manifest().name}")

    def compile_plugins(self, fail_fast: bool = False):
        """
        编译插件（执行静态冲突检查）

        Args:
            fail_fast: 遇到错误是否立即失败
        """
        self.plugin_bus.compile(fail_fast=fail_fast)

    def emit_event(self, event: str, step: int, context_data: Optional[Dict[str, Any]] = None) -> EventContext:
        """
        派发事件到插件系统

        Args:
            event: 事件名称
            step: 当前步数
            context_data: 上下文数据

        Returns:
            事件上下文
        """
        return self.plugin_bus.emit(event, step, context_data)

    def get_plugin(self, name: str) -> Optional[PluginBase]:
        """
        获取插件实例

        Args:
            name: 插件名称

        Returns:
            插件实例，如果未注册则返回 None
        """
        return self.plugin_bus.get_plugin(name)

    def enable_plugin(self, name: str):
        """
        启用插件

        Args:
            name: 插件名称
        """
        self.plugin_bus.enable_plugin(name)
        logger.info(f"Plugin '{name}' enabled")

    def disable_plugin(self, name: str, reason: str = "manual"):
        """
        禁用插件

        Args:
            name: 插件名称
            reason: 禁用原因
        """
        self.plugin_bus.disable_plugin(name, reason)
        logger.info(f"Plugin '{name}' disabled: {reason}")

    def get_plugin_statistics(self) -> Dict[str, Any]:
        """
        获取插件统计信息

        Returns:
            统计信息字典
        """
        return self.plugin_bus.get_statistics()

    def print_plugin_status(self):
        """打印插件状态"""
        self.plugin_bus.print_status()

    # ========================================================================
    # 系统方法
    # ========================================================================

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

        # 插件统计
        plugin_stats = self.plugin_bus.get_statistics()
        print(f"Active Plugins: {plugin_stats['active_plugins']}")
        print(f"Total Plugins: {plugin_stats['total_plugins']}")

        print(f"Registered Commands: {len(self.command_registry._commands)}")
        print(f"EQI Enabled: {'Yes' if self.eqi_manager else 'No'}")
        print("="*80)

        # 打印模块状态
        self.module_manager.print_status()

        # 打印插件状态
        if plugin_stats['total_plugins'] > 0:
            self.plugin_bus.print_status()

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

    # ========================================================================
    # APX集成与自动插件加载（新增）
    # ========================================================================

    def register_plugin_class(self, name: str, plugin_class: type):
        """
        注册插件类到注册表

        Args:
            name: 插件名称
            plugin_class: 插件类
        """
        self.plugin_registry[name] = plugin_class
        logger.info(f"Registered plugin class: {name}")

    def load_apx_model(
        self,
        apx_path: Path,
        auto_configure_plugins: bool = True,
        score_threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """
        加载APX模型并自动配置插件

        Args:
            apx_path: APX文件路径
            auto_configure_plugins: 是否自动配置插件
            score_threshold: 插件相关性分数阈值

        Returns:
            APX信息字典

        Example:
            >>> core = ConsoleCore()
            >>> apx_info = core.load_apx_model(
            ...     Path("models/mixtral-moe.apx"),
            ...     auto_configure_plugins=True
            ... )
            >>> # 自动检测到moe能力，加载route_optimizer插件
        """
        if auto_configure_plugins:
            apx_info, plugins = self.apx_loader.load_with_auto_plugins(
                apx_path,
                self.auto_loader,
                auto_enable=True,
                score_threshold=score_threshold,
            )

            # 注册自动加载的插件
            for plugin in plugins:
                try:
                    self.register_plugin(plugin)
                    logger.info(f"Auto-registered plugin: {plugin.get_manifest().name}")
                except Exception as e:
                    logger.error(f"Failed to register auto-loaded plugin: {e}")

            logger.info(f"Auto-registered {len(plugins)} plugins for APX model")

            # 重新编译插件（包括新加载的）
            self.plugin_bus.compile(fail_fast=False)

        else:
            apx_info = self.apx_loader.load(apx_path)

        return apx_info

    def analyze_model_for_plugins(
        self,
        model_path: Path = None,
        capabilities: List[str] = None,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """
        分析模型并推荐插件（不加载）

        Args:
            model_path: 模型目录路径（与capabilities二选一）
            capabilities: 直接指定能力列表（与model_path二选一）
            dry_run: 是否仅分析（不实例化插件）

        Returns:
            分析结果

        Example:
            >>> analysis = core.analyze_model_for_plugins(
            ...     model_path=Path("models/mixtral-8x7b")
            ... )
            >>> print(analysis['available_plugins'])
            [{'name': 'route_optimizer', 'score': 1.0}]
        """
        if capabilities is None:
            if model_path is None:
                raise ValueError("Either model_path or capabilities must be provided")

            # 从模型目录检测能力
            try:
                from apt_model.tools.apx.detectors import detect_capabilities
                capabilities = detect_capabilities(model_path)
            except Exception as e:
                logger.error(f"Failed to detect capabilities: {e}")
                capabilities = []

        return self.auto_loader.analyze_model(capabilities)

    def get_plugin_recommendations(
        self,
        capabilities: List[str],
        format: str = "text"
    ) -> str:
        """
        获取插件推荐报告

        Args:
            capabilities: 模型能力列表
            format: 输出格式 ("text" 或 "json")

        Returns:
            格式化的推荐报告

        Example:
            >>> report = core.get_plugin_recommendations(["moe", "tva"])
            >>> print(report)
            ============================================================
            Plugin Recommendation Report
            ============================================================
            Model Capabilities: moe, tva
            ...
        """
        return self.auto_loader.get_recommendations_report(capabilities, format=format)


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
