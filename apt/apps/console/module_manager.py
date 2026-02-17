#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module Manager (模块管理器)

负责管理所有 APT 核心模块的注册、初始化、生命周期和通信。
"""

import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import importlib

logger = logging.getLogger(__name__)


class ModuleStatus(Enum):
    """模块状态"""
    UNLOADED = "unloaded"      # 未加载
    LOADING = "loading"         # 加载中
    LOADED = "loaded"           # 已加载
    INITIALIZING = "initializing"  # 初始化中
    READY = "ready"             # 就绪
    ERROR = "error"             # 错误
    DISABLED = "disabled"       # 已禁用


@dataclass
class ModuleMetadata:
    """模块元数据"""
    name: str                          # 模块名称
    version: str                       # 版本号
    description: str                   # 描述
    module_path: str                   # 模块路径（Python导入路径）
    dependencies: List[str] = field(default_factory=list)  # 依赖模块列表
    optional_dependencies: List[str] = field(default_factory=list)  # 可选依赖
    init_function: Optional[str] = None  # 初始化函数名
    config_key: Optional[str] = None   # 配置键名
    category: str = "core"             # 类别: core, plugin, runtime, etc.
    enabled: bool = True               # 是否启用
    auto_load: bool = True             # 是否自动加载


class ModuleManager:
    """
    模块管理器

    负责：
    1. 模块注册与发现
    2. 依赖解析
    3. 模块加载与初始化
    4. 生命周期管理
    5. 模块间通信
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化模块管理器

        Args:
            config: 全局配置字典
        """
        self.config = config or {}
        self.modules: Dict[str, ModuleMetadata] = {}  # 模块元数据
        self.instances: Dict[str, Any] = {}           # 模块实例
        self.status: Dict[str, ModuleStatus] = {}     # 模块状态
        self.init_order: List[str] = []               # 初始化顺序

        # 注册核心模块
        self._register_core_modules()

    def _register_core_modules(self):
        """注册所有核心模块"""

        # VFT/TVA - Vein-Flow Transformer
        self.register_module(ModuleMetadata(
            name="vft_tva",
            version="1.0.0",
            description="Vein-Flow Transformer / Tri-Vein Attention",
            module_path="vft_tva",
            dependencies=[],
            category="core",
            auto_load=False,
        ))

        # EQI - Evidence Qualitative Inference
        self.register_module(ModuleMetadata(
            name="eqi",
            version="1.0.0",
            description="Evidence Qualitative Inference Plugin",
            module_path="eqi",
            dependencies=[],
            category="core",
            auto_load=False,
        ))

        # Reasoning Controller
        self.register_module(ModuleMetadata(
            name="reasoning",
            version="1.0.0",
            description="Multi-step Reasoning Controller",
            module_path="apt_model.runtime.decoder.reasoning_controller",
            dependencies=["vft_tva"],
            category="runtime",
            auto_load=False,
        ))

        # Codec System
        self.register_module(ModuleMetadata(
            name="codec",
            version="1.0.0",
            description="Codec System for Multiple Languages",
            module_path="apt_model.codecs",
            dependencies=[],
            category="core",
            auto_load=True,
        ))

        # Plugin System
        self.register_module(ModuleMetadata(
            name="plugins",
            version="1.0.0",
            description="Plugin Management System",
            module_path="apt.plugins",
            dependencies=[],
            category="core",
            auto_load=True,
        ))

        # Multilingual Support
        self.register_module(ModuleMetadata(
            name="multilingual",
            version="1.0.0",
            description="Multilingual Support System",
            module_path="apt.multilingual",
            dependencies=["codec"],
            category="core",
            auto_load=True,
        ))

        # Core Registry
        self.register_module(ModuleMetadata(
            name="registry",
            version="1.0.0",
            description="Core Provider Registry System",
            module_path="apt.core.registry",
            dependencies=[],
            category="core",
            auto_load=True,
        ))

        # Training System
        self.register_module(ModuleMetadata(
            name="training",
            version="1.0.0",
            description="Model Training System with Callbacks",
            module_path="apt_model.training",
            dependencies=["codec", "multilingual"],
            category="core",
            auto_load=False,
        ))

        # RAG System
        self.register_module(ModuleMetadata(
            name="rag",
            version="1.0.0",
            description="Retrieval Augmented Generation System",
            module_path="apt.core.providers.retrieval",
            dependencies=["registry"],
            category="core",
            auto_load=False,
        ))

        # Hardware & Resources
        self.register_module(ModuleMetadata(
            name="hardware",
            version="1.0.0",
            description="Hardware Detection and Resource Management",
            module_path="apt.core.hardware",
            dependencies=[],
            category="infrastructure",
            auto_load=True,
        ))

    def register_module(self, metadata: ModuleMetadata):
        """
        注册模块

        Args:
            metadata: 模块元数据
        """
        if metadata.name in self.modules:
            logger.warning(f"Module '{metadata.name}' already registered, overwriting...")

        self.modules[metadata.name] = metadata
        self.status[metadata.name] = ModuleStatus.UNLOADED
        logger.info(f"Registered module: {metadata.name} v{metadata.version}")

    def get_module_info(self, name: str) -> Optional[ModuleMetadata]:
        """获取模块信息"""
        return self.modules.get(name)

    def list_modules(self, category: Optional[str] = None,
                    status: Optional[ModuleStatus] = None) -> List[ModuleMetadata]:
        """
        列出模块

        Args:
            category: 按类别筛选
            status: 按状态筛选

        Returns:
            模块元数据列表
        """
        result = []
        for name, metadata in self.modules.items():
            if category and metadata.category != category:
                continue
            if status and self.status[name] != status:
                continue
            result.append(metadata)
        return result

    def resolve_dependencies(self, module_name: str) -> List[str]:
        """
        解析模块依赖（拓扑排序）

        Args:
            module_name: 模块名称

        Returns:
            按依赖顺序排列的模块列表
        """
        if module_name not in self.modules:
            raise ValueError(f"Module '{module_name}' not found")

        visited = set()
        order = []

        def visit(name: str):
            if name in visited:
                return
            if name not in self.modules:
                logger.warning(f"Dependency '{name}' not found, skipping...")
                return

            visited.add(name)
            metadata = self.modules[name]

            # 先访问依赖
            for dep in metadata.dependencies:
                visit(dep)

            # 再添加当前模块
            order.append(name)

        visit(module_name)
        return order

    def load_module(self, name: str, force_reload: bool = False) -> bool:
        """
        加载模块

        Args:
            name: 模块名称
            force_reload: 是否强制重新加载

        Returns:
            是否加载成功
        """
        if name not in self.modules:
            logger.error(f"Module '{name}' not registered")
            return False

        metadata = self.modules[name]

        # 检查是否已加载
        if not force_reload and self.status[name] in [ModuleStatus.LOADED, ModuleStatus.READY]:
            logger.info(f"Module '{name}' already loaded")
            return True

        # 检查是否禁用
        if not metadata.enabled:
            logger.info(f"Module '{name}' is disabled")
            self.status[name] = ModuleStatus.DISABLED
            return False

        try:
            # 更新状态
            self.status[name] = ModuleStatus.LOADING

            # 加载依赖
            for dep in metadata.dependencies:
                if not self.load_module(dep):
                    logger.error(f"Failed to load dependency '{dep}' for module '{name}'")
                    self.status[name] = ModuleStatus.ERROR
                    return False

            # 导入模块
            logger.info(f"Loading module: {name} from {metadata.module_path}")
            module = importlib.import_module(metadata.module_path)
            self.instances[name] = module

            # 调用初始化函数（如果有）
            if metadata.init_function:
                init_func = getattr(module, metadata.init_function, None)
                if init_func:
                    self.status[name] = ModuleStatus.INITIALIZING
                    init_func(self.config.get(metadata.config_key, {}))

            # 更新状态
            self.status[name] = ModuleStatus.READY
            logger.info(f"Module '{name}' loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load module '{name}': {e}")
            logger.exception(e)
            self.status[name] = ModuleStatus.ERROR
            return False

    def load_all(self, auto_only: bool = True) -> Dict[str, bool]:
        """
        加载所有模块

        Args:
            auto_only: 是否只加载 auto_load=True 的模块

        Returns:
            模块加载结果字典 {module_name: success}
        """
        results = {}
        for name, metadata in self.modules.items():
            if auto_only and not metadata.auto_load:
                continue
            results[name] = self.load_module(name)
        return results

    def enable_module(self, name: str):
        """启用模块"""
        if name in self.modules:
            self.modules[name].enabled = True
            logger.info(f"Module '{name}' enabled")

    def disable_module(self, name: str):
        """禁用模块"""
        if name in self.modules:
            self.modules[name].enabled = False
            self.status[name] = ModuleStatus.DISABLED
            logger.info(f"Module '{name}' disabled")

    def get_instance(self, name: str) -> Optional[Any]:
        """获取模块实例"""
        return self.instances.get(name)

    def get_status(self, name: str) -> Optional[ModuleStatus]:
        """获取模块状态"""
        return self.status.get(name)

    def print_status(self):
        """打印所有模块状态"""
        print("\n" + "="*80)
        print(" APT Module Status")
        print("="*80)
        print(f"{'Module':<20} {'Version':<10} {'Status':<15} {'Category':<15}")
        print("-"*80)

        for name in sorted(self.modules.keys()):
            metadata = self.modules[name]
            status = self.status[name]
            status_str = status.value.upper()

            # 根据状态添加颜色标记
            if status == ModuleStatus.READY:
                status_str = f"✓ {status_str}"
            elif status == ModuleStatus.ERROR:
                status_str = f"✗ {status_str}"
            elif status == ModuleStatus.DISABLED:
                status_str = f"○ {status_str}"

            print(f"{name:<20} {metadata.version:<10} {status_str:<15} {metadata.category:<15}")

        print("="*80 + "\n")
