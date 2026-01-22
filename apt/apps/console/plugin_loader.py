"""
插件加载器

动态加载和管理APG插件包
"""

import zipfile
import tempfile
import importlib.util
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Type
import yaml
import shutil
import logging

from apt.apps.console.plugin_standards import PluginBase

logger = logging.getLogger(__name__)


class PluginLoader:
    """APG插件加载器"""

    def __init__(self, plugin_dir: Optional[Path] = None):
        """
        Args:
            plugin_dir: 插件安装目录（None则使用默认 ~/.apt/plugins）
        """
        if plugin_dir is None:
            plugin_dir = Path.home() / ".apt" / "plugins"

        self.plugin_dir = plugin_dir
        self.plugin_dir.mkdir(parents=True, exist_ok=True)

        self._loaded_plugins: Dict[str, PluginBase] = {}
        self._loaded_modules: Dict[str, Any] = {}
        self._temp_dirs: List[Path] = []

        logger.debug(f"PluginLoader initialized with plugin_dir: {plugin_dir}")

    def install(
        self,
        apg_path: Path,
        force: bool = False,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """
        安装APG插件包

        Args:
            apg_path: APG文件路径
            force: 是否强制覆盖已安装插件
            validate: 是否验证插件包

        Returns:
            插件manifest字典

        Raises:
            ValueError: 如果安装失败
        """
        apg_path = Path(apg_path)

        if not apg_path.exists():
            raise ValueError(f"APG file not found: {apg_path}")

        # 1. 解压到临时目录并读取manifest
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)

            with zipfile.ZipFile(apg_path, 'r') as zf:
                zf.extractall(tmp_root)

            # 2. 读取manifest
            manifest_path = tmp_root / "plugin.yaml"
            if not manifest_path.exists():
                raise ValueError(f"Invalid APG: plugin.yaml not found in {apg_path}")

            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = yaml.safe_load(f)

            # 3. 验证manifest
            if validate:
                self._validate_manifest(manifest)

            # 4. 检查是否已安装
            plugin_name = manifest["name"]
            plugin_version = manifest["version"]
            plugin_install_dir = self.plugin_dir / plugin_name

            if plugin_install_dir.exists():
                if not force:
                    raise ValueError(
                        f"Plugin {plugin_name} already installed at {plugin_install_dir}. "
                        f"Use force=True to overwrite."
                    )
                else:
                    logger.warning(f"Overwriting existing plugin: {plugin_name}")
                    shutil.rmtree(plugin_install_dir)

            # 5. 安装到插件目录
            shutil.copytree(tmp_root, plugin_install_dir)

            logger.info(f"Plugin {plugin_name} v{plugin_version} installed to {plugin_install_dir}")

            return manifest

    def uninstall(self, plugin_name: str):
        """
        卸载插件

        Args:
            plugin_name: 插件名称

        Raises:
            ValueError: 如果插件未安装
        """
        plugin_install_dir = self.plugin_dir / plugin_name

        if not plugin_install_dir.exists():
            raise ValueError(f"Plugin {plugin_name} not installed")

        # 先卸载运行时插件
        if plugin_name in self._loaded_plugins:
            self.unload(plugin_name)

        # 删除插件目录
        shutil.rmtree(plugin_install_dir)
        logger.info(f"Plugin {plugin_name} uninstalled")

    def load(
        self,
        plugin_name: str,
        reload_if_loaded: bool = False,
    ) -> PluginBase:
        """
        加载已安装的插件

        Args:
            plugin_name: 插件名称
            reload_if_loaded: 如果已加载是否重新加载

        Returns:
            插件实例

        Raises:
            ValueError: 如果插件未安装或加载失败
        """
        # 如果已加载，直接返回
        if plugin_name in self._loaded_plugins:
            if not reload_if_loaded:
                logger.debug(f"Plugin {plugin_name} already loaded")
                return self._loaded_plugins[plugin_name]
            else:
                self.unload(plugin_name)

        # 1. 查找插件目录
        plugin_install_dir = self.plugin_dir / plugin_name
        if not plugin_install_dir.exists():
            raise ValueError(
                f"Plugin {plugin_name} not installed. "
                f"Install it first using loader.install()"
            )

        # 2. 读取manifest
        manifest_path = plugin_install_dir / "plugin.yaml"
        if not manifest_path.exists():
            raise ValueError(f"Invalid plugin: plugin.yaml not found in {plugin_install_dir}")

        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = yaml.safe_load(f)

        # 3. 动态导入插件模块
        plugin_module_path = plugin_install_dir / "plugin" / "__init__.py"
        if not plugin_module_path.exists():
            raise ValueError(f"Plugin module not found: {plugin_module_path}")

        module_name = f"apt_plugins.{plugin_name}"

        try:
            spec = importlib.util.spec_from_file_location(
                module_name,
                plugin_module_path,
            )
            if spec is None or spec.loader is None:
                raise ValueError(f"Failed to create module spec for {plugin_module_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            self._loaded_modules[plugin_name] = module

        except Exception as e:
            raise ValueError(f"Failed to import plugin module {plugin_name}: {e}") from e

        # 4. 实例化插件
        if not hasattr(module, "Plugin"):
            raise ValueError(
                f"Plugin class not found in {plugin_name}. "
                f"Expected class name: Plugin"
            )

        plugin_class = module.Plugin

        # 验证插件类继承自PluginBase
        if not issubclass(plugin_class, PluginBase):
            raise ValueError(
                f"Plugin class in {plugin_name} must inherit from PluginBase"
            )

        try:
            plugin_instance = plugin_class()
        except Exception as e:
            raise ValueError(f"Failed to instantiate plugin {plugin_name}: {e}") from e

        # 5. 缓存插件实例
        self._loaded_plugins[plugin_name] = plugin_instance

        logger.info(f"Plugin {plugin_name} loaded successfully")

        return plugin_instance

    def unload(self, plugin_name: str):
        """
        卸载运行时插件（从内存中移除）

        Args:
            plugin_name: 插件名称
        """
        if plugin_name not in self._loaded_plugins:
            logger.warning(f"Plugin {plugin_name} not loaded")
            return

        # 调用插件清理方法
        plugin = self._loaded_plugins[plugin_name]
        if hasattr(plugin, "cleanup"):
            try:
                plugin.cleanup()
                logger.debug(f"Plugin {plugin_name} cleanup completed")
            except Exception as e:
                logger.error(f"Error during plugin {plugin_name} cleanup: {e}")

        # 从缓存中移除
        del self._loaded_plugins[plugin_name]

        # 从模块缓存中移除
        if plugin_name in self._loaded_modules:
            module_name = f"apt_plugins.{plugin_name}"
            if module_name in sys.modules:
                del sys.modules[module_name]
            del self._loaded_modules[plugin_name]

        logger.info(f"Plugin {plugin_name} unloaded")

    def get_loaded_plugin(self, plugin_name: str) -> Optional[PluginBase]:
        """
        获取已加载的插件实例

        Args:
            plugin_name: 插件名称

        Returns:
            插件实例，如果未加载则返回None
        """
        return self._loaded_plugins.get(plugin_name)

    def list_installed(self) -> List[Dict[str, Any]]:
        """
        列出已安装的插件

        Returns:
            插件manifest列表
        """
        installed = []

        if not self.plugin_dir.exists():
            return installed

        for plugin_dir in self.plugin_dir.iterdir():
            if plugin_dir.is_dir():
                manifest_path = plugin_dir / "plugin.yaml"
                if manifest_path.exists():
                    try:
                        with open(manifest_path, 'r', encoding='utf-8') as f:
                            manifest = yaml.safe_load(f)
                        installed.append(manifest)
                    except Exception as e:
                        logger.error(f"Error reading manifest from {plugin_dir}: {e}")

        return installed

    def list_loaded(self) -> List[str]:
        """
        列出已加载的插件名称

        Returns:
            插件名称列表
        """
        return list(self._loaded_plugins.keys())

    def _validate_manifest(self, manifest: Dict[str, Any]):
        """验证manifest基本字段"""
        required_fields = ["name", "version", "description", "author"]

        missing = []
        for field in required_fields:
            if field not in manifest:
                missing.append(field)

        if missing:
            raise ValueError(
                f"Manifest missing required fields: {', '.join(missing)}"
            )

    def cleanup(self):
        """清理所有加载的插件"""
        for plugin_name in list(self._loaded_plugins.keys()):
            self.unload(plugin_name)

        logger.debug("All plugins unloaded")

    def __del__(self):
        """析构时自动清理"""
        try:
            self.cleanup()
        except Exception:
            pass  # Suppress errors during cleanup in destructor
