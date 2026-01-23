"""
插件注册表

管理插件元数据、版本和依赖关系
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
import yaml
import logging
from packaging import version as pkg_version

logger = logging.getLogger(__name__)


class PluginRegistry:
    """插件注册表 - 管理插件元数据和依赖"""

    def __init__(self, registry_file: Optional[Path] = None):
        """
        Args:
            registry_file: 注册表文件路径（None则使用默认 ~/.apt/plugin_registry.yaml）
        """
        if registry_file is None:
            registry_file = Path.home() / ".apt" / "plugin_registry.yaml"

        self.registry_file = registry_file
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)

        self._registry: Dict[str, Dict[str, Any]] = {}
        self._load_registry()

        logger.debug(f"PluginRegistry initialized with file: {registry_file}")

    def _load_registry(self):
        """从文件加载注册表"""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r', encoding='utf-8') as f:
                    self._registry = yaml.safe_load(f) or {}
                logger.debug(f"Loaded {len(self._registry)} plugins from registry")
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
                self._registry = {}
        else:
            self._registry = {}
            logger.debug("Registry file not found, starting with empty registry")

    def _save_registry(self):
        """保存注册表到文件"""
        try:
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                yaml.dump(self._registry, f, allow_unicode=True, sort_keys=False)
            logger.debug(f"Registry saved to {self.registry_file}")
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
            raise

    def register(self, manifest: Dict[str, Any], enabled: bool = True):
        """
        注册插件

        Args:
            manifest: 插件manifest字典
            enabled: 是否启用插件

        Raises:
            ValueError: 如果manifest无效
        """
        # 验证必需字段
        required_fields = ["name", "version"]
        for field in required_fields:
            if field not in manifest:
                raise ValueError(f"Manifest missing required field: {field}")

        plugin_name = manifest["name"]
        plugin_version = manifest["version"]

        # 创建插件条目（如果不存在）
        if plugin_name not in self._registry:
            self._registry[plugin_name] = {
                "versions": {},
                "latest": plugin_version,
            }

        # 添加版本信息
        self._registry[plugin_name]["versions"][plugin_version] = {
            "manifest": manifest,
            "installed": True,
            "enabled": enabled,
        }

        # 更新latest版本
        current_latest = self._registry[plugin_name]["latest"]
        if self._compare_versions(plugin_version, current_latest) > 0:
            self._registry[plugin_name]["latest"] = plugin_version

        self._save_registry()
        logger.info(f"Registered plugin: {plugin_name} v{plugin_version}")

    def unregister(self, plugin_name: str, version: Optional[str] = None):
        """
        注销插件

        Args:
            plugin_name: 插件名称
            version: 插件版本（None则注销所有版本）

        Raises:
            ValueError: 如果插件未注册
        """
        if plugin_name not in self._registry:
            raise ValueError(f"Plugin {plugin_name} not registered")

        if version is None:
            # 删除整个插件
            del self._registry[plugin_name]
            logger.info(f"Unregistered plugin: {plugin_name} (all versions)")
        else:
            # 删除特定版本
            if version not in self._registry[plugin_name]["versions"]:
                raise ValueError(f"Plugin {plugin_name} version {version} not registered")

            del self._registry[plugin_name]["versions"][version]
            logger.info(f"Unregistered plugin version: {plugin_name} v{version}")

            # 如果没有版本了，删除整个插件
            if not self._registry[plugin_name]["versions"]:
                del self._registry[plugin_name]
                logger.debug(f"Removed plugin {plugin_name} (no versions left)")
            else:
                # 更新latest版本
                versions = list(self._registry[plugin_name]["versions"].keys())
                self._registry[plugin_name]["latest"] = max(
                    versions,
                    key=lambda v: pkg_version.parse(v)
                )

        self._save_registry()

    def get_plugin_info(
        self,
        plugin_name: str,
        version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        获取插件信息

        Args:
            plugin_name: 插件名称
            version: 插件版本（None则返回latest版本）

        Returns:
            插件信息字典，如果不存在则返回None
        """
        if plugin_name not in self._registry:
            return None

        if version is None:
            version = self._registry[plugin_name]["latest"]

        return self._registry[plugin_name]["versions"].get(version)

    def get_manifest(
        self,
        plugin_name: str,
        version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        获取插件manifest

        Args:
            plugin_name: 插件名称
            version: 插件版本（None则返回latest版本）

        Returns:
            插件manifest字典，如果不存在则返回None
        """
        info = self.get_plugin_info(plugin_name, version)
        if info is None:
            return None
        return info.get("manifest")

    def list_plugins(
        self,
        enabled_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        列出所有注册的插件

        Args:
            enabled_only: 是否只列出启用的插件

        Returns:
            插件信息列表
        """
        plugins = []

        for plugin_name, plugin_data in self._registry.items():
            latest_version = plugin_data["latest"]
            latest_info = plugin_data["versions"][latest_version]

            if enabled_only and not latest_info.get("enabled", True):
                continue

            plugins.append({
                "name": plugin_name,
                "version": latest_version,
                "enabled": latest_info.get("enabled", True),
                "manifest": latest_info["manifest"],
            })

        return plugins

    def is_enabled(self, plugin_name: str, version: Optional[str] = None) -> bool:
        """
        检查插件是否启用

        Args:
            plugin_name: 插件名称
            version: 插件版本（None则检查latest版本）

        Returns:
            是否启用
        """
        info = self.get_plugin_info(plugin_name, version)
        if info is None:
            return False
        return info.get("enabled", True)

    def set_enabled(
        self,
        plugin_name: str,
        enabled: bool,
        version: Optional[str] = None
    ):
        """
        设置插件启用状态

        Args:
            plugin_name: 插件名称
            enabled: 是否启用
            version: 插件版本（None则设置latest版本）

        Raises:
            ValueError: 如果插件未注册
        """
        if plugin_name not in self._registry:
            raise ValueError(f"Plugin {plugin_name} not registered")

        if version is None:
            version = self._registry[plugin_name]["latest"]

        if version not in self._registry[plugin_name]["versions"]:
            raise ValueError(f"Plugin {plugin_name} version {version} not registered")

        self._registry[plugin_name]["versions"][version]["enabled"] = enabled
        self._save_registry()

        status = "enabled" if enabled else "disabled"
        logger.info(f"Plugin {plugin_name} v{version} {status}")

    def resolve_dependencies(
        self,
        plugin_name: str,
        version: Optional[str] = None
    ) -> List[Tuple[str, Optional[str]]]:
        """
        解析插件依赖链

        Args:
            plugin_name: 插件名称
            version: 插件版本（None则使用latest版本）

        Returns:
            按加载顺序排列的 (plugin_name, version) 元组列表（依赖在前）

        Raises:
            ValueError: 如果插件未注册或存在循环依赖
        """
        visited: Set[str] = set()
        order: List[Tuple[str, Optional[str]]] = []
        visiting: Set[str] = set()  # 用于检测循环依赖

        def visit(name: str, ver: Optional[str] = None):
            # 检测循环依赖
            if name in visiting:
                raise ValueError(f"Circular dependency detected: {name}")

            if name in visited:
                return

            visiting.add(name)
            visited.add(name)

            # 获取插件信息
            info = self.get_plugin_info(name, ver)
            if info is None:
                raise ValueError(f"Plugin {name} not found in registry")

            manifest = info["manifest"]

            # 递归访问依赖
            deps = manifest.get("dependencies", {}).get("plugins", [])
            for dep in deps:
                # 依赖可以是字符串或字典 {"name": "foo", "version": "1.0.0"}
                if isinstance(dep, str):
                    dep_name = dep
                    dep_version = None
                elif isinstance(dep, dict):
                    dep_name = dep["name"]
                    dep_version = dep.get("version")
                else:
                    raise ValueError(f"Invalid dependency format: {dep}")

                visit(dep_name, dep_version)

            # 添加到加载顺序
            order.append((name, ver))

            visiting.remove(name)

        visit(plugin_name, version)
        return order

    def check_conflicts(
        self,
        plugin_name: str,
        loaded_plugins: List[str],
        version: Optional[str] = None
    ) -> List[str]:
        """
        检查插件冲突

        Args:
            plugin_name: 插件名称
            loaded_plugins: 已加载的插件列表
            version: 插件版本（None则使用latest版本）

        Returns:
            冲突的插件列表
        """
        manifest = self.get_manifest(plugin_name, version)
        if manifest is None:
            return []

        # 获取冲突插件列表
        plugin_bus_config = manifest.get("plugin_bus", {})
        conflicting_plugins = plugin_bus_config.get("conflicting_plugins", [])

        # 检查已加载的插件中是否有冲突
        conflicts = []
        for loaded in loaded_plugins:
            if loaded in conflicting_plugins:
                conflicts.append(loaded)

        return conflicts

    def _compare_versions(self, v1: str, v2: str) -> int:
        """
        比较版本号

        Args:
            v1: 版本号1
            v2: 版本号2

        Returns:
            1表示v1>v2, 0表示相等, -1表示v1<v2
        """
        try:
            ver1 = pkg_version.parse(v1)
            ver2 = pkg_version.parse(v2)

            if ver1 > ver2:
                return 1
            elif ver1 < ver2:
                return -1
            else:
                return 0
        except Exception as e:
            logger.warning(f"Error comparing versions {v1} and {v2}: {e}")
            # 如果解析失败，使用字符串比较
            if v1 > v2:
                return 1
            elif v1 < v2:
                return -1
            else:
                return 0

    def clear(self):
        """清空注册表"""
        self._registry = {}
        self._save_registry()
        logger.info("Registry cleared")
