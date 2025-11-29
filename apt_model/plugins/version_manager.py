#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
插件版本管理系统

提供插件版本控制、依赖解析和兼容性检查功能。
支持语义化版本(Semantic Versioning)规范。
"""

import re
import json
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
from datetime import datetime


@dataclass
class Version:
    """
    语义化版本类

    遵循Semantic Versioning 2.0.0规范: MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]
    - MAJOR: 不兼容的API修改
    - MINOR: 向后兼容的功能性新增
    - PATCH: 向后兼容的问题修复
    """
    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build: Optional[str] = None

    @classmethod
    def parse(cls, version_string: str) -> "Version":
        """
        解析版本字符串

        支持格式:
        - "1.2.3"
        - "1.2.3-alpha"
        - "1.2.3-beta.1"
        - "1.2.3+build123"
        - "1.2.3-rc.1+build.456"
        """
        # 正则表达式匹配语义化版本
        pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.-]+))?(?:\+([a-zA-Z0-9.-]+))?$'
        match = re.match(pattern, version_string.strip())

        if not match:
            raise ValueError(f"Invalid version string: {version_string}")

        major, minor, patch, prerelease, build = match.groups()

        return cls(
            major=int(major),
            minor=int(minor),
            patch=int(patch),
            prerelease=prerelease,
            build=build
        )

    def __str__(self) -> str:
        """返回版本字符串"""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __eq__(self, other: "Version") -> bool:
        """版本相等比较（忽略build元数据）"""
        return (
            self.major == other.major and
            self.minor == other.minor and
            self.patch == other.patch and
            self.prerelease == other.prerelease
        )

    def __lt__(self, other: "Version") -> bool:
        """版本小于比较"""
        # 比较主版本号
        if self.major != other.major:
            return self.major < other.major

        # 比较次版本号
        if self.minor != other.minor:
            return self.minor < other.minor

        # 比较修订号
        if self.patch != other.patch:
            return self.patch < other.patch

        # 处理预发布版本
        # 如果一个有预发布版本而另一个没有，没有预发布版本的更大
        if self.prerelease is None and other.prerelease is not None:
            return False
        if self.prerelease is not None and other.prerelease is None:
            return True

        # 两个都有预发布版本，按字符串比较
        if self.prerelease is not None and other.prerelease is not None:
            return self.prerelease < other.prerelease

        # 完全相同
        return False

    def __le__(self, other: "Version") -> bool:
        return self == other or self < other

    def __gt__(self, other: "Version") -> bool:
        return not self <= other

    def __ge__(self, other: "Version") -> bool:
        return not self < other

    def is_compatible_with(self, other: "Version", mode: str = "minor") -> bool:
        """
        检查版本兼容性

        参数:
            other: 另一个版本
            mode: 兼容性模式
                - "major": 主版本号必须相同
                - "minor": 主版本号和次版本号必须相同
                - "patch": 完全相同版本
                - "caret": Caret范围 (^1.2.3 = >=1.2.3 <2.0.0)
                - "tilde": Tilde范围 (~1.2.3 = >=1.2.3 <1.3.0)

        返回:
            是否兼容
        """
        if mode == "major":
            return self.major == other.major
        elif mode == "minor":
            return self.major == other.major and self.minor == other.minor
        elif mode == "patch":
            return self == other
        elif mode == "caret":
            # ^1.2.3 := >=1.2.3 <2.0.0
            if self.major == 0:
                # ^0.2.3 := >=0.2.3 <0.3.0
                return (
                    self.major == other.major and
                    self.minor == other.minor and
                    self >= other
                )
            return self.major == other.major and self >= other
        elif mode == "tilde":
            # ~1.2.3 := >=1.2.3 <1.3.0
            return (
                self.major == other.major and
                self.minor == other.minor and
                self >= other
            )
        else:
            raise ValueError(f"Unknown compatibility mode: {mode}")


@dataclass
class PluginDependency:
    """插件依赖"""
    name: str
    version_constraint: str  # e.g., ">=1.0.0,<2.0.0" or "^1.2.3"
    optional: bool = False

    def is_satisfied_by(self, version: Version) -> bool:
        """
        检查版本是否满足依赖约束

        支持的约束格式:
        - "1.2.3": 精确版本
        - ">=1.2.3": 大于等于
        - "<=1.2.3": 小于等于
        - ">1.2.3": 大于
        - "<1.2.3": 小于
        - "^1.2.3": Caret范围
        - "~1.2.3": Tilde范围
        - ">=1.0.0,<2.0.0": 范围组合
        """
        # 处理组合约束（逗号分隔）
        if ',' in self.version_constraint:
            constraints = [c.strip() for c in self.version_constraint.split(',')]
            return all(self._check_single_constraint(c, version) for c in constraints)

        return self._check_single_constraint(self.version_constraint, version)

    def _check_single_constraint(self, constraint: str, version: Version) -> bool:
        """检查单个约束"""
        constraint = constraint.strip()

        # Caret范围
        if constraint.startswith('^'):
            required = Version.parse(constraint[1:])
            return version.is_compatible_with(required, mode="caret")

        # Tilde范围
        if constraint.startswith('~'):
            required = Version.parse(constraint[1:])
            return version.is_compatible_with(required, mode="tilde")

        # 大于等于
        if constraint.startswith('>='):
            required = Version.parse(constraint[2:])
            return version >= required

        # 小于等于
        if constraint.startswith('<='):
            required = Version.parse(constraint[2:])
            return version <= required

        # 大于
        if constraint.startswith('>'):
            required = Version.parse(constraint[1:])
            return version > required

        # 小于
        if constraint.startswith('<'):
            required = Version.parse(constraint[1:])
            return version < required

        # 精确匹配
        required = Version.parse(constraint)
        return version == required


@dataclass
class PluginMetadata:
    """插件元数据"""
    name: str
    version: Version
    description: str = ""
    author: str = ""
    license: str = ""
    homepage: str = ""
    dependencies: List[PluginDependency] = None
    apt_version_required: str = ">=1.0.0"  # APT框架版本要求
    python_version_required: str = ">=3.7"
    tags: List[str] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.tags is None:
            self.tags = []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginMetadata":
        """从字典创建元数据"""
        # 解析版本
        version = Version.parse(data['version'])

        # 解析依赖
        dependencies = []
        for dep in data.get('dependencies', []):
            if isinstance(dep, dict):
                dependencies.append(PluginDependency(**dep))
            else:
                # 简化格式: "name:version_constraint"
                parts = dep.split(':', 1)
                dependencies.append(PluginDependency(
                    name=parts[0],
                    version_constraint=parts[1] if len(parts) > 1 else "*"
                ))

        return cls(
            name=data['name'],
            version=version,
            description=data.get('description', ''),
            author=data.get('author', ''),
            license=data.get('license', ''),
            homepage=data.get('homepage', ''),
            dependencies=dependencies,
            apt_version_required=data.get('apt_version_required', '>=1.0.0'),
            python_version_required=data.get('python_version_required', '>=3.7'),
            tags=data.get('tags', [])
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'version': str(self.version),
            'description': self.description,
            'author': self.author,
            'license': self.license,
            'homepage': self.homepage,
            'dependencies': [
                {
                    'name': dep.name,
                    'version_constraint': dep.version_constraint,
                    'optional': dep.optional
                }
                for dep in self.dependencies
            ],
            'apt_version_required': self.apt_version_required,
            'python_version_required': self.python_version_required,
            'tags': self.tags
        }


class PluginVersionManager:
    """
    插件版本管理器

    功能:
    - 插件注册和版本跟踪
    - 依赖解析和冲突检测
    - 版本兼容性检查
    - 插件升级和降级管理
    """

    def __init__(self, registry_path: str = ".cache/plugins/registry.json"):
        """
        初始化版本管理器

        参数:
            registry_path: 插件注册表路径
        """
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        # 插件注册表: {plugin_name: [PluginMetadata, ...]}
        self.plugins: Dict[str, List[PluginMetadata]] = defaultdict(list)

        # 已安装插件: {plugin_name: PluginMetadata}
        self.installed: Dict[str, PluginMetadata] = {}

        # 加载注册表
        self._load_registry()

    def _load_registry(self):
        """从文件加载插件注册表"""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 恢复插件列表
                for plugin_name, versions in data.get('plugins', {}).items():
                    self.plugins[plugin_name] = [
                        PluginMetadata.from_dict(v) for v in versions
                    ]

                # 恢复已安装插件
                for plugin_name, metadata in data.get('installed', {}).items():
                    self.installed[plugin_name] = PluginMetadata.from_dict(metadata)

            except Exception as e:
                print(f"Warning: Failed to load plugin registry: {e}")

    def _save_registry(self):
        """保存插件注册表到文件"""
        data = {
            'plugins': {
                name: [m.to_dict() for m in versions]
                for name, versions in self.plugins.items()
            },
            'installed': {
                name: metadata.to_dict()
                for name, metadata in self.installed.items()
            },
            'last_updated': datetime.now().isoformat()
        }

        with open(self.registry_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def register_plugin(self, metadata: PluginMetadata):
        """
        注册插件版本

        参数:
            metadata: 插件元数据
        """
        # 检查是否已存在相同版本
        existing_versions = self.plugins[metadata.name]
        for existing in existing_versions:
            if existing.version == metadata.version:
                raise ValueError(
                    f"Plugin {metadata.name} version {metadata.version} already registered"
                )

        # 添加到注册表
        self.plugins[metadata.name].append(metadata)

        # 按版本排序
        self.plugins[metadata.name].sort(key=lambda m: m.version, reverse=True)

        # 保存
        self._save_registry()

    def install_plugin(self, name: str, version: Optional[str] = None) -> PluginMetadata:
        """
        安装插件

        参数:
            name: 插件名称
            version: 指定版本（None表示安装最新版本）

        返回:
            安装的插件元数据
        """
        # 查找可用版本
        available_versions = self.plugins.get(name, [])

        if not available_versions:
            raise ValueError(f"Plugin '{name}' not found in registry")

        # 选择版本
        if version is None:
            # 安装最新稳定版本（跳过预发布版本）
            stable_versions = [m for m in available_versions if m.version.prerelease is None]
            if not stable_versions:
                # 如果没有稳定版本，使用最新版本
                metadata = available_versions[0]
            else:
                metadata = stable_versions[0]
        else:
            # 安装指定版本
            target_version = Version.parse(version)
            metadata = None
            for m in available_versions:
                if m.version == target_version:
                    metadata = m
                    break

            if metadata is None:
                raise ValueError(
                    f"Plugin '{name}' version '{version}' not found. "
                    f"Available: {[str(m.version) for m in available_versions]}"
                )

        # 检查依赖
        self._check_dependencies(metadata)

        # 安装
        self.installed[name] = metadata
        self._save_registry()

        return metadata

    def uninstall_plugin(self, name: str):
        """卸载插件"""
        if name not in self.installed:
            raise ValueError(f"Plugin '{name}' is not installed")

        # 检查是否有其他插件依赖此插件
        dependents = self._find_dependents(name)
        if dependents:
            raise ValueError(
                f"Cannot uninstall '{name}': required by {', '.join(dependents)}"
            )

        del self.installed[name]
        self._save_registry()

    def upgrade_plugin(self, name: str, target_version: Optional[str] = None) -> PluginMetadata:
        """
        升级插件

        参数:
            name: 插件名称
            target_version: 目标版本（None表示升级到最新版本）

        返回:
            升级后的插件元数据
        """
        if name not in self.installed:
            raise ValueError(f"Plugin '{name}' is not installed")

        current_metadata = self.installed[name]

        # 查找目标版本
        available_versions = self.plugins.get(name, [])

        if target_version is None:
            # 升级到最新版本
            candidates = [
                m for m in available_versions
                if m.version > current_metadata.version and m.version.prerelease is None
            ]
        else:
            # 升级到指定版本
            target = Version.parse(target_version)
            candidates = [m for m in available_versions if m.version == target]

        if not candidates:
            raise ValueError(f"No upgrade available for '{name}'")

        new_metadata = candidates[0]

        # 检查依赖
        self._check_dependencies(new_metadata)

        # 升级
        self.installed[name] = new_metadata
        self._save_registry()

        return new_metadata

    def _check_dependencies(self, metadata: PluginMetadata):
        """检查插件依赖是否满足"""
        for dep in metadata.dependencies:
            if dep.optional:
                continue

            # 检查依赖是否已安装
            if dep.name not in self.installed:
                raise ValueError(
                    f"Missing dependency: {dep.name} {dep.version_constraint}"
                )

            # 检查版本是否满足
            installed_version = self.installed[dep.name].version
            if not dep.is_satisfied_by(installed_version):
                raise ValueError(
                    f"Dependency version conflict: {dep.name} requires {dep.version_constraint}, "
                    f"but {installed_version} is installed"
                )

    def _find_dependents(self, plugin_name: str) -> List[str]:
        """查找依赖指定插件的其他插件"""
        dependents = []

        for name, metadata in self.installed.items():
            if name == plugin_name:
                continue

            for dep in metadata.dependencies:
                if dep.name == plugin_name and not dep.optional:
                    dependents.append(name)
                    break

        return dependents

    def get_installed_plugins(self) -> Dict[str, PluginMetadata]:
        """获取已安装插件列表"""
        return self.installed.copy()

    def get_available_versions(self, name: str) -> List[Version]:
        """获取插件的所有可用版本"""
        if name not in self.plugins:
            return []
        return [m.version for m in self.plugins[name]]

    def check_compatibility(self, plugin1: str, plugin2: str) -> Tuple[bool, str]:
        """
        检查两个插件是否兼容

        返回:
            (是否兼容, 原因说明)
        """
        if plugin1 not in self.installed or plugin2 not in self.installed:
            return False, "One or both plugins are not installed"

        meta1 = self.installed[plugin1]
        meta2 = self.installed[plugin2]

        # 检查是否有相互依赖
        for dep in meta1.dependencies:
            if dep.name == plugin2:
                if not dep.is_satisfied_by(meta2.version):
                    return False, f"{plugin1} requires {plugin2} {dep.version_constraint}"

        for dep in meta2.dependencies:
            if dep.name == plugin1:
                if not dep.is_satisfied_by(meta1.version):
                    return False, f"{plugin2} requires {plugin1} {dep.version_constraint}"

        return True, "Compatible"

    # ========================================================================
    # 🔮 WebUI/API Export Interface
    # ========================================================================

    def export_for_webui(self, export_path: str = None) -> Dict[str, Any]:
        """
        导出插件管理数据供WebUI/API使用

        未来API端点:
        - GET /api/plugins/installed
        - GET /api/plugins/available
        - POST /api/plugins/install
        - POST /api/plugins/uninstall
        - POST /api/plugins/upgrade
        - GET /api/plugins/{name}/versions

        参数:
            export_path: JSON文件导出路径（可选）

        返回:
            完整的插件管理数据
        """
        # 已安装插件
        installed_list = [
            {
                **metadata.to_dict(),
                'installed': True,
                'installed_at': datetime.now().isoformat()
            }
            for metadata in self.installed.values()
        ]

        # 可用插件（所有版本）
        available_list = []
        for name, versions in self.plugins.items():
            for metadata in versions:
                available_list.append({
                    **metadata.to_dict(),
                    'installed': name in self.installed,
                    'is_latest': metadata == versions[0],
                    'is_stable': metadata.version.prerelease is None
                })

        # 依赖关系图
        dependency_graph = {}
        for name, metadata in self.installed.items():
            dependency_graph[name] = [
                {'name': dep.name, 'constraint': dep.version_constraint, 'optional': dep.optional}
                for dep in metadata.dependencies
            ]

        # 统计信息
        statistics = {
            'total_installed': len(self.installed),
            'total_available': len(self.plugins),
            'total_versions': sum(len(versions) for versions in self.plugins.values()),
            'plugins_by_tag': self._get_plugins_by_tag()
        }

        data = {
            'installed': installed_list,
            'available': available_list,
            'dependency_graph': dependency_graph,
            'statistics': statistics,
            'generated_at': datetime.now().isoformat()
        }

        # 导出到JSON文件
        if export_path:
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        return data

    def _get_plugins_by_tag(self) -> Dict[str, int]:
        """按标签统计插件"""
        tag_counts = defaultdict(int)
        for metadata in self.installed.values():
            for tag in metadata.tags:
                tag_counts[tag] += 1
        return dict(tag_counts)

    def generate_plugin_report(self, output_path: str = None) -> str:
        """
        生成插件管理报告（Markdown格式）

        参数:
            output_path: 报告保存路径（可选）

        返回:
            Markdown格式的报告内容
        """
        report = []
        report.append("# 插件管理报告")
        report.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 已安装插件
        report.append("## 已安装插件\n")
        if self.installed:
            report.append("| 名称 | 版本 | 描述 | 作者 | 依赖数 |")
            report.append("|------|------|------|------|--------|")
            for name, metadata in self.installed.items():
                report.append(
                    f"| {metadata.name} | {metadata.version} | "
                    f"{metadata.description[:30]}... | {metadata.author} | "
                    f"{len(metadata.dependencies)} |"
                )
        else:
            report.append("*暂无已安装插件*")

        # 可用插件
        report.append("\n## 可用插件\n")
        if self.plugins:
            for name, versions in sorted(self.plugins.items()):
                latest = versions[0]
                report.append(f"\n### {name}")
                report.append(f"- **最新版本**: {latest.version}")
                report.append(f"- **描述**: {latest.description}")
                report.append(f"- **可用版本**: {', '.join(str(v.version) for v in versions[:5])}")
                if len(versions) > 5:
                    report.append(f"  ...及 {len(versions) - 5} 个其他版本")

        # 依赖关系
        report.append("\n## 依赖关系\n")
        for name, metadata in self.installed.items():
            if metadata.dependencies:
                report.append(f"\n**{name}** 依赖:")
                for dep in metadata.dependencies:
                    optional_tag = " (可选)" if dep.optional else ""
                    report.append(f"- {dep.name} {dep.version_constraint}{optional_tag}")

        report_text = '\n'.join(report)

        # 保存报告
        if output_path:
            report_file = Path(output_path)
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_text)

        return report_text
