#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
插件版本管理系统测试

测试版本解析、依赖管理、兼容性检查等功能
"""

import os
import json
import pytest
import tempfile
from pathlib import Path

from apt.apt_model.plugins.version_manager import (
    Version,
    PluginDependency,
    PluginMetadata,
    PluginVersionManager
)


@pytest.fixture
def temp_registry():
    """临时注册表文件fixture"""
    with tempfile.TemporaryDirectory() as temp_dir:
        registry_path = Path(temp_dir) / "registry.json"
        yield registry_path


@pytest.fixture
def version_manager(temp_registry):
    """PluginVersionManager实例fixture"""
    return PluginVersionManager(registry_path=str(temp_registry))


class TestVersionParsing:
    """测试版本解析"""

    def test_parse_simple_version(self):
        """测试解析简单版本"""
        v = Version.parse("1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3
        assert v.prerelease is None
        assert v.build is None

    def test_parse_prerelease_version(self):
        """测试解析预发布版本"""
        v = Version.parse("1.2.3-alpha")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3
        assert v.prerelease == "alpha"

        v2 = Version.parse("1.2.3-beta.1")
        assert v2.prerelease == "beta.1"

        v3 = Version.parse("2.0.0-rc.1")
        assert v3.prerelease == "rc.1"

    def test_parse_build_metadata(self):
        """测试解析构建元数据"""
        v = Version.parse("1.2.3+build123")
        assert v.build == "build123"

        v2 = Version.parse("1.2.3-alpha+build.456")
        assert v2.prerelease == "alpha"
        assert v2.build == "build.456"

    def test_parse_invalid_version(self):
        """测试解析无效版本"""
        with pytest.raises(ValueError):
            Version.parse("1.2")

        with pytest.raises(ValueError):
            Version.parse("1.2.3.4")

        with pytest.raises(ValueError):
            Version.parse("invalid")

    def test_version_string_representation(self):
        """测试版本字符串表示"""
        assert str(Version.parse("1.2.3")) == "1.2.3"
        assert str(Version.parse("1.2.3-alpha")) == "1.2.3-alpha"
        assert str(Version.parse("1.2.3+build")) == "1.2.3+build"
        assert str(Version.parse("1.2.3-rc.1+build.2")) == "1.2.3-rc.1+build.2"


class TestVersionComparison:
    """测试版本比较"""

    def test_version_equality(self):
        """测试版本相等"""
        v1 = Version.parse("1.2.3")
        v2 = Version.parse("1.2.3")
        assert v1 == v2

        # build元数据不影响相等性
        v3 = Version.parse("1.2.3+build1")
        v4 = Version.parse("1.2.3+build2")
        assert v3 == v4

    def test_version_ordering(self):
        """测试版本排序"""
        v1 = Version.parse("1.0.0")
        v2 = Version.parse("1.1.0")
        v3 = Version.parse("1.1.1")
        v4 = Version.parse("2.0.0")

        assert v1 < v2 < v3 < v4
        assert v4 > v3 > v2 > v1

    def test_prerelease_ordering(self):
        """测试预发布版本排序"""
        v_stable = Version.parse("1.0.0")
        v_alpha = Version.parse("1.0.0-alpha")
        v_beta = Version.parse("1.0.0-beta")
        v_rc = Version.parse("1.0.0-rc")

        # 预发布版本 < 稳定版本
        assert v_alpha < v_stable
        assert v_beta < v_stable
        assert v_rc < v_stable

        # 预发布版本之间按字母顺序
        assert v_alpha < v_beta < v_rc

    def test_version_comparison_operators(self):
        """测试版本比较运算符"""
        v1 = Version.parse("1.0.0")
        v2 = Version.parse("2.0.0")

        assert v1 < v2
        assert v1 <= v2
        assert v2 > v1
        assert v2 >= v1
        assert v1 <= v1
        assert v1 >= v1


class TestVersionCompatibility:
    """测试版本兼容性"""

    def test_major_compatibility(self):
        """测试主版本兼容性"""
        v1 = Version.parse("1.2.3")
        v2 = Version.parse("1.5.0")
        v3 = Version.parse("2.0.0")

        assert v1.is_compatible_with(v2, mode="major")
        assert not v1.is_compatible_with(v3, mode="major")

    def test_minor_compatibility(self):
        """测试次版本兼容性"""
        v1 = Version.parse("1.2.3")
        v2 = Version.parse("1.2.5")
        v3 = Version.parse("1.3.0")

        assert v1.is_compatible_with(v2, mode="minor")
        assert not v1.is_compatible_with(v3, mode="minor")

    def test_caret_compatibility(self):
        """测试Caret范围兼容性"""
        base = Version.parse("1.2.3")

        assert Version.parse("1.2.3").is_compatible_with(base, mode="caret")
        assert Version.parse("1.2.4").is_compatible_with(base, mode="caret")
        assert Version.parse("1.5.0").is_compatible_with(base, mode="caret")
        assert not Version.parse("2.0.0").is_compatible_with(base, mode="caret")

        # 0.x.y特殊处理
        base_zero = Version.parse("0.2.3")
        assert Version.parse("0.2.3").is_compatible_with(base_zero, mode="caret")
        assert Version.parse("0.2.5").is_compatible_with(base_zero, mode="caret")
        assert not Version.parse("0.3.0").is_compatible_with(base_zero, mode="caret")

    def test_tilde_compatibility(self):
        """测试Tilde范围兼容性"""
        base = Version.parse("1.2.3")

        assert Version.parse("1.2.3").is_compatible_with(base, mode="tilde")
        assert Version.parse("1.2.5").is_compatible_with(base, mode="tilde")
        assert not Version.parse("1.3.0").is_compatible_with(base, mode="tilde")


class TestPluginDependency:
    """测试插件依赖"""

    def test_exact_version_constraint(self):
        """测试精确版本约束"""
        dep = PluginDependency(name="plugin-a", version_constraint="1.2.3")

        assert dep.is_satisfied_by(Version.parse("1.2.3"))
        assert not dep.is_satisfied_by(Version.parse("1.2.4"))
        assert not dep.is_satisfied_by(Version.parse("1.2.2"))

    def test_range_constraint(self):
        """测试范围约束"""
        dep = PluginDependency(name="plugin-a", version_constraint=">=1.0.0,<2.0.0")

        assert dep.is_satisfied_by(Version.parse("1.0.0"))
        assert dep.is_satisfied_by(Version.parse("1.5.0"))
        assert not dep.is_satisfied_by(Version.parse("2.0.0"))
        assert not dep.is_satisfied_by(Version.parse("0.9.0"))

    def test_caret_constraint(self):
        """测试Caret约束"""
        dep = PluginDependency(name="plugin-a", version_constraint="^1.2.3")

        assert dep.is_satisfied_by(Version.parse("1.2.3"))
        assert dep.is_satisfied_by(Version.parse("1.5.0"))
        assert not dep.is_satisfied_by(Version.parse("2.0.0"))

    def test_tilde_constraint(self):
        """测试Tilde约束"""
        dep = PluginDependency(name="plugin-a", version_constraint="~1.2.3")

        assert dep.is_satisfied_by(Version.parse("1.2.3"))
        assert dep.is_satisfied_by(Version.parse("1.2.5"))
        assert not dep.is_satisfied_by(Version.parse("1.3.0"))

    def test_comparison_constraints(self):
        """测试比较运算符约束"""
        dep_gte = PluginDependency(name="plugin-a", version_constraint=">=1.0.0")
        assert dep_gte.is_satisfied_by(Version.parse("1.0.0"))
        assert dep_gte.is_satisfied_by(Version.parse("2.0.0"))
        assert not dep_gte.is_satisfied_by(Version.parse("0.9.0"))

        dep_lt = PluginDependency(name="plugin-a", version_constraint="<2.0.0")
        assert dep_lt.is_satisfied_by(Version.parse("1.5.0"))
        assert not dep_lt.is_satisfied_by(Version.parse("2.0.0"))


class TestPluginMetadata:
    """测试插件元数据"""

    def test_create_metadata(self):
        """测试创建元数据"""
        metadata = PluginMetadata(
            name="test-plugin",
            version=Version.parse("1.0.0"),
            description="A test plugin",
            author="Test Author"
        )

        assert metadata.name == "test-plugin"
        assert metadata.version == Version.parse("1.0.0")
        assert metadata.description == "A test plugin"
        assert metadata.author == "Test Author"

    def test_metadata_with_dependencies(self):
        """测试带依赖的元数据"""
        metadata = PluginMetadata(
            name="test-plugin",
            version=Version.parse("1.0.0"),
            dependencies=[
                PluginDependency(name="dep-a", version_constraint="^1.0.0"),
                PluginDependency(name="dep-b", version_constraint="~2.1.0", optional=True)
            ]
        )

        assert len(metadata.dependencies) == 2
        assert metadata.dependencies[0].name == "dep-a"
        assert metadata.dependencies[1].optional

    def test_metadata_serialization(self):
        """测试元数据序列化"""
        metadata = PluginMetadata(
            name="test-plugin",
            version=Version.parse("1.2.3"),
            description="Test",
            dependencies=[
                PluginDependency(name="dep-a", version_constraint="^1.0.0")
            ],
            tags=["ai", "nlp"]
        )

        # 转为字典
        data = metadata.to_dict()
        assert data['name'] == "test-plugin"
        assert data['version'] == "1.2.3"
        assert len(data['dependencies']) == 1
        assert data['tags'] == ["ai", "nlp"]

        # 从字典恢复
        restored = PluginMetadata.from_dict(data)
        assert restored.name == metadata.name
        assert restored.version == metadata.version
        assert len(restored.dependencies) == len(metadata.dependencies)


class TestPluginVersionManager:
    """测试插件版本管理器"""

    def test_register_plugin(self, version_manager):
        """测试注册插件"""
        metadata = PluginMetadata(
            name="test-plugin",
            version=Version.parse("1.0.0"),
            description="Test plugin"
        )

        version_manager.register_plugin(metadata)

        assert "test-plugin" in version_manager.plugins
        assert len(version_manager.plugins["test-plugin"]) == 1

    def test_register_multiple_versions(self, version_manager):
        """测试注册多个版本"""
        for version in ["1.0.0", "1.1.0", "2.0.0"]:
            metadata = PluginMetadata(
                name="test-plugin",
                version=Version.parse(version)
            )
            version_manager.register_plugin(metadata)

        versions = version_manager.get_available_versions("test-plugin")
        assert len(versions) == 3
        # 应该按版本降序排列
        assert versions[0] == Version.parse("2.0.0")
        assert versions[1] == Version.parse("1.1.0")
        assert versions[2] == Version.parse("1.0.0")

    def test_install_plugin_latest(self, version_manager):
        """测试安装最新版本"""
        # 注册多个版本
        for version in ["1.0.0", "1.1.0", "2.0.0-beta", "1.5.0"]:
            version_manager.register_plugin(PluginMetadata(
                name="test-plugin",
                version=Version.parse(version)
            ))

        # 安装最新稳定版（跳过beta）
        installed = version_manager.install_plugin("test-plugin")

        assert installed.version == Version.parse("1.5.0")
        assert "test-plugin" in version_manager.installed

    def test_install_specific_version(self, version_manager):
        """测试安装指定版本"""
        for version in ["1.0.0", "1.1.0", "2.0.0"]:
            version_manager.register_plugin(PluginMetadata(
                name="test-plugin",
                version=Version.parse(version)
            ))

        installed = version_manager.install_plugin("test-plugin", version="1.1.0")

        assert installed.version == Version.parse("1.1.0")

    def test_install_with_dependencies(self, version_manager):
        """测试安装带依赖的插件"""
        # 注册依赖插件
        version_manager.register_plugin(PluginMetadata(
            name="dep-plugin",
            version=Version.parse("1.0.0")
        ))

        # 注册主插件（依赖dep-plugin）
        version_manager.register_plugin(PluginMetadata(
            name="main-plugin",
            version=Version.parse("1.0.0"),
            dependencies=[
                PluginDependency(name="dep-plugin", version_constraint="^1.0.0")
            ]
        ))

        # 先安装依赖
        version_manager.install_plugin("dep-plugin")

        # 再安装主插件
        installed = version_manager.install_plugin("main-plugin")
        assert installed.name == "main-plugin"

    def test_install_missing_dependency(self, version_manager):
        """测试安装缺失依赖的插件"""
        version_manager.register_plugin(PluginMetadata(
            name="main-plugin",
            version=Version.parse("1.0.0"),
            dependencies=[
                PluginDependency(name="missing-dep", version_constraint="^1.0.0")
            ]
        ))

        # 应该抛出异常
        with pytest.raises(ValueError, match="Missing dependency"):
            version_manager.install_plugin("main-plugin")

    def test_uninstall_plugin(self, version_manager):
        """测试卸载插件"""
        version_manager.register_plugin(PluginMetadata(
            name="test-plugin",
            version=Version.parse("1.0.0")
        ))

        version_manager.install_plugin("test-plugin")
        assert "test-plugin" in version_manager.installed

        version_manager.uninstall_plugin("test-plugin")
        assert "test-plugin" not in version_manager.installed

    def test_uninstall_with_dependents(self, version_manager):
        """测试卸载被依赖的插件"""
        version_manager.register_plugin(PluginMetadata(
            name="dep-plugin",
            version=Version.parse("1.0.0")
        ))

        version_manager.register_plugin(PluginMetadata(
            name="main-plugin",
            version=Version.parse("1.0.0"),
            dependencies=[
                PluginDependency(name="dep-plugin", version_constraint="^1.0.0")
            ]
        ))

        version_manager.install_plugin("dep-plugin")
        version_manager.install_plugin("main-plugin")

        # 不能卸载被依赖的插件
        with pytest.raises(ValueError, match="required by"):
            version_manager.uninstall_plugin("dep-plugin")

    def test_upgrade_plugin(self, version_manager):
        """测试升级插件"""
        for version in ["1.0.0", "1.1.0", "2.0.0"]:
            version_manager.register_plugin(PluginMetadata(
                name="test-plugin",
                version=Version.parse(version)
            ))

        # 安装旧版本
        version_manager.install_plugin("test-plugin", version="1.0.0")
        assert version_manager.installed["test-plugin"].version == Version.parse("1.0.0")

        # 升级到最新版本
        upgraded = version_manager.upgrade_plugin("test-plugin")
        assert upgraded.version == Version.parse("2.0.0")

    def test_upgrade_to_specific_version(self, version_manager):
        """测试升级到指定版本"""
        for version in ["1.0.0", "1.1.0", "2.0.0"]:
            version_manager.register_plugin(PluginMetadata(
                name="test-plugin",
                version=Version.parse(version)
            ))

        version_manager.install_plugin("test-plugin", version="1.0.0")
        upgraded = version_manager.upgrade_plugin("test-plugin", target_version="1.1.0")

        assert upgraded.version == Version.parse("1.1.0")

    def test_check_compatibility(self, version_manager):
        """测试检查插件兼容性"""
        version_manager.register_plugin(PluginMetadata(
            name="plugin-a",
            version=Version.parse("1.0.0"),
            dependencies=[
                PluginDependency(name="plugin-b", version_constraint="^1.0.0")
            ]
        ))

        version_manager.register_plugin(PluginMetadata(
            name="plugin-b",
            version=Version.parse("1.5.0")
        ))

        version_manager.install_plugin("plugin-b")
        version_manager.install_plugin("plugin-a")

        # 应该兼容
        compatible, reason = version_manager.check_compatibility("plugin-a", "plugin-b")
        assert compatible

    def test_persistence(self, temp_registry):
        """测试注册表持久化"""
        # 创建第一个管理器并注册插件
        vm1 = PluginVersionManager(registry_path=str(temp_registry))
        vm1.register_plugin(PluginMetadata(
            name="test-plugin",
            version=Version.parse("1.0.0")
        ))
        vm1.install_plugin("test-plugin")

        # 创建第二个管理器，应该能加载已注册的插件
        vm2 = PluginVersionManager(registry_path=str(temp_registry))

        assert "test-plugin" in vm2.plugins
        assert "test-plugin" in vm2.installed


class TestWebUIExport:
    """测试WebUI/API数据导出"""

    def test_export_for_webui(self, version_manager):
        """测试WebUI数据导出"""
        # 准备测试数据
        version_manager.register_plugin(PluginMetadata(
            name="plugin-a",
            version=Version.parse("1.0.0"),
            tags=["ai", "nlp"]
        ))

        version_manager.register_plugin(PluginMetadata(
            name="plugin-a",
            version=Version.parse("2.0.0"),
            tags=["ai", "nlp"]
        ))

        version_manager.install_plugin("plugin-a", version="1.0.0")

        # 导出数据
        data = version_manager.export_for_webui()

        # 验证数据结构
        assert 'installed' in data
        assert 'available' in data
        assert 'dependency_graph' in data
        assert 'statistics' in data
        assert 'generated_at' in data

        # 验证已安装插件
        assert len(data['installed']) == 1
        assert data['installed'][0]['name'] == 'plugin-a'
        assert data['installed'][0]['version'] == '1.0.0'

        # 验证可用插件
        assert len(data['available']) == 2

        # 验证统计信息
        assert data['statistics']['total_installed'] == 1
        assert data['statistics']['total_available'] == 1

    def test_export_to_json_file(self, version_manager, temp_registry):
        """测试导出到JSON文件"""
        version_manager.register_plugin(PluginMetadata(
            name="test-plugin",
            version=Version.parse("1.0.0")
        ))

        export_path = temp_registry.parent / "export.json"
        data = version_manager.export_for_webui(export_path=str(export_path))

        # 验证文件已创建
        assert export_path.exists()

        # 验证文件内容
        with open(export_path, 'r', encoding='utf-8') as f:
            file_data = json.load(f)

        assert file_data == data

    def test_generate_plugin_report(self, version_manager):
        """测试生成Markdown报告"""
        version_manager.register_plugin(PluginMetadata(
            name="plugin-a",
            version=Version.parse("1.0.0"),
            description="Test plugin A",
            author="Author A"
        ))

        version_manager.install_plugin("plugin-a")

        # 生成报告
        report = version_manager.generate_plugin_report()

        # 验证报告内容
        assert "# 插件管理报告" in report
        assert "已安装插件" in report
        assert "可用插件" in report
        assert "plugin-a" in report


class TestAPIReadiness:
    """测试未来API端点就绪性"""

    def test_api_get_installed(self, version_manager):
        """
        测试获取已安装插件API

        未来API: GET /api/plugins/installed
        """
        version_manager.register_plugin(PluginMetadata(
            name="plugin-a",
            version=Version.parse("1.0.0")
        ))
        version_manager.install_plugin("plugin-a")

        # 模拟API响应
        installed = version_manager.get_installed_plugins()

        assert isinstance(installed, dict)
        assert "plugin-a" in installed
        assert installed["plugin-a"].version == Version.parse("1.0.0")

    def test_api_get_available_versions(self, version_manager):
        """
        测试获取可用版本API

        未来API: GET /api/plugins/{name}/versions
        """
        for version in ["1.0.0", "1.1.0", "2.0.0"]:
            version_manager.register_plugin(PluginMetadata(
                name="plugin-a",
                version=Version.parse(version)
            ))

        # 模拟API响应
        versions = version_manager.get_available_versions("plugin-a")

        assert len(versions) == 3
        assert Version.parse("2.0.0") in versions

    def test_api_install_plugin(self, version_manager):
        """
        测试安装插件API

        未来API: POST /api/plugins/install
        """
        version_manager.register_plugin(PluginMetadata(
            name="plugin-a",
            version=Version.parse("1.0.0")
        ))

        # 模拟API请求
        result = version_manager.install_plugin("plugin-a")

        assert result.name == "plugin-a"
        assert "plugin-a" in version_manager.installed

    def test_api_uninstall_plugin(self, version_manager):
        """
        测试卸载插件API

        未来API: POST /api/plugins/uninstall
        """
        version_manager.register_plugin(PluginMetadata(
            name="plugin-a",
            version=Version.parse("1.0.0")
        ))
        version_manager.install_plugin("plugin-a")

        # 模拟API请求
        version_manager.uninstall_plugin("plugin-a")

        assert "plugin-a" not in version_manager.installed

    def test_api_upgrade_plugin(self, version_manager):
        """
        测试升级插件API

        未来API: POST /api/plugins/upgrade
        """
        for version in ["1.0.0", "2.0.0"]:
            version_manager.register_plugin(PluginMetadata(
                name="plugin-a",
                version=Version.parse(version)
            ))

        version_manager.install_plugin("plugin-a", version="1.0.0")

        # 模拟API请求
        result = version_manager.upgrade_plugin("plugin-a")

        assert result.version == Version.parse("2.0.0")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
