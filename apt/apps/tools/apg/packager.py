"""
APG插件打包器

将插件目录打包为APG (APT Plugin Package) 格式
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import zipfile
import yaml
import tempfile
import shutil
import re
import logging

logger = logging.getLogger(__name__)


class PluginPackager:
    """APG插件打包器"""

    def pack(
        self,
        plugin_dir: Path,
        output: Path,
        manifest: Optional[Dict[str, Any]] = None,
        include_tests: bool = False,
        validate: bool = True,
    ) -> Path:
        """
        打包插件目录为APG文件

        Args:
            plugin_dir: 插件源代码目录
            output: 输出APG文件路径
            manifest: 插件元数据（如果None则从plugin.yaml读取）
            include_tests: 是否包含测试文件
            validate: 是否验证插件结构

        Returns:
            生成的APG文件路径

        Raises:
            ValueError: 如果插件目录或manifest无效
        """
        plugin_dir = Path(plugin_dir)
        output = Path(output)

        # 1. 验证插件目录结构
        if validate:
            self._validate_plugin_dir(plugin_dir)

        # 2. 读取或验证manifest
        if manifest is None:
            manifest_path = plugin_dir / "plugin.yaml"
            if not manifest_path.exists():
                raise ValueError(f"plugin.yaml not found in {plugin_dir}")
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = yaml.safe_load(f)

        # 3. 验证manifest
        if validate:
            self._validate_manifest(manifest)

        # 4. 创建临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)

            # 5. 复制插件文件
            self._copy_plugin_files(
                plugin_dir,
                tmp_root,
                include_tests=include_tests,
            )

            # 6. 写入manifest
            with open(tmp_root / "plugin.yaml", 'w', encoding='utf-8') as f:
                yaml.dump(manifest, f, allow_unicode=True, sort_keys=False)

            # 7. 创建ZIP包
            output.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
                for file_path in tmp_root.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(tmp_root)
                        zf.write(file_path, arcname)

        file_size = output.stat().st_size
        logger.info(f"Plugin packaged: {output} ({file_size:,} bytes)")

        return output

    def _validate_plugin_dir(self, plugin_dir: Path):
        """验证插件目录结构"""
        if not plugin_dir.exists():
            raise ValueError(f"Plugin directory not found: {plugin_dir}")

        if not plugin_dir.is_dir():
            raise ValueError(f"Not a directory: {plugin_dir}")

        # 必需文件/目录
        required_items = {
            "plugin": "Plugin code directory",
            "plugin.yaml": "Plugin manifest file",
        }

        missing = []
        for item, description in required_items.items():
            if not (plugin_dir / item).exists():
                missing.append(f"{item} ({description})")

        if missing:
            raise ValueError(
                f"Required items missing in {plugin_dir}:\n" +
                "\n".join(f"  - {item}" for item in missing)
            )

        # 验证plugin目录包含__init__.py
        plugin_init = plugin_dir / "plugin" / "__init__.py"
        if not plugin_init.exists():
            raise ValueError(f"plugin/__init__.py not found in {plugin_dir}")

    def _validate_manifest(self, manifest: Dict[str, Any]):
        """验证manifest完整性"""
        # 必需字段
        required_fields = {
            "name": str,
            "version": str,
            "description": str,
            "author": str,
        }

        errors = []

        for field, expected_type in required_fields.items():
            if field not in manifest:
                errors.append(f"Required field missing: {field}")
            elif not isinstance(manifest[field], expected_type):
                errors.append(
                    f"Field '{field}' must be {expected_type.__name__}, "
                    f"got {type(manifest[field]).__name__}"
                )

        if errors:
            raise ValueError("Manifest validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        # 验证版本格式
        version = manifest["version"]
        if not self._is_valid_version(version):
            raise ValueError(
                f"Invalid version format: {version}. "
                f"Expected semantic version (e.g., 1.0.0)"
            )

        # 验证engine版本格式（如果存在）
        if "engine" in manifest:
            engine_req = manifest["engine"]
            if not self._is_valid_version_requirement(engine_req):
                raise ValueError(
                    f"Invalid engine version requirement: {engine_req}. "
                    f"Expected format: >=1.0.0, ~=1.2.0, or ==1.0.0"
                )

    def _is_valid_version(self, version: str) -> bool:
        """验证版本号格式（semantic versioning）"""
        pattern = r'^\d+\.\d+\.\d+(?:-[a-zA-Z0-9]+)?(?:\+[a-zA-Z0-9]+)?$'
        return bool(re.match(pattern, version))

    def _is_valid_version_requirement(self, requirement: str) -> bool:
        """验证版本要求格式"""
        # 支持 >=1.0.0, ~=1.2.0, ==1.0.0 或单独的版本号
        patterns = [
            r'^>=\d+\.\d+\.\d+$',
            r'^~=\d+\.\d+\.\d+$',
            r'^==\d+\.\d+\.\d+$',
            r'^\d+\.\d+\.\d+$',
        ]
        return any(re.match(pattern, requirement) for pattern in patterns)

    def _copy_plugin_files(
        self,
        src: Path,
        dst: Path,
        include_tests: bool,
    ):
        """复制插件文件到目标目录"""
        # 需要复制的目录
        dirs_to_copy = ["plugin", "commands", "adapters"]
        if include_tests:
            dirs_to_copy.append("tests")

        for dir_name in dirs_to_copy:
            src_dir = src / dir_name
            if src_dir.exists() and src_dir.is_dir():
                dst_dir = dst / dir_name
                shutil.copytree(src_dir, dst_dir)
                logger.debug(f"Copied directory: {dir_name}")

        # 复制单个文件
        files_to_copy = ["plugin.yaml", "requirements.txt", "README.md", "LICENSE"]
        for file_name in files_to_copy:
            src_file = src / file_name
            if src_file.exists() and src_file.is_file():
                shutil.copy2(src_file, dst / file_name)
                logger.debug(f"Copied file: {file_name}")

    def unpack(self, apg_path: Path, output_dir: Path) -> Dict[str, Any]:
        """
        解包APG文件

        Args:
            apg_path: APG文件路径
            output_dir: 输出目录

        Returns:
            插件manifest字典

        Raises:
            ValueError: 如果APG文件无效
        """
        apg_path = Path(apg_path)
        output_dir = Path(output_dir)

        if not apg_path.exists():
            raise ValueError(f"APG file not found: {apg_path}")

        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)

        # 解压ZIP
        with zipfile.ZipFile(apg_path, 'r') as zf:
            zf.extractall(output_dir)

        logger.info(f"Unpacked APG to: {output_dir}")

        # 读取manifest
        manifest_path = output_dir / "plugin.yaml"
        if not manifest_path.exists():
            raise ValueError(f"Invalid APG: plugin.yaml not found in {apg_path}")

        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = yaml.safe_load(f)

        return manifest

    def get_manifest(self, apg_path: Path) -> Dict[str, Any]:
        """
        读取APG文件的manifest（不解压）

        Args:
            apg_path: APG文件路径

        Returns:
            插件manifest字典

        Raises:
            ValueError: 如果APG文件无效
        """
        apg_path = Path(apg_path)

        if not apg_path.exists():
            raise ValueError(f"APG file not found: {apg_path}")

        # 从ZIP中读取plugin.yaml
        with zipfile.ZipFile(apg_path, 'r') as zf:
            if "plugin.yaml" not in zf.namelist():
                raise ValueError(f"Invalid APG: plugin.yaml not found in {apg_path}")

            with zf.open("plugin.yaml") as f:
                manifest = yaml.safe_load(f)

        return manifest


def pack_plugin(
    src_dir: Path,
    out_apg: Path,
    name: str,
    version: str,
    description: str,
    author: str,
    **kwargs
) -> Path:
    """
    便捷函数：打包插件

    Args:
        src_dir: 插件源代码目录
        out_apg: 输出APG文件路径
        name: 插件名称
        version: 插件版本
        description: 插件描述
        author: 插件作者
        **kwargs: 其他manifest字段

    Returns:
        生成的APG文件路径
    """
    # 构建manifest
    manifest = {
        "name": name,
        "version": version,
        "description": description,
        "author": author,
        **kwargs
    }

    packager = PluginPackager()
    return packager.pack(
        plugin_dir=src_dir,
        output=out_apg,
        manifest=manifest,
    )
