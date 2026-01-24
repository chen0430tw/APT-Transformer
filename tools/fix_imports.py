#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动修复 APT-Transformer 导入路径

用法:
    python tools/fix_imports.py --dry-run  # 预览修改
    python tools/fix_imports.py            # 执行修改
    python tools/fix_imports.py --file apt/apps/cli/commands.py  # 修复单个文件
"""

import re
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# 导入路径映射表
IMPORT_MAPPINGS = {
    # 模型相关
    'apt.apt_model.modeling.apt_model': 'apt.model.architectures.apt_model',
    'apt.apt_model.modeling.chinese_tokenizer_integration': 'apt.model.tokenization.chinese_tokenizer_integration',
    'apt.apt_model.modeling.moe_optimized': 'apt.model.layers.moe_optimized',

    # 训练相关
    'apt.apt_model.training.trainer': 'apt.trainops.engine.trainer',
    'apt.apt_model.training.checkpoint': 'apt.trainops.checkpoints.checkpoint',

    # 工具类（需要注释掉或替换）
    'apt.apt_model.utils.logging_utils': None,  # 使用 logging 标准库
    'apt.apt_model.utils.resource_monitor': 'apt.trainops.eval.training_monitor',
    'apt.apt_model.utils.language_manager': None,  # 已废弃
    'apt.apt_model.utils.hardware_check': 'apt.core.hardware',
    'apt.apt_model.utils.cache_manager': None,  # 已废弃
    'apt.apt_model.utils.common': None,  # 已废弃
    'apt.apt_model.utils': 'apt.core',

    # 插件
    'apt.apt_model.plugins.visual_distillation_plugin': 'apt.apps.plugins.distillation.visual_distillation_plugin',
    'apt.apt_model.plugins.teacher_api': 'apt.apps.plugins.distillation.teacher_api',

    # Codecs（已废弃）
    'apt.apt_model.codecs': None,  # 使用 transformers.AutoTokenizer
}


class ImportFixer:
    """导入路径修复器"""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.stats = {
            'files_scanned': 0,
            'files_modified': 0,
            'imports_fixed': 0,
            'imports_commented': 0
        }

    def fix_file(self, file_path: str) -> bool:
        """修复单个文件"""
        self.stats['files_scanned'] += 1

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"❌ 读取失败 {file_path}: {e}")
            return False

        original_content = content
        modified = False

        # 逐行处理
        lines = content.split('\n')
        new_lines = []

        for line in lines:
            new_line, line_modified = self._fix_line(line, file_path)
            new_lines.append(new_line)
            if line_modified:
                modified = True

        if not modified:
            return False

        new_content = '\n'.join(new_lines)

        if self.dry_run:
            print(f"📝 将修改 {file_path}")
            return True

        # 写入修改
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            self.stats['files_modified'] += 1
            print(f"✅ 已修复 {file_path}")
            return True
        except Exception as e:
            print(f"❌ 写入失败 {file_path}: {e}")
            return False

    def _fix_line(self, line: str, file_path: str) -> Tuple[str, bool]:
        """修复单行导入"""
        original_line = line

        # 匹配 from apt.apt_model.xxx import yyy
        from_pattern = re.compile(r'from\s+(apt\.apt_model\.[a-zA-Z0-9_.]+)\s+import')
        match = from_pattern.search(line)

        if match:
            old_module = match.group(1)

            # 查找映射
            new_module = None
            for old_prefix, new_prefix in IMPORT_MAPPINGS.items():
                if old_module.startswith(old_prefix):
                    if new_prefix is None:
                        # 需要注释掉
                        self.stats['imports_commented'] += 1
                        return f"# DEPRECATED: {line}  # {old_prefix} 已废弃", True
                    else:
                        new_module = old_module.replace(old_prefix, new_prefix, 1)
                        break

            if new_module:
                new_line = line.replace(old_module, new_module)
                self.stats['imports_fixed'] += 1
                return new_line, True

        # 匹配 import apt.apt_model.xxx
        import_pattern = re.compile(r'import\s+(apt\.apt_model\.[a-zA-Z0-9_.]+)')
        match = import_pattern.search(line)

        if match:
            old_module = match.group(1)

            # 查找映射
            new_module = None
            for old_prefix, new_prefix in IMPORT_MAPPINGS.items():
                if old_module.startswith(old_prefix):
                    if new_prefix is None:
                        self.stats['imports_commented'] += 1
                        return f"# DEPRECATED: {line}  # {old_prefix} 已废弃", True
                    else:
                        new_module = old_module.replace(old_prefix, new_prefix, 1)
                        break

            if new_module:
                new_line = line.replace(old_module, new_module)
                self.stats['imports_fixed'] += 1
                return new_line, True

        return line, False

    def fix_directory(self, directory: str, pattern: str = "*.py"):
        """修复整个目录"""
        root_path = Path(directory)

        for file_path in root_path.rglob(pattern):
            # 跳过特殊目录
            if any(skip in str(file_path) for skip in ['__pycache__', '.git', 'archived', '.pytest_cache']):
                continue

            self.fix_file(str(file_path))

    def print_stats(self):
        """打印统计信息"""
        print()
        print("=" * 80)
        print("📊 修复统计")
        print("=" * 80)
        print(f"扫描文件: {self.stats['files_scanned']}")
        print(f"修改文件: {self.stats['files_modified']}")
        print(f"修复导入: {self.stats['imports_fixed']}")
        print(f"注释导入: {self.stats['imports_commented']}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='修复 APT-Transformer 导入路径')
    parser.add_argument('--dry-run', action='store_true', help='预览修改，不实际写入')
    parser.add_argument('--file', type=str, help='修复单个文件')
    parser.add_argument('--dir', type=str, default='apt', help='修复目录（默认: apt）')
    args = parser.parse_args()

    fixer = ImportFixer(dry_run=args.dry_run)

    if args.dry_run:
        print("🔍 预览模式 (--dry-run)")
        print()

    if args.file:
        print(f"修复单个文件: {args.file}")
        fixer.fix_file(args.file)
    else:
        print(f"修复目录: {args.dir}")
        print()
        fixer.fix_directory(args.dir)

    fixer.print_stats()


if __name__ == '__main__':
    main()
