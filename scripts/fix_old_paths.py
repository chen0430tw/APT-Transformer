#!/usr/bin/env python3
"""
批量修复旧路径引用

将项目中的旧路径引用替换为新路径：
- apt_model.core → apt.core
- apt_model.perf → apt.perf
- apt_model.memory → apt.memory
- apt_model.apps → apt.apps

作者: APT-Transformer Team
日期: 2026-01-22
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).parent.parent

# 路径映射 (只映射真正需要替换的旧路径)
PATH_MAPPINGS = {
    'apt_model.core': 'apt.core',
    'apt_model.perf': 'apt.perf',
    'apt_model.memory': 'apt.memory',
    'apt_model.apps': 'apt.apps',
}


def fix_file(filepath: Path, dry_run=False) -> Tuple[int, List[str]]:
    """
    修复单个文件的旧路径引用

    Returns:
        (替换次数, 替换详情列表)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        return 0, []

    original_content = content
    replacements = []

    for old_path, new_path in PATH_MAPPINGS.items():
        if old_path in content:
            # 计算替换次数
            count = content.count(old_path)

            # 执行替换
            content = content.replace(old_path, new_path)

            replacements.append(f"{old_path} → {new_path} ({count} 处)")

    if content != original_content:
        if not dry_run:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
        return len(replacements), replacements

    return 0, []


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='批量修复旧路径引用')
    parser.add_argument('--dry-run', action='store_true',
                       help='只显示需要修复的内容，不实际修改文件')
    parser.add_argument('--path', type=str, default=None,
                       help='指定要修复的路径（默认：整个项目）')
    args = parser.parse_args()

    print("=" * 70)
    print("APT-Transformer 旧路径引用修复工具")
    print("=" * 70)

    if args.dry_run:
        print("\n⚠️  DRY RUN 模式 - 不会实际修改文件\n")

    # 确定搜索路径
    if args.path:
        search_path = ROOT / args.path
        if not search_path.exists():
            print(f"错误: 路径不存在: {search_path}")
            return 1
    else:
        search_path = ROOT

    # 查找所有 Python 文件
    py_files = list(search_path.rglob('*.py'))
    py_files = [f for f in py_files if '__pycache__' not in str(f)]

    print(f"扫描 {len(py_files)} 个文件...\n")

    # 修复文件
    fixed_files = []
    total_replacements = 0

    for filepath in py_files:
        count, replacements = fix_file(filepath, dry_run=args.dry_run)

        if count > 0:
            fixed_files.append((filepath, replacements))
            total_replacements += count

    # 显示结果
    if fixed_files:
        print(f"{'=' * 70}")
        print(f"发现 {len(fixed_files)} 个文件需要修复:")
        print(f"{'=' * 70}\n")

        for filepath, replacements in fixed_files:
            print(f"📄 {filepath.relative_to(ROOT)}")
            for replacement in replacements:
                print(f"   • {replacement}")
            print()

        print(f"{'=' * 70}")
        if args.dry_run:
            print(f"⚠️  DRY RUN: 将修复 {len(fixed_files)} 个文件，共 {total_replacements} 处引用")
            print(f"\n运行 'python scripts/fix_old_paths.py' 以实际修复")
        else:
            print(f"✓ 已修复 {len(fixed_files)} 个文件，共 {total_replacements} 处引用")
        print(f"{'=' * 70}")

        return 0
    else:
        print("✓ 未发现需要修复的文件")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
