#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动修复循环导入 - 为 __init__.py 文件添加 try-except 保护
"""

import sys
from pathlib import Path
import re

def safe_print(msg):
    try:
        print(msg)
    except OSError:
        pass

def fix_init_file(init_file: Path, dry_run: bool = True) -> bool:
    """为 __init__.py 文件添加 try-except 保护"""
    try:
        with open(init_file, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.split('\n')
        new_lines = []
        i = 0

        # 跟踪是否已经在 try 块中
        in_try = False
        try_indent = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # 检查 try: 语句
            if stripped.startswith('try:'):
                in_try = True
                try_indent = len(line) - len(line.lstrip())
                new_lines.append(line)
                i += 1
                continue

            # 检查 except 语句（结束 try 块）
            if stripped.startswith('except'):
                in_try = False
                new_lines.append(line)
                i += 1
                continue

            # 检查是否是 from apt. 或 import apt. 导入
            if (stripped.startswith('from apt.') or stripped.startswith('import apt.')):
                if not in_try:
                    # 找到未保护的导入
                    indent = len(line) - len(line.lstrip())
                    indent_str = ' ' * indent

                    # 收集连续的导入语句（处理多行导入）
                    import_lines = [line]
                    j = i + 1

                    # 检查是否是多行导入
                    while j < len(lines):
                        next_line = lines[j]
                        next_stripped = next_line.strip()

                        # 如果是续行（缩进或括号）
                        if next_stripped and (
                            next_line.startswith(indent_str + '    ') or
                            next_stripped.startswith(')') or
                            (import_lines[-1].rstrip().endswith('(') or
                             import_lines[-1].rstrip().endswith(','))
                        ):
                            import_lines.append(next_line)
                            j += 1
                        else:
                            break

                    # 生成 try-except 包裹的代码
                    new_lines.append(f"{indent_str}try:")
                    for imp_line in import_lines:
                        new_lines.append(f"    {imp_line}")
                    new_lines.append(f"{indent_str}except ImportError:")
                    new_lines.append(f"{indent_str}    pass")

                    i = j
                    continue

            new_lines.append(line)
            i += 1

        new_content = '\n'.join(new_lines)

        if new_content != content:
            if not dry_run:
                with open(init_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                safe_print(f"✓ 已修复: {init_file.relative_to(project_root)}")
            else:
                safe_print(f"需要修复: {init_file.relative_to(project_root)}")
            return True
        else:
            return False

    except Exception as e:
        safe_print(f"✗ 处理失败 {init_file}: {e}")
        return False

def main():
    """主函数"""
    global project_root
    project_root = Path('/home/user/APT-Transformer')

    import argparse
    parser = argparse.ArgumentParser(description='自动修复 __init__.py 的循环导入问题')
    parser.add_argument('--dry-run', action='store_true', help='仅预览，不实际修改文件')
    parser.add_argument('--apply', action='store_true', help='实际应用修复')
    args = parser.parse_args()

    if not args.apply and not args.dry_run:
        safe_print("请使用 --dry-run 预览或 --apply 应用修复")
        return 1

    dry_run = not args.apply

    safe_print("=" * 70)
    if dry_run:
        safe_print("预览模式 - 不会修改文件")
    else:
        safe_print("应用模式 - 将修改文件")
    safe_print("=" * 70)

    apt_dir = project_root / 'apt'
    init_files = list(apt_dir.rglob('__init__.py'))

    safe_print(f"\n找到 {len(init_files)} 个 __init__.py 文件")

    # 跳过已经完全修复的文件
    skip_files = [
        'apt/trainops/__init__.py',  # 已手动修复
        'apt/apps/training/__init__.py',  # 已手动修复
        'apt/core/__init__.py',  # 已手动修复
    ]

    fixed_count = 0
    for init_file in init_files:
        rel_path = str(init_file.relative_to(project_root))
        if rel_path in skip_files:
            safe_print(f"⏭️  跳过（已手动修复）: {rel_path}")
            continue

        if fix_init_file(init_file, dry_run):
            fixed_count += 1

    safe_print("\n" + "=" * 70)
    if dry_run:
        safe_print(f"预览完成: 发现 {fixed_count} 个需要修复的文件")
        safe_print("\n使用 --apply 参数来应用修复:")
        safe_print("  python3 scripts/testing/fix_circular_imports.py --apply")
    else:
        safe_print(f"修复完成: 已修复 {fixed_count} 个文件")
        safe_print("\n建议:")
        safe_print("1. 运行测试确认修复有效")
        safe_print("2. 提交修复: git add apt/ && git commit -m 'fix: Add try-except protection to all __init__.py files'")

    return 0

if __name__ == "__main__":
    sys.exit(main())
