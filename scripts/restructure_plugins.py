#!/usr/bin/env python3
"""
APT-Transformer Plugins 重构脚本

当前问题：
- apt/plugins/         (插件框架)
- apt/apps/plugins/    (具体插件)
- apt_model/plugins/   (重复文件)
- legacy_plugins/      (遗留插件)

重构方案：
1. apt/plugins/ → apt/apps/plugin_system/ (框架层属于 L3)
2. apt/apps/plugins/ 保留 (具体插件实现)
3. 删除 apt_model/plugins/ (完全重复)
4. legacy_plugins/ → archived/legacy_plugins/ (归档)
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent


def restructure_plugins(dry_run=False):
    """执行 plugins 重构"""
    print("=" * 60)
    print("APT-Transformer Plugins 重构")
    print("=" * 60)
    print()

    actions = []

    # ========== 1. 移动插件框架到 L3 ==========
    src_framework = ROOT / "apt" / "plugins"
    dst_framework = ROOT / "apt" / "apps" / "plugin_system"

    if src_framework.exists():
        if dst_framework.exists():
            print("⚠️  目标已存在: apt/apps/plugin_system/")
            print("    跳过框架迁移")
        else:
            actions.append((
                "移动插件框架",
                lambda: shutil.move(str(src_framework), str(dst_framework)),
                f"apt/plugins/ → apt/apps/plugin_system/"
            ))
    else:
        print("⚠️  源不存在: apt/plugins/")

    # ========== 2. 删除重复的插件 ==========
    dup_plugins = ROOT / "apt_model" / "plugins"

    if dup_plugins.exists():
        actions.append((
            "删除重复插件",
            lambda: shutil.rmtree(str(dup_plugins)),
            f"删除 apt_model/plugins/ (8个重复文件)"
        ))
    else:
        print("⚠️  apt_model/plugins/ 不存在，跳过删除")

    # ========== 3. 归档遗留插件 ==========
    legacy = ROOT / "legacy_plugins"
    archived_legacy = ROOT / "archived" / "legacy_plugins"

    if legacy.exists():
        archived_legacy.parent.mkdir(exist_ok=True)

        if archived_legacy.exists():
            print("⚠️  归档目标已存在: archived/legacy_plugins/")
            print("    跳过归档")
        else:
            actions.append((
                "归档遗留插件",
                lambda: shutil.move(str(legacy), str(archived_legacy)),
                f"legacy_plugins/ → archived/legacy_plugins/"
            ))
    else:
        print("⚠️  legacy_plugins/ 不存在，跳过归档")

    print()
    print("=" * 60)
    print(f"计划执行 {len(actions)} 个操作:")
    print("=" * 60)

    for i, (name, func, desc) in enumerate(actions, 1):
        print(f"{i}. {name}")
        print(f"   {desc}")
        print()

    if dry_run:
        print("=" * 60)
        print("这是 DRY RUN 模式，没有实际修改文件。")
        print("执行实际重构，请运行: python scripts/restructure_plugins.py")
        return

    # 执行操作
    print("=" * 60)
    print("开始执行...")
    print("=" * 60)
    print()

    for name, func, desc in actions:
        try:
            func()
            print(f"✓ {name}: {desc}")
        except Exception as e:
            print(f"✗ {name} 失败: {e}")

    print()
    print("=" * 60)
    print("重构完成！")
    print("=" * 60)
    print()
    print("新的插件结构:")
    print("  apt/apps/plugin_system/  - 插件系统框架 (base, manager, hooks)")
    print("  apt/apps/plugins/        - 具体插件实现 (8个插件)")
    print("  archived/legacy_plugins/ - 遗留插件归档")
    print()
    print("apt_model/plugins/ 已删除 (重复)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="APT-Transformer Plugins 重构脚本")
    parser.add_argument("--dry-run", action="store_true", help="仅显示重构计划，不实际执行")

    args = parser.parse_args()

    restructure_plugins(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
