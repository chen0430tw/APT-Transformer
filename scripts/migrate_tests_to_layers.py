#!/usr/bin/env python3
"""
APT-Transformer 测试分层迁移脚本

将 tests/ 根目录下的 30 个测试文件按照 L0/L1/L2/L3 归类到：
- tests/l0_kernel/       (L0 核心)
- tests/l1_performance/  (L1 性能)
- tests/l2_memory/       (L2 记忆)
- tests/l3_product/      (L3 应用)
- tests/integration/     (集成测试)
"""

import shutil
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).parent.parent
TESTS = ROOT / "tests"

# 测试分类映射表（根据测试内容）
TEST_CLASSIFICATION = {
    # ========== L0 Kernel (核心模型和训练) ==========
    "l0_kernel": [
        "test_small_apt_model.py",
        "test_trainer_complete.py",
        "test_smoke.py",
        "test_dbc_optimization.py",
        "test_dbc_optimization_numpy.py",
        "test_dbc_acceleration.py",
        "test_hlbd_quick_learning.py",
        "test_vft_tva.py",
        "test_multimodal.py",
        "test_multilingual.py",
        "test_core_imports.py",
        "test_bert_tokenizer.py",
        "test_callbacks.py",
        "test_legacy_adapters.py",
    ],

    # ========== L1 Performance (性能优化) ==========
    "l1_performance": [
        "test_virtual_blackwell.py",
        "test_vb_basic.py",
        "test_compression_minimal.py",
        "test_compression_plugin.py",
        "test_compression_plugins.py",
    ],

    # ========== L2 Memory (记忆系统) ==========
    "l2_memory": [
        # 目前没有专门的 L2 测试，待补充
    ],

    # ========== L3 Product (应用层) ==========
    "l3_product": [
        "test_plugin_system.py",
        "test_plugin_system_standalone.py",
        "test_plugin_version_manager.py",
        "test_console.py",
        "test_admin_mode_structure.py",
    ],

    # ========== Integration (集成测试) ==========
    "integration": [
        "test_error_persistence.py",
        "test_terminator_logic.py",
        "test_terminator_scenario.py",
    ],
}

# 保留在根目录的特殊文件
KEEP_IN_ROOT = [
    "conftest.py",
    "README.md",
    "README_HLBD_MODELS.md",
    "delete_old_models.py",
    "load_hlbd_model.py",
]


def migrate_tests(dry_run=False):
    """执行测试迁移"""
    print("=" * 60)
    print("APT-Transformer 测试分层迁移")
    print("=" * 60)
    print()

    total_migrated = 0
    total_skipped = 0

    for category, tests in TEST_CLASSIFICATION.items():
        if not tests:
            print(f"📁 {category}/ (空，跳过)")
            continue

        print(f"📁 迁移到 tests/{category}/")

        target_dir = TESTS / category
        target_dir.mkdir(exist_ok=True)

        for test in tests:
            src = TESTS / test
            dst = target_dir / test

            if not src.exists():
                print(f"  ⚠️  源文件不存在，跳过: {test}")
                total_skipped += 1
                continue

            if dst.exists():
                print(f"  ⚠️  目标已存在，跳过: {test}")
                total_skipped += 1
                continue

            if dry_run:
                print(f"  [DRY RUN] {test} → {category}/")
            else:
                shutil.move(str(src), str(dst))
                print(f"  ✓ {test} → {category}/")

            total_migrated += 1

        print()

    # 汇总统计
    print("=" * 60)
    print(f"迁移完成:")
    print(f"  ✓ 成功: {total_migrated}")
    print(f"  ⚠️  跳过: {total_skipped}")
    print(f"  📌 保留在根目录: {len(KEEP_IN_ROOT)}")
    print("=" * 60)

    if dry_run:
        print("\n这是 DRY RUN 模式，没有实际修改文件。")
        print("执行实际迁移，请运行: python scripts/migrate_tests_to_layers.py")
    else:
        print("\n✅ 测试已按层级重新组织！")
        print("\n新的测试结构:")
        print("  tests/l0_kernel/       - L0 核心算法测试")
        print("  tests/l1_performance/  - L1 性能优化测试")
        print("  tests/l2_memory/       - L2 记忆系统测试")
        print("  tests/l3_product/      - L3 应用层测试")
        print("  tests/integration/     - 集成测试")
        print()
        print("运行分层测试:")
        print("  pytest tests/l0_kernel/       # 只测试核心")
        print("  pytest tests/l1_performance/  # 只测试性能")
        print("  pytest tests/l3_product/      # 只测试应用")
        print("  pytest tests/integration/     # 只测试集成")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="APT-Transformer 测试分层迁移脚本")
    parser.add_argument("--dry-run", action="store_true", help="仅显示迁移计划，不实际执行")

    args = parser.parse_args()

    migrate_tests(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
