#!/usr/bin/env python3
"""
APT-Transformer 文档分层迁移脚本

将 docs/ 根目录下的 44 个文档按照 L0/L1/L2/L3 归类到：
- docs/kernel/       (L0 核心)
- docs/performance/  (L1 性能)
- docs/memory/       (L2 记忆)
- docs/product/      (L3 应用)
- docs/guides/       (跨层指南)
"""

import shutil
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).parent.parent
DOCS = ROOT / "docs"

# 文档分类映射表（根据内容和职责）
DOC_CLASSIFICATION = {
    # ========== L0 Kernel (核心算法) ==========
    "kernel": [
        "APT_MODEL_HANDBOOK.md",
        "DBC_DAC_OPTIMIZATION_GUIDE.md",
        "LEFT_SPIN_SMOOTH_INTEGRATION.md",
        "DATA_PREPROCESSING_GUIDE.md",
        "CONTEXT_AND_ROPE_OPTIMIZATION.md",
        "DEEPSEEK_TRAINING_GUIDE.md",
        "FINE_TUNING_GUIDE.md",
        "HLBD.md",
        "DEBUG_MODE_GUIDE.md",
    ],

    # ========== L1 Performance (性能优化) ==========
    "performance": [
        "VIRTUAL_BLACKWELL_COMPLETE_GUIDE.md",
        "VIRTUAL_BLACKWELL_COMPLETE.md",
        "VGPU_STACK_ARCHITECTURE.md",
        "VGPU_QUICK_START.md",
        "ENABLE_VIRTUAL_BLACKWELL.md",
        "GPU_FLASH_OPTIMIZATION_GUIDE.txt",
        "GPU_FLASH_SUCCESS_ANALYSIS.md",
        "CLOUD_NPU_GUIDE.md",
        "NPU_INTEGRATION_GUIDE.md",
        "EXTREME_OPTIMIZATIONS_GUIDE.md",
        "ELASTIC_APT_INTEGRATION.md",
        "TRAINING_BACKENDS.md",
    ],

    # ========== L2 Memory (记忆系统) ==========
    "memory": [
        "AIM_MEMORY_GUIDE.md",
        "AIM_NC_GUIDE.md",
        "HIERARCHICAL_MEMORY_GUIDE.md",
        "MEMORY_SYSTEM_GUIDE.md",
        "GRAPH_BRAIN_TRAINING_GUIDE.md",
        "KNOWLEDGE_GRAPH_GUIDE.md",
    ],

    # ========== L3 Product (应用层) ==========
    "product": [
        "PLUGIN_SYSTEM_GUIDE.md",
        "AGENT_SYSTEM_GUIDE.md",
        "WEB_SEARCH_PLUGIN_GUIDE.md",
        "VISUALIZATION_GUIDE.md",
        "VISUAL_DISTILLATION_GUIDE.md",
        "DISTILLATION_PRINCIPLE.md",
        "TEACHER_API_GUIDE.md",
        "MCP_INTEGRATION_GUIDE.md",
        "LAUNCHER_README.md",
        "API_PROVIDERS_GUIDE.md",
        "CLAUDE4_MODEL_GUIDE.md",
        "GPT_MODELS_GUIDE.md",
        "RL_PRETRAINING_GUIDE.md",
        "OPTUNA_GUIDE.md",
    ],

    # ========== Guides (跨层指南) ==========
    "guides": [
        "COMPLETE_TECH_SUMMARY.md",
        "INTEGRATION_SUMMARY.md",
        "APX.md",
        "README.md",
    ],
}


def migrate_docs(dry_run=False):
    """执行文档迁移"""
    print("=" * 60)
    print("APT-Transformer 文档分层迁移")
    print("=" * 60)
    print()

    total_migrated = 0
    total_skipped = 0

    for category, docs in DOC_CLASSIFICATION.items():
        print(f"📁 迁移到 docs/{category}/")

        target_dir = DOCS / category
        target_dir.mkdir(exist_ok=True)

        for doc in docs:
            src = DOCS / doc
            dst = target_dir / doc

            if not src.exists():
                print(f"  ⚠️  源文件不存在，跳过: {doc}")
                total_skipped += 1
                continue

            if dst.exists():
                print(f"  ⚠️  目标已存在，跳过: {doc}")
                total_skipped += 1
                continue

            if dry_run:
                print(f"  [DRY RUN] {doc} → {category}/")
            else:
                shutil.move(str(src), str(dst))
                print(f"  ✓ {doc} → {category}/")

            total_migrated += 1

        print()

    # 汇总统计
    print("=" * 60)
    print(f"迁移完成:")
    print(f"  ✓ 成功: {total_migrated}")
    print(f"  ⚠️  跳过: {total_skipped}")
    print("=" * 60)

    if dry_run:
        print("\n这是 DRY RUN 模式，没有实际修改文件。")
        print("执行实际迁移，请运行: python scripts/migrate_docs_to_layers.py")
    else:
        print("\n✅ 文档已按层级重新组织！")
        print("\n新的文档结构:")
        print("  docs/kernel/      - L0 核心算法文档")
        print("  docs/performance/ - L1 性能优化文档")
        print("  docs/memory/      - L2 记忆系统文档")
        print("  docs/product/     - L3 应用层文档")
        print("  docs/guides/      - 跨层指南和总结")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="APT-Transformer 文档分层迁移脚本")
    parser.add_argument("--dry-run", action="store_true", help="仅显示迁移计划，不实际执行")

    args = parser.parse_args()

    migrate_docs(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
