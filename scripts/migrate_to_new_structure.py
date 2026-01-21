#!/usr/bin/env python3
"""
APT-Transformer 目录重构迁移脚本

将 apt_model/ 下的文件按照分层架构迁移到：
- apt/core/ (L0 Kernel)
- apt/perf/ (L1 Performance)
- apt/memory/ (L2 Memory)
- apt/apps/ (L3 Product)
"""

import os
import shutil
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).parent.parent

# 迁移映射表
MIGRATIONS = {
    # L0 Kernel: 核心模型和训练
    "L0_core": [
        # modeling
        ("apt_model/modeling/apt_model.py", "apt/core/modeling/apt_model.py"),
        ("apt_model/modeling/multimodal_model.py", "apt/core/modeling/multimodal_model.py"),
        ("apt_model/modeling/hlbd_moe_model.py", "apt/core/modeling/hlbd_moe_model.py"),
        ("apt_model/modeling/embeddings.py", "apt/core/modeling/embeddings.py"),
        ("apt_model/modeling/blocks/", "apt/core/modeling/blocks/"),

        # generation
        ("apt_model/generation/", "apt/core/generation/"),

        # training (基础训练循环)
        ("apt_model/training/trainer.py", "apt/core/training/trainer.py"),
        ("apt_model/training/optimizer.py", "apt/core/training/optimizer.py"),
        ("apt_model/training/data_loading.py", "apt/core/training/data_loading.py"),
        ("apt_model/training/schedules.py", "apt/core/training/schedules.py"),

        # runtime
        ("apt_model/runtime/", "apt/core/runtime/"),

        # config
        ("apt_model/config/", "apt/core/config/"),

        # codecs
        ("apt_model/codecs/", "apt/core/codecs/"),

        # data processing
        ("apt_model/data/", "apt/core/data/"),

        # exceptions
        ("apt_model/exceptions.py", "apt/core/exceptions.py"),
    ],

    # L1 Performance: Virtual Blackwell 和性能优化
    "L1_performance": [
        # optimization (Virtual Blackwell stack)
        ("apt_model/optimization/", "apt/perf/optimization/"),

        # infrastructure (NPU, distributed)
        ("apt_model/infrastructure/", "apt/perf/infrastructure/"),

        # training (性能优化相关)
        ("apt_model/training/mixed_precision.py", "apt/perf/training/mixed_precision.py"),
        ("apt_model/training/checkpoint.py", "apt/perf/training/checkpoint.py"),
        ("apt_model/training/distributed.py", "apt/perf/training/distributed.py"),
    ],

    # L2 Memory: AIM-Memory, GraphRAG
    "L2_memory": [
        # graph_rag
        ("apt_model/core/graph_rag/", "apt/memory/graph_rag/"),

        # knowledge graph
        ("apt_model/modeling/knowledge_graph.py", "apt/memory/knowledge_graph.py"),
        ("apt_model/modeling/kg_rag_integration.py", "apt/memory/kg_rag_integration.py"),
        ("apt_model/modeling/rag_integration.py", "apt/memory/rag_integration.py"),

        # memory module
        ("apt_model/memory/", "apt/memory/aim/"),
    ],

    # L3 Product: WebUI, API, Plugins, Observability
    "L3_product": [
        # webui
        ("apt_model/webui/", "apt/apps/webui/"),

        # api
        ("apt_model/api/", "apt/apps/api/"),

        # cli
        ("apt_model/cli/", "apt/apps/cli/"),

        # interactive
        ("apt_model/interactive/", "apt/apps/cli/interactive/"),

        # console
        ("apt_model/console/", "apt/apps/console/"),

        # plugins
        ("apt_model/plugins/", "apt/apps/plugins/"),

        # agent
        ("apt_model/agent/", "apt/apps/agent/"),

        # evaluation
        ("apt_model/evaluation/", "apt/apps/evaluation/"),

        # rl
        ("apt_model/rl/", "apt/apps/rl/"),
    ],
}


def ensure_parent_dir(filepath):
    """确保父目录存在"""
    parent = filepath.parent
    if not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ 创建目录: {parent}")


def migrate_file(src, dst, dry_run=False):
    """迁移单个文件或目录"""
    src_path = ROOT / src
    dst_path = ROOT / dst

    # 检查源是否存在
    if not src_path.exists():
        print(f"  ⚠️  源不存在，跳过: {src}")
        return False

    # 如果目标已存在，跳过
    if dst_path.exists():
        print(f"  ⚠️  目标已存在，跳过: {dst}")
        return False

    # 确保目标父目录存在
    ensure_parent_dir(dst_path)

    # 执行迁移
    if not dry_run:
        if src_path.is_dir():
            shutil.copytree(src_path, dst_path)
            print(f"  ✓ 迁移目录: {src} → {dst}")
        else:
            shutil.copy2(src_path, dst_path)
            print(f"  ✓ 迁移文件: {src} → {dst}")
    else:
        print(f"  [DRY RUN] {src} → {dst}")

    return True


def migrate_layer(layer_name, migrations, dry_run=False):
    """迁移指定层的所有文件"""
    print(f"\n{'='*60}")
    print(f"开始迁移: {layer_name}")
    print(f"{'='*60}\n")

    success_count = 0
    skip_count = 0

    for src, dst in migrations:
        if migrate_file(src, dst, dry_run):
            success_count += 1
        else:
            skip_count += 1

    print(f"\n{layer_name} 迁移完成:")
    print(f"  ✓ 成功: {success_count}")
    print(f"  ⚠️  跳过: {skip_count}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="APT-Transformer 目录重构迁移脚本")
    parser.add_argument("--dry-run", action="store_true", help="仅显示迁移计划，不实际执行")
    parser.add_argument("--layer", choices=["L0", "L1", "L2", "L3", "all"], default="all",
                        help="指定迁移哪一层 (L0/L1/L2/L3/all)")

    args = parser.parse_args()

    print(f"\nAPT-Transformer 目录重构迁移脚本")
    print(f"根目录: {ROOT}")
    print(f"模式: {'DRY RUN' if args.dry_run else 'EXECUTE'}")
    print(f"层级: {args.layer}")

    # 选择要迁移的层
    if args.layer == "all":
        layers_to_migrate = [
            ("L0_core", MIGRATIONS["L0_core"]),
            ("L1_performance", MIGRATIONS["L1_performance"]),
            ("L2_memory", MIGRATIONS["L2_memory"]),
            ("L3_product", MIGRATIONS["L3_product"]),
        ]
    else:
        layer_map = {
            "L0": ("L0_core", MIGRATIONS["L0_core"]),
            "L1": ("L1_performance", MIGRATIONS["L1_performance"]),
            "L2": ("L2_memory", MIGRATIONS["L2_memory"]),
            "L3": ("L3_product", MIGRATIONS["L3_product"]),
        }
        layers_to_migrate = [layer_map[args.layer]]

    # 执行迁移
    for layer_name, migrations in layers_to_migrate:
        migrate_layer(layer_name, migrations, dry_run=args.dry_run)

    print(f"\n{'='*60}")
    print("迁移完成！")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("这是 DRY RUN 模式，没有实际修改文件。")
        print("执行实际迁移，请运行: python scripts/migrate_to_new_structure.py")


if __name__ == "__main__":
    main()
