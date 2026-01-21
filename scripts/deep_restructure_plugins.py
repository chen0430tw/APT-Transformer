#!/usr/bin/env python3
"""
APT-Transformer Plugins 深度重构脚本

分析结果：
- Legacy 中有4个重复插件（蒸馏/剪枝已被current包含）
- Legacy 中有3个有价值插件（高级调试/数据处理/多模态）
- 当前8个插件需要按功能分类

深度重构方案：
1. 提取有价值的legacy插件 → apt/apps/plugins/experimental/
2. 删除重复的legacy插件
3. 当前插件按功能分类 → core/integration/distillation/
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ========== 1. 有价值的Legacy插件（保留并现代化）==========
VALUABLE_LEGACY = {
    "plugin_8_advanced_debugging.py": "高级调试 (梯度监控/激活分析)",
    "plugin_7_data_processors.py": "数据处理 (清洗/增强/采样)",
    "plugin_6_multimodal_training.py": "多模态训练",
}

# ========== 2. 重复的Legacy插件（删除）==========
DUPLICATE_LEGACY = {
    "model_distillation_plugin.py": "已被 compression_plugin.py 包含",
    "model_pruning_plugin.py": "已被 compression_plugin.py 包含",
}

# ========== 3. 可选的Legacy插件（评估后决定）==========
OPTIONAL_LEGACY = {
    "cloud_storage_plugin.py": "云存储集成 (S3/Azure/GCS)",
    "huggingface_integration_plugin.py": "HuggingFace Hub 集成",
}

# ========== 4. 当前插件重新分类 ==========
PLUGIN_CLASSIFICATION = {
    "core": {
        "desc": "核心插件 - 训练和优化必需",
        "plugins": [
            "compression_plugin.py",
            "training_monitor_plugin.py",
            "version_manager.py",
        ]
    },
    "integration": {
        "desc": "集成插件 - 外部服务和工具",
        "plugins": [
            "graph_rag_plugin.py",
            "ollama_export_plugin.py",
            "web_search_plugin.py",
        ]
    },
    "distillation": {
        "desc": "蒸馏套件 - 知识蒸馏相关",
        "plugins": [
            "teacher_api.py",
            "visual_distillation_plugin.py",
        ]
    },
}


def deep_restructure_plugins(dry_run=False):
    """执行深度重构"""
    print("=" * 80)
    print("APT-Transformer Plugins 深度重构")
    print("=" * 80)
    print()

    archived = ROOT / "archived" / "legacy_plugins"
    current_plugins = ROOT / "apt" / "apps" / "plugins"
    experimental = current_plugins / "experimental"

    actions = []

    # ========== 阶段1: 提取有价值的Legacy插件 ==========
    print("📦 阶段1: 提取有价值的Legacy插件")
    print("-" * 80)

    if not dry_run:
        experimental.mkdir(exist_ok=True)

    for plugin_name, desc in VALUABLE_LEGACY.items():
        # 尝试从 batch1 和 batch2 中查找
        src = None
        if (archived / "batch2" / plugin_name).exists():
            src = archived / "batch2" / plugin_name
        elif (archived / "batch1" / plugin_name).exists():
            src = archived / "batch1" / plugin_name

        if src:
            dst = experimental / plugin_name
            if dst.exists():
                print(f"  ⚠️  已存在，跳过: {plugin_name}")
            else:
                actions.append((
                    "提取",
                    lambda s=src, d=dst: shutil.copy2(str(s), str(d)),
                    f"提取 {plugin_name} → experimental/ ({desc})"
                ))
                print(f"  ✓ {plugin_name:45} | {desc}")
        else:
            print(f"  ⚠️  未找到: {plugin_name}")

    print()

    # ========== 阶段2: 说明重复的Legacy插件（保持归档状态）==========
    print("📦 阶段2: 重复的Legacy插件（保持归档）")
    print("-" * 80)
    for plugin_name, reason in DUPLICATE_LEGACY.items():
        print(f"  ℹ️  {plugin_name:45} | {reason}")
    print("     → 这些插件保持在 archived/ 中，不做处理")
    print()

    # ========== 阶段3: 当前插件分类重组 ==========
    print("📦 阶段3: 当前插件分类重组")
    print("-" * 80)

    for category, info in PLUGIN_CLASSIFICATION.items():
        print(f"\n  📁 {category}/ - {info['desc']}")

        category_dir = current_plugins / category
        if not dry_run:
            category_dir.mkdir(exist_ok=True)

        for plugin in info['plugins']:
            src = current_plugins / plugin
            dst = category_dir / plugin

            if not src.exists():
                print(f"    ⚠️  源不存在: {plugin}")
                continue

            if dst.exists():
                print(f"    ⚠️  已存在: {plugin}")
                continue

            actions.append((
                "分类",
                lambda s=src, d=dst: shutil.move(str(s), str(d)),
                f"    ✓ {plugin} → {category}/"
            ))

    print()

    # ========== 阶段4: 创建 __init__.py ==========
    if not dry_run:
        for category in PLUGIN_CLASSIFICATION.keys():
            init_file = current_plugins / category / "__init__.py"
            if not init_file.exists():
                actions.append((
                    "创建",
                    lambda f=init_file: f.write_text(
                        f'"""{PLUGIN_CLASSIFICATION[category]["desc"]}"""'
                    ),
                    f"创建 {category}/__init__.py"
                ))

        # 创建 experimental/__init__.py
        exp_init = experimental / "__init__.py"
        if not exp_init.exists():
            actions.append((
                "创建",
                lambda: exp_init.write_text('"""实验性插件 - 从Legacy提取，需要评估和现代化"""'),
                "创建 experimental/__init__.py"
            ))

    # ========== 执行所有操作 ==========
    print("=" * 80)
    print(f"共计划 {len(actions)} 个操作")
    print("=" * 80)
    print()

    if dry_run:
        for action_type, _, desc in actions:
            print(f"  [{action_type}] {desc}")
        print()
        print("这是 DRY RUN 模式，没有实际修改文件。")
        print("执行实际重构，请运行: python scripts/deep_restructure_plugins.py")
        return

    print("开始执行...")
    print()

    for action_type, func, desc in actions:
        try:
            func()
            print(f"✓ {desc}")
        except Exception as e:
            print(f"✗ {desc} 失败: {e}")

    print()
    print("=" * 80)
    print("深度重构完成！")
    print("=" * 80)
    print()
    print("新的插件结构:")
    print("""
    apt/apps/plugins/
    ├── core/              ✅ 核心插件 (3个)
    │   ├── compression_plugin.py
    │   ├── training_monitor_plugin.py
    │   └── version_manager.py
    ├── integration/       ✅ 集成插件 (3个)
    │   ├── graph_rag_plugin.py
    │   ├── ollama_export_plugin.py
    │   └── web_search_plugin.py
    ├── distillation/      ✅ 蒸馏套件 (2个)
    │   ├── teacher_api.py
    │   └── visual_distillation_plugin.py
    └── experimental/      ✅ 实验性插件 (3个 from legacy)
        ├── plugin_8_advanced_debugging.py
        ├── plugin_7_data_processors.py
        └── plugin_6_multimodal_training.py

    archived/legacy_plugins/  ✅ 归档 (重复插件保持不变)
    """)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="APT-Transformer Plugins 深度重构脚本")
    parser.add_argument("--dry-run", action="store_true", help="仅显示重构计划，不实际执行")

    args = parser.parse_args()

    deep_restructure_plugins(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
