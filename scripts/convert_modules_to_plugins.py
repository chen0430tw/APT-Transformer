#!/usr/bin/env python3
"""
APT-Transformer 模块转插件脚本

将核心模块转换为插件，提升架构清晰度和可选性。

分层转换计划:
- Tier 1 (立即转换): 6个高价值低成本模块
- Tier 2 (中期转换): 15个高价值中成本模块
- Tier 3 (长期转换): 12个复杂研究模块
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ========== Tier 1: 立即转换 (高价值，低成本) ==========
TIER1_CONVERSIONS = {
    "monitoring": {
        "desc": "监控和诊断插件",
        "modules": [
            {
                "name": "gradient_monitor",
                "src": "apt_model/training/gradient_monitor.py",
                "dst": "gradient_monitor_plugin.py",
                "reason": "梯度监控 - 可选调试功能",
            },
            {
                "name": "resource_monitor",
                "src": "apt_model/utils/resource_monitor.py",
                "dst": "resource_monitor_plugin.py",
                "reason": "资源监控 - 可选系统监控",
            },
        ],
    },
    "visualization": {
        "desc": "可视化插件",
        "modules": [
            {
                "name": "model_visualization",
                "src": "apt_model/utils/visualization.py",
                "dst": "model_visualization_plugin.py",
                "reason": "模型可视化 - 可选分析工具",
            },
        ],
    },
    "evaluation": {
        "desc": "评估和基准测试插件",
        "modules": [
            {
                "name": "model_evaluator",
                "src": "apt/apps/evaluation/model_evaluator.py",
                "dst": "model_evaluator_plugin.py",
                "reason": "模型评估 - 可选基准测试",
            },
            {
                "name": "model_comparison",
                "src": "apt/apps/evaluation/comparison.py",
                "dst": "model_comparison_plugin.py",
                "reason": "模型对比 - 可选分析工具",
            },
        ],
    },
    "infrastructure": {
        "desc": "基础设施插件",
        "modules": [
            {
                "name": "logging",
                "src": "apt/perf/infrastructure/logging.py",
                "dst": "logging_plugin.py",
                "reason": "集中式日志 - 可选增强",
            },
        ],
    },
}

# ========== Tier 2: 中期转换 (高价值，中等成本) ==========
# 注意: 只转换真正应该是插件的模块
# - APX转换器是打包工具，应保持为工具
# - 数据处理/管道是核心功能，应保持为模块
TIER2_CONVERSIONS = {
    "optimization": {
        "desc": "性能优化插件",
        "modules": [
            {
                "name": "mxfp4_quantization",
                "src": "apt/perf/optimization/mxfp4_quantization.py",
                "dst": "mxfp4_quantization_plugin.py",
                "reason": "MXFP4量化 - 可选优化技术",
            },
        ],
    },
    "rl": {
        "desc": "强化学习插件 - 可选的对齐训练方法",
        "modules": [
            {
                "name": "rlhf_trainer",
                "src": "apt/apps/rl/rlhf_trainer.py",
                "dst": "rlhf_trainer_plugin.py",
                "reason": "RLHF训练 - 可选对齐方法",
            },
            {
                "name": "dpo_trainer",
                "src": "apt/apps/rl/dpo_trainer.py",
                "dst": "dpo_trainer_plugin.py",
                "reason": "DPO训练 - 可选对齐方法",
            },
            {
                "name": "grpo_trainer",
                "src": "apt/apps/rl/grpo_trainer.py",
                "dst": "grpo_trainer_plugin.py",
                "reason": "GRPO训练 - 可选对齐方法",
            },
            {
                "name": "reward_model",
                "src": "apt/apps/rl/reward_model.py",
                "dst": "reward_model_plugin.py",
                "reason": "奖励模型 - RL训练工具",
            },
        ],
    },
    "protocol": {
        "desc": "协议集成插件 - 外部协议支持",
        "modules": [
            {
                "name": "mcp_integration",
                "src": "apt_model/modeling/mcp_integration.py",
                "dst": "mcp_integration_plugin.py",
                "reason": "MCP协议 - 外部协议集成",
            },
        ],
    },
    "retrieval": {
        "desc": "检索增强插件 - 可选的RAG功能",
        "modules": [
            {
                "name": "rag_integration",
                "src": "apt_model/modeling/rag_integration.py",
                "dst": "rag_integration_plugin.py",
                "reason": "RAG集成 - 可选检索增强",
            },
            {
                "name": "kg_rag_integration",
                "src": "apt_model/modeling/kg_rag_integration.py",
                "dst": "kg_rag_integration_plugin.py",
                "reason": "KG+RAG融合 - 可选知识图谱检索",
            },
        ],
    },
}

# ========== 不应转换为插件的模块 ==========
# 这些应该保持为模块/工具
NOT_PLUGINS = {
    "tools": [
        "apt_model/tools/apx/converter.py",  # 打包工具，不是插件
    ],
    "core_modules": [
        "apt/core/data/data_processor.py",  # 核心数据处理
        "apt/core/data/pipeline.py",  # 核心数据管道
    ],
    "infrastructure": [
        # 某些基础设施应该保持在 apt/perf 或 apt/core
    ],
}


def convert_tier1_modules(dry_run=False):
    """转换 Tier 1 模块为插件"""
    print("=" * 80)
    print("APT-Transformer 模块转插件 - Tier 1 (立即转换)")
    print("=" * 80)
    print()

    plugins_root = ROOT / "apt" / "apps" / "plugins"
    actions = []
    total_modules = 0

    for category, info in TIER1_CONVERSIONS.items():
        print(f"📦 类别: {category}/")
        print(f"   描述: {info['desc']}")
        print(f"   模块数: {len(info['modules'])}")
        print()

        category_dir = plugins_root / category

        if not dry_run:
            category_dir.mkdir(exist_ok=True)

        for module in info['modules']:
            src_path = ROOT / module['src']
            dst_path = category_dir / module['dst']

            total_modules += 1

            if not src_path.exists():
                print(f"  ⚠️  源文件不存在，跳过: {module['src']}")
                continue

            if dst_path.exists():
                print(f"  ⚠️  目标已存在，跳过: {module['dst']}")
                continue

            actions.append({
                'type': 'copy',
                'src': src_path,
                'dst': dst_path,
                'name': module['name'],
                'category': category,
                'reason': module['reason'],
            })

            print(f"  ✓ {module['name']}")
            print(f"     原始: {module['src']}")
            print(f"     目标: plugins/{category}/{module['dst']}")
            print(f"     原因: {module['reason']}")
            print()

        # 创建 __init__.py
        if not dry_run:
            init_file = category_dir / "__init__.py"
            if not init_file.exists():
                actions.append({
                    'type': 'create_init',
                    'path': init_file,
                    'category': category,
                    'desc': info['desc'],
                })

    # 汇总
    print("=" * 80)
    print(f"转换汇总:")
    print(f"  总模块数: {total_modules}")
    print(f"  计划操作: {len([a for a in actions if a['type'] == 'copy'])}")
    print(f"  新建类别: {len(TIER1_CONVERSIONS)}")
    print("=" * 80)
    print()

    if dry_run:
        print("这是 DRY RUN 模式，没有实际修改文件。")
        print("执行实际转换，请运行: python scripts/convert_modules_to_plugins.py")
        return

    # 执行操作
    print("开始执行转换...")
    print()

    success_count = 0
    fail_count = 0

    for action in actions:
        try:
            if action['type'] == 'copy':
                shutil.copy2(str(action['src']), str(action['dst']))
                print(f"✓ 已转换: {action['name']} → plugins/{action['category']}/")
                success_count += 1
            elif action['type'] == 'create_init':
                action['path'].write_text(
                    f'"""{action["desc"]}"""\n'
                )
                print(f"✓ 已创建: plugins/{action['category']}/__init__.py")
        except Exception as e:
            print(f"✗ 失败: {action.get('name', action.get('category', 'unknown'))} - {e}")
            fail_count += 1

    print()
    print("=" * 80)
    print("Tier 1 转换完成！")
    print("=" * 80)
    print()
    print(f"✓ 成功: {success_count}")
    print(f"✗ 失败: {fail_count}")
    print()
    print("新的插件结构:")
    print("""
    apt/apps/plugins/
    ├── core/              (3 plugins) - 核心插件
    ├── integration/       (3 plugins) - 集成插件
    ├── distillation/      (2 plugins) - 蒸馏插件
    ├── experimental/      (3 plugins) - 实验插件
    ├── monitoring/        (2 plugins) - 监控插件 ✨ NEW
    ├── visualization/     (1 plugin)  - 可视化插件 ✨ NEW
    ├── evaluation/        (2 plugins) - 评估插件 ✨ NEW
    └── infrastructure/    (1 plugin)  - 基础设施插件 ✨ NEW
    """)
    print()
    print("下一步:")
    print("  1. 运行测试: pytest tests/")
    print("  2. 更新导入: 检查受影响的模块")
    print("  3. 提交更改: git commit")


def show_tier2_plan():
    """显示 Tier 2 转换计划"""
    print()
    print("=" * 80)
    print("📋 Tier 2 转换计划 (中期转换 - 高价值，中等成本)")
    print("=" * 80)
    print()

    total = 0
    for category, info in TIER2_CONVERSIONS.items():
        print(f"  📁 {category}/ - {info['desc']}")
        print(f"     模块数: {len(info['modules'])}")
        for module in info['modules']:
            print(f"       • {module['name']}: {module['reason']}")
            total += 1
        print()

    print(f"总计: {total} 个模块")
    print()
    print("运行 Tier 2 转换:")
    print("  python scripts/convert_modules_to_plugins.py --tier2")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="APT-Transformer 模块转插件脚本")
    parser.add_argument("--dry-run", action="store_true", help="仅显示转换计划，不实际执行")
    parser.add_argument("--tier2", action="store_true", help="显示 Tier 2 转换计划")

    args = parser.parse_args()

    if args.tier2:
        show_tier2_plan()
    else:
        convert_tier1_modules(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
