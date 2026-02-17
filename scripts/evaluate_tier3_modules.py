#!/usr/bin/env python3
"""
APT-Transformer Tier 3 模块评估

评估复杂模块是否应该做插件
"""

from pathlib import Path

ROOT = Path(__file__).parent.parent

# ========== Tier 3 评估 ==========

TIER3_EVALUATION = {
    "hardware": {
        "desc": "硬件模拟和适配插件",
        "modules": [
            {
                "name": "virtual_blackwell",
                "src": "apt/perf/optimization/virtual_blackwell_adapter.py",
                "dst": "virtual_blackwell_plugin.py",
                "reason": "虚拟Blackwell GPU模拟 - 实验性硬件仿真",
                "should_plugin": True,  # ✅ 实验性功能，可选
                "priority": "High",
            },
            {
                "name": "npu_backend",
                "src": "apt/perf/optimization/npu_backend.py",
                "dst": "npu_backend_plugin.py",
                "reason": "NPU加速后端 - 可选硬件支持",
                "should_plugin": True,  # ✅ 特定硬件，可选
                "priority": "Medium",
            },
            {
                "name": "cloud_npu_adapter",
                "src": "apt/perf/optimization/cloud_npu_adapter.py",
                "dst": "cloud_npu_adapter_plugin.py",
                "reason": "云NPU适配器 - 云环境专用",
                "should_plugin": True,  # ✅ 云环境专用，可选
                "priority": "Medium",
            },
        ],
    },
    "deployment": {
        "desc": "部署和虚拟化插件",
        "modules": [
            {
                "name": "microvm_compression",
                "src": "apt/perf/optimization/microvm_compression.py",
                "dst": "microvm_compression_plugin.py",
                "reason": "MicroVM压缩 - 可选部署方案",
                "should_plugin": True,  # ✅ 特定部署场景
                "priority": "Low",
            },
            {
                "name": "vgpu_stack",
                "src": "apt/perf/optimization/vgpu_stack.py",
                "dst": "vgpu_stack_plugin.py",
                "reason": "虚拟GPU管理 - 云/容器环境",
                "should_plugin": True,  # ✅ 虚拟化环境专用
                "priority": "Medium",
            },
        ],
    },
    "memory": {
        "desc": "高级记忆系统插件",
        "modules": [
            {
                "name": "aim_memory",
                "src": "apt/memory/aim/aim_memory.py",
                "dst": "aim_memory_plugin.py",
                "reason": "AIM Memory - 高级上下文记忆系统",
                "should_plugin": True,  # ✅ 可选增强功能
                "priority": "High",
            },
        ],
    },
}

# ========== 不应该做插件的 Tier 3 模块 ==========
NOT_TIER3_PLUGINS = {
    "core_optimization": [
        {
            "file": "apt/perf/optimization/gpu_flash_optimization.py",
            "reason": "GPU Flash优化 - 核心性能优化，不是可选功能",
        },
        {
            "file": "apt/perf/optimization/extreme_scale_training.py",
            "reason": "极端规模训练 - 核心训练能力，不是可选功能",
        },
    ],
    "already_plugins": [
        {
            "file": "apt/apps/plugins/integration/graph_rag_plugin.py",
            "reason": "GraphRAG - 已经是插件了",
        },
    ],
    "core_systems": [
        {
            "file": "apt/memory/knowledge_graph.py",
            "reason": "知识图谱 - L2核心功能，不是可选",
        },
        {
            "file": "apt/core/data/external_data.py",
            "reason": "外部数据加载 - 核心数据能力",
        },
    ],
}


def show_tier3_evaluation():
    """显示 Tier 3 评估结果"""
    print("=" * 80)
    print("APT-Transformer Tier 3 模块评估")
    print("=" * 80)
    print()

    # 应该做插件的
    print("✅ 应该做插件的模块:")
    print("-" * 80)
    total_should = 0

    for category, info in TIER3_EVALUATION.items():
        plugins = [m for m in info['modules'] if m['should_plugin']]
        if plugins:
            print(f"\n📦 {category}/ - {info['desc']}")
            print(f"   模块数: {len(plugins)}")
            for module in plugins:
                total_should += 1
                print(f"   ✓ {module['name']}")
                print(f"      原因: {module['reason']}")
                print(f"      优先级: {module['priority']}")

    print()
    print(f"总计: {total_should} 个模块应该转换为插件")
    print()

    # 不应该做插件的
    print("=" * 80)
    print("❌ 不应该做插件的模块:")
    print("-" * 80)

    for category, items in NOT_TIER3_PLUGINS.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  ❌ {item['file']}")
            print(f"     原因: {item['reason']}")

    print()
    print("=" * 80)
    print("决策依据:")
    print("=" * 80)
    print("""
应该做插件 ✅:
- 实验性硬件模拟（Virtual Blackwell）
- 可选硬件支持（NPU）
- 特定部署场景（MicroVM, vGPU）
- 可选增强功能（AIM Memory）

不应该做插件 ❌:
- 核心性能优化（GPU Flash, Extreme Scale）
- 已经是插件的（GraphRAG）
- 核心系统功能（Knowledge Graph, Data Loading）

建议: Tier 3 转换 6 个模块
    """)


if __name__ == "__main__":
    show_tier3_evaluation()
