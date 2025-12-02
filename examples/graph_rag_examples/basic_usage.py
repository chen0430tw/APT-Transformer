#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GraphRAG基础使用示例

演示如何使用GraphRAG插件构建和查询知识图谱
"""

import logging
logging.basicConfig(level=logging.INFO)

from apt_model.plugins.graph_rag_plugin import create_graph_rag_plugin


def main():
    print("=" * 70)
    print("GraphRAG 基础使用示例")
    print("=" * 70)

    # 1. 创建GraphRAG插件
    print("\n[步骤1] 创建GraphRAG插件...")
    plugin = create_graph_rag_plugin({
        'max_dimension': 2,
        'enable_brain': True,
        'enable_spectral': True
    })

    # 2. 添加知识三元组
    print("\n[步骤2] 添加知识三元组...")
    triples = [
        # AI基础
        ("Python", "是", "编程语言"),
        ("Python", "用于", "AI开发"),
        ("Python", "特点", "简洁易学"),

        # 深度学习框架
        ("PyTorch", "是", "深度学习框架"),
        ("PyTorch", "基于", "Python"),
        ("TensorFlow", "是", "深度学习框架"),
        ("TensorFlow", "基于", "Python"),

        # APT相关
        ("APT", "是", "模型架构"),
        ("APT", "使用", "PyTorch"),
        ("APT", "支持", "知识图谱"),

        # 知识图谱
        ("知识图谱", "用于", "知识表示"),
        ("知识图谱", "包含", "实体和关系"),
        ("APT", "集成", "知识图谱"),
    ]

    for s, p, o in triples:
        plugin.add_triple(s, p, o)

    print(f"  已添加 {len(triples)} 个三元组")

    # 3. 构建索引
    print("\n[步骤3] 构建索引...")
    plugin.build_indices()

    # 4. 查询
    print("\n[步骤4] 执行查询...")

    queries = [
        ("Python AI", "hybrid"),
        ("深度学习框架", "spectral"),
        ("APT 知识图谱", "brain"),
    ]

    for query, mode in queries:
        print(f"\n  查询: '{query}' (模式={mode})")
        results = plugin.query(query, mode=mode, top_k=5)

        for i, res in enumerate(results, 1):
            print(f"    {i}. {res['entity']}: {res['score']:.4f}")

    # 5. 统计信息
    print("\n[步骤5] 查看统计信息...")
    plugin.print_statistics()

    # 6. 保存
    print("\n[步骤6] 保存知识图谱...")
    plugin.save("./graph_rag_demo.pkl")
    print("  已保存到 graph_rag_demo.pkl")

    print("\n" + "=" * 70)
    print("示例完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
