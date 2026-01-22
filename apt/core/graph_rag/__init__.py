"""
APT GraphRAG - 基于泛图分析的知识图谱系统

核心模块:
- GeneralizedGraph: 泛图数据结构
- HodgeLaplacian: 谱分析引擎
- GraphBrainEngine: 图脑动力学
- GraphRAGManager: 统一管理接口

理论基础:
- 泛图分析 (GGA)
- Hodge-Laplacian 谱理论
- 非平衡态统计物理
- 认知拓扑动力学

作者: chen0430tw
"""

__version__ = "0.1.0"
__author__ = "chen0430tw"

from .generalized_graph import GeneralizedGraph, Cell
from .hodge_laplacian import HodgeLaplacian
from .graph_brain import GraphBrainEngine, GraphBrainState
from .graph_rag_manager import GraphRAGManager

__all__ = [
    # 核心类
    "GeneralizedGraph",
    "Cell",
    "HodgeLaplacian",
    "GraphBrainEngine",
    "GraphBrainState",
    "GraphRAGManager",
    
    # 版本信息
    "__version__",
    "__author__"
]

# 便捷导入
def create_rag_system(max_dimension=2, enable_brain=True, enable_spectral=True):
    """
    创建GraphRAG系统的便捷函数
    
    Args:
        max_dimension: 最大维度 (0=点, 1=边, 2=面, 3=体)
        enable_brain: 启用图脑动力学
        enable_spectral: 启用谱分析
    
    Returns:
        GraphRAGManager实例
    
    Example:
        >>> import apt_graph_rag as agr
        >>> rag = agr.create_rag_system()
        >>> rag.add_triple("实体1", "关系", "实体2")
        >>> rag.build_indices()
        >>> results = rag.query("实体1")
    """
    return GraphRAGManager(
        max_dimension=max_dimension,
        enable_brain=enable_brain,
        enable_spectral=enable_spectral
    )


# 快速开始示例
def quick_start_example():
    """
    快速开始示例
    
    演示如何使用GraphRAG系统
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("APT GraphRAG - 快速开始示例")
    print("=" * 70)
    
    # 1. 创建系统
    print("\n1. 创建GraphRAG系统...")
    rag = create_rag_system(
        max_dimension=2,
        enable_brain=True,
        enable_spectral=True
    )
    
    # 2. 添加知识
    print("\n2. 添加知识三元组...")
    triples = [
        ("Python", "是", "编程语言"),
        ("Python", "用于", "AI开发"),
        ("深度学习", "需要", "Python"),
        ("PyTorch", "是", "深度学习框架"),
        ("PyTorch", "基于", "Python"),
    ]
    
    for s, p, o in triples:
        rag.add_triple(s, p, o)
    
    # 3. 构建索引
    print("\n3. 构建索引...")
    rag.build_indices()
    
    # 4. 查询
    print("\n4. 执行查询: 'Python AI'")
    results = rag.query("Python AI", mode="hybrid", top_k=5)
    
    print("\n查询结果:")
    for i, res in enumerate(results, 1):
        print(f"  {i}. {res['entity']} (score={res['score']:.4f})")
    
    # 5. 统计
    print("\n5. 系统统计:")
    stats = rag.get_statistics()
    print(f"  实体数: {stats['num_entities']}")
    print(f"  关系数: {stats['num_relations']}")
    print(f"  事实数: {stats['num_facts']}")
    
    if 'topology' in stats:
        betti = stats['topology']['betti_numbers']
        print(f"  Betti数: {betti}")
    
    print("\n" + "=" * 70)
    print("示例完成!")
    print("=" * 70)


if __name__ == "__main__":
    quick_start_example()
