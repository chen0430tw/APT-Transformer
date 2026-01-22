#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GraphRAG插件 - 增强的知识图谱系统

提供:
- 泛图数据结构 (支持高阶关系)
- Hodge-Laplacian谱分析
- 图脑动力学引擎
- 多模式查询 (谱推理、图脑、混合)

集成方式:
1. 独立使用
2. 与现有轻量级KG集成
3. 与API提供商集成 (用于构建知识)

作者: chen0430tw
"""

from typing import Dict, List, Any, Optional
import logging

# 导入GraphRAG核心
from apt.core.graph_rag import (
    GraphRAGManager,
    GeneralizedGraph,
    HodgeLaplacian,
    GraphBrainEngine,
    create_rag_system
)

logger = logging.getLogger(__name__)


class GraphRAGPlugin:
    """
    APT GraphRAG插件

    封装GraphRAG系统，提供与APT项目的无缝集成
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化GraphRAG插件

        Args:
            config: 配置字典
                - max_dimension: 最大维度 (默认2)
                - enable_brain: 启用图脑 (默认True)
                - enable_spectral: 启用谱分析 (默认True)
                - use_api: 是否使用API构建 (默认False)
                - api_provider: API提供商
                - api_key: API密钥
                - api_model: API模型名
        """
        self.config = config or {}
        self.rag = create_rag_system(
            max_dimension=self.config.get('max_dimension', 2),
            enable_brain=self.config.get('enable_brain', True),
            enable_spectral=self.config.get('enable_spectral', True)
        )

        self.api_provider = None
        if self.config.get('use_api', False):
            self._init_api_provider()

        logger.info(f"[GraphRAG] 已初始化 (维度={self.rag.max_dimension})")

    def _init_api_provider(self):
        """初始化API提供商"""
        try:
            from apt.core.api_providers import create_api_provider

            self.api_provider = create_api_provider(
                provider=self.config['api_provider'],
                api_key=self.config['api_key'],
                model_name=self.config.get('api_model')
            )

            logger.info(f"[GraphRAG] API提供商已配置: {self.config['api_provider']}")
        except Exception as e:
            logger.warning(f"[GraphRAG] API提供商初始化失败: {e}")

    def add_triple(self, subject: str, predicate: str, object: str):
        """
        添加三元组

        Args:
            subject: 主语
            predicate: 谓语/关系
            object: 宾语
        """
        self.rag.add_triple(subject, predicate, object)

    def add_triples_batch(self, triples: List[tuple]):
        """
        批量添加三元组

        Args:
            triples: [(s, p, o), ...]
        """
        for s, p, o in triples:
            self.rag.add_triple(s, p, o)

    def build_indices(self):
        """构建索引"""
        logger.info("[GraphRAG] 构建索引中...")
        self.rag.build_indices()
        logger.info("[GraphRAG] 索引构建完成")

    def query(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        查询知识图谱

        Args:
            query: 查询文本
            mode: 查询模式 ('spectral', 'brain', 'hybrid')
            top_k: 返回前K个结果

        Returns:
            查询结果列表
        """
        return self.rag.query(query, mode=mode, top_k=top_k)

    def build_from_documents_with_api(
        self,
        documents: List[str],
        max_triples_per_doc: int = 10
    ):
        """
        使用API从文档构建知识图谱

        Args:
            documents: 文档列表
            max_triples_per_doc: 每个文档最多提取的三元组数
        """
        if not self.api_provider:
            raise ValueError("未配置API提供商，请在config中设置use_api=True")

        logger.info(f"[GraphRAG] 从 {len(documents)} 个文档构建知识图谱...")

        total_triples = 0
        for i, doc in enumerate(documents):
            # 用API提取三元组
            prompt = f"""从以下文本中提取知识三元组，每行一个，格式为: (实体1, 关系, 实体2)

文本: {doc[:500]}

输出格式示例:
(Python, 是, 编程语言)
(Python, 用于, AI开发)

请提取最多{max_triples_per_doc}个三元组:"""

            try:
                response = self.api_provider.generate_text(
                    prompt,
                    max_tokens=300,
                    temperature=0.3
                )

                # 解析三元组 (简单实现)
                triples = self._parse_triples_from_text(response)

                # 添加到图谱
                for s, p, o in triples:
                    self.rag.add_triple(s, p, o)
                    total_triples += 1

                if (i + 1) % 10 == 0:
                    logger.info(f"[GraphRAG] 已处理 {i+1}/{len(documents)} 文档")

            except Exception as e:
                logger.warning(f"[GraphRAG] 文档 {i} 处理失败: {e}")
                continue

        # 构建索引
        self.build_indices()

        logger.info(f"[GraphRAG] 构建完成，提取了 {total_triples} 个三元组")
        logger.info(f"[GraphRAG] API成本: ${self.api_provider.stats['total_cost']:.4f}")

        return total_triples

    def _parse_triples_from_text(self, text: str) -> List[tuple]:
        """
        从文本解析三元组

        简单实现：查找 (X, Y, Z) 格式

        Args:
            text: 包含三元组的文本

        Returns:
            [(s, p, o), ...]
        """
        import re

        triples = []
        pattern = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'

        matches = re.findall(pattern, text)
        for match in matches:
            s, p, o = [x.strip() for x in match]
            if s and p and o:  # 确保都非空
                triples.append((s, p, o))

        return triples

    def integrate_with_lightweight_kg(self, kg):
        """
        与现有轻量级KG集成

        将轻量级KG的三元组导入到GraphRAG

        Args:
            kg: KnowledgeGraph实例 (from apt.apt_model.modeling.knowledge_graph)
        """
        logger.info("[GraphRAG] 与轻量级KG集成...")

        # 导入所有三元组
        count = 0
        for triple in kg.triples:
            self.rag.add_triple(triple.head, triple.relation, triple.tail)
            count += 1

        # 重建索引
        self.build_indices()

        logger.info(f"[GraphRAG] 已导入 {count} 个三元组")

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.rag.get_statistics()

    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()

        print("\n" + "=" * 70)
        print("GraphRAG 统计信息")
        print("=" * 70)
        print(f"  实体数量: {stats.get('num_entities', 0)}")
        print(f"  关系数量: {stats.get('num_relations', 0)}")
        print(f"  事实数量: {stats.get('num_facts', 0)}")

        if 'topology' in stats:
            print(f"\n  拓扑特征:")
            betti = stats['topology'].get('betti_numbers', [])
            if len(betti) > 0:
                print(f"    连通分支: {betti[0]}")
            if len(betti) > 1:
                print(f"    循环数: {betti[1]}")
            if len(betti) > 2:
                print(f"    空洞数: {betti[2]}")

        print("=" * 70)

    def save(self, filepath: str):
        """保存知识图谱"""
        self.rag.save(filepath)
        logger.info(f"[GraphRAG] 已保存到 {filepath}")

    def load(self, filepath: str):
        """加载知识图谱"""
        self.rag.load(filepath)
        logger.info(f"[GraphRAG] 已从 {filepath} 加载")


# ==================== 便捷函数 ====================

def create_graph_rag_plugin(config: Optional[Dict[str, Any]] = None) -> GraphRAGPlugin:
    """
    创建GraphRAG插件的便捷函数

    Args:
        config: 配置字典

    Returns:
        GraphRAGPlugin实例

    Example:
        >>> plugin = create_graph_rag_plugin({'max_dimension': 2})
        >>> plugin.add_triple("Python", "是", "编程语言")
        >>> plugin.build_indices()
        >>> results = plugin.query("Python")
    """
    return GraphRAGPlugin(config)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("【GraphRAG插件演示】\n")

    # 示例1: 基础使用
    print("=" * 60)
    print("[示例1] 基础使用")
    print("=" * 60)

    plugin = create_graph_rag_plugin({
        'max_dimension': 2,
        'enable_brain': True,
        'enable_spectral': True
    })

    # 添加知识
    triples = [
        ("Python", "是", "编程语言"),
        ("Python", "用于", "AI开发"),
        ("PyTorch", "是", "深度学习框架"),
        ("PyTorch", "基于", "Python"),
    ]

    for s, p, o in triples:
        plugin.add_triple(s, p, o)

    # 构建索引
    plugin.build_indices()

    # 查询
    results = plugin.query("Python AI", mode="hybrid", top_k=3)
    print("\n查询结果 'Python AI':")
    for res in results:
        print(f"  - {res['entity']}: {res['score']:.4f}")

    # 统计
    plugin.print_statistics()

    print("\n[提示] 查看 MODULE_INTEGRATION_PLAN.md 了解完整集成方案")
