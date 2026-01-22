#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
知识图谱检索Provider

与现有retrieval系统集成，提供基于KG的检索能力
"""

from apt_model.utils.fake_torch import get_torch
torch = get_torch()
from apt_model.utils.fake_torch import get_torch
torch = get_torch()
nn = torch.nn
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from apt.core.base_provider import BaseProvider
from apt_model.modeling.knowledge_graph import KnowledgeGraph, Triple


class KGRetriever(nn.Module):
    """
    知识图谱检索器

    支持：
    - 实体检索
    - 关系检索
    - 子图检索
    - 多跳推理
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        d_model: int = 768,
        max_hops: int = 2,
        top_k: int = 5
    ):
        super().__init__()
        self.kg = kg
        self.d_model = d_model
        self.max_hops = max_hops
        self.top_k = top_k

        # 实体到ID的映射
        self.entity2id = {entity: idx for idx, entity in enumerate(kg.entities)}
        self.id2entity = {idx: entity for entity, idx in self.entity2id.items()}

        # 关系到ID的映射
        self.relation2id = {rel: idx for idx, rel in enumerate(kg.relations)}
        self.id2relation = {idx: rel for rel, idx in self.relation2id.items()}

        # 实体嵌入（可训练）
        num_entities = len(kg.entities)
        num_relations = len(kg.relations)

        self.entity_embeddings = nn.Embedding(num_entities, d_model)
        self.relation_embeddings = nn.Embedding(num_relations, d_model)

        # 初始化
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

    def encode_entity(self, entity: str) -> Optional[torch.Tensor]:
        """编码实体为向量"""
        if entity not in self.entity2id:
            return None

        entity_id = self.entity2id[entity]
        return self.entity_embeddings(torch.tensor(entity_id))

    def encode_entities(self, entities: List[str]) -> torch.Tensor:
        """批量编码实体"""
        entity_ids = [self.entity2id.get(e, 0) for e in entities]
        return self.entity_embeddings(torch.tensor(entity_ids))

    def find_entities_from_query(
        self,
        query_embedding: torch.Tensor,
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        从查询向量找到最相关的实体

        Args:
            query_embedding: 查询向量 [d_model]
            top_k: 返回top-k个实体

        Returns:
            (实体, 相似度分数) 列表
        """
        if top_k is None:
            top_k = self.top_k

        # 计算与所有实体的相似度
        all_entity_embs = self.entity_embeddings.weight  # [num_entities, d_model]

        # Cosine相似度
        query_norm = torch.nn.functional.normalize(query_embedding, dim=-1)
        entity_norms = torch.nn.functional.normalize(all_entity_embs, dim=-1)
        similarities = torch.matmul(entity_norms, query_norm)  # [num_entities]

        # Top-k
        top_scores, top_indices = torch.topk(similarities, min(top_k, len(self.entity2id)))

        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            entity = self.id2entity[idx]
            results.append((entity, score))

        return results

    def retrieve_subgraph(
        self,
        entities: List[str],
        max_hops: Optional[int] = None
    ) -> KnowledgeGraph:
        """检索实体相关的子图"""
        if max_hops is None:
            max_hops = self.max_hops

        return self.kg.get_subgraph(entities, max_hops)

    def multi_hop_reasoning(
        self,
        start_entity: str,
        relations: List[str],
        max_results: int = 10
    ) -> List[List[str]]:
        """多跳推理"""
        return self.kg.multi_hop_query(start_entity, relations, max_results)

    def forward(
        self,
        query_embedding: torch.Tensor,
        top_k: Optional[int] = None,
        return_subgraph: bool = True
    ) -> Tuple[List[str], List[Triple], torch.Tensor]:
        """
        前向传播：从查询向量检索相关知识

        Args:
            query_embedding: 查询向量 [batch, seq_len, d_model] or [d_model]
            top_k: 检索top-k个实体
            return_subgraph: 是否返回子图

        Returns:
            entities: 检索到的实体列表
            triples: 相关三元组
            scores: 相关性分数
        """
        # 处理不同维度的输入
        if query_embedding.dim() == 3:
            # [batch, seq_len, d_model] -> 使用平均池化
            query_embedding = query_embedding.mean(dim=1)  # [batch, d_model]

        if query_embedding.dim() == 2:
            # [batch, d_model] -> 使用第一个
            query_embedding = query_embedding[0]  # [d_model]

        # 找到相关实体
        entity_scores = self.find_entities_from_query(query_embedding, top_k)
        entities = [e for e, _ in entity_scores]
        scores = torch.tensor([s for _, s in entity_scores])

        # 检索相关三元组
        triples = []
        if return_subgraph:
            subgraph = self.retrieve_subgraph(entities, max_hops=1)
            triples = list(subgraph.triples)
        else:
            # 只返回直接相关的三元组
            for entity in entities:
                triples.extend(self.kg.query_by_head(entity))
                triples.extend(self.kg.query_by_tail(entity))

        return entities, triples, scores


class KGRetrievalProvider(BaseProvider):
    """知识图谱检索Provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.kg_path = config.get('kg_path')
        self.d_model = config.get('d_model', 768)
        self.max_hops = config.get('max_hops', 2)
        self.top_k = config.get('top_k', 5)

        # 加载知识图谱
        if self.kg_path and os.path.exists(self.kg_path):
            self.kg = KnowledgeGraph.load(self.kg_path)
        else:
            self.kg = KnowledgeGraph()

    def create_retriever(self, d_model: int, top_k: int) -> KGRetriever:
        """创建检索器"""
        return KGRetriever(
            kg=self.kg,
            d_model=d_model,
            max_hops=self.max_hops,
            top_k=top_k
        )

    def retrieve(
        self,
        retriever: KGRetriever,
        query: torch.Tensor,
        top_k: Optional[int] = None
    ) -> Tuple[List[str], torch.Tensor]:
        """
        检索知识

        Args:
            retriever: KG检索器
            query: 查询向量
            top_k: 检索数量

        Returns:
            知识文本列表, 相关性分数
        """
        entities, triples, scores = retriever(query, top_k, return_subgraph=True)

        # 将三元组转换为文本
        knowledge_texts = []
        for triple in triples[:top_k or self.top_k]:
            text = f"{triple.head} {triple.relation} {triple.tail}"
            knowledge_texts.append(text)

        return knowledge_texts, scores

    def build_index(
        self,
        corpus: List[str],
        embedding_model: Optional[nn.Module] = None
    ) -> Optional[torch.Tensor]:
        """
        构建索引（对于KG，主要是加载三元组）

        Args:
            corpus: 三元组文本列表（格式：head\trelation\ttail）
            embedding_model: 可选的嵌入模型

        Returns:
            实体嵌入（如果使用embedding_model）
        """
        # 解析三元组并添加到KG
        for line in corpus:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                head, relation, tail = parts[0], parts[1], parts[2]
                self.kg.add_triple(head, relation, tail)

        print(f"[KG] 构建索引完成: {len(self.kg.entities)} 实体, {len(self.kg.triples)} 三元组")

        # 如果提供了embedding模型，可以预计算实体嵌入
        if embedding_model is not None:
            # TODO: 使用embedding_model编码实体名称
            pass

        return None


import os


def register_kg_provider():
    """注册KG检索provider到registry"""
    from apt.core.registry import registry

    try:
        registry.register(
            'retrieval',
            'kg',
            KGRetrievalProvider,
            override=True
        )
        print("[KG] KG Retrieval Provider已注册到registry")
    except Exception as e:
        print(f"[KG] 注册失败: {e}")


# 自动注册
register_kg_provider()
