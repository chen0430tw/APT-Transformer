#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
知识图谱 + RAG 集成模块

结合结构化知识（KG）和非结构化文档（RAG）：
- KG提供精确的实体关系知识
- RAG提供丰富的上下文信息
- 两者融合增强生成质量
"""

from apt_model.utils.fake_torch import get_torch
torch = get_torch()
from apt_model.utils.fake_torch import get_torch
torch = get_torch()
nn = torch.nn
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass

from apt.core.registry import registry
from apt.core.system import get_device
from apt.core.modeling.rag_integration import RAGWrapper, RAGConfig
from apt.core.modeling.knowledge_graph import KnowledgeGraph
from apt_model.infrastructure.logging import get_progress_logger

logger = get_progress_logger()


@dataclass
class KGRAGConfig:
    """KG-RAG集成配置"""

    # KG配置
    kg_path: Optional[str] = None  # 知识图谱文件路径
    kg_max_hops: int = 2  # 多跳推理最大跳数
    kg_top_k: int = 5  # KG检索top-k

    # RAG配置
    rag_provider: str = 'exact_cosine'  # RAG检索provider
    rag_top_k: int = 5  # RAG检索top-k
    corpus_path: Optional[str] = None  # 文档语料路径

    # 融合配置
    fusion_method: str = 'weighted'  # 'weighted', 'concatenate', 'gate'
    kg_weight: float = 0.5  # KG知识权重
    rag_weight: float = 0.5  # RAG文档权重

    # 模型配置
    d_model: int = 768
    cache_dir: str = './cache/kg_rag'


class KGRAGWrapper(nn.Module):
    """
    KG + RAG 集成包装器

    同时使用知识图谱和文档检索增强生成
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: KGRAGConfig,
        kg: Optional[KnowledgeGraph] = None,
        corpus: Optional[List[str]] = None
    ):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.device = get_device()

        # 加载或创建知识图谱
        if kg is not None:
            self.kg = kg
        elif config.kg_path:
            self.kg = KnowledgeGraph.load(config.kg_path)
        else:
            self.kg = KnowledgeGraph()

        # 创建KG检索器
        try:
            kg_provider_class = registry.get('retrieval', 'kg')
            self.kg_provider = kg_provider_class({
                'kg_path': config.kg_path,
                'd_model': config.d_model,
                'max_hops': config.kg_max_hops,
                'top_k': config.kg_top_k,
            })
            self.kg_retriever = self.kg_provider.create_retriever(
                d_model=config.d_model,
                top_k=config.kg_top_k
            )
            self.kg_retriever = self.kg_retriever.to(self.device)
            self.use_kg = True
            logger.info("[KG-RAG] KG检索器已初始化")
        except Exception as e:
            logger.warning(f"[KG-RAG] KG检索器初始化失败: {e}")
            self.use_kg = False
            self.kg_provider = None
            self.kg_retriever = None

        # 创建RAG检索器
        try:
            rag_config = RAGConfig(
                provider_name=config.rag_provider,
                top_k=config.rag_top_k,
                d_model=config.d_model,
                corpus_path=config.corpus_path,
                cache_dir=config.cache_dir
            )
            self.rag_wrapper = RAGWrapper(base_model, rag_config, corpus)
            self.use_rag = True
            logger.info("[KG-RAG] RAG检索器已初始化")
        except Exception as e:
            logger.warning(f"[KG-RAG] RAG检索器初始化失败: {e}")
            self.use_rag = False
            self.rag_wrapper = None

        # 融合层
        if config.fusion_method == 'gate':
            self.fusion_gate = nn.Linear(config.d_model * 2, 2)
        elif config.fusion_method == 'weighted':
            # 可学习的权重
            self.kg_weight_param = nn.Parameter(torch.tensor(config.kg_weight))
            self.rag_weight_param = nn.Parameter(torch.tensor(config.rag_weight))

        logger.info(f"[KG-RAG] 融合方法: {config.fusion_method}")
        logger.info(f"[KG-RAG] KG启用: {self.use_kg}, RAG启用: {self.use_rag}")

    def build_kg_index(self, triples: List[Tuple[str, str, str]]):
        """构建KG索引"""
        if not self.use_kg:
            logger.warning("[KG-RAG] KG未启用")
            return

        self.kg.add_triples_batch(triples)
        # 重新创建检索器以使用更新的KG
        self.kg_retriever = self.kg_provider.create_retriever(
            d_model=self.config.d_model,
            top_k=self.config.kg_top_k
        )
        self.kg_retriever = self.kg_retriever.to(self.device)
        logger.info(f"[KG-RAG] KG索引已构建: {len(self.kg.entities)} 实体, {len(self.kg.triples)} 三元组")

    def build_rag_index(
        self,
        corpus: List[str],
        embedding_model: Optional[nn.Module] = None
    ):
        """构建RAG索引"""
        if not self.use_rag:
            logger.warning("[KG-RAG] RAG未启用")
            return

        self.rag_wrapper.build_index(corpus, embedding_model=embedding_model)
        logger.info(f"[KG-RAG] RAG索引已构建: {len(corpus)} 文档")

    def retrieve(
        self,
        query: torch.Tensor,
        use_kg: bool = True,
        use_rag: bool = True
    ) -> Dict[str, Any]:
        """
        检索知识

        Args:
            query: 查询向量 [batch, seq_len, d_model]
            use_kg: 是否使用KG
            use_rag: 是否使用RAG

        Returns:
            检索结果字典
        """
        results = {
            'kg_knowledge': [],
            'rag_docs': [],
            'kg_scores': None,
            'rag_scores': None,
        }

        # KG检索
        if use_kg and self.use_kg and self.kg_retriever:
            try:
                kg_texts, kg_scores = self.kg_provider.retrieve(
                    self.kg_retriever,
                    query,
                    top_k=self.config.kg_top_k
                )
                results['kg_knowledge'] = kg_texts
                results['kg_scores'] = kg_scores
            except Exception as e:
                logger.warning(f"[KG-RAG] KG检索失败: {e}")

        # RAG检索
        if use_rag and self.use_rag and self.rag_wrapper:
            try:
                rag_results = self.rag_wrapper.retrieve(query, top_k=self.config.rag_top_k)
                results['rag_docs'] = rag_results[0]
                results['rag_scores'] = rag_results[1]
            except Exception as e:
                logger.warning(f"[KG-RAG] RAG检索失败: {e}")

        return results

    def fuse_knowledge(
        self,
        kg_knowledge: List[str],
        rag_docs: List[str],
        kg_scores: Optional[torch.Tensor] = None,
        rag_scores: Optional[torch.Tensor] = None
    ) -> str:
        """
        融合KG和RAG的知识

        Args:
            kg_knowledge: KG检索的知识
            rag_docs: RAG检索的文档
            kg_scores: KG相关性分数
            rag_scores: RAG相关性分数

        Returns:
            融合后的上下文文本
        """
        if self.config.fusion_method == 'concatenate':
            # 简单拼接
            context_parts = []
            if kg_knowledge:
                context_parts.append("知识图谱: " + " ; ".join(kg_knowledge))
            if rag_docs:
                context_parts.append("文档内容: " + " ".join(rag_docs))
            return "\n".join(context_parts)

        elif self.config.fusion_method == 'weighted':
            # 加权融合
            kg_weight = torch.sigmoid(self.kg_weight_param).item() if hasattr(self, 'kg_weight_param') else self.config.kg_weight
            rag_weight = torch.sigmoid(self.rag_weight_param).item() if hasattr(self, 'rag_weight_param') else self.config.rag_weight

            # 归一化
            total = kg_weight + rag_weight
            kg_weight /= total
            rag_weight /= total

            context_parts = []
            if kg_knowledge:
                # 根据权重选择top-k
                kg_k = max(1, int(len(kg_knowledge) * kg_weight / 0.5))
                context_parts.append("知识: " + " ; ".join(kg_knowledge[:kg_k]))
            if rag_docs:
                rag_k = max(1, int(len(rag_docs) * rag_weight / 0.5))
                context_parts.append("上下文: " + " ".join(rag_docs[:rag_k]))
            return "\n".join(context_parts)

        else:  # gate
            # TODO: 实现基于gate的融合
            return self.fuse_knowledge(kg_knowledge, rag_docs, kg_scores, rag_scores)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_kg: bool = True,
        use_rag: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            use_kg: 是否使用KG
            use_rag: 是否使用RAG

        Returns:
            输出字典
        """
        # 获取基础模型的hidden states
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )

        # 检索知识
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            # 使用中间层的hidden states作为查询
            query = outputs.hidden_states[len(outputs.hidden_states) // 2]
            retrieval_results = self.retrieve(query, use_kg, use_rag)
        else:
            retrieval_results = {
                'kg_knowledge': [],
                'rag_docs': [],
                'kg_scores': None,
                'rag_scores': None,
            }

        # 融合知识
        fused_context = self.fuse_knowledge(
            retrieval_results['kg_knowledge'],
            retrieval_results['rag_docs'],
            retrieval_results['kg_scores'],
            retrieval_results['rag_scores']
        )

        return {
            'logits': outputs.logits if hasattr(outputs, 'logits') else outputs,
            'kg_knowledge': retrieval_results['kg_knowledge'],
            'rag_docs': retrieval_results['rag_docs'],
            'fused_context': fused_context,
            'kg_scores': retrieval_results['kg_scores'],
            'rag_scores': retrieval_results['rag_scores'],
        }


def create_kg_rag_model(
    base_model: nn.Module,
    kg_path: Optional[str] = None,
    corpus_path: Optional[str] = None,
    kg: Optional[KnowledgeGraph] = None,
    corpus: Optional[List[str]] = None,
    fusion_method: str = 'weighted',
    **kwargs
) -> KGRAGWrapper:
    """
    创建KG-RAG模型的便捷函数

    Args:
        base_model: 基础语言模型
        kg_path: 知识图谱文件路径
        corpus_path: 文档语料路径
        kg: 已加载的知识图谱（可选）
        corpus: 文档列表（可选）
        fusion_method: 融合方法
        **kwargs: 其他配置

    Returns:
        KG-RAG模型
    """
    config = KGRAGConfig(
        kg_path=kg_path,
        corpus_path=corpus_path,
        fusion_method=fusion_method,
        **kwargs
    )

    model = KGRAGWrapper(base_model, config, kg, corpus)

    # 自动构建索引
    if kg and not kg_path:
        # 如果提供了KG对象但没有路径，使用KG对象
        model.build_kg_index([(t.head, t.relation, t.tail) for t in kg.triples])

    if corpus:
        model.build_rag_index(corpus)

    return model


# 便捷函数：快速创建KG-RAG模型
def quick_kg_rag(
    model: nn.Module,
    kg_triples: Optional[List[Tuple[str, str, str]]] = None,
    corpus: Optional[List[str]] = None,
    **kwargs
) -> KGRAGWrapper:
    """
    快速创建KG-RAG模型

    Args:
        model: 基础模型
        kg_triples: 三元组列表
        corpus: 文档列表
        **kwargs: 其他配置

    Returns:
        KG-RAG模型
    """
    kg = KnowledgeGraph()
    if kg_triples:
        kg.add_triples_batch(kg_triples)

    return create_kg_rag_model(
        model,
        kg=kg,
        corpus=corpus,
        **kwargs
    )
