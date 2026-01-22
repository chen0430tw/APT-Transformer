#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT 模型模块

包含所有APT模型的实现：
- APT核心模型（自生成变换器）
- 多模态模型
- 各种特定任务模型（Claude4, GPT5等）
- 模型组件（embeddings, encoders, blocks等）
"""

# 核心模型
from apt_model.modeling.apt_model import APTLargeModel
from apt_model.modeling.multimodal_model import MultimodalAPTModel
from apt_model.modeling.embeddings import APTEmbedding

# 分词器
from apt_model.modeling.chinese_tokenizer import ChineseTokenizer
from apt_model.modeling.chinese_tokenizer_integration import (
    integrate_chinese_tokenizer,
    detect_language,
    get_appropriate_tokenizer,
)

# 特定模型
from apt_model.modeling.claude4_model import Claude4Model
from apt_model.modeling.gpt5_model import GPT5Model
from apt_model.modeling.gpt4o_model import GPT4oModel
from apt_model.modeling.gpto3_model import GPTO3Model
from apt_model.modeling.vft_tva_model import VFTTVAModel

# 高级特性
from apt_model.modeling.advanced_rope import AdvancedRoPE
from apt_model.modeling.apt_control import APTController
from apt_model.modeling.elastic_transformer import ElasticTransformer
from apt_model.modeling.moe_optimized import MoEOptimized
from apt_model.modeling.left_spin_smooth import (
    LeftSpinStep,
    LeftSpinResidual,
    AdaptiveLeftSpinStep,
)
from apt_model.modeling.memory_augmented_smooth import MemoryAugmentedSmooth

# RAG和知识图谱
from apt_model.modeling.rag_integration import RAGIntegration
from apt_model.modeling.kg_rag_integration import KGRAGIntegration
from apt_model.modeling.knowledge_graph import KnowledgeGraph

# MCP集成
from apt_model.modeling.mcp_integration import MCPIntegration

__all__ = [
    # 核心模型
    'APTLargeModel',
    'MultimodalAPTModel',
    'APTEmbedding',

    # 分词器
    'ChineseTokenizer',
    'integrate_chinese_tokenizer',
    'detect_language',
    'get_appropriate_tokenizer',

    # 特定模型
    'Claude4Model',
    'GPT5Model',
    'GPT4oModel',
    'GPTO3Model',
    'VFTTVAModel',

    # 高级特性
    'AdvancedRoPE',
    'APTController',
    'ElasticTransformer',
    'MoEOptimized',
    'LeftSpinStep',
    'LeftSpinResidual',
    'AdaptiveLeftSpinStep',
    'MemoryAugmentedSmooth',

    # RAG和知识图谱
    'RAGIntegration',
    'KGRAGIntegration',
    'KnowledgeGraph',

    # MCP集成
    'MCPIntegration',
]
