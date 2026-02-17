#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model Domain

模型定义域：包含所有模型架构、层、分词器、损失函数和优化器

子模块：
- architectures: 模型架构定义（APT, Claude4, GPT5等）
- layers: 基础层组件（Attention, FFN, Embedding等）
- tokenization: 分词器实现
- losses: 损失函数
- optim: 优化器
- extensions: 核心扩展（RAG, KG, MCP等）

使用示例：
    try:
        from apt.model.architectures import APTLargeModel
    except ImportError:
        APTLargeModel = None
    try:
        from apt.model.tokenization import ChineseTokenizer
    except ImportError:
        ChineseTokenizer = None
    try:
        from apt.model.losses import APTLoss
    except ImportError:
        APTLoss = None
"""

__version__ = '2.0.0-alpha'

# 主要模块导出
try:
    try:
        from apt.model.architectures import APTLargeModel, MultimodalAPTModel
    except ImportError:
        APTLargeModel = None
        MultimodalAPTModel = None
except ImportError as e:
    # 如果导入失败，定义占位符
    APTLargeModel = None
    MultimodalAPTModel = None

try:
    try:
        from apt.model.tokenization import ChineseTokenizer
    except ImportError:
        ChineseTokenizer = None
except ImportError:
    ChineseTokenizer = None

try:
    try:
        from apt.model.layers import PositionalEncoding, TokenEmbedding, AdvancedRoPE
    except ImportError:
        PositionalEncoding = None
        TokenEmbedding = None
        AdvancedRoPE = None
except ImportError:
    PositionalEncoding = None
    TokenEmbedding = None
    AdvancedRoPE = None

try:
    try:
        from apt.model.extensions import RAGIntegration, KnowledgeGraph, MCPIntegration
    except ImportError:
        RAGIntegration = None
        KnowledgeGraph = None
        MCPIntegration = None
except ImportError:
    RAGIntegration = None
    KnowledgeGraph = None
    MCPIntegration = None

__all__ = [
    # Architectures
    'APTLargeModel',
    'MultimodalAPTModel',
    # Tokenization
    'ChineseTokenizer',
    # Layers
    'PositionalEncoding',
    'TokenEmbedding',
    'AdvancedRoPE',
    # Extensions
    'RAGIntegration',
    'KnowledgeGraph',
    'MCPIntegration',
]
