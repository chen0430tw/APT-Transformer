#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Extensions

核心扩展（由核心团队维护，深度集成）：
- RAG (Retrieval-Augmented Generation): 检索增强生成
- KG (Knowledge Graph): 知识图谱集成
- MCP (Model Context Protocol): 模型上下文协议
- Graph RAG: 图检索增强生成

扩展特点：
- 可以修改模型架构
- 编译时集成
- 深度访问内部API
- 核心团队维护
"""

try:
    from apt.model.extensions.rag_integration import RAGIntegration
except ImportError:
    RAGIntegration = None
try:
    from apt.model.extensions.knowledge_graph import KnowledgeGraph
except ImportError:
    KnowledgeGraph = None
try:
    from apt.model.extensions.kg_rag_integration import KGRAGIntegration
except ImportError:
    KGRAGIntegration = None
try:
    from apt.model.extensions.mcp_integration import MCPIntegration
except ImportError:
    MCPIntegration = None

__all__ = [
    'RAGIntegration',
    'KnowledgeGraph',
    'KGRAGIntegration',
    'MCPIntegration',
]
