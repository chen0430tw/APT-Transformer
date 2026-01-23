#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Memory Module (L2 Memory Layer)

记忆层 - AIM-Memory 系统：
- AIM-Memory (惯性路由 + 时间镜像 + 锚点修正)
- AIM-NC (N-gram + Trie 集成 + 锚点主权)
- GraphRAG (知识图谱 + 检索增强)
- Long Context (超长上下文支持)

使用示例:
    >>> import apt
    >>> apt.enable('pro')  # 加载 L0 + L2
    >>> from apt.memory import AIMMemory, GraphRAG
    >>> memory = AIMMemory(mode='aim-nc')
"""

# ═══════════════════════════════════════════════════════════
# AIM-Memory System
# ═══════════════════════════════════════════════════════════
try:
    from apt.memory.aim.aim_memory import AIMMemory
except ImportError:
    AIMMemory = None

try:
    from apt.memory.aim.aim_nc import AIMNC
except ImportError:
    AIMNC = None

try:
    from apt.memory.aim.anchor_fields import AnchorField
except ImportError:
    AnchorField = None

try:
    from apt.memory.aim.evidence_feedback import EvidenceFeedback
except ImportError:
    EvidenceFeedback = None

try:
    from apt.memory.aim.tiered_memory import TieredMemory
except ImportError:
    TieredMemory = None

# ═══════════════════════════════════════════════════════════
# GraphRAG System
# ═══════════════════════════════════════════════════════════
try:
    from apt.memory.graph_rag.graph_brain import GraphBrain
except ImportError:
    GraphBrain = None

try:
    from apt.memory.graph_rag.graph_rag_manager import GraphRAGManager as GraphRAG
except ImportError:
    GraphRAG = None

try:
    from apt.memory.knowledge_graph import KnowledgeGraph
except ImportError:
    KnowledgeGraph = None

try:
    from apt.memory.kg_rag_integration import KGRAGIntegration
except ImportError:
    KGRAGIntegration = None

try:
    from apt.memory.rag_integration import RAGIntegration
except ImportError:
    RAGIntegration = None

# ═══════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════
__all__ = [
    # AIM-Memory
    'AIMMemory',
    'AIMNC',
    'AnchorField',
    'EvidenceFeedback',
    'TieredMemory',

    # GraphRAG
    'GraphBrain',
    'GraphRAG',
    'KnowledgeGraph',
    'KGRAGIntegration',
    'RAGIntegration',
]