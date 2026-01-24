#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Provider Interfaces

This package defines the abstract interfaces that plugins must implement
to integrate with the APT core system.

Available Provider Types:
- AttentionProvider: Attention mechanism implementations
- FFNProvider: Feed-forward network implementations
- RouterProvider: MoE routing implementations
- AlignProvider: Bistate alignment implementations
- RetrievalProvider: RAG retrieval implementations
"""

try:
    from apt.core.providers.attention import AttentionProvider
except ImportError:
    pass
try:
    from apt.core.providers.ffn import FFNProvider
except ImportError:
    pass
try:
    from apt.core.providers.router import RouterProvider
except ImportError:
    pass
try:
    from apt.core.providers.align import AlignProvider
except ImportError:
    pass
try:
    from apt.core.providers.retrieval import RetrievalProvider
except ImportError:
    pass

__all__ = [
    'AttentionProvider',
    'FFNProvider',
    'RouterProvider',
    'AlignProvider',
    'RetrievalProvider',
]
