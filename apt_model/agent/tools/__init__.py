#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agent Tools Collection

预定义的工具集合
"""

from .web_search import (
    WebSearchTool,
    SearchResult, SearchResponse,
    SearchEngine,
    DuckDuckGoSearch, MockSearchEngine,
    create_web_search_tool
)

__all__ = [
    "WebSearchTool",
    "SearchResult", "SearchResponse",
    "SearchEngine",
    "DuckDuckGoSearch", "MockSearchEngine",
    "create_web_search_tool"
]
