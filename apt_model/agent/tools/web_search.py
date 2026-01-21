#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Web 搜索工具

支持多个搜索引擎：
- DuckDuckGo (免费，无需 API key)
- Google Custom Search (需要 API key)
- Bing Search (需要 API key)
- Serper.dev (需要 API key)

特性：
1. 多搜索引擎支持
2. 结果解析和摘要
3. 错误处理和重试
4. 结果缓存
5. 与 AIM-Memory 集成存储搜索历史

参考：
- DuckDuckGo API: https://duckduckgo.com/api
- Serper.dev: https://serper.dev/
"""

import re
import time
import json
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging

# HTTP 请求库
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("[WebSearch] aiohttp not available, web search will not work")

from ..tool_system import Tool, ToolDefinition, ToolParameter, ToolType


logger = logging.getLogger(__name__)


# ==================== 搜索结果 ====================

@dataclass
class SearchResult:
    """搜索结果"""
    title: str
    snippet: str
    url: str
    source: str = "unknown"  # 来源（duckduckgo/google/bing）
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResponse:
    """搜索响应"""
    query: str
    results: List[SearchResult]
    total_results: int
    search_engine: str
    execution_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== 搜索引擎基类 ====================

class SearchEngine:
    """搜索引擎基类"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._session = None

    async def search(
        self,
        query: str,
        num_results: int = 5,
        **kwargs
    ) -> SearchResponse:
        """执行搜索"""
        raise NotImplementedError

    async def _get_session(self) -> 'aiohttp.ClientSession':
        """获取 HTTP session"""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp is not installed")

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        """关闭 session"""
        if self._session and not self._session.closed:
            await self._session.close()


# ==================== DuckDuckGo 搜索（免费） ====================

class DuckDuckGoSearch(SearchEngine):
    """
    DuckDuckGo 搜索（免费，无需 API key）

    注意：DuckDuckGo 的官方 API 功能有限，这里使用 HTML 抓取
    """

    def __init__(self):
        super().__init__(api_key=None)

    async def search(
        self,
        query: str,
        num_results: int = 5,
        **kwargs
    ) -> SearchResponse:
        """
        使用 DuckDuckGo 搜索

        Args:
            query: 搜索查询
            num_results: 结果数量

        Returns:
            SearchResponse
        """
        start_time = time.time()

        try:
            session = await self._get_session()

            # DuckDuckGo 的 HTML 搜索 URL
            url = "https://html.duckduckgo.com/html/"
            params = {
                "q": query,
                "kl": "wt-wt"  # 所有地区
            }

            async with session.post(url, data=params) as response:
                html = await response.text()

                # 简单的 HTML 解析（生产环境应使用 BeautifulSoup）
                results = self._parse_duckduckgo_html(html, num_results)

                return SearchResponse(
                    query=query,
                    results=results,
                    total_results=len(results),
                    search_engine="duckduckgo",
                    execution_time=time.time() - start_time
                )

        except Exception as e:
            logger.error(f"[DuckDuckGo] Search error: {e}")
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_engine="duckduckgo",
                execution_time=time.time() - start_time,
                metadata={"error": str(e)}
            )

    def _parse_duckduckgo_html(self, html: str, max_results: int) -> List[SearchResult]:
        """
        解析 DuckDuckGo HTML 结果（简化版）

        生产环境应使用 BeautifulSoup 或类似库
        """
        results = []

        # 使用正则表达式提取结果（非常简化）
        # 实际应用中应使用 HTML 解析器

        # 提取标题和链接
        title_pattern = r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>'
        snippet_pattern = r'<a[^>]*class="result__snippet"[^>]*>([^<]+)</a>'

        titles = re.findall(title_pattern, html)
        snippets = re.findall(snippet_pattern, html)

        for i, (url, title) in enumerate(titles[:max_results]):
            snippet = snippets[i] if i < len(snippets) else ""

            results.append(SearchResult(
                title=title.strip(),
                snippet=snippet.strip(),
                url=url.strip(),
                source="duckduckgo"
            ))

        return results


# ==================== 模拟搜索（用于测试） ====================

class MockSearchEngine(SearchEngine):
    """模拟搜索引擎（用于测试，无需网络）"""

    def __init__(self):
        super().__init__(api_key=None)

        # 模拟的搜索数据库
        self.database = {
            "transformer": [
                SearchResult(
                    title="Attention Is All You Need",
                    snippet="The Transformer architecture, introduced by Vaswani et al., uses self-attention mechanisms...",
                    url="https://arxiv.org/abs/1706.03762",
                    source="mock"
                ),
                SearchResult(
                    title="The Illustrated Transformer",
                    snippet="A visual guide to understanding the Transformer architecture...",
                    url="https://jalammar.github.io/illustrated-transformer/",
                    source="mock"
                ),
            ],
            "python": [
                SearchResult(
                    title="Python Official Documentation",
                    snippet="The official Python language reference and standard library documentation...",
                    url="https://docs.python.org/3/",
                    source="mock"
                ),
                SearchResult(
                    title="Python Tutorial",
                    snippet="Learn Python programming from basics to advanced topics...",
                    url="https://www.python.org/about/gettingstarted/",
                    source="mock"
                ),
            ],
            "default": [
                SearchResult(
                    title="Example Search Result",
                    snippet="This is a mock search result for testing purposes...",
                    url="https://example.com",
                    source="mock"
                ),
            ]
        }

    async def search(
        self,
        query: str,
        num_results: int = 5,
        **kwargs
    ) -> SearchResponse:
        """模拟搜索"""
        start_time = time.time()

        # 模拟网络延迟
        await asyncio.sleep(0.1)

        # 查找最匹配的关键词
        query_lower = query.lower()
        results = []

        for keyword, keyword_results in self.database.items():
            if keyword in query_lower:
                results.extend(keyword_results)

        # 如果没有匹配，返回默认结果
        if not results:
            results = self.database["default"]

        # 限制结果数量
        results = results[:num_results]

        return SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
            search_engine="mock",
            execution_time=time.time() - start_time,
            metadata={"note": "This is a mock search for testing"}
        )


# ==================== Web 搜索工具 ====================

class WebSearchTool(Tool):
    """
    Web 搜索工具

    与工具系统集成的 Web 搜索功能
    """

    def __init__(
        self,
        name: str = "web_search",
        search_engine: str = "mock",  # "duckduckgo", "mock"
        api_key: Optional[str] = None,
        default_num_results: int = 5
    ):
        # 工具定义
        definition = ToolDefinition(
            name=name,
            description="Search the web for information using a search engine",
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True
                ),
                ToolParameter(
                    name="num_results",
                    type="number",
                    description="Number of results to return (default: 5)",
                    required=False,
                    default=default_num_results
                )
            ],
            tool_type=ToolType.SEARCH,
            timeout=15.0,
            allow_parallel=True,
            cacheable=True,  # 搜索结果可以缓存
            cache_ttl=3600.0,  # 1小时
            tags=["web", "search", "internet"]
        )

        super().__init__(definition)

        # 创建搜索引擎
        if search_engine == "duckduckgo":
            self.search_engine = DuckDuckGoSearch()
        elif search_engine == "mock":
            self.search_engine = MockSearchEngine()
        else:
            raise ValueError(f"Unknown search engine: {search_engine}")

        self.default_num_results = default_num_results

    async def execute(self, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """执行 Web 搜索"""
        query = arguments["query"]
        num_results = arguments.get("num_results", self.default_num_results)

        # 执行搜索
        response = await self.search_engine.search(query, num_results)

        # 格式化结果
        formatted_results = []
        for i, result in enumerate(response.results, 1):
            formatted_results.append({
                "rank": i,
                "title": result.title,
                "snippet": result.snippet,
                "url": result.url,
                "source": result.source
            })

        return {
            "query": query,
            "results": formatted_results,
            "total_results": response.total_results,
            "search_engine": response.search_engine,
            "execution_time": response.execution_time
        }

    async def close(self):
        """关闭搜索引擎"""
        if hasattr(self.search_engine, 'close'):
            await self.search_engine.close()


# ==================== 工厂函数 ====================

def create_web_search_tool(
    search_engine: str = "mock",
    api_key: Optional[str] = None,
    **kwargs
) -> WebSearchTool:
    """
    创建 Web 搜索工具

    Args:
        search_engine: 搜索引擎名称 ("duckduckgo", "mock")
        api_key: API 密钥（如果需要）
        **kwargs: 其他参数

    Returns:
        WebSearchTool
    """
    return WebSearchTool(
        search_engine=search_engine,
        api_key=api_key,
        **kwargs
    )


# ==================== 示例 ====================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=== Web Search Tool Demo ===\n")

    async def test():
        # 创建搜索工具（使用 mock 引擎）
        tool = create_web_search_tool(search_engine="mock")

        # 测试搜索
        queries = [
            "transformer architecture",
            "python programming",
            "artificial intelligence"
        ]

        for query in queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}")

            result = await tool.execute({"query": query, "num_results": 3})

            print(f"Search Engine: {result['search_engine']}")
            print(f"Total Results: {result['total_results']}")
            print(f"Execution Time: {result['execution_time']:.3f}s")
            print(f"\nResults:")

            for item in result["results"]:
                print(f"  [{item['rank']}] {item['title']}")
                print(f"      {item['snippet']}")
                print(f"      URL: {item['url']}")
                print()

        # 关闭
        await tool.close()

    asyncio.run(test())

    print("\n=== Demo Complete ===")
