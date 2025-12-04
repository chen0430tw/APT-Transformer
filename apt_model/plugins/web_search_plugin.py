#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI Web Search Plugin

整合主流 AI 搜索方案，支持多个后端:
- Tavily (AI-native, 专为 agents 设计)
- Perplexity (快速, <400ms)
- DuckDuckGo (免费, 隐私友好)
- Serper.dev (Google SERP API)
- Brave Search (隐私优先)

基于 2025 年最新搜索技术栈。
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import logging
from abc import ABC, abstractmethod

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False

logger = logging.getLogger(__name__)


class SearchProvider(Enum):
    """Supported search providers"""
    TAVILY = "tavily"
    PERPLEXITY = "perplexity"
    DUCKDUCKGO = "duckduckgo"
    SERPER = "serper"
    BRAVE = "brave"


@dataclass
class SearchResult:
    """统一的搜索结果格式"""
    title: str
    url: str
    snippet: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResponse:
    """搜索响应"""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time: float
    provider: str
    raw_response: Optional[Dict] = None


class BaseSearchBackend(ABC):
    """搜索后端基类"""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.config = kwargs
        self.stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'failed_searches': 0,
            'total_time': 0.0
        }

    @abstractmethod
    def search(self, query: str, max_results: int = 10, **kwargs) -> SearchResponse:
        """执行搜索"""
        pass

    def _update_stats(self, success: bool, duration: float):
        """更新统计信息"""
        self.stats['total_searches'] += 1
        if success:
            self.stats['successful_searches'] += 1
        else:
            self.stats['failed_searches'] += 1
        self.stats['total_time'] += duration

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        avg_time = self.stats['total_time'] / max(1, self.stats['total_searches'])
        return {
            **self.stats,
            'avg_search_time': avg_time,
            'success_rate': self.stats['successful_searches'] / max(1, self.stats['total_searches'])
        }


class TavilyBackend(BaseSearchBackend):
    """
    Tavily AI Search

    专为 AI agents 设计的搜索 API。
    - 每月 1000 免费 credits
    - $0.008/credit
    - 返回结构化 JSON with summaries

    文档: https://docs.tavily.com/
    """

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.api_url = "https://api.tavily.com/search"

    def search(self, query: str, max_results: int = 10, **kwargs) -> SearchResponse:
        if not HAS_REQUESTS:
            raise RuntimeError("requests library not installed. Run: pip install requests")

        start_time = time.time()

        try:
            payload = {
                "api_key": self.api_key,
                "query": query,
                "search_depth": kwargs.get('search_depth', 'basic'),  # 'basic' or 'advanced'
                "include_answer": kwargs.get('include_answer', True),
                "include_images": kwargs.get('include_images', False),
                "include_raw_content": kwargs.get('include_raw_content', False),
                "max_results": max_results
            }

            response = requests.post(
                self.api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            duration = time.time() - start_time

            # Parse results
            results = []
            for item in data.get('results', []):
                results.append(SearchResult(
                    title=item.get('title', ''),
                    url=item.get('url', ''),
                    snippet=item.get('content', ''),
                    score=item.get('score', 0.0),
                    metadata={'raw_content': item.get('raw_content')}
                ))

            self._update_stats(True, duration)

            return SearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                search_time=duration,
                provider='tavily',
                raw_response=data
            )

        except Exception as e:
            duration = time.time() - start_time
            self._update_stats(False, duration)
            logger.error(f"Tavily search error: {e}")
            raise


class PerplexityBackend(BaseSearchBackend):
    """
    Perplexity AI Search

    强调速度和深度过滤。
    - 中位数 <400ms
    - $5/1000 请求
    - 零数据保留政策

    文档: https://docs.perplexity.ai/
    """

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.api_url = "https://api.perplexity.ai/search"

    def search(self, query: str, max_results: int = 10, **kwargs) -> SearchResponse:
        if not HAS_REQUESTS:
            raise RuntimeError("requests library not installed")

        start_time = time.time()

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "query": query,
                "max_results": max_results,
                "search_recency_filter": kwargs.get('recency', 'month')  # 'day', 'week', 'month', 'year'
            }

            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=10  # Perplexity is fast
            )
            response.raise_for_status()

            data = response.json()
            duration = time.time() - start_time

            # Parse results
            results = []
            for item in data.get('results', []):
                results.append(SearchResult(
                    title=item.get('title', ''),
                    url=item.get('url', ''),
                    snippet=item.get('snippet', ''),
                    score=item.get('relevance_score', 0.0),
                    metadata=item.get('metadata', {})
                ))

            self._update_stats(True, duration)

            return SearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                search_time=duration,
                provider='perplexity',
                raw_response=data
            )

        except Exception as e:
            duration = time.time() - start_time
            self._update_stats(False, duration)
            logger.error(f"Perplexity search error: {e}")
            raise


class DuckDuckGoBackend(BaseSearchBackend):
    """
    DuckDuckGo Search

    免费、注重隐私、LangChain 直接支持。
    - 无需 API key
    - 无请求限制
    - 隐私优先

    文档: https://pypi.org/project/duckduckgo-search/
    """

    def __init__(self, **kwargs):
        super().__init__(api_key=None, **kwargs)
        if not HAS_DDGS:
            logger.warning("duckduckgo_search not installed. Install: pip install duckduckgo-search")

    def search(self, query: str, max_results: int = 10, **kwargs) -> SearchResponse:
        if not HAS_DDGS:
            raise RuntimeError("duckduckgo_search not installed. Run: pip install duckduckgo-search")

        start_time = time.time()

        try:
            with DDGS() as ddgs:
                search_results = list(ddgs.text(
                    query,
                    max_results=max_results,
                    region=kwargs.get('region', 'wt-wt'),  # worldwide
                    safesearch=kwargs.get('safesearch', 'moderate'),
                    timelimit=kwargs.get('timelimit', None)  # 'd', 'w', 'm', 'y'
                ))

            duration = time.time() - start_time

            # Parse results
            results = []
            for item in search_results:
                results.append(SearchResult(
                    title=item.get('title', ''),
                    url=item.get('href', ''),
                    snippet=item.get('body', ''),
                    score=1.0,  # DuckDuckGo doesn't provide scores
                    metadata={'published': item.get('published')}
                ))

            self._update_stats(True, duration)

            return SearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                search_time=duration,
                provider='duckduckgo',
                raw_response={"results": search_results}
            )

        except Exception as e:
            duration = time.time() - start_time
            self._update_stats(False, duration)
            logger.error(f"DuckDuckGo search error: {e}")
            raise


class SerperBackend(BaseSearchBackend):
    """
    Serper.dev - Google SERP API

    返回原始 Google 搜索结果。
    - ~2 秒返回
    - LangChain 内置支持
    - 访问 Google 搜索数据

    文档: https://serper.dev/
    """

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.api_url = "https://google.serper.dev/search"

    def search(self, query: str, max_results: int = 10, **kwargs) -> SearchResponse:
        if not HAS_REQUESTS:
            raise RuntimeError("requests library not installed")

        start_time = time.time()

        try:
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }

            payload = {
                "q": query,
                "num": max_results,
                "gl": kwargs.get('gl', 'us'),  # country
                "hl": kwargs.get('hl', 'en')   # language
            }

            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            duration = time.time() - start_time

            # Parse organic results
            results = []
            for item in data.get('organic', [])[:max_results]:
                results.append(SearchResult(
                    title=item.get('title', ''),
                    url=item.get('link', ''),
                    snippet=item.get('snippet', ''),
                    score=float(item.get('position', 0)),
                    metadata={'position': item.get('position')}
                ))

            self._update_stats(True, duration)

            return SearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                search_time=duration,
                provider='serper',
                raw_response=data
            )

        except Exception as e:
            duration = time.time() - start_time
            self._update_stats(False, duration)
            logger.error(f"Serper search error: {e}")
            raise


class BraveBackend(BaseSearchBackend):
    """
    Brave Search API

    隐私优先的搜索引擎。
    - 独立索引
    - 隐私保护
    - 免费层可用

    文档: https://brave.com/search/api/
    """

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.api_url = "https://api.search.brave.com/res/v1/web/search"

    def search(self, query: str, max_results: int = 10, **kwargs) -> SearchResponse:
        if not HAS_REQUESTS:
            raise RuntimeError("requests library not installed")

        start_time = time.time()

        try:
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key
            }

            params = {
                "q": query,
                "count": max_results,
                "safesearch": kwargs.get('safesearch', 'moderate')
            }

            response = requests.get(
                self.api_url,
                headers=headers,
                params=params,
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            duration = time.time() - start_time

            # Parse results
            results = []
            for item in data.get('web', {}).get('results', [])[:max_results]:
                results.append(SearchResult(
                    title=item.get('title', ''),
                    url=item.get('url', ''),
                    snippet=item.get('description', ''),
                    score=float(item.get('age', 0)),
                    metadata={'age': item.get('age')}
                ))

            self._update_stats(True, duration)

            return SearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                search_time=duration,
                provider='brave',
                raw_response=data
            )

        except Exception as e:
            duration = time.time() - start_time
            self._update_stats(False, duration)
            logger.error(f"Brave search error: {e}")
            raise


class WebSearchPlugin:
    """
    统一的 Web 搜索插件接口

    支持多个搜索后端，自动回退机制。
    """

    def __init__(
        self,
        provider: Union[str, SearchProvider] = SearchProvider.DUCKDUCKGO,
        api_key: Optional[str] = None,
        fallback_providers: Optional[List[Union[str, SearchProvider]]] = None,
        **kwargs
    ):
        """
        Args:
            provider: 主搜索提供商
            api_key: API密钥 (如需要)
            fallback_providers: 备用提供商列表
            **kwargs: 额外配置
        """
        self.primary_provider = SearchProvider(provider) if isinstance(provider, str) else provider
        self.fallback_providers = [
            SearchProvider(p) if isinstance(p, str) else p
            for p in (fallback_providers or [])
        ]

        # 初始化后端
        self.backends: Dict[SearchProvider, BaseSearchBackend] = {}
        self._init_backend(self.primary_provider, api_key, **kwargs)

        # 初始化备用后端
        for fb_provider in self.fallback_providers:
            try:
                self._init_backend(fb_provider, api_key, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to init fallback provider {fb_provider}: {e}")

        logger.info(f"WebSearchPlugin initialized with provider={self.primary_provider.value}")

    def _init_backend(self, provider: SearchProvider, api_key: Optional[str], **kwargs):
        """初始化搜索后端"""
        backend_map = {
            SearchProvider.TAVILY: TavilyBackend,
            SearchProvider.PERPLEXITY: PerplexityBackend,
            SearchProvider.DUCKDUCKGO: DuckDuckGoBackend,
            SearchProvider.SERPER: SerperBackend,
            SearchProvider.BRAVE: BraveBackend,
        }

        backend_class = backend_map.get(provider)
        if not backend_class:
            raise ValueError(f"Unknown provider: {provider}")

        # DuckDuckGo不需要API key
        if provider == SearchProvider.DUCKDUCKGO:
            self.backends[provider] = backend_class(**kwargs)
        else:
            if not api_key:
                raise ValueError(f"API key required for provider: {provider}")
            self.backends[provider] = backend_class(api_key, **kwargs)

    def search(
        self,
        query: str,
        max_results: int = 10,
        use_fallback: bool = True,
        **kwargs
    ) -> SearchResponse:
        """
        执行搜索

        Args:
            query: 搜索查询
            max_results: 最大结果数
            use_fallback: 是否使用备用提供商
            **kwargs: 额外参数

        Returns:
            SearchResponse
        """
        # 尝试主提供商
        try:
            backend = self.backends[self.primary_provider]
            return backend.search(query, max_results, **kwargs)
        except Exception as e:
            logger.error(f"Primary provider {self.primary_provider.value} failed: {e}")

            if not use_fallback:
                raise

            # 尝试备用提供商
            for fb_provider in self.fallback_providers:
                if fb_provider not in self.backends:
                    continue

                try:
                    logger.info(f"Trying fallback provider: {fb_provider.value}")
                    backend = self.backends[fb_provider]
                    return backend.search(query, max_results, **kwargs)
                except Exception as fb_error:
                    logger.error(f"Fallback provider {fb_provider.value} failed: {fb_error}")

            # 所有提供商都失败
            raise RuntimeError(f"All search providers failed for query: {query}")

    def get_stats(self) -> Dict[str, Any]:
        """获取所有后端的统计信息"""
        stats = {}
        for provider, backend in self.backends.items():
            stats[provider.value] = backend.get_stats()
        return stats


# ==================== 便捷函数 ====================

def quick_search(
    query: str,
    provider: str = 'duckduckgo',
    api_key: Optional[str] = None,
    max_results: int = 10
) -> List[Dict[str, str]]:
    """
    快速搜索函数

    Args:
        query: 搜索查询
        provider: 搜索提供商
        api_key: API密钥
        max_results: 最大结果数

    Returns:
        结果列表 (简化格式)
    """
    plugin = WebSearchPlugin(provider=provider, api_key=api_key)
    response = plugin.search(query, max_results=max_results)

    return [
        {
            'title': r.title,
            'url': r.url,
            'snippet': r.snippet,
            'score': r.score
        }
        for r in response.results
    ]


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("Web Search Plugin Demo")
    print("=" * 70)

    # Demo 1: DuckDuckGo (免费)
    print("\n1. Testing DuckDuckGo (free, no API key needed)")
    print("-" * 70)

    try:
        plugin = WebSearchPlugin(provider='duckduckgo')
        response = plugin.search("Python machine learning", max_results=3)

        print(f"Query: {response.query}")
        print(f"Provider: {response.provider}")
        print(f"Search time: {response.search_time:.2f}s")
        print(f"Results: {response.total_results}\n")

        for i, result in enumerate(response.results, 1):
            print(f"{i}. {result.title}")
            print(f"   URL: {result.url}")
            print(f"   {result.snippet[:100]}...\n")

    except Exception as e:
        print(f"Error: {e}")

    # Demo 2: 快速搜索
    print("\n2. Quick Search Demo")
    print("-" * 70)

    try:
        results = quick_search("GPT models", max_results=2)
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['title']}")
            print(f"   {r['url']}\n")
    except Exception as e:
        print(f"Error: {e}")

    print("=" * 70)
    print("Demo Complete")
    print("=" * 70)
