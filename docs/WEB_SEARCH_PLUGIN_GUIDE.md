# AI 联网搜索插件使用指南

## 📋 概览

基于 2025 年最新技术栈的 AI 联网搜索插件，整合了业界主流的搜索 API，为 AI 模型提供实时网络信息检索能力。

## ✨ 支持的搜索后端

### 1. **Tavily** (AI-Native，推荐用于生产)
- ⭐ 专为 AI agents 设计
- 💰 每月 1000 免费 credits，$0.008/credit
- ⚡ 返回结构化 JSON with summaries
- 🎯 包含 search/extract/crawl 统一 API
- 📚 [官方文档](https://docs.tavily.com/)

### 2. **Perplexity** (速度最快)
- ⚡ 中位数 <400ms 响应时间
- 💰 $5/1000 请求
- 🔒 零数据保留政策
- 🧠 多个 Sonar 模型 (basic/pro/reasoning)
- 📚 [官方文档](https://docs.perplexity.ai/)

### 3. **DuckDuckGo** (免费，隐私友好)
- 🆓 完全免费，无需 API key
- 🔒 注重隐私
- 🔌 LangChain 原生支持
- ⚠️ 无请求速率限制，但结果质量稍低
- 📚 [Python 库](https://pypi.org/project/duckduckgo-search/)

### 4. **Serper.dev** (Google SERP)
- 🔍 访问 Google 搜索结果
- ⚡ ~2 秒返回
- 🔌 LangChain 内置支持
- 💰 按请求计费
- 📚 [官方文档](https://serper.dev/)

### 5. **Brave Search** (独立索引)
- 🦁 独立搜索索引
- 🔒 隐私优先
- 🆓 免费层可用
- 📚 [API 文档](https://brave.com/search/api/)

---

## 🚀 快速开始

### 安装依赖

```bash
# 基础依赖
pip install requests

# DuckDuckGo (推荐，免费)
pip install duckduckgo-search

# 可选：其他库
pip install beautifulsoup4  # HTML 解析
pip install aiohttp         # 异步请求
```

### 1. 使用 DuckDuckGo (免费，无需 API key)

```python
from apt_model.plugins.web_search_plugin import WebSearchPlugin

# 创建插件 (默认使用 DuckDuckGo)
plugin = WebSearchPlugin(provider='duckduckgo')

# 搜索
response = plugin.search("Python machine learning tutorials", max_results=5)

# 查看结果
print(f"找到 {response.total_results} 个结果 (耗时 {response.search_time:.2f}s)")
for i, result in enumerate(response.results, 1):
    print(f"\n{i}. {result.title}")
    print(f"   URL: {result.url}")
    print(f"   摘要: {result.snippet}")
    print(f"   分数: {result.score}")
```

### 2. 使用 Tavily (AI-Native)

```python
from apt_model.plugins.web_search_plugin import WebSearchPlugin

# 需要 API key (注册: https://tavily.com/)
plugin = WebSearchPlugin(
    provider='tavily',
    api_key='tvly-YOUR_API_KEY'
)

# 搜索（Tavily 支持更多选项）
response = plugin.search(
    query="Latest AI research 2025",
    max_results=5,
    search_depth='advanced',     # 'basic' or 'advanced'
    include_answer=True,          # 是否包含 AI 生成的答案
    include_images=False,         # 是否包含图片
    include_raw_content=False     # 是否包含原始 HTML
)

print(f"查询: {response.query}")
for result in response.results:
    print(f"\n标题: {result.title}")
    print(f"链接: {result.url}")
    print(f"内容: {result.snippet}")
    print(f"相关性: {result.score:.3f}")
```

### 3. 快速搜索函数

```python
from apt_model.plugins.web_search_plugin import quick_search

# 一行代码搜索
results = quick_search("GPT models", provider='duckduckgo', max_results=3)

for r in results:
    print(f"{r['title']} - {r['url']}")
```

---

## 🔧 高级用法

### 1. 多后端 + 自动回退

```python
from apt_model.plugins.web_search_plugin import WebSearchPlugin

# 主后端 + 备用后端
plugin = WebSearchPlugin(
    provider='tavily',
    api_key='your_tavily_key',
    fallback_providers=['duckduckgo', 'brave']  # 自动回退
)

# 如果 Tavily 失败，会自动尝试 DuckDuckGo，然后 Brave
response = plugin.search("AI news", max_results=5)
```

### 2. Perplexity with 时间过滤

```python
plugin = WebSearchPlugin(
    provider='perplexity',
    api_key='your_perplexity_key'
)

# 只搜索最近一天的结果
response = plugin.search(
    query="breaking AI news",
    max_results=10,
    recency='day'  # 'day', 'week', 'month', 'year'
)
```

### 3. Serper.dev (Google SERP)

```python
plugin = WebSearchPlugin(
    provider='serper',
    api_key='your_serper_key'
)

# Google 搜索结果
response = plugin.search(
    query="machine learning papers",
    max_results=10,
    gl='us',  # 国家代码
    hl='en'   # 语言
)
```

### 4. 获取统计信息

```python
plugin = WebSearchPlugin(provider='duckduckgo')

# 执行多次搜索
plugin.search("AI", max_results=5)
plugin.search("ML", max_results=5)
plugin.search("DL", max_results=5)

# 查看统计
stats = plugin.get_stats()
print(stats)
# {
#   'duckduckgo': {
#     'total_searches': 3,
#     'successful_searches': 3,
#     'failed_searches': 0,
#     'total_time': 1.23,
#     'avg_search_time': 0.41,
#     'success_rate': 1.0
#   }
# }
```

---

## 🎯 与 GPT 模型集成

### 示例 1: 搜索增强生成 (RAG)

```python
import torch
from apt_model.modeling.gpt4o_model import GPT4oModel
from apt_model.plugins.web_search_plugin import WebSearchPlugin

# 初始化模型和搜索
model = GPT4oModel(vocab_size=50257, d_model=768, n_layers=12)
search_plugin = WebSearchPlugin(provider='duckduckgo')

def search_augmented_generation(query: str, model, tokenizer, search_plugin):
    """使用搜索结果增强生成"""

    # 1. 搜索相关信息
    search_response = search_plugin.search(query, max_results=3)

    # 2. 构建上下文
    context = f"Query: {query}\n\nRelevant information:\n"
    for i, result in enumerate(search_response.results, 1):
        context += f"{i}. {result.title}\n{result.snippet}\n\n"

    # 3. 使用上下文生成回答
    input_ids = tokenizer.encode(context, return_tensors='pt')

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=100,
            temperature=0.8,
            top_p=0.9
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response, search_response

# 使用
response, search_results = search_augmented_generation(
    "What are the latest developments in AI?",
    model,
    tokenizer,
    search_plugin
)

print(f"Generated response:\n{response}\n")
print(f"Sources:")
for r in search_results.results:
    print(f"- {r.title}: {r.url}")
```

### 示例 2: 实时问答系统

```python
from apt_model.plugins.web_search_plugin import WebSearchPlugin

class RealtimeQA:
    """实时问答系统，结合搜索和生成"""

    def __init__(self, model, tokenizer, search_provider='duckduckgo', api_key=None):
        self.model = model
        self.tokenizer = tokenizer
        self.search = WebSearchPlugin(provider=search_provider, api_key=api_key)

    def answer(self, question: str, use_search: bool = True) -> dict:
        """回答问题"""

        result = {
            'question': question,
            'answer': '',
            'sources': [],
            'used_search': use_search
        }

        if use_search:
            # 搜索相关信息
            search_response = self.search.search(question, max_results=5)

            # 提取最相关的片段
            context_snippets = [r.snippet for r in search_response.results[:3]]
            context = "\n\n".join(context_snippets)

            # 保存来源
            result['sources'] = [
                {'title': r.title, 'url': r.url}
                for r in search_response.results
            ]
        else:
            context = question

        # 生成回答
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        output = self.model.generate(
            input_ids,
            max_new_tokens=150,
            temperature=0.7
        )

        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        result['answer'] = answer.split('Answer:')[-1].strip()

        return result

# 使用
qa_system = RealtimeQA(model, tokenizer, search_provider='duckduckgo')

response = qa_system.answer("What is quantum computing?")
print(f"Q: {response['question']}")
print(f"A: {response['answer']}")
print(f"\nSources:")
for src in response['sources']:
    print(f"  - {src['title']}: {src['url']}")
```

### 示例 3: 多查询聚合

```python
def multi_query_search(queries: list, plugin: WebSearchPlugin, max_per_query: int = 3):
    """
    执行多个查询并聚合结果
    """
    all_results = []
    seen_urls = set()

    for query in queries:
        response = plugin.search(query, max_results=max_per_query)

        for result in response.results:
            # 去重
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                all_results.append({
                    'query': query,
                    'title': result.title,
                    'url': result.url,
                    'snippet': result.snippet,
                    'score': result.score
                })

    # 按分数排序
    all_results.sort(key=lambda x: x['score'], reverse=True)

    return all_results

# 使用
plugin = WebSearchPlugin(provider='duckduckgo')

queries = [
    "transformer architecture",
    "attention mechanism explained",
    "self-attention tutorial"
]

results = multi_query_search(queries, plugin, max_per_query=2)

print(f"Found {len(results)} unique results across {len(queries)} queries:\n")
for i, r in enumerate(results[:5], 1):
    print(f"{i}. [{r['query']}] {r['title']}")
    print(f"   {r['url']}\n")
```

---

## 📊 性能对比 (2025)

| Provider | 平均响应时间 | 结果质量 | 成本 | 隐私 | 推荐场景 |
|----------|------------|---------|------|------|---------|
| **Tavily** | ~1.5s | ⭐⭐⭐⭐⭐ | $0.008/req | 中 | 生产环境、AI agents |
| **Perplexity** | <400ms | ⭐⭐⭐⭐ | $0.005/req | 高 | 实时应用、速度优先 |
| **DuckDuckGo** | ~2s | ⭐⭐⭐ | 免费 | 高 | 开发测试、隐私优先 |
| **Serper** | ~2s | ⭐⭐⭐⭐⭐ | 按量计费 | 中 | Google 搜索结果 |
| **Brave** | ~1.8s | ⭐⭐⭐⭐ | 免费层 | 高 | 独立索引、隐私优先 |

---

## 🛠️ 故障排查

### 问题 1: 导入错误

**症状**: `ImportError: No module named 'duckduckgo_search'`

**解决方案**:
```bash
pip install duckduckgo-search
```

### 问题 2: API key 无效

**症状**: `401 Unauthorized`

**解决方案**:
1. 检查 API key 是否正确
2. 确认 API key 有效期
3. 检查账户余额（付费服务）

### 问题 3: 请求超时

**症状**: `requests.exceptions.Timeout`

**解决方案**:
```python
# 增加超时时间
plugin = WebSearchPlugin(
    provider='tavily',
    api_key='your_key',
    timeout=60  # 60 秒
)
```

### 问题 4: 速率限制

**症状**: `429 Too Many Requests`

**解决方案**:
```python
import time

# 添加重试逻辑
def search_with_retry(plugin, query, max_retries=3):
    for attempt in range(max_retries):
        try:
            return plugin.search(query)
        except Exception as e:
            if '429' in str(e) and attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避
                print(f"Rate limited, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
```

---

## 📚 API 参考

### WebSearchPlugin

#### `__init__(provider, api_key=None, fallback_providers=None, **kwargs)`

创建搜索插件实例。

**参数**:
- `provider` (str | SearchProvider): 主搜索提供商
- `api_key` (str, optional): API 密钥
- `fallback_providers` (list, optional): 备用提供商列表
- `**kwargs`: 额外配置选项

**示例**:
```python
plugin = WebSearchPlugin(
    provider='tavily',
    api_key='your_key',
    fallback_providers=['duckduckgo']
)
```

---

#### `search(query, max_results=10, use_fallback=True, **kwargs)`

执行搜索。

**参数**:
- `query` (str): 搜索查询
- `max_results` (int): 最大结果数
- `use_fallback` (bool): 是否使用备用提供商
- `**kwargs`: 提供商特定参数

**返回**: `SearchResponse`

**示例**:
```python
response = plugin.search("AI news", max_results=5)
```

---

#### `get_stats()`

获取统计信息。

**返回**: Dict[str, Any]

**示例**:
```python
stats = plugin.get_stats()
print(f"Success rate: {stats['duckduckgo']['success_rate']:.1%}")
```

---

### SearchResponse

搜索响应数据类。

**字段**:
- `query` (str): 搜索查询
- `results` (List[SearchResult]): 搜索结果列表
- `total_results` (int): 结果总数
- `search_time` (float): 搜索耗时（秒）
- `provider` (str): 使用的提供商
- `raw_response` (dict, optional): 原始响应数据

---

### SearchResult

单个搜索结果。

**字段**:
- `title` (str): 标题
- `url` (str): URL
- `snippet` (str): 摘要/片段
- `score` (float): 相关性分数
- `metadata` (dict): 额外元数据

---

## 🌟 最佳实践

### 1. 选择合适的提供商

```python
# 开发/测试 → DuckDuckGo (免费)
dev_plugin = WebSearchPlugin(provider='duckduckgo')

# 生产环境 → Tavily (AI-optimized)
prod_plugin = WebSearchPlugin(provider='tavily', api_key=TAVILY_KEY)

# 速度优先 → Perplexity (<400ms)
fast_plugin = WebSearchPlugin(provider='perplexity', api_key=PERPLEXITY_KEY)
```

### 2. 实现缓存

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_search(query: str, provider: str = 'duckduckgo'):
    plugin = WebSearchPlugin(provider=provider)
    response = plugin.search(query, max_results=5)
    return [(r.title, r.url, r.snippet) for r in response.results]

# 重复查询会使用缓存
results1 = cached_search("machine learning")
results2 = cached_search("machine learning")  # 从缓存获取
```

### 3. 错误处理

```python
def safe_search(plugin, query, default_results=None):
    """安全的搜索，带错误处理"""
    try:
        response = plugin.search(query, max_results=5)
        return response.results
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return default_results or []

# 使用
results = safe_search(plugin, "AI news", default_results=[])
```

### 4. 结果后处理

```python
def filter_results(results: list, min_score: float = 0.5):
    """过滤低质量结果"""
    return [r for r in results if r.score >= min_score]

def deduplicate_results(results: list):
    """去重（基于 URL）"""
    seen = set()
    unique = []
    for r in results:
        if r.url not in seen:
            seen.add(r.url)
            unique.append(r)
    return unique
```

---

## 🔮 未来计划

- [ ] 支持异步搜索 (`async/await`)
- [ ] 添加结果缓存持久化
- [ ] 集成更多搜索引擎 (Bing, Yandex)
- [ ] 支持图片/视频搜索
- [ ] 添加搜索结果排序/过滤
- [ ] 集成 LangChain Tools

---

## 📖 参考资料

### 文章和对比

- [Perplexity vs Tavily 对比](https://alphacorp.ai/perplexity-search-api-vs-tavily-the-better-choice-for-rag-and-agents-in-2025/)
- [Tavily 深度解析](https://skywork.ai/skypage/en/unlocking-agentic-ai-tavily-search/1977931655987253248)
- [Tavily 融资新闻](https://techcrunch.com/2025/08/06/tavily-raises-25m-to-connect-ai-agents-to-the-web/)
- [Top 5 Anthropic 搜索替代方案](https://www.scrapeless.com/en/blog/anthropic-web-search-alternatives)
- [8 个最佳搜索 API 工具](https://data4ai.com/blog/tool-comparisons/best-search-api-tools/)

### 官方文档

- [Tavily API 文档](https://docs.tavily.com/)
- [Perplexity AI 文档](https://docs.perplexity.ai/)
- [DuckDuckGo Search Python](https://pypi.org/project/duckduckgo-search/)
- [Serper.dev 文档](https://serper.dev/)
- [Brave Search API](https://brave.com/search/api/)
- [LangChain DuckDuckGo Tool](https://python.langchain.com/v0.2/docs/integrations/tools/ddg/)
- [DataCamp: Building GPT with Browsing](https://www.datacamp.com/tutorial/building-a-gpt-model-with-browsing-capabilities-using-lang-chain-tools)

---

## 💡 贡献指南

欢迎贡献！提交 PR 前请：

1. 添加新的搜索后端时，继承 `BaseSearchBackend`
2. 确保所有测试通过
3. 更新本文档
4. 遵循代码风格

---

## 📧 支持

遇到问题？欢迎：
- 提交 Issue
- 查看文档
- 参考示例代码

---

**Happy Searching! 🔍**
