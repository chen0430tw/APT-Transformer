# APT Agent System 技术指南

**自主工具调用和决策系统**

版本: 1.0
日期: 2026-01-21
作者: Claude + 430

---

## 概述

APT Agent System 是一个完整的 Agent 工具调用和决策系统，让模型能够：

1. **自主判断**何时需要使用工具
2. **安全执行** Python 代码（沙盒环境）
3. **联网搜索**获取最新信息
4. **ReAct 循环**：推理-行动-观察的决策循环

### 核心特性

- ✅ **工具系统**：灵活的工具注册、发现和调用
- ✅ **Python 沙盒**：多层安全机制的代码执行
- ✅ **Web 搜索**：支持多种搜索引擎
- ✅ **Agent 循环**：ReAct 风格的自主决策
- ✅ **并行调用**：支持并行执行多个工具
- ✅ **MCP 兼容**：符合 Model Context Protocol 标准

---

## 参考资料

本系统参考了以下最新技术和最佳实践：

### Agent 架构
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Model Context Protocol (MCP) 2025-11-25 Spec](https://modelcontextprotocol.io/)
- [OpenAI Function Calling API](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use API](https://docs.anthropic.com/claude/docs/tool-use)

### 代码沙盒
- [E2B: The Enterprise AI Agent Cloud](https://e2b.dev/)
- [Modal Sandboxes](https://modal.com/docs/examples/agent)
- [langchain-sandbox](https://github.com/langchain-ai/langchain-sandbox)
- [RestrictedPython](https://github.com/zopefoundation/RestrictedPython)

### 搜索引擎
- [DuckDuckGo API](https://duckduckgo.com/api)
- [Serper.dev](https://serper.dev/)

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     APT Agent System                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌─────────────────┐                  │
│  │  Tool System │      │  Agent Loop     │                  │
│  ├──────────────┤      ├─────────────────┤                  │
│  │ • Registry   │◄─────┤ • ReAct         │                  │
│  │ • Executor   │      │ • Reasoning     │                  │
│  │ • Cache      │      │ • Tool Selection│                  │
│  └──────────────┘      └─────────────────┘                  │
│         ▲                       ▲                            │
│         │                       │                            │
│  ┌──────┴───────────────────────┴──────┐                   │
│  │           Tools                      │                   │
│  ├──────────────────────────────────────┤                   │
│  │ ┌─────────────┐  ┌────────────────┐ │                   │
│  │ │Python Sandbox│  │  Web Search    │ │                   │
│  │ ├─────────────┤  ├────────────────┤ │                   │
│  │ │• AST Check  │  │• DuckDuckGo    │ │                   │
│  │ │• Resource   │  │• Google        │ │                   │
│  │ │• Timeout    │  │• Bing          │ │                   │
│  │ └─────────────┘  └────────────────┘ │                   │
│  │                                      │                   │
│  │ ┌─────────────┐  ┌────────────────┐ │                   │
│  │ │AIM-Memory   │  │  Custom Tools  │ │                   │
│  │ │Integration  │  │  (Extensible)  │ │                   │
│  │ └─────────────┘  └────────────────┘ │                   │
│  └──────────────────────────────────────┘                   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 安装依赖

```bash
# 核心依赖（已包含在项目中）
# 可选：安装 aiohttp 用于真实的 Web 搜索
pip install aiohttp

# 可选：安装 RestrictedPython 增强沙盒安全性
pip install RestrictedPython
```

### 基础示例

```python
import asyncio
from apt_model.agent import (
    register_tool, tool, ToolParameter, ToolType,
    ToolExecutor, get_tool_registry
)

# 1. 定义工具（使用装饰器）
@tool(
    name="calculator",
    description="Perform arithmetic calculations",
    tool_type=ToolType.COMPUTATION,
    parameters=[
        ToolParameter("expression", "string", "Math expression")
    ]
)
async def calculator(expression: str):
    return eval(expression, {"__builtins__": {}}, {})

# 2. 执行工具
async def main():
    executor = ToolExecutor(get_tool_registry())
    result = await executor.execute_single(
        "calculator",
        {"expression": "3 + 5 * 2"}
    )
    print(f"Result: {result.result}")

asyncio.run(main())
```

---

## 核心组件

### 1. 工具系统 (`tool_system.py`)

#### 工具定义

```python
from apt_model.agent import ToolDefinition, ToolParameter, ToolType

definition = ToolDefinition(
    name="my_tool",
    description="Tool description",
    parameters=[
        ToolParameter(
            name="param1",
            type="string",  # "string", "number", "boolean", "array", "object"
            description="Parameter description",
            required=True
        )
    ],
    tool_type=ToolType.CUSTOM,
    timeout=10.0,
    allow_parallel=True,
    cacheable=False
)
```

#### 工具注册

**方法 1: 使用装饰器（推荐）**

```python
from apt_model.agent import tool, ToolType, ToolParameter

@tool(
    name="my_tool",
    description="My custom tool",
    tool_type=ToolType.CUSTOM,
    parameters=[
        ToolParameter("input", "string", "Input text")
    ]
)
async def my_tool(input: str):
    return f"Processed: {input}"
```

**方法 2: 手动注册**

```python
from apt_model.agent import Tool, register_tool

class MyTool(Tool):
    async def execute(self, arguments, context=None):
        return {"result": "success"}

tool_instance = MyTool(definition)
register_tool(tool_instance)
```

#### 工具调用

**单次调用**

```python
executor = ToolExecutor(registry)
result = await executor.execute_single(
    tool_name="my_tool",
    arguments={"input": "hello"}
)

if result.success:
    print(f"Result: {result.result}")
else:
    print(f"Error: {result.error}")
```

**并行调用**

```python
calls = [
    {"tool_name": "tool1", "arguments": {"arg": "value1"}},
    {"tool_name": "tool2", "arguments": {"arg": "value2"}},
]

results = await executor.execute_parallel(calls)

for result in results:
    print(f"{result.tool_name}: {result.result}")
```

---

### 2. Python 沙盒 (`python_sandbox.py`)

#### 安全机制

1. **AST 静态检查**：在执行前检查代码安全性
2. **受限命名空间**：只包含安全的内置函数
3. **资源限制**：CPU、内存、时间限制
4. **输出截断**：防止输出过大

#### 使用示例

```python
from apt_model.agent import PythonSandbox, SandboxConfig

# 创建沙盒
config = SandboxConfig(
    max_execution_time=5.0,  # 秒
    max_memory_mb=100,       # MB
    max_output_size=10000,   # 字符
    allow_imports=False,     # 禁止 import
    allow_file_ops=False     # 禁止文件操作
)

sandbox = PythonSandbox(config)

# 执行代码
code = """
result = sum([1, 2, 3, 4, 5])
print(f"Sum: {result}")
__result__ = result
"""

success, result, error = sandbox.execute(code)

if success:
    print(f"Result: {result}")
else:
    print(f"Error: {error}")
```

#### 作为工具使用

```python
from apt_model.agent import PythonCodeExecutorTool, register_tool

# 注册 Python 执行工具
python_tool = PythonCodeExecutorTool()
register_tool(python_tool)

# 通过工具系统调用
result = await executor.execute_single(
    "python_executor",
    {
        "code": "result = 2 ** 10\n__result__ = result",
        "mode": "script"  # or "expression"
    }
)
```

---

### 3. Web 搜索 (`web_search.py`)

#### 支持的搜索引擎

- **MockSearchEngine**: 模拟搜索（用于测试，无需网络）
- **DuckDuckGoSearch**: DuckDuckGo 搜索（免费，无需 API key）
- **更多引擎**: 可扩展支持 Google、Bing、Serper.dev 等

#### 使用示例

```python
from apt_model.agent.tools.web_search import create_web_search_tool
from apt_model.agent import register_tool

# 创建并注册搜索工具
web_search = create_web_search_tool(
    search_engine="mock",  # or "duckduckgo"
    api_key=None  # 如果需要
)
register_tool(web_search)

# 执行搜索
result = await executor.execute_single(
    "web_search",
    {
        "query": "transformer architecture",
        "num_results": 5
    }
)

search_result = result.result
for item in search_result['results']:
    print(f"{item['title']}: {item['url']}")
```

---

### 4. ReAct Agent (`agent_loop.py`)

#### ReAct 循环

```
┌──────────────────────────────────────────────┐
│  User Question                                │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Step 1: Thought                              │
│  "I need to calculate the result"            │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Step 2: Action + Action Input               │
│  Action: python_executor                     │
│  Input: {"code": "..."}                      │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Step 3: Observation                         │
│  Result: 13                                  │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│  Step 4: Thought                              │
│  "I have the answer now"                     │
│  Final Answer: The result is 13              │
└──────────────────────────────────────────────┘
```

#### 使用示例

```python
from apt_model.agent.agent_loop import create_react_agent, AgentConfig

# 创建 Agent
config = AgentConfig(
    max_steps=10,
    max_execution_time=60.0
)

agent = create_react_agent(
    enable_python=True,
    enable_web_search=True,
    config=config,
    llm_callable=your_llm_function  # 可选：提供 LLM 函数
)

# 运行 Agent
result = await agent.run("Calculate 3 + 5 * 2")

if result.success:
    print(f"Answer: {result.final_answer}")

    # 查看执行步骤
    for step in result.steps:
        print(f"Step {step.step_num}: {step.thought}")
        if step.action:
            print(f"  Action: {step.action}({step.action_input})")
        if step.observation:
            print(f"  Observation: {step.observation}")
else:
    print(f"Error: {result.error}")
```

---

## 与 APT 其他组件集成

### 集成 AIM-Memory

```python
from apt_model.memory.aim_memory_nc import create_aim_memory_nc
from apt_model.agent import Tool, ToolDefinition, ToolParameter, ToolType

class AIMMemoryTool(Tool):
    def __init__(self):
        definition = ToolDefinition(
            name="memory_search",
            description="Search in AIM-Memory for relevant information",
            parameters=[
                ToolParameter("query", "string", "Search query")
            ],
            tool_type=ToolType.RETRIEVAL
        )
        super().__init__(definition)
        self.aim = create_aim_memory_nc()

    async def execute(self, arguments, context=None):
        query = arguments["query"]
        selected, refill = self.aim.route_memory(query, mode='fast')

        results = [
            {"summary": node.summary, "fields": node.fields}
            for node in selected
        ]

        return {
            "results": results,
            "num_results": len(results)
        }

# 注册
memory_tool = AIMMemoryTool()
register_tool(memory_tool)
```

### 集成 MCP

```python
# APT Agent System 已经与 MCP 兼容

# 获取 OpenAI Function Calling 格式
definitions = registry.get_definitions(format="openai")

# 获取 MCP 格式
definitions = registry.get_definitions(format="mcp")
```

---

## 高级用法

### 自定义工具

```python
from apt_model.agent import Tool, ToolDefinition, ToolParameter, ToolType

class DatabaseQueryTool(Tool):
    def __init__(self, db_connection):
        definition = ToolDefinition(
            name="db_query",
            description="Query database",
            parameters=[
                ToolParameter("sql", "string", "SQL query")
            ],
            tool_type=ToolType.DATABASE,
            timeout=10.0
        )
        super().__init__(definition)
        self.db = db_connection

    async def execute(self, arguments, context=None):
        sql = arguments["sql"]
        # 执行查询
        results = await self.db.execute(sql)
        return {"rows": results}
```

### 工具缓存

```python
# 启用缓存
definition = ToolDefinition(
    name="expensive_tool",
    description="A tool with expensive operations",
    parameters=[...],
    cacheable=True,      # 启用缓存
    cache_ttl=3600.0     # 缓存生存时间（秒）
)
```

### 并行限制

```python
# 限制并行数量
executor = ToolExecutor(registry, max_parallel=3)

# 有些工具不允许并行
definition = ToolDefinition(
    name="sequential_tool",
    allow_parallel=False  # 禁止并行
)
```

---

## 配置和调优

### Agent 配置

```python
config = AgentConfig(
    max_steps=10,              # 最大步骤数
    max_execution_time=60.0,   # 最大执行时间（秒）
    enable_python=True,        # 启用 Python 执行
    enable_web_search=True,    # 启用 Web 搜索
    enable_memory=True,        # 启用记忆系统
    decision_mode="auto",      # 决策模式 ("auto"/"manual")
    require_confirmation=False # 是否需要用户确认
)
```

### 沙盒配置

```python
sandbox_config = SandboxConfig(
    max_execution_time=5.0,    # 执行超时
    max_memory_mb=100,         # 内存限制
    max_output_size=10000,     # 输出限制
    allow_imports=False,       # 是否允许 import
    allowed_modules=[          # 允许的模块白名单
        "math", "random", "datetime", "json", "re"
    ],
    allow_file_ops=False,      # 是否允许文件操作
    allow_network=False        # 是否允许网络操作
)
```

---

## 完整示例

运行完整示例：

```bash
python examples/agent_demo.py
```

这个示例展示了：
1. 工具注册和调用
2. Python 代码沙盒执行
3. Web 搜索
4. 并行工具调用
5. ReAct Agent 决策循环
6. 工具统计信息

---

## 性能和安全

### 性能优化

1. **并行调用**: 使用 `execute_parallel` 并行执行多个工具
2. **缓存**: 对昂贵的操作启用缓存
3. **超时控制**: 设置合理的超时时间
4. **资源限制**: 限制内存和 CPU 使用

### 安全措施

**Python 沙盒**：
- ✅ AST 静态检查
- ✅ 受限命名空间
- ✅ 资源限制（时间、内存）
- ✅ 禁止危险操作（文件、网络、eval）
- ✅ 输出截断

**工具调用**：
- ✅ 参数验证
- ✅ 超时控制
- ✅ 错误处理
- ✅ 执行隔离

---

## 限制和改进方向

### 当前限制

1. **Python 沙盒**:
   - Windows 上部分限制功能不可用（signal）
   - 未集成 Docker/gVisor 等容器化隔离

2. **Web 搜索**:
   - DuckDuckGo HTML 解析较简单
   - 未集成所有主流搜索引擎 API

3. **Agent 循环**:
   - LLM 调用需要自定义实现
   - 未集成多模态能力

### 改进方向

1. **增强沙盒**: 集成 E2B/Modal/Pyodide
2. **更多工具**: 文件操作、数据库、API 调用
3. **多模态**: 图像理解、语音识别
4. **分布式**: 支持分布式工具执行
5. **可视化**: Agent 执行过程可视化

---

## 技术来源

- **作者**: Claude + 430
- **版本**: 1.0
- **日期**: 2026-01-21

**参考资料** (见文档开头)

---

## 相关文档

- [AIM-Memory 技术指南](AIM_MEMORY_GUIDE.md)
- [AIM-NC 技术指南](AIM_NC_GUIDE.md)
- [集成总结文档](INTEGRATION_SUMMARY.md)

---

## FAQ

**Q: 如何添加自定义工具？**

A: 继承 `Tool` 类或使用 `@tool` 装饰器。参见"自定义工具"章节。

**Q: Python 沙盒安全吗？**

A: 在 Unix 系统上提供多层安全保护（AST检查 + 资源限制 + 命名空间隔离）。生产环境建议使用 Docker/gVisor。

**Q: 如何集成真实的 LLM？**

A: 在创建 Agent 时传入 `llm_callable` 函数：
```python
async def my_llm(prompt: str) -> str:
    # 调用 GPT/Claude/Llama 等
    return response

agent = create_react_agent(llm_callable=my_llm)
```

**Q: 支持哪些搜索引擎？**

A: 目前支持 MockSearchEngine（测试用）和 DuckDuckGo。可扩展支持 Google、Bing、Serper.dev 等。

**Q: 如何与 AIM-Memory/AIM-NC 集成？**

A: 创建自定义工具继承 AIM-Memory 的 `route_memory` 功能。参见"集成 AIM-Memory"章节。
