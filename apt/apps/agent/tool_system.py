#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Agent 工具调用系统

基于 MCP (Model Context Protocol) 和最新的 Agent 架构设计，
提供自主工具发现、调用和结果处理能力。

核心特性：
1. 工具注册和发现
2. 自动工具选择（Function Calling）
3. 并行工具调用
4. 结果缓存和错误处理
5. 与 AIM-Memory/AIM-NC 集成

参考文献：
- Model Context Protocol (MCP) 2025-11-25 Spec
- OpenAI Function Calling API
- Anthropic Tool Use API
- LangChain Tool System
"""

import json
import asyncio
import inspect
from typing import Dict, Any, List, Optional, Callable, Union, Literal
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import logging
import time

logger = logging.getLogger(__name__)


# ==================== 工具定义 ====================

class ToolType(Enum):
    """工具类型"""
    COMPUTATION = "computation"      # 计算工具（Python 沙盒）
    SEARCH = "search"                # 搜索工具（Web 搜索）
    RETRIEVAL = "retrieval"          # 检索工具（AIM-Memory）
    API = "api"                      # API 调用
    DATABASE = "database"            # 数据库查询
    FILE = "file"                    # 文件操作
    CUSTOM = "custom"                # 自定义工具


@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None  # 枚举值


@dataclass
class ToolDefinition:
    """工具定义（MCP 兼容）"""
    name: str
    description: str
    parameters: List[ToolParameter]
    tool_type: ToolType = ToolType.CUSTOM

    # 执行配置
    timeout: float = 30.0  # 秒
    allow_parallel: bool = True  # 是否允许并行调用
    require_confirmation: bool = False  # 是否需要用户确认

    # 缓存配置
    cacheable: bool = False
    cache_ttl: float = 300.0  # 缓存生存时间（秒）

    # 元数据
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_openai_format(self) -> Dict[str, Any]:
        """转换为 OpenAI Function Calling 格式"""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                properties[param.name]["enum"] = param.enum
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

    def to_mcp_format(self) -> Dict[str, Any]:
        """转换为 MCP 工具格式"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": {
                "type": "object",
                "properties": {
                    param.name: {
                        "type": param.type,
                        "description": param.description
                    } for param in self.parameters
                },
                "required": [p.name for p in self.parameters if p.required]
            }
        }


@dataclass
class ToolCallRequest:
    """工具调用请求"""
    tool_name: str
    arguments: Dict[str, Any]
    call_id: str  # 唯一调用 ID
    timestamp: float = field(default_factory=time.time)
    context: Optional[Dict[str, Any]] = None  # 上下文信息


@dataclass
class ToolCallResult:
    """工具调用结果"""
    call_id: str
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0  # 执行时间（秒）
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time,
            "metadata": self.metadata
        }


# ==================== 工具基类 ====================

class Tool(ABC):
    """工具抽象基类"""

    def __init__(self, definition: ToolDefinition):
        self.definition = definition
        self._cache = {}  # 简单缓存
        self._call_count = 0
        self._total_time = 0.0

    @abstractmethod
    async def execute(self, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
        """
        执行工具

        Args:
            arguments: 工具参数
            context: 执行上下文

        Returns:
            执行结果
        """
        pass

    def validate_arguments(self, arguments: Dict[str, Any]) -> bool:
        """验证参数"""
        for param in self.definition.parameters:
            if param.required and param.name not in arguments:
                raise ValueError(f"Missing required parameter: {param.name}")
        return True

    async def call(self, request: ToolCallRequest) -> ToolCallResult:
        """
        调用工具（带缓存和错误处理）

        Args:
            request: 工具调用请求

        Returns:
            工具调用结果
        """
        start_time = time.time()

        try:
            # 验证参数
            self.validate_arguments(request.arguments)

            # 检查缓存
            if self.definition.cacheable:
                cache_key = self._get_cache_key(request.arguments)
                if cache_key in self._cache:
                    cached_result, cache_time = self._cache[cache_key]
                    if time.time() - cache_time < self.definition.cache_ttl:
                        logger.info(f"[Tool] Cache hit for {self.definition.name}")
                        return ToolCallResult(
                            call_id=request.call_id,
                            tool_name=self.definition.name,
                            success=True,
                            result=cached_result,
                            execution_time=time.time() - start_time,
                            metadata={"cached": True}
                        )

            # 执行工具
            result = await asyncio.wait_for(
                self.execute(request.arguments, request.context),
                timeout=self.definition.timeout
            )

            # 更新缓存
            if self.definition.cacheable:
                cache_key = self._get_cache_key(request.arguments)
                self._cache[cache_key] = (result, time.time())

            # 更新统计
            execution_time = time.time() - start_time
            self._call_count += 1
            self._total_time += execution_time

            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                success=True,
                result=result,
                execution_time=execution_time
            )

        except asyncio.TimeoutError:
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                success=False,
                error=f"Tool execution timed out after {self.definition.timeout}s",
                execution_time=time.time() - start_time
            )

        except Exception as e:
            logger.error(f"[Tool] Error executing {self.definition.name}: {e}")
            return ToolCallResult(
                call_id=request.call_id,
                tool_name=self.definition.name,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )

    def _get_cache_key(self, arguments: Dict[str, Any]) -> str:
        """生成缓存键"""
        return json.dumps(arguments, sort_keys=True)

    def get_stats(self) -> Dict[str, Any]:
        """获取工具统计信息"""
        return {
            "name": self.definition.name,
            "call_count": self._call_count,
            "total_time": self._total_time,
            "avg_time": self._total_time / self._call_count if self._call_count > 0 else 0.0,
            "cache_size": len(self._cache) if self.definition.cacheable else 0
        }


# ==================== 工具注册表 ====================

class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._tools_by_type: Dict[ToolType, List[str]] = {t: [] for t in ToolType}
        self._tools_by_tag: Dict[str, List[str]] = {}

    def register(self, tool: Tool):
        """注册工具"""
        if tool.definition.name in self._tools:
            logger.warning(f"[ToolRegistry] Tool {tool.definition.name} already registered, overwriting")

        self._tools[tool.definition.name] = tool

        # 按类型索引
        tool_type = tool.definition.tool_type
        if tool.definition.name not in self._tools_by_type[tool_type]:
            self._tools_by_type[tool_type].append(tool.definition.name)

        # 按标签索引
        for tag in tool.definition.tags:
            if tag not in self._tools_by_tag:
                self._tools_by_tag[tag] = []
            if tool.definition.name not in self._tools_by_tag[tag]:
                self._tools_by_tag[tag].append(tool.definition.name)

        logger.info(f"[ToolRegistry] Registered tool: {tool.definition.name} ({tool_type.value})")

    def unregister(self, tool_name: str):
        """注销工具"""
        if tool_name not in self._tools:
            return

        tool = self._tools[tool_name]

        # 从类型索引移除
        tool_type = tool.definition.tool_type
        if tool_name in self._tools_by_type[tool_type]:
            self._tools_by_type[tool_type].remove(tool_name)

        # 从标签索引移除
        for tag in tool.definition.tags:
            if tag in self._tools_by_tag and tool_name in self._tools_by_tag[tag]:
                self._tools_by_tag[tag].remove(tool_name)

        del self._tools[tool_name]
        logger.info(f"[ToolRegistry] Unregistered tool: {tool_name}")

    def get(self, tool_name: str) -> Optional[Tool]:
        """获取工具"""
        return self._tools.get(tool_name)

    def list_all(self) -> List[str]:
        """列出所有工具名称"""
        return list(self._tools.keys())

    def list_by_type(self, tool_type: ToolType) -> List[str]:
        """按类型列出工具"""
        return self._tools_by_type.get(tool_type, [])

    def list_by_tag(self, tag: str) -> List[str]:
        """按标签列出工具"""
        return self._tools_by_tag.get(tag, [])

    def get_definitions(self, format: Literal["openai", "mcp", "native"] = "native") -> List[Dict[str, Any]]:
        """
        获取所有工具定义

        Args:
            format: 输出格式（openai/mcp/native）

        Returns:
            工具定义列表
        """
        definitions = []

        for tool in self._tools.values():
            if format == "openai":
                definitions.append(tool.definition.to_openai_format())
            elif format == "mcp":
                definitions.append(tool.definition.to_mcp_format())
            else:
                definitions.append(asdict(tool.definition))

        return definitions

    def get_stats(self) -> Dict[str, Any]:
        """获取注册表统计信息"""
        return {
            "total_tools": len(self._tools),
            "by_type": {t.value: len(names) for t, names in self._tools_by_type.items()},
            "by_tag": {tag: len(names) for tag, names in self._tools_by_tag.items()},
            "tool_stats": {name: tool.get_stats() for name, tool in self._tools.items()}
        }


# ==================== 工具调用器 ====================

class ToolExecutor:
    """工具执行器（支持并行调用）"""

    def __init__(self, registry: ToolRegistry, max_parallel: int = 5):
        self.registry = registry
        self.max_parallel = max_parallel
        self._call_counter = 0

    async def execute_single(self, tool_name: str, arguments: Dict[str, Any],
                           context: Optional[Dict[str, Any]] = None) -> ToolCallResult:
        """执行单个工具调用"""
        tool = self.registry.get(tool_name)
        if tool is None:
            return ToolCallResult(
                call_id=self._gen_call_id(),
                tool_name=tool_name,
                success=False,
                error=f"Tool '{tool_name}' not found"
            )

        request = ToolCallRequest(
            tool_name=tool_name,
            arguments=arguments,
            call_id=self._gen_call_id(),
            context=context
        )

        return await tool.call(request)

    async def execute_parallel(self, calls: List[Dict[str, Any]],
                             context: Optional[Dict[str, Any]] = None) -> List[ToolCallResult]:
        """
        并行执行多个工具调用

        Args:
            calls: 调用列表，每个元素包含 {"tool_name": str, "arguments": dict}
            context: 共享上下文

        Returns:
            结果列表
        """
        # 创建任务
        tasks = []
        for call in calls:
            tool_name = call["tool_name"]
            arguments = call.get("arguments", {})

            tool = self.registry.get(tool_name)
            if tool is None:
                # 工具不存在，直接返回错误结果
                tasks.append(asyncio.create_task(self._create_error_result(
                    tool_name, f"Tool '{tool_name}' not found"
                )))
                continue

            # 检查是否允许并行
            if not tool.definition.allow_parallel:
                logger.warning(f"[ToolExecutor] Tool {tool_name} does not allow parallel execution")

            request = ToolCallRequest(
                tool_name=tool_name,
                arguments=arguments,
                call_id=self._gen_call_id(),
                context=context
            )

            tasks.append(asyncio.create_task(tool.call(request)))

        # 执行任务（带并发限制）
        results = []
        for i in range(0, len(tasks), self.max_parallel):
            batch = tasks[i:i+self.max_parallel]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    results.append(ToolCallResult(
                        call_id=self._gen_call_id(),
                        tool_name="unknown",
                        success=False,
                        error=str(result)
                    ))
                else:
                    results.append(result)

        return results

    async def _create_error_result(self, tool_name: str, error: str) -> ToolCallResult:
        """创建错误结果"""
        return ToolCallResult(
            call_id=self._gen_call_id(),
            tool_name=tool_name,
            success=False,
            error=error
        )

    def _gen_call_id(self) -> str:
        """生成调用 ID"""
        self._call_counter += 1
        return f"call_{self._call_counter}_{int(time.time() * 1000)}"


# ==================== 全局注册表 ====================

_global_registry = ToolRegistry()

def get_tool_registry() -> ToolRegistry:
    """获取全局工具注册表"""
    return _global_registry

def register_tool(tool: Tool):
    """注册工具到全局注册表"""
    _global_registry.register(tool)

def get_tool(tool_name: str) -> Optional[Tool]:
    """从全局注册表获取工具"""
    return _global_registry.get(tool_name)


# ==================== 工具装饰器 ====================

def tool(name: str, description: str, tool_type: ToolType = ToolType.CUSTOM,
        parameters: Optional[List[ToolParameter]] = None, **kwargs):
    """
    工具装饰器（简化工具注册）

    使用示例:
    @tool(
        name="calculator",
        description="Perform mathematical calculations",
        tool_type=ToolType.COMPUTATION,
        parameters=[
            ToolParameter("expression", "string", "Mathematical expression to evaluate")
        ]
    )
    async def calculator(expression: str):
        return eval(expression)
    """
    def decorator(func: Callable):
        # 自动推断参数（如果未提供）
        if parameters is None:
            sig = inspect.signature(func)
            auto_params = []
            for param_name, param in sig.parameters.items():
                if param_name in ['self', 'cls', 'context']:
                    continue

                param_type = "string"  # 默认类型
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == int:
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif param.annotation == list:
                        param_type = "array"
                    elif param.annotation == dict:
                        param_type = "object"

                auto_params.append(ToolParameter(
                    name=param_name,
                    type=param_type,
                    description=f"Parameter {param_name}",
                    required=param.default == inspect.Parameter.empty
                ))

            tool_params = auto_params
        else:
            tool_params = parameters

        # 创建工具定义
        definition = ToolDefinition(
            name=name,
            description=description,
            parameters=tool_params,
            tool_type=tool_type,
            **kwargs
        )

        # 创建工具类
        class DecoratedTool(Tool):
            async def execute(self, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Any:
                if inspect.iscoroutinefunction(func):
                    return await func(**arguments)
                else:
                    return func(**arguments)

        # 注册工具
        tool_instance = DecoratedTool(definition)
        register_tool(tool_instance)

        return func

    return decorator


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=== Tool System Demo ===\n")

    # 定义一个简单工具
    @tool(
        name="add",
        description="Add two numbers",
        tool_type=ToolType.COMPUTATION,
        parameters=[
            ToolParameter("a", "number", "First number"),
            ToolParameter("b", "number", "Second number")
        ],
        cacheable=True
    )
    async def add(a: float, b: float) -> float:
        return a + b

    # 测试工具调用
    async def test():
        registry = get_tool_registry()
        executor = ToolExecutor(registry)

        # 单次调用
        result = await executor.execute_single("add", {"a": 3, "b": 5})
        print(f"Result: {result.result}, Time: {result.execution_time:.4f}s")

        # 并行调用
        calls = [
            {"tool_name": "add", "arguments": {"a": 1, "b": 2}},
            {"tool_name": "add", "arguments": {"a": 3, "b": 4}},
            {"tool_name": "add", "arguments": {"a": 5, "b": 6}},
        ]
        results = await executor.execute_parallel(calls)
        print(f"\nParallel results:")
        for r in results:
            print(f"  {r.tool_name}({r.call_id}): {r.result}")

        # 统计信息
        stats = registry.get_stats()
        print(f"\nStats: {json.dumps(stats, indent=2)}")

    asyncio.run(test())

    print("\n=== Demo Complete ===")
