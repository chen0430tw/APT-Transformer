#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Agent System 完整示例

展示如何使用 Agent 系统的所有功能：
1. 工具注册和调用
2. Python 代码执行（沙盒）
3. Web 搜索
4. ReAct Agent 循环
5. 与 AIM-Memory 集成（可选）

使用方式：
    python examples/agent_demo.py
"""

import sys
import os
import asyncio
import logging

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from apt_model.agent import (
    Tool, ToolDefinition, ToolParameter, ToolType,
    ToolRegistry, ToolExecutor,
    PythonCodeExecutorTool, PythonSandbox,
    register_tool, get_tool_registry, tool
)

from apt_model.agent.tools.web_search import WebSearchTool, create_web_search_tool
from apt_model.agent.agent_loop import ReactAgent, create_react_agent, AgentConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== 示例 1: 基础工具使用 ====================

async def demo_basic_tools():
    """示例 1: 基础工具注册和调用"""
    print("\n" + "="*70)
    print("示例 1: 基础工具注册和调用")
    print("="*70 + "\n")

    # 使用装饰器注册工具
    @tool(
        name="calculator",
        description="Perform basic arithmetic calculations",
        tool_type=ToolType.COMPUTATION,
        parameters=[
            ToolParameter("expression", "string", "Arithmetic expression (e.g., '3 + 5 * 2')")
        ]
    )
    async def calculator(expression: str):
        """计算器工具"""
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return {"result": result, "expression": expression}
        except Exception as e:
            return {"error": str(e)}

    # 获取注册表和执行器
    registry = get_tool_registry()
    executor = ToolExecutor(registry)

    # 测试工具调用
    expressions = ["3 + 5", "10 * 2 + 3", "2 ** 10"]

    for expr in expressions:
        result = await executor.execute_single("calculator", {"expression": expr})
        print(f"计算: {expr}")
        print(f"  结果: {result.result}")
        print(f"  执行时间: {result.execution_time:.4f}s\n")


# ==================== 示例 2: Python 沙盒 ====================

async def demo_python_sandbox():
    """示例 2: Python 代码沙盒执行"""
    print("\n" + "="*70)
    print("示例 2: Python 代码沙盒执行")
    print("="*70 + "\n")

    # 注册 Python 执行工具
    python_tool = PythonCodeExecutorTool()
    register_tool(python_tool)

    executor = ToolExecutor(get_tool_registry())

    # 测试代码示例
    test_cases = [
        {
            "name": "简单计算",
            "code": "result = sum([1, 2, 3, 4, 5])\nprint(f'Sum: {result}')\n__result__ = result"
        },
        {
            "name": "列表推导",
            "code": "squares = [x**2 for x in range(10)]\n__result__ = squares"
        },
        {
            "name": "字符串处理",
            "code": "text = 'Hello, Agent!'\n__result__ = text.upper()"
        }
    ]

    for test in test_cases:
        print(f"\n测试: {test['name']}")
        print(f"代码:\n{test['code']}\n")

        result = await executor.execute_single(
            "python_executor",
            {"code": test["code"], "mode": "script"}
        )

        if result.success:
            print(f"✓ 执行成功")
            print(f"结果: {result.result}")
        else:
            print(f"✗ 执行失败")
            print(f"错误: {result.error}")


# ==================== 示例 3: Web 搜索 ====================

async def demo_web_search():
    """示例 3: Web 搜索功能"""
    print("\n" + "="*70)
    print("示例 3: Web 搜索功能")
    print("="*70 + "\n")

    # 创建并注册 Web 搜索工具
    web_search_tool = create_web_search_tool(search_engine="mock")
    register_tool(web_search_tool)

    executor = ToolExecutor(get_tool_registry())

    # 测试搜索
    queries = [
        "transformer architecture deep learning",
        "python programming tutorial",
        "artificial intelligence 2025"
    ]

    for query in queries:
        print(f"\n搜索: {query}")

        result = await executor.execute_single(
            "web_search",
            {"query": query, "num_results": 3}
        )

        if result.success:
            search_result = result.result
            print(f"  搜索引擎: {search_result['search_engine']}")
            print(f"  结果数: {search_result['total_results']}")
            print(f"  执行时间: {search_result['execution_time']:.3f}s")
            print(f"\n  前3个结果:")
            for item in search_result['results'][:3]:
                print(f"    [{item['rank']}] {item['title']}")
                print(f"        {item['snippet'][:80]}...")
        else:
            print(f"  ✗ 搜索失败: {result.error}")


# ==================== 示例 4: ReAct Agent ====================

async def demo_react_agent():
    """示例 4: ReAct Agent 决策循环"""
    print("\n" + "="*70)
    print("示例 4: ReAct Agent 决策循环")
    print("="*70 + "\n")

    # 创建 Agent
    config = AgentConfig(
        max_steps=5,
        max_execution_time=30.0
    )

    agent = create_react_agent(
        enable_python=True,
        enable_web_search=True,
        config=config
    )

    # 测试问题
    questions = [
        "计算 (10 + 5) * 3 的结果",
        "搜索关于 transformer 架构的信息",
        "Python 是什么编程语言？"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"问题 {i}: {question}")
        print(f"{'='*60}\n")

        result = await agent.run(question)

        print(f"执行状态: {'成功' if result.success else '失败'}")
        print(f"总步骤数: {result.total_steps}\n")

        # 显示每一步
        for step in result.steps:
            print(f"  [步骤 {step.step_num}]")
            print(f"    思考: {step.thought}")
            if step.action:
                print(f"    行动: {step.action}")
                print(f"    输入: {step.action_input}")
            if step.observation:
                obs_preview = str(step.observation)[:200]
                print(f"    观察: {obs_preview}{'...' if len(str(step.observation)) > 200 else ''}")
            print()

        if result.success and result.final_answer:
            print(f"最终答案: {result.final_answer}\n")
        elif result.error:
            print(f"错误: {result.error}\n")


# ==================== 示例 5: 并行工具调用 ====================

async def demo_parallel_tools():
    """示例 5: 并行工具调用"""
    print("\n" + "="*70)
    print("示例 5: 并行工具调用")
    print("="*70 + "\n")

    executor = ToolExecutor(get_tool_registry(), max_parallel=3)

    # 并行执行多个计算
    calls = [
        {"tool_name": "calculator", "arguments": {"expression": "10 + 20"}},
        {"tool_name": "calculator", "arguments": {"expression": "5 * 6"}},
        {"tool_name": "calculator", "arguments": {"expression": "100 / 4"}},
        {"tool_name": "python_executor", "arguments": {
            "code": "__result__ = [x**2 for x in range(5)]",
            "mode": "script"
        }},
    ]

    print("并行执行 4 个工具调用...\n")

    results = await executor.execute_parallel(calls)

    for i, result in enumerate(results, 1):
        print(f"调用 {i}: {result.tool_name}")
        print(f"  状态: {'成功' if result.success else '失败'}")
        print(f"  结果: {result.result}")
        print(f"  执行时间: {result.execution_time:.4f}s\n")


# ==================== 示例 6: 工具统计 ====================

def demo_tool_stats():
    """示例 6: 工具统计信息"""
    print("\n" + "="*70)
    print("示例 6: 工具统计信息")
    print("="*70 + "\n")

    registry = get_tool_registry()
    stats = registry.get_stats()

    print(f"工具总数: {stats['total_tools']}")
    print(f"\n按类型分组:")
    for tool_type, count in stats['by_type'].items():
        if count > 0:
            print(f"  {tool_type}: {count}")

    print(f"\n按标签分组:")
    for tag, count in stats['by_tag'].items():
        print(f"  {tag}: {count}")

    print(f"\n工具详细统计:")
    for tool_name, tool_stats in stats['tool_stats'].items():
        print(f"  {tool_name}:")
        print(f"    调用次数: {tool_stats['call_count']}")
        print(f"    总时间: {tool_stats['total_time']:.4f}s")
        print(f"    平均时间: {tool_stats['avg_time']:.4f}s")


# ==================== 主函数 ====================

async def main():
    """运行所有示例"""
    print("="*70)
    print("APT Agent System - 完整示例")
    print("="*70)
    print("\n这个示例展示了 Agent 系统的所有功能:")
    print("  1. 工具注册和调用")
    print("  2. Python 代码沙盒执行")
    print("  3. Web 搜索")
    print("  4. ReAct Agent 决策循环")
    print("  5. 并行工具调用")
    print("  6. 工具统计信息")
    print("\n按 Enter 开始...")
    input()

    try:
        await demo_basic_tools()
        await demo_python_sandbox()
        await demo_web_search()
        await demo_parallel_tools()
        await demo_react_agent()
        demo_tool_stats()

    except KeyboardInterrupt:
        print("\n\n示例被用户中断")
    except Exception as e:
        logger.error(f"示例执行出错: {e}", exc_info=True)
    finally:
        print("\n" + "="*70)
        print("示例完成")
        print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
