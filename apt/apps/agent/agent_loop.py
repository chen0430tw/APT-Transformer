#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Agent 决策循环

实现 ReAct (Reasoning + Acting) 风格的 Agent 循环：
1. Thought: Agent 思考下一步行动
2. Action: Agent 选择并执行工具
3. Observation: Agent 观察工具执行结果
4. Repeat until done

参考架构：
- ReAct: https://arxiv.org/abs/2210.03629
- LangChain Agents
- AutoGPT
- BabyAGI

集成：
- APT-Transformer 作为推理引擎
- AIM-Memory/AIM-NC 作为长期记忆
- Tool System 作为行动能力
"""

import re
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Literal
from dataclasses import dataclass, field
from enum import Enum
import logging

from .tool_system import ToolRegistry, ToolExecutor, ToolCallResult
from .python_sandbox import PythonCodeExecutorTool
from .tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)


# ==================== Agent 状态 ====================

class AgentState(Enum):
    """Agent 状态"""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentStep:
    """Agent 执行步骤"""
    step_num: int
    thought: str  # 思考内容
    action: Optional[str] = None  # 行动（工具名称）
    action_input: Optional[Dict[str, Any]] = None  # 行动输入
    observation: Optional[str] = None  # 观察结果
    state: AgentState = AgentState.THINKING


@dataclass
class AgentResult:
    """Agent 执行结果"""
    success: bool
    final_answer: Optional[str] = None
    steps: List[AgentStep] = field(default_factory=list)
    total_steps: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==================== Agent 配置 ====================

@dataclass
class AgentConfig:
    """Agent 配置"""

    # 执行限制
    max_steps: int = 10  # 最大步骤数
    max_execution_time: float = 60.0  # 最大执行时间（秒）

    # 工具选择
    enable_python: bool = True  # 是否启用 Python 执行
    enable_web_search: bool = True  # 是否启用 Web 搜索
    enable_memory: bool = True  # 是否启用记忆系统

    # 决策策略
    decision_mode: Literal["auto", "manual"] = "auto"  # 自动或手动决策
    require_confirmation: bool = False  # 是否需要确认

    # Prompt 模板
    system_prompt: Optional[str] = None
    react_prompt_template: Optional[str] = None


# ==================== Prompt 模板 ====================

DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant with access to various tools.

You can use the following tools:
{tool_descriptions}

To use a tool, you should format your response as:

Thought: [Your reasoning about what to do next]
Action: [The tool name]
Action Input: [The input to the tool as a JSON object]

After using a tool, you will receive an observation with the result.
You can then continue thinking and acting, or provide a final answer.

When you have enough information to answer the user's question, respond with:

Thought: [Your final reasoning]
Final Answer: [Your answer to the user]
"""

REACT_PROMPT_TEMPLATE = """Question: {question}

{agent_scratchpad}

Let's think step by step.
"""


# ==================== Agent ====================

class ReactAgent:
    """
    ReAct Agent

    实现 Reasoning + Acting 循环
    """

    def __init__(
        self,
        registry: ToolRegistry,
        config: Optional[AgentConfig] = None,
        llm_callable: Optional[callable] = None
    ):
        """
        Args:
            registry: 工具注册表
            config: Agent 配置
            llm_callable: LLM 调用函数 (query: str) -> str
        """
        self.registry = registry
        self.config = config or AgentConfig()
        self.executor = ToolExecutor(registry)

        # LLM 调用函数（如果未提供，使用 mock）
        self.llm_callable = llm_callable or self._mock_llm

        # 状态
        self.state = AgentState.IDLE
        self.steps: List[AgentStep] = []

    async def run(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResult:
        """
        运行 Agent

        Args:
            question: 用户问题
            context: 额外上下文

        Returns:
            AgentResult
        """
        self.state = AgentState.THINKING
        self.steps = []

        start_time = asyncio.get_event_loop().time()

        try:
            for step_num in range(1, self.config.max_steps + 1):
                # 检查超时
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > self.config.max_execution_time:
                    return AgentResult(
                        success=False,
                        steps=self.steps,
                        total_steps=len(self.steps),
                        error=f"Execution timeout after {elapsed:.1f}s"
                    )

                # 执行一步
                step_result = await self._execute_step(question, step_num, context)

                self.steps.append(step_result)

                # 检查是否完成
                if step_result.state == AgentState.DONE:
                    # 提取最终答案
                    final_answer = self._extract_final_answer(step_result.thought)

                    return AgentResult(
                        success=True,
                        final_answer=final_answer,
                        steps=self.steps,
                        total_steps=len(self.steps)
                    )

                elif step_result.state == AgentState.ERROR:
                    return AgentResult(
                        success=False,
                        steps=self.steps,
                        total_steps=len(self.steps),
                        error=step_result.observation
                    )

            # 达到最大步骤数
            return AgentResult(
                success=False,
                steps=self.steps,
                total_steps=len(self.steps),
                error=f"Reached maximum steps ({self.config.max_steps})"
            )

        except Exception as e:
            logger.error(f"[Agent] Error: {e}")
            return AgentResult(
                success=False,
                steps=self.steps,
                total_steps=len(self.steps),
                error=str(e)
            )

    async def _execute_step(
        self,
        question: str,
        step_num: int,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentStep:
        """执行单步"""

        # 构建 prompt
        agent_scratchpad = self._format_scratchpad()
        prompt = REACT_PROMPT_TEMPLATE.format(
            question=question,
            agent_scratchpad=agent_scratchpad
        )

        # 调用 LLM
        response = await self._call_llm(prompt)

        # 解析响应
        parsed = self._parse_response(response)

        step = AgentStep(step_num=step_num)

        # Thought
        step.thought = parsed.get("thought", "")

        # Final Answer (结束条件)
        if "final_answer" in parsed:
            step.state = AgentState.DONE
            return step

        # Action
        if "action" in parsed and "action_input" in parsed:
            step.action = parsed["action"]
            step.action_input = parsed["action_input"]
            step.state = AgentState.ACTING

            # 执行工具
            try:
                result = await self.executor.execute_single(
                    tool_name=step.action,
                    arguments=step.action_input,
                    context=context
                )

                if result.success:
                    step.observation = self._format_observation(result)
                    step.state = AgentState.OBSERVING
                else:
                    step.observation = f"Error: {result.error}"
                    step.state = AgentState.ERROR

            except Exception as e:
                step.observation = f"Tool execution failed: {e}"
                step.state = AgentState.ERROR

        else:
            # 没有 action，继续思考
            step.state = AgentState.THINKING

        return step

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        解析 LLM 响应

        期望格式：
        Thought: ...
        Action: tool_name
        Action Input: {"arg1": "value1", ...}

        或：
        Thought: ...
        Final Answer: ...
        """
        parsed = {}

        # 提取 Thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer)|\Z)", response, re.DOTALL)
        if thought_match:
            parsed["thought"] = thought_match.group(1).strip()

        # 提取 Final Answer
        final_answer_match = re.search(r"Final Answer:\s*(.+)", response, re.DOTALL)
        if final_answer_match:
            parsed["final_answer"] = final_answer_match.group(1).strip()
            return parsed  # 找到 Final Answer，不再解析 Action

        # 提取 Action
        action_match = re.search(r"Action:\s*(\w+)", response)
        if action_match:
            parsed["action"] = action_match.group(1).strip()

        # 提取 Action Input
        action_input_match = re.search(r"Action Input:\s*(\{.+?\})", response, re.DOTALL)
        if action_input_match:
            try:
                parsed["action_input"] = json.loads(action_input_match.group(1))
            except json.JSONDecodeError:
                logger.warning("[Agent] Failed to parse Action Input as JSON")
                parsed["action_input"] = {}

        return parsed

    def _format_scratchpad(self) -> str:
        """格式化执行历史（scratchpad）"""
        if not self.steps:
            return ""

        scratchpad_parts = []
        for step in self.steps:
            scratchpad_parts.append(f"Thought: {step.thought}")
            if step.action:
                scratchpad_parts.append(f"Action: {step.action}")
                scratchpad_parts.append(f"Action Input: {json.dumps(step.action_input)}")
            if step.observation:
                scratchpad_parts.append(f"Observation: {step.observation}")

        return "\n".join(scratchpad_parts)

    def _format_observation(self, result: ToolCallResult) -> str:
        """格式化观察结果"""
        if isinstance(result.result, dict):
            return json.dumps(result.result, indent=2, ensure_ascii=False)
        elif isinstance(result.result, str):
            return result.result
        else:
            return str(result.result)

    def _extract_final_answer(self, thought: str) -> str:
        """从 thought 中提取最终答案"""
        # 尝试提取 "Final Answer:" 后的内容
        match = re.search(r"Final Answer:\s*(.+)", thought, re.DOTALL)
        if match:
            return match.group(1).strip()
        return thought

    async def _call_llm(self, prompt: str) -> str:
        """调用 LLM"""
        if asyncio.iscoroutinefunction(self.llm_callable):
            return await self.llm_callable(prompt)
        else:
            return self.llm_callable(prompt)

    def _mock_llm(self, prompt: str) -> str:
        """Mock LLM（用于测试）"""
        # 简单的模式匹配
        if "计算" in prompt or "calculate" in prompt.lower():
            return """Thought: 我需要执行计算
Action: python_executor
Action Input: {"code": "result = 3 + 5 * 2; print(result)", "mode": "script"}"""

        elif "搜索" in prompt or "search" in prompt.lower():
            return """Thought: 我需要在网上搜索信息
Action: web_search
Action Input: {"query": "transformer architecture", "num_results": 3}"""

        else:
            return """Thought: 我已经有足够的信息来回答问题了
Final Answer: 这是一个测试回答。"""


# ==================== Agent 工厂 ====================

def create_react_agent(
    enable_python: bool = True,
    enable_web_search: bool = True,
    config: Optional[AgentConfig] = None,
    llm_callable: Optional[callable] = None
) -> ReactAgent:
    """
    创建 ReAct Agent

    Args:
        enable_python: 是否启用 Python 执行
        enable_web_search: 是否启用 Web 搜索
        config: Agent 配置
        llm_callable: LLM 调用函数

    Returns:
        ReactAgent
    """
    from .tool_system import get_tool_registry, register_tool

    registry = get_tool_registry()

    # 注册 Python 执行工具
    if enable_python:
        python_tool = PythonCodeExecutorTool()
        register_tool(python_tool)
        logger.info("[Agent] Registered Python executor tool")

    # 注册 Web 搜索工具
    if enable_web_search:
        web_search_tool = WebSearchTool(search_engine="mock")  # 使用 mock 引擎
        register_tool(web_search_tool)
        logger.info("[Agent] Registered Web search tool")

    # 创建 Agent
    agent = ReactAgent(
        registry=registry,
        config=config,
        llm_callable=llm_callable
    )

    logger.info(f"[Agent] Created ReAct agent with {len(registry.list_all())} tools")

    return agent


# ==================== 示例 ====================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=== ReAct Agent Demo ===\n")

    async def test():
        # 创建 Agent
        agent = create_react_agent(
            enable_python=True,
            enable_web_search=True
        )

        # 测试问题
        questions = [
            "计算 3 + 5 * 2 的结果",
            "搜索 transformer 架构的信息",
            "什么是人工智能？"
        ]

        for i, question in enumerate(questions, 1):
            print(f"\n{'='*60}")
            print(f"Question {i}: {question}")
            print(f"{'='*60}\n")

            result = await agent.run(question)

            print(f"Success: {result.success}")
            print(f"Total Steps: {result.total_steps}")

            print(f"\nSteps:")
            for step in result.steps:
                print(f"  Step {step.step_num}:")
                print(f"    Thought: {step.thought[:100]}...")
                if step.action:
                    print(f"    Action: {step.action}({step.action_input})")
                if step.observation:
                    print(f"    Observation: {step.observation[:100]}...")

            if result.success and result.final_answer:
                print(f"\nFinal Answer: {result.final_answer}")
            elif result.error:
                print(f"\nError: {result.error}")

    asyncio.run(test())

    print("\n=== Demo Complete ===")
