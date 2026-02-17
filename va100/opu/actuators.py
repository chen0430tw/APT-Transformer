"""
OPU Actuators — 动作执行器接口
═══════════════════════════════

VA100 实现此接口, OPU 不碰实现细节。

GPT-5.2: "OPU 只输出动作，不直接操作内部对象。"
"""

from __future__ import annotations
from typing import List, Protocol, runtime_checkable

from opu.actions import OPUAction


@runtime_checkable
class ActionExecutor(Protocol):
    """
    VA100 引擎实现此协议。
    OPU 调用 execute(actions), 引擎翻译成:
      · vram 的 promote/evict/prefetch
      · kv_adapter 的 materialize 策略调整
      · ghost move 的队列窗口调整

    像"机场地勤": OPU 发指令, 地勤执行, 飞机(推理)不知道。
    """

    def execute_evict(self, target_free_ratio: float, **kw) -> None: ...
    def execute_prefetch(self, window: int, coalesce: bool = False, **kw) -> None: ...
    def execute_tighten(self, hot_ratio: float, warm_ratio: float, **kw) -> None: ...
    def execute_relax(self, hot_ratio: float, **kw) -> None: ...
    def execute_gate_compute(self, gate_level: int, **kw) -> None: ...
    def execute_quality_escalation(self, **kw) -> None: ...


def dispatch_actions(executor: ActionExecutor,
                     actions: List[OPUAction]) -> None:
    """
    通用动作分发器。
    按 priority 降序执行, 把 OPUAction 翻译成 executor 方法调用。
    """
    for a in sorted(actions, key=lambda x: -x.priority):
        p = a.payload
        if a.type == 'evict':
            executor.execute_evict(
                target_free_ratio=p.get('target_free_ratio', 0.1))
        elif a.type == 'prefetch':
            executor.execute_prefetch(
                window=p.get('window', 2),
                coalesce=p.get('coalesce', False))
        elif a.type == 'tighten':
            executor.execute_tighten(
                hot_ratio=p.get('hot_ratio', 0.5),
                warm_ratio=p.get('warm_ratio', 0.3))
        elif a.type == 'relax':
            executor.execute_relax(
                hot_ratio=p.get('hot_ratio', 0.6))
        elif a.type == 'gate_compute':
            executor.execute_gate_compute(
                gate_level=p.get('gate_level', 1))
        elif a.type == 'quality_escalation':
            executor.execute_quality_escalation()
        # health / noop: 只记账不执行
