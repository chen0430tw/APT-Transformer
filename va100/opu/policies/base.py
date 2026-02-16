"""
OPU Policy Base — 策略基类 (v2)
════════════════════════════════

v2 修复:
  · evaluate() 严格返回 List[OPUAction], 不允许返回 tuple
  · state_changed 通过 property 暴露, core.py 在 evaluate() 后读取
  · 新增 name 属性用于动作追责
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from opu.actions import OPUAction
    from opu.config import OPUConfig


class PolicyBase(ABC):
    """
    策略基类。

    每个 policy 只负责"一个闭环"的决策：
      ResourcePolicy  → 资源闭环 (tighten/relax)
      FrictionPolicy  → 摩擦闭环 (gate/prefetch)
      QualityPolicy   → 质量闭环 (escalation)

    接口约定:
      · evaluate(ledger) → List[OPUAction]  (严格返回列表, 不许返回 tuple)
      · state_changed: bool  (evaluate 后读取, 用于 σ 追踪)
      · name: str  (用于动作追责)
    """

    def __init__(self, cfg: 'OPUConfig'):
        self.cfg = cfg
        self._state_changed: bool = False

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def state_changed(self) -> bool:
        """上一次 evaluate() 是否导致策略内部状态变化 (用于 σ 追踪)"""
        return self._state_changed

    @abstractmethod
    def evaluate(self, ledger: Dict[str, Any]) -> List['OPUAction']:
        """
        根据 ledger (EMA 状态) 输出动作列表。

        重要: 必须返回 List[OPUAction], 不允许返回 tuple。
        如果策略状态有变化, 设置 self._state_changed = True。
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """重置策略内部状态"""
        ...
