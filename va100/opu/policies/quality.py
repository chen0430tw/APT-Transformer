"""
质量闭环策略 (Loop C) — v2
═════════════════════════════

v2 修复:
  · QualityEscalation 附带 reason + source
  · 质量恶化时 suppress_evict=True (抑制后续驱逐)
  · state_changed 用于 σ 追踪

GPT-5.2 R2 §1C + §5:
  "一旦质量信号恶化 → 强制提升关键 tile 精度/常驻/降低门控强度"
"""

from __future__ import annotations
from typing import Any, Dict, List

from opu.actions import OPUAction, QualityEscalation
from opu.config import OPUConfig
from opu.policies.base import PolicyBase


class QualityPolicy(PolicyBase):
    """
    质量守门员。

    · quality_score EMA 低于阈值 → alarm + escalation + suppress_evict
    · 恢复后自动解除 alarm
    · 恶化时降低 gate_level (通过 core.py 策略交互)
    """

    def __init__(self, cfg: OPUConfig):
        super().__init__(cfg)
        self._quality_ema: float = 1.0
        self._alarm: bool = False

    def evaluate(self, ledger: Dict[str, Any]) -> List[OPUAction]:
        acts: List[OPUAction] = []
        self._state_changed = False

        # 不受 cooldown 限制: 质量守门员优先级最高
        quality = ledger.get('quality', self._quality_ema)

        if quality < self.cfg.quality_alarm_threshold:
            if not self._alarm:
                self._state_changed = True
            self._alarm = True
            acts.append(QualityEscalation(
                quality_score=quality,
                suppress_evict=True,
                reason=f"q_ema={quality:.3f}<{self.cfg.quality_alarm_threshold};promote+suppress_evict",
                source=self.name,
            ))

        elif quality > self.cfg.quality_recover_threshold and self._alarm:
            self._alarm = False
            self._state_changed = True

        return acts

    def update_ema(self, quality_score: float, alpha: float = 0.15):
        """由 OPU core 调用, 更新质量 EMA"""
        self._quality_ema = self._quality_ema * (1 - alpha) + quality_score * alpha

    @property
    def alarm(self) -> bool:
        return self._alarm

    @property
    def quality_ema(self) -> float:
        return self._quality_ema

    def reset(self):
        self._quality_ema = 1.0
        self._alarm = False
        self._state_changed = False
