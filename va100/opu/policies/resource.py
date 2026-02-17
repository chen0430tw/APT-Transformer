"""
资源闭环策略 (Loop A) — v2
═════════════════════════════

v2 修复:
  · evaluate() 返回 List[OPUAction] (不再返回 tuple, 符合 ABC 契约)
  · 状态变化通过 self._state_changed 暴露
  · Evict/Tighten/Relax 动作附带 reason + source

阈值 + 迟滞 + 冷却期 + 限速器，缺一不可。
"""

from __future__ import annotations
from typing import Any, Dict, List

from opu.actions import OPUAction, Evict, Tighten, Relax
from opu.config import OPUConfig
from opu.policies.base import PolicyBase


class ResourcePolicy(PolicyBase):
    """
    资源闭环: 基于显存压力的 tighten/relax/evict。

    状态:
      _pressure_level: 'normal' | 'high' (hysteresis)
      _tighten_count:  连续 tighten 次数 (限速器)
      _relax_count:    连续 relax 次数
    """

    def __init__(self, cfg: OPUConfig):
        super().__init__(cfg)
        self._pressure_level = 'normal'
        self._tighten_count = 0
        self._relax_count = 0
        self._effective_hot_ratio = cfg.hot_ratio
        self._effective_warm_ratio = cfg.warm_ratio

    def evaluate(self, ledger: Dict[str, Any]) -> List[OPUAction]:
        acts: List[OPUAction] = []
        self._state_changed = False

        if ledger.get('cooldown_left', 0) > 0:
            return acts

        pressure = ledger.get('hot_pressure', 0.0)
        faults = ledger.get('faults', 0.0)
        mu = ledger.get('mu', 0.0)
        tau = ledger.get('tau', 0.0)

        # ── 迟滞: 压力状态切换 ──
        old = self._pressure_level
        if pressure >= self.cfg.high_water:
            self._pressure_level = 'high'
        elif pressure <= self.cfg.low_water:
            self._pressure_level = 'normal'

        if self._pressure_level != old:
            self._state_changed = True

        # ── 高压: evict + tighten (限速器) ──
        if self._pressure_level == 'high':
            reason = (f"pressure={pressure:.2f}>{self.cfg.high_water}")
            acts.append(Evict(
                target_free_ratio=pressure - self.cfg.low_water,
                pressure=pressure,
                reason=reason,
                source=self.name,
            ))
            if self._tighten_count < self.cfg.max_tighten_streak:
                new_hot = max(0.2, self._effective_hot_ratio * 0.85)
                new_warm = max(0.1, self._effective_warm_ratio * 0.90)
                acts.append(Tighten(
                    hot_ratio=new_hot, warm_ratio=new_warm,
                    reason=f"memory;streak={self._tighten_count+1}/{self.cfg.max_tighten_streak}",
                    source=self.name,
                ))
                self._effective_hot_ratio = new_hot
                self._effective_warm_ratio = new_warm
                self._tighten_count += 1
                self._relax_count = 0

        # ── 正常 + 所有 KPI 健康: 放松 (限速) ──
        elif (self._pressure_level == 'normal'
              and faults < 0.5 and mu < 0.05 and tau < 0.05):
            if self._relax_count < self.cfg.max_relax_streak:
                new_hot = min(0.7, self._effective_hot_ratio * 1.05)
                acts.append(Relax(
                    hot_ratio=new_hot,
                    reason=f"healthy;p={pressure:.2f},f={faults:.1f},μ={mu:.3f},τ={tau:.3f}",
                    source=self.name,
                ))
                self._effective_hot_ratio = new_hot
                self._relax_count += 1
                self._tighten_count = 0

        return acts

    def reset(self):
        self._pressure_level = 'normal'
        self._tighten_count = 0
        self._relax_count = 0
        self._effective_hot_ratio = self.cfg.hot_ratio
        self._effective_warm_ratio = self.cfg.warm_ratio
        self._state_changed = False
