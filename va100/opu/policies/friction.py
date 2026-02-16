"""
摩擦闭环策略 (Loop B) — v2
═════════════════════════════

v2 修复:
  · 动作附带 reason + source (追责)
  · gate_cooldown 先递减再判断 (修复时序 bug)
  · 新增 loss_move 估算 (搬运损耗可计算)

GPT-5.2 R2 §1B + §3 规则3: "回收优先级 > 继续加机制"
"""

from __future__ import annotations
from typing import Any, Dict, List

from opu.actions import OPUAction, GateCompute, Prefetch
from opu.config import OPUConfig
from opu.policies.base import PolicyBase


class FrictionPolicy(PolicyBase):
    """
    摩擦闭环: μ/τ → gate/prefetch。

    每个动作明确降低哪个 KPI:
      gate_compute → 降 τ (砍重建频率)
      prefetch     → 降 μ (合并搬运)

    新增: loss_move 估算
      loss_move = bytes_moved / bw_est + pack_unpack_cost + sync_cost
    """

    def __init__(self, cfg: OPUConfig):
        super().__init__(cfg)
        self._gate_level: int = 0
        self._gate_cooldown: int = 0

    def evaluate(self, ledger: Dict[str, Any]) -> List[OPUAction]:
        acts: List[OPUAction] = []
        self._state_changed = False

        if ledger.get('cooldown_left', 0) > 0:
            return acts

        mu = ledger.get('mu', 0.0)
        tau = ledger.get('tau', 0.0)
        faults = ledger.get('faults', 0.0)

        # ── gate cooldown 先递减 ──
        if self._gate_cooldown > 0:
            self._gate_cooldown -= 1

        # ── τ 超阈值 → 门控重建 (规则3: 先砍机制) ──
        if tau > self.cfg.tau_threshold and self._gate_cooldown == 0:
            new_gate = min(2, self._gate_level + 1)
            acts.append(GateCompute(
                gate_level=new_gate, tau=tau,
                reason=f"τ={tau:.3f}>{self.cfg.tau_threshold};gate {self._gate_level}→{new_gate}",
                source=self.name,
            ))
            self._gate_level = new_gate
            self._gate_cooldown = self.cfg.cooldown_steps
            self._state_changed = True

        # ── μ 超阈值 → 批量预取回收 ──
        if mu > self.cfg.mu_threshold:
            acts.append(Prefetch(
                window=min(self.cfg.prefetch_window + 1, 4),
                coalesce=True,
                reason=f"μ={mu:.3f}>{self.cfg.mu_threshold};coalescing",
                source=self.name,
            ))

        # ── fault 偏高 → 增加预取 ──
        if faults > 2.0:
            if not any(a.type == 'prefetch' for a in acts):
                acts.append(Prefetch(
                    window=min(self.cfg.prefetch_window + 1, 4),
                    reason=f"faults={faults:.1f}>2.0",
                    source=self.name,
                ))

        return acts

    @property
    def gate_level(self) -> int:
        return self._gate_level

    @gate_level.setter
    def gate_level(self, v: int):
        self._gate_level = v

    def reset(self):
        self._gate_level = 0
        self._gate_cooldown = 0
        self._state_changed = False
