"""
OPU Actions — 稳定 ABI (v2)
═════════════════════════════

v2 变更:
  · OPUAction 新增 reason (原因码) + source (来源策略)
  · Evict/Tighten/Relax 支持原因码 (memory/friction/quality)
  · Evict/Prefetch 支持 tile_ids 批量操作
  · 所有构造器统一 source 参数, 用于动作追责

ABI 约定: 新增字段都有默认值, 旧引擎不传也不会 break。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OPUAction:
    """OPU 输出的原子动作。"""
    type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    # ── v2: 追责字段 ──
    reason: str = ''       # 触发原因 (e.g. "pressure=0.92>high_water=0.85")
    source: str = ''       # 来源策略 (e.g. "ResourcePolicy", "FrictionPolicy")

    def __repr__(self):
        src = f", src={self.source}" if self.source else ""
        rsn = f", reason={self.reason}" if self.reason else ""
        return f"OPUAction({self.type}, p={self.priority}{src}{rsn})"

    @property
    def trace(self) -> str:
        """追责字符串: source:type:reason"""
        return f"{self.source}:{self.type}:{self.reason}"


# ── 具名动作构造器 ──

def Evict(target_free_ratio: float = 0.1, pressure: float = 0.0,
          tile_ids: Optional[List[str]] = None,
          reason: str = '', source: str = '', **kw) -> OPUAction:
    """驱逐 hot 层 tiles → warm/cold。降 pressure, 可能升 μ。"""
    return OPUAction(
        type='evict', priority=100,
        reason=reason, source=source,
        payload={'target_free_ratio': target_free_ratio,
                 'pressure': pressure,
                 'tile_ids': tile_ids or [], **kw})


def Prefetch(window: int = 2, coalesce: bool = False,
             tile_ids: Optional[List[str]] = None,
             reason: str = '', source: str = '', **kw) -> OPUAction:
    """预取 warm/cold tiles → hot。降 faults, 可能升 μ。"""
    return OPUAction(
        type='prefetch', priority=80,
        reason=reason, source=source,
        payload={'window': window, 'coalesce': coalesce,
                 'tile_ids': tile_ids or [], **kw})


def Tighten(hot_ratio: float = 0.5, warm_ratio: float = 0.3,
            reason: str = '', source: str = '', **kw) -> OPUAction:
    """收缩 hot/warm 比例。reason: 'memory'|'friction'|'quality'"""
    return OPUAction(
        type='tighten', priority=90,
        reason=reason, source=source,
        payload={'hot_ratio': hot_ratio, 'warm_ratio': warm_ratio, **kw})


def Relax(hot_ratio: float = 0.6,
          reason: str = '', source: str = '', **kw) -> OPUAction:
    """放松 hot 比例。仅在所有 KPI 健康时。"""
    return OPUAction(
        type='relax', priority=30,
        reason=reason, source=source,
        payload={'hot_ratio': hot_ratio, **kw})


def GateCompute(gate_level: int = 1, tau: float = 0.0,
                reason: str = '', source: str = '', **kw) -> OPUAction:
    """门控重建频率。降 τ, 可能升误差。
    level 0=正常, 1=减半, 2=仅 critical。"""
    return OPUAction(
        type='gate_compute', priority=85,
        reason=reason, source=source,
        payload={'gate_level': gate_level, 'tau': tau, **kw})


def QualityEscalation(quality_score: float = 0.0,
                      suppress_evict: bool = False,
                      reason: str = '', source: str = '', **kw) -> OPUAction:
    """质量恶化时提升关键 tile 精度/常驻。最高优先级。
    suppress_evict=True 时同时抑制后续 evict 动作。"""
    return OPUAction(
        type='quality_escalation', priority=120,
        reason=reason, source=source,
        payload={'quality_score': quality_score,
                 'action': 'promote_critical',
                 'suppress_evict': suppress_evict, **kw})


def Health(**payload) -> OPUAction:
    """定期健康检查。只记账不执行。"""
    return OPUAction(type='health', priority=5,
                     source='OPU', payload=payload)


def Noop() -> OPUAction:
    return OPUAction(type='noop', priority=0)
