"""
OPU Config — 治理器独立配置
═══════════════════════════
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class OPUConfig:
    """OPU 配置 (与推理引擎配置解耦)"""

    enabled: bool = True
    ema_alpha: float = 0.15

    # ── 资源闭环 ──
    high_water: float = 0.85
    low_water: float = 0.6
    cooldown_steps: int = 6
    max_tighten_streak: int = 5
    max_relax_streak: int = 3

    # ── 摩擦闭环 ──
    mu_threshold: float = 0.1     # μ > 此值 → 增加预取
    tau_threshold: float = 0.15   # τ > 此值 → 门控重建
    prefetch_window: int = 2

    # ── 质量闭环 ──
    quality_alarm_threshold: float = 0.5
    quality_recover_threshold: float = 0.7

    # ── 资源比例 (OPU 可动态调整) ──
    hot_ratio: float = 0.6
    warm_ratio: float = 0.3

    # ── 健康检查 ──
    health_interval: int = 16

    @classmethod
    def from_infer_config(cls, icfg) -> 'OPUConfig':
        """从 InferConfig 提取 OPU 相关字段"""
        return cls(
            enabled=getattr(icfg, 'opu_enabled', True),
            high_water=getattr(icfg, 'opu_high_water', 0.85),
            low_water=getattr(icfg, 'opu_low_water', 0.6),
            cooldown_steps=getattr(icfg, 'opu_cooldown', 6),
            prefetch_window=getattr(icfg, 'prefetch_window', 2),
            hot_ratio=getattr(icfg, 'hot_ratio', 0.6),
            warm_ratio=getattr(icfg, 'warm_ratio', 0.3),
        )
