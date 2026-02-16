"""
OPU StepStats — 每步可观测信号 (v2)
════════════════════════════════════════

v2 变更 (基于 GPT-5.2 R2 任务书):
  · 新增 tile 级统计 (tiles_hot/warm/cold, 搬运计数)
  · 新增 ghost_move 事件 (次数/字节/overhead)
  · 新增 stall_reason 枚举
  · 新增 phase 标记 (prefill/decode)
  · 新增 action 追责字段
  · 所有搬运/耗时字段明确为 "当步增量", 不是累积值

切口规则 1: "切口必须可计量"
loss = wait_time + copy_bytes/bw + unpack_cost + rebuild_cost
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class StallReason(Enum):
    """每步的主瓶颈原因 (互斥, 取最大耗时项)"""
    NONE = 'none'
    KV_TRANSFER = 'kv_transfer'
    WEIGHT_TRANSFER = 'weight_transfer'
    PACK_UNPACK = 'pack_unpack'
    REBUILD = 'rebuild'
    ALLOCATOR = 'allocator'
    KERNEL_WAIT = 'kernel_wait'


@dataclass
class GhostMoveEvent:
    """单次 Ghost Move 搬运事件"""
    layer_id: int = 0
    tiles_moved: int = 0
    bytes_moved: int = 0
    reason: str = ''
    from_tier: str = ''
    to_tier: str = ''


@dataclass
class StepStats:
    """
    每一步的可观测信号。
    VA100 负责采集, OPU 负责解读。

    重要: 所有时间/字节字段必须是 **当步增量**, 不是累积值。
    """
    step: int = 0
    step_time_s: float = 0.0
    phase: str = 'decode'

    # ── 显存 ──
    hot_usage_mb: float = 0.0
    hot_pressure: float = 0.0
    warm_usage_mb: float = 0.0
    cold_usage_mb: float = 0.0
    gpu_alloc_peak_mb: float = 0.0
    gpu_reserved_peak_mb: float = 0.0

    # ── KV 分层字节 ──
    kv_bytes_hot: int = 0
    kv_bytes_warm: int = 0
    kv_bytes_cold: int = 0

    # ── 搬运 (当步增量) ──
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    bw_est_gbs: float = 0.0

    # ── Tile 级统计 ──
    tiles_hot_count: int = 0
    tiles_warm_count: int = 0
    tiles_cold_count: int = 0
    tiles_evicted: int = 0
    tiles_prefetched: int = 0
    tiles_promoted: int = 0
    tiles_demoted: int = 0

    # ── Tile 操作耗时 (当步, ms) ──
    tile_pack_ms: float = 0.0
    tile_unpack_ms: float = 0.0
    tile_gather_ms: float = 0.0

    # ── 预取命中 ──
    prefetch_hits: int = 0
    prefetch_misses: int = 0

    # ── 重建 ──
    rebuild_count: int = 0
    rebuild_time_s: float = 0.0

    # ── 异常 ──
    faults: int = 0

    # ── 切口损耗分解 (当步增量) ──
    wait_time_s: float = 0.0
    copy_bytes: int = 0
    unpack_cost_s: float = 0.0
    rebuild_cost_s: float = 0.0

    @property
    def aperture_loss_s(self) -> float:
        """切口总损耗 = 等待 + 解包 + 重建"""
        return self.wait_time_s + self.unpack_cost_s + self.rebuild_cost_s

    # ── 瓶颈原因 ──
    stall_reason: StallReason = StallReason.NONE

    # ── Ghost Move 事件 ──
    ghost_move_events: List[GhostMoveEvent] = field(default_factory=list)
    ghost_move_overhead_ms: float = 0.0
    ghost_move_hide_rate: float = 1.0

    # ── 质量信号 ──
    logits_entropy: float = 0.0
    repeat_rate: float = 0.0
    quality_score: float = 1.0

    # ── 动作追责 ──
    actions_taken: List[str] = field(default_factory=list)

    def classify_stall(self) -> StallReason:
        """根据当步耗时分解, 自动判定主瓶颈"""
        candidates = [
            (self.wait_time_s, StallReason.KV_TRANSFER),
            (self.unpack_cost_s, StallReason.PACK_UNPACK),
            (self.rebuild_cost_s, StallReason.REBUILD),
        ]
        worst_time, worst_reason = max(candidates, key=lambda x: x[0])
        if worst_time < 1e-6:
            return StallReason.NONE
        return worst_reason
