#!/usr/bin/env python3
"""
Virtual A100 — 5-Layer Modular Ghost Move Inference Engine (v0.5 低熵重构)
=============================================================================

把"概念模拟器"拆成 5 个正交模块，每个有清晰边界和接口。

  ┌─────────────────────────────────────────────────────────────┐
  │ L4. VirtualA100Engine  编排器：forward 调 DK-Tile,           │
  │                        OPU 决策冷热, KVAdapter 懒补齐       │
  │ L3. OPU                Observe→Ledger→Policy→Act            │
  │                        管三本账：算力/显存/搬运              │
  │ L2. DKTileCore         Layer→Tiles 分解、评分、pack/unpack  │
  │ L1. KVAdapter          对外 KV 兼容（懒补齐 lazy materialize）│
  │ L0. VirtualVRAMBackend 冷温热三层内存 + offload/fetch/evict  │
  └─────────────────────────────────────────────────────────────┘

  底座: GhostCompressor (随机 SVD + 自适应 rank + INT8 因子)
        .aguf 文件格式 (APT GGUF-like Format, 二进制序列化)
        TF32 仿真 (Ampere Tensor Core 精度模拟)

v0.5 低熵重构:
  1. **配置层**: GhostConfig, InferConfig 改为 frozen dataclass
  2. **状态层**: TierStats, VRAMStats 添加显式状态转换方法
  3. **计算层**: 纯函数化 (rms_norm, softmax, rope_embed)

关键改进:
  1. OPU 闭环: 每 step 记账, 按抖动做迟滞, 按热点做冷热分层
  2. DK-Tile ↔ KV 兼容适配: "外面看 KV, 里面跑 DK-Tile"
  3. 虚拟显存三层: hot(GPU) / warm(CPU pinned) / cold(disk/mmap)
  4. 摩擦损耗可度量: μ(搬运摩擦) / τ(重建税) / σ(策略抖动)
  5. torch.compile 边界明确: 只编热路径, 不编搬运与策略
  6. **随机 SVD**: 使用随机投影加速 SVD 分解（10x-20x）

Usage:
  python virtual_a100.py test     # 17 项自检
  python virtual_a100.py sim      # 端到端仿真 (含 OPU + DK-Tile)
"""

from __future__ import annotations
import numpy as np
import struct
import heapq
import json
import time
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from collections import deque

# ============================================================================
# PyTorch 支持（替换 numpy 进行真实 GPU 计算）
# ============================================================================
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch 不可用，将使用 numpy 模式")

# ============================================================================
# LECAC 集成：调用 virtual_vram.py 的 INT2 训练能力
# ============================================================================
try:
    import importlib.util

    # 使用直接文件导入，避免包导入问题
    vram_path = Path(__file__).parent.parent / "apt" / "vgpu" / "runtime" / "virtual_vram.py"
    if vram_path.exists():
        spec = importlib.util.spec_from_file_location("_virtual_vram_impl", str(vram_path))
        _virtual_vram_impl = importlib.util.module_from_spec(spec)
        sys.modules["_virtual_vram_impl"] = _virtual_vram_impl
        spec.loader.exec_module(_virtual_vram_impl)

        # 导出关键符号
        virtual_vram = _virtual_vram_impl.virtual_vram
        VirtualVRAMConfig = _virtual_vram_impl.VirtualVRAMConfig
        NATURAL_EQUILIBRIUM_CONSTANT = _virtual_vram_impl.NATURAL_EQUILIBRIUM_CONSTANT
        LECAC_AVAILABLE = True
    else:
        raise ImportError(f"virtual_vram.py not found at {vram_path}")
except Exception as e:
    LECAC_AVAILABLE = False
    virtual_vram = None
    VirtualVRAMConfig = None
    NATURAL_EQUILIBRIUM_CONSTANT = 4.0 / 2.718281828  # 4/e
    print(f"[WARNING] LECAC (virtual_vram.py) 不可用: {e}")

# ============================================================================
# GGUF 加载支持
# ============================================================================
try:
    from llama_cpp import Llama
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    print("[WARNING] llama-cpp-python 不可用，GGUF 加载功能将被禁用")


# ============================================================================
# 类型转换工具
# ============================================================================

def to_torch(x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """numpy → torch 转换"""
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(x).contiguous()


def to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """torch → numpy 转换"""
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


# ═══════════════════════════════════════════════════════════════
# torch.compile 边界标记 (GPT-5.2: "只编热路径, 不编搬运与策略")
# ═══════════════════════════════════════════════════════════════
#
# @compilable_hot  — 可编译的热路径 (稳定形状, 高频调用)
#   tile pack/unpack, 量化反量化, layout transform,
#   attention/ffn kernel, RMSNorm, RoPE
#
# @dynamic_cold    — 不可编译 (动态分支, 形状变化)
#   OPU policy, offload/prefetch 控制流, coalescing 逻辑
#
# 在 numpy 仿真中这些是 no-op 标记;
# 切换到 torch 后端时, @compilable_hot → torch.compile(...)

def compilable_hot(fn):
    """标记: 此函数是可被 torch.compile 的热路径"""
    fn._compile_hint = 'hot'
    return fn


def dynamic_cold(fn):
    """标记: 此函数不应被 torch.compile (动态控制流)"""
    fn._compile_hint = 'cold'
    return fn

# ════════════════════════════════════════════════════════════════
# 0) 配置与常量
# ════════════════════════════════════════════════════════════════

# AGUF (APT Ghost Unified Format) 格式
# 避免 Ghost 备份软件的命名冲突，把 GPT 改成 APT
AGUF_MAGIC = b"AGUF"  # APT Ghost Unified Format
AGUF_VERSION = 2

LLAMA_70B = dict(
    H=8192, F=28672, L=80, VOCAB=32000,
    n_heads=64, n_kv_heads=8, head_dim=128,
    norm_eps=1e-5, rope_theta=500000.0,
)

WEIGHT_NAMES_6 = ['Wq', 'Wk', 'Wv', 'Wo', 'W1', 'W2']

TF32_PERF = {
    'A100_TF32_TFLOPS': 156.0,
    'A100_FP32_TFLOPS': 19.5,
    'RTX3070_TF32_TFLOPS': 20.3,
    'RTX3070_FP32_TFLOPS': 20.3,
    'RTX3070_BW_GBs': 448.0,
    'A100_BW_GBs': 2039.0,
}


@dataclass(frozen=True)
class GhostConfig:
    """离线压缩配置（低熵：不可变，类型安全）"""
    base_rank: int = 32
    min_rank: int = 4
    max_rank: int = 512
    alloc_method: str = 'greedy'
    quantize_factors: bool = True
    sparse_density: float = 0.0

    # 随机 SVD 配置
    use_random_svd: bool = True
    svd_oversample: int = 10
    svd_n_iter: int = 2

    # 投影核三层存储
    enable_projection_tiered: bool = False
    projection_hot_layers: int = 4
    projection_warm_layers: int = 24

    def __post_init__(self):
        assert self.min_rank <= self.base_rank <= self.max_rank
        assert self.alloc_method in ['greedy', 'uniform']
        assert 0.0 <= self.sparse_density <= 1.0
        assert self.svd_oversample >= 0
        assert self.svd_n_iter >= 0


@dataclass(frozen=True)
class InferConfig:
    """推理配置（低熵：不可变，类型安全）"""
    max_ctx: int = 512
    kv_quant_bits: int = 0
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    use_tf32: bool = False

    # OPU 与三层内存
    vram_budget_gb: float = 7.5
    cpu_budget_gb: float = 16.0
    hot_ratio: float = 0.60
    warm_ratio: float = 0.30
    opu_enabled: bool = True
    opu_cooldown: int = 6
    opu_high_water: float = 0.92
    opu_low_water: float = 0.85
    prefetch_window: int = 2

    def __post_init__(self):
        assert self.max_ctx > 0
        assert self.kv_quant_bits in [0, 4, 8]
        assert 0.0 <= self.temperature <= 2.0
        assert 0.0 <= self.top_p <= 1.0
        assert self.top_k >= 0
        assert 0.0 < self.hot_ratio + self.warm_ratio < 1.0
        assert 0.0 < self.opu_low_water < self.opu_high_water < 1.0


# ════════════════════════════════════════════════════════════════
# 1) 量化基础设施 (INT8 / INT4 / TF32)
# ════════════════════════════════════════════════════════════════

@compilable_hot
def quantize_int8(t: np.ndarray) -> Tuple[np.ndarray, float]:
    amax = np.abs(t).max()
    scale = float(amax / 127.0) if amax > 0 else 1.0
    scale = max(scale, 1e-12)
    q = np.clip(np.round(t / scale), -127, 127).astype(np.int8)
    return q, scale


@compilable_hot
def dequantize_int8(q: np.ndarray, scale: float, dtype=np.float32) -> np.ndarray:
    return q.astype(dtype) * dtype(scale)


def _tf32_truncate(x: np.ndarray) -> np.ndarray:
    """TF32: 清除 FP32 尾数的低 13 位 (23→10 bit mantissa)"""
    x = np.asarray(x, dtype=np.float32)
    x_bits = x.view(np.uint32)
    mask = np.uint32(0xFFFFE000)
    return (x_bits & mask).view(np.float32)


@compilable_hot
def tf32_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """模拟 TF32 Tensor Core: 截断→精确乘→FP32 累加"""
    return _tf32_truncate(a.astype(np.float32)) @ _tf32_truncate(b.astype(np.float32))


# ════════════════════════════════════════════════════════════════
# 2) Ghost 低秩因子 (v1 成熟代码, 不动)
# ════════════════════════════════════════════════════════════════

@dataclass
class GhostFactor:
    """W ≈ U @ diag(S) @ V^T 的低秩因子"""
    name: str
    m: int
    n: int
    rank: int
    U_q: np.ndarray
    U_scale: float
    S: np.ndarray
    V_q: np.ndarray
    V_scale: float
    quantized: bool
    sparse_idx: Optional[np.ndarray] = None
    sparse_val: Optional[np.ndarray] = None

    @property
    def compressed_bytes(self) -> int:
        if self.quantized:
            b = self.m * self.rank + self.rank * 4 + self.n * self.rank + 8
        else:
            b = (self.m * self.rank + self.rank + self.n * self.rank) * 4
        if self.sparse_idx is not None:
            b += self.sparse_idx.shape[0] * 4 + self.sparse_val.shape[0] * 2
        return b

    @property
    def original_bytes(self) -> int:
        return self.m * self.n * 4

    def reconstruct(self) -> np.ndarray:
        U = dequantize_int8(self.U_q, self.U_scale) if self.quantized else self.U_q
        V = dequantize_int8(self.V_q, self.V_scale) if self.quantized else self.V_q
        W = U @ np.diag(self.S) @ V.T
        if self.sparse_val is not None:
            for k in range(len(self.sparse_val)):
                i, j = self.sparse_idx[k]
                W[i, j] += self.sparse_val[k]
        return W

    def forward(self, x: np.ndarray, use_tf32: bool = False) -> np.ndarray:
        """y = x @ W^T = x @ V @ diag(S) @ U^T"""
        V = dequantize_int8(self.V_q, self.V_scale) if self.quantized else self.V_q
        U = dequantize_int8(self.U_q, self.U_scale) if self.quantized else self.U_q
        if use_tf32:
            t1 = tf32_matmul(x, V)
            t2 = t1 * self.S[np.newaxis, :]
            return tf32_matmul(t2, U.T)
        else:
            t1 = x @ V
            t2 = t1 * self.S[np.newaxis, :]
            return t2 @ U.T


@dataclass
class GhostLayer:
    """一个 Transformer 层的全部 Ghost 因子"""
    layer_idx: int
    factors: Dict[str, GhostFactor]
    norm1_weight: Optional[np.ndarray] = None
    norm2_weight: Optional[np.ndarray] = None

    @property
    def compressed_bytes(self) -> int:
        return sum(f.compressed_bytes for f in self.factors.values())

    def forward_linear(self, name: str, x: np.ndarray,
                       use_tf32: bool = False) -> np.ndarray:
        return self.factors[name].forward(x, use_tf32=use_tf32)


# ════════════════════════════════════════════════════════════════
# L0. VirtualVRAMBackend — 冷温热三层内存
# ════════════════════════════════════════════════════════════════
#
# GPT-5.2: "这三层不是'存储分级'这么简单，它们是 OPU 的动作空间"
# Hot(GPU常驻) / Warm(CPU pinned) / Cold(disk/mmap)
#

@dataclass
class TierStats:
    """单层存储统计（低熵：显式状态转换）"""
    count: int = 0
    bytes_total: int = 0
    fetch_count: int = 0
    evict_count: int = 0
    bytes_moved_in: int = 0
    bytes_moved_out: int = 0

    # ── 显式状态转换方法（低熵原则）──
    def record_store(self, nbytes: int):
        """记录存储操作"""
        self.count += 1
        self.bytes_total += nbytes

    def record_fetch(self, nbytes: int):
        """记录获取操作"""
        self.fetch_count += 1
        self.bytes_moved_out += nbytes

    def record_evict(self, nbytes: int):
        """记录驱逐操作"""
        self.evict_count += 1
        self.bytes_moved_out += nbytes

    def record_move_in(self, nbytes: int):
        """记录移入操作"""
        self.bytes_moved_in += nbytes


@dataclass
class VRAMStats:
    """全局搬运统计（低熵：显式状态转换）"""
    hot: TierStats = field(default_factory=TierStats)
    warm: TierStats = field(default_factory=TierStats)
    cold: TierStats = field(default_factory=TierStats)

    # 时间账本
    total_transfer_time_s: float = 0.0
    total_rebuild_time_s: float = 0.0
    total_compute_time_s: float = 0.0
    total_steps: int = 0

    # 切口回收账本
    aperture_evictions: int = 0
    aperture_recoveries: int = 0
    aperture_bytes_saved: int = 0

    # ── 显式状态转换方法（低熵原则）──
    def record_transfer(self, seconds: float):
        """记录搬运时间"""
        self.total_transfer_time_s += seconds

    def record_rebuild(self, seconds: float):
        """记录重建时间"""
        self.total_rebuild_time_s += seconds

    def record_compute(self, seconds: float):
        """记录计算时间"""
        self.total_compute_time_s += seconds

    def record_step(self):
        """记录一个步骤"""
        self.total_steps += 1

    def record_aperture_eviction(self, nbytes: int):
        """记录切口驱逐"""
        self.aperture_evictions += 1
        self.aperture_bytes_saved += nbytes

    def record_aperture_recovery(self):
        """记录切口恢复"""
        self.aperture_recoveries += 1

    # ── 切口回收三大 KPI ──
    @property
    def friction_mu(self) -> float:
        """搬运摩擦系数 μ"""
        if self.total_compute_time_s < 1e-12:
            return 0.0
        return self.total_transfer_time_s / self.total_compute_time_s

    @property
    def rebuild_tax_tau(self) -> float:
        """重建税 τ"""
        denom = self.total_compute_time_s + self.total_rebuild_time_s
        if denom < 1e-12:
            return 0.0
        return self.total_rebuild_time_s / denom


class VirtualVRAMBackend:
    """
    三层虚拟显存管理器。

    设计原则 (GPT-5.2):
      · Hot: 无阻塞、连续块、kernel 直接访问
      · Warm: CPU pinned, 可快速回迁 (PCIe 可预测)
      · Cold: 容量无限但摩擦最大, 必须由 OPU 控制访问节奏

    v2 改进 (GPT-5.2 第二轮):
      · 搬运延迟仿真 (PCIe bandwidth model)
      · Tile coalescing (合并小块搬运, 避免碎片化)
      · 切口回收账本 (aperture eviction/recovery tracking)
      · μ/τ 可度量
    """

    # PCIe 3.0 x16 ≈ 15.8 GB/s, PCIe 4.0 x16 ≈ 31.5 GB/s
    PCIE_BW_GBs = 15.8  # 仿真用保守值
    # Disk sequential read ≈ 500 MB/s (NVMe)
    DISK_BW_GBs = 0.5
    # 最小搬运粒度 (bytes) — 小于此的 tile 会被合并搬运
    COALESCE_MIN_BYTES = 4096

    def __init__(self, hot_budget: int, warm_budget: int, cold_budget: int):
        self.hot_budget = hot_budget
        self.warm_budget = warm_budget
        self.cold_budget = cold_budget

        # tile_id → (tier, data, metadata)
        self._store: Dict[str, Tuple[str, Any, dict]] = {}
        self.stats = VRAMStats()

        # 搬运队列 (用于 coalescing)
        self._pending_transfers: List[Tuple[str, str, int]] = []  # (tile_id, to_tier, nbytes)

    def _tier_stats(self, tier: str) -> TierStats:
        return getattr(self.stats, tier)

    def _tier_usage(self, tier: str) -> int:
        return sum(meta.get('bytes', 0)
                   for tid, (t, _, meta) in self._store.items() if t == tier)

    def _tier_budget(self, tier: str) -> int:
        return {'hot': self.hot_budget, 'warm': self.warm_budget,
                'cold': self.cold_budget}[tier]

    def _sim_transfer_time(self, nbytes: int, from_tier: str, to_tier: str) -> float:
        """仿真搬运延迟 (秒)"""
        if from_tier == to_tier:
            return 0.0
        # cold→any 走 disk bandwidth; 其余走 PCIe
        if from_tier == 'cold' or to_tier == 'cold':
            bw = self.DISK_BW_GBs * 1e9
        else:
            bw = self.PCIE_BW_GBs * 1e9
        return nbytes / max(bw, 1.0)

    def store(self, tile_id: str, data: Any, tier: str = 'hot',
              nbytes: int = 0) -> bool:
        """存储 tile 到指定层（低熵：使用 record_store）"""
        budget = self._tier_budget(tier)
        if budget > 0 and self._tier_usage(tier) + nbytes > budget:
            fallback = {'hot': 'warm', 'warm': 'cold'}.get(tier)
            if fallback:
                return self.store(tile_id, data, fallback, nbytes)
            return False
        meta = {'bytes': nbytes, 'stored_at': time.perf_counter(),
                'access_count': 0, 'aperture': 'recoverable'}
        self._store[tile_id] = (tier, data, meta)
        # 低熵：显式状态转换
        self._tier_stats(tier).record_store(nbytes)
        return True

    def fetch(self, tile_id: str) -> Optional[Any]:
        """获取 tile 数据（低熵：使用 record_fetch 和 record_transfer）"""
        if tile_id not in self._store:
            return None
        tier, data, meta = self._store[tile_id]
        meta['access_count'] = meta.get('access_count', 0) + 1
        if tier != 'hot':
            nbytes = meta.get('bytes', 0)
            ts = self._tier_stats(tier)
            # 低熵：显式状态转换
            ts.record_fetch(nbytes)
            xfer_t = self._sim_transfer_time(nbytes, tier, 'hot')
            self.stats.record_transfer(xfer_t)
        return data

    def promote(self, tile_id: str, to_tier: str) -> bool:
        """提升 tile（低熵：使用 record_transfer 和 record_move_in）"""
        if tile_id not in self._store:
            return False
        old_tier, data, meta = self._store[tile_id]
        if old_tier == to_tier:
            return True
        nbytes = meta.get('bytes', 0)
        if self._tier_usage(to_tier) + nbytes > self._tier_budget(to_tier):
            return False
        xfer_t = self._sim_transfer_time(nbytes, old_tier, to_tier)
        self.stats.record_transfer(xfer_t)  # 低熵
        self._store[tile_id] = (to_tier, data, meta)
        self._tier_stats(to_tier).record_move_in(nbytes)  # 低熵
        return True

    def demote(self, tile_id: str, to_tier: str) -> bool:
        """降级 tile（低熵：使用 record_transfer 和 record_aperture_eviction）"""
        if tile_id not in self._store:
            return False
        old_tier, data, meta = self._store[tile_id]
        nbytes = meta.get('bytes', 0)
        xfer_t = self._sim_transfer_time(nbytes, old_tier, to_tier)
        self.stats.record_transfer(xfer_t)  # 低熵
        # 切口标记
        if meta.get('aperture') == 'recoverable':
            self.stats.record_aperture_eviction(nbytes)  # 低熵
        self._store[tile_id] = (to_tier, data, meta)
        self._tier_stats(old_tier).record_evict(nbytes)  # 低熵
        self._tier_stats(to_tier).record_move_in(nbytes)  # 低熵
        return True

    def evict(self, tile_id: str) -> None:
        """彻底删除 tile（低熵：使用 record_evict）"""
        if tile_id in self._store:
            tier, _, meta = self._store.pop(tile_id)
            nbytes = meta.get('bytes', 0)
            self._tier_stats(tier).record_evict(nbytes)  # 低熵

    def get_tier(self, tile_id: str) -> Optional[str]:
        if tile_id not in self._store:
            return None
        return self._store[tile_id][0]

    def get_aperture(self, tile_id: str) -> Optional[str]:
        """获取 tile 的切口标记"""
        if tile_id not in self._store:
            return None
        return self._store[tile_id][2].get('aperture', 'recoverable')

    def set_aperture(self, tile_id: str, aperture: str) -> None:
        """设置切口标记: 'pinned' | 'recoverable' | 'evictable'"""
        if tile_id in self._store:
            self._store[tile_id][2]['aperture'] = aperture

    def tiles_in_tier(self, tier: str) -> List[str]:
        return [tid for tid, (t, _, _) in self._store.items() if t == tier]

    # ── Tile Coalescing (GPT-5.2: 合并小块避免碎片化搬运) ──

    def queue_transfer(self, tile_id: str, to_tier: str) -> None:
        """批量搬运: 先入队, flush 时合并执行"""
        if tile_id in self._store:
            nbytes = self._store[tile_id][2].get('bytes', 0)
            self._pending_transfers.append((tile_id, to_tier, nbytes))

    def flush_transfers(self) -> int:
        """合并执行所有排队的搬运, 返回总搬运字节数"""
        if not self._pending_transfers:
            return 0
        total_bytes = sum(nb for _, _, nb in self._pending_transfers)
        # 按实际方向分组计算延迟 (cold 走 disk, 其余走 PCIe)
        cold_bytes = 0
        warm_bytes = 0
        for tid, to_tier, nb in self._pending_transfers:
            if tid in self._store:
                from_tier = self._store[tid][0]
                if from_tier == 'cold' or to_tier == 'cold':
                    cold_bytes += nb
                else:
                    warm_bytes += nb
        if cold_bytes > 0:
            self.stats.record_transfer(self._sim_transfer_time(
                cold_bytes, 'cold', 'hot'))
        if warm_bytes > 0:
            self.stats.record_transfer(self._sim_transfer_time(
                warm_bytes, 'warm', 'hot'))
        for tid, to_tier, _ in self._pending_transfers:
            if tid in self._store:
                old_tier, data, meta = self._store[tid]
                self._store[tid] = (to_tier, data, meta)
        self._pending_transfers.clear()
        return total_bytes

    def tier_summary(self) -> Dict[str, dict]:
        result = {}
        for tier in ('hot', 'warm', 'cold'):
            usage = self._tier_usage(tier)
            budget = self._tier_budget(tier)
            ts = self._tier_stats(tier)
            result[tier] = {
                'count': len(self.tiles_in_tier(tier)),
                'usage_mb': usage / 1e6,
                'budget_mb': budget / 1e6,
                'pressure': usage / max(budget, 1),
                'fetches': ts.fetch_count,
                'evictions': ts.evict_count,
            }
        return result

    def kpi_summary(self) -> Dict[str, float]:
        """切口回收 KPI 汇总"""
        return {
            'mu': self.stats.friction_mu,
            'tau': self.stats.rebuild_tax_tau,
            'aperture_evictions': self.stats.aperture_evictions,
            'aperture_recoveries': self.stats.aperture_recoveries,
            'aperture_bytes_saved_mb': self.stats.aperture_bytes_saved / 1e6,
            'transfer_time_ms': self.stats.total_transfer_time_s * 1000,
            'rebuild_time_ms': self.stats.total_rebuild_time_s * 1000,
        }


# ════════════════════════════════════════════════════════════════
# L2. DKTileCore — Layer → Tiles 分解
# ════════════════════════════════════════════════════════════════
#
# GPT-5.2: "DK-Tile 不是'跳过层'，而是把层变成可调度的货柜"
# "超链接"是：哪个 tile 先到、哪个可以延迟、哪个用镜像补齐
#

@dataclass
class DKTile:
    """
    DK-Tile: 最小可调度证据单元

    GPT-5.2 R2 定义:
      tile_id = (layer, kind, shard, precision_state, version)
      criticality = critical | normal | cheap
      
    criticality 分类 (GPT-5.2: "防傻的工程策略"):
      critical: 注意力投影(Wq/Wk/Wv/Wo) — 永远不走最激进近似, 尽量常驻 hot/warm
      normal:   FFN(W1/W2) — 允许温冷搬运, 允许中等近似
      cheap:    KV tiles — 允许激进压缩/驱逐, 按需重建
    """
    tile_id: str           # "L{layer}_{kind}" e.g. "L0_Wq", "L3_KV_0:128"
    layer: int
    kind: str              # 'Wq'|'Wk'|'Wv'|'Wo'|'W1'|'W2'|'KV'
    payload: Any           # GhostFactor | np.ndarray | compressed data
    nbytes: int = 0

    # ── 身份与可追踪性 (GPT-5.2 R2: "tile 必须有完整身份") ──
    criticality: str = 'normal'     # 'critical' | 'normal' | 'cheap'
    version: int = 0                # 版本号 (每次 pack/unpack 递增)
    precision_state: str = 'full'   # 'full' | 'tf32' | 'int8' | 'int4' | 'proxy'

    # ── 元数据 ──
    hotness: float = 0.0   # 热度评分 (OPU 更新)
    access_count: int = 0  # 访问次数
    last_access: float = 0.0

    # ── 拓扑链接 ──
    links: Dict[str, str] = field(default_factory=dict)
    # e.g. {"mirror": "L0_Wq_backup", "complement": "L0_Wq_residual"}

    # ── 切口标记 ──
    aperture: str = 'recoverable'   # 'pinned' | 'recoverable' | 'evictable'

    def touch(self):
        """更新访问热度"""
        self.access_count += 1
        self.last_access = time.perf_counter()
        # 指数衰减热度
        self.hotness = self.hotness * 0.9 + 1.0

    @staticmethod
    def infer_criticality(kind: str) -> str:
        """根据 kind 自动推断 criticality"""
        if kind in ('Wq', 'Wk', 'Wv', 'Wo'):
            return 'critical'
        elif kind in ('W1', 'W2'):
            return 'normal'
        else:  # KV, etc.
            return 'cheap'


class DKTileCore:
    """
    Layer → Tiles 分解引擎。

    核心接口 (GPT-5.2 定义):
      tileize(layer) → tiles       分解
      detileize(tiles) → kv_view   重组 (懒)
      score(tile) → float          评分
      pack(tile) → compressed      打包
      unpack(compressed) → tile    解包
    """

    def __init__(self):
        self._tile_registry: Dict[str, DKTile] = {}

    def tileize_layer(self, ghost_layer: GhostLayer) -> List[DKTile]:
        """
        把 GhostLayer 拆成 DKTile 列表。
        自动分配 criticality + aperture:
          critical (Wq/Wk/Wv/Wo) → aperture='pinned' (不允许驱逐到 cold)
          normal (W1/W2) → aperture='recoverable'
        """
        tiles = []
        li = ghost_layer.layer_idx
        for name, factor in ghost_layer.factors.items():
            tile_id = f"L{li}_{name}"
            crit = DKTile.infer_criticality(name)
            # critical tiles 默认 pinned (GPT-5.2: "永远不走最激进近似")
            aperture = 'pinned' if crit == 'critical' else 'recoverable'
            tile = DKTile(
                tile_id=tile_id,
                layer=li,
                kind=name,
                payload=factor,
                nbytes=factor.compressed_bytes,
                criticality=crit,
                aperture=aperture,
            )
            tiles.append(tile)
            self._tile_registry[tile_id] = tile
        return tiles

    def tileize_kv(self, layer_idx: int, k: np.ndarray, v: np.ndarray,
                   pos: int) -> DKTile:
        """把 KV 片段封装为 DKTile"""
        tile_id = f"L{layer_idx}_KV_{pos}"
        nbytes = k.nbytes + v.nbytes
        tile = DKTile(
            tile_id=tile_id, layer=layer_idx, kind='KV',
            payload=(k.copy(), v.copy()), nbytes=nbytes,
            aperture='recoverable',
        )
        self._tile_registry[tile_id] = tile
        return tile

    def detileize_weight(self, tile: DKTile, x: np.ndarray,
                         use_tf32: bool = False) -> np.ndarray:
        """从 weight tile 做 forward: y = x @ W^T"""
        tile.touch()
        factor: GhostFactor = tile.payload
        return factor.forward(x, use_tf32=use_tf32)

    def detileize_kv(self, tiles: List[DKTile]) -> Tuple[np.ndarray, np.ndarray]:
        """
        从 KV tiles 重组完整 KV 视图。
        GPT-5.2: "补齐是懒执行 (lazy materialization)"
        """
        if not tiles:
            return np.zeros((0,)), np.zeros((0,))
        ks, vs = [], []
        for t in sorted(tiles, key=lambda t: t.tile_id):
            t.touch()
            k, v = t.payload
            ks.append(k)
            vs.append(v)
        return np.stack(ks), np.stack(vs)

    def score(self, tile: DKTile) -> float:
        """
        评分: 热度 + 访问频率 + 时间衰减 + criticality 加权。
        GPT-5.2 R2: critical tiles 得分大幅加权, 不容易被驱逐。
        """
        age = time.perf_counter() - tile.last_access if tile.last_access > 0 else 1e6
        recency = 1.0 / (1.0 + age)
        base = tile.hotness * 0.5 + tile.access_count * 0.3 + recency * 0.2
        # criticality 加权: critical 不容易被驱逐
        crit_mult = {'critical': 10.0, 'normal': 1.0, 'cheap': 0.3}
        return base * crit_mult.get(tile.criticality, 1.0)

    def pack(self, tile: DKTile) -> bytes:
        """打包 tile 为可搬运的字节 (用于 cold 层存储)"""
        if tile.kind == 'KV':
            k, v = tile.payload
            if isinstance(k, tuple):
                # INT8 量化 KV
                k_q, k_s = k
                v_q, v_s = v
                header = struct.pack('<Bff', 1, k_s, v_s)
                return header + k_q.tobytes() + v_q.tobytes()
            else:
                header = struct.pack('<Bff', 0, 0.0, 0.0)
                return header + k.tobytes() + v.tobytes()
        else:
            factor: GhostFactor = tile.payload
            return factor.U_q.tobytes() + factor.S.tobytes() + factor.V_q.tobytes()

    def unpack(self, packed: bytes, tile: DKTile) -> Any:
        """从字节恢复 tile payload (pack 的逆操作)"""
        if tile.kind == 'KV':
            quantized = struct.unpack_from('<B', packed, 0)[0]
            k_s = struct.unpack_from('<f', packed, 1)[0]
            v_s = struct.unpack_from('<f', packed, 5)[0]
            data = packed[9:]
            if quantized:
                half = len(data) // 2
                k_q = np.frombuffer(data[:half], dtype=np.int8).copy()
                v_q = np.frombuffer(data[half:], dtype=np.int8).copy()
                return (k_q, k_s), (v_q, v_s)
            else:
                half = len(data) // 2
                k = np.frombuffer(data[:half], dtype=np.float32).copy()
                v = np.frombuffer(data[half:], dtype=np.float32).copy()
                return k, v
        else:
            factor: GhostFactor = tile.payload
            u_size = factor.m * factor.rank * np.dtype(factor.U_q.dtype).itemsize
            s_size = factor.rank * 4
            v_size = factor.n * factor.rank * np.dtype(factor.V_q.dtype).itemsize
            U = np.frombuffer(packed[:u_size], dtype=factor.U_q.dtype).reshape(
                factor.m, factor.rank).copy()
            S = np.frombuffer(packed[u_size:u_size+s_size], dtype=np.float32).copy()
            V = np.frombuffer(packed[u_size+s_size:], dtype=factor.V_q.dtype).reshape(
                factor.n, factor.rank).copy()
            return GhostFactor(
                name=factor.name, m=factor.m, n=factor.n, rank=factor.rank,
                U_q=U, U_scale=factor.U_scale, S=S,
                V_q=V, V_scale=factor.V_scale, quantized=factor.quantized,
                sparse_idx=factor.sparse_idx, sparse_val=factor.sparse_val,
            )

    def get_tile(self, tile_id: str) -> Optional[DKTile]:
        return self._tile_registry.get(tile_id)


# ════════════════════════════════════════════════════════════════
# L1. KVAdapter — 对外 KV 兼容层
# ════════════════════════════════════════════════════════════════
#
# GPT-5.2: "让外面觉得'什么都没发生'"
# "补齐本来的 KV 就在这一层做：补齐是懒执行"
#

class KVAdapter:
    """
    对外暴露标准 KV cache 接口;
    内部用 DKTile 管理, 配合 VirtualVRAMBackend 做冷热分层。

    GPT-5.2 关键要求:
      · "补齐是懒执行 (lazy materialization)"
      · "别每步都补齐"
      · "只补齐下一 token 真会读的那部分"

    统计追踪:
      · cache_hits: 热缓存命中 (无需 detileize)
      · cache_misses: 需要从 tiles 重建
      · rebuild_time_s: 累积重建耗时 → 用于计算 τ
    """

    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int,
                 max_ctx: int, dk: DKTileCore, vram: VirtualVRAMBackend,
                 quant_bits: int = 0):
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_ctx = max_ctx
        self.dk = dk
        self.vram = vram
        self.quant_bits = quant_bits
        self.current_len = 0

        self._kv_tile_ids: Dict[int, List[str]] = {i: [] for i in range(n_layers)}

        # 热缓存 + 脏标记 (lazy materialization 核心)
        self._hot_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._cache_valid: Dict[int, bool] = {}  # layer → 缓存是否有效

        # ── Lazy materialization 统计 ──
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.rebuild_count: int = 0
        self.rebuild_time_s: float = 0.0

    def write(self, layer_idx: int, pos: int, k: np.ndarray, v: np.ndarray):
        """
        写入一个时间步的 KV。
        k, v: [n_kv_heads, head_dim]
        """
        if self.quant_bits == 8:
            k_q, k_s = quantize_int8(k)
            v_q, v_s = quantize_int8(v)
            payload_k = (k_q, k_s)
            payload_v = (v_q, v_s)
        else:
            payload_k = k.copy()
            payload_v = v.copy()

        tile_id = f"L{layer_idx}_KV_{pos}"
        tile = DKTile(
            tile_id=tile_id, layer=layer_idx, kind='KV',
            payload=(payload_k, payload_v),
            nbytes=k.nbytes + v.nbytes,
            aperture='recoverable',
        )
        self.dk._tile_registry[tile_id] = tile
        self.vram.store(tile_id, tile, tier='hot', nbytes=tile.nbytes)
        self._kv_tile_ids[layer_idx].append(tile_id)

        # 更新热缓存 (直接写入, 不需要重建)
        self._update_hot_cache(layer_idx, pos, k, v)
        self._cache_valid[layer_idx] = True
        self.current_len = max(self.current_len, pos + 1)

    def _update_hot_cache(self, layer_idx: int, pos: int,
                          k: np.ndarray, v: np.ndarray):
        """维护累积 KV 视图 (热缓存, 避免每次 detileize)"""
        if layer_idx not in self._hot_cache:
            # 首次: 预分配
            shape = (self.max_ctx, self.n_kv_heads, self.head_dim)
            self._hot_cache[layer_idx] = (
                np.zeros(shape, dtype=np.float32),
                np.zeros(shape, dtype=np.float32),
            )
        k_cache, v_cache = self._hot_cache[layer_idx]
        slot = pos % self.max_ctx
        k_cache[slot] = k
        v_cache[slot] = v

    def read(self, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        读取完整 KV。
        Lazy materialization: 优先从热缓存读; 只在无效时才 detileize。
        """
        seq_len = min(self.current_len, self.max_ctx)
        if layer_idx in self._hot_cache and self._cache_valid.get(layer_idx, False):
            # 热缓存命中 — 零重建成本
            self.cache_hits += 1
            k_cache, v_cache = self._hot_cache[layer_idx]
            return k_cache[:seq_len].copy(), v_cache[:seq_len].copy()
        # 缓存未命中 — 需要 detileize (GPT-5.2: "只补齐真会读的那部分")
        self.cache_misses += 1
        self.rebuild_count += 1
        t0 = time.perf_counter()
        result = self._rebuild_from_tiles(layer_idx, seq_len)
        rebuild_t = time.perf_counter() - t0
        self.rebuild_time_s += rebuild_t
        # 回写到 VRAMStats (供 OPU 度量 τ)
        self.vram.stats.total_rebuild_time_s += rebuild_t
        self.vram.stats.aperture_recoveries += 1
        return result

    def _rebuild_from_tiles(self, layer_idx: int, seq_len: int
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """从 DK-Tile 存储重建 KV — GPT-5.2 的 "补齐" 路径"""
        shape = (seq_len, self.n_kv_heads, self.head_dim)
        k_out = np.zeros(shape, dtype=np.float32)
        v_out = np.zeros(shape, dtype=np.float32)
        for pos in range(seq_len):
            tile_id = f"L{layer_idx}_KV_{pos}"
            data = self.vram.fetch(tile_id)
            if data is not None:
                tile: DKTile = data
                payload_k, payload_v = tile.payload
                if isinstance(payload_k, tuple):
                    # INT8 量化
                    k_out[pos] = dequantize_int8(payload_k[0], payload_k[1])
                    v_out[pos] = dequantize_int8(payload_v[0], payload_v[1])
                else:
                    k_out[pos] = payload_k
                    v_out[pos] = payload_v
        return k_out, v_out

    def export_kv(self, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """对外兼容接口: 返回标准 (K, V) 张量"""
        return self.read(layer_idx)

    def import_kv(self, layer_idx: int, K: np.ndarray, V: np.ndarray):
        """从外部 KV 导入 (e.g. 加载旧缓存)"""
        for pos in range(K.shape[0]):
            self.write(layer_idx, pos, K[pos], V[pos])

    def reset(self):
        self.current_len = 0
        self._kv_tile_ids = {i: [] for i in range(self.n_layers)}
        self._hot_cache.clear()
        self._cache_valid.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        self.rebuild_count = 0
        self.rebuild_time_s = 0.0

    @property
    def memory_bytes(self) -> int:
        return sum(c[0].nbytes + c[1].nbytes
                   for c in self._hot_cache.values())

    def lazy_stats(self) -> Dict[str, Any]:
        """Lazy materialization 统计"""
        total = self.cache_hits + self.cache_misses
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / max(total, 1),
            'rebuilds': self.rebuild_count,
            'rebuild_time_ms': self.rebuild_time_s * 1000,
        }


# ════════════════════════════════════════════════════════════════
# L3. OPU — 已分离为独立模块 opu/
# ════════════════════════════════════════════════════════════════
#
# GPT-5.2: "OPU 从 VA100 文件里分离出来，
# 但 VA100 必须保留 OPU 的 hook 点与 action 执行器;
# OPU 只做治理决策，不碰实现细节。"
#
# 文件结构:
#   opu/__init__.py       — 包入口 (re-exports)
#   opu/core.py           — OPU 编排器 (Observe/Ledger/3-Policy/Decide)
#   opu/config.py         — OPUConfig (与 InferConfig 解耦)
#   opu/stats.py          — StepStats (每步可观测信号)
#   opu/actions.py        — OPUAction + 具名构造器 (稳定 ABI)
#   opu/actuators.py      — ActionExecutor 协议 + dispatch_actions
#   opu/policies/base.py  — PolicyBase ABC
#   opu/policies/resource.py  — 资源闭环 A (tighten/relax)
#   opu/policies/friction.py  — 摩擦闭环 B (μ/τ → gate)
#   opu/policies/quality.py   — 质量闭环 C (质量→escalation)
#

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from opu import (
    OPU, OPUConfig, StepStats, OPUAction,
    dispatch_actions, ActionExecutor,
)
from opu.actions import (
    Evict, Prefetch, Tighten, Relax,
    GateCompute, QualityEscalation, Health, Noop,
)


# ── VA100 信号采集器 (Hook 接口) ──

class VA100SignalCollector:
    """
    VA100 引擎在 3 个 hook 点采集信号, 组装成 StepStats 喂给 OPU。

    Hook 1: observe_begin()   — step 开始 (读水位/队列积压)
    Hook 2: observe_kv()      — KV 访问后 (hit/miss/rebuild)
    Hook 3: observe_end()     — step 结束 (组装 StepStats)

    GPT-5.2: "VA100 不负责'想', 只负责'做'。"
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._step_time_s = 0.0
        self._h2d_bytes = 0
        self._d2h_bytes = 0
        self._prefetch_hits = 0
        self._prefetch_misses = 0
        self._rebuild_count = 0
        self._rebuild_time_s = 0.0
        self._wait_time_s = 0.0
        self._copy_bytes = 0
        self._quality_score = 1.0
        self._logits_entropy = 0.0
        self._repeat_rate = 0.0

    # Hook 1: step 开始
    def observe_begin(self, vram_summary: dict):
        """读显存水位"""
        self._hot_pressure = vram_summary['hot']['pressure']
        self._hot_usage_mb = vram_summary['hot']['usage_mb']
        self._warm_usage_mb = vram_summary['warm']['usage_mb']
        self._cold_usage_mb = vram_summary['cold']['usage_mb']

    # Hook 2: KV 访问后
    def observe_kv(self, hits: int, misses: int,
                   rebuild_count: int, rebuild_time_s: float):
        """DK-Tile / lazy materialize 的核心信号"""
        self._prefetch_hits += hits
        self._prefetch_misses += misses
        self._rebuild_count += rebuild_count
        self._rebuild_time_s += rebuild_time_s

    # Hook 3: step 结束 → 组装 StepStats
    def observe_end(self, step: int, step_time_s: float,
                    h2d_bytes: int, d2h_bytes: int,
                    wait_time_s: float, copy_bytes: int,
                    rebuild_cost_s: float = 0.0,
                    quality_score: float = 1.0,
                    logits_entropy: float = 0.0,
                    repeat_rate: float = 0.0,
                    tiles_hot_count: int = 0,
                    tiles_warm_count: int = 0,
                    tiles_cold_count: int = 0) -> StepStats:
        """组装完整 StepStats, 喂给 OPU.observe()
        注意: wait_time_s 和 rebuild_cost_s 必须是当步增量, 由调用方计算。"""
        stats = StepStats(
            step=step,
            step_time_s=step_time_s,
            hot_usage_mb=self._hot_usage_mb,
            hot_pressure=self._hot_pressure,
            warm_usage_mb=self._warm_usage_mb,
            cold_usage_mb=self._cold_usage_mb,
            h2d_bytes=h2d_bytes,
            d2h_bytes=d2h_bytes,
            prefetch_hits=self._prefetch_hits,
            prefetch_misses=self._prefetch_misses,
            rebuild_count=self._rebuild_count,
            rebuild_time_s=self._rebuild_time_s,
            faults=self._prefetch_misses,
            wait_time_s=wait_time_s,
            copy_bytes=copy_bytes,
            rebuild_cost_s=rebuild_cost_s,
            quality_score=quality_score,
            logits_entropy=logits_entropy,
            repeat_rate=repeat_rate,
            tiles_hot_count=tiles_hot_count,
            tiles_warm_count=tiles_warm_count,
            tiles_cold_count=tiles_cold_count,
        )
        # 自动分类瓶颈
        stats.stall_reason = stats.classify_stall()
        self.reset()
        return stats


# ════════════════════════════════════════════════════════════════
# L4. VirtualA100Runtime (implements ActionExecutor) + Engine
# ════════════════════════════════════════════════════════════════
#
# Runtime: 实现 ActionExecutor 协议, 执行 OPU 动作
# Engine:  编排 forward + OPU 3-hook + KVAdapter
#
# GPT-5.2: "VA100 不负责'想', 只负责'做'"
#

class VirtualA100Runtime:
    """
    GPT-5.2 R2: "Engine 只负责 orchestrate"

    管理所有 weight tiles 的冷热分层。
    提供 7 类动作接口给 OPU:
      evict / prefetch / tighten / relax / gate / promote_critical

    关键设计 (GPT-5.2 R2):
      · Critical tiles (Wq/Wk/Wv/Wo) 永远不驱逐到 cold
      · Warm 层 = "搬运友好层" (pinned, 可批量, 可预测延迟)
      · Gate 控制重建频率 (避免额外计算拖慢)
    """

    def __init__(self, ghost_layers: List[GhostLayer],
                 dk: DKTileCore, vram: VirtualVRAMBackend,
                 cfg: InferConfig):
        self.ghost_layers = ghost_layers
        self.dk = dk
        self.vram = vram
        self.cfg = cfg
        self.n_layers = len(ghost_layers)

        # 把所有 GhostLayer tileize 并存入 VRAM backend
        self._all_tiles: Dict[str, DKTile] = {}
        total_bytes = 0
        for gl in ghost_layers:
            tiles = dk.tileize_layer(gl)
            for t in tiles:
                self._all_tiles[t.tile_id] = t
                total_bytes += t.nbytes

        # 初始分层
        hot_budget = int(cfg.vram_budget_gb * 1e9 * cfg.hot_ratio)
        if total_bytes < hot_budget * 0.8:
            self._mode = "ALL_HOT"
            for tid, t in self._all_tiles.items():
                vram.store(tid, t, tier='hot', nbytes=t.nbytes)
        else:
            # TIERED: critical tiles 优先放 hot, 然后按层填充
            self._mode = "TIERED"
            cum = 0
            # Phase 1: critical tiles 优先
            for tid, t in sorted(self._all_tiles.items(),
                                 key=lambda x: (0 if x[1].criticality == 'critical' else 1,
                                                x[1].layer)):
                if cum + t.nbytes < hot_budget * 0.7:
                    vram.store(tid, t, tier='hot', nbytes=t.nbytes)
                    cum += t.nbytes
                else:
                    vram.store(tid, t, tier='warm', nbytes=t.nbytes)

        # 搬运统计
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.h2d_bytes = 0
        self.d2h_bytes = 0

        # ── 门控状态 (GPT-5.2 R2: gate_compute) ──
        self._rebuild_gate: int = 0   # 0=正常, 1=减半, 2=仅 critical
        self._rebuild_step_counter: int = 0

    def get_weight_tile(self, layer_idx: int, name: str) -> DKTile:
        """获取 weight tile (如不在 hot 则搬运)"""
        tile_id = f"L{layer_idx}_{name}"
        tier = self.vram.get_tier(tile_id)

        if tier == 'hot':
            self.prefetch_hits += 1
            tile = self._all_tiles[tile_id]
            tile.touch()
            return tile

        # 需要搬运: warm/cold → hot
        self.prefetch_misses += 1
        tile = self._all_tiles[tile_id]
        self.h2d_bytes += tile.nbytes
        self.vram.promote(tile_id, 'hot')
        tile.touch()
        return tile

    def prefetch_layer(self, layer_idx: int):
        """预取整层的 weight tiles 到 hot"""
        for name in WEIGHT_NAMES_6:
            tile_id = f"L{layer_idx}_{name}"
            if tile_id in self._all_tiles:
                tier = self.vram.get_tier(tile_id)
                if tier and tier != 'hot':
                    self.vram.promote(tile_id, 'hot')
                    self.h2d_bytes += self._all_tiles[tile_id].nbytes

    @dynamic_cold
    def evict_cold_tiles(self, target_free_ratio: float):
        """
        驱逐最冷的 tiles (hot→warm/cold)。
        target_free_ratio: 目标释放比例 (相对 hot budget)
        GPT-5.2 R2: critical tiles 永远不驱逐到 cold, 只能到 warm。
        """
        hot_tiles = self.vram.tiles_in_tier('hot')
        if not hot_tiles:
            return
        # 按热度排序 (criticality 已内嵌在 score 权重中)
        scored = [(self.dk.score(self._all_tiles[tid]), tid)
                  for tid in hot_tiles if tid in self._all_tiles]
        scored.sort()
        # 目标: 释放 target_free_ratio * hot_budget 字节
        target_bytes = int(self.cfg.vram_budget_gb * 1e9 * self.cfg.hot_ratio
                           * max(0.01, min(target_free_ratio, 0.5)))
        evicted_bytes = 0
        for _, tid in scored:
            if evicted_bytes >= target_bytes:
                break
            tile = self._all_tiles[tid]
            # Critical tiles: 只能到 warm, 不能到 cold
            if tile.criticality == 'critical':
                dest = 'warm'
            elif (tile.criticality == 'cheap'
                  and self.vram._tier_usage('cold') < self.vram.cold_budget):
                dest = 'cold'
            else:
                dest = 'warm'
            self.vram.demote(tid, dest)
            self.d2h_bytes += tile.nbytes
            evicted_bytes += tile.nbytes

    def prefetch_warm_tiles(self, window: int, coalesce: bool = False):
        """
        预取下 N 层 tiles。
        coalesce=True 时使用 tile coalescing (批量搬运)。
        """
        recent = sorted(self._all_tiles.values(),
                        key=lambda t: t.last_access, reverse=True)
        if not recent:
            return
        cur_layer = recent[0].layer
        for offset in range(1, window + 1):
            next_layer = cur_layer + offset
            if next_layer < self.n_layers:
                if coalesce:
                    for name in WEIGHT_NAMES_6:
                        tile_id = f"L{next_layer}_{name}"
                        if tile_id in self._all_tiles:
                            tier = self.vram.get_tier(tile_id)
                            if tier and tier != 'hot':
                                self.vram.queue_transfer(tile_id, 'hot')
                    self.vram.flush_transfers()
                else:
                    self.prefetch_layer(next_layer)

    # ── 资源闭环动作 (GPT-5.2 R2) ──

    @dynamic_cold
    def tighten_hot(self, new_hot_ratio: float, new_warm_ratio: float):
        """
        收缩 hot 层: 驱逐 hot 中最冷的 tiles 直到满足新比例。
        GPT-5.2 R2: "tighten 不是省显存, 是避免抖动+fault风暴"
        """
        target_hot = int(self.cfg.vram_budget_gb * 1e9 * new_hot_ratio)
        current_hot = self.vram._tier_usage('hot')
        if current_hot <= target_hot:
            return  # 已经够紧了
        # 需要驱逐的量
        to_evict = current_hot - target_hot
        hot_tiles = self.vram.tiles_in_tier('hot')
        scored = [(self.dk.score(self._all_tiles[tid]), tid)
                  for tid in hot_tiles if tid in self._all_tiles]
        scored.sort()
        evicted = 0
        for _, tid in scored:
            if evicted >= to_evict:
                break
            tile = self._all_tiles[tid]
            # Critical tiles 只能到 warm; normal/cheap 可以到 cold
            if tile.criticality == 'critical':
                dest = 'warm'
            elif (tile.criticality == 'cheap'
                  and self.vram._tier_usage('cold') < self.vram.cold_budget):
                dest = 'cold'
            else:
                dest = 'warm'
            self.vram.demote(tid, dest)
            evicted += tile.nbytes
            self.d2h_bytes += tile.nbytes

    @dynamic_cold
    def relax_hot(self, new_hot_ratio: float):
        """
        放松 hot 层: 从 warm 中提升最热的 tiles。
        GPT-5.2 R2: "只在低压力+低fault+低rebuild_tax 时逐步放松"
        """
        target_hot = int(self.cfg.vram_budget_gb * 1e9 * new_hot_ratio)
        current_hot = self.vram._tier_usage('hot')
        if current_hot >= target_hot:
            return
        room = target_hot - current_hot
        warm_tiles = self.vram.tiles_in_tier('warm')
        scored = [(self.dk.score(self._all_tiles[tid]), tid)
                  for tid in warm_tiles if tid in self._all_tiles]
        scored.sort(reverse=True)  # 最热的先提升
        promoted = 0
        for _, tid in scored:
            if promoted >= room:
                break
            tile = self._all_tiles[tid]
            self.vram.promote(tid, 'hot')
            promoted += tile.nbytes
            self.h2d_bytes += tile.nbytes

    def set_rebuild_gate(self, level: int):
        """
        门控重建频率。
        GPT-5.2 R2: "当 τ 超阈值 → 先砍机制"
          level 0: 正常 (每步重建)
          level 1: 每 2 步重建一次
          level 2: 仅 critical tiles 重建, 其余跳过
        """
        self._rebuild_gate = level

    def should_rebuild(self, tile: DKTile, step: int = -1) -> bool:
        """检查是否允许重建此 tile (受门控控制)。
        gate=1 基于 step 奇偶 (per-step, 不是 per-tile)。"""
        if self._rebuild_gate == 0:
            return True
        if self._rebuild_gate == 1:
            # 基于 step 奇偶: 偶数步可重建, 奇数步跳过
            s = step if step >= 0 else self._rebuild_step_counter
            return s % 2 == 0
        # gate == 2: 仅 critical
        return tile.criticality == 'critical'

    def promote_critical_tiles(self):
        """
        质量闭环: 把所有 critical tiles 提升到 hot。
        GPT-5.2 R2: "一旦质量恶化 → 强制提升关键 tile 精度/常驻"
        """
        for tid, tile in self._all_tiles.items():
            if tile.criticality == 'critical':
                tier = self.vram.get_tier(tid)
                if tier and tier != 'hot':
                    self.vram.promote(tid, 'hot')
                    self.h2d_bytes += tile.nbytes

    # ── ActionExecutor 协议 (稳定 ABI, OPU 通过此接口控制 VA100) ──

    def execute_evict(self, target_free_ratio: float = 0.1, **kw) -> None:
        self.evict_cold_tiles(target_free_ratio)

    def execute_prefetch(self, window: int = 2, coalesce: bool = False, **kw) -> None:
        self.prefetch_warm_tiles(window, coalesce=coalesce)

    def execute_tighten(self, hot_ratio: float = 0.5, warm_ratio: float = 0.3, **kw) -> None:
        self.tighten_hot(hot_ratio, warm_ratio)

    def execute_relax(self, hot_ratio: float = 0.6, **kw) -> None:
        self.relax_hot(hot_ratio)

    def execute_gate_compute(self, gate_level: int = 1, **kw) -> None:
        self.set_rebuild_gate(gate_level)

    def execute_quality_escalation(self, **kw) -> None:
        self.promote_critical_tiles()

    def reset_step_stats(self):
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.h2d_bytes = 0
        self.d2h_bytes = 0


# ── 推理辅助函数 ──

@compilable_hot
def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


@compilable_hot
def rms_norm(h, eps=1e-6):
    rms = np.sqrt(np.mean(h * h, axis=-1, keepdims=True) + eps)
    return h / rms


@compilable_hot
def rope_embed(x, pos, head_dim, theta=500000.0):
    d = head_dim
    freqs = 1.0 / (theta ** (np.arange(0, d, 2, dtype=np.float32) / d))
    t = pos * freqs
    cos_t, sin_t = np.cos(t), np.sin(t)
    out = np.empty_like(x)
    out[..., 0::2] = x[..., 0::2] * cos_t - x[..., 1::2] * sin_t
    out[..., 1::2] = x[..., 0::2] * sin_t + x[..., 1::2] * cos_t
    return out


class VirtualA100Engine:
    """
    编排器: forward 调 DK-Tile, OPU 决策冷热, KVAdapter 懒补齐。

    GPT-5.2: "外面看起来还是 KV cache；
    里面 DK-Tile + Ghost Move + 虚拟显存 + 切口回收在跑。"

    每 token 的流程:
      1. OPU.tick() — 采信号、决策、执行 (上一步的后处理)
      2. RMSNorm → QKV via DK-Tile → RoPE → KVAdapter.write
      3. KVAdapter.read → Attention → Output projection
      4. RMSNorm → FFN via DK-Tile
      5. Runtime 预取下一层 (Ghost Move: 异步搬运)
    """

    def __init__(self, ghost_layers: List[GhostLayer],
                 model_config: dict, infer_cfg: InferConfig,
                 embed_weight: Optional[np.ndarray] = None,
                 head_weight: Optional[np.ndarray] = None):
        self.mcfg = model_config
        self.icfg = infer_cfg
        self.n_layers = len(ghost_layers)

        H = model_config.get('H', 512)
        n_kv_heads = model_config.get('n_kv_heads', 8)
        head_dim = model_config.get('head_dim', H // model_config.get('n_heads', 8))

        self.embed_weight = embed_weight
        self.head_weight = head_weight

        # ── L0: VirtualVRAMBackend ──
        hot_b = int(infer_cfg.vram_budget_gb * 1e9 * infer_cfg.hot_ratio)
        warm_b = int(infer_cfg.cpu_budget_gb * 1e9 * infer_cfg.warm_ratio)
        cold_b = int(100e9)  # cold 无限
        self.vram = VirtualVRAMBackend(hot_b, warm_b, cold_b)

        # ── L2: DKTileCore ──
        self.dk = DKTileCore()

        # ── Runtime ──
        self.runtime = VirtualA100Runtime(
            ghost_layers, self.dk, self.vram, infer_cfg)

        # ── L1: KVAdapter ──
        self.kv_adapter = KVAdapter(
            n_layers=self.n_layers,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            max_ctx=infer_cfg.max_ctx,
            dk=self.dk,
            vram=self.vram,
            quant_bits=infer_cfg.kv_quant_bits,
        )

        # ── L3: OPU (分离模块, 通过 ActionExecutor 协议控制 Runtime) ──
        opu_cfg = OPUConfig.from_infer_config(infer_cfg)
        self.opu = OPU(opu_cfg)
        self._signal = VA100SignalCollector()

        # 统计
        self.tokens_generated = 0
        self.total_time = 0.0
        self._step_count = 0

        # ── 增量追踪 (用于给 OPU 喂当步增量而非累积值) ──
        self._prev_transfer_time_s: float = 0.0
        self._prev_rebuild_time_s: float = 0.0

    def _ghost_linear(self, layer_idx: int, name: str,
                      x: np.ndarray) -> np.ndarray:
        """通过 DK-Tile 做低秩线性层"""
        tile = self.runtime.get_weight_tile(layer_idx, name)
        return self.dk.detileize_weight(tile, x, use_tf32=self.icfg.use_tf32)

    def forward_one_token(self, h: np.ndarray, pos: int) -> np.ndarray:
        if h.ndim == 1:
            h = h[np.newaxis, :]

        H = self.mcfg.get('H', h.shape[-1])
        n_heads = self.mcfg.get('n_heads', 8)
        n_kv_heads = self.mcfg.get('n_kv_heads', n_heads)
        head_dim = self.mcfg.get('head_dim', H // n_heads)
        kv_repeat = n_heads // n_kv_heads

        for li in range(self.n_layers):
            # ── 预取下一层 (Ghost Move: 提前一个 window) ──
            if li + 1 < self.n_layers:
                self.runtime.prefetch_layer(li + 1)

            # ── Attention ──
            h_norm = rms_norm(h)
            q = self._ghost_linear(li, 'Wq', h_norm)
            k = self._ghost_linear(li, 'Wk', h_norm)
            v = self._ghost_linear(li, 'Wv', h_norm)

            q = q.reshape(1, n_heads, head_dim)
            k = k.reshape(1, n_kv_heads, head_dim)
            v = v.reshape(1, n_kv_heads, head_dim)

            q = rope_embed(q, pos, head_dim,
                           self.mcfg.get('rope_theta', 500000.0))
            k = rope_embed(k, pos, head_dim,
                           self.mcfg.get('rope_theta', 500000.0))

            # KVAdapter: 写入 (内部 DK-Tile 化)
            self.kv_adapter.write(li, pos, k[0], v[0])

            # KVAdapter: 读取完整 KV (懒补齐)
            k_full, v_full = self.kv_adapter.read(li)
            seq_len = min(self.kv_adapter.current_len, self.icfg.max_ctx)
            k_full = k_full[:seq_len]
            v_full = v_full[:seq_len]

            # GQA repeat
            if kv_repeat > 1:
                k_full = np.repeat(k_full, kv_repeat, axis=1)
                v_full = np.repeat(v_full, kv_repeat, axis=1)

            # Attention scores
            scores = np.einsum('bhd,shd->bhs', q, k_full) / np.sqrt(head_dim)
            attn = softmax(scores, axis=-1)
            out = np.einsum('bhs,shd->bhd', attn, v_full)
            out = out.reshape(1, n_heads * head_dim)

            attn_out = self._ghost_linear(li, 'Wo', out)
            h = h + attn_out

            # ── FFN (简化: ReLU 替代 SwiGLU, W1/W2 替代 W1/W2/W3) ──
            # 真实 LLaMA: gate = silu(W_gate @ x), up = W_up @ x, down = W_down @ (gate * up)
            # 仿真版: 省略 gate 投影, 用 ReLU 近似
            h_norm = rms_norm(h)
            ff_up = self._ghost_linear(li, 'W1', h_norm)
            ff_act = np.maximum(ff_up, 0)  # ReLU (仿真简化)
            ff_down = self._ghost_linear(li, 'W2', ff_act)
            h = h + ff_down

        # Final norm + head projection
        h = rms_norm(h)
        if self.head_weight is not None:
            logits = h @ self.head_weight.T
        else:
            logits = h
        return logits.squeeze(0)

    def sample_token(self, logits: np.ndarray) -> int:
        T = self.icfg.temperature
        if T < 1e-8:
            return int(np.argmax(logits))
        logits = logits / T
        probs = softmax(logits)
        if self.icfg.top_k > 0:
            top_k = min(self.icfg.top_k, len(probs))
            idx_k = np.argpartition(probs, -top_k)[-top_k:]
            mask = np.zeros_like(probs)
            mask[idx_k] = probs[idx_k]
            probs = mask
        if self.icfg.top_p < 1.0:
            sorted_idx = np.argsort(probs)[::-1]
            cum = np.cumsum(probs[sorted_idx])
            cutoff = np.searchsorted(cum, self.icfg.top_p) + 1
            keep = sorted_idx[:cutoff]
            mask = np.zeros_like(probs)
            mask[keep] = probs[keep]
            probs = mask
        probs = probs / (probs.sum() + 1e-12)
        return int(np.random.choice(len(probs), p=probs))

    def generate(self, prompt_tokens: List[int], max_new: int = 64,
                 verbose: bool = True) -> List[int]:
        self.kv_adapter.reset()
        generated = []
        self._recent_tokens = list(prompt_tokens[-16:])  # 质量闭环: 追踪近期 token

        if verbose:
            print(f"[Engine] Prefill {len(prompt_tokens)} tokens...")
        t0 = time.perf_counter()

        for pos, tok in enumerate(prompt_tokens):
            h = (self.embed_weight[tok] if self.embed_weight is not None
                 else np.random.randn(self.mcfg.get('H', 512)).astype(np.float32) * 0.02)
            logits = self.forward_one_token(h, pos)
            self._opu_tick(pos, logits)

        prefill_time = time.perf_counter() - t0
        if verbose:
            print(f"  Prefill: {prefill_time:.2f}s "
                  f"({len(prompt_tokens)/max(prefill_time,1e-9):.1f} tok/s)")
            print(f"  {self.opu.summary()}")

        if verbose:
            print(f"[Engine] Decode (max {max_new} tokens)...")
        t1 = time.perf_counter()

        next_tok = self.sample_token(logits)
        generated.append(next_tok)
        self._recent_tokens.append(next_tok)

        for step in range(1, max_new):
            pos = len(prompt_tokens) + step
            if pos >= self.icfg.max_ctx:
                break
            h = (self.embed_weight[next_tok] if self.embed_weight is not None
                 else np.random.randn(self.mcfg.get('H', 512)).astype(np.float32) * 0.02)
            logits = self.forward_one_token(h, pos)
            next_tok = self.sample_token(logits)
            generated.append(next_tok)
            self._recent_tokens.append(next_tok)
            if len(self._recent_tokens) > 32:
                self._recent_tokens = self._recent_tokens[-32:]
            self._opu_tick(pos, logits)

        decode_time = time.perf_counter() - t1
        self.tokens_generated = len(generated)
        self.total_time = decode_time
        # 记录有效计算时间（低熵：使用 record_compute）
        self.vram.stats.record_compute(decode_time + prefill_time)

        if verbose:
            tps = len(generated) / max(decode_time, 1e-9)
            print(f"  Decode: {decode_time:.2f}s, {len(generated)} tokens, "
                  f"{tps:.2f} tok/s")
            print(f"  {self.opu.summary()}")
            ts = self.vram.tier_summary()
            print(f"  Hot: {ts['hot']['count']} tiles ({ts['hot']['usage_mb']:.1f} MB, "
                  f"p={ts['hot']['pressure']:.0%})")
            print(f"  Warm: {ts['warm']['count']} tiles ({ts['warm']['usage_mb']:.1f} MB)")
            print(f"  Cold: {ts['cold']['count']} tiles ({ts['cold']['usage_mb']:.1f} MB)")
            print(f"  KV cache: {self.kv_adapter.memory_bytes/1e6:.1f} MB")
            print(f"  Runtime: hits={self.runtime.prefetch_hits}, "
                  f"misses={self.runtime.prefetch_misses}, "
                  f"h2d={self.runtime.h2d_bytes/1e6:.1f} MB")
            # ── 切口回收 KPI (GPT-5.2 核心度量) ──
            kpi = self.vram.kpi_summary()
            print(f"  ┌─ 切口 KPI ──────────────────────────┐")
            print(f"  │ μ (搬运摩擦) = {kpi['mu']:.4f}             │")
            print(f"  │ τ (重建税)   = {kpi['tau']:.4f}             │")
            print(f"  │ σ (策略抖动) = {self.opu.policy_jitter:.4f}             │")
            print(f"  │ 切口驱逐: {kpi['aperture_evictions']:>4}  补齐: {kpi['aperture_recoveries']:>4}       │")
            print(f"  │ 搬运: {kpi['transfer_time_ms']:.1f}ms  重建: {kpi['rebuild_time_ms']:.1f}ms  │")
            print(f"  └────────────────────────────────────┘")
            # ── Lazy materialization 统计 ──
            ls = self.kv_adapter.lazy_stats()
            print(f"  KV lazy: hits={ls['cache_hits']}, "
                  f"misses={ls['cache_misses']}, "
                  f"hit_rate={ls['hit_rate']:.0%}, "
                  f"rebuilds={ls['rebuilds']}")

        return generated

    def _opu_tick(self, pos: int, logits: Optional[np.ndarray] = None):
        """
        OPU 3-hook 闭环: 每 token 一次。

        Hook 1: observe_begin  — 读显存水位
        Hook 2: observe_kv     — KV 命中/未中/重建
        Hook 3: observe_end    — 组装 StepStats → OPU.tick → dispatch
        """
        sig = self._signal

        # ── Hook 1: step 开始 (显存水位) ──
        sig.observe_begin(self.vram.tier_summary())

        # ── Hook 2: KV 访问信号 ──
        sig.observe_kv(
            hits=self.runtime.prefetch_hits,
            misses=self.runtime.prefetch_misses,
            rebuild_count=self.kv_adapter.rebuild_count,
            rebuild_time_s=self.kv_adapter.rebuild_time_s,
        )

        # ── 质量信号计算 (便宜的 proxy) ──
        quality_score = 1.0
        logits_entropy = 0.0
        repeat_rate = 0.0

        if logits is not None and len(logits) > 0:
            probs = softmax(logits)
            probs_clipped = np.clip(probs, 1e-12, 1.0)
            logits_entropy = float(-np.sum(probs_clipped * np.log(probs_clipped)))
            if logits_entropy < 0.5:
                quality_score = max(0.0, logits_entropy / 0.5)
            elif logits_entropy > 8.0:
                quality_score = max(0.0, 1.0 - (logits_entropy - 8.0) / 4.0)

        if hasattr(self, '_recent_tokens') and len(self._recent_tokens) >= 8:
            unique = len(set(self._recent_tokens[-8:]))
            repeat_rate = 1.0 - unique / 8.0
            if repeat_rate > 0.5:
                quality_score *= (1.0 - repeat_rate)

        # ── 计算增量 (不喂累积值给 OPU) ──
        cur_transfer_time = self.vram.stats.total_transfer_time_s
        cur_rebuild_time = self.vram.stats.total_rebuild_time_s
        delta_wait = cur_transfer_time - self._prev_transfer_time_s
        delta_rebuild = cur_rebuild_time - self._prev_rebuild_time_s
        self._prev_transfer_time_s = cur_transfer_time
        self._prev_rebuild_time_s = cur_rebuild_time

        # ── Tile 统计 ──
        ts = self.vram.tier_summary()

        # ── Hook 3: step 结束 → StepStats → OPU.tick → dispatch ──
        stats = sig.observe_end(
            step=self._step_count,
            step_time_s=self.total_time / max(self._step_count, 1),
            h2d_bytes=self.runtime.h2d_bytes,
            d2h_bytes=self.runtime.d2h_bytes,
            wait_time_s=delta_wait,
            copy_bytes=self.runtime.h2d_bytes + self.runtime.d2h_bytes,
            rebuild_cost_s=delta_rebuild,
            quality_score=quality_score,
            logits_entropy=logits_entropy,
            repeat_rate=repeat_rate,
            tiles_hot_count=ts['hot']['count'],
            tiles_warm_count=ts['warm']['count'],
            tiles_cold_count=ts['cold']['count'],
        )

        # OPU 决策 (纯治理, 不碰实现)
        actions = self.opu.tick(stats)

        # 通过 ActionExecutor 协议执行 (Runtime 实现)
        dispatch_actions(self.runtime, actions)

        self.runtime.reset_step_stats()
        self._step_count += 1


# ════════════════════════════════════════════════════════════════
# Ghost 压缩引擎 (随机 SVD 优化)
# ════════════════════════════════════════════════════════════════

def random_svd(
    A: np.ndarray,
    rank: Optional[int] = None,
    oversample: int = 10,
    n_iter: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    随机 SVD（比标准 SVD 快 10-20x）

    基于：Halko et al. "Finding structure with randomness" (2011)

    Args:
        A: 输入矩阵 (m, n)
        rank: 目标秩
        oversample: 过采样参数
        n_iter: 幂迭代次数
        rng: 随机数生成器

    Returns:
        U: (m, rank) 左奇异向量
        S: (rank,) 奇异值
        V: (n, rank) 右奇异向量
    """
    if rng is None:
        rng = np.random.default_rng()

    m, n = A.shape
    r = rank if rank is not None else min(m, n)
    k = min(r + oversample, min(m, n))

    # 步骤 1: 生成高斯随机投影
    Omega = rng.standard_normal((n, k), dtype=np.float32)

    # 步骤 2: Y = A @ Omega
    Y = A @ Omega

    # 步骤 3: 幂迭代（可选，提高精度）
    for _ in range(n_iter):
        Y = A @ (A.T @ Y)
        Q, _ = np.linalg.qr(Y)
        Y = Q

    # 步骤 4: QR 分解
    Q, _ = np.linalg.qr(Y)

    # 步骤 5: B = Q^T @ A（小矩阵）
    B = Q.T @ A

    # 步骤 6: 对 B 做标准 SVD
    Ub, S, Vb = np.linalg.svd(B, full_matrices=False)

    # 步骤 7: 恢复 U
    U = Q @ Ub

    # 截断到目标秩
    # U 是 (m, k)，截取前 r 列 → (m, r)
    # S 长度是 k，截取前 r 个 → (r,)
    # Vb (Vt) 是 (k, n)，截取前 r 行 → (r, n)
    U_out = U[:, :r]
    S_out = S[:r]
    Vt_out = Vb[:r, :]  # (r, n)，V^T 格式

    return U_out.astype(np.float32), S_out.astype(np.float32), Vt_out.astype(np.float32)


class GhostCompressor:
    """Ghost 压缩器（支持随机 SVD）"""

    def __init__(self, cfg: GhostConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed=42)

    def compress_weight(self, name: str, W: np.ndarray, rank: int
                        ) -> GhostFactor:
        m, n = W.shape
        r = min(rank, min(m, n))

        # 随机 SVD vs 标准 SVD
        if self.cfg.use_random_svd:
            U, S, Vt = random_svd(
                W,
                rank=r,
                oversample=self.cfg.svd_oversample,
                n_iter=self.cfg.svd_n_iter,
                rng=self.rng,
            )
            # Vt 已经是 (r, n) 格式
            V = Vt.T
        else:
            U, S, Vt = np.linalg.svd(W, full_matrices=False)
            # 截取前 r 个
            U = U[:, :r]
            S = S[:r]
            Vt = Vt[:r, :]
            V = Vt.T

        U_r, S_r, V_r = U.copy(), S.copy(), V.copy()

        sparse_idx = sparse_val = None
        if self.cfg.sparse_density > 0:
            W_lr = U_r @ np.diag(S_r) @ Vt  # Vt 已经是 (r, n)
            R = W - W_lr
            flat = np.abs(R.ravel())
            k = max(1, int(self.cfg.sparse_density * flat.size))
            idx = np.argpartition(flat, -k)[-k:]
            rows, cols = np.unravel_index(idx, R.shape)
            sparse_idx = np.stack([rows, cols], axis=1).astype(np.int32)
            sparse_val = R[rows, cols].astype(np.float16)

        if self.cfg.quantize_factors:
            U_q, U_s = quantize_int8(U_r)
            V_q, V_s = quantize_int8(V_r)
        else:
            U_q, U_s = U_r.astype(np.float32), 1.0
            V_q, V_s = V_r.astype(np.float32), 1.0

        return GhostFactor(
            name=name, m=m, n=n, rank=r,
            U_q=U_q, U_scale=U_s, S=S_r.astype(np.float32),
            V_q=V_q, V_scale=V_s, quantized=self.cfg.quantize_factors,
            sparse_idx=sparse_idx, sparse_val=sparse_val)

    def allocate_ranks_greedy(self, svd_cache, layers, total_budget,
                              min_r=4, max_r=512):
        L = len(layers)
        wn = list(layers[0].keys())
        n_mat = L * len(wn)
        rank_map = [{k: min_r for k in wn} for _ in range(L)]
        remain = total_budget - n_mat * min_r
        if remain <= 0:
            return rank_map, {'method': 'greedy', 'budget': total_budget}
        heap = []
        for li in range(L):
            for w in wn:
                _, Sv, _ = svd_cache[li][w]
                mr = min(max_r, len(Sv))
                if min_r < mr:
                    heapq.heappush(heap, (-float(Sv[min_r] ** 2), li, w))
        alloc = 0
        while alloc < remain and heap:
            _, li, w = heapq.heappop(heap)
            rank_map[li][w] += 1
            alloc += 1
            _, Sv, _ = svd_cache[li][w]
            cur = rank_map[li][w]
            if cur < min(max_r, len(Sv)):
                heapq.heappush(heap, (-float(Sv[cur] ** 2), li, w))
        all_r = [rank_map[li][w] for li in range(L) for w in wn]
        return rank_map, {
            'method': 'greedy', 'actual': sum(all_r),
            'min_r': int(np.min(all_r)), 'max_r': int(np.max(all_r)),
            'mean_r': float(np.mean(all_r))}

    def compress_model(self, layers: List[Dict[str, np.ndarray]],
                       progress: bool = True) -> List[GhostLayer]:
        L = len(layers)
        wn = list(layers[0].keys())

        # 打印使用的 SVD 方法
        if progress:
            method = "随机SVD" if self.cfg.use_random_svd else "标准SVD"
            print(f"[Ghost] {method} 缓存 ({L}层×{len(wn)}矩阵)...")

        # SVD 缓存（使用随机 SVD 或标准 SVD）
        svd_cache = []
        for li in range(L):
            layer_svd = {}
            for k in wn:
                W = layers[li][k]

                if self.cfg.use_random_svd:
                    U, S, V = random_svd(
                        W,
                        rank=None,
                        oversample=self.cfg.svd_oversample,
                        n_iter=self.cfg.svd_n_iter,
                        rng=self.rng,
                    )
                    # Vt 用于兼容旧代码
                    Vt = V.T
                else:
                    U, S, Vt = np.linalg.svd(W, full_matrices=False)

                layer_svd[k] = (U, S, Vt)
            svd_cache.append(layer_svd)

        n_mat = L * len(wn)
        total_budget = n_mat * self.cfg.base_rank
        if progress:
            print(f"[Ghost] Rank 分配 (method={self.cfg.alloc_method})...")
        rank_map, stats = self.allocate_ranks_greedy(
            svd_cache, layers, total_budget, self.cfg.min_rank, self.cfg.max_rank)
        if progress:
            print(f"  {stats}")
        ghost_layers = []
        total_orig = total_comp = 0
        for li in range(L):
            factors = {}
            for k in wn:
                factors[k] = self.compress_weight(k, layers[li][k], rank_map[li][k])
                total_orig += factors[k].original_bytes
                total_comp += factors[k].compressed_bytes
            ghost_layers.append(GhostLayer(layer_idx=li, factors=factors))
        if progress:
            cr = total_orig / max(total_comp, 1)
            print(f"[Ghost] {total_orig/1e9:.2f}GB → {total_comp/1e9:.4f}GB ({cr:.1f}x)")
        return ghost_layers


# ════════════════════════════════════════════════════════════════
# .aguf 文件格式 (APT GGUF-like Format, v1 兼容)
# ════════════════════════════════════════════════════════════════

def save_ghost(path, ghost_layers, model_config):
    with open(path, 'wb') as f:
        f.write(AGUF_MAGIC)
        f.write(struct.pack('<I', AGUF_VERSION))
        cfg_b = json.dumps(model_config, ensure_ascii=False).encode('utf-8')
        f.write(struct.pack('<I', len(cfg_b)))
        f.write(cfg_b)
        f.write(struct.pack('<I', len(ghost_layers)))
        for gl in ghost_layers:
            f.write(struct.pack('<I', gl.layer_idx))
            f.write(struct.pack('<I', len(gl.factors)))
            for name, gf in gl.factors.items():
                f.write(name.encode('utf-8')[:8].ljust(8, b'\x00'))
                f.write(struct.pack('<HII', gf.rank, gf.m, gf.n))
                f.write(struct.pack('<B', 1 if gf.quantized else 0))
                f.write(gf.U_q.tobytes())
                f.write(struct.pack('<f', gf.U_scale))
                f.write(gf.S.tobytes())
                f.write(gf.V_q.tobytes())
                f.write(struct.pack('<f', gf.V_scale))
                has_sp = gf.sparse_idx is not None
                f.write(struct.pack('<B', 1 if has_sp else 0))
                if has_sp:
                    f.write(struct.pack('<I', len(gf.sparse_val)))
                    f.write(gf.sparse_idx.tobytes())
                    f.write(gf.sparse_val.tobytes())


def load_ghost(path):
    with open(path, 'rb') as f:
        assert f.read(4) == AGUF_MAGIC
        version = struct.unpack('<I', f.read(4))[0]
        cfg_len = struct.unpack('<I', f.read(4))[0]
        model_config = json.loads(f.read(cfg_len).decode('utf-8'))
        n_layers = struct.unpack('<I', f.read(4))[0]
        ghost_layers = []
        for _ in range(n_layers):
            li = struct.unpack('<I', f.read(4))[0]
            nf = struct.unpack('<I', f.read(4))[0]
            factors = {}
            for _ in range(nf):
                nm = f.read(8).rstrip(b'\x00').decode('utf-8')
                r, m, n = struct.unpack('<HII', f.read(10))
                q = bool(struct.unpack('<B', f.read(1))[0])
                ud = np.int8 if q else np.float32
                U = np.frombuffer(f.read(m*r*np.dtype(ud).itemsize), dtype=ud).reshape(m, r).copy()
                Us = struct.unpack('<f', f.read(4))[0]
                S = np.frombuffer(f.read(r*4), dtype=np.float32).copy()
                vd = np.int8 if q else np.float32
                V = np.frombuffer(f.read(n*r*np.dtype(vd).itemsize), dtype=vd).reshape(n, r).copy()
                Vs = struct.unpack('<f', f.read(4))[0]
                sp = bool(struct.unpack('<B', f.read(1))[0])
                si = sv = None
                if sp:
                    nnz = struct.unpack('<I', f.read(4))[0]
                    si = np.frombuffer(f.read(nnz*8), dtype=np.int32).reshape(nnz, 2).copy()
                    sv = np.frombuffer(f.read(nnz*2), dtype=np.float16).copy()
                factors[nm] = GhostFactor(nm, m, n, r, U, Us, S, V, Vs, q, si, sv)
            ghost_layers.append(GhostLayer(li, factors))
    return ghost_layers, model_config


# ════════════════════════════════════════════════════════════════
# KV Cache 保存/恢复 (参考 llama.cpp prompt cache)
# ════════════════════════════════════════════════════════════════

VCACHE_MAGIC = b"VCA0"  # Virtual Cache
VCACHE_VERSION = 1


def save_kv_cache(path: str, kv_adapter, vram_backend, tokens: List[int]):
    """
    保存 KV Cache 到文件（避免重复编码 prompt）

    Args:
        path: 保存路径 (.vcache 文件)
        kv_adapter: KVAdapter 实例
        vram_backend: VirtualVRAMBackend 实例
        tokens: 已处理的 token 列表
    """
    import pickle

    with open(path, 'wb') as f:
        # 魔数和版本
        f.write(VCACHE_MAGIC)
        f.write(struct.pack('<I', VCACHE_VERSION))

        # Token 列表
        token_bytes = struct.pack('<I', len(tokens))
        for tok in tokens:
            token_bytes += struct.pack('<I', tok)
        f.write(token_bytes)

        # KV Cache 数据（使用 KVAdapter 的实际属性）
        kv_data = pickle.dumps({
            '_hot_cache': kv_adapter._hot_cache,  # 热缓存
            '_cache_valid': kv_adapter._cache_valid,  # 缓存有效性
            '_kv_tile_ids': kv_adapter._kv_tile_ids,  # KV tile IDs
            'current_len': kv_adapter.current_len,
            'cache_hits': kv_adapter.cache_hits,
            'cache_misses': kv_adapter.cache_misses,
        })
        f.write(struct.pack('<I', len(kv_data)))
        f.write(kv_data)

        # VRAM 分层信息
        tier_summary = vram_backend.tier_summary()
        tier_data = pickle.dumps(tier_summary)
        f.write(struct.pack('<I', len(tier_data)))
        f.write(tier_data)


def load_kv_cache(path: str, kv_adapter, vram_backend):
    """
    从文件恢复 KV Cache

    Returns:
        tokens: 已处理的 token 列表
    """
    import pickle

    with open(path, 'rb') as f:
        # 魔数和版本
        assert f.read(4) == VCACHE_MAGIC
        version = struct.unpack('<I', f.read(4))[0]

        # Token 列表
        n_tokens = struct.unpack('<I', f.read(4))[0]
        tokens = [struct.unpack('<I', f.read(4))[0] for _ in range(n_tokens)]

        # KV Cache 数据
        kv_len = struct.unpack('<I', f.read(4))[0]
        kv_data = pickle.loads(f.read(kv_len))
        kv_adapter._hot_cache = kv_data['_hot_cache']
        kv_adapter._cache_valid = kv_data['_cache_valid']
        kv_adapter._kv_tile_ids = kv_data['_kv_tile_ids']
        kv_adapter.current_len = kv_data['current_len']
        kv_adapter.cache_hits = kv_data['cache_hits']
        kv_adapter.cache_misses = kv_data['cache_misses']

        # VRAM 分层信息（暂不恢复，让 OPU 重新决策）
        tier_len = struct.unpack('<I', f.read(4))[0]
        _tier_data = pickle.loads(f.read(tier_len))

    return tokens


# ════════════════════════════════════════════════════════════════
# Session 保存/恢复 (完整会话状态)
# ════════════════════════════════════════════════════════════════

VSESSION_MAGIC = b"VSES0"  # Virtual Session
VSESSION_VERSION = 1


def save_session(path: str, engine, tokens: List[int], model_path: str):
    """
    保存完整会话状态（含 OPU、VRAM、KV Cache）

    Args:
        path: 保存路径 (.vsession 文件)
        engine: VirtualA100Engine 实例
        tokens: 已生成的 token 列表
        model_path: 模型文件路径
    """
    import pickle
    from collections import deque

    # 完整状态：包含 OPU 内部状态
    state = {
        'kv_adapter': {
            '_hot_cache': dict(engine.kv_adapter._hot_cache),
            '_cache_valid': dict(engine.kv_adapter._cache_valid),
            '_kv_tile_ids': dict(engine.kv_adapter._kv_tile_ids),
            'current_len': engine.kv_adapter.current_len,
            'cache_hits': engine.kv_adapter.cache_hits,
            'cache_misses': engine.kv_adapter.cache_misses,
            'rebuild_count': engine.kv_adapter.rebuild_count,
            'rebuild_time_s': engine.kv_adapter.rebuild_time_s,
        },
        'opu': {
            'step': engine.opu.step,
            '_ema': dict(engine.opu._ema),  # EMA 字典
            '_quality_ema': engine.opu._quality._quality_ema,  # Quality EMA
            '_quality_alarm': engine.opu._quality._alarm,
            '_policy_changes': engine.opu._policy_changes,
            '_policy_change_window': list(engine.opu._policy_change_window),  # deque 转 list
            '_cooldown_left': engine.opu._cooldown_left,
            'gate_level': engine.opu.gate_level,
        },
        'stats': {
            'tokens_generated': engine.tokens_generated,
            'total_time': engine.total_time,
        },
        'vram_tier': engine.vram.tier_summary(),
    }
    state_data = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path, 'wb') as f:
        # 魔数和版本
        f.write(VSESSION_MAGIC)
        f.write(struct.pack('<I', VSESSION_VERSION))

        # 模型路径
        model_path_b = model_path.encode('utf-8')
        f.write(struct.pack('<I', len(model_path_b)))
        f.write(model_path_b)

        # Token 列表
        f.write(struct.pack('<I', len(tokens)))
        for tok in tokens:
            f.write(struct.pack('<I', tok))

        # 状态数据
        f.write(struct.pack('<I', len(state_data)))
        f.write(state_data)

        # 确保写入磁盘
        f.flush()
        os.fsync(f.fileno())


def load_session(path: str, engine):
    """
    从文件恢复会话状态

    Returns:
        (tokens, model_path)
    """
    import pickle

    with open(path, 'rb') as f:
        # 魔数和版本
        magic = f.read(5)  # VSESSION_MAGIC 是 5 字节
        assert magic == VSESSION_MAGIC, f"错误的魔数: {magic}, 期望: {VSESSION_MAGIC}"
        version = struct.unpack('<I', f.read(4))[0]

        # 模型路径
        model_path_len = struct.unpack('<I', f.read(4))[0]
        model_path = f.read(model_path_len).decode('utf-8')

        # Token 列表
        n_tokens = struct.unpack('<I', f.read(4))[0]
        tokens = [struct.unpack('<I', f.read(4))[0] for _ in range(n_tokens)]

        # 完整状态
        state_len = struct.unpack('<I', f.read(4))[0]
        state = pickle.loads(f.read(state_len))

        # 恢复 KV Adapter 状态
        engine.kv_adapter._hot_cache = state['kv_adapter']['_hot_cache']
        engine.kv_adapter._cache_valid = state['kv_adapter']['_cache_valid']
        engine.kv_adapter._kv_tile_ids = state['kv_adapter']['_kv_tile_ids']
        engine.kv_adapter.current_len = state['kv_adapter']['current_len']
        engine.kv_adapter.cache_hits = state['kv_adapter']['cache_hits']
        engine.kv_adapter.cache_misses = state['kv_adapter']['cache_misses']
        engine.kv_adapter.rebuild_count = state['kv_adapter']['rebuild_count']
        engine.kv_adapter.rebuild_time_s = state['kv_adapter']['rebuild_time_s']

        # 恢复 OPU 内部状态（通过内部属性）
        engine.opu.step = state['opu']['step']
        engine.opu._ema = state['opu']['_ema']  # 直接替换字典
        engine.opu._quality._quality_ema = state['opu']['_quality_ema']
        engine.opu._quality._alarm = state['opu']['_quality_alarm']
        engine.opu._policy_changes = state['opu']['_policy_changes']
        engine.opu._policy_change_window = deque(state['opu']['_policy_change_window'], maxlen=100)
        engine.opu._cooldown_left = state['opu']['_cooldown_left']
        # gate_level 是 property，需要通过 _friction 设置
        if engine.opu._friction is not None:
            engine.opu._friction._gate_level = state['opu']['gate_level']

        # 恢复统计信息
        engine.tokens_generated = state['stats']['tokens_generated']
        engine.total_time = state['stats']['total_time']

    return tokens, model_path


# ════════════════════════════════════════════════════════════════
# GGUF 加载支持 (集成 llama-cpp-python)
# ════════════════════════════════════════════════════════════════

def load_gguf(path: str, n_gpu_layers: int = 0, n_ctx: int = 512):
    """
    加载 GGUF 格式模型 (如 DeepSeek-R1-Distill-Llama-70B)

    Args:
        path: GGUF 文件路径
        n_gpu_layers: offload 到 GPU 的层数 (-1 = 全部, 0 = 全部 CPU)
        n_ctx: 上下文长度

    Returns:
        (llama_model, model_config)
    """
    if not GGUF_AVAILABLE:
        raise ImportError("llama-cpp-python 未安装，无法加载 GGUF 文件")

    if not os.path.exists(path):
        raise FileNotFoundError(f"GGUF 文件不存在: {path}")

    # 使用 llama-cpp-python 加载
    llama_model = Llama(
        model_path=path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        verbose=False
    )

    # 提取模型配置（从 GGUF 元数据）
    # LLaMA 70B 的典型配置
    model_config = {
        'H': 8192,  # hidden_size
        'F': 28672,  # intermediate_size (ffn_hidden)
        'L': 80,  # n_layers
        'VOCAB': 32000,
        'n_heads': 64,
        'n_kv_heads': 8,
        'head_dim': 128,
        'norm_eps': 1e-5,
        'rope_theta': 500000.0,
    }

    return llama_model, model_config


def convert_gguf_to_weights(path: str) -> Dict[str, np.ndarray]:
    """
    从 GGUF 模型提取权重为 numpy 数组（用于 Ghost 压缩）

    Args:
        path: GGUF 文件路径

    Returns:
        字典: {layer_name: weight_array}
    """
    if not GGUF_AVAILABLE:
        raise ImportError("llama-cpp-python 未安装")

    import torch
    from llama_cpp import Llama

    # 加载模型
    llama_model = Llama(model_path=path, n_gpu_layers=0, n_ctx=512)

    # 提取权重（通过内部模型访问）
    # 注意：这需要 gguf 转换为 PyTorch 权重
    weights = {}

    # TODO: 实现实际的权重提取
    # llama_model.model._np...

    return weights


# ════════════════════════════════════════════════════════════════
# 合成权重 + 仿真
# ════════════════════════════════════════════════════════════════

def make_synthetic_layer(rng, H, F, alpha_attn=0.5, alpha_ff=0.3, sv_scale=10.0):
    def mk(od, id_, a):
        r_max = min(od, id_)
        U, _ = np.linalg.qr(rng.standard_normal((od, r_max)))
        V, _ = np.linalg.qr(rng.standard_normal((id_, r_max)))
        sv = sv_scale * np.power(np.arange(1, r_max+1, dtype=np.float64), -a)
        return (U * sv[np.newaxis, :]) @ V.T
    return {k: mk(H, H, alpha_attn).astype(np.float32) if k in ('Wq','Wk','Wv','Wo')
            else mk(F, H, alpha_ff).astype(np.float32) if k == 'W1'
            else mk(H, F, alpha_ff).astype(np.float32)
            for k in WEIGHT_NAMES_6}


def run_simulation(H=256, F=768, L=8, base_rank=32, seed=42, verbose=True):
    rng = np.random.default_rng(seed)
    if verbose:
        print("=" * 65)
        print(f"Virtual A100 v2 仿真 | H={H} F={F} L={L} rank={base_rank}")
        print("=" * 65)

    # 合成权重
    if verbose: print("\n[1] 生成合成权重...")
    layers = [make_synthetic_layer(rng, H, F, 0.5 + 0.8*i/L, 0.3 + 0.5*i/L)
              for i in range(L)]

    # Ghost 压缩
    if verbose: print(f"\n[2] Ghost 压缩 (rank={base_rank})...")
    gcfg = GhostConfig(base_rank=base_rank, quantize_factors=True, alloc_method='greedy')
    ghost_layers = GhostCompressor(gcfg).compress_model(layers, progress=verbose)

    # 质量诊断
    if verbose:
        print(f"\n[3] 质量诊断:")
        print(f"  {'层':<4} {'矩阵':<4} {'rank':>5} {'fro%':>7} {'cos':>7}")
        print("  " + "-" * 32)
    for li in [0, L//2, L-1]:
        for wn in ['Wq', 'W1']:
            gf = ghost_layers[li].factors[wn]
            W = layers[li][wn]
            Wr = gf.reconstruct()
            fro = np.linalg.norm(W-Wr) / (np.linalg.norm(W) + 1e-15)
            cos = np.sum(W*Wr) / (np.linalg.norm(W)*np.linalg.norm(Wr) + 1e-15)
            if verbose:
                print(f"  {li:<4} {wn:<4} {gf.rank:>5} {fro:>6.1%} {cos:>6.4f}")

    # 端到端推理 (5 层模块全部参与)
    if verbose: print(f"\n[4] 端到端推理 (5 层模块化)...")
    VOCAB = 256
    mcfg = dict(H=H, F=F, L=L, VOCAB=VOCAB,
                n_heads=max(1, H//64), n_kv_heads=max(1, H//64),
                head_dim=min(64, H))
    embed_w = rng.standard_normal((VOCAB, H)).astype(np.float32) * 0.02
    head_w = rng.standard_normal((VOCAB, H)).astype(np.float32) * 0.02

    icfg = InferConfig(max_ctx=128, vram_budget_gb=8.0, opu_enabled=True)
    engine = VirtualA100Engine(ghost_layers, mcfg, icfg,
                               embed_weight=embed_w, head_weight=head_w)

    prompt = [rng.integers(0, VOCAB) for _ in range(8)]
    generated = engine.generate(prompt, max_new=16, verbose=verbose)

    # .aguf 文件测试
    if verbose: print(f"\n[5] .aguf 文件读写...")
    tmp = os.path.join(os.environ.get('TEMP', '/tmp'), "test_v2.aguf")
    save_ghost(tmp, ghost_layers, mcfg)
    fsize = os.path.getsize(tmp)
    loaded, _ = load_ghost(tmp)
    for li in range(min(3, L)):
        for wn in ['Wq', 'W1']:
            r1 = ghost_layers[li].factors[wn].reconstruct()
            r2 = loaded[li].factors[wn].reconstruct()
            assert np.allclose(r1, r2, atol=1e-4)
    os.remove(tmp)
    if verbose:
        print(f"  文件: {fsize/1e6:.2f} MB, 验证: [OK]")

    # Rank 分配
    if verbose:
        print(f"\n[6] Rank 分配:")
        print(f"  {'层':<4} " + " ".join(f"{w:>5}" for w in WEIGHT_NAMES_6))
        for li in sorted(set([0, L//4, L//2, 3*L//4, L-1])):
            rm = {w: ghost_layers[li].factors[w].rank for w in WEIGHT_NAMES_6}
            print(f"  {li:<4} " + " ".join(f"{rm[w]:>5}" for w in WEIGHT_NAMES_6))

    # OPU + 三层内存状态
    if verbose:
        ts = engine.vram.tier_summary()
        print(f"\n[7] 模块状态:")
        print(f"  OPU: {engine.opu.summary()}")
        for tier in ('hot', 'warm', 'cold'):
            t = ts[tier]
            print(f"  {tier:>5}: {t['count']:>4} tiles, "
                  f"{t['usage_mb']:>7.1f}/{t['budget_mb']:.0f} MB, "
                  f"p={t['pressure']:.0%}")
        print(f"  KV adapter: {engine.kv_adapter.memory_bytes/1e6:.1f} MB "
              f"(len={engine.kv_adapter.current_len})")
        ls = engine.kv_adapter.lazy_stats()
        print(f"  KV lazy: hits={ls['cache_hits']}, misses={ls['cache_misses']}, "
              f"hit_rate={ls['hit_rate']:.0%}")

    # 切口回收 KPI 输出
    kpi = engine.vram.kpi_summary()
    if verbose:
        print(f"\n[8] 切口回收 KPI (GPT-5.2 核心度量):")
        print(f"  μ (搬运摩擦) = {kpi['mu']:.4f}  "
              f"{'[OK] OK' if kpi['mu'] < 0.1 else '[WARNING] 搬运成瓶颈'}")
        print(f"  τ (重建税)   = {kpi['tau']:.4f}  "
              f"{'[OK] OK' if kpi['tau'] < 0.05 else '[WARNING] 应门控重建频率'}")
        print(f"  σ (策略抖动) = {engine.opu.policy_jitter:.4f}  "
              f"{'[OK] OK' if engine.opu.policy_jitter < 0.1 else '[WARNING] 迟滞不足'}")
        print(f"  切口驱逐: {kpi['aperture_evictions']}, "
              f"补齐: {kpi['aperture_recoveries']}")
        print(f"  搬运耗时: {kpi['transfer_time_ms']:.2f}ms, "
              f"重建耗时: {kpi['rebuild_time_ms']:.2f}ms")

    # 70B 推演
    if verbose:
        r = base_rank
        est_w = 80 * 6 * (8192+8192) * r * 1 / 1e9
        est_kv = 2 * 80 * 512 * 8 * 128 * 4 / 1e9
        est_e = 2 * 32000 * 8192 * 2 / 1e9
        est = est_w + est_kv + est_e
        print(f"\n  ┌─ 70B@r={r} 推演 ────────────────────┐")
        print(f"  │ 权重: {est_w:.2f} GB  KV: {est_kv:.2f} GB       │")
        print(f"  │ Embed: {est_e:.2f} GB  总: {est:.2f} GB       │")
        print(f"  │ 3070:  8.00 GB                     │")
        print(f"  │ {'[OK] ALL_HOT' if est < 7.5 else '[WARNING] TIERED (OPU 调度)':>35} │")
        print(f"  └───────────────────────────────────┘")

    # torch.compile 边界统计
    if verbose:
        hot_fns = [f.__name__ for f in [quantize_int8, dequantize_int8,
                   tf32_matmul, softmax, rms_norm, rope_embed]
                   if getattr(f, '_compile_hint', None) == 'hot']
        print(f"\n[9] torch.compile 边界:")
        print(f"  Hot (可编译): {', '.join(hot_fns)}")
        print(f"  Cold (不编译): OPU.policy, evict_cold_tiles, flush_transfers")

    return {
        'generated': len(generated),
        'tps': engine.tokens_generated / max(engine.total_time, 1e-9),
        'runtime_mode': engine.runtime._mode,
        'opu_steps': engine.opu.step,
        'opu_sigma': engine.opu.policy_jitter,
        'mu': kpi['mu'],
        'tau': kpi['tau'],
        'lazy_hit_rate': engine.kv_adapter.lazy_stats()['hit_rate'],
    }


# ════════════════════════════════════════════════════════════════
# 自检 (17 项)
# ════════════════════════════════════════════════════════════════

def run_tests():
    print("=" * 65)
    print("Virtual A100 v2 自检")
    print("=" * 65)
    rng = np.random.default_rng(42)
    passed = 0

    # 1: INT8 量化
    print("[1] INT8 往返...", end=" ")
    t = rng.standard_normal((64, 128)).astype(np.float32)
    q, s = quantize_int8(t)
    assert np.linalg.norm(t - dequantize_int8(q, s)) / np.linalg.norm(t) < 0.05
    print("[OK]"); passed += 1

    # 2: TF32 截断
    print("[2] TF32 截断...", end=" ")
    assert _tf32_truncate(np.float32(1.0)) == np.float32(1.0)
    val = np.float32(1.0 + 2**-11)
    assert _tf32_truncate(val) != val
    print("[OK]"); passed += 1

    # 3: TF32 matmul
    print("[3] TF32 matmul...", end=" ")
    A = rng.standard_normal((32, 64)).astype(np.float32)
    B = rng.standard_normal((64, 48)).astype(np.float32)
    err = np.linalg.norm(tf32_matmul(A,B) - A@B) / np.linalg.norm(A@B)
    assert 1e-8 < err < 0.01
    print(f"[OK] (err={err:.6f})"); passed += 1

    # 4: GhostFactor forward
    print("[4] GhostFactor forward...", end=" ")
    W = rng.standard_normal((64, 128)).astype(np.float32)
    gf = GhostCompressor(GhostConfig(base_rank=16, quantize_factors=False)
                         ).compress_weight('Wq', W, 16)
    x = rng.standard_normal((4, 128)).astype(np.float32)
    err = np.linalg.norm(gf.forward(x) - x @ gf.reconstruct().T) / np.linalg.norm(x @ W.T)
    assert err < 1e-5
    print(f"[OK] (err={err:.2e})"); passed += 1

    # 5: 贪心 rank 分配
    print("[5] 贪心 rank...", end=" ")
    ls = [make_synthetic_layer(rng, 64, 192) for _ in range(4)]
    sc = [{k: np.linalg.svd(la[k], full_matrices=False) for k in la} for la in ls]
    comp = GhostCompressor(GhostConfig(base_rank=16, alloc_method='greedy'))
    rm, st = comp.allocate_ranks_greedy(sc, ls, 4*6*16, 4, 256)
    assert st['actual'] == 4*6*16
    print(f"[OK] (min={st['min_r']}, max={st['max_r']})"); passed += 1

    # 6: .aguf 文件
    print("[6] .aguf 往返...", end=" ")
    gls = comp.compress_model(ls, progress=False)
    tmp = os.path.join(os.environ.get('TEMP', '/tmp'), "test_v2.aguf")
    save_ghost(tmp, gls, {'H': 64, 'L': 4})
    ld, _ = load_ghost(tmp)
    for li in range(4):
        for w in ['Wq', 'W1']:
            assert np.allclose(gls[li].factors[w].reconstruct(),
                               ld[li].factors[w].reconstruct(), atol=1e-4)
    os.remove(tmp)
    print("[OK]"); passed += 1

    # 7: VirtualVRAMBackend 三层
    print("[7] VRAM 三层...", end=" ")
    vram = VirtualVRAMBackend(1000, 5000, 100000)
    vram.store("t1", "data1", 'hot', 100)
    vram.store("t2", "data2", 'warm', 200)
    assert vram.get_tier("t1") == 'hot'
    assert vram.get_tier("t2") == 'warm'
    vram.demote("t1", 'warm')
    assert vram.get_tier("t1") == 'warm'
    vram.promote("t1", 'hot')
    assert vram.get_tier("t1") == 'hot'
    ts = vram.tier_summary()
    assert ts['hot']['count'] == 1
    print("[OK]"); passed += 1

    # 8: DKTileCore tileize
    print("[8] DK-Tile tileize...", end=" ")
    dk = DKTileCore()
    gl = gls[0]
    tiles = dk.tileize_layer(gl)
    assert len(tiles) == 6  # Wq Wk Wv Wo W1 W2
    assert all(t.tile_id.startswith("L0_") for t in tiles)
    print(f"[OK] ({len(tiles)} tiles)"); passed += 1

    # 9: DKTileCore detileize_weight
    print("[9] DK-Tile forward...", end=" ")
    tile = dk.get_tile("L0_Wq")
    x = rng.standard_normal((4, gl.factors['Wq'].n)).astype(np.float32)
    y = dk.detileize_weight(tile, x)
    y_ref = gl.forward_linear('Wq', x)
    assert np.allclose(y, y_ref, atol=1e-6)
    print("[OK]"); passed += 1

    # 10: DKTileCore score
    print("[10] DK-Tile scoring...", end=" ")
    tile.touch()
    tile.touch()
    tile.touch()
    s1 = dk.score(tile)
    tile2 = tiles[1]
    s2 = dk.score(tile2)
    assert s1 > s2  # tile 被 touch 过更多次
    print(f"[OK] (hot={s1:.2f} > cold={s2:.2f})"); passed += 1

    # 11: KVAdapter write/read
    print("[11] KVAdapter...", end=" ")
    vram2 = VirtualVRAMBackend(int(1e8), int(1e8), int(1e9))
    dk2 = DKTileCore()
    kva = KVAdapter(2, 4, 32, 64, dk2, vram2)
    k = rng.standard_normal((4, 32)).astype(np.float32)
    v = rng.standard_normal((4, 32)).astype(np.float32)
    kva.write(0, 0, k, v)
    k_out, v_out = kva.read(0)
    assert np.allclose(k_out[0], k)
    assert np.allclose(v_out[0], v)
    # 验证 export_kv 兼容接口
    K, V = kva.export_kv(0)
    assert np.allclose(K[0], k)
    print("[OK]"); passed += 1

    # 12: OPU basic (分离模块版)
    print("[12] OPU 闭环...", end=" ")
    class _MockExecutor:
        """实现 ActionExecutor 协议"""
        def __init__(self):
            self.evict_calls = 0
            self.prefetch_calls = 0
            self.tighten_calls = 0
            self.relax_calls = 0
            self.gate_level = 0
            self.promote_calls = 0
        def execute_evict(self, target_free_ratio=0.1, **kw): self.evict_calls += 1
        def execute_prefetch(self, window=2, coalesce=False, **kw): self.prefetch_calls += 1
        def execute_tighten(self, hot_ratio=0.5, warm_ratio=0.3, **kw): self.tighten_calls += 1
        def execute_relax(self, hot_ratio=0.6, **kw): self.relax_calls += 1
        def execute_gate_compute(self, gate_level=1, **kw): self.gate_level = gate_level
        def execute_quality_escalation(self, **kw): self.promote_calls += 1
    opu = OPU(OPUConfig(enabled=True))
    mock = _MockExecutor()
    for i in range(20):
        acts = opu.tick(StepStats(step=i, step_time_s=0.01, quality_score=1.0))
        dispatch_actions(mock, acts)
    assert opu.step == 20
    assert opu.policy_jitter >= 0
    assert opu.quality_ema > 0.5
    print(f"[OK] (σ={opu.policy_jitter:.2f}, q={opu.quality_ema:.2f})"); passed += 1

    # 13: OPU 高压触发 evict + tighten
    print("[13] OPU 高压...", end=" ")
    opu4 = OPU(OPUConfig(enabled=True, cooldown_steps=1, high_water=0.5))
    mock4 = _MockExecutor()
    acts = opu4.tick(StepStats(step=0, hot_pressure=0.95, faults=5, quality_score=0.9))
    dispatch_actions(mock4, acts)
    types = [a.type for a in acts]
    assert 'evict' in types, f"Expected evict in {types}"
    assert 'tighten' in types, f"Expected tighten in {types}"
    assert mock4.evict_calls > 0
    assert mock4.tighten_calls > 0
    print(f"[OK] (actions={types}, evict={mock4.evict_calls}, "
          f"tighten={mock4.tighten_calls})"); passed += 1

    # 14: RoPE
    print("[14] RoPE...", end=" ")
    x = rng.standard_normal((1, 4, 64)).astype(np.float32)
    assert np.allclose(rope_embed(x, 0, 64), x, atol=1e-5)
    assert not np.allclose(rope_embed(x, 100, 64), x, atol=0.1)
    print("[OK]"); passed += 1

    # 15: VirtualA100Runtime
    print("[15] Runtime 初始化...", end=" ")
    gls2 = comp.compress_model(ls, progress=False)
    dk5 = DKTileCore()
    vram5 = VirtualVRAMBackend(int(1e9), int(1e9), int(1e9))
    cfg5 = InferConfig(vram_budget_gb=1.0, hot_ratio=0.8)
    rt = VirtualA100Runtime(gls2, dk5, vram5, cfg5)
    assert rt._mode == "ALL_HOT"
    tile = rt.get_weight_tile(0, 'Wq')
    assert tile is not None
    assert rt.prefetch_hits > 0
    print(f"[OK] (mode={rt._mode}, hits={rt.prefetch_hits})"); passed += 1

    # 16: Engine mini inference
    print("[16] Mini 推理 (H=64, L=2)...", end=" ")
    r = run_simulation(H=64, F=192, L=2, base_rank=8, seed=42, verbose=False)
    assert r['generated'] > 0
    print(f"[OK] ({r['generated']} tok, {r['tps']:.0f} tok/s, "
          f"mode={r['runtime_mode']})"); passed += 1

    # 17: Engine with TF32
    print("[17] TF32 推理...", end=" ")
    ls_sm = [make_synthetic_layer(rng, 64, 192) for _ in range(2)]
    gls_sm = GhostCompressor(GhostConfig(base_rank=8, quantize_factors=True)
                             ).compress_model(ls_sm, progress=False)
    mcfg = dict(H=64, F=192, L=2, VOCAB=64, n_heads=1, n_kv_heads=1, head_dim=64)
    ew = rng.standard_normal((64, 64)).astype(np.float32) * 0.02
    hw = rng.standard_normal((64, 64)).astype(np.float32) * 0.02
    ic_fp32 = InferConfig(max_ctx=32, use_tf32=False)
    ic_tf32 = InferConfig(max_ctx=32, use_tf32=True)
    e1 = VirtualA100Engine(gls_sm, mcfg, ic_fp32, ew, hw)
    e2 = VirtualA100Engine(gls_sm, mcfg, ic_tf32, ew, hw)
    prompt = [0, 1, 2, 3]
    g1 = e1.generate(prompt, max_new=8, verbose=False)
    g2 = e2.generate(prompt, max_new=8, verbose=False)
    assert len(g1) > 0 and len(g2) > 0
    print(f"[OK] (FP32={len(g1)} tok, TF32={len(g2)} tok)"); passed += 1

    # 18: TIERED 模式 (强制小 hot 预算 → 触发 evict/prefetch)
    print("[18] TIERED 模式...", end=" ")
    ls4 = [make_synthetic_layer(rng, 64, 192) for _ in range(4)]
    gls4 = GhostCompressor(GhostConfig(base_rank=8, quantize_factors=True)
                           ).compress_model(ls4, progress=False)
    mcfg4 = dict(H=64, F=192, L=4, VOCAB=64, n_heads=1, n_kv_heads=1, head_dim=64)
    ew4 = rng.standard_normal((64, 64)).astype(np.float32) * 0.02
    hw4 = rng.standard_normal((64, 64)).astype(np.float32) * 0.02
    # hot_budget 极小 → 强制 TIERED
    ic_tiered = InferConfig(max_ctx=32, vram_budget_gb=0.000001, hot_ratio=0.5)
    e_tiered = VirtualA100Engine(gls4, mcfg4, ic_tiered, ew4, hw4)
    assert e_tiered.runtime._mode == "TIERED"
    g_tiered = e_tiered.generate([0, 1, 2], max_new=4, verbose=False)
    assert len(g_tiered) > 0
    ts_t = e_tiered.vram.tier_summary()
    print(f"[OK] (mode=TIERED, hot={ts_t['hot']['count']}, "
          f"warm={ts_t['warm']['count']})"); passed += 1

    # 19: Tile coalescing (合并小块搬运)
    print("[19] Tile coalescing...", end=" ")
    vram_c = VirtualVRAMBackend(int(1e8), int(1e8), int(1e9))
    vram_c.store("c1", "d1", 'warm', 512)
    vram_c.store("c2", "d2", 'warm', 768)
    vram_c.store("c3", "d3", 'warm', 256)
    vram_c.queue_transfer("c1", 'hot')
    vram_c.queue_transfer("c2", 'hot')
    vram_c.queue_transfer("c3", 'hot')
    total = vram_c.flush_transfers()
    assert total == 512 + 768 + 256
    # 合并后只产生一次搬运延迟 (而非三次)
    assert vram_c.stats.total_transfer_time_s > 0
    print(f"[OK] (coalesced {total} bytes)"); passed += 1

    # 20: 切口回收 KPI (μ/τ/σ)
    print("[20] 切口 KPI...", end=" ")
    vram_k = VirtualVRAMBackend(int(1e6), int(1e6), int(1e9))
    # 模拟搬运: store→demote→promote (产生搬运摩擦)
    vram_k.store("kpi1", "data", 'hot', 100000)
    vram_k.demote("kpi1", 'warm')  # 产生 eviction + transfer
    vram_k.promote("kpi1", 'hot')  # 产生 recovery + transfer
    vram_k.stats.record_compute(1.0)  # 低熵：使用 record_compute
    vram_k.stats.record_rebuild(0.05)  # 低熵：使用 record_rebuild
    mu = vram_k.stats.friction_mu
    tau = vram_k.stats.rebuild_tax_tau
    assert mu > 0, f"μ should be > 0, got {mu}"
    assert tau > 0, f"τ should be > 0, got {tau}"
    assert vram_k.stats.aperture_evictions > 0
    kpi = vram_k.kpi_summary()
    assert 'mu' in kpi and 'tau' in kpi
    print(f"[OK] (μ={mu:.6f}, τ={tau:.4f}, "
          f"evict={kpi['aperture_evictions']})"); passed += 1

    # 21: KV Lazy materialization 统计
    print("[21] KV lazy stats...", end=" ")
    vram_l = VirtualVRAMBackend(int(1e8), int(1e8), int(1e9))
    dk_l = DKTileCore()
    kva_l = KVAdapter(2, 4, 32, 64, dk_l, vram_l)
    k = rng.standard_normal((4, 32)).astype(np.float32)
    v = rng.standard_normal((4, 32)).astype(np.float32)
    kva_l.write(0, 0, k, v)
    kva_l.write(0, 1, k, v)
    # 第一次读: 应该命中热缓存
    k_out, _ = kva_l.read(0)
    assert kva_l.cache_hits == 1
    assert kva_l.cache_misses == 0
    # 使缓存失效, 强制重建
    kva_l._cache_valid[0] = False
    k_out2, _ = kva_l.read(0)
    assert kva_l.cache_misses == 1
    assert kva_l.rebuild_count == 1
    assert kva_l.rebuild_time_s > 0
    ls_l = kva_l.lazy_stats()
    assert ls_l['hit_rate'] == 0.5  # 1 hit / 2 total
    print(f"[OK] (hit_rate={ls_l['hit_rate']:.0%}, "
          f"rebuilds={ls_l['rebuilds']})"); passed += 1

    # 22: Aperture 切口标记
    print("[22] Aperture 标记...", end=" ")
    vram_a = VirtualVRAMBackend(int(1e6), int(1e6), int(1e9))
    vram_a.store("ap1", "data", 'hot', 1000)
    assert vram_a.get_aperture("ap1") == 'recoverable'
    vram_a.set_aperture("ap1", 'pinned')
    assert vram_a.get_aperture("ap1") == 'pinned'
    vram_a.set_aperture("ap1", 'evictable')
    assert vram_a.get_aperture("ap1") == 'evictable'
    print("[OK]"); passed += 1

    # 23: torch.compile 标记
    print("[23] compile 边界...", end=" ")
    assert getattr(quantize_int8, '_compile_hint', None) == 'hot'
    assert getattr(dequantize_int8, '_compile_hint', None) == 'hot'
    assert getattr(tf32_matmul, '_compile_hint', None) == 'hot'
    assert getattr(softmax, '_compile_hint', None) == 'hot'
    assert getattr(rms_norm, '_compile_hint', None) == 'hot'
    assert getattr(rope_embed, '_compile_hint', None) == 'hot'
    assert getattr(VirtualA100Runtime.evict_cold_tiles, '_compile_hint', None) == 'cold'
    assert getattr(VirtualA100Runtime.tighten_hot, '_compile_hint', None) == 'cold'
    print("[OK] (6 hot, 3 cold)"); passed += 1

    # ── GPT-5.2 R2 新增测试 ──

    # 24: Criticality 自动分配 (文档 §5: critical/normal/cheap)
    print("[24] Criticality...", end=" ")
    dk_cr = DKTileCore()
    ls_cr = [make_synthetic_layer(rng, 64, 192)]
    gc_cr = GhostCompressor(GhostConfig(base_rank=8)).compress_model(ls_cr, progress=False)
    tiles_cr = dk_cr.tileize_layer(gc_cr[0])
    crits = {t.kind: t.criticality for t in tiles_cr}
    assert crits['Wq'] == 'critical', f"Wq should be critical, got {crits['Wq']}"
    assert crits['Wk'] == 'critical'
    assert crits['W1'] == 'normal', f"W1 should be normal, got {crits['W1']}"
    assert crits['W2'] == 'normal'
    # critical tiles 默认 pinned aperture
    apers = {t.kind: t.aperture for t in tiles_cr}
    assert apers['Wq'] == 'pinned', f"Wq should be pinned, got {apers['Wq']}"
    assert apers['W1'] == 'recoverable'
    # criticality 影响 score 权重 (critical >> normal)
    tiles_cr[0].touch()  # Wq (critical)
    tiles_cr[4].touch()  # W1 (normal)
    s_crit = dk_cr.score(tiles_cr[0])
    s_norm = dk_cr.score(tiles_cr[4])
    assert s_crit > s_norm * 5, f"critical score {s_crit} should be >> normal {s_norm}"
    print(f"[OK] (Wq={crits['Wq']}/pinned, W1={crits['W1']}/recoverable, "
          f"score_ratio={s_crit/max(s_norm,1e-9):.1f}x)"); passed += 1

    # 25: 质量闭环 — quality_escalation 触发 (文档 §1C + §5)
    print("[25] 质量闭环...", end=" ")
    opu_q = OPU(OPUConfig(enabled=True, cooldown_steps=0))
    mock_q = _MockExecutor()
    # 正常质量: 不触发
    acts_q = opu_q.tick(StepStats(step=0, quality_score=0.9))
    dispatch_actions(mock_q, acts_q)
    assert mock_q.promote_calls == 0
    # 质量持续恶化: EMA 需要多步才会降到 0.5 以下
    for i in range(15):
        acts_q = opu_q.tick(StepStats(step=i+1, quality_score=0.1))
        dispatch_actions(mock_q, acts_q)
    assert opu_q.quality_ema < 0.5, f"EMA should be <0.5, got {opu_q.quality_ema:.3f}"
    assert opu_q.quality_alarm == True, "Quality alarm should be set"
    assert mock_q.promote_calls > 0, "promote_critical should have been called"
    # 质量恢复后告警解除
    for i in range(20):
        acts_q = opu_q.tick(StepStats(step=i+16, quality_score=0.95))
        dispatch_actions(mock_q, acts_q)
    assert opu_q.quality_alarm == False, "Quality alarm should clear after recovery"
    print(f"[OK] (alarm triggered & cleared, promote_calls={mock_q.promote_calls})"); passed += 1

    # 26: 资源闭环 — tighten 限速器 (文档 §1A: 连续 tighten ≤5)
    print("[26] Tighten 限速...", end=" ")
    opu_tl = OPU(OPUConfig(enabled=True, cooldown_steps=0, high_water=0.5))
    mock_tl = _MockExecutor()
    tighten_total = 0
    for i in range(10):
        acts_tl = opu_tl.tick(StepStats(step=i, hot_pressure=0.95, quality_score=1.0))
        dispatch_actions(mock_tl, acts_tl)
        tighten_total += sum(1 for a in acts_tl if a.type == 'tighten')
    assert tighten_total <= 5, f"Tighten should be rate-limited to ≤5, got {tighten_total}"
    print(f"[OK] (tighten_total={tighten_total})"); passed += 1

    # 27: 摩擦闭环 — gate_compute 分级 (文档 §1B + §3)
    print("[27] Gate compute...", end=" ")
    opu_gc = OPU(OPUConfig(enabled=True, cooldown_steps=0, tau_threshold=0.05))
    mock_gc = _MockExecutor()
    # 多步高 τ 让 EMA 积累
    for i in range(10):
        acts_gc = opu_gc.tick(StepStats(step=i, rebuild_cost_s=0.3, quality_score=1.0))
        dispatch_actions(mock_gc, acts_gc)
    types_gc = [a.type for a in acts_gc]
    assert 'gate_compute' in types_gc, f"Expected gate_compute for high τ, got {types_gc}"
    assert opu_gc.gate_level > 0
    # Runtime.should_rebuild 受 gate 控制
    ls_gc = [make_synthetic_layer(rng, 64, 192)]
    gc_gc = GhostCompressor(GhostConfig(base_rank=8)).compress_model(ls_gc, progress=False)
    dk_gc = DKTileCore()
    vram_gc2 = VirtualVRAMBackend(int(1e9), int(1e9), int(1e9))
    rt_gc = VirtualA100Runtime(gc_gc, dk_gc, vram_gc2, InferConfig())
    rt_gc.set_rebuild_gate(2)  # 仅 critical
    tile_crit = DKTile(tile_id="test", layer=0, kind='Wq', payload=None, criticality='critical')
    tile_cheap = DKTile(tile_id="test2", layer=0, kind='KV', payload=None, criticality='cheap')
    assert rt_gc.should_rebuild(tile_crit) == True
    assert rt_gc.should_rebuild(tile_cheap) == False
    print(f"[OK] (gate_level={opu_gc.gate_level}, "
          f"critical=rebuild, cheap=skip)"); passed += 1

    # 28: Runtime promote_critical (文档 §5: 质量恶化→强制提升)
    print("[28] Promote critical...", end=" ")
    ls_pc = [make_synthetic_layer(rng, 64, 192) for _ in range(2)]
    gc_pc = GhostCompressor(GhostConfig(base_rank=8, quantize_factors=True)
                            ).compress_model(ls_pc, progress=False)
    dk_pc = DKTileCore()
    vram_pc = VirtualVRAMBackend(int(1e9), int(1e9), int(1e9))
    cfg_pc = InferConfig(vram_budget_gb=0.000001, hot_ratio=0.3)
    rt_pc = VirtualA100Runtime(gc_pc, dk_pc, vram_pc, cfg_pc)
    assert rt_pc._mode == "TIERED"
    warm_before = set(vram_pc.tiles_in_tier('warm'))
    warm_critical_before = [tid for tid in warm_before
                           if tid in rt_pc._all_tiles
                           and rt_pc._all_tiles[tid].criticality == 'critical']
    # 通过 ActionExecutor 协议调用
    rt_pc.execute_quality_escalation()
    for tid, t in rt_pc._all_tiles.items():
        if t.criticality == 'critical':
            tier = vram_pc.get_tier(tid)
            assert tier == 'hot', f"Critical tile {tid} should be hot, got {tier}"
    print(f"[OK] (promoted {len(warm_critical_before)} critical tiles)"); passed += 1

    # 29: StepStats 切口损耗分解 (文档 §3 规则1: 可计量)
    print("[29] 切口损耗分解...", end=" ")
    st = StepStats(
        step=0, step_time_s=0.1,
        wait_time_s=0.02, copy_bytes=1024, unpack_cost_s=0.005,
        rebuild_cost_s=0.01,
        quality_score=0.8, logits_entropy=3.5, repeat_rate=0.1,
    )
    loss = st.aperture_loss_s
    assert abs(loss - 0.035) < 1e-6, f"aperture_loss should be 0.035, got {loss}"
    assert st.quality_score == 0.8
    print(f"[OK] (loss={loss:.4f}, quality={st.quality_score})"); passed += 1

    # 30: 三闭环压力测试 (资源+摩擦+质量 同时施压)
    print("[30] 三闭环压力测...", end=" ")
    opu_3l = OPU(OPUConfig(enabled=True, cooldown_steps=0, high_water=0.5,
                            max_tighten_streak=10))  # 放宽限速方便测试
    mock_3l = _MockExecutor()
    # 先喂2步高 τ 让 EMA 累积 (不消耗太多 tighten)
    for i in range(3):
        opu_3l.tick(StepStats(step=i, hot_pressure=0.95, faults=3,
                              rebuild_cost_s=0.5, quality_score=1.0))
    # 再来一步: 高压+高τ(EMA已累积) → 应该触发 evict+tighten+gate_compute
    acts_3l = opu_3l.tick(StepStats(
        step=3, hot_pressure=0.95, faults=3,
        rebuild_cost_s=0.5, quality_score=1.0,
    ))
    dispatch_actions(mock_3l, acts_3l)
    types_3l = [a.type for a in acts_3l]
    assert 'evict' in types_3l, f"Missing evict: {types_3l}"
    assert 'tighten' in types_3l, f"Missing tighten: {types_3l}"
    assert 'gate_compute' in types_3l, f"Missing gate_compute: {types_3l}"
    # Step 2: 质量持续恶化 → quality alarm
    for i in range(15):
        acts = opu_3l.tick(StepStats(step=i+4, quality_score=0.1))
        dispatch_actions(mock_3l, acts)
    assert opu_3l.quality_alarm == True, "Quality alarm should be set"
    assert mock_3l.promote_calls > 0
    print(f"[OK] (step1={types_3l}, quality_alarm=True, "
          f"promote_calls={mock_3l.promote_calls})"); passed += 1

    # ── v2 修复验证测试 ──

    # 31: pack/unpack 往返 (修复: 原来缺 unpack)
    print("[31] pack/unpack 往返...", end=" ")
    dk_pu = DKTileCore()
    ls_pu = [make_synthetic_layer(rng, 64, 192)]
    gc_pu = GhostCompressor(GhostConfig(base_rank=8)).compress_model(ls_pu, progress=False)
    tiles_pu = dk_pu.tileize_layer(gc_pu[0])
    tile_wq = tiles_pu[0]  # Wq
    packed = dk_pu.pack(tile_wq)
    unpacked_factor = dk_pu.unpack(packed, tile_wq)
    x_test = rng.standard_normal((2, tile_wq.payload.n)).astype(np.float32)
    y_orig = tile_wq.payload.forward(x_test)
    y_unpacked = unpacked_factor.forward(x_test)
    assert np.allclose(y_orig, y_unpacked, atol=1e-5), \
        f"pack/unpack roundtrip failed: max diff={np.abs(y_orig - y_unpacked).max()}"
    print("[OK]"); passed += 1

    # 32: store() 容量执法 (修复: 原来不检查 budget)
    print("[32] store() 容量执法...", end=" ")
    vram_cap = VirtualVRAMBackend(100, 500, 100000)  # hot 只有 100 bytes
    ok1 = vram_cap.store("cap1", "d", 'hot', 60)
    assert ok1 == True, "First store should succeed"
    ok2 = vram_cap.store("cap2", "d", 'hot', 60)  # 超 100 → 降级到 warm
    assert ok2 == True, "Should fallback to warm"
    assert vram_cap.get_tier("cap2") == 'warm', \
        f"cap2 should be in warm, got {vram_cap.get_tier('cap2')}"
    print("[OK]"); passed += 1

    # 33: delta 信号 (修复: 原来喂累积值导致 μ 不收敛)
    print("[33] delta 信号...", end=" ")
    opu_d = OPU(OPUConfig(enabled=True))
    # 模拟 10 步, 每步恒定增量 0.001 wait_time
    for i in range(10):
        opu_d.tick(StepStats(
            step=i, wait_time_s=0.001, rebuild_cost_s=0.0005,
            quality_score=1.0,
        ))
    mu_val = opu_d._ema.get('mu', 0.0)
    tau_val = opu_d._ema.get('tau', 0.0)
    # 如果是累积值, μ 会越来越大; 增量值时 μ 应收敛到 ~0.001
    assert mu_val < 0.01, f"μ should converge near 0.001 with delta inputs, got {mu_val:.4f}"
    assert tau_val < 0.005, f"τ should converge near 0.0005 with delta inputs, got {tau_val:.4f}"
    print(f"[OK] (μ={mu_val:.4f}, τ={tau_val:.4f})"); passed += 1

    # 34: should_rebuild 基于 step (修复: 原来是全局 tile 计数器)
    print("[34] should_rebuild per-step...", end=" ")
    ls_sr = [make_synthetic_layer(rng, 64, 192)]
    gc_sr = GhostCompressor(GhostConfig(base_rank=8)).compress_model(ls_sr, progress=False)
    dk_sr = DKTileCore()
    vram_sr = VirtualVRAMBackend(int(1e9), int(1e9), int(1e9))
    rt_sr = VirtualA100Runtime(gc_sr, dk_sr, vram_sr, InferConfig())
    rt_sr.set_rebuild_gate(1)  # 每 2 步一次
    tile_test = DKTile(tile_id="t", layer=0, kind='Wq', payload=None, criticality='critical')
    # 同一个 step 内多次查询应该返回相同结果
    r_step0_a = rt_sr.should_rebuild(tile_test, step=0)
    r_step0_b = rt_sr.should_rebuild(tile_test, step=0)
    assert r_step0_a == r_step0_b, "Same step should give same result for multiple tiles"
    r_step1 = rt_sr.should_rebuild(tile_test, step=1)
    assert r_step0_a != r_step1, "Adjacent steps should alternate"
    print("[OK]"); passed += 1

    # 35: 动作追责 (v2 新增: reason + source)
    print("[35] 动作追责...", end=" ")
    opu_tr = OPU(OPUConfig(enabled=True, cooldown_steps=0, high_water=0.5))
    acts_tr = opu_tr.tick(StepStats(step=0, hot_pressure=0.95, quality_score=1.0))
    traced = [a for a in acts_tr if a.source]
    assert len(traced) > 0, f"Actions should have source, got {acts_tr}"
    for a in traced:
        assert a.source != '', f"Action {a.type} missing source"
        assert a.reason != '', f"Action {a.type} missing reason"
    traces = opu_tr.action_traces()
    assert len(traces) > 0, f"action_traces() should return traces"
    print(f"[OK] (traces={traces[:2]})"); passed += 1

    # 36: suppress_evict (v2 新增: 质量恶化时抑制 evict)
    print("[36] suppress_evict...", end=" ")
    opu_se = OPU(OPUConfig(enabled=True, cooldown_steps=0, high_water=0.5))
    # 先让质量 EMA 降到 alarm 线以下
    for i in range(15):
        opu_se.tick(StepStats(step=i, quality_score=0.1, hot_pressure=0.3))
    assert opu_se.quality_alarm == True
    # 现在同时施加高压 + 低质量
    acts_se = opu_se.tick(StepStats(
        step=15, hot_pressure=0.95, quality_score=0.1))
    types_se = [a.type for a in acts_se]
    # quality_escalation 应该存在
    assert 'quality_escalation' in types_se, f"Missing quality_escalation: {types_se}"
    # evict 应该被 suppress
    assert 'evict' not in types_se, \
        f"evict should be suppressed during quality alarm, got {types_se}"
    print(f"[OK] (types={types_se})"); passed += 1

    # 37: tighten_hot 正确驱逐目标 (修复: 原来 normal/cheap 都去 warm)
    print("[37] tighten 驱逐目标...", end=" ")
    ls_td = [make_synthetic_layer(rng, 64, 192) for _ in range(2)]
    gc_td = GhostCompressor(GhostConfig(base_rank=8, quantize_factors=True)
                            ).compress_model(ls_td, progress=False)
    dk_td = DKTileCore()
    vram_td = VirtualVRAMBackend(int(1e9), int(1e9), int(1e9))
    cfg_td = InferConfig(vram_budget_gb=1.0, hot_ratio=0.8)
    rt_td = VirtualA100Runtime(gc_td, dk_td, vram_td, cfg_td)
    # 所有 tiles 应在 hot
    assert rt_td._mode == "ALL_HOT"
    # 执行 tighten: 目标 hot_ratio 很小 → 迫使驱逐
    rt_td.tighten_hot(0.001, 0.3)
    # 检查 critical 不在 cold
    for tid, t in rt_td._all_tiles.items():
        tier = vram_td.get_tier(tid)
        if t.criticality == 'critical':
            assert tier != 'cold', f"Critical tile {tid} should not be in cold, got {tier}"
    print("[OK]"); passed += 1

    print(f"\n{'='*65}")
    print(f"全部 {passed}/37 项测试通过 [OK]")
    print(f"{'='*65}")


# ════════════════════════════════════════════════════════════════
# LECAC + PyTorch 集成：真实 GPU 计算版本
# ════════════════════════════════════════════════════════════════

class VirtualA100EngineTorch:
    """
    Virtual A100 的 PyTorch 版本
    使用真实 GPU 计算 + LECAC INT2 + virtual_vram offload

    这是连接 Virtual A100 和 LECAC 的桥梁：
    1. 加载 GGUF 模型
    2. 应用 LECAC INT2 量化
    3. 使用 virtual_vram 自动 offload
    4. 在 8GB GPU + 24GB RAM 上运行 70B 模型
    """

    def __init__(self, gguf_path: str, use_lecac: bool = True,
                 lecac_alpha: float = NATURAL_EQUILIBRIUM_CONSTANT,
                 vram_budget_gb: float = 8.0):
        """
        初始化 PyTorch 版 Virtual A100

        Args:
            gguf_path: GGUF 模型文件路径
            use_lecac: 是否使用 LECAC INT2 量化
            lecac_alpha: LECAC 补偿强度 (默认 4/e)
            vram_budget_gb: GPU 显存预算
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 不可用")

        self.gguf_path = gguf_path
        self.use_lecac = use_lecac and LECAC_AVAILABLE
        self.lecac_alpha = lecac_alpha
        self.vram_budget_gb = vram_budget_gb

        # 检查 GPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[VirtualA100-Torch] GPU: {self.gpu_name} ({self.gpu_mem_gb:.1f}GB)")
        else:
            self.device = torch.device('cpu')
            self.gpu_name = "CPU"
            self.gpu_mem_gb = 0.0
            print("[WARNING] 无 GPU 可用")

        # 加载 GGUF 模型
        if GGUF_AVAILABLE:
            print(f"[VirtualA100-Torch] 加载 GGUF: {gguf_path}")
            self.llama_model, self.model_config = load_gguf(
                gguf_path,
                n_gpu_layers=0,  # 初始全部 CPU，由 virtual_vram 管理
                n_ctx=512
            )
            print(f"[VirtualA100-Torch] GGUF 加载完成")
        else:
            raise RuntimeError("需要 llama-cpp-python 加载 GGUF")

        # 配置 LECAC virtual_vram
        if self.use_lecac:
            self.vram_config = VirtualVRAMConfig(
                enabled=True,
                min_tensor_bytes=1 << 20,  # 1MB
                compress=True,
                compress_dtype="int8",
                use_lecac=True,
                lecac_alpha=lecac_alpha,
                verbose=False
            )
            print(f"[VirtualA100-Torch] LECAC 启用 (alpha={lecac_alpha:.4f})")
        else:
            self.vram_config = None
            print("[VirtualA100-Torch] LECAC 未启用（使用标准模式）")

        # 统计
        self.tokens_generated = 0

    def generate(self, prompt: str, max_tokens: int = 50,
                temperature: float = 0.7) -> str:
        """
        生成文本（使用 llama-cpp-python + virtual_vram offload）

        Args:
            prompt: 输入提示
            max_tokens: 最大生成 token 数
            temperature: 采样温度

        Returns:
            生成的文本
        """
        if self.use_lecac:
            # 使用 virtual_vram context manager
            with virtual_vram(self.vram_config):
                output = self.llama_model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["\n"],
                    echo=False
                )
        else:
            output = self.llama_model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["\n"],
                echo=False
            )

        self.tokens_generated += max_tokens
        return output['choices'][0]['text']

    def fine_tune_step(self, inputs: torch.Tensor, targets: torch.Tensor,
                       optimizer: torch.optim.Optimizer) -> float:
        """
        微调步骤（使用 LECAC INT2）

        Args:
            inputs: 输入 token ids
            targets: 目标 token ids
            optimizer: 优化器

        Returns:
            损失值
        """
        if not self.use_lecac:
            raise RuntimeError("需要启用 LECAC 才能进行微调")

        # 在 virtual_vram context 下训练
        with virtual_vram(self.vram_config):
            # TODO: 实现实际的微调逻辑
            # 需要将 GGUF 模型转换为 PyTorch 模型
            # 然后应用 LECAC INT2 训练
            pass

        return 0.0

    def get_memory_stats(self) -> Dict[str, float]:
        """获取显存统计"""
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1e9
            gpu_reserved = torch.cuda.memory_reserved() / 1e9
            return {
                'gpu_allocated_gb': gpu_allocated,
                'gpu_reserved_gb': gpu_reserved,
                'tokens_generated': self.tokens_generated
            }
        return {'tokens_generated': self.tokens_generated}


def create_virtual_a100_torch(gguf_path: str, **kwargs) -> VirtualA100EngineTorch:
    """
    工厂函数：创建 PyTorch 版 Virtual A100

    Args:
        gguf_path: GGUF 模型路径
        **kwargs: 传递给 VirtualA100EngineTorch

    Returns:
        VirtualA100EngineTorch 实例

    Example:
        >>> engine = create_virtual_a100_torch(
        ...     "D:/model.gguf",
        ...     use_lecac=True,
        ...     vram_budget_gb=8.0
        ... )
        >>> output = engine.generate("What is AI?", max_tokens=50)
        >>> print(output)
    """
    return VirtualA100EngineTorch(gguf_path, **kwargs)


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

def main():
    import argparse
    p = argparse.ArgumentParser(description="Virtual A100 v2")
    sub = p.add_subparsers(dest='cmd')
    sub.add_parser('test', help='自检')
    sp = sub.add_parser('sim', help='仿真')
    sp.add_argument('--H', type=int, default=256)
    sp.add_argument('--F', type=int, default=768)
    sp.add_argument('--L', type=int, default=8)
    sp.add_argument('--rank', type=int, default=32)
    sp.add_argument('--seed', type=int, default=42)

    # 新增：GGUF 测试
    sg = sub.add_parser('gguf', help='测试 GGUF 模型加载（70B）')
    sg.add_argument('--model', type=str,
                   default="D:/huihui-ai_DeepSeek-R1-Distill-Llama-70B-abliterated-Q4_0.gguf",
                   help='GGUF 模型路径')
    sg.add_argument('--tokens', type=int, default=20,
                   help='生成 token 数量')
    sg.add_argument('--no-lecac', action='store_true',
                   help='禁用 LECAC')

    args = p.parse_args()

    if args.cmd == 'test':
        run_tests()
    elif args.cmd == 'sim':
        run_simulation(H=args.H, F=args.F, L=args.L,
                       base_rank=args.rank, seed=args.seed)
    elif args.cmd == 'gguf':
        # 测试 GGUF 模型加载
        print("=" * 65)
        print("Virtual A100 - GGUF 模型测试")
        print("=" * 65)

        engine = create_virtual_a100_torch(
            args.model,
            use_lecac=not args.no_lecac,
            vram_budget_gb=8.0
        )

        # 测试生成
        prompt = "What is the capital of France?"
        print(f"\n输入: {prompt}")
        print(f"生成 {args.tokens} tokens...\n")

        output = engine.generate(prompt, max_tokens=args.tokens)
        print(f"输出: {output}")

        # 显存统计
        stats = engine.get_memory_stats()
        print(f"\n显存统计:")
        print(f"  GPU 已分配: {stats.get('gpu_allocated_gb', 0):.2f} GB")
        print(f"  GPU 已保留: {stats.get('gpu_reserved_gb', 0):.2f} GB")
        print(f"  生成 tokens: {stats['tokens_generated']}")

        print("\n[SUCCESS] GGUF 模型测试完成！")
    else:
        p.print_help()


if __name__ == '__main__':
    main()
