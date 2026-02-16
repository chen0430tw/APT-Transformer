#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Virtual A100 配置 - RTX 3070 Laptop 8GB "70B 专用版"
======================================================

硬件现实：
  - GPU: RTX 3070 Laptop 8GB VRAM
  - RAM: 25GB 系统内存
  - PCIe: 3.0 x8 (笔记本缩水版)

目标：在 3070 上跑 70B 模型
  - Ghost 低秩压缩 (rank=16) → 70B INT8 压到 ~20GB
  - 80 层三层存储：4层热 + 24层温 + 52层冷
  - 流水线预取 + 激进 offload

内存计算：
  70B FP16 原始: ~140 GB
  Ghost INT8 压缩 (rank=16): ~20 GB
    - Hot (GPU 4层):   ~1 GB
    - Warm (CPU 24层):  ~6 GB (pinned)
    - Cold (CPU 52层):  ~13 GB (mmap)

作者：GPT-5.2 R2 + 麦当劳商学院
版本：2.0.0-3070-70B
"""

import sys
import io

# Windows 控制台编码修复
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from dataclasses import dataclass
from typing import Optional


@dataclass
class McDonald3070_70BConfig:
    """
    RTX 3070 专用配置 — 70B 模型版

    核心策略：
      1. Ghost 低秩压缩 → 70B 从 140GB 压到 ~20GB
      2. 三层存储 → 4层热 + 24层温 + 52层冷
      3. 激进 offload → 每步只保持 4 层在 GPU
      4. 流水线预取 → 隐藏 PCIe 传输延迟
    """

    # 硬件预算
    vram_budget_gb: float = 6.0     # GPU: 6GB 给权重，2GB 给 KV/buffer
    cpu_budget_gb: float = 22.0      # CPU: 22GB 给模型，3GB 给系统

    # 70B 模型 80 层分配
    hot_layers: int = 4              # GPU 常驻 4 层 (~1GB)
    warm_layers: int = 24            # CPU pinned 24 层 (~6GB)
    # 剩余 52 层放 cold 层按需 mmap (~13GB)

    # 冷热比例
    hot_ratio: float = 0.05          # 热层 5%
    warm_ratio: float = 0.30         # 温层 30%

    # OPU 店长看板
    opu_enabled: bool = True
    opu_high_water: float = 0.70     # 70% 就开始清理
    opu_low_water: float = 0.50      # 50% 停止清理
    opu_cooldown: int = 2            # 2 步冷却（频繁调度）
    mu_threshold: float = 0.08       # 80ms 就预取
    tau_threshold: float = 0.12      # 120ms 就门控
    prefetch_window: int = 3         # 预取未来 3 层
    quality_alarm_threshold: float = 0.45
    quality_recover_threshold: float = 0.65

    # 压缩策略
    compress_dtype: str = "int8"     # INT8 4x 压缩
    min_tensor_bytes: int = 256 * 1024  # 256KB 以上就压缩
    kv_quant_bits: int = 4           # INT4 KV cache

    # Ghost 低秩压缩
    ghost_base_rank: int = 16        # 基础秩 16
    ghost_min_rank: int = 4          # 最小秩 4
    ghost_max_rank: int = 32         # 最大秩 32（关键层）
    ghost_alloc_method: str = 'greedy'
    ghost_quantize_factors: bool = True   # 因子 INT8
    ghost_sparse_density: float = 0.0

    # 模型结构
    model_layers: int = 80           # LLaMA 70B 有 80 层
    model_hidden: int = 8192
    model_ffn: int = 28672

    # 推理配置
    window_batch_size: int = 1
    window_max_ctx: int = 4096
    window_prefetch: bool = True     # 必须开启
    window_aggressive_offload: bool = True

    # 生成参数
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    use_tf32: bool = False

    # 调试
    verbose: bool = True
    print_stats_every: int = 16
    health_interval: int = 32

    @classmethod
    def for_chat(cls) -> 'McDonald3070_70BConfig':
        """对话模式：低延迟优先"""
        cfg = cls()
        cfg.window_max_ctx = 2048
        cfg.prefetch_window = 2
        cfg.hot_layers = 6
        cfg.warm_layers = 30
        cfg.opu_high_water = 0.65
        return cfg

    @classmethod
    def for_long_context(cls) -> 'McDonald3070_70BConfig':
        """长上下文模式：8K 上下文"""
        cfg = cls()
        cfg.window_max_ctx = 8192
        cfg.kv_quant_bits = 4
        cfg.hot_layers = 3
        cfg.warm_layers = 20
        cfg.prefetch_window = 4
        return cfg

    @classmethod
    def for_quality(cls) -> 'McDonald3070_70BConfig':
        """质量优先：提高 Ghost rank"""
        cfg = cls()
        cfg.ghost_base_rank = 24
        cfg.ghost_min_rank = 8
        cfg.ghost_max_rank = 48
        cfg.hot_layers = 5
        cfg.warm_layers = 28
        cfg.opu_high_water = 0.75
        return cfg

    def to_opu_config(self):
        """转换为 OPUConfig"""
        from opu.config import OPUConfig
        return OPUConfig(
            enabled=self.opu_enabled,
            ema_alpha=0.15,
            high_water=self.opu_high_water,
            low_water=self.opu_low_water,
            cooldown_steps=self.opu_cooldown,
            prefetch_window=self.prefetch_window,
            mu_threshold=self.mu_threshold,
            tau_threshold=self.tau_threshold,
            quality_alarm_threshold=self.quality_alarm_threshold,
            quality_recover_threshold=self.quality_recover_threshold,
            hot_ratio=self.hot_ratio,
            warm_ratio=self.warm_ratio,
            health_interval=self.health_interval,
        )

    def to_infer_config(self):
        """转换为 InferConfig"""
        from virtual_a100 import InferConfig
        return InferConfig(
            max_ctx=self.window_max_ctx,
            kv_quant_bits=self.kv_quant_bits,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            use_tf32=self.use_tf32,
            vram_budget_gb=self.vram_budget_gb,
            cpu_budget_gb=self.cpu_budget_gb,
            hot_ratio=self.hot_ratio,
            warm_ratio=self.warm_ratio,
            opu_enabled=self.opu_enabled,
            opu_cooldown=self.opu_cooldown,
            opu_high_water=self.opu_high_water,
            opu_low_water=self.opu_low_water,
            prefetch_window=self.prefetch_window,
        )

    def to_vvram_config(self):
        """转换为 VirtualVRAMConfig"""
        from virtual_vram import VirtualVRAMConfig
        return VirtualVRAMConfig(
            enabled=True,
            min_tensor_bytes=self.min_tensor_bytes,
            compress=True,
            compress_dtype=self.compress_dtype,
            stream_prefetch=self.window_prefetch,
            track_dependencies=True,
            verbose=self.verbose,
        )

    def to_ghost_config(self):
        """转换为 GhostConfig"""
        from virtual_a100 import GhostConfig
        return GhostConfig(
            base_rank=self.ghost_base_rank,
            min_rank=self.ghost_min_rank,
            max_rank=self.ghost_max_rank,
            alloc_method=self.ghost_alloc_method,
            quantize_factors=self.ghost_quantize_factors,
            sparse_density=self.ghost_sparse_density,
        )


# 预设配置
DEFAULT_70B = McDonald3070_70BConfig()
CHAT_70B = McDonald3070_70BConfig.for_chat()
LONG_CTX_70B = McDonald3070_70BConfig.for_long_context()
QUALITY_70B = McDonald3070_70BConfig.for_quality()


def estimate_70b_memory(cfg: McDonald3070_70BConfig) -> dict:
    """
    估算 70B 模型的内存占用

    Returns:
        dict: {
            'ghost_compressed_gb': float,  # Ghost INT8 压缩后
            'hot_gb': float,               # 热层占用
            'warm_gb': float,              # 温层占用
            'cold_gb': float,              # 冷层占用
            'total_gb': float,             # 总占用
        }
    """
    # 单层大小（Ghost INT8 压缩后）
    # Ghost rank=16: (8192*16 + 28672*16) * 1 * 6 ≈ 3.5 MB per layer
    bytes_per_layer = (
        (cfg.model_hidden + cfg.model_ffn) * cfg.ghost_base_rank * 2 * 6
    ) * 1  # INT8

    total_gb = (bytes_per_layer * cfg.model_layers) / (1024**3)
    hot_gb = (bytes_per_layer * cfg.hot_layers) / (1024**3)
    warm_gb = (bytes_per_layer * cfg.warm_layers) / (1024**3)
    cold_gb = total_gb - hot_gb - warm_gb

    return {
        'ghost_compressed_gb': total_gb,
        'hot_gb': hot_gb,
        'warm_gb': warm_gb,
        'cold_gb': cold_gb,
        'total_gb': total_gb,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Virtual A100 - RTX 3070 Laptop 70B 配置")
    print("=" * 70)
    print()

    cfg = DEFAULT_70B
    mem = estimate_70b_memory(cfg)

    print("硬件：")
    print(f"  GPU (VRAM):     {cfg.vram_budget_gb} GB")
    print(f"  CPU (RAM):      {cfg.cpu_budget_gb} GB")
    print()

    print("70B 模型内存占用（Ghost INT8 rank={}）：".format(cfg.ghost_base_rank))
    print(f"  压缩后总大小:    {mem['ghost_compressed_gb']:.2f} GB")
    print(f"  Hot (GPU):      {mem['hot_gb']:.2f} GB  ({cfg.hot_layers} 层)")
    print(f"  Warm (Pinned):   {mem['warm_gb']:.2f} GB  ({cfg.warm_layers} 层)")
    print(f"  Cold (Mmap):     {mem['cold_gb']:.2f} GB  ({cfg.model_layers - cfg.hot_layers - cfg.warm_layers} 层)")
    print()

    print("OPU 店长看板：")
    print(f"  资源高水位:     {cfg.opu_high_water*100:.0f}%")
    print(f"  资源低水位:     {cfg.opu_low_water*100:.0f}%")
    print(f"  预取窗口:       {cfg.prefetch_window} 层")
    print()

    print("=" * 70)
    print("预设配置：")
    print("  - DEFAULT_70B:   平衡模式")
    print("  - CHAT_70B:      对话优先 (低延迟)")
    print("  - LONG_CTX_70B:  长上下文 (8K)")
    print("  - QUALITY_70B:   质量优先 (高 rank)")
    print("=" * 70)
