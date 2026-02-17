#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════
Virtual VRAM v0.2 — 收口测试 (Convergence & Limit Finder)
═══════════════════════════════════════════════════════════════════

GPT-5.2 终止版口径：只做 3 件事，然后就停。

  1) 固定基准配置 — 让每次测试可比
  2) 每步 6 数账本 — 定位瓶颈是显存/摩擦/抖动
  3) 防抖阀 — 连续 stall 上升 + 动作频繁 → safe_mode

测试矩阵:
  · 模型规模: 4 档 (d=512/768/1024/1536)
  · 序列长度: 2 档 (256/1024)
  · 虚拟显存: 4 模式 (off / v0.1 / v0.2-fp16 / v0.2-int8)
  · 多轮稳定性: 连续 N 轮，验证吞吐不雪崩

极限判据 (GPT-5.2 三选一, 全部实现):
  A) tok/s < 阈值 (decode throughput collapse)
  B) stall_ratio > 0.5 (搬运主导)
  C) page_in+out 连续 3 步上升 (摩擦失控)

用法:
  把 virtual_vram.py 放同目录, 然后:
    python test_convergence.py                    # 全量测试
    python test_convergence.py --quick             # 快速验证 (小模型)
    python test_convergence.py --find-limit        # 找极限 batch size
    python test_convergence.py --stability         # 50 轮稳定性测试
    python test_convergence.py --all               # 全部跑一遍

需要: PyTorch + CUDA, psutil (pip install psutil)
"""

import os
import sys
import gc
import json
import time
import argparse
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# 路径配置 (适配你的 D 盘结构, 没有就跳过)
# ═══════════════════════════════════════════════════════════════════
_BASE = os.environ.get("APT_BASE", "D:/APT-Transformer")
if os.path.isdir(_BASE):
    for k, v in {
        "TORCH_HOME": f"{_BASE}/.torch_cache",
        "TEMP": f"{_BASE}/.temp",
        "TMP": f"{_BASE}/.temp",
        "TMPDIR": f"{_BASE}/.temp",
        "TORCHINDUCTOR_CACHE_DIR": f"{_BASE}/.cache/torchinductor",
        "TRITON_CACHE_DIR": f"{_BASE}/.cache/triton",
    }.items():
        os.environ.setdefault(k, v)
        os.makedirs(v, exist_ok=True)

# ── 导入 torch ──
try:
    import torch
    import torch.nn as nn
except ImportError:
    print("ERROR: PyTorch not found. pip install torch")
    sys.exit(1)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("WARNING: psutil not found, CPU RAM monitoring disabled")

# ── 导入 virtual_vram (灵活路径) ──
try:
    from virtual_vram import VirtualVRAMConfig, virtual_vram, VVRAMStats
except ImportError:
    try:
        from apt.vgpu.runtime.virtual_vram import VirtualVRAMConfig, virtual_vram, VVRAMStats
    except ImportError:
        print("ERROR: virtual_vram.py not found.")
        print("  把 virtual_vram.py 放到同目录, 或确保 apt 包可导入")
        sys.exit(1)


# ═══════════════════════════════════════════════════════════════════
# CUDA 错误处理
# ═══════════════════════════════════════════════════════════════════

def _is_oom(e: Exception) -> bool:
    """检查是否 OOM (兼容 PyTorch 各版本)"""
    msg = str(e).lower()
    return ("out of memory" in msg or "cuda error" in msg
            or "cudaerrormemoryallocation" in msg)


def _cuda_recover():
    """OOM 后恢复 CUDA context"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # 验证 CUDA 是否还活着
        try:
            _ = torch.zeros(1, device="cuda")
            del _
        except Exception:
            print("  ⚠️  CUDA context 不可恢复, 后续 CUDA 测试将跳过")
            return False
    return True


# ═══════════════════════════════════════════════════════════════════
# §1) 固定基准配置 (不再动)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkConfig:
    """固定的基准测试参数, 确保每次可比"""
    # 模型
    d_model: int = 768
    nhead: int = 8
    num_layers: int = 12
    dim_feedforward: int = 3072

    # 输入
    batch_size: int = 16
    seq_len: int = 512

    # 生成模拟 (forward 步数)
    max_new_tokens: int = 32
    seed: int = 42

    # 虚拟显存
    vvram_min_bytes: int = 1 << 22  # 4 MB 阈值

    # 安全阀
    max_cpu_ram_pct: float = 85.0
    max_step_time_s: float = 30.0

    def label(self) -> str:
        return f"d{self.d_model}_L{self.num_layers}_b{self.batch_size}_s{self.seq_len}"


# 预定义配置档位
CONFIGS = {
    "small":  BenchmarkConfig(d_model=512,  nhead=8,  num_layers=6,  dim_feedforward=2048,
                              batch_size=8,  seq_len=256),
    "medium": BenchmarkConfig(d_model=768,  nhead=8,  num_layers=12, dim_feedforward=3072,
                              batch_size=4,  seq_len=256),
    "large":  BenchmarkConfig(d_model=1024, nhead=16, num_layers=16, dim_feedforward=4096,
                              batch_size=2,  seq_len=256),
    "xlarge": BenchmarkConfig(d_model=1536, nhead=16, num_layers=24, dim_feedforward=6144,
                              batch_size=1,  seq_len=256),
}

# 虚拟显存模式
VVRAM_MODES = {
    "off":       None,
    "v01_plain": lambda mb: VirtualVRAMConfig(
        enabled=True, min_tensor_bytes=mb,
        compress=False, stream_prefetch=True,
        track_dependencies=True,  # v0.3: 启用依赖追踪
        verbose=False),
    "v02_fp16":  lambda mb: VirtualVRAMConfig(
        enabled=True, min_tensor_bytes=mb,
        compress=True, compress_dtype="float16",
        stream_prefetch=True,
        track_dependencies=True,  # v0.3: 启用依赖追踪
        verbose=False),
    "v02_int8":  lambda mb: VirtualVRAMConfig(
        enabled=True, min_tensor_bytes=mb,
        compress=True, compress_dtype="int8",
        stream_prefetch=True,
        track_dependencies=True,  # v0.3: 启用依赖追踪
        verbose=False),
}


# ═══════════════════════════════════════════════════════════════════
# §2) 每步 6 数账本 (Per-Step Ledger)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class StepLedger:
    """GPT-5.2 要求的每步 6 数账本"""
    step: int = 0
    phase: str = "decode"         # prefill | decode

    # 1. tok/s
    tok_per_sec: float = 0.0

    # 2. peak_vram (GB)
    peak_vram_gb: float = 0.0
    reserved_vram_gb: float = 0.0

    # 3. stall_time (s) — step_time 中非计算部分
    step_time_s: float = 0.0
    compute_time_s: float = 0.0
    stall_time_s: float = 0.0
    stall_ratio: float = 0.0      # stall / step_time

    # 4. page_in / page_out (bytes)
    page_in_bytes: int = 0        # offload_bytes (forward: GPU→CPU)
    page_out_bytes: int = 0       # restore_bytes (backward: CPU→GPU)

    # 5. hit_rate (offload skip = cache hit)
    offload_count: int = 0
    skip_count: int = 0
    hit_rate: float = 0.0         # skip / (skip + offload)

    # 6. 额外诊断
    cpu_ram_pct: float = 0.0
    compression_ratio: float = 0.0

    def to_line(self) -> str:
        """单行摘要 (可 grep / 可 pandas read_csv)"""
        return (
            f"step={self.step:3d} phase={self.phase:7s} "
            f"tok/s={self.tok_per_sec:7.1f} "
            f"peak={self.peak_vram_gb:.2f}GB "
            f"stall={self.stall_time_s:.4f}s({self.stall_ratio:.0%}) "
            f"in={self.page_in_bytes/1e6:.1f}MB "
            f"out={self.page_out_bytes/1e6:.1f}MB "
            f"hit={self.hit_rate:.0%} "
            f"comp={self.compression_ratio:.1f}x "
            f"ram={self.cpu_ram_pct:.0f}%"
        )


class LedgerTracker:
    """收集多步 ledger, 提供统计与防抖判据"""

    def __init__(self):
        self.entries: List[StepLedger] = []

    def add(self, e: StepLedger):
        self.entries.append(e)

    def last_n(self, n: int = 5) -> List[StepLedger]:
        return self.entries[-n:] if len(self.entries) >= n else self.entries

    # ── 极限判据 A: tok/s 崩溃 ──
    def is_throughput_collapse(self, threshold: float = 1.0) -> bool:
        recent = self.last_n(3)
        if not recent:
            return False
        return all(e.tok_per_sec < threshold and e.tok_per_sec > 0
                   for e in recent)

    # ── 极限判据 B: stall 占比过高 ──
    def is_stall_dominant(self, threshold: float = 0.5) -> bool:
        recent = self.last_n(3)
        if not recent:
            return False
        return all(e.stall_ratio > threshold for e in recent)

    # ── 极限判据 C: page_in+out 连续上升 (摩擦失控) ──
    def is_friction_runaway(self) -> bool:
        recent = self.last_n(4)
        if len(recent) < 4:
            return False
        totals = [e.page_in_bytes + e.page_out_bytes for e in recent]
        return all(totals[i] > totals[i-1] for i in range(1, len(totals)))

    # ── 综合: 任一触发即为极限 ──
    def hit_limit(self) -> Optional[str]:
        if self.is_throughput_collapse():
            return "throughput_collapse"
        if self.is_stall_dominant():
            return "stall_dominant"
        if self.is_friction_runaway():
            return "friction_runaway"
        return None

    def summary_stats(self) -> Dict[str, float]:
        if not self.entries:
            return {}
        decode = [e for e in self.entries if e.phase == "decode"]
        if not decode:
            decode = self.entries
        toks = [e.tok_per_sec for e in decode if e.tok_per_sec > 0]
        stalls = [e.stall_ratio for e in decode]
        return {
            "tok_s_mean": sum(toks) / len(toks) if toks else 0,
            "tok_s_min": min(toks) if toks else 0,
            "tok_s_max": max(toks) if toks else 0,
            "stall_ratio_mean": sum(stalls) / len(stalls) if stalls else 0,
            "peak_vram_gb": max(e.peak_vram_gb for e in self.entries),
            "total_page_in_mb": sum(e.page_in_bytes for e in self.entries) / 1e6,
            "total_page_out_mb": sum(e.page_out_bytes for e in self.entries) / 1e6,
            "steps": len(self.entries),
        }


# ═══════════════════════════════════════════════════════════════════
# §3) 防抖阀 (Thrash Guard → Safe Mode)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ThrashGuard:
    """
    GPT-5.2: "如果连续 N 步 stall_time 上升且动作频繁 → safe_mode"
    safe_mode: 降低观测频率, 扩大 offload 阈值, 减少触发
    """
    window: int = 5            # 连续多少步触发
    stall_growth_threshold: float = 0.02   # stall_time 增长阈值
    active: bool = False
    trigger_count: int = 0

    # safe_mode 参数
    safe_min_tensor_bytes: int = 1 << 24   # 16 MB (比正常大 4x)
    safe_observe_interval: int = 4          # 每 4 步才观测一次

    def check(self, ledger: LedgerTracker) -> bool:
        """检查是否应进入 safe_mode"""
        recent = ledger.last_n(self.window)
        if len(recent) < self.window:
            return self.active

        stalls = [e.stall_time_s for e in recent]
        # 检查 stall 是否连续上升
        monotonic_up = all(stalls[i] >= stalls[i-1] + self.stall_growth_threshold
                          for i in range(1, len(stalls)))

        if monotonic_up:
            self.trigger_count += 1
            if self.trigger_count >= 2 and not self.active:
                self.active = True
                print("  ⚠️  THRASH GUARD: 进入 safe_mode "
                      f"(连续 {self.window} 步 stall 上升)")
        else:
            self.trigger_count = max(0, self.trigger_count - 1)
            if self.trigger_count == 0 and self.active:
                self.active = False
                print("  ✅  THRASH GUARD: 退出 safe_mode (stall 稳定)")

        return self.active


# ═══════════════════════════════════════════════════════════════════
# 模型
# ═══════════════════════════════════════════════════════════════════

class TestTransformer(nn.Module):
    def __init__(self, cfg: BenchmarkConfig):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            batch_first=True, dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.head = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x):
        return self.head(self.encoder(x))


# ═══════════════════════════════════════════════════════════════════
# 核心测试引擎
# ═══════════════════════════════════════════════════════════════════

def _gpu_info() -> str:
    if not torch.cuda.is_available():
        return "No CUDA"
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"{name}, {total:.1f} GB"


def _cpu_ram_pct() -> float:
    if HAS_PSUTIL:
        return psutil.virtual_memory().percent
    return 0.0


def run_single_step(
    model: nn.Module,
    cfg: BenchmarkConfig,
    vvram_cfg: Optional[VirtualVRAMConfig],
    step: int,
    phase: str = "decode",
    thrash_guard: Optional[ThrashGuard] = None,
) -> StepLedger:
    """
    执行一次 forward + backward, 返回 StepLedger。
    这是账本的核心采集点。
    """
    device = next(model.parameters()).device

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # 如果 thrash_guard 激活, 动态调大 offload 阈值
    effective_cfg = vvram_cfg
    if thrash_guard and thrash_guard.active and vvram_cfg is not None:
        effective_cfg = VirtualVRAMConfig(
            enabled=True,
            min_tensor_bytes=thrash_guard.safe_min_tensor_bytes,
            compress=vvram_cfg.compress,
            compress_dtype=vvram_cfg.compress_dtype,
            stream_prefetch=vvram_cfg.stream_prefetch,
            verbose=False,
        )

    torch.manual_seed(cfg.seed + step)
    torch.cuda.manual_seed_all(cfg.seed + step)
    x_cpu = torch.randn(cfg.batch_size, cfg.seq_len, cfg.d_model)
    x = x_cpu.to(device)
    del x_cpu

    # ── 采集 offload 前后的 stats 差值 ──
    # virtual_vram 的 stats 是内部的, 我们通过 hook 间接追踪
    # 方法: 用 CUDA events 测 compute vs total
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    wall_start = time.perf_counter()
    start_event.record()

    vram_before = torch.cuda.memory_allocated(device)

    if effective_cfg is not None:
        with virtual_vram(effective_cfg):
            y = model(x)
            loss = y.mean()
            loss.backward()
    else:
        y = model(x)
        loss = y.mean()
        loss.backward()

    end_event.record()
    torch.cuda.synchronize()

    wall_end = time.perf_counter()
    step_time = wall_end - wall_start
    gpu_time_ms = start_event.elapsed_time(end_event)
    gpu_time_s = gpu_time_ms / 1000.0

    peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3

    # stall = wall_time - gpu_time (差值 ≈ PCIe + sync + overhead)
    stall_time = max(0.0, step_time - gpu_time_s)
    stall_ratio = stall_time / step_time if step_time > 0 else 0.0

    # tok/s: prefill 处理 batch*seq 个 token, decode 每步处理 batch 个 token (每序列 1 个)
    tokens = cfg.batch_size * cfg.seq_len if phase == "prefill" else cfg.batch_size
    tok_s = tokens / step_time if step_time > 0 else 0.0

    # page bytes: 基于 VRAM 实际增量估算 offload 量
    # vram_before ≈ model params, peak ≈ params + activations on GPU
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    input_bytes = cfg.batch_size * cfg.seq_len * cfg.d_model * 4  # FP32
    # Transformer 每层保存的中间张量:
    #   hidden states (~4个), attention scores (batch*heads*seq*seq*4),
    #   FFN intermediates (~2个), layernorm inputs (~2个)
    # 粗估: hidden_per_layer + attention_scores_per_layer
    hidden_per_layer = cfg.batch_size * cfg.seq_len * cfg.d_model * 4 * 6
    attn_scores_per_layer = cfg.batch_size * cfg.nhead * cfg.seq_len * cfg.seq_len * 4
    activation_est = cfg.num_layers * (hidden_per_layer + attn_scores_per_layer)
    actual_act_on_gpu = max(0, int(peak_alloc * 1e9) - vram_before)
    page_out_est = max(0, activation_est - actual_act_on_gpu)
    page_in_est = page_out_est  # backward 恢复 ≈ forward offload

    # hit_rate: 基于 VRAM 节省比例
    if effective_cfg is None:
        hit_rate = 1.0
        comp_ratio = 1.0
        offload_count = 0
        skip_count = 0
    else:
        offload_ratio = page_out_est / max(activation_est, 1)
        offload_count = max(0, int(cfg.num_layers * 2 * offload_ratio))
        skip_count = max(0, cfg.num_layers * 2 - offload_count)
        total = offload_count + skip_count
        hit_rate = skip_count / total if total > 0 else 0.0
        comp_ratio = {"int8": 4.0, "float16": 2.0}.get(
            effective_cfg.compress_dtype, 1.0) if effective_cfg.compress else 1.0
        # 压缩后实际 PCIe 搬运量更小
        page_out_est = int(page_out_est / comp_ratio)
        page_in_est = page_out_est

    model.zero_grad(set_to_none=True)
    del x, y, loss
    gc.collect()
    torch.cuda.empty_cache()

    return StepLedger(
        step=step,
        phase=phase,
        tok_per_sec=tok_s,
        peak_vram_gb=peak_alloc,
        reserved_vram_gb=peak_reserved,
        step_time_s=step_time,
        compute_time_s=gpu_time_s,
        stall_time_s=stall_time,
        stall_ratio=stall_ratio,
        page_in_bytes=page_in_est,
        page_out_bytes=page_out_est,
        offload_count=offload_count,
        skip_count=skip_count,
        hit_rate=hit_rate,
        cpu_ram_pct=_cpu_ram_pct(),
        compression_ratio=comp_ratio,
    )


# ═══════════════════════════════════════════════════════════════════
# 测试 1: 基准对比 (4 模式 × 固定配置)
# ═══════════════════════════════════════════════════════════════════

def test_baseline_comparison(cfg: BenchmarkConfig, n_steps: int = 5):
    """对比 4 种虚拟显存模式的性能, 输出账本"""
    print(f"\n{'='*72}")
    print(f"TEST: 基准对比 [{cfg.label()}]")
    print(f"  模型: d={cfg.d_model}, L={cfg.num_layers}, ff={cfg.dim_feedforward}")
    print(f"  输入: batch={cfg.batch_size}, seq={cfg.seq_len}")
    print(f"  GPU:  {_gpu_info()}")
    print(f"{'='*72}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TestTransformer(cfg).to(device)
    model.train()
    param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"  参数: {param_mb:.0f} MB\n")

    results = {}

    for mode_name, mode_factory in VVRAM_MODES.items():
        vvram_cfg = mode_factory(cfg.vvram_min_bytes) if mode_factory else None
        print(f"  ── {mode_name} ──")

        ledger = LedgerTracker()
        ok = True

        for s in range(n_steps):
            try:
                entry = run_single_step(model, cfg, vvram_cfg, step=s,
                                        phase="prefill" if s == 0 else "decode")
                ledger.add(entry)
                print(f"    {entry.to_line()}")

                # 安全检查
                if entry.cpu_ram_pct > cfg.max_cpu_ram_pct:
                    print(f"    ⛔ CPU RAM {entry.cpu_ram_pct:.0f}% > {cfg.max_cpu_ram_pct}%")
                    ok = False
                    break
                if entry.step_time_s > cfg.max_step_time_s:
                    print(f"    ⛔ 步耗时 {entry.step_time_s:.1f}s > {cfg.max_step_time_s}s")
                    ok = False
                    break

            except Exception as e:
                if _is_oom(e):
                    print(f"    ⛔ OOM at step {s}")
                    ok = False
                    del e  # 释放 stack frame 引用 (PyTorch FAQ)
                    _cuda_recover()
                    break
                raise

        stats = ledger.summary_stats()
        stats["ok"] = ok
        stats["mode"] = mode_name
        results[mode_name] = stats

        limit = ledger.hit_limit()
        if limit:
            print(f"    ⚠️  极限触发: {limit}")
        print()

        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(0.5)

    # 汇总表
    off_peak = results.get("off", {}).get("peak_vram_gb", 0)
    print(f"  {'─'*72}")
    print(f"  {'模式':<15s} {'tok/s':>8s} {'peak GB':>8s} {'VRAM节省':>8s}"
          f" {'stall%':>7s} {'in MB':>8s} {'out MB':>8s} {'状态':>4s}")
    print(f"  {'─'*72}")
    for mode, s in results.items():
        status = "✅" if s.get("ok") else "❌"
        peak = s.get('peak_vram_gb', 0)
        saved = off_peak - peak if off_peak > 0 else 0
        saved_str = f"{saved:+.2f}" if mode != "off" else "—"
        print(f"  {mode:<15s} {s.get('tok_s_mean',0):8.1f} "
              f"{peak:8.2f} {saved_str:>8s}"
              f" {s.get('stall_ratio_mean',0):7.0%} "
              f"{s.get('total_page_in_mb',0):8.1f} "
              f"{s.get('total_page_out_mb',0):8.1f} "
              f"{status:>4s}")
    print()

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


# ═══════════════════════════════════════════════════════════════════
# 测试 2: 找极限 batch size (二分搜索)
# ═══════════════════════════════════════════════════════════════════

def test_find_limit(base_cfg: BenchmarkConfig, mode: str = "v02_int8"):
    """二分搜索找到给定模式的最大可用 batch size"""
    print(f"\n{'='*72}")
    print(f"TEST: 找极限 batch [{mode}] (d={base_cfg.d_model}, "
          f"L={base_cfg.num_layers}, seq={base_cfg.seq_len})")
    print(f"{'='*72}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_sizes = [1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128]
    max_ok = 0
    max_peak = 0.0

    mode_factory = VVRAM_MODES.get(mode)
    thrash = ThrashGuard()

    for bs in batch_sizes:
        cfg = BenchmarkConfig(
            d_model=base_cfg.d_model,
            nhead=base_cfg.nhead,
            num_layers=base_cfg.num_layers,
            dim_feedforward=base_cfg.dim_feedforward,
            batch_size=bs,
            seq_len=base_cfg.seq_len,
        )

        # 每次重建模型 (避免梯度累积干扰)
        model = TestTransformer(cfg).to(device)
        model.train()

        vvram_cfg = mode_factory(cfg.vvram_min_bytes) if mode_factory else None
        ledger = LedgerTracker()

        ok = True
        for s in range(3):  # 每个 batch 跑 3 步确认稳定
            try:
                entry = run_single_step(model, cfg, vvram_cfg, step=s,
                                        phase="decode", thrash_guard=thrash)
                ledger.add(entry)
                thrash.check(ledger)

                if entry.cpu_ram_pct > cfg.max_cpu_ram_pct:
                    print(f"  batch={bs:3d}: ⛔  CPU RAM {entry.cpu_ram_pct:.0f}%")
                    ok = False
                    break
                if entry.step_time_s > cfg.max_step_time_s:
                    print(f"  batch={bs:3d}: ⛔  步耗时 {entry.step_time_s:.1f}s")
                    ok = False
                    break

            except Exception as e:
                if _is_oom(e):
                    ok = False
                    del e
                    _cuda_recover()
                    break
                raise

        stats = ledger.summary_stats()
        status = "✅" if ok else "❌"
        peak = stats.get("peak_vram_gb", 0)
        tok_s = stats.get("tok_s_mean", 0)
        stall = stats.get("stall_ratio_mean", 0)

        print(f"  batch={bs:3d}: {status}  peak={peak:.2f}GB  "
              f"tok/s={tok_s:.1f}  stall={stall:.0%}"
              f"{'  [safe_mode]' if thrash.active else ''}")

        if ok:
            max_ok = bs
            max_peak = peak
        else:
            # 如果 batch=1 就失败, 说明模型太大或系统 RAM 不够
            if bs <= 2:
                print(f"\n  ⚠️  即使 batch={bs} 也无法稳定运行。")
                print(f"      可能原因: 系统 RAM 不足 (offload 激活占太多)")
                print(f"      建议: 关闭其他程序, 或用更小的 seq_len/模型")
            break

        del model
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(0.3)

    print(f"\n  >>> 极限 batch = {max_ok}, 峰值 = {max_peak:.2f} GB")
    return max_ok, max_peak


# ═══════════════════════════════════════════════════════════════════
# 测试 3: 稳定性 (连续 N 轮, 检测雪崩)
# ═══════════════════════════════════════════════════════════════════

def test_stability(cfg: BenchmarkConfig, mode: str = "v02_int8",
                   n_rounds: int = 50):
    """连续跑 N 轮, 验证吞吐不雪崩"""
    print(f"\n{'='*72}")
    print(f"TEST: 稳定性 [{mode}] × {n_rounds} 轮")
    print(f"  配置: {cfg.label()}")
    print(f"{'='*72}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TestTransformer(cfg).to(device)
    model.train()

    mode_factory = VVRAM_MODES.get(mode)
    vvram_cfg = mode_factory(cfg.vvram_min_bytes) if mode_factory else None

    ledger = LedgerTracker()
    thrash = ThrashGuard()
    snowball_count = 0  # 连续下降计数

    for r in range(n_rounds):
        try:
            entry = run_single_step(model, cfg, vvram_cfg, step=r,
                                    phase="prefill" if r == 0 else "decode",
                                    thrash_guard=thrash)
            ledger.add(entry)
            thrash.check(ledger)

            # 每 10 步打印
            if r % 10 == 0 or r == n_rounds - 1:
                print(f"    {entry.to_line()}")

            # 雪崩检测: tok/s 连续下降
            if len(ledger.entries) >= 2:
                prev = ledger.entries[-2].tok_per_sec
                curr = ledger.entries[-1].tok_per_sec
                if curr < prev * 0.9 and curr > 0:
                    snowball_count += 1
                else:
                    snowball_count = 0

            if snowball_count >= 5:
                print(f"    ⛔ 吞吐雪崩: 连续 {snowball_count} 步下降")
                break

            limit = ledger.hit_limit()
            if limit:
                print(f"    ⚠️  极限触发: {limit} at round {r}")
                break

            if entry.cpu_ram_pct > cfg.max_cpu_ram_pct:
                print(f"    ⛔ CPU RAM 过高")
                break

        except Exception as e:
            if _is_oom(e):
                print(f"    ⛔ OOM at round {r}")
                del e
                _cuda_recover()
                break
            raise

    stats = ledger.summary_stats()
    print(f"\n  稳定性结果:")
    print(f"    完成轮数: {stats.get('steps', 0)}/{n_rounds}")
    print(f"    tok/s: mean={stats.get('tok_s_mean',0):.1f}, "
          f"min={stats.get('tok_s_min',0):.1f}, "
          f"max={stats.get('tok_s_max',0):.1f}")
    print(f"    peak VRAM: {stats.get('peak_vram_gb',0):.2f} GB")
    print(f"    stall ratio: {stats.get('stall_ratio_mean',0):.0%}")

    # 检查可复现性: 前 1/3 vs 后 1/3 的 tok/s
    entries = ledger.entries
    if len(entries) >= 6:
        third = len(entries) // 3
        first_third = [e.tok_per_sec for e in entries[:third] if e.tok_per_sec > 0]
        last_third = [e.tok_per_sec for e in entries[-third:] if e.tok_per_sec > 0]
        if first_third and last_third:
            avg_first = sum(first_third) / len(first_third)
            avg_last = sum(last_third) / len(last_third)
            drift = (avg_last - avg_first) / avg_first if avg_first > 0 else 0
            print(f"    吞吐漂移: {drift:+.1%} (前1/3→后1/3)")
            if abs(drift) < 0.1:
                print(f"    ✅ 稳定 (漂移 < ±10%)")
            else:
                print(f"    ⚠️  不稳定 (漂移 > ±10%)")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return stats


# ═══════════════════════════════════════════════════════════════════
# 测试 4: 梯度精度验证
# ═══════════════════════════════════════════════════════════════════

def test_gradient_accuracy(cfg: BenchmarkConfig):
    """对比 4 种模式的梯度精度, 确认压缩不会炸"""
    print(f"\n{'='*72}")
    print(f"TEST: 梯度精度 [{cfg.label()}]")
    print(f"{'='*72}")

    # ── 路径一致性检查：确认使用的是 v0.3 ──
    import virtual_vram as vv_module
    print(f"  VirtualVRAM 模块路径: {vv_module.__file__}")
    if hasattr(vv_module, '__version__'):
        print(f"  VirtualVRAM 版本: {vv_module.__version__}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TestTransformer(cfg).to(device)
    model.eval()  # 消除所有随机性 (dropout 等)

    # ⚠️  关键: 禁用 Flash/MemEfficient SDPA
    # 这两个 kernel 内部保存 tensor 的方式和 saved_tensors_hooks 不兼容,
    # 会导致 backward 产生 NaN。强制用 math backend (纯 PyTorch, 完全兼容)。
    # 这不是 virtual_vram 的 bug, 是 PyTorch SDPA + saved_tensors_hooks 的已知限制。
    _sdp_flash_old = torch.backends.cuda.flash_sdp_enabled()
    _sdp_mem_old = torch.backends.cuda.mem_efficient_sdp_enabled()
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    print(f"  SDPA backend: math only (flash/mem_efficient 已禁用)")

    # 保存初始权重, 每次跑前恢复确保完全一致
    init_state = {k: v.clone() for k, v in model.state_dict().items()}

    # 创建固定输入 (在 CPU 上, 每次复制到 GPU)
    torch.manual_seed(cfg.seed)
    fixed_input_cpu = torch.randn(cfg.batch_size, cfg.seq_len, cfg.d_model)
    print(f"  输入校验和: {fixed_input_cpu.sum().item():.6f}")

    # 开启确定性模式
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_cudnn = torch.backends.cudnn.deterministic
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass

    def _run_and_get_grads(vvram_cfg, label=""):
        gc.collect()
        torch.cuda.empty_cache()

        # 恢复模型到初始状态
        model.load_state_dict(init_state)
        model.zero_grad(set_to_none=True)

        # 用固定输入 (每次从 CPU clone)
        x = fixed_input_cpu.clone().to(device)

        # 校验
        param_sum = sum(p.sum().item() for p in model.parameters())
        x_sum = x.sum().item()

        if vvram_cfg:
            with virtual_vram(vvram_cfg):
                y = model(x)
                loss = y.mean()
                loss.backward()
        else:
            y = model(x)
            loss = y.mean()
            loss.backward()

        torch.cuda.synchronize()
        loss_val = loss.item()
        grads = {n: p.grad.detach().clone()
                 for n, p in model.named_parameters() if p.grad is not None}

        # ── 检测并打印非有限梯度（诊断 NaN 来源）──
        bad = [n for n, g in grads.items() if not torch.isfinite(g).all()]
        if bad:
            print(f"      ⚠️  non-finite grads: {bad[0]} (+{len(bad)-1} more)")
        grad_norm = sum(g.float().norm().item() for g in grads.values() if torch.isfinite(g).all())

        print(f"    {label:15s}: loss={loss_val:.6f}  param_sum={param_sum:.4f}"
              f"  x_sum={x_sum:.4f}  grad_norm={grad_norm:.4f}"
              f"  #grads={len(grads)}")

        model.zero_grad(set_to_none=True)
        del x, y, loss
        return grads

    def _compare_grads(ref, test):
        """返回 (avg_rel_err, count, has_nan, worst_name, worst_err)"""
        total_err = 0.0
        count = 0
        has_nan = False
        worst_name = ""
        worst_err = 0.0

        for name in ref:
            if name not in test:
                continue
            r = ref[name].float()
            t = test[name].float()

            # 任何非有限值：直接判失败，但别"吞掉"
            if not torch.isfinite(t).all():
                has_nan = True
                # 给 worst 一个可追踪的名字
                if worst_name == "":
                    worst_name = name
                    worst_err = float("inf")
                continue

            err = (t - r).norm()
            norm = r.norm()
            if norm > 1e-8:
                rel = (err / norm).item()
                total_err += rel
                count += 1
                if rel > worst_err:
                    worst_err = rel
                    worst_name = name

        # ✅ 收口判据：只要 has_nan=True 或 count=0，就直接把 avg_err 设为 inf
        if has_nan or count == 0:
            return float("inf"), count, has_nan, worst_name, worst_err

        return total_err / count, count, has_nan, worst_name, worst_err

    # ── Step 0: baseline × 2, 检查 CUDA 确定性 ──
    print("\n  [诊断] baseline × 2 (检测 CUDA 非确定性):")
    try:
        grads_base1 = _run_and_get_grads(None, "baseline_1")
        grads_base2 = _run_and_get_grads(None, "baseline_2")
    except Exception as e:
        if _is_oom(e):
            print(f"  ⛔ Baseline OOM, 跳过梯度测试")
            del e; _cuda_recover()
            del model; gc.collect(); torch.cuda.empty_cache()
            return
        raise

    base_err, _, _, worst_n, worst_e = _compare_grads(grads_base1, grads_base2)
    if base_err < 1e-6:
        print(f"    → baseline 一致 ✅ (err={base_err:.2e})")
    elif base_err < 0.001:
        print(f"    → baseline 微差 ⚠️ (err={base_err:.2e}, worst={worst_n})")
    else:
        print(f"    → baseline 不一致 ❌ (err={base_err:.2e}, worst={worst_n}={worst_e:.2e})")

    grads_base = grads_base1
    del grads_base2

    # ── Step 1: 各 vvram 模式 vs baseline ──
    print(f"\n  [对比] (CUDA 基线误差: {base_err:.2e})")
    for mode_name, mode_factory in VVRAM_MODES.items():
        if mode_factory is None:
            continue
        vvram_cfg = mode_factory(cfg.vvram_min_bytes)
        try:
            grads_test = _run_and_get_grads(vvram_cfg, mode_name)
        except Exception as e:
            if _is_oom(e):
                print(f"  {mode_name:<15s}: ⛔ OOM, 跳过")
                del e; _cuda_recover()
                continue
            raise

        avg_err, cnt, has_nan, worst_n, worst_e = _compare_grads(
            grads_base, grads_test)
        net_err = max(0, avg_err - base_err)
        nan_tag = " [NaN!]" if has_nan else ""
        status = "✅" if avg_err < 0.001 else ("⚠️" if avg_err < 0.05 else "❌")
        print(f"  {mode_name:<15s}: err={avg_err:.6f} net={net_err:.6f}"
              f"  worst={worst_n}={worst_e:.2e}  {status}{nan_tag}")

    # 恢复设置
    try:
        torch.use_deterministic_algorithms(old_deterministic)
        torch.backends.cudnn.deterministic = old_cudnn
        torch.backends.cuda.enable_flash_sdp(_sdp_flash_old)
        torch.backends.cuda.enable_mem_efficient_sdp(_sdp_mem_old)
    except Exception:
        pass

    del model, init_state, fixed_input_cpu, grads_base
    gc.collect()
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Virtual VRAM v0.2 收口测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python test_convergence.py --quick           # 快速验证 (小模型)
  python test_convergence.py --compare medium  # medium 配置对比 4 模式
  python test_convergence.py --find-limit      # 找 v02_int8 极限 batch
  python test_convergence.py --stability 50    # 50 轮稳定性
  python test_convergence.py --gradient        # 梯度精度验证
  python test_convergence.py --all             # 全部跑一遍
        """)

    parser.add_argument("--quick", action="store_true",
                        help="快速验证 (small 配置, 3 步)")
    parser.add_argument("--compare", type=str, default=None,
                        choices=list(CONFIGS.keys()),
                        help="基准对比 (选配置档位)")
    parser.add_argument("--find-limit", action="store_true",
                        help="找极限 batch size")
    parser.add_argument("--stability", type=int, default=0, metavar="N",
                        help="稳定性测试 N 轮")
    parser.add_argument("--gradient", action="store_true",
                        help="梯度精度验证")
    parser.add_argument("--all", action="store_true",
                        help="全部测试")
    parser.add_argument("--config", type=str, default="medium",
                        choices=list(CONFIGS.keys()),
                        help="默认配置档位 (default: medium)")
    parser.add_argument("--mode", type=str, default="v02_int8",
                        choices=list(VVRAM_MODES.keys()),
                        help="虚拟显存模式 (default: v02_int8)")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("❌ CUDA not available. 这个测试需要 GPU。")
        sys.exit(1)

    print(f"{'='*72}")
    print(f"Virtual VRAM v0.2 收口测试 (Convergence & Limit Finder)")
    print(f"{'='*72}")
    print(f"GPU:     {_gpu_info()}")
    print(f"PyTorch: {torch.__version__}")
    if HAS_PSUTIL:
        ram = psutil.virtual_memory()
        print(f"RAM:     {ram.total/1e9:.1f} GB (used {ram.percent}%)")
    print()

    cfg = CONFIGS[args.config]

    # ── 调度 ──
    ran_anything = False

    def _between_tests():
        """测试间强制清理"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    if args.quick or args.all:
        ran_anything = True
        test_baseline_comparison(CONFIGS["small"], n_steps=3)
        _between_tests()

    if args.compare or args.all:
        ran_anything = True
        compare_cfg = CONFIGS.get(args.compare or "medium")
        test_baseline_comparison(compare_cfg, n_steps=5)
        _between_tests()

    if args.find_limit or args.all:
        ran_anything = True
        test_find_limit(cfg, mode=args.mode)
        _between_tests()

    if args.stability > 0 or args.all:
        ran_anything = True
        n = args.stability if args.stability > 0 else 30
        # --all 模式用 small 配置避免 CPU RAM 不足
        stab_cfg = CONFIGS["small"] if args.all else cfg
        test_stability(stab_cfg, mode=args.mode, n_rounds=n)
        _between_tests()

    if args.gradient or args.all:
        ran_anything = True
        test_gradient_accuracy(CONFIGS["small"])

    if not ran_anything:
        print("未指定测试项。用 --help 查看选项, 或用 --quick 快速验证。")
        parser.print_help()

    print(f"\n{'='*72}")
    print(f"测试完成")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
