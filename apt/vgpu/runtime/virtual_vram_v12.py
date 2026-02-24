# virtual_vram.py
# -*- coding: utf-8 -*-
"""
Virtual VRAM v1.2 (DeepSeek-Inspired Advanced Offload)

v1.2 新增特性（灵感来自 DeepSeek FlashMLA）：
  1. 提高min_tensor_bytes阈值：1MB → 5MB（减少小tensor开销）
  2. 选择性offload：只offload 5MB-50MB的tensor（避免超大tensor交换开销）
  3. Block-based offload：分块处理大tensor（类似paged KV cache）
  4. 异步H2D预取：在forward结束前预取下一批（减少等待时间）
  5. LECaC集成：量化后offload（实验性，需要LECaC可用）

设计原则：
  - 参考DeepSeek的paged KV cache：64 tokens per block
  - 参考FlashMLA的稀疏策略：只处理有价值的tensor
  - 保持v1.1的稳定性：.cpu()为主，避免pinned memory的坑

Usage:
    from apt.vgpu.runtime.virtual_vram import VirtualVRAMConfig, virtual_vram

    cfg = VirtualVRAMConfig(
        min_tensor_bytes=5<<20,      # 5MB
        max_tensor_bytes=50<<20,     # 50MB
        block_size=64,
        enable_prefetch=True,
        enable_quantization=False
    )
    with virtual_vram(cfg):
        loss = model(x).sum()
        loss.backward()
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import torch

# 尝试导入LECaC
try:
    from apt.vgpu.runtime.lecac import lecac_quantize, lecac_dequantize
    LECaC_AVAILABLE = True
except ImportError:
    LECaC_AVAILABLE = False
    lecac_quantize = None
    lecac_dequantize = None

# 自然平衡常数：4/e ≈ 1.4715，用于 LECaC 量化补偿
NATURAL_EQUILIBRIUM_CONSTANT: float = 4.0 / math.e


@dataclass
class VirtualVRAMConfig:
    enabled: bool = True
    # Tensor size filtering（选择性offload）
    min_tensor_bytes: int = 5 << 20   # 5MB — 最小offload大小
    max_tensor_bytes: int = 50 << 20  # 50MB — 最大offload大小（0=无限制）

    # Block-based offload（类似paged KV cache）
    block_size: int = 64               # 每块的token/元素数

    # 预取优化
    enable_prefetch: bool = True      # 启用异步预取
    prefetch_batches: int = 2         # 预取未来N个tensor

    # 量化优化（实验性）
    enable_quantization: bool = False # 启用LECaC量化
    quantization_bits: int = 8       # 量化位数

    # 性能监控
    verbose: bool = False

    # 已废弃，保留向后兼容
    min_storage_bytes: int = 0


class _Packed:
    """pack_hook 返回值：CPU tensor + 恢复用元数据 + 分块信息"""
    __slots__ = ('cpu_tensor', 'device', 'dtype', 'requires_grad',
                 'block_size', 'block_offsets', 'quantized')

    def __init__(self, cpu_tensor: torch.Tensor,
                 device: torch.device,
                 dtype: torch.dtype,
                 requires_grad: bool,
                 block_size: int = 0,
                 block_offsets: Optional[List[int]] = None,
                 quantized: bool = False):
        self.cpu_tensor = cpu_tensor
        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.block_size = block_size
        self.block_offsets = block_offsets or []
        self.quantized = quantized


class _PrefetchQueue:
    """异步预取队列（参考DeepSeek的异步H2D）"""
    def __init__(self, max_size: int = 10):
        self.queue: List[Any] = []
        self.max_size = max_size
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

    def start(self):
        """启动预取线程"""
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.thread.start()

    def stop(self):
        """停止预取线程"""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)

    def add(self, packed: _Packed):
        """添加待预取的tensor"""
        with self.lock:
            if len(self.queue) < self.max_size:
                self.queue.append(packed)

    def _prefetch_worker(self):
        """预取工作线程"""
        while not self.stop_event.is_set():
            with self.lock:
                if not self.queue:
                    time.sleep(0.001)
                    continue
                packed = self.queue.pop(0)

            try:
                # 预取H2D
                restored = packed.cpu_tensor.to(
                    device=packed.device,
                    non_blocking=False,
                )
                restored.requires_grad_(packed.requires_grad)
                packed.prefetched = restored
            except Exception as e:
                if hasattr(packed, 'verbose') and packed.verbose:
                    print(f"[VirtualVRAM] ⚠️  Prefetch failed: {e}")
                packed.prefetched = None


@contextmanager
def virtual_vram(cfg: VirtualVRAMConfig):
    """
    DeepSeek-inspired 激活值 CPU offload context manager。
    """
    if not cfg.enabled:
        yield
        return

    min_bytes = cfg.min_tensor_bytes or cfg.min_storage_bytes or (5 << 20)
    max_bytes = cfg.max_tensor_bytes or 0

    # 查表：避免对同一 view 重复 offload
    cache: Dict[Tuple, _Packed] = {}

    # 预取队列（如果启用）
    prefetch_queue: Optional[_PrefetchQueue] = None
    if cfg.enable_prefetch:
        prefetch_queue = _PrefetchQueue(max_size=cfg.prefetch_batches)
        prefetch_queue.start()

    def _make_key(t: torch.Tensor) -> Tuple:
        return (
            t.untyped_storage().data_ptr(),
            tuple(t.shape),
            tuple(t.stride()),
            t.storage_offset(),
        )

    def _split_into_blocks(t: torch.Tensor, block_size: int) -> List[torch.Tensor]:
        """将tensor分成多个块（类似paged KV cache）"""
        if block_size <= 0 or t.numel() <= block_size:
            return [t]

        blocks = []
        num_blocks = (t.numel() + block_size - 1) // block_size

        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, t.numel())

            # 展平后分块，再恢复原始形状
            flat = t.flatten()[start_idx:end_idx]
            blocks.append(flat)

        return blocks

    def pack_hook(t: torch.Tensor) -> Any:
        # ── 快速筛选 ────────────────────────────────────────────────────
        if not torch.is_tensor(t) or not t.is_cuda:
            return t

        # bool 类型必须跳过
        if t.dtype == torch.bool:
            return t

        nbytes = t.numel() * t.element_size()

        # 太小或太大的tensor都跳过（选择性offload）
        if nbytes < min_bytes:
            return t
        if max_bytes > 0 and nbytes > max_bytes:
            return t

        # ── 查表：同一 view 只搬一次 ────────────────────────────────────
        key = _make_key(t)
        if key in cache:
            if cfg.verbose:
                print(f"[VirtualVRAM] 🔗 cache hit {tuple(t.shape)}")
            return cache[key]

        # ── Block-based D2H（类似paged KV cache）────────────────────────
        try:
            original_shape = t.shape
            original_stride = t.stride()
            dtype = t.dtype
            requires_grad = bool(t.requires_grad)

            # 分块处理
            blocks = _split_into_blocks(t, cfg.block_size)

            # 量化（如果启用）
            quantized = False
            if cfg.enable_quantization and LECaC_AVAILABLE:
                # LECaC量化每个block
                quantized_blocks = []
                for block in blocks:
                    q_block = lecac_quantize(
                        block,
                        bits=cfg.quantization_bits,
                        constant=NATURAL_EQUILIBRIUM_CONSTANT
                    )
                    quantized_blocks.append(q_block)
                cpu_blocks = [b.cpu() for b in quantized_blocks]
                quantized = True
            else:
                # 直接CPU搬运
                cpu_blocks = [b.detach().cpu() for b in blocks]

            # 恢复原始形状（在CPU上重组）
            if len(blocks) == 1:
                cpu_tensor = cpu_blocks[0].reshape(original_shape)
            else:
                # 重新拼接
                cpu_flat = torch.cat(cpu_blocks)
                cpu_tensor = cpu_flat.reshape(original_shape)

            block_offsets = list(range(len(blocks)))

            packed = _Packed(
                cpu_tensor, t.device, dtype, requires_grad,
                block_size=cfg.block_size,
                block_offsets=block_offsets,
                quantized=quantized
            )
            cache[key] = packed

            if cfg.verbose:
                mb = nbytes / 1024 / 1024
                print(f"[VirtualVRAM] ✅ D2H {mb:.2f}MB {tuple(t.shape)} "
                      f"dtype={dtype} blocks={len(blocks)} "
                      f"quantized={quantized}")

            # 添加到预取队列
            if prefetch_queue:
                prefetch_queue.add(packed)

            return packed

        except Exception as e:
            if cfg.verbose:
                print(f"[VirtualVRAM] ❌ D2H 失败: {e}")
            return t

    def unpack_hook(packed: Any) -> torch.Tensor:
        if not isinstance(packed, _Packed):
            return packed

        # 检查是否已经预取
        if hasattr(packed, 'prefetched') and packed.prefetched is not None:
            restored = packed.prefetched
            if cfg.verbose:
                mb = restored.numel() * restored.element_size() / 1024 / 1024
                print(f"[VirtualVRAM] ⚡ H2D (prefetched) {mb:.2f}MB {tuple(restored.shape)}")
            return restored

        # H2D：同步恢复
        try:
            cpu_tensor = packed.cpu_tensor

            # 反量化（如果需要）
            if packed.quantized and LECaC_AVAILABLE:
                # 分块反量化
                if packed.block_size > 0 and len(packed.block_offsets) > 1:
                    # 分块反量化
                    blocks = []
                    flat = cpu_tensor.flatten()
                    block_size = packed.block_size

                    for i, offset in enumerate(packed.block_offsets):
                        start_idx = offset * block_size
                        end_idx = min(start_idx + block_size, flat.numel())
                        block = flat[start_idx:end_idx]

                        dq_block = lecac_dequantize(
                            block,
                            bits=cfg.quantization_bits,
                            original_shape=(),
                            original_dtype=packed.dtype,
                            constant=NATURAL_EQUILIBRIUM_CONSTANT
                        )
                        blocks.append(dq_block)

                    restored_flat = torch.cat(blocks)
                    restored = restored_flat.reshape(cpu_tensor.shape)
                else:
                    # 整体反量化
                    restored = lecac_dequantize(
                        cpu_tensor,
                        bits=cfg.quantization_bits,
                        original_shape=cpu_tensor.shape,
                        original_dtype=packed.dtype,
                        constant=NATURAL_EQUILIBRIUM_CONSTANT
                    )
            else:
                restored = cpu_tensor

            # 移回GPU
            restored = restored.to(device=packed.device, non_blocking=False)
            restored.requires_grad_(packed.requires_grad)

        except Exception as e:
            raise RuntimeError(f"Failed to restore tensor from CPU: {e}")

        if cfg.verbose:
            mb = restored.numel() * restored.element_size() / 1024 / 1024
            print(f"[VirtualVRAM] ↩️  H2D {mb:.2f}MB {tuple(restored.shape)}")

        return restored

    try:
        with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
            yield
    finally:
        # 清理资源
        cache.clear()
        if prefetch_queue:
            prefetch_queue.stop()
