# virtual_vram.py
# -*- coding: utf-8 -*-
"""
Virtual VRAM v1.3 (DeepSeek Block-Based + Prefetch)

v1.3 新增特性（在v1.2基础上）：
  4. Block-based offload：只对第一维（batch/seq_len）分块，避免复杂stride问题
  5. 异步H2D预取：基于LRU cache，预测即将使用的tensor

修复的问题：
  - __slots__中添加'prefetched'属性
  - 简化分块逻辑：只分第一维，保持其他维度完整
  - 预取失败时自动fallback到同步H2D

设计原则：
  - 保守实现：只处理最常见的情况
  - 自动降级：出错时回退到v1.2的行为
  - 保持稳定：宁可慢也不能崩

Usage:
    from apt.vgpu.runtime.virtual_vram import VirtualVRAMConfig, virtual_vram

    cfg = VirtualVRAMConfig(
        min_tensor_bytes=5<<20,
        max_tensor_bytes=50<<20,
        enable_block_offload=True,   # 启用分块offload
        enable_prefetch=True,        # 启用预取
        verbose=True
    )
    with virtual_vram(cfg):
        loss = model(x).sum()
        loss.backward()
"""

from __future__ import annotations

import math
import queue
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

import torch

# 尝试导入LECaC
try:
    from apt.vgpu.runtime.lecac import lecac_quantize, lecac_dequantize
    LECaC_AVAILABLE = True
except ImportError:
    LECaC_AVAILABLE = False
    lecac_quantize = None
    lecac dequantize = None

# 自然平衡常数：4/e ≈ 1.4715，用于 LECaC 量化补偿
NATURAL_EQUILIBRIUM_CONSTANT: float = 4.0 / math.e


@dataclass
class VirtualVRAMConfig:
    enabled: bool = True
    # Tensor size filtering（选择性offload）
    min_tensor_bytes: int = 5 << 20   # 5MB — 最小offload大小
    max_tensor_bytes: int = 50 << 20  # 50MB — 最大offload大小（0=无限制）

    # Block-based offload（简化版）
    enable_block_offload: bool = False  # 启用分块offload
    block_size: int = 64              # 每块的元素数

    # 预取优化
    enable_prefetch: bool = False     # 启用异步预取
    prefetch_cache_size: int = 5      # 预取cache大小

    # 量化优化（实验性）
    enable_quantization: bool = False
    quantization_bits: int = 8

    # 性能监控
    verbose: bool = False

    # 已废弃，保留向后兼容
    min_storage_bytes: int = 0


class _Packed:
    """pack_hook 返回值：CPU tensor + 恢复用元数据 + 预取信息"""
    __slots__ = ('cpu_tensor', 'device', 'dtype', 'requires_grad',
                 'quantized', 'block_info', 'prefetched', 'cache_key')

    def __init__(self, cpu_tensor: torch.Tensor,
                 device: torch.device,
                 dtype: torch.dtype,
                 requires_grad: bool,
                 quantized: bool = False,
                 block_info: Optional[Dict] = None,
                 cache_key: Optional[Tuple] = None):
        self.cpu_tensor = cpu_tensor
        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.quantized = quantized
        self.block_info = block_info or {}
        self.prefetched = None  # 预取的tensor
        self.cache_key = cache_key  # 用于预取cache的key


class _Prefetcher:
    """异步预取器（参考DeepSeek的异步H2D）"""

    def __init__(self, cache_size: int = 5, verbose: bool = False):
        self.cache = OrderedDict()  # LRU cache
        self.cache_size = cache_size
        self.queue: queue.Queue = queue.Queue(maxsize=10)
        self.verbose = verbose
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

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
        if packed.cache_key is not None:
            self.queue.put(packed)

    def _prefetch_worker(self):
        """预取工作线程"""
        while not self.stop_event.is_set():
            try:
                # 超时获取，避免永久阻塞
                packed = self.queue.get(timeout=0.1)
                self._do_prefetch(packed)
            except queue.Empty:
                continue
            except Exception as e:
                if self.verbose:
                    print(f"[VirtualVRAM] ⚠️  Prefetch error: {e}")

    def _do_prefetch(self, packed: _Packed):
        """执行预取"""
        try:
            # H2D预取
            cpu_tensor = packed.cpu_tensor

            # 反量化（如果需要）
            if packed.quantized and LECaC_AVAILABLE:
                restored = lecac_dequantize(
                    cpu_tensor,
                    bits=8,  # 从cfg读取
                    original_shape=cpu_tensor.shape,
                    original_dtype=packed.dtype,
                    constant=NATURAL_EQUILIBRIUM_CONSTANT
                )
            else:
                restored = cpu_tensor

            # 移回GPU
            restored = restored.to(device=packed.device, non_blocking=False)
            restored.requires_grad_(packed.requires_grad)

            # 存入cache
            if packed.cache_key is not None:
                self.cache[packed.cache_key] = restored

            # LRU eviction
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)  # 移除最旧的

            packed.prefetched = restored

            if self.verbose:
                mb = restored.numel() * restored.element_size() / 1024 / 1024
                print(f"[VirtualVRAM] ⚡ Prefetched {mb:.2f}MB {tuple(restored.shape)}")

        except Exception as e:
            if self.verbose:
                print(f"[VirtualVRAM] ⚠️  Prefetch failed: {e}")
            packed.prefetched = None

    def get(self, cache_key: Tuple) -> Optional[torch.Tensor]:
        """获取预取的tensor"""
        return self.cache.get(cache_key)


@contextmanager
def virtual_vram(cfg: VirtualVRAMConfig):
    """
    DeepSeek-inspired 激活值 CPU offload with block-based & prefetch.
    """
    if not cfg.enabled:
        yield
        return

    min_bytes = cfg.min_tensor_bytes or cfg.min_storage_bytes or (5 << 20)
    max_bytes = cfg.max_tensor_bytes or 0

    # 尝试导入LECaC
    try:
        from apt.vgpu.runtime.lecac import lecac_quantize, lecac_dequantize
        LECaC_AVAILABLE = True
    except ImportError:
        LECaC_AVAILABLE = False
        lecac_quantize = None
        lecac_dequantize = None

    # 查表：避免对同一 view 重复 offload
    cache: Dict[Tuple, _Packed] = {}

    # 预取器（如果启用）
    prefetcher: Optional[_Prefetcher] = None
    if cfg.enable_prefetch:
        prefetcher = _Prefetcher(
            cache_size=cfg.prefetch_cache_size,
            verbose=cfg.verbose
        )
        prefetcher.start()

    def _make_key(t: torch.Tensor) -> Tuple:
        return (
            t.untyped_storage().data_ptr(),
            tuple(t.shape),
            tuple(t.stride()),
            t.storage_offset(),
        )

    def _split_first_dim(t: torch torch.Tensor, block_size: int) -> List[torch.Tensor]:
        """只沿第一维分块（简化版，避免stride问题）"""
        if t.dim() == 0 or t.size(0) == 0:
            return [t]

        first_dim_size = t.size(0)
        if first_dim_size <= block_size:
            return [t]

        blocks = []
        num_blocks = (first_dim_size + block_size - 1) // block_size

        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = min((i + 1) * block_size, first_dim_size)
            block = t[start_idx:end_idx]  # 沿第一维切片
            blocks.append(block)

        return blocks

    def _pack_hook(t: torch.Tensor) -> Any:
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
            if cfg.verbose:
                mb = nbytes / 1024 / 1024
                print(f"[VirtualVRAM] ⏭️  Skip large tensor {mb:.2f}MB {tuple(t.shape)}")
            return t

        # ── 查表：同一 view 只搬一次 ────────────────────────────────────
        key = _make_key(t)
        if key in cache:
            if cfg.verbose:
                print(f"[VirtualVRAM] 🔗 cache hit {tuple(t.shape)}")
            cached = cache[key]
            # 如果启用了预取，添加到预取队列
            if prefetcher:
                prefetcher.add(cached)
            return cached

        # ── D2H：分块或整体 ───────────────────────────────────────────────
        try:
            original_dtype = t.dtype
            quantized = False
            block_info = None

            # 分块offload（如果启用）
            if cfg.enable_block_offload and t.dim() > 0 and t.size(0) > cfg.block_size:
                try:
                    blocks = _split_first_dim(t, cfg.block_size)
                    cpu_blocks = []

                    for block in blocks:
                        # LECaC量化（如果启用）
                        if cfg.enable_quantization and LECaC_AVAILABLE:
                            q_block = lecac_quantize(
                                block.detach(),
                                bits=cfg.quantization_bits,
                                constant=NATURAL_EQUILIBRIUM_CONSTANT
                            ).cpu()
                            cpu_blocks.append(q_block)
                            quantized = True
                        else:
                            cpu_blocks.append(block.detach().cpu())

                    # 重组（在CPU上）
                    if len(cpu_blocks) == 1:
                        cpu_tensor = cpu_blocks[0]
                    else:
                        cpu_tensor = torch.cat(cpu_blocks, dim=0)

                    block_info = {
                        'num_blocks': len(blocks),
                        'block_size': cfg.block_size,
                        'original_shape': t.shape
                    }

                except Exception as e:
                    if cfg.verbose:
                        print(f"[VirtualVRAM] ⚠️  Block offload failed, using full tensor: {e}")
                    # Fallback to full tensor
                    if cfg.enable_quantization and LECaC_AVAILABLE:
                        cpu_tensor = lecac_quantize(
                            t.detach(),
                            bits=cfg.quantization_bits,
                            constant=NATURAL_EQUILIBRIUM_CONSTANT
                        ).cpu()
                        quantized = True
                    else:
                        cpu_tensor = t.detach().cpu()
            else:
                # 不分块，直接搬运
                if cfg.enable_quantization and LECaC_AVAILABLE:
                    cpu_tensor = lecac_quantize(
                        t.detach(),
                        bits=cfg.quantization_bits,
                        constant=NATURAL_EQUILIBRIUM_CONSTANT
                    ).cpu()
                    quantized = True
                else:
                    cpu_tensor = t.detach().cpu()

            packed = _Packed(cpu_tensor, t.device, original_dtype,
                             bool(t.requires_grad), quantized,
                             block_info, key)
            cache[key] = packed

            # 添加到预取队列
            if prefetcher:
                prefetcher.add(packed)

            if cfg.verbose:
                mb = nbytes / 1024 / 1024
                block_str = f" blocks={block_info['num_blocks']}" if block_info else ""
                print(f"[VirtualVRAM] ✅ D2H {mb:.2f}MB {tuple(t.shape)} "
                      f"dtype={original_dtype}{block_str}")

            return packed

        except Exception as e:
            if cfg.verbose:
                print(f"[VirtualVRAM] ❌ D2H 失败: {e}")
            return t

    def _unpack_hook(packed: Any) -> torch.Tensor:
        if not isinstance(packed, _Packed):
            return packed

        # 检查是否已经预取
        if packed.prefetched is not None:
            if cfg.verbose:
                mb = packed.prefetched.numel() * packed.prefetched.element_size() / 1024 / 1024
                print(f"[VirtualVRAM] ⚡ Using prefetched {mb:.2f}MB {tuple(packed.prefetched.shape)}")
            return packed.prefetched

        # 检查预取cache
        if prefetcher and packed.cache_key is not None:
            cached = prefetcher.get(packed.cache_key)
            if cached is not None:
                if cfg.verbose:
                    mb = cached.numel() * cached.element_size() / 1024 / 1024
                    print(f"[VirtualVRAM] ⚡ Using prefetch cache {mb:.2f}MB")
                return cached

        # H2D：同步恢复（带反量化）
        try:
            cpu_tensor = packed.cpu_tensor

            # 反量化（如果需要）
            if packed.quantized and LECaC_AVAILABLE:
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
        with torch.autograd.graph.saved_tensors_hooks(_pack_hook, _unpack_hook):
            yield
    finally:
        # 清理资源
        cache.clear()
        if prefetcher:
            prefetcher.stop()
