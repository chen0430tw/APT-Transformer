# virtual_vram_v14.py
# -*- coding: utf-8 -*-
"""
Virtual VRAM v1.4 (Paged Memory Management)

v1.4 新增特性（在v1.3基础上）：
  6. Paged Memory Management: 类似 PagedAttention 的固定页大小管理
     - 固定页大小（Page Size: 2MB 默认）
     - 页表（PageTable）管理虚拟页到物理页的映射
     - 页分配器（PageAllocator）管理物理页的分配和释放
     - LRU 页面淘汰策略

设计原则：
  - 参考 vLLM PagedAttention 的设计
  - 将 CPU 内存分成固定大小的页
  - 减少内存碎片，提高内存利用率
  - 支持页面共享（多个 tensor 可以引用同一页）

Usage:
    from apt.vgpu.runtime.virtual_vram import VirtualVRAMConfig, virtual_vram

    cfg = VirtualVRAMConfig(
        min_tensor_bytes=5<<20,
        max_tensor_bytes=50<<20,
        enable_paged_memory=True,    # 启用分页内存管理
        page_size=2<<20,             # 页大小 2MB
        max_pages=1000,              # 最大物理页数
        enable_block_offload=True,
        enable_prefetch=True,
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
from typing import Any, Dict, List, Optional, Tuple, Set

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

    # Block-based offload（简化版）
    enable_block_offload: bool = False  # 启用分块offload
    block_size: int = 64              # 每块的元素数

    # 预取优化
    enable_prefetch: bool = False     # 启用异步预取
    prefetch_cache_size: int = 5      # 预取cache大小

    # 量化优化（实验性）
    enable_quantization: bool = False
    quantization_bits: int = 8

    # ===== v1.4 新增：Paged Memory Management =====
    enable_paged_memory: bool = False  # 启用分页内存管理
    page_size: int = 2 << 20           # 页大小 2MB（默认）
    max_pages: int = 1000              # 最大物理页数（2GB CPU 内存）
    page_cache_size: int = 100         # 页缓存大小（GPU 端缓存常用页）

    # 性能监控
    verbose: bool = False

    # 已废弃，保留向后兼容
    min_storage_bytes: int = 0


class _PageAllocator:
    """
    物理页分配器（类似 PagedAttention 的 Block Manager）

    管理固定大小的物理页，支持分配和释放操作。
    使用位图或空闲列表来跟踪空闲页。
    """

    def __init__(self, page_size: int, max_pages: int, verbose: bool = False):
        self.page_size = page_size
        self.max_pages = max_pages
        self.verbose = verbose

        # 物理页存储：每个页是一个 CPU tensor
        self.pages: List[Optional[torch.Tensor]] = [None] * max_pages

        # 空闲页列表（使用 free list 管理空闲页）
        self.free_pages: List[int] = list(range(max_pages))

        # 页引用计数（用于页面共享）
        self.page_refcount: List[int] = [0] * max_pages

        # 统计信息
        self.stats = {
            'allocations': 0,
            'deallocations': 0,
            'evictions': 0,
            'peak_usage': 0
        }

    def allocate(self, num_pages: int) -> Optional[List[int]]:
        """
        分配多个物理页

        返回: 物理页号列表，失败返回 None
        """
        if len(self.free_pages) < num_pages:
            if self.verbose:
                print(f"[PageAllocator] ⚠️  无法分配 {num_pages} 页，剩余 {len(self.free_pages)} 页")
            return None

        allocated = []
        for _ in range(num_pages):
            page_id = self.free_pages.pop()
            allocated.append(page_id)
            self.page_refcount[page_id] = 1

        self.stats['allocations'] += num_pages
        current_usage = self.max_pages - len(self.free_pages)
        self.stats['peak_usage'] = max(self.stats['peak_usage'], current_usage)

        if self.verbose:
            print(f"[PageAllocator] ✅ 分配 {num_pages} 页: {allocated}")

        return allocated

    def deallocate(self, page_ids: List[int]):
        """
        释放物理页
        """
        for page_id in page_ids:
            if page_id < 0 or page_id >= self.max_pages:
                continue

            self.page_refcount[page_id] -= 1

            # 引用计数为 0 时才真正释放
            if self.page_refcount[page_id] <= 0:
                self.pages[page_id] = None
                self.free_pages.append(page_id)
                self.stats['deallocations'] += 1

                if self.verbose:
                    print(f"[PageAllocator] 🗑️  释放页 {page_id}")

    def get_page(self, page_id: int) -> Optional[torch.Tensor]:
        """获取物理页的数据"""
        if page_id < 0 or page_id >= self.max_pages:
            return None
        return self.pages[page_id]

    def set_page(self, page_id: int, data: torch.Tensor):
        """设置物理页的数据"""
        if page_id < 0 or page_id >= self.max_pages:
            return
        self.pages[page_id] = data

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'free_pages': len(self.free_pages),
            'used_pages': self.max_pages - len(self.free_pages),
            'usage_ratio': (self.max_pages - len(self.free_pages)) / self.max_pages
        }


class _PageTable:
    """
    页表（类似 PagedAttention 的 Page Table）

    管理虚拟页到物理页的映射关系。
    支持多级页表结构（这里简化为单级）。
    """

    def __init__(self):
        # 虚拟页 → 物理页的映射
        # key: (tensor_key, page_index)
        # value: physical_page_id
        self.mapping: Dict[Tuple[Tuple, int], int] = {}

        # 反向映射：物理页 → 虚拟页列表（用于 LRU 淘汰）
        self.reverse_mapping: Dict[int, Set[Tuple[Tuple, int]]] = {}

    def map(self, tensor_key: Tuple, page_index: int, physical_page: int):
        """建立虚拟页到物理页的映射"""
        virtual_key = (tensor_key, page_index)
        self.mapping[virtual_key] = physical_page

        if physical_page not in self.reverse_mapping:
            self.reverse_mapping[physical_page] = set()
        self.reverse_mapping[physical_page].add(virtual_key)

    def unmap(self, tensor_key: Tuple, page_index: int) -> Optional[int]:
        """删除映射"""
        virtual_key = (tensor_key, page_index)
        if virtual_key not in self.mapping:
            return None

        physical_page = self.mapping.pop(virtual_key)

        # 更新反向映射
        if physical_page in self.reverse_mapping:
            self.reverse_mapping[physical_page].discard(virtual_key)
            if not self.reverse_mapping[physical_page]:
                del self.reverse_mapping[physical_page]

        return physical_page

    def lookup(self, tensor_key: Tuple, page_index: int) -> Optional[int]:
        """查找虚拟页对应的物理页"""
        virtual_key = (tensor_key, page_index)
        return self.mapping.get(virtual_key)

    def clear(self):
        """清空页表"""
        self.mapping.clear()
        self.reverse_mapping.clear()


class _Packed:
    """pack_hook 返回值：v1.4 增加分页信息"""
    __slots__ = ('cpu_tensor', 'device', 'dtype', 'requires_grad',
                 'quantized', 'block_info', 'prefetched', 'cache_key',
                 'page_ids', 'page_offset', 'paged_mode')

    def __init__(self, cpu_tensor: torch.Tensor,
                 device: torch.device,
                 dtype: torch.dtype,
                 requires_grad: bool,
                 quantized: bool = False,
                 block_info: Optional[Dict] = None,
                 cache_key: Optional[Tuple] = None,
                 page_ids: Optional[List[int]] = None,
                 page_offset: int = 0,
                 paged_mode: bool = False):
        self.cpu_tensor = cpu_tensor
        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.quantized = quantized
        self.block_info = block_info or {}
        self.prefetched = None  # 预取的tensor
        self.cache_key = cache_key

        # v1.4: 分页信息
        self.page_ids = page_ids or []  # 占用的物理页号列表
        self.page_offset = page_offset  # 在第一个页内的字节偏移
        self.paged_mode = paged_mode    # 是否使用分页模式


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
    DeepSeek-inspired 激活值 CPU offload with paged memory management.
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

    # v1.4: 分页内存管理组件
    page_allocator: Optional[_PageAllocator] = None
    page_table: Optional[_PageTable] = None

    if cfg.enable_paged_memory:
        page_allocator = _PageAllocator(
            page_size=cfg.page_size,
            max_pages=cfg.max_pages,
            verbose=cfg.verbose
        )
        page_table = _PageTable()

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

    def _split_first_dim(t: torch.Tensor, block_size: int) -> List[torch.Tensor]:
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

    def _store_in_pages(t: torch.Tensor, tensor_key: Tuple) -> Optional[_Packed]:
        """
        将 tensor 数据存储到固定大小的页中

        简化实现：每个 tensor 完整存储在一个页中
        """
        if page_allocator is None or page_table is None:
            return None

        try:
            # 计算需要的页数（每个 tensor 完整存一页）
            num_pages = 1

            # 分配物理页
            page_ids = page_allocator.allocate(num_pages)
            if page_ids is None:
                if cfg.verbose:
                    print(f"[VirtualVRAM] ⚠️  分页分配失败，fallback 到普通模式")
                return None

            # 将整个 tensor 展平存储到页中（保持原始 dtype 和形状信息）
            cpu_copy = t.detach().cpu()
            page_allocator.set_page(page_ids[0], cpu_copy)

            # 更新页表映射
            page_table.map(tensor_key, 0, page_ids[0])

            if cfg.verbose:
                stats = page_allocator.get_stats()
                print(f"[VirtualVRAM] 📄 分页存储: {num_pages} 页 "
                      f"({nbytes/1024/1024:.2f}MB) → 页 {page_ids}, "
                      f"内存使用: {stats['usage_ratio']:.1%}")

            # 创建 paged mode 的 _Packed
            packed = _Packed(
                cpu_tensor=t.detach().cpu(),  # 保留完整副本用于 fallback
                device=t.device,
                dtype=t.dtype,
                requires_grad=bool(t.requires_grad),
                quantized=False,
                cache_key=tensor_key,
                page_ids=page_ids,
                page_offset=0,
                paged_mode=True
            )

            return packed

        except Exception as e:
            if cfg.verbose:
                print(f"[VirtualVRAM] ⚠️  分页存储失败: {e}")
            return None

    def _load_from_pages(packed: _Packed) -> Optional[torch.Tensor]:
        """
        从页中恢复 tensor 数据（简化版：每个 tensor 完整存储）
        """
        if page_allocator is None or page_table is None:
            return None

        if not packed.paged_mode or not packed.page_ids:
            return None

        try:
            tensor_key = packed.cache_key
            if tensor_key is None:
                return None

            # 从页加载完整的 tensor
            page_data = page_allocator.get_page(packed.page_ids[0])
            if page_data is None:
                if cfg.verbose:
                    print(f"[VirtualVRAM] ⚠️  页 {packed.page_ids[0]} 数据丢失")
                return None

            # 直接恢复（保持形状和 dtype）
            restored = page_data.to(device=packed.device, non_blocking=False)
            restored.requires_grad_(packed.requires_grad)

            if cfg.verbose:
                mb = restored.numel() * restored.element_size() / 1024 / 1024
                print(f"[VirtualVRAM] 📖 分页加载: {mb:.2f}MB 从 {len(packed.page_ids)} 页")

            return restored

        except Exception as e:
            if cfg.verbose:
                print(f"[VirtualVRAM] ⚠️  分页加载失败: {e}")
            return None

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

        # ── v1.4: 优先尝试分页存储 ─────────────────────────────────────
        if cfg.enable_paged_memory:
            try:
                paged_packed = _store_in_pages(t, key)
                if paged_packed is not None:
                    cache[key] = paged_packed

                    # 添加到预取队列
                    if prefetcher:
                        prefetcher.add(paged_packed)

                    return paged_packed
            except Exception as e:
                if cfg.verbose:
                    print(f"[VirtualVRAM] ⚠️  分页存储失败，fallback: {e}")

        # ── D2H：分块或整体（fallback）───────────────────────────────────
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

        # v1.4: 优先尝试从分页加载
        if packed.paged_mode:
            try:
                restored = _load_from_pages(packed)
                if restored is not None:
                    return restored
            except Exception as e:
                if cfg.verbose:
                    print(f"[VirtualVRAM] ⚠️  分页加载失败，fallback: {e}")

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
        if page_table:
            page_table.clear()
        if prefetcher:
            prefetcher.stop()

        # 打印统计信息
        if page_allocator and cfg.verbose:
            stats = page_allocator.get_stats()
            print(f"[VirtualVRAM] 📊 Page Allocator 统计:")
            print(f"  - 分配次数: {stats['allocations']}")
            print(f"  - 释放次数: {stats['deallocations']}")
            print(f"  - 淘汰次数: {stats['evictions']}")
            print(f"  - 峰值使用: {stats['peak_usage']} 页 ({stats['peak_usage']*cfg.page_size/1024/1024:.2f}MB)")
            print(f"  - 当前使用: {stats['used_pages']} 页 ({stats['usage_ratio']:.1%})")
