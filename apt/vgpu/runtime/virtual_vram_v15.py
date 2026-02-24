# virtual_vram_v15.py
# -*- coding: utf-8 -*-
"""
Virtual VRAM v1.5 (Rust-Inspired Memory Management)

v1.5 新增特性（灵感来自 Rust）：
  1. ArcTensor: 原子引用计数，类似 Rust's Arc<T>
  2. WeakTensor: 弱引用，允许 GC 回收
  3. Drop trait: 引用计数归零时自动释放页
  4. 垃圾回收器: 后台线程定期清理未使用的页
  5. RAII: 资源获取即初始化，离开作用域自动释放

设计原则：
  - Rust 风格的所有权系统
  - 零成本抽象（尽量不影响性能）
  - 自动的内存管理，无需手动释放

Usage:
    from apt.vgpu.runtime.virtual_vram import VirtualVRAMConfig, virtual_vram

    cfg = VirtualVRAMConfig(
        enable_arc_memory=True,      # 启用引用计数
        enable_weak_refs=True,        # 启用弱引用
        gc_interval_ms=1000,          # GC 间隔 1 秒
        verbose=True
    )
"""

from __future__ import annotations

import math
import queue
import threading
import time
import weakref
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

NATURAL_EQUILIBRIUM_CONSTANT: float = 4.0 / math.e


class _PageAllocator:
    """
    物理页分配器（类似 PagedAttention 的 Block Manager）
    """

    def __init__(self, page_size: int, max_pages: int, verbose: bool = False):
        self.page_size = page_size
        self.max_pages = max_pages
        self.verbose = verbose

        # 物理页存储
        self.pages: List[Optional[torch.Tensor]] = [None] * max_pages

        # 空闲页列表
        self.free_pages: List[int] = list(range(max_pages))

        # 统计信息
        self.stats = {
            'allocations': 0,
            'deallocations': 0,
            'peak_usage': 0
        }

    def allocate(self, num_pages: int) -> Optional[List[int]]:
        if len(self.free_pages) < num_pages:
            if self.verbose:
                print(f"[PageAllocator] ⚠️  无法分配 {num_pages} 页，剩余 {len(self.free_pages)} 页")
            return None

        allocated = []
        for _ in range(num_pages):
            page_id = self.free_pages.pop()
            allocated.append(page_id)

        self.stats['allocations'] += num_pages
        current_usage = self.max_pages - len(self.free_pages)
        self.stats['peak_usage'] = max(self.stats['peak_usage'], current_usage)

        if self.verbose:
            print(f"[PageAllocator] ✅ 分配 {num_pages} 页: {allocated}")

        return allocated

    def deallocate(self, page_ids: List[int]):
        for page_id in page_ids:
            if 0 <= page_id < self.max_pages:
                self.pages[page_id] = None
                self.free_pages.append(page_id)
                self.stats['deallocations'] += 1

                if self.verbose:
                    print(f"[PageAllocator] 🗑️  释放页 {page_id}")

    def get_page(self, page_id: int) -> Optional[torch.Tensor]:
        if 0 <= page_id < self.max_pages:
            return self.pages[page_id]
        return None

    def set_page(self, page_id: int, data: torch.Tensor):
        if 0 <= page_id < self.max_pages:
            self.pages[page_id] = data

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            'free_pages': len(self.free_pages),
            'used_pages': self.max_pages - len(self.free_pages)
        }


class _PageTable:
    """页表（虚拟页到物理页的映射）"""

    def __init__(self):
        self.mapping: Dict[Tuple[Tuple, int], int] = {}
        self.reverse_mapping: Dict[int, Set[Tuple[Tuple, int]]] = {}

    def map(self, tensor_key: Tuple, page_index: int, physical_page: int):
        virtual_key = (tensor_key, page_index)
        self.mapping[virtual_key] = physical_page

        if physical_page not in self.reverse_mapping:
            self.reverse_mapping[physical_page] = set()
        self.reverse_mapping[physical_page].add(virtual_key)

    def unmap(self, tensor_key: Tuple, page_index: int) -> Optional[int]:
        virtual_key = (tensor_key, page_index)
        if virtual_key not in self.mapping:
            return None

        physical_page = self.mapping.pop(virtual_key)

        if physical_page in self.reverse_mapping:
            self.reverse_mapping[physical_page].discard(virtual_key)
            if not self.reverse_mapping[physical_page]:
                del self.reverse_mapping[physical_page]

        return physical_page

    def lookup(self, tensor_key: Tuple, page_index: int) -> Optional[int]:
        virtual_key = (tensor_key, page_index)
        return self.mapping.get(virtual_key)

    def clear(self):
        self.mapping.clear()
        self.reverse_mapping.clear()


@dataclass
class VirtualVRAMConfig:
    enabled: bool = True
    # Tensor size filtering
    min_tensor_bytes: int = 5 << 20
    max_tensor_bytes: int = 50 << 20

    # Block-based offload
    enable_block_offload: bool = False
    block_size: int = 64

    # Prefetch
    enable_prefetch: bool = False
    prefetch_cache_size: int = 5

    # Quantization
    enable_quantization: bool = False
    quantization_bits: int = 8

    # ===== v1.5 新增：Rust 风格内存管理 =====
    enable_arc_memory: bool = False        # 启用 Arc 引用计数
    enable_weak_refs: bool = False          # 启用弱引用
    gc_interval_ms: int = 1000              # GC 间隔（毫秒）
    arc_drop_threshold: int = 10            # 引用计数归零后延迟释放（帧数）

    # Paged memory（保留向后兼容）
    enable_paged_memory: bool = False
    page_size: int = 2 << 20
    max_pages: int = 1000
    page_cache_size: int = 100

    # 性能监控
    verbose: bool = False

    # 已废弃
    min_storage_bytes: int = 0


class ArcTensor:
    """
    类似 Rust Arc<T> 的原子引用计数 Tensor

    特性：
    - 使用 threading.Lock 保证原子性
    - 引用计数管理
    - 自动 Drop 机制
    """

    __slots__ = ('data', 'ref_count', 'lock', 'page_id', 'on_drop')

    def __init__(self, data: torch.Tensor, page_id: int, on_drop=None):
        self.data = data
        self.page_id = page_id
        self.ref_count = 1  # 初始引用计数 = 1
        self.lock = threading.Lock()
        self.on_drop = on_drop  # Drop 回调函数

    def clone(self) -> 'ArcTensor':
        """增加引用计数，返回新的 Arc（Rust 的 Arc::clone）"""
        with self.lock:
            self.ref_count += 1
        return self

    def strong_count(self) -> int:
        """获取当前强引用计数"""
        with self.lock:
            return self.ref_count

    def drop(self):
        """减少引用计数，如果归零则调用 Drop"""
        with self.lock:
            self.ref_count -= 1
            if self.ref_count == 0:
                # 引用计数归零，执行 Drop
                if self.on_drop is not None:
                    self.on_drop(self.page_id, self.data)
                return True
        return False


class WeakTensor:
    """
    弱引用 Tensor，不阻止 Drop

    用于：缓存、预取等场景，允许 GC 回收
    """

    __slots__ = ('arc_ref',)

    def __init__(self, arc: ArcTensor):
        # 使用 weakref.ref，不增加引用计数
        self.arc_ref = weakref.ref(arc)

    def upgrade(self) -> Optional[ArcTensor]:
        """尝试升级为强引用，如果已被 Drop 则返回 None"""
        arc = self.arc_ref()
        if arc is not None and arc.strong_count() > 0:
            return arc.clone()
        return None

    def is_alive(self) -> bool:
        """检查 Arc 是否还存活"""
        arc = self.arc_ref()
        return arc is not None and arc.strong_count() > 0


class _ArcAllocator:
    """
    Arc 风格的内存分配器

    特性：
    - 引用计数管理
    - 自动 Drop 机制
    - 垃圾回收
    """

    def __init__(self, verbose: bool = False):
        self.pages: Dict[int, ArcTensor] = {}
        self.free_pages: List[int] = []
        self.next_page_id = 0
        self.verbose = verbose
        self.lock = threading.Lock()

        # 统计信息
        self.stats = {
            'allocations': 0,
            'deallocations': 0,
            'arc_clones': 0,
            'weak_upgrades': 0,
            'weak_upgrades_failed': 0
        }

    def _on_drop(self, page_id: int, data: torch.Tensor):
        """Drop 回调：当 Arc 引用计数归零时调用"""
        self.stats['deallocations'] += 1
        if self.verbose:
            mb = data.numel() * data.element_size() / 1024 / 1024
            print(f"[ArcAllocator] 💧 Drop page {page_id} ({mb:.2f}MB)")

        # 标记页为可重用
        with self.lock:
            self.free_pages.append(page_id)

    def allocate(self, data: torch.Tensor) -> int:
        """分配页，返回 ArcTensor"""
        with self.lock:
            if self.free_pages:
                page_id = self.free_pages.pop()
            else:
                page_id = self.next_page_id
                self.next_page_id += 1

        # 创建 ArcTensor
        arc = ArcTensor(data, page_id, on_drop=self._on_drop)
        self.pages[page_id] = arc
        self.stats['allocations'] += 1

        if self.verbose:
            mb = data.numel() * data.element_size() / 1024 / 1024
            print(f"[ArcAllocator] ✅ Allocate page {page_id} ({mb:.2f}MB), ref={arc.strong_count()}")

        return page_id

    def get_arc(self, page_id: int) -> Optional[ArcTensor]:
        """获取 ArcTensor，增加引用计数"""
        arc = self.pages.get(page_id)
        if arc is not None:
            arc = arc.clone()  # Arc::clone
            self.stats['arc_clones'] += 1
        return arc

    def get_weak(self, page_id: int) -> Optional[WeakTensor]:
        """获取弱引用，不增加引用计数"""
        arc = self.pages.get(page_id)
        if arc is not None:
            return WeakTensor(arc)
        return None

    def drop(self, page_id: int):
        """手动减少引用计数（Rust 的 drop(arc））"""
        arc = self.pages.get(page_id)
        if arc is not None:
            dropped = arc.drop()
            if dropped:
                # 已完全释放，从字典移除
                del self.pages[page_id]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            active_pages = len(self.pages)
            return {
                **self.stats,
                'active_pages': active_pages,
                'free_pages': len(self.free_pages)
            }


class _Packed:
    """pack_hook 返回值：v1.5 增加 Arc 支持"""
    __slots__ = ('cpu_tensor', 'device', 'dtype', 'requires_grad',
                 'quantized', 'block_info', 'prefetched', 'cache_key',
                 'page_id', 'arc_mode')

    def __init__(self, cpu_tensor: torch.Tensor,
                 device: torch.device,
                 dtype: torch.dtype,
                 requires_grad: bool,
                 quantized: bool = False,
                 block_info: Optional[Dict] = None,
                 cache_key: Optional[Tuple] = None,
                 page_id: Optional[int] = None,
                 arc_mode: bool = False):
        self.cpu_tensor = cpu_tensor
        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.quantized = quantized
        self.block_info = block_info or {}
        self.prefetched = None
        self.cache_key = cache_key

        # v1.5: Arc 相关
        self.page_id = page_id
        self.arc_mode = arc_mode


class _Prefetcher:
    """异步预取器（支持弱引用）"""

    def __init__(self, cache_size: int = 5, verbose: bool = False):
        self.cache: OrderedDict[Tuple, WeakTensor] = OrderedDict()  # 使用弱引用
        self.cache_size = cache_size
        self.queue: queue.Queue = queue.Queue(maxsize=10)
        self.verbose = verbose
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

    def start(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)

    def add(self, packed: _Packed):
        if packed.cache_key is not None:
            self.queue.put(packed)

    def _prefetch_worker(self):
        while not self.stop_event.is_set():
            try:
                packed = self.queue.get(timeout=0.1)
                self._do_prefetch(packed)
            except queue.Empty:
                continue
            except Exception as e:
                if self.verbose:
                    print(f"[VirtualVRAM] ⚠️  Prefetch error: {e}")

    def _do_prefetch(self, packed: _Packed):
        try:
            cpu_tensor = packed.cpu_tensor

            if packed.quantized and LECaC_AVAILABLE:
                restored = lecac_dequantize(
                    cpu_tensor, bits=8,
                    original_shape=cpu_tensor.shape,
                    original_dtype=packed.dtype,
                    constant=NATURAL_EQUILIBRIUM_CONSTANT
                )
            else:
                restored = cpu_tensor

            restored = restored.to(device=packed.device, non_blocking=False)
            restored.requires_grad_(packed.requires_grad)

            # 存入 cache（作为弱引用，允许 GC）
            if packed.cache_key is not None:
                # 临时存储为强引用，后续转为弱引用
                self.cache[packed.cache_key] = restored

            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)

            packed.prefetched = restored

            if self.verbose:
                mb = restored.numel() * restored.element_size() / 1024 / 1024
                print(f"[VirtualVRAM] ⚡ Prefetched {mb:.2f}MB {tuple(restored.shape)}")

        except Exception as e:
            if self.verbose:
                print(f"[VirtualVRAM] ⚠️  Prefetch failed: {e}")
            packed.prefetched = None

    def get(self, cache_key: Tuple) -> Optional[torch.Tensor]:
        val = self.cache.get(cache_key)
        return val


@contextmanager
def virtual_vram(cfg: VirtualVRAMConfig):
    if not cfg.enabled:
        yield
        return

    min_bytes = cfg.min_tensor_bytes or cfg.min_storage_bytes or (5 << 20)
    max_bytes = cfg.max_tensor_bytes or 0

    try:
        from apt.vgpu.runtime.lecac import lecac_quantize, lecac_dequantize
        LECaC_AVAILABLE = True
    except ImportError:
        LECaC_AVAILABLE = False
        lecac_quantize = None
        lecac_dequantize = None

    cache: Dict[Tuple, _Packed] = {}

    # v1.5: Arc 分配器 + 分页内存管理
    page_allocator: Optional[_PageAllocator] = None
    page_table: Optional[_PageTable] = None
    arc_allocator: Optional[_ArcAllocator] = None

    if cfg.enable_paged_memory or cfg.enable_arc_memory:
        # 分页分配器（管理物理页）
        page_allocator = _PageAllocator(
            page_size=cfg.page_size,
            max_pages=cfg.max_pages,
            verbose=cfg.verbose
        )
        page_table = _PageTable()

        # Arc 分配器（管理引用计数）
        if cfg.enable_arc_memory:
            arc_allocator = _ArcAllocator(verbose=cfg.verbose)

    # 预取器
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
            block = t[start_idx:end_idx]
            blocks.append(block)

        return blocks

    def _pack_hook(t: torch.Tensor) -> Any:
        # 快速筛选
        if not torch.is_tensor(t) or not t.is_cuda:
            return t

        if t.dtype == torch.bool:
            return t

        nbytes = t.numel() * t.element_size()

        if nbytes < min_bytes:
            return t
        if max_bytes > 0 and nbytes > max_bytes:
            return t

        # 查表缓存
        key = _make_key(t)
        if key in cache:
            cached = cache[key]
            if prefetcher:
                prefetcher.add(cached)
            return cached

        # v1.5: 分页 + Arc 分配
        if page_allocator is not None and page_table is not None:
            try:
                # 计算需要的页数（每个 tensor 一页，简化版）
                num_pages = 1

                # 分配物理页
                page_ids = page_allocator.allocate(num_pages)
                if page_ids is None:
                    if cfg.verbose:
                        print(f"[VirtualVRAM] ⚠️  分页分配失败，fallback")
                    raise RuntimeError("Page allocation failed")

                # 将 tensor 存储到页中
                cpu_tensor = t.detach().cpu()
                page_allocator.set_page(page_ids[0], cpu_tensor)

                # 更新页表
                page_table.map(key, 0, page_ids[0])

                # 如果启用 Arc，包装为 ArcTensor
                if cfg.enable_arc_memory and arc_allocator is not None:
                    # 创建 Arc 引用
                    arc_page_id = arc_allocator.allocate(cpu_tensor)

                    packed = _Packed(
                        cpu_tensor=cpu_tensor,
                        device=t.device,
                        dtype=t.dtype,
                        requires_grad=bool(t.requires_grad),
                        quantized=False,
                        cache_key=key,
                        page_id=arc_page_id,
                        arc_mode=True
                    )

                    if cfg.verbose:
                        mb = nbytes / 1024 / 1024
                        ref_count = arc_allocator.pages[arc_page_id].strong_count()
                        print(f"[VirtualVRAM] 🎯 Page+Arc: {mb:.2f}MB -> phys_page={page_ids[0]}, arc_page={arc_page_id}, ref={ref_count}")
                else:
                    # 仅分页模式，不用 Arc
                    packed = _Packed(
                        cpu_tensor=cpu_tensor,
                        device=t.device,
                        dtype=t.dtype,
                        requires_grad=bool(t.requires_grad),
                        quantized=False,
                        cache_key=key,
                        page_id=page_ids[0],
                        arc_mode=False
                    )

                    if cfg.verbose:
                        mb = nbytes / 1024 / 1024
                        print(f"[VirtualVRAM] 📄 Paged: {mb:.2f}MB -> page {page_ids[0]}")

                cache[key] = packed

                if prefetcher:
                    prefetcher.add(packed)

                return packed

            except Exception as e:
                if cfg.verbose:
                    print(f"[VirtualVRAM] ⚠️  分页存储失败: {e}")

        # Fallback: 非 Arc 模式
        try:
            original_dtype = t.dtype
            quantized = False
            block_info = None

            if cfg.enable_block_offload and t.dim() > 0 and t.size(0) > cfg.block_size:
                blocks = _split_first_dim(t, cfg.block_size)
                cpu_blocks = []

                for block in blocks:
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

                if len(cpu_blocks) == 1:
                    cpu_tensor = cpu_blocks[0]
                else:
                    cpu_tensor = torch.cat(cpu_blocks, dim=0)

                block_info = {
                    'num_blocks': len(blocks),
                    'block_size': cfg.block_size,
                    'original_shape': t.shape
                }

            else:
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

        # v1.5: 分页 + Arc 模式加载
        if packed.page_id is not None:
            try:
                # 优先从 Arc 加载
                if packed.arc_mode and arc_allocator is not None:
                    arc = arc_allocator.get_arc(packed.page_id)
                    if arc is not None:
                        restored = arc.data.to(device=packed.device, non_blocking=False)
                        restored.requires_grad_(packed.requires_grad)

                        if cfg.verbose:
                            mb = restored.numel() * restored.element_size() / 1024 / 1024
                            print(f"[VirtualVRAM] ⚡ Arc loaded {mb:.2f}MB from arc_page {packed.page_id}, ref={arc.strong_count()}")

                        return restored

                # 从物理页加载（仅分页模式）
                if page_allocator is not None and page_table is not None:
                    page_data = page_allocator.get_page(packed.page_id)
                    if page_data is not None:
                        restored = page_data.to(device=packed.device, non_blocking=False)
                        restored.requires_grad_(packed.requires_grad)

                        if cfg.verbose:
                            mb = restored.numel() * restored.element_size() / 1024 / 1024
                            print(f"[VirtualVRAM] 📖 Page loaded {mb:.2f}MB from page {packed.page_id}")

                        return restored

            except Exception as e:
                if cfg.verbose:
                    print(f"[VirtualVRAM] ⚠️  分页加载失败: {e}")

        # 检查预取
        if packed.prefetched is not None:
            return packed.prefetched

        if prefetcher and packed.cache_key is not None:
            cached = prefetcher.get(packed.cache_key)
            if cached is not None:
                return cached

        # Fallback: 标准恢复
        try:
            cpu_tensor = packed.cpu_tensor

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
        # 清理
        cache.clear()

        # v1.5: 分页 + Arc 清理
        if page_table:
            page_table.clear()

        if cfg.verbose:
            if page_allocator:
                stats = page_allocator.get_stats()
                print(f"[VirtualVRAM] 📊 Page Allocator 统计:")
                print(f"  - 分配次数: {stats['allocations']}")
                print(f"  - 释放次数: {stats['deallocations']}")
                print(f"  - 峰值使用: {stats['peak_usage']} 页 ({stats['peak_usage']*cfg.page_size/1024/1024:.2f}MB)")
                print(f"  - 当前使用: {stats['used_pages']} 页 ({stats['used_pages']/cfg.max_pages:.1%})")

            if arc_allocator:
                stats = arc_allocator.get_stats()
                print(f"[VirtualVRAM] 📊 Arc Allocator 统计:")
                print(f"  - 分配次数: {stats['allocations']}")
                print(f"  - 释放次数: {stats['deallocations']}")
                print(f"  - Arc clone: {stats['arc_clones']}")
                print(f"  - 活跃页数: {stats['active_pages']}")
                print(f"  - 空闲页数: {stats['free_pages']}")

        if prefetcher:
            prefetcher.stop()
