# virtual_vram_v16.py
# -*- coding: utf-8 -*-
"""
Virtual VRAM v1.6 (Linear Scaling Nested Architecture)

v1.6 核心改进：线性缩放嵌套结构
  - LECaC → Page → Block → Arc (嵌套，非独立)
  - Arc 嵌入到 Block 里
  - Block 嵌入到 Page 里
  - Page 嵌入到 LECaC 里

设计原则：
  - 减少往返和管理开销
  - 单一数据结构，不是多层独立管理
  - 参考 FlashAttention 的解耦快速设计

与 v1.5 的区别：
  v1.5: Arc、Page、Block 是独立的管理层（独立开关）
  v1.6: Arc⊂Block⊂Page⊂LECaC（线性嵌套）

Usage:
    from apt.vgpu.runtime.virtual_vram import VirtualVRAMConfig, virtual_vram

    cfg = VirtualVRAMConfig(
        enable_nested_v16=True,       # 启用 v1.6 嵌套架构
        min_tensor_bytes=5 << 20,
        verbose=True
    )
"""

from __future__ import annotations

import math
import queue
import threading
import time
import weakref
import heapq
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

# ── 性能诊断统计（始终累积，极低开销；调用 reset_vram_perf_stats() 清零） ──
# pack_ms  = D2H CPU 侧提交耗时（non_blocking=True：pinned 分配 + DMA 提交，不含实际 DMA 等待）
# unpack_n = H2D 提交次数（non_blocking=True，提交即返回，不含 DMA 等待时间）
_vram_perf: Dict[str, float] = {
    "pack_n":      0.0,   # 实际发生 D2H 的次数
    "pack_bytes":  0.0,   # D2H 总字节数
    "pack_ms":     0.0,   # D2H CPU 侧提交耗时（ms，pinned 分配 + non_blocking copy_ 提交时间）
    "skip_n":      0.0,   # _pack_hook 快速跳过次数（INT8/bool/size 过滤）
    "unpack_n":    0.0,   # H2D 提交次数（non_blocking，不含实际 DMA 时间）
    "unpack_bytes":0.0,   # H2D 总字节数（估算）
}


def reset_vram_perf_stats() -> None:
    """每步（或每 log_interval）调用一次，重置累积统计。"""
    for k in _vram_perf:
        _vram_perf[k] = 0.0


def get_vram_perf_stats() -> Dict[str, float]:
    """返回当前统计快照（不重置）。"""
    return dict(_vram_perf)


class _HeatPriorityQueue:
    """
    全局热度优先队列（O(log n)更新和查询）
    只在热度变化时更新，不在unpack时遍历
    """
    def __init__(self):
        # 最小堆（存储负热度实现最大堆）
        self.heap: List[Tuple[float, int, '_NestedArcBlock']] = []
        # key到heap index的映射（用于快速更新）
        self.key_to_index: Dict[Tuple, int] = {}
        self.lock = threading.Lock()
        self._dirty = True  # 标记堆需要重建

    def add_or_update(self, block: '_NestedArcBlock', key: Tuple):
        """添加或更新块的热度"""
        with self.lock:
            # 计算当前热度
            heat = block.arc_heat_score()

            # 检查是否已存在
            if key in self.key_to_index:
                # 标记为脏，延迟重建（避免每次更新都heapify）
                self._dirty = True
            else:
                # 新块，加入堆
                heapq.heappush(self.heap, (-heat, id(block), block))
                self.key_to_index[key] = len(self.heap) - 1

    def get_top_k(self, k: int) -> List['_NestedArcBlock']:
        """获取Top K热度块（不包含当前块）"""
        with self.lock:
            if self._dirty:
                # 重建堆（O(n)，但只在脏时执行）
                self._rebuild_heap()

            # 取Top K（注意是最小堆，所以取负的）
            result = []
            temp_heap = []
            for _ in range(min(k, len(self.heap))):
                if not self.heap:
                    break
                neg_heat, _, block = heapq.heappop(self.heap)
                result.append(block)
                temp_heap.append((neg_heat, id(block), block))

            # 把取出来的放回去
            for item in temp_heap:
                heapq.heappush(self.heap, item)

            return result

    def _rebuild_heap(self):
        """重建堆（当有更新时调用）"""
        # 根据最新热度重新建堆
        new_heap = []
        for neg_heat, _, block in self.heap:
            heat = block.arc_heat_score()
            heapq.heappush(new_heap, (-heat, id(block), block))
        self.heap = new_heap
        self._dirty = False


# 全局热度优先队列
_heat_pq = _HeatPriorityQueue()


class _NestedArcBlock:
    """
    v1.6: 线性缩放嵌套块
    LECaC → Page → Block → Arc (嵌套结构)

    设计：单一数据结构，不是多层独立管理
    - quantized: LECaC 层（最外层）
    - page_id: Page 层
    - block_id: Block 层
    - arc_ref_count: Arc 层（最内层，嵌入到 Block 里）
    - last_access: 最后访问时间（用于热度预取）
    """

    __slots__ = (
        # LECaC 层（最外层）
        'quantized', 'quantization_bits',

        # Page 层
        'page_id', 'page_offset',

        # Block 层
        'block_id', 'block_offset', 'block_size',

        # Arc 层（最内层，嵌入到 Block 里）
        'arc_ref_count', 'arc_weak_refs',

        # 预取优化
        'last_access',  # 最后访问时间戳
        'access_count',  # 访问次数（热度）

        # 基础数据（共享）
        'cpu_tensor', 'device', 'dtype', 'requires_grad',

        # 异步 D2H 完成事件（non_blocking copy_ 后 record，unpack 时 synchronize）
        'dma_event',

        # 锁（用于 Arc 原子操作）
        '_lock'
    )

    def __init__(self,
                 cpu_tensor: torch.Tensor,
                 device: torch.device,
                 dtype: torch.dtype,
                 requires_grad: bool,
                 quantized: bool = False,
                 quantization_bits: int = 8,
                 page_id: int = -1,
                 page_offset: int = 0,
                 block_id: int = -1,
                 block_offset: int = 0,
                 block_size: int = 0):
        import threading
        import time

        # LECaC 层
        self.quantized = quantized
        self.quantization_bits = quantization_bits

        # Page 层
        self.page_id = page_id
        self.page_offset = page_offset

        # Block 层
        self.block_id = block_id
        self.block_offset = block_offset
        self.block_size = block_size

        # Arc 层（嵌入到 Block 里）
        self.arc_ref_count = 1  # 初始引用计数 = 1
        self.arc_weak_refs = []  # 弱引用列表

        # 预取优化：热度追踪
        self.last_access = time.time()  # 最后访问时间
        self.access_count = 1  # 初始访问次数

        # 基础数据
        self.cpu_tensor = cpu_tensor
        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad

        # 异步 D2H 完成事件：pack 时 record，unpack/prefetch 时 synchronize
        # None 表示 D2H 是同步完成的（fallback 路径），无需等待
        self.dma_event: Optional[torch.cuda.Event] = None

        # Arc 原子操作锁
        self._lock = threading.Lock()

    def arc_clone(self, key: Tuple) -> '_NestedArcBlock':
        """Arc::clone - 增加引用计数，更新全局热度队列"""
        with self._lock:
            self.arc_ref_count += 1
            self.access_count += 1  # 增加访问次数
            self.last_access = time.time()  # 更新访问时间

        # 更新全局热度优先队列（O(1)标记脏）
        _heat_pq.add_or_update(self, key)
        return self

    def arc_drop(self) -> bool:
        """Arc::drop - 减少引用计数，返回是否归零"""
        with self._lock:
            self.arc_ref_count -= 1
            return self.arc_ref_count == 0

    def arc_strong_count(self) -> int:
        """获取强引用计数"""
        with self._lock:
            return self.arc_ref_count

    def arc_heat_score(self) -> float:
        """
        计算热度分数（用于预取优先级）
        使用**曲线增长**而非线性：热度随引用次数超线性增长

        热度 = (access_count * arc_ref_count)^2 * time_decay
        或者：exp(access_count) * arc_ref_count * time_decay
        """
        import time
        with self._lock:
            age = time.time() - self.last_access
            # 时间衰减：越新的访问权重越高
            time_decay = 1.0 / (1.0 + age * 0.1)

            # **曲线增长**：引用越多，热度呈指数增长
            # 使用平方函数：heat = (count * ref)^2
            linear_score = self.access_count * self.arc_ref_count
            heat = (linear_score ** 2) * time_decay  # 平方增长

            # 或者使用对数指数增长（更平滑）
            # heat = math.exp(linear_score * 0.1) * self.arc_ref_count * time_decay

            return heat

    def arc_add_weak_ref(self, ref):
        """添加弱引用"""
        self.arc_weak_refs.append(ref)


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

    # ===== v1.6 新增：线性缩放嵌套架构 =====
    enable_nested_v16: bool = False        # 启用 v1.6 嵌套架构 (LECaC→Page→Block→Arc)
    nested_block_size: int = 64            # 嵌套块大小
    nested_quantization_bits: int = 8      # ⚠️  已弃用：saved_tensors_hooks 中量化会破坏 backward
                                           # 保留字段仅为接口兼容，实际不再使用

    # Block-based offload (保留向后兼容)
    enable_block_offload: bool = False
    block_size: int = 64

    # Prefetch
    enable_prefetch: bool = False
    prefetch_cache_size: int = 5

    # Quantization (已弃用：saved_tensors_hooks 中有损量化会破坏 backward 数值稳定性)
    enable_quantization: bool = False      # ⚠️  已弃用，设为 True 不再生效
    quantization_bits: int = 8             # ⚠️  已弃用

    # ===== v1.5 保留：向后兼容 =====
    enable_arc_memory: bool = False        # v1.5: 启用 Arc 引用计数
    enable_weak_refs: bool = False          # v1.5: 启用弱引用
    gc_interval_ms: int = 1000              # v1.5: GC 间隔
    arc_drop_threshold: int = 10            # v1.5: 延迟释放

    # Paged memory（保留向后兼容）
    enable_paged_memory: bool = False
    page_size: int = 2 << 20
    max_pages: int = 1000
    page_cache_size: int = 100

    # 性能监控
    verbose: bool = False
    # 调试模式：开启后打印 [DEBUG PACK/UNPACK/PREFETCH] 详细追踪日志
    # 训练时务必保持 False，否则每次 backward 均会打印大量日志
    debug: bool = False

    # 主 cache 最大条目数（LRU 淘汰）
    # 每个训练步骤的激活 tensor 都有新的 data_ptr，cache 会无限增长。
    # 设置上限防止 CPU RAM 耗尽：默认 2000 条，覆盖 ~2-5 步的激活值，
    # 足够 autograd 一步 forward+backward 内去重，多余的旧条目被淘汰。
    max_cache_entries: int = 2000

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
    """
    pack_hook 返回值：v1.6 增加嵌套结构支持

    支持三种模式：
    1. v1.5 独立模式：page_id, arc_mode（向后兼容）
    2. v1.6 嵌套模式：nested_block（单一嵌套结构）
    3. 传统模式：仅有 cpu_tensor
    """
    __slots__ = ('cpu_tensor', 'device', 'dtype', 'requires_grad',
                 'quantized', 'block_info', 'prefetched', 'cache_key',
                 'page_id', 'arc_mode',
                 'nested_block')  # v1.6: 嵌套块

    def __init__(self, cpu_tensor: torch.Tensor,
                 device: torch.device,
                 dtype: torch.dtype,
                 requires_grad: bool,
                 quantized: bool = False,
                 block_info: Optional[Dict] = None,
                 cache_key: Optional[Tuple] = None,
                 page_id: Optional[int] = None,
                 arc_mode: bool = False,
                 nested_block: Optional[_NestedArcBlock] = None):  # v1.6
        self.cpu_tensor = cpu_tensor
        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.quantized = quantized
        self.block_info = block_info or {}
        self.prefetched = None
        self.cache_key = cache_key

        # v1.5: 独立 Arc 模式（向后兼容）
        self.page_id = page_id
        self.arc_mode = arc_mode

        # v1.6: 嵌套块模式（新）
        self.nested_block = nested_block


class _Prefetcher:
    """异步预取器（支持弱引用）"""

    def __init__(self, cache_size: int = 5, verbose: bool = False, debug: bool = False):
        self.cache: OrderedDict[Tuple, WeakTensor] = OrderedDict()  # 使用弱引用
        self.cache_size = cache_size
        self.queue: queue.Queue = queue.Queue(maxsize=10)
        self.verbose = verbose
        self.debug = debug
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
            # v1.6: 支持嵌套块模式
            if packed.nested_block is not None:
                # 从嵌套块获取数据
                nested = packed.nested_block

                # 等待异步 D2H 完成（prefetch 线程可能早于 unpack_hook 访问 cpu_tensor）
                # 先存局部变量再 synchronize()，避免主线程并发置 None 后拿到 NoneType。
                _ev = nested.dma_event
                if _ev is not None:
                    _ev.synchronize()
                    nested.dma_event = None

                cpu_tensor = nested.cpu_tensor
                target_device = nested.device
                requires_grad = nested.requires_grad
                original_dtype = nested.dtype

                # LECaC 去量化
                if nested.quantized and LECaC_AVAILABLE:
                    restored = lecac_dequantize(
                        cpu_tensor,
                        bits=nested.quantization_bits,
                        original_shape=cpu_tensor.shape,
                        original_dtype=original_dtype,
                        constant=NATURAL_EQUILIBRIUM_CONSTANT
                    )
                else:
                    restored = cpu_tensor

                if self.debug:
                    print(f'[DEBUG _do_prefetch NESTED] Before to: tensor dtype={restored.dtype}, target dtype={original_dtype}, device={target_device}')
                restored = restored.to(device=target_device, dtype=original_dtype, non_blocking=False)
                if self.debug:
                    print(f'[DEBUG _do_prefetch NESTED] After to: tensor dtype={restored.dtype}, shape={restored.shape}, requires_grad={restored.requires_grad}')
                restored.requires_grad_(requires_grad)
                if self.debug:
                    print(f'[DEBUG _do_prefetch NESTED] After requires_grad: dtype={restored.dtype}, requires_grad={restored.requires_grad}')
            else:
                # 传统模式
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

                if self.debug:
                    print(f'[DEBUG _do_prefetch LEGACY] Before to: tensor dtype={restored.dtype}, target dtype={packed.dtype}, device={packed.device}')
                restored = restored.to(device=packed.device, dtype=packed.dtype, non_blocking=False)
                if self.debug:
                    print(f'[DEBUG _do_prefetch LEGACY] After to: tensor dtype={restored.dtype}, shape={restored.shape}, requires_grad={restored.requires_grad}')
                restored.requires_grad_(packed.requires_grad)
                if self.debug:
                    print(f'[DEBUG _do_prefetch LEGACY] After requires_grad: dtype={restored.dtype}, requires_grad={restored.requires_grad}')

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

    cache: "OrderedDict[Tuple, _Packed]" = OrderedDict()

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
            verbose=cfg.verbose,
            debug=cfg.debug,
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
            _vram_perf["skip_n"] += 1
            return t

        # INT8/uint8 tensor（如 LECaC 的 x_q）已是紧凑整数存储（1 byte/element），
        # 再走 VirtualVRAM 会触发二次量化（GPU kernel + .item() 强制同步）+ D2H/H2D，
        # 开销远超收益，直接跳过。
        if t.dtype in (torch.int8, torch.uint8):
            _vram_perf["skip_n"] += 1
            return t

        nbytes = t.numel() * t.element_size()

        if nbytes < min_bytes:
            _vram_perf["skip_n"] += 1
            return t
        if max_bytes > 0 and nbytes > max_bytes:
            _vram_perf["skip_n"] += 1
            return t

        # 不 offload nn.Parameter 的 view/slice
        # MoELayer 里 W1_e = self.W1[e_idx] 是 Parameter 的 index slice，
        # 其 grad_fn 的直接父节点是 AccumulateGrad(Parameter)。
        # 这类 tensor 已常驻 GPU，offload 只会带来 D2H/H2D 往返开销，无收益。
        # 激活值（attention 输出、FFN 中间结果）的 grad_fn 父节点不会是 AccumulateGrad，
        # 不受此筛选影响。
        if t.requires_grad and t.grad_fn is not None:
            for _fn, _ in t.grad_fn.next_functions:
                if _fn is not None and type(_fn).__name__ == 'AccumulateGrad':
                    _vram_perf["skip_n"] += 1
                    return t

        # 通过所有过滤器 → 将进行实际 D2H，开始计时
        _pack_t0 = time.perf_counter()

        # 查表缓存
        key = _make_key(t)
        if key in cache:
            cached = cache[key]
            # v1.6: 如果是嵌套块，增加引用计数并更新热度队列
            if cached.nested_block is not None:
                cached.nested_block.arc_clone(key)
                if cfg.verbose:
                    print(f"[VirtualVRAM v1.6] 🔗 Cache hit + Arc clone: ref={cached.nested_block.arc_strong_count()}")
            if prefetcher:
                prefetcher.add(cached)
            return cached

        # ===== v1.6: 嵌套模式（LECaC → Page → Block → Arc） =====
        if cfg.enable_nested_v16:
            try:
                block_size = cfg.nested_block_size
                original_dtype = t.dtype
                tensor_size_mb = nbytes / (1024 * 1024)

                # v1.7 量化策略：
                # 浮点 tensor（Attention Q/K/V 等激活值）→ 不量化，无损 D2H
                #   理由：float→INT8→float 有损还原引入误差 ε，大模型梯度被放大 → FlashAttn NaN
                # 整数 tensor（LECACLinear 的 x_q，已被 pack_hook 早期过滤跳过）→ 理论可量化
                #   实际：int8/uint8 在早期过滤处已 return t，此处 should_quantize 永远为 False
                is_float_dtype = t.dtype in (torch.float32, torch.float16, torch.bfloat16)
                should_quantize = (LECaC_AVAILABLE and
                                   cfg.nested_quantization_bits > 0 and
                                   tensor_size_mb >= 5.0 and
                                   not is_float_dtype)

                # ── 异步 D2H ──────────────────────────────────────────────────
                # non_blocking=True：copy_ 提交 DMA 后立即返回，GPU 计算继续；
                # dma_event.record() 记录提交点，unpack_hook 里 synchronize() 等待完成。
                # 效果：D2H 与后续 forward 层的 GPU 计算重叠，消除同步阻塞。
                dma_event: Optional[torch.cuda.Event] = None

                if should_quantize:
                    # INT 类型量化（实际上不会走到这里，见上方注释）
                    quantized_gpu = lecac_quantize(
                        t.detach(),
                        bits=cfg.nested_quantization_bits,
                        constant=NATURAL_EQUILIBRIUM_CONSTANT
                    )
                    try:
                        cpu_tensor = torch.empty(quantized_gpu.shape, dtype=quantized_gpu.dtype, pin_memory=True)
                        cpu_tensor.copy_(quantized_gpu, non_blocking=True)
                        dma_event = torch.cuda.Event()
                        dma_event.record()
                    except Exception:
                        cpu_tensor = quantized_gpu.cpu()
                    quantized = True
                    if cfg.verbose:
                        print(f"[VirtualVRAM v1.6] 🔢 INT量化D2H(async): {tensor_size_mb:.2f}MB {t.dtype}→INT{cfg.nested_quantization_bits}")
                else:
                    # 浮点激活值：pinned memory + 异步 D2H
                    try:
                        cpu_tensor = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
                        cpu_tensor.copy_(t.detach(), non_blocking=True)  # 异步 D2H，不阻塞 GPU
                        dma_event = torch.cuda.Event()
                        dma_event.record()  # 记录 DMA 提交位置
                    except Exception:
                        cpu_tensor = t.detach().cpu()  # fallback：pinned 内存不足，同步 D2H
                    quantized = False
                    if cfg.verbose:
                        print(f"[VirtualVRAM v1.6] 📤 无损D2H(async pinned): {tensor_size_mb:.2f}MB {t.dtype}")

                # 创建嵌套块（包含 Page → Block → Arc）
                nested_block = _NestedArcBlock(
                    cpu_tensor=cpu_tensor,
                    device=t.device,
                    dtype=original_dtype,  # 保存真实dtype
                    requires_grad=bool(t.requires_grad),
                    quantized=quantized,
                    quantization_bits=cfg.nested_quantization_bits,
                    page_id=len(cache),  # 简化：用 cache 索引作为 page_id
                    page_offset=0,
                    block_id=len(cache),  # 简化：用 cache 索引作为 block_id
                    block_offset=0,
                    block_size=block_size
                )
                nested_block.dma_event = dma_event  # 异步 D2H 完成事件

                packed = _Packed(
                    cpu_tensor=cpu_tensor,
                    device=t.device,
                    dtype=original_dtype,  # 保存真实dtype
                    requires_grad=bool(t.requires_grad),
                    quantized=quantized,
                    cache_key=key,
                    nested_block=nested_block  # v1.6: 使用嵌套块
                )

                cache[key] = packed
                # LRU 淘汰：防止 cache 跨步骤无限增长（每步激活 tensor 的 data_ptr 不同）
                if len(cache) > cfg.max_cache_entries:
                    cache.popitem(last=False)

                # 添加到全局热度优先队列
                _heat_pq.add_or_update(nested_block, key)

                if prefetcher:
                    prefetcher.add(packed)

                _vram_perf["pack_n"]     += 1
                _vram_perf["pack_bytes"] += nbytes
                _vram_perf["pack_ms"]    += (time.perf_counter() - _pack_t0) * 1000

                if cfg.verbose:
                    mb = nbytes / 1024 / 1024
                    print(f"[VirtualVRAM v1.6] ✅ Nested D2H: {mb:.2f}MB {tuple(t.shape)} "
                          f"block={nested_block.block_id} ref={nested_block.arc_strong_count()} "
                          f"heat={nested_block.arc_heat_score():.2f}")

                return packed

            except Exception as e:
                if cfg.verbose:
                    print(f"[VirtualVRAM v1.6] ⚠️ 嵌套存储失败: {e}")

        # ===== v1.5: 独立分页 + Arc 分配（向后兼容） =====
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
                # LRU 淘汰
                if len(cache) > cfg.max_cache_entries:
                    cache.popitem(last=False)

                if prefetcher:
                    prefetcher.add(packed)

                _vram_perf["pack_n"]     += 1
                _vram_perf["pack_bytes"] += nbytes
                _vram_perf["pack_ms"]    += (time.perf_counter() - _pack_t0) * 1000
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
                    # 浮点 tensor 不量化（避免 backward NaN），只做无损 CPU offload
                    _is_float = block.dtype in (torch.float32, torch.float16, torch.bfloat16)
                    if cfg.enable_quantization and LECaC_AVAILABLE and not _is_float:
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
                # 浮点 tensor 不量化（避免 backward NaN），只做无损 CPU offload
                _is_float = t.dtype in (torch.float32, torch.float16, torch.bfloat16)
                if cfg.enable_quantization and LECaC_AVAILABLE and not _is_float:
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
            # LRU 淘汰
            if len(cache) > cfg.max_cache_entries:
                cache.popitem(last=False)

            if prefetcher:
                prefetcher.add(packed)

            _vram_perf["pack_n"]     += 1
            _vram_perf["pack_bytes"] += nbytes
            _vram_perf["pack_ms"]    += (time.perf_counter() - _pack_t0) * 1000

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

        # 统计 H2D 提交次数和字节数（non_blocking=True：提交即返回，不含 DMA 等待）
        if packed.nested_block is not None:
            _nb = packed.nested_block
            _vram_perf["unpack_n"]     += 1
            _vram_perf["unpack_bytes"] += _nb.cpu_tensor.numel() * _nb.cpu_tensor.element_size()
        elif packed.cpu_tensor is not None:
            _vram_perf["unpack_n"]     += 1
            _vram_perf["unpack_bytes"] += packed.cpu_tensor.numel() * packed.cpu_tensor.element_size()

        # ===== v1.6: 嵌套模式加载 =====
        if packed.nested_block is not None:
            try:
                nested = packed.nested_block

                # ⚡ 优先消费预取结果（prefetcher 线程已在 forward 期间完成 H2D）
                # ⚠️ 必须先取局部引用再清零：防止 prefetcher 线程并发写入时的竞争
                # ⚠️ 清零后原来的 GPU 强引用由 t_pre 持有，函数返回后自动释放，防止泄漏
                _pre = packed.prefetched
                if _pre is not None:
                    packed.prefetched = None  # 消费后立即释放强引用，防止 GPU tensor 滞留泄漏
                    nested.access_count += 1
                    nested.last_access = time.time()
                    if cfg.verbose:
                        mb = _pre.numel() * _pre.element_size() / 1024 / 1024
                        print(f"[VirtualVRAM v1.6] ⚡ Prefetch hit: {mb:.2f}MB "
                              f"block={nested.block_id} heat={nested.arc_heat_score():.2f}")
                    return _pre

                # 预取未就绪，走同步路径：
                # 等待异步 D2H 完成（pack 时 non_blocking=True，unpack 前必须 synchronize）
                # 正常训练中 backward 发生在若干 forward 层之后，DMA 通常早已完成，此处几乎无等待
                # 注意：先把引用存入局部变量再调用 synchronize()，避免 prefetcher 线程并发把
                #       nested.dma_event 置 None 后主线程在 check 通过后拿到 NoneType。
                _ev = nested.dma_event
                if _ev is not None:
                    _ev.synchronize()
                    nested.dma_event = None  # 释放 Event，允许 GC

                # LECaC 去量化
                if nested.quantized and LECaC_AVAILABLE:
                    restored = lecac_dequantize(
                        nested.cpu_tensor,
                        bits=nested.quantization_bits,
                        original_shape=nested.cpu_tensor.shape,
                        original_dtype=nested.dtype,
                        constant=NATURAL_EQUILIBRIUM_CONSTANT
                    )

                    # 🔍 Debug: 检查去量化后的tensor是否有NaN
                    if cfg.debug and torch.isnan(restored).any():
                        print(f"[VRAM DEBUG] ❌ restored tensor has NaN after lecac_dequantize!")
                        print(f"  shape={restored.shape}, bits={nested.quantization_bits}, constant={NATURAL_EQUILIBRIUM_CONSTANT:.4f}")
                        print(f"  cpu_tensor dtype={nested.cpu_tensor.dtype}, cpu_tensor has NaN={torch.isnan(nested.cpu_tensor).any()}")
                else:
                    restored = nested.cpu_tensor

                # H2D 传输：non_blocking=True — H2D DMA 与后续 CUDA kernel 同流排队，
                # CPU 不阻塞，GPU 不因等待 DMA 而空转（CUDA stream 保证 DMA 先于 kernel 执行）
                restored = restored.to(device=nested.device, dtype=nested.dtype, non_blocking=True)
                restored.requires_grad_(nested.requires_grad)

                # 更新热度（访问完成）
                nested.access_count += 1
                nested.last_access = time.time()

                if cfg.verbose:
                    mb = restored.numel() * restored.element_size() / 1024 / 1024
                    print(f"[VirtualVRAM v1.6] ↩️  Nested load: {mb:.2f}MB "
                          f"block={nested.block_id} heat={nested.arc_heat_score():.2f} ref={nested.arc_strong_count()}")

                return restored

            except Exception as e:
                if cfg.verbose:
                    print(f"[VirtualVRAM v1.6] ⚠️ 嵌套加载失败: {e}")
                # Fallback 到传统模式

        # ===== v1.5: 独立分页 + Arc 模式加载（向后兼容） =====
        if packed.page_id is not None:
            try:
                # 优先从 Arc 加载
                if packed.arc_mode and arc_allocator is not None:
                    arc = arc_allocator.get_arc(packed.page_id)
                    if arc is not None:
                        restored = arc.data.to(device=packed.device, non_blocking=True)
                        restored.requires_grad_(packed.requires_grad)

                        if cfg.verbose:
                            mb = restored.numel() * restored.element_size() / 1024 / 1024
                            print(f"[VirtualVRAM] ⚡ Arc loaded {mb:.2f}MB from arc_page {packed.page_id}, ref={arc.strong_count()}")

                        return restored

                # 从物理页加载（仅分页模式）
                if page_allocator is not None and page_table is not None:
                    page_data = page_allocator.get_page(packed.page_id)
                    if page_data is not None:
                        restored = page_data.to(device=packed.device, non_blocking=True)
                        restored.requires_grad_(packed.requires_grad)

                        if cfg.verbose:
                            mb = restored.numel() * restored.element_size() / 1024 / 1024
                            print(f"[VirtualVRAM] 📖 Page loaded {mb:.2f}MB from page {packed.page_id}")

                        return restored

            except Exception as e:
                if cfg.verbose:
                    print(f"[VirtualVRAM] ⚠️  分页加载失败: {e}")

        # 检查预取
        # ⚠️ 必须在 return 前将 packed.prefetched 清零！
        # 若不清零，GPU tensor 会以强引用留在 _Packed 里，而 _Packed 在主 cache 中
        # 永远存活 → 每步预取的 GPU tensor 全部泄漏。
        if packed.prefetched is not None:
            t_pre = packed.prefetched
            packed.prefetched = None  # 消费后立即释放 GPU 强引用
            return t_pre

        if prefetcher and packed.cache_key is not None:
            cached = prefetcher.get(packed.cache_key)
            if cached is not None:
                # 消费后从预取缓存移除，避免 GPU tensor 在 prefetcher.cache 里滞留
                prefetcher.cache.pop(packed.cache_key, None)
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

            restored = restored.to(device=packed.device, dtype=packed.dtype, non_blocking=True)
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

        # v1.6: 清理全局热度优先队列
        if cfg.enable_nested_v16:
            global _heat_pq
            _heat_pq = _HeatPriorityQueue()  # 重置队列

        # v1.5: 分页 + Arc 清理
        if page_table:
            page_table.clear()

        if cfg.verbose:
            # v1.6: 嵌套模式统计
            if cfg.enable_nested_v16:
                nested_blocks = [p.nested_block for p in cache.values() if p.nested_block is not None]
                if nested_blocks:
                    total_refs = sum(b.arc_strong_count() for b in nested_blocks)
                    total_access = sum(b.access_count for b in nested_blocks)
                    avg_heat = sum(b.arc_heat_score() for b in nested_blocks) / len(nested_blocks)

                    # 找出最热的块
                    hottest = max(nested_blocks, key=lambda b: b.arc_heat_score())

                    print(f"[VirtualVRAM v1.6] 📊 嵌套架构统计:")
                    print(f"  - 嵌套块数: {len(nested_blocks)}")
                    print(f"  - 总引用计数: {total_refs}")
                    print(f"  - 平均引用: {total_refs / len(nested_blocks):.2f}")
                    print(f"  - 总访问次数: {total_access}")
                    print(f"  - 平均热度: {avg_heat:.2f}")
                    print(f"  - 最热块: block={hottest.block_id} heat={hottest.arc_heat_score():.2f} "
                          f"access={hottest.access_count}")
                    print(f"  - 量化块数: {sum(1 for b in nested_blocks if b.quantized)}")
                    print(f"  - 热度队列大小: {len(_heat_pq.heap)}")

            # v1.5: Page Allocator 统计
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
