# virtual_vram.py
# -*- coding: utf-8 -*-
"""
Virtual VRAM v0.2 (Storage-Aware Activation Offload)

核心问题（v0.1 的致命缺陷）：
  Autograd 对同一 storage 的多个 view（reshape/transpose/slice 产生）
  分别调用 pack_hook，v0.1 对每个 view 独立做 contiguous().to("cpu")，
  破坏了 view 之间的别名关系（aliasing）。Backward 时 autograd 期望
  这些 view 共享同一块 CUDA 内存，但 v0.1 恢复出的是互相独立的副本
  → CUDA illegal memory access。

v0.2 修复方案：Storage 级别的生命周期注册表
  Pack 阶段：
    - 用 id(tensor.untyped_storage()) 作为唯一 key（Python 对象 id，
      不是 CUDA 物理地址，不会因内存复用而冲突）
    - 首次见到某 storage：把整块 storage 搬到 CPU，存入注册表
    - 再次见到同一 storage：直接引用已有的 CPU storage，只记录
      该 view 的 shape/stride/storage_offset（"灵魂密码"）

  Unpack 阶段：
    - 首次有人要这块 storage：把 CPU storage 整体搬回 CUDA，
      注册新的 CUDA storage 对象
    - 后续 view：复用已恢复的 CUDA storage
    - 所有 view 统一用 as_strided 从共享 CUDA storage 重建，
      完整还原别名关系

Usage:
    from apt.vgpu.runtime.virtual_vram import VirtualVRAMConfig, virtual_vram

    cfg = VirtualVRAMConfig(enabled=True, min_storage_bytes=1<<20)
    with virtual_vram(cfg):
        loss = model(x).sum()
        loss.backward()
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import torch

# 自然平衡常数：4/e ≈ 1.4715，用于 LECaC 量化补偿
NATURAL_EQUILIBRIUM_CONSTANT: float = 4.0 / math.e


@dataclass
class VirtualVRAMConfig:
    enabled: bool = True
    # 按 storage 大小过滤（整块 storage，不是单个 tensor 的 numel*itemsize）
    min_storage_bytes: int = 1 << 20   # 1MB
    verbose: bool = False
    # 兼容旧字段名
    min_tensor_bytes: int = 0  # 已废弃，保留向后兼容


# ─────────────────────────────────────────────────────────────────────────────
# Storage 注册表（每次 virtual_vram context 独立一份，线程局部）
# ─────────────────────────────────────────────────────────────────────────────

class _StorageRecord:
    """一块 CUDA storage 在 CPU 上的完整副本 + 恢复状态"""
    __slots__ = ('cpu_storage', 'nbytes', 'dtype', 'device',
                 'cuda_storage', 'lock')

    def __init__(self, cpu_storage: torch.Storage,
                 nbytes: int, dtype: torch.dtype, device: torch.device):
        self.cpu_storage = cpu_storage   # CPU 上的 storage（字节级）
        self.nbytes = nbytes
        self.dtype = dtype
        self.device = device
        self.cuda_storage: Optional[torch.Storage] = None  # 恢复后填充
        self.lock = threading.Lock()     # 防止多线程重复恢复


class _ViewPacked:
    """
    pack_hook 返回值：描述一个 view 如何从共享 storage 重建。
    """
    __slots__ = ('storage_key', 'shape', 'stride', 'storage_offset',
                 'dtype', 'device', 'requires_grad')

    def __init__(self, storage_key: int,
                 shape: Tuple, stride: Tuple, storage_offset: int,
                 dtype: torch.dtype, device: torch.device,
                 requires_grad: bool):
        self.storage_key = storage_key
        self.shape = shape
        self.stride = stride
        self.storage_offset = storage_offset
        self.dtype = dtype
        self.device = device
        self.requires_grad = requires_grad


# ─────────────────────────────────────────────────────────────────────────────
# Context manager
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def virtual_vram(cfg: VirtualVRAMConfig):
    """
    Storage-aware activation offload context manager.
    """
    if not cfg.enabled:
        yield
        return

    # 兼容旧字段名
    min_bytes = cfg.min_storage_bytes or cfg.min_tensor_bytes or (1 << 20)

    # 本次 context 的 storage 注册表（storage_id → _StorageRecord）
    registry: Dict[int, _StorageRecord] = {}

    def pack_hook(t: torch.Tensor):
        # ── 快速筛选 ──────────────────────────────────────────────────────
        if not torch.is_tensor(t) or not t.is_cuda:
            return t

        try:
            raw_storage = t.untyped_storage()
        except Exception:
            return t

        storage_nbytes = raw_storage.nbytes()
        if storage_nbytes < min_bytes:
            return t

        storage_key = id(raw_storage)

        # ── 首次见到此 storage：搬到 CPU ─────────────────────────────────
        if storage_key not in registry:
            try:
                with torch.no_grad():
                    # 把整块 CUDA storage 以字节形式拷到 CPU
                    cpu_raw = raw_storage.cpu()   # 返回 CPU UntypedStorage
            except Exception as e:
                if cfg.verbose:
                    print(f"[VirtualVRAM] ❌ storage 转移失败: {e}")
                return t

            registry[storage_key] = _StorageRecord(
                cpu_storage=cpu_raw,
                nbytes=storage_nbytes,
                dtype=t.dtype,
                device=t.device,
            )
            if cfg.verbose:
                mb = storage_nbytes / 1024 / 1024
                print(f"[VirtualVRAM] ✅ offload storage {storage_key} "
                      f"{mb:.2f}MB → cpu, view shape={tuple(t.shape)}")
        else:
            if cfg.verbose:
                print(f"[VirtualVRAM] 🔗 alias storage {storage_key}, "
                      f"view shape={tuple(t.shape)}")

        # ── 记录该 view 的"灵魂密码" ─────────────────────────────────────
        return _ViewPacked(
            storage_key=storage_key,
            shape=tuple(t.shape),
            stride=tuple(t.stride()),
            storage_offset=t.storage_offset(),
            dtype=t.dtype,
            device=t.device,
            requires_grad=bool(t.requires_grad),
        )

    def unpack_hook(packed):
        # 非我们的包装对象直接返回
        if not isinstance(packed, _ViewPacked):
            return packed

        key = packed.storage_key
        if key not in registry:
            # 找不到 storage，说明该 tensor 小于阈值被跳过了（不应发生）
            raise RuntimeError(
                f"[VirtualVRAM] unpack: storage {key} 不在注册表中")

        record = registry[key]

        # ── 确保 CUDA storage 已恢复（只恢复一次，后续 view 共享）────────
        with record.lock:
            if record.cuda_storage is None:
                try:
                    with torch.no_grad():
                        # CPU storage → CUDA storage（整块恢复）
                        record.cuda_storage = record.cpu_storage.cuda(
                            packed.device.index
                        )
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to restore storage from CPU: {e}")
                if cfg.verbose:
                    mb = record.nbytes / 1024 / 1024
                    print(f"[VirtualVRAM] ↩️  restore storage {key} "
                          f"{mb:.2f}MB → cuda")

        # ── 用 as_strided 从共享 CUDA storage 重建此 view ────────────────
        # as_strided 不拷贝数据，只改变解释方式，完整还原别名关系
        with torch.no_grad():
            # 先造一个和 storage 同 dtype 的 1D 占位 tensor
            # storage 字节数 / dtype 字节数 = 元素数
            elem_size = torch.tensor([], dtype=packed.dtype).element_size()
            n_elems = record.nbytes // elem_size
            base = torch.empty(n_elems, dtype=packed.dtype,
                               device=packed.device)
            # 把 base 的 storage 替换为我们恢复的 cuda_storage
            base.set_(record.cuda_storage, 0,
                      (n_elems,), (1,))
            # 用原始 shape/stride/offset 重建 view
            restored = torch.as_strided(
                base,
                size=packed.shape,
                stride=packed.stride,
                storage_offset=packed.storage_offset,
            )

        restored = restored.requires_grad_(packed.requires_grad)

        if cfg.verbose:
            print(f"[VirtualVRAM] 🔧 as_strided view {tuple(packed.shape)} "
                  f"stride={packed.stride} offset={packed.storage_offset}")

        return restored

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        yield

    # context 退出后清理注册表（释放 CPU 内存）
    registry.clear()
