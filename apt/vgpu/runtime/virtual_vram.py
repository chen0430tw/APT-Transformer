# virtual_vram.py
# -*- coding: utf-8 -*-
"""
Virtual VRAM v1.0 (Pinned-Memory Async Activation Offload)

架构：
  Pack:  pinned CPU tensor + non_blocking D2H on 专用 copy_stream
         → 与 forward compute 重叠，不阻塞默认流
  Unpack: 等待 D2H event → H2D 恢复（同步，backward 立即需要数据）

Key 设计（采纳 Opus v1.0）：
  tensor_key = (storage_data_ptr, shape, stride, offset)
  每个唯一 view 存一份 CPU tensor，不做 contiguous()，保留原始 stride。
  比 v0.2 的 as_strided 方案更简单、更鲁棒。

Usage:
    from apt.vgpu.runtime.virtual_vram import VirtualVRAMConfig, virtual_vram

    cfg = VirtualVRAMConfig(enabled=True, min_tensor_bytes=1<<20)
    with virtual_vram(cfg):
        loss = model(x).sum()
        loss.backward()
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

import torch

# 自然平衡常数：4/e ≈ 1.4715，用于 LECaC 量化补偿
NATURAL_EQUILIBRIUM_CONSTANT: float = 4.0 / math.e


@dataclass
class VirtualVRAMConfig:
    enabled: bool = True
    min_tensor_bytes: int = 1 << 20   # 1MB — 只 offload >= 此大小的 tensor
    # 已废弃，保留向后兼容
    min_storage_bytes: int = 0
    verbose: bool = False


class _Packed:
    """pack_hook 返回值：CPU tensor + 恢复用元数据 + D2H 完成事件"""
    __slots__ = ('cpu_tensor', 'device', 'dtype', 'requires_grad', 'event')

    def __init__(self, cpu_tensor: torch.Tensor,
                 device: torch.device, dtype: torch.dtype,
                 requires_grad: bool,
                 event: Optional[torch.cuda.Event]):
        self.cpu_tensor = cpu_tensor
        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.event = event     # D2H 完成信号（非 CUDA 环境时为 None）


@contextmanager
def virtual_vram(cfg: VirtualVRAMConfig):
    """
    Pinned-memory async activation offload.
    """
    if not cfg.enabled:
        yield
        return

    min_bytes = cfg.min_tensor_bytes or cfg.min_storage_bytes or (1 << 20)

    # 查表：避免对同一 view 重复 offload
    # key = (storage_data_ptr, shape, stride, offset) → _Packed
    cache: Dict[Tuple, _Packed] = {}

    # D2H 专用流：与默认 compute stream 并行搬运
    copy_stream: Optional[torch.cuda.Stream] = None
    if torch.cuda.is_available():
        try:
            copy_stream = torch.cuda.Stream()
        except Exception:
            pass

    def _make_key(t: torch.Tensor) -> Tuple:
        """为一个 tensor view 生成唯一标识"""
        return (
            t.untyped_storage().data_ptr(),
            tuple(t.shape),
            tuple(t.stride()),
            t.storage_offset(),
        )

    def pack_hook(t: torch.Tensor) -> Any:
        # ── 快速筛选 ────────────────────────────────────────────────────
        if not torch.is_tensor(t) or not t.is_cuda:
            return t

        nbytes = t.numel() * t.element_size()
        if nbytes < min_bytes:
            return t

        # ── 查表：同一 view 只搬一次 ────────────────────────────────────
        key = _make_key(t)
        if key in cache:
            if cfg.verbose:
                print(f"[VirtualVRAM] 🔗 cache hit {tuple(t.shape)}")
            return cache[key]

        # ── D2H：pinned memory + async copy ─────────────────────────────
        device = t.device
        dtype = t.dtype
        requires_grad = bool(t.requires_grad)

        try:
            # 分配 pinned CPU tensor（DMA 直达，比 pageable 快 2-3 倍）
            # 不做 contiguous()！保留原始 stride，这是 v0.1 崩的根因修复
            cpu_tensor = torch.empty(
                t.shape, dtype=dtype, device='cpu', pin_memory=True,
            )

            if copy_stream is not None:
                # 关键：让 copy_stream 等待 default stream 当前队列中的所有工作
                # PyTorch 切换流时不自动插入跨流依赖！不加这行，copy_stream
                # 可能在 compute kernel 还未写完 t 时就开始 D2H → 读到 NaN
                copy_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(copy_stream):
                    cpu_tensor.copy_(t, non_blocking=True)
                event = copy_stream.record_event()
            else:
                # 无 CUDA 或 stream 创建失败，退化到同步
                cpu_tensor.copy_(t)
                event = None

        except Exception as e:
            if cfg.verbose:
                print(f"[VirtualVRAM] ❌ D2H 失败: {e}")
            return t

        packed = _Packed(cpu_tensor, device, dtype, requires_grad, event)
        cache[key] = packed

        if cfg.verbose:
            mb = nbytes / 1024 / 1024
            print(f"[VirtualVRAM] ✅ D2H {mb:.2f}MB {tuple(t.shape)} "
                  f"stride={tuple(t.stride())} async={'yes' if event else 'no'}")

        return packed

    def unpack_hook(packed: Any) -> torch.Tensor:
        if not isinstance(packed, _Packed):
            return packed

        # ── 等待 D2H 完成 ────────────────────────────────────────────────
        if packed.event is not None:
            packed.event.synchronize()

        # ── H2D：pinned → CUDA（同步，backward 立即需要数据）────────────
        try:
            with torch.no_grad():
                restored = packed.cpu_tensor.to(
                    device=packed.device,
                    non_blocking=False,
                )
            restored.requires_grad_(packed.requires_grad)
        except Exception as e:
            raise RuntimeError(f"Failed to restore tensor from CPU: {e}")

        if cfg.verbose:
            mb = restored.numel() * restored.element_size() / 1024 / 1024
            print(f"[VirtualVRAM] ↩️  H2D {mb:.2f}MB {tuple(restored.shape)}")

        return restored

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        yield

    # context 退出：清理 cache（释放 pinned CPU 内存）
    cache.clear()
