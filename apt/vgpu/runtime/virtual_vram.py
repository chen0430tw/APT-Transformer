# virtual_vram.py
# -*- coding: utf-8 -*-
"""
Virtual VRAM v0.1 (Activation Offload via saved_tensors_hooks)

修复版本：
- 不要在 hook 内部做 .detach()，会破坏梯度
- 简化实现，去掉 pin_memory/non_blocking
- 直接在 pack 时转移到 CPU，unpack 时恢复到 GPU

Goal:
- Reduce peak CUDA memory by offloading autograd-saved tensors (activations)
  to CPU memory during forward, restoring to CUDA on backward.

This is "virtual VRAM" in the sense:
- VRAM content becomes a recoverable state, not always-resident bytes.
- Torch.compile does NOT need to "know" this; it lives in autograd hooks.

Usage:
    from virtual_vram import VirtualVRAMConfig, virtual_vram

    cfg = VirtualVRAMConfig(
        enabled=True,
        min_tensor_bytes=1<<20,     # only offload tensors >= 1MB
        verbose=False,
    )

    with virtual_vram(cfg):
        loss = model(x).sum()
        loss.backward()
"""

from __future__ import annotations

import uuid
import tempfile
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any, Tuple

import torch


@dataclass
class VirtualVRAMConfig:
    enabled: bool = True
    min_tensor_bytes: int = 1 << 20  # 1MB
    verbose: bool = False


def _tensor_nbytes(t: torch.Tensor) -> int:
    try:
        return t.numel() * t.element_size()
    except Exception:
        return 0


class _CPUPacked:
    """包装一个 CPU tensor，保存元数据用于恢复"""
    def __init__(self, cpu_tensor: torch.Tensor, device: torch.device, dtype: torch.dtype, requires_grad: bool):
        self.cpu_tensor = cpu_tensor
        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad


@contextmanager
def virtual_vram(cfg: VirtualVRAMConfig):
    """
    Context manager enabling activation offload to CPU.
    """
    if not cfg.enabled:
        yield
        return

    def pack_hook(t: torch.Tensor) -> _CPUPacked:
        # 只处理 CUDA 张量
        if not torch.is_tensor(t) or not t.is_cuda:
            # 非 CUDA tensor 直接返回（不改变行为）
            return t

        nbytes = _tensor_nbytes(t)
        if nbytes < int(cfg.min_tensor_bytes):
            # 太小了，不值得搬
            return t

        # 保存元数据
        device = t.device
        dtype = t.dtype
        requires_grad = bool(t.requires_grad)

        # 关键：不要让 offload 进入 autograd 图（不可微搬运）
        # .contiguous() 确保 stride 布局标准化为行主序，
        # 避免非连续 tensor（expand/slice/transpose 产生的异常 stride）
        # 恢复到 CUDA 后触发 FlexAttention / SDPA 的 stride 4-对齐检查失败
        try:
            with torch.no_grad():
                cpu_tensor = t.detach().contiguous().to("cpu")
        except Exception as e:
            # 转移失败，返回原始 tensor
            if cfg.verbose:
                print(f"[VirtualVRAM] ❌ 转移失败: {e}")
            return t

        if cfg.verbose:
            print(f"[VirtualVRAM] ✅ offload->cpu {nbytes/1024/1024:.2f} MB, "
                  f"dtype={dtype}, shape={tuple(t.shape)}, requires_grad={requires_grad}")

        return _CPUPacked(cpu_tensor, device, dtype, requires_grad)

    def unpack_hook(packed) -> torch.Tensor:
        # 如果不是我们的包装对象，直接返回
        if not isinstance(packed, _CPUPacked):
            return packed

        # 从 CPU 恢复到 CUDA（同样不进入 autograd 图）
        try:
            with torch.no_grad():
                restored = packed.cpu_tensor.to(device=packed.device, dtype=packed.dtype)
            # 保持原有的 requires_grad 设置
            restored.requires_grad_(packed.requires_grad)
            if cfg.verbose:
                nbytes = restored.numel() * restored.element_size()
                print(f"[VirtualVRAM] ↩️  restore->cuda {nbytes/1024/1024:.2f} MB, "
                      f"dtype={packed.dtype}, shape={tuple(restored.shape)}")
            return restored
        except Exception as e:
            # 恢复失败，这是个严重问题
            raise RuntimeError(f"Failed to restore tensor from CPU: {e}")

    # 启用 hooks
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        yield
