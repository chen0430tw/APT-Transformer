# virtual_vram.py
# -*- coding: utf-8 -*-
"""
Virtual VRAM v1.1 (Stable Activation Offload)

历史教训（v1.0 的坑）：
  pinned memory + async D2H 理论上更快，但实测踩了三个坑：
  1. copy_stream 缺少跨流依赖 → D2H 在 compute kernel 写完前就开始 → NaN
  2. bool mask (dropout) 被 offload → 恢复时 dtype 错误 → Mask should be Bool
  3. 即使加了 wait_stream + event.synchronize()，某些 tensor 仍数据损坏
  实测: pinned+async 28s/step，.detach().cpu() 6.44s/step（快 4.5x）

v1.1 设计：
  Pack:   bool 类型直接跳过；其余 .detach().cpu()（PyTorch 内部优化，稳定可靠）
  Unpack: .to(device, non_blocking=False)（同步 H2D，backward 立即需要数据）
  Cache:  per-view key 去重（storage_ptr, shape, stride, offset），避免同一 view 重复搬运

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
from typing import Any, Dict, Tuple

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
    """pack_hook 返回值：CPU tensor + 恢复用元数据"""
    __slots__ = ('cpu_tensor', 'device', 'requires_grad')

    def __init__(self, cpu_tensor: torch.Tensor,
                 device: torch.device,
                 requires_grad: bool):
        self.cpu_tensor = cpu_tensor
        self.device = device
        self.requires_grad = requires_grad


@contextmanager
def virtual_vram(cfg: VirtualVRAMConfig):
    """
    稳定的激活值 CPU offload context manager。
    """
    if not cfg.enabled:
        yield
        return

    min_bytes = cfg.min_tensor_bytes or cfg.min_storage_bytes or (1 << 20)

    # 查表：避免对同一 view 重复 offload
    # key = (storage_data_ptr, shape, stride, offset) → _Packed
    cache: Dict[Tuple, _Packed] = {}

    def _make_key(t: torch.Tensor) -> Tuple:
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

        # bool 类型（如 dropout mask）必须跳过：
        # offload 后恢复时 .to(device) 不保证保留 bool dtype，
        # 会导致 "Mask should be Bool" 类型错误
        if t.dtype == torch.bool:
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

        # ── D2H：detach().cpu()（同步，稳定可靠）────────────────────────
        # 教训：pinned memory + async copy 在多流环境下数据损坏问题极难追查。
        # torch.Tensor.cpu() 内部走 PyTorch 标准 D2H 路径，
        # 会等待 tensor 所在流完成写入，不会产生 NaN。
        # 实测比 pinned+async 快 4.5x（6.44s vs 28s per step）。
        try:
            cpu_tensor = t.detach().cpu()
        except Exception as e:
            if cfg.verbose:
                print(f"[VirtualVRAM] ❌ D2H 失败: {e}")
            return t

        packed = _Packed(cpu_tensor, t.device, bool(t.requires_grad))
        cache[key] = packed

        if cfg.verbose:
            mb = nbytes / 1024 / 1024
            print(f"[VirtualVRAM] ✅ D2H {mb:.2f}MB {tuple(t.shape)} "
                  f"dtype={t.dtype} stride={tuple(t.stride())}")

        return packed

    def unpack_hook(packed: Any) -> torch.Tensor:
        if not isinstance(packed, _Packed):
            return packed

        # H2D：同步恢复（backward 立即需要数据）
        try:
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

    # context 退出：清理 cache（释放 CPU 内存）
    cache.clear()
