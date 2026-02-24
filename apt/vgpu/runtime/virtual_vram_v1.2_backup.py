# virtual_vram.py
# -*- coding: utf-8 -*-
"""
Virtual VRAM v1.2 (DeepSeek-Inspired Conservative Optimization)

v1.2 改进（参考 DeepSeek FlashMLA）：
  1. 提高min_tensor_bytes阈值：1MB → 5MB（减少小tensor开销）
  2. 选择性offload：只offload 5MB-50MB的tensor（避免超大tensor）
  3. LECaC集成：量化后offload（可选，实验性）

保持v1.1的稳定性：
  - .detach().cpu() 为主（避免pinned memory的坑）
  - 简单可靠，不过度工程化

Usage:
    from apt.vgpu.runtime.virtual_vram import VirtualVRAMConfig, virtual_vram

    cfg = VirtualVRAMConfig(
        min_tensor_bytes=5<<20,     # 5MB
        max_tensor_bytes=50<<20,    # 50MB (0=无限制)
        enable_quantization=False  # 实验性
    )
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
    # Tensor size filtering（选择性offload，参考DeepSeek）
    min_tensor_bytes: int = 5 << 20   # 5MB — 最小offload大小
    max_tensor_bytes: int = 50 << 20  # 50MB — 最大offload大小（0=无限制）

    # 量化优化（实验性，需要LECaC）
    enable_quantization: bool = False
    quantization_bits: int = 8

    # 已废弃，保留向后兼容
    min_storage_bytes: int = 0
    verbose: bool = False


class _Packed:
    """pack_hook 返回值：CPU tensor + 恢复用元数据"""
    __slots__ = ('cpu_tensor', 'device', 'dtype', 'requires_grad', 'quantized')

    def __init__(self, cpu_tensor: torch.Tensor,
                 device: torch.device,
                 dtype: torch.dtype,
                 requires_grad: bool,
                 quantized: bool = False):
        self.cpu_tensor = cpu_tensor
        self.device = device
        self.dtype = dtype
        self.requires_grad = requires_grad
        self.quantized = quantized


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
            return cache[key]

        # ── D2H：.cpu() 或 LECaC量化 ────────────────────────────────────────
        try:
            original_dtype = t.dtype
            quantized = False

            # 尝试LECaC量化（如果启用且可用）
            if cfg.enable_quantization and LECaC_AVAILABLE:
                try:
                    cpu_tensor = lecac_quantize(
                        t.detach(),
                        bits=cfg.quantization_bits,
                        constant=NATURAL_EQUILIBRIUM_CONSTANT
                    ).cpu()
                    quantized = True
                except Exception as e:
                    if cfg.verbose:
                        print(f"[VirtualVRAM] ⚠️  Quantization failed, using .cpu(): {e}")
                    cpu_tensor = t.detach().cpu()
            else:
                cpu_tensor = t.detach().cpu()

            packed = _Packed(cpu_tensor, t.device, original_dtype,
                             bool(t.requires_grad), quantized)
            cache[key] = packed

            if cfg.verbose:
                mb = nbytes / 1024 / 1024
                print(f"[VirtualVRAM] ✅ D2H {mb:.2f}MB {tuple(t.shape)} "
                      f"dtype={original_dtype} quantized={quantized}")

            return packed

        except Exception as e:
            if cfg.verbose:
                print(f"[VirtualVRAM] ❌ D2H 失败: {e}")
            return t

    def unpack_hook(packed: Any) -> torch.Tensor:
        if not isinstance(packed, _Packed):
            return packed

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

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        yield

    # context 退出：清理 cache（释放 CPU 内存）
    cache.clear()
