# virtual_vram.py
# -*- coding: utf-8 -*-
"""
Virtual VRAM v0.6 "查表机制：收束搬运的叢集再查表"

核心設計哲學：大道至簡
- 只搬運 storage（底層肉體），不搬運 tensor（視圖靈魂）
- Pack 時只發提貨單（ticket），Unpack 時按表索驥重塑靈魂
- 同一 storage 的多個 tensor（view/reshape 別名）只搬運一次
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch

NATURAL_EQUILIBRIUM_CONSTANT: float = 4.0 / math.e


@dataclass
class VirtualVRAMConfig:
    enabled: bool = True
    min_tensor_bytes: int = 1 << 20
    verbose: bool = False
    ultra_debug: bool = True

    def __post_init__(self):
        if self.ultra_debug:
            print(f"[VirtualVRAM] CREATED: enabled={self.enabled}, min_bytes={self.min_tensor_bytes}")


def _tensor_nbytes(t: torch.Tensor) -> int:
    try:
        return t.numel() * t.element_size()
    except Exception:
        return 0


@contextmanager
def virtual_vram(cfg: VirtualVRAMConfig):
    if not cfg.enabled:
        yield
        return

    # ========== 兩本總帳：tensor_id -> CPU tensor ==========
    cpu_tensors = {}  # tensor_id (storage_id+shape+stride) -> CPU tensor

    def pack_hook(t: torch.Tensor):
        if not torch.is_tensor(t) or not t.is_cuda:
            return t

        nbytes = _tensor_nbytes(t)
        if nbytes < int(cfg.min_tensor_bytes):
            return t

        try:
            storage = t.untyped_storage()
            storage_id = storage.data_ptr()
        except Exception:
            return t

        # ========== 創建唯一 tensor ID（基於 storage + shape + stride）==========
        tensor_id = (storage_id, tuple(t.size()), tuple(t.stride()), t.storage_offset())

        # ========== 收束叢集：查表 ==========
        if tensor_id not in cpu_tensors:
            # 第一次見到這個 tensor（特定視圖），搬運並保存
            cpu_tensor = t.detach().cpu()
            cpu_tensors[tensor_id] = cpu_tensor

            if cfg.ultra_debug:
                print(f"[PACK] Offload tensor {tensor_id[0]}...{hex(tensor_id[1][-1])} ({nbytes/1024/1024:.2f} MB)")
        elif cfg.ultra_debug:
            print(f"[PACK] Cache hit! Reuse tensor {tensor_id[0]}...{hex(tensor_id[1][-1])}")

        # ========== 只發提貨單 ==========
        ticket = (tensor_id, t.device, t.dtype)

        return ticket

    def unpack_hook(ticket):
        if not isinstance(ticket, tuple) or len(ticket) != 3:
            return ticket

        tensor_id, device, dtype = ticket

        # ========== 按表索驥：從 CPU 恢復 ==========
        cpu_tensor = cpu_tensors[tensor_id]

        # 恢復到 GPU（避免 backward 污染，指定 dtype）
        restored = cpu_tensor.clone().to(device, dtype=dtype)

        if cfg.verbose:
            nbytes = restored.numel() * restored.element_size()
            print(f"[VirtualVRAM] ↩️  restore {nbytes/1024/1024:.2f} MB")

        return restored

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        try:
            yield
        finally:
            if cfg.ultra_debug:
                print(f"\n[VirtualVRAM] Summary: {len(cpu_tensors)} unique storages offloaded\n")
            cpu_tensors.clear()
