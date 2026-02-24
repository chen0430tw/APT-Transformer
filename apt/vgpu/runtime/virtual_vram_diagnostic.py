# virtual_vram_diagnostic.py
# -*- coding: utf-8 -*-
"""
Virtual VRAM 诊断脚本 - 分析相同 storage_id 的 tensor 之间的关系
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from collections import defaultdict

import torch

NATURAL_EQUILIBRIUM_CONSTANT: float = 4.0 / math.e


@dataclass
class VirtualVRAMConfig:
    enabled: bool = True
    min_tensor_bytes: int = 1 << 20
    verbose: bool = False
    ultra_debug: bool = True


def _tensor_nbytes(t: torch.Tensor) -> int:
    try:
        return t.numel() * t.element_size()
    except Exception:
        return 0


@contextmanager
def virtual_vram_diagnostic(cfg: VirtualVRAMConfig):
    if not cfg.enabled:
        yield
        return

    # ========== 诊断数据结构 ==========
    # storage_id -> [tensor_info列表]
    storage_groups = defaultdict(list)

    # storage_id -> 是否已打印过详细信息
    printed_details = set()

    pack_count = [0]

    def pack_hook(t: torch.Tensor):
        if not torch.is_tensor(t) or not t.is_cuda:
            return t

        nbytes = _tensor_nbytes(t)
        if nbytes < int(cfg.min_tensor_bytes):
            return t

        device = t.device
        dtype = t.dtype
        shape = tuple(t.shape)
        strides = t.stride()
        offset = t.storage_offset()

        # 获取 storage 信息
        try:
            storage = t.untyped_storage()
            storage_id = storage.data_ptr()
            storage_size = storage.size()
        except Exception:
            return t

        pack_count[0] += 1
        pack_id = pack_count[0]

        # ========== 收集诊断信息 ==========
        tensor_info = {
            'pack_id': pack_id,
            'shape': shape,
            'strides': strides,
            'offset': offset,
            'dtype': dtype,
            'nbytes': nbytes,
            'storage_size': storage_size,
            'numel': t.numel(),
            'is_contiguous': t.is_contiguous(),
        }

        storage_groups[storage_id].append(tensor_info)

        # ========== 打印诊断信息 ==========
        print(f"\n{'='*80}")
        print(f"[PACK #{pack_id}] Tensor Info")
        print(f"-" * 80)
        print(f"  Storage ID        : {storage_id}")
        print(f"  Shape             : {shape}")
        print(f"  Strides           : {strides}")
        print(f"  Offset            : {offset}")
        print(f"  Dtype             : {dtype}")
        print(f"  Numel             : {t.numel()}")
        print(f"  Bytes             : {nbytes} ({nbytes/1024/1024:.2f} MB)")
        print(f"  Storage Size      : {storage_size} bytes ({storage_size/1024/1024:.2f} MB)")
        print(f"  Is Contiguous     : {t.is_contiguous()}")
        print(f"  Storage Matches?  : {storage_size == nbytes}")

        # 计算需要的 storage 大小（基于 stride）
        if len(shape) > 0:
            # 找到最大的索引
            max_offset = offset
            for i in range(len(shape)):
                max_offset += (shape[i] - 1) * strides[i]
            required_bytes = (max_offset + 1) * t.element_size()
            print(f"  Required Storage  : {required_bytes} bytes ({required_bytes/1024/1024:.2f} MB)")
            print(f"  Fits in Storage?  : {required_bytes <= storage_size}")

            if required_bytes > storage_size:
                print(f"  ⚠️  WARNING: View exceeds storage bounds by {required_bytes - storage_size} bytes!")

        print(f"{'='*80}")

        # ========== 如果这个 storage 有多个 tensor，打印对比 ==========
        if len(storage_groups[storage_id]) > 1 and storage_id not in printed_details:
            printed_details.add(storage_id)
            print(f"\n{'!'*80}")
            print(f"[DIAGNOSIS] Storage ID {storage_id} has {len(storage_groups[storage_id])} tensors!")
            print(f"{'!'*80}")

            for i, info in enumerate(storage_groups[storage_id]):
                print(f"\n  Tensor #{i+1} (PACK #{info['pack_id']}):")
                print(f"    Shape           : {info['shape']}")
                print(f"    Strides         : {info['strides']}")
                print(f"    Offset          : {info['offset']}")
                print(f"    Dtype           : {info['dtype']}")
                print(f"    Numel           : {info['numel']}")
                print(f"    Bytes           : {info['nbytes']/1024/1024:.2f} MB")
                print(f"    Storage Size    : {info['storage_size']/1024/1024:.2f} MB")
                print(f"    Is Contiguous   : {info['is_contiguous']}")

                # 计算需要的 storage
                if len(info['shape']) > 0:
                    max_offset = info['offset']
                    for j in range(len(info['shape'])):
                        max_offset += (info['shape'][j] - 1) * info['strides'][j]
                    required = (max_offset + 1) * torch.finfo(info['dtype']).bits // 8 if info['dtype'] in [torch.float32, torch.float16, torch.bfloat16] else (max_offset + 1) * 2
                    print(f"    Required Storage: {required/1024/1024:.2f} MB")

            # 分析：这些 tensor 是否真的共享同一块 storage？
            print(f"\n  Analysis:")
            all_same_storage = all(info['storage_size'] == storage_groups[storage_id][0]['storage_size']
                                   for info in storage_groups[storage_id])
            print(f"    All same storage size? {all_same_storage}")

            if all_same_storage:
                print(f"    ✅ These tensors ARE aliases (same storage)")
            else:
                print(f"    ❌ These tensors have DIFFERENT storage sizes!")
                print(f"    This means they are NOT true aliases!")

            print(f"\n{'!'*80}\n")

        # 不做任何 offload，直接返回原 tensor
        return t

    def unpack_hook(t):
        # 不做任何处理
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        yield

    # ========== 最终诊断报告 ==========
    print(f"\n{'#'*80}")
    print(f"[FINAL DIAGNOSIS REPORT]")
    print(f"{'#'*80}")
    print(f"Total unique storages: {len(storage_groups)}")
    print(f"Total packs: {pack_count[0]}")

    # 找出有多个 tensor 的 storage
    multi_tensor_storages = {sid: infos for sid, infos in storage_groups.items() if len(infos) > 1}
    print(f"Storages with multiple tensors: {len(multi_tensor_storages)}")

    if multi_tensor_storages:
        print(f"\n⚠️  Found {len(multi_tensor_storages)} storage(s) with multiple tensors!")
        print(f"This indicates potential view/reshape aliasing.\n")
    else:
        print(f"\n✅ No storage aliasing detected - each tensor has unique storage\n")

    print(f"{'#'*80}\n")
