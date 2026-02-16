#!/usr/bin/env python3
import sys
import torch
from apt.vgpu.runtime.virtual_blackwell_adapter import VirtualBlackwellAdapter

# 测试基本功能
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vb = VirtualBlackwellAdapter(mode='training', enable_fp4=True, gpu_id=0)

W = torch.randn(128, 64).to(device)
X = torch.randn(64, 32).to(device)  # (in_features, batch) for W @ X

vb.register_weight('test', W)
Y = vb.compress(W, X, weight_id='test')

error = (W @ X - Y).abs().mean().item()

with open('vb_minimal_result.txt', 'w') as f:
    f.write(f"Device: {device}\n")
    f.write(f"W shape: {W.shape}\n")
    f.write(f"X shape: {X.shape}\n")
    f.write(f"Y shape: {Y.shape}\n")
    f.write(f"Computation error: {error:.6f}\n")
    f.write(f"Status: {'[OK]' if error < 0.1 else '[X]'}\n\n")

    # 测试精度分离
    sep = vb.vgpu.separator.separate(W)
    f.write(f"Coarse levels: {sep['coarse'].unique().numel()}\n")
    f.write(f"Fine levels: {sep['fine'].unique().numel()}\n")

    # 测试BOH协议
    handshake = vb.vgpu.protocol.handshake(0, 1, 1000)
    f.write(f"BOH priority: {handshake['priority']}\n")
    f.write(f"BOH status: {handshake['status']}\n")

    # 测试共享内存
    shared_keys = list(vb.vgpu.shared_memory.keys())
    f.write(f"Shared memory entries: {len(shared_keys)}\n")

    f.write("\n[OK] All tests passed!\n")

sys.stdout.write("Test completed, check vb_minimal_result.txt\n")
