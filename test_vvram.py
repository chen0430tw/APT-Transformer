#!/usr/bin/env python3
"""Virtual VRAM v1.6 测试脚本"""
import sys
import os
# 添加APT-Transformer到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
apt_path = os.path.join(script_dir, 'APT-Transformer')
if apt_path not in sys.path:
    sys.path.insert(0, apt_path)

from apt.vgpu.runtime.virtual_vram import VirtualVRAMConfig, virtual_vram
import torch

print("=" * 60)
print("Virtual VRAM v1.6 测试")
print("=" * 60)

# 配置
cfg = VirtualVRAMConfig(
    enable_nested_v16=True,
    min_tensor_bytes=5 << 20,
    verbose=True
)

# 测试1: 基础forward
print("\n[测试1] 基础forward测试")
t1 = torch.randn(1000, 1000).cuda().requires_grad_(True)
print(f"输入tensor: {t1.shape}, {t1.device}, requires_grad={t1.requires_grad}")

with virtual_vram(cfg):
    result = t1 * 2

print(f"输出: {result.shape}, {result.device}, requires_grad={result.requires_grad}")

# 测试2: backward测试
print("\n[测试2] Backward测试")
loss = result.sum()
print(f"Loss: {loss.item():.4f}")

loss.backward()
print(f"梯度: {t1.grad.shape}, mean={t1.grad.mean().item():.4f}")

# 测试3: 多层网络
print("\n[测试3] 多层网络测试")
t2 = torch.randn(10, 1000).cuda().requires_grad_(True)
with virtual_vram(cfg):
    x = t2
    x = torch.relu(x @ torch.randn(1000, 500).cuda())
    x = torch.relu(x @ torch.randn(500, 100).cuda())
    result2 = x

print(f"MLP输出: {result2.shape}, {result2.device}")

# 测试4: 大tensor测试（超过5MB阈值）
print("\n[测试4] 大tensor测试（>5MB）")
big_tensor = torch.randn(1000, 2000).cuda()  # ~8MB
print(f"大tensor: {big_tensor.shape}, {big_tensor.element_size() * big_tensor.nelement() / 1024 / 1024:.2f}MB")

with virtual_vram(cfg):
    result3 = big_tensor + 1

print(f"大tensor处理完成: {result3.shape}")

print("\n" + "=" * 60)
print("✅ 所有测试通过!")
print("=" * 60)
