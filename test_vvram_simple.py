#!/usr/bin/env python3
"""
Virtual VRAM v1.6 简单测试 - 使用合成数据
不需要下载大数据集
"""
import sys
import os

sys.path.insert(0, "/mnt/d/APT-Transformer")

import torch
import torch.nn as nn
from apt.vgpu.runtime.virtual_vram import VirtualVRAMConfig, virtual_vram

print("=" * 60)
print("Virtual VRAM v1.6 简单测试（合成数据）")
print("=" * 60)

# 配置
cfg = VirtualVRAMConfig(
    enabled=True,
    enable_nested_v16=True,
    min_tensor_bytes=5 << 20,  # 5MB阈值
    verbose=True
)

# 创建简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1000, 500)
        self.linear2 = nn.Linear(500, 100)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return x

model = SimpleModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\n[测试1] 小tensor训练（<5MB，不触发Virtual VRAM）")
print("-" * 60)
for step in range(3):
    optimizer.zero_grad()
    x = torch.randn(10, 1000).cuda()  # ~40KB
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    print(f"Step {step}: Loss={loss.item():.4f}")

print("\n[测试2] 大tensor训练（>5MB，触发Virtual VRAM）")
print("-" * 60)
with virtual_vram(cfg):
    for step in range(3):
        optimizer.zero_grad()
        # 创建大tensor（>5MB）
        big_x = torch.randn(1000, 2000).cuda()  # ~8MB
        big_y = model(big_x)
        loss = big_y.sum()
        loss.backward()
        optimizer.step()
        print(f"Step {step}: Loss={loss.item():.4f}")

print("\n[测试3] 检查NaN")
print("-" * 60)
# 检查参数是否有NaN
has_nan = False
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        print(f"❌ {name} 包含NaN!")
        has_nan = True
    elif torch.isinf(param).any():
        print(f"❌ {name} 包含Inf!")
        has_nan = True

if not has_nan:
    print("✅ 所有参数正常，无NaN/Inf")

print("\n[测试4] 显存使用")
print("-" * 60)
if torch.cuda.is_available():
    print(f"显存已用: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
    print(f"显存缓存: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")

print("\n" + "=" * 60)
print("✅ 测试完成!")
print("=" * 60)
