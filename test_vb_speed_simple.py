#!/usr/bin/env python3
import sys
def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except (OSError, IOError):
        pass
"""简化版VB速度测试（跳过大层output_head）"""

import torch
import torch.nn as nn
import time
import sys

safe_print("="*80)
safe_print("Virtual Blackwell 简化速度测试（CUDA）")
safe_print("="*80)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
safe_print(f"设备: {device}")
safe_print()

# 创建一个简单的小模型（避免output_head大层）
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(256, 1024),  # Layer 1
            nn.ReLU(),
            nn.Linear(1024, 1024), # Layer 2
            nn.ReLU(),
            nn.Linear(1024, 256),  # Layer 3
        )

    def forward(self, x):
        return self.layers(x)

safe_print("创建简单模型...")
model = SimpleModel().to(device)
total_params = sum(p.numel() for p in model.parameters())
safe_print(f"模型参数: {total_params:,}")
safe_print()

# 应用VB
safe_print("应用Virtual Blackwell...")
sys.stdout.flush()

from apt.vgpu.runtime.vb_integration import VBModelWrapper

wrapper = VBModelWrapper(
    model,
    mode='training',
    enable_fp4=True,
    enable_quantization=True,
    replace_pattern='all'
)

safe_print(f"已替换 {len(wrapper.replaced_layers)} 层")
safe_print()

# 创建优化器
safe_print("创建优化器...")
sys.stdout.flush()

optimizer = torch.optim.Adam(wrapper.parameters(), lr=0.001)
criterion = nn.MSELoss()

safe_print("优化器创建完成")
safe_print()

# 测试训练
batch_size = 8
num_batches = 10

safe_print(f"开始训练测试 ({num_batches} batches)...")
safe_print("-"*80)

batch_times = []
total_start = time.time()

for batch_idx in range(num_batches):
    batch_start = time.time()

    # 随机输入
    x = torch.randn(batch_size, 256).to(device)
    target = torch.randn(batch_size, 256).to(device)

    # 前向+反向
    optimizer.zero_grad()
    output = wrapper(x)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    batch_time = time.time() - batch_start
    batch_times.append(batch_time)

    if (batch_idx + 1) % 5 == 0:
        avg_time = sum(batch_times[-5:]) / 5
        safe_print(f"Batch {batch_idx+1}/{num_batches} | "
              f"Loss: {loss.item():.4f} | "
              f"Time: {batch_time:.3f}s | "
              f"Avg: {avg_time:.3f}s")

total_time = time.time() - total_start

safe_print("-"*80)
safe_print()
safe_print("="*80)
safe_print("性能统计")
safe_print("="*80)
safe_print(f"总时间: {total_time:.2f}s")
safe_print(f"平均batch: {sum(batch_times)/len(batch_times):.3f}s")
safe_print(f"首批: {batch_times[0]:.3f}s")
safe_print(f"后续平均: {sum(batch_times[1:])/len(batch_times[1:]):.3f}s")
safe_print()

# VB统计
all_stats = wrapper.get_all_stats()
total_cache_hits = 0
total_computes = 0

for name, stats in all_stats.items():
    if 'layer1_vgpu' in stats:
        vgpu = stats['layer1_vgpu']
        total_computes += vgpu.get('total', 0)
        total_cache_hits += vgpu.get('cache_hits', 0)

if total_computes > 0:
    safe_print(f"总计算: {total_computes}")
    safe_print(f"缓存命中: {total_cache_hits} ({total_cache_hits/total_computes*100:.1f}%)")

safe_print()
safe_print("="*80)
safe_print("测试完成！")
safe_print("="*80)
