#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试虚拟Blackwell训练加速效果
对比标准PyTorch vs Virtual Blackwell (Flash Attention + FP4)
"""

import torch
import torch.nn as nn
import time
from apt.vgpu.runtime.vb_integration import enable_vb_optimization

# 创建测试模型
class SimpleTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=4, dim_feedforward=2048):
        super().__init__()
        self.embedding = nn.Embedding(10000, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, 10000)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc_out(x)

def train_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    start_time = time.time()

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, 10000), target.view(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx >= 20:  # 只训练20个batch作为测试
            break

    elapsed = time.time() - start_time
    return total_loss / (batch_idx + 1), elapsed

def create_dummy_data(batch_size=32, seq_len=128, num_batches=25):
    """创建虚拟训练数据"""
    data = []
    for _ in range(num_batches):
        x = torch.randint(0, 10000, (batch_size, seq_len))
        y = torch.randint(0, 10000, (batch_size, seq_len))
        data.append((x, y))
    return data

def main():
    print("=" * 80)
    print("虚拟Blackwell训练加速测试")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")

    # 创建数据
    print("\n创建训练数据...")
    dataloader = create_dummy_data(batch_size=16, seq_len=64, num_batches=25)

    # ========== 测试1: 标准PyTorch ==========
    print("\n" + "=" * 80)
    print("测试1: 标准PyTorch (无优化)")
    print("=" * 80)

    model_standard = SimpleTransformer(d_model=256, nhead=8, num_layers=3).to(device)
    optimizer_standard = torch.optim.Adam(model_standard.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("\n开始训练...")
    loss_standard, time_standard = train_epoch(
        model_standard, dataloader, optimizer_standard, criterion, device
    )

    print(f"\n结果:")
    print(f"  平均Loss: {loss_standard:.4f}")
    print(f"  训练时间: {time_standard:.2f}秒")
    print(f"  吞吐量: {20/time_standard:.2f} batch/s")

    # ========== 测试2: Virtual Blackwell (FP4启用) ==========
    print("\n" + "=" * 80)
    print("测试2: Virtual Blackwell (Flash Attention + FP4)")
    print("=" * 80)

    model_vb = SimpleTransformer(d_model=256, nhead=8, num_layers=3).to(device)

    # 启用Virtual Blackwell优化
    model_vb = enable_vb_optimization(
        model_vb,
        mode='training',
        enable_fp4=True,
        enable_quantization=False,  # 不启用BOH量化，只测FP4
        replace_pattern='all'
    )

    optimizer_vb = torch.optim.Adam(model_vb.parameters(), lr=0.001)

    print("开始训练...")
    loss_vb, time_vb = train_epoch(
        model_vb, dataloader, optimizer_vb, criterion, device
    )

    print(f"\n结果:")
    print(f"  平均Loss: {loss_vb:.4f}")
    print(f"  训练时间: {time_vb:.2f}秒")
    print(f"  吞吐量: {20/time_vb:.2f} batch/s")

    # 打印VB统计
    print("\nVirtual Blackwell统计:")
    model_vb.print_all_stats()

    # ========== 性能对比 ==========
    print("\n" + "=" * 80)
    print("性能对比总结")
    print("=" * 80)

    speedup = time_standard / time_vb
    throughput_improvement = (20/time_vb) / (20/time_standard)

    print(f"\n标准PyTorch:")
    print(f"  训练时间: {time_standard:.2f}秒")
    print(f"  吞吐量: {20/time_standard:.2f} batch/s")

    print(f"\nVirtual Blackwell (FP4):")
    print(f"  训练时间: {time_vb:.2f}秒")
    print(f"  吞吐量: {20/time_vb:.2f} batch/s")

    print(f"\n加速效果:")
    print(f"  {'[>>] ' if speedup > 1 else '[!]  '}时间加速: {speedup:.2f}×")
    print(f"  {'[>>] ' if throughput_improvement > 1 else '[!]  '}吞吐量提升: {throughput_improvement:.2f}×")

    if speedup > 1:
        print(f"\n[OK] Virtual Blackwell 比标准PyTorch快 {(speedup-1)*100:.1f}%")
    else:
        print(f"\n[!]  Virtual Blackwell 比标准PyTorch慢 {(1-speedup)*100:.1f}%")
        print(f"   (这可能是因为模型太小，FP4编解码开销超过了收益)")

    # 精度对比
    print(f"\n精度对比:")
    print(f"  标准PyTorch Loss: {loss_standard:.6f}")
    print(f"  Virtual Blackwell Loss: {loss_vb:.6f}")
    print(f"  Loss差异: {abs(loss_standard - loss_vb):.6f}")

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
