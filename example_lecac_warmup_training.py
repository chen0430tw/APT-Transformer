#!/usr/bin/env python3
"""
LECaC Soft Warmup 训练示例
==========================
演示如何在训练中使用Alpha Compensation Warmup解决低学习率不稳定问题
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# 导入LECaC和Warmup调度器
from apt.vgpu.runtime.lecac import LECACLinear, replace_linear_with_lecac
from apt.vgpu.runtime.lecac_warmup import (
    LECACAlphaScheduler,
    update_lecac_alpha
)

print("=" * 70)
print("LECaC Soft Warmup 训练示例")
print("=" * 70)

# ============================================================================
# 1. 创建模型（使用LECaC）
# ============================================================================

class SimpleLM(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = LECACLinear(embed_dim, hidden_dim, bits=2)  # INT2量化
        self.fc2 = LECACLinear(hidden_dim, vocab_size, bits=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleLM().to(device)

print(f"\n[模型] SimpleLM with LECaC INT2")
print(f"  - 设备: {device}")
print(f"  - LECaC层数: 2")

# ============================================================================
# 2. 配置训练超参数（模拟TWCC环境）
# ============================================================================

# 训练配置
total_steps = 1000
lr_warmup_steps = 100        # 学习率warmup步数
lecac_warmup_steps = 100     # LECaC alpha warmup步数（建议与lr_warmup对齐）

# 学习率配置（TWCC失败的配置）
min_lr = 3e-6   # Warmup起始lr（很低，容易触发量化不稳定）
max_lr = 3e-4   # 目标lr
base_lr = 3e-4

# 优化器
optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)

# 学习率调度器（linear warmup）
def lr_lambda(step):
    if step < lr_warmup_steps:
        # Linear warmup
        return (min_lr + (max_lr - min_lr) * step / lr_warmup_steps) / base_lr
    else:
        return 1.0

lr_scheduler = LambdaLR(optimizer, lr_lambda)

print(f"\n[训练配置]")
print(f"  - 总步数: {total_steps}")
print(f"  - 学习率: {min_lr:.2e} → {max_lr:.2e} (warmup {lr_warmup_steps} steps)")

# ============================================================================
# 3. 创建LECaC Alpha Warmup调度器 🔑
# ============================================================================

alpha_scheduler = LECACAlphaScheduler(
    warmup_steps=lecac_warmup_steps,
    base_alpha=4.0 / 2.718,  # 4/e ≈ 1.47
    warmup_multiplier=3.0,   # Warmup期间: 1.47 × 3 = 4.41
    schedule="cosine"        # 平滑过渡
)

print(f"\n[LECaC Warmup] ✨ 关键优化")
print(f"  - Alpha范围: 4.41 → 1.47 (warmup {lecac_warmup_steps} steps)")
print(f"  - 调度策略: cosine")
print(f"  - 物理意义: 低lr时增强梯度补偿，对抗量化噪声")

# ============================================================================
# 4. 训练循环（带Alpha Warmup）
# ============================================================================

print(f"\n{'='*70}")
print("开始训练")
print(f"{'='*70}\n")

criterion = nn.CrossEntropyLoss()
nan_count = 0

for step in range(total_steps):
    # ========================================================================
    # 🔑 核心：更新LECaC alpha
    # ========================================================================
    current_alpha = alpha_scheduler.get_alpha(step)
    num_updated = update_lecac_alpha(model, current_alpha)

    # 训练步骤
    optimizer.zero_grad()

    # 生成随机数据
    batch_size = 32
    seq_len = 64
    x = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
    target = torch.randint(0, 1000, (batch_size, seq_len)).to(device)

    # Forward
    output = model(x)

    # 计算loss
    loss = criterion(output.view(-1, 1000), target.view(-1))

    # Backward
    loss.backward()
    optimizer.step()
    lr_scheduler.step()

    # 检查NaN
    if torch.isnan(loss):
        nan_count += 1

    # 日志输出
    if step % 10 == 0 or step < 10:
        current_lr = optimizer.param_groups[0]['lr']
        status = "❌ NaN" if torch.isnan(loss) else "✅"

        print(f"Step {step:4d}/{total_steps} | "
              f"Loss: {loss.item():.4f} {status} | "
              f"LR: {current_lr:.2e} | "
              f"Alpha: {current_alpha:.4f}")

        # Warmup结束标记
        if step == lr_warmup_steps:
            print(f"\n{'='*70}")
            print(f"✅ Warmup完成 (step {step})")
            print(f"{'='*70}\n")

# ============================================================================
# 5. 训练结果总结
# ============================================================================

print(f"\n{'='*70}")
print("训练完成")
print(f"{'='*70}")

print(f"\n[结果统计]")
print(f"  - 总步数: {total_steps}")
print(f"  - NaN步数: {nan_count}")
print(f"  - 成功率: {(1 - nan_count/total_steps) * 100:.1f}%")

if nan_count == 0:
    print(f"\n🎉 成功！LECaC Soft Warmup有效解决了低学习率不稳定问题")
else:
    print(f"\n⚠️  仍有 {nan_count} 步出现NaN，可能需要调整warmup参数")

# 检查最终参数健康度
print(f"\n[参数健康度检查]")
has_issue = False
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        print(f"  ❌ {name}: 包含NaN")
        has_issue = True
    elif torch.isinf(param).any():
        print(f"  ❌ {name}: 包含Inf")
        has_issue = True

if not has_issue:
    print(f"  ✅ 所有参数正常")

# ============================================================================
# 6. Alpha调度可视化（可选）
# ============================================================================

print(f"\n[Alpha调度曲线]")
print(f"{'Step':<8} {'Alpha':<8} {'LR':<12}")
print("-" * 30)

for step in [0, 10, 20, 50, 99, 100, 200, 500, 999]:
    alpha = alpha_scheduler.get_alpha(step)
    lr = lr_lambda(step) * base_lr if step < total_steps else base_lr
    print(f"{step:<8} {alpha:<8.4f} {lr:<12.2e}")

print(f"\n{'='*70}")
print("示例完成")
print(f"{'='*70}")

print(f"\n[使用建议]")
print(f"1. LECaC warmup步数建议与学习率warmup对齐")
print(f"2. warmup_multiplier建议范围: 2.0-4.0")
print(f"3. 如果仍有NaN: 提高multiplier或延长warmup_steps")
print(f"4. Cosine调度比linear更平滑，推荐使用")
