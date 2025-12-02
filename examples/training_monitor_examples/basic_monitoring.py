#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SOSA训练监控基础示例

演示如何使用SOSA监控训练过程
"""

import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO)

from apt_model.plugins.training_monitor_plugin import create_training_monitor_plugin


# 简单的模型
class SimpleModel(nn.Module):
    def __init__(self, input_dim=10, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def main():
    print("=" * 70)
    print("SOSA 训练监控基础示例")
    print("=" * 70)

    # 1. 创建模型和优化器
    print("\n[步骤1] 创建模型和优化器...")
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 2. 创建SOSA监控插件
    print("\n[步骤2] 创建SOSA监控插件...")
    plugin = create_training_monitor_plugin(
        model=model,
        optimizer=optimizer,
        config={
            'auto_fix': True,  # 启用自动修复
            'checkpoint_dir': './sosa_checkpoints',
            'window_seconds': 10.0,
            'max_fixes_per_error': 3
        }
    )

    # 3. 定义前向函数
    def forward_fn(model, batch):
        pred = model(batch['x'])
        loss = nn.functional.mse_loss(pred, batch['y'])
        return loss

    # 4. 模拟训练
    print("\n[步骤3] 开始训练...")
    num_steps = 100

    for step in range(num_steps):
        # 创建假数据
        batch = {
            'x': torch.randn(8, 10),
            'y': torch.randn(8, 1)
        }

        # 执行训练步 (自动监控和修复)
        loss = plugin.training_step(batch, forward_fn)

        # 每10步打印
        if step % 10 == 0:
            print(f"  Step {step}: Loss = {loss.item():.4f}")

        # 每30步打印报告
        if step % 30 == 0 and step > 0:
            print(f"\n  [报告 @ Step {step}]")
            plugin.print_report()

    # 5. 最终报告
    print("\n" + "=" * 70)
    print("训练完成 - 最终报告")
    print("=" * 70)
    plugin.print_report()

    # 6. 获取统计
    stats = plugin.get_statistics()
    print("\n关键统计:")
    print(f"  总步数: {stats['global_step']}")
    print(f"  最佳Loss: {stats['best_loss']:.4f}")
    print(f"  异常检测: {stats['total_errors']} 次")
    print(f"  自动修复: {stats['successful_fixes']} 次")

    print("\n" + "=" * 70)
    print("示例完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
