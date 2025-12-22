#!/usr/bin/env python3
"""
可视化系统演示脚本
生成模拟训练数据，展示可视化效果
"""

import json
import math
from pathlib import Path
import time

def generate_demo_data(epochs=50):
    """生成模拟训练数据"""

    # 创建演示目录
    demo_dir = Path('demo_visualization')
    demo_dir.mkdir(exist_ok=True)

    # 模拟训练数据
    control_losses = []
    autopoietic_losses = []

    print("🎬 生成演示数据...")
    print(f"   模拟 {epochs} 个epoch的训练过程\n")

    for epoch in range(1, epochs + 1):
        # 对照组: 下降较慢，有些波动
        control_loss = 5.0 * math.exp(-epoch/30) + 0.5 + 0.2 * math.sin(epoch/5)

        # 实验组: 下降更快，更平滑（展示APT优势）
        autopoietic_loss = 5.0 * math.exp(-epoch/20) + 0.3 + 0.1 * math.sin(epoch/7)

        control_losses.append(float(control_loss))
        autopoietic_losses.append(float(autopoietic_loss))

        # 保存中间报告（模拟实时训练）
        report = {
            'control_losses': control_losses,
            'autopoietic_losses': autopoietic_losses,
            'current_epoch': epoch,
            'timestamp': time.time()
        }

        report_path = demo_dir / 'experiment_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # 进度显示
        if epoch % 10 == 0:
            print(f"   ✓ Epoch {epoch}/{epochs} | "
                  f"Control: {control_loss:.4f} | "
                  f"APT: {autopoietic_loss:.4f}")

    print(f"\n✅ 演示数据已保存到: {demo_dir}/")
    print(f"   对照组最终Loss: {control_losses[-1]:.4f}")
    print(f"   实验组最终Loss: {autopoietic_losses[-1]:.4f}")
    print(f"   APT改进: {(control_losses[-1] - autopoietic_losses[-1]) / control_losses[-1] * 100:.1f}%")

    print("\n🚀 启动可视化:")
    print(f"   python visualize_training.py --log-dir demo_visualization --offline")

if __name__ == "__main__":
    generate_demo_data(epochs=50)
