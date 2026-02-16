#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CompAct 随机投影核 - 使用示例
=============================

演示如何在 Virtual Blackwell 训练中集成随机投影核

作者：GPT-5.2 R2
版本：1.0.0
"""

import sys
import io

# Windows 控制台编码修复
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
import torch.nn as nn
from random_projection_kernel import (
    RandomProjectionKernel,
    ProjectionKernelConfig,
    CompActLinear,
    compact_act_forward,
    estimate_memory,
)


def demo_memory_estimation():
    """演示内存估算"""
    print("=" * 70)
    print("💾 随机投影核内存估算")
    print("=" * 70)
    print()

    models = [
        ("LLaMA-7B", 32, 4096),
        ("LLaMA-13B", 40, 5120),
        ("LLaMA-30B", 60, 6656),
        ("LLaMA-65B", 80, 8192),
    ]

    ranks = [128, 256, 512, 1024]

    print(f"{'模型':<15} {'层数':<6} {'hidden':<8} {'rank=128':<12} {'rank=256':<12} {'rank=512':<12}")
    print("-" * 70)

    for name, layers, hidden in models:
        row = f"{name:<15} {layers:<6} {hidden:<8}"
        for rank in ranks:
            mem = estimate_memory(layers, hidden, rank)
            row += f"{mem['total_gb']:>6.2f} GB  "
        print(row)

    print()

    # 对比：原 CompAct vs 我们的优化
    print("对比：原 CompAct vs 随机投影核优化")
    print()
    print("  原 CompAct:")
    print("    - 不存储 P")
    print("    - 每次反向传播重新生成（慢）")
    print("    - 内存: 0 MB（但计算开销大）")
    print()
    print("  随机投影核（本实现）:")
    print("    - 存储 P^T")
    print("    - 反向传播直接使用（快）")
    print(f"    - 内存: LLaMA-65B (rank=512) → {estimate_memory(80, 8192, 512)['total_gb']:.2f} GB")
    print()


def demo_basic_usage():
    """演示基本使用"""
    print("=" * 70)
    print("🚀 基本使用示例")
    print("=" * 70)
    print()

    # 创建配置
    config = ProjectionKernelConfig(
        rank=512,
        distribution="gaussian",
        seed_mode="per_layer",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 创建核管理器
    kernel_mgr = RandomProjectionKernel(config, global_seed=42)

    # 注册层
    hidden_dim = 4096
    for i in range(32):  # LLaMA-7B 有 32 层
        layer_id = f"transformer.h.{i}"
        P = kernel_mgr.register_layer(layer_id, n=hidden_dim)

        if i == 0:
            print(f"注册层 {layer_id}:")
            print(f"  投影矩阵 P 形状: {P.shape}")
            print(f"  伴随矩阵 P^T 形状: {kernel_mgr.get_adjoint(layer_id).shape}")
            print()

    # 前向传播
    B, T = 2, 128
    x = torch.randn(B, T, hidden_dim, device=config.device)

    print(f"输入激活: {x.shape}")
    z = compact_act_forward(x, "transformer.h.0", kernel_mgr)
    print(f"压缩激活: {z.shape}")
    print(f"压缩比: {hidden_dim / config.rank:.1f}x")
    print()


def demo_global_kernel():
    """演示全局核优化"""
    print("=" * 70)
    print("🌐 全局核优化（极致节省内存）")
    print("=" * 70)
    print()

    config = ProjectionKernelConfig(
        rank=512,
        seed_mode="global",
        global_transform="roll",  # 循环移位变换
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    kernel_mgr = RandomProjectionKernel(config, global_seed=42)

    # 初始化全局核
    hidden_dim = 8192
    rank = 512
    kernel_mgr.init_global_kernel(n=hidden_dim, r=rank)

    print(f"全局核形状: {kernel_mgr.global_kernel.shape}")
    print(f"全局核内存: {rank * hidden_dim * 4 / (1024**2):.2f} MB")
    print()

    # 注册多层（都从全局核派生）
    num_layers = 80
    for i in range(num_layers):
        layer_id = f"layer.{i}"
        kernel_mgr.register_layer(layer_id, n=hidden_dim)

    print(f"注册 {num_layers} 层后:")
    print(f"  总内存: {kernel_mgr.memory_usage_mb():.2f} MB")
    print(f"  对比（独立核）: {num_layers * rank * hidden_dim * 4 / (1024**2):.2f} MB")
    print(f"  节省: {100 * (1 - kernel_mgr.memory_usage_mb() / (num_layers * rank * hidden_dim * 4 / (1024**2))):.1f}%")
    print()


def demo_linear_layer():
    """演示集成到 nn.Linear"""
    print("=" * 70)
    print("🔧 集成到 nn.Linear")
    print("=" * 70)
    print()

    config = ProjectionKernelConfig(
        rank=256,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    kernel_mgr = RandomProjectionKernel(config)

    # 创建 CompAct 线性层
    layer = CompActLinear(
        in_features=4096,
        out_features=4096,
        kernel_manager=kernel_mgr,
        layer_id="test_layer",
        device=config.device,
    )

    print(f"CompAct 线性层:")
    print(f"  输入维度: {layer.in_features}")
    print(f"  输出维度: {layer.out_features}")
    print(f"  压缩秩: {config.rank}")
    print()

    # 前向传播
    x = torch.randn(2, 128, 4096, device=config.device)
    layer.train()
    y = layer(x)

    print(f"输入: {x.shape}")
    print(f"输出: {y.shape}")
    print()


def demo_gradient_flow():
    """演示梯度流"""
    print("=" * 70)
    print("📊 梯度流验证")
    print("=" * 70)
    print()

    config = ProjectionKernelConfig(
        rank=128,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    kernel_mgr = RandomProjectionKernel(config)

    # 注册层
    layer_id = "test_layer"
    hidden_dim = 1024
    kernel_mgr.register_layer(layer_id, n=hidden_dim)

    # 前向传播
    x = torch.randn(1, 16, hidden_dim, device=config.device, requires_grad=True)
    z = compact_act_forward(x, layer_id, kernel_mgr)

    # 损失
    loss = z.pow(2).mean()

    # 反向传播
    loss.backward()

    print(f"梯度检查:")
    print(f"  输入梯度形状: {x.grad.shape}")
    print(f"  输入梯度范数: {x.grad.norm():.6f}")
    print(f"  梯度是否为 NaN: {torch.isnan(x.grad).any()}")
    print(f"  梯度是否为 Inf: {torch.isinf(x.grad).any()}")
    print()

    # 验证伴随矩阵的使用
    P = kernel_mgr.get_projection(layer_id)
    P_adjoint = kernel_mgr.get_adjoint(layer_id)

    print(f"投影核验证:")
    print(f"  P 形状: {P.shape}")
    print(f"  P^T 形状: {P_adjoint.shape}")
    print(f"  P^T 是否等于 P.T: {torch.allclose(P_adjoint, P.T)}")
    print()


def demo_comparison_with_compact():
    """对比原 CompAct 方法"""
    print("=" * 70)
    print("⚡ 性能对比：原 CompAct vs 随机投影核")
    print("=" * 70)
    print()

    import time

    config = ProjectionKernelConfig(
        rank=512,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    kernel_mgr = RandomProjectionKernel(config)
    hidden_dim = 8192
    layer_id = "benchmark"

    # 注册层
    kernel_mgr.register_layer(layer_id, n=hidden_dim)

    # 测试数据
    B, T = 4, 256
    x = torch.randn(B, T, hidden_dim, device=config.device)

    # 暖身
    for _ in range(10):
        z = compact_act_forward(x, layer_id, kernel_mgr)

    # 测试前向传播
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(100):
        z = compact_act_forward(x, layer_id, kernel_mgr)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    forward_time = time.time() - start

    # 测试反向传播
    grad_output = torch.randn_like(z)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(100):
        grad_input = torch.autograd.grad(z, x, grad_output, retain_graph=True)[0]
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    backward_time = time.time() - start

    print(f"100 次迭代时间:")
    print(f"  前向传播: {forward_time*1000:.2f} ms")
    print(f"  反向传播: {backward_time*1000:.2f} ms")
    print(f"  总计: {(forward_time+backward_time)*1000:.2f} ms")
    print()

    print("优势分析:")
    print("  ✓ 反向传播不需要重新生成随机矩阵")
    print("  ✓ 直接使用存储的伴随矩阵 P^T")
    print("  ✓ 避免了随机数生成开销")
    print()


def main():
    print("""
╔════════════════════════════════════════════════════════════════╗
║            🎯 随机投影核 (Random Projection Kernel) 🎯          ║
║                     CompAct 优化实现 v1.0                       ║
╠════════════════════════════════════════════════════════════════╣
║  核心：存储伴随矩阵 P^T，反向传播直接使用                       ║
║  优势：避免重新生成随机矩阵，提升训练速度                      ║
╚════════════════════════════════════════════════════════════════╝
    """)

    demos = [
        ("内存估算", demo_memory_estimation),
        ("基本使用", demo_basic_usage),
        ("全局核优化", demo_global_kernel),
        ("集成 nn.Linear", demo_linear_layer),
        ("梯度流验证", demo_gradient_flow),
        ("性能对比", demo_comparison_with_compact),
    ]

    while True:
        print("\n🎯 选择演示:")
        for i, (name, _) in enumerate(demos, 1):
            print(f"  {i}. {name}")
        print("  0. 退出")
        print()

        choice = input("请选择 (0-6): ").strip()

        if choice == "0":
            print("\n👋 恭喜！你已掌握随机投影核优化！")
            break
        elif choice.isdigit() and 1 <= int(choice) <= len(demos):
            _, func = demos[int(choice) - 1]
            func()
        else:
            print("❌ 无效选择，请重试")


if __name__ == "__main__":
    main()
