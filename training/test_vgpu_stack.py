"""
测试虚拟Blackwell堆叠技术

测试场景：
1. 基础堆叠：多层缓存管理
2. 自动提升：热数据向Level 0迁移
3. 负载测试：大量tensor管理
4. 集成测试：与神经网络层集成
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import time
from apt_model.optimization.vgpu_stack import (
    create_vgpu_stack,
    VGPUStack,
    VGPUStackLinear
)


def test_basic_stack():
    """测试1：基础堆叠功能"""
    print("="*70)
    print("测试1: 基础堆叠功能")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建小容量堆叠（测试淘汰）
    config = {
        'levels': [
            {'capacity_mb': 10, 'device': device, 'speed_gbps': 900},   # Level 0: 10MB
            {'capacity_mb': 50, 'device': 'cpu', 'speed_gbps': 50},     # Level 1: 50MB
            {'capacity_mb': 200, 'device': 'ssd', 'speed_gbps': 7}      # Level 2: 200MB
        ]
    }
    stack = VGPUStack(config)

    # 注册20个矩阵（每个约1MB，总共20MB）
    print(f"\n注册20个1MB矩阵（Level 0只有10MB）...")
    tensors = []
    for i in range(20):
        # 256x512 float32 ≈ 0.5MB
        W = torch.randn(256, 512, device=device) * 0.02
        stack.register(f'weight_{i}', W)
        tensors.append(W)

    print("\n初始状态:")
    stack.print_stats()

    # 访问热点数据
    print("\n频繁访问前5个矩阵...")
    for _ in range(50):
        for i in range(5):
            W = stack.access(f'weight_{i}')

    print("\n热点访问后:")
    stack.print_stats()

    print("\n✅ 测试1完成\n")


def test_promotion():
    """测试2：自动提升机制"""
    print("="*70)
    print("测试2: 自动提升机制")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    stack = create_vgpu_stack()

    # 注册到不同层级
    W_cold = torch.randn(128, 128, device=device)
    W_warm = torch.randn(128, 128, device=device)
    W_hot = torch.randn(128, 128, device=device)

    stack.register('cold', W_cold, priority=1)
    stack.register('warm', W_warm, priority=5)
    stack.register('hot', W_hot, priority=9)

    print("\n初始注册:")
    print(f"  cold在Level {stack.tensor_directory.get('cold', -1)}")
    print(f"  warm在Level {stack.tensor_directory.get('warm', -1)}")
    print(f"  hot在Level {stack.tensor_directory.get('hot', -1)}")

    # 访问cold 100次，让它提升
    print("\n访问'cold' 100次...")
    for _ in range(100):
        stack.access('cold')

    print("\n访问后:")
    print(f"  cold在Level {stack.tensor_directory.get('cold', -1)} (应该提升到Level 0)")
    print(f"  warm在Level {stack.tensor_directory.get('warm', -1)}")
    print(f"  hot在Level {stack.tensor_directory.get('hot', -1)}")

    stack.print_stats()

    print("\n✅ 测试2完成\n")


def test_load():
    """测试3：负载测试"""
    print("="*70)
    print("测试3: 负载测试（1000个tensor）")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    stack = create_vgpu_stack()

    # 注册1000个小矩阵
    print("\n注册1000个矩阵...")
    start = time.time()
    for i in range(1000):
        W = torch.randn(64, 64, device=device) * 0.02
        stack.register(f'w_{i}', W)
    register_time = time.time() - start
    print(f"  注册耗时: {register_time:.2f}s")

    # 随机访问测试
    print("\n随机访问10000次...")
    import random
    start = time.time()
    for _ in range(10000):
        key = f'w_{random.randint(0, 999)}'
        W = stack.access(key)
    access_time = time.time() - start
    print(f"  访问耗时: {access_time:.2f}s")
    print(f"  平均访问: {access_time/10000*1000:.2f}ms")

    stack.print_stats()

    print("\n✅ 测试3完成\n")


def test_nn_integration():
    """测试4：神经网络集成"""
    print("="*70)
    print("测试4: 神经网络集成")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建堆叠
    stack = create_vgpu_stack()

    # 创建使用VGPU堆叠的网络
    class TestNet(nn.Module):
        def __init__(self, vgpu_stack):
            super().__init__()
            self.fc1 = VGPUStackLinear(512, 1024, vgpu_stack)
            self.fc2 = VGPUStackLinear(1024, 1024, vgpu_stack)
            self.fc3 = VGPUStackLinear(1024, 512, vgpu_stack)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = TestNet(stack).to(device)
    print(f"\n创建3层网络，设备: {device}")

    # 前向传播测试
    print("\n前向传播100次...")
    x = torch.randn(32, 512, device=device)

    start = time.time()
    for i in range(100):
        with torch.no_grad():
            y = model(x)
        if (i + 1) % 25 == 0:
            print(f"  Batch {i+1}/100")
    forward_time = time.time() - start

    print(f"\n耗时: {forward_time:.2f}s")
    print(f"平均: {forward_time/100*1000:.2f}ms/batch")

    stack.print_stats()

    print("\n✅ 测试4完成\n")


def test_comparison():
    """测试5：对比标准PyTorch"""
    print("="*70)
    print("测试5: 性能对比（VGPU Stack vs 标准PyTorch）")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 标准PyTorch
    print("\n[标准PyTorch]")
    model_std = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512)
    ).to(device)

    x = torch.randn(32, 512, device=device)

    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            y = model_std(x)
    time_std = time.time() - start
    print(f"  100次前向传播: {time_std:.3f}s")

    # VGPU Stack
    print("\n[VGPU Stack]")
    stack = create_vgpu_stack()

    class VGPUNet(nn.Module):
        def __init__(self, vgpu_stack):
            super().__init__()
            self.fc1 = VGPUStackLinear(512, 1024, vgpu_stack)
            self.fc2 = VGPUStackLinear(1024, 1024, vgpu_stack)
            self.fc3 = VGPUStackLinear(1024, 512, vgpu_stack)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model_vgpu = VGPUNet(stack).to(device)

    # 复制权重（公平对比）
    with torch.no_grad():
        model_vgpu.fc1.weight.copy_(model_std[0].weight)
        model_vgpu.fc1.bias.copy_(model_std[0].bias)
        model_vgpu.fc2.weight.copy_(model_std[2].weight)
        model_vgpu.fc2.bias.copy_(model_std[2].bias)
        model_vgpu.fc3.weight.copy_(model_std[4].weight)
        model_vgpu.fc3.bias.copy_(model_std[4].bias)

    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            y = model_vgpu(x)
    time_vgpu = time.time() - start
    print(f"  100次前向传播: {time_vgpu:.3f}s")

    print("\n[性能对比]")
    print(f"  标准PyTorch: {time_std:.3f}s")
    print(f"  VGPU Stack:  {time_vgpu:.3f}s")
    overhead = (time_vgpu - time_std) / time_std * 100
    print(f"  开销: {overhead:+.1f}%")

    if overhead < 20:
        print("  ✅ 开销可接受（<20%）")
    else:
        print("  ⚠️ 开销较高，需要优化")

    stack.print_stats()

    print("\n✅ 测试5完成\n")


def main():
    """运行所有测试"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║            虚拟Blackwell堆叠技术测试套件                            ║")
    print("║            VGPU Stack - Multi-Level Memory Hierarchy               ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print("\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"检测到设备: {device}")
    if device == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print("\n")

    tests = [
        ("基础堆叠", test_basic_stack),
        ("自动提升", test_promotion),
        ("负载测试", test_load),
        ("网络集成", test_nn_integration),
        ("性能对比", test_comparison),
    ]

    for name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n❌ 测试 '{name}' 失败: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("="*70)
    print("所有测试完成！")
    print("="*70)


if __name__ == "__main__":
    main()
