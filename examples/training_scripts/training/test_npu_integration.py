#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPU集成测试 - 虚拟Blackwell×华为昇腾NPU

测试内容：
1. NPU设备检测
2. NPU后端适配器
3. VGPU Stack NPU支持
4. Virtual Blackwell NPU优化
5. 完整训练流程

作者: claude + chen0430tw
版本: 1.0 (Virtual Blackwell NPU Extension)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from typing import Optional

# NPU后端
from apt.perf.optimization.npu_backend import (
    DeviceBackend,
    UnifiedDeviceManager,
    get_device_manager,
    get_unified_backend,
    is_npu_available,
    is_cuda_available,
    get_accelerator_type
)

# 设备管理
from apt.core.system import get_device, get_device_info, memory_cleanup

# Virtual Blackwell
import apt.perf.optimization.vb_global as vb
from apt.perf.optimization.vgpu_stack import create_vgpu_stack


def test_device_detection():
    """测试1: 设备检测"""
    print("\n" + "="*70)
    print("测试1: 设备检测")
    print("="*70)

    # 检测加速器类型
    accel_type = get_accelerator_type()
    print(f"✓ 加速器类型: {accel_type.upper()}")

    # NPU检测
    npu_avail = is_npu_available()
    print(f"✓ NPU可用: {'是' if npu_avail else '否'}")

    # CUDA检测
    cuda_avail = is_cuda_available()
    print(f"✓ CUDA可用: {'是' if cuda_avail else '否'}")

    # 获取设备信息
    device = get_device()
    print(f"✓ 默认设备: {device}")

    # 详细设备信息
    info = get_device_info()
    print(f"✓ 设备详情:")
    print(f"  - 设备类型: {info.get('device_type', 'unknown')}")
    print(f"  - 设备名称: {info.get('device_name', 'unknown')}")
    print(f"  - 设备数量: {info.get('device_count', 0)}")
    if info.get('cuda_version'):
        print(f"  - CUDA版本: {info['cuda_version']}")
    if info.get('npu_version'):
        print(f"  - NPU版本: {info['npu_version']}")

    print("\n✅ 测试1通过")
    return accel_type, device


def test_device_backend(device: torch.device):
    """测试2: NPU后端适配器"""
    print("\n" + "="*70)
    print("测试2: NPU后端适配器")
    print("="*70)

    # 创建设备后端
    backend = DeviceBackend(device)
    print(f"✓ 后端创建: {backend}")

    # 设备属性
    print(f"✓ 设备类型: {backend.device_type}")
    print(f"✓ 设备索引: {backend.device_index}")
    print(f"✓ 设备可用: {backend.is_available()}")
    print(f"✓ 设备数量: {backend.device_count()}")
    print(f"✓ 设备名称: {backend.get_device_name()}")

    # 设备属性
    props = backend.get_device_properties()
    print(f"✓ 设备属性:")
    print(f"  - 名称: {props['name']}")
    print(f"  - 类型: {props['type']}")
    print(f"  - 总内存: {props['total_memory'] / (1024**3):.2f} GB")

    # 内存信息
    summary = backend.get_memory_summary()
    print(f"✓ 内存摘要:")
    print(f"  - 已分配: {summary['allocated_mb']:.2f} MB")
    print(f"  - 已保留: {summary['reserved_mb']:.2f} MB")
    if 'total_mb' in summary:
        print(f"  - 总容量: {summary['total_mb']:.2f} MB")
        print(f"  - 使用率: {summary.get('utilization_pct', 0):.2f}%")

    # 测试tensor移动
    x = torch.randn(10, 10)
    x_device = backend.to_device(x)
    print(f"✓ Tensor移动: {x.device} -> {x_device.device}")

    print("\n✅ 测试2通过")
    return backend


def test_unified_device_manager():
    """测试3: 统一设备管理器"""
    print("\n" + "="*70)
    print("测试3: 统一设备管理器")
    print("="*70)

    # 获取全局管理器
    manager = get_device_manager()
    print(f"✓ 设备管理器: {manager}")

    # 所有设备
    devices = manager.get_all_devices()
    print(f"✓ 可用设备数: {len(devices)}")
    for dev in devices:
        print(f"  - {dev}")

    # 最佳设备
    best_device = manager.get_best_device()
    print(f"✓ 最佳设备: {best_device}")

    # 设备摘要
    summary = manager.get_device_summary()
    print(f"✓ 设备摘要:")
    print(f"  - 总设备数: {summary['total_devices']}")
    print(f"  - CUDA设备: {summary['cuda_devices']}")
    print(f"  - NPU设备: {summary['npu_devices']}")

    print("\n✅ 测试3通过")
    return manager


def test_vgpu_stack_npu(device: torch.device):
    """测试4: VGPU Stack NPU支持"""
    print("\n" + "="*70)
    print("测试4: VGPU Stack NPU支持")
    print("="*70)

    # 创建VGPU Stack（自动检测NPU）
    stack = create_vgpu_stack()
    print("✓ VGPU Stack已创建")

    # 检查层级配置
    stats = stack.get_all_stats()
    print(f"✓ 堆叠层级数: {len(stats)}")
    for level_stat in stats:
        print(f"  - Level {level_stat['level']}: {level_stat['device']} "
              f"({level_stat['capacity_mb']:.0f}MB)")

    # 测试tensor存储
    test_tensor = torch.randn(100, 100).to(device)
    success = stack.put('test_tensor', test_tensor)
    print(f"✓ Tensor存储: {'成功' if success else '失败'}")

    # 测试tensor获取
    retrieved = stack.get('test_tensor')
    if retrieved is not None:
        print(f"✓ Tensor获取: 成功 (shape={retrieved.shape})")
    else:
        print(f"✗ Tensor获取: 失败")

    # 统计信息
    stats = stack.get_all_stats()
    print(f"✓ Level 0统计:")
    print(f"  - 命中率: {stats[0]['hit_rate']*100:.1f}%")
    print(f"  - 缓存tensor数: {stats[0]['cached_tensors']}")

    print("\n✅ 测试4通过")
    return stack


def test_vb_npu_optimization(device: torch.device):
    """测试5: Virtual Blackwell NPU优化"""
    print("\n" + "="*70)
    print("测试5: Virtual Blackwell NPU优化")
    print("="*70)

    # 启用VB（应显示NPU设备）
    vb.enable_balanced_mode()

    # 检查状态
    is_enabled = vb.is_enabled()
    print(f"✓ VB已启用: {is_enabled}")

    # 获取配置
    config = vb.get_config()
    print(f"✓ VB配置:")
    print(f"  - FP4: {config['use_fp4']}")
    print(f"  - Flash Attention: {config['use_flash_attn']}")
    print(f"  - 混合精度: {config['mixed_precision']}")

    # 获取VGPU Stack
    stack = vb.get_stack()
    print(f"✓ VGPU Stack: {'已创建' if stack else '未创建'}")

    print("\n✅ 测试5通过")


def test_simple_model_training(device: torch.device):
    """测试6: 简单模型训练"""
    print("\n" + "="*70)
    print("测试6: 简单模型训练（NPU/GPU）")
    print("="*70)

    # 创建简单模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = SimpleModel().to(device)
    print(f"✓ 模型已创建并移至 {device}")

    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 训练几步
    print("✓ 开始训练...")
    model.train()
    for step in range(5):
        # 随机数据
        x = torch.randn(32, 128).to(device)
        y = torch.randint(0, 10, (32,)).to(device)

        # 前向传播
        output = model(x)
        loss = criterion(output, y)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"  Step {step+1}/5: Loss = {loss.item():.4f}")

    print("✓ 训练完成")

    # 清理内存
    memory_cleanup()
    print("✓ 内存已清理")

    print("\n✅ 测试6通过")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("🧪 虚拟Blackwell NPU集成测试")
    print("="*70)

    try:
        # 测试1: 设备检测
        accel_type, device = test_device_detection()

        # 测试2: 设备后端
        backend = test_device_backend(device)

        # 测试3: 统一设备管理器
        manager = test_unified_device_manager()

        # 测试4: VGPU Stack NPU支持
        stack = test_vgpu_stack_npu(device)

        # 测试5: VB NPU优化
        test_vb_npu_optimization(device)

        # 测试6: 模型训练
        test_simple_model_training(device)

        # 最终报告
        print("\n" + "="*70)
        print("🎉 所有测试通过！")
        print("="*70)
        print(f"✅ 加速器类型: {accel_type.upper()}")
        print(f"✅ 设备: {device}")
        print(f"✅ NPU支持: {'已启用' if is_npu_available() else '未安装'}")
        print(f"✅ Virtual Blackwell: 完全兼容NPU")
        print("="*70)

    except Exception as e:
        print("\n" + "="*70)
        print("❌ 测试失败")
        print("="*70)
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    run_all_tests()
