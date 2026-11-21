#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速验证 virtual_vram（apt/vgpu/runtime/virtual_vram.py, v1.6.2）

v0.3 的 compress / compress_dtype / stream_prefetch 参数已移除：
  - 量化压缩已弃用（saved_tensors_hooks 中量化会破坏 backward）
  - stream_prefetch → enable_prefetch（v1.6.2 backward 倒序预取）
"""

import os
import sys
import torch
import torch.nn as nn

# 从仓库根导入 apt 包（va100/ 的上一级）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apt.vgpu.runtime.virtual_vram import VirtualVRAMConfig, virtual_vram


def test_basic_functionality():
    """测试基本功能"""
    print("="*70)
    print("测试 1: 基本功能验证")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("❌ 需要 CUDA")
        return False
    
    # 简单模型
    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
    ).cuda()
    
    x = torch.randn(16, 256, device="cuda", requires_grad=True)
    
    # 测试配置
    config = VirtualVRAMConfig(
        enabled=True,
        min_tensor_bytes=1 << 10,  # 1 KB
        enable_prefetch=True,
        verbose=True,
    )
    
    print("\n[Baseline]")
    y = model(x)
    loss = y.mean()
    loss.backward()
    
    baseline_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    print(f"  grad_norm = {baseline_grad_norm:.4f}")
    
    model.zero_grad()
    x.grad = None
    
    print("\n[Virtual VRAM v1.6.2]")
    with virtual_vram(config):
        y = model(x)
        loss = y.mean()
        loss.backward()
    
    vvram_grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    has_nan = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
    
    print(f"  grad_norm = {vvram_grad_norm:.4f}")
    print(f"  has_nan = {has_nan}")
    
    # 检查
    if has_nan:
        print("\n❌ 测试失败：仍然有 NaN！")
        return False
    elif abs(vvram_grad_norm - baseline_grad_norm) / baseline_grad_norm > 1e-5:
        # 无量化：offload 是无损拷贝，梯度应与 baseline 一致
        print(f"\n❌ 测试失败：梯度与 baseline 不一致 ({abs(vvram_grad_norm - baseline_grad_norm) / baseline_grad_norm:.2e})")
        return False
    else:
        print("\n✅ 测试通过：无 NaN，梯度一致")
        return True


def test_all_modes():
    """测试所有模式"""
    print("\n" + "="*70)
    print("测试 2: 所有模式验证")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("❌ 需要 CUDA")
        return False
    
    torch.manual_seed(42)
    
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
    ).cuda()
    
    x = torch.randn(8, 128, device="cuda")
    
    # 测试模式
    modes = [
        ("legacy (同步)", VirtualVRAMConfig(
            enabled=True, min_tensor_bytes=1<<10)),
        ("legacy (预取)", VirtualVRAMConfig(
            enabled=True, min_tensor_bytes=1<<10, enable_prefetch=True)),
        ("nested (同步)", VirtualVRAMConfig(
            enabled=True, min_tensor_bytes=1<<10, enable_nested_v16=True)),
        ("nested (预取)", VirtualVRAMConfig(
            enabled=True, min_tensor_bytes=1<<10, enable_nested_v16=True,
            enable_prefetch=True)),
    ]
    
    all_passed = True
    
    for name, cfg in modes:
        model.zero_grad()
        
        try:
            with virtual_vram(cfg):
                y = model(x)
                loss = y.mean()
                loss.backward()
            
            grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
            has_nan = any(torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                         for p in model.parameters() if p.grad is not None)
            
            status = "✅" if not has_nan else "❌"
            print(f"  {name:15s}: grad_norm={grad_norm:.4f}  {status}")
            
            if has_nan:
                all_passed = False
                
        except Exception as e:
            print(f"  {name:15s}: ❌ 异常 - {e}")
            all_passed = False
    
    return all_passed


def main():
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "Virtual VRAM v1.6.2 快速验证" + " "*13 + "║")
    print("╚" + "="*68 + "╝")
    print()
    
    results = []
    
    # 测试 1
    try:
        results.append(("基本功能", test_basic_functionality()))
    except Exception as e:
        print(f"\n❌ 测试 1 失败: {e}")
        results.append(("基本功能", False))
    
    # 测试 2
    try:
        results.append(("所有模式", test_all_modes()))
    except Exception as e:
        print(f"\n❌ 测试 2 失败: {e}")
        results.append(("所有模式", False))
    
    # 总结
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:15s}: {status}")
    
    all_passed = all(r[1] for r in results)
    print()
    if all_passed:
        print("🎉 所有测试通过！")
    else:
        print("⚠️  部分测试失败，需要进一步检查。")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
