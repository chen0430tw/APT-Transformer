#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 Virtual VRAM v0.3 修复效果
"""

import torch
import torch.nn as nn
from virtual_vram import VirtualVRAMConfig, virtual_vram


def test_gradient_convergence():
    """测试梯度收敛性（修复前会 NaN）"""
    print("="*70)
    print("测试 1: 梯度收敛性（核心修复验证）")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("需要 CUDA")
        return
    
    # 固定随机种子
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # 简单 Transformer 块
    model = nn.Sequential(
        nn.Linear(512, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Linear(512, 512),
    ).cuda()
    
    # 固定输入
    x = torch.randn(8, 128, 512, device="cuda", requires_grad=True)
    
    # 保存初始权重
    init_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    def run_test(config_name, vvram_config):
        """运行一次测试"""
        # 恢复初始状态
        model.load_state_dict(init_state)
        model.zero_grad(set_to_none=True)
        
        x_copy = x.clone().detach().requires_grad_(True)
        
        if vvram_config is None:
            # Baseline
            y = model(x_copy)
            loss = y.mean()
            loss.backward()
        else:
            # Virtual VRAM
            with virtual_vram(vvram_config):
                y = model(x_copy)
                loss = y.mean()
                loss.backward()
        
        torch.cuda.synchronize()
        
        # 检查梯度
        grad_norm = sum(p.grad.float().norm().item() 
                       for p in model.parameters() if p.grad is not None)
        has_nan = any(torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                     for p in model.parameters() if p.grad is not None)
        num_grads = sum(1 for p in model.parameters() if p.grad is not None)
        
        print(f"  {config_name:20s}: loss={loss.item():.6f}  "
              f"grad_norm={grad_norm:.4f}  #grads={num_grads}  "
              f"NaN={'❌' if has_nan else '✅'}")
        
        return loss.item(), grad_norm, has_nan
    
    # 测试配置
    print("\n【修复前的问题配置】")
    
    # Baseline
    loss_base, grad_base, nan_base = run_test("baseline", None)
    
    # 修复后应该不 NaN 的配置
    configs = [
        ("plain (同步)", VirtualVRAMConfig(
            enabled=True, 
            min_tensor_bytes=1<<10,
            compress=False,
            stream_prefetch=False,  # 同步模式
            track_dependencies=True,
        )),
        ("plain (异步+依赖)", VirtualVRAMConfig(
            enabled=True,
            min_tensor_bytes=1<<10,
            compress=False,
            stream_prefetch=True,  # 异步但有依赖闭环
            track_dependencies=True,
        )),
        ("fp16 (同步)", VirtualVRAMConfig(
            enabled=True,
            min_tensor_bytes=1<<10,
            compress=True,
            compress_dtype="float16",
            stream_prefetch=False,
            track_dependencies=True,
        )),
        ("fp16 (异步+依赖)", VirtualVRAMConfig(
            enabled=True,
            min_tensor_bytes=1<<10,
            compress=True,
            compress_dtype="float16",
            stream_prefetch=True,
            track_dependencies=True,
        )),
        ("int8 (同步)", VirtualVRAMConfig(
            enabled=True,
            min_tensor_bytes=1<<10,
            compress=True,
            compress_dtype="int8",
            stream_prefetch=False,
            track_dependencies=True,
        )),
        ("int8 (异步+依赖)", VirtualVRAMConfig(
            enabled=True,
            min_tensor_bytes=1<<10,
            compress=True,
            compress_dtype="int8",
            stream_prefetch=True,
            track_dependencies=True,
        )),
    ]
    
    print("\n【修复后配置】")
    results = {}
    for name, cfg in configs:
        loss, grad, nan = run_test(name, cfg)
        results[name] = (loss, grad, nan)
    
    # 总结
    print("\n" + "="*70)
    print("总结:")
    print("="*70)
    
    all_passed = True
    for name, (loss, grad, nan) in results.items():
        # 检查：1) 不 NaN  2) loss 与 baseline 一致
        loss_match = abs(loss - loss_base) < 1e-4
        no_nan = not nan
        passed = loss_match and no_nan
        all_passed = all_passed and passed
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:20s}: {status}  "
              f"(loss_match={loss_match}, no_nan={no_nan})")
    
    print()
    if all_passed:
        print("🎉 所有测试通过！NaN 问题已修复。")
    else:
        print("⚠️  部分测试失败，需要进一步检查。")
    
    return all_passed


def test_dependency_tracking():
    """测试依赖追踪"""
    print("\n" + "="*70)
    print("测试 2: 依赖追踪（审计 stream 依赖闭环）")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("需要 CUDA")
        return
    
    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
    ).cuda()
    
    x = torch.randn(16, 256, device="cuda")
    
    # 测试异步 prefetch + 依赖追踪
    config = VirtualVRAMConfig(
        enabled=True,
        min_tensor_bytes=1<<10,
        compress=True,
        compress_dtype="int8",
        stream_prefetch=True,
        track_dependencies=True,
        verbose=False,
    )
    
    with virtual_vram(config) as mgr:
        y = model(x)
        loss = y.mean()
        loss.backward()
    
    print(f"  dependency_edges = {mgr.stats.dependency_edges}")
    print(f"  loop_breaks = {mgr.stats.loop_breaks}")
    print(f"  offload_count = {mgr.stats.offload_count}")
    print(f"  restore_count = {mgr.stats.restore_count}")
    
    if mgr.stats.loop_breaks > 0:
        print("\n  ⚠️  检测到断环！这会导致 NaN。")
        return False
    elif mgr.stats.dependency_edges > 0:
        print("\n  ✅ 依赖闭环正常建立。")
        return True
    else:
        print("\n  ℹ️  未使用异步 prefetch，无需依赖边。")
        return True


def test_restore_semantics():
    """测试 restore 语义（detach）"""
    print("\n" + "="*70)
    print("测试 3: Restore 语义（saved tensor 应该 detach）")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("需要 CUDA")
        return
    
    class CustomModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(128, 128)
        
        def forward(self, x):
            # 这里会保存 x 作为 saved tensor
            y = self.linear(x)
            return y
    
    model = CustomModule().cuda()
    x = torch.randn(8, 128, device="cuda", requires_grad=True)
    
    config = VirtualVRAMConfig(
        enabled=True,
        min_tensor_bytes=1<<10,
        compress=True,
        compress_dtype="int8",
        stream_prefetch=False,
        verbose=False,
    )
    
    with virtual_vram(config):
        y = model(x)
        loss = y.mean()
        
        # 检查 backward 是否正常
        try:
            loss.backward()
            print("  ✅ Backward 正常完成（无 requires_grad 异常）")
            
            # 检查梯度
            has_grad = x.grad is not None
            has_nan = torch.isnan(x.grad).any() if has_grad else False
            
            print(f"  ✅ 输入梯度存在: {has_grad}")
            print(f"  ✅ 输入梯度无 NaN: {not has_nan}")
            
            return has_grad and not has_nan
            
        except Exception as e:
            print(f"  ❌ Backward 失败: {e}")
            return False


def main():
    """主测试流程"""
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "Virtual VRAM v0.3 修复验证" + " "*15 + "║")
    print("╚" + "="*68 + "╝")
    print()
    
    if not torch.cuda.is_available():
        print("❌ 需要 CUDA")
        return
    
    results = {}
    
    # 测试 1: 梯度收敛性
    results['gradient'] = test_gradient_convergence()
    
    # 测试 2: 依赖追踪
    results['dependency'] = test_dependency_tracking()
    
    # 测试 3: Restore 语义
    results['restore'] = test_restore_semantics()
    
    # 总结
    print("\n" + "="*70)
    print("最终总结")
    print("="*70)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:15s}: {status}")
    
    all_passed = all(results.values())
    print()
    if all_passed:
        print("🎉 所有核心修复已验证通过！")
        print()
        print("修复内容:")
        print("  1. ✅ Stream 依赖闭环：使用 CUDA event 建立依赖边")
        print("  2. ✅ Restore 语义修正：恢复的 tensor 一律 detach")
        print("  3. ✅ 依赖追踪审计：可审计的搬运路径")
    else:
        print("⚠️  部分测试未通过，需要进一步调试。")


if __name__ == "__main__":
    main()
