#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 Virtual VRAM 修复效果（apt/vgpu/runtime/virtual_vram.py, v1.6.2）

原 v0.3 版的 compress / compress_dtype / stream_prefetch / track_dependencies
参数和 mgr.stats 已不存在：量化压缩已弃用，异步预取改为 enable_prefetch
（backward 倒序预取，CUDA event + record_stream 同步），统计改用
get_vram_perf_stats()。
"""

import os
import sys
import torch
import torch.nn as nn

# 从仓库根导入 apt 包（va100/ 的上一级）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from apt.vgpu.runtime.virtual_vram import (
    VirtualVRAMConfig, virtual_vram, get_vram_perf_stats, reset_vram_perf_stats,
)


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
        ("legacy (同步)", VirtualVRAMConfig(
            enabled=True, min_tensor_bytes=1<<10,
        )),
        ("legacy (预取)", VirtualVRAMConfig(
            enabled=True, min_tensor_bytes=1<<10, enable_prefetch=True,
        )),
        ("nested (同步)", VirtualVRAMConfig(
            enabled=True, min_tensor_bytes=1<<10, enable_nested_v16=True,
        )),
        ("nested (预取)", VirtualVRAMConfig(
            enabled=True, min_tensor_bytes=1<<10, enable_nested_v16=True,
            enable_prefetch=True,
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
        # 检查：1) 不 NaN  2) loss 与 baseline 一致  3) 梯度与 baseline 一致
        # （offload 只影响 backward，loss 必然一致；梯度才能暴露恢复错误）
        loss_match = abs(loss - loss_base) < 1e-4
        grad_match = abs(grad - grad_base) <= 1e-5 * max(abs(grad_base), 1.0)
        no_nan = not nan
        passed = loss_match and grad_match and no_nan
        all_passed = all_passed and passed

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name:20s}: {status}  "
              f"(loss_match={loss_match}, grad_match={grad_match}, no_nan={no_nan})")
    
    print()
    if all_passed:
        print("🎉 所有测试通过！NaN 问题已修复。")
    else:
        print("⚠️  部分测试失败，需要进一步检查。")
    
    return all_passed


def test_dependency_tracking():
    """测试异步预取的流同步（v1.6.2：D2H dma_event → 预取流 H2D → 计算流 wait_event）"""
    print("\n" + "="*70)
    print("测试 2: 异步预取（审计 offload / restore / 预取命中）")
    print("="*70)

    if not torch.cuda.is_available():
        print("需要 CUDA")
        return

    # 多层模型：倒序预取需要多个 saved tensor 才有"下一个"可预取
    torch.manual_seed(0)
    layers = []
    for _ in range(6):
        layers += [nn.Linear(256, 256), nn.ReLU()]
    model = nn.Sequential(*layers).cuda()

    x = torch.randn(64, 256, device="cuda")

    def grads():
        return torch.cat([p.grad.flatten() for p in model.parameters()])

    model.zero_grad(set_to_none=True)
    model(x).pow(2).mean().backward()
    g_ref = grads()

    config = VirtualVRAMConfig(
        enabled=True,
        min_tensor_bytes=1<<10,
        enable_prefetch=True,
        prefetch_cache_size=2,
        verbose=False,
    )

    model.zero_grad(set_to_none=True)
    reset_vram_perf_stats()
    with virtual_vram(config):
        model(x).pow(2).mean().backward()
    torch.cuda.synchronize()
    st = get_vram_perf_stats()
    g = grads()

    max_err = (g - g_ref).abs().max().item()
    print(f"  offload (D2H) = {int(st['pack_n'])}")
    print(f"  restore (H2D) = {int(st['unpack_n'])}")
    print(f"  prefetch issued/hit/miss = {int(st['prefetch_issued'])}/"
          f"{int(st['prefetch_hit'])}/{int(st['prefetch_miss'])}")
    print(f"  max|grad - baseline| = {max_err:.2e}")

    if st['pack_n'] == 0:
        print("\n  ❌ 没有发生 offload，测试无效。")
        return False
    if max_err > 1e-6:
        print("\n  ❌ 预取恢复的数据与 baseline 不一致（流同步有问题）。")
        return False
    if st['prefetch_hit'] == 0:
        print("\n  ❌ 开启了预取但没有命中，预取路径未生效。")
        return False
    print("\n  ✅ 预取命中且梯度与 baseline 一致。")
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
    print("║" + " "*15 + "Virtual VRAM v1.6.2 修复验证" + " "*13 + "║")
    print("╚" + "="*68 + "╝")
    print()
    
    if not torch.cuda.is_available():
        print("❌ 需要 CUDA")
        return
    
    results = {}
    
    # 测试 1: 梯度收敛性
    results['gradient'] = test_gradient_convergence()
    
    # 测试 2: 异步预取
    results['prefetch'] = test_dependency_tracking()
    
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
        print("  1. ✅ 各 offload 路径梯度与 baseline 一致")
        print("  2. ✅ 倒序预取：CUDA event + record_stream 流同步正确")
        print("  3. ✅ Restore 语义：backward 正常完成")
    else:
        print("⚠️  部分测试未通过，需要进一步调试。")
    return all_passed


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
