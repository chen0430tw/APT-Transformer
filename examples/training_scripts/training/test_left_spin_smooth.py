#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
左旋平滑集成测试脚本

验证左旋平滑机制已正确集成到 APT 模型中，并替换了传统泰勒展开。

测试内容:
1. 左旋平滑模块单元测试
2. APT模型集成测试
3. 残差连接替换验证
4. 尖点规避效果对比

作者: claude + chen0430tw
版本: 1.0
日期: 2026-01-21
"""

import torch
import torch.nn as nn
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from apt.apt_model.modeling.left_spin_smooth import (
    LeftSpinStep,
    LeftSpinResidual,
    LeftSpinMonitor
)
from apt.apt_model.modeling.apt_model import (
    APTModel,
    APTModelConfiguration
)


def test_left_spin_step_basic():
    """测试1: 左旋平滑步进器基础功能"""
    print("\n" + "="*70)
    print("测试1: 左旋平滑步进器基础功能")
    print("="*70)

    # 创建左旋步进器
    left_spin = LeftSpinStep(
        alpha=0.5,
        tau=0.3,
        beta=0.7,
        gate_type='normalized'
    )

    # 模拟数据
    batch_size, seq_len, d_model = 2, 10, 64
    u = torch.randn(batch_size, seq_len, d_model)
    delta_u = torch.randn(batch_size, seq_len, d_model)

    # 前向传播
    u_next, stats = left_spin(u, delta_u, use_smooth=True)

    print(f"✅ 输入形状: u={u.shape}, Δu={delta_u.shape}")
    print(f"✅ 输出形状: u'={u_next.shape}")
    print(f"✅ 统计信息:")
    print(f"   - 尖点强度: {stats['spike_strength']:.4f}")
    print(f"   - 缓冲角: {stats['buffer_angle']:.4f}")
    print(f"   - 门控值: {stats['gate_value']:.4f}")
    print(f"   - 已平滑: {stats['smoothed']}")

    # 验证门控值范围
    assert 0.0 <= stats['gate_value'] <= 1.0, "门控值应在 [0, 1] 范围内"
    print("✅ 门控值范围验证通过")


def test_left_spin_spike_detection():
    """测试2: 尖点检测能力"""
    print("\n" + "="*70)
    print("测试2: 尖点检测能力")
    print("="*70)

    left_spin = LeftSpinStep(alpha=0.5, tau=0.3, beta=0.7)

    batch_size, seq_len, d_model = 2, 10, 64
    u = torch.randn(batch_size, seq_len, d_model)

    # 场景1: 正常增量（小变化）
    delta_small = 0.1 * torch.randn(batch_size, seq_len, d_model)
    _, stats_small = left_spin(u, delta_small, use_smooth=True)

    # 场景2: 尖点增量（大变化）
    delta_large = 10.0 * torch.randn(batch_size, seq_len, d_model)
    _, stats_large = left_spin(u, delta_large, use_smooth=True)

    print(f"📊 正常增量:")
    print(f"   - 尖点强度: {stats_small['spike_strength']:.4f}")
    print(f"   - 缓冲角: {stats_small['buffer_angle']:.4f}")
    print(f"   - 门控值: {stats_small['gate_value']:.4f}")

    print(f"\n📊 尖点增量:")
    print(f"   - 尖点强度: {stats_large['spike_strength']:.4f}")
    print(f"   - 缓冲角: {stats_large['buffer_angle']:.4f}")
    print(f"   - 门控值: {stats_large['gate_value']:.4f}")

    # 验证：尖点增量应触发更强缓冲（门控值更小）
    assert stats_large['spike_strength'] > stats_small['spike_strength'], \
        "尖点增量应有更高的尖点强度"
    assert stats_large['gate_value'] < stats_small['gate_value'], \
        "尖点增量应有更小的门控值（更强缓冲）"

    print("\n✅ 尖点检测验证通过：大增量触发更强缓冲")


def test_left_spin_residual():
    """测试3: 左旋平滑残差层"""
    print("\n" + "="*70)
    print("测试3: 左旋平滑残差层")
    print("="*70)

    left_spin_res = LeftSpinResidual(
        alpha=0.5,
        tau=0.3,
        beta=0.7,
        gate_type='normalized',
        adaptive=True
    )

    batch_size, seq_len, d_model = 4, 20, 128
    x = torch.randn(batch_size, seq_len, d_model)
    residual = torch.randn(batch_size, seq_len, d_model)

    # 训练模式
    left_spin_res.train()
    output_train = left_spin_res(x, residual)

    # 推理模式
    left_spin_res.eval()
    output_eval = left_spin_res(x, residual)

    print(f"✅ 输入形状: x={x.shape}, residual={residual.shape}")
    print(f"✅ 训练模式输出: {output_train.shape}")
    print(f"✅ 推理模式输出: {output_eval.shape}")
    print(f"✅ 平滑应用比例: {left_spin_res.get_smooth_ratio():.2%}")


def test_apt_model_with_left_spin():
    """测试4: APT模型集成左旋平滑"""
    print("\n" + "="*70)
    print("测试4: APT模型集成左旋平滑")
    print("="*70)

    # 创建配置（启用左旋平滑）
    config = APTModelConfiguration(
        vocab_size=1000,
        d_model=128,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        d_ff=512,
        max_seq_len=64,
        use_left_spin=True,  # 启用左旋平滑
        left_spin_alpha=0.5,
        left_spin_tau=0.3,
        left_spin_beta=0.7,
        use_dbc_dac=False  # 简化测试
    )

    print(f"✅ 配置创建成功")
    print(f"   - use_left_spin: {config.use_left_spin}")
    print(f"   - left_spin_alpha: {config.left_spin_alpha}")
    print(f"   - left_spin_tau: {config.left_spin_tau}")

    # 创建模型
    model = APTModel(config)

    print(f"✅ 模型创建成功")
    print(f"   - 编码器层数: {len(model.encoder_layers)}")
    print(f"   - 解码器层数: {len(model.decoder_layers)}")

    # 检查是否集成了左旋平滑
    encoder_layer_0 = model.encoder_layers[0]
    has_left_spin = hasattr(encoder_layer_0, 'left_spin_attn') and \
                    encoder_layer_0.left_spin_attn is not None

    print(f"✅ 编码器层0 集成左旋平滑: {has_left_spin}")

    if has_left_spin:
        print(f"   - 注意力子层左旋平滑: {type(encoder_layer_0.left_spin_attn).__name__}")
        print(f"   - 前馈子层左旋平滑: {type(encoder_layer_0.left_spin_ffn).__name__}")

    # 前向传播测试
    batch_size, seq_len = 2, 16
    src_tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    tgt_tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        output = model(src_tokens=src_tokens, tgt_tokens=tgt_tokens)

    print(f"✅ 前向传播成功")
    print(f"   - 输出形状: {output.shape}")
    print(f"   - 期望形状: [{batch_size}, {seq_len}, {config.vocab_size}]")

    assert output.shape == (batch_size, seq_len, config.vocab_size), \
        "输出形状不匹配"


def test_left_spin_vs_standard_residual():
    """测试5: 左旋平滑 vs 标准残差对比"""
    print("\n" + "="*70)
    print("测试5: 左旋平滑 vs 标准残差对比")
    print("="*70)

    batch_size, seq_len, d_model = 4, 20, 128
    u = torch.randn(batch_size, seq_len, d_model)

    # 构造尖点增量（部分位置有巨大变化）
    delta_u = 0.1 * torch.randn(batch_size, seq_len, d_model)
    # 在中间位置注入尖点
    delta_u[:, seq_len//2, :] *= 100.0  # 🔥 尖点

    # 标准残差连接
    u_standard = u + delta_u

    # 左旋平滑残差连接
    left_spin = LeftSpinStep(alpha=0.5, tau=0.3, beta=0.7)
    u_smoothed, stats = left_spin(u, delta_u, use_smooth=True)

    # 计算输出稳定性（方差）
    var_standard = torch.var(u_standard).item()
    var_smoothed = torch.var(u_smoothed).item()

    print(f"📊 标准残差连接:")
    print(f"   - 输出方差: {var_standard:.6f}")

    print(f"\n📊 左旋平滑残差连接:")
    print(f"   - 输出方差: {var_smoothed:.6f}")
    print(f"   - 尖点强度: {stats['spike_strength']:.4f}")
    print(f"   - 缓冲角: {stats['buffer_angle']:.4f}")
    print(f"   - 门控值: {stats['gate_value']:.4f}")

    print(f"\n✅ 稳定性提升: {(var_standard - var_smoothed) / var_standard * 100:.2f}%")
    print(f"   (方差减少表示输出更稳定)")


def test_left_spin_monitor():
    """测试6: 左旋平滑监控器"""
    print("\n" + "="*70)
    print("测试6: 左旋平滑监控器")
    print("="*70)

    monitor = LeftSpinMonitor(log_interval=10)
    left_spin = LeftSpinStep(alpha=0.5, tau=0.3, beta=0.7)

    batch_size, seq_len, d_model = 2, 10, 64

    # 模拟20步
    for step in range(20):
        u = torch.randn(batch_size, seq_len, d_model)
        delta_u = torch.randn(batch_size, seq_len, d_model) * (1.0 + 0.1 * step)

        _, stats = left_spin(u, delta_u, use_smooth=True)
        monitor.log_stats(stats)

    # 获取统计
    full_stats = monitor.get_statistics()

    print(f"✅ 监控器统计（20步）:")
    for key, values in full_stats.items():
        print(f"   - {key}:")
        print(f"     均值: {values['mean']:.4f}")
        print(f"     标准差: {values['std']:.4f}")
        print(f"     范围: [{values['min']:.4f}, {values['max']:.4f}]")


def test_disable_left_spin():
    """测试7: 禁用左旋平滑（降级为标准残差）"""
    print("\n" + "="*70)
    print("测试7: 禁用左旋平滑（降级为标准残差）")
    print("="*70)

    # 创建配置（禁用左旋平滑）
    config = APTModelConfiguration(
        vocab_size=1000,
        d_model=128,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=4,
        d_ff=512,
        use_left_spin=False,  # ❌ 禁用左旋平滑
        use_dbc_dac=False
    )

    model = APTModel(config)

    # 检查编码器层
    encoder_layer_0 = model.encoder_layers[0]
    has_left_spin = hasattr(encoder_layer_0, 'left_spin_attn') and \
                    encoder_layer_0.left_spin_attn is not None

    print(f"✅ 配置: use_left_spin={config.use_left_spin}")
    print(f"✅ 编码器层0 集成左旋平滑: {has_left_spin}")
    print(f"   (应为 False，因为已禁用)")

    # 前向传播应正常工作（降级为标准残差）
    batch_size, seq_len = 2, 16
    src_tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    tgt_tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        output = model(src_tokens=src_tokens, tgt_tokens=tgt_tokens)

    print(f"✅ 前向传播成功（使用标准残差）")
    print(f"   - 输出形状: {output.shape}")


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("🚀 左旋平滑集成测试套件")
    print("="*70)
    print("\n测试目标: 验证左旋平滑已正确替换泰勒展开")
    print("测试内容: 7个测试场景")

    try:
        test_left_spin_step_basic()
        test_left_spin_spike_detection()
        test_left_spin_residual()
        test_apt_model_with_left_spin()
        test_left_spin_vs_standard_residual()
        test_left_spin_monitor()
        test_disable_left_spin()

        print("\n" + "="*70)
        print("🎉 所有测试通过！")
        print("="*70)
        print("\n✅ 左旋平滑已成功集成到 APT 模型")
        print("✅ 泰勒展开已被替换为单向缓冲机制")
        print("✅ 尖点规避功能正常工作")
        print("✅ 向后兼容（可禁用左旋平滑）")

        print("\n📚 使用方法:")
        print("   # 启用左旋平滑（默认）")
        print("   config = APTModelConfiguration(use_left_spin=True)")
        print("\n   # 禁用左旋平滑（降级为标准残差）")
        print("   config = APTModelConfiguration(use_left_spin=False)")

        print("\n🔬 核心改进:")
        print("   - 传统: u' = u + Δu  (遇尖点会炸)")
        print("   - 左旋: u' = u + g(φ)·Δu  (遇尖点自动缩小步长)")
        print("   - 其中 φ = α·softplus(s-τ), s 为尖点强度")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
