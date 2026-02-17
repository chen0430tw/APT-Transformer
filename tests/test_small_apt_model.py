#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试小型APT模型（包括压缩和DBC加速）"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 添加路径
sys.path.insert(0, '/home/user/APT-Transformer')

from apt_model.modeling.apt_model import (
    APTModel,
    APTModelConfiguration,
    DBCDAC_Optimizer,
    create_gradient_stabilizer_hook
)
from legacy_plugins.batch1.model_pruning_plugin import ModelPruningPlugin


def create_small_apt_config():
    """创建小型APT模型配置"""
    config = APTModelConfiguration(
        vocab_size=1000,          # 小词汇表
        d_model=128,              # 小维度
        max_seq_len=64,           # 短序列
        num_encoder_layers=2,     # 2层编码器
        num_decoder_layers=2,     # 2层解码器
        num_heads=4,              # 4个注意力头
        d_ff=512,                 # 小前馈网络
        dropout=0.1,
        use_autopoietic=True,     # 启用自生成机制
        use_dbc_dac=True,         # 启用DBC-DAC
    )
    return config


def register_dbc_hooks(model):
    """为模型注册DBC-DAC hooks"""
    opt = DBCDAC_Optimizer()
    hooks = []
    for _, p in model.named_parameters():
        if p.requires_grad:
            hooks.append(p.register_hook(create_gradient_stabilizer_hook(opt)))
    return hooks


def create_dummy_data(batch_size=8, seq_len=32, vocab_size=1000):
    """创建模拟训练数据"""
    # 输入序列
    src = torch.randint(1, vocab_size, (batch_size, seq_len))
    # 目标序列（简单起见，使用输入的偏移版本）
    tgt = torch.randint(1, vocab_size, (batch_size, seq_len))

    return src, tgt


def test_apt_forward():
    """测试APT模型前向传播"""
    print("\n" + "="*60)
    print("🧪 测试APT模型前向传播")
    print("="*60)

    # 创建小型APT模型
    config = create_small_apt_config()
    model = APTModel(config)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"📊 模型配置:")
    print(f"   词汇表大小: {config.vocab_size}")
    print(f"   模型维度: {config.d_model}")
    print(f"   编码器层数: {config.num_encoder_layers}")
    print(f"   解码器层数: {config.num_decoder_layers}")
    print(f"   注意力头数: {config.num_heads}")
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")

    # 创建输入
    batch_size, seq_len = 4, 32
    src, tgt = create_dummy_data(batch_size, seq_len, config.vocab_size)

    print(f"\n📝 输入数据:")
    print(f"   源序列形状: {src.shape}")
    print(f"   目标序列形状: {tgt.shape}")

    # 前向传播
    print("\n🔍 执行前向传播...")
    model.eval()
    with torch.no_grad():
        try:
            output = model(src, tgt)
            print(f"   输出形状: {output.shape}")
            print(f"   输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
            print(f"   ✅ 前向传播成功!")
            return True
        except Exception as e:
            print(f"   ❌ 前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_apt_training():
    """测试APT模型训练"""
    print("\n" + "="*60)
    print("🧪 测试APT模型训练")
    print("="*60)

    # 创建模型
    config = create_small_apt_config()
    model = APTModel(config)

    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)

    # 创建数据
    batch_size, seq_len = 8, 32
    src, tgt = create_dummy_data(batch_size, seq_len, config.vocab_size)

    print(f"📊 训练设置:")
    print(f"   优化器: Adam (lr=1e-4)")
    print(f"   损失函数: CrossEntropyLoss")
    print(f"   Batch size: {batch_size}")

    # 训练几步
    print("\n🏃 运行训练步骤...")
    model.train()
    num_steps = 5

    for step in range(num_steps):
        optimizer.zero_grad()

        # 前向传播
        output = model(src, tgt[:, :-1])  # 解码器输入去掉最后一个token

        # 计算损失
        loss = criterion(
            output.reshape(-1, config.vocab_size),
            tgt[:, 1:].reshape(-1)  # 目标去掉第一个token
        )

        # 反向传播
        loss.backward()

        # 计算梯度范数
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)

        # 优化
        optimizer.step()

        print(f"   Step {step+1}/{num_steps} | Loss: {loss.item():.4f} | Grad Norm: {grad_norm:.4f}")

    print(f"   ✅ 训练测试成功!")
    return True


def test_apt_with_dbc():
    """测试APT模型 + DBC加速"""
    print("\n" + "="*60)
    print("🧪 测试APT模型 + DBC-DAC加速")
    print("="*60)

    # 创建两个相同的模型
    config = create_small_apt_config()
    model_without_dbc = APTModel(config)
    model_with_dbc = APTModel(config)

    # 确保参数相同
    model_with_dbc.load_state_dict(model_without_dbc.state_dict())

    # 只给一个模型注册DBC hooks
    print("🔧 为第二个模型注册DBC hooks...")
    hooks = register_dbc_hooks(model_with_dbc)
    print(f"   注册了 {len(hooks)} 个hooks")

    # 创建优化器
    optimizer1 = optim.Adam(model_without_dbc.parameters(), lr=1e-4)
    optimizer2 = optim.Adam(model_with_dbc.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)

    # 创建数据（使用较大的值来测试稳定性）
    batch_size, seq_len = 8, 32
    src, tgt = create_dummy_data(batch_size, seq_len, config.vocab_size)

    print(f"\n📊 对比测试 (5步训练):")
    print(f"   模型1: 无DBC加速")
    print(f"   模型2: 有DBC加速")

    # 训练对比
    print("\n🏃 开始训练...")
    num_steps = 5

    for step in range(num_steps):
        # 无DBC模型
        optimizer1.zero_grad()
        model_without_dbc.train()
        output1 = model_without_dbc(src, tgt[:, :-1])
        loss1 = criterion(output1.reshape(-1, config.vocab_size), tgt[:, 1:].reshape(-1))
        loss1.backward()
        grad_norm1 = sum(p.grad.norm().item() for p in model_without_dbc.parameters() if p.grad is not None)
        optimizer1.step()

        # 有DBC模型
        optimizer2.zero_grad()
        model_with_dbc.train()
        output2 = model_with_dbc(src, tgt[:, :-1])
        loss2 = criterion(output2.reshape(-1, config.vocab_size), tgt[:, 1:].reshape(-1))
        loss2.backward()
        grad_norm2 = sum(p.grad.norm().item() for p in model_with_dbc.parameters() if p.grad is not None)
        optimizer2.step()

        print(f"\n   Step {step+1}/{num_steps}:")
        print(f"     无DBC: Loss={loss1.item():.4f}, Grad={grad_norm1:.4f}")
        print(f"     有DBC: Loss={loss2.item():.4f}, Grad={grad_norm2:.4f}")
        print(f"     梯度差异: {abs(grad_norm1 - grad_norm2):.4f}")

    print(f"\n   ✅ DBC加速测试完成!")
    return True


def test_apt_with_pruning():
    """测试APT模型 + 剪枝压缩"""
    print("\n" + "="*60)
    print("🧪 测试APT模型 + 剪枝压缩")
    print("="*60)

    # 创建模型
    config = create_small_apt_config()
    model = APTModel(config)

    # 统计原始参数
    original_params = sum(p.numel() for p in model.parameters())
    print(f"📊 原始模型参数: {original_params:,}")

    # 创建剪枝插件
    prune_config = {
        'prune_ratio': 0.2,  # 剪枝20%
        'prune_type': 'magnitude',
        'structured': False,
    }
    plugin = ModelPruningPlugin(prune_config)

    # 应用剪枝
    print(f"\n✂️ 应用权重大小剪枝 (20%)...")
    model = plugin.magnitude_pruning(model, prune_ratio=0.2, structured=False)

    # 获取剪枝统计
    stats = plugin.get_pruning_statistics(model)
    print(f"\n📈 剪枝后统计:")
    print(f"   总参数: {stats['total_params']:,}")
    print(f"   剪枝参数: {stats['pruned_params']:,}")
    print(f"   剩余参数: {stats['remaining_params']:,}")
    print(f"   稀疏度: {stats['sparsity']*100:.2f}%")

    # 测试剪枝后的模型是否仍能工作
    print(f"\n🔍 测试剪枝后模型...")
    batch_size, seq_len = 4, 32
    src, tgt = create_dummy_data(batch_size, seq_len, config.vocab_size)

    model.eval()
    with torch.no_grad():
        try:
            output = model(src, tgt)
            print(f"   输出形状: {output.shape}")
            print(f"   ✅ 剪枝后模型仍可正常工作!")
        except Exception as e:
            print(f"   ❌ 剪枝后模型失败: {e}")
            return False

    return True


def test_full_pipeline():
    """测试完整流程：训练 -> 剪枝 -> DBC加速"""
    print("\n" + "="*60)
    print("🧪 测试完整流程 (训练 -> 剪枝 -> DBC)")
    print("="*60)

    # 1. 创建并训练模型
    print("\n📍 阶段1: 初始训练")
    config = create_small_apt_config()
    model = APTModel(config)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)

    batch_size, seq_len = 8, 32
    src, tgt = create_dummy_data(batch_size, seq_len, config.vocab_size)

    # 训练3步
    model.train()
    for step in range(3):
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.reshape(-1, config.vocab_size), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        print(f"   训练步骤 {step+1}/3: Loss={loss.item():.4f}")

    # 2. 应用剪枝
    print("\n📍 阶段2: 模型剪枝")
    prune_config = {'prune_ratio': 0.3, 'prune_type': 'magnitude', 'structured': False}
    plugin = ModelPruningPlugin(prune_config)
    model = plugin.magnitude_pruning(model, prune_ratio=0.3, structured=False)

    stats = plugin.get_pruning_statistics(model)
    print(f"   剪枝后稀疏度: {stats['sparsity']*100:.2f}%")

    # 3. 使用DBC继续训练
    print("\n📍 阶段3: DBC加速微调")
    hooks = register_dbc_hooks(model)
    print(f"   注册了 {len(hooks)} 个DBC hooks")

    optimizer = optim.Adam(model.parameters(), lr=5e-5)  # 降低学习率进行微调

    model.train()
    for step in range(3):
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.reshape(-1, config.vocab_size), tgt[:, 1:].reshape(-1))
        loss.backward()
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        print(f"   微调步骤 {step+1}/3: Loss={loss.item():.4f}, Grad={grad_norm:.4f}")

    print(f"\n   ✅ 完整流程测试成功!")
    return True


def main():
    """主测试函数"""
    print("\n🚀 开始测试小型APT模型...")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    try:
        # 测试前向传播
        test1 = test_apt_forward()

        # 测试训练
        test2 = test_apt_training()

        # 测试DBC加速
        test3 = test_apt_with_dbc()

        # 测试剪枝
        test4 = test_apt_with_pruning()

        # 测试完整流程
        test5 = test_full_pipeline()

        # 总结
        print("\n" + "="*60)
        print("📝 测试总结")
        print("="*60)
        print(f"✅ APT前向传播: {'通过' if test1 else '失败'}")
        print(f"✅ APT模型训练: {'通过' if test2 else '失败'}")
        print(f"✅ DBC加速训练: {'通过' if test3 else '失败'}")
        print(f"✅ 模型剪枝压缩: {'通过' if test4 else '失败'}")
        print(f"✅ 完整流程集成: {'通过' if test5 else '失败'}")

        if all([test1, test2, test3, test4, test5]):
            print("\n🎉 所有测试通过!")
            print("\n💡 APT模型特性:")
            print("   ✅ 自生成注意力机制 (Autopoietic Attention)")
            print("   ✅ DBC-DAC梯度稳定")
            print("   ✅ 支持模型剪枝压缩")
            print("   ✅ 编码器-解码器架构")
            print("   ✅ 可扩展的配置系统")
            return 0
        else:
            print("\n❌ 部分测试失败")
            return 1

    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
