#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试模型压缩插件"""

import sys
import torch
import torch.nn as nn

# 添加路径
sys.path.insert(0, '/home/user/APT-Transformer')

from legacy_plugins.batch1.model_pruning_plugin import ModelPruningPlugin
from legacy_plugins.batch1.model_distillation_plugin import ModelDistillationPlugin


class TestModel(nn.Module):
    """简单的测试模型"""
    def __init__(self, input_size=100, hidden_size=50, output_size=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def test_pruning_plugin():
    """测试模型剪枝插件"""
    print("\n" + "="*60)
    print("🧪 测试模型剪枝插件")
    print("="*60)

    # 配置
    config = {
        'prune_ratio': 0.3,
        'prune_type': 'magnitude',
        'structured': False,
    }

    # 创建插件和模型
    plugin = ModelPruningPlugin(config)
    model = TestModel()

    # 统计原始参数
    original_params = sum(p.numel() for p in model.parameters())
    print(f"📊 原始模型参数量: {original_params:,}")

    # 应用权重大小剪枝
    print("\n✂️ 应用权重大小剪枝...")
    model = plugin.magnitude_pruning(model, prune_ratio=0.3, structured=False)

    # 获取剪枝统计
    stats = plugin.get_pruning_statistics(model)
    print(f"\n📈 剪枝统计:")
    print(f"   总参数: {stats['total_params']:,}")
    print(f"   剪枝参数: {stats['pruned_params']:,}")
    print(f"   剩余参数: {stats['remaining_params']:,}")
    print(f"   稀疏度: {stats['sparsity']*100:.2f}%")

    # 测试模型前向传播
    print("\n🔍 测试前向传播...")
    test_input = torch.randn(8, 100)
    with torch.no_grad():
        output = model(test_input)
    print(f"   输入形状: {test_input.shape}")
    print(f"   输出形状: {output.shape}")
    print(f"   ✅ 前向传播成功!")

    return True


def test_distillation_plugin():
    """测试知识蒸馏插件"""
    print("\n" + "="*60)
    print("🧪 测试知识蒸馏插件")
    print("="*60)

    # 配置
    config = {
        'temperature': 4.0,
        'alpha': 0.7,
        'beta': 0.3,
        'distill_type': 'response',
    }

    # 创建插件
    plugin = ModelDistillationPlugin(config)

    # 创建模拟数据
    batch_size, seq_len, vocab_size = 8, 128, 1000
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    print(f"📊 测试数据:")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_len}")
    print(f"   Vocab size: {vocab_size}")

    # 测试响应蒸馏
    print("\n🎓 测试响应蒸馏...")
    response_loss = plugin.response_distillation_loss(
        student_logits, teacher_logits, labels
    )
    print(f"   响应蒸馏损失: {response_loss.item():.4f}")
    print(f"   ✅ 响应蒸馏测试成功!")

    # 测试特征蒸馏
    print("\n🎓 测试特征蒸馏...")
    student_features = torch.randn(batch_size, seq_len, 512)
    teacher_features = torch.randn(batch_size, seq_len, 512)
    feature_loss = plugin.feature_distillation_loss(
        student_features, teacher_features
    )
    print(f"   特征蒸馏损失: {feature_loss.item():.4f}")
    print(f"   ✅ 特征蒸馏测试成功!")

    # 测试关系蒸馏
    print("\n🎓 测试关系蒸馏...")
    student_outputs = torch.randn(batch_size, 512)
    teacher_outputs = torch.randn(batch_size, 512)
    relation_loss = plugin.relation_distillation_loss(
        student_outputs, teacher_outputs
    )
    print(f"   关系蒸馏损失: {relation_loss.item():.4f}")
    print(f"   ✅ 关系蒸馏测试成功!")

    return True


def main():
    """主测试函数"""
    print("\n🚀 开始测试模型压缩插件...")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    try:
        # 测试剪枝
        pruning_ok = test_pruning_plugin()

        # 测试蒸馏
        distillation_ok = test_distillation_plugin()

        # 总结
        print("\n" + "="*60)
        print("📝 测试总结")
        print("="*60)
        print(f"✅ 模型剪枝插件: {'通过' if pruning_ok else '失败'}")
        print(f"✅ 知识蒸馏插件: {'通过' if distillation_ok else '失败'}")

        if pruning_ok and distillation_ok:
            print("\n🎉 所有测试通过!")
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
