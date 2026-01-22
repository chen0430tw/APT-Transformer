#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比学习预训练示例

演示如何使用SimCLR/MoCo风格的对比学习进行预训练

作者: chen0430tw
"""

import torch
import torch.nn as nn
import logging
from apt_model.pretraining import create_contrastive_pretrainer, ContrastiveConfig
from apt.core.pretraining.contrastive_pretrain import TextAugmentation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simple_encoder(vocab_size=1000, hidden_size=128):
    """创建简单的编码器"""

    class SimpleEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)

        def forward(self, input_ids, attention_mask=None):
            hidden = self.embedding(input_ids)

            class Output:
                def __init__(self, hidden):
                    self.last_hidden_state = hidden

            return Output(hidden)

    return SimpleEncoder()


def create_augmented_pairs(batch_size=8, seq_len=20, vocab_size=1000):
    """
    创建增强的样本对

    在实际应用中，x1和x2应该是同一样本的两个不同增强
    """
    # 原始数据
    original = torch.randint(5, vocab_size, (batch_size, seq_len))

    # 增强1: 随机mask一些token
    x1 = TextAugmentation.random_mask(
        original.clone(),
        mask_token_id=4,
        mask_prob=0.15
    )

    # 增强2: 随机交换一些token
    x2 = TextAugmentation.random_swap(
        original.clone(),
        swap_prob=0.1
    )

    return x1, x2


def main():
    print("=" * 70)
    print("对比学习预训练示例")
    print("=" * 70)

    # 1. SimCLR风格对比学习
    print("\n[模式1] SimCLR风格对比学习")
    print("=" * 70)

    # 创建编码器
    print("\n[步骤1] 创建编码器...")
    encoder = create_simple_encoder()
    print("  编码器参数量:", sum(p.numel() for p in encoder.parameters()))

    # 创建训练器
    print("\n[步骤2] 创建SimCLR训练器...")
    config = ContrastiveConfig(
        temperature=0.07,
        projection_dim=64,
        hidden_dim=128,
        use_momentum_encoder=False,  # SimCLR
        symmetric_loss=True
    )

    trainer = create_contrastive_pretrainer(
        encoder=encoder,
        hidden_size=128,
        config=config,
        device='cpu'
    )

    print(f"  温度参数: {config.temperature}")
    print(f"  投影维度: {config.projection_dim}")
    print(f"  使用对称损失: {config.symmetric_loss}")

    # 训练
    print("\n[步骤3] 开始SimCLR训练...")
    num_steps = 20

    for step in range(num_steps):
        # 创建增强对
        x1, x2 = create_augmented_pairs(batch_size=8)

        # 训练步骤
        stats = trainer.train_step(x1, x2)

        # 打印进度
        if step % 5 == 0:
            print(f"  Step {step}/{num_steps}:")
            print(f"    Loss: {stats['loss']:.4f}")
            print(f"    Accuracy: {stats['accuracy']:.2%}")

    print("\n  SimCLR训练完成！")

    # 2. MoCo风格对比学习
    print("\n\n[模式2] MoCo风格对比学习")
    print("=" * 70)

    # 创建编码器
    print("\n[步骤1] 创建编码器...")
    encoder_moco = create_simple_encoder()

    # 创建训练器
    print("\n[步骤2] 创建MoCo训练器...")
    config_moco = ContrastiveConfig(
        temperature=0.07,
        projection_dim=64,
        use_momentum_encoder=True,  # MoCo
        momentum=0.999,
        queue_size=256
    )

    trainer_moco = create_contrastive_pretrainer(
        encoder=encoder_moco,
        hidden_size=128,
        config=config_moco,
        device='cpu'
    )

    print(f"  使用动量编码器: {config_moco.use_momentum_encoder}")
    print(f"  动量系数: {config_moco.momentum}")
    print(f"  队列大小: {config_moco.queue_size}")

    # 训练
    print("\n[步骤3] 开始MoCo训练...")

    for step in range(num_steps):
        x1, x2 = create_augmented_pairs(batch_size=8)
        stats = trainer_moco.train_step(x1, x2)

        if step % 5 == 0:
            print(f"  Step {step}/{num_steps}:")
            print(f"    Loss: {stats['loss']:.4f}")
            print(f"    Accuracy: {stats['accuracy']:.2%}")

    print("\n  MoCo训练完成！")

    # 3. 数据增强演示
    print("\n\n[演示] 数据增强方法")
    print("=" * 70)

    original_ids = torch.randint(5, 100, (4, 20))
    print("\n原始序列:")
    print(original_ids[:2])

    # Random Mask
    print("\n1. Random Mask (mask_prob=0.15):")
    masked = TextAugmentation.random_mask(
        original_ids.clone(),
        mask_token_id=4,
        mask_prob=0.15
    )
    print(masked[:2])

    # Random Swap
    print("\n2. Random Swap (swap_prob=0.1):")
    swapped = TextAugmentation.random_swap(
        original_ids.clone(),
        swap_prob=0.1
    )
    print(swapped[:2])

    # Random Delete
    print("\n3. Random Delete (delete_prob=0.1):")
    deleted, mask = TextAugmentation.random_delete(
        original_ids.clone(),
        delete_prob=0.1
    )
    print(deleted[:2])
    print("Mask:", mask[:2])

    # 4. 对比不同温度参数
    print("\n\n[实验] 不同温度参数的影响")
    print("=" * 70)

    temperatures = [0.05, 0.07, 0.1, 0.2]

    for temp in temperatures:
        print(f"\n温度 τ = {temp}:")
        encoder_temp = create_simple_encoder()
        trainer_temp = create_contrastive_pretrainer(
            encoder=encoder_temp,
            hidden_size=128,
            config={'temperature': temp, 'projection_dim': 64},
            device='cpu'
        )

        # 训练5步
        losses = []
        for step in range(5):
            x1, x2 = create_augmented_pairs(batch_size=8)
            stats = trainer_temp.train_step(x1, x2)
            losses.append(stats['loss'])

        avg_loss = sum(losses) / len(losses)
        print(f"  平均损失: {avg_loss:.4f}")

    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    print("\n对比学习关键点:")
    print("  1. SimCLR: 使用batch内样本作为负样本，需要大batch size")
    print("  2. MoCo: 使用动量编码器+队列，可用小batch size")
    print("  3. 数据增强: 创建正样本对的关键")
    print("  4. 温度参数: 控制损失函数的锐度")
    print("  5. 无需标注数据，学习通用表示")


if __name__ == "__main__":
    main()
