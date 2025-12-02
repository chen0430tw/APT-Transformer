#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GRPO训练示例

演示如何使用GRPO (Group Relative Policy Optimization) 训练模型

作者: chen0430tw
"""

import torch
import torch.nn as nn
import logging
from apt_model.rl import create_grpo_trainer, GRPOConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simple_model(vocab_size=1000, hidden_size=128):
    """创建简单的语言模型"""

    class SimpleLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            self.lm_head = nn.Linear(hidden_size, vocab_size)

        def forward(self, input_ids, attention_mask=None):
            hidden = self.embedding(input_ids)
            logits = self.lm_head(hidden)

            class Output:
                def __init__(self, logits):
                    self.logits = logits

            return Output(logits)

    return SimpleLanguageModel()


def create_reward_model():
    """创建简单的奖励模型"""

    class SimpleRewardModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input_ids, attention_mask=None):
            batch_size = input_ids.size(0)
            # 模拟奖励: 随机但有趋势
            base_rewards = torch.randn(batch_size, 1)
            trend = torch.arange(batch_size).float().unsqueeze(1) * 0.05
            rewards = base_rewards + trend
            return {'rewards': rewards}

    return SimpleRewardModel()


def generate_grouped_responses(group_size=4, num_groups=2, seq_len=20, vocab_size=1000):
    """
    生成分组的响应

    Args:
        group_size: 每组的样本数
        num_groups: 组数
        seq_len: 序列长度
        vocab_size: 词表大小

    Returns:
        (responses, response_masks)
    """
    batch_size = group_size * num_groups
    responses = torch.randint(5, vocab_size, (batch_size, seq_len))
    response_masks = torch.ones_like(responses)

    return responses, response_masks


def main():
    print("=" * 70)
    print("GRPO训练示例")
    print("=" * 70)

    # 1. 创建模型
    print("\n[步骤1] 创建策略模型和奖励模型...")
    policy_model = create_simple_model()
    reward_model = create_reward_model()

    print("  策略模型参数量:", sum(p.numel() for p in policy_model.parameters()))

    # 2. 创建GRPO训练器 - 相对优势模式
    print("\n[步骤2] 创建GRPO训练器 (相对优势模式)...")
    config = GRPOConfig(
        group_size=4,
        advantage_type="relative",
        learning_rate=1e-5
    )

    trainer = create_grpo_trainer(
        policy_model=policy_model,
        reward_model=reward_model,
        config=config,
        device='cpu'
    )

    print(f"  组大小: {config.group_size}")
    print(f"  优势类型: {config.advantage_type}")
    print(f"  学习率: {config.learning_rate}")

    # 3. 训练
    print("\n[步骤3] 开始GRPO训练...")
    num_steps = 20

    for step in range(num_steps):
        # 生成分组响应
        # 每次生成2组，每组4个响应
        responses, response_masks = generate_grouped_responses(
            group_size=config.group_size,
            num_groups=2
        )

        # 训练步骤
        stats = trainer.train_step(
            responses=responses,
            response_masks=response_masks
        )

        # 打印进度
        if step % 5 == 0:
            print(f"\n  Step {step}/{num_steps}:")
            print(f"    Mean Reward: {stats['mean_reward']:.4f}")
            print(f"    Group Variance: {stats['group_variance']:.4f}")
            print(f"    Policy Loss: {stats['policy_loss']:.4f}")
            print(f"    Mean Advantage: {stats['mean_advantage']:.4f}")

    # 4. 测试不同的优势类型
    print("\n" + "=" * 70)
    print("测试不同的优势类型")
    print("=" * 70)

    advantage_types = ["relative", "normalized", "rank"]

    for adv_type in advantage_types:
        print(f"\n[优势类型: {adv_type}]")

        config = GRPOConfig(
            group_size=4,
            advantage_type=adv_type
        )

        model = create_simple_model()
        trainer = create_grpo_trainer(
            policy_model=model,
            reward_model=reward_model,
            config=config,
            device='cpu'
        )

        # 训练几步
        for step in range(5):
            responses, response_masks = generate_grouped_responses(
                group_size=4, num_groups=2
            )
            stats = trainer.train_step(responses, response_masks)

            if step % 2 == 0:
                print(f"  Step {step}: Loss={stats['policy_loss']:.4f}, "
                      f"Reward={stats['mean_reward']:.4f}")

    # 5. 无奖励模型模式
    print("\n" + "=" * 70)
    print("无奖励模型模式 (使用外部奖励)")
    print("=" * 70)

    policy_model3 = create_simple_model()
    trainer_no_rm = create_grpo_trainer(
        policy_model=policy_model3,
        reward_model=None,  # 不使用奖励模型
        config={'group_size': 4},
        device='cpu'
    )

    print("\n开始训练...")
    for step in range(10):
        responses, response_masks = generate_grouped_responses(
            group_size=4, num_groups=2
        )

        # 提供外部计算的奖励
        external_rewards = torch.randn(8)

        stats = trainer_no_rm.train_step(
            responses=responses,
            response_masks=response_masks,
            rewards=external_rewards
        )

        if step % 5 == 0:
            print(f"  Step {step}: Loss={stats['policy_loss']:.4f}")

    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    print("\nGRPO关键点:")
    print("  1. 基于分组相对优势更新策略")
    print("  2. 支持多种优势计算方式")
    print("  3. 可以使用或不使用奖励模型")
    print("  4. 适合在线学习场景")
    print("  5. 计算效率高于PPO")


if __name__ == "__main__":
    main()
