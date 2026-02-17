#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DPO训练示例

演示如何使用DPO (Direct Preference Optimization) 训练模型

作者: chen0430tw
"""

import torch
import torch.nn as nn
import logging
from apt.apps.rl import create_dpo_trainer, DPOConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simple_model(vocab_size=1000, hidden_size=128):
    """创建简单的语言模型用于演示"""

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


def create_preference_data(batch_size=4, seq_len=20, vocab_size=1000):
    """
    创建模拟的偏好数据

    Returns:
        (chosen_ids, rejected_ids, chosen_mask, rejected_mask)
    """
    # 选中的响应
    chosen_ids = torch.randint(5, vocab_size, (batch_size, seq_len))
    chosen_mask = torch.ones_like(chosen_ids)

    # 拒绝的响应
    rejected_ids = torch.randint(5, vocab_size, (batch_size, seq_len))
    rejected_mask = torch.ones_like(rejected_ids)

    return chosen_ids, rejected_ids, chosen_mask, rejected_mask


def main():
    print("=" * 70)
    print("DPO训练示例")
    print("=" * 70)

    # 1. 创建模型
    print("\n[步骤1] 创建策略模型和参考模型...")
    policy_model = create_simple_model()
    ref_model = create_simple_model()

    # 加载相同的初始权重
    ref_model.load_state_dict(policy_model.state_dict())
    ref_model.eval()  # 参考模型固定

    print("  策略模型参数量:", sum(p.numel() for p in policy_model.parameters()))
    print("  参考模型参数量:", sum(p.numel() for p in ref_model.parameters()))

    # 2. 创建DPO训练器
    print("\n[步骤2] 创建DPO训练器...")
    config = DPOConfig(
        beta=0.1,  # 温度参数
        label_smoothing=0.0,
        learning_rate=1e-5,
        reference_free=False
    )

    trainer = create_dpo_trainer(
        policy_model=policy_model,
        ref_policy_model=ref_model,
        config=config,
        device='cpu'
    )

    print(f"  Beta (温度): {config.beta}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  无参考模式: {config.reference_free}")

    # 3. 训练
    print("\n[步骤3] 开始DPO训练...")
    num_steps = 20

    for step in range(num_steps):
        # 获取偏好数据
        chosen_ids, rejected_ids, chosen_mask, rejected_mask = create_preference_data()

        # 训练步骤
        stats = trainer.train_step(
            chosen_ids=chosen_ids,
            rejected_ids=rejected_ids,
            chosen_mask=chosen_mask,
            rejected_mask=rejected_mask
        )

        # 打印进度
        if step % 5 == 0:
            print(f"\n  Step {step}/{num_steps}:")
            print(f"    Loss: {stats['loss']:.4f}")
            print(f"    Accuracy: {stats['accuracy']:.2%}")
            print(f"    Chosen Reward: {stats['chosen_reward']:.4f}")
            print(f"    Rejected Reward: {stats['rejected_reward']:.4f}")
            print(f"    Reward Margin: {stats['reward_margin']:.4f}")

    # 4. 最终统计
    print("\n[步骤4] 训练完成！")
    final_stats = trainer.get_statistics()
    print(f"  总训练步数: {final_stats['total_steps']}")
    print(f"  最终准确率: {final_stats['accuracy']:.2%}")

    # 5. 对比：无参考模式
    print("\n" + "=" * 70)
    print("无参考模式DPO训练")
    print("=" * 70)

    policy_model2 = create_simple_model()
    config_rf = DPOConfig(
        beta=0.1,
        reference_free=True  # 无参考模式
    )

    trainer_rf = create_dpo_trainer(
        policy_model=policy_model2,
        ref_policy_model=None,
        config=config_rf,
        device='cpu'
    )

    print("\n开始训练...")
    for step in range(10):
        chosen_ids, rejected_ids, chosen_mask, rejected_mask = create_preference_data()
        stats = trainer_rf.train_step(chosen_ids, rejected_ids, chosen_mask, rejected_mask)

        if step % 5 == 0:
            print(f"  Step {step}: Loss={stats['loss']:.4f}, Acc={stats['accuracy']:.2%}")

    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    print("\nDPO关键点:")
    print("  1. 无需单独训练奖励模型")
    print("  2. 训练更简单、更稳定")
    print("  3. 可以使用无参考模式进一步简化")
    print("  4. Beta参数控制优化强度")


if __name__ == "__main__":
    main()
