#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MLM预训练示例

演示如何使用BERT风格的遮蔽语言模型进行预训练

作者: chen0430tw
"""

import torch
import torch.nn as nn
import logging
from apt_model.pretraining import create_mlm_pretrainer, MLMConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simple_language_model(vocab_size=1000, hidden_size=128):
    """创建简单的语言模型"""

    class SimpleLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)

        def forward(self, input_ids, attention_mask=None, output_hidden_states=True):
            hidden = self.embedding(input_ids)

            class Output:
                def __init__(self, hidden):
                    self.last_hidden_state = hidden
                    self.hidden_states = (hidden,)

            return Output(hidden)

    return SimpleLanguageModel()


def create_training_data(batch_size=4, seq_len=20, vocab_size=1000):
    """
    创建训练数据

    Returns:
        (input_ids, attention_mask)
    """
    # 避开特殊token (0-4)
    input_ids = torch.randint(5, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)

    return input_ids, attention_mask


def main():
    print("=" * 70)
    print("MLM预训练示例")
    print("=" * 70)

    # 1. 仅MLM任务 (RoBERTa风格)
    print("\n[模式1] 仅MLM任务 (RoBERTa风格)")
    print("=" * 70)

    # 创建模型
    print("\n[步骤1] 创建语言模型...")
    model = create_simple_language_model()
    print("  模型参数量:", sum(p.numel() for p in model.parameters()))

    # 创建训练器
    print("\n[步骤2] 创建MLM训练器...")
    config = MLMConfig(
        vocab_size=1000,
        mask_prob=0.15,
        mask_token_prob=0.8,
        random_token_prob=0.1,
        keep_token_prob=0.1,
        use_nsp=False  # RoBERTa不使用NSP
    )

    trainer = create_mlm_pretrainer(
        model=model,
        hidden_size=128,
        config=config,
        device='cpu'
    )

    print(f"  遮蔽概率: {config.mask_prob}")
    print(f"  [MASK]概率: {config.mask_token_prob}")
    print(f"  随机token概率: {config.random_token_prob}")
    print(f"  保持不变概率: {config.keep_token_prob}")

    # 训练
    print("\n[步骤3] 开始MLM训练...")
    num_steps = 20

    for step in range(num_steps):
        # 获取训练数据
        input_ids, attention_mask = create_training_data()

        # 训练步骤
        stats = trainer.train_step(input_ids, attention_mask)

        # 打印进度
        if step % 5 == 0:
            print(f"\n  Step {step}/{num_steps}:")
            print(f"    Total Loss: {stats['total_loss']:.4f}")
            print(f"    MLM Loss: {stats['mlm_loss']:.4f}")
            print(f"    MLM Accuracy: {stats['mlm_accuracy']:.2%}")
            print(f"    Masked Tokens: {stats['num_masked']}")

    print("\n  RoBERTa风格MLM训练完成！")

    # 2. MLM + NSP任务 (BERT风格)
    print("\n\n[模式2] MLM + NSP任务 (BERT风格)")
    print("=" * 70)

    # 创建模型
    print("\n[步骤1] 创建语言模型...")
    model_bert = create_simple_language_model()

    # 创建训练器
    print("\n[步骤2] 创建BERT训练器...")
    config_bert = MLMConfig(
        vocab_size=1000,
        mask_prob=0.15,
        use_nsp=True  # BERT使用NSP
    )

    trainer_bert = create_mlm_pretrainer(
        model=model_bert,
        hidden_size=128,
        config=config_bert,
        device='cpu'
    )

    print(f"  使用NSP任务: {config_bert.use_nsp}")

    # 训练
    print("\n[步骤3] 开始BERT训练...")

    for step in range(num_steps):
        input_ids, attention_mask = create_training_data()

        # NSP标签 (0=连续, 1=不连续)
        nsp_labels = torch.randint(0, 2, (4,))

        # 训练步骤
        stats = trainer_bert.train_step(
            input_ids, attention_mask, nsp_labels
        )

        # 打印进度
        if step % 5 == 0:
            print(f"\n  Step {step}/{num_steps}:")
            print(f"    Total Loss: {stats['total_loss']:.4f}")
            print(f"    MLM Loss: {stats['mlm_loss']:.4f}")
            print(f"    MLM Accuracy: {stats['mlm_accuracy']:.2%}")
            print(f"    NSP Loss: {stats['nsp_loss']:.4f}")
            print(f"    NSP Accuracy: {stats['nsp_accuracy']:.2%}")

    print("\n  BERT风格训练完成！")

    # 3. 实验：不同遮蔽概率
    print("\n\n[实验] 不同遮蔽概率的影响")
    print("=" * 70)

    mask_probs = [0.10, 0.15, 0.20, 0.25]

    for mask_prob in mask_probs:
        print(f"\n遮蔽概率 = {mask_prob}:")

        model_exp = create_simple_language_model()
        trainer_exp = create_mlm_pretrainer(
            model=model_exp,
            hidden_size=128,
            config={'vocab_size': 1000, 'mask_prob': mask_prob},
            device='cpu'
        )

        # 训练5步
        losses = []
        accuracies = []
        for step in range(5):
            input_ids, attention_mask = create_training_data()
            stats = trainer_exp.train_step(input_ids, attention_mask)
            losses.append(stats['mlm_loss'])
            accuracies.append(stats['mlm_accuracy'])

        avg_loss = sum(losses) / len(losses)
        avg_acc = sum(accuracies) / len(accuracies)
        print(f"  平均损失: {avg_loss:.4f}")
        print(f"  平均准确率: {avg_acc:.2%}")

    # 4. 遮蔽策略演示
    print("\n\n[演示] BERT遮蔽策略")
    print("=" * 70)

    model_demo = create_simple_language_model()
    trainer_demo = create_mlm_pretrainer(
        model=model_demo,
        hidden_size=128,
        config={'vocab_size': 1000, 'mask_prob': 0.15},
        device='cpu'
    )

    # 创建示例数据
    original_ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]])
    print("\n原始序列:")
    print(original_ids)

    # 创建遮蔽
    masked_input, labels, mask_positions = trainer_demo.create_mlm_mask(original_ids)

    print("\n遮蔽后的序列:")
    print(masked_input)

    print("\n标签 (-100表示不需要预测):")
    print(labels)

    print("\n遮蔽位置:")
    print(mask_positions.int())

    print("\n遮蔽策略说明:")
    print("  - 15%的token被选中")
    print("  - 其中80%替换为[MASK](4)")
    print("  - 10%替换为随机token")
    print("  - 10%保持不变")

    # 5. 学习率调度演示
    print("\n\n[演示] 学习率warmup")
    print("=" * 70)

    model_warmup = create_simple_language_model()
    config_warmup = MLMConfig(
        vocab_size=1000,
        mask_prob=0.15,
        learning_rate=1e-4,
        warmup_steps=100
    )

    trainer_warmup = create_mlm_pretrainer(
        model=model_warmup,
        hidden_size=128,
        config=config_warmup,
        device='cpu'
    )

    print(f"  初始学习率: {config_warmup.learning_rate}")
    print(f"  Warmup步数: {config_warmup.warmup_steps}")

    print("\n训练前10步的学习率变化:")
    for step in range(10):
        input_ids, attention_mask = create_training_data()
        stats = trainer_warmup.train_step(input_ids, attention_mask)

        current_lr = trainer_warmup.optimizer.param_groups[0]['lr']
        print(f"  Step {step}: lr={current_lr:.6f}")

    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    print("\nMLM预训练关键点:")
    print("  1. BERT遮蔽策略: 80% [MASK], 10% 随机, 10% 不变")
    print("  2. RoBERTa改进: 移除NSP任务，动态遮蔽")
    print("  3. 遮蔽概率: 通常15%")
    print("  4. 学习率warmup: 避免训练初期不稳定")
    print("  5. 双向上下文: 学习更强的语言表示")


if __name__ == "__main__":
    main()
