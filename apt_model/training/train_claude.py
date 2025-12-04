#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Claude Model Trainer - 带反思层的训练器

专为Claude模型设计的训练器，考虑了：
1. 反思层(Reflection Layer)的训练
2. HHH原则(Helpful, Harmless, Honest)的优化
3. Constitutional AI训练策略
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import argparse

from apt_model.modeling.claude_model import ClaudeModel, create_claude_model
from apt_model.utils import get_device
from apt_model.modeling.chinese_tokenizer_integration import (
    get_appropriate_tokenizer,
    save_tokenizer,
)


class ClaudeDataset(Dataset):
    """
    Claude训练数据集

    包含文本及其对应的HHH标签（可选）
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 256,
        hhh_labels: Optional[List[Dict[str, float]]] = None
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.hhh_labels = hhh_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # 编码文本
        try:
            if hasattr(self.tokenizer, 'encode'):
                tokens = self.tokenizer.encode(
                    text,
                    max_length=self.max_length,
                    truncation=True
                )
            else:
                tokens = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors='pt'
                )
                tokens = tokens['input_ids'].squeeze(0).tolist()
        except:
            # Fallback
            tokens = [ord(c) % 50000 for c in text[:self.max_length]]

        if len(tokens) == 0:
            tokens = [0]

        # Padding
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]

        result = {'input_ids': torch.tensor(tokens, dtype=torch.long)}

        # 添加HHH标签（如果有）
        if self.hhh_labels is not None:
            label = self.hhh_labels[idx]
            result['hhh_label'] = torch.tensor([
                label.get('helpful', 0.5),
                label.get('harmless', 0.5),
                label.get('honest', 0.5)
            ], dtype=torch.float32)

        return result


def get_claude_training_texts() -> List[str]:
    """
    获取Claude风格的训练文本

    这些文本体现了Claude的HHH原则
    """
    return [
        # Helpful - 有帮助的回答
        "Hello! I'm Claude, an AI assistant created by Anthropic. How can I help you today?",
        "I'd be happy to help you understand this concept. Let me break it down step by step.",
        "That's a great question! Let me provide you with a comprehensive answer.",
        "I can assist you with that. Here's what I recommend...",
        "Let me help you solve this problem systematically.",

        # Harmless - 无害的拒绝
        "I'm not able to help with that request as it could potentially cause harm.",
        "I don't think I should provide information on that topic, as it could be misused.",
        "I'd rather not assist with that. Instead, I can help you with...",
        "I need to decline that request, but I'm happy to help with other questions.",

        # Honest - 诚实的承认限制
        "I'm not sure about that. Let me explain what I do know...",
        "I don't have access to real-time information, so I can't tell you the current...",
        "I'm uncertain about some details here. What I can say with confidence is...",
        "I may be wrong about this, but based on my training...",
        "I don't know the answer to that specific question.",

        # 中文 - Helpful
        "你好！我是Claude，一个AI助手。我很乐意帮助你。",
        "这是一个很好的问题。让我为你详细解答。",
        "我可以帮你解决这个问题。这是我的建议...",
        "让我一步一步地为你分析这个概念。",

        # 中文 - Harmless
        "抱歉，我不能帮助完成这个请求，因为它可能造成伤害。",
        "我认为我不应该提供这方面的信息。",
        "我需要拒绝这个请求，但我很乐意帮助解决其他问题。",

        # 中文 - Honest
        "我不太确定这个。让我解释一下我所知道的...",
        "我没有实时信息访问能力，所以无法告诉你当前的...",
        "关于这个我不太确定。我可以确信的是...",
        "我不知道这个具体问题的答案。",

        # 复杂对话 - 体现HHH平衡
        "I understand you're asking about [topic]. While I want to be helpful, I should note that [limitation]. What I can do is [alternative].",
        "That's a nuanced question. Let me give you a balanced perspective...",
        "I appreciate your question. To give you the most helpful answer, I need to first clarify...",
        "I aim to be both helpful and honest here. The situation is more complex than...",

        # 技术内容
        "Machine learning models learn patterns from data through training.",
        "The Transformer architecture uses self-attention mechanisms.",
        "Claude is based on Constitutional AI principles for safety and alignment.",
        "Large language models generate text by predicting the next token.",

        # 更多训练数据
        "I'm designed to be helpful, harmless, and honest in all interactions.",
        "Let me know if you need any clarification on my previous response.",
        "I'll do my best to assist you within my capabilities and ethical guidelines.",
        "Thank you for your patience. Let me try to explain this differently.",
    ]


class ClaudeTrainer:
    """
    Claude模型训练器

    特殊功能：
    1. 标准语言模型损失
    2. HHH反思损失
    3. 修正质量损失
    """

    def __init__(
        self,
        model: ClaudeModel,
        tokenizer,
        device: str = 'cuda',
        learning_rate: float = 3e-4,
        # 损失权重
        lm_weight: float = 1.0,
        reflection_weight: float = 0.1,
        hhh_target: float = 0.8  # HHH分数的目标值
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.lm_weight = lm_weight
        self.reflection_weight = reflection_weight
        self.hhh_target = hhh_target

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )

        # 将模型移到设备
        self.model.to(device)

    def compute_lm_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        vocab_size: int
    ) -> torch.Tensor:
        """计算语言模型损失"""
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            ignore_index=0
        )
        return loss

    def compute_reflection_loss(
        self,
        hhh_scores: Dict[str, torch.Tensor],
        target: float = None
    ) -> torch.Tensor:
        """
        计算反思损失 - 鼓励高HHH分数

        Args:
            hhh_scores: {'helpful': [batch, 1], 'harmless': [batch, 1], 'honest': [batch, 1]}
            target: 目标分数（默认self.hhh_target）

        Returns:
            loss: 反思损失
        """
        if target is None:
            target = self.hhh_target

        target_tensor = torch.tensor(target, device=hhh_scores['helpful'].device)

        # 每个HHH维度都应该接近target
        losses = []
        for score in hhh_scores.values():
            loss = F.mse_loss(score.squeeze(-1), target_tensor.expand_as(score.squeeze(-1)))
            losses.append(loss)

        return sum(losses) / len(losses)

    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        单个训练步骤

        Args:
            batch: {'input_ids': [batch, seq_len], 'hhh_label': [batch, 3] (optional)}

        Returns:
            losses: 损失字典
        """
        input_ids = batch['input_ids'].to(self.device)
        batch_size, seq_len = input_ids.shape

        # 准备输入和目标
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]

        # 前向传播（带反思信息）
        logits, reflection_info = self.model(inputs, return_reflection=True)
        vocab_size = logits.size(-1)

        # 1. 语言模型损失
        lm_loss = self.compute_lm_loss(logits, targets, vocab_size)

        # 2. 反思损失（鼓励高HHH分数）
        if 'hhh_scores' in reflection_info:
            reflection_loss = self.compute_reflection_loss(reflection_info['hhh_scores'])
        else:
            reflection_loss = torch.tensor(0.0, device=self.device)

        # 总损失
        total_loss = (
            self.lm_weight * lm_loss +
            self.reflection_weight * reflection_loss
        )

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # 返回损失信息
        losses = {
            'total': total_loss.item(),
            'lm': lm_loss.item(),
            'reflection': reflection_loss.item()
        }

        # 添加HHH分数信息
        if 'hhh_scores' in reflection_info:
            for key, value in reflection_info['hhh_scores'].items():
                losses[f'hhh_{key}'] = value.mean().item()

        return losses

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        total_epochs: int
    ) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()

        total_losses = {
            'total': 0.0,
            'lm': 0.0,
            'reflection': 0.0,
            'hhh_helpful': 0.0,
            'hhh_harmless': 0.0,
            'hhh_honest': 0.0
        }

        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{total_epochs}"
        )

        for batch_idx, batch in enumerate(progress_bar):
            losses = self.train_step(batch)

            # 累积损失
            for key, value in losses.items():
                if key in total_losses:
                    total_losses[key] += value

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{losses['total']:.4f}",
                'H': f"{losses.get('hhh_helpful', 0):.3f}",
                'H2': f"{losses.get('hhh_harmless', 0):.3f}",
                'H3': f"{losses.get('hhh_honest', 0):.3f}"
            })

        # 计算平均损失
        num_batches = len(dataloader)
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}

        return avg_losses

    def generate_sample(
        self,
        prompt: str = "Hello, I'm Claude.",
        max_length: int = 50,
        temperature: float = 0.8
    ) -> str:
        """生成样本文本"""
        self.model.eval()

        try:
            # 编码输入
            if hasattr(self.tokenizer, 'encode'):
                input_ids = self.tokenizer.encode(prompt, max_length=max_length)
            else:
                tokens = self.tokenizer(prompt, return_tensors='pt')
                input_ids = tokens['input_ids'].squeeze(0).tolist()

            input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)

            # 生成
            with torch.no_grad():
                for _ in range(max_length):
                    logits = self.model(input_ids)
                    next_token_logits = logits[0, -1, :] / temperature
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

                    # 停止条件
                    if next_token.item() == 0:  # EOS token
                        break

            # 解码
            if hasattr(self.tokenizer, 'decode'):
                generated_text = self.tokenizer.decode(input_ids[0].tolist())
            else:
                generated_text = self.tokenizer.decode(input_ids[0].tolist())

            return generated_text

        except Exception as e:
            return f"[生成失败: {e}]"


def train_claude_model(
    model_size: str = 'small',
    epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    reflection_weight: float = 0.1,
    save_path: str = None,
    texts: Optional[List[str]] = None,
    device: str = None
) -> Tuple[ClaudeModel, any]:
    """
    训练Claude模型的主函数

    Args:
        model_size: 'small', 'base', 'large'
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        reflection_weight: 反思损失权重
        save_path: 保存路径
        texts: 训练文本列表
        device: 设备

    Returns:
        (model, tokenizer)
    """
    # 设备
    if device is None:
        device = get_device()

    print(f"\n{'='*70}")
    print(f"训练 Claude 模型（带反思层）")
    print(f"{'='*70}")
    print(f"设备: {device}")
    print(f"模型大小: {model_size}")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"学习率: {learning_rate}")
    print(f"反思权重: {reflection_weight}")

    # 加载tokenizer
    print("\n[1/6] 加载tokenizer...")
    tokenizer = get_appropriate_tokenizer(language='en')
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50000
    print(f"词汇表大小: {vocab_size}")

    # 创建模型
    print(f"\n[2/6] 创建Claude模型（{model_size}）...")
    model = create_claude_model(
        vocab_size=vocab_size,
        model_size=model_size,
        use_reflection=True
    )
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 准备数据
    print("\n[3/6] 准备训练数据...")
    if texts is None:
        texts = get_claude_training_texts()
    print(f"训练样本数: {len(texts)}")

    dataset = ClaudeDataset(texts, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 创建训练器
    print("\n[4/6] 初始化训练器...")
    trainer = ClaudeTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=learning_rate,
        reflection_weight=reflection_weight
    )

    # 训练循环
    print(f"\n[5/6] 开始训练...\n")

    best_loss = float('inf')

    for epoch in range(epochs):
        # 训练一个epoch
        losses = trainer.train_epoch(dataloader, epoch, epochs)

        # 打印epoch总结
        print(f"\nEpoch {epoch+1}/{epochs} 总结:")
        print(f"  总损失: {losses['total']:.4f}")
        print(f"  语言模型损失: {losses['lm']:.4f}")
        print(f"  反思损失: {losses['reflection']:.4f}")
        print(f"  HHH分数:")
        print(f"    Helpful:  {losses['hhh_helpful']:.4f}")
        print(f"    Harmless: {losses['hhh_harmless']:.4f}")
        print(f"    Honest:   {losses['hhh_honest']:.4f}")

        # 每5个epoch生成样本
        if (epoch + 1) % 5 == 0:
            print("\n生成样本:")
            prompts = [
                "Hello, I'm Claude.",
                "你好，我是Claude。",
                "Can you help me?"
            ]
            for prompt in prompts:
                generated = trainer.generate_sample(prompt, max_length=30)
                print(f"  输入: {prompt}")
                print(f"  输出: {generated[:100]}...")
                print()

        # 保存最佳模型
        if losses['total'] < best_loss:
            best_loss = losses['total']
            print(f"✓ 新的最佳模型（损失: {best_loss:.4f}）")

    # 保存模型
    print(f"\n[6/6] 保存模型...")
    if save_path is None:
        save_path = f"./claude_model_{model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    os.makedirs(save_path, exist_ok=True)

    # 保存模型权重
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_size': model_size,
        'vocab_size': vocab_size,
        'epoch': epochs,
        'best_loss': best_loss
    }, os.path.join(save_path, 'claude_model.pt'))

    # 保存tokenizer
    save_tokenizer(tokenizer, save_path)

    print(f"模型已保存到: {save_path}")
    print(f"\n{'='*70}")
    print(f"Claude模型训练完成！")
    print(f"{'='*70}\n")

    return model, tokenizer


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='训练Claude模型（带反思层）')
    parser.add_argument('--model-size', type=str, default='small',
                       choices=['small', 'base', 'large'],
                       help='模型大小')
    parser.add_argument('--epochs', type=int, default=30,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--reflection-weight', type=float, default=0.1,
                       help='反思损失权重')
    parser.add_argument('--save-path', type=str, default=None,
                       help='模型保存路径')
    parser.add_argument('--device', type=str, default=None,
                       help='设备 (cuda/cpu)')

    args = parser.parse_args()

    train_claude_model(
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        reflection_weight=args.reflection_weight,
        save_path=args.save_path,
        device=args.device
    )


if __name__ == '__main__':
    main()
