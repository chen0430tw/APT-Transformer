#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一的GPT模型变体训练器
支持训练: GPT-4o, GPT-o3, GPT-5模型

这个训练器整合了项目中的三个高级GPT模型变体，提供统一的训练接口。
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
from typing import Optional, Dict, Any, List
import argparse

# 导入各个GPT模型
try:
    from apt_model.modeling.gpt4o_model import GPT4oModel
except ImportError:
    GPT4oModel = None
    print("Warning: GPT4oModel not available")

try:
    from apt_model.modeling.gpto3_model import GPTo3Model
except ImportError:
    GPTo3Model = None
    print("Warning: GPTo3Model not available")

try:
    from apt_model.modeling.gpt5_model import GPT5Model
except ImportError:
    GPT5Model = None
    print("Warning: GPT5Model not available")

from apt_model.utils import get_device
from apt_model.modeling.chinese_tokenizer_integration import (
    get_appropriate_tokenizer,
    save_tokenizer,
)


class TextDataset(Dataset):
    """简单的文本数据集"""
    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # 编码文本
        try:
            if hasattr(self.tokenizer, 'encode'):
                tokens = self.tokenizer.encode(text, max_length=self.max_length, truncation=True)
            else:
                tokens = self.tokenizer(text, max_length=self.max_length, truncation=True, return_tensors='pt')
                tokens = tokens['input_ids'].squeeze(0).tolist()
        except:
            # 简单fallback
            tokens = [ord(c) % 50000 for c in text[:self.max_length]]

        # 确保至少有一些tokens
        if len(tokens) == 0:
            tokens = [0]

        # Padding
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]

        return torch.tensor(tokens, dtype=torch.long)


def get_training_texts() -> List[str]:
    """获取训练文本 - Claude风格的对话数据"""
    return [
        # Claude风格对话
        "Hello! I'm Claude, an AI assistant created by Anthropic.",
        "I'm designed to be helpful, harmless, and honest in my interactions.",
        "How can I assist you today?",
        "I'd be happy to help you with questions about various topics.",
        "While I have many capabilities, I also have important limitations I should be transparent about.",
        "I cannot access the internet or real-time information.",
        "I aim to provide thoughtful, nuanced responses based on my training.",
        "If I'm uncertain about something, I'll let you know rather than guessing.",
        "My goal is to be as helpful as possible while staying within ethical boundaries.",
        "I'm here to assist through conversation and analysis.",

        # 中文Claude风格
        "你好！我是Claude，一个由Anthropic开发的AI助手。",
        "我专注于提供有帮助、无害且诚实的对话。",
        "我可以帮助你解决问题、回答疑问或者进行对话。",
        "作为AI助手，我会尽力提供准确的信息。",
        "我也会诚实地承认我不知道的事情。",
        "我理解中文和英文，可以用两种语言交流。",
        "我的目标是在保持道德边界的同时尽可能提供帮助。",

        # 技术内容
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand human language.",
        "Transformers are the foundation of modern language models.",
        "Attention mechanisms allow models to focus on relevant information.",

        # 更多训练数据
        "The weather is lovely today.",
        "Thank you for your question.",
        "I understand what you're asking.",
        "Let me help you with that.",
        "That's an interesting perspective.",
        "机器学习是人工智能的一个子领域。",
        "深度学习使用多层神经网络。",
        "自然语言处理让计算机理解人类语言。",
        "Transformer架构是现代语言模型的基础。",
        "注意力机制让模型能够关注相关信息。",
    ]


def create_model(model_type: str, vocab_size: int = 50000, device: str = 'cuda') -> nn.Module:
    """
    创建指定类型的GPT模型

    Args:
        model_type: 模型类型 ('gpt4o', 'gpto3', 'gpt5')
        vocab_size: 词汇表大小
        device: 设备

    Returns:
        模型实例
    """
    if model_type == 'gpt4o':
        if GPT4oModel is None:
            raise ImportError("GPT4oModel is not available")
        model = GPT4oModel(
            vocab_size=vocab_size,
            d_model=512,
            num_heads=8,
            num_layers=6,
            d_ff=2048,
            max_seq_len=128,
            dropout=0.1
        )

    elif model_type == 'gpto3':
        if GPTo3Model is None:
            raise ImportError("GPTo3Model is not available")
        model = GPTo3Model(
            vocab_size=vocab_size,
            d_model=512,
            num_heads=8,
            num_layers=6,
            d_ff=2048,
            max_seq_len=128,
            dropout=0.1,
            # o3特有参数
            reasoning_steps=3,
            halt_threshold=0.95
        )

    elif model_type == 'gpt5':
        if GPT5Model is None:
            raise ImportError("GPT5Model is not available")
        model = GPT5Model(
            vocab_size=vocab_size,
            d_model=512,
            num_heads=8,
            num_layers=6,
            d_ff=2048,
            max_seq_len=128,
            dropout=0.1,
            # MoE参数
            num_experts=8,
            top_k=2
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from 'gpt4o', 'gpto3', 'gpt5'")

    return model.to(device)


def train_gpt_model(
    model_type: str = 'gpt4o',
    epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    save_path: str = None,
    texts: Optional[List[str]] = None,
    device: str = None
):
    """
    训练GPT模型的主函数

    Args:
        model_type: 模型类型 ('gpt4o', 'gpto3', 'gpt5')
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        save_path: 保存路径
        texts: 训练文本列表
        device: 设备
    """
    # 设置设备
    if device is None:
        device = get_device()

    print(f"\n{'='*60}")
    print(f"开始训练 {model_type.upper()} 模型")
    print(f"{'='*60}")
    print(f"设备: {device}")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"学习率: {learning_rate}")

    # 获取tokenizer
    print("\n[1/5] 加载tokenizer...")
    tokenizer = get_appropriate_tokenizer(language='en')
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50000
    print(f"词汇表大小: {vocab_size}")

    # 创建模型
    print(f"\n[2/5] 创建{model_type.upper()}模型...")
    model = create_model(model_type, vocab_size=vocab_size, device=device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 准备数据
    print("\n[3/5] 准备训练数据...")
    if texts is None:
        texts = get_training_texts()
    print(f"训练样本数: {len(texts)}")

    dataset = TextDataset(texts, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 设置优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 训练循环
    print(f"\n[4/5] 开始训练...")
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            batch = batch.to(device)

            # 准备输入和目标
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]

            # 前向传播
            optimizer.zero_grad()

            # 不同模型可能有不同的调用方式
            try:
                if model_type == 'gpto3':
                    # GPTo3可能需要text_ids参数
                    logits = model(text_ids=input_ids)
                else:
                    logits = model(input_ids)
            except Exception as e:
                print(f"\n模型前向传播错误: {e}")
                # Fallback: 尝试通用调用
                logits = model(input_ids)

            # 计算损失
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1),
                ignore_index=0  # 忽略padding
            )

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - 平均损失: {avg_loss:.4f}")

        # 每5个epoch生成样本
        if (epoch + 1) % 5 == 0:
            print("\n生成样本:")
            model.eval()
            with torch.no_grad():
                sample_input = torch.randint(0, vocab_size, (1, 10)).to(device)
                try:
                    if model_type == 'gpto3':
                        sample_output = model(text_ids=sample_input)
                    else:
                        sample_output = model(sample_input)
                    print(f"样本输出形状: {sample_output.shape}")
                except Exception as e:
                    print(f"生成样本时出错: {e}")
            model.train()

    # 保存模型
    print(f"\n[5/5] 保存模型...")
    if save_path is None:
        save_path = f"./{model_type}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    os.makedirs(save_path, exist_ok=True)

    # 保存模型权重
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': model_type,
        'vocab_size': vocab_size,
        'epoch': epochs,
    }, os.path.join(save_path, 'model.pt'))

    # 保存tokenizer
    save_tokenizer(tokenizer, save_path)

    print(f"模型已保存到: {save_path}")
    print(f"\n{'='*60}")
    print(f"{model_type.upper()}模型训练完成！")
    print(f"{'='*60}\n")

    return model, tokenizer


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='训练GPT模型变体')
    parser.add_argument('--model', type=str, default='gpt4o',
                       choices=['gpt4o', 'gpto3', 'gpt5'],
                       help='模型类型')
    parser.add_argument('--epochs', type=int, default=20,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--save-path', type=str, default=None,
                       help='模型保存路径')
    parser.add_argument('--device', type=str, default=None,
                       help='设备 (cuda/cpu)')

    args = parser.parse_args()

    train_gpt_model(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_path,
        device=args.device
    )


if __name__ == '__main__':
    main()
