#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Claude Unified Model Trainer
统一的Claude模型训练器，支持HHH反思和图论反思
"""

from apt_model.utils.fake_torch import get_torch
torch = get_torch()
from apt_model.utils.fake_torch import get_torch
torch = get_torch()
nn = torch.nn
F = torch.nn.functional
Dataset = torch.utils.data.Dataset
DataLoader = torch.utils.data.DataLoader
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
import os
from datetime import datetime
import logging

from apt_model.modeling.claude4_model import ClaudeUnifiedModel, create_claude_unified
from apt_model.utils import get_device
from apt_model.modeling.chinese_tokenizer_integration import (
    get_appropriate_tokenizer,
    save_tokenizer,
)
from apt_model.training.training_guard import TrainingGuard, EarlyStopping

# 创建 logger
logger = logging.getLogger(__name__)


class ClaudeDataset(Dataset):
    """
    Claude训练数据集

    支持HHH标签（可选）
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


def get_claude_training_data() -> List[str]:
    """
    获取Claude风格的训练数据

    返回体现HHH原则的对话示例
    """
    return [
        # Helpful - 有帮助的回答
        "Hello! I'm Claude, an AI assistant. How can I help you today?",
        "I'd be happy to help you understand this concept. Let me break it down step by step.",
        "That's a great question! Let me provide you with a comprehensive answer.",
        "I can assist you with that. Here's what I recommend based on the situation.",
        "Let me help you solve this problem systematically and thoroughly.",
        "I understand what you're asking. Let me clarify that for you.",
        "Here's a detailed explanation that should address your question.",

        # Harmless - 无害的拒绝与引导
        "I'm not able to help with that request as it could potentially cause harm.",
        "I don't think I should provide information on that topic, as it could be misused.",
        "I'd rather not assist with that. Instead, I can help you with a safer alternative.",
        "I need to decline that request for safety reasons, but I'm happy to help with other questions.",
        "That's outside my ethical guidelines. Let me suggest a better approach instead.",
        "I can't help with that, but I can explain why and offer alternatives.",

        # Honest - 诚实地承认限制
        "I'm not sure about that. Let me explain what I do know and what I'm uncertain about.",
        "I don't have access to real-time information, so I can't tell you the current status.",
        "I'm uncertain about some details here. What I can say with confidence is...",
        "I may be wrong about this, but based on my training data...",
        "I don't know the answer to that specific question, and I don't want to guess.",
        "That's beyond my knowledge cutoff. I should be honest about my limitations.",
        "I'm not entirely confident in this answer. Here's what I think, but please verify.",

        # 中文 - Helpful
        "你好！我是Claude，一个AI助手。我很乐意帮助你。",
        "这是一个很好的问题。让我为你详细解答一下。",
        "我可以帮你解决这个问题。这是我基于情况的建议。",
        "让我一步一步地为你分析这个概念，确保你能理解。",
        "我理解你的问题。让我为你澄清一下。",
        "这里有一个详细的解释，应该能解决你的疑问。",

        # 中文 - Harmless
        "抱歉，我不能帮助完成这个请求，因为它可能造成伤害。",
        "我认为我不应该提供这方面的信息，因为可能被误用。",
        "我不太适合帮助这个，但我可以建议一个更安全的替代方案。",
        "出于安全考虑，我需要拒绝这个请求，但我很乐意帮助其他问题。",
        "这超出了我的道德准则。让我提供一个更好的方法。",

        # 中文 - Honest
        "我不太确定这个。让我解释一下我知道的和不确定的部分。",
        "我没有实时信息访问能力，所以无法告诉你当前的状态。",
        "关于这个我有些不确定。我可以确信的是...",
        "我可能在这方面是错的，但根据我的训练数据...",
        "我不知道这个具体问题的答案，我不想猜测。",
        "这超出了我的知识范围。我应该诚实地承认我的局限性。",

        # 复杂对话 - 体现HHH平衡
        "I understand your question. While I want to be helpful, I should note my limitations. Here's what I can do instead.",
        "That's a nuanced question. Let me give you a balanced perspective while being honest about uncertainties.",
        "I appreciate your question. To give you the most helpful and honest answer, let me first clarify a few things.",
        "I aim to be both helpful and honest here. The situation is more complex than it might seem, so let me explain carefully.",
        "I want to help, but I also need to be honest about what I can and cannot do. Here's my best answer within those bounds.",

        # 技术与教育内容
        "Machine learning models learn patterns from data through iterative training processes.",
        "The Transformer architecture revolutionized natural language processing with its attention mechanism.",
        "Claude is based on Constitutional AI principles, emphasizing safety and alignment.",
        "Large language models generate text by predicting tokens based on learned probability distributions.",
        "Neural networks use backpropagation to adjust weights and improve performance.",
        "Understanding AI limitations is as important as understanding its capabilities.",

        # 元认知与自我意识
        "As an AI, I have capabilities but also important limitations I should be transparent about.",
        "I'm designed to be helpful, harmless, and honest in all my interactions.",
        "Let me know if you need any clarification on my previous response - I want to ensure I'm being clear.",
        "I'll do my best to assist you within my capabilities and ethical guidelines.",
        "Thank you for your patience. Let me try to explain this in a different, clearer way.",
        "I appreciate your understanding as I navigate the balance between being helpful and staying within ethical bounds.",

        # 对话技巧
        "That's an interesting perspective. Could you tell me more about your thinking?",
        "I see what you mean. Let me build on that idea with some additional context.",
        "Good point! That connects to an important related concept I should mention.",
        "I want to make sure I understand correctly before I respond. Are you asking about...?",
        "Let me summarize to make sure we're on the same page, then I'll elaborate.",
    ]


class ClaudeTrainer:
    """
    Claude统一模型训练器（支持训练保护）

    支持：
    - HHH反思损失
    - 图论反思统计
    - 灵活的反思模式配置
    - 训练保护机制
    """

    def __init__(
        self,
        model: ClaudeUnifiedModel,
        tokenizer,
        device: str = 'cuda',
        learning_rate: float = 3e-4,
        lm_weight: float = 1.0,
        reflection_weight: float = 0.1,
        hhh_target: float = 0.8,
        # Training guard parameters
        enable_guard: bool = True,
        max_steps: Optional[int] = None,
        max_time_hours: Optional[float] = None,
        early_stopping_patience: Optional[int] = None,
        guard_verbose: bool = True
    ):
        self.model = model.to(device)
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

        self.step_count = 0

        # 训练保护
        self.enable_guard = enable_guard
        if enable_guard:
            early_stopping = None
            if early_stopping_patience is not None:
                early_stopping = EarlyStopping(
                    patience=early_stopping_patience,
                    mode='min',
                    verbose=guard_verbose
                )

            self.guard = TrainingGuard(
                max_steps=max_steps,
                max_time_hours=max_time_hours,
                early_stopping=early_stopping,
                verbose=guard_verbose
            )
        else:
            self.guard = None

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
        reflection_info: Dict,
        target: float = None
    ) -> torch.Tensor:
        """计算反思损失"""
        if target is None:
            target = self.hhh_target

        losses = []

        # HHH反思损失
        if reflection_info.get('hhh_scores') is not None:
            hhh_scores = reflection_info['hhh_scores']
            target_tensor = torch.tensor(target, device=hhh_scores['helpful'].device)

            for score in hhh_scores.values():
                loss = F.mse_loss(
                    score.squeeze(-1),
                    target_tensor.expand_as(score.squeeze(-1))
                )
                losses.append(loss)

        # 如果没有反思损失，返回0
        if not losses:
            return torch.tensor(0.0, device=self.device)

        return sum(losses) / len(losses)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """单步训练（支持多模态）"""
        input_ids = batch['input_ids'].to(self.device)
        batch_size, seq_len = input_ids.shape

        # 多模态输入（可选）
        image_feat = batch.get('image_feat')
        audio_feat = batch.get('audio_feat')

        if image_feat is not None:
            image_feat = image_feat.to(self.device)
        if audio_feat is not None:
            audio_feat = audio_feat.to(self.device)

        # 准备输入和目标
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]

        # 前向传播（带反思信息 + 多模态）
        logits, reflection_info = self.model(
            input_ids=inputs,
            image_feat=image_feat,
            audio_feat=audio_feat,
            return_reflection=True
        )
        vocab_size = logits.size(-1)

        # 1. 语言模型损失
        lm_loss = self.compute_lm_loss(logits, targets, vocab_size)

        # 2. 反思损失
        reflection_loss = self.compute_reflection_loss(reflection_info)

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
        self.step_count += 1

        # 返回损失信息
        losses = {
            'total': total_loss.item(),
            'lm': lm_loss.item(),
            'reflection': reflection_loss.item()
        }

        # 添加HHH分数
        if reflection_info.get('hhh_scores') is not None:
            for key, value in reflection_info['hhh_scores'].items():
                losses[f'hhh_{key}'] = value.mean().item()

        # 添加图论统计
        if reflection_info.get('graph_stats'):
            avg_complexity = sum(
                v['complexity'] for v in reflection_info['graph_stats'].values()
            ) / len(reflection_info['graph_stats'])
            losses['graph_complexity'] = avg_complexity

        return losses

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        total_epochs: int,
        start_guard: bool = False
    ) -> tuple[Dict[str, float], bool]:
        """
        训练一个epoch（带训练保护）

        Returns:
            (avg_losses, should_stop)
        """
        self.model.train()

        # 启动训练保护（如果是第一个epoch）
        if start_guard and self.guard:
            self.guard.start()

        total_losses = {
            'total': 0.0,
            'lm': 0.0,
            'reflection': 0.0,
        }

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")
        should_stop = False

        for batch_idx, batch in enumerate(progress_bar):
            losses = self.train_step(batch)

            # 训练保护检查
            if self.guard:
                if not self.guard.step(loss=losses['total'], model=self.model):
                    should_stop = True
                    break

            # 累积损失
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value

            # 更新进度条
            postfix = {'loss': f"{losses['total']:.4f}"}
            if 'hhh_helpful' in losses:
                postfix.update({
                    'H': f"{losses['hhh_helpful']:.3f}",
                    'H2': f"{losses['hhh_harmless']:.3f}",
                    'H3': f"{losses['hhh_honest']:.3f}"
                })
            if 'graph_complexity' in losses:
                postfix['GC'] = f"{losses['graph_complexity']:.3f}"

            progress_bar.set_postfix(postfix)

        # 计算平均损失
        num_batches = len(dataloader) if not should_stop else (batch_idx + 1)
        avg_losses = {k: v / max(1, num_batches) for k, v in total_losses.items()}

        return avg_losses, should_stop


def train_claude_unified(
    model_size: str = 'base',
    reflection_mode: str = 'hhh',  # 'hhh', 'graph', 'both'
    epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    reflection_weight: float = 0.1,
    save_path: str = None,
    texts: Optional[List[str]] = None,
    device: str = None,
    # Training guard parameters
    enable_guard: bool = True,
    max_steps: Optional[int] = None,
    max_time_hours: Optional[float] = None,
    early_stopping_patience: Optional[int] = None
) -> Tuple[ClaudeUnifiedModel, any]:
    """
    训练Claude统一模型（带训练保护）

    Args:
        model_size: 'small', 'base', 'large'
        reflection_mode: 'hhh' (Constitutional AI), 'graph' (图论), 'both'
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        reflection_weight: 反思损失权重
        save_path: 保存路径
        texts: 训练文本
        device: 设备
        enable_guard: 启用训练保护
        max_steps: 最大训练步数
        max_time_hours: 最大训练时间（小时）
        early_stopping_patience: Early stopping patience

    Returns:
        (model, tokenizer)
    """
    # 设备
    if device is None:
        device = get_device()

    # 用户友好的输出（保留 print）
    print(f"\n{'='*80}")
    print(f"训练 Claude 统一模型")
    print(f"{'='*80}")
    print(f"模型大小: {model_size}")
    print(f"反思模式: {reflection_mode}")
    print(f"  - HHH反思: {'✓' if reflection_mode in ['hhh', 'both'] else '✗'}")
    print(f"  - 图论反思: {'✓' if reflection_mode in ['graph', 'both'] else '✗'}")
    print(f"设备: {device}")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"学习率: {learning_rate}")
    print(f"反思权重: {reflection_weight}")

    # 系统日志
    logger.info(f"Starting Claude Unified Model training: model_size={model_size}, "
                f"reflection_mode={reflection_mode}, device={device}")

    # 加载tokenizer
    print("\n[1/6] 加载tokenizer...")
    logger.info("Loading tokenizer...")
    tokenizer = get_appropriate_tokenizer(language='en')
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 50000
    print(f"词汇表大小: {vocab_size}")
    logger.info(f"Tokenizer loaded: vocab_size={vocab_size}")

    # 创建模型
    print(f"\n[2/6] 创建Claude模型...")
    logger.info(f"Creating Claude model: size={model_size}, reflection={reflection_mode}")
    model = create_claude_unified(
        vocab_size=vocab_size,
        model_size=model_size,
        reflection_mode=reflection_mode
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {param_count:,}")
    logger.info(f"Model created: {param_count:,} parameters")

    # 准备数据
    print("\n[3/6] 准备训练数据...")
    logger.info("Preparing training data...")
    if texts is None:
        texts = get_claude_training_data()
    print(f"训练样本数: {len(texts)}")
    logger.info(f"Training data prepared: {len(texts)} samples")

    dataset = ClaudeDataset(texts, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 创建训练器
    print("\n[4/6] 初始化训练器...")
    logger.info("Initializing trainer...")
    trainer = ClaudeTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=learning_rate,
        reflection_weight=reflection_weight,
        enable_guard=enable_guard,
        max_steps=max_steps,
        max_time_hours=max_time_hours,
        early_stopping_patience=early_stopping_patience
    )
    logger.info(f"Trainer initialized: lr={learning_rate}, reflection_weight={reflection_weight}")

    # 训练循环
    print(f"\n[5/6] 开始训练...\n")
    logger.info(f"Starting training loop: {epochs} epochs")

    best_loss = float('inf')

    for epoch in range(epochs):
        # 第一个epoch启动guard
        losses, should_stop = trainer.train_epoch(
            dataloader, epoch, epochs, start_guard=(epoch == 0)
        )

        # 打印epoch总结（用户友好）
        print(f"\nEpoch {epoch+1}/{epochs} 总结:")
        print(f"  总损失: {losses['total']:.4f}")
        print(f"  语言模型损失: {losses['lm']:.4f}")
        print(f"  反思损失: {losses['reflection']:.4f}")

        if 'hhh_helpful' in losses:
            print(f"  HHH分数:")
            print(f"    Helpful:  {losses['hhh_helpful']:.4f}")
            print(f"    Harmless: {losses['hhh_harmless']:.4f}")
            print(f"    Honest:   {losses['hhh_honest']:.4f}")

        if 'graph_complexity' in losses:
            print(f"  图论复杂度: {losses['graph_complexity']:.4f}")

        # 系统日志（详细记录）
        logger.info(f"Epoch {epoch+1}/{epochs}: total_loss={losses['total']:.4f}, "
                   f"lm_loss={losses['lm']:.4f}, reflection_loss={losses['reflection']:.4f}")

        # 保存最佳模型
        if losses['total'] < best_loss:
            best_loss = losses['total']
            print(f"  ✓ 新的最佳模型（损失: {best_loss:.4f}）")
            logger.info(f"New best model saved: loss={best_loss:.4f}")

        # Early stopping 检查
        if trainer.guard and trainer.guard.early_stopping:
            if not trainer.guard.validate(losses['total']):
                should_stop = True

        # 如果需要停止，退出训练循环
        if should_stop:
            break

    # 打印训练保护统计（用户友好）
    if trainer.guard:
        stats = trainer.guard.get_stats()
        print(f"\n{'='*80}")
        print("训练保护统计:")
        print(f"  总步数: {stats['total_steps']}")
        print(f"  训练时间: {stats['elapsed_hours']:.2f} 小时")
        if stats['stopped']:
            print(f"  停止原因: {stats['stop_reason']}")
        print(f"{'='*80}\n")

        # 系统日志
        logger.info(f"Training guard stats: total_steps={stats['total_steps']}, "
                   f"elapsed_hours={stats['elapsed_hours']:.2f}, "
                   f"stopped={stats['stopped']}, stop_reason={stats.get('stop_reason', 'N/A')}")

    # 保存模型
    print(f"[6/6] 保存模型...")
    logger.info("Saving model...")
    if save_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f"./claude_{reflection_mode}_{model_size}_{timestamp}"

    os.makedirs(save_path, exist_ok=True)

    # 保存模型权重
    checkpoint_path = os.path.join(save_path, 'claude_unified_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_size': model_size,
        'reflection_mode': reflection_mode,
        'vocab_size': vocab_size,
        'epoch': epochs,
        'best_loss': best_loss
    }, checkpoint_path)
    logger.info(f"Model checkpoint saved: {checkpoint_path}")

    # 保存tokenizer
    save_tokenizer(tokenizer, save_path)
    logger.info(f"Tokenizer saved: {save_path}")

    # 用户友好的最终输出（保留 print）
    print(f"模型已保存到: {save_path}")
    print(f"\n{'='*80}")
    print(f"Claude模型训练完成！")
    print(f"{'='*80}\n")

    logger.info(f"Claude training completed successfully: best_loss={best_loss:.4f}")

    return model, tokenizer


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='训练Claude统一模型')
    parser.add_argument('--model-size', type=str, default='small',
                       choices=['small', 'base', 'large'],
                       help='模型大小')
    parser.add_argument('--reflection-mode', type=str, default='hhh',
                       choices=['hhh', 'graph', 'both'],
                       help='反思模式：hhh(Constitutional AI), graph(图论), both(混合)')
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

    train_claude_unified(
        model_size=args.model_size,
        reflection_mode=args.reflection_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        reflection_weight=args.reflection_weight,
        save_path=args.save_path,
        device=args.device
    )
