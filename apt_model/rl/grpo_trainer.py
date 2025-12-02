#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GRPO训练器 (Group Relative Policy Optimization)

分组相对策略优化 - 一种高效的在线RL方法

核心思想:
- 将样本分组
- 计算组内相对优势
- 使用相对优势更新策略
- 比PPO更简单，比DPO需要奖励模型

论文参考:
- DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

作者: chen0430tw
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
import logging
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """GRPO配置"""
    # 分组参数
    group_size: int = 4  # 每组的样本数

    # 优势计算
    advantage_type: str = "relative"  # "relative", "normalized", "rank"
    temperature: float = 1.0  # 优势温度

    # 策略更新
    learning_rate: float = 1e-5
    max_grad_norm: float = 1.0
    clip_range: float = 0.2  # 类似PPO的裁剪

    # KL惩罚
    kl_coef: float = 0.1  # KL散度系数
    target_kl: Optional[float] = None  # 目标KL (自适应调整kl_coef)

    # 其他
    value_loss_coef: float = 0.5  # 价值损失系数 (如果使用critic)
    use_critic: bool = False  # 是否使用独立的critic


class GRPOTrainer:
    """
    GRPO训练器

    实现分组相对策略优化算法

    算法流程:
    1. 对于每个prompt，生成group_size个响应
    2. 使用奖励模型对响应评分
    3. 计算组内相对优势
    4. 使用优势更新策略

    优势:
    - 不需要参考模型 (相比DPO)
    - 比PPO更简单
    - 适合在线学习
    - 计算效率高
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reward_model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        config: Optional[GRPOConfig] = None,
        device: str = "cuda"
    ):
        """
        初始化GRPO训练器

        Args:
            policy_model: 策略模型 (要训练的模型)
            reward_model: 奖励模型 (可选，用于评分)
            optimizer: 优化器
            config: GRPO配置
            device: 设备
        """
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.config = config or GRPOConfig()
        self.device = device

        # 优化器
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                policy_model.parameters(),
                lr=self.config.learning_rate
            )
        else:
            self.optimizer = optimizer

        # 移动到设备
        self.policy_model.to(device)
        if self.reward_model is not None:
            self.reward_model.to(device)
            self.reward_model.eval()

        # 统计
        self.stats = {
            'total_steps': 0,
            'mean_reward': 0.0,
            'group_variance': 0.0,
            'policy_loss': 0.0,
            'kl_divergence': 0.0
        }

        logger.info(f"[GRPO] 训练器初始化完成 (group_size={self.config.group_size})")

    def get_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        获取序列的对数概率

        Args:
            model: 模型
            input_ids: [batch, seq_len]
            attention_mask: mask

        Returns:
            log_probs: [batch, seq_len]
        """
        # 前向传播
        outputs = model(input_ids, attention_mask=attention_mask)

        # 获取logits
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs

        # 计算log_probs
        log_probs = F.log_softmax(logits, dim=-1)

        # 收集对应token的log_probs
        gathered_log_probs = torch.gather(
            log_probs[:, :-1, :],  # 去掉最后一个位置
            dim=-1,
            index=input_ids[:, 1:].unsqueeze(-1)  # 使用下一个token作为标签
        ).squeeze(-1)

        return gathered_log_probs

    def compute_rewards(
        self,
        responses: torch.Tensor,
        response_masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        使用奖励模型计算奖励

        Args:
            responses: [batch, seq_len]
            response_masks: [batch, seq_len]

        Returns:
            rewards: [batch]
        """
        if self.reward_model is None:
            # 如果没有奖励模型，返回随机奖励 (仅用于演示)
            return torch.randn(responses.size(0), device=self.device)

        with torch.no_grad():
            reward_output = self.reward_model(responses, response_masks)
            rewards = reward_output['rewards'].squeeze(-1)  # [batch]

        return rewards

    def compute_group_advantages(
        self,
        rewards: torch.Tensor,
        group_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算分组相对优势

        Args:
            rewards: [batch] 或 [num_groups * group_size]
            group_size: 每组大小 (如果None，使用config中的)

        Returns:
            (advantages, stats)
        """
        if group_size is None:
            group_size = self.config.group_size

        batch_size = rewards.size(0)
        num_groups = batch_size // group_size

        if num_groups == 0:
            # 不够一组，使用全局优势
            advantages = rewards - rewards.mean()
            stats = {
                'group_variance': rewards.var().item(),
                'mean_reward': rewards.mean().item()
            }
            return advantages, stats

        # 重塑为 [num_groups, group_size]
        grouped_rewards = rewards[:num_groups * group_size].view(num_groups, group_size)

        if self.config.advantage_type == "relative":
            # 相对优势: r - mean(r_group)
            group_means = grouped_rewards.mean(dim=1, keepdim=True)
            advantages = grouped_rewards - group_means

        elif self.config.advantage_type == "normalized":
            # 归一化优势: (r - mean) / std
            group_means = grouped_rewards.mean(dim=1, keepdim=True)
            group_stds = grouped_rewards.std(dim=1, keepdim=True) + 1e-8
            advantages = (grouped_rewards - group_means) / group_stds

        elif self.config.advantage_type == "rank":
            # 基于排名的优势
            # 排名越高，优势越大
            ranks = torch.argsort(torch.argsort(grouped_rewards, dim=1), dim=1)
            advantages = ranks.float() - (group_size - 1) / 2
            advantages = advantages / (group_size / 2)  # 归一化到[-1, 1]

        else:
            raise ValueError(f"Unknown advantage type: {self.config.advantage_type}")

        # 应用温度
        advantages = advantages / self.config.temperature

        # 展平回 [num_groups * group_size]
        advantages = advantages.view(-1)

        # 统计
        stats = {
            'group_variance': grouped_rewards.var(dim=1).mean().item(),
            'mean_reward': grouped_rewards.mean().item()
        }

        return advantages, stats

    def compute_policy_loss(
        self,
        responses: torch.Tensor,
        response_masks: torch.Tensor,
        advantages: torch.Tensor,
        old_log_probs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算策略损失

        GRPO损失: -E[A * log π(a|s)]

        如果提供old_log_probs，使用PPO风格的裁剪

        Args:
            responses: [batch, seq_len]
            response_masks: [batch, seq_len]
            advantages: [batch]
            old_log_probs: [batch, seq_len] (可选)

        Returns:
            (loss, metrics)
        """
        # 获取当前策略的log_probs
        current_log_probs = self.get_log_probs(
            self.policy_model, responses, response_masks
        )

        # 计算序列总log_prob
        if response_masks is not None:
            mask = response_masks[:, 1:].float()  # 去掉第一个位置
            current_log_prob = (current_log_probs * mask).sum(dim=-1)
        else:
            current_log_prob = current_log_probs.sum(dim=-1)

        if old_log_probs is not None:
            # PPO风格裁剪
            if response_masks is not None:
                mask = response_masks[:, 1:].float()
                old_log_prob = (old_log_probs * mask).sum(dim=-1)
            else:
                old_log_prob = old_log_probs.sum(dim=-1)

            # 计算比率
            ratio = torch.exp(current_log_prob - old_log_prob)

            # 裁剪
            clipped_ratio = torch.clamp(
                ratio,
                1.0 - self.config.clip_range,
                1.0 + self.config.clip_range
            )

            # 策略损失
            surr1 = ratio * advantages
            surr2 = clipped_ratio * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # KL散度 (近似)
            kl_div = (old_log_prob - current_log_prob).mean()

        else:
            # 简单策略梯度
            policy_loss = -(current_log_prob * advantages).mean()
            kl_div = torch.tensor(0.0)

        metrics = {
            'policy_loss': policy_loss.item(),
            'kl_divergence': kl_div.item() if isinstance(kl_div, torch.Tensor) else 0.0,
            'mean_advantage': advantages.mean().item()
        }

        return policy_loss, metrics

    def train_step(
        self,
        responses: torch.Tensor,
        response_masks: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        old_log_probs: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        执行一次GRPO训练步骤

        Args:
            responses: 生成的响应 [batch, seq_len]
            response_masks: mask [batch, seq_len]
            rewards: 奖励 [batch] (如果None，使用reward_model计算)
            old_log_probs: 旧的log_probs [batch, seq_len] (用于PPO风格裁剪)

        Returns:
            统计信息
        """
        self.policy_model.train()

        # 1. 计算奖励 (如果未提供)
        if rewards is None:
            rewards = self.compute_rewards(responses, response_masks)

        # 2. 计算分组优势
        advantages, group_stats = self.compute_group_advantages(rewards)

        # 确保advantages和responses大小匹配
        if advantages.size(0) < responses.size(0):
            # 如果有余数，只使用完整的组
            num_samples = advantages.size(0)
            responses = responses[:num_samples]
            if response_masks is not None:
                response_masks = response_masks[:num_samples]
            if old_log_probs is not None:
                old_log_probs = old_log_probs[:num_samples]

        # 3. 计算策略损失
        policy_loss, policy_metrics = self.compute_policy_loss(
            responses, response_masks, advantages, old_log_probs
        )

        # 4. 反向传播
        self.optimizer.zero_grad()
        policy_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(),
            self.config.max_grad_norm
        )

        self.optimizer.step()

        # 5. 更新统计
        self.stats['total_steps'] += 1
        self.stats['mean_reward'] = group_stats['mean_reward']
        self.stats['group_variance'] = group_stats['group_variance']
        self.stats['policy_loss'] = policy_metrics['policy_loss']
        self.stats['kl_divergence'] = policy_metrics['kl_divergence']

        return {
            **group_stats,
            **policy_metrics,
            'total_loss': policy_loss.item()
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()


# ==================== 便捷函数 ====================

def create_grpo_trainer(
    policy_model: nn.Module,
    reward_model: Optional[nn.Module] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> GRPOTrainer:
    """
    创建GRPO训练器的便捷函数

    Args:
        policy_model: 策略模型
        reward_model: 奖励模型 (可选)
        config: 配置字典
        **kwargs: 其他参数

    Returns:
        GRPOTrainer实例

    Example:
        >>> trainer = create_grpo_trainer(model, reward_model)
        >>> stats = trainer.train_step(responses)
    """
    if config is not None:
        grpo_config = GRPOConfig(**config)
    else:
        grpo_config = GRPOConfig()

    return GRPOTrainer(
        policy_model=policy_model,
        reward_model=reward_model,
        config=grpo_config,
        **kwargs
    )


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("GRPO训练器演示")
    print("=" * 70)

    # 创建假模型
    class FakeModel(nn.Module):
        def __init__(self, vocab_size=1000, hidden_size=128):
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

    class FakeRewardModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input_ids, attention_mask=None):
            batch_size = input_ids.size(0)
            # 模拟奖励: 随机但有一定模式
            rewards = torch.randn(batch_size, 1) + torch.arange(batch_size).float().unsqueeze(1) * 0.1
            return {'rewards': rewards}

    policy_model = FakeModel()
    reward_model = FakeRewardModel()

    # 创建GRPO训练器
    print("\n[相对优势模式]")
    trainer = create_grpo_trainer(
        policy_model=policy_model,
        reward_model=reward_model,
        config={
            'group_size': 4,
            'advantage_type': 'relative',
            'learning_rate': 1e-5
        },
        device='cpu'
    )

    # 模拟训练数据
    # 每个组有group_size个响应
    print("\n开始GRPO训练...")
    for step in range(10):
        # 模拟生成的响应 (实际应该由模型生成)
        # 这里我们生成2组，每组4个响应
        batch_size = 8
        seq_len = 20

        responses = torch.randint(0, 1000, (batch_size, seq_len))
        response_masks = torch.ones_like(responses)

        stats = trainer.train_step(responses, response_masks)

        if step % 3 == 0:
            print(f"\nStep {step}:")
            print(f"  Mean Reward: {stats['mean_reward']:.4f}")
            print(f"  Group Variance: {stats['group_variance']:.4f}")
            print(f"  Policy Loss: {stats['policy_loss']:.4f}")
            print(f"  Mean Advantage: {stats['mean_advantage']:.4f}")

    # 测试不同的优势类型
    print("\n" + "=" * 70)
    print("[归一化优势模式]")
    print("=" * 70)

    trainer_norm = create_grpo_trainer(
        policy_model=FakeModel(),
        reward_model=reward_model,
        config={
            'group_size': 4,
            'advantage_type': 'normalized'
        },
        device='cpu'
    )

    print("\n开始训练...")
    for step in range(5):
        responses = torch.randint(0, 1000, (8, 20))
        response_masks = torch.ones_like(responses)

        stats = trainer_norm.train_step(responses, response_masks)

        if step % 2 == 0:
            print(f"\nStep {step}:")
            print(f"  Mean Reward: {stats['mean_reward']:.4f}")
            print(f"  Policy Loss: {stats['policy_loss']:.4f}")

    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    print("\nGRPO优势:")
    print("  ✓ 比PPO更简单")
    print("  ✓ 不需要参考模型")
    print("  ✓ 适合在线学习")
    print("  ✓ 计算效率高")
    print("  ✓ 分组相对优势更稳定")
