#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RLHF训练器 (Reinforcement Learning from Human Feedback)

基于PPO (Proximal Policy Optimization) 的RLHF实现

作者: chen0430tw
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RLHFConfig:
    """RLHF配置"""
    # PPO参数
    ppo_epochs: int = 4  # PPO内部训练轮数
    clip_epsilon: float = 0.2  # PPO裁剪参数
    value_loss_coef: float = 0.1  # 价值损失系数
    entropy_coef: float = 0.01  # 熵正则化系数

    # 奖励相关
    kl_coef: float = 0.1  # KL散度惩罚系数
    reward_scale: float = 1.0  # 奖励缩放

    # 训练参数
    batch_size: int = 4
    max_length: int = 512
    learning_rate: float = 1e-5

    # 其他
    gamma: float = 0.99  # 折扣因子
    gae_lambda: float = 0.95  # GAE参数
    max_grad_norm: float = 1.0  # 梯度裁剪


class RLHFTrainer:
    """
    RLHF训练器

    实现基于PPO的RLHF训练流程:
    1. 使用策略模型生成响应
    2. 使用奖励模型评分
    3. 计算优势函数
    4. 使用PPO更新策略
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reward_model: nn.Module,
        ref_policy_model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        config: Optional[RLHFConfig] = None,
        device: str = "cuda"
    ):
        """
        初始化RLHF训练器

        Args:
            policy_model: 策略模型 (要训练的模型)
            reward_model: 奖励模型 (固定)
            ref_policy_model: 参考策略模型 (用于KL惩罚，可选)
            optimizer: 优化器
            config: RLHF配置
            device: 设备
        """
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.ref_policy_model = ref_policy_model or policy_model  # 默认使用自己作为参考
        self.config = config or RLHFConfig()
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
        self.reward_model.to(device)
        self.ref_policy_model.to(device)

        # 统计
        self.stats = {
            'total_steps': 0,
            'total_episodes': 0,
            'mean_reward': 0.0,
            'mean_kl': 0.0
        }

        logger.info("[RLHF] 训练器初始化完成")

    def generate_responses(
        self,
        prompts: torch.Tensor,
        prompt_masks: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        使用策略模型生成响应

        Args:
            prompts: 提示 [batch, prompt_len]
            prompt_masks: 提示mask
            max_new_tokens: 最大生成长度
            temperature: 温度

        Returns:
            (responses, response_masks, log_probs)
        """
        self.policy_model.eval()

        with torch.no_grad():
            # 简化实现：这里应该使用实际的生成逻辑
            # 为了演示，我们生成随机的token ids和log_probs
            batch_size = prompts.size(0)

            # 假设生成
            generated_ids = torch.randint(
                0, 50000,
                (batch_size, max_new_tokens),
                device=self.device
            )

            # 合并prompt和generated
            responses = torch.cat([prompts, generated_ids], dim=1)

            # 生成masks
            response_masks = torch.ones_like(responses)

            # 生成log_probs (这里是占位符)
            log_probs = torch.randn(batch_size, responses.size(1), device=self.device)

        return responses, response_masks, log_probs

    def compute_rewards(
        self,
        responses: torch.Tensor,
        response_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        使用奖励模型计算奖励

        Args:
            responses: 响应 [batch, seq_len]
            response_masks: mask

        Returns:
            rewards: [batch, seq_len]
        """
        self.reward_model.eval()

        with torch.no_grad():
            reward_output = self.reward_model(responses, response_masks)
            rewards = reward_output['rewards']  # [batch, 1]

            # 扩展到整个序列
            rewards_per_token = rewards.unsqueeze(1).expand(-1, responses.size(1))

        return rewards_per_token

    def compute_kl_penalty(
        self,
        responses: torch.Tensor,
        response_masks: torch.Tensor,
        policy_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        计算KL散度惩罚

        KL(policy || ref_policy)

        Args:
            responses: 响应
            response_masks: mask
            policy_log_probs: 策略的log概率

        Returns:
            kl_divergence: [batch, seq_len]
        """
        self.ref_policy_model.eval()

        with torch.no_grad():
            # 获取参考策略的log_probs
            # 简化实现：实际需要通过模型前向传播获取
            ref_log_probs = policy_log_probs - torch.randn_like(policy_log_probs) * 0.1

        # KL散度: E[log(policy) - log(ref)]
        kl_div = policy_log_probs - ref_log_probs

        return kl_div

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算GAE优势函数

        Args:
            rewards: [batch, seq_len]
            values: [batch, seq_len]
            masks: [batch, seq_len]

        Returns:
            (advantages, returns)
        """
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)

        # 计算returns和advantages
        gae = 0
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = 0
            else:
                next_value = values[:, t + 1]

            delta = rewards[:, t] + self.config.gamma * next_value - values[:, t]
            gae = delta + self.config.gamma * self.config.gae_lambda * gae * masks[:, t]

            advantages[:, t] = gae
            returns[:, t] = advantages[:, t] + values[:, t]

        # 归一化advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def ppo_step(
        self,
        responses: torch.Tensor,
        response_masks: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> Dict[str, float]:
        """
        执行一次PPO更新

        Args:
            responses: 响应
            response_masks: mask
            old_log_probs: 旧的log概率
            advantages: 优势函数
            returns: 回报

        Returns:
            统计信息
        """
        self.policy_model.train()

        # 前向传播获取当前log_probs和values
        # 简化实现：实际需要通过模型获取
        current_log_probs = old_log_probs + torch.randn_like(old_log_probs) * 0.01
        values = torch.randn_like(returns)

        # 计算概率比
        ratio = torch.exp(current_log_probs - old_log_probs)

        # PPO裁剪损失
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1.0 - self.config.clip_epsilon,
            1.0 + self.config.clip_epsilon
        ) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()

        # 价值损失
        value_loss = F.mse_loss(values, returns)

        # 熵正则化 (鼓励探索)
        # 简化：使用log_probs的方差作为熵的代理
        entropy = -(current_log_probs * torch.exp(current_log_probs)).sum(dim=-1).mean()

        # 总损失
        loss = (
            policy_loss +
            self.config.value_loss_coef * value_loss -
            self.config.entropy_coef * entropy
        )

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(),
            self.config.max_grad_norm
        )
        self.optimizer.step()

        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }

    def train_step(
        self,
        prompts: torch.Tensor,
        prompt_masks: torch.Tensor
    ) -> Dict[str, Any]:
        """
        完整的RLHF训练步骤

        1. 生成响应
        2. 计算奖励
        3. 计算KL惩罚
        4. 计算优势
        5. PPO更新

        Args:
            prompts: 提示
            prompt_masks: mask

        Returns:
            统计信息
        """
        # 1. 生成响应
        responses, response_masks, old_log_probs = self.generate_responses(
            prompts, prompt_masks
        )

        # 2. 计算奖励
        rewards = self.compute_rewards(responses, response_masks)

        # 3. KL惩罚
        kl_penalty = self.compute_kl_penalty(responses, response_masks, old_log_probs)
        rewards_with_kl = rewards - self.config.kl_coef * kl_penalty

        # 4. 计算values (简化: 使用随机值)
        values = torch.randn_like(rewards_with_kl)

        # 5. 计算优势
        advantages, returns = self.compute_advantages(
            rewards_with_kl, values, response_masks
        )

        # 6. PPO更新 (多个epoch)
        ppo_stats = []
        for epoch in range(self.config.ppo_epochs):
            stats = self.ppo_step(
                responses, response_masks, old_log_probs,
                advantages, returns
            )
            ppo_stats.append(stats)

        # 统计
        mean_reward = rewards.mean().item()
        mean_kl = kl_penalty.mean().item()

        self.stats['total_steps'] += 1
        self.stats['mean_reward'] = mean_reward
        self.stats['mean_kl'] = mean_kl

        return {
            'mean_reward': mean_reward,
            'mean_kl': mean_kl,
            'ppo_loss': sum(s['loss'] for s in ppo_stats) / len(ppo_stats),
            'policy_loss': sum(s['policy_loss'] for s in ppo_stats) / len(ppo_stats),
            'value_loss': sum(s['value_loss'] for s in ppo_stats) / len(ppo_stats),
            'entropy': sum(s['entropy'] for s in ppo_stats) / len(ppo_stats),
        }


# ==================== 便捷函数 ====================

def create_rlhf_trainer(
    policy_model: nn.Module,
    reward_model: nn.Module,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> RLHFTrainer:
    """
    创建RLHF训练器的便捷函数

    Args:
        policy_model: 策略模型
        reward_model: 奖励模型
        config: 配置字典
        **kwargs: 其他参数

    Returns:
        RLHFTrainer实例

    Example:
        >>> trainer = create_rlhf_trainer(policy_model, reward_model)
        >>> stats = trainer.train_step(prompts, prompt_masks)
    """
    if config is not None:
        rlhf_config = RLHFConfig(**config)
    else:
        rlhf_config = RLHFConfig()

    return RLHFTrainer(
        policy_model=policy_model,
        reward_model=reward_model,
        config=rlhf_config,
        **kwargs
    )


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("RLHF训练器演示")
    print("=" * 70)

    # 创建假模型
    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

        def forward(self, input_ids, attention_mask=None):
            return None

    class FakeRewardModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, input_ids, attention_mask=None):
            batch_size = input_ids.size(0)
            return {'rewards': torch.randn(batch_size, 1)}

    policy_model = FakeModel()
    reward_model = FakeRewardModel()

    # 创建RLHF训练器
    trainer = create_rlhf_trainer(
        policy_model=policy_model,
        reward_model=reward_model,
        config={'ppo_epochs': 2, 'kl_coef': 0.1},
        device='cpu'
    )

    # 模拟训练
    print("\n开始RLHF训练...")
    prompts = torch.randint(0, 1000, (4, 20))
    prompt_masks = torch.ones_like(prompts)

    for step in range(5):
        stats = trainer.train_step(prompts, prompt_masks)

        print(f"\nStep {step}:")
        print(f"  奖励: {stats['mean_reward']:.4f}")
        print(f"  KL: {stats['mean_kl']:.4f}")
        print(f"  PPO Loss: {stats['ppo_loss']:.4f}")
        print(f"  熵: {stats['entropy']:.4f}")

    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
