#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DPO训练器 (Direct Preference Optimization)

无需奖励模型的直接偏好优化方法

论文: Direct Preference Optimization: Your Language Model is Secretly a Reward Model
链接: https://arxiv.org/abs/2305.18290

作者: chen0430tw
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DPOConfig:
    """DPO配置"""
    beta: float = 0.1  # DPO温度参数
    label_smoothing: float = 0.0  # 标签平滑
    learning_rate: float = 1e-6
    max_grad_norm: float = 1.0
    reference_free: bool = False  # 是否使用无参考模式


class DPOTrainer:
    """
    DPO训练器

    直接从偏好数据优化策略模型，无需训练单独的奖励模型

    核心思想:
    - 最大化选中响应的概率
    - 最小化拒绝响应的概率
    - 使用参考模型防止过拟合
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_policy_model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        config: Optional[DPOConfig] = None,
        device: str = "cuda"
    ):
        """
        初始化DPO训练器

        Args:
            policy_model: 策略模型 (要训练的模型)
            ref_policy_model: 参考策略模型 (固定，用于KL约束)
            optimizer: 优化器
            config: DPO配置
            device: 设备
        """
        self.policy_model = policy_model
        self.ref_policy_model = ref_policy_model
        self.config = config or DPOConfig()
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
        if self.ref_policy_model is not None:
            self.ref_policy_model.to(device)
            self.ref_policy_model.eval()  # 参考模型始终eval模式

        # 统计
        self.stats = {
            'total_steps': 0,
            'mean_chosen_reward': 0.0,
            'mean_rejected_reward': 0.0,
            'accuracy': 0.0
        }

        logger.info(f"[DPO] 训练器初始化完成 (beta={self.config.beta})")

    def get_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        获取序列的对数概率

        Args:
            model: 模型
            input_ids: [batch, seq_len]
            attention_mask: mask
            labels: 标签 (如果None，使用input_ids)

        Returns:
            log_probs: [batch, seq_len]
        """
        if labels is None:
            labels = input_ids

        # 前向传播
        outputs = model(input_ids, attention_mask=attention_mask)

        # 获取logits
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs

        # 计算log_probs
        log_probs = F.log_softmax(logits, dim=-1)

        # 收集对应标签的log_probs
        gathered_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=labels.unsqueeze(-1)
        ).squeeze(-1)

        return gathered_log_probs

    def compute_dpo_loss(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        计算DPO损失

        DPO损失公式:
        L = -log(sigmoid(β * (log π_θ(y_w|x) - log π_θ(y_l|x)
                              - log π_ref(y_w|x) + log π_ref(y_l|x))))

        其中:
        - y_w: 选中的响应 (chosen)
        - y_l: 拒绝的响应 (rejected)
        - π_θ: 策略模型
        - π_ref: 参考模型
        - β: 温度参数

        Args:
            chosen_ids: 选中的响应 [batch, seq_len]
            rejected_ids: 拒绝的响应 [batch, seq_len]
            chosen_mask: mask
            rejected_mask: mask

        Returns:
            (loss, metrics)
        """
        # 获取策略模型的log_probs
        policy_chosen_log_probs = self.get_log_probs(
            self.policy_model, chosen_ids, chosen_mask
        )
        policy_rejected_log_probs = self.get_log_probs(
            self.policy_model, rejected_ids, rejected_mask
        )

        # 对mask求和得到序列总log_prob
        if chosen_mask is not None:
            policy_chosen_log_prob = (policy_chosen_log_probs * chosen_mask).sum(dim=-1)
            policy_rejected_log_prob = (policy_rejected_log_probs * rejected_mask).sum(dim=-1)
        else:
            policy_chosen_log_prob = policy_chosen_log_probs.sum(dim=-1)
            policy_rejected_log_prob = policy_rejected_log_probs.sum(dim=-1)

        # 获取参考模型的log_probs (如果有)
        if self.ref_policy_model is not None and not self.config.reference_free:
            with torch.no_grad():
                ref_chosen_log_probs = self.get_log_probs(
                    self.ref_policy_model, chosen_ids, chosen_mask
                )
                ref_rejected_log_probs = self.get_log_probs(
                    self.ref_policy_model, rejected_ids, rejected_mask
                )

                if chosen_mask is not None:
                    ref_chosen_log_prob = (ref_chosen_log_probs * chosen_mask).sum(dim=-1)
                    ref_rejected_log_prob = (ref_rejected_log_probs * rejected_mask).sum(dim=-1)
                else:
                    ref_chosen_log_prob = ref_chosen_log_probs.sum(dim=-1)
                    ref_rejected_log_prob = ref_rejected_log_probs.sum(dim=-1)
        else:
            # 无参考模式
            ref_chosen_log_prob = 0.0
            ref_rejected_log_prob = 0.0

        # 计算隐式奖励
        policy_logratios = policy_chosen_log_prob - policy_rejected_log_prob
        ref_logratios = ref_chosen_log_prob - ref_rejected_log_prob

        logits = policy_logratios - ref_logratios

        # DPO损失
        losses = -F.logsigmoid(self.config.beta * logits)

        # 标签平滑
        if self.config.label_smoothing > 0:
            smooth_loss = -F.logsigmoid(-self.config.beta * logits)
            losses = (1 - self.config.label_smoothing) * losses + \
                    self.config.label_smoothing * smooth_loss

        loss = losses.mean()

        # 计算准确率 (chosen的奖励 > rejected的奖励)
        with torch.no_grad():
            chosen_rewards = self.config.beta * (policy_chosen_log_prob - ref_chosen_log_prob).detach()
            rejected_rewards = self.config.beta * (policy_rejected_log_prob - ref_rejected_log_prob).detach()
            accuracy = (chosen_rewards > rejected_rewards).float().mean()

        metrics = {
            'chosen_reward': chosen_rewards.mean().item(),
            'rejected_reward': rejected_rewards.mean().item(),
            'reward_margin': (chosen_rewards - rejected_rewards).mean().item(),
            'accuracy': accuracy.item()
        }

        return loss, metrics

    def train_step(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        执行一次训练步骤

        Args:
            chosen_ids: 选中的响应
            rejected_ids: 拒绝的响应
            chosen_mask: mask
            rejected_mask: mask

        Returns:
            统计信息
        """
        self.policy_model.train()

        # 计算损失
        loss, metrics = self.compute_dpo_loss(
            chosen_ids, rejected_ids,
            chosen_mask, rejected_mask
        )

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(),
            self.config.max_grad_norm
        )

        self.optimizer.step()

        # 更新统计
        self.stats['total_steps'] += 1
        self.stats['mean_chosen_reward'] = metrics['chosen_reward']
        self.stats['mean_rejected_reward'] = metrics['rejected_reward']
        self.stats['accuracy'] = metrics['accuracy']

        return {
            'loss': loss.item(),
            **metrics
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()


# ==================== 便捷函数 ====================

def create_dpo_trainer(
    policy_model: nn.Module,
    ref_policy_model: Optional[nn.Module] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> DPOTrainer:
    """
    创建DPO训练器的便捷函数

    Args:
        policy_model: 策略模型
        ref_policy_model: 参考模型 (可选)
        config: 配置字典
        **kwargs: 其他参数

    Returns:
        DPOTrainer实例

    Example:
        >>> trainer = create_dpo_trainer(model)
        >>> stats = trainer.train_step(chosen_ids, rejected_ids)
    """
    if config is not None:
        dpo_config = DPOConfig(**config)
    else:
        dpo_config = DPOConfig()

    return DPOTrainer(
        policy_model=policy_model,
        ref_policy_model=ref_policy_model,
        config=dpo_config,
        **kwargs
    )


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("DPO训练器演示")
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

    policy_model = FakeModel()
    ref_model = FakeModel()

    # 加载相同权重作为参考
    ref_model.load_state_dict(policy_model.state_dict())

    # 创建DPO训练器
    trainer = create_dpo_trainer(
        policy_model=policy_model,
        ref_policy_model=ref_model,
        config={'beta': 0.1, 'learning_rate': 1e-5},
        device='cpu'
    )

    # 模拟训练数据
    print("\n开始DPO训练...")
    chosen_ids = torch.randint(0, 1000, (4, 20))
    rejected_ids = torch.randint(0, 1000, (4, 20))

    for step in range(10):
        stats = trainer.train_step(chosen_ids, rejected_ids)

        if step % 3 == 0:
            print(f"\nStep {step}:")
            print(f"  Loss: {stats['loss']:.4f}")
            print(f"  Accuracy: {stats['accuracy']:.2%}")
            print(f"  Chosen Reward: {stats['chosen_reward']:.4f}")
            print(f"  Rejected Reward: {stats['rejected_reward']:.4f}")
            print(f"  Reward Margin: {stats['reward_margin']:.4f}")

    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    print("\nDPO优势:")
    print("  ✓ 无需单独训练奖励模型")
    print("  ✓ 训练更稳定")
    print("  ✓ 实现更简单")
    print("  ✓ 性能与RLHF相当")
