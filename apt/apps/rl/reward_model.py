#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
奖励模型 (Reward Model)

用于强化学习中的奖励预测，特别是RLHF (Reinforcement Learning from Human Feedback)

作者: chen0430tw
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RewardModel(nn.Module):
    """
    奖励模型

    基于预训练语言模型，添加价值头来预测奖励分数

    用途:
    - RLHF中的奖励信号
    - 偏好学习
    - 响应质量评分
    """

    def __init__(
        self,
        base_model: nn.Module,
        hidden_size: int = 768,
        num_layers: int = 1,
        dropout: float = 0.1,
        use_pooling: str = "last"  # "last", "mean", "max"
    ):
        """
        初始化奖励模型

        Args:
            base_model: 基础语言模型 (如APT, GPT等)
            hidden_size: 隐藏层大小
            num_layers: 价值头的层数
            dropout: Dropout率
            use_pooling: 池化方式 ("last", "mean", "max")
        """
        super().__init__()
        self.base_model = base_model
        self.hidden_size = hidden_size
        self.use_pooling = use_pooling

        # 价值头 (Value Head)
        layers = []
        input_size = hidden_size

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        # 最后一层输出标量奖励
        layers.append(nn.Linear(input_size if num_layers == 1 else hidden_size, 1))

        self.value_head = nn.Sequential(*layers)

        logger.info(f"[RewardModel] 初始化完成 (pooling={use_pooling}, layers={num_layers})")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            return_hidden_states: 是否返回隐藏状态

        Returns:
            字典包含:
                - rewards: [batch_size, 1] 奖励分数
                - hidden_states: (可选) 隐藏状态
        """
        # 获取基础模型输出
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # 获取隐藏状态
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden]
        elif hasattr(outputs, 'hidden_states'):
            hidden_states = outputs.hidden_states[-1]
        else:
            raise ValueError("基础模型输出不包含hidden_states")

        # 池化
        if self.use_pooling == "last":
            # 使用最后一个token的表示
            if attention_mask is not None:
                # 找到每个序列的最后一个有效token
                sequence_lengths = attention_mask.sum(dim=1) - 1
                pooled = hidden_states[torch.arange(hidden_states.size(0)), sequence_lengths]
            else:
                pooled = hidden_states[:, -1, :]

        elif self.use_pooling == "mean":
            # 平均池化
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_hidden / sum_mask
            else:
                pooled = hidden_states.mean(dim=1)

        elif self.use_pooling == "max":
            # 最大池化
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                hidden_states = hidden_states.masked_fill(mask_expanded == 0, float('-inf'))
            pooled = hidden_states.max(dim=1)[0]

        else:
            raise ValueError(f"不支持的池化方式: {self.use_pooling}")

        # 通过价值头得到奖励
        rewards = self.value_head(pooled)  # [batch, 1]

        result = {'rewards': rewards}
        if return_hidden_states:
            result['hidden_states'] = hidden_states

        return result

    def get_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        便捷函数：直接获取奖励分数

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            rewards: [batch_size, 1]
        """
        return self.forward(input_ids, attention_mask)['rewards']

    def compare_responses(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        比较两个响应的奖励

        Args:
            chosen_ids: 选中的响应 [batch, seq_len]
            rejected_ids: 拒绝的响应 [batch, seq_len]
            chosen_mask: 选中响应的mask
            rejected_mask: 拒绝响应的mask

        Returns:
            (chosen_rewards, rejected_rewards)
        """
        chosen_rewards = self.get_rewards(chosen_ids, chosen_mask)
        rejected_rewards = self.get_rewards(rejected_ids, rejected_mask)

        return chosen_rewards, rejected_rewards


class RewardModelTrainer:
    """
    奖励模型训练器

    用于从人类偏好数据训练奖励模型
    """

    def __init__(
        self,
        reward_model: RewardModel,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        margin: float = 0.0
    ):
        """
        初始化训练器

        Args:
            reward_model: 奖励模型
            optimizer: 优化器
            device: 设备
            margin: 对比损失的边界
        """
        self.model = reward_model
        self.optimizer = optimizer
        self.device = device
        self.margin = margin

        self.model.to(device)

    def compute_loss(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算对比损失

        使用Bradley-Terry模型: P(chosen > rejected) = sigmoid(r_chosen - r_rejected)

        Args:
            chosen_ids: 选中的响应
            rejected_ids: 拒绝的响应
            chosen_mask: mask
            rejected_mask: mask

        Returns:
            loss
        """
        # 获取奖励
        chosen_rewards, rejected_rewards = self.model.compare_responses(
            chosen_ids, rejected_ids, chosen_mask, rejected_mask
        )

        # Bradley-Terry损失
        # 我们希望 r_chosen > r_rejected
        logits = chosen_rewards - rejected_rewards - self.margin
        loss = -F.logsigmoid(logits).mean()

        return loss

    def train_step(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        训练一步

        Args:
            chosen_ids: 选中的响应
            rejected_ids: 拒绝的响应
            chosen_mask: mask
            rejected_mask: mask

        Returns:
            统计信息
        """
        self.model.train()
        self.optimizer.zero_grad()

        # 计算损失
        loss = self.compute_loss(chosen_ids, rejected_ids, chosen_mask, rejected_mask)

        # 反向传播
        loss.backward()
        self.optimizer.step()

        # 统计
        with torch.no_grad():
            chosen_rewards, rejected_rewards = self.model.compare_responses(
                chosen_ids, rejected_ids, chosen_mask, rejected_mask
            )

            accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'chosen_reward_mean': chosen_rewards.mean().item(),
            'rejected_reward_mean': rejected_rewards.mean().item(),
            'reward_margin': (chosen_rewards - rejected_rewards).mean().item()
        }


# ==================== 便捷函数 ====================

def create_reward_model(
    base_model: nn.Module,
    hidden_size: int = 768,
    **kwargs
) -> RewardModel:
    """
    创建奖励模型的便捷函数

    Args:
        base_model: 基础语言模型
        hidden_size: 隐藏层大小
        **kwargs: 其他参数

    Returns:
        RewardModel实例

    Example:
        >>> from transformers import AutoModel
        >>> base_model = AutoModel.from_pretrained("gpt2")
        >>> reward_model = create_reward_model(base_model)
    """
    return RewardModel(base_model, hidden_size, **kwargs)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("奖励模型演示")
    print("=" * 70)

    # 创建假的基础模型
    class FakeBaseModel(nn.Module):
        def __init__(self, hidden_size=768):
            super().__init__()
            self.embedding = nn.Embedding(1000, hidden_size)

        def forward(self, input_ids, attention_mask=None, output_hidden_states=True):
            hidden = self.embedding(input_ids)

            class Output:
                def __init__(self, hidden):
                    self.last_hidden_state = hidden
                    self.hidden_states = (hidden,)

            return Output(hidden)

    # 创建奖励模型
    base_model = FakeBaseModel(hidden_size=128)
    reward_model = create_reward_model(
        base_model=base_model,
        hidden_size=128,
        num_layers=2
    )

    # 模拟数据
    chosen_ids = torch.randint(0, 1000, (4, 20))
    rejected_ids = torch.randint(0, 1000, (4, 20))

    # 比较响应
    chosen_rewards, rejected_rewards = reward_model.compare_responses(
        chosen_ids, rejected_ids
    )

    print(f"\n选中响应奖励: {chosen_rewards.squeeze()}")
    print(f"拒绝响应奖励: {rejected_rewards.squeeze()}")
    print(f"奖励差: {(chosen_rewards - rejected_rewards).squeeze()}")

    # 训练
    print("\n训练奖励模型...")
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-4)
    trainer = RewardModelTrainer(reward_model, optimizer, device="cpu")

    for step in range(10):
        stats = trainer.train_step(chosen_ids, rejected_ids)

        if step % 3 == 0:
            print(f"Step {step}: loss={stats['loss']:.4f}, "
                  f"acc={stats['accuracy']:.2%}, "
                  f"margin={stats['reward_margin']:.4f}")

    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
