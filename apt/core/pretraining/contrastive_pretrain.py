#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比学习预训练 (Contrastive Learning Pretraining)

基于SimCLR/MoCo思想的对比学习自监督预训练

核心思想:
- 通过数据增强创建正样本对
- 最大化正样本对的相似度
- 最小化负样本对的相似度
- 使用InfoNCE损失训练

论文参考:
- SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
- MoCo: Momentum Contrast for Unsupervised Visual Representation Learning

作者: chen0430tw
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Callable, Tuple
import logging
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ContrastiveConfig:
    """对比学习配置"""
    # 温度参数
    temperature: float = 0.07  # NT-Xent loss温度

    # 投影头配置
    projection_dim: int = 128  # 投影到的维度
    hidden_dim: int = 2048  # 投影头隐藏层维度
    use_projection: bool = True  # 是否使用投影头

    # 训练参数
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    max_grad_norm: float = 1.0

    # 动量编码器 (MoCo风格)
    use_momentum_encoder: bool = False
    momentum: float = 0.999  # 动量系数
    queue_size: int = 65536  # 负样本队列大小

    # 对称损失
    symmetric_loss: bool = True  # 使用对称版本的损失


class ProjectionHead(nn.Module):
    """
    投影头

    将表示映射到对比学习空间
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 128
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContrastivePretrainer:
    """
    对比学习预训练器

    实现SimCLR/MoCo风格的对比学习预训练

    用法:
        >>> pretrainer = ContrastivePretrainer(encoder_model)
        >>> stats = pretrainer.train_step(batch_x1, batch_x2)
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int = 768,
        optimizer: Optional[torch.optim.Optimizer] = None,
        config: Optional[ContrastiveConfig] = None,
        device: str = "cuda"
    ):
        """
        初始化对比学习预训练器

        Args:
            encoder: 编码器模型 (要训练的模型)
            hidden_size: 编码器输出维度
            optimizer: 优化器
            config: 对比学习配置
            device: 设备
        """
        self.encoder = encoder
        self.config = config or ContrastiveConfig()
        self.device = device
        self.hidden_size = hidden_size

        # 投影头
        if self.config.use_projection:
            self.projection = ProjectionHead(
                input_dim=hidden_size,
                hidden_dim=self.config.hidden_dim,
                output_dim=self.config.projection_dim
            ).to(device)
        else:
            self.projection = None

        # 动量编码器 (MoCo风格)
        if self.config.use_momentum_encoder:
            self.encoder_momentum = self._create_momentum_encoder()
            if self.config.use_projection:
                self.projection_momentum = ProjectionHead(
                    input_dim=hidden_size,
                    hidden_dim=self.config.hidden_dim,
                    output_dim=self.config.projection_dim
                ).to(device)
                self.projection_momentum.load_state_dict(self.projection.state_dict())

            # 负样本队列
            self.register_queue()
        else:
            self.encoder_momentum = None
            self.projection_momentum = None

        # 优化器
        if optimizer is None:
            params = list(self.encoder.parameters())
            if self.projection is not None:
                params += list(self.projection.parameters())

            self.optimizer = torch.optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            self.optimizer = optimizer

        # 移动到设备
        self.encoder.to(device)

        # 统计
        self.stats = {
            'total_steps': 0,
            'mean_loss': 0.0,
            'mean_accuracy': 0.0
        }

        logger.info(f"[ContrastivePretrainer] 初始化完成 (temperature={self.config.temperature})")

    def _create_momentum_encoder(self) -> nn.Module:
        """创建动量编码器"""
        import copy
        encoder_momentum = copy.deepcopy(self.encoder)
        encoder_momentum.to(self.device)

        # 冻结参数
        for param in encoder_momentum.parameters():
            param.requires_grad = False

        return encoder_momentum

    def register_queue(self):
        """注册负样本队列"""
        self.queue = torch.randn(
            self.config.projection_dim,
            self.config.queue_size,
            device=self.device
        )
        self.queue = F.normalize(self.queue, dim=0)
        self.queue_ptr = 0

    @torch.no_grad()
    def _momentum_update(self):
        """更新动量编码器"""
        if not self.config.use_momentum_encoder:
            return

        m = self.config.momentum

        # 更新编码器
        for param_q, param_k in zip(self.encoder.parameters(),
                                     self.encoder_momentum.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)

        # 更新投影头
        if self.projection is not None:
            for param_q, param_k in zip(self.projection.parameters(),
                                         self.projection_momentum.parameters()):
                param_k.data = param_k.data * m + param_q.data * (1. - m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """更新负样本队列"""
        batch_size = keys.shape[0]

        ptr = self.queue_ptr

        # 替换队列中的keys
        if ptr + batch_size <= self.config.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # 循环队列
            remain = self.config.queue_size - ptr
            self.queue[:, ptr:] = keys[:remain].T
            self.queue[:, :batch_size - remain] = keys[remain:].T

        ptr = (ptr + batch_size) % self.config.queue_size
        self.queue_ptr = ptr

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_momentum: bool = False
    ) -> torch.Tensor:
        """
        编码输入序列

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            use_momentum: 是否使用动量编码器

        Returns:
            embeddings: [batch, projection_dim] 或 [batch, hidden_size]
        """
        encoder = self.encoder_momentum if use_momentum else self.encoder
        projection = self.projection_momentum if use_momentum else self.projection

        # 获取编码器输出
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)

        # 获取句子表示 (使用[CLS]或平均池化)
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states'):
            hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = outputs

        # 池化
        if attention_mask is not None:
            # 平均池化
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embeddings = sum_hidden / sum_mask
        else:
            # 使用[CLS] token (第一个token)
            embeddings = hidden_states[:, 0, :]

        # 投影
        if projection is not None:
            embeddings = projection(embeddings)

        # L2归一化
        embeddings = F.normalize(embeddings, dim=-1)

        return embeddings

    def compute_infonce_loss(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        queue: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        计算InfoNCE损失 (NT-Xent)

        L = -log(exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ))

        Args:
            z_i: 第一个视图的表示 [batch, dim]
            z_j: 第二个视图的表示 [batch, dim]
            queue: 负样本队列 (MoCo) [dim, queue_size]

        Returns:
            (loss, accuracy)
        """
        batch_size = z_i.shape[0]

        if queue is None:
            # SimCLR风格：使用batch内的样本作为负样本
            # 拼接两个视图
            z = torch.cat([z_i, z_j], dim=0)  # [2*batch, dim]

            # 计算相似度矩阵
            sim_matrix = torch.matmul(z, z.T) / self.config.temperature  # [2*batch, 2*batch]

            # 创建标签: 每个样本的正样本是另一个视图中的对应样本
            labels = torch.arange(batch_size, device=self.device)
            labels = torch.cat([labels + batch_size, labels], dim=0)

            # 移除自己和自己的相似度
            mask = torch.eye(2 * batch_size, device=self.device).bool()
            sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

            # 计算损失
            loss = F.cross_entropy(sim_matrix, labels)

            # 计算准确率
            with torch.no_grad():
                pred = sim_matrix.argmax(dim=1)
                accuracy = (pred == labels).float().mean().item()

        else:
            # MoCo风格：使用队列中的样本作为负样本
            # 正样本对的相似度
            pos_sim = torch.sum(z_i * z_j, dim=-1, keepdim=True) / self.config.temperature  # [batch, 1]

            # 负样本对的相似度
            neg_sim = torch.matmul(z_i, queue) / self.config.temperature  # [batch, queue_size]

            # 拼接
            logits = torch.cat([pos_sim, neg_sim], dim=1)  # [batch, 1 + queue_size]

            # 标签为0 (正样本在第一个位置)
            labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)

            # 计算损失
            loss = F.cross_entropy(logits, labels)

            # 计算准确率
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                accuracy = (pred == labels).float().mean().item()

        return loss, accuracy

    def train_step(
        self,
        input_ids_1: torch.Tensor,
        input_ids_2: torch.Tensor,
        attention_mask_1: Optional[torch.Tensor] = None,
        attention_mask_2: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        执行一次训练步骤

        Args:
            input_ids_1: 第一个增强视图 [batch, seq_len]
            input_ids_2: 第二个增强视图 [batch, seq_len]
            attention_mask_1: 第一个视图的mask
            attention_mask_2: 第二个视图的mask

        Returns:
            统计信息
        """
        self.encoder.train()
        if self.projection is not None:
            self.projection.train()

        # 编码两个视图
        z_i = self.encode(input_ids_1, attention_mask_1, use_momentum=False)

        if self.config.use_momentum_encoder:
            # MoCo: 使用动量编码器编码第二个视图
            with torch.no_grad():
                z_j = self.encode(input_ids_2, attention_mask_2, use_momentum=True)
            queue = self.queue
        else:
            # SimCLR: 使用相同编码器
            z_j = self.encode(input_ids_2, attention_mask_2, use_momentum=False)
            queue = None

        # 计算损失
        loss, accuracy = self.compute_infonce_loss(z_i, z_j, queue)

        # 对称损失 (可选)
        if self.config.symmetric_loss and not self.config.use_momentum_encoder:
            loss_sym, accuracy_sym = self.compute_infonce_loss(z_j, z_i, queue)
            loss = (loss + loss_sym) / 2
            accuracy = (accuracy + accuracy_sym) / 2

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.encoder.parameters(),
            self.config.max_grad_norm
        )
        if self.projection is not None:
            torch.nn.utils.clip_grad_norm_(
                self.projection.parameters(),
                self.config.max_grad_norm
            )

        self.optimizer.step()

        # 更新动量编码器
        if self.config.use_momentum_encoder:
            self._momentum_update()
            # 更新队列
            self._dequeue_and_enqueue(z_j)

        # 更新统计
        self.stats['total_steps'] += 1
        self.stats['mean_loss'] = loss.item()
        self.stats['mean_accuracy'] = accuracy

        return {
            'loss': loss.item(),
            'accuracy': accuracy
        }

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()


# ==================== 数据增强 ====================

class TextAugmentation:
    """
    文本数据增强

    用于生成对比学习的正样本对
    """

    @staticmethod
    def random_mask(
        input_ids: torch.Tensor,
        mask_token_id: int,
        mask_prob: float = 0.15
    ) -> torch.Tensor:
        """随机mask一些token"""
        masked_input = input_ids.clone()
        mask = torch.rand(input_ids.shape, device=input_ids.device) < mask_prob
        masked_input[mask] = mask_token_id
        return masked_input

    @staticmethod
    def random_delete(
        input_ids: torch.Tensor,
        delete_prob: float = 0.1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """随机删除一些token"""
        keep_mask = torch.rand(input_ids.shape, device=input_ids.device) > delete_prob
        # 至少保留一个token
        keep_mask[:, 0] = True

        # 删除tokens
        batch_size = input_ids.size(0)
        max_len = keep_mask.sum(dim=1).max().item()

        output_ids = torch.zeros(batch_size, max_len, dtype=input_ids.dtype, device=input_ids.device)
        attention_mask = torch.zeros(batch_size, max_len, device=input_ids.device)

        for i in range(batch_size):
            kept_tokens = input_ids[i][keep_mask[i]]
            output_ids[i, :len(kept_tokens)] = kept_tokens
            attention_mask[i, :len(kept_tokens)] = 1

        return output_ids, attention_mask

    @staticmethod
    def random_swap(
        input_ids: torch.Tensor,
        swap_prob: float = 0.1
    ) -> torch.Tensor:
        """随机交换相邻token"""
        swapped_input = input_ids.clone()
        seq_len = input_ids.size(1)

        for i in range(1, seq_len - 1):
            if torch.rand(1).item() < swap_prob:
                # 交换i和i+1
                swapped_input[:, i], swapped_input[:, i+1] = \
                    swapped_input[:, i+1].clone(), swapped_input[:, i].clone()

        return swapped_input


# ==================== 便捷函数 ====================

def create_contrastive_pretrainer(
    encoder: nn.Module,
    hidden_size: int = 768,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> ContrastivePretrainer:
    """
    创建对比学习预训练器的便捷函数

    Args:
        encoder: 编码器模型
        hidden_size: 编码器输出维度
        config: 配置字典
        **kwargs: 其他参数

    Returns:
        ContrastivePretrainer实例

    Example:
        >>> pretrainer = create_contrastive_pretrainer(model, hidden_size=768)
        >>> stats = pretrainer.train_step(x1, x2)
    """
    if config is not None:
        contrastive_config = ContrastiveConfig(**config)
    else:
        contrastive_config = ContrastiveConfig()

    return ContrastivePretrainer(
        encoder=encoder,
        hidden_size=hidden_size,
        config=contrastive_config,
        **kwargs
    )


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("对比学习预训练演示")
    print("=" * 70)

    # 创建假的编码器模型
    class FakeEncoder(nn.Module):
        def __init__(self, vocab_size=1000, hidden_size=128):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)

        def forward(self, input_ids, attention_mask=None):
            hidden = self.embedding(input_ids)

            class Output:
                def __init__(self, hidden):
                    self.last_hidden_state = hidden

            return Output(hidden)

    encoder = FakeEncoder(vocab_size=1000, hidden_size=128)

    # 创建对比学习预训练器
    print("\n[SimCLR风格]")
    pretrainer = create_contrastive_pretrainer(
        encoder=encoder,
        hidden_size=128,
        config={
            'temperature': 0.07,
            'projection_dim': 64,
            'use_momentum_encoder': False
        },
        device='cpu'
    )

    # 模拟训练数据
    batch_size = 8
    seq_len = 20

    print("\n开始训练...")
    for step in range(10):
        # 生成两个增强视图 (实际应该是数据增强)
        x1 = torch.randint(0, 1000, (batch_size, seq_len))
        x2 = torch.randint(0, 1000, (batch_size, seq_len))

        stats = pretrainer.train_step(x1, x2)

        if step % 3 == 0:
            print(f"\nStep {step}:")
            print(f"  Loss: {stats['loss']:.4f}")
            print(f"  Accuracy: {stats['accuracy']:.2%}")

    # MoCo风格
    print("\n" + "=" * 70)
    print("[MoCo风格]")
    print("=" * 70)

    encoder2 = FakeEncoder(vocab_size=1000, hidden_size=128)
    pretrainer_moco = create_contrastive_pretrainer(
        encoder=encoder2,
        hidden_size=128,
        config={
            'temperature': 0.07,
            'projection_dim': 64,
            'use_momentum_encoder': True,
            'queue_size': 256
        },
        device='cpu'
    )

    print("\n开始训练...")
    for step in range(10):
        x1 = torch.randint(0, 1000, (batch_size, seq_len))
        x2 = torch.randint(0, 1000, (batch_size, seq_len))

        stats = pretrainer_moco.train_step(x1, x2)

        if step % 3 == 0:
            print(f"\nStep {step}:")
            print(f"  Loss: {stats['loss']:.4f}")
            print(f"  Accuracy: {stats['accuracy']:.2%}")

    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    print("\n对比学习优势:")
    print("  ✓ 无需标注数据")
    print("  ✓ 学习通用表示")
    print("  ✓ 提升下游任务性能")
    print("  ✓ 支持SimCLR/MoCo等方法")
