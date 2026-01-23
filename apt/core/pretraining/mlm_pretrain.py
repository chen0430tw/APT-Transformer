#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MLM预训练 (Masked Language Modeling Pretraining)

基于BERT的遮蔽语言模型预训练

核心思想:
- 随机遮蔽输入序列中的一些token
- 模型预测被遮蔽的token
- 通过自监督学习学习语言表示

论文参考:
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- RoBERTa: A Robustly Optimized BERT Pretraining Approach

作者: chen0430tw
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import logging
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)


@dataclass
class MLMConfig:
    """MLM配置"""
    # 遮蔽策略
    mask_prob: float = 0.15  # 遮蔽概率
    mask_token_prob: float = 0.8  # 被[MASK]替换的概率
    random_token_prob: float = 0.1  # 被随机token替换的概率
    keep_token_prob: float = 0.1  # 保持不变的概率

    # 训练参数
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 10000

    # 词表
    vocab_size: int = 50000
    mask_token_id: int = 4  # [MASK] token的ID
    pad_token_id: int = 0  # [PAD] token的ID
    cls_token_id: int = 2  # [CLS] token的ID
    sep_token_id: int = 3  # [SEP] token的ID

    # 是否使用NSP (Next Sentence Prediction)
    use_nsp: bool = False  # NSP任务 (BERT原始有，RoBERTa移除了)

    # 损失权重
    mlm_loss_weight: float = 1.0
    nsp_loss_weight: float = 1.0


class MLMHead(nn.Module):
    """
    MLM预测头

    从隐藏状态预测被遮蔽的token
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        hidden_act: str = "gelu"
    ):
        super().__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

        if hidden_act == "gelu":
            self.activation = nn.GELU()
        elif hidden_act == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()

        self.decoder = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.layer_norm(x)
        logits = self.decoder(x)

        return logits


class NSPHead(nn.Module):
    """
    NSP (Next Sentence Prediction) 预测头

    二分类任务: 判断两个句子是否连续
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled_output: [batch, hidden_size] (通常是[CLS]的表示)

        Returns:
            logits: [batch, 2]
        """
        return self.classifier(pooled_output)


class MLMPretrainer:
    """
    MLM预训练器

    实现BERT风格的遮蔽语言模型预训练

    用法:
        >>> pretrainer = MLMPretrainer(model, vocab_size=50000)
        >>> stats = pretrainer.train_step(input_ids)
    """

    def __init__(
        self,
        model: nn.Module,
        hidden_size: int = 768,
        optimizer: Optional[torch.optim.Optimizer] = None,
        config: Optional[MLMConfig] = None,
        device: str = "cuda"
    ):
        """
        初始化MLM预训练器

        Args:
            model: 语言模型 (要训练的模型)
            hidden_size: 模型隐藏层大小
            optimizer: 优化器
            config: MLM配置
            device: 设备
        """
        self.model = model
        self.config = config or MLMConfig()
        self.device = device
        self.hidden_size = hidden_size

        # MLM预测头
        self.mlm_head = MLMHead(
            hidden_size=hidden_size,
            vocab_size=self.config.vocab_size
        ).to(device)

        # NSP预测头 (可选)
        if self.config.use_nsp:
            self.nsp_head = NSPHead(hidden_size=hidden_size).to(device)
        else:
            self.nsp_head = None

        # 优化器
        if optimizer is None:
            params = list(self.model.parameters()) + list(self.mlm_head.parameters())
            if self.nsp_head is not None:
                params += list(self.nsp_head.parameters())

            self.optimizer = torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            self.optimizer = optimizer

        # 学习率调度器 (带warmup)
        self.scheduler = None
        if self.config.warmup_steps > 0:
            from torch.optim.lr_scheduler import LambdaLR

            def lr_lambda(current_step):
                if current_step < self.config.warmup_steps:
                    return float(current_step) / float(max(1, self.config.warmup_steps))
                return 1.0

            self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        # 移动到设备
        self.model.to(device)

        # 统计
        self.stats = {
            'total_steps': 0,
            'mean_mlm_loss': 0.0,
            'mean_mlm_accuracy': 0.0,
            'mean_nsp_loss': 0.0,
            'mean_nsp_accuracy': 0.0
        }

        logger.info(f"[MLMPretrainer] 初始化完成 (mask_prob={self.config.mask_prob})")

    def create_mlm_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        创建MLM遮蔽

        策略 (BERT):
        - 15%的token会被选中
        - 其中80%替换为[MASK]
        - 10%替换为随机token
        - 10%保持不变

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            (masked_input_ids, labels, mask_positions)
            - masked_input_ids: 遮蔽后的输入
            - labels: 原始token (用于计算损失)
            - mask_positions: 被遮蔽的位置
        """
        batch_size, seq_len = input_ids.shape

        # 复制输入
        masked_input = input_ids.clone()
        labels = input_ids.clone()

        # 创建遮蔽mask
        probability_matrix = torch.full(input_ids.shape, self.config.mask_prob, device=self.device)

        # 不遮蔽特殊token
        special_tokens_mask = (
            (input_ids == self.config.pad_token_id) |
            (input_ids == self.config.cls_token_id) |
            (input_ids == self.config.sep_token_id)
        )
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # 如果有attention_mask，不遮蔽padding位置
        if attention_mask is not None:
            probability_matrix.masked_fill_(attention_mask == 0, value=0.0)

        # 随机选择要遮蔽的token
        mask_positions = torch.bernoulli(probability_matrix).bool()

        # 不遮蔽的位置，label设为-100 (忽略)
        labels[~mask_positions] = -100

        # 80%的mask位置替换为[MASK]
        indices_replaced = torch.bernoulli(
            torch.full(input_ids.shape, self.config.mask_token_prob, device=self.device)
        ).bool() & mask_positions
        masked_input[indices_replaced] = self.config.mask_token_id

        # 10%的mask位置替换为随机token
        indices_random = torch.bernoulli(
            torch.full(input_ids.shape, self.config.random_token_prob / (1 - self.config.mask_token_prob), device=self.device)
        ).bool() & mask_positions & ~indices_replaced
        random_tokens = torch.randint(
            low=0,
            high=self.config.vocab_size,
            size=input_ids.shape,
            dtype=input_ids.dtype,
            device=self.device
        )
        masked_input[indices_random] = random_tokens[indices_random]

        # 剩余10%保持不变

        return masked_input, labels, mask_positions

    def compute_mlm_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        计算MLM损失

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            labels: [batch, seq_len] (被遮蔽位置的真实token)

        Returns:
            (loss, accuracy)
        """
        # 预测
        logits = self.mlm_head(hidden_states)  # [batch, seq_len, vocab_size]

        # 计算损失 (自动忽略label=-100的位置)
        loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            labels.view(-1),
            ignore_index=-100
        )

        # 计算准确率
        with torch.no_grad():
            mask = labels != -100
            if mask.sum() > 0:
                pred = logits.argmax(dim=-1)
                accuracy = (pred[mask] == labels[mask]).float().mean().item()
            else:
                accuracy = 0.0

        return loss, accuracy

    def compute_nsp_loss(
        self,
        pooled_output: torch.Tensor,
        nsp_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        计算NSP损失

        Args:
            pooled_output: [batch, hidden_size]
            nsp_labels: [batch] (0=连续, 1=不连续)

        Returns:
            (loss, accuracy)
        """
        logits = self.nsp_head(pooled_output)  # [batch, 2]

        loss = F.cross_entropy(logits, nsp_labels)

        # 计算准确率
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            accuracy = (pred == nsp_labels).float().mean().item()

        return loss, accuracy

    def train_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        nsp_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        执行一次训练步骤

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            nsp_labels: [batch] (如果use_nsp=True)

        Returns:
            统计信息
        """
        self.model.train()
        self.mlm_head.train()
        if self.nsp_head is not None:
            self.nsp_head.train()

        # 创建MLM遮蔽
        masked_input_ids, labels, mask_positions = self.create_mlm_mask(input_ids, attention_mask)

        # 前向传播
        outputs = self.model(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # 获取隐藏状态
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states'):
            hidden_states = outputs.hidden_states[-1]
        else:
            hidden_states = outputs

        # MLM损失
        mlm_loss, mlm_accuracy = self.compute_mlm_loss(hidden_states, labels)
        total_loss = self.config.mlm_loss_weight * mlm_loss

        # NSP损失 (可选)
        nsp_loss = torch.tensor(0.0, device=self.device)
        nsp_accuracy = 0.0

        if self.config.use_nsp and nsp_labels is not None:
            # 获取[CLS]的表示作为句子表示
            pooled_output = hidden_states[:, 0, :]  # [batch, hidden_size]

            nsp_loss, nsp_accuracy = self.compute_nsp_loss(pooled_output, nsp_labels)
            total_loss += self.config.nsp_loss_weight * nsp_loss

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()

        # 梯度裁剪
        all_params = list(self.model.parameters()) + list(self.mlm_head.parameters())
        if self.nsp_head is not None:
            all_params += list(self.nsp_head.parameters())

        torch.nn.utils.clip_grad_norm_(all_params, self.config.max_grad_norm)

        self.optimizer.step()

        # 更新学习率
        if self.scheduler is not None:
            self.scheduler.step()

        # 更新统计
        self.stats['total_steps'] += 1
        self.stats['mean_mlm_loss'] = mlm_loss.item()
        self.stats['mean_mlm_accuracy'] = mlm_accuracy
        self.stats['mean_nsp_loss'] = nsp_loss.item() if isinstance(nsp_loss, torch.Tensor) else 0.0
        self.stats['mean_nsp_accuracy'] = nsp_accuracy

        result = {
            'total_loss': total_loss.item(),
            'mlm_loss': mlm_loss.item(),
            'mlm_accuracy': mlm_accuracy,
            'num_masked': mask_positions.sum().item()
        }

        if self.config.use_nsp:
            result['nsp_loss'] = nsp_loss.item()
            result['nsp_accuracy'] = nsp_accuracy

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()


# ==================== 数据处理 ====================

class NSPDataBuilder:
    """
    构建NSP训练数据

    从句子对创建NSP任务数据
    """

    @staticmethod
    def create_nsp_pairs(
        sentences: list,
        tokenizer,
        max_length: int = 512,
        negative_ratio: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        创建NSP句子对

        Args:
            sentences: 句子列表
            tokenizer: 分词器
            max_length: 最大长度
            negative_ratio: 负样本比例

        Returns:
            (input_ids, attention_mask, nsp_labels)
        """
        num_pairs = len(sentences) - 1
        input_ids_list = []
        attention_mask_list = []
        nsp_labels_list = []

        for i in range(num_pairs):
            # 随机决定是正样本还是负样本
            is_negative = random.random() < negative_ratio

            sent_a = sentences[i]

            if is_negative:
                # 负样本: 随机选择一个不相邻的句子
                random_idx = random.choice([j for j in range(len(sentences)) if abs(j - i) > 1])
                sent_b = sentences[random_idx]
                label = 1  # 不连续
            else:
                # 正样本: 选择下一个句子
                sent_b = sentences[i + 1]
                label = 0  # 连续

            # 拼接: [CLS] sent_a [SEP] sent_b [SEP]
            # (这里简化，实际应该使用tokenizer)
            # encoded = tokenizer(sent_a, sent_b, max_length=max_length, ...)

            # 占位符 (实际实现需要真实的tokenizer)
            input_ids_list.append(torch.zeros(max_length, dtype=torch.long))
            attention_mask_list.append(torch.ones(max_length, dtype=torch.long))
            nsp_labels_list.append(label)

        input_ids = torch.stack(input_ids_list)
        attention_mask = torch.stack(attention_mask_list)
        nsp_labels = torch.tensor(nsp_labels_list, dtype=torch.long)

        return input_ids, attention_mask, nsp_labels


# ==================== 便捷函数 ====================

def create_mlm_pretrainer(
    model: nn.Module,
    hidden_size: int = 768,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> MLMPretrainer:
    """
    创建MLM预训练器的便捷函数

    Args:
        model: 语言模型
        hidden_size: 隐藏层大小
        config: 配置字典
        **kwargs: 其他参数

    Returns:
        MLMPretrainer实例

    Example:
        >>> pretrainer = create_mlm_pretrainer(model, hidden_size=768)
        >>> stats = pretrainer.train_step(input_ids)
    """
    if config is not None:
        mlm_config = MLMConfig(**config)
    else:
        mlm_config = MLMConfig()

    return MLMPretrainer(
        model=model,
        hidden_size=hidden_size,
        config=mlm_config,
        **kwargs
    )


# ==================== 使用示例 ====================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("MLM预训练演示")
    print("=" * 70)

    # 创建假的语言模型
    class FakeLanguageModel(nn.Module):
        def __init__(self, vocab_size=1000, hidden_size=128):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)

        def forward(self, input_ids, attention_mask=None, output_hidden_states=True):
            hidden = self.embedding(input_ids)

            class Output:
                def __init__(self, hidden):
                    self.last_hidden_state = hidden
                    self.hidden_states = (hidden,)

            return Output(hidden)

    model = FakeLanguageModel(vocab_size=1000, hidden_size=128)

    # 创建MLM预训练器
    print("\n[仅MLM任务]")
    pretrainer = create_mlm_pretrainer(
        model=model,
        hidden_size=128,
        config={
            'vocab_size': 1000,
            'mask_prob': 0.15,
            'use_nsp': False
        },
        device='cpu'
    )

    # 模拟训练数据
    batch_size = 4
    seq_len = 20

    print("\n开始MLM预训练...")
    for step in range(10):
        input_ids = torch.randint(5, 1000, (batch_size, seq_len))  # 避开特殊token
        attention_mask = torch.ones_like(input_ids)

        stats = pretrainer.train_step(input_ids, attention_mask)

        if step % 3 == 0:
            print(f"\nStep {step}:")
            print(f"  MLM Loss: {stats['mlm_loss']:.4f}")
            print(f"  MLM Accuracy: {stats['mlm_accuracy']:.2%}")
            print(f"  Masked Tokens: {stats['num_masked']}")

    # MLM + NSP
    print("\n" + "=" * 70)
    print("[MLM + NSP任务 (BERT风格)]")
    print("=" * 70)

    model2 = FakeLanguageModel(vocab_size=1000, hidden_size=128)
    pretrainer_bert = create_mlm_pretrainer(
        model=model2,
        hidden_size=128,
        config={
            'vocab_size': 1000,
            'mask_prob': 0.15,
            'use_nsp': True
        },
        device='cpu'
    )

    print("\n开始BERT风格预训练...")
    for step in range(10):
        input_ids = torch.randint(5, 1000, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)
        nsp_labels = torch.randint(0, 2, (batch_size,))  # 0或1

        stats = pretrainer_bert.train_step(input_ids, attention_mask, nsp_labels)

        if step % 3 == 0:
            print(f"\nStep {step}:")
            print(f"  Total Loss: {stats['total_loss']:.4f}")
            print(f"  MLM Loss: {stats['mlm_loss']:.4f}")
            print(f"  MLM Accuracy: {stats['mlm_accuracy']:.2%}")
            print(f"  NSP Loss: {stats['nsp_loss']:.4f}")
            print(f"  NSP Accuracy: {stats['nsp_accuracy']:.2%}")

    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    print("\nMLM预训练优势:")
    print("  ✓ 简单有效的自监督学习")
    print("  ✓ 学习双向上下文表示")
    print("  ✓ BERT证明的有效性")
    print("  ✓ 可扩展到大规模数据")
