#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Claude Model - Transformer with Reflection Layer
带反思层的Claude模型实现

设计理念：
- Constitutional AI：通过反思层实现自我批评和改进
- Helpful, Harmless, Honest：三大核心原则
- Reflection Layer：在生成后进行反思和修正

架构特点：
1. 基础Transformer编码器
2. 反思层(Reflection Layer)：评估输出的安全性和质量
3. 修正层(Correction Layer)：基于反思结果进行修正
4. 对齐层(Alignment Layer)：确保输出符合HHH原则
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class ReflectionLayer(nn.Module):
    """
    反思层 - 评估生成内容的质量和安全性

    功能：
    - 评估输出是否有帮助(Helpful)
    - 检查输出是否无害(Harmless)
    - 验证输出是否诚实(Honest)
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # 自注意力：让模型反思自己的输出
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # HHH评估器
        self.helpful_score = nn.Linear(d_model, 1)
        self.harmless_score = nn.Linear(d_model, 1)
        self.honest_score = nn.Linear(d_model, 1)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: [batch, seq_len, d_model] - 模型的原始输出

        Returns:
            reflected_x: 反思后的表示
            scores: HHH评分字典
        """
        # 自我反思（自注意力）
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN
        ffn_out = self.ffn(x)
        reflected_x = self.norm2(x + ffn_out)

        # 计算HHH分数（在序列维度上平均）
        pooled = reflected_x.mean(dim=1)  # [batch, d_model]

        scores = {
            'helpful': torch.sigmoid(self.helpful_score(pooled)),    # [batch, 1]
            'harmless': torch.sigmoid(self.harmless_score(pooled)),  # [batch, 1]
            'honest': torch.sigmoid(self.honest_score(pooled))       # [batch, 1]
        }

        return reflected_x, scores


class CorrectionLayer(nn.Module):
    """
    修正层 - 基于反思分数修正输出

    如果HHH分数过低，修正层会调整输出
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()

        # 修正投影
        self.correction_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        # 门控机制：决定保留多少原始输出vs修正输出
        self.gate = nn.Sequential(
            nn.Linear(d_model + 3, d_model),  # +3 for HHH scores
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        reflected_x: torch.Tensor,
        scores: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            x: 原始输出 [batch, seq_len, d_model]
            reflected_x: 反思后的表示 [batch, seq_len, d_model]
            scores: HHH分数字典

        Returns:
            corrected_x: 修正后的输出
        """
        batch_size, seq_len, d_model = x.shape

        # 生成修正版本
        correction = self.correction_proj(reflected_x)

        # 准备门控输入：扩展分数到序列长度
        score_tensor = torch.cat([
            scores['helpful'],
            scores['harmless'],
            scores['honest']
        ], dim=-1)  # [batch, 3]

        score_expanded = score_tensor.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, 3]
        gate_input = torch.cat([reflected_x, score_expanded], dim=-1)  # [batch, seq_len, d_model+3]

        # 计算门控权重
        gate_weight = self.gate(gate_input)  # [batch, seq_len, d_model]

        # 混合原始和修正版本
        corrected = gate_weight * correction + (1 - gate_weight) * x

        return self.norm(corrected)


class ClaudeTransformerBlock(nn.Module):
    """
    Claude Transformer Block
    标准Transformer block + 反思能力
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

        # 标准Transformer组件
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class ClaudeModel(nn.Module):
    """
    Claude Model - Constitutional AI with Reflection

    架构：
    1. Token Embedding + Positional Encoding
    2. N × Transformer Blocks
    3. Reflection Layer（反思层）
    4. Correction Layer（修正层）
    5. Output Projection

    训练策略：
    - 主损失：标准语言模型损失
    - 反思损失：鼓励高HHH分数
    - 修正损失：确保修正提高质量
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        use_reflection: bool = True  # 是否使用反思层
    ):
        super().__init__()

        self.d_model = d_model
        self.use_reflection = use_reflection

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ClaudeTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 反思和修正层
        if use_reflection:
            self.reflection_layer = ReflectionLayer(d_model, num_heads, dropout)
            self.correction_layer = CorrectionLayer(d_model, dropout)

        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重共享
        self.token_embedding.weight = self.lm_head.weight

        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

        # 初始化线性层
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_reflection: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        """
        Args:
            input_ids: [batch, seq_len]
            return_reflection: 是否返回反思信息

        Returns:
            logits: [batch, seq_len, vocab_size]
            或 (logits, reflection_info) 如果return_reflection=True
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        token_emb = self.token_embedding(input_ids)  # [batch, seq_len, d_model]
        pos_emb = self.pos_embedding[:, :seq_len, :]
        x = self.dropout(token_emb + pos_emb)

        # 创建因果mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(input_ids.device)
        mask = mask.masked_fill(mask == True, float('-inf'))

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # 反思和修正
        reflection_info = {}
        if self.use_reflection:
            original_x = x
            reflected_x, hhh_scores = self.reflection_layer(x)
            x = self.correction_layer(original_x, reflected_x, hhh_scores)

            reflection_info = {
                'hhh_scores': hhh_scores,
                'reflected': reflected_x,
                'original': original_x
            }

        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]

        if return_reflection:
            return logits, reflection_info
        return logits

    def get_hhh_scores(self, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        获取HHH分数（不计算梯度）

        Args:
            input_ids: [batch, seq_len]

        Returns:
            hhh_scores: {'helpful': tensor, 'harmless': tensor, 'honest': tensor}
        """
        with torch.no_grad():
            _, reflection_info = self.forward(input_ids, return_reflection=True)
            return reflection_info['hhh_scores']


def create_claude_model(
    vocab_size: int = 50000,
    model_size: str = 'base',
    **kwargs
) -> ClaudeModel:
    """
    创建Claude模型的便捷函数

    Args:
        vocab_size: 词汇表大小
        model_size: 'small', 'base', 'large'
        **kwargs: 其他参数

    Returns:
        ClaudeModel实例
    """
    configs = {
        'small': {
            'd_model': 512,
            'num_heads': 8,
            'num_layers': 6,
            'd_ff': 2048,
        },
        'base': {
            'd_model': 768,
            'num_heads': 12,
            'num_layers': 12,
            'd_ff': 3072,
        },
        'large': {
            'd_model': 1024,
            'num_heads': 16,
            'num_layers': 24,
            'd_ff': 4096,
        }
    }

    config = configs.get(model_size, configs['base'])
    config.update(kwargs)

    return ClaudeModel(vocab_size=vocab_size, **config)


if __name__ == '__main__':
    # 测试代码
    print("创建Claude模型...")
    model = create_claude_model(vocab_size=50000, model_size='small')

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 50000, (batch_size, seq_len))

    print(f"\n输入形状: {input_ids.shape}")

    # 不带反思信息
    logits = model(input_ids)
    print(f"输出形状: {logits.shape}")

    # 带反思信息
    logits, reflection_info = model(input_ids, return_reflection=True)
    print(f"\nHHH分数:")
    for key, value in reflection_info['hhh_scores'].items():
        print(f"  {key}: {value.mean().item():.4f}")

    print("\nClaude模型测试完成！")
