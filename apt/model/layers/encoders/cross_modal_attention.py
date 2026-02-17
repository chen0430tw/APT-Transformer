#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跨模态注意力模块
Cross-modal attention mechanisms for multimodal fusion
"""

from apt.core.fake_torch import get_torch
torch = get_torch()
from apt.core.fake_torch import get_torch
torch = get_torch()
nn = torch.nn
F = torch.nn.functional
from typing import Optional, Tuple


class CrossModalAttention(nn.Module):
    """
    跨模态注意力机制
    允许一个模态(query)关注另一个模态(key/value)的信息
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            dropout: Dropout概率
            bias: 是否使用bias
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Query来自模态1
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Key和Value来自模态2
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.dropout = nn.Dropout(dropout)

        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [B, N1, D] - 来自模态1
            key: [B, N2, D] - 来自模态2
            value: [B, N2, D] - 来自模态2
            attention_mask: [B, N1, N2] - 注意力掩码

        Returns:
            output: [B, N1, D]
            attention_weights: [B, num_heads, N1, N2]
        """
        batch_size, seq_len_q, _ = query.size()
        seq_len_kv = key.size(1)

        # 投影到多头
        Q = self.q_proj(query)  # [B, N1, D]
        K = self.k_proj(key)    # [B, N2, D]
        V = self.v_proj(value)  # [B, N2, D]

        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        # 现在: [B, num_heads, seq_len, head_dim]

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # [B, num_heads, N1, N2]

        # 应用注意力掩码
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # 应用注意力到value
        context = torch.matmul(attention_probs, V)
        # [B, num_heads, N1, head_dim]

        # 合并多头
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len_q, self.embed_dim)
        # [B, N1, D]

        # 输出投影
        output = self.out_proj(context)
        output = self.dropout(output)

        return output, attention_probs


class BiDirectionalCrossAttention(nn.Module):
    """
    双向跨模态注意力
    两个模态相互关注对方的信息
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()

        # 模态1关注模态2
        self.cross_attn_1to2 = CrossModalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # 模态2关注模态1
        self.cross_attn_2to1 = CrossModalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Layer Norm
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        modal1_features: torch.Tensor,
        modal2_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            modal1_features: [B, N1, D]
            modal2_features: [B, N2, D]

        Returns:
            enhanced_modal1: [B, N1, D]
            enhanced_modal2: [B, N2, D]
        """
        # 模态1使用模态2的信息
        modal1_cross, _ = self.cross_attn_1to2(
            query=modal1_features,
            key=modal2_features,
            value=modal2_features
        )
        enhanced_modal1 = self.norm1(modal1_features + modal1_cross)

        # 模态2使用模态1的信息
        modal2_cross, _ = self.cross_attn_2to1(
            query=modal2_features,
            key=modal1_features,
            value=modal1_features
        )
        enhanced_modal2 = self.norm2(modal2_features + modal2_cross)

        return enhanced_modal1, enhanced_modal2


class MultiModalFusionLayer(nn.Module):
    """
    多模态融合层
    支持多种融合策略
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        fusion_method: str = 'attention',
        dropout: float = 0.1
    ):
        """
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
            fusion_method: 融合方法 ('attention', 'concatenate', 'add', 'gated')
            dropout: Dropout概率
        """
        super().__init__()

        self.fusion_method = fusion_method
        self.embed_dim = embed_dim

        if fusion_method == 'attention':
            # 使用跨模态注意力融合
            self.cross_attention = CrossModalAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            )

        elif fusion_method == 'concatenate':
            # 拼接后降维
            self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim)

        elif fusion_method == 'gated':
            # 门控融合
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid()
            )
            self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim)

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        modal1_features: torch.Tensor,
        modal2_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            modal1_features: [B, N1, D] 或 [B, D]
            modal2_features: [B, N2, D] 或 [B, D]

        Returns:
            fused_features: [B, N, D] 或 [B, D]
        """
        # 确保维度一致
        if modal1_features.dim() == 2:
            modal1_features = modal1_features.unsqueeze(1)  # [B, 1, D]
        if modal2_features.dim() == 2:
            modal2_features = modal2_features.unsqueeze(1)  # [B, 1, D]

        if self.fusion_method == 'attention':
            # 跨模态注意力融合
            fused, _ = self.cross_attention(
                query=modal1_features,
                key=modal2_features,
                value=modal2_features
            )
            fused = self.norm(modal1_features + fused)

        elif self.fusion_method == 'add':
            # 简单相加
            fused = modal1_features + modal2_features
            fused = self.norm(fused)

        elif self.fusion_method == 'concatenate':
            # 拼接后降维
            # 先平均池化到相同长度
            modal1_pooled = modal1_features.mean(dim=1, keepdim=True)  # [B, 1, D]
            modal2_pooled = modal2_features.mean(dim=1, keepdim=True)  # [B, 1, D]

            concat = torch.cat([modal1_pooled, modal2_pooled], dim=-1)  # [B, 1, 2D]
            fused = self.fusion_proj(concat)  # [B, 1, D]
            fused = self.norm(fused)

        elif self.fusion_method == 'gated':
            # 门控融合
            modal1_pooled = modal1_features.mean(dim=1, keepdim=True)
            modal2_pooled = modal2_features.mean(dim=1, keepdim=True)

            concat = torch.cat([modal1_pooled, modal2_pooled], dim=-1)
            gate = self.gate(concat)  # [B, 1, D]

            fused = gate * modal1_pooled + (1 - gate) * modal2_pooled
            fused = self.norm(fused)

        return fused


class TriModalFusionLayer(nn.Module):
    """
    三模态融合层
    融合文本、图像、音频三种模态
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()

        # 三个跨模态注意力
        self.text_to_vision = CrossModalAttention(embed_dim, num_heads, dropout)
        self.text_to_audio = CrossModalAttention(embed_dim, num_heads, dropout)
        self.vision_audio_fusion = BiDirectionalCrossAttention(embed_dim, num_heads, dropout)

        # 最终融合
        self.final_fusion = nn.Linear(embed_dim * 3, embed_dim)

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        audio_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            text_features: [B, N_text, D]
            vision_features: [B, N_vision, D]
            audio_features: [B, N_audio, D]

        Returns:
            fused_features: [B, D]
        """
        # 文本关注视觉和音频
        text_vision, _ = self.text_to_vision(text_features, vision_features, vision_features)
        text_audio, _ = self.text_to_audio(text_features, audio_features, audio_features)

        # 视觉和音频相互关注
        enhanced_vision, enhanced_audio = self.vision_audio_fusion(vision_features, audio_features)

        # 平均池化到固定长度
        text_pooled = (text_features + text_vision + text_audio).mean(dim=1)  # [B, D]
        vision_pooled = enhanced_vision.mean(dim=1)  # [B, D]
        audio_pooled = enhanced_audio.mean(dim=1)  # [B, D]

        # 拼接并融合
        concat = torch.cat([text_pooled, vision_pooled, audio_pooled], dim=-1)  # [B, 3D]
        fused = self.final_fusion(concat)  # [B, D]
        fused = self.norm(fused)

        return fused
