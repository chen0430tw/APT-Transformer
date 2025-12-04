#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Claude Unified Model - 整合图论反思和HHH反思的Claude模型

整合了两种反思机制：
1. Graph-based Reflection (Claude4): 图连通度、最短路径、镜像复杂度
2. Constitutional AI Reflection: HHH原则（Helpful, Harmless, Honest）

提供灵活的配置，可以选择使用哪种反思机制或同时使用两种。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List
from collections import deque


# ==============================================================================
# HHH Reflection Components (Constitutional AI)
# ==============================================================================

class HHHReflectionLayer(nn.Module):
    """
    HHH反思层 - Constitutional AI风格
    评估输出的Helpful, Harmless, Honest属性
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # 自注意力用于反思
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # HHH评分器
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
            x: [batch, seq_len, d_model]
        Returns:
            reflected_x, hhh_scores
        """
        # 自我反思
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN
        ffn_out = self.ffn(x)
        reflected_x = self.norm2(x + ffn_out)

        # 计算HHH分数
        pooled = reflected_x.mean(dim=1)  # [batch, d_model]

        hhh_scores = {
            'helpful': torch.sigmoid(self.helpful_score(pooled)),
            'harmless': torch.sigmoid(self.harmless_score(pooled)),
            'honest': torch.sigmoid(self.honest_score(pooled))
        }

        return reflected_x, hhh_scores


class CorrectionLayer(nn.Module):
    """修正层 - 基于HHH分数修正输出"""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()

        self.correction_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        # 门控：决定保留多少原始vs修正
        self.gate = nn.Sequential(
            nn.Linear(d_model + 3, d_model),  # +3 for HHH scores
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        reflected_x: torch.Tensor,
        hhh_scores: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """修正输出"""
        batch_size, seq_len, d_model = x.shape

        # 生成修正版本
        correction = self.correction_proj(reflected_x)

        # 准备门控输入
        score_tensor = torch.cat([
            hhh_scores['helpful'],
            hhh_scores['harmless'],
            hhh_scores['honest']
        ], dim=-1)  # [batch, 3]

        score_expanded = score_tensor.unsqueeze(1).expand(-1, seq_len, -1)
        gate_input = torch.cat([reflected_x, score_expanded], dim=-1)

        # 计算门控权重
        gate_weight = self.gate(gate_input)

        # 混合
        corrected = gate_weight * correction + (1 - gate_weight) * x

        return self.norm(corrected)


# ==============================================================================
# Graph Reflection Components (Claude4 style)
# ==============================================================================

class GraphConnectivityAnalyzer(nn.Module):
    """图连通度分析器 - 分析注意力图的连通性"""

    def __init__(self, d_model: int, threshold: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.threshold = threshold

    def compute_connectivity(
        self,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """计算连通度分数"""
        B, H, T, _ = attention_weights.shape

        # 二值化注意力图
        adj_matrix = (attention_weights > self.threshold).float()

        # 计算度数
        degree = adj_matrix.sum(dim=-1)  # [B, H, T]

        # 简化的连通度估计：平均度数 / 序列长度
        connectivity = degree / (T + 1e-8)

        return connectivity.mean(dim=1)  # [B, T]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """前向传播"""
        # 计算连通度
        connectivity = self.compute_connectivity(attention_weights)  # [B, T]

        # 调制hidden states
        connectivity_weight = connectivity.unsqueeze(-1)  # [B, T, 1]
        modulated = hidden_states * (1.0 + 0.1 * connectivity_weight)

        return modulated


class MirrorComplexityAnalyzer(nn.Module):
    """镜像复杂度分析器"""

    def __init__(self, d_model: int, num_mirrors: int = 3):
        super().__init__()
        self.num_mirrors = num_mirrors

        # 镜像投影
        self.mirror_projs = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_mirrors)
        ])

    def compute_complexity(self, x: torch.Tensor, mirrors: List[torch.Tensor]) -> torch.Tensor:
        """计算复杂度"""
        # 计算与每个镜像的距离
        distances = []
        for mirror in mirrors:
            dist = torch.norm(x - mirror, dim=-1)  # [B, T]
            distances.append(dist)

        # 复杂度 = 平均距离
        complexity = torch.stack(distances, dim=0).mean(dim=0)  # [B, T]

        return complexity

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 创建镜像
        mirrors = [proj(x) for proj in self.mirror_projs]

        # 计算复杂度
        complexity = self.compute_complexity(x, mirrors)

        # 选择最复杂的镜像
        complexity_weight = F.softmax(complexity.unsqueeze(-1), dim=1)
        weighted_mirror = sum(m * complexity_weight for m in mirrors)

        return weighted_mirror, complexity.mean(dim=1)  # [B, D], [B]


# ==============================================================================
# Unified Claude Model
# ==============================================================================

class ClaudeTransformerBlock(nn.Module):
    """Claude Transformer Block"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()

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
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention
        attn_out, attn_weights = self.self_attn(x, x, x, attn_mask=mask, need_weights=return_attention)
        x = self.norm1(x + attn_out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        if return_attention:
            return x, attn_weights
        return x, None


class ClaudeUnifiedModel(nn.Module):
    """
    Claude统一模型 - 整合图论反思和HHH反思

    Args:
        vocab_size: 词汇表大小
        d_model: 模型维度
        num_heads: 注意力头数
        num_layers: Transformer层数
        d_ff: FFN维度
        max_seq_len: 最大序列长度
        dropout: Dropout率
        use_hhh_reflection: 是否使用HHH反思
        use_graph_reflection: 是否使用图论反思
        reflection_at_layers: 在哪些层使用反思（如[6,9,12]表示第6,9,12层）
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
        use_hhh_reflection: bool = True,
        use_graph_reflection: bool = False,
        reflection_at_layers: Optional[List[int]] = None,
        enable_multimodal: bool = False,
        image_dim: int = 1024,
        audio_dim: int = 512
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.use_hhh_reflection = use_hhh_reflection
        self.use_graph_reflection = use_graph_reflection
        self.enable_multimodal = enable_multimodal

        # 默认在后半段层使用反思
        if reflection_at_layers is None:
            reflection_at_layers = list(range(num_layers // 2, num_layers))
        self.reflection_at_layers = set(reflection_at_layers)

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        # Multimodal projections (optional)
        if enable_multimodal:
            self.image_proj = nn.Linear(image_dim, d_model)
            self.audio_proj = nn.Linear(audio_dim, d_model)
            nn.init.normal_(self.image_proj.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.audio_proj.weight, mean=0.0, std=0.02)
            if self.image_proj.bias is not None:
                nn.init.zeros_(self.image_proj.bias)
            if self.audio_proj.bias is not None:
                nn.init.zeros_(self.audio_proj.bias)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ClaudeTransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Reflection layers
        if use_hhh_reflection:
            self.hhh_reflection = HHHReflectionLayer(d_model, num_heads, dropout)
            self.correction_layer = CorrectionLayer(d_model, dropout)

        if use_graph_reflection:
            self.graph_connectivity = GraphConnectivityAnalyzer(d_model)
            self.mirror_complexity = MirrorComplexityAnalyzer(d_model)

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

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        image_feat: Optional[torch.Tensor] = None,
        audio_feat: Optional[torch.Tensor] = None,
        return_reflection: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        """
        前向传播（支持多模态）

        Args:
            input_ids: [batch, seq_len] 文本token IDs (可选)
            image_feat: [batch, D_img] 图像特征 (可选)
            audio_feat: [batch, D_aud] 音频特征 (可选)
            return_reflection: 是否返回反思信息

        Returns:
            logits 或 (logits, reflection_info)
        """
        # 构建多模态嵌入
        embeddings_list = []

        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
            # Text embeddings
            text_emb = self.token_embedding(input_ids)
            text_emb = text_emb + self.pos_embedding[:, :seq_len, :]
            embeddings_list.append(text_emb)

        if self.enable_multimodal and image_feat is not None:
            # Image embeddings
            img_emb = self.image_proj(image_feat)  # [B, D]
            if img_emb.dim() == 2:
                img_emb = img_emb.unsqueeze(1)  # [B, 1, D]
            embeddings_list.append(img_emb)

        if self.enable_multimodal and audio_feat is not None:
            # Audio embeddings
            aud_emb = self.audio_proj(audio_feat)  # [B, D]
            if aud_emb.dim() == 2:
                aud_emb = aud_emb.unsqueeze(1)  # [B, 1, D]
            embeddings_list.append(aud_emb)

        if not embeddings_list:
            raise ValueError("At least one modality (text/image/audio) must be provided")

        # Concatenate all modalities
        if len(embeddings_list) == 1:
            x = embeddings_list[0]
        else:
            x = torch.cat(embeddings_list, dim=1)  # [B, T_total, D]

        batch_size, seq_len = x.size(0), x.size(1)
        x = self.dropout(x)

        # 因果mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(x.device)
        mask = mask.masked_fill(mask == True, float('-inf'))

        # Transformer blocks with reflection
        reflection_info = {
            'hhh_scores': None,
            'graph_stats': {},
            'reflected_at': []
        }

        for i, block in enumerate(self.blocks):
            # 是否在这一层使用反思
            use_reflection_here = i in self.reflection_at_layers

            # 前向传播（可能需要注意力权重）
            x, attn_weights = block(x, mask, return_attention=use_reflection_here and self.use_graph_reflection)

            # 应用反思
            if use_reflection_here:
                original_x = x

                # HHH反思
                if self.use_hhh_reflection:
                    reflected_x, hhh_scores = self.hhh_reflection(x)
                    x = self.correction_layer(original_x, reflected_x, hhh_scores)

                    if return_reflection:
                        reflection_info['hhh_scores'] = hhh_scores
                        reflection_info['reflected_at'].append(('hhh', i))

                # 图论反思
                if self.use_graph_reflection and attn_weights is not None:
                    # 连通度分析
                    x = self.graph_connectivity(x, attn_weights.unsqueeze(1))

                    # 镜像复杂度
                    mirrored_x, complexity = self.mirror_complexity(x)
                    x = 0.9 * x + 0.1 * mirrored_x  # 轻微混合

                    if return_reflection:
                        reflection_info['graph_stats'][f'layer_{i}'] = {
                            'complexity': complexity.mean().item()
                        }
                        reflection_info['reflected_at'].append(('graph', i))

        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if return_reflection:
            return logits, reflection_info
        return logits

    def get_reflection_summary(self, input_ids: torch.Tensor) -> Dict:
        """获取反思摘要（不计算梯度）"""
        with torch.no_grad():
            _, reflection_info = self.forward(input_ids, return_reflection=True)

            summary = {
                'reflection_type': [],
                'num_reflection_layers': len(reflection_info['reflected_at'])
            }

            if reflection_info['hhh_scores'] is not None:
                summary['reflection_type'].append('HHH')
                summary['hhh_scores'] = {
                    k: v.mean().item() for k, v in reflection_info['hhh_scores'].items()
                }

            if reflection_info['graph_stats']:
                summary['reflection_type'].append('Graph')
                summary['avg_complexity'] = sum(
                    v['complexity'] for v in reflection_info['graph_stats'].values()
                ) / len(reflection_info['graph_stats'])

            return summary


def create_claude_unified(
    vocab_size: int = 50000,
    model_size: str = 'base',
    reflection_mode: str = 'hhh',  # 'hhh', 'graph', 'both'
    **kwargs
) -> ClaudeUnifiedModel:
    """
    创建Claude统一模型的便捷函数

    Args:
        vocab_size: 词汇表大小
        model_size: 'small', 'base', 'large'
        reflection_mode: 'hhh', 'graph', 'both'
        **kwargs: 其他参数

    Returns:
        ClaudeUnifiedModel实例
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

    # 设置反思模式
    use_hhh = reflection_mode in ['hhh', 'both']
    use_graph = reflection_mode in ['graph', 'both']

    return ClaudeUnifiedModel(
        vocab_size=vocab_size,
        use_hhh_reflection=use_hhh,
        use_graph_reflection=use_graph,
        **config
    )


if __name__ == '__main__':
    print("="*70)
    print("Claude Unified Model - 测试")
    print("="*70)

    # 测试不同配置
    configs_to_test = [
        ('HHH反思', 'hhh'),
        ('图论反思', 'graph'),
        ('混合反思', 'both')
    ]

    for name, mode in configs_to_test:
        print(f"\n测试: {name} ({mode})")
        model = create_claude_unified(
            vocab_size=50000,
            model_size='small',
            reflection_mode=mode
        )

        print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")

        # 测试前向传播
        batch_size = 2
        seq_len = 32
        input_ids = torch.randint(0, 50000, (batch_size, seq_len))

        # 不带反思信息
        logits = model(input_ids)
        print(f"  输出形状: {logits.shape}")

        # 带反思信息
        logits, reflection_info = model(input_ids, return_reflection=True)
        print(f"  反思层数: {len(reflection_info['reflected_at'])}")

        if reflection_info['hhh_scores']:
            print(f"  HHH分数: {reflection_info['hhh_scores']}")

        if reflection_info['graph_stats']:
            print(f"  图论统计: {len(reflection_info['graph_stats'])} 层")

    print("\n" + "="*70)
    print("测试完成！")
