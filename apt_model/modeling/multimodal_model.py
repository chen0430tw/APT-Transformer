#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多模态APT模型
Multimodal APT model supporting text, vision, and audio inputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, Union, List
from apt.core.modeling.apt_model import APTLargeModel
from apt_model.config.multimodal_config import MultimodalConfig
from apt.core.modeling.encoders.vision_encoder import VisionEncoder
from apt.core.modeling.encoders.audio_encoder import AudioEncoder
from apt.core.modeling.encoders.cross_modal_attention import (
    CrossModalAttention,
    BiDirectionalCrossAttention,
    MultiModalFusionLayer,
    TriModalFusionLayer
)


class MultimodalAPTModel(APTLargeModel):
    """
    多模态APT模型
    支持文本、图像、音频三种模态的组合
    """

    def __init__(
        self,
        config,
        multimodal_config: Optional[MultimodalConfig] = None,
        vision_encoder_type: str = 'simple',
        audio_encoder_type: str = 'simple',
        freeze_encoders: bool = False,
        fusion_method: str = 'cross_attention'
    ):
        """
        Args:
            config: APT模型配置
            multimodal_config: 多模态配置
            vision_encoder_type: 视觉编码器类型 ('simple', 'clip', 'vit', 'resnet')
            audio_encoder_type: 音频编码器类型 ('simple', 'wav2vec2', 'hubert', 'whisper')
            freeze_encoders: 是否冻结预训练编码器
            fusion_method: 融合方法 ('cross_attention', 'tri_modal', 'concatenate', 'add', 'gated')
        """
        super().__init__(config)

        # 多模态配置
        if multimodal_config is None:
            multimodal_config = MultimodalConfig()
        self.multimodal_config = multimodal_config

        self.vision_encoder_type = vision_encoder_type
        self.audio_encoder_type = audio_encoder_type
        self.fusion_method = fusion_method

        # 初始化多模态编码器
        self._init_encoders(freeze_encoders)

        # 初始化融合层
        self._init_fusion_layers()

        # 模态类型嵌入
        self.modality_type_embed = nn.Embedding(4, config.d_model)  # 0=text, 1=vision, 2=audio, 3=fused

    def _init_encoders(self, freeze_encoders: bool):
        """初始化各模态编码器"""
        d_model = self.config.d_model

        # 视觉编码器
        if self.multimodal_config.enable_image:
            self.vision_encoder = VisionEncoder(
                encoder_type=self.vision_encoder_type,
                output_dim=d_model,
                freeze_encoder=freeze_encoders
            )
        else:
            self.vision_encoder = None

        # 音频编码器
        if self.multimodal_config.enable_audio:
            self.audio_encoder = AudioEncoder(
                encoder_type=self.audio_encoder_type,
                output_dim=d_model,
                freeze_encoder=freeze_encoders
            )
        else:
            self.audio_encoder = None

    def _init_fusion_layers(self):
        """初始化多模态融合层"""
        d_model = self.config.d_model

        if self.fusion_method == 'cross_attention':
            # 跨模态注意力融合
            self.text_vision_fusion = CrossModalAttention(
                embed_dim=d_model,
                num_heads=self.config.num_attention_heads,
                dropout=self.config.dropout
            )
            self.text_audio_fusion = CrossModalAttention(
                embed_dim=d_model,
                num_heads=self.config.num_attention_heads,
                dropout=self.config.dropout
            )
            self.vision_audio_fusion = BiDirectionalCrossAttention(
                embed_dim=d_model,
                num_heads=self.config.num_attention_heads,
                dropout=self.config.dropout
            )

        elif self.fusion_method == 'tri_modal':
            # 三模态融合
            self.tri_modal_fusion = TriModalFusionLayer(
                embed_dim=d_model,
                num_heads=self.config.num_attention_heads,
                dropout=self.config.dropout
            )

        else:
            # 其他融合方法 (concatenate, add, gated)
            self.multimodal_fusion = MultiModalFusionLayer(
                embed_dim=d_model,
                num_heads=self.config.num_attention_heads,
                fusion_method=self.fusion_method,
                dropout=self.config.dropout
            )

        # 融合后的投影层
        self.fusion_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(self.config.dropout)
        )

    def encode_vision(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        编码视觉输入

        Args:
            pixel_values: [B, C, H, W]

        Returns:
            vision_features: [B, D]
        """
        if self.vision_encoder is None:
            raise ValueError("Vision encoder not initialized")

        return self.vision_encoder(pixel_values)

    def encode_audio(self, audio_values: torch.Tensor) -> torch.Tensor:
        """
        编码音频输入

        Args:
            audio_values: [B, T] 或 [B, n_mels, T]

        Returns:
            audio_features: [B, D]
        """
        if self.audio_encoder is None:
            raise ValueError("Audio encoder not initialized")

        return self.audio_encoder(audio_values)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        编码文本输入

        Args:
            input_ids: [B, L]
            attention_mask: [B, L]

        Returns:
            text_features: [B, L, D]
        """
        # 使用父类的嵌入层
        text_emb = self.token_embedding(input_ids)
        text_emb = self.positional_encoding(text_emb)

        # 添加模态类型编码
        batch_size, seq_len = input_ids.size()
        modality_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=input_ids.device)
        modality_emb = self.modality_type_embed(modality_ids)
        text_emb = text_emb + modality_emb

        # 通过编码器
        if attention_mask is not None:
            # 转换attention_mask格式: [B, L] -> [B, 1, 1, L]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0

        text_features = self.encoder(text_emb, attention_mask=attention_mask)

        return text_features

    def fuse_modalities(
        self,
        text_features: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        融合多种模态

        Args:
            text_features: [B, L, D] 或 None
            vision_features: [B, D] 或 None
            audio_features: [B, D] 或 None

        Returns:
            fused_features: [B, L, D] 或 [B, D]
        """
        # 统计可用模态数量
        available_modalities = []
        if text_features is not None:
            available_modalities.append('text')
        if vision_features is not None:
            available_modalities.append('vision')
        if audio_features is not None:
            available_modalities.append('audio')

        if len(available_modalities) == 0:
            raise ValueError("At least one modality must be provided")

        # 单模态情况
        if len(available_modalities) == 1:
            if text_features is not None:
                return text_features
            elif vision_features is not None:
                return vision_features.unsqueeze(1)  # [B, 1, D]
            else:
                return audio_features.unsqueeze(1)  # [B, 1, D]

        # 确保vision和audio有序列维度
        if vision_features is not None and vision_features.dim() == 2:
            vision_features = vision_features.unsqueeze(1)  # [B, 1, D]

        if audio_features is not None and audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(1)  # [B, 1, D]

        # 多模态融合
        if self.fusion_method == 'cross_attention':
            return self._fuse_cross_attention(text_features, vision_features, audio_features)

        elif self.fusion_method == 'tri_modal':
            return self._fuse_tri_modal(text_features, vision_features, audio_features)

        else:
            return self._fuse_other(text_features, vision_features, audio_features)

    def _fuse_cross_attention(
        self,
        text_features: Optional[torch.Tensor],
        vision_features: Optional[torch.Tensor],
        audio_features: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """使用跨模态注意力融合"""
        # 文本关注视觉和音频
        fused = text_features

        if text_features is not None and vision_features is not None:
            text_vision, _ = self.text_vision_fusion(
                query=text_features,
                key=vision_features,
                value=vision_features
            )
            fused = fused + text_vision

        if text_features is not None and audio_features is not None:
            text_audio, _ = self.text_audio_fusion(
                query=text_features,
                key=audio_features,
                value=audio_features
            )
            fused = fused + text_audio

        # 视觉和音频相互关注
        if vision_features is not None and audio_features is not None:
            enhanced_vision, enhanced_audio = self.vision_audio_fusion(
                vision_features,
                audio_features
            )
            # 将增强的视觉和音频信息也融入文本
            if text_features is not None:
                # 平均池化并添加到文本特征
                vision_pooled = enhanced_vision.mean(dim=1, keepdim=True)  # [B, 1, D]
                audio_pooled = enhanced_audio.mean(dim=1, keepdim=True)  # [B, 1, D]
                fused = fused + vision_pooled + audio_pooled

        return self.fusion_projection(fused)

    def _fuse_tri_modal(
        self,
        text_features: Optional[torch.Tensor],
        vision_features: Optional[torch.Tensor],
        audio_features: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """使用三模态融合层"""
        if text_features is None or vision_features is None or audio_features is None:
            # 退回到其他融合方法
            return self._fuse_other(text_features, vision_features, audio_features)

        # 三模态融合返回 [B, D]，需要扩展到 [B, L, D]
        fused_pooled = self.tri_modal_fusion(text_features, vision_features, audio_features)  # [B, D]

        # 广播到序列长度
        batch_size = text_features.size(0)
        seq_len = text_features.size(1)
        fused = fused_pooled.unsqueeze(1).expand(batch_size, seq_len, -1)  # [B, L, D]

        # 与原始文本特征残差连接
        fused = text_features + fused

        return self.fusion_projection(fused)

    def _fuse_other(
        self,
        text_features: Optional[torch.Tensor],
        vision_features: Optional[torch.Tensor],
        audio_features: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """使用其他融合方法 (concatenate, add, gated)"""
        # 依次融合
        fused = text_features

        if text_features is not None and vision_features is not None:
            fused = self.multimodal_fusion(fused, vision_features)

        if fused is not None and audio_features is not None:
            fused = self.multimodal_fusion(fused, audio_features)

        return self.fusion_projection(fused)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        audio_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        多模态前向传播

        Args:
            input_ids: 文本输入 [B, L]
            attention_mask: 文本注意力掩码 [B, L]
            pixel_values: 图像输入 [B, C, H, W]
            audio_values: 音频输入 [B, T] 或 [B, n_mels, T]
            labels: 标签 [B] 或 [B, L]
            return_dict: 是否返回字典

        Returns:
            如果return_dict=True，返回字典包含:
            - logits: 输出logits
            - loss: 损失 (如果提供labels)
            - text_features: 文本特征
            - vision_features: 视觉特征 (如果有)
            - audio_features: 音频特征 (如果有)
            - fused_features: 融合特征
            否则返回logits张量
        """
        # 编码各模态
        text_features = None
        vision_features = None
        audio_features = None

        if input_ids is not None:
            text_features = self.encode_text(input_ids, attention_mask)

        if pixel_values is not None and self.vision_encoder is not None:
            vision_features = self.encode_vision(pixel_values)

        if audio_values is not None and self.audio_encoder is not None:
            audio_features = self.encode_audio(audio_values)

        # 融合多模态
        fused_features = self.fuse_modalities(text_features, vision_features, audio_features)

        # 输出投影
        logits = self.output_projection(fused_features)

        # 计算损失
        loss = None
        if labels is not None:
            if labels.dim() == 1:
                # 分类任务: [B]
                # 池化序列维度
                pooled_logits = logits.mean(dim=1)  # [B, D] -> [B, vocab_size]
                loss = F.cross_entropy(pooled_logits, labels)
            else:
                # 序列任务: [B, L]
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )

        if return_dict:
            return {
                'logits': logits,
                'loss': loss,
                'text_features': text_features,
                'vision_features': vision_features,
                'audio_features': audio_features,
                'fused_features': fused_features
            }
        else:
            return logits

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        audio_values: Optional[torch.Tensor] = None,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        **kwargs
    ) -> torch.Tensor:
        """
        生成文本序列

        Args:
            input_ids: 输入token IDs [B, L]
            pixel_values: 图像输入 [B, C, H, W]
            audio_values: 音频输入 [B, T]
            max_length: 最大生成长度
            temperature: 温度参数
            top_k: Top-k采样
            top_p: Top-p采样

        Returns:
            generated_ids: 生成的token IDs [B, max_length]
        """
        batch_size = input_ids.size(0) if input_ids is not None else 1
        device = input_ids.device if input_ids is not None else next(self.parameters()).device

        # 编码多模态输入（只需编码一次）
        vision_features = None
        audio_features = None

        if pixel_values is not None:
            vision_features = self.encode_vision(pixel_values)

        if audio_values is not None:
            audio_features = self.encode_audio(audio_values)

        # 初始化生成序列
        if input_ids is None:
            # 使用特殊起始token
            generated = torch.full((batch_size, 1), self.config.bos_token_id, dtype=torch.long, device=device)
        else:
            generated = input_ids

        # 自回归生成
        for _ in range(max_length):
            # 编码当前文本
            text_features = self.encode_text(generated)

            # 融合模态
            fused_features = self.fuse_modalities(text_features, vision_features, audio_features)

            # 获取最后一个位置的logits
            logits = self.output_projection(fused_features)  # [B, L, vocab_size]
            next_token_logits = logits[:, -1, :] / temperature  # [B, vocab_size]

            # Top-k 和 Top-p 采样
            filtered_logits = self._top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

            # 采样下一个token
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

            # 添加到生成序列
            generated = torch.cat([generated, next_token], dim=1)

            # 检查是否所有序列都已结束
            if (next_token == self.config.eos_token_id).all():
                break

        return generated

    def _top_k_top_p_filtering(
        self,
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float('Inf')
    ) -> torch.Tensor:
        """Top-k和top-p过滤"""
        # Top-k
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        # Top-p
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # 移除累积概率超过top_p的tokens
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value

        return logits


# 工厂函数

def create_multimodal_model(
    config,
    multimodal_config: Optional[MultimodalConfig] = None,
    vision_encoder: str = 'simple',
    audio_encoder: str = 'simple',
    fusion_method: str = 'cross_attention',
    **kwargs
) -> MultimodalAPTModel:
    """
    创建多模态APT模型的工厂函数

    Args:
        config: APT配置
        multimodal_config: 多模态配置
        vision_encoder: 视觉编码器类型
        audio_encoder: 音频编码器类型
        fusion_method: 融合方法
        **kwargs: 其他参数

    Returns:
        MultimodalAPTModel实例
    """
    return MultimodalAPTModel(
        config=config,
        multimodal_config=multimodal_config,
        vision_encoder_type=vision_encoder,
        audio_encoder_type=audio_encoder,
        fusion_method=fusion_method,
        **kwargs
    )
