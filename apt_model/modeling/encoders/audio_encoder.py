#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
音频编码器模块
Audio encoders for audio processing in multimodal models
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class SimpleAudioEncoder(nn.Module):
    """
    简单的音频编码器
    使用1D卷积处理音频特征
    """

    def __init__(self, output_dim: int = 768, n_mels: int = 80):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # 第一层: n_mels → 128
            nn.Conv1d(n_mels, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # 第二层: 128 → 256
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            # 第三层: 256 → 512
            nn.Conv1d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),

            # 全局平均池化
            nn.AdaptiveAvgPool1d(1)
        )

        self.projection = nn.Linear(512, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 音频特征张量 [B, n_mels, T] 或 [B, T, n_mels]

        Returns:
            特征张量 [B, output_dim]
        """
        # 确保输入是 [B, C, T] 格式
        if x.dim() == 3 and x.size(1) != 80:  # 如果是 [B, T, n_mels]
            x = x.transpose(1, 2)

        features = self.conv_layers(x)  # [B, 512, 1]
        features = features.squeeze(-1)  # [B, 512]
        output = self.projection(features)  # [B, output_dim]

        return output


class AudioEncoder(nn.Module):
    """
    高级音频编码器
    支持多种预训练模型 (Wav2Vec2, HuBERT等)
    """

    def __init__(
        self,
        encoder_type: str = 'wav2vec2',
        model_name: Optional[str] = None,
        output_dim: int = 768,
        freeze_encoder: bool = False
    ):
        """
        Args:
            encoder_type: 编码器类型 ('wav2vec2', 'hubert', 'whisper', 'simple')
            model_name: 预训练模型名称
            output_dim: 输出维度
            freeze_encoder: 是否冻结编码器参数
        """
        super().__init__()

        self.encoder_type = encoder_type
        self.output_dim = output_dim

        if encoder_type == 'wav2vec2':
            self._init_wav2vec2_encoder(model_name)
        elif encoder_type == 'hubert':
            self._init_hubert_encoder(model_name)
        elif encoder_type == 'whisper':
            self._init_whisper_encoder(model_name)
        elif encoder_type == 'simple':
            self.encoder = SimpleAudioEncoder(output_dim=output_dim)
            self.processor = None
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # 冻结encoder参数
        if freeze_encoder and hasattr(self, 'encoder'):
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _init_wav2vec2_encoder(self, model_name: Optional[str]):
        """初始化Wav2Vec2编码器"""
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Processor

            model_name = model_name or 'facebook/wav2vec2-base'
            self.encoder = Wav2Vec2Model.from_pretrained(model_name)
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)

            # 投影到目标维度
            encoder_dim = self.encoder.config.hidden_size
            if encoder_dim != self.output_dim:
                self.projection = nn.Linear(encoder_dim, self.output_dim)
            else:
                self.projection = nn.Identity()

        except ImportError:
            raise ImportError(
                "Wav2Vec2 requires transformers library. "
                "Install with: pip install transformers"
            )

    def _init_hubert_encoder(self, model_name: Optional[str]):
        """初始化HuBERT编码器"""
        try:
            from transformers import HubertModel, Wav2Vec2Processor

            model_name = model_name or 'facebook/hubert-base-ls960'
            self.encoder = HubertModel.from_pretrained(model_name)
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)

            # 投影到目标维度
            encoder_dim = self.encoder.config.hidden_size
            if encoder_dim != self.output_dim:
                self.projection = nn.Linear(encoder_dim, self.output_dim)
            else:
                self.projection = nn.Identity()

        except ImportError:
            raise ImportError(
                "HuBERT requires transformers library. "
                "Install with: pip install transformers"
            )

    def _init_whisper_encoder(self, model_name: Optional[str]):
        """初始化Whisper编码器"""
        try:
            from transformers import WhisperModel, WhisperProcessor

            model_name = model_name or 'openai/whisper-base'
            full_model = WhisperModel.from_pretrained(model_name)
            self.encoder = full_model.encoder  # 只使用编码器部分
            self.processor = WhisperProcessor.from_pretrained(model_name)

            # 投影到目标维度
            encoder_dim = full_model.config.d_model
            if encoder_dim != self.output_dim:
                self.projection = nn.Linear(encoder_dim, self.output_dim)
            else:
                self.projection = nn.Identity()

        except ImportError:
            raise ImportError(
                "Whisper requires transformers library. "
                "Install with: pip install transformers"
            )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_values: 音频张量 [B, T] 或处理器输出

        Returns:
            特征张量 [B, output_dim]
        """
        if self.encoder_type in ['wav2vec2', 'hubert']:
            # Wav2Vec2/HuBERT编码器
            outputs = self.encoder(input_values=input_values)
            # 使用平均池化
            features = outputs.last_hidden_state.mean(dim=1)  # [B, encoder_dim]
            features = self.projection(features)  # [B, output_dim]

        elif self.encoder_type == 'whisper':
            # Whisper编码器
            outputs = self.encoder(input_values)
            # 使用平均池化
            features = outputs.last_hidden_state.mean(dim=1)  # [B, encoder_dim]
            features = self.projection(features)  # [B, output_dim]

        elif self.encoder_type == 'simple':
            # 简单音频编码器
            features = self.encoder(input_values)  # [B, output_dim]

        return features

    def process_audio(self, audio_path: str, sampling_rate: int = 16000):
        """
        处理音频文件

        Args:
            audio_path: 音频文件路径
            sampling_rate: 采样率

        Returns:
            处理后的tensor
        """
        import torchaudio

        # 加载音频
        waveform, sr = torchaudio.load(audio_path)

        # 重采样到目标采样率
        if sr != sampling_rate:
            resampler = torchaudio.transforms.Resample(sr, sampling_rate)
            waveform = resampler(waveform)

        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0)  # [T]

        if self.processor is not None:
            # 使用transformers processor
            inputs = self.processor(
                waveform.numpy(),
                sampling_rate=sampling_rate,
                return_tensors="pt"
            )
            return inputs['input_values']
        else:
            # 简单处理：计算Mel频谱
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sampling_rate,
                n_mels=80
            )
            mel_spec = mel_transform(waveform)  # [n_mels, T]
            return mel_spec.unsqueeze(0)  # [1, n_mels, T]


# 工厂函数
def create_audio_encoder(
    encoder_type: str = 'simple',
    **kwargs
) -> AudioEncoder:
    """
    创建音频编码器的工厂函数

    Args:
        encoder_type: 编码器类型 ('wav2vec2', 'hubert', 'whisper', 'simple')
        **kwargs: 传递给AudioEncoder的其他参数

    Returns:
        AudioEncoder实例
    """
    return AudioEncoder(encoder_type=encoder_type, **kwargs)
