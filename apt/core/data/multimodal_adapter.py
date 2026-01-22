#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多模态数据适配器
统一 MultimodalDataset 和各训练器的接口
"""

from apt.apt_model.utils.fake_torch import get_torch
torch = get_torch()
from apt.apt_model.utils.fake_torch import get_torch
torch = get_torch()
nn = torch.nn
from typing import Dict, Optional, Any, List
Dataset = torch.utils.data.Dataset
DataLoader = torch.utils.data.DataLoader


class MultimodalDataAdapter(Dataset):
    """
    多模态数据适配器

    将 MultimodalDataset 的输出格式转换为 GPT/Claude trainer 期望的格式

    转换规则:
    - text_input_ids -> input_ids
    - pixel_values -> image_feat (通过 vision_encoder)
    - audio_values -> audio_feat (通过 audio_encoder)
    """

    def __init__(
        self,
        dataset: Dataset,
        vision_encoder: Optional[nn.Module] = None,
        audio_encoder: Optional[nn.Module] = None,
        convert_keys: bool = True,
        device: str = 'cpu'
    ):
        """
        Args:
            dataset: 原始多模态数据集
            vision_encoder: 图像编码器 (将 pixel_values 转为 image_feat)
            audio_encoder: 音频编码器 (将 audio_values 转为 audio_feat)
            convert_keys: 是否转换键名
            device: 编码器运行设备
        """
        self.dataset = dataset
        self.vision_encoder = vision_encoder
        self.audio_encoder = audio_encoder
        self.convert_keys = convert_keys
        self.device = device

        # 将编码器移到设备上并设为评估模式
        if self.vision_encoder is not None:
            self.vision_encoder = self.vision_encoder.to(device)
            self.vision_encoder.eval()

        if self.audio_encoder is not None:
            self.audio_encoder = self.audio_encoder.to(device)
            self.audio_encoder.eval()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取并转换一个样本
        """
        item = self.dataset[idx]
        result = {}

        # 1. 转换文本相关键
        if 'text_input_ids' in item:
            if self.convert_keys:
                result['input_ids'] = item['text_input_ids']
            else:
                result['text_input_ids'] = item['text_input_ids']

            # 保留 attention_mask
            if 'text_attention_mask' in item:
                result['attention_mask'] = item['text_attention_mask']

        # 2. 处理图像: pixel_values -> image_feat
        if 'pixel_values' in item:
            pixel_values = item['pixel_values']

            if self.vision_encoder is not None:
                # 编码为特征向量
                with torch.no_grad():
                    if pixel_values.dim() == 3:
                        pixel_values = pixel_values.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]

                    pixel_values = pixel_values.to(self.device)
                    image_feat = self.vision_encoder(pixel_values)

                    # 如果返回的是字典（如 CLIP）
                    if isinstance(image_feat, dict):
                        image_feat = image_feat.get('pooler_output',
                                                    image_feat.get('last_hidden_state'))

                    # 确保是 [D] 或 [1, D]
                    if image_feat.dim() == 3:  # [B, L, D]
                        image_feat = image_feat.mean(dim=1)  # Pool

                    image_feat = image_feat.squeeze(0).cpu()  # [D]

                result['image_feat'] = image_feat
            else:
                # 没有编码器，直接保留原始像素值
                result['pixel_values'] = pixel_values

        # 3. 处理音频: audio_values -> audio_feat
        if 'audio_values' in item:
            audio_values = item['audio_values']

            if self.audio_encoder is not None:
                # 编码为特征向量
                with torch.no_grad():
                    if audio_values.dim() == 1:
                        audio_values = audio_values.unsqueeze(0)  # [T] -> [1,T]
                    elif audio_values.dim() == 2 and audio_values.size(0) > 1:
                        # 可能是 [n_mels, T]，不需要额外维度
                        pass

                    audio_values = audio_values.to(self.device)
                    audio_feat = self.audio_encoder(audio_values)

                    # 如果返回的是字典
                    if isinstance(audio_feat, dict):
                        audio_feat = audio_feat.get('pooler_output',
                                                    audio_feat.get('last_hidden_state'))

                    # 确保是 [D]
                    if audio_feat.dim() == 3:  # [B, L, D]
                        audio_feat = audio_feat.mean(dim=1)  # Pool

                    audio_feat = audio_feat.squeeze(0).cpu()  # [D]

                result['audio_feat'] = audio_feat
            else:
                # 没有编码器，直接保留原始音频值
                result['audio_values'] = audio_values

        # 4. 保留标签
        if 'labels' in item:
            result['labels'] = item['labels']
        elif 'label' in item:
            result['labels'] = item['label']

        # 5. 保留元数据
        if 'metadata' in item:
            result['metadata'] = item['metadata']

        return result


class SimpleVisionEncoder(nn.Module):
    """
    简单的视觉编码器
    将图像像素值编码为特征向量
    """

    def __init__(self, input_channels: int = 3, output_dim: int = 1024):
        super().__init__()

        # 简单的 CNN 编码器
        self.encoder = nn.Sequential(
            # [B, 3, 224, 224] -> [B, 64, 112, 112]
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # [B, 64, 56, 56] -> [B, 128, 28, 28]
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # [B, 128, 14, 14] -> [B, 256, 7, 7]
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # [B, 256, 1, 1]
        )

        # 投影到目标维度
        self.projection = nn.Linear(256, output_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, C, H, W]

        Returns:
            features: [B, D]
        """
        x = self.encoder(pixel_values)  # [B, 256, 1, 1]
        x = x.flatten(1)  # [B, 256]
        x = self.projection(x)  # [B, D]
        return x


class SimpleAudioEncoder(nn.Module):
    """
    简单的音频编码器
    将音频波形/频谱编码为特征向量
    """

    def __init__(self, input_dim: int = 80, output_dim: int = 512):
        super().__init__()

        # 假设输入是 Mel 频谱: [B, n_mels, T]
        self.encoder = nn.Sequential(
            # [B, 80, T] -> [B, 128, T/2]
            nn.Conv1d(input_dim, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # [B, 128, T/2] -> [B, 256, T/4]
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            # [B, 256, T/4] -> [B, 256, T/8]
            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            # Global pooling
            nn.AdaptiveAvgPool1d(1)  # [B, 256, 1]
        )

        # 投影到目标维度
        self.projection = nn.Linear(256, output_dim)

    def forward(self, audio_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_values: [B, n_mels, T] 或 [B, T]

        Returns:
            features: [B, D]
        """
        # 如果是1D波形，转换为2D
        if audio_values.dim() == 2:
            # 简单处理：假设是波形，添加通道维度
            audio_values = audio_values.unsqueeze(1)  # [B, 1, T]
            # 调整编码器输入维度
            if self.encoder[0].in_channels != 1:
                # 需要重新初始化第一层
                pass  # 简化处理，假设输入格式正确

        x = self.encoder(audio_values)  # [B, 256, 1]
        x = x.squeeze(-1)  # [B, 256]
        x = self.projection(x)  # [B, D]
        return x


def create_multimodal_adapter(
    dataset: Dataset,
    use_vision: bool = True,
    use_audio: bool = True,
    vision_encoder_type: str = 'simple',
    audio_encoder_type: str = 'simple',
    vision_output_dim: int = 1024,
    audio_output_dim: int = 512,
    device: str = 'cpu'
) -> MultimodalDataAdapter:
    """
    创建多模态数据适配器的工厂函数

    Args:
        dataset: 原始数据集
        use_vision: 是否使用视觉编码器
        use_audio: 是否使用音频编码器
        vision_encoder_type: 视觉编码器类型 ('simple', 'clip', 'vit')
        audio_encoder_type: 音频编码器类型 ('simple', 'wav2vec2', 'hubert')
        vision_output_dim: 视觉特征维度
        audio_output_dim: 音频特征维度
        device: 设备

    Returns:
        MultimodalDataAdapter 实例
    """
    vision_encoder = None
    audio_encoder = None

    # 创建视觉编码器
    if use_vision:
        if vision_encoder_type == 'simple':
            vision_encoder = SimpleVisionEncoder(output_dim=vision_output_dim)

        elif vision_encoder_type == 'clip':
            try:
                from transformers import CLIPVisionModel
                vision_encoder = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32')
            except ImportError:
                print("⚠️ transformers 未安装，使用简单编码器")
                vision_encoder = SimpleVisionEncoder(output_dim=vision_output_dim)

        elif vision_encoder_type == 'vit':
            try:
                from transformers import ViTModel
                vision_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
            except ImportError:
                print("⚠️ transformers 未安装，使用简单编码器")
                vision_encoder = SimpleVisionEncoder(output_dim=vision_output_dim)

    # 创建音频编码器
    if use_audio:
        if audio_encoder_type == 'simple':
            audio_encoder = SimpleAudioEncoder(output_dim=audio_output_dim)

        elif audio_encoder_type == 'wav2vec2':
            try:
                from transformers import Wav2Vec2Model
                audio_encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
            except ImportError:
                print("⚠️ transformers 未安装，使用简单编码器")
                audio_encoder = SimpleAudioEncoder(output_dim=audio_output_dim)

        elif audio_encoder_type == 'hubert':
            try:
                from transformers import HubertModel
                audio_encoder = HubertModel.from_pretrained('facebook/hubert-base-ls960')
            except ImportError:
                print("⚠️ transformers 未安装，使用简单编码器")
                audio_encoder = SimpleAudioEncoder(output_dim=audio_output_dim)

    return MultimodalDataAdapter(
        dataset=dataset,
        vision_encoder=vision_encoder,
        audio_encoder=audio_encoder,
        device=device
    )


def create_multimodal_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    use_vision: bool = True,
    use_audio: bool = True,
    **adapter_kwargs
) -> DataLoader:
    """
    创建多模态数据加载器

    Args:
        dataset: 原始数据集
        batch_size: 批大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        use_vision: 是否使用视觉编码
        use_audio: 是否使用音频编码
        **adapter_kwargs: 传递给适配器的参数

    Returns:
        DataLoader 实例
    """
    # 创建适配器
    adapter = create_multimodal_adapter(
        dataset=dataset,
        use_vision=use_vision,
        use_audio=use_audio,
        **adapter_kwargs
    )

    # 创建数据加载器
    dataloader = DataLoader(
        adapter,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader
