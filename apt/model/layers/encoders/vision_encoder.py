#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视觉编码器模块
Vision encoders for image processing in multimodal models
"""

from apt.core.fake_torch import get_torch
torch = get_torch()
from apt.core.fake_torch import get_torch
torch = get_torch()
nn = torch.nn
from typing import Optional, Tuple


class SimpleCNNEncoder(nn.Module):
    """
    简单的CNN图像编码器
    用于快速原型和轻量级应用
    """

    def __init__(self, output_dim: int = 768, input_channels: int = 3):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # 第一层: 3 → 64
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # 第二层: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 第三层: 128 → 256
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.projection = nn.Linear(256, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 图像张量 [B, C, H, W]

        Returns:
            特征张量 [B, output_dim]
        """
        features = self.conv_layers(x)
        features = features.flatten(1)  # [B, 256]
        output = self.projection(features)  # [B, output_dim]

        return output


class VisionEncoder(nn.Module):
    """
    高级视觉编码器
    支持多种预训练模型 (CLIP, ViT等)
    """

    def __init__(
        self,
        encoder_type: str = 'clip',
        model_name: Optional[str] = None,
        output_dim: int = 768,
        freeze_encoder: bool = False
    ):
        """
        Args:
            encoder_type: 编码器类型 ('clip', 'vit', 'resnet', 'simple')
            model_name: 预训练模型名称
            output_dim: 输出维度
            freeze_encoder: 是否冻结编码器参数
        """
        super().__init__()

        self.encoder_type = encoder_type
        self.output_dim = output_dim

        if encoder_type == 'clip':
            self._init_clip_encoder(model_name)
        elif encoder_type == 'vit':
            self._init_vit_encoder(model_name)
        elif encoder_type == 'resnet':
            self._init_resnet_encoder(model_name)
        elif encoder_type == 'simple':
            self.encoder = SimpleCNNEncoder(output_dim=output_dim)
            self.processor = None
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # 冻结encoder参数
        if freeze_encoder and hasattr(self, 'encoder'):
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _init_clip_encoder(self, model_name: Optional[str]):
        """初始化CLIP视觉编码器"""
        try:
            from transformers import CLIPVisionModel, CLIPImageProcessor

            model_name = model_name or 'openai/clip-vit-base-patch32'
            self.encoder = CLIPVisionModel.from_pretrained(model_name)
            self.processor = CLIPImageProcessor.from_pretrained(model_name)

            # 投影到目标维度
            encoder_dim = self.encoder.config.hidden_size
            if encoder_dim != self.output_dim:
                self.projection = nn.Linear(encoder_dim, self.output_dim)
            else:
                self.projection = nn.Identity()

        except ImportError:
            raise ImportError(
                "CLIP requires transformers library. "
                "Install with: pip install transformers"
            )

    def _init_vit_encoder(self, model_name: Optional[str]):
        """初始化ViT编码器"""
        try:
            from transformers import ViTModel, ViTImageProcessor

            model_name = model_name or 'google/vit-base-patch16-224'
            self.encoder = ViTModel.from_pretrained(model_name)
            self.processor = ViTImageProcessor.from_pretrained(model_name)

            # 投影到目标维度
            encoder_dim = self.encoder.config.hidden_size
            if encoder_dim != self.output_dim:
                self.projection = nn.Linear(encoder_dim, self.output_dim)
            else:
                self.projection = nn.Identity()

        except ImportError:
            raise ImportError(
                "ViT requires transformers library. "
                "Install with: pip install transformers"
            )

    def _init_resnet_encoder(self, model_name: Optional[str]):
        """初始化ResNet编码器"""
        try:
            from torchvision import models
            from torchvision.models import ResNet50_Weights

            # 使用预训练的ResNet50
            self.encoder = models.resnet50(weights=ResNet50_Weights.DEFAULT)

            # 移除最后的分类层
            self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])

            # 投影到目标维度
            encoder_dim = 2048  # ResNet50最后一层输出
            self.projection = nn.Linear(encoder_dim, self.output_dim)

            self.processor = None  # ResNet可以直接接受标准化的图像

        except ImportError:
            raise ImportError(
                "ResNet requires torchvision library. "
                "Install with: pip install torchvision"
            )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: 图像张量 [B, C, H, W] 或处理器输出

        Returns:
            特征张量 [B, output_dim]
        """
        if self.encoder_type in ['clip', 'vit']:
            # 使用transformers编码器
            outputs = self.encoder(pixel_values=pixel_values)
            # 使用[CLS] token或池化输出
            features = outputs.last_hidden_state[:, 0]  # [B, encoder_dim]
            features = self.projection(features)  # [B, output_dim]

        elif self.encoder_type == 'resnet':
            # ResNet编码器
            features = self.encoder(pixel_values)  # [B, 2048, 1, 1]
            features = features.flatten(1)  # [B, 2048]
            features = self.projection(features)  # [B, output_dim]

        elif self.encoder_type == 'simple':
            # 简单CNN编码器
            features = self.encoder(pixel_values)  # [B, output_dim]

        return features

    def process_image(self, image):
        """
        处理PIL Image或图像路径

        Args:
            image: PIL Image对象 或 图像路径

        Returns:
            处理后的tensor
        """
        if self.processor is not None:
            # 使用transformers processor
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs['pixel_values']
        else:
            # 使用简单的预处理
            from torchvision import transforms

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            if isinstance(image, str):
                from PIL import Image
                image = Image.open(image).convert('RGB')

            return transform(image).unsqueeze(0)


# 为了向后兼容，创建工厂函数
def create_vision_encoder(
    encoder_type: str = 'simple',
    **kwargs
) -> VisionEncoder:
    """
    创建视觉编码器的工厂函数

    Args:
        encoder_type: 编码器类型 ('clip', 'vit', 'resnet', 'simple')
        **kwargs: 传递给VisionEncoder的其他参数

    Returns:
        VisionEncoder实例
    """
    return VisionEncoder(encoder_type=encoder_type, **kwargs)
