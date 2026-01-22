#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多模态APT模型推理示例
Multimodal APT model inference examples
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from typing import Optional, Union, List
from PIL import Image
import logging

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from apt_model.config.config import APTConfig
from apt_model.config.multimodal_config import MultimodalConfig
from apt.core.modeling.multimodal_model import MultimodalAPTModel, create_multimodal_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultimodalInference:
    """多模态推理器"""

    def __init__(
        self,
        model: MultimodalAPTModel,
        tokenizer=None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Args:
            model: 多模态模型
            tokenizer: 文本tokenizer
            device: 设备
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def predict_text_only(
        self,
        text: str,
        max_length: int = 50
    ) -> str:
        """
        仅文本输入的预测

        Args:
            text: 输入文本
            max_length: 最大生成长度

        Returns:
            生成的文本
        """
        # Tokenize
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for text inference")

        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True
        )

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # 生成
        generated_ids = self.model.generate(
            input_ids=input_ids,
            max_length=max_length
        )

        # 解码
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return generated_text

    @torch.no_grad()
    def predict_text_image(
        self,
        text: str,
        image_path: Union[str, Image.Image],
        max_length: int = 50
    ) -> str:
        """
        文本+图像输入的预测

        Args:
            text: 输入文本
            image_path: 图像路径或PIL Image
            max_length: 最大生成长度

        Returns:
            生成的文本
        """
        # 处理文本
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for text inference")

        text_inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True
        )

        input_ids = text_inputs['input_ids'].to(self.device)

        # 处理图像
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path

        pixel_values = self.model.vision_encoder.process_image(image)
        pixel_values = pixel_values.to(self.device)

        # 生成
        generated_ids = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_length=max_length
        )

        # 解码
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return generated_text

    @torch.no_grad()
    def predict_text_audio(
        self,
        text: str,
        audio_path: str,
        max_length: int = 50
    ) -> str:
        """
        文本+音频输入的预测

        Args:
            text: 输入文本
            audio_path: 音频文件路径
            max_length: 最大生成长度

        Returns:
            生成的文本
        """
        # 处理文本
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for text inference")

        text_inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True
        )

        input_ids = text_inputs['input_ids'].to(self.device)

        # 处理音频
        audio_values = self.model.audio_encoder.process_audio(audio_path)
        audio_values = audio_values.to(self.device)

        # 生成
        generated_ids = self.model.generate(
            input_ids=input_ids,
            audio_values=audio_values,
            max_length=max_length
        )

        # 解码
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return generated_text

    @torch.no_grad()
    def predict_all_modalities(
        self,
        text: str,
        image_path: Union[str, Image.Image],
        audio_path: str,
        max_length: int = 50
    ) -> str:
        """
        文本+图像+音频输入的预测

        Args:
            text: 输入文本
            image_path: 图像路径或PIL Image
            audio_path: 音频文件路径
            max_length: 最大生成长度

        Returns:
            生成的文本
        """
        # 处理文本
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for text inference")

        text_inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True
        )

        input_ids = text_inputs['input_ids'].to(self.device)

        # 处理图像
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path

        pixel_values = self.model.vision_encoder.process_image(image)
        pixel_values = pixel_values.to(self.device)

        # 处理音频
        audio_values = self.model.audio_encoder.process_audio(audio_path)
        audio_values = audio_values.to(self.device)

        # 生成
        generated_ids = self.model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            audio_values=audio_values,
            max_length=max_length
        )

        # 解码
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return generated_text

    @torch.no_grad()
    def extract_features(
        self,
        text: Optional[str] = None,
        image_path: Optional[Union[str, Image.Image]] = None,
        audio_path: Optional[str] = None
    ) -> dict:
        """
        提取多模态特征（不生成文本）

        Args:
            text: 输入文本
            image_path: 图像路径或PIL Image
            audio_path: 音频文件路径

        Returns:
            特征字典
        """
        features = {}

        # 文本特征
        if text is not None and self.tokenizer is not None:
            text_inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True
            )
            input_ids = text_inputs['input_ids'].to(self.device)
            attention_mask = text_inputs['attention_mask'].to(self.device)

            text_features = self.model.encode_text(input_ids, attention_mask)
            features['text'] = text_features.cpu()

        # 视觉特征
        if image_path is not None:
            if isinstance(image_path, str):
                image = Image.open(image_path).convert('RGB')
            else:
                image = image_path

            pixel_values = self.model.vision_encoder.process_image(image)
            pixel_values = pixel_values.to(self.device)

            vision_features = self.model.encode_vision(pixel_values)
            features['vision'] = vision_features.cpu()

        # 音频特征
        if audio_path is not None:
            audio_values = self.model.audio_encoder.process_audio(audio_path)
            audio_values = audio_values.to(self.device)

            audio_features = self.model.encode_audio(audio_values)
            features['audio'] = audio_features.cpu()

        return features

    @torch.no_grad()
    def compute_similarity(
        self,
        text: Optional[str] = None,
        image_path: Optional[Union[str, Image.Image]] = None,
        audio_path: Optional[str] = None
    ) -> dict:
        """
        计算模态间的相似度

        Args:
            text: 输入文本
            image_path: 图像路径或PIL Image
            audio_path: 音频文件路径

        Returns:
            相似度字典
        """
        # 提取特征
        features = self.extract_features(text, image_path, audio_path)

        similarities = {}

        # 文本-视觉相似度
        if 'text' in features and 'vision' in features:
            text_feat = features['text'].mean(dim=1)  # [B, D]
            vision_feat = features['vision']  # [B, D]

            # 余弦相似度
            similarity = torch.nn.functional.cosine_similarity(text_feat, vision_feat, dim=-1)
            similarities['text_vision'] = similarity.item()

        # 文本-音频相似度
        if 'text' in features and 'audio' in features:
            text_feat = features['text'].mean(dim=1)  # [B, D]
            audio_feat = features['audio']  # [B, D]

            similarity = torch.nn.functional.cosine_similarity(text_feat, audio_feat, dim=-1)
            similarities['text_audio'] = similarity.item()

        # 视觉-音频相似度
        if 'vision' in features and 'audio' in features:
            vision_feat = features['vision']  # [B, D]
            audio_feat = features['audio']  # [B, D]

            similarity = torch.nn.functional.cosine_similarity(vision_feat, audio_feat, dim=-1)
            similarities['vision_audio'] = similarity.item()

        return similarities


def demo_text_only():
    """仅文本的示例"""
    logger.info("=== Text-Only Inference Demo ===")

    # 创建模型（简化示例）
    config = APTConfig(d_model=768, num_layers=6, num_attention_heads=12, vocab_size=50000)
    multimodal_config = MultimodalConfig(enable_image=False, enable_audio=False)

    model = create_multimodal_model(
        config=config,
        multimodal_config=multimodal_config
    )

    # 创建推理器
    # 注意：这里需要实际的tokenizer
    inference = MultimodalInference(model, tokenizer=None)

    logger.info("Text-only model created successfully")


def demo_text_image():
    """文本+图像的示例"""
    logger.info("=== Text-Image Inference Demo ===")

    config = APTConfig(d_model=768, num_layers=6, num_attention_heads=12, vocab_size=50000)
    multimodal_config = MultimodalConfig(enable_image=True, enable_audio=False)

    model = create_multimodal_model(
        config=config,
        multimodal_config=multimodal_config,
        vision_encoder='simple'
    )

    inference = MultimodalInference(model, tokenizer=None)

    logger.info("Text-image model created successfully")

    # 如果有实际的图像文件，可以这样使用：
    # result = inference.predict_text_image(
    #     text="Describe this image:",
    #     image_path="path/to/image.jpg"
    # )


def demo_all_modalities():
    """所有模态的示例"""
    logger.info("=== All Modalities Inference Demo ===")

    config = APTConfig(d_model=768, num_layers=6, num_attention_heads=12, vocab_size=50000)
    multimodal_config = MultimodalConfig(enable_image=True, enable_audio=True)

    model = create_multimodal_model(
        config=config,
        multimodal_config=multimodal_config,
        vision_encoder='simple',
        audio_encoder='simple',
        fusion_method='cross_attention'
    )

    inference = MultimodalInference(model, tokenizer=None)

    logger.info("Multi-modal model created successfully")

    # 如果有实际的图像和音频文件，可以这样使用：
    # result = inference.predict_all_modalities(
    #     text="Describe what you see and hear:",
    #     image_path="path/to/image.jpg",
    #     audio_path="path/to/audio.wav"
    # )


def demo_feature_extraction():
    """特征提取示例"""
    logger.info("=== Feature Extraction Demo ===")

    config = APTConfig(d_model=768, num_layers=6, num_attention_heads=12, vocab_size=50000)
    multimodal_config = MultimodalConfig(enable_image=True, enable_audio=True)

    model = create_multimodal_model(
        config=config,
        multimodal_config=multimodal_config,
        vision_encoder='simple',
        audio_encoder='simple'
    )

    inference = MultimodalInference(model, tokenizer=None)

    logger.info("Model created for feature extraction")

    # 示例：提取特征
    # features = inference.extract_features(
    #     text="Sample text",
    #     image_path="path/to/image.jpg",
    #     audio_path="path/to/audio.wav"
    # )
    #
    # logger.info(f"Extracted features: {features.keys()}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Multimodal Inference Examples')
    parser.add_argument('--demo', type=str, default='all',
                        choices=['text', 'text_image', 'all_modalities', 'features', 'all'],
                        help='Demo to run')

    args = parser.parse_args()

    if args.demo == 'text' or args.demo == 'all':
        demo_text_only()
        print()

    if args.demo == 'text_image' or args.demo == 'all':
        demo_text_image()
        print()

    if args.demo == 'all_modalities' or args.demo == 'all':
        demo_all_modalities()
        print()

    if args.demo == 'features' or args.demo == 'all':
        demo_feature_extraction()
        print()

    logger.info("All demos completed successfully!")


if __name__ == '__main__':
    main()
