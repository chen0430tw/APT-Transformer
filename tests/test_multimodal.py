#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多模态组件单元测试
Unit tests for multimodal components
"""

import os
import sys
import unittest
import torch
import torch.nn as nn
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from apt_model.config.config import APTConfig
from apt_model.config.multimodal_config import MultimodalConfig
from apt_model.modeling.multimodal_model import MultimodalAPTModel, create_multimodal_model
from apt_model.modeling.encoders.vision_encoder import VisionEncoder, SimpleCNNEncoder
from apt_model.modeling.encoders.audio_encoder import AudioEncoder, SimpleAudioEncoder
from apt_model.modeling.encoders.cross_modal_attention import (
    CrossModalAttention,
    BiDirectionalCrossAttention,
    MultiModalFusionLayer,
    TriModalFusionLayer
)


class TestVisionEncoder(unittest.TestCase):
    """测试视觉编码器"""

    def setUp(self):
        """设置测试环境"""
        self.batch_size = 2
        self.image_size = (224, 224)
        self.output_dim = 768

    def test_simple_cnn_encoder(self):
        """测试简单CNN编码器"""
        encoder = SimpleCNNEncoder(output_dim=self.output_dim)

        # 创建输入
        x = torch.randn(self.batch_size, 3, *self.image_size)

        # 前向传播
        output = encoder(x)

        # 检查输出形状
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_vision_encoder_simple(self):
        """测试简单视觉编码器"""
        encoder = VisionEncoder(
            encoder_type='simple',
            output_dim=self.output_dim
        )

        # 创建输入
        pixel_values = torch.randn(self.batch_size, 3, *self.image_size)

        # 前向传播
        features = encoder(pixel_values)

        # 检查输出形状
        self.assertEqual(features.shape, (self.batch_size, self.output_dim))

    def test_vision_encoder_invalid_type(self):
        """测试无效的编码器类型"""
        with self.assertRaises(ValueError):
            VisionEncoder(encoder_type='invalid_type')


class TestAudioEncoder(unittest.TestCase):
    """测试音频编码器"""

    def setUp(self):
        """设置测试环境"""
        self.batch_size = 2
        self.n_mels = 80
        self.time_steps = 100
        self.output_dim = 768

    def test_simple_audio_encoder(self):
        """测试简单音频编码器"""
        encoder = SimpleAudioEncoder(output_dim=self.output_dim)

        # 创建输入 [B, n_mels, T]
        x = torch.randn(self.batch_size, self.n_mels, self.time_steps)

        # 前向传播
        output = encoder(x)

        # 检查输出形状
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_simple_audio_encoder_transpose_input(self):
        """测试转置输入格式 [B, T, n_mels]"""
        encoder = SimpleAudioEncoder(output_dim=self.output_dim)

        # 创建转置输入
        x = torch.randn(self.batch_size, self.time_steps, self.n_mels)

        # 前向传播
        output = encoder(x)

        # 检查输出形状
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))

    def test_audio_encoder_simple(self):
        """测试简单音频编码器"""
        encoder = AudioEncoder(
            encoder_type='simple',
            output_dim=self.output_dim
        )

        # 创建输入
        input_values = torch.randn(self.batch_size, self.n_mels, self.time_steps)

        # 前向传播
        features = encoder(input_values)

        # 检查输出形状
        self.assertEqual(features.shape, (self.batch_size, self.output_dim))

    def test_audio_encoder_invalid_type(self):
        """测试无效的编码器类型"""
        with self.assertRaises(ValueError):
            AudioEncoder(encoder_type='invalid_type')


class TestCrossModalAttention(unittest.TestCase):
    """测试跨模态注意力"""

    def setUp(self):
        """设置测试环境"""
        self.batch_size = 2
        self.seq_len_q = 10
        self.seq_len_kv = 8
        self.embed_dim = 768
        self.num_heads = 12

    def test_cross_modal_attention_forward(self):
        """测试跨模态注意力前向传播"""
        attention = CrossModalAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads
        )

        # 创建输入
        query = torch.randn(self.batch_size, self.seq_len_q, self.embed_dim)
        key = torch.randn(self.batch_size, self.seq_len_kv, self.embed_dim)
        value = torch.randn(self.batch_size, self.seq_len_kv, self.embed_dim)

        # 前向传播
        output, attention_weights = attention(query, key, value)

        # 检查输出形状
        self.assertEqual(output.shape, (self.batch_size, self.seq_len_q, self.embed_dim))
        self.assertEqual(
            attention_weights.shape,
            (self.batch_size, self.num_heads, self.seq_len_q, self.seq_len_kv)
        )

    def test_cross_modal_attention_with_mask(self):
        """测试带掩码的跨模态注意力"""
        attention = CrossModalAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads
        )

        # 创建输入
        query = torch.randn(self.batch_size, self.seq_len_q, self.embed_dim)
        key = torch.randn(self.batch_size, self.seq_len_kv, self.embed_dim)
        value = torch.randn(self.batch_size, self.seq_len_kv, self.embed_dim)

        # 创建注意力掩码
        attention_mask = torch.zeros(self.batch_size, self.seq_len_q, self.seq_len_kv)

        # 前向传播
        output, attention_weights = attention(query, key, value, attention_mask)

        # 检查输出形状
        self.assertEqual(output.shape, (self.batch_size, self.seq_len_q, self.embed_dim))

    def test_bidirectional_cross_attention(self):
        """测试双向跨模态注意力"""
        attention = BiDirectionalCrossAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads
        )

        # 创建输入
        modal1 = torch.randn(self.batch_size, self.seq_len_q, self.embed_dim)
        modal2 = torch.randn(self.batch_size, self.seq_len_kv, self.embed_dim)

        # 前向传播
        enhanced_modal1, enhanced_modal2 = attention(modal1, modal2)

        # 检查输出形状
        self.assertEqual(enhanced_modal1.shape, modal1.shape)
        self.assertEqual(enhanced_modal2.shape, modal2.shape)


class TestMultiModalFusion(unittest.TestCase):
    """测试多模态融合"""

    def setUp(self):
        """设置测试环境"""
        self.batch_size = 2
        self.seq_len1 = 10
        self.seq_len2 = 8
        self.embed_dim = 768
        self.num_heads = 12

    def test_fusion_attention(self):
        """测试注意力融合"""
        fusion = MultiModalFusionLayer(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            fusion_method='attention'
        )

        # 创建输入
        modal1 = torch.randn(self.batch_size, self.seq_len1, self.embed_dim)
        modal2 = torch.randn(self.batch_size, self.seq_len2, self.embed_dim)

        # 融合
        fused = fusion(modal1, modal2)

        # 检查输出形状
        self.assertEqual(len(fused.shape), 3)
        self.assertEqual(fused.size(0), self.batch_size)
        self.assertEqual(fused.size(-1), self.embed_dim)

    def test_fusion_concatenate(self):
        """测试拼接融合"""
        fusion = MultiModalFusionLayer(
            embed_dim=self.embed_dim,
            fusion_method='concatenate'
        )

        modal1 = torch.randn(self.batch_size, self.seq_len1, self.embed_dim)
        modal2 = torch.randn(self.batch_size, self.seq_len2, self.embed_dim)

        fused = fusion(modal1, modal2)

        self.assertEqual(len(fused.shape), 3)
        self.assertEqual(fused.size(-1), self.embed_dim)

    def test_fusion_add(self):
        """测试相加融合"""
        fusion = MultiModalFusionLayer(
            embed_dim=self.embed_dim,
            fusion_method='add'
        )

        modal1 = torch.randn(self.batch_size, self.seq_len1, self.embed_dim)
        modal2 = torch.randn(self.batch_size, self.seq_len1, self.embed_dim)  # 相同长度

        fused = fusion(modal1, modal2)

        self.assertEqual(fused.shape, modal1.shape)

    def test_fusion_gated(self):
        """测试门控融合"""
        fusion = MultiModalFusionLayer(
            embed_dim=self.embed_dim,
            fusion_method='gated'
        )

        modal1 = torch.randn(self.batch_size, self.seq_len1, self.embed_dim)
        modal2 = torch.randn(self.batch_size, self.seq_len2, self.embed_dim)

        fused = fusion(modal1, modal2)

        self.assertEqual(len(fused.shape), 3)
        self.assertEqual(fused.size(-1), self.embed_dim)

    def test_tri_modal_fusion(self):
        """测试三模态融合"""
        fusion = TriModalFusionLayer(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads
        )

        # 创建三个模态的输入
        text = torch.randn(self.batch_size, 10, self.embed_dim)
        vision = torch.randn(self.batch_size, 8, self.embed_dim)
        audio = torch.randn(self.batch_size, 6, self.embed_dim)

        # 融合
        fused = fusion(text, vision, audio)

        # 检查输出形状 [B, D]
        self.assertEqual(fused.shape, (self.batch_size, self.embed_dim))


class TestMultimodalAPTModel(unittest.TestCase):
    """测试多模态APT模型"""

    def setUp(self):
        """设置测试环境"""
        self.batch_size = 2
        self.seq_len = 10
        self.vocab_size = 1000
        self.image_size = (224, 224)
        self.audio_time_steps = 100

        # 创建配置
        self.config = APTConfig(
            vocab_size=self.vocab_size,
            d_model=256,  # 减小以加快测试
            num_layers=2,
            num_attention_heads=4,
            max_position_embeddings=512
        )

        self.multimodal_config = MultimodalConfig(
            enable_image=True,
            enable_audio=True
        )

    def test_model_creation(self):
        """测试模型创建"""
        model = create_multimodal_model(
            config=self.config,
            multimodal_config=self.multimodal_config,
            vision_encoder='simple',
            audio_encoder='simple'
        )

        self.assertIsInstance(model, MultimodalAPTModel)
        self.assertIsNotNone(model.vision_encoder)
        self.assertIsNotNone(model.audio_encoder)

    def test_text_only_forward(self):
        """测试仅文本前向传播"""
        multimodal_config = MultimodalConfig(enable_image=False, enable_audio=False)

        model = create_multimodal_model(
            config=self.config,
            multimodal_config=multimodal_config
        )

        # 创建输入
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))

        # 前向传播
        outputs = model(input_ids=input_ids, return_dict=True)

        # 检查输出
        self.assertIn('logits', outputs)
        self.assertEqual(outputs['logits'].size(0), self.batch_size)
        self.assertEqual(outputs['logits'].size(1), self.seq_len)

    def test_text_image_forward(self):
        """测试文本+图像前向传播"""
        multimodal_config = MultimodalConfig(enable_image=True, enable_audio=False)

        model = create_multimodal_model(
            config=self.config,
            multimodal_config=multimodal_config,
            vision_encoder='simple'
        )

        # 创建输入
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        pixel_values = torch.randn(self.batch_size, 3, *self.image_size)

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            return_dict=True
        )

        # 检查输出
        self.assertIn('logits', outputs)
        self.assertIn('text_features', outputs)
        self.assertIn('vision_features', outputs)
        self.assertIsNotNone(outputs['vision_features'])

    def test_text_audio_forward(self):
        """测试文本+音频前向传播"""
        multimodal_config = MultimodalConfig(enable_image=False, enable_audio=True)

        model = create_multimodal_model(
            config=self.config,
            multimodal_config=multimodal_config,
            audio_encoder='simple'
        )

        # 创建输入
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        audio_values = torch.randn(self.batch_size, 80, self.audio_time_steps)

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            audio_values=audio_values,
            return_dict=True
        )

        # 检查输出
        self.assertIn('logits', outputs)
        self.assertIn('audio_features', outputs)
        self.assertIsNotNone(outputs['audio_features'])

    def test_all_modalities_forward(self):
        """测试所有模态前向传播"""
        model = create_multimodal_model(
            config=self.config,
            multimodal_config=self.multimodal_config,
            vision_encoder='simple',
            audio_encoder='simple',
            fusion_method='cross_attention'
        )

        # 创建输入
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        pixel_values = torch.randn(self.batch_size, 3, *self.image_size)
        audio_values = torch.randn(self.batch_size, 80, self.audio_time_steps)

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            audio_values=audio_values,
            return_dict=True
        )

        # 检查输出
        self.assertIn('logits', outputs)
        self.assertIn('text_features', outputs)
        self.assertIn('vision_features', outputs)
        self.assertIn('audio_features', outputs)
        self.assertIn('fused_features', outputs)

    def test_model_with_labels(self):
        """测试带标签的训练"""
        model = create_multimodal_model(
            config=self.config,
            multimodal_config=self.multimodal_config,
            vision_encoder='simple',
            audio_encoder='simple'
        )

        # 创建输入和标签
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        labels = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))

        # 前向传播
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            return_dict=True
        )

        # 检查损失
        self.assertIn('loss', outputs)
        self.assertIsNotNone(outputs['loss'])
        self.assertTrue(outputs['loss'].item() > 0)

    def test_different_fusion_methods(self):
        """测试不同的融合方法"""
        fusion_methods = ['cross_attention', 'tri_modal', 'concatenate', 'add', 'gated']

        for fusion_method in fusion_methods:
            with self.subTest(fusion_method=fusion_method):
                model = create_multimodal_model(
                    config=self.config,
                    multimodal_config=self.multimodal_config,
                    vision_encoder='simple',
                    audio_encoder='simple',
                    fusion_method=fusion_method
                )

                # 创建输入
                input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
                pixel_values = torch.randn(self.batch_size, 3, *self.image_size)
                audio_values = torch.randn(self.batch_size, 80, self.audio_time_steps)

                # 前向传播
                outputs = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    audio_values=audio_values,
                    return_dict=True
                )

                # 检查输出存在
                self.assertIn('logits', outputs)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestVisionEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestAudioEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestCrossModalAttention))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiModalFusion))
    suite.addTests(loader.loadTestsFromTestCase(TestMultimodalAPTModel))

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == '__main__':
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
