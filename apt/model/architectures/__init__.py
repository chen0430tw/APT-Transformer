#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Architectures

包含所有模型架构定义：
- APTLargeModel: APT核心模型
- MultimodalAPTModel: 多模态模型
- Claude4Model, GPT5Model等特定模型实现
"""

# 核心模型
from apt.model.architectures.apt_model import APTLargeModel
from apt.model.architectures.multimodal_model import MultimodalAPTModel
from apt.model.architectures.elastic_transformer import ElasticTransformer

# 特定模型实现
from apt.model.architectures.claude4_model import Claude4Model
from apt.model.architectures.gpt5_model import GPT5Model
from apt.model.architectures.gpt4o_model import GPT4oModel
from apt.model.architectures.gpto3_model import GPTo3Model
from apt.model.architectures.vft_tva_model import VFTTVAModel

__all__ = [
    # Core models
    'APTLargeModel',
    'MultimodalAPTModel',
    'ElasticTransformer',
    # Specific models
    'Claude4Model',
    'GPT5Model',
    'GPT4oModel',
    'GPTo3Model',
    'VFTTVAModel',
]
