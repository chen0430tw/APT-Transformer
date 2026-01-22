#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model Domain

模型定义域：包含所有模型架构、层、分词器、损失函数和优化器

子模块：
- architectures: 模型架构定义（APT, Claude4, GPT5等）
- layers: 基础层组件（Attention, FFN, Embedding等）
- tokenization: 分词器实现
- losses: 损失函数
- optim: 优化器
- extensions: 核心扩展（RAG, KG, MCP等）

使用示例：
    from apt.model.architectures import APTLargeModel
    from apt.model.tokenization import ChineseTokenizer
    from apt.model.losses import APTLoss
"""

__version__ = '2.0.0-alpha'

# 此模块将在后续PR中从apt_model迁移内容
__all__ = []
