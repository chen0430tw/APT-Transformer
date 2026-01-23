#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
English GPT2 Codec Plugin

英文GPT2编解码器插件。

基于transformers的GPT2Tokenizer，提供BPE分词。

特点：
- BPE分词算法
- 支持英文和其他拉丁语系
- 预训练词汇表（50257 tokens）
"""

from apt.apt_model.codecs.plugins.en_gpt2.codec import EnGPT2Codec

__all__ = ['EnGPT2Codec']
