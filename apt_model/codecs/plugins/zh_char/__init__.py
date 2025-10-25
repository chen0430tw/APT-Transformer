#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chinese Character-level Codec Plugin

中文字符级编解码器插件。

基于现有的ChineseTokenizer，提供字符级中文分词。

特点：
- 字符级分词
- 支持简体/繁体中文
- 词汇表自动构建
"""

from apt_model.codecs.plugins.zh_char.codec import ZhCharCodec

__all__ = ['ZhCharCodec']
