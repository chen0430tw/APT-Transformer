#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Core Codecs

语言编解码器核心系统。

核心组件：
- Codec: 编解码器抽象基类
- CodecConfig: 编解码器配置
- CodecRegistry: 全局注册表
- UnicodeNormalizer: Unicode规范化

使用方式:
    from apt.core.codecs import Codec, register_codec, get_codec

    # 注册codec
    register_codec(MyCodec())

    # 获取codec
    codec = get_codec("my_codec")

    # 编码/解码
    ids = codec.encode("Hello world")
    text = codec.decode(ids)
"""

# API接口
from apt.core.codecs.api import (
    Codec,
    CodecConfig,
)

# 注册表
from apt.core.codecs.registry import (
    CodecRegistry,
    codec_registry,
    register_codec,
    get_codec,
    get_codec_for_language,
    list_codecs,
    list_languages,
)

# Unicode规范化
from apt.core.codecs.unicode_norm import (
    UnicodeNormalizer,
    default_normalizer,
    normalize_unicode,
    nfc,
    nfkc,
    remove_accents,
    to_halfwidth,
)

__all__ = [
    # API
    'Codec',
    'CodecConfig',

    # Registry
    'CodecRegistry',
    'codec_registry',
    'register_codec',
    'get_codec',
    'get_codec_for_language',
    'list_codecs',
    'list_languages',

    # Unicode normalization
    'UnicodeNormalizer',
    'default_normalizer',
    'normalize_unicode',
    'nfc',
    'nfkc',
    'remove_accents',
    'to_halfwidth',
]
