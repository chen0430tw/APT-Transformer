#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model Codecs

语言编解码器插件系统。

自动加载和注册所有可用的codec插件。

可用插件：
- zh_char: 中文字符级分词
- en_gpt2: 英文GPT2 BPE分词
- ja_mecab: 日文MeCab分词（占位）

使用方式:
    from apt.apt_model.codecs import get_codec_for_language, list_available_codecs

    # 获取中文codec
    codec = get_codec_for_language("zh")

    # 获取英文codec
    codec = get_codec_for_language("en")

    # 列出所有可用codec
    codecs = list_available_codecs()
"""

import logging
from typing import Optional, List

from apt.core.codecs import (
    codec_registry,
    register_codec,
    get_codec,
    get_codec_for_language as _get_codec_for_language,
    list_codecs,
)

logger = logging.getLogger(__name__)


# ============================================================================
# 自动注册所有插件
# ============================================================================

def _register_all_plugins():
    """
    自动注册所有可用的codec插件

    使用延迟加载策略：
    - 只注册工厂函数，不立即实例化
    - 第一次使用时才加载实际codec
    """
    # 中文插件
    def create_zh_char():
        from apt.apt_model.codecs.plugins.zh_char import ZhCharCodec
        from apt.core.codecs import CodecConfig
        config = CodecConfig(name="zh_char", vocab_size=21128)
        return ZhCharCodec(config)

    register_codec(factory=create_zh_char, name="zh_char")

    # 英文插件
    def create_en_gpt2():
        from apt.apt_model.codecs.plugins.en_gpt2 import EnGPT2Codec
        from apt.core.codecs import CodecConfig
        config = CodecConfig(name="en_gpt2", vocab_size=50257)
        return EnGPT2Codec(config)

    register_codec(factory=create_en_gpt2, name="en_gpt2")

    # 日文插件（占位）
    def create_ja_mecab():
        from apt.apt_model.codecs.plugins.ja_mecab import JaMecabCodec
        from apt.core.codecs import CodecConfig
        config = CodecConfig(name="ja_mecab", vocab_size=32000)
        return JaMecabCodec(config)

    register_codec(factory=create_ja_mecab, name="ja_mecab")

    logger.info("Registered 3 codec plugins: zh_char, en_gpt2, ja_mecab")


# 模块导入时自动注册
_register_all_plugins()


# ============================================================================
# 便捷函数
# ============================================================================

def get_codec_for_language(
    language: str,
    prefer: Optional[str] = None,
    fallback: Optional[str] = "en_gpt2"
) -> Optional['Codec']:
    """
    根据语言代码获取codec

    参数:
        language: 语言代码 (如 'zh', 'ja', 'en')
        prefer: 首选codec名称
        fallback: 回退codec名称（默认en_gpt2）

    返回:
        Codec实例

    示例:
        >>> # 获取中文codec
        >>> codec = get_codec_for_language("zh")
        >>>
        >>> # 获取英文codec
        >>> codec = get_codec_for_language("en")
        >>>
        >>> # 获取日文codec，找不到用英文
        >>> codec = get_codec_for_language("ja", fallback="en_gpt2")
    """
    return _get_codec_for_language(language, prefer, fallback)


def list_available_codecs(language: Optional[str] = None) -> List[str]:
    """
    列出所有可用的codec

    参数:
        language: 可选，按语言过滤

    返回:
        Codec名称列表

    示例:
        >>> # 列出所有codec
        >>> all_codecs = list_available_codecs()
        >>> print(all_codecs)
        ['zh_char', 'en_gpt2', 'ja_mecab']
        >>>
        >>> # 列出支持中文的codec
        >>> zh_codecs = list_available_codecs(language="zh")
        >>> print(zh_codecs)
        ['zh_char']
    """
    return list_codecs(language)


def get_default_codec() -> 'Codec':
    """
    获取默认codec（英文GPT2）

    返回:
        Codec实例
    """
    codec = get_codec("en_gpt2")
    if codec is None:
        raise RuntimeError("Default codec 'en_gpt2' not available")
    return codec


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    # 从core导出
    'codec_registry',
    'register_codec',
    'get_codec',
    'list_codecs',

    # 本模块函数
    'get_codec_for_language',
    'list_available_codecs',
    'get_default_codec',
]
