#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Core Codec API

定义语言编解码器的统一接口。所有语言插件必须实现此接口。

核心原则：
- 接口简单：encode/decode/tokenize 三个核心方法
- 语言无关：适用于所有语言和分词策略
- 可扩展：插件可以添加额外方法和属性
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class CodecConfig:
    """
    Codec配置

    通用配置项，各插件可以扩展
    """
    name: str                           # Codec名称，如 "zh_char", "ja_mecab"
    vocab_size: int = 50000            # 词汇表大小
    max_length: int = 512              # 最大序列长度
    special_tokens: Dict[str, str] = None  # 特殊标记
    lowercase: bool = False            # 是否小写化
    strip_accents: bool = False        # 是否去除重音
    extra_config: Dict[str, Any] = None    # 扩展配置

    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = {
                'pad': '<|pad|>',
                'unk': '<|unk|>',
                'bos': '<|endoftext|>',
                'eos': '<|endoftext|>',
            }
        if self.extra_config is None:
            self.extra_config = {}


class Codec(ABC):
    """
    语言编解码器抽象基类

    所有语言插件必须继承此类并实现所有抽象方法。

    属性:
        name: Codec名称
        langs: 支持的语言代码列表 (如 ['zh', 'zh-cn'])
        vocab_size: 词汇表大小
        config: Codec配置

    核心方法:
        encode: 文本 → token IDs
        decode: token IDs → 文本
        tokenize: 文本 → tokens

    示例:
        >>> codec = get_codec("zh_char")
        >>> ids = codec.encode("你好世界")
        >>> text = codec.decode(ids)
        >>> tokens = codec.tokenize("你好世界")
    """

    def __init__(self, config: CodecConfig):
        """
        初始化Codec

        参数:
            config: Codec配置
        """
        self.config = config
        self.name = config.name
        self.vocab_size = config.vocab_size

    @property
    @abstractmethod
    def langs(self) -> List[str]:
        """
        支持的语言代码列表

        返回:
            语言代码列表，如 ['zh', 'zh-cn', 'zh-tw']
        """
        pass

    @abstractmethod
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        **kwargs
    ) -> List[int]:
        """
        将文本编码为token ID序列

        参数:
            text: 输入文本
            add_special_tokens: 是否添加特殊标记 (BOS/EOS)
            max_length: 最大长度（None则使用config.max_length）
            truncation: 是否截断
            **kwargs: 额外参数

        返回:
            token ID列表

        示例:
            >>> codec.encode("Hello world")
            [101, 7592, 2088, 102]
        """
        pass

    @abstractmethod
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> str:
        """
        将token ID序列解码为文本

        参数:
            token_ids: token ID列表
            skip_special_tokens: 是否跳过特殊标记
            **kwargs: 额外参数

        返回:
            解码后的文本

        示例:
            >>> codec.decode([101, 7592, 2088, 102])
            "Hello world"
        """
        pass

    @abstractmethod
    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        将文本分词为token列表

        参数:
            text: 输入文本
            **kwargs: 额外参数

        返回:
            token列表

        示例:
            >>> codec.tokenize("Hello world")
            ["Hello", "world"]
        """
        pass

    # ========================================================================
    # 辅助方法（可选实现）
    # ========================================================================

    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = True,
        **kwargs
    ) -> List[List[int]]:
        """
        批量编码文本

        参数:
            texts: 文本列表
            add_special_tokens: 是否添加特殊标记
            max_length: 最大长度
            padding: 是否填充到max_length
            truncation: 是否截断
            **kwargs: 额外参数

        返回:
            token ID列表的列表
        """
        encoded = [
            self.encode(text, add_special_tokens, max_length, truncation, **kwargs)
            for text in texts
        ]

        if padding and max_length:
            pad_id = self.get_token_id('pad')
            encoded = [
                ids + [pad_id] * (max_length - len(ids))
                for ids in encoded
            ]

        return encoded

    def batch_decode(
        self,
        token_ids_list: List[List[int]],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> List[str]:
        """
        批量解码token IDs

        参数:
            token_ids_list: token ID列表的列表
            skip_special_tokens: 是否跳过特殊标记
            **kwargs: 额外参数

        返回:
            解码后的文本列表
        """
        return [
            self.decode(ids, skip_special_tokens, **kwargs)
            for ids in token_ids_list
        ]

    def get_token_id(self, token_type: str) -> int:
        """
        获取特殊token的ID

        参数:
            token_type: token类型 ('pad', 'unk', 'bos', 'eos')

        返回:
            token ID

        示例:
            >>> codec.get_token_id('pad')
            0
        """
        # 子类可以重写此方法提供具体实现
        # 默认返回简单映射
        mapping = {
            'pad': 0,
            'unk': 1,
            'bos': 2,
            'eos': 2,
        }
        return mapping.get(token_type, 1)

    @property
    def pad_token_id(self) -> int:
        """填充token ID"""
        return self.get_token_id('pad')

    @property
    def unk_token_id(self) -> int:
        """未知token ID"""
        return self.get_token_id('unk')

    @property
    def bos_token_id(self) -> int:
        """开始token ID"""
        return self.get_token_id('bos')

    @property
    def eos_token_id(self) -> int:
        """结束token ID"""
        return self.get_token_id('eos')

    @property
    def pad_token(self) -> str:
        """填充token字符串"""
        return self.config.special_tokens.get('pad', '<|pad|>')

    @property
    def unk_token(self) -> str:
        """未知token字符串"""
        return self.config.special_tokens.get('unk', '<|unk|>')

    @property
    def bos_token(self) -> str:
        """开始token字符串"""
        return self.config.special_tokens.get('bos', '<|endoftext|>')

    @property
    def eos_token(self) -> str:
        """结束token字符串"""
        return self.config.special_tokens.get('eos', '<|endoftext|>')

    def save_pretrained(self, path: str):
        """
        保存codec到目录

        参数:
            path: 保存目录路径
        """
        # 子类应该实现具体的保存逻辑
        raise NotImplementedError(
            f"Codec '{self.name}' does not implement save_pretrained"
        )

    @classmethod
    def from_pretrained(cls, path: str) -> 'Codec':
        """
        从目录加载codec

        参数:
            path: 加载目录路径

        返回:
            Codec实例
        """
        # 子类应该实现具体的加载逻辑
        raise NotImplementedError(
            f"Codec does not implement from_pretrained"
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, langs={self.langs}, vocab_size={self.vocab_size})"
