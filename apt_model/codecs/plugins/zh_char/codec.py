#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chinese Character-level Codec Implementation

中文字符级编解码器实现。

包装现有的ChineseTokenizer，实现Codec接口。
"""

from typing import List, Optional
import logging

from apt.core.codecs import Codec, CodecConfig
from apt.core.codecs.unicode_norm import nfc
from apt_model.modeling.chinese_tokenizer import ChineseTokenizer

logger = logging.getLogger(__name__)


class ZhCharCodec(Codec):
    """
    中文字符级编解码器

    基于ChineseTokenizer实现，提供字符级中文分词。

    特点：
    - 字符级分词，每个汉字作为一个token
    - 支持简体/繁体中文
    - 自动构建词汇表
    - 与现有ChineseTokenizer兼容

    使用方式:
        config = CodecConfig(name="zh_char", vocab_size=21128)
        codec = ZhCharCodec(config)

        # 编码
        ids = codec.encode("你好世界")

        # 解码
        text = codec.decode(ids)
    """

    def __init__(self, config: Optional[CodecConfig] = None, texts: Optional[List[str]] = None):
        """
        初始化中文字符级codec

        参数:
            config: Codec配置
            texts: 用于构建词汇表的文本列表（可选）
        """
        if config is None:
            config = CodecConfig(
                name="zh_char",
                vocab_size=21128,  # 常用汉字数量
            )

        super().__init__(config)

        # 创建底层ChineseTokenizer
        self._tokenizer = ChineseTokenizer(
            vocab_file=None,
            mode="char",
            vocab_size=config.vocab_size,
            texts=texts
        )

        # 构建token ID映射
        self._build_token_mapping()

        logger.info(f"ZhCharCodec initialized with vocab_size={self.vocab_size}")

    def _build_token_mapping(self):
        """构建特殊token的ID映射"""
        # 从底层tokenizer获取特殊token IDs
        try:
            self._pad_id = self._tokenizer.encoder.get(
                self._tokenizer.special_tokens.get("pad_token", "<|pad|>"), 0
            )
            self._unk_id = self._tokenizer.encoder.get(
                self._tokenizer.special_tokens.get("unk_token", "<|unk|>"), 1
            )
            self._bos_id = self._tokenizer.encoder.get(
                self._tokenizer.special_tokens.get("bos_token", "<|endoftext|>"), 2
            )
            self._eos_id = self._bos_id  # BOS和EOS使用同一个token
        except Exception as e:
            logger.warning(f"Failed to build token mapping: {e}, using defaults")
            self._pad_id = 0
            self._unk_id = 1
            self._bos_id = 2
            self._eos_id = 2

    @property
    def langs(self) -> List[str]:
        """支持的语言代码"""
        return ['zh', 'zh-cn', 'zh-tw', 'zh-hans', 'zh-hant']

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        **kwargs
    ) -> List[int]:
        """
        编码文本为token IDs

        参数:
            text: 输入文本
            add_special_tokens: 是否添加特殊标记（当前未实现BOS/EOS）
            max_length: 最大长度
            truncation: 是否截断
            **kwargs: 额外参数

        返回:
            token ID列表
        """
        # Unicode规范化
        text = nfc(text)

        # 使用底层tokenizer编码
        try:
            # ChineseTokenizer.encode 接受 return_tensors 参数
            # 我们不需要tensor，直接调用内部逻辑
            token_ids = self._tokenizer.encode(text)

            # 处理截断
            if max_length and truncation and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]

            return token_ids

        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            # 返回UNK token
            return [self._unk_id]

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> str:
        """
        解码token IDs为文本

        参数:
            token_ids: token ID列表
            skip_special_tokens: 是否跳过特殊标记
            **kwargs: 额外参数

        返回:
            解码后的文本
        """
        try:
            # 使用底层tokenizer解码
            text = self._tokenizer.decode(token_ids)

            # 移除特殊token（如果需要）
            if skip_special_tokens:
                for special_token in self._tokenizer.special_tokens.values():
                    text = text.replace(special_token, '')

            return text.strip()

        except Exception as e:
            logger.error(f"Decoding failed: {e}")
            return ""

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        分词为token列表

        参数:
            text: 输入文本
            **kwargs: 额外参数

        返回:
            token列表
        """
        # Unicode规范化
        text = nfc(text)

        # 字符级分词：简单地分割每个字符
        # 但保留特殊字符和空格的处理
        tokens = []
        for char in text:
            if char.strip():  # 非空白字符
                tokens.append(char)
            elif char == ' ':
                tokens.append(char)

        return tokens

    def get_token_id(self, token_type: str) -> int:
        """
        获取特殊token的ID

        参数:
            token_type: token类型 ('pad', 'unk', 'bos', 'eos')

        返回:
            token ID
        """
        mapping = {
            'pad': self._pad_id,
            'unk': self._unk_id,
            'bos': self._bos_id,
            'eos': self._eos_id,
        }
        return mapping.get(token_type, self._unk_id)

    def save_pretrained(self, path: str):
        """
        保存codec到目录

        参数:
            path: 保存目录路径
        """
        import os
        import json

        os.makedirs(path, exist_ok=True)

        # 保存配置
        config_path = os.path.join(path, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'name': self.name,
                'vocab_size': self.vocab_size,
                'special_tokens': self.config.special_tokens,
            }, f, ensure_ascii=False, indent=2)

        # 保存词汇表
        vocab_path = os.path.join(path, "vocab.json")
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self._tokenizer.encoder, f, ensure_ascii=False, indent=2)

        logger.info(f"ZhCharCodec saved to {path}")

    @classmethod
    def from_pretrained(cls, path: str) -> 'ZhCharCodec':
        """
        从目录加载codec

        参数:
            path: 加载目录路径

        返回:
            ZhCharCodec实例
        """
        import os
        import json

        # 加载配置
        config_path = os.path.join(path, "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        # 创建配置对象
        config = CodecConfig(
            name=config_dict['name'],
            vocab_size=config_dict['vocab_size'],
            special_tokens=config_dict.get('special_tokens'),
        )

        # 创建codec实例
        codec = cls(config)

        # 加载词汇表
        vocab_path = os.path.join(path, "vocab.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, 'r', encoding='utf-8') as f:
                encoder = json.load(f)
            codec._tokenizer.encoder = encoder
            codec._tokenizer.decoder = {v: k for k, v in encoder.items()}
            codec._tokenizer.vocab_size = len(encoder)
            codec._build_token_mapping()

        logger.info(f"ZhCharCodec loaded from {path}")
        return codec
