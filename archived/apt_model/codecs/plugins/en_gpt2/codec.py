#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
English GPT2 Codec Implementation

英文GPT2编解码器实现。

包装transformers的GPT2Tokenizer，实现Codec接口。
"""

from typing import List, Optional
import logging
import torch

from apt.core.codecs import Codec, CodecConfig
from apt.core.codecs.unicode_norm import nfc

logger = logging.getLogger(__name__)


class EnGPT2Codec(Codec):
    """
    英文GPT2编解码器

    基于GPT2Tokenizer实现，提供BPE分词。

    特点：
    - BPE (Byte Pair Encoding) 分词
    - 支持英文及其他拉丁语系
    - 预训练词汇表（50257 tokens）
    - 与GPT2/GPT3模型兼容

    使用方式:
        config = CodecConfig(name="en_gpt2", vocab_size=50257)
        codec = EnGPT2Codec(config)

        # 编码
        ids = codec.encode("Hello world!")

        # 解码
        text = codec.decode(ids)
    """

    def __init__(self, config: Optional[CodecConfig] = None, cache_dir: Optional[str] = None):
        """
        初始化英文GPT2 codec

        参数:
            config: Codec配置
            cache_dir: Hugging Face模型缓存目录
        """
        if config is None:
            config = CodecConfig(
                name="en_gpt2",
                vocab_size=50257,  # GPT2标准词汇表大小
            )

        super().__init__(config)

        # 尝试加载GPT2Tokenizer
        try:
            from transformers import GPT2Tokenizer
            self._tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir=cache_dir)

            # 设置pad_token（GPT2默认没有）
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._available = True
            logger.info(f"EnGPT2Codec initialized with GPT2Tokenizer")

        except Exception as e:
            logger.warning(f"Failed to load GPT2Tokenizer: {e}")
            logger.warning("EnGPT2Codec will use fallback mode")
            self._tokenizer = None
            self._available = False

    @property
    def langs(self) -> List[str]:
        """支持的语言代码"""
        return ['en', 'en-us', 'en-gb']

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
            add_special_tokens: 是否添加特殊标记
            max_length: 最大长度
            truncation: 是否截断
            **kwargs: 额外参数

        返回:
            token ID列表
        """
        if not self._available:
            logger.warning("GPT2Tokenizer not available, using fallback")
            return self._fallback_encode(text)

        # Unicode规范化
        text = nfc(text)

        try:
            # 使用GPT2Tokenizer编码
            token_ids = self._tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length or self.config.max_length,
                truncation=truncation,
                **kwargs
            )
            return token_ids

        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            return self._fallback_encode(text)

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
        if not self._available:
            logger.warning("GPT2Tokenizer not available, using fallback")
            return self._fallback_decode(token_ids)

        try:
            # 使用GPT2Tokenizer解码
            text = self._tokenizer.decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                **kwargs
            )
            return text.strip()

        except Exception as e:
            logger.error(f"Decoding failed: {e}")
            return self._fallback_decode(token_ids)

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        分词为token列表

        参数:
            text: 输入文本
            **kwargs: 额外参数

        返回:
            token列表
        """
        if not self._available:
            logger.warning("GPT2Tokenizer not available, using fallback")
            return text.split()

        # Unicode规范化
        text = nfc(text)

        try:
            # 使用GPT2Tokenizer分词
            tokens = self._tokenizer.tokenize(text, **kwargs)
            return tokens

        except Exception as e:
            logger.error(f"Tokenization failed: {e}")
            return text.split()

    def get_token_id(self, token_type: str) -> int:
        """
        获取特殊token的ID

        参数:
            token_type: token类型 ('pad', 'unk', 'bos', 'eos')

        返回:
            token ID
        """
        if not self._available:
            return super().get_token_id(token_type)

        try:
            if token_type == 'pad':
                return self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
            elif token_type == 'unk':
                return self._tokenizer.unk_token_id if hasattr(self._tokenizer, 'unk_token_id') else 0
            elif token_type == 'bos':
                return self._tokenizer.bos_token_id if hasattr(self._tokenizer, 'bos_token_id') else self._tokenizer.eos_token_id
            elif token_type == 'eos':
                return self._tokenizer.eos_token_id
            else:
                return 0
        except:
            return super().get_token_id(token_type)

    def _fallback_encode(self, text: str) -> List[int]:
        """
        简单的备用编码（当GPT2Tokenizer不可用时）

        参数:
            text: 输入文本

        返回:
            token ID列表
        """
        # 非常简单的编码：按空格分词，然后hash
        words = text.split()
        return [hash(word) % 50000 + 100 for word in words]

    def _fallback_decode(self, token_ids: List[int]) -> str:
        """
        简单的备用解码（当GPT2Tokenizer不可用时）

        参数:
            token_ids: token ID列表

        返回:
            解码后的文本
        """
        # 无法真正解码，返回token ID的字符串表示
        return " ".join(f"[{tid}]" for tid in token_ids)

    def save_pretrained(self, path: str):
        """
        保存codec到目录

        参数:
            path: 保存目录路径
        """
        if not self._available:
            raise RuntimeError("Cannot save: GPT2Tokenizer not available")

        import os
        os.makedirs(path, exist_ok=True)

        # 保存tokenizer
        self._tokenizer.save_pretrained(path)
        logger.info(f"EnGPT2Codec saved to {path}")

    @classmethod
    def from_pretrained(cls, path: str) -> 'EnGPT2Codec':
        """
        从目录加载codec

        参数:
            path: 加载目录路径

        返回:
            EnGPT2Codec实例
        """
        try:
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained(path)

            # 创建codec
            config = CodecConfig(name="en_gpt2", vocab_size=len(tokenizer))
            codec = cls.__new__(cls)
            codec.config = config
            codec.name = config.name
            codec.vocab_size = config.vocab_size
            codec._tokenizer = tokenizer
            codec._available = True

            if codec._tokenizer.pad_token is None:
                codec._tokenizer.pad_token = codec._tokenizer.eos_token

            logger.info(f"EnGPT2Codec loaded from {path}")
            return codec

        except Exception as e:
            logger.error(f"Failed to load EnGPT2Codec from {path}: {e}")
            raise
