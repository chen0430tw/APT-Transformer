#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Japanese MeCab Codec Implementation (Placeholder)

日文MeCab编解码器占位实现。

这是一个占位插件，提供基本接口但没有完整功能。
要使用完整功能，需要安装 fugashi 和 MeCab 词典。
"""

from typing import List, Optional
import logging

from apt.core.codecs import Codec, CodecConfig
from apt.core.codecs.unicode_norm import nfc

logger = logging.getLogger(__name__)


class JaMecabCodec(Codec):
    """
    日文MeCab编解码器（占位）

    这是一个占位实现。要使用完整MeCab分词功能，需要：
    1. 安装: pip install fugashi[unidic-lite]
    2. 替换此实现为完整版本

    当前行为：
    - 简单字符级分词（占位）
    - 不依赖外部库

    使用方式:
        config = CodecConfig(name="ja_mecab", vocab_size=32000)
        codec = JaMecabCodec(config)

        # 占位实现的基本编码/解码
        ids = codec.encode("こんにちは世界")
        text = codec.decode(ids)
    """

    def __init__(self, config: Optional[CodecConfig] = None):
        """
        初始化日文MeCab codec（占位）

        参数:
            config: Codec配置
        """
        if config is None:
            config = CodecConfig(
                name="ja_mecab",
                vocab_size=32000,  # 日文常用词汇量
            )

        super().__init__(config)

        # 检查是否有MeCab可用
        self._mecab_available = False
        try:
            import fugashi
            self._tagger = fugashi.Tagger()
            self._mecab_available = True
            logger.info("JaMecabCodec initialized with MeCab")
        except ImportError:
            logger.warning(
                "MeCab not available. JaMecabCodec is using placeholder mode. "
                "Install with: pip install fugashi[unidic-lite]"
            )
            self._tagger = None

        # 简单的字符映射（占位用）
        self._char_to_id = {}
        self._id_to_char = {}
        self._next_id = 100  # 预留0-99给特殊token

        logger.info(f"JaMecabCodec initialized (placeholder={'not ' if self._mecab_available else ''}active)")

    @property
    def langs(self) -> List[str]:
        """支持的语言代码"""
        return ['ja', 'ja-jp']

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        **kwargs
    ) -> List[int]:
        """
        编码文本为token IDs（占位实现）

        参数:
            text: 输入文本
            add_special_tokens: 是否添加特殊标记
            max_length: 最大长度
            truncation: 是否截断
            **kwargs: 额外参数

        返回:
            token ID列表
        """
        # Unicode规范化
        text = nfc(text)

        if self._mecab_available:
            # 使用MeCab分词
            return self._mecab_encode(text, max_length, truncation)
        else:
            # 占位：简单字符级编码
            return self._placeholder_encode(text, max_length, truncation)

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> str:
        """
        解码token IDs为文本（占位实现）

        参数:
            token_ids: token ID列表
            skip_special_tokens: 是否跳过特殊标记
            **kwargs: 额外参数

        返回:
            解码后的文本
        """
        if self._mecab_available:
            # 使用MeCab解码
            return self._mecab_decode(token_ids)
        else:
            # 占位：简单字符级解码
            return self._placeholder_decode(token_ids)

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        分词为token列表（占位实现）

        参数:
            text: 输入文本
            **kwargs: 额外参数

        返回:
            token列表
        """
        # Unicode规范化
        text = nfc(text)

        if self._mecab_available:
            # 使用MeCab分词
            return [word.surface for word in self._tagger(text)]
        else:
            # 占位：字符级分词
            return list(text)

    def _mecab_encode(self, text: str, max_length: Optional[int], truncation: bool) -> List[int]:
        """使用MeCab编码"""
        try:
            # 分词
            words = [word.surface for word in self._tagger(text)]

            # 转换为IDs（这里需要实际的词汇表，当前简化处理）
            token_ids = []
            for word in words:
                # 简化：使用hash
                tid = hash(word) % (self.vocab_size - 100) + 100
                token_ids.append(tid)

            # 截断
            if max_length and truncation and len(token_ids) > max_length:
                token_ids = token_ids[:max_length]

            return token_ids

        except Exception as e:
            logger.error(f"MeCab encoding failed: {e}")
            return self._placeholder_encode(text, max_length, truncation)

    def _mecab_decode(self, token_ids: List[int]) -> str:
        """使用MeCab解码（占位）"""
        # MeCab解码需要完整的词汇表，当前返回占位
        return "".join(f"[{tid}]" for tid in token_ids)

    def _placeholder_encode(self, text: str, max_length: Optional[int], truncation: bool) -> List[int]:
        """占位编码：字符级"""
        token_ids = []
        for char in text:
            if char not in self._char_to_id:
                self._char_to_id[char] = self._next_id
                self._id_to_char[self._next_id] = char
                self._next_id += 1

            token_ids.append(self._char_to_id[char])

        # 截断
        if max_length and truncation and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        return token_ids

    def _placeholder_decode(self, token_ids: List[int]) -> str:
        """占位解码：字符级"""
        chars = []
        for tid in token_ids:
            if tid in self._id_to_char:
                chars.append(self._id_to_char[tid])
            else:
                chars.append('?')

        return ''.join(chars)

    def get_token_id(self, token_type: str) -> int:
        """
        获取特殊token的ID

        参数:
            token_type: token类型 ('pad', 'unk', 'bos', 'eos')

        返回:
            token ID
        """
        mapping = {
            'pad': 0,
            'unk': 1,
            'bos': 2,
            'eos': 2,
        }
        return mapping.get(token_type, 1)
