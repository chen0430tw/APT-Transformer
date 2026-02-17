#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Codec Compatibility Layer

提供Codec与旧tokenizer接口的兼容层。

用于逐步迁移：让新Codec可以像旧tokenizer一样使用。
"""

from typing import List, Union, Optional
import torch
import logging

from apt.core.codecs import Codec

logger = logging.getLogger(__name__)


class CodecTokenizerWrapper:
    """
    Codec到Tokenizer接口的适配器

    将Codec包装成类似transformers tokenizer的接口，
    以便与现有代码（如TextDataset）兼容。

    使用方式:
        codec = get_codec("zh_char")
        tokenizer = CodecTokenizerWrapper(codec)

        # 像tokenizer一样使用
        ids = tokenizer.encode("你好", return_tensors="pt")
        text = tokenizer.decode(ids)
    """

    def __init__(self, codec: Codec):
        """
        初始化包装器

        参数:
            codec: Codec实例
        """
        self.codec = codec
        self._name = codec.name

        # 提供tokenizer风格的属性
        self.vocab_size = codec.vocab_size
        self.pad_token_id = codec.pad_token_id
        self.eos_token_id = codec.eos_token_id
        self.bos_token_id = codec.bos_token_id
        self.unk_token_id = codec.unk_token_id

        self.pad_token = codec.pad_token
        self.eos_token = codec.eos_token
        self.bos_token = codec.bos_token
        self.unk_token = codec.unk_token

        logger.debug(f"CodecTokenizerWrapper created for {codec.name}")

    def encode(
        self,
        text: str,
        return_tensors: Optional[str] = None,
        max_length: Optional[int] = None,
        truncation: bool = True,
        add_special_tokens: bool = True,
        **kwargs
    ) -> Union[List[int], torch.Tensor]:
        """
        编码文本（tokenizer风格接口）

        参数:
            text: 输入文本
            return_tensors: 返回格式 ("pt" 为PyTorch tensor)
            max_length: 最大长度
            truncation: 是否截断
            add_special_tokens: 是否添加特殊标记
            **kwargs: 额外参数

        返回:
            token IDs (list或tensor)
        """
        # 使用codec编码
        token_ids = self.codec.encode(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            truncation=truncation,
            **kwargs
        )

        # 转换返回格式
        if return_tensors == "pt":
            return torch.tensor(token_ids, dtype=torch.long)
        else:
            return token_ids

    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> str:
        """
        解码token IDs（tokenizer风格接口）

        参数:
            token_ids: token ID列表或tensor
            skip_special_tokens: 是否跳过特殊标记
            **kwargs: 额外参数

        返回:
            解码后的文本
        """
        # 转换tensor到list
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        # 使用codec解码
        return self.codec.decode(token_ids, skip_special_tokens, **kwargs)

    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        分词（tokenizer风格接口）

        参数:
            text: 输入文本
            **kwargs: 额外参数

        返回:
            token列表
        """
        return self.codec.tokenize(text, **kwargs)

    def batch_encode(
        self,
        texts: List[str],
        return_tensors: Optional[str] = None,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = True,
        **kwargs
    ) -> Union[List[List[int]], torch.Tensor]:
        """
        批量编码（tokenizer风格接口）

        参数:
            texts: 文本列表
            return_tensors: 返回格式
            max_length: 最大长度
            padding: 是否填充
            truncation: 是否截断
            **kwargs: 额外参数

        返回:
            token IDs列表（list或tensor）
        """
        # 使用codec批量编码
        token_ids_list = self.codec.batch_encode(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            **kwargs
        )

        # 转换返回格式
        if return_tensors == "pt":
            return torch.tensor(token_ids_list, dtype=torch.long)
        else:
            return token_ids_list

    def save_pretrained(self, path: str):
        """保存到目录"""
        self.codec.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str, codec_class) -> 'CodecTokenizerWrapper':
        """从目录加载"""
        codec = codec_class.from_pretrained(path)
        return cls(codec)

    def __repr__(self):
        return f"CodecTokenizerWrapper({self.codec})"


def wrap_codec_as_tokenizer(codec: Codec) -> CodecTokenizerWrapper:
    """
    将Codec包装为tokenizer接口

    参数:
        codec: Codec实例

    返回:
        CodecTokenizerWrapper实例
    """
    return CodecTokenizerWrapper(codec)
