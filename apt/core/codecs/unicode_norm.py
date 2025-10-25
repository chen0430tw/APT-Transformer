#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Core Codec - Unicode Normalization

Unicode文本规范化工具。

功能：
- NFC/NFKC规范化
- 全角/半角转换
- 控制字符清理
- 空白字符规范化
"""

import unicodedata
import re
from typing import Optional


class UnicodeNormalizer:
    """
    Unicode规范化器

    提供多种Unicode文本规范化方法。

    使用方式:
        normalizer = UnicodeNormalizer()
        text = normalizer.normalize("ｈｅｌｌｏ")  # 半角化
        text = normalizer.nfc("café")  # NFC规范化
    """

    # 控制字符范围
    CONTROL_CHARS_RE = re.compile(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]')

    # 多个空白字符
    MULTI_SPACES_RE = re.compile(r'\s+')

    def __init__(self, form: str = "NFC", remove_control: bool = True):
        """
        初始化规范化器

        参数:
            form: Unicode规范化形式 ('NFC', 'NFKC', 'NFD', 'NFKD')
            remove_control: 是否移除控制字符
        """
        self.form = form
        self.remove_control = remove_control

    def normalize(self, text: str, form: Optional[str] = None) -> str:
        """
        完整的Unicode规范化流程

        参数:
            text: 输入文本
            form: Unicode规范化形式（None则使用初始化时的form）

        返回:
            规范化后的文本

        示例:
            >>> normalizer.normalize("ｈｅｌｌｏ")
            "hello"
        """
        if not text:
            return text

        # 1. Unicode规范化
        form = form or self.form
        text = self.apply_form(text, form)

        # 2. 移除控制字符
        if self.remove_control:
            text = self.remove_control_chars(text)

        # 3. 规范化空白
        text = self.normalize_whitespace(text)

        return text

    def apply_form(self, text: str, form: str = "NFC") -> str:
        """
        应用Unicode规范化形式

        参数:
            text: 输入文本
            form: 规范化形式
                - 'NFC': Canonical Decomposition + Canonical Composition
                - 'NFKC': Compatibility Decomposition + Canonical Composition
                - 'NFD': Canonical Decomposition
                - 'NFKD': Compatibility Decomposition

        返回:
            规范化后的文本

        示例:
            >>> normalizer.apply_form("café", "NFC")
            "café"
        """
        return unicodedata.normalize(form, text)

    def nfc(self, text: str) -> str:
        """
        NFC规范化（最常用）

        保留字符的原始含义，只做组合。
        推荐用于多数场景。

        参数:
            text: 输入文本

        返回:
            NFC规范化后的文本
        """
        return unicodedata.normalize('NFC', text)

    def nfkc(self, text: str) -> str:
        """
        NFKC规范化（兼容性）

        将兼容字符转换为标准形式（如全角→半角）。
        推荐用于搜索和比较。

        参数:
            text: 输入文本

        返回:
            NFKC规范化后的文本

        示例:
            >>> normalizer.nfkc("ｈｅｌｌｏ")
            "hello"
        """
        return unicodedata.normalize('NFKC', text)

    def remove_control_chars(self, text: str) -> str:
        """
        移除控制字符

        移除ASCII控制字符（0x00-0x1F, 0x7F），但保留制表符、换行符。

        参数:
            text: 输入文本

        返回:
            清理后的文本
        """
        return self.CONTROL_CHARS_RE.sub('', text)

    def normalize_whitespace(self, text: str, single_space: bool = False) -> str:
        """
        规范化空白字符

        参数:
            text: 输入文本
            single_space: 是否将多个空白合并为单个空格

        返回:
            规范化后的文本

        示例:
            >>> normalizer.normalize_whitespace("hello  world", single_space=True)
            "hello world"
        """
        # 去除首尾空白
        text = text.strip()

        # 合并多个空白
        if single_space:
            text = self.MULTI_SPACES_RE.sub(' ', text)

        return text

    def to_halfwidth(self, text: str) -> str:
        """
        全角→半角转换

        将全角字符转换为半角字符。

        参数:
            text: 输入文本

        返回:
            半角化后的文本

        示例:
            >>> normalizer.to_halfwidth("ＨＥＬ  ＬＯ")
            "HELLO"
        """
        # NFKC会自动处理全角→半角
        return unicodedata.normalize('NFKC', text)

    def to_fullwidth(self, text: str) -> str:
        """
        半角→全角转换（部分字符）

        注意：此方法只处理ASCII字符。

        参数:
            text: 输入文本

        返回:
            全角化后的文本
        """
        result = []
        for char in text:
            code = ord(char)
            # ASCII可打印字符范围
            if 0x21 <= code <= 0x7E:
                # 转换为全角（偏移0xFEE0）
                result.append(chr(code + 0xFEE0))
            else:
                result.append(char)
        return ''.join(result)

    def remove_accents(self, text: str) -> str:
        """
        移除重音符号

        将 "café" → "cafe"

        参数:
            text: 输入文本

        返回:
            无重音的文本

        示例:
            >>> normalizer.remove_accents("café")
            "cafe"
        """
        # 分解字符
        nfd = unicodedata.normalize('NFD', text)
        # 移除组合标记（重音）
        result = ''.join(
            char for char in nfd
            if unicodedata.category(char) != 'Mn'  # Mn = Nonspacing Mark
        )
        # 重新组合
        return unicodedata.normalize('NFC', result)

    def casefold(self, text: str) -> str:
        """
        大小写折叠（更强的小写化）

        比lower()更彻底，处理特殊字符（如德语ß）。

        参数:
            text: 输入文本

        返回:
            折叠后的文本

        示例:
            >>> normalizer.casefold("Straße")
            "strasse"
        """
        return text.casefold()


# ============================================================================
# 全局单例
# ============================================================================

# 默认规范化器
default_normalizer = UnicodeNormalizer(form="NFC", remove_control=True)


# ============================================================================
# 便捷函数
# ============================================================================

def normalize_unicode(text: str, form: str = "NFC", remove_control: bool = True) -> str:
    """
    快捷Unicode规范化

    参数:
        text: 输入文本
        form: 规范化形式
        remove_control: 是否移除控制字符

    返回:
        规范化后的文本
    """
    normalizer = UnicodeNormalizer(form=form, remove_control=remove_control)
    return normalizer.normalize(text)


def nfc(text: str) -> str:
    """NFC规范化（快捷方法）"""
    return default_normalizer.nfc(text)


def nfkc(text: str) -> str:
    """NFKC规范化（快捷方法）"""
    return default_normalizer.nfkc(text)


def remove_accents(text: str) -> str:
    """移除重音（快捷方法）"""
    return default_normalizer.remove_accents(text)


def to_halfwidth(text: str) -> str:
    """全角→半角（快捷方法）"""
    return default_normalizer.to_halfwidth(text)
