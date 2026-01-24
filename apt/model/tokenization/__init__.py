#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tokenization

分词器实现：
- ChineseTokenizer: 中文分词器
- Tokenizer integration utilities
- Language detection
"""

try:
    from apt.model.tokenization.chinese_tokenizer import ChineseTokenizer
except ImportError:
    ChineseTokenizer = None
try:
    from apt.model.tokenization.chinese_tokenizer_integration import (
        integrate_chinese_tokenizer,
        detect_language,
        get_appropriate_tokenizer,
    )
except ImportError:
    integrate_chinese_tokenizer = None
    detect_language = None
    get_appropriate_tokenizer = None

__all__ = [
    'ChineseTokenizer',
    'integrate_chinese_tokenizer',
    'detect_language',
    'get_appropriate_tokenizer',
]
