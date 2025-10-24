#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Multilingual Support

Comprehensive multilingual support for APT, including:
- Language definitions with metadata
- Tokenizer provider interface
- Language registry
- Language detection
- Support for 12+ languages

Key components:
- Language: Language definition class
- TokenizerProvider: Abstract tokenizer interface
- LanguageRegistry: Language management
- LanguageDetector: Automatic language detection

Usage:
    from apt.multilingual import (
        Language,
        get_language,
        list_languages,
        detect_language,
        TokenizerProvider
    )

    # Get a language
    chinese = get_language('zh')
    print(chinese.name)  # "Chinese (Simplified)"
    print(chinese.vocab_size)  # 21128

    # Detect language
    lang_code = detect_language("你好世界")
    print(lang_code)  # "zh"

    # List all languages
    for lang in list_languages():
        print(f"{lang.code}: {lang.name}")
"""

# Language definitions
from apt.multilingual.language import (
    Language,
    Script,
    Direction,
    LanguageFeatures,
    # Predefined languages
    ENGLISH,
    CHINESE_SIMPLIFIED,
    CHINESE_TRADITIONAL,
    JAPANESE,
    KOREAN,
    SPANISH,
    FRENCH,
    GERMAN,
    RUSSIAN,
    ARABIC,
    HINDI,
    MULTILINGUAL,
    # Language groups
    PREDEFINED_LANGUAGES,
    EAST_ASIAN_LANGUAGES,
    EUROPEAN_LANGUAGES,
    RTL_LANGUAGES,
)

# Tokenizer interface
from apt.multilingual.tokenizer import (
    TokenizerProvider,
    TokenizerConfig,
    get_tokenizer_for_language,
)

# Language registry
from apt.multilingual.registry import (
    LanguageRegistry,
    language_registry,
    get_language,
    list_languages,
    get_vocab_size,
    is_language_supported,
)

# Language detection
from apt.multilingual.detector import (
    LanguageDetector,
    language_detector,
    detect_language,
    detect_script,
    is_mixed_language,
)

__all__ = [
    # Core classes
    'Language',
    'Script',
    'Direction',
    'LanguageFeatures',

    # Predefined languages
    'ENGLISH',
    'CHINESE_SIMPLIFIED',
    'CHINESE_TRADITIONAL',
    'JAPANESE',
    'KOREAN',
    'SPANISH',
    'FRENCH',
    'GERMAN',
    'RUSSIAN',
    'ARABIC',
    'HINDI',
    'MULTILINGUAL',

    # Language groups
    'PREDEFINED_LANGUAGES',
    'EAST_ASIAN_LANGUAGES',
    'EUROPEAN_LANGUAGES',
    'RTL_LANGUAGES',

    # Tokenizer
    'TokenizerProvider',
    'TokenizerConfig',
    'get_tokenizer_for_language',

    # Registry
    'LanguageRegistry',
    'language_registry',
    'get_language',
    'list_languages',
    'get_vocab_size',
    'is_language_supported',

    # Detection
    'LanguageDetector',
    'language_detector',
    'detect_language',
    'detect_script',
    'is_mixed_language',
]
