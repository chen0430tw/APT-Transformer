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
    try:
        from apt.multilingual import (
            Language,
            get_language,
            list_languages,
            detect_language,
            TokenizerProvider
        )
    except ImportError:
        Language = None
        get_language = None
        list_languages = None
        detect_language = None
        TokenizerProvider = None

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
try:
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
except ImportError:
    Language = None
    Script = None
    Direction = None
    LanguageFeatures = None
    # Predefined languages ENGLISH = None
    CHINESE_SIMPLIFIED = None
    CHINESE_TRADITIONAL = None
    JAPANESE = None
    KOREAN = None
    SPANISH = None
    FRENCH = None
    GERMAN = None
    RUSSIAN = None
    ARABIC = None
    HINDI = None
    MULTILINGUAL = None
    # Language groups PREDEFINED_LANGUAGES = None
    EAST_ASIAN_LANGUAGES = None
    EUROPEAN_LANGUAGES = None
    RTL_LANGUAGES = None

# Tokenizer interface
try:
    from apt.multilingual.tokenizer import (
        TokenizerProvider,
        TokenizerConfig,
        get_tokenizer_for_language,
    )
except ImportError:
    TokenizerProvider = None
    TokenizerConfig = None
    get_tokenizer_for_language = None

# Language registry
try:
    from apt.multilingual.registry import (
        LanguageRegistry,
        language_registry,
        get_language,
        list_languages,
        get_vocab_size,
        is_language_supported,
    )
except ImportError:
    LanguageRegistry = None
    language_registry = None
    get_language = None
    list_languages = None
    get_vocab_size = None
    is_language_supported = None

# Language detection
try:
    from apt.multilingual.detector import (
        LanguageDetector,
        language_detector,
        detect_language,
        detect_script,
        is_mixed_language,
    )
except ImportError:
    LanguageDetector = None
    language_detector = None
    detect_language = None
    detect_script = None
    is_mixed_language = None

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
