#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Multilingual Support - Language Definition

Defines the Language class for representing different languages
and their characteristics.

Key features:
- ISO 639-1 language codes
- Language metadata (name, script, direction)
- Special token definitions
- Vocabulary size recommendations
- Language-specific features
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from enum import Enum


class Script(Enum):
    """Writing system/script types."""
    LATIN = "latin"           # English, French, Spanish, etc.
    CHINESE = "chinese"       # Simplified/Traditional Chinese
    JAPANESE = "japanese"     # Hiragana, Katakana, Kanji
    KOREAN = "korean"         # Hangul
    ARABIC = "arabic"         # Arabic script
    CYRILLIC = "cyrillic"     # Russian, Ukrainian, etc.
    DEVANAGARI = "devanagari" # Hindi, Sanskrit
    THAI = "thai"             # Thai script
    HEBREW = "hebrew"         # Hebrew script
    GREEK = "greek"           # Greek script
    MIXED = "mixed"           # Multiple scripts


class Direction(Enum):
    """Text direction."""
    LTR = "ltr"  # Left-to-right
    RTL = "rtl"  # Right-to-left


@dataclass
class Language:
    """
    Language definition with metadata and characteristics.

    Attributes:
        code: ISO 639-1 language code (e.g., 'en', 'zh', 'ja')
        name: Language name (e.g., 'English', 'Chinese', 'Japanese')
        native_name: Native language name (e.g., 'English', '中文', '日本語')
        script: Writing system
        direction: Text direction
        vocab_size: Recommended vocabulary size
        special_tokens: Language-specific special tokens
        features: Language-specific features (set of strings)
        aliases: Alternative codes or names
    """

    code: str
    name: str
    native_name: str = ""
    script: Script = Script.LATIN
    direction: Direction = Direction.LTR
    vocab_size: int = 50000
    special_tokens: Dict[str, str] = field(default_factory=dict)
    features: Set[str] = field(default_factory=set)
    aliases: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Post-initialization validation."""
        if not self.native_name:
            self.native_name = self.name

        # Validate code (at least 2 characters)
        if len(self.code) < 2:
            raise ValueError(f"Invalid language code: {self.code}")

    def has_feature(self, feature: str) -> bool:
        """
        Check if language has a specific feature.

        Args:
            feature: Feature name (e.g., 'tones', 'cases', 'articles')

        Returns:
            True if language has the feature
        """
        return feature in self.features

    def get_special_token(self, token_type: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a special token for this language.

        Args:
            token_type: Token type (e.g., 'pad', 'unk', 'cls')
            default: Default value if token not found

        Returns:
            Special token string or default
        """
        return self.special_tokens.get(token_type, default)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'code': self.code,
            'name': self.name,
            'native_name': self.native_name,
            'script': self.script.value,
            'direction': self.direction.value,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'features': list(self.features),
            'aliases': self.aliases
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Language':
        """Create from dictionary."""
        data = data.copy()
        data['script'] = Script(data['script'])
        data['direction'] = Direction(data['direction'])
        data['features'] = set(data.get('features', []))
        return cls(**data)

    def __repr__(self):
        return f"Language({self.code}, {self.name}, {self.script.value})"


# ============================================================================
# Language Feature Constants
# ============================================================================

class LanguageFeatures:
    """Common language features."""

    # Morphological features
    CASES = "cases"                    # Grammatical cases
    GENDER = "gender"                  # Grammatical gender
    ARTICLES = "articles"              # Definite/indefinite articles
    INFLECTION = "inflection"          # Word inflection

    # Phonological features
    TONES = "tones"                    # Tonal language
    STRESS = "stress"                  # Stress accent
    PITCH_ACCENT = "pitch_accent"      # Pitch accent

    # Writing system features
    SPACES = "spaces"                  # Uses spaces between words
    NO_SPACES = "no_spaces"            # No spaces (e.g., Chinese)
    MIXED_SCRIPT = "mixed_script"      # Multiple scripts in use

    # Syntactic features
    SVO = "svo"                        # Subject-Verb-Object order
    SOV = "sov"                        # Subject-Object-Verb order
    VSO = "vso"                        # Verb-Subject-Object order

    # Character features
    LOGOGRAPHIC = "logographic"        # Logographic writing (Chinese)
    ALPHABETIC = "alphabetic"          # Alphabetic writing
    SYLLABIC = "syllabic"              # Syllabary (Japanese kana)

    # Processing features
    NEEDS_SEGMENTATION = "needs_segmentation"  # Requires word segmentation
    COMPLEX_MORPHOLOGY = "complex_morphology"  # Complex morphology
    AGGLUTINATIVE = "agglutinative"            # Agglutinative language


# ============================================================================
# Predefined Languages
# ============================================================================

# English
ENGLISH = Language(
    code="en",
    name="English",
    native_name="English",
    script=Script.LATIN,
    direction=Direction.LTR,
    vocab_size=50000,
    features={
        LanguageFeatures.SPACES,
        LanguageFeatures.ALPHABETIC,
        LanguageFeatures.SVO,
        LanguageFeatures.ARTICLES,
        LanguageFeatures.STRESS
    },
    aliases=["eng"]
)

# Chinese (Simplified)
CHINESE_SIMPLIFIED = Language(
    code="zh",
    name="Chinese (Simplified)",
    native_name="简体中文",
    script=Script.CHINESE,
    direction=Direction.LTR,
    vocab_size=21128,
    features={
        LanguageFeatures.NO_SPACES,
        LanguageFeatures.LOGOGRAPHIC,
        LanguageFeatures.TONES,
        LanguageFeatures.SVO,
        LanguageFeatures.NEEDS_SEGMENTATION
    },
    aliases=["zh-cn", "zh-hans", "chi", "zho"]
)

# Chinese (Traditional)
CHINESE_TRADITIONAL = Language(
    code="zh-tw",
    name="Chinese (Traditional)",
    native_name="繁體中文",
    script=Script.CHINESE,
    direction=Direction.LTR,
    vocab_size=21128,
    features={
        LanguageFeatures.NO_SPACES,
        LanguageFeatures.LOGOGRAPHIC,
        LanguageFeatures.TONES,
        LanguageFeatures.SVO,
        LanguageFeatures.NEEDS_SEGMENTATION
    },
    aliases=["zh-hant"]
)

# Japanese
JAPANESE = Language(
    code="ja",
    name="Japanese",
    native_name="日本語",
    script=Script.JAPANESE,
    direction=Direction.LTR,
    vocab_size=32000,
    features={
        LanguageFeatures.NO_SPACES,
        LanguageFeatures.MIXED_SCRIPT,
        LanguageFeatures.LOGOGRAPHIC,
        LanguageFeatures.SYLLABIC,
        LanguageFeatures.SOV,
        LanguageFeatures.PITCH_ACCENT,
        LanguageFeatures.AGGLUTINATIVE
    },
    aliases=["jpn"]
)

# Korean
KOREAN = Language(
    code="ko",
    name="Korean",
    native_name="한국어",
    script=Script.KOREAN,
    direction=Direction.LTR,
    vocab_size=30000,
    features={
        LanguageFeatures.SPACES,
        LanguageFeatures.ALPHABETIC,
        LanguageFeatures.SOV,
        LanguageFeatures.AGGLUTINATIVE
    },
    aliases=["kor"]
)

# Spanish
SPANISH = Language(
    code="es",
    name="Spanish",
    native_name="Español",
    script=Script.LATIN,
    direction=Direction.LTR,
    vocab_size=50000,
    features={
        LanguageFeatures.SPACES,
        LanguageFeatures.ALPHABETIC,
        LanguageFeatures.SVO,
        LanguageFeatures.GENDER,
        LanguageFeatures.ARTICLES,
        LanguageFeatures.INFLECTION,
        LanguageFeatures.STRESS
    },
    aliases=["spa"]
)

# French
FRENCH = Language(
    code="fr",
    name="French",
    native_name="Français",
    script=Script.LATIN,
    direction=Direction.LTR,
    vocab_size=50000,
    features={
        LanguageFeatures.SPACES,
        LanguageFeatures.ALPHABETIC,
        LanguageFeatures.SVO,
        LanguageFeatures.GENDER,
        LanguageFeatures.ARTICLES,
        LanguageFeatures.INFLECTION
    },
    aliases=["fra", "fre"]
)

# German
GERMAN = Language(
    code="de",
    name="German",
    native_name="Deutsch",
    script=Script.LATIN,
    direction=Direction.LTR,
    vocab_size=50000,
    features={
        LanguageFeatures.SPACES,
        LanguageFeatures.ALPHABETIC,
        LanguageFeatures.SVO,
        LanguageFeatures.GENDER,
        LanguageFeatures.CASES,
        LanguageFeatures.ARTICLES,
        LanguageFeatures.INFLECTION,
        LanguageFeatures.COMPLEX_MORPHOLOGY
    },
    aliases=["deu", "ger"]
)

# Russian
RUSSIAN = Language(
    code="ru",
    name="Russian",
    native_name="Русский",
    script=Script.CYRILLIC,
    direction=Direction.LTR,
    vocab_size=50000,
    features={
        LanguageFeatures.SPACES,
        LanguageFeatures.ALPHABETIC,
        LanguageFeatures.SVO,
        LanguageFeatures.GENDER,
        LanguageFeatures.CASES,
        LanguageFeatures.INFLECTION,
        LanguageFeatures.COMPLEX_MORPHOLOGY,
        LanguageFeatures.STRESS
    },
    aliases=["rus"]
)

# Arabic
ARABIC = Language(
    code="ar",
    name="Arabic",
    native_name="العربية",
    script=Script.ARABIC,
    direction=Direction.RTL,
    vocab_size=50000,
    features={
        LanguageFeatures.SPACES,
        LanguageFeatures.ALPHABETIC,
        LanguageFeatures.VSO,
        LanguageFeatures.GENDER,
        LanguageFeatures.CASES,
        LanguageFeatures.INFLECTION,
        LanguageFeatures.COMPLEX_MORPHOLOGY
    },
    aliases=["ara"]
)

# Hindi
HINDI = Language(
    code="hi",
    name="Hindi",
    native_name="हिन्दी",
    script=Script.DEVANAGARI,
    direction=Direction.LTR,
    vocab_size=50000,
    features={
        LanguageFeatures.SPACES,
        LanguageFeatures.ALPHABETIC,
        LanguageFeatures.SOV,
        LanguageFeatures.GENDER,
        LanguageFeatures.CASES,
        LanguageFeatures.INFLECTION
    },
    aliases=["hin"]
)

# Multilingual (special case)
MULTILINGUAL = Language(
    code="multi",
    name="Multilingual",
    native_name="Multilingual",
    script=Script.MIXED,
    direction=Direction.LTR,
    vocab_size=250000,
    features={
        LanguageFeatures.SPACES,
        LanguageFeatures.MIXED_SCRIPT
    },
    aliases=["multilingual", "xx"]
)


# ============================================================================
# Language Collections
# ============================================================================

# All predefined languages
PREDEFINED_LANGUAGES = [
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
    MULTILINGUAL
]

# Language groups
EAST_ASIAN_LANGUAGES = [CHINESE_SIMPLIFIED, CHINESE_TRADITIONAL, JAPANESE, KOREAN]
EUROPEAN_LANGUAGES = [ENGLISH, SPANISH, FRENCH, GERMAN, RUSSIAN]
RTL_LANGUAGES = [ARABIC]
