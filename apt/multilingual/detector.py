#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Multilingual Support - Language Detector

Simple language detection based on character ranges and patterns.

Key features:
- Character-based language detection
- Script detection
- Confidence scoring
- Fast heuristic methods
"""

import re
from typing import List, Tuple, Optional, Dict
from collections import Counter
import logging

from apt.multilingual.language import Language, Script
from apt.multilingual.registry import language_registry

logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Simple language detector based on character analysis.

    This detector uses character ranges and patterns to identify
    the language of a text. It's fast but not as accurate as
    statistical models.

    Usage:
        detector = LanguageDetector()

        # Detect language
        lang = detector.detect("Hello world!")  # Returns 'en'

        # Detect with confidence
        lang, confidence = detector.detect_with_confidence("你好世界")
        # Returns ('zh', 0.95)
    """

    # Unicode ranges for different scripts
    CHAR_RANGES = {
        Script.LATIN: (0x0000, 0x024F),
        Script.CHINESE: (0x4E00, 0x9FFF),  # CJK Unified Ideographs
        Script.JAPANESE: [(0x3040, 0x309F), (0x30A0, 0x30FF)],  # Hiragana + Katakana
        Script.KOREAN: (0xAC00, 0xD7AF),  # Hangul
        Script.ARABIC: (0x0600, 0x06FF),
        Script.CYRILLIC: (0x0400, 0x04FF),
        Script.DEVANAGARI: (0x0900, 0x097F),
        Script.THAI: (0x0E00, 0x0E7F),
        Script.HEBREW: (0x0590, 0x05FF),
        Script.GREEK: (0x0370, 0x03FF),
    }

    def __init__(self):
        """Initialize language detector."""
        self.registry = language_registry

    def detect_script(self, text: str) -> Script:
        """
        Detect the primary script of a text.

        Args:
            text: Input text

        Returns:
            Detected script
        """
        if not text:
            return Script.LATIN

        # Count characters by script
        script_counts = Counter()

        for char in text:
            if char.isspace() or char in ",.!?;:()[]{}":
                continue

            code_point = ord(char)

            for script, ranges in self.CHAR_RANGES.items():
                if isinstance(ranges, tuple):
                    ranges = [ranges]

                for start, end in ranges:
                    if start <= code_point <= end:
                        script_counts[script] += 1
                        break

        if not script_counts:
            return Script.LATIN

        # Return most common script
        return script_counts.most_common(1)[0][0]

    def detect(self, text: str) -> Optional[str]:
        """
        Detect language of text.

        Args:
            text: Input text

        Returns:
            Language code or None

        Example:
            lang = detector.detect("Hello world!")  # 'en'
            lang = detector.detect("你好世界")  # 'zh'
        """
        lang, _ = self.detect_with_confidence(text)
        return lang

    def detect_with_confidence(
        self,
        text: str
    ) -> Tuple[Optional[str], float]:
        """
        Detect language with confidence score.

        Args:
            text: Input text

        Returns:
            Tuple of (language_code, confidence)

        Example:
            lang, conf = detector.detect_with_confidence("你好")
            # ('zh', 0.95)
        """
        if not text or not text.strip():
            return None, 0.0

        # Detect script first
        script = self.detect_script(text)

        # Get languages with this script
        languages = self.registry.list_languages(script=script.value)

        if not languages:
            return None, 0.0

        # Simple heuristic based on script
        if script == Script.CHINESE:
            # Check for traditional vs simplified
            if self._has_traditional_chars(text):
                return 'zh-tw', 0.85
            else:
                return 'zh', 0.85

        elif script == Script.JAPANESE:
            return 'ja', 0.85

        elif script == Script.KOREAN:
            return 'ko', 0.85

        elif script == Script.ARABIC:
            return 'ar', 0.80

        elif script == Script.CYRILLIC:
            return 'ru', 0.75  # Could be other Cyrillic languages

        elif script == Script.DEVANAGARI:
            return 'hi', 0.75

        elif script == Script.LATIN:
            # Need more sophisticated detection for Latin scripts
            # For now, default to English
            return 'en', 0.60

        else:
            # Return first language with this script
            if languages:
                return languages[0].code, 0.50
            return None, 0.0

    def _has_traditional_chars(self, text: str) -> bool:
        """
        Check if text contains traditional Chinese characters.

        This is a simple heuristic and not 100% accurate.
        """
        # Some characters that commonly appear in traditional but not simplified
        traditional_indicators = ['們', '說', '國', '學', '會', '長', '來']
        return any(char in text for char in traditional_indicators)

    def detect_multiple(
        self,
        text: str,
        top_n: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Detect multiple possible languages.

        Args:
            text: Input text
            top_n: Number of results to return

        Returns:
            List of (language_code, confidence) tuples

        Example:
            results = detector.detect_multiple("Hello world")
            # [('en', 0.60), ('fr', 0.40), ('es', 0.35)]
        """
        # For now, just return single detection
        # More sophisticated implementation would analyze patterns
        lang, conf = self.detect_with_confidence(text)

        if lang:
            return [(lang, conf)]
        return []

    def is_mixed_language(self, text: str, threshold: float = 0.2) -> bool:
        """
        Check if text contains multiple languages/scripts.

        Args:
            text: Input text
            threshold: Minimum proportion to count as mixed

        Returns:
            True if text appears to be mixed language
        """
        if not text:
            return False

        # Count characters by script
        script_counts = Counter()
        total_chars = 0

        for char in text:
            if char.isspace() or char in ",.!?;:()[]{}":
                continue

            total_chars += 1
            code_point = ord(char)

            for script, ranges in self.CHAR_RANGES.items():
                if isinstance(ranges, tuple):
                    ranges = [ranges]

                for start, end in ranges:
                    if start <= code_point <= end:
                        script_counts[script] += 1
                        break

        if total_chars == 0:
            return False

        # Check if multiple scripts exceed threshold
        significant_scripts = [
            script for script, count in script_counts.items()
            if count / total_chars >= threshold
        ]

        return len(significant_scripts) > 1

    def get_script_distribution(self, text: str) -> Dict[str, float]:
        """
        Get distribution of scripts in text.

        Args:
            text: Input text

        Returns:
            Dictionary mapping script names to proportions

        Example:
            dist = detector.get_script_distribution("Hello 你好")
            # {'latin': 0.5, 'chinese': 0.5}
        """
        if not text:
            return {}

        script_counts = Counter()
        total_chars = 0

        for char in text:
            if char.isspace() or char in ",.!?;:()[]{}":
                continue

            total_chars += 1
            code_point = ord(char)

            for script, ranges in self.CHAR_RANGES.items():
                if isinstance(ranges, tuple):
                    ranges = [ranges]

                for start, end in ranges:
                    if start <= code_point <= end:
                        script_counts[script] += 1
                        break

        if total_chars == 0:
            return {}

        return {
            script.value: count / total_chars
            for script, count in script_counts.items()
        }


# ============================================================================
# Global Detector Instance
# ============================================================================

# Global detector instance
language_detector = LanguageDetector()


# ============================================================================
# Convenience Functions
# ============================================================================

def detect_language(text: str) -> Optional[str]:
    """
    Detect language of text.

    Args:
        text: Input text

    Returns:
        Language code or None
    """
    return language_detector.detect(text)


def detect_script(text: str) -> Script:
    """
    Detect script of text.

    Args:
        text: Input text

    Returns:
        Detected script
    """
    return language_detector.detect_script(text)


def is_mixed_language(text: str) -> bool:
    """
    Check if text is mixed language.

    Args:
        text: Input text

    Returns:
        True if mixed language detected
    """
    return language_detector.is_mixed_language(text)
