#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Multilingual Support - Language Registry

Manages language definitions and provides lookup functionality.

Key features:
- Register and lookup languages
- Language detection
- Language code normalization
- Language recommendations
"""

from typing import Dict, List, Optional, Set
import logging

from apt.multilingual.language import Language, PREDEFINED_LANGUAGES

logger = logging.getLogger(__name__)


class LanguageRegistry:
    """
    Registry for language definitions.

    The LanguageRegistry manages all supported languages and provides
    lookup functionality by code, name, or alias.

    Usage:
        registry = LanguageRegistry()

        # Register language
        registry.register(my_language)

        # Lookup by code
        lang = registry.get_language('zh')

        # List all languages
        languages = registry.list_languages()
    """

    def __init__(self):
        """Initialize language registry."""
        # Storage: {code: Language}
        self._languages: Dict[str, Language] = {}

        # Alias mapping: {alias: code}
        self._aliases: Dict[str, str] = {}

        # Name mapping: {name.lower(): code}
        self._names: Dict[str, str] = {}

        logger.info("LanguageRegistry initialized")

    def register(self, language: Language, override: bool = False) -> None:
        """
        Register a language.

        Args:
            language: Language instance
            override: Allow overriding existing language

        Raises:
            ValueError: If language already registered and override=False
        """
        code = language.code.lower()

        if code in self._languages and not override:
            raise ValueError(
                f"Language '{code}' already registered. Use override=True to replace."
            )

        # Register language
        self._languages[code] = language

        # Register aliases
        for alias in language.aliases:
            alias = alias.lower()
            self._aliases[alias] = code

        # Register name
        name_key = language.name.lower()
        self._names[name_key] = code

        logger.info(f"Registered language: {code} ({language.name})")

    def get_language(
        self,
        identifier: str,
        default: Optional[Language] = None
    ) -> Optional[Language]:
        """
        Get language by code, name, or alias.

        Args:
            identifier: Language code, name, or alias
            default: Default language if not found

        Returns:
            Language instance or default

        Example:
            lang = registry.get_language('zh')
            lang = registry.get_language('Chinese')
            lang = registry.get_language('zh-cn')
        """
        identifier = identifier.lower()

        # Try direct code lookup
        if identifier in self._languages:
            return self._languages[identifier]

        # Try alias lookup
        if identifier in self._aliases:
            code = self._aliases[identifier]
            return self._languages[code]

        # Try name lookup
        if identifier in self._names:
            code = self._names[identifier]
            return self._languages[code]

        # Not found
        if default:
            logger.warning(f"Language '{identifier}' not found, using default")
            return default

        logger.warning(f"Language '{identifier}' not found")
        return None

    def has_language(self, identifier: str) -> bool:
        """
        Check if language is registered.

        Args:
            identifier: Language code, name, or alias

        Returns:
            True if language is registered
        """
        return self.get_language(identifier) is not None

    def list_languages(self, script: Optional[str] = None) -> List[Language]:
        """
        List all registered languages.

        Args:
            script: Filter by script type (optional)

        Returns:
            List of Language instances
        """
        languages = list(self._languages.values())

        if script:
            script = script.lower()
            languages = [
                lang for lang in languages
                if lang.script.value.lower() == script
            ]

        return languages

    def list_codes(self) -> List[str]:
        """
        List all language codes.

        Returns:
            List of language codes
        """
        return list(self._languages.keys())

    def get_languages_by_feature(self, feature: str) -> List[Language]:
        """
        Get languages that have a specific feature.

        Args:
            feature: Feature name (e.g., 'tones', 'cases')

        Returns:
            List of Language instances with the feature
        """
        return [
            lang for lang in self._languages.values()
            if lang.has_feature(feature)
        ]

    def get_rtl_languages(self) -> List[Language]:
        """Get all right-to-left languages."""
        from apt.multilingual.language import Direction
        return [
            lang for lang in self._languages.values()
            if lang.direction == Direction.RTL
        ]

    def get_recommended_vocab_size(self, language_code: str) -> int:
        """
        Get recommended vocabulary size for a language.

        Args:
            language_code: Language code

        Returns:
            Recommended vocabulary size
        """
        lang = self.get_language(language_code)
        if lang:
            return lang.vocab_size
        else:
            return 50000  # Default

    def normalize_code(self, code: str) -> Optional[str]:
        """
        Normalize language code to standard form.

        Args:
            code: Language code or alias

        Returns:
            Normalized code or None

        Example:
            registry.normalize_code('zh-cn')  # Returns 'zh'
            registry.normalize_code('Chinese')  # Returns 'zh'
        """
        lang = self.get_language(code)
        return lang.code if lang else None

    def get_info(self, identifier: str) -> Optional[Dict]:
        """
        Get detailed information about a language.

        Args:
            identifier: Language code, name, or alias

        Returns:
            Language info dictionary or None
        """
        lang = self.get_language(identifier)
        return lang.to_dict() if lang else None

    def register_all_predefined(self) -> None:
        """Register all predefined languages."""
        for lang in PREDEFINED_LANGUAGES:
            try:
                self.register(lang, override=True)
            except Exception as e:
                logger.error(f"Failed to register {lang.code}: {e}")

        logger.info(f"Registered {len(self._languages)} predefined languages")

    def unregister(self, language_code: str) -> bool:
        """
        Unregister a language.

        Args:
            language_code: Language code

        Returns:
            True if language was removed
        """
        code = language_code.lower()

        if code not in self._languages:
            return False

        lang = self._languages[code]

        # Remove aliases
        for alias in lang.aliases:
            self._aliases.pop(alias.lower(), None)

        # Remove name
        self._names.pop(lang.name.lower(), None)

        # Remove language
        del self._languages[code]

        logger.info(f"Unregistered language: {code}")
        return True

    def search(self, query: str) -> List[Language]:
        """
        Search for languages by name or code.

        Args:
            query: Search query

        Returns:
            List of matching languages
        """
        query = query.lower()
        results = []

        for lang in self._languages.values():
            # Check code
            if query in lang.code.lower():
                results.append(lang)
                continue

            # Check name
            if query in lang.name.lower():
                results.append(lang)
                continue

            # Check native name
            if query in lang.native_name.lower():
                results.append(lang)
                continue

            # Check aliases
            if any(query in alias.lower() for alias in lang.aliases):
                results.append(lang)
                continue

        return results

    def __len__(self):
        """Get number of registered languages."""
        return len(self._languages)

    def __repr__(self):
        return f"LanguageRegistry({len(self._languages)} languages)"


# ============================================================================
# Global Language Registry
# ============================================================================

# Global registry instance
language_registry = LanguageRegistry()

# Auto-register predefined languages
language_registry.register_all_predefined()


# ============================================================================
# Convenience Functions
# ============================================================================

def get_language(identifier: str) -> Optional[Language]:
    """
    Get language from global registry.

    Args:
        identifier: Language code, name, or alias

    Returns:
        Language instance or None
    """
    return language_registry.get_language(identifier)


def list_languages() -> List[Language]:
    """
    List all registered languages.

    Returns:
        List of Language instances
    """
    return language_registry.list_languages()


def get_vocab_size(language_code: str) -> int:
    """
    Get recommended vocabulary size for a language.

    Args:
        language_code: Language code

    Returns:
        Recommended vocabulary size
    """
    return language_registry.get_recommended_vocab_size(language_code)


def is_language_supported(language_code: str) -> bool:
    """
    Check if a language is supported.

    Args:
        language_code: Language code

    Returns:
        True if language is supported
    """
    return language_registry.has_language(language_code)
