#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Multilingual Support - Tokenizer Provider Interface

Defines the TokenizerProvider interface for language-specific tokenization.

Key features:
- Provider pattern for tokenizers
- Language-aware tokenization
- Support for different tokenization strategies
- Integration with core Provider system
"""

from abc import abstractmethod
from typing import Dict, Any, List, Optional
import logging

from apt.core.registry import Provider

logger = logging.getLogger(__name__)


class TokenizerProvider(Provider):
    """
    Abstract interface for tokenizer providers.

    Tokenizer providers handle language-specific text tokenization,
    encoding, and decoding. Different implementations can support
    different tokenization strategies (BPE, WordPiece, SentencePiece, etc.).

    Configuration keys (example):
        - vocab_size: Vocabulary size
        - model_path: Path to pretrained tokenizer
        - special_tokens: Additional special tokens
        - lowercase: Whether to lowercase text
        - strip_accents: Whether to strip accents
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize tokenizer provider.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.vocab_size = config.get('vocab_size', 50000)
        self.language = config.get('language', 'en')

    @abstractmethod
    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Tokenize text into tokens.

        Args:
            text: Input text
            **kwargs: Additional arguments

        Returns:
            List of tokens

        Example:
            tokens = tokenizer.tokenize("Hello world!")
            # ['Hello', 'world', '!']
        """
        pass

    @abstractmethod
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        **kwargs
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens (BOS, EOS)
            max_length: Maximum sequence length
            **kwargs: Additional arguments

        Returns:
            List of token IDs

        Example:
            ids = tokenizer.encode("Hello world!")
            # [101, 7592, 2088, 999, 102]
        """
        pass

    @abstractmethod
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional arguments

        Returns:
            Decoded text

        Example:
            text = tokenizer.decode([101, 7592, 2088, 999, 102])
            # "Hello world!"
        """
        pass

    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        **kwargs
    ) -> List[List[int]]:
        """
        Encode multiple texts.

        Args:
            texts: List of texts
            add_special_tokens: Whether to add special tokens
            max_length: Maximum sequence length
            padding: Whether to pad to max_length
            **kwargs: Additional arguments

        Returns:
            List of token ID lists
        """
        encoded = [
            self.encode(text, add_special_tokens, max_length, **kwargs)
            for text in texts
        ]

        if padding and max_length:
            pad_id = self.get_pad_token_id()
            encoded = [
                ids + [pad_id] * (max_length - len(ids))
                for ids in encoded
            ]

        return encoded

    def batch_decode(
        self,
        token_ids_list: List[List[int]],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> List[str]:
        """
        Decode multiple token ID sequences.

        Args:
            token_ids_list: List of token ID lists
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional arguments

        Returns:
            List of decoded texts
        """
        return [
            self.decode(ids, skip_special_tokens, **kwargs)
            for ids in token_ids_list
        ]

    @abstractmethod
    def get_vocab_size(self) -> int:
        """
        Get vocabulary size.

        Returns:
            Vocabulary size
        """
        pass

    @abstractmethod
    def get_vocab(self) -> Dict[str, int]:
        """
        Get vocabulary mapping.

        Returns:
            Dictionary mapping tokens to IDs
        """
        pass

    def get_pad_token_id(self) -> int:
        """Get padding token ID."""
        return 0

    def get_unk_token_id(self) -> int:
        """Get unknown token ID."""
        return 1

    def get_bos_token_id(self) -> int:
        """Get beginning-of-sequence token ID."""
        return 2

    def get_eos_token_id(self) -> int:
        """Get end-of-sequence token ID."""
        return 3

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Convert tokens to IDs.

        Args:
            tokens: List of tokens

        Returns:
            List of token IDs
        """
        vocab = self.get_vocab()
        unk_id = self.get_unk_token_id()
        return [vocab.get(token, unk_id) for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """
        Convert IDs to tokens.

        Args:
            ids: List of token IDs

        Returns:
            List of tokens
        """
        id_to_token = {v: k for k, v in self.get_vocab().items()}
        unk_token = "<unk>"
        return [id_to_token.get(id, unk_token) for id in ids]

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save tokenizer to directory.

        Args:
            save_directory: Directory to save tokenizer
        """
        raise NotImplementedError("save_pretrained not implemented")

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Load tokenizer from pretrained model.

        Args:
            model_path: Path to model directory
            **kwargs: Additional configuration

        Returns:
            Tokenizer instance
        """
        raise NotImplementedError("from_pretrained not implemented")

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate tokenizer-specific configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if configuration is valid
        """
        if 'vocab_size' in config and config['vocab_size'] <= 0:
            return False
        return True

    def supports_language(self, language_code: str) -> bool:
        """
        Check if tokenizer supports a language.

        Args:
            language_code: Language code (e.g., 'en', 'zh')

        Returns:
            True if language is supported
        """
        return True  # Override in subclasses

    def get_language(self) -> str:
        """
        Get the language this tokenizer is configured for.

        Returns:
            Language code
        """
        return self.language

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"lang={self.language}, "
            f"vocab_size={self.vocab_size})"
        )


# ============================================================================
# Tokenizer Utilities
# ============================================================================

class TokenizerConfig:
    """Configuration for tokenizer providers."""

    def __init__(
        self,
        vocab_size: int = 50000,
        language: str = 'en',
        lowercase: bool = False,
        strip_accents: bool = False,
        special_tokens: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Initialize tokenizer configuration.

        Args:
            vocab_size: Vocabulary size
            language: Language code
            lowercase: Whether to lowercase
            strip_accents: Whether to strip accents
            special_tokens: Special token mapping
            **kwargs: Additional configuration
        """
        self.vocab_size = vocab_size
        self.language = language
        self.lowercase = lowercase
        self.strip_accents = strip_accents
        self.special_tokens = special_tokens or {}
        self.extra = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'language': self.language,
            'lowercase': self.lowercase,
            'strip_accents': self.strip_accents,
            'special_tokens': self.special_tokens,
            **self.extra
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TokenizerConfig':
        """Create from dictionary."""
        return cls(**data)


def get_tokenizer_for_language(language_code: str, **kwargs) -> Optional[TokenizerProvider]:
    """
    Get a tokenizer for a specific language.

    Args:
        language_code: Language code
        **kwargs: Additional configuration

    Returns:
        Tokenizer instance or None

    Example:
        tokenizer = get_tokenizer_for_language('zh', vocab_size=21128)
    """
    from apt.core.registry import registry

    # Try to get language-specific tokenizer
    tokenizer_name = f"tokenizer_{language_code}"
    try:
        config = TokenizerConfig(language=language_code, **kwargs).to_dict()
        return registry.get('tokenizer', tokenizer_name, config, fallback=True)
    except Exception as e:
        logger.warning(f"Failed to get tokenizer for {language_code}: {e}")
        return None
