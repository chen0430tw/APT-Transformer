#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Attention Provider Interface

Defines the interface for attention mechanism implementations.
All attention plugins must implement this interface.
"""

from abc import abstractmethod
from typing import Dict, Any, Optional
import torch.nn as nn

from apt.core.registry import Provider


class AttentionProvider(Provider):
    """
    Abstract interface for attention mechanism providers.

    Attention providers are responsible for creating attention layers
    that can be used in transformer models. Different implementations
    can provide different attention mechanisms (e.g., TVA, Flash, Linear, etc.).

    Configuration keys (example):
        - d_model: Model dimension
        - num_heads: Number of attention heads
        - dropout: Dropout probability
        - Implementation-specific parameters
    """

    @abstractmethod
    def create_layer(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        **kwargs
    ) -> nn.Module:
        """
        Create an attention layer instance.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            **kwargs: Additional implementation-specific parameters

        Returns:
            PyTorch attention layer module

        Example:
            provider = registry.get('attention', 'tva_default')
            attention = provider.create_layer(d_model=768, num_heads=12)
        """
        pass

    def get_output_dim(self, d_model: int) -> int:
        """
        Get the output dimension of the attention layer.

        Args:
            d_model: Input model dimension

        Returns:
            Output dimension (typically same as d_model)
        """
        return d_model

    def supports_masking(self) -> bool:
        """
        Check if this attention implementation supports causal masking.

        Returns:
            True if masking is supported
        """
        return True

    def supports_kv_cache(self) -> bool:
        """
        Check if this attention implementation supports KV caching for inference.

        Returns:
            True if KV cache is supported
        """
        return False

    def get_flops_estimate(
        self,
        seq_len: int,
        d_model: int,
        num_heads: int
    ) -> int:
        """
        Estimate FLOPs for this attention mechanism.

        Args:
            seq_len: Sequence length
            d_model: Model dimension
            num_heads: Number of heads

        Returns:
            Estimated FLOPs count
        """
        # Standard attention: O(n^2 * d)
        return 4 * seq_len * seq_len * d_model

    def get_memory_estimate(
        self,
        seq_len: int,
        d_model: int,
        num_heads: int,
        batch_size: int = 1
    ) -> int:
        """
        Estimate memory usage in bytes.

        Args:
            seq_len: Sequence length
            d_model: Model dimension
            num_heads: Number of heads
            batch_size: Batch size

        Returns:
            Estimated memory in bytes
        """
        # QKV matrices + attention scores
        # Assuming float32 (4 bytes)
        qkv_memory = 3 * batch_size * seq_len * d_model * 4
        scores_memory = batch_size * num_heads * seq_len * seq_len * 4
        return qkv_memory + scores_memory

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate attention-specific configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if configuration is valid
        """
        # Basic validation - subclasses should override
        if 'd_model' in config:
            if config['d_model'] <= 0:
                return False
        if 'num_heads' in config:
            if config['num_heads'] <= 0:
                return False
            if 'd_model' in config and config['d_model'] % config['num_heads'] != 0:
                return False
        return True
