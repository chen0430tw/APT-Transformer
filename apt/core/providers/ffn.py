#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FFN (Feed-Forward Network) Provider Interface

Defines the interface for FFN implementations.
"""

from abc import abstractmethod
from typing import Dict, Any, Optional
import torch.nn as nn

from apt.core.registry import Provider


class FFNProvider(Provider):
    """
    Abstract interface for feed-forward network providers.

    FFN providers create the feed-forward layers used in transformer blocks.
    Different implementations can provide different architectures (e.g.,
    standard MLP, GLU variants, SwiGLU, etc.).

    Configuration keys (example):
        - d_model: Model dimension
        - d_ff: FFN hidden dimension
        - dropout: Dropout probability
        - activation: Activation function name
    """

    @abstractmethod
    def create_layer(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = 'gelu',
        **kwargs
    ) -> nn.Module:
        """
        Create a feed-forward network layer.

        Args:
            d_model: Model dimension (input/output)
            d_ff: Hidden dimension
            dropout: Dropout probability
            activation: Activation function ('gelu', 'relu', 'swish', etc.)
            **kwargs: Additional implementation-specific parameters

        Returns:
            PyTorch FFN module

        Example:
            provider = registry.get('ffn', 'default')
            ffn = provider.create_layer(d_model=768, d_ff=3072)
        """
        pass

    def get_output_dim(self, d_model: int, d_ff: int) -> int:
        """
        Get the output dimension of the FFN layer.

        Args:
            d_model: Input model dimension
            d_ff: Hidden dimension

        Returns:
            Output dimension (typically same as d_model)
        """
        return d_model

    def get_parameter_count(self, d_model: int, d_ff: int) -> int:
        """
        Estimate parameter count for this FFN.

        Args:
            d_model: Model dimension
            d_ff: Hidden dimension

        Returns:
            Estimated number of parameters
        """
        # Standard FFN: W1 (d_model x d_ff) + b1 + W2 (d_ff x d_model) + b2
        return 2 * d_model * d_ff + d_ff + d_model

    def supports_gating(self) -> bool:
        """
        Check if this FFN implementation supports gating mechanisms (GLU-style).

        Returns:
            True if gating is supported
        """
        return False

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate FFN-specific configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if configuration is valid
        """
        if 'd_model' in config and config['d_model'] <= 0:
            return False
        if 'd_ff' in config and config['d_ff'] <= 0:
            return False
        return True
