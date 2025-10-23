#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model Builder

The ModelBuilder is the core component for assembling APT models using
the provider pattern. It enables flexible, configuration-driven model
construction with swappable components.

Key features:
- Provider-based component creation
- Configuration-driven assembly
- Automatic fallback handling
- Plugin integration
"""

from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
import logging

from apt.core.registry import registry
from apt.core.config import APTConfig

logger = logging.getLogger(__name__)


class ModelBuilder:
    """
    Model assembly builder using the Provider pattern.

    The ModelBuilder uses the registry to look up and instantiate providers,
    then uses those providers to create model components. This enables
    swappable implementations and plugin architecture.

    Usage:
        config = APTConfig.from_yaml('profiles/gpt5_moe_reasoning.yaml')
        builder = ModelBuilder(config)

        # Build individual components
        attention = builder.build_attention(d_model=768, num_heads=12)
        ffn = builder.build_ffn(d_model=768, d_ff=3072)

        # Build entire model
        model = builder.build_model()
    """

    def __init__(self, config: APTConfig):
        """
        Initialize ModelBuilder with configuration.

        Args:
            config: APT configuration
        """
        self.config = config
        self._providers = {}  # Cache for provider instances

        logger.info(f"ModelBuilder initialized with config: {config}")

    def build_attention(
        self,
        d_model: Optional[int] = None,
        num_heads: Optional[int] = None,
        dropout: Optional[float] = None,
        **kwargs
    ) -> nn.Module:
        """
        Build attention layer using configured provider.

        Args:
            d_model: Model dimension (defaults to config.d_model)
            num_heads: Number of attention heads (defaults to config.num_heads)
            dropout: Dropout rate (defaults to config.dropout)
            **kwargs: Additional provider-specific parameters

        Returns:
            Attention layer module

        Example:
            attention = builder.build_attention(d_model=768, num_heads=12)
        """
        # Use config defaults if not specified
        d_model = d_model or self.config.d_model
        num_heads = num_heads or self.config.num_heads
        dropout = dropout or self.config.dropout

        # Get provider
        provider = self._get_provider('attention', self.config.attention_name)

        # Build provider config
        provider_config = self.config.get_provider_config('attention')
        provider_config.update(kwargs)

        # Create layer
        try:
            layer = provider.create_layer(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                **provider_config
            )
            logger.info(
                f"Built attention layer: {self.config.attention_name} "
                f"(d_model={d_model}, num_heads={num_heads})"
            )
            return layer

        except Exception as e:
            logger.error(f"Failed to build attention layer: {e}")
            raise

    def build_ffn(
        self,
        d_model: Optional[int] = None,
        d_ff: Optional[int] = None,
        dropout: Optional[float] = None,
        activation: Optional[str] = None,
        **kwargs
    ) -> nn.Module:
        """
        Build feed-forward network using configured provider.

        Args:
            d_model: Model dimension (defaults to config.d_model)
            d_ff: FFN hidden dimension (defaults to config.d_ff)
            dropout: Dropout rate (defaults to config.dropout)
            activation: Activation function (defaults to config.activation)
            **kwargs: Additional provider-specific parameters

        Returns:
            FFN module

        Example:
            ffn = builder.build_ffn(d_model=768, d_ff=3072)
        """
        d_model = d_model or self.config.d_model
        d_ff = d_ff or self.config.d_ff
        dropout = dropout or self.config.dropout
        activation = activation or self.config.activation

        provider = self._get_provider('ffn', self.config.ffn_name)

        provider_config = self.config.get_provider_config('ffn')
        provider_config.update(kwargs)

        try:
            layer = provider.create_layer(
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                **provider_config
            )
            logger.info(
                f"Built FFN layer: {self.config.ffn_name} "
                f"(d_model={d_model}, d_ff={d_ff})"
            )
            return layer

        except Exception as e:
            logger.error(f"Failed to build FFN layer: {e}")
            raise

    def build_router(
        self,
        d_model: Optional[int] = None,
        num_experts: Optional[int] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> nn.Module:
        """
        Build MoE router using configured provider.

        Args:
            d_model: Model dimension (defaults to config.d_model)
            num_experts: Number of experts
            top_k: Experts selected per token
            **kwargs: Additional provider-specific parameters

        Returns:
            Router module

        Example:
            router = builder.build_router(d_model=768, num_experts=64, top_k=2)
        """
        d_model = d_model or self.config.d_model

        # Get MoE config from extra
        moe_config = self.config.extra.get('moe', {})
        num_experts = num_experts or moe_config.get('experts', 64)
        top_k = top_k or moe_config.get('top_k', 2)

        provider = self._get_provider('router', self.config.router_name)

        provider_config = self.config.get_provider_config('router')
        provider_config.update(kwargs)

        try:
            router = provider.create_router(
                d_model=d_model,
                num_experts=num_experts,
                top_k=top_k,
                **provider_config
            )
            logger.info(
                f"Built router: {self.config.router_name} "
                f"(num_experts={num_experts}, top_k={top_k})"
            )
            return router

        except Exception as e:
            logger.error(f"Failed to build router: {e}")
            raise

    def build_aligner(
        self,
        d_model: Optional[int] = None,
        **kwargs
    ) -> nn.Module:
        """
        Build bistate aligner using configured provider.

        Args:
            d_model: Model dimension (defaults to config.d_model)
            **kwargs: Additional provider-specific parameters

        Returns:
            Aligner module

        Example:
            aligner = builder.build_aligner(d_model=768)
        """
        d_model = d_model or self.config.d_model

        # Get bistate config from extra
        bistate_config = self.config.extra.get('bistate', {})
        alpha = bistate_config.get('alpha', 0.35)
        beta = bistate_config.get('beta', 0.20)

        provider = self._get_provider('align', self.config.align_name)

        provider_config = self.config.get_provider_config('align')
        provider_config.update({'alpha': alpha, 'beta': beta})
        provider_config.update(kwargs)

        try:
            aligner = provider.create_aligner(
                d_model=d_model,
                **provider_config
            )
            logger.info(
                f"Built aligner: {self.config.align_name} "
                f"(d_model={d_model})"
            )
            return aligner

        except Exception as e:
            logger.error(f"Failed to build aligner: {e}")
            raise

    def build_retriever(
        self,
        d_model: Optional[int] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> nn.Module:
        """
        Build retriever using configured provider.

        Args:
            d_model: Model dimension (defaults to config.d_model)
            top_k: Number of documents to retrieve
            **kwargs: Additional provider-specific parameters

        Returns:
            Retriever module

        Example:
            retriever = builder.build_retriever(d_model=768, top_k=5)
        """
        d_model = d_model or self.config.d_model
        top_k = top_k or 5

        provider = self._get_provider('retrieval', self.config.retrieval_name)

        provider_config = self.config.get_provider_config('retrieval')
        provider_config.update(kwargs)

        try:
            retriever = provider.create_retriever(
                d_model=d_model,
                top_k=top_k,
                **provider_config
            )
            logger.info(
                f"Built retriever: {self.config.retrieval_name} "
                f"(d_model={d_model}, top_k={top_k})"
            )
            return retriever

        except Exception as e:
            logger.error(f"Failed to build retriever: {e}")
            raise

    def build_block(
        self,
        d_model: Optional[int] = None,
        num_heads: Optional[int] = None,
        d_ff: Optional[int] = None,
        dropout: Optional[float] = None,
        **kwargs
    ) -> nn.Module:
        """
        Build a complete transformer block.

        A block consists of:
        - Attention layer
        - FFN layer
        - Layer normalization
        - Residual connections

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: FFN hidden dimension
            dropout: Dropout rate
            **kwargs: Additional parameters

        Returns:
            Transformer block module

        Example:
            block = builder.build_block(d_model=768, num_heads=12, d_ff=3072)
        """
        d_model = d_model or self.config.d_model
        num_heads = num_heads or self.config.num_heads
        d_ff = d_ff or self.config.d_ff
        dropout = dropout or self.config.dropout

        # Build components
        attention = self.build_attention(d_model, num_heads, dropout)
        ffn = self.build_ffn(d_model, d_ff, dropout)

        # Create block
        block = TransformerBlock(
            attention=attention,
            ffn=ffn,
            d_model=d_model,
            dropout=dropout,
            layer_norm_eps=self.config.layer_norm_eps
        )

        logger.info(f"Built transformer block (d_model={d_model})")
        return block

    def build_model(self, **kwargs) -> nn.Module:
        """
        Build complete APT model.

        This method assembles the full model architecture using all
        configured providers and plugins.

        Args:
            **kwargs: Override configuration parameters

        Returns:
            Complete APT model

        Example:
            model = builder.build_model()
        """
        logger.info("Building complete APT model...")

        # TODO: Implement full model assembly
        # This will be implemented in the next phase
        raise NotImplementedError(
            "Full model building not yet implemented. "
            "Use build_attention(), build_ffn(), build_block() for now."
        )

    def _get_provider(self, kind: str, name: str):
        """
        Get provider instance from registry (with caching).

        Args:
            kind: Provider type
            name: Provider name

        Returns:
            Provider instance
        """
        key = f"{kind}:{name}"

        if key not in self._providers:
            provider_config = self.config.get_provider_config(kind)
            provider = registry.get(kind, name, provider_config)
            self._providers[key] = provider
            logger.debug(f"Cached provider: {key}")

        return self._providers[key]

    def list_components(self) -> Dict[str, str]:
        """
        List all configured components.

        Returns:
            Dictionary of component types to provider names
        """
        return {
            'attention': self.config.attention_name,
            'ffn': self.config.ffn_name,
            'router': self.config.router_name,
            'align': self.config.align_name,
            'retrieval': self.config.retrieval_name,
        }


# ============================================================================
# Transformer Block
# ============================================================================

class TransformerBlock(nn.Module):
    """
    Standard transformer block with attention and FFN.

    Architecture:
        x -> LayerNorm -> Attention -> Residual
          -> LayerNorm -> FFN -> Residual
    """

    def __init__(
        self,
        attention: nn.Module,
        ffn: nn.Module,
        d_model: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5
    ):
        """
        Initialize transformer block.

        Args:
            attention: Attention module
            ffn: Feed-forward network module
            d_model: Model dimension
            dropout: Dropout rate
            layer_norm_eps: Layer norm epsilon
        """
        super().__init__()

        self.attention = attention
        self.ffn = ffn

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Attention mask [batch, seq_len, seq_len]
            **kwargs: Additional arguments

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Attention sub-layer
        residual = x
        x = self.norm1(x)
        x = self.attention(x, mask=mask, **kwargs)
        x = self.dropout1(x)
        x = x + residual

        # FFN sub-layer
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = x + residual

        return x
