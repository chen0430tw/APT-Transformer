#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Block Registry System

Factory pattern registry for attention and FFN implementations.
Allows runtime selection via config: --attn.impl tva --ffn.impl vft
"""

from typing import Dict, Callable, Any, Optional
from apt.apt_model.utils.fake_torch import get_torch
torch = get_torch()
nn = torch.nn


# ============================================================================
# Registry
# ============================================================================

class BlockRegistry:
    """Registry for attention and FFN implementations."""

    def __init__(self):
        self._attention_registry: Dict[str, Callable] = {}
        self._ffn_registry: Dict[str, Callable] = {}

    def register_attention(self, name: str):
        """
        Decorator to register an attention implementation.

        Usage:
            @register_attn("tva")
            class TVAAttention(nn.Module):
                ...
        """
        def decorator(cls):
            self._attention_registry[name] = cls
            return cls
        return decorator

    def register_ffn(self, name: str):
        """
        Decorator to register an FFN implementation.

        Usage:
            @register_ffn("vft")
            class VFTFeedForward(nn.Module):
                ...
        """
        def decorator(cls):
            self._ffn_registry[name] = cls
            return cls
        return decorator

    def get_attention(self, name: str, **kwargs) -> nn.Module:
        """
        Get attention implementation by name.

        Args:
            name: Attention implementation name
            **kwargs: Arguments for attention constructor

        Returns:
            Attention module instance

        Example:
            attn = registry.get_attention('tva', d_model=768, n_heads=12, rank=4)
        """
        if name not in self._attention_registry:
            raise ValueError(
                f"Unknown attention implementation: {name}. "
                f"Available: {list(self._attention_registry.keys())}"
            )
        return self._attention_registry[name](**kwargs)

    def get_ffn(self, name: str, **kwargs) -> nn.Module:
        """
        Get FFN implementation by name.

        Args:
            name: FFN implementation name
            **kwargs: Arguments for FFN constructor

        Returns:
            FFN module instance

        Example:
            ffn = registry.get_ffn('vft', d_model=768, rank=4)
        """
        if name not in self._ffn_registry:
            raise ValueError(
                f"Unknown FFN implementation: {name}. "
                f"Available: {list(self._ffn_registry.keys())}"
            )
        return self._ffn_registry[name](**kwargs)

    def list_attention(self):
        """List all registered attention implementations."""
        return list(self._attention_registry.keys())

    def list_ffn(self):
        """List all registered FFN implementations."""
        return list(self._ffn_registry.keys())


# Global registry instance
_global_registry = BlockRegistry()


# ============================================================================
# Convenience Decorators
# ============================================================================

def register_attn(name: str):
    """
    Decorator to register attention implementation.

    Usage:
        @register_attn("tva")
        class TVAAttention(nn.Module):
            ...
    """
    return _global_registry.register_attention(name)


def register_ffn(name: str):
    """
    Decorator to register FFN implementation.

    Usage:
        @register_ffn("vft")
        class VFTFeedForward(nn.Module):
            ...
    """
    return _global_registry.register_ffn(name)


def get_attention(name: str, **kwargs) -> nn.Module:
    """Get attention by name."""
    return _global_registry.get_attention(name, **kwargs)


def get_ffn(name: str, **kwargs) -> nn.Module:
    """Get FFN by name."""
    return _global_registry.get_ffn(name, **kwargs)


def list_attention():
    """List registered attention implementations."""
    return _global_registry.list_attention()


def list_ffn():
    """List registered FFN implementations."""
    return _global_registry.list_ffn()


# ============================================================================
# Standard Implementations
# ============================================================================

def register_standard_blocks():
    """Register standard attention and FFN implementations."""

    # Import here to avoid circular dependency
    from apt.apt_model.modeling.blocks.vft_tva import TVAAttention, VFTFeedForward

    # Register TVA variants
    register_attn("tva")(TVAAttention)

    # Register VFT variants
    register_ffn("vft")(VFTFeedForward)

    # Standard multi-head attention (for comparison)
    @register_attn("standard")
    class StandardMultiHeadAttention(nn.Module):
        """Standard multi-head attention for baseline."""
        def __init__(
            self,
            d_model: int,
            n_heads: int,
            dropout: float = 0.0,
            **kwargs  # Ignore extra args for compatibility
        ):
            super().__init__()
            assert d_model % n_heads == 0
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_head = d_model // n_heads

            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, attn_mask=None, **kwargs):
            B, T, D = x.shape
            q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
            k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

            import math
            scale = 1.0 / math.sqrt(self.d_head)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if attn_mask is not None:
                scores = scores + attn_mask

            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(B, T, D)
            out = self.out_proj(out)
            return out, None

    # Standard FFN (for comparison)
    @register_ffn("standard")
    class StandardFFN(nn.Module):
        """Standard feed-forward network for baseline."""
        def __init__(
            self,
            d_model: int,
            d_ff: Optional[int] = None,
            dropout: float = 0.0,
            activation: str = 'gelu',
            **kwargs  # Ignore extra args for compatibility
        ):
            super().__init__()
            d_ff = d_ff or 4 * d_model

            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout)

            if activation == 'gelu':
                self.act = nn.GELU()
            elif activation == 'relu':
                self.act = nn.ReLU()
            elif activation == 'silu':
                self.act = nn.SiLU()
            else:
                self.act = nn.GELU()

        def forward(self, x):
            h = self.fc1(x)
            h = self.act(h)
            h = self.dropout(h)
            h = self.fc2(h)
            return h


# Auto-register on import
register_standard_blocks()
