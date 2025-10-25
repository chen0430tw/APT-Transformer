#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model Blocks

Core building blocks for APT models:
- Vein subspace projection
- VFT/TVA attention and FFN
- Block registry system
"""

from apt_model.modeling.blocks.vein import (
    VeinProjector,
    VeinSubspaceShared,  # Alias for backward compatibility
    create_vein_projector,
)

from apt_model.modeling.blocks.vft_tva import (
    TVAAttention,
    VFTFeedForward,
    NormalCompensator,
    VFTBlock,
    create_vft_block,
)

from apt_model.modeling.blocks.registry import (
    register_attn,
    register_ffn,
    get_attention,
    get_ffn,
    list_attention,
    list_ffn,
)

__all__ = [
    # Vein
    'VeinProjector',
    'VeinSubspaceShared',
    'create_vein_projector',

    # VFT/TVA
    'TVAAttention',
    'VFTFeedForward',
    'NormalCompensator',
    'VFTBlock',
    'create_vft_block',

    # Registry
    'register_attn',
    'register_ffn',
    'get_attention',
    'get_ffn',
    'list_attention',
    'list_ffn',
]
