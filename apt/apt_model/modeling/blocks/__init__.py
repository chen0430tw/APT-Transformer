#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VFT/TVA Blocks Module

Refactored VFT/TVA implementation from original vft_tva.py.
"""

from apt.apt_model.modeling.blocks.vein import (
    VeinProjector,
    VeinSubspaceShared,  # Alias for backward compatibility
)

from apt.apt_model.modeling.blocks.vft_tva import (
    TVAAttention,
    VFTFeedForward,
    NormalCompensator,
    VFTBlock,
    create_vft_block,
    _stable_softmax,
    _off_plane_eps,
)

__all__ = [
    # Vein projector
    'VeinProjector',
    'VeinSubspaceShared',
    # VFT/TVA components
    'TVAAttention',
    'VFTFeedForward',
    'NormalCompensator',
    'VFTBlock',
    # Factory functions
    'create_vft_block',
    # Utilities
    '_stable_softmax',
    '_off_plane_eps',
]
