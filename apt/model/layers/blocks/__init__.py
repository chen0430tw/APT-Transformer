#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VFT/TVA Blocks Module

Refactored VFT/TVA implementation from original vft_tva.py.
"""

try:
    from apt.model.layers.blocks.vein import (
        VeinProjector,
        VeinSubspaceShared,  # Alias for backward compatibility
    )
except ImportError:
    pass

try:
    from apt.model.layers.blocks.vft_tva import (
        TVAAttention,
        VFTFeedForward,
        NormalCompensator,
        VFTBlock,
        create_vft_block,
        _stable_softmax,
        _off_plane_eps,
    )
except ImportError:
    pass

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
