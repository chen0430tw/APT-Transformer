#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vein Subspace Projector

Low-rank subspace projection for VFT/TVA architecture.
Based on the original vft_tva.py implementation.
"""

from apt.core.fake_torch import get_torch
torch = get_torch()
from apt.core.fake_torch import get_torch
torch = get_torch()
nn = torch.nn


class VeinProjector(nn.Module):
    """
    Low-rank projector used by VFT/TVA; provides project() / reconstruct().
    U: R^r -> R^d, V: R^d -> R^r with orthogonal init.

    This is the original implementation from vft_tva.py.
    """
    def __init__(self, d_model: int, rank: int):
        super().__init__()
        assert 1 <= rank < d_model, "rank must be in [1, d_model-1]"
        self.U = nn.Linear(rank, d_model, bias=False)
        self.V = nn.Linear(d_model, rank, bias=False)
        nn.init.orthogonal_(self.U.weight)
        nn.init.orthogonal_(self.V.weight)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """x [B,T,D] -> z [B,T,r]"""
        return self.V(x)

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        """z [B,T,r] -> x_hat [B,T,D]"""
        return self.U(z)


# Alias for backward compatibility with existing code
VeinSubspaceShared = VeinProjector
