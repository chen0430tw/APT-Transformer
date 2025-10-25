#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vein Subspace Projector

Low-rank subspace projection for VFT/TVA architecture.
Provides efficient computation in r-dimensional vein subspace.
"""

import math
import torch
import torch.nn as nn
from typing import Literal


class VeinProjector(nn.Module):
    """
    Low-rank vein subspace projector.

    Projects from d-dimensional space to r-dimensional vein subspace and back.

    Two implementations:
    - 'linear': Use nn.Linear layers (supports bias, orthogonal init)
    - 'parameter': Use nn.Parameter (memory efficient, custom init)

    Both are functionally equivalent but differ in implementation details.
    """

    def __init__(
        self,
        d_model: int,
        rank: int,
        implementation: Literal['linear', 'parameter'] = 'linear',
        init_method: Literal['orthogonal', 'normal', 'xavier'] = 'orthogonal',
        bias: bool = False,
    ):
        """
        Args:
            d_model: Full model dimension
            rank: Vein subspace rank (1 <= rank < d_model)
            implementation: 'linear' or 'parameter'
            init_method: Initialization method
            bias: Whether to use bias (only for 'linear')
        """
        super().__init__()
        assert 1 <= rank < d_model, f"rank must be in [1, {d_model}), got {rank}"

        self.d_model = d_model
        self.rank = rank
        self.implementation = implementation
        self.init_method = init_method

        if implementation == 'linear':
            # Use nn.Linear: U: r -> d, V: d -> r
            self.U = nn.Linear(rank, d_model, bias=bias)
            self.V = nn.Linear(d_model, rank, bias=bias)
            self._init_linear_weights()

        elif implementation == 'parameter':
            # Use nn.Parameter: U: d x r, V: d x r
            self.U = nn.Parameter(torch.empty(d_model, rank))
            self.V = nn.Parameter(torch.empty(d_model, rank))
            self._init_parameter_weights()

        else:
            raise ValueError(f"Unknown implementation: {implementation}")

    def _init_linear_weights(self):
        """Initialize linear layer weights."""
        if self.init_method == 'orthogonal':
            nn.init.orthogonal_(self.U.weight)
            nn.init.orthogonal_(self.V.weight)
        elif self.init_method == 'normal':
            std = 1.0 / math.sqrt(self.d_model)
            nn.init.normal_(self.U.weight, mean=0.0, std=std)
            nn.init.normal_(self.V.weight, mean=0.0, std=std)
        elif self.init_method == 'xavier':
            nn.init.xavier_uniform_(self.U.weight)
            nn.init.xavier_uniform_(self.V.weight)
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")

    def _init_parameter_weights(self):
        """Initialize parameter weights."""
        if self.init_method == 'orthogonal':
            # Orthogonal initialization for parameters
            nn.init.orthogonal_(self.U)
            nn.init.orthogonal_(self.V)
        elif self.init_method == 'normal':
            std = 1.0 / math.sqrt(self.d_model)
            nn.init.normal_(self.U, mean=0.0, std=std)
            nn.init.normal_(self.V, mean=0.0, std=std)
        elif self.init_method == 'xavier':
            nn.init.xavier_uniform_(self.U)
            nn.init.xavier_uniform_(self.V)
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project from full space to vein subspace.

        Args:
            x: Input tensor [..., d_model]

        Returns:
            z: Projected tensor [..., rank]
        """
        if self.implementation == 'linear':
            return self.V(x)
        else:  # parameter
            return x @ self.V

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct from vein subspace to full space.

        Args:
            z: Vein tensor [..., rank]

        Returns:
            x: Reconstructed tensor [..., d_model]
        """
        if self.implementation == 'linear':
            return self.U(z)
        else:  # parameter
            return z @ self.U.T

    def compute_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute reconstruction error (off-plane distance).

        Args:
            x: Input tensor [..., d_model]

        Returns:
            Error per token [...]
        """
        z = self.project(x)
        x_hat = self.reconstruct(z)
        return torch.norm(x - x_hat, dim=-1)

    def get_compression_ratio(self) -> float:
        """Get parameter compression ratio compared to full rank."""
        # Vein uses 2 * d * r parameters (U and V)
        # Full rank would use d * d parameters
        vein_params = 2 * self.d_model * self.rank
        full_params = self.d_model * self.d_model
        return full_params / vein_params

    def extra_repr(self) -> str:
        """Extra representation for print."""
        return (
            f'd_model={self.d_model}, rank={self.rank}, '
            f'implementation={self.implementation}, '
            f'compression_ratio={self.get_compression_ratio():.1f}x'
        )


# Alias for backward compatibility
VeinSubspaceShared = VeinProjector


def create_vein_projector(
    d_model: int,
    rank: int,
    use_linear: bool = True,
    orthogonal_init: bool = True,
) -> VeinProjector:
    """
    Factory function to create vein projector with common configurations.

    Args:
        d_model: Model dimension
        rank: Vein rank
        use_linear: Use linear implementation (True) or parameter (False)
        orthogonal_init: Use orthogonal initialization

    Returns:
        VeinProjector instance
    """
    implementation = 'linear' if use_linear else 'parameter'
    init_method = 'orthogonal' if orthogonal_init else 'normal'

    return VeinProjector(
        d_model=d_model,
        rank=rank,
        implementation=implementation,
        init_method=init_method,
    )
