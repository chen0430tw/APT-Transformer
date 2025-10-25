#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Expert Routing for Reasoning

Implements token-wise expert routing in low-rank (vein) subspace.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List


def _init_linear(module: nn.Linear, std: float = 0.02):
    """Initialize linear layer weights."""
    nn.init.normal_(module.weight, mean=0.0, std=std)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class ExpertRouter(nn.Module):
    """
    Token-wise expert router in vein subspace.

    Routes each token to top-k experts based on router scores computed
    in the low-rank vein subspace.

    This is more efficient than full-dimensional routing while maintaining
    expressiveness.
    """

    def __init__(
        self,
        rank: int,
        num_experts: int = 4,
        top_k: int = 2,
        noise_std: float = 0.0,
        load_balancing: bool = True,
    ):
        """
        Args:
            rank: Vein subspace dimension
            num_experts: Number of experts
            top_k: Number of experts to route to per token
            noise_std: Noise std for load balancing (default: 0.0)
            load_balancing: Whether to apply load balancing loss
        """
        super().__init__()
        self.rank = rank
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std
        self.load_balancing = load_balancing

        # Router scores: rank -> num_experts
        self.score = nn.Linear(rank, num_experts, bias=False)
        _init_linear(self.score)

    def forward(
        self,
        z: torch.Tensor,
        training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts.

        Args:
            z: Token representations in vein subspace [batch, seq_len, rank]
            training: Whether in training mode (for noise injection)

        Returns:
            Tuple of:
                - top_weights: Weights for top-k experts [batch, seq_len, top_k]
                - top_indices: Indices of top-k experts [batch, seq_len, top_k]
                - all_probs: Full probability distribution [batch, seq_len, num_experts]
        """
        # Compute router logits
        logits = self.score(z)  # [batch, seq_len, num_experts]

        # Add noise for load balancing (during training)
        if training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Softmax to get probabilities
        probs = F.softmax(logits, dim=-1)  # [batch, seq_len, num_experts]

        # Select top-k
        top_weights, top_indices = probs.topk(self.top_k, dim=-1)

        return top_weights, top_indices, probs

    def compute_load_balancing_loss(
        self,
        all_probs: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss.

        Encourages uniform distribution of tokens across experts.

        Args:
            all_probs: Full probability distribution [batch, seq_len, num_experts]
            expert_mask: Optional mask for valid tokens [batch, seq_len]

        Returns:
            Load balancing loss (scalar)
        """
        if not self.load_balancing:
            return torch.tensor(0.0, device=all_probs.device)

        # Average probability per expert
        if expert_mask is not None:
            # Mask out invalid tokens
            all_probs = all_probs * expert_mask.unsqueeze(-1)
            num_tokens = expert_mask.sum()
        else:
            num_tokens = all_probs.shape[0] * all_probs.shape[1]

        mean_probs = all_probs.sum(dim=(0, 1)) / num_tokens  # [num_experts]

        # Ideal uniform distribution
        target = 1.0 / self.num_experts

        # L2 loss to encourage uniformity
        loss = F.mse_loss(mean_probs, torch.full_like(mean_probs, target))

        return loss


class MiniExpert(nn.Module):
    """
    Small expert network operating in vein subspace.

    A lightweight FFN that processes tokens in the low-rank vein subspace.
    Much more parameter-efficient than full-dimensional experts.
    """

    def __init__(
        self,
        rank: int,
        hidden_dim: Optional[int] = None,
        activation: str = 'silu',
        dropout: float = 0.0,
    ):
        """
        Args:
            rank: Vein subspace dimension
            hidden_dim: Hidden dimension (default: 4 * rank)
            activation: Activation function ('silu', 'gelu', 'relu')
            dropout: Dropout probability
        """
        super().__init__()
        self.rank = rank
        self.hidden_dim = hidden_dim or max(64, 4 * rank)

        # Two-layer FFN
        self.fc1 = nn.Linear(rank, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, rank)

        # Activation
        if activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        _init_linear(self.fc1)
        _init_linear(self.fc2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Process tokens in vein subspace.

        Args:
            z: Tokens in vein subspace [batch, seq_len, rank] or [num_tokens, rank]

        Returns:
            Processed tokens [same shape as input]
        """
        h = self.fc1(z)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return h


class MoELayer(nn.Module):
    """
    Mixture-of-Experts layer in vein subspace.

    Combines ExpertRouter with multiple MiniExperts.
    Efficiently routes tokens to specialized experts.
    """

    def __init__(
        self,
        rank: int,
        num_experts: int = 4,
        top_k: int = 2,
        expert_hidden_dim: Optional[int] = None,
        activation: str = 'silu',
        dropout: float = 0.0,
        load_balancing: bool = True,
        load_balance_weight: float = 0.01,
    ):
        """
        Args:
            rank: Vein subspace dimension
            num_experts: Number of experts
            top_k: Number of experts per token
            expert_hidden_dim: Hidden dim for each expert
            activation: Activation function
            dropout: Dropout probability
            load_balancing: Whether to use load balancing
            load_balance_weight: Weight for load balancing loss
        """
        super().__init__()
        self.rank = rank
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight

        # Router
        self.router = ExpertRouter(
            rank=rank,
            num_experts=num_experts,
            top_k=top_k,
            load_balancing=load_balancing,
        )

        # Experts
        self.experts = nn.ModuleList([
            MiniExpert(
                rank=rank,
                hidden_dim=expert_hidden_dim,
                activation=activation,
                dropout=dropout,
            )
            for _ in range(num_experts)
        ])

    def forward(
        self,
        z: torch.Tensor,
        return_aux_loss: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        MoE forward pass.

        Args:
            z: Input in vein subspace [batch, seq_len, rank]
            return_aux_loss: Whether to return auxiliary loss

        Returns:
            Tuple of:
                - output: Processed tokens [batch, seq_len, rank]
                - aux_loss: Load balancing loss (if return_aux_loss=True)
        """
        batch_size, seq_len, rank = z.shape

        # Route tokens
        top_weights, top_indices, all_probs = self.router(z, training=self.training)
        # top_weights: [batch, seq_len, top_k]
        # top_indices: [batch, seq_len, top_k]

        # Process with experts (scatter-gather pattern)
        z_out = torch.zeros_like(z)

        for k in range(self.top_k):
            # Get weights and indices for this k
            weights_k = top_weights[..., k:k+1]  # [batch, seq_len, 1]
            indices_k = top_indices[..., k]  # [batch, seq_len]

            # Process each expert separately
            for expert_id, expert in enumerate(self.experts):
                # Mask for tokens routed to this expert
                mask = (indices_k == expert_id)  # [batch, seq_len]

                if mask.any():
                    # Select tokens for this expert
                    z_selected = z[mask]  # [num_selected, rank]

                    # Process with expert
                    z_processed = expert(z_selected)  # [num_selected, rank]

                    # Scatter back with routing weights
                    z_out[mask] = z_out[mask] + weights_k[mask] * z_processed

        # Compute auxiliary loss
        aux_loss = None
        if return_aux_loss and self.training:
            aux_loss = self.router.compute_load_balancing_loss(all_probs)
            aux_loss = aux_loss * self.load_balance_weight

        return z_out, aux_loss


class SwitchRouter(ExpertRouter):
    """
    Switch Transformer routing (top-1 expert per token).

    Simpler and faster than standard MoE, with comparable performance.
    """

    def __init__(
        self,
        rank: int,
        num_experts: int = 4,
        capacity_factor: float = 1.25,
        **kwargs
    ):
        """
        Args:
            rank: Vein subspace dimension
            num_experts: Number of experts
            capacity_factor: Expert capacity factor
            **kwargs: Additional arguments for ExpertRouter
        """
        super().__init__(
            rank=rank,
            num_experts=num_experts,
            top_k=1,  # Switch uses top-1
            **kwargs
        )
        self.capacity_factor = capacity_factor

    def forward(self, z: torch.Tensor, training: bool = True):
        """Override to enforce capacity constraints."""
        # Use parent's forward for basic routing
        top_weights, top_indices, all_probs = super().forward(z, training)

        # Enforce expert capacity (drop tokens if capacity exceeded)
        batch_size, seq_len = z.shape[:2]
        capacity = int(self.capacity_factor * seq_len / self.num_experts)

        # TODO: Implement capacity enforcement
        # For now, just return standard routing

        return top_weights, top_indices, all_probs
