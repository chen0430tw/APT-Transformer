#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Router Provider Interface

Defines the interface for MoE (Mixture of Experts) routing implementations.
"""

from abc import abstractmethod
from typing import Dict, Any, Optional, Tuple
from apt.core.fake_torch import get_torch
torch = get_torch()
from apt.core.fake_torch import get_torch
torch = get_torch()
nn = torch.nn

from apt.core.registry import Provider


class RouterProvider(Provider):
    """
    Abstract interface for MoE router providers.

    Router providers implement the logic for selecting which experts to use
    in a Mixture of Experts layer. Different routing strategies can be
    implemented (e.g., top-k, learned, random, etc.).

    Configuration keys (example):
        - num_experts: Total number of experts
        - top_k: Number of experts to select per token
        - capacity_factor: Expert capacity multiplier
        - temperature: Routing temperature
        - load_balance_weight: Load balancing loss weight
    """

    @abstractmethod
    def create_router(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        **kwargs
    ) -> nn.Module:
        """
        Create a router module for MoE.

        Args:
            d_model: Model dimension (router input)
            num_experts: Total number of experts
            top_k: Number of experts to select per token
            **kwargs: Additional implementation-specific parameters

        Returns:
            PyTorch router module that outputs expert selection and weights

        Example:
            provider = registry.get('router', 'topk_default')
            router = provider.create_router(d_model=768, num_experts=64, top_k=2)
        """
        pass

    @abstractmethod
    def route(
        self,
        router: nn.Module,
        inputs: torch.Tensor,
        train: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform routing operation.

        Args:
            router: Router module instance
            inputs: Input tensor [batch, seq_len, d_model]
            train: Whether in training mode

        Returns:
            Tuple of:
            - expert_indices: Selected expert indices [batch, seq_len, top_k]
            - expert_weights: Selection weights [batch, seq_len, top_k]
            - aux_losses: Dictionary of auxiliary losses (e.g., load balance)

        Example:
            indices, weights, losses = provider.route(router, hidden_states)
        """
        pass

    def compute_load_balance_loss(
        self,
        router_logits: torch.Tensor,
        expert_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss.

        Args:
            router_logits: Raw router logits [batch, seq_len, num_experts]
            expert_mask: Expert selection mask [batch, seq_len, num_experts]

        Returns:
            Load balance loss scalar
        """
        # Default implementation - subclasses can override
        num_experts = router_logits.size(-1)

        # Fraction of tokens routed to each expert
        router_probs = torch.softmax(router_logits, dim=-1)
        expert_usage = expert_mask.float().mean(dim=[0, 1])  # [num_experts]

        # Target uniform distribution
        target = torch.ones_like(expert_usage) / num_experts

        # L2 loss
        loss = torch.mean((expert_usage - target) ** 2)
        return loss

    def get_capacity(
        self,
        batch_size: int,
        seq_len: int,
        num_experts: int,
        top_k: int,
        capacity_factor: float = 1.25
    ) -> int:
        """
        Calculate expert capacity (max tokens per expert).

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            num_experts: Number of experts
            top_k: Experts selected per token
            capacity_factor: Capacity multiplier

        Returns:
            Expert capacity (number of tokens)
        """
        total_tokens = batch_size * seq_len
        tokens_per_expert = (total_tokens * top_k) / num_experts
        capacity = int(tokens_per_expert * capacity_factor)
        return max(capacity, top_k)  # At least top_k

    def supports_dynamic_capacity(self) -> bool:
        """
        Check if this router supports dynamic capacity adjustment.

        Returns:
            True if dynamic capacity is supported
        """
        return False

    def supports_expert_choice(self) -> bool:
        """
        Check if this router supports expert-choice routing (experts choose tokens).

        Returns:
            True if expert-choice is supported
        """
        return False

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate router-specific configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if configuration is valid
        """
        if 'num_experts' in config:
            if config['num_experts'] <= 0:
                return False

        if 'top_k' in config:
            if config['top_k'] <= 0:
                return False
            if 'num_experts' in config and config['top_k'] > config['num_experts']:
                return False

        if 'capacity_factor' in config:
            if config['capacity_factor'] <= 0:
                return False

        return True
