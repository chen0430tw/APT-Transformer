#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Align Provider Interface

Defines the interface for bistate alignment implementations.
"""

from abc import abstractmethod
from typing import Dict, Any, Optional, Tuple
from apt_model.utils.fake_torch import get_torch
torch = get_torch()
from apt_model.utils.fake_torch import get_torch
torch = get_torch()
nn = torch.nn

from apt.core.registry import Provider


class AlignProvider(Provider):
    """
    Abstract interface for bistate alignment providers.

    Alignment providers implement mechanisms to align model states during
    training, typically used for consistency regularization or multi-state
    training strategies.

    Configuration keys (example):
        - d_model: Model dimension
        - alpha: Stable state weight
        - beta: Align state weight
        - tau_align: Alignment temperature
        - align_loss_weight: Weight for alignment loss
    """

    @abstractmethod
    def create_aligner(
        self,
        d_model: int,
        alpha: float = 0.35,
        beta: float = 0.20,
        **kwargs
    ) -> nn.Module:
        """
        Create an alignment module.

        Args:
            d_model: Model dimension
            alpha: Stable state weight
            beta: Align state weight
            **kwargs: Additional implementation-specific parameters

        Returns:
            PyTorch alignment module

        Example:
            provider = registry.get('align', 'bistate_default')
            aligner = provider.create_aligner(d_model=768)
        """
        pass

    @abstractmethod
    def compute_alignment(
        self,
        aligner: nn.Module,
        state_1: torch.Tensor,
        state_2: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute alignment between two states.

        Args:
            aligner: Alignment module instance
            state_1: First state tensor [batch, seq_len, d_model]
            state_2: Second state tensor [batch, seq_len, d_model]
            **kwargs: Additional parameters

        Returns:
            Tuple of:
            - aligned_state: Aligned output [batch, seq_len, d_model]
            - alignment_loss: Alignment loss scalar

        Example:
            aligned, loss = provider.compute_alignment(aligner, stable_state, align_state)
        """
        pass

    def compute_consistency_loss(
        self,
        state_1: torch.Tensor,
        state_2: torch.Tensor,
        metric: str = 'cosine'
    ) -> torch.Tensor:
        """
        Compute consistency loss between two states.

        Args:
            state_1: First state tensor
            state_2: Second state tensor
            metric: Distance metric ('cosine', 'l2', 'kl')

        Returns:
            Consistency loss scalar
        """
        if metric == 'cosine':
            # Cosine similarity loss
            cos_sim = torch.nn.functional.cosine_similarity(
                state_1.flatten(0, 1),
                state_2.flatten(0, 1),
                dim=-1
            )
            loss = 1.0 - cos_sim.mean()
            return loss

        elif metric == 'l2':
            # L2 distance
            loss = torch.nn.functional.mse_loss(state_1, state_2)
            return loss

        elif metric == 'kl':
            # KL divergence (assuming log probabilities)
            loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(state_1, dim=-1),
                torch.nn.functional.softmax(state_2, dim=-1),
                reduction='batchmean'
            )
            return loss

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def get_interpolation_weight(
        self,
        epoch: int,
        max_epochs: int,
        warmup: int = 5
    ) -> float:
        """
        Get interpolation weight for alignment (curriculum schedule).

        Args:
            epoch: Current epoch
            max_epochs: Total epochs
            warmup: Number of warmup epochs

        Returns:
            Weight in [0, 1]
        """
        if epoch < warmup:
            return 0.0
        else:
            # Linear increase after warmup
            return min(1.0, (epoch - warmup) / (max_epochs - warmup))

    def supports_multi_state(self) -> bool:
        """
        Check if this aligner supports more than 2 states.

        Returns:
            True if multi-state alignment is supported
        """
        return False

    def supports_asymmetric_alignment(self) -> bool:
        """
        Check if this aligner supports asymmetric alignment (different weights for states).

        Returns:
            True if asymmetric alignment is supported
        """
        return True

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate alignment-specific configuration.

        Args:
            config: Configuration dictionary

        Returns:
            True if configuration is valid
        """
        if 'd_model' in config and config['d_model'] <= 0:
            return False

        if 'alpha' in config:
            if not (0.0 <= config['alpha'] <= 1.0):
                return False

        if 'beta' in config:
            if not (0.0 <= config['beta'] <= 1.0):
                return False

        if 'tau_align' in config:
            if config['tau_align'] <= 0:
                return False

        return True
