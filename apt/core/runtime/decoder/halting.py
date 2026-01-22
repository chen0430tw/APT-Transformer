#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Halting Mechanisms for Reasoning

Provides learned halting signals for adaptive computation in reasoning loops.
"""

from apt.apt_model.utils.fake_torch import get_torch
torch = get_torch()
from apt.apt_model.utils.fake_torch import get_torch
torch = get_torch()
nn = torch.nn
F = torch.nn.functional
from typing import Optional, Dict, Any


def _init_linear(module: nn.Linear, std: float = 0.02):
    """Initialize linear layer weights."""
    nn.init.normal_(module.weight, mean=0.0, std=std)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


class HaltingUnit(nn.Module):
    """
    Learned stop signal for adaptive computation.

    Outputs a probability in (0,1) indicating whether to stop reasoning.
    Higher values indicate higher confidence that reasoning should halt.

    Based on Adaptive Computation Time (ACT) but simplified for reasoning loops.
    """

    def __init__(self, d_model: int, hidden_dim: Optional[int] = None):
        """
        Args:
            d_model: Model dimension
            hidden_dim: Hidden dimension for halting network (default: d_model // 4)
        """
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim or (d_model // 4)

        # Simple 1-layer projection to halt probability
        self.fc = nn.Linear(d_model, 1)
        _init_linear(self.fc)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute halting probability.

        Args:
            h: Hidden states [batch, seq_len, d_model] or [batch, d_model]

        Returns:
            Halting probability [batch, seq_len] or [batch]
        """
        halt_logit = self.fc(h)  # [..., 1]
        halt_prob = torch.sigmoid(halt_logit)
        return halt_prob.squeeze(-1)


class MultiCriteriaHalting(nn.Module):
    """
    Multi-criteria halting mechanism.

    Combines multiple stopping criteria:
    - KL divergence between consecutive predictions
    - Hidden state change magnitude
    - Prediction entropy change
    - Learned halting signal

    Stops when majority of criteria are satisfied OR learned halt probability is high.
    """

    def __init__(
        self,
        d_model: int,
        eps_kl: float = 0.02,
        eps_state: float = 0.03,
        eps_entropy: float = 0.05,
        halt_thresh: float = 0.8,
        use_learned_halt: bool = True,
    ):
        """
        Args:
            d_model: Model dimension
            eps_kl: KL divergence threshold
            eps_state: Hidden state change threshold (relative)
            eps_entropy: Entropy change threshold (relative)
            halt_thresh: Learned halt probability threshold
            use_learned_halt: Whether to use learned halting unit
        """
        super().__init__()
        self.eps_kl = eps_kl
        self.eps_state = eps_state
        self.eps_entropy = eps_entropy
        self.halt_thresh = halt_thresh
        self.use_learned_halt = use_learned_halt

        if use_learned_halt:
            self.halt_unit = HaltingUnit(d_model)
        else:
            self.halt_unit = None

    @staticmethod
    def _kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Compute KL divergence: KL(p || q)

        Args:
            p: First distribution [...]
            q: Second distribution [...]
            eps: Small constant for numerical stability

        Returns:
            KL divergence [...]
        """
        p = p.clamp_min(eps)
        q = q.clamp_min(eps)
        return (p * (p.log() - q.log())).sum(dim=-1)

    @staticmethod
    def _entropy(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Compute entropy of distribution.

        Args:
            p: Distribution [...]
            eps: Small constant for numerical stability

        Returns:
            Entropy [...]
        """
        p = p.clamp_min(eps)
        return -(p * p.log()).sum(dim=-1)

    def forward(
        self,
        h_prev: torch.Tensor,
        h_curr: torch.Tensor,
        logits_prev: Optional[torch.Tensor] = None,
        logits_curr: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute halting decision based on multiple criteria.

        Args:
            h_prev: Previous hidden states [batch, seq_len, d_model]
            h_curr: Current hidden states [batch, seq_len, d_model]
            logits_prev: Previous logits [batch, seq_len, vocab_size] (optional)
            logits_curr: Current logits [batch, seq_len, vocab_size] (optional)

        Returns:
            Dict with:
                - should_halt: Boolean mask [batch, seq_len]
                - kl: KL divergence values
                - state_change: Hidden state change (relative)
                - entropy_change: Entropy change (relative)
                - halt_prob: Learned halt probability (if used)
        """
        # State change criterion
        state_diff = (h_curr - h_prev).norm(dim=-1)  # [batch, seq_len]
        state_norm = h_prev.norm(dim=-1).clamp_min(1e-6)
        state_change = state_diff / state_norm
        state_criterion = state_change < self.eps_state

        result = {
            'state_change': state_change,
            'state_criterion': state_criterion,
        }

        # Prediction criteria (if logits provided)
        if logits_prev is not None and logits_curr is not None:
            p_prev = F.softmax(logits_prev, dim=-1)
            p_curr = F.softmax(logits_curr, dim=-1)

            # KL divergence criterion
            kl = self._kl_divergence(p_curr, p_prev)
            kl_criterion = kl < self.eps_kl

            # Entropy change criterion
            ent_prev = self._entropy(p_prev)
            ent_curr = self._entropy(p_curr)
            ent_change = (ent_prev - ent_curr).abs() / (ent_prev.clamp_min(1e-6))
            ent_criterion = ent_change < self.eps_entropy

            result.update({
                'kl': kl,
                'kl_criterion': kl_criterion,
                'entropy_prev': ent_prev,
                'entropy_curr': ent_curr,
                'entropy_change': ent_change,
                'entropy_criterion': ent_criterion,
            })

            # Combine criteria
            should_halt = state_criterion & kl_criterion & ent_criterion
        else:
            # Only state criterion available
            should_halt = state_criterion

        # Learned halting signal (if used)
        if self.use_learned_halt and self.halt_unit is not None:
            halt_prob = self.halt_unit(h_curr)
            halt_criterion = halt_prob > self.halt_thresh
            should_halt = should_halt | halt_criterion
            result['halt_prob'] = halt_prob
            result['halt_criterion'] = halt_criterion

        result['should_halt'] = should_halt

        return result


class BudgetedHalting(nn.Module):
    """
    Halting with global budget constraint.

    Only allows a certain percentage of tokens to enter extended reasoning.
    Useful for controlling computational cost.
    """

    def __init__(
        self,
        global_budget: float = 0.15,
        entropy_trigger: float = 2.0,
    ):
        """
        Args:
            global_budget: Fraction of tokens allowed to reason (0-1)
            entropy_trigger: Minimum entropy to trigger reasoning
        """
        super().__init__()
        self.global_budget = global_budget
        self.entropy_trigger = entropy_trigger

    def select_tokens(
        self,
        logits: torch.Tensor,
        max_tokens: Optional[int] = None
    ) -> torch.Tensor:
        """
        Select which tokens should enter reasoning based on entropy and budget.

        Args:
            logits: Model logits [batch, seq_len, vocab_size]
            max_tokens: Maximum number of tokens to select (overrides budget)

        Returns:
            Boolean mask [batch, seq_len] indicating which tokens to reason about
        """
        # Compute entropy
        p = F.softmax(logits, dim=-1).clamp_min(1e-8)
        entropy = -(p * p.log()).sum(dim=-1)  # [batch, seq_len]

        batch_size, seq_len = entropy.shape
        total_tokens = batch_size * seq_len

        # Determine budget
        if max_tokens is None:
            k = max(1, int(self.global_budget * total_tokens))
        else:
            k = min(max_tokens, total_tokens)

        # Get top-k high-entropy tokens
        flat_entropy = entropy.flatten()
        if k < flat_entropy.numel():
            threshold = flat_entropy.topk(k).values.min()
        else:
            threshold = flat_entropy.min()

        # Apply both entropy trigger and budget threshold
        final_threshold = max(self.entropy_trigger, float(threshold))
        mask = entropy >= final_threshold

        return mask
