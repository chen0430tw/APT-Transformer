#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reasoning Controller

Orchestrates multi-step iterative reasoning with adaptive halting.
"""

from apt_model.utils.fake_torch import get_torch
torch = get_torch()
from apt_model.utils.fake_torch import get_torch
torch = get_torch()
nn = torch.nn
from typing import Optional, Dict, Any, Tuple

from apt.core.runtime.decoder.structured_reasoner import StructuredReasoner
from apt.core.runtime.decoder.halting import MultiCriteriaHalting, BudgetedHalting


class ReasoningController(nn.Module):
    """
    Multi-step reasoning controller with adaptive halting.

    Controls the iterative reasoning loop:
    1. Run structured reasoning steps
    2. Check multiple halting criteria (KL, state change, entropy, learned signal)
    3. Stop when convergence detected or max steps reached
    4. Track patience (consecutive non-convergence steps)

    Based on GPT-o3's reasoning controller.
    """

    def __init__(
        self,
        vein_projector: nn.Module,
        max_steps: int = 6,
        patience: int = 2,
        eps_kl: float = 0.02,
        eps_state: float = 0.03,
        eps_entropy: float = 0.05,
        halt_thresh: float = 0.8,
        num_experts: int = 4,
        top_k_experts: int = 2,
        expert_hidden_dim: int = 128,
    ):
        """
        Args:
            vein_projector: VeinSubspaceShared module
            max_steps: Maximum reasoning steps
            patience: Max consecutive non-convergence steps before forcing stop
            eps_kl: KL divergence threshold for convergence
            eps_state: State change threshold for convergence
            eps_entropy: Entropy change threshold for convergence
            halt_thresh: Learned halt probability threshold
            num_experts: Number of experts in reasoner
            top_k_experts: Number of experts to route to
            expert_hidden_dim: Hidden dimension for experts
        """
        super().__init__()
        self.vein = vein_projector
        self.max_steps = max(1, int(max_steps))
        self.patience = max(1, int(patience))

        # Structured reasoner
        self.reasoner = StructuredReasoner(
            vein_projector=vein_projector,
            num_experts=num_experts,
            top_k=top_k_experts,
            expert_hidden_dim=expert_hidden_dim,
            use_halting=True,
        )

        # Multi-criteria halting
        self.halting = MultiCriteriaHalting(
            d_model=vein_projector.d_model,
            eps_kl=eps_kl,
            eps_state=eps_state,
            eps_entropy=eps_entropy,
            halt_thresh=halt_thresh,
            use_learned_halt=True,
        )

    def forward(
        self,
        h: torch.Tensor,
        lm_head: nn.Module,
        return_details: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Run multi-step reasoning with adaptive halting.

        Args:
            h: Initial hidden states [batch, seq_len, d_model] or [num_tokens, d_model]
            lm_head: Language model head for computing logits
            return_details: Whether to return detailed step-by-step info

        Returns:
            Tuple of:
                - h_final: Final hidden states (same shape as input)
                - info: Dict with:
                    - steps: Number of steps taken
                    - converged: Whether convergence was reached
                    - step_details: List of per-step info (if return_details=True)
        """
        # Handle 2D input [num_tokens, d_model]
        reshape_back = False
        if h.dim() == 2:
            h = h.unsqueeze(0)  # [1, num_tokens, d_model]
            reshape_back = True

        batch_size, seq_len, d_model = h.shape

        # Initial prediction
        logits_prev = lm_head(h)  # [batch, seq_len, vocab_size]
        h_prev = h

        # Track stall count per token
        stall_count = torch.zeros(batch_size, seq_len, dtype=torch.long, device=h.device)

        # Step details (if requested)
        step_details = [] if return_details else None

        # Reasoning loop
        converged = False
        steps_taken = 0

        for step in range(self.max_steps):
            steps_taken = step + 1

            # Perform one reasoning step
            h, reasoner_meta = self.reasoner(h, return_aux_loss=True)

            # Compute new logits
            logits = lm_head(h)

            # Check halting criteria
            halt_info = self.halting(
                h_prev=h_prev,
                h_curr=h,
                logits_prev=logits_prev,
                logits_curr=logits,
            )

            should_halt = halt_info['should_halt']  # [batch, seq_len]

            # Update stall count
            # Increment for tokens that haven't converged
            stall_count = stall_count + (~should_halt).long()

            # Check if all tokens either converged or exceeded patience
            all_done = (should_halt | (stall_count >= self.patience)).all()

            # Save step details
            if return_details:
                step_details.append({
                    'step': step,
                    'converged_mask': should_halt,
                    'stall_count': stall_count.clone(),
                    'kl': halt_info.get('kl'),
                    'state_change': halt_info.get('state_change'),
                    'entropy_change': halt_info.get('entropy_change'),
                    'halt_prob': halt_info.get('halt_prob'),
                    'aux_loss': reasoner_meta.get('aux_loss'),
                })

            if all_done:
                converged = should_halt.all().item()
                break

            # Update prev state
            h_prev = h
            logits_prev = logits

        # Reshape back if needed
        if reshape_back:
            h = h.squeeze(0)

        info = {
            'steps': steps_taken,
            'converged': converged,
        }

        if return_details:
            info['step_details'] = step_details

        return h, info


class BudgetedReasoningController(nn.Module):
    """
    Reasoning controller with global budget constraints.

    Only allows a fraction of tokens to enter reasoning, based on:
    1. Prediction entropy (uncertain tokens)
    2. Global budget (computational cost)

    Useful for controlling inference cost while maintaining quality
    on difficult tokens.
    """

    def __init__(
        self,
        vein_projector: nn.Module,
        global_budget: float = 0.15,
        entropy_trigger: float = 2.0,
        max_steps: int = 6,
        patience: int = 2,
        **controller_kwargs
    ):
        """
        Args:
            vein_projector: VeinSubspaceShared module
            global_budget: Fraction of tokens allowed to reason
            entropy_trigger: Minimum entropy to trigger reasoning
            max_steps: Maximum reasoning steps
            patience: Max consecutive non-convergence steps
            **controller_kwargs: Additional arguments for ReasoningController
        """
        super().__init__()
        self.global_budget = global_budget
        self.entropy_trigger = entropy_trigger

        # Budget selector
        self.budget_selector = BudgetedHalting(
            global_budget=global_budget,
            entropy_trigger=entropy_trigger,
        )

        # Reasoning controller
        self.controller = ReasoningController(
            vein_projector=vein_projector,
            max_steps=max_steps,
            patience=patience,
            **controller_kwargs
        )

    def forward(
        self,
        h: torch.Tensor,
        lm_head: nn.Module,
        return_details: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Run budgeted reasoning.

        Args:
            h: Hidden states [batch, seq_len, d_model]
            lm_head: Language model head
            return_details: Whether to return detailed info

        Returns:
            Tuple of:
                - h_final: Final hidden states
                - info: Dict with reasoning info
        """
        batch_size, seq_len, d_model = h.shape

        # Compute initial logits
        logits = lm_head(h)

        # Select tokens for reasoning based on budget
        reasoning_mask = self.budget_selector.select_tokens(logits)  # [batch, seq_len]

        if not reasoning_mask.any():
            # No tokens selected for reasoning
            return h, {
                'steps': 0,
                'converged': True,
                'num_reasoning_tokens': 0,
            }

        # Extract tokens for reasoning
        indices = reasoning_mask.nonzero(as_tuple=False)  # [num_selected, 2]
        h_selected = h[indices[:, 0], indices[:, 1], :]  # [num_selected, d_model]

        # Run reasoning on selected tokens
        h_updated, info = self.controller(
            h_selected,
            lm_head,
            return_details=return_details
        )

        # Scatter updated tokens back
        h_final = h.clone()
        h_final[indices[:, 0], indices[:, 1], :] = h_updated

        # Add budget info
        info['num_reasoning_tokens'] = indices.shape[0]
        info['reasoning_fraction'] = indices.shape[0] / (batch_size * seq_len)
        info['reasoning_mask'] = reasoning_mask

        return h_final, info


class AdaptiveBudgetController(nn.Module):
    """
    Adaptive budget controller.

    Dynamically adjusts the reasoning budget based on model performance.
    - Increase budget when model is uncertain
    - Decrease budget when model is confident
    """

    def __init__(
        self,
        vein_projector: nn.Module,
        init_budget: float = 0.15,
        min_budget: float = 0.05,
        max_budget: float = 0.50,
        adaptation_rate: float = 0.1,
        **controller_kwargs
    ):
        """
        Args:
            vein_projector: VeinSubspaceShared module
            init_budget: Initial budget
            min_budget: Minimum budget
            max_budget: Maximum budget
            adaptation_rate: How quickly to adapt budget
            **controller_kwargs: Additional arguments
        """
        super().__init__()
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.adaptation_rate = adaptation_rate

        # Current budget (learnable parameter)
        self.register_buffer('current_budget', torch.tensor(init_budget))

        # Budgeted controller
        self.controller = BudgetedReasoningController(
            vein_projector=vein_projector,
            global_budget=init_budget,
            **controller_kwargs
        )

    def adapt_budget(self, avg_entropy: float, avg_steps: float):
        """
        Adapt budget based on recent performance.

        Args:
            avg_entropy: Average prediction entropy
            avg_steps: Average reasoning steps taken
        """
        # High entropy -> increase budget
        # Low entropy -> decrease budget
        # Many steps -> increase budget (model struggling)
        # Few steps -> decrease budget (model confident)

        # Entropy target (log of vocab_size / 2)
        entropy_target = 2.0

        # Step target
        step_target = 3.0

        # Compute adjustment
        entropy_factor = (avg_entropy - entropy_target) / entropy_target
        step_factor = (avg_steps - step_target) / step_target

        adjustment = (entropy_factor + step_factor) / 2 * self.adaptation_rate

        # Update budget
        new_budget = self.current_budget + adjustment
        new_budget = torch.clamp(new_budget, self.min_budget, self.max_budget)

        self.current_budget.copy_(new_budget)
        self.controller.global_budget = float(new_budget)

    def forward(self, h: torch.Tensor, lm_head: nn.Module, **kwargs):
        """Forward pass (delegates to budgeted controller)."""
        return self.controller(h, lm_head, **kwargs)
