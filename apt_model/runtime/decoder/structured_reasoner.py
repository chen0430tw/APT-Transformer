#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Structured Reasoner

Implements one step of structured reasoning using vein subspace projection,
expert routing, and learned halting.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from apt_model.runtime.decoder.routing import MoELayer, ExpertRouter, MiniExpert
from apt_model.runtime.decoder.halting import HaltingUnit


class StructuredReasoner(nn.Module):
    """
    One reasoning step in o3 style.

    Process:
    1. Project hidden states to vein subspace: z = V^T h
    2. Route to top-k experts: (weights, indices) = Router(z)
    3. Mix expert outputs: z_new = sum_k weight_k * Expert_k(z)
    4. Reconstruct to full dimension: h_new = U z_new
    5. Compute halting probability: p_halt = sigmoid(W h_new)

    The vein subspace allows efficient routing and expert computation
    in a low-rank space while maintaining expressiveness.
    """

    def __init__(
        self,
        vein_projector: nn.Module,
        num_experts: int = 4,
        top_k: int = 2,
        expert_hidden_dim: int = 128,
        use_halting: bool = True,
        residual_blend: bool = True,
        max_blend: float = 0.9,
    ):
        """
        Args:
            vein_projector: VeinSubspaceShared module for projection
            num_experts: Number of experts
            top_k: Number of experts to route to per token
            expert_hidden_dim: Hidden dimension for each expert
            use_halting: Whether to include halting unit
            residual_blend: Whether to blend with residual connection
            max_blend: Maximum blend factor for expert output
        """
        super().__init__()
        self.vein = vein_projector
        self.num_experts = num_experts
        self.top_k = top_k
        self.residual_blend = residual_blend
        self.max_blend = max_blend

        rank = vein_projector.rank

        # MoE layer in vein subspace
        self.moe = MoELayer(
            rank=rank,
            num_experts=num_experts,
            top_k=top_k,
            expert_hidden_dim=expert_hidden_dim,
            load_balancing=True,
        )

        # Halting unit (optional)
        if use_halting:
            self.halt_unit = HaltingUnit(vein_projector.d_model)
        else:
            self.halt_unit = None

    def forward(
        self,
        h: torch.Tensor,
        return_aux_loss: bool = False
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """
        Perform one reasoning step.

        Args:
            h: Hidden states [batch, seq_len, d_model]
            return_aux_loss: Whether to return auxiliary loss

        Returns:
            Tuple of:
                - h_new: Updated hidden states [batch, seq_len, d_model]
                - metadata: Dict with intermediate values
        """
        batch_size, seq_len, d_model = h.shape

        # 1. Project to vein subspace
        z = self.vein.project(h)  # [batch, seq_len, rank]

        # 2. Route to experts and process
        z_new, aux_loss = self.moe(z, return_aux_loss=return_aux_loss)

        # 3. Residual blend (keep some of original z)
        if self.residual_blend:
            # Use router weights as blend factor
            # For simplicity, use a fixed blend based on top_k
            blend_factor = min(self.top_k / self.num_experts, self.max_blend)
            z_final = z_new * blend_factor + z * (1.0 - blend_factor)
        else:
            z_final = z_new

        # 4. Reconstruct to full dimension
        h_new = self.vein.reconstruct(z_final)  # [batch, seq_len, d_model]

        # 5. Compute halting probability
        metadata = {
            'z_old': z,
            'z_new': z_final,
        }

        if self.halt_unit is not None:
            p_halt = self.halt_unit(h_new)  # [batch, seq_len]
            metadata['p_halt'] = p_halt

        if aux_loss is not None:
            metadata['aux_loss'] = aux_loss

        return h_new, metadata


class ChainOfThoughtReasoner(nn.Module):
    """
    Chain-of-Thought style reasoning.

    Performs multiple sequential reasoning steps without explicit halting,
    following a fixed chain of thought.
    """

    def __init__(
        self,
        vein_projector: nn.Module,
        num_steps: int = 3,
        num_experts: int = 4,
        top_k: int = 2,
        expert_hidden_dim: int = 128,
    ):
        """
        Args:
            vein_projector: VeinSubspaceShared module
            num_steps: Number of reasoning steps
            num_experts: Number of experts per step
            top_k: Number of experts to route to
            expert_hidden_dim: Hidden dimension for experts
        """
        super().__init__()
        self.num_steps = num_steps

        # Create one reasoner per step (parameter sharing optional)
        self.reasoners = nn.ModuleList([
            StructuredReasoner(
                vein_projector=vein_projector,
                num_experts=num_experts,
                top_k=top_k,
                expert_hidden_dim=expert_hidden_dim,
                use_halting=False,  # No halting in CoT
            )
            for _ in range(num_steps)
        ])

    def forward(
        self,
        h: torch.Tensor,
        return_intermediates: bool = False
    ) -> tuple[torch.Tensor, Optional[list]]:
        """
        Perform chain-of-thought reasoning.

        Args:
            h: Hidden states [batch, seq_len, d_model]
            return_intermediates: Whether to return intermediate states

        Returns:
            Tuple of:
                - h_final: Final hidden states
                - intermediates: List of intermediate states (if requested)
        """
        intermediates = [] if return_intermediates else None

        for step, reasoner in enumerate(self.reasoners):
            h, metadata = reasoner(h, return_aux_loss=False)

            if return_intermediates:
                intermediates.append({
                    'step': step,
                    'hidden_states': h,
                    'metadata': metadata,
                })

        return h, intermediates


class SelfConsistencyReasoner(nn.Module):
    """
    Self-consistency reasoning with voting.

    Performs multiple independent reasoning chains and aggregates
    their outputs through majority voting or weighted averaging.
    """

    def __init__(
        self,
        vein_projector: nn.Module,
        num_chains: int = 5,
        num_steps_per_chain: int = 3,
        aggregation: str = 'mean',  # 'mean', 'max', 'vote'
        **reasoner_kwargs
    ):
        """
        Args:
            vein_projector: VeinSubspaceShared module
            num_chains: Number of independent reasoning chains
            num_steps_per_chain: Steps per chain
            aggregation: How to aggregate chain outputs
            **reasoner_kwargs: Additional arguments for reasoner
        """
        super().__init__()
        self.num_chains = num_chains
        self.aggregation = aggregation

        # Create independent reasoning chains
        self.chains = nn.ModuleList([
            ChainOfThoughtReasoner(
                vein_projector=vein_projector,
                num_steps=num_steps_per_chain,
                **reasoner_kwargs
            )
            for _ in range(num_chains)
        ])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Perform self-consistency reasoning.

        Args:
            h: Hidden states [batch, seq_len, d_model]

        Returns:
            Aggregated hidden states [batch, seq_len, d_model]
        """
        # Run all chains
        chain_outputs = []
        for chain in self.chains:
            h_chain, _ = chain(h, return_intermediates=False)
            chain_outputs.append(h_chain)

        # Stack outputs [num_chains, batch, seq_len, d_model]
        stacked = torch.stack(chain_outputs, dim=0)

        # Aggregate
        if self.aggregation == 'mean':
            h_final = stacked.mean(dim=0)
        elif self.aggregation == 'max':
            h_final, _ = stacked.max(dim=0)
        elif self.aggregation == 'vote':
            # For voting, we'd need logits - not applicable to hidden states
            # Fall back to mean
            h_final = stacked.mean(dim=0)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return h_final


class TreeOfThoughtsReasoner(nn.Module):
    """
    Tree-of-Thoughts style reasoning.

    Explores multiple reasoning paths simultaneously and prunes
    low-probability branches.

    NOTE: This is a simplified implementation. Full ToT requires
    search algorithms like BFS or beam search.
    """

    def __init__(
        self,
        vein_projector: nn.Module,
        num_branches: int = 3,
        max_depth: int = 3,
        prune_threshold: float = 0.1,
        **reasoner_kwargs
    ):
        """
        Args:
            vein_projector: VeinSubspaceShared module
            num_branches: Number of branches per node
            max_depth: Maximum tree depth
            prune_threshold: Minimum probability to keep a branch
            **reasoner_kwargs: Additional arguments
        """
        super().__init__()
        self.num_branches = num_branches
        self.max_depth = max_depth
        self.prune_threshold = prune_threshold

        # Single reasoner shared across tree
        self.reasoner = StructuredReasoner(
            vein_projector=vein_projector,
            use_halting=True,
            **reasoner_kwargs
        )

    def forward(self, h: torch.Tensor, lm_head: Optional[nn.Module] = None) -> torch.Tensor:
        """
        Perform tree-of-thoughts reasoning.

        Args:
            h: Hidden states [batch, seq_len, d_model]
            lm_head: Language model head for scoring branches

        Returns:
            Best path hidden states [batch, seq_len, d_model]
        """
        # Simplified implementation: just run num_branches independent chains
        # and select the one with highest halt probability

        batch_size, seq_len, d_model = h.shape

        # Expand to multiple branches
        h_expanded = h.unsqueeze(0).expand(self.num_branches, -1, -1, -1)
        h_expanded = h_expanded.reshape(self.num_branches * batch_size, seq_len, d_model)

        # Run reasoning
        best_h = h
        best_score = float('-inf')

        for depth in range(self.max_depth):
            h_new, metadata = self.reasoner(h_expanded, return_aux_loss=False)

            # Score each branch (use halting probability as score)
            if 'p_halt' in metadata:
                scores = metadata['p_halt'].mean(dim=1)  # [num_branches * batch]
                scores = scores.view(self.num_branches, batch_size).mean(dim=1)

                # Find best branch
                best_idx = scores.argmax()
                if scores[best_idx] > best_score:
                    best_score = scores[best_idx]
                    start_idx = best_idx * batch_size
                    end_idx = start_idx + batch_size
                    best_h = h_new[start_idx:end_idx]

            h_expanded = h_new

        return best_h
