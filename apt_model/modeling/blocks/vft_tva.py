#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VFT/TVA: Vein-Flow Transformer / Tri-Vein Attention

Core implementation of VFT/TVA architecture with:
- TVAAttention: Attention computed entirely in r-dim vein subspace
- VFTFeedForward: Factorized FFN in vein subspace
- NormalCompensator: Sparse corrections for off-manifold tokens
- VFTBlock: Complete transformer block with VFT/TVA

Complexity: O(B * H * T² * r) instead of O(B * H * T² * d)
where r << d (typically r=4-32, d=768-4096)
"""

import math
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from apt_model.modeling.blocks.vein import VeinProjector


# ============================================================================
# TVA Attention
# ============================================================================

class TVAAttention(nn.Module):
    """
    Tri-Vein Attention: Attention computed entirely in r-dimensional vein space.

    Flow:
      1. Project Q, K, V to vein space: Q_r = V^T Q, K_r = V^T K, V_r = V^T V
      2. Compute attention in r-dim: A = softmax(Q_r K_r^T / sqrt(r))
      3. Apply attention: Y_r = A V_r
      4. Reconstruct: Y = U Y_r

    Complexity:
      - Standard attention: O(B * H * T² * d)
      - TVA: O(B * H * T² * r) + projection overhead
      - Speedup when r << d (typically 10-100x fewer FLOPs for attention)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rank: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        vein_projector: Optional[VeinProjector] = None,
        use_separate_projector_per_head: bool = False,
    ):
        """
        Args:
            d_model: Model dimension (must be divisible by n_heads)
            n_heads: Number of attention heads
            rank: Vein subspace rank
            attn_dropout: Dropout for attention weights
            proj_dropout: Dropout for output projection
            vein_projector: Shared vein projector (created if None)
            use_separate_projector_per_head: Whether each head has own projector
        """
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.rank = rank
        self.use_separate_projector_per_head = use_separate_projector_per_head

        # QKV projections (standard)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Vein projector(s)
        if vein_projector is not None:
            self.vein = vein_projector
        else:
            # Create shared projector for d_head dimension
            self.vein = VeinProjector(
                d_model=self.d_head,
                rank=rank,
                implementation='linear',
                init_method='orthogonal',
            )

        # Dropout
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        # Scale factor for attention
        self.scale = 1.0 / math.sqrt(self.rank)

    def _shape_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape [B, T, D] -> [B, H, T, d_head]
        """
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape [B, H, T, d_head] -> [B, T, D]
        """
        B, H, T, d_head = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * d_head)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input [B, T, D]
            attn_mask: Attention mask [B, 1, T, T] or [B, H, T, T]
                       (0 for keep, -inf for mask)
            return_attention_weights: Whether to return attention weights

        Returns:
            Tuple of:
                - output: [B, T, D]
                - attn_weights: [B, H, T, T] if return_attention_weights else None
        """
        B, T, D = x.shape

        # Standard QKV projections
        q = self._shape_for_heads(self.q_proj(x))  # [B, H, T, d_head]
        k = self._shape_for_heads(self.k_proj(x))
        v = self._shape_for_heads(self.v_proj(x))

        # Project to vein subspace (per head)
        # Flatten heads for vein projection
        q_flat = q.reshape(B * self.n_heads, T, self.d_head)
        k_flat = k.reshape(B * self.n_heads, T, self.d_head)
        v_flat = v.reshape(B * self.n_heads, T, self.d_head)

        q_r = self.vein.project(q_flat)  # [B*H, T, r]
        k_r = self.vein.project(k_flat)
        v_r = self.vein.project(v_flat)

        # Compute attention in vein space
        attn_scores = torch.matmul(q_r, k_r.transpose(-2, -1)) * self.scale
        attn_scores = attn_scores.view(B, self.n_heads, T, T)

        # Apply mask
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to v_r
        attn_weights_flat = attn_weights.view(B * self.n_heads, T, T)
        y_r = torch.matmul(attn_weights_flat, v_r)  # [B*H, T, r]

        # Reconstruct from vein space
        y_flat = self.vein.reconstruct(y_r)  # [B*H, T, d_head]
        y = y_flat.view(B, self.n_heads, T, self.d_head)

        # Merge heads
        y = self._merge_heads(y)  # [B, T, D]

        # Output projection
        output = self.out_proj(y)
        output = self.proj_dropout(output)

        if return_attention_weights:
            return output, attn_weights
        else:
            return output, None


# ============================================================================
# VFT Feed-Forward
# ============================================================================

class VFTFeedForward(nn.Module):
    """
    VFT Feed-Forward: FFN computed in vein subspace.

    Flow:
      1. Project to vein: z = V h
      2. Two-layer FFN in r-dim: g = W2(act(W1(z)))
      3. Reconstruct: y = U g
      4. Add direct path for stability: output = y + stabilizer(h)

    Parameters:
      - Standard FFN: 2 * d * d_ff (typically d_ff = 4d)
      - VFT-FFN: 2 * r * r_hidden (typically r_hidden = 2r-4r)
      - Compression: ~(d/r)² reduction in parameters
    """

    def __init__(
        self,
        d_model: int,
        rank: int,
        r_hidden: Optional[int] = None,
        activation: str = 'silu',
        dropout: float = 0.0,
        vein_projector: Optional[VeinProjector] = None,
        use_stabilizer: bool = True,
    ):
        """
        Args:
            d_model: Model dimension
            rank: Vein subspace rank
            r_hidden: Hidden dimension in vein space (default: 4 * rank)
            activation: Activation function ('silu', 'gelu', 'relu')
            dropout: Dropout probability
            vein_projector: Shared vein projector
            use_stabilizer: Whether to add direct path stabilizer
        """
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.r_hidden = r_hidden if r_hidden is not None else max(4 * rank, rank + 1)

        # Vein projector
        if vein_projector is not None:
            self.vein = vein_projector
        else:
            self.vein = VeinProjector(
                d_model=d_model,
                rank=rank,
                implementation='linear',
                init_method='orthogonal',
            )

        # Two-layer FFN in vein space
        self.w1 = nn.Linear(rank, self.r_hidden, bias=True)
        self.w2 = nn.Linear(self.r_hidden, rank, bias=True)
        self.dropout = nn.Dropout(dropout)

        # Activation
        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Stabilizer (small direct residual)
        self.use_stabilizer = use_stabilizer
        if use_stabilizer:
            self.stabilizer = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model, bias=False),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input [B, T, D]

        Returns:
            output: [B, T, D]
        """
        # Project to vein
        z = self.vein.project(x)  # [B, T, r]

        # FFN in vein space
        h = self.w1(z)
        h = self.act(h)
        h = self.dropout(h)
        z_out = self.w2(h)  # [B, T, r]

        # Reconstruct
        y = self.vein.reconstruct(z_out)  # [B, T, D]

        # Add stabilizer
        if self.use_stabilizer:
            y = y + self.stabilizer(x)

        return y


# ============================================================================
# Normal Compensator
# ============================================================================

class NormalCompensator(nn.Module):
    """
    Sparse normal compensation for off-manifold tokens.

    For tokens with large reconstruction error (off-plane distance ε > τ),
    apply sparse rank-s corrections:
        Δy_i = Σ_{j=1..s} α_{ij} * u_j * (v_j^T h_i)

    where:
    - {u_j, v_j} are learnable basis vectors (shared across all tokens)
    - α_{ij} are token-specific weights from a learned gate
    - s is small (typically 1-3)
    - τ is the threshold for applying corrections
    """

    def __init__(
        self,
        d_model: int,
        s: int = 1,
        tau: float = 0.18,
        alpha_scale: float = 0.5,
    ):
        """
        Args:
            d_model: Model dimension
            s: Number of correction basis vectors
            tau: Threshold for off-plane distance
            alpha_scale: Scale factor for correction weights
        """
        super().__init__()
        self.d_model = d_model
        self.s = s
        self.tau = tau
        self.alpha_scale = alpha_scale

        if s > 0:
            # Learnable basis vectors
            self.U = nn.Parameter(torch.randn(s, d_model) * 0.02)
            self.V = nn.Parameter(torch.randn(s, d_model) * 0.02)

            # Gate network to compute α
            self.gate = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, s),
            )

    def forward(
        self,
        x: torch.Tensor,
        eps: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply normal compensation.

        Args:
            x: Input [B, T, D]
            eps: Off-plane distance [B, T]

        Returns:
            output: Corrected output [B, T, D]
        """
        if self.s == 0:
            return x

        B, T, D = x.shape

        # Mask: only apply to tokens with ε > τ
        mask = (eps > self.tau).float().unsqueeze(-1)  # [B, T, 1]

        # Compute correction weights: α = sigmoid(gate(h)) * alpha_scale
        alpha = torch.sigmoid(self.gate(x)) * self.alpha_scale  # [B, T, s]

        # Compute corrections: Σ_j α_j u_j (v_j^T h)
        vh = torch.einsum('btd,sd->bts', x, self.V)  # [B, T, s]
        correction = torch.einsum('bts,sd->btd', alpha * vh, self.U)  # [B, T, D]

        # Apply with mask
        output = x + correction * mask

        return output


# ============================================================================
# VFT Block
# ============================================================================

class VFTBlock(nn.Module):
    """
    Complete VFT/TVA Transformer block.

    Components:
      1. Pre-norm + TVA Attention
      2. Residual connection
      3. Pre-norm + VFT Feed-Forward
      4. Residual connection
      5. Normal Compensation (for off-manifold tokens)

    All computation happens in shared r-dimensional vein subspace.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        rank: int = 32,
        r_hidden: Optional[int] = None,
        s_normals: int = 1,
        tau: float = 0.18,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        activation: str = 'silu',
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            rank: Vein subspace rank
            r_hidden: Hidden dimension for VFT-FFN
            s_normals: Number of normal compensation basis vectors
            tau: Threshold for normal compensation
            attn_dropout: Dropout for attention
            ffn_dropout: Dropout for FFN
            activation: Activation function
        """
        super().__init__()
        self.d_model = d_model
        self.rank = rank

        # Shared vein projector for the entire block
        self.vein = VeinProjector(
            d_model=d_model,
            rank=rank,
            implementation='linear',
            init_method='orthogonal',
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # TVA Attention (uses d_head-dimensional projector internally)
        self.attn = TVAAttention(
            d_model=d_model,
            n_heads=n_heads,
            rank=rank,
            attn_dropout=attn_dropout,
            vein_projector=None,  # Creates own projector
        )

        # VFT Feed-Forward
        self.ffn = VFTFeedForward(
            d_model=d_model,
            rank=rank,
            r_hidden=r_hidden,
            activation=activation,
            dropout=ffn_dropout,
            vein_projector=self.vein,  # Share block's projector
        )

        # Normal Compensator
        self.normals = NormalCompensator(
            d_model=d_model,
            s=s_normals,
            tau=tau,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Forward pass.

        Args:
            x: Input [B, T, D]
            attn_mask: Attention mask
            return_metrics: Whether to return diagnostic metrics

        Returns:
            Tuple of:
                - output: [B, T, D]
                - metrics: Dict with diagnostic info (if return_metrics)
        """
        # Attention block
        h = self.norm1(x)
        attn_out, _ = self.attn(h, attn_mask=attn_mask)
        x = x + attn_out

        # Compute off-plane distance for normal compensation
        h2 = self.norm2(x)
        eps = self.vein.compute_reconstruction_error(h2)  # [B, T]

        # Feed-forward block
        ffn_out = self.ffn(h2)
        x = x + ffn_out

        # Normal compensation (only for off-manifold tokens)
        x = self.normals(x, eps)

        # Metrics
        metrics = None
        if return_metrics:
            metrics = {
                'eps_mean': float(eps.mean().item()),
                'eps_max': float(eps.max().item()),
                'eps_frac_over_tau': float((eps > self.normals.tau).float().mean().item()),
                'rank': self.rank,
            }

        return x, metrics


# ============================================================================
# Factory Functions
# ============================================================================

def create_vft_block(
    d_model: int,
    n_heads: int = 8,
    rank: int = 32,
    **kwargs
) -> VFTBlock:
    """
    Factory function to create VFT block with common configurations.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        rank: Vein subspace rank
        **kwargs: Additional arguments for VFTBlock

    Returns:
        VFTBlock instance
    """
    return VFTBlock(
        d_model=d_model,
        n_heads=n_heads,
        rank=rank,
        **kwargs
    )
