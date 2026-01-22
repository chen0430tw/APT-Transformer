#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VFT/TVA: Vein-Flow Transformer / Tri-Vein Attention

Core implementation based on original vft_tva.py:
- TVAAttention: attention computed wholly in r-dim vein subspace
- VFTFeedForward: factorized FFN in the same subspace
- NormalCompensator: sparse normal corrections for off-manifold tokens
- VFTBlock: TVA + VFT-FFN + normal compensation + unified tau gating

Complexity: O(B * H * T² * r) instead of O(B * H * T² * d)
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import math
from apt.apt_model.utils.fake_torch import get_torch
torch = get_torch()
from apt.apt_model.utils.fake_torch import get_torch
torch = get_torch()
nn = torch.nn
F = torch.nn.functional

from apt.apt_model.modeling.blocks.vein import VeinProjector


# --------------------------- utilities ---------------------------

def _stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(x, dim=dim)

def _off_plane_eps(h: torch.Tensor, proj: VeinProjector) -> torch.Tensor:
    """离面率 ε = ||h - U(Vh)||_2 per token [B,T]."""
    z = proj.project(h)
    h_hat = proj.reconstruct(z)
    return torch.norm(h - h_hat, dim=-1)


# ------------------------- TVA attention -------------------------

class TVAAttention(nn.Module):
    """
    Tri-Vein Attention: compute attention entirely in r-dim vein space.

    Flow:
      Q,K,V -> project to r:  \tilde Q=V^T Q, \tilde K=V^T K, \tilde V=U^T V
      A = softmax( ( \tilde Q \tilde K^T ) / sqrt(r) )      [B,H,T,T]
      Y_base = U( A \tilde V )

    Complexity: O(B * H * T^2 * r) + two proj/reconstruct passes.
    """
    def __init__(self, d_model: int, n_heads: int, rank: int, attn_dropout: float = 0.0, proj: Optional[VeinProjector] = None):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.rank = rank

        # projections for Q,K,V
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(attn_dropout)
        self.proj = proj if proj is not None else VeinProjector(d_model, rank)

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B,T,D] -> [B,H,T,d_head]
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # [B,H,T,d_head] -> [B,T,D]
        B, H, T, Dh = x.shape
        return x.permute(0, 2, 1, 3).reshape(B, T, H * Dh)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B,T,D], attn_mask (optional): [B,1,T,T] additive mask (0 keep, -inf mask)
        """
        B, T, D = x.shape

        # standard input projections
        q = self._shape_heads(self.q(x))  # [B,H,T,d]
        k = self._shape_heads(self.k(x))
        v = self._shape_heads(self.v(x))

        # vein-space projections per head: (share same projector for simplicity)
        # flatten heads for projection, then restore
        qf = q.reshape(B * self.n_heads, T, self.d_head)
        kf = k.reshape(B * self.n_heads, T, self.d_head)
        vf = v.reshape(B * self.n_heads, T, self.d_head)

        q_r = self.proj.project(qf)   # [B*H,T,r]
        k_r = self.proj.project(kf)
        v_r = self.proj.project(vf)   # NOTE: here we project v with V, then later lift with U

        # attention in r-dim
        scale = 1.0 / math.sqrt(self.rank)
        attn_scores = torch.matmul(q_r, k_r.transpose(-2, -1)) * scale  # [B*H,T,T]
        attn_scores = attn_scores.view(B, self.n_heads, T, T)

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        attn = _stable_softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        # apply to v_r
        y_r = torch.matmul(attn.view(B * self.n_heads, T, T), v_r)       # [B*H,T,r]
        # lift back
        y = self.proj.reconstruct(y_r).view(B, self.n_heads, T, self.d_head)
        y = self._merge_heads(y)                                         # [B,T,D]
        return self.o(y)


# --------------------- factorized VFT-FFN ------------------------

class VFTFeedForward(nn.Module):
    """
    FFN in the same shared vein subspace:
      z = V h
      g = act( W1_r z + b1 )   (work in r or 2r)
      y = U ( W2_r g )         (lift back)
    Optionally add a small direct path for stability.

    By default we choose hidden size r_hidden = 2r for some nonlinearity capacity.
    """
    def __init__(self, d_model: int, rank: int, r_hidden: Optional[int] = None, act: str = "silu", drop: float = 0.0, proj: Optional[VeinProjector] = None):
        super().__init__()
        self.rank = rank
        self.r_hidden = r_hidden if r_hidden is not None else max(rank * 2, rank + 1)
        self.proj = proj if proj is not None else VeinProjector(d_model, rank)

        self.w1 = nn.Linear(rank, self.r_hidden, bias=True)
        self.w2 = nn.Linear(self.r_hidden, rank, bias=True)
        self.drop = nn.Dropout(drop)

        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.SiLU()

        # small direct residual stabilizer
        self.stab = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj.project(x)             # [B,T,r]
        g = self.act(self.w1(z))
        g = self.drop(g)
        z2 = self.w2(g)                      # [B,T,r]
        y = self.proj.reconstruct(z2)        # [B,T,D]
        return y + self.stab(x)


# --------------------- normal compensation -----------------------

class NormalCompensator(nn.Module):
    """
    Sparse normal compensation (law-of-small-s corrections):
      For tokens with off-plane ε > τ, add at most s outer-product style increments:
        Δy_i = Σ_{j=1..s} α_{ij} * u_j * (v_j^T h_i)

    Implementation:
      - global learnable {u_j, v_j}, shared across tokens (small s)
      - token-wise α via a tiny gate from h (or from ε)
      - apply only to masked positions
    """
    def __init__(self, d_model: int, s: int = 1, tau: float = 0.18, alpha_scale: float = 0.5):
        super().__init__()
        assert s >= 0
        self.s = s
        self.tau = float(tau)
        self.alpha_scale = float(alpha_scale)

        if s > 0:
            self.U = nn.Parameter(torch.randn(s, d_model) * 0.02)
            self.V = nn.Parameter(torch.randn(s, d_model) * 0.02)
            self.gate = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, s),
            )

    def forward(self, x: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """
        x:   [B,T,D]
        eps: [B,T] off-plane magnitude
        """
        if self.s == 0:
            return x

        B, T, D = x.shape
        mask = (eps > self.tau).float().unsqueeze(-1)  # [B,T,1]

        # α = sigmoid(gate(h)) * alpha_scale
        alpha = torch.sigmoid(self.gate(x)) * self.alpha_scale   # [B,T,s]
        # compute Σ_j α_j u_j (v_j^T h)
        # first compute (v_j^T h) for all j: [B,T,s]
        vh = torch.einsum("btd,sd->bts", x, self.V)              # dot with each v_j
        inc = torch.einsum("bts,sd->btd", alpha * vh, self.U)    # weight u_j and sum
        return x + inc * mask


# -------------------------- VFT block ----------------------------

class VFTBlock(nn.Module):
    """
    One Transformer block built with:
      - TVA attention in vein space
      - VFT feed-forward in the same vein space
      - unified tau gating: apply NormalCompensator only when ε>τ
    """
    def __init__(self,
                 d_model: int,
                 n_heads: int = 8,
                 rank: int = 32,
                 s_normals: int = 1,
                 tau: float = 0.18,
                 attn_dropout: float = 0.0,
                 ffn_dropout: float = 0.0,
                 ):
        super().__init__()
        self.proj = VeinProjector(d_model, rank)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = TVAAttention(d_model, n_heads, rank, attn_dropout, proj=self.proj)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = VFTFeedForward(d_model, rank, drop=ffn_dropout, proj=self.proj)
        self.normals = NormalCompensator(d_model, s=s_normals, tau=tau)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # pre-norm -> TVA
        h = self.norm1(x)
        y = self.attn(h, attn_mask=attn_mask)
        x = x + y

        # compute off-plane ε for gating normals
        eps = _off_plane_eps(self.norm2(x), self.proj)  # [B,T]
        # FFN in vein space + residual
        h2 = self.norm2(x)
        y2 = self.ffn(h2)
        x = x + y2
        # normals only for off-manifold tokens
        x = self.normals(x, eps)

        info = {
            "eps_mean": float(eps.mean().item()),
            "eps_frac_over_tau": float((eps > self.normals.tau).float().mean().item()),
            "rank": self.proj.V.out_features
        }
        return x, info


# ----------------------- factory function ------------------------

def create_vft_block(d_model: int, n_heads: int = 8, rank: int = 32, **kwargs) -> VFTBlock:
    """
    Factory function to create VFT block with common defaults.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        rank: Vein subspace rank
        **kwargs: Additional arguments for VFTBlock

    Returns:
        VFTBlock instance
    """
    return VFTBlock(d_model=d_model, n_heads=n_heads, rank=rank, **kwargs)
