#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VFT-TVA Model - Vein-Flow Transformer with Tri-Vein Attention

Complete model using VFT blocks for efficient low-rank attention and FFN.
Supports text-only and multimodal (text + image + audio) inputs.

Key Features:
- Tri-Vein Attention (TVA): attention computed in low-rank vein subspace
- VFT FeedForward: factorized FFN in same vein space
- Normal Compensation: sparse corrections for off-manifold tokens
- Multimodal support: text + image + audio via projection layers
"""

from apt.apt_model.utils.fake_torch import get_torch
torch = get_torch()
from apt.apt_model.utils.fake_torch import get_torch
torch = get_torch()
nn = torch.nn
F = torch.nn.functional
import math
from typing import Optional, Tuple, Dict, Any, List


# ==================== VFT/TVA Core Components ====================

class VeinProjector(nn.Module):
    """
    Low-rank projector: provides project() / reconstruct().
    U: R^r -> R^d, V: R^d -> R^r with orthogonal init.
    """
    def __init__(self, d_model: int, rank: int):
        super().__init__()
        assert 1 <= rank < d_model, "rank must be in [1, d_model-1]"
        self.d_model = d_model
        self.rank = rank
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


class TVAAttention(nn.Module):
    """
    Tri-Vein Attention: compute attention entirely in r-dim vein space.
    Complexity: O(B * H * T^2 * r) instead of O(B * H * T^2 * d)
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rank: int,
        attn_dropout: float = 0.0,
        proj: Optional[VeinProjector] = None
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.rank = rank

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(attn_dropout)
        self.proj = proj if proj is not None else VeinProjector(d_model, rank)

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.d_head).permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, Dh = x.shape
        return x.permute(0, 2, 1, 3).reshape(B, T, H * Dh)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T, D = x.shape

        q = self._shape_heads(self.q(x))
        k = self._shape_heads(self.k(x))
        v = self._shape_heads(self.v(x))

        # Project to vein space
        qf = q.reshape(B * self.n_heads, T, self.d_head)
        kf = k.reshape(B * self.n_heads, T, self.d_head)
        vf = v.reshape(B * self.n_heads, T, self.d_head)

        q_r = self.proj.project(qf)
        k_r = self.proj.project(kf)
        v_r = self.proj.project(vf)

        # Attention in r-dim
        scale = 1.0 / math.sqrt(self.rank)
        attn_scores = torch.matmul(q_r, k_r.transpose(-2, -1)) * scale
        attn_scores = attn_scores.view(B, self.n_heads, T, T)

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)

        # Apply to v_r and reconstruct
        y_r = torch.matmul(attn.view(B * self.n_heads, T, T), v_r)
        y = self.proj.reconstruct(y_r).view(B, self.n_heads, T, self.d_head)
        y = self._merge_heads(y)
        return self.o(y)


class VFTFeedForward(nn.Module):
    """
    FFN in vein subspace:
    z = V h -> g = act(W1 z) -> y = U(W2 g)
    """
    def __init__(
        self,
        d_model: int,
        rank: int,
        r_hidden: Optional[int] = None,
        act: str = "silu",
        drop: float = 0.0,
        proj: Optional[VeinProjector] = None
    ):
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

        self.stab = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj.project(x)
        g = self.act(self.w1(z))
        g = self.drop(g)
        z2 = self.w2(g)
        y = self.proj.reconstruct(z2)
        return y + self.stab(x)


class NormalCompensator(nn.Module):
    """
    Sparse normal compensation for tokens with off-plane ε > τ
    """
    def __init__(
        self,
        d_model: int,
        s: int = 1,
        tau: float = 0.18,
        alpha_scale: float = 0.5
    ):
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
        if self.s == 0:
            return x

        B, T, D = x.shape
        mask = (eps > self.tau).float().unsqueeze(-1)

        alpha = torch.sigmoid(self.gate(x)) * self.alpha_scale
        vh = torch.einsum("btd,sd->bts", x, self.V)
        inc = torch.einsum("bts,sd->btd", alpha * vh, self.U)
        return x + inc * mask


def _off_plane_eps(h: torch.Tensor, proj: VeinProjector) -> torch.Tensor:
    """Compute off-plane magnitude ε = ||h - U(Vh)||_2"""
    z = proj.project(h)
    h_hat = proj.reconstruct(z)
    return torch.norm(h - h_hat, dim=-1)


class VFTBlock(nn.Module):
    """
    One VFT Transformer block with TVA attention and VFT FFN
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        rank: int = 32,
        s_normals: int = 1,
        tau: float = 0.18,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0
    ):
        super().__init__()
        self.proj = VeinProjector(d_model, rank)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = TVAAttention(d_model, n_heads, rank, attn_dropout, proj=self.proj)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = VFTFeedForward(d_model, rank, drop=ffn_dropout, proj=self.proj)
        self.normals = NormalCompensator(d_model, s=s_normals, tau=tau)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # TVA attention
        h = self.norm1(x)
        y = self.attn(h, attn_mask=attn_mask)
        x = x + y

        # FFN in vein space
        eps = _off_plane_eps(self.norm2(x), self.proj)
        h2 = self.norm2(x)
        y2 = self.ffn(h2)
        x = x + y2

        # Normal compensation
        x = self.normals(x, eps)

        info = {
            "eps_mean": float(eps.mean().item()),
            "eps_frac_over_tau": float((eps > self.normals.tau).float().mean().item()),
            "rank": self.proj.rank
        }
        return x, info


# ==================== Complete VFT-TVA Model ====================

class VFTTVAModel(nn.Module):
    """
    Complete VFT-TVA model with optional multimodal support

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        n_layers: Number of VFT blocks
        n_heads: Number of attention heads
        rank: Vein subspace rank
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
        enable_multimodal: Enable image and audio inputs
        image_dim: Image feature dimension
        audio_dim: Audio feature dimension
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        rank: int = 32,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        s_normals: int = 1,
        tau: float = 0.18,
        enable_multimodal: bool = False,
        image_dim: int = 1024,
        audio_dim: int = 512
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.enable_multimodal = enable_multimodal

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        # Multimodal projections
        if enable_multimodal:
            self.image_proj = nn.Linear(image_dim, d_model)
            self.audio_proj = nn.Linear(audio_dim, d_model)
            nn.init.normal_(self.image_proj.weight, std=0.02)
            nn.init.normal_(self.audio_proj.weight, std=0.02)

        self.dropout = nn.Dropout(dropout)

        # VFT blocks
        self.blocks = nn.ModuleList([
            VFTBlock(
                d_model=d_model,
                n_heads=n_heads,
                rank=rank,
                s_normals=s_normals,
                tau=tau,
                attn_dropout=dropout,
                ffn_dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.token_embedding.weight = self.lm_head.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        image_feat: Optional[torch.Tensor] = None,
        audio_feat: Optional[torch.Tensor] = None,
        return_info: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, List[Dict]]:
        """
        Forward pass with optional multimodal inputs

        Args:
            input_ids: [B, T] text token IDs (optional)
            image_feat: [B, D_img] image features (optional)
            audio_feat: [B, D_aud] audio features (optional)
            return_info: Return diagnostic info from each block

        Returns:
            logits [B, T, vocab_size] or (logits, block_info_list)
        """
        # Build embeddings
        embeddings_list = []

        if input_ids is not None:
            batch_size, seq_len = input_ids.shape
            text_emb = self.token_embedding(input_ids)
            text_emb = text_emb + self.pos_embedding[:, :seq_len, :]
            embeddings_list.append(text_emb)

        if self.enable_multimodal and image_feat is not None:
            img_emb = self.image_proj(image_feat)
            if img_emb.dim() == 2:
                img_emb = img_emb.unsqueeze(1)
            embeddings_list.append(img_emb)

        if self.enable_multimodal and audio_feat is not None:
            aud_emb = self.audio_proj(audio_feat)
            if aud_emb.dim() == 2:
                aud_emb = aud_emb.unsqueeze(1)
            embeddings_list.append(aud_emb)

        if not embeddings_list:
            raise ValueError("At least one modality must be provided")

        # Concatenate modalities
        if len(embeddings_list) == 1:
            x = embeddings_list[0]
        else:
            x = torch.cat(embeddings_list, dim=1)

        batch_size, seq_len = x.size(0), x.size(1)
        x = self.dropout(x)

        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(x.device)
        mask = mask.masked_fill(mask == True, float('-inf'))
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]

        # VFT blocks
        block_infos = []
        for block in self.blocks:
            x, info = block(x, attn_mask=mask)
            if return_info:
                block_infos.append(info)

        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if return_info:
            return logits, block_infos
        return logits

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        image_feat: Optional[torch.Tensor] = None,
        audio_feat: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95
    ) -> torch.Tensor:
        """
        Autoregressive generation

        Args:
            input_ids: [B, L] initial tokens
            image_feat: [B, D_img] image features
            audio_feat: [B, D_aud] audio features
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling

        Returns:
            generated_ids: [B, L + max_new_tokens]
        """
        if input_ids is None:
            batch_size = 1
            device = next(self.parameters()).device
            input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        else:
            device = input_ids.device

        generated = input_ids

        for _ in range(max_new_tokens):
            # Get logits (multimodal features only used once)
            logits = self.forward(
                input_ids=generated,
                image_feat=image_feat if _ == 0 else None,
                audio_feat=audio_feat if _ == 0 else None
            )

            # Sample next token
            next_token_logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            if (next_token == 2).all():  # Assuming 2 is EOS token
                break

        return generated


# ==================== Factory Functions ====================

def create_vft_tva_model(
    model_size: str = 'base',
    enable_multimodal: bool = False,
    **kwargs
) -> VFTTVAModel:
    """
    Create VFT-TVA model with predefined sizes

    Args:
        model_size: 'small', 'base', 'large'
        enable_multimodal: Enable multimodal support
        **kwargs: Override default parameters

    Returns:
        VFTTVAModel instance
    """
    configs = {
        'small': {
            'vocab_size': 32000,
            'd_model': 384,
            'n_layers': 6,
            'n_heads': 6,
            'rank': 24
        },
        'base': {
            'vocab_size': 50000,
            'd_model': 512,
            'n_layers': 8,
            'n_heads': 8,
            'rank': 32
        },
        'large': {
            'vocab_size': 50000,
            'd_model': 768,
            'n_layers': 12,
            'n_heads': 12,
            'rank': 48
        }
    }

    config = configs.get(model_size, configs['base'])
    config.update(kwargs)
    config['enable_multimodal'] = enable_multimodal

    return VFTTVAModel(**config)
