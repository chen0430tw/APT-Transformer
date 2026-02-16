# -*- coding: utf-8 -*-
"""
gpt5_model.py  —  All-in-one (CPU-friendly)
- Codebook MoE (top-k + shared expert)
- Leaf-Vote (K=2)
- Streaming retrieval stub + memory buckets
- Bi-state precision align (vein subspace)
- Composite feedback (entropy, ΔKL)

Refactored to use unified VFT/TVA modules from apt.model.layers.blocks
"""

from __future__ import annotations
from typing import Any, Dict, Tuple, List, Optional
import math

# Try to import the fake_torch from the external apt package.  If it is
# unavailable (e.g. in environments where the apt package is not installed),
# fall back to the real torch.  The get_torch() function returns either the
# fake torch or the actual torch module as appropriate.
try:
    from apt.core.fake_torch import get_torch  # type: ignore
except Exception:
    import torch as _torch  # type: ignore
    def get_torch() -> Any:
        return _torch

# Initialise torch once.  The duplicate call below was redundant and
# could lead to confusion; we remove it.
torch = get_torch()
nn = torch.nn
F = torch.nn.functional

# Use refactored VeinProjector from blocks module
# Try to import the VeinProjector from the external apt package.  If
# that fails (for example, in environments where the apt package is
# unavailable), fall back to a simple linear projector implementation
# that exposes the same interface (`project` and `reconstruct` methods).
try:
    from apt.model.layers.blocks import VeinProjector  # type: ignore
except Exception:
    class VeinProjector(nn.Module):  # type: ignore
        """Fallback vein projector: projects to a low-rank space via linear layers.

        This simple implementation mimics the interface of the external
        VeinProjector by providing a `rank` attribute as well as
        `project()` and `reconstruct()` methods.  It does not depend on
        the apt package and can be used as a drop-in replacement when
        running on systems without that dependency.
        """
        def __init__(self, d_model: int, rank: int, implementation: str = 'linear', init_method: str = 'orthogonal'):
            super().__init__()
            self.rank = int(rank)
            # Simple linear projection down and up.  Using bias=False
            # ensures a clean projection; orthogonal initialisation
            # encourages stability in the subspace.
            self.proj = nn.Linear(d_model, self.rank, bias=False)
            self.rec = nn.Linear(self.rank, d_model, bias=False)
            # Initialise the projection matrix orthogonally
            nn.init.orthogonal_(self.proj.weight)
            # Initialise the reconstruction matrix to approximate the pseudoinverse
            # by using the transpose of the projection.  This preserves information
            # rather than zeroing out the reconstruction weights (which caused
            # outputs to collapse to near‑zero).
            with torch.no_grad():
                if self.proj.weight.size(0) == 0 or self.proj.weight.size(1) == 0:
                    nn.init.orthogonal_(self.rec.weight)
                else:
                    self.rec.weight.copy_(self.proj.weight.data.t())

        def project(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

        def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
            return self.rec(z)


# ========================= utils =========================
def token_entropy_scalar(logits: torch.Tensor) -> torch.Tensor:
    """Mean token entropy over [B,T,V] logits."""
    p = F.softmax(logits, dim=-1)
    return -(p * (p.clamp_min(1e-9).log())).sum(-1).mean()


# ========================= moe ===========================
class CodebookRouter(nn.Module):
    """Linear -> logits gates (softmax is applied inside MoELayer)."""
    def __init__(self, d_model: int, d_route: int, num_skills: int, temperature: float = 1.0):
        super().__init__()
        self.proj = nn.Linear(d_model, num_skills)
        self.temperature = float(temperature)

    def route(self, h: torch.Tensor) -> torch.Tensor:
        # Return logits; MoELayer applies softmax once.
        return self.proj(h) / max(1e-6, self.temperature)  # [B,T,E]


class MiniExpert(nn.Module):
    def __init__(self, d_in: int, d_h: int, d_out: int):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(d_in, d_h), nn.SiLU(), nn.Linear(d_h, d_out))

    def forward(self, x):  # [B,T,D] -> [B,T,D]
        return self.ff(x)


class SharedExpert(MiniExpert):
    def __init__(self, d_in, d_h, d_out, scale: float = 0.25):
        super().__init__(d_in, d_h, d_out)
        self.scale = float(scale)

    def forward(self, x):
        return super().forward(x) * self.scale


class MoELayer(nn.Module):
    """Top-k MoE with **vectorized token dispatch** (GPU-friendly).

    - No Python branching on tensors (no implicit sync).
    - Computes only K experts per token via gathered expert weights.
    - Designed to be compatible with `torch.compile` (optional).
    """
    def __init__(self, d_model: int, num_skills: int, d_hidden: Optional[int] = None, init_method: str = "xavier"):
        super().__init__()
        self.d_model = int(d_model)
        self.num_skills = int(num_skills)
        self.d_hidden = int(d_hidden) if d_hidden is not None else int(4 * d_model)

        # Expert weights packed as tensors:
        # W1: [E, D, H], b1: [E, H], W2: [E, H, D], b2: [E, D]
        self.W1 = nn.Parameter(torch.empty(self.num_skills, self.d_model, self.d_hidden))
        self.b1 = nn.Parameter(torch.zeros(self.num_skills, self.d_hidden))
        self.W2 = nn.Parameter(torch.empty(self.num_skills, self.d_hidden, self.d_model))
        self.b2 = nn.Parameter(torch.zeros(self.num_skills, self.d_model))

        # Shared expert path (always applied)
        self.shared = nn.Sequential(
            nn.Linear(self.d_model, self.d_hidden),
            nn.GELU(),
            nn.Linear(self.d_hidden, self.d_model),
        )

        if init_method == "xavier":
            nn.init.xavier_uniform_(self.W1)
            nn.init.xavier_uniform_(self.W2)

        self._compiled = None  # lazy torch.compile cache

    def _forward_core(self, h: torch.Tensor, router: "CodebookRouter", top_k: int = 2) -> Tuple[torch.Tensor, Dict[str, Any]]:
        gate = router.route(h)  # [B,T,E]
        k = min(int(top_k), self.num_skills)

        # Top-k routing probabilities
        probs = torch.softmax(gate, dim=-1)                  # [B,T,E]
        topv, topi = torch.topk(probs, k=k, dim=-1)          # [B,T,K], [B,T,K]

        # Gather expert parameters per token per k
        # Shapes:
        #   topi -> [B,T,K]
        #   W1_g -> [B,T,K,D,H]
        W1_g = self.W1[topi]                                 # advanced indexing
        b1_g = self.b1[topi]                                 # [B,T,K,H]
        W2_g = self.W2[topi]                                 # [B,T,K,H,D]
        b2_g = self.b2[topi]                                 # [B,T,K,D]

        # Compute expert outputs:
        # h: [B,T,D] -> [B,T,1,D,1] for matmul with [B,T,K,D,H]
        pre = torch.einsum('btd,btkdh->btkh', h, W1_g)  # [B,T,K,H]
        pre = pre + b1_g
        act = torch.nn.functional.gelu(pre)
        out = torch.einsum('btkh,btkhd->btkd', act, W2_g) + b2_g  # [B,T,K,D]

        # Weighted sum of top-k experts
        mix = (out * topv.unsqueeze(-1)).sum(dim=2)           # [B,T,D]

        # Add shared path (dense)
        shared = self.shared(h)

        y = mix + shared

        aux = {"gate_mean": gate.mean()}  # tensor, no host sync
        return y, aux

    def forward(self, h: torch.Tensor, router: "CodebookRouter", top_k: int = 2, compile_dispatch: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Best-effort compile: if inductor toolchain isn't available (common on minimal systems),
        # fall back to eager without breaking training.
        if compile_dispatch and (self._compiled is None):
            try:
                import torch._dynamo
                torch._dynamo.config.suppress_errors = True
                self._compiled = torch.compile(self._forward_core, backend="aot_eager", dynamic=True)
            except Exception:
                self._compiled = self._forward_core
        fn = self._compiled if (compile_dispatch and self._compiled is not None) else self._forward_core
        try:
            return fn(h, router, top_k)
        except Exception:
            # runtime backend failure -> eager fallback
            self._compiled = self._forward_core
            return self._forward_core(h, router, top_k)



class ToyEmbed(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 4096,
                 use_rope_embed: bool = False):
        """Simple text embedding with optional RoPE on the embedding itself.

        Args:
            vocab_size: size of the vocabulary.
            d_model: hidden size of the embedding.
            max_len: maximum sequence length for positional embeddings.
            use_rope_embed: whether to apply rotary positional embeddings to the token+position embedding.
        """
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        # Store flag for applying RoPE on the embedding.  The suffix
        # ``_embed`` distinguishes this from attention‑level RoPE.
        self.use_rope_embed = bool(use_rope_embed)

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary positional embeddings to [B,T,D] tensor, trimming any padded dimension."""
        B, T, orig_D = x.size()
        # Pad one dimension if the hidden size is odd
        if orig_D % 2 != 0:
            pad = x.new_zeros(B, T, 1)
            x = torch.cat([x, pad], dim=-1)
            D = orig_D + 1
        else:
            D = orig_D
        half = D // 2
        pos = torch.arange(T, device=x.device, dtype=x.dtype)
        freqs = torch.arange(0, half, device=x.device, dtype=x.dtype)
        inv_freq = 1.0 / (10000 ** (freqs / float(half)))
        sinusoid = pos.unsqueeze(1) * inv_freq.unsqueeze(0)
        sin = sinusoid.sin().unsqueeze(0)
        cos = sinusoid.cos().unsqueeze(0)
        x1 = x[..., :half]
        x2 = x[..., half:half*2]
        x_rot_first = x1 * cos - x2 * sin
        x_rot_second = x1 * sin + x2 * cos
        x_rot = torch.cat([x_rot_first, x_rot_second], dim=-1)
        # Trim the padded dimension if we added one
        if D != orig_D:
            x_rot = x_rot[..., :orig_D]
        return x_rot

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        B, T = x_ids.shape
        pos_ids = torch.arange(T, device=x_ids.device).unsqueeze(0).expand(B, T)
        emb = self.tok(x_ids) + self.pos(pos_ids)
        # Apply RoPE at the embedding level only if explicitly requested.
        if self.use_rope_embed:
            emb = self._apply_rope(emb)
        return emb


class MultimodalEmbed(nn.Module):
    """Multimodal embedding layer for GPT-5 (text + image + audio)."""
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 4096,
                 image_dim: int = 1024, audio_dim: int = 512,
                 use_rope_embed: bool = False):
        """Embed text, image and audio into a common latent space.

        Each modality is projected into the same dimensionality ``d_model``.
        Rotary positional embeddings may be optionally applied to the combined
        token and position embeddings of the text modality via ``use_rope_embed``.

        Args:
            vocab_size: size of the token vocabulary.
            d_model: hidden dimension for all modalities.
            max_len: maximum sequence length for position embeddings.
            image_dim: dimensionality of image features.
            audio_dim: dimensionality of audio features.
            use_rope_embed: whether to apply RoPE to the token+position embedding (text only).
        """
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.image_proj = nn.Linear(image_dim, d_model)
        self.audio_proj = nn.Linear(audio_dim, d_model)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        nn.init.normal_(self.image_proj.weight, std=0.02)
        nn.init.normal_(self.audio_proj.weight, std=0.02)
        # Flag to determine whether to apply RoPE on the text embedding.
        self.use_rope_embed = bool(use_rope_embed)

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        B, T, orig_D = x.size()
        # Pad when hidden dimension is odd
        if orig_D % 2 != 0:
            pad = x.new_zeros(B, T, 1)
            x = torch.cat([x, pad], dim=-1)
            D = orig_D + 1
        else:
            D = orig_D
        half = D // 2
        pos = torch.arange(T, device=x.device, dtype=x.dtype)
        freqs = torch.arange(0, half, device=x.device, dtype=x.dtype)
        inv_freq = 1.0 / (10000 ** (freqs / float(half)))
        sinusoid = pos.unsqueeze(1) * inv_freq.unsqueeze(0)
        sin = sinusoid.sin().unsqueeze(0)
        cos = sinusoid.cos().unsqueeze(0)
        x1 = x[..., :half]
        x2 = x[..., half:half*2]
        x_rot_first = x1 * cos - x2 * sin
        x_rot_second = x1 * sin + x2 * cos
        x_rot = torch.cat([x_rot_first, x_rot_second], dim=-1)
        # Trim any padded dimension
        if D != orig_D:
            x_rot = x_rot[..., :orig_D]
        return x_rot

    def forward(self, x_ids: Optional[torch.Tensor] = None,
                image_feat: Optional[torch.Tensor] = None,
                audio_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        parts: List[torch.Tensor] = []

        if x_ids is not None:
            B, T = x_ids.shape
            pos_ids = torch.arange(T, device=x_ids.device).unsqueeze(0).expand(B, T)
            txt = self.tok(x_ids) + self.pos(pos_ids)
            # Apply RoPE at the embedding level only if requested.  Note
            # that attention‑level RoPE is controlled separately via ``GPT5Attention``.
            if self.use_rope_embed:
                txt = self._apply_rope(txt)
            parts.append(txt)

        if image_feat is not None:
            img_emb = self.image_proj(image_feat)
            # Ensure the sequence dimension is present on image features.  If a
            # single image representation is provided per example, unsqueeze
            # dimension 1 to make it a length‑1 sequence.
            if img_emb.dim() == 2:
                img_emb = img_emb.unsqueeze(1)
            parts.append(img_emb)

        if audio_feat is not None:
            aud_emb = self.audio_proj(audio_feat)
            if aud_emb.dim() == 2:
                aud_emb = aud_emb.unsqueeze(1)
            parts.append(aud_emb)

        if not parts:
            raise ValueError("At least one modality (text/image/audio) must be provided")

        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=1)


# ======================= attention =======================
class GPT5Attention(nn.Module):
    """Multi‑head self‑attention supporting RoPE, causal and sliding window masks, and grouped‑query heads.

    This module implements a standard scaled dot‑product attention over a sequence with a
    variable number of query heads (``n_heads``).  Optional rotary positional embeddings
    (RoPE) may be applied to the query and key projections to encode relative positions.
    A sliding window attention can be enabled via ``window_size`` to restrict each token
    to attend only to a local neighbourhood in addition to the usual causal mask.  If
    ``num_kv_heads`` is provided and is smaller than ``n_heads``, grouped‑query attention
    (GQA) is performed by sharing key/value heads across groups of query heads.

    Args:
        d_model: The dimensionality of the input and output representations.
        n_heads: The number of query heads to use.  Must divide ``d_model``.
        window_size: If >0, each query may only attend to the previous ``window_size`` tokens.
        num_kv_heads: If specified and < ``n_heads``, the number of distinct key/value heads
            for grouped‑query attention.  ``n_heads`` must be divisible by ``num_kv_heads``.
        use_rope: Whether to apply rotary positional embeddings to the query and key.
    """

    def __init__(self, d_model: int, n_heads: int = 8,
                 window_size: int = 0, num_kv_heads: Optional[int] = None,
                 use_rope_attn: bool = False, compile_moe_dispatch: bool = False):
        # NOTE: compile_moe_dispatch is unused in attention (kept for API symmetry).

        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.head_dim = self.d_model // self.n_heads
        self.window_size = int(window_size)
        # If num_kv_heads not provided, default to n_heads (no GQA)
        self.num_kv_heads = int(num_kv_heads) if num_kv_heads is not None else self.n_heads
        assert self.n_heads % self.num_kv_heads == 0, "n_heads must be divisible by num_kv_heads"
        # Whether to apply rotary positional embeddings at the attention level.
        self.use_rope_attn = bool(use_rope_attn)

        # Projection matrices
        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)
        self.W_o = nn.Linear(self.d_model, self.d_model)

        # Initialise weights similar to GPT‑style initialisation
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        if self.W_q.bias is not None:
            nn.init.zeros_(self.W_q.bias)
        if self.W_k.bias is not None:
            nn.init.zeros_(self.W_k.bias)
        if self.W_v.bias is not None:
            nn.init.zeros_(self.W_v.bias)
        if self.W_o.bias is not None:
            nn.init.zeros_(self.W_o.bias)

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary positional embeddings to a 4‑D tensor [B,H,T,D], trimming any padding."""
        B, H, T, D = x.size()
        orig_D = D
        # Pad one dimension if head_dim is odd
        if orig_D % 2 != 0:
            pad = x.new_zeros(B, H, T, 1)
            x = torch.cat([x, pad], dim=-1)
            D = orig_D + 1
        else:
            D = orig_D
        half = D // 2
        pos = torch.arange(T, device=x.device, dtype=x.dtype)
        freqs = torch.arange(0, half, device=x.device, dtype=x.dtype)
        inv_freq = 1.0 / (10000 ** (freqs / float(half)))
        sinusoid = pos.unsqueeze(1) * inv_freq.unsqueeze(0)
        sin = sinusoid.sin().unsqueeze(0).unsqueeze(0)       # [1,1,T,half]
        cos = sinusoid.cos().unsqueeze(0).unsqueeze(0)       # [1,1,T,half]
        x1 = x[..., :half]
        x2 = x[..., half:half*2]
        x_rot_first = x1 * cos - x2 * sin
        x_rot_second = x1 * sin + x2 * cos
        x_rot = torch.cat([x_rot_first, x_rot_second], dim=-1)
        # Trim padding if any
        if D != orig_D:
            x_rot = x_rot[..., :orig_D]
        return x_rot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute multi‑head attention on the input sequence.

        Args:
            x: Tensor of shape [B, T, d_model].
        Returns:
            Tensor of shape [B, T, d_model] containing the attended features.
        """
        B, T, D = x.size()
        H = self.n_heads
        hd = self.head_dim
        # Project to query, key and value, reshape to [B,H,T,hd]
        q = self.W_q(x).view(B, T, H, hd).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, hd).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, hd).transpose(1, 2)
        # Apply RoPE on q/k if enabled.  At this level, RoPE encodes
        # relative positions for the attention heads.  Embedding‑level
        # RoPE is controlled separately via the embedding layers.
        if self.use_rope_attn:
            q = self._apply_rope(q)
            k = self._apply_rope(k)
        # Grouped‑query attention: share KV heads across groups
        if self.num_kv_heads < H:
            group_size = H // self.num_kv_heads
            # Reshape to [B, num_kv_heads, group_size, T, hd]
            k_reshaped = k.reshape(B, self.num_kv_heads, group_size, T, hd)
            v_reshaped = v.reshape(B, self.num_kv_heads, group_size, T, hd)
            # Take the first head in each group as the shared KV head
            k_small = k_reshaped[:, :, 0, :, :]
            v_small = v_reshaped[:, :, 0, :, :]
            # Broadcast back to full head count
            k = k_small.unsqueeze(2).repeat(1, 1, group_size, 1, 1).reshape(B, H, T, hd)
            v = v_small.unsqueeze(2).repeat(1, 1, group_size, 1, 1).reshape(B, H, T, hd)
        # Compute scaled dot‑product attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(max(hd, 1))  # [B,H,T,T]
        # Compose causal and sliding window masks
        # Causal mask: disallow attending to future positions
        causal_mask = torch.triu(torch.ones((T, T), dtype=torch.bool, device=att.device), diagonal=1)
        mask = causal_mask
        # Sliding window mask: disallow attention beyond window_size
        if self.window_size and self.window_size > 0:
            i = torch.arange(T, device=att.device).view(T, 1)
            j = torch.arange(T, device=att.device).view(1, T)
            local = j < (i - (self.window_size - 1))
            mask = local if mask is None else (mask | local)
        if mask is not None:
            att = att.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        # Softmax over last dimension
        att = F.softmax(att, dim=-1)
        # Compute attention output
        y = att @ v  # [B,H,T,hd]
        # Concatenate heads and project back to d_model
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        out = self.W_o(y)
        return out


# ======================= GPT-5 block ======================
class GPT5Block(nn.Module):
    """A single GPT‑5 block comprising self‑attention followed by a mixture‑of‑experts (MoE) and feed‑forward network.

    This block applies a pre‑layer‑norm before the attention, then adds the attention output via a residual connection.
    A second pre‑layer‑norm is applied before the MoE, which itself returns an auxiliary dictionary.  The result of
    the MoE is combined with its feed‑forward projection and added to the residual.  A low‑rank vein projector
    is attached for diagnostic purposes but not directly used in the forward path.

    Args:
        d_model: Hidden dimensionality.
        num_skills: Number of expert skills in the MoE.
        d_route: Hidden size for the router module.
        top_k: Number of top experts selected by the MoE router.
        rank: Rank for the vein projector.
        n_heads: Number of attention heads.
        window_size: Local attention window size; 0 disables sliding window.
        num_kv_heads: Number of KV heads for GQA; if None, defaults to n_heads.
        use_rope_attn: Whether to apply rotary positional embeddings in the attention layer (RoPE on Q/K).
    """

    def __init__(self, d_model: int, num_skills: int = 64, d_route: int = 64,
                 top_k: int = 2, rank: int = 32,
                 n_heads: int = 8, window_size: int = 0,
                 num_kv_heads: Optional[int] = None,
                 use_rope_attn: bool = False, compile_moe_dispatch: bool = False):
        # NOTE: compile_moe_dispatch is unused in attention (kept for API symmetry).

        super().__init__()
        # Self‑attention with its own layer norm
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = GPT5Attention(
            d_model=d_model,
            n_heads=n_heads,
            window_size=window_size,
            num_kv_heads=num_kv_heads,
            use_rope_attn=use_rope_attn,
            compile_moe_dispatch=compile_moe_dispatch,
        )
        # MoE components
        self.norm_moe = nn.LayerNorm(d_model)
        self.router = CodebookRouter(d_model=d_model, d_route=d_route,
                                     num_skills=num_skills, temperature=0.7)
        self.moe = MoELayer(d_model=d_model, num_skills=num_skills, d_hidden=4 * d_model, init_method="xavier")
        # Feed‑forward network with its own layer norm
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.top_k = int(top_k)
        self.compile_moe_dispatch = bool(compile_moe_dispatch)
        # Diagnostic projector
        self.proj = VeinProjector(d_model, rank, implementation='linear', init_method='orthogonal')

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Self‑attention branch
        attn_in = self.norm_attn(x)
        attn_out = self.attn(attn_in)
        x = x + attn_out
        # Mixture‑of‑experts branch
        moe_in = self.norm_moe(x)
        h, aux = self.moe(moe_in, self.router, top_k=self.top_k, compile_dispatch=self.compile_moe_dispatch)
        x = x + h + self.ff(h)
        return x, {"moe": aux, "proj_rank": self.proj.rank}


# ======================= GPT-5 model ======================
class GPT5Model(nn.Module):
    """GPT‑5 model with self‑attention, mixture‑of‑experts and optional multimodal embeddings.

    Args:
        vocab_size: Vocabulary size for token embeddings and the LM head.
        d_model: Hidden dimensionality of the model.
        n_layers: Number of transformer/MoE blocks.
        num_skills: Number of expert skills per block.
        d_route: Hidden size for the router in the MoE.
        top_k: Top‑k experts to select in the MoE.
        rank: Rank used in diagnostic vein projectors.
        enable_multimodal: If True, use ``MultimodalEmbed`` for text/image/audio; otherwise use ``ToyEmbed``.
        image_dim: Dimensionality of image features if multimodal.
        audio_dim: Dimensionality of audio features if multimodal.
        use_rope_embed: Whether to apply rotary embeddings at the embedding layer (token+position).
        use_rope_attn: Whether to apply rotary embeddings at the attention layer (on query/key).
        n_heads: Number of attention heads in each block.
        window_size: Local attention window size; 0 disables sliding window.
        num_kv_heads: Number of KV heads for GQA; if None, defaults to ``n_heads``.
    """

    def __init__(self, vocab_size: int = 200000, d_model: int = 512,
                 n_layers: int = 4, num_skills: int = 64,
                 d_route: int = 64, top_k: int = 2, rank: int = 32,
                 enable_multimodal: bool = False,
                 image_dim: int = 1024, audio_dim: int = 512,
                 use_rope_embed: bool = False,
                 use_rope_attn: bool = False,
                 n_heads: int = 8, window_size: int = 0,
                 num_kv_heads: Optional[int] = None, compile_moe_dispatch: bool = False):
        super().__init__()
        # Embedding layer: multimodal or text‑only.  Distinguish between
        # embedding‑level RoPE (use_rope_embed) and attention‑level RoPE (use_rope_attn).
        if enable_multimodal:
            self.emb = MultimodalEmbed(vocab_size, d_model,
                                       image_dim=image_dim, audio_dim=audio_dim,
                                       use_rope_embed=use_rope_embed)
        else:
            self.emb = ToyEmbed(vocab_size, d_model, use_rope_embed=use_rope_embed)
        self.enable_multimodal = enable_multimodal

        # Transformer/MoE blocks with attention.  Pass the attention‑level
        # RoPE flag (use_rope_attn) into each block.  The embedding‑level
        # RoPE has already been applied in the embedding layer.
        self.blocks = nn.ModuleList([
            GPT5Block(
                d_model=d_model,
                num_skills=num_skills,
                d_route=d_route,
                top_k=top_k,
                rank=rank,
                n_heads=n_heads,
                window_size=window_size,
                num_kv_heads=num_kv_heads,
                use_rope_attn=use_rope_attn,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # NOTE: vote/retrieval/feedback logic is intentionally NOT stored on the
        # model. Use GPT5InferenceHelper and pass it into `forward_step(...)`.

    
    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                image_feat: Optional[torch.Tensor] = None,
                audio_feat: Optional[torch.Tensor] = None,
                *,
                return_info: bool = False) -> Any:
        """Training/eager forward pass (supports autograd).

        This mirrors the core path of ``forward_step`` but **does not** use
        ``torch.no_grad`` and skips the optional vote/retrieval side paths.
        Use ``forward_step`` for inference-time diagnostics.

        Args:
            input_ids: [B, T] token ids (required when ``enable_multimodal`` is False).
            image_feat: optional image features.
            audio_feat: optional audio features.
            return_info: if True, also return a dict of auxiliary MoE stats.

        Returns:
            logits [B, T, V] (and optionally info dict).
        """
        if self.enable_multimodal:
            h = self.emb(x_ids=input_ids, image_feat=image_feat, audio_feat=audio_feat)
        else:
            if input_ids is None:
                raise ValueError("input_ids must be provided when multimodal is disabled")
            h = self.emb(input_ids)

        aux_list: List[Dict[str, Any]] = []
        for blk in self.blocks:
            h, aux = blk(h)
            if aux is not None:
                aux_list.append(aux)

        h = self.norm(h)
        logits = self.lm_head(h)

        if return_info:
            return logits, {"moe_aux": aux_list}
        return logits

    @torch.no_grad()
    def forward_step(self,
                     input_ids: Optional[torch.Tensor] = None,
                     image_feat: Optional[torch.Tensor] = None,
                     audio_feat: Optional[torch.Tensor] = None,
                     step_idx: int = 0,
                     schema_required: bool = False,
                     *,
                     inference: Optional["GPT5InferenceHelper"] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Inference-only step wrapper.

        This method is deliberately a thin wrapper and **requires** an external
        :class:`GPT5InferenceHelper` instance. This keeps vote/retrieval/feedback
        logic out of the training model state and prevents accidental inclusion
        in the training graph.

        Usage:
            helper = GPT5InferenceHelper(d_model=..., rank=...)
            logits, info = model.forward_step(input_ids, step_idx=..., inference=helper)
        """
        if inference is None:
            raise RuntimeError(
                "forward_step requires an external `GPT5InferenceHelper` instance. "
                "Create one via `GPT5InferenceHelper(d_model=..., rank=...)` and pass it "
                "as `inference=...`."
            )
        return inference.forward_step(self,
                                      input_ids=input_ids,
                                      image_feat=image_feat,
                                      audio_feat=audio_feat,
                                      step_idx=step_idx,
                                      schema_required=schema_required)


# ============================================================
# ===================== INFERENCE-ONLY HELPERS =====================
# (vote / retrieval / feedback)
# IMPORTANT: These are intentionally isolated from GPT5Model.forward().
# Do NOT call them from the training step.
# =====================================================================
# These are NOT referenced by GPT5Model.forward() and are
# safe to keep in the same file as long as your training loop
# only uses GPT5Model.forward().
# ============================================================

def inference_token_entropy(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Per-position entropy of a logits tensor (returns tensor, no host sync)."""
    probs = torch.softmax(logits, dim=dim)
    logp = torch.log(torch.clamp(probs, min=1e-9))
    ent = -(probs * logp).sum(dim=dim)
    return ent.mean()  # scalar tensor


def inference_delta_kl(p_logits: torch.Tensor, q_logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """KL(p||q) for two logits tensors, returns a scalar tensor."""
    p = torch.softmax(p_logits, dim=dim)
    q = torch.softmax(q_logits, dim=dim)
    logp = torch.log(torch.clamp(p, min=1e-9))
    logq = torch.log(torch.clamp(q, min=1e-9))
    kl = (p * (logp - logq)).sum(dim=dim)
    return kl.mean()

class InferenceVeinProjector(nn.Module):
    """Lightweight low-rank projector used by inference-time alignment.

    This is intentionally minimal and CPU/GPU friendly. It is NOT required for training.
    """
    def __init__(self, d_model: int, rank: int, implementation: str = "linear", init_method: str = "xavier"):
        super().__init__()
        self.d_model = int(d_model)
        self.rank = int(rank)
        # simple linear down/up projection
        self.down = nn.Linear(self.d_model, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.d_model, bias=False)
        if init_method == "xavier":
            nn.init.xavier_uniform_(self.down.weight)
            nn.init.xavier_uniform_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        return self.up(z)

class InferenceVoteHead(nn.Module):
    """K=2 轻量投票：外部传入 generator/ checker。

    This module is only used at inference time inside ``GPT5InferenceHelper``.
    The public ``forward`` returns an object with the chosen logits and scores.
    """
    def __init__(self, K: int = 2):
        super().__init__()
        self.K = int(K)

    def forward(self, hidden: torch.Tensor, generator, checker):  # type: ignore[override]
        # Generate multiple candidate logits via the provided generator
        logits_list = generator(hidden, self.K)
        # Evaluate each candidate using the checker and pick the one with minimum score
        scores = [float(checker(lg)) for lg in logits_list]
        best = int(torch.tensor(scores).argmin().item())
        return type("VoteResult", (), {"chosen": logits_list[best], "scores": scores})


class InferenceStreamingRetriever:
    """占位检索器：按需替换为 MCP/RAG 客户端，接口保持 retrieve_async / poll。"""
    def __init__(self, threshold: float = 0.6):
        self.thr = float(threshold)
        self._last = None

    def retrieve_async(self, h: torch.Tensor):
        self._last = None

    def poll(self):
        return None  # object: ok, confidence, evidence_emb


class InferenceMoEController:
    """统一门控：熵触发投票；按固定周期触发检索。"""
    def __init__(self, entropy_trig: float = 2.2, retrieval_every: int = 192):
        self.entropy_trig = float(entropy_trig)
        self.retrieval_every = max(1, int(retrieval_every))

    def should_vote(self, logits, step_idx: int, schema_required: bool) -> bool:
        p = torch.softmax(logits[:, -1, :], dim=-1)
        # Note: .item() is acceptable here because this method is only used
        # during inference via GPT5InferenceHelper and is never invoked in
        # the training forward path.  Converting the entropy to a Python
        # float allows quick scalar comparison without introducing
        # synchronisation overhead in training.
        H = -(p * (p.clamp_min(1e-9).log())).sum(-1).mean().item()
        return bool(schema_required or (H >= self.entropy_trig))

    def should_retrieve(self, step_idx: int) -> bool:
        return (int(step_idx) % self.retrieval_every) == 0


class InferenceFeedbackEvaluator:
    """ΔKL + entropy（用于外层调度的反馈）。"""
    def __init__(self):
        self._prev = None

    def step(self, logits: torch.Tensor) -> Dict[str, float]:
        p = torch.softmax(logits[:, -1, :], dim=-1)
        # Convert mean entropy to a Python float; see MoEController.should_vote
        H = float((-(p * (p.clamp_min(1e-9).log())).sum(-1)).mean().item())
        dkl = 0.0
        if self._prev is not None:
            q = self._prev
            dkl = float((p * (p.clamp_min(1e-9).log() - q.clamp_min(1e-9).log())).sum(-1).mean().item())
        self._prev = p.detach()
        return {"entropy": round(H, 4), "dkl": round(dkl, 4)}


class InferenceMemoryBucket:
    """短期记忆桶；可扩展为短→长迁移。"""
    def __init__(self, max_short: int = 8):
        self.short: List[torch.Tensor] = []
        self.max_short = int(max_short)

    def update(self, emb: torch.Tensor, meta: Dict[str, Any], to_long: bool = False):
        if emb is None:
            return
        self.short.append(emb.detach())
        if len(self.short) > self.max_short:
            self.short.pop(0)


class InferencePrecisionAligner:
    """双态数对齐（Vein 子空间 convex mix + 主空间融合）。"""
    def __init__(self, projector, alpha: float = 0.35, beta: float = 0.2, tau: float = 0.15):
        self.proj = projector
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tau = float(tau)

    def align(self, h: torch.Tensor, evidence: torch.Tensor):
        if evidence is None or evidence.shape != h.shape:
            evidence = torch.zeros_like(h)
        z_h = self.proj.project(h)
        z_e = self.proj.project(evidence)
        z = (1 - self.alpha) * z_h + self.alpha * z_e
        h2 = self.proj.reconstruct(z)
        out = (1 - self.beta) * h + self.beta * h2
        return out, {"alpha": self.alpha, "beta": self.beta}


# =================== inference helper ====================
class GPT5InferenceHelper:
    """Inference‑only helper for GPT‑5.

    This class encapsulates vote, retrieval, feedback and alignment logic
    that are used solely during inference.  By moving these operations
    into a dedicated helper, the training forward path of ``GPT5Model``
    remains free of side effects such as random sampling, device
    synchronisation via ``.item()``, and memory updates.  The helper
    maintains its own projector, aligner and memory bucket, so that
    training does not inadvertently share state across inference runs.

    Args:
        d_model: Hidden dimensionality of the GPT‑5 model.
        rank: Rank for the vein projector used in alignment.
        K: Number of candidates for voting.
        thresh: Threshold for streaming retriever; evidence with
            confidence below this threshold is ignored.
        entropy_trig: Entropy trigger for voting.
        retrieval_every: Interval at which retrieval is attempted.
        align_alpha: Weight used in convex mixing of subspace projections.
        align_beta: Weight used in convex mixing of reconstructed and
            original hidden states.
        align_tau: Additional parameter for the precision aligner (not
            currently used but preserved for completeness).
        mem_max_short: Maximum number of short‑term memory vectors.
    """
    def __init__(self,
                 d_model: int,
                 rank: int,
                 *,
                 K: int = 2,
                 thresh: float = 0.6,
                 entropy_trig: float = 2.2,
                 retrieval_every: int = 192,
                 align_alpha: float = 0.35,
                 align_beta: float = 0.2,
                 align_tau: float = 0.15,
                 mem_max_short: int = 8) -> None:
        # Inference‑time modules
        self.vote = InferenceVoteHead(K=K)
        self.retriever = InferenceStreamingRetriever(threshold=thresh)
        self.ctrl = InferenceMoEController(entropy_trig=entropy_trig, retrieval_every=retrieval_every)
        self.feedback = InferenceFeedbackEvaluator()
        # Use a private projector for inference alignment, rather than
        # sharing the training projector.  This avoids cross‑contamination
        # of any learned state and ensures deterministic behaviour.
        self.projector = InferenceVeinProjector(d_model, rank, implementation='linear', init_method='orthogonal')
        self.align = InferencePrecisionAligner(self.projector, alpha=align_alpha, beta=align_beta, tau=align_tau)
        self.memory = InferenceMemoryBucket(max_short=mem_max_short)
        # Store retrieval threshold to avoid depending on private attribute
        self._retrieval_thresh = float(thresh)

    def postprocess(self,
                    model: nn.Module,
                    h: torch.Tensor,
                    logits: torch.Tensor,
                    step_idx: int,
                    schema_required: bool) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Postprocess logits with vote, retrieval, alignment and feedback.

        Args:
            model: The GPT‑5 model owning this helper (provides ``lm_head``).
            h: Hidden state [B,T,D] on which voting/retrieval operate.
            logits: Pre‑softmax logits [B,T,V] to be potentially replaced.
            step_idx: Current decoding step index.
            schema_required: Flag passed through to ``MoEController.should_vote``.

        Returns:
            Tuple of (possibly modified logits, info dict).
        """
        chosen_logits = logits
        align_info: Optional[Dict[str, Any]] = None
        # Voting: sample K noisy candidates and pick the lowest‑entropy one
        if self.ctrl.should_vote(chosen_logits, step_idx, schema_required):
            def gen(hidden: torch.Tensor, K: int) -> List[torch.Tensor]:
                outs: List[torch.Tensor] = []
                for _ in range(K):
                    noise = torch.randn_like(hidden) * 0.01
                    outs.append(model.lm_head(hidden + noise))
                return outs
            def check(l: torch.Tensor) -> float:
                # token_entropy returns a scalar tensor; convert to float for comparison
                return float(inference_token_entropy(l).item())
            vr = self.vote(h, gen, check)
            chosen_logits = vr.chosen
        # Retrieval and alignment: optionally update hidden state and recompute logits
        if self.ctrl.should_retrieve(step_idx):
            self.retriever.retrieve_async(h)
            res = self.retriever.poll()
            # res may be None or an object with attributes ok, confidence, evidence_emb
            if res and getattr(res, "ok", False) and getattr(res, "confidence", 0.0) >= self._retrieval_thresh:
                ev = getattr(res, "evidence_emb", None)
                # Fallback to zero evidence if missing or shape mismatch
                if ev is None or ev.shape != h.shape:
                    ev = torch.zeros_like(h)
                # Update short‑term memory
                self.memory.update(ev, meta=dict(step=step_idx), to_long=False)
                # Align hidden state to the evidence
                h, align_info = self.align.align(h, ev)
                # Recompute logits with the aligned hidden state
                chosen_logits = model.lm_head(h)
        # Compute feedback statistics (entropy and ΔKL)
        fb = self.feedback.step(chosen_logits)
        info = {"feedback": fb, "align": align_info, "mem_len": len(self.memory.short)}
        return chosen_logits, info

    @torch.no_grad()
    def forward_step(self,
                     model: "GPT5Model",
                     input_ids: Optional[torch.Tensor] = None,
                     image_feat: Optional[torch.Tensor] = None,
                     audio_feat: Optional[torch.Tensor] = None,
                     step_idx: int = 0,
                     schema_required: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Inference-only forward step.

        Runs the model's core path to obtain hidden states and logits, then
        applies vote/retrieval/feedback via :meth:`postprocess`.
        """
        # Core path (mirrors GPT5Model.forward but keeps the last hidden)
        if model.enable_multimodal:
            h = model.emb(x_ids=input_ids, image_feat=image_feat, audio_feat=audio_feat)
        else:
            if input_ids is None:
                raise ValueError("input_ids must be provided when multimodal is disabled")
            h = model.emb(input_ids)

        for blk in model.blocks:
            h, _aux = blk(h)

        h = model.norm(h)
        logits = model.lm_head(h)
        return self.postprocess(model, h, logits, step_idx=step_idx, schema_required=schema_required)


# ================== embed =================
# Note: VeinProjector now imported from apt.model.layers.blocks
