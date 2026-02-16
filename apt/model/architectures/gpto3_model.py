
# GPTo3 Model — Structured Reasoning atop GPT‑4o backbone
# --------------------------------------------------------
# This file is self-contained. It includes minimal GPT‑4o building blocks
# (DynamicTau, VeinSubspaceShared, TriVeinAttention, HybridFFN, OmniInputEncoder)
# and adds o3-specific pieces:
#   - HaltingUnit (learned stop signal)
#   - ExpertRouter / MiniExpert (token-wise MoE in vein subspace)
#   - StructuredReasoner (a single reasoning step)
#   - ReasoningController (multi-metric halting with budget)
#   - GPTo3Model (only high-entropy tokens enter structured reasoning)
#
# The external interface is simple:
#   logits = GPTo3Model(...)(text_ids=...)
#
# Notes:
# - CPU-only friendly; no CUDA-specific calls are required.
# - This is a compact reference implementation intended for research prototyping.

import math
from typing import Optional, Tuple, List

# Try to import the fake_torch from the external apt package.  If that
# fails (for example, when the apt package is not available), fall
# back to the real torch.  The get_torch function returns either
# the fake torch or the actual torch module as appropriate.
try:
    from apt.core.fake_torch import get_torch  # type: ignore
except Exception:
    import torch as _torch  # type: ignore
    def get_torch():
        return _torch
torch = get_torch()
nn = torch.nn
F = torch.nn.functional


# ----------------------------------------------------------
# Utilities
# ----------------------------------------------------------

def _init_linear(module: nn.Linear, std: float = 0.02):
    nn.init.normal_(module.weight, mean=0.0, std=std)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


# ----------------------------------------------------------
# Dynamic τ Gating (from 4o)
# ----------------------------------------------------------

class DynamicTau(nn.Module):
    def __init__(self, init_tau=0.18, min_tau=0.05, max_tau=0.35, adapt_rate=0.05):
        super().__init__()
        self.register_buffer('tau', torch.tensor(float(init_tau)))
        self.min_tau = float(min_tau)
        self.max_tau = float(max_tau)
        self.adapt_rate = float(adapt_rate)

    def forward(self, load_factor: float):
        # Small smooth adaptation (no state explosion)
        new_tau = self.tau * (1.0 - self.adapt_rate * (float(load_factor) - 1.0))
        new_tau = torch.clamp(new_tau, self.min_tau, self.max_tau)
        self.tau.copy_(new_tau)
        return self.tau


# ----------------------------------------------------------
# Low-rank shared subspace (Vein) (from 4o)
# ----------------------------------------------------------

class VeinSubspaceShared(nn.Module):
    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.d_model = int(d_model)
        self.rank = int(rank)
        self.U = nn.Parameter(torch.empty(d_model, rank))
        self.V = nn.Parameter(torch.empty(d_model, rank))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        # x[..., D] -> [..., r]
        return x @ self.V

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        # z[..., r] -> [..., D]
        return z @ self.U.t()


# ----------------------------------------------------------
# Tri‑Vein Attention (main low‑rank attention in shared subspace) (from 4o)
# ----------------------------------------------------------

class TriVeinAttention(nn.Module):
    """
    Tri‑Vein attention layer operating in a low‑rank subspace.
    This version supports optional rotary positional encodings (RoPE),
    sliding window attention (SWA) and grouped‑query attention (GQA).

    Args:
        d_model: hidden dimension of the model
        n_heads: number of attention heads
        rank: low‑rank dimension for vein subspace
        window_size: if > 0, restrict attention to a causal window of this size
        num_kv_heads: if set and < n_heads, groups KV heads and repeats to Q heads (GQA)
        use_rope: whether to apply rotary positional embeddings to Q and K
    """
    def __init__(self, d_model: int, n_heads: int, rank: int,
                 window_size: int = 0,
                 num_kv_heads: Optional[int] = None,
                 use_rope: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.rank = int(rank)
        self.head_dim = self.d_model // self.n_heads
        self.window_size = int(window_size)
        self.num_kv_heads = int(num_kv_heads) if num_kv_heads is not None else self.n_heads
        # Ensure heads divisible when GQA is used
        assert self.n_heads % self.num_kv_heads == 0, "n_heads must be divisible by num_kv_heads"
        self.use_rope = bool(use_rope)

        # Linear projections
        self.W_q = nn.Linear(self.d_model, self.d_model)
        self.W_k = nn.Linear(self.d_model, self.d_model)
        self.W_v = nn.Linear(self.d_model, self.d_model)
        self.W_o = nn.Linear(self.d_model, self.d_model)
        for m in [self.W_q, self.W_k, self.W_v, self.W_o]:
            _init_linear(m)

        # Shared vein subspace for low‑rank projections
        self.subspace = VeinSubspaceShared(self.head_dim, self.rank)

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary positional embeddings to x [B,H,T,D], trimming any padding."""
        B, H, T, D = x.size()
        orig_D = D
        # Pad one dimension for odd head_dim
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
        sinusoid = pos.unsqueeze(1) * inv_freq.unsqueeze(0)  # [T,half]
        sin = sinusoid.sin().unsqueeze(0).unsqueeze(0)       # [1,1,T,half]
        cos = sinusoid.cos().unsqueeze(0).unsqueeze(0)       # [1,1,T,half]
        x1 = x[..., :half]
        x2 = x[..., half:half*2]
        x_rot_first = x1 * cos - x2 * sin
        x_rot_second = x1 * sin + x2 * cos
        x_rot = torch.cat([x_rot_first, x_rot_second], dim=-1)
        # Trim to original dimension if padded
        if D != orig_D:
            x_rot = x_rot[..., :orig_D]
        return x_rot

    def forward(self, x: torch.Tensor, load_factor: float = 1.0, *, is_causal: bool = True) -> torch.Tensor:
        B, T, D = x.size()
        H = self.n_heads
        # Linear projections and reshape to [B,H,T,D]
        q = self.W_q(x).view(B, T, H, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, self.head_dim).transpose(1, 2)

        # Apply RoPE if enabled
        if self.use_rope:
            q = self._apply_rope(q)
            k = self._apply_rope(k)

        # GQA: group KV heads and replicate
        if self.num_kv_heads < self.n_heads:
            group_size = self.n_heads // self.num_kv_heads
            # Reshape to [B, num_kv_heads, group_size, T, head_dim]
            k_reshaped = k.reshape(B, self.num_kv_heads, group_size, T, self.head_dim)
            v_reshaped = v.reshape(B, self.num_kv_heads, group_size, T, self.head_dim)
            # Take first head in each group
            k_small = k_reshaped[:, :, 0, :, :]
            v_small = v_reshaped[:, :, 0, :, :]
            # Repeat to full head count
            k = k_small.unsqueeze(2).repeat(1, 1, group_size, 1, 1).reshape(B, self.n_heads, T, self.head_dim)
            v = v_small.unsqueeze(2).repeat(1, 1, group_size, 1, 1).reshape(B, self.n_heads, T, self.head_dim)

        # Project into low‑rank subspace
        zq = self.subspace.project(q)
        zk = self.subspace.project(k)
        zv = self.subspace.project(v)

        # Compute attention in subspace
        att = (zq @ zk.transpose(-2, -1)) / math.sqrt(max(self.rank, 1))

        # Compose masks: causal + sliding window
        mask = None
        if is_causal:
            causal = torch.triu(torch.ones((T, T), dtype=torch.bool, device=att.device), diagonal=1)
            mask = causal
        if self.window_size and self.window_size > 0:
            i = torch.arange(T, device=att.device).view(T, 1)
            j = torch.arange(T, device=att.device).view(1, T)
            local = j < (i - (self.window_size - 1))
            mask = local if mask is None else (mask | local)
        if mask is not None:
            att = att.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Softmax
        att = F.softmax(att, dim=-1)

        # Low‑rank output
        y_sub = att @ zv
        # Reconstruct to full dimension
        y = self.subspace.reconstruct(y_sub)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(y)


# ----------------------------------------------------------
# Hybrid FeedForward (Mini‑MoE) (from 4o)
# ----------------------------------------------------------

class HybridFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.SiLU(),
                nn.Linear(d_ff, d_model),
            ) for _ in range(num_experts)
        ])
        for e in self.experts:
            for m in e:
                if isinstance(m, nn.Linear):
                    _init_linear(m)
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        _init_linear(self.gate)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(self.gate(x), dim=-1)  # [B,T,E]
        out = 0.0
        for i, expert in enumerate(self.experts):
            w = probs[..., i:i+1]               # [B,T,1]
            out = out + w * expert(x)
        return self.drop(out)


# ----------------------------------------------------------
# Omni Input Encoder (from 4o; text‑only is fine)
# ----------------------------------------------------------

class OmniInputEncoder(nn.Module):
    def __init__(self, d_model: int, vocab_size: int = 200000, image_dim: int = 1024, audio_dim: int = 512):
        super().__init__()
        self.text_emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.text_emb.weight, mean=0.0, std=0.02)
        self.image_proj = nn.Linear(image_dim, d_model)
        self.audio_proj = nn.Linear(audio_dim, d_model)
        _init_linear(self.image_proj); _init_linear(self.audio_proj)

    def forward(self, text_ids=None, image_feat=None, audio_feat=None):
        parts = []
        if text_ids is not None:
            parts.append(self.text_emb(text_ids))
        if image_feat is not None:
            parts.append(self.image_proj(image_feat))
        if audio_feat is not None:
            parts.append(self.audio_proj(audio_feat))
        assert len(parts) > 0, "At least one modality must be provided"
        return sum(parts) / len(parts)


# ----------------------------------------------------------
# GPT‑4o Block (attention + hybrid FFN)
# ----------------------------------------------------------

class GPT4oBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, rank: int,
                 *, window_size: int = 0, num_kv_heads: Optional[int] = None, use_rope: bool = False,
                 dropout: float = 0.0):
        """
        A GPT‑4o block consisting of a single TriVeinAttention followed by a HybridFFN.

        Args:
            d_model: hidden dimension
            n_heads: number of attention heads
            d_ff: feedforward width
            rank: low‑rank dimension for vein subspace
            window_size: sliding window size for attention (0 = global)
            num_kv_heads: number of KV heads for GQA (None = same as Q heads)
            use_rope: whether to apply RoPE to Q/K
            dropout: feedforward dropout
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = TriVeinAttention(d_model, n_heads, rank,
                                     window_size=window_size,
                                     num_kv_heads=num_kv_heads,
                                     use_rope=use_rope)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = HybridFFN(d_model, d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor, load_factor: float = 1.0):
        x = x + self.attn(self.norm1(x), load_factor=load_factor)
        x = x + self.ffn(self.norm2(x))
        return x


# ==========================================================
# o3 NEW MODULES
# ==========================================================

class HaltingUnit(nn.Module):
    """A small learned stop signal in (0,1)."""
    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, 1)
        _init_linear(self.fc)
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(h))  # [B,T,1]


class ExpertRouter(nn.Module):
    """Token-wise router in the vein subspace."""
    def __init__(self, r: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.top_k = int(top_k)
        self.score = nn.Linear(r, num_experts, bias=False)
        _init_linear(self.score)
    def forward(self, z: torch.Tensor):
        # z: [B,T,r]
        probs = F.softmax(self.score(z), dim=-1)   # [B,T,E]
        topv, topi = probs.topk(self.top_k, dim=-1)
        return topv, topi, probs


class MiniExpert(nn.Module):
    """Tiny expert operating in the vein subspace r."""
    def __init__(self, r: int, width: int = 128):
        super().__init__()
        h = max(64, int(width))
        self.net = nn.Sequential(
            nn.Linear(r, h), nn.SiLU(),
            nn.Linear(h, r),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                _init_linear(m)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class StructuredReasoner(nn.Module):
    """
    One reasoning step:
      z = V^T h
      (alpha, idx) = Router(z)
      z_new = sum_k alpha_k * Expert_k(z)
      h_new = U z_new
      p_halt = sigmoid(W h_new)
    """
    def __init__(self, vein: VeinSubspaceShared, num_experts: int = 4, top_k: int = 2, width: int = 128):
        super().__init__()
        self.vein = vein
        r = vein.rank
        self.router = ExpertRouter(r, num_experts=num_experts, top_k=top_k)
        self.experts = nn.ModuleList([MiniExpert(r, width) for _ in range(num_experts)])
        self.halt = HaltingUnit(vein.d_model)

    def step(self, h: torch.Tensor):
        B, T, D = h.shape
        z = self.vein.project(h)                  # [B,T,r]

        topv, topi, full = self.router(z)         # [B,T,K], [B,T,K], [B,T,E]
        # Aggregate experts - Fixed logic
        z_new = torch.zeros_like(z)
        for k in range(topi.size(-1)):
            idx = topi[..., k]                    # [B,T]
            w = topv[..., k:k+1]                  # [B,T,1]
            # Process each expert for tokens that selected it
            for e_id, expert in enumerate(self.experts):
                mask = (idx == e_id)              # [B,T]
                if mask.any():
                    z_sel = z[mask]
                    z_upd = expert(z_sel)
                    # Accumulate weighted expert outputs
                    z_new[mask] = z_new[mask] + w[mask] * z_upd
        # residual blend (keep some of original z)
        blend = topv.sum(dim=-1, keepdim=True).clamp(max=0.9)
        z_final = z_new * blend + z * (1.0 - blend)

        h_new = self.vein.reconstruct(z_final)    # [B,T,D]
        p_halt = self.halt(h_new).squeeze(-1)     # [B,T]
        return h_new, {"router_probs": full, "p_halt": p_halt, "z_old": z, "z_new": z_final}


class ReasoningController(nn.Module):
    """Multi‑metric halting with patience and max steps."""
    def __init__(self, vein: VeinSubspaceShared,
                 max_steps: int = 6, patience: int = 2,
                 eps_kl: float = 0.02, eps_vein: float = 0.03, eps_entropy: float = 0.05, halt_thresh: float = 0.8,
                 topk_experts: int = 2):
        super().__init__()
        self.vein = vein
        self.max_steps = max(1, int(max_steps))
        self.patience = max(1, int(patience))
        self.eps_kl = float(eps_kl)
        self.eps_vein = float(eps_vein)
        self.eps_entropy = float(eps_entropy)
        self.halt_thresh = float(halt_thresh)
        self.reasoner = StructuredReasoner(vein, top_k=topk_experts)

    @staticmethod
    def _kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8):
        p = p.clamp_min(eps); q = q.clamp_min(eps)
        return (p * (p.log() - q.log())).sum(dim=-1)

    def forward(self, h: torch.Tensor, lm_head: nn.Linear):
        # h: [N,D] or [B,T,D]; we also accept [N,D] and treat as batch 1
        reshape_back = False
        if h.dim() == 2:
            h = h.unsqueeze(0)  # [1,N,D]
            reshape_back = True

        B, T, D = h.shape
        logits_prev = lm_head(h)                  # [B,T,V]
        p_prev = F.softmax(logits_prev, dim=-1)
        ent_prev = -(p_prev * p_prev.clamp_min(1e-8).log()).sum(dim=-1)  # [B,T]
        z_prev = self.vein.project(h)

        stall = torch.zeros(B, T, dtype=torch.long, device=h.device)
        steps = 0

        # Iterate up to ``max_steps`` reasoning steps.  We deliberately avoid
        # converting tensors to Python scalars inside the loop to prevent
        # device synchronisation.  Early termination based on the patience
        # threshold has been removed; instead, the loop always executes
        # ``max_steps`` iterations (unless a break condition is triggered
        # before the loop, such as an empty selection of tokens).  This
        # simplifies the control flow and avoids `.item()` calls.
        for t in range(self.max_steps):
            steps = t + 1
            h, meta = self.reasoner.step(h)
            logits = lm_head(h)
            p = F.softmax(logits, dim=-1)

            # Compute per‑token metrics
            kl = self._kl(p, p_prev)                             # [B,T]
            ent = -(p * p.clamp_min(1e-8).log()).sum(dim=-1)     # [B,T]
            z_new = meta["z_new"]
            vein_rel = (z_new - z_prev).norm(dim=-1) / (z_prev.norm(dim=-1) + 1e-6)
            halt = meta["p_halt"]

            # Determine which tokens have converged or should halt.  We keep
            # these computations in tensor form and update ``stall`` to
            # accumulate the count of consecutive non‑converged steps.
            done = ((kl < self.eps_kl) &
                    (vein_rel < self.eps_vein) &
                    (((ent_prev - ent) / (ent_prev + 1e-6)) < self.eps_entropy)) | (halt > self.halt_thresh)
            stall = stall + (~done).long()
            # Update previous state for the next iteration
            logits_prev, p_prev, ent_prev, z_prev = logits, p, ent, z_new
            # We intentionally do not break early based on patience; the
            # loop will continue for the full ``max_steps`` so that the
            # reasoning is deterministic and free of Python scalar conversions.

        if reshape_back:
            h = h.squeeze(0)   # [N,D] again
        return h, {"steps": steps}


# ----------------------------------------------------------
# GPTo3 Model
# ----------------------------------------------------------

class GPTo3Model(nn.Module):
    """
    o3 = 4o + structured reasoning (token‑wise, budgeted).
    - Backbone: GPT‑4o (TriVeinAttention + HybridFFN), L layers
    - Controller: only high‑entropy tokens enter reasoning loop
    """
    def __init__(self,
                 vocab_size: int = 200000,
                 d_model: int = 2048,
                 n_heads: int = 16,
                 d_ff: int = 8192,
                 num_layers: int = 24,
                 rank: int = 4,
                 # optional TriVein enhancements
                 window_size: int = 0,
                 num_kv_heads: Optional[int] = None,
                 use_rope: bool = False,
                 # controller params
                 entropy_trig: float = 2.0,
                 global_budget: float = 0.15,
                 max_reason_steps: int = 6,
                 patience: int = 2,
                 eps_kl: float = 0.02,
                 eps_vein: float = 0.03,
                 eps_entropy: float = 0.05,
                 halt_thresh: float = 0.8,
                 topk_experts: int = 2):
        super().__init__()
        # Encoder for text/image/audio with the specified vocabulary size
        self.encoder = OmniInputEncoder(d_model, vocab_size=vocab_size)
        # Adaptive τ scheduler
        self.tau = DynamicTau()
        # Transformer blocks with TriVein attention and hybrid FFN; pass window_size/gqa/rope params
        self.blocks = nn.ModuleList([
            GPT4oBlock(d_model, n_heads, d_ff, rank,
                       window_size=window_size,
                       num_kv_heads=num_kv_heads,
                       use_rope=use_rope)
            for _ in range(num_layers)
        ])
        # Final normalisation and language head
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        _init_linear(self.lm_head, std=0.02)

        # Initialise a separate vein subspace for structured reasoning.
        # We use the full model dimension ``d_model`` rather than the head
        # dimension from the first block.  Sharing the attention subspace
        # (of size head_dim) with the reasoning components leads to a
        # mismatch when projecting the hidden state ``h`` of shape
        # [N,D] into a space of shape [*, head_dim].  Using a new
        # ``VeinSubspaceShared(d_model, rank)`` ensures the projection
        # and reconstruction operations match the full hidden size.
        self.vein = VeinSubspaceShared(d_model, rank)
        self.controller = ReasoningController(
            self.vein, max_steps=max_reason_steps, patience=patience,
            eps_kl=eps_kl, eps_vein=eps_vein, eps_entropy=eps_entropy,
            halt_thresh=halt_thresh, topk_experts=topk_experts
        )
        self.entropy_trig = float(entropy_trig)
        self.global_budget = float(global_budget)

    def _token_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Calculate token entropy (gradient-enabled for training)."""
        p = logits.softmax(-1).clamp_min(1e-8)
        return -(p * p.log()).sum(-1)  # [B,T]

    def forward(self, text_ids=None, image_feat=None, audio_feat=None, load_factor: float = 1.0):
        x = self.encoder(text_ids, image_feat, audio_feat)              # [B,T,D]
        for blk in self.blocks:
            x = blk(x, load_factor=load_factor)
        x = self.norm(x)
        logits = self.lm_head(x)                                        # [B,T,V]

        # ---- Structured reasoning only for high‑entropy tokens ----
        ent = self._token_entropy(logits)                               # [B,T]
        B, T = ent.shape
        # Determine the entropy threshold to trigger structured reasoning.
        k = max(1, int(self.global_budget * B * T))
        flat = ent.flatten()
        if k < flat.numel():
            thresh = flat.topk(k).values.min()
        else:
            thresh = flat.min()
        # Use tensor operations to compute the maximum between the fixed
        # entropy trigger and the observed threshold.  Avoid casting to
        # Python floats, which would cause device synchronisation.
        trig_tensor = torch.tensor(self.entropy_trig, dtype=ent.dtype, device=ent.device)
        trig = torch.maximum(trig_tensor, thresh)
        mask = ent >= trig

        # Select the indices of tokens exceeding the entropy trigger.  Use
        # ``nonzero`` to obtain a 2‑D index tensor.  We avoid ``mask.any()``
        # to sidestep the truth‑value ambiguity of tensors; instead, we
        # inspect the size of the resulting index tensor.
        idx = mask.nonzero(as_tuple=False)  # [N,2] (or empty if no tokens)
        if idx.numel() > 0:
            # Gather the hidden states for the selected tokens, run the
            # reasoning controller, and scatter the updated representations
            # back into the sequence.  The logits are recomputed only
            # when reasoning has been applied.
            h_sel = x[idx[:, 0], idx[:, 1], :]
            h_upd, info = self.controller(h_sel, self.lm_head)
            x[idx[:, 0], idx[:, 1], :] = h_upd
            logits = self.lm_head(x)

        return logits


# ----------------------------------------------------------
# Quick self-test (CPU)
# ----------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    # Instantiate a small GPTo3Model for a smoke test.  Use the
    # configured vocabulary size rather than assuming a hard-coded
    # 32k vocabulary.  This ensures the embedding and output head
    # dimensions align when the default configuration uses a 200k vocab.
    model = GPTo3Model(d_model=512, n_heads=8, d_ff=2048, num_layers=4, rank=4)
    vocab_sz = model.lm_head.weight.size(0)
    # Randomly sample a short sequence of token IDs from the model's
    # vocabulary.  Keeping the sequence short makes the test fast while
    # validating the forward pass across the embedding and transformer.
    inp = torch.randint(0, vocab_sz, (1, 64))
    with torch.no_grad():
        out = model(text_ids=inp)
    print("Logits:", tuple(out.shape))
