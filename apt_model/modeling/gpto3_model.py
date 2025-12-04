
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
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, d_model: int, n_heads: int, rank: int, tau_module: DynamicTau):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.rank = rank
        self.head_dim = d_model // n_heads
        self.tau_module = tau_module

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        for m in [self.W_q, self.W_k, self.W_v, self.W_o]:
            _init_linear(m)

        self.subspace = VeinSubspaceShared(self.head_dim, rank)

    def forward(self, x: torch.Tensor, load_factor: float = 1.0):
        B, T, D = x.size()
        H = self.n_heads

        q = self.W_q(x).view(B, T, H, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, H, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, H, self.head_dim).transpose(1, 2)

        zq = self.subspace.project(q)
        zk = self.subspace.project(k)
        zv = self.subspace.project(v)

        att = (zq @ zk.transpose(-2, -1)) / math.sqrt(max(self.rank, 1))
        att = F.softmax(att, dim=-1)
        y_r = att @ zv
        y = self.subspace.reconstruct(y_r)
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
    def __init__(self, d_model: int, vocab_size: int = 32000, image_dim: int = 1024, audio_dim: int = 512):
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
    def __init__(self, d_model: int, n_heads: int, d_ff: int, rank: int, tau_module: DynamicTau, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = TriVeinAttention(d_model, n_heads, rank, tau_module)
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

        for t in range(self.max_steps):
            steps = t + 1
            h, meta = self.reasoner.step(h)
            logits = lm_head(h)
            p = F.softmax(logits, dim=-1)

            kl = self._kl(p, p_prev)                             # [B,T]
            ent = -(p * p.clamp_min(1e-8).log()).sum(dim=-1)     # [B,T]
            z_new = meta["z_new"]
            vein_rel = (z_new - z_prev).norm(dim=-1) / (z_prev.norm(dim=-1) + 1e-6)
            halt = meta["p_halt"]

            done = ((kl < self.eps_kl) &
                    (vein_rel < self.eps_vein) &
                    (((ent_prev - ent) / (ent_prev + 1e-6)) < self.eps_entropy)) | (halt > self.halt_thresh)

            stall = stall + (~done).long()
            if (stall >= self.patience).sum() == 0:
                break

            # update prev state
            logits_prev, p_prev, ent_prev, z_prev = logits, p, ent, z_new

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
                 vocab_size: int = 32000,
                 d_model: int = 2048,
                 n_heads: int = 16,
                 d_ff: int = 8192,
                 num_layers: int = 24,
                 rank: int = 4,
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
        self.encoder = OmniInputEncoder(d_model, vocab_size=vocab_size)
        self.tau = DynamicTau()
        self.blocks = nn.ModuleList([
            GPT4oBlock(d_model, n_heads, d_ff, rank, self.tau) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        _init_linear(self.lm_head, std=0.02)

        # Use the first block's subspace as the shared vein (all blocks share head_dim)
        self.vein = self.blocks[0].attn.subspace
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
        k = max(1, int(self.global_budget * B * T))                     # global budget
        flat = ent.flatten()
        if k < flat.numel():
            thresh = flat.topk(k).values.min()
        else:
            thresh = flat.min()
        trig = max(self.entropy_trig, float(thresh))
        mask = (ent >= trig)                                            # [B,T]

        if mask.any():
            idx = mask.nonzero(as_tuple=False)                          # [N,2]
            h_sel = x[idx[:,0], idx[:,1], :]                            # [N,D]
            h_upd, info = self.controller(h_sel, self.lm_head)          # [N,D]
            x[idx[:,0], idx[:,1], :] = h_upd                            # scatter back
            logits = self.lm_head(x)

        return logits


# ----------------------------------------------------------
# Quick self-test (CPU)
# ----------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    model = GPTo3Model(d_model=512, n_heads=8, d_ff=2048, num_layers=4, rank=4)
    inp = torch.randint(0, 32000, (1, 64))
    with torch.no_grad():
        out = model(text_ids=inp)
    print("Logits:", tuple(out.shape))
