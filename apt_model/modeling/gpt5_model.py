# -*- coding: utf-8 -*-
"""
gpt5_model.py  —  All-in-one (CPU-friendly)
- Codebook MoE (top-k + shared expert)
- Leaf-Vote (K=2)
- Streaming retrieval stub + memory buckets
- Bi-state precision align (vein subspace)
- Composite feedback (entropy, ΔKL)

Updated to use refactored VFT/TVA modules from apt_model.modeling.blocks
"""

from __future__ import annotations
from typing import Any, Dict, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Use refactored VFT/TVA modules
from apt_model.modeling.blocks import VeinProjector


# ========================= utils =========================
def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Mean token entropy over [B,T,V] logits."""
    p = F.softmax(logits, dim=-1)
    return -(p * (p.clamp_min(1e-9).log())).sum(-1).mean()


# ========================= moe ===========================
class CodebookRouter(nn.Module):
    """Linear -> softmax gates."""
    def __init__(self, d_model: int, d_route: int, num_skills: int, temperature: float = 1.0):
        super().__init__()
        self.proj = nn.Linear(d_model, num_skills)
        self.temperature = float(temperature)

    def route(self, h: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.proj(h) / max(1e-6, self.temperature), dim=-1)  # [B,T,K]


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
    """Top-k MoE（CPU 友好版，mask 混合，不做 token dispatch）。"""
    def __init__(self, d_model: int, experts: nn.ModuleList, shared: nn.Module):
        super().__init__()
        self.experts = experts
        self.shared = shared
        self.num_skills = len(experts)

    def forward(self, h: torch.Tensor, router: CodebookRouter, top_k: int = 2) -> Tuple[torch.Tensor, Dict[str, Any]]:
        gate = router.route(h)                              # [B,T,K]
        k = min(int(top_k), gate.size(-1))
        vals, idx = torch.topk(gate, k=k, dim=-1)          # [B,T,k]
        out = torch.zeros_like(h)
        for j in range(k):
            w = vals[..., j:j+1]                           # [B,T,1]
            e = idx[..., j]                                # [B,T]
            mix = torch.zeros_like(h)
            for eid, expert in enumerate(self.experts):
                mask = (e == eid).float().unsqueeze(-1)    # [B,T,1]
                if mask.any():
                    mix = mix + expert(h) * mask
            out = out + w * mix
        out = out + self.shared(h)
        aux = {"gate_mean": float(gate.mean().item())}
        return out, aux


# ======================== runtime ========================
class VoteHead(nn.Module):
    """K=2 轻量投票：外部传入 generator/ checker。"""
    def __init__(self, K: int = 2):
        super().__init__()
        self.K = int(K)

    def forward(self, hidden: torch.Tensor, generator, checker):
        logits_list = generator(hidden, self.K)
        scores = [float(checker(lg)) for lg in logits_list]
        best = int(torch.tensor(scores).argmin().item())
        return type("VoteResult", (), {"chosen": logits_list[best], "scores": scores})


class StreamingRetriever:
    """占位检索器：按需替换为 MCP/RAG 客户端，接口保持 retrieve_async / poll。"""
    def __init__(self, threshold: float = 0.6):
        self.thr = float(threshold)
        self._last = None

    def retrieve_async(self, h: torch.Tensor):
        self._last = None

    def poll(self):
        return None  # object: ok, confidence, evidence_emb


class MoEController:
    """统一门控：熵触发投票；按固定周期触发检索。"""
    def __init__(self, entropy_trig: float = 2.2, retrieval_every: int = 192):
        self.entropy_trig = float(entropy_trig)
        self.retrieval_every = max(1, int(retrieval_every))

    def should_vote(self, logits, step_idx: int, schema_required: bool) -> bool:
        p = torch.softmax(logits[:, -1, :], dim=-1)
        H = -(p * (p.clamp_min(1e-9).log())).sum(-1).mean().item()
        return bool(schema_required or (H >= self.entropy_trig))

    def should_retrieve(self, step_idx: int) -> bool:
        return (int(step_idx) % self.retrieval_every) == 0


class FeedbackEvaluator:
    """ΔKL + entropy（用于外层调度的反馈）。"""
    def __init__(self):
        self._prev = None

    def step(self, logits: torch.Tensor) -> Dict[str, float]:
        p = torch.softmax(logits[:, -1, :], dim=-1)
        H = float((-(p * (p.clamp_min(1e-9).log())).sum(-1)).mean().item())
        dkl = 0.0
        if self._prev is not None:
            q = self._prev
            dkl = float((p * (p.clamp_min(1e-9).log() - q.clamp_min(1e-9).log())).sum(-1).mean().item())
        self._prev = p.detach()
        return {"entropy": round(H, 4), "dkl": round(dkl, 4)}


class MemoryBucket:
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


class PrecisionAligner:
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


# ================== embed =================
# Note: VeinProjector is now imported from apt_model.modeling.blocks

class ToyEmbed(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 4096):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        B, T = x_ids.shape
        pos = torch.arange(T, device=x_ids.device).unsqueeze(0).expand(B, T)
        return self.tok(x_ids) + self.pos(pos)


# ======================= GPT-5 block ======================
class GPT5Block(nn.Module):
    def __init__(self, d_model: int, num_skills: int = 64, d_route: int = 64,
                 top_k: int = 2, rank: int = 32):
        super().__init__()
        self.norm_in = nn.LayerNorm(d_model)
        self.router = CodebookRouter(d_model=d_model, d_route=d_route,
                                     num_skills=num_skills, temperature=0.7)
        experts = nn.ModuleList([MiniExpert(d_model, 4 * d_model, d_model) for _ in range(num_skills)])
        shared = SharedExpert(d_model, 2 * d_model, d_model, scale=0.25)
        self.moe = MoELayer(d_model, experts, shared)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.top_k = int(top_k)
        self.proj = VeinProjector(d_model, rank)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        h = self.norm_in(x)
        h, aux = self.moe(h, self.router, top_k=self.top_k)
        x = x + h + self.ff(h)
        return x, {"moe": aux, "proj_rank": self.proj.V.out_features}


# ======================= GPT-5 model ======================
class GPT5Model(nn.Module):
    def __init__(self, vocab_size: int = 32000, d_model: int = 512,
                 n_layers: int = 4, num_skills: int = 64,
                 d_route: int = 64, top_k: int = 2, rank: int = 32):
        super().__init__()
        self.emb = ToyEmbed(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            GPT5Block(d_model, num_skills, d_route, top_k, rank) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # orchestration
        self.vote = VoteHead(K=2)
        self.retriever = StreamingRetriever(threshold=0.6)
        self.ctrl = MoEController(entropy_trig=2.2, retrieval_every=192)
        self.feedback = FeedbackEvaluator()
        self.projector = VeinProjector(d_model, rank)
        self.align = PrecisionAligner(self.projector, alpha=0.35, beta=0.2, tau=0.15)
        self.memory = MemoryBucket(max_short=8)

    @torch.no_grad()
    def forward_step(self, input_ids: torch.Tensor, step_idx: int = 0,
                     schema_required: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """One incremental step; returns logits [B,T,V] and diagnostic info."""
        h = self.emb(input_ids)
        for blk in self.blocks:
            h, _ = blk(h)
        h = self.norm(h)
        logits = self.lm_head(h)

        # Leaf-Vote when needed
        if self.ctrl.should_vote(logits, step_idx, schema_required):
            def gen(hidden, K):
                outs = []
                for _ in range(K):
                    noise = torch.randn_like(hidden) * 0.01
                    outs.append(self.lm_head(hidden + noise))
                return outs
            def check(l):
                return float(token_entropy(l).item())
            vr = self.vote(h, gen, check)
            logits = vr.chosen

        # Retrieval + memory + bi-state align
        align_info = None
        if self.ctrl.should_retrieve(step_idx):
            self.retriever.retrieve_async(h)
            res = self.retriever.poll()
            if res and getattr(res, "ok", False) and getattr(res, "confidence", 0.0) >= 0.6:
                ev = getattr(res, "evidence_emb", None)
                if ev is None or ev.shape != h.shape:
                    ev = torch.zeros_like(h)
                self.memory.update(ev, meta=dict(step=step_idx), to_long=False)
                h, align_info = self.align.align(h, ev)
                logits = self.lm_head(h)

        fb = self.feedback.step(logits)
        info = {"feedback": fb, "align": align_info, "mem_len": len(self.memory.short)}
        return logits, info


# ========================== demo ==========================
if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, V = 1, 24, 4096
    x = torch.randint(0, V, (B, T))
    model = GPT5Model(vocab_size=V, d_model=256, n_layers=3,
                      num_skills=16, d_route=32, top_k=2, rank=16)
    logits, info = model.forward_step(x, step_idx=0, schema_required=False)
    print("logits:", tuple(logits.shape))
    print("info:", info)
