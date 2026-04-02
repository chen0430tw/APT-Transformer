# -*- coding: utf-8 -*-
"""
standard_transformer.py — 内置标准因果语言模型 Transformer

无外部依赖，可在任何环境直接使用。
架构：Token Embedding + RoPE + N × (CausalSelfAttention + SwiGLU FFN) + LM Head

用法（通过 quickcook --model-arch standard-transformer）:
    create_model("standard-transformer", vocab_size=65536, d_model=1536,
                 num_heads=16, num_layers=18, max_seq_len=3072)
"""

from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── RoPE ─────────────────────────────────────────────────────────────────────

def _build_rope_cache(seq_len: int, head_dim: int, device: torch.device,
                      dtype: torch.dtype, base: int = 10000):
    half = head_dim // 2
    theta = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, theta)                       # [T, half]
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)
    return cos, sin                                     # [T, half]


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, D]  cos/sin: [T, D//2]"""
    B, H, T, D = x.shape
    half = D // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos[:T].unsqueeze(0).unsqueeze(0)              # [1, 1, T, half]
    s = sin[:T].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


# ─── Attention ────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.dropout   = dropout

        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        q, k, v = self.qkv(x).split(C, dim=-1)             # each [B, T, C]
        q = q.view(B, T, H, D).transpose(1, 2)             # [B, H, T, D]
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )                                                   # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


# ─── SwiGLU FFN ───────────────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        super().__init__()
        d_ff = d_ff or int(d_model * 8 / 3)
        d_ff = (d_ff + 63) // 64 * 64          # round up to multiple of 64
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff,    d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ─── Transformer Block ────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                 d_ff: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.ln1  = nn.RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, dropout)
        self.ln2  = nn.RMSNorm(d_model)
        self.ffn  = SwiGLUFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.ffn(self.ln2(x))
        return x


# ─── Top-level Model ──────────────────────────────────────────────────────────

class StandardTransformer(nn.Module):
    """
    标准因果语言模型 Transformer。

    接口与 APTModel 兼容：
        forward(input_ids) -> logits [B, T, vocab_size]
        forward(input_ids, labels) -> (loss, logits)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int    = 1024,
        num_heads: int  = 16,
        num_layers: int = 12,
        max_seq_len: int = 2048,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model     = d_model
        self.num_heads   = num_heads
        self.max_seq_len = max_seq_len

        self.embed   = nn.Embedding(vocab_size, d_model)
        self.drop    = nn.Dropout(dropout)
        self.layers  = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.ln_f    = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # weight tying
        self.lm_head.weight = self.embed.weight

        # RoPE cache (re-built if seq_len > max_seq_len at forward time)
        head_dim = d_model // num_heads
        cos, sin = _build_rope_cache(max_seq_len, head_dim,
                                     device=torch.device("cpu"),
                                     dtype=torch.float32)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self):
        std = 0.02
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=std)

    def _ensure_rope(self, seq_len: int):
        if seq_len <= self.rope_cos.size(0):
            return
        head_dim = self.d_model // self.num_heads
        cos, sin = _build_rope_cache(seq_len, head_dim,
                                     device=self.embed.weight.device,
                                     dtype=self.embed.weight.dtype)
        self.rope_cos = cos
        self.rope_sin = sin

    def forward(
        self,
        input_ids: torch.Tensor,                    # [B, T]
        labels:    Optional[torch.Tensor] = None,   # [B, T]
    ):
        B, T = input_ids.shape
        self._ensure_rope(T)

        x = self.drop(self.embed(input_ids))        # [B, T, C]
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]

        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.ln_f(x)
        logits = self.lm_head(x)                    # [B, T, vocab_size]

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            return loss, logits

        return logits
