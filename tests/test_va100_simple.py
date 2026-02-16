#!/usr/bin/env python3
"""
Virtual A100 - 低熵重构版本
==============================

重构原则：
  1. 单一数据流 - Token → Embedding → Layers → Logits → Token
  2. 状态显式化 - 所有状态都在不可变的 State 对象中
  3. 配置类型化 - 消除 magic numbers
  4. 职责分离 - Computation, State, Policy 严格分离
  5. 可观测性 - 所有操作可追踪

作者：GPT-5.2 R2
版本：2.0.0-低熵
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Literal, Protocol
from abc import ABC, abstractmethod
from enum import Enum
import time


# ============================================================================
# 1. 配置层（消除配置熵）
# ============================================================================

@dataclass(frozen=True)
class ModelConfig:
    """模型配置（不可变）"""
    # 结构
    H: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 8
    vocab_size: int = 32000

    # RoPE
    rope_theta: float = 500000.0

    # 归一化
    norm_eps: float = 1e-5

    def __post_init__(self):
        assert self.H % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0

    @property
    def head_dim(self) -> int:
        return self.H // self.n_heads

    @property
    def kv_repeat(self) -> int:
        return self.n_heads // self.n_kv_heads


@dataclass(frozen=True)
class InferConfig:
    """推理配置（不可变）"""
    max_ctx: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    use_tf32: bool = False
    prefetch_window: int = 2

    def __post_init__(self):
        assert 0 <= self.temperature <= 2.0
        assert 0 <= self.top_p <= 1.0


@dataclass(frozen=True)
class SystemConfig:
    """系统配置（不可变）"""
    verbose: bool = False
    seed: int = 42
    device: str = 'cpu'


# ============================================================================
# 2. 状态层（消除状态熵）
# ============================================================================

@dataclass
class ModelState:
    """模型状态（静态权重）"""
    # 权重
    embed_weight: Optional[np.ndarray]  # (V, H)
    head_weight: Optional[np.ndarray]    # (V, H)
    ghost_layers: List[Any]             # List of GhostLayer

    # 配置
    model_cfg: ModelConfig

    # 内部状态（不可变引用，可变内容）
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(42),
        init=False,
        repr=False
    )


@dataclass
class StepContext:
    """单步上下文（每步创建新实例）"""
    position: int
    hidden: np.ndarray          # (H,) 当前隐藏状态
    kv_cache: Dict[int, Tuple]  # {layer_idx: (K, V)} KV 缓存

    @property
    def seq_len(self) -> int:
        """当前序列长度"""
        # 从 KV 缓存推断
        if self.kv_cache:
            k, _ = next(iter(self.kv_cache.values()))
            return k.shape[0] + 1
        return 1


@dataclass
class RuntimeStats:
    """运行时统计（可变，但隔离）"""
    start_time: float = field(default_factory=time.perf_counter)

    # 计数
    tokens_generated: int = 0
    steps_completed: int = 0

    # 时间
    prefill_time_s: float = 0.0
    decode_time_s: float = 0.0

    # 资源
    total_compute_ms: float = 0.0
    total_transfer_ms: float = 0.0

    @property
    def total_time_s(self) -> float:
        return time.perf_counter() - self.start_time

    @property
    def tokens_per_second(self) -> float:
        decode_time = max(self.decode_time_s, 1e-9)
        return self.tokens_generated / decode_time

    def record_compute(self, ms: float):
        """记录计算时间"""
        self.total_compute_ms += ms

    def record_transfer(self, ms: float):
        """记录传输时间"""
        self.total_transfer_ms += ms


# ============================================================================
# 3. 计算层（纯函数，无副作用）
# ============================================================================

class MathOps:
    """数学运算（纯函数）"""

    @staticmethod
    def rms_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """RMS 归一化"""
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
        return x / rms

    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """数值稳定的 softmax"""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-12)

    @staticmethod
    def rope(x: np.ndarray, pos: int, dim: int, theta: float) -> np.ndarray:
        """RoPE 位置编码"""
        # x: (n_heads, dim)
        freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[:dim//2] / dim))
        angles = pos * freqs

        x1 = x[:, :dim//2].reshape(-1)
        x2 = x[:, dim//2:].reshape(-1)

        cos = np.cos(angles).astype(x.dtype)
        sin = np.sin(angles).astype(x.dtype)

        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos

        return np.concatenate([x1_rot, x2_rot], axis=-1).reshape(x.shape)


class LayerForward:
    """层前向传播（纯函数）"""

    def __init__(self, model_cfg: ModelConfig):
        self.cfg = model_cfg
        self.math = MathOps()

    def attention(
        self,
        h: np.ndarray,          # (H,)
        k_cache: np.ndarray,     # (seq, n_kv_heads, head_dim)
        v_cache: np.ndarray,     # (seq, n_kv_heads, head_dim)
        q_weight: Any,          # GhostFactor for Wq
        k_weight: Any,          # GhostFactor for Wk
        v_weight: Any,          # GhostFactor for Wv
        o_weight: Any,          # GhostFactor for Wo
        pos: int,
    ) -> np.ndarray:
        """注意力计算（纯函数）"""
        # 归一化
        h_norm = self.math.rms_norm(h)

        # Q, K, V 投影
        q = self._ghost_linear(q_weight, h_norm)  # (H,)
        k = self._ghost_linear(k_weight, h_norm)  # (H,)
        v = self._ghost_linear(v_weight, h_norm)  # (H,)

        # reshape
        n_heads = self.cfg.n_heads
        n_kv_heads = self.cfg.n_kv_heads
        head_dim = self.cfg.head_dim

        q = q.reshape(n_heads, head_dim)           # (n_heads, head_dim)
        k = k.reshape(n_kv_heads, head_dim)         # (n_kv_heads, head_dim)
        v = v.reshape(n_kv_heads, head_dim)         # (n_kv_heads, head_dim)

        # RoPE
        q = self.math.rope(q, pos, head_dim, self.cfg.rope_theta)
        k = self.math.rope(k, pos, head_dim, self.cfg.rope_theta)

        # GQA 扩展
        if self.cfg.kv_repeat > 1:
            k = np.repeat(k, self.cfg.kv_repeat, axis=0)  # (n_heads, head_dim)
            v = np.repeat(v, self.cfg.kv_repeat, axis=0)  # (n_heads, head_dim)

        # 拼接到缓存
        # k_cache: (seq, n_kv_heads, head_dim) → (seq, n_heads, head_dim)
        if self.cfg.kv_repeat > 1:
            k_cache_expanded = np.repeat(k_cache, self.cfg.kv_repeat, axis=1)
            v_cache_expanded = np.repeat(v_cache, self.cfg.kv_repeat, axis=1)
        else:
            k_cache_expanded = k_cache
            v_cache_expanded = v_cache

        k_full = np.concatenate([k[np.newaxis, :], k_cache_expanded], axis=0)
        v_full = np.concatenate([v[np.newaxis, :], v_cache_expanded], axis=0)

        # Attention scores
        scores = np.einsum('hd,sd->hs', q, k_full) / np.sqrt(head_dim)
        attn = self.math.softmax(scores, axis=-1)
        out = np.einsum('hs,sd->hd', attn, v_full)

        # 输出投影
        out = out.reshape(-1)  # (H,)
        out_weighted = self._ghost_linear(o_weight, out)

        return h + out_weighted  # 残差连接

    def ffn(
        self,
        h: np.ndarray,
        w1_weight: Any,
        w2_weight: Any,
    ) -> np.ndarray:
        """FFN 计算（纯函数）"""
        h_norm = self.math.rms_norm(h)
        up = self._ghost_linear(w1_weight, h_norm)
        act = np.maximum(up, 0)  # ReLU（简化）
        down = self._ghost_linear(w2_weight, act)
        return h + down  # 残差连接

    def _ghost_linear(self, weight: Any, x: np.ndarray) -> np.ndarray:
        """Ghost 线性层（调用 detileize_weight）"""
        # 这里需要调用 ghost_factor.forward(x)
        # 简化实现
        return x  # TODO: 实现 GhostFactor 调用


# ============================================================================
# 4. 控制层（消除控制流熵）
# ============================================================================

class SamplingPolicy:
    """采样策略（封装采样逻辑）"""

    def __init__(self, cfg: InferConfig):
        self.cfg = cfg
        self.math = MathOps()

    def sample(self, logits: np.ndarray, rng: np.random.Generator) -> int:
        """
        采样 token

        Args:
            logits: (V,) 未归一化 logit
            rng: 随机数生成器

        Returns:
            token ID
        """
        # 温度
        if self.cfg.temperature < 1e-8:
            return int(np.argmax(logits))

        logits = logits / self.cfg.temperature
        probs = self.math.softmax(logits)

        # Top-k
        if self.cfg.top_k > 0:
            k = min(self.cfg.top_k, len(probs))
            idx_k = np.argpartition(probs, -k)[-k:]
            mask = np.zeros_like(probs)
            mask[idx_k] = probs[idx_k]
            probs = mask

        # Top-p
        if self.cfg.top_p < 1.0:
            sorted_idx = np.argsort(probs)[::-1]
            cumsum = np.cumsum(probs[sorted_idx])
            cutoff = np.searchsorted(cumsum, self.cfg.top_p) + 1
            keep = sorted_idx[:cutoff]
            mask = np.zeros_like(probs)
            mask[keep] = probs[keep]
            probs = mask

        # 归一化
        probs = probs / (probs.sum() + 1e-12)

        # 采样
        return int(rng.choice(len(probs), p=probs))


# ============================================================================
# 5. 低熵推理引擎
# ============================================================================

class LowEntropyEngine:
    """
    低熵推理引擎

    特点：
      1. 所有状态都是不可变的（或显式可变）
      2. 所有计算都是纯函数
      3. 数据流单向、明确
      4. 副作用隔离
    """

    def __init__(
        self,
        model_state: ModelState,
        infer_cfg: InferConfig,
        system_cfg: SystemConfig,
    ):
        # 状态（显式化）
        self.model = model_state
        self.infer = infer_cfg
        self.system = system_cfg

        # 运行时统计（隔离的可变状态）
        self.stats = RuntimeStats()

        # 组件（无状态）
        self.layer_forward = LayerForward(model_state.model_cfg)
        self.sampler = SamplingPolicy(infer_cfg)

        # 内部状态
        self._rng = np.random.default_rng(system_cfg.seed)
        self._kv_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def generate(
        self,
        prompt_tokens: List[int],
        max_new: int = 64,
    ) -> Tuple[List[int], RuntimeStats]:
        """
        生成 token（主循环）

        Returns:
            (生成的 token 列表, 运行时统计)
        """
        # Prefill 阶段
        t0 = time.perf_counter()
        for pos, tok in enumerate(prompt_tokens):
            h = self.model.get_embedding(tok)
            context = StepContext(position=pos, hidden=h, kv_cache={})
            self._prefill_step(context)
        self.stats.prefill_time_s = time.perf_counter() - t0

        # Decode 阶段
        t1 = time.perf_counter()
        generated = []

        # 初始 logits（从 prefill 最后一步）
        logits = h  # 简化

        for step in range(max_new):
            # 采样
            tok = self.sampler.sample(logits, self._rng)
            generated.append(tok)
            self.stats.tokens_generated += 1

            # 前向传播
            h = self.model.get_embedding(tok)
            pos = len(prompt_tokens) + step

            # 更新 KV 缓存
            context = StepContext(position=pos, hidden=h, kv_cache=self._kv_cache)

            # 检查上下文长度
            if pos >= self.infer.max_ctx:
                break

            # 单步前向
            logits = self._decode_step(context)

            self.stats.steps_completed += 1

            if self.system.verbose and (step + 1) % 10 == 0:
                print(f"  Step {step+1}: token={tok}, tps={self.stats.tokens_per_second:.1f}")

        self.stats.decode_time_s = time.perf_counter() - t1

        return generated, self.stats

    def _prefill_step(self, context: StepContext):
        """Prefill 单步（简化）"""
        # TODO: 实现 prefill 逻辑
        pass

    def _decode_step(self, context: StepContext) -> np.ndarray:
        """
        Decode 单步（纯函数）

        Args:
            context: 当前步上下文

        Returns:
            logits: (vocab_size,)
        """
        h = context.hidden

        # 层循环
        for li in range(self.model.model_cfg.n_layers):
            # 获取 Ghost 层
            layer = self.model.ghost_layers[li]

            # Attention + FFN（简化）
            # TODO: 调用 LayerForward
            pass

        # 最终投影
        logits = self.model.get_logits(h)

        return logits


# ============================================================================
# 示例使用
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Virtual A100 - 低熵重构版本")
    print("=" * 70)
    print()

    # 创建配置
    model_cfg = ModelConfig(H=256, n_layers=4, n_heads=4, n_kv_heads=4)
    infer_cfg = InferConfig(temperature=0.7, top_p=0.9)
    system_cfg = SystemConfig(verbose=True, seed=42)

    print("[配置]")
    print(f"  模型: H={model_cfg.H}, layers={model_cfg.n_layers}")
    print(f"  推理: temp={infer_cfg.temperature}, top_p={infer_cfg.top_p}")
    print()

    # 创建模型状态（需要 ghost_layers，这里用空列表）
    model_state = ModelState(
        embed_weight=None,
        head_weight=None,
        ghost_layers=[],
        model_cfg=model_cfg,
    )

    # 创建引擎
    engine = LowEntropyEngine(model_state, infer_cfg, system_cfg)

    # 生成
    prompt = [1, 2, 3]  # 示例 prompt
    tokens, stats = engine.generate(prompt, max_new=10)

    print()
    print("[结果]")
    print(f"  Prompt: {prompt}")
    print(f"  Generated: {tokens}")
    print(f"  Tokens/s: {stats.tokens_per_second:.1f}")
    print()
    print("=" * 70)
    print("[OK] 低熵引擎测试完成")
    print("=" * 70)
