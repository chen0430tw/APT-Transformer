#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LECaC Soft Warmup Scheduler
============================
解决LECaC量化在低学习率warmup期间的梯度不稳定问题

核心机制：
1. Alpha Compensation Warmup - 动态调整补偿强度
2. Progressive Bits Warmup - 渐进降低量化精度（可选）

参考文献：
- Progressive Quantization: https://www.emergentmind.com/topics/progressive-quantization
- Soft-then-Hard Quantization: http://proceedings.mlr.press/v139/guo21c/guo21c.pdf
- Annealing: https://aclanthology.org/2021.eacl-main.212.pdf
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch.nn as nn

# 自然均衡常数（LECaC默认值）
_ALPHA_4_OVER_E: float = 4.0 / math.e  # ≈ 1.4715


@dataclass
class LECACWarmupConfig:
    """LECaC Warmup配置"""
    warmup_steps: int = 100                    # Warmup步数（与学习率warmup对齐）
    base_alpha: float = _ALPHA_4_OVER_E        # 基础补偿强度（默认1.47）
    warmup_alpha_multiplier: float = 3.0       # Warmup期间alpha倍数（1.47 → 4.41）
    progressive_bits: bool = False             # 是否启用渐进bits warmup
    target_bits: int = 2                       # 目标量化精度
    warmup_schedule: str = "linear"            # Warmup调度类型："linear", "cosine", "exponential"


class LECACAlphaScheduler:
    """
    LECaC Alpha补偿强度调度器

    在训练初期增强补偿强度，对抗低学习率下的量化噪声

    Example:
        scheduler = LECACAlphaScheduler(warmup_steps=100)

        for step in range(1000):
            # 获取当前alpha
            current_alpha = scheduler.get_alpha(step)

            # 更新所有LECaCLinear层的alpha
            for module in model.modules():
                if isinstance(module, LECACLinear):
                    module.alpha = current_alpha

            # 正常训练
            loss.backward()
            optimizer.step()
    """

    def __init__(
        self,
        warmup_steps: int = 100,
        base_alpha: float = _ALPHA_4_OVER_E,
        warmup_multiplier: float = 3.0,
        schedule: str = "linear"
    ):
        """
        Args:
            warmup_steps: Warmup步数
            base_alpha: 基础alpha（训练稳定后的值）
            warmup_multiplier: Warmup期间的倍数（建议2-4x）
            schedule: 调度类型 ["linear", "cosine", "exponential"]
        """
        self.warmup_steps = warmup_steps
        self.base_alpha = base_alpha
        self.warmup_multiplier = warmup_multiplier
        self.schedule = schedule

        # 计算warmup初始alpha
        self.warmup_start_alpha = base_alpha * warmup_multiplier

        print(f"[LECaC Warmup] Alpha调度器初始化")
        print(f"  - Warmup步数: {warmup_steps}")
        print(f"  - Alpha范围: {self.warmup_start_alpha:.4f} → {base_alpha:.4f}")
        print(f"  - 调度类型: {schedule}")

    def get_alpha(self, step: int) -> float:
        """
        获取当前step的alpha值

        Args:
            step: 当前训练步（0-based）

        Returns:
            当前alpha值
        """
        if step >= self.warmup_steps:
            return self.base_alpha

        # Warmup进度 [0, 1]
        progress = step / self.warmup_steps

        # 根据调度类型计算alpha
        if self.schedule == "linear":
            # 线性衰减：warmup_start_alpha → base_alpha
            alpha = self.warmup_start_alpha - (self.warmup_start_alpha - self.base_alpha) * progress

        elif self.schedule == "cosine":
            # Cosine退火：平滑过渡
            alpha = self.base_alpha + (self.warmup_start_alpha - self.base_alpha) * 0.5 * (1 + math.cos(math.pi * progress))

        elif self.schedule == "exponential":
            # 指数衰减：快速降低
            decay_rate = math.log(self.warmup_multiplier) / self.warmup_steps
            alpha = self.base_alpha * math.exp(decay_rate * (self.warmup_steps - step))

        else:
            raise ValueError(f"未知的调度类型: {self.schedule}")

        return alpha

    def state_dict(self):
        """保存调度器状态"""
        return {
            "warmup_steps": self.warmup_steps,
            "base_alpha": self.base_alpha,
            "warmup_multiplier": self.warmup_multiplier,
            "schedule": self.schedule,
        }

    def load_state_dict(self, state_dict):
        """加载调度器状态"""
        self.warmup_steps = state_dict["warmup_steps"]
        self.base_alpha = state_dict["base_alpha"]
        self.warmup_multiplier = state_dict["warmup_multiplier"]
        self.schedule = state_dict["schedule"]
        self.warmup_start_alpha = self.base_alpha * self.warmup_multiplier


class LECACBitsScheduler:
    """
    LECaC Bits渐进调度器（实验性）

    逐步降低量化精度：8-bit → 4-bit → 2-bit

    ⚠️ 注意：需要模型支持动态bits切换
    """

    def __init__(
        self,
        warmup_steps: int = 300,
        target_bits: int = 2
    ):
        """
        Args:
            warmup_steps: Warmup总步数
            target_bits: 目标量化精度（2或4）
        """
        self.warmup_steps = warmup_steps
        self.target_bits = target_bits

        # 阶段划分：[0, 1/3) → 8-bit, [1/3, 2/3) → 4-bit, [2/3, end) → target_bits
        self.phase1_end = warmup_steps // 3
        self.phase2_end = 2 * warmup_steps // 3

        print(f"[LECaC Warmup] Bits渐进调度器初始化")
        print(f"  - Phase 1 (8-bit): steps 0-{self.phase1_end}")
        print(f"  - Phase 2 (4-bit): steps {self.phase1_end}-{self.phase2_end}")
        print(f"  - Phase 3 ({target_bits}-bit): steps {self.phase2_end}+")

    def get_bits(self, step: int) -> int:
        """
        获取当前step的量化精度

        Args:
            step: 当前训练步

        Returns:
            当前bits值 (2, 4, or 8)
        """
        if step < self.phase1_end:
            return 8
        elif step < self.phase2_end:
            return 4
        else:
            return self.target_bits


def update_lecac_alpha(model: nn.Module, new_alpha: float) -> int:
    """
    更新模型中所有LECaCLinear层的alpha值

    Args:
        model: 包含LECaCLinear层的模型
        new_alpha: 新的alpha值

    Returns:
        更新的层数
    """
    from apt.vgpu.runtime.lecac import LECACLinear

    count = 0
    for module in model.modules():
        if isinstance(module, LECACLinear):
            module.alpha = new_alpha
            count += 1

    return count


class SoftQuantizationScheduler:
    """
    Soft Quantization with Temperature Annealing（备用方案）

    使用温度参数控制量化的"软硬"程度：
    - 高温（τ >> 1）: 软量化，接近原始值，保持梯度流
    - 低温（τ → 0）: 硬量化，接近离散舍入

    参考：Soft-then-Hard Quantization
    http://proceedings.mlr.press/v139/guo21c/guo21c.pdf
    """

    def __init__(
        self,
        warmup_steps: int = 100,
        start_temperature: float = 10.0,
        end_temperature: float = 0.01,
        schedule: str = "exponential"
    ):
        """
        Args:
            warmup_steps: Warmup步数
            start_temperature: 初始温度（高温→软量化）
            end_temperature: 最终温度（低温→硬量化）
            schedule: 退火策略 ["linear", "exponential", "cosine"]
        """
        self.warmup_steps = warmup_steps
        self.start_temp = start_temperature
        self.end_temp = end_temperature
        self.schedule = schedule

        print(f"[LECaC Warmup] Soft Quantization调度器初始化（备用）")
        print(f"  - 温度范围: {start_temperature:.2f} → {end_temperature:.4f}")
        print(f"  - 退火策略: {schedule}")

    def get_temperature(self, step: int) -> float:
        """
        获取当前step的温度值

        Args:
            step: 当前训练步

        Returns:
            当前温度τ
        """
        if step >= self.warmup_steps:
            return self.end_temp

        progress = step / self.warmup_steps

        if self.schedule == "linear":
            temp = self.start_temp - (self.start_temp - self.end_temp) * progress

        elif self.schedule == "exponential":
            # τ(t) = τ_end * (τ_start / τ_end)^(1 - progress)
            ratio = self.start_temp / self.end_temp
            temp = self.end_temp * (ratio ** (1.0 - progress))

        elif self.schedule == "cosine":
            temp = self.end_temp + (self.start_temp - self.end_temp) * 0.5 * (1 + math.cos(math.pi * progress))

        else:
            raise ValueError(f"未知的退火策略: {self.schedule}")

        return temp


def soft_quantize_int2_with_temperature(x, temperature: float = 1.0):
    """
    带温度参数的软量化（INT2）- 备用接口

    使用tanh近似round操作，temperature控制"软硬"程度

    Args:
        x: 输入tensor (FP32/BF16)
        temperature: 温度参数（τ > 1软量化，τ → 0硬量化）

    Returns:
        (quantized_tensor, scale): INT8存储的INT2 + 缩放因子

    Example:
        # Warmup初期（τ=10）：软量化
        x_q, scale = soft_quantize_int2_with_temperature(x, temperature=10.0)

        # Warmup后期（τ=0.01）：接近硬量化
        x_q, scale = soft_quantize_int2_with_temperature(x, temperature=0.01)
    """
    import torch

    with torch.no_grad():
        # 计算scale
        amax = x.abs().max()
        scale = torch.clamp(amax / 1.0, min=1e-6)
        x_scaled = x / scale

        # 软舍入 vs 硬舍入
        if temperature > 0.1:
            # 软舍入：tanh近似round
            # round(x) ≈ x + tanh(x/τ) - x * tanh(1/τ)
            x_soft_round = x_scaled + torch.tanh(x_scaled / temperature) - x_scaled * math.tanh(1.0 / temperature)
        else:
            # 硬舍入
            x_soft_round = x_scaled.round()

        # Clamp到INT2范围 [-2, 1]
        x_quantized = x_soft_round.clamp(-2, 1).to(torch.int8)

    return x_quantized, scale


def soft_quantize_int4_with_temperature(x, temperature: float = 1.0):
    """
    带温度参数的软量化（INT4）- 备用接口

    Args:
        x: 输入tensor
        temperature: 温度参数

    Returns:
        (quantized_tensor, scale): INT8存储的INT4 + 缩放因子
    """
    import torch

    with torch.no_grad():
        amax = x.abs().max()
        scale = torch.clamp(amax / 7.0, min=1e-6)
        x_scaled = x / scale

        if temperature > 0.1:
            x_soft_round = x_scaled + torch.tanh(x_scaled / temperature) - x_scaled * math.tanh(1.0 / temperature)
        else:
            x_soft_round = x_scaled.round()

        x_quantized = x_soft_round.clamp(-8, 7).to(torch.int8)

    return x_quantized, scale


# ============================================================================
# 公共导出
# ============================================================================

__all__ = [
    "LECACWarmupConfig",
    "LECACAlphaScheduler",
    "LECACBitsScheduler",
    "SoftQuantizationScheduler",
    "update_lecac_alpha",
    "soft_quantize_int2_with_temperature",
    "soft_quantize_int4_with_temperature",
    "_ALPHA_4_OVER_E",
]
