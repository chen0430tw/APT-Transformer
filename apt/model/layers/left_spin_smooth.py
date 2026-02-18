#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
左旋平滑模块 (Left-Spin Smooth)

替换传统泰勒展开的"线性外推"，改用"单向缓冲+尖点规避"机制。

核心思想：
- 传统：u' = u + Δu  (遇尖点会炸)
- 左旋：u' = u + g(φ)·Δu  (遇尖点自动缩小步长，方向不变)

作者: claude + chen0430tw
版本: 1.0 (Real-valued Left-Spin for Transformers)
日期: 2026-01-21
"""

from apt.core.fake_torch import get_torch
torch = get_torch()
from apt.core.fake_torch import get_torch
torch = get_torch()
nn = torch.nn
F = torch.nn.functional
from typing import Optional, Tuple, Dict


class LeftSpinStep(nn.Module):
    """
    左旋平滑步进器（实数版）

    替换 u' = u + Δu 为 u' = u + g(φ)·Δu
    其中 φ 由尖点强度 s 计算得出

    参数:
        alpha: 缓冲强度系数（控制 φ 的缩放）
        tau: 尖点阈值（s > τ 时才启用缓冲）
        beta: 惯性系数（滞后平滑，防止抖动）
        w1: 一阶变化强度权重
        w2: 二阶加速度权重
        gate_type: 门控函数类型 ('cosine' | 'normalized')
        eps: 数值稳定性小量
    """

    def __init__(
        self,
        alpha: float = 0.5,
        tau: float = 0.3,
        beta: float = 0.7,
        w1: float = 1.0,
        w2: float = 0.5,
        gate_type: str = 'normalized',  # 'cosine' 或 'normalized'
        eps: float = 1e-8
    ):
        super().__init__()

        # 可学习参数（可选：固定值或可训练）
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self.tau = nn.Parameter(torch.tensor(tau), requires_grad=False)
        self.beta = beta
        self.w1 = w1
        self.w2 = w2
        self.gate_type = gate_type
        self.eps = eps

        # 缓冲角历史（用于惯性平滑）
        # 训练中会被动态 resize 到 [batch, seq_len]，
        # 恢复训练时通过 _load_from_state_dict 自动处理 shape 差异
        self.register_buffer('phi_prev', torch.tensor(0.0))
        # 上一次的增量（用于计算二阶导数/加速度）
        self.register_buffer('delta_prev', None)

    def compute_spike_strength(
        self,
        u: torch.Tensor,
        delta_u: torch.Tensor,
        delta_prev: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算尖点强度 s

        s = w1 * d + w2 * a
        其中:
            d = ||Δu|| / (ε + ||u||)  一阶变化强度
            a = ||Δu - Δu_prev|| / (ε + ||Δu|| + ||Δu_prev||)  二阶加速度

        参数:
            u: 当前状态 [*, d_model]
            delta_u: 当前增量 [*, d_model]
            delta_prev: 上一次增量 [*, d_model] (可选)

        返回:
            s: 尖点强度 [*]
        """
        # 一阶：相对变化强度
        norm_u = torch.norm(u, p=2, dim=-1, keepdim=False)
        norm_delta = torch.norm(delta_u, p=2, dim=-1, keepdim=False)
        d = norm_delta / (self.eps + norm_u)

        # 二阶：加速度（曲率近似）
        if delta_prev is not None and delta_prev.shape == delta_u.shape:
            norm_delta_prev = torch.norm(delta_prev, p=2, dim=-1, keepdim=False)
            delta_diff = delta_u - delta_prev
            norm_delta_diff = torch.norm(delta_diff, p=2, dim=-1, keepdim=False)
            a = norm_delta_diff / (self.eps + norm_delta + norm_delta_prev)
        else:
            a = torch.zeros_like(d)

        # 组合
        s = self.w1 * d + self.w2 * a

        return s

    def compute_buffer_angle(self, s: torch.Tensor) -> torch.Tensor:
        """
        计算缓冲角 φ

        φ = α · softplus(s - τ)
        并应用惯性平滑: φ_t = (1-β)φ_{t-1} + βφ_t

        参数:
            s: 尖点强度 [*]

        返回:
            phi: 缓冲角 [*]  (保证 ≥ 0)
        """
        # softplus(s - τ) 当 s > τ 时快速增长，s < τ 时接近0
        phi_raw = self.alpha * F.softplus(s - self.tau)

        # 惯性平滑（使用全局 phi_prev，适用于 batch 维度）
        if self.phi_prev.numel() == 1 or self.phi_prev.shape != phi_raw.shape:
            # 初始化或重新初始化为与 phi_raw 相同形状
            # 处理变长序列：当序列长度变化时重新初始化
            self.phi_prev = torch.zeros_like(phi_raw)

        phi = (1 - self.beta) * self.phi_prev + self.beta * phi_raw

        # 更新历史
        self.phi_prev = phi.detach()

        # 确保非负
        phi = torch.clamp(phi, min=0.0)

        return phi

    def apply_gate(self, phi: torch.Tensor) -> torch.Tensor:
        """
        门控函数 g(φ)

        两种实现:
        A. 余弦门控：g(φ) = cos(φ)
        B. 归一化门控：g(φ) = 1 / √(1 + φ²)

        参数:
            phi: 缓冲角 [*]

        返回:
            g: 门控系数 [*]  范围 (0, 1]
        """
        if self.gate_type == 'cosine':
            # 余弦门控
            g = torch.cos(phi)
            # 限制范围防止负值
            g = torch.clamp(g, min=0.0, max=1.0)
        elif self.gate_type == 'normalized':
            # 归一化门控（更稳定，不会完全归零）
            g = 1.0 / torch.sqrt(1.0 + phi ** 2)
        else:
            raise ValueError(f"Unknown gate_type: {self.gate_type}")

        return g

    def forward(
        self,
        u: torch.Tensor,
        delta_u: torch.Tensor,
        use_smooth: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        左旋平滑前向传播

        参数:
            u: 当前状态 [batch, seq_len, d_model]
            delta_u: 残差增量 [batch, seq_len, d_model]
            use_smooth: 是否启用左旋平滑（False则退化为标准残差）

        返回:
            u_next: 更新后的状态 [batch, seq_len, d_model]
            stats: 统计信息字典
        """
        if not use_smooth:
            # 退化为标准残差连接
            return u + delta_u, {
                'spike_strength': torch.tensor(0.0),
                'buffer_angle': torch.tensor(0.0),
                'gate_value': torch.tensor(1.0),
                'smoothed': False
            }

        # 1. 计算尖点强度
        s = self.compute_spike_strength(u, delta_u, self.delta_prev)

        # 2. 计算缓冲角
        phi = self.compute_buffer_angle(s)

        # 3. 应用门控
        g = self.apply_gate(phi)

        # 4. 左旋平滑步进
        # g: [batch, seq_len] -> [batch, seq_len, 1] 以匹配 delta_u 维度
        g_expanded = g.unsqueeze(-1)
        delta_u_eff = g_expanded * delta_u
        u_next = u + delta_u_eff

        # 5. 更新历史
        self.delta_prev = delta_u.detach()

        # 6. 统计信息
        stats = {
            'spike_strength': s.mean().item(),
            'buffer_angle': phi.mean().item(),
            'gate_value': g.mean().item(),
            'smoothed': True
        }

        return u_next, stats

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """处理 phi_prev / delta_prev 的 shape 差异

        这两个 buffer 在训练中会被动态 resize (从 scalar 扩展为 [batch, seq_len])，
        checkpoint 保存时带着训练时的 shape。加载到新模型时 shape 可能不同
        （不同 batch_size 或序列长度），此时重置为 scalar 而不是报错。
        同理 delta_prev 可能是 None 也可能是 [batch, seq_len, d_model]。

        类似自生成注意力 (auto_u/auto_v) 的外挂层模式：
        模块自身负责兼容性，不需要外部 strip。
        """
        phi_key = prefix + 'phi_prev'
        delta_key = prefix + 'delta_prev'

        # phi_prev: 如果 shape 不匹配，重置为 scalar（首次 forward 会重新 resize）
        if phi_key in state_dict:
            saved = state_dict[phi_key]
            if saved.shape != self.phi_prev.shape:
                # shape 不同说明 batch/seq 变了，惯性记忆失效，重置
                state_dict[phi_key] = torch.tensor(0.0, dtype=saved.dtype)

        # delta_prev: 如果目标是 None 但 checkpoint 里有值，跳过
        if delta_key in state_dict:
            saved = state_dict[delta_key]
            if self.delta_prev is None and saved is not None:
                # 当前模型 delta_prev=None，checkpoint 里有训练时的值
                # 直接丢弃 — 新的 forward 会重新计算
                del state_dict[delta_key]

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )

    def reset_state(self):
        """重置内部状态（用于新序列开始）"""
        self.phi_prev = torch.tensor(0.0, device=self.phi_prev.device)
        self.delta_prev = None


class LeftSpinResidual(nn.Module):
    """
    左旋平滑残差连接层

    封装标准的残差连接，替换为左旋平滑版本
    使用方式:
        # 原始: h = h + dropout(sublayer(h))
        # 左旋: h = left_spin_residual(h, dropout(sublayer(h)))
    """

    def __init__(
        self,
        alpha: float = 0.5,
        tau: float = 0.3,
        beta: float = 0.7,
        gate_type: str = 'normalized',
        adaptive: bool = True
    ):
        """
        参数:
            alpha, tau, beta: LeftSpinStep 参数
            gate_type: 门控类型
            adaptive: 是否根据训练/推理自动调整
        """
        super().__init__()

        self.left_spin = LeftSpinStep(
            alpha=alpha,
            tau=tau,
            beta=beta,
            gate_type=gate_type
        )
        self.adaptive = adaptive

        # 统计计数器（用于监控）
        self.register_buffer('total_steps', torch.tensor(0))
        self.register_buffer('smoothed_steps', torch.tensor(0))

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor
    ) -> torch.Tensor:
        """
        参数:
            x: 原始输入（残差基准）
            residual: 子层输出（残差增量）

        返回:
            output: 左旋平滑后的输出
        """
        # 自适应控制：训练时启用，推理时可选
        use_smooth = True
        if self.adaptive and not self.training:
            # 推理时可以禁用（减少计算）
            use_smooth = False

        output, stats = self.left_spin(x, residual, use_smooth=use_smooth)

        # 更新统计
        self.total_steps += 1
        if stats['smoothed']:
            self.smoothed_steps += 1

        return output

    def get_smooth_ratio(self) -> float:
        """获取平滑应用比例"""
        if self.total_steps == 0:
            return 0.0
        return (self.smoothed_steps / self.total_steps).item()


def replace_residual_with_leftspin(
    original_output: torch.Tensor,
    residual: torch.Tensor,
    left_spin_layer: Optional[LeftSpinResidual] = None,
    **left_spin_kwargs
) -> torch.Tensor:
    """
    便捷函数：替换标准残差连接为左旋版本

    使用示例:
        # 原始代码
        x = x + dropout(sublayer(x))

        # 替换为
        x = replace_residual_with_leftspin(x, dropout(sublayer(x)))

    参数:
        original_output: 残差基准（原始 x）
        residual: 残差增量（子层输出）
        left_spin_layer: 已初始化的 LeftSpinResidual 层（可选）
        **left_spin_kwargs: LeftSpinResidual 初始化参数

    返回:
        output: 左旋平滑后的输出
    """
    if left_spin_layer is None:
        # 创建临时层
        left_spin_layer = LeftSpinResidual(**left_spin_kwargs)
        left_spin_layer.to(original_output.device)

    return left_spin_layer(original_output, residual)


# ============================================================================
# 可视化和诊断工具
# ============================================================================

class LeftSpinMonitor:
    """
    左旋平滑监控器

    用于跟踪和可视化左旋平滑的效果
    """

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval
        self.step_count = 0
        self.history = {
            'spike_strength': [],
            'buffer_angle': [],
            'gate_value': []
        }

    def log_stats(self, stats: Dict[str, float]):
        """记录统计信息"""
        self.step_count += 1

        for key in ['spike_strength', 'buffer_angle', 'gate_value']:
            if key in stats:
                self.history[key].append(stats[key])

        # 定期打印
        if self.step_count % self.log_interval == 0:
            self.print_summary()

    def print_summary(self):
        """打印统计摘要"""
        print(f"\n[LeftSpin Monitor] Step {self.step_count}")
        for key, values in self.history.items():
            if values:
                avg = sum(values[-self.log_interval:]) / min(len(values), self.log_interval)
                print(f"  {key}: {avg:.4f}")

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """获取完整统计"""
        import numpy as np

        stats = {}
        for key, values in self.history.items():
            if values:
                stats[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }

        return stats


# ============================================================================
# 实验性：可学习的左旋参数
# ============================================================================

class AdaptiveLeftSpinStep(LeftSpinStep):
    """
    自适应左旋平滑（可学习参数版本）

    将 alpha, tau 设为可学习参数，让模型自动调整缓冲强度
    """

    def __init__(
        self,
        alpha: float = 0.5,
        tau: float = 0.3,
        beta: float = 0.7,
        w1: float = 1.0,
        w2: float = 0.5,
        gate_type: str = 'normalized',
        eps: float = 1e-8,
        learnable: bool = True
    ):
        super().__init__(alpha, tau, beta, w1, w2, gate_type, eps)

        # 将参数设为可学习
        if learnable:
            self.alpha.requires_grad = True
            self.tau.requires_grad = True

            # 添加约束（通过 sigmoid 映射到合理范围）
            self.register_buffer('alpha_min', torch.tensor(0.1))
            self.register_buffer('alpha_max', torch.tensor(2.0))
            self.register_buffer('tau_min', torch.tensor(0.1))
            self.register_buffer('tau_max', torch.tensor(1.0))

    def compute_buffer_angle(self, s: torch.Tensor) -> torch.Tensor:
        """重写以支持参数约束"""
        # 约束 alpha 和 tau 到合理范围
        alpha_constrained = torch.sigmoid(self.alpha) * (self.alpha_max - self.alpha_min) + self.alpha_min
        tau_constrained = torch.sigmoid(self.tau) * (self.tau_max - self.tau_min) + self.tau_min

        phi_raw = alpha_constrained * F.softplus(s - tau_constrained)

        if self.phi_prev.numel() == 1:
            self.phi_prev = torch.zeros_like(phi_raw)

        phi = (1 - self.beta) * self.phi_prev + self.beta * phi_raw
        self.phi_prev = phi.detach()

        return torch.clamp(phi, min=0.0)


# ============================================================================
# 导出接口
# ============================================================================

__all__ = [
    'LeftSpinStep',
    'LeftSpinResidual',
    'replace_residual_with_leftspin',
    'LeftSpinMonitor',
    'AdaptiveLeftSpinStep'
]
