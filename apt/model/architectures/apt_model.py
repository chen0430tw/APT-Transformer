#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT核心模型集成实现
集成自生成注意力机制(Autopoietic Attention)到APT模型框架
集成DBC-DAC压缩优化，提高训练稳定性
"""

import os
# 检查是否应该屏蔽自创生变换器的警告
SUPPRESS_APT_WARNINGS = os.environ.get('SUPPRESS_APT_WARNINGS', 'False').lower() in ('true', '1', 'yes')
from apt.core.fake_torch import get_torch
torch = get_torch()
nn = torch.nn
F = torch.nn.functional
import math
import warnings
import sys
from typing import Optional, Tuple, List, Dict, Union

# 导入左旋平滑模块
from apt.model.layers.left_spin_smooth import (
    LeftSpinStep,
    LeftSpinResidual,
    AdaptiveLeftSpinStep
)


class DBCDAC_Optimizer:
    """
    维度平衡压缩法(DBC)与维度伴随补偿法(DAC)结合的优化器
    用于APT模型训练过程中的梯度稳定和参数优化
    """
    
    def __init__(self, rank_ratio_proj=0.1, rank_ratio_res=0.05, 
                 threshold=1e-6, iterations=1, use_quantization=False,
                 quant_bits=8, apply_to_gradients=True):
        """
        初始化DBC-DAC优化器
        
        参数:
            rank_ratio_proj: float, 初始低秩正交投影的比例
            rank_ratio_res: float, 残差补偿时的秩比率
            threshold: float, 维度平衡矩阵的数值稳定性阈值
            iterations: int, 残差补偿的迭代次数
            use_quantization: bool, 是否使用量化
            quant_bits: int, 量化位数
            apply_to_gradients: bool, 是否应用于梯度稳定
        """
        self.rank_ratio_proj = rank_ratio_proj
        self.rank_ratio_res = rank_ratio_res
        self.threshold = threshold
        self.iterations = iterations
        self.use_quantization = use_quantization
        self.quant_bits = quant_bits
        self.apply_to_gradients = apply_to_gradients
        self.res_scale = 0.1  # 残差缩放因子
    
    def compute_balance_vector(self, W):
        """计算矩阵W每一行的和作为平衡向量，若绝对值低于threshold则置为1"""
        row_sums = W.sum(dim=1)
        D_vec = torch.where(
            row_sums.abs() > self.threshold, 
            row_sums, 
            torch.ones_like(row_sums) * self.threshold * torch.sign(row_sums)
        )
        # 处理零值情况
        D_vec = torch.where(row_sums == 0, torch.ones_like(row_sums) * self.threshold, D_vec)
        return D_vec

    def low_rank_approx(self, A, rank_ratio):
        """
        对矩阵A进行低秩近似，使用低熵导向原则优化 (进一步优化版)

        优化点:
        1. 自适应秩选择 - 根据能量分布动态调整
        2. 稀疏随机投影 - 投影复杂度从O(mnr)降到O(nnz·r)
        3. 早停机制 - 能量保留达标即停止
        4. 幂迭代加速 - 小秩情况下避免完整特征值分解

        复杂度: O(nnz·r + mr² + k²) where k << r
        """
        h, w = A.shape[-2], A.shape[-1]
        max_rank = int(min(h, w))
        r = int(max(1, min(max_rank-1, getattr(self, 'rank_ratio_proj', 0.1) * max_rank)))

        try:
            # 确保A是float32类型
            original_dtype = A.dtype
            if A.dtype == torch.float16 or A.dtype == torch.bfloat16:
                A = A.to(torch.float32)

            m, n = A.shape
            r_init = max(1, int(min(m, n) * rank_ratio))

            # 🚀 优化1: 自适应秩选择 - 根据Frobenius范数预估
            A_norm = torch.norm(A, 'fro')
            energy_threshold = 0.95  # 保留95%能量

            # 🚀 优化2: 快速密集随机投影 (GPU 优化)
            # GPU 讨厌稀疏内存访问，密集矩阵反而更快
            if m >= n:
                # 行数多，投影到列空间
                # 直接生成密集高斯随机矩阵
                Q = torch.randn(n, r_init, device=A.device, dtype=A.dtype)

                # 快速正交化 (QR 分解)
                Q, _ = torch.linalg.qr(Q)
                Y = A @ Q  # 投影，利用稀疏性

                # 协方差矩阵
                C = Y.T @ Y

                # 🚀 优化3: 自适应特征值计算
                # 如果r很小(<50)，用幂迭代；否则用eigh
                if r_init <= 50:
                    # 幂迭代法计算前k个特征值 (更快)
                    k = min(r_init, int(r_init * 0.8))  # 只计算80%
                    eigenvalues, eigenvectors = self._power_iteration(C, k)
                else:
                    # 完整特征值分解
                    eigenvalues, eigenvectors = torch.linalg.eigh(C)

                    # 🚀 优化4: 早停 - 只保留足够能量的特征值
                    eigenvalues_sorted, idx = torch.sort(eigenvalues, descending=True)
                    cumsum_energy = torch.cumsum(eigenvalues_sorted, dim=0)
                    total_energy = eigenvalues_sorted.sum()

                    # 找到保留95%能量所需的维度
                    k = torch.searchsorted(cumsum_energy, energy_threshold * total_energy).item() + 1
                    k = min(k, r_init)

                    # 只保留前k个
                    idx = idx[:k]
                    eigenvalues = eigenvalues[idx]
                    eigenvectors = eigenvectors[:, idx]

                # 重构低秩近似
                U_r = Y @ eigenvectors
                V_r = Q @ eigenvectors
                S_r = torch.diag(torch.sqrt(eigenvalues.clamp(min=0)))

                A_approx = U_r @ S_r @ V_r.T

            else:
                # 列数多，投影到行空间 (对称处理)
                # 直接生成密集高斯随机矩阵
                Q = torch.randn(m, r_init, device=A.device, dtype=A.dtype)

                Q, _ = torch.linalg.qr(Q)
                Y = A.T @ Q

                C = Y.T @ Y

                if r_init <= 50:
                    k = min(r_init, int(r_init * 0.8))
                    eigenvalues, eigenvectors = self._power_iteration(C, k)
                else:
                    eigenvalues, eigenvectors = torch.linalg.eigh(C)

                    eigenvalues_sorted, idx = torch.sort(eigenvalues, descending=True)
                    cumsum_energy = torch.cumsum(eigenvalues_sorted, dim=0)
                    total_energy = eigenvalues_sorted.sum()

                    k = torch.searchsorted(cumsum_energy, energy_threshold * total_energy).item() + 1
                    k = min(k, r_init)

                    idx = idx[:k]
                    eigenvalues = eigenvalues[idx]
                    eigenvectors = eigenvectors[:, idx]

                V_r = Y @ eigenvectors
                U_r = Q @ eigenvectors
                S_r = torch.diag(torch.sqrt(eigenvalues.clamp(min=0)))

                A_approx = U_r @ S_r @ V_r.T

            # 恢复原始数据类型
            if original_dtype == torch.float16:
                A_approx = A_approx.to(torch.float16)
            elif original_dtype == torch.bfloat16:
                A_approx = A_approx.to(torch.bfloat16)

            return A_approx, (U_r, S_r, V_r)

        except Exception as e:
            print(f"低秩近似计算错误: {e}")
            return A, (None, None, None)

    def _power_iteration(self, C, k, max_iter=20):
        """
        幂迭代法计算矩阵C的前k个特征值和特征向量

        复杂度: O(k²·iter) << O(k³)
        适用于小k的情况
        """
        n = C.shape[0]
        device = C.device
        dtype = C.dtype

        # 初始化随机向量
        V = torch.randn(n, k, device=device, dtype=dtype)
        V, _ = torch.linalg.qr(V)

        for _ in range(max_iter):
            # 幂迭代: V = C @ V
            V_new = C @ V

            # QR分解保持正交性
            V_new, R = torch.linalg.qr(V_new)

            # 检查收敛 (可选)
            if torch.allclose(V, V_new, atol=1e-5):
                break

            V = V_new

        # 计算特征值: λ = V^T C V
        eigenvalues = torch.diag(V.T @ C @ V)

        return eigenvalues, V


    # 同时，在stabilize_matrix方法中也需要添加类型转换:
    
    def stabilize_matrix(self, W):
        """
        使用DBC-DAC方法稳定矩阵，减少数值不稳定问题
        
        参数:
            W: torch.Tensor, 输入矩阵
            
        返回:
            W_stabilized: torch.Tensor, 稳定化后的矩阵
        """
        if not isinstance(W, torch.Tensor):
            W = torch.tensor(W, dtype=torch.float32)
        
        # 保存原始数据类型，以便最终恢复
        original_dtype = W.dtype
        
        # 如果是半精度，转为float32进行计算
        if W.dtype == torch.float16 or W.dtype == torch.bfloat16:
            W = W.to(torch.float32)
        
        # 检查输入是否有NaN或Inf
        if torch.isnan(W).any() or torch.isinf(W).any():
            W = torch.nan_to_num(W, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 维度平衡处理
        D_vec = self.compute_balance_vector(W)
        # D_inv = torch.diag(1.0 / D_vec)
        # W_norm = D_inv @ W  <-- 原始代码 (OOM 凶手)
        W_norm = (1.0 / D_vec).unsqueeze(1) * W
        
        # 低秩近似
        W_proj, _ = self.low_rank_approx(W_norm, self.rank_ratio_proj)
        
        # 可选的残差处理
        if self.iterations > 0:
            R = W_norm - W_proj
            W_res_total = torch.zeros_like(W_norm)
            
            for i in range(self.iterations):
                if torch.norm(R) < 1e-8:
                    break
                
                W_res, _ = self.low_rank_approx(R, self.rank_ratio_res)
                W_res_total = W_res_total + self.res_scale * W_res
                R = R - W_res
            
            W_norm_stabilized = W_proj + W_res_total
        else:
            W_norm_stabilized = W_proj
        
        # 应用维度平衡恢复
        W_stabilized = D_vec.unsqueeze(1) * W_norm_stabilized
        
        # 恢复原始数据类型
        W_stabilized = W_stabilized.to(original_dtype)

        return W_stabilized

    def stabilize_matrix_fast(self, W):
        """
        简化版矩阵稳定：减少迭代，移除冗余计算

        参数:
            W: torch.Tensor, 输入矩阵

        返回:
            W_stabilized: torch.Tensor, 稳定化后的矩阵
        """
        # 类型转换
        original_dtype = W.dtype
        if W.dtype in [torch.float16, torch.bfloat16]:
            W = W.to(torch.float32)

        # 维度平衡 (DBC)
        # 🚀 优化3: 移除阈值判断中的 item() 调用
        # 直接运算，不通过 Python if 检查
        row_sums = W.sum(dim=1, keepdim=True)
        # 🚀 修复: 处理零梯度的符号问题，防止 sign(0)=0 导致除零
        rs_sign = torch.sign(row_sums)
        rs_sign[rs_sign == 0] = 1.0  # 强制让 0 的符号为 1，避免乘积为 0
        # 避免除零的软阈值处理
        D_vec = rs_sign * torch.maximum(
            row_sums.abs(),
            torch.tensor(self.threshold, device=W.device, dtype=W.dtype)
        )
        W_norm = W / D_vec

        # 低秩近似 (DAC)
        # 🚀 优化4: 仅做一次投影 (One-pass)，不做残差迭代
        # 残差迭代(iterations>0)会让计算量翻倍，但在梯度平滑任务中收益递减
        W_proj, _ = self.low_rank_approx(W_norm, self.rank_ratio_proj)

        # 恢复
        W_stabilized = W_proj * D_vec
        return W_stabilized.to(original_dtype)

    def stabilize_gradients(self, grad):
        """
        极速版梯度稳定：随机触发 + 过滤小参数 + 基础清洗
        """
        if not isinstance(grad, torch.Tensor) or grad is None:
            return grad

        # 1. 🚀 修改点A：提高门槛到 150000
        # 只有非常大的矩阵才值得做分解，中等矩阵直接放行
        if grad.numel() < 150000:
             return torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 2. 大参数基础清洗
        grad = torch.nan_to_num(grad, nan=0.0, posinf=1.0, neginf=-1.0)

        # 3. 维度过滤
        if grad.ndim < 2:
            return grad

        # 🚀 修改点B：把 0.25 改为 0.05
        # 意思：只有 5% 的概率往下走，95% 的概率直接 return (跳过)
        import random
        if random.random() > 0.05: 
            return grad

        # --- 以下是昂贵的 DBC 计算 (现在很少触发了) ---
        original_shape = grad.shape

        if len(original_shape) == 2:
            stabilized_grad = self.stabilize_matrix_fast(grad)
        else:
            reshaped_grad = grad.reshape(original_shape[0], -1)
            stabilized_grad = self.stabilize_matrix_fast(reshaped_grad)
            stabilized_grad = stabilized_grad.reshape(original_shape)

        return stabilized_grad

def create_gradient_stabilizer_hook(dbc_dac_optimizer):
    """创建用于稳定梯度的钩子函数（已优化：无同步）"""
    def hook(grad):
        if grad is None:
            return None

        # 🚀 优化: 移除 CPU-GPU 同步检查
        # NaN/Inf 处理已经在 stabilize_gradients 中完成
        # 不再使用 if torch.isnan(grad).any()

        # 使用DBC-DAC优化器稳定梯度
        return dbc_dac_optimizer.stabilize_gradients(grad)

    return hook


def add_gradient_hooks_to_model(model, dbc_dac_optimizer):
    """
    为模型的所有参数添加梯度稳定钩子
    
    参数:
        model: APT模型实例
        dbc_dac_optimizer: DBCDAC_Optimizer实例
    """
    hooks = []
    
    # 为每个参数添加钩子
    for name, param in model.named_parameters():
        if param.requires_grad:
            hook = param.register_hook(create_gradient_stabilizer_hook(dbc_dac_optimizer))
            hooks.append(hook)
    
    return hooks


class RMSNorm(nn.Module):
    """RMSNorm (Root Mean Square LayerNorm), no mean subtraction."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        norm = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: (xW1) * silu(xW2) then W3."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w12 = nn.Linear(in_dim, hidden_dim * 2)
        self.w3 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        # silu = x * sigmoid(x)
        x = x1 * torch.sigmoid(x2)  # lightweight SiLU-ish gating (silu(x2) * x1 approximated)
        x = self.dropout(x)
        return self.w3(x)

# ---------------------------------------------------------------------------
# MoE (Mixture-of-Experts) 组件
# ---------------------------------------------------------------------------

class TopKRouter(nn.Module):
    """Top-K 门控路由器（Switch/ST-MoE 风格）

    职责:
      1. 将 token hidden state 映射到 num_experts 维 logits
      2. 选出 top_k 专家并返回归一化权重
      3. 计算 load-balancing aux loss + router z-loss

    参数:
        d_model:        隐藏维度
        num_experts:    专家总数
        top_k:          每个 token 选几个专家
        noisy_gating:   训练时是否加噪声探索
        noise_std:      噪声标准差
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 1,
        noisy_gating: bool = True,
        noise_std: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noisy_gating = noisy_gating
        self.noise_std = noise_std

        self.gate = nn.Linear(d_model, num_experts, bias=False)
        if noisy_gating:
            self.noise_linear = nn.Linear(d_model, num_experts, bias=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        参数:
            x: (B, T, D)  token hidden states

        返回:
            weights:   (B, T, top_k)   归一化路由权重
            indices:   (B, T, top_k)   所选专家下标
            aux_loss:  scalar          辅助损失 (balance + z-loss)
        """
        # logits: (B, T, E)
        logits = self.gate(x)

        # 训练时添加可学习噪声以促进探索
        if self.training and self.noisy_gating:
            noise = torch.randn_like(logits) * F.softplus(self.noise_linear(x)) * self.noise_std
            logits = logits + noise

        # --- 辅助损失 ---
        aux_loss = self._compute_aux_loss(logits)

        # top-k 选择
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)  # (B,T,k)
        top_k_weights = torch.softmax(top_k_logits, dim=-1)                    # (B,T,k)

        return top_k_weights, top_k_indices, aux_loss

    def _compute_aux_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """计算 Switch-style load-balancing loss + router z-loss。

        balance loss = N * sum_i(f_i * P_i)
        z-loss       = mean(logsumexp(logits)^2)
        """
        # logits: (B, T, E)
        num_experts = logits.size(-1)
        probs = torch.softmax(logits, dim=-1)  # (B,T,E)

        # f_i: 每个专家被 top-k 选中的 token 比例
        _, top_indices = torch.topk(logits, self.top_k, dim=-1)          # (B,T,k)
        expert_mask = torch.zeros_like(probs)
        expert_mask.scatter_(-1, top_indices, 1.0)                        # (B,T,E)
        f = expert_mask.float().mean(dim=(0, 1))                          # (E,)

        # P_i: 每个专家获得的平均路由概率
        P = probs.mean(dim=(0, 1))                                        # (E,)

        # Switch Transformer balance loss
        balance_loss = num_experts * (f * P).sum()

        # Router z-loss: 防止 logits 过大导致数值不稳
        z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()

        return balance_loss + z_loss


class MoEFFN(nn.Module):
    """MoE-Ready FFN：支持 Dense / MoE 可切换的前馈网络

    当 use_moe=False 时退化为普通 Dense FFN (aux_loss=0)。
    当 use_moe=True 时：
      - 创建 num_experts 个 FFN 专家
      - 用 TopKRouter 选专家
      - 可选 shared_expert（always-on 的 dense 专家当底盘）
      - forward 返回 (output, aux_loss)

    后端策略（可扩展）:
      - 默认: 纯 PyTorch loop dispatch (兼容性最好)
      - 可选: megablocks / scattermoe / tutel (安装后自动启用)
    """

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        use_swiglu: bool = True,
        # --- MoE 参数 ---
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 1,
        capacity_factor: float = 1.25,
        shared_expert: bool = True,
        noisy_gating: bool = True,
    ):
        super().__init__()
        self.use_moe = use_moe
        self.d_model = d_model
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.use_shared_expert = shared_expert and use_moe

        if not use_moe:
            # --- Dense 模式 ---
            if use_swiglu:
                self.dense_ffn = SwiGLU(d_model, dim_feedforward, d_model, dropout=dropout)
            else:
                self.dense_ffn = nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.GELU() if activation == "gelu" else nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, d_model),
                )
        else:
            # --- MoE 模式 ---
            self.router = TopKRouter(
                d_model=d_model,
                num_experts=num_experts,
                top_k=top_k,
                noisy_gating=noisy_gating,
            )

            # 创建 num_experts 个独立 FFN 专家
            experts = []
            for _ in range(num_experts):
                if use_swiglu:
                    experts.append(SwiGLU(d_model, dim_feedforward, d_model, dropout=dropout))
                else:
                    experts.append(nn.Sequential(
                        nn.Linear(d_model, dim_feedforward),
                        nn.GELU() if activation == "gelu" else nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(dim_feedforward, d_model),
                    ))
            self.experts = nn.ModuleList(experts)

            # 可选 shared expert (always-on dense 专家)
            if self.use_shared_expert:
                if use_swiglu:
                    self.shared_expert_ffn = SwiGLU(d_model, dim_feedforward, d_model, dropout=dropout)
                else:
                    self.shared_expert_ffn = nn.Sequential(
                        nn.Linear(d_model, dim_feedforward),
                        nn.GELU() if activation == "gelu" else nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(dim_feedforward, d_model),
                    )
                # 融合门控：控制 shared_expert 和 routed_expert 的比例
                self.shared_gate = nn.Linear(d_model, 1, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            x: (B, T, D)

        返回:
            output:   (B, T, D)
            aux_loss: scalar (Dense 模式下为 0)
        """
        if not self.use_moe:
            return self.dense_ffn(x), torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # --- MoE dispatch ---
        B, T, D = x.shape
        weights, indices, aux_loss = self.router(x)  # (B,T,k), (B,T,k), scalar

        # Token-level dispatch: 对每个 token 加权组合 top_k 专家输出
        # 使用高效的 batch 实现，避免 python-level loop over tokens
        x_flat = x.view(B * T, D)                                  # (N, D)
        weights_flat = weights.view(B * T, self.top_k)              # (N, k)
        indices_flat = indices.view(B * T, self.top_k)              # (N, k)

        # 收集所有专家需要处理的 token 并批量执行
        output = torch.zeros_like(x_flat)  # (N, D)

        for k_idx in range(self.top_k):
            expert_indices_k = indices_flat[:, k_idx]   # (N,)
            weights_k = weights_flat[:, k_idx]           # (N,)

            for e_idx in range(len(self.experts)):
                mask = (expert_indices_k == e_idx)       # (N,)
                if mask.any():
                    expert_input = x_flat[mask]          # (n_e, D)
                    expert_output = self.experts[e_idx](expert_input)  # (n_e, D)
                    output[mask] += weights_k[mask].unsqueeze(-1) * expert_output

        output = output.view(B, T, D)

        # --- Shared expert ---
        if self.use_shared_expert:
            shared_out = self.shared_expert_ffn(x)                      # (B,T,D)
            gate = torch.sigmoid(self.shared_gate(x.mean(dim=1, keepdim=True)))  # (B,1,1)
            output = gate * shared_out + (1.0 - gate) * output

        return output, aux_loss


class PositionalEncoding(nn.Module):
    """位置编码实现，支持动态扩展"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        # 预先计算 max_len 个位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        参数:
            x: [batch_size, seq_len, embedding_dim]
        返回:
            x 加上位置编码
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            device = x.device
            extra_len = seq_len - self.pe.size(1)
            # 生成额外的位置编码
            pe_extra = torch.zeros(extra_len, self.d_model, device=device)
            position = torch.arange(self.pe.size(1), seq_len, dtype=torch.float, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / self.d_model))
            pe_extra[:, 0::2] = torch.sin(position * div_term)
            pe_extra[:, 1::2] = torch.cos(position * div_term)
            pe_extra = pe_extra.unsqueeze(0)  # shape: [1, extra_len, d_model]
            pe = torch.cat([self.pe, pe_extra], dim=1)
        else:
            pe = self.pe
        return self.dropout(x + pe[:, :seq_len, :])


# 你可以在此处定义一个全局日志文件名，确保每次都往同一个文件追加写入
DEBUG_LOG_FILE = "autopoietic_debug.log"

class AutopoieticAttention(nn.Module):
    """自生成注意力机制（优化版）

    设计目标：
    - 热路径优先：尽量让注意力走到 PyTorch SDPA（Flash/Math/Memory-efficient）实现
    - 保留“自生成”味道：用低秩自生成分量 + 门控温度 τ 对注意力输出做可学习扰动
    - 保持接口兼容：forward 返回 (attn_out, attn_weights_or_None)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        eps: float = 1e-6,
        alpha: float = 0.1,
        init_tau: float = 1.0,
        sr_ratio: int = 4,
        use_autopoietic: bool = True,
        batch_first: bool = True,
        use_dbc_dac: bool = True,
        debug_mode: bool = False,
        rank_ratio_proj: float = 0.1,
        rank_ratio_res: float = 0.05,
        dbc_threshold: float = 1e-6,
        dbc_iterations: int = 1,
        use_fused_qkv: bool = True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = float(dropout)
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        self.use_autopoietic = use_autopoietic
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.debug_mode = bool(debug_mode)

        # fused QKV for better kernel shapes
        self.use_fused_qkv = bool(use_fused_qkv)
        if self.use_fused_qkv:
            self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
            self.q_proj = self.k_proj = self.v_proj = None
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
            self.qkv_proj = None

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # 可学习温度（用于“自生成门控”）
        self.tau = nn.Parameter(torch.ones(1) * float(init_tau))

        # 低秩“自生成”分量：Δ = (x U) V
        # sr_ratio 越大，rank 越小（更省 FLOPs）
        r = max(4, embed_dim // max(1, int(sr_ratio)))
        self.auto_u = nn.Linear(embed_dim, r, bias=False)
        self.auto_v = nn.Linear(r, embed_dim, bias=False)
        self.auto_gate = nn.Linear(embed_dim, num_heads, bias=True)

        self.dropout_layer = nn.Dropout(self.dropout)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        # (B,T,C) -> (B,H,T,D)
        b, t, c = x.shape
        x = x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        return x

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        # (B,H,T,D) -> (B,T,C)
        b, h, t, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * d)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        is_causal: bool = False,
    ):
        # 统一 batch_first
        if not self.batch_first:
            # (T,B,C) -> (B,T,C)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        b, t, c = query.shape

        # Patch 5: 只有真正 self-attn（q/k/v 指向同一张量）才走 fused QKV
        is_self_attn = (query is key) and (key is value)
        use_fused = self.use_fused_qkv and is_self_attn

        if use_fused:
            qkv = self.qkv_proj(query)  # (B,T,3C)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            if self.use_fused_qkv:
                # fused proj 不适用 cross-attn，退化：用同一矩阵分别投影 q/k/v
                q = self.qkv_proj(query)[..., :c]
                k = self.qkv_proj(key)[..., c:2*c]
                v = self.qkv_proj(value)[..., 2*c:]
            else:
                q = self.q_proj(query)
                k = self.k_proj(key)
                v = self.v_proj(value)

        q = self._shape(q)
        k = self._shape(k)
        v = self._shape(v)

        # key_padding_mask: (B,T) -> (B,1,1,T) additive mask for SDPA when needed
        # SDPA in PyTorch supports attn_mask; we compose a boolean mask if possible.
        composed_mask = None
        if key_padding_mask is not None:
            # True means "pad" (masked)
            kpm = key_padding_mask.view(b, 1, 1, -1).to(dtype=torch.bool)
            composed_mask = kpm if composed_mask is None else (composed_mask | kpm)

        if attn_mask is not None:
            # accept (T,T), (B,T,T), or already broadcastable
            am = attn_mask
            # 统一转为 bool：-inf / 非零 → True（遮掉），0.0 → False（放通）
            # 这样 float causal mask 和 bool kpm 可以安全地用 | 合并，
            # 避免 float mask 在 composed_mask 已存在时被静默丢弃。
            if am.dtype != torch.bool:
                am = am.bool()
            # broadcast to (B,1,T,T)
            if am.dim() == 2:
                am = am.view(1, 1, am.size(0), am.size(1)).expand(b, 1, -1, -1).contiguous()
            elif am.dim() == 3:
                am = am.view(b, 1, am.size(-2), am.size(-1)).contiguous()
            composed_mask = am if composed_mask is None else (composed_mask | am)

        # SDPA fast path
        if hasattr(F, "scaled_dot_product_attention"):
            # dropout only during training
            dropout_p = self.dropout if self.training else 0.0
            attn_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=composed_mask,
                dropout_p=dropout_p,
                is_causal=is_causal
            )  # (B,H,T,D)
            attn_weights = None
        else:
            # fallback: explicit softmax
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,H,T,T)
            if composed_mask is not None:
                if composed_mask.dtype == torch.bool:
                    scores = scores.masked_fill(composed_mask, float("-inf"))
                else:
                    scores = scores + composed_mask
            if is_causal:
                causal = torch.triu(torch.ones(t, t, device=scores.device, dtype=torch.bool), diagonal=1)
                scores = scores.masked_fill(causal.view(1,1,t,t), float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout_layer(attn)
            attn_out = torch.matmul(attn, v)
            attn_weights = attn if need_weights else None

        out = self._merge(attn_out)  # (B,T,C)

        # 自生成扰动：低秩分量 + 门控温度
        if self.use_autopoietic:
            # gate per-head from mean token embedding
            g = torch.sigmoid(self.auto_gate(query.mean(dim=1)))  # (B,H)
            g = g.view(b, self.num_heads, 1, 1)
            delta = self._shape(self.auto_v(self.auto_u(query)))  # (B,H,T,D)
            # 温度 τ 与 alpha 控制扰动强度
            out = out + self.alpha * torch.tanh(self.tau) * self._merge(delta * g)

        out = self.out_proj(out)

        if not self.batch_first:
            out = out.transpose(0, 1)

        return out, attn_weights

class APTEncoderLayer(nn.Module):
    """
    APT编码器层
    集成自生成注意力机制 + 左旋平滑残差连接 + MoE-Ready FFN
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        batch_first: bool = True,
        eps: float = 1e-6,
        alpha: float = 0.1,
        init_tau: float = 1.0,
        sr_ratio: int = 4,
        use_autopoietic: bool = True,
        # DBC-DAC相关参数
        use_dbc_dac: bool = True,
        rank_ratio_proj: float = 0.1,
        rank_ratio_res: float = 0.05,
        dbc_threshold: float = 1e-6,
        dbc_iterations: int = 1,
        # 左旋平滑参数
        use_left_spin: bool = True,
        left_spin_alpha: float = 0.5,
        left_spin_tau: float = 0.3,
        left_spin_beta: float = 0.7,
        # 现代化组件开关
        use_rmsnorm: bool = True,
        use_swiglu: bool = True,
        # FFN 扩展倍率（Phi-3 风格）
        ffn_ratio: Optional[float] = None,
        # MoE 参数（默认全关闭，不影响现有训练）
        use_moe: bool = False,
        moe_num_experts: int = 8,
        moe_top_k: int = 1,
        moe_capacity_factor: float = 1.25,
        moe_shared_expert: bool = True,
        moe_noisy_gating: bool = True,
    ):
        super().__init__()
        self.use_rmsnorm = use_rmsnorm
        self.use_swiglu = use_swiglu

        # FFN 扩展倍率覆盖
        if ffn_ratio is not None:
            dim_feedforward = round(d_model * ffn_ratio)

        # 自生成注意力层
        self.self_attn = AutopoieticAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            eps=eps,
            alpha=alpha,
            init_tau=init_tau,
            sr_ratio=sr_ratio,
            use_autopoietic=use_autopoietic,
            batch_first=batch_first,
            use_dbc_dac=use_dbc_dac,
            rank_ratio_proj=rank_ratio_proj,
            rank_ratio_res=rank_ratio_res,
            dbc_threshold=dbc_threshold,
            dbc_iterations=dbc_iterations
        )

        # 前馈网络：统一走 MoEFFN (Dense / MoE 可切换)
        self.ffn = MoEFFN(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            use_swiglu=use_swiglu,
            use_moe=use_moe,
            num_experts=moe_num_experts,
            top_k=moe_top_k,
            capacity_factor=moe_capacity_factor,
            shared_expert=moe_shared_expert,
            noisy_gating=moe_noisy_gating,
        )
        # 兼容旧代码的字段引用
        self.swiglu = None
        self.linear1 = None
        self.linear2 = None
        self.dropout = None

        # 层归一化
        self.norm1 = RMSNorm(d_model, eps=eps) if self.use_rmsnorm else nn.LayerNorm(d_model)
        self.norm2 = RMSNorm(d_model, eps=eps) if self.use_rmsnorm else nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 激活函数
        self.activation_fn = F.gelu if activation == "gelu" else F.relu

        # 配置
        self.batch_first = batch_first
        self.use_left_spin = use_left_spin

        # 辅助损失缓存（每层独立）
        self._aux_loss = torch.tensor(0.0)

        # 🚀 左旋平滑残差连接（替换传统泰勒展开）
        if use_left_spin:
            self.left_spin_attn = LeftSpinResidual(
                alpha=alpha,  # 修复：使用与 AutopoieticAttention 一致的 alpha
                tau=init_tau,  # 修复：使用与 AutopoieticAttention 一致的 init_tau
                beta=left_spin_beta,
                gate_type='normalized',
                adaptive=True
            )
            self.left_spin_ffn = LeftSpinResidual(
                alpha=alpha,  # 修复：使用与 AutopoieticAttention 一致的 alpha
                tau=init_tau,  # 修复：使用与 AutopoieticAttention 一致的 init_tau
                beta=left_spin_beta,
                gate_type='normalized',
                adaptive=True
            )
        else:
            self.left_spin_attn = None
            self.left_spin_ffn = None

        self.debug_mode = False

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        编码器层前向传播（集成左旋平滑 + MoE）

        参数:
            src: 输入张量 [seq_len, batch_size, d_model] 或 [batch_size, seq_len, d_model]
            src_mask: 序列掩码 [seq_len, seq_len] 或 [batch_size, seq_len, seq_len]
            src_key_padding_mask: 填充掩码 [batch_size, seq_len]

        返回:
            output: 编码器层输出
        """
        # 🚀 自注意力子层（左旋平滑残差连接）
        src2, _ = self.self_attn(
            query=src,
            key=src,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src2_dropout = self.dropout1(src2)

        if self.use_left_spin and self.left_spin_attn is not None:
            src = self.left_spin_attn(src, src2_dropout)
        else:
            src = src + src2_dropout

        src = self.norm1(src)

        # 🚀 前馈网络子层（MoE-Ready: 返回 (output, aux_loss)）
        src2, aux_loss = self.ffn(src)
        self._aux_loss = aux_loss
        src2_dropout = self.dropout2(src2)

        if self.use_left_spin and self.left_spin_ffn is not None:
            src = self.left_spin_ffn(src, src2_dropout)
        else:
            src = src + src2_dropout

        src = self.norm2(src)

        return src


class APTDecoderLayer(nn.Module):
    """
    APT解码器层
    集成自生成注意力机制 + 左旋平滑残差连接 + MoE-Ready FFN
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        batch_first: bool = True,
        eps: float = 1e-6,
        alpha: float = 0.1,
        init_tau: float = 1.0,
        sr_ratio: int = 4,
        use_autopoietic: bool = True,
        # DBC-DAC相关参数
        use_dbc_dac: bool = True,
        rank_ratio_proj: float = 0.1,
        rank_ratio_res: float = 0.05,
        dbc_threshold: float = 1e-6,
        dbc_iterations: int = 1,
        # 左旋平滑参数
        use_left_spin: bool = True,
        left_spin_alpha: float = 0.5,
        left_spin_tau: float = 0.3,
        left_spin_beta: float = 0.7,
        # 现代化组件开关
        use_rmsnorm: bool = True,
        use_swiglu: bool = True,
        # GPT-only 开关：False 时 cross-attn 被旁路
        use_cross_attn: bool = True,
        # FFN 扩展倍率（Phi-3 风格）
        ffn_ratio: Optional[float] = None,
        # MoE 参数（默认全关闭，不影响现有训练）
        use_moe: bool = False,
        moe_num_experts: int = 8,
        moe_top_k: int = 1,
        moe_capacity_factor: float = 1.25,
        moe_shared_expert: bool = True,
        moe_noisy_gating: bool = True,
    ):
        super().__init__()
        self.use_rmsnorm = use_rmsnorm
        self.use_swiglu = use_swiglu
        self.use_cross_attn = use_cross_attn

        # FFN 扩展倍率覆盖
        if ffn_ratio is not None:
            dim_feedforward = round(d_model * ffn_ratio)

        # 自注意力层(掩码)
        self.self_attn = AutopoieticAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            eps=eps,
            alpha=alpha,
            init_tau=init_tau,
            sr_ratio=sr_ratio,
            use_autopoietic=use_autopoietic,
            batch_first=batch_first,
            use_dbc_dac=use_dbc_dac,
            rank_ratio_proj=rank_ratio_proj,
            rank_ratio_res=rank_ratio_res,
            dbc_threshold=dbc_threshold,
            dbc_iterations=dbc_iterations
        )

        # 编码器-解码器注意力层
        self.multihead_attn = AutopoieticAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            eps=eps,
            alpha=alpha,
            init_tau=init_tau,
            sr_ratio=sr_ratio,
            use_autopoietic=use_autopoietic,
            batch_first=batch_first,
            use_dbc_dac=use_dbc_dac,
            rank_ratio_proj=rank_ratio_proj,
            rank_ratio_res=rank_ratio_res,
            dbc_threshold=dbc_threshold,
            dbc_iterations=dbc_iterations
        )

        # 前馈网络：统一走 MoEFFN (Dense / MoE 可切换)
        self.ffn = MoEFFN(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            use_swiglu=use_swiglu,
            use_moe=use_moe,
            num_experts=moe_num_experts,
            top_k=moe_top_k,
            capacity_factor=moe_capacity_factor,
            shared_expert=moe_shared_expert,
            noisy_gating=moe_noisy_gating,
        )
        # 兼容旧代码的字段引用
        self.swiglu = None
        self.linear1 = None
        self.linear2 = None
        self.dropout = None

        # 层归一化
        self.norm1 = RMSNorm(d_model, eps=eps) if self.use_rmsnorm else nn.LayerNorm(d_model)
        self.norm2 = RMSNorm(d_model, eps=eps) if self.use_rmsnorm else nn.LayerNorm(d_model)
        self.norm3 = RMSNorm(d_model, eps=eps) if self.use_rmsnorm else nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # 激活函数
        self.activation_fn = F.gelu if activation == "gelu" else F.relu

        # 配置
        self.batch_first = batch_first
        self.use_left_spin = use_left_spin

        # 辅助损失缓存（每层独立）
        self._aux_loss = torch.tensor(0.0)

        # 🚀 左旋平滑残差连接（3个子层）
        if use_left_spin:
            self.left_spin_self_attn = LeftSpinResidual(
                alpha=alpha,  # 修复：使用与 AutopoieticAttention 一致的 alpha
                tau=init_tau,  # 修复：使用与 AutopoieticAttention 一致的 init_tau
                beta=left_spin_beta,
                gate_type='normalized',
                adaptive=True
            )
            self.left_spin_cross_attn = LeftSpinResidual(
                alpha=alpha,  # 修复：使用与 AutopoieticAttention 一致的 alpha
                tau=init_tau,  # 修复：使用与 AutopoieticAttention 一致的 init_tau
                beta=left_spin_beta,
                gate_type='normalized',
                adaptive=True
            )
            self.left_spin_ffn = LeftSpinResidual(
                alpha=alpha,  # 修复：使用与 AutopoieticAttention 一致的 alpha
                tau=init_tau,  # 修复：使用与 AutopoieticAttention 一致的 init_tau
                beta=left_spin_beta,
                gate_type='normalized',
                adaptive=True
            )
        else:
            self.left_spin_self_attn = None
            self.left_spin_cross_attn = None
            self.left_spin_ffn = None

    def forward(
        self,
        tgt: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        解码器层前向传播（集成左旋平滑 + MoE）

        参数:
            tgt: 目标序列 [seq_len, batch_size, d_model] 或 [batch_size, seq_len, d_model]
            memory: 编码器输出（可为 None，此时完全跳过 cross-attn，等价 GPT block）
            tgt_mask: 目标序列掩码 [tgt_len, tgt_len] 或 [batch_size, tgt_len, tgt_len]
            memory_mask: 记忆掩码 [tgt_len, src_len]
            tgt_key_padding_mask: 目标填充掩码 [batch_size, tgt_len]
            memory_key_padding_mask: 记忆填充掩码 [batch_size, src_len]

        返回:
            output: 解码器层输出
        """
        # 🚀 自注意力子层（左旋平滑残差连接）
        tgt2, _ = self.self_attn(
            query=tgt,
            key=tgt,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt2_dropout = self.dropout1(tgt2)

        if self.use_left_spin and self.left_spin_self_attn is not None:
            tgt = self.left_spin_self_attn(tgt, tgt2_dropout)
        else:
            tgt = tgt + tgt2_dropout

        tgt = self.norm1(tgt)

        # 🚀 编码器-解码器注意力子层（旁路式：memory=None 或 use_cross_attn=False 时跳过）
        do_cross = (
            memory is not None
            and getattr(self, "use_cross_attn", True)
            and getattr(self, "multihead_attn", None) is not None
        )
        if do_cross:
            tgt2, _ = self.multihead_attn(
                query=tgt,
                key=memory,
                value=memory,
                attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask
            )
            tgt2_dropout = self.dropout2(tgt2)

            if self.use_left_spin and self.left_spin_cross_attn is not None:
                tgt = self.left_spin_cross_attn(tgt, tgt2_dropout)
            else:
                tgt = tgt + tgt2_dropout

            tgt = self.norm2(tgt)

        # 🚀 前馈网络子层（MoE-Ready: 返回 (output, aux_loss)）
        tgt2, aux_loss = self.ffn(tgt)
        self._aux_loss = aux_loss
        tgt2_dropout = self.dropout3(tgt2)

        if self.use_left_spin and self.left_spin_ffn is not None:
            tgt = self.left_spin_ffn(tgt, tgt2_dropout)
        else:
            tgt = tgt + tgt2_dropout

        tgt = self.norm3(tgt)

        return tgt


class APTModelConfiguration:
    """APT模型配置类（集成左旋平滑 + MoE）"""
    def __init__(
        self,
        vocab_size: int = 30522,  # 词汇表大小
        d_model: int = 768,  # 模型维度
        max_seq_len: int = 2048,  # 最大序列长度
        num_encoder_layers: int = 12,  # 编码器层数
        num_decoder_layers: int = 12,  # 解码器层数
        num_heads: int = 12,  # 注意力头数
        d_ff: int = 3072,  # 前馈网络维度
        dropout: float = 0.1,  # Dropout比率
        activation: str = "gelu",  # 激活函数
        epsilon: float = 1e-6,  # 自生成无穷倒数缩放因子
        alpha: float = 0.1,  # 泰勒展开系数（已被左旋平滑替换）
        beta: float = 0.01,  # 动态调节系数
        init_tau: float = 1.0,  # 初始温度
        sr_ratio: int = 4,  # 自生成矩阵压缩比
        use_autopoietic: bool = True,  # 是否使用自生成机制
        base_lr: float = 3e-5,  # 基准学习率(用于动态参数调整)
        batch_first: bool = True,  # 是否使用batch_first格式
        pad_token_id: int = 0,  # 填充token ID
        bos_token_id: int = 101,  # 开始token ID
        eos_token_id: int = 102,  # 结束token ID
        # DBC-DAC 相关参数
        use_dbc_dac: bool = True,  # 是否使用DBC-DAC稳定
        rank_ratio_proj: float = 0.1,  # DBC投影比例
        rank_ratio_res: float = 0.05,  # DAC残差比例
        dbc_threshold: float = 1e-6,  # DBC阈值
        dbc_iterations: int = 1,  # DAC迭代次数
        # 🚀 左旋平滑相关参数（替换泰勒展开）
        use_left_spin: bool = True,  # 是否使用左旋平滑残差
        left_spin_alpha: float = 0.5,  # 缓冲强度系数
        left_spin_tau: float = 0.3,  # 尖点阈值
        left_spin_beta: float = 0.7,  # 惯性系数
        # GPT-only 开关（旁路式，保留 Encoder 结构不删除）
        decoder_only: bool = True,   # True=GPT-only forward；False=seq2seq forward
        use_cross_attn: bool = False,  # DecoderLayer 是否启用 cross-attn
        # FFN 扩展倍率（Phi-3 风格，None 保持 d_ff 不变）
        ffn_ratio: Optional[float] = None,
        # MoE 参数（默认全关闭，不影响现有训练）
        use_moe: bool = False,              # 是否启用 MoE FFN
        moe_num_experts: int = 8,           # 专家总数（常用 8/16）
        moe_top_k: int = 1,                 # 每 token 选几个专家（先 Top-1 最稳）
        moe_capacity_factor: float = 1.25,  # 容量因子（防溢出）
        moe_aux_weight: float = 0.01,       # 辅助损失权重（训练时用）
        moe_shared_expert: bool = True,     # 是否启用 shared expert（always-on 底盘）
        moe_noisy_gating: bool = True,      # 训练时路由器加噪声探索
        moe_router_z_loss: float = 0.0,     # z-loss 额外缩放（0 = 走默认）
        **kwargs  # 其他参数
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.init_tau = init_tau
        self.sr_ratio = sr_ratio
        self.use_autopoietic = use_autopoietic
        self.base_lr = base_lr
        self.batch_first = batch_first
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        # DBC-DAC相关参数
        self.use_dbc_dac = use_dbc_dac
        self.rank_ratio_proj = rank_ratio_proj
        self.rank_ratio_res = rank_ratio_res
        self.dbc_threshold = dbc_threshold
        self.dbc_iterations = dbc_iterations

        # 🚀 左旋平滑相关参数
        self.use_left_spin = use_left_spin
        self.left_spin_alpha = left_spin_alpha
        self.left_spin_tau = left_spin_tau
        self.left_spin_beta = left_spin_beta

        # GPT-only 开关
        self.decoder_only = decoder_only
        self.use_cross_attn = use_cross_attn

        # FFN 扩展倍率
        self.ffn_ratio = ffn_ratio

        # MoE 参数
        self.use_moe = use_moe
        self.moe_num_experts = moe_num_experts
        self.moe_top_k = moe_top_k
        self.moe_capacity_factor = moe_capacity_factor
        self.moe_aux_weight = moe_aux_weight
        self.moe_shared_expert = moe_shared_expert
        self.moe_noisy_gating = moe_noisy_gating
        self.moe_router_z_loss = moe_router_z_loss

        # 添加任何额外参数
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self):
        """将配置转换为字典"""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict):
        """从字典创建配置"""
        return cls(**config_dict)
    
    def save_pretrained(self, save_directory):
        """保存配置到指定目录"""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        config_file = os.path.join(save_directory, "config.json")
        
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_path):
        """从预训练目录加载配置"""
        import os
        import json
        
        config_file = os.path.join(model_path, "config.json")
        
        if not os.path.exists(config_file):
            raise ValueError(f"在 {model_path} 中找不到config.json")
        
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)


class APTModel(nn.Module):
    """
    自生成变换器(APT)模型
    集成了自生成注意力机制和DBC-DAC稳定技术的完整Transformer模型
    支持编码器-解码器架构，适用于各种序列到序列任务
    """
    def __init__(self, config: APTModelConfiguration):
        super().__init__()
        self.config = config
        use_rmsnorm = getattr(config, 'use_rmsnorm', True)
        use_swiglu = getattr(config, 'use_swiglu', True)
        # 词嵌入
        self.token_embedding = nn.Embedding(
            config.vocab_size, 
            config.d_model, 
            padding_idx=config.pad_token_id
        )
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(
            config.d_model,
            max_len=config.max_seq_len,
            dropout=config.dropout
        )
        
        # 兼容缺失的可选配置项，提供合理的默认值
        use_autopoietic = getattr(config, "use_autopoietic", True)
        use_dbc_dac = getattr(config, "use_dbc_dac", False)
        rank_ratio_proj = getattr(config, "rank_ratio_proj", 0.1)
        rank_ratio_res = getattr(config, "rank_ratio_res", 0.05)
        dbc_threshold = getattr(config, "dbc_threshold", 1e-6)
        dbc_iterations = getattr(config, "dbc_iterations", 1)

        # 🚀 左旋平滑参数
        use_left_spin = getattr(config, "use_left_spin", True)
        left_spin_alpha = getattr(config, "left_spin_alpha", 0.5)
        left_spin_tau = getattr(config, "left_spin_tau", 0.3)
        left_spin_beta = getattr(config, "left_spin_beta", 0.7)

        # GPT-only 开关
        self.decoder_only = bool(getattr(config, "decoder_only", True))
        use_cross_attn = bool(getattr(config, "use_cross_attn", False))

        # MoE 参数
        ffn_ratio = getattr(config, "ffn_ratio", None)
        use_moe = getattr(config, "use_moe", False)
        moe_num_experts = getattr(config, "moe_num_experts", 8)
        moe_top_k = getattr(config, "moe_top_k", 1)
        moe_capacity_factor = getattr(config, "moe_capacity_factor", 1.25)
        moe_shared_expert = getattr(config, "moe_shared_expert", True)
        moe_noisy_gating = getattr(config, "moe_noisy_gating", True)
        self.moe_aux_weight = float(getattr(config, "moe_aux_weight", 0.01))

        # 创建编码器层
        encoder_layers = []
        for _ in range(config.num_encoder_layers):
            encoder_layers.append(
                APTEncoderLayer(d_model=config.d_model,
                    nhead=config.num_heads,
                    dim_feedforward=config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                    batch_first=config.batch_first,
                    eps=config.epsilon,
                    alpha=config.alpha,
                    init_tau=config.init_tau,
                    sr_ratio=config.sr_ratio,
                    use_autopoietic=use_autopoietic,
                    use_dbc_dac=use_dbc_dac,
                    rank_ratio_proj=rank_ratio_proj,
                    rank_ratio_res=rank_ratio_res,
                    dbc_threshold=dbc_threshold,
                    dbc_iterations=dbc_iterations,
                    # 🚀 左旋平滑参数
                    use_left_spin=use_left_spin,
                    left_spin_alpha=left_spin_alpha,
                    left_spin_tau=left_spin_tau,
                    left_spin_beta=left_spin_beta,
                    use_rmsnorm=use_rmsnorm,
                    use_swiglu=use_swiglu,
                    # MoE 参数
                    ffn_ratio=ffn_ratio,
                    use_moe=use_moe,
                    moe_num_experts=moe_num_experts,
                    moe_top_k=moe_top_k,
                    moe_capacity_factor=moe_capacity_factor,
                    moe_shared_expert=moe_shared_expert,
                    moe_noisy_gating=moe_noisy_gating)
            )

        # 创建解码器层
        decoder_layers = []
        for _ in range(config.num_decoder_layers):
            decoder_layers.append(
                APTDecoderLayer(d_model=config.d_model,
                    nhead=config.num_heads,
                    dim_feedforward=config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                    batch_first=config.batch_first,
                    eps=config.epsilon,
                    alpha=config.alpha,
                    init_tau=config.init_tau,
                    sr_ratio=config.sr_ratio,
                    use_autopoietic=use_autopoietic,
                    use_dbc_dac=use_dbc_dac,
                    rank_ratio_proj=rank_ratio_proj,
                    rank_ratio_res=rank_ratio_res,
                    dbc_threshold=dbc_threshold,
                    dbc_iterations=dbc_iterations,
                    # 🚀 左旋平滑参数
                    use_left_spin=use_left_spin,
                    left_spin_alpha=left_spin_alpha,
                    left_spin_tau=left_spin_tau,
                    left_spin_beta=left_spin_beta,
                    use_rmsnorm=use_rmsnorm,
                    use_swiglu=use_swiglu,
                    use_cross_attn=use_cross_attn,
                    # MoE 参数
                    ffn_ratio=ffn_ratio,
                    use_moe=use_moe,
                    moe_num_experts=moe_num_experts,
                    moe_top_k=moe_top_k,
                    moe_capacity_factor=moe_capacity_factor,
                    moe_shared_expert=moe_shared_expert,
                    moe_noisy_gating=moe_noisy_gating)
            )
        
        # 编码器和解码器
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.decoder_layers = nn.ModuleList(decoder_layers)
        
        # 最终层归一化
        self.encoder_norm = RMSNorm(config.d_model, eps=getattr(config, 'layer_norm_eps', 1e-6)) if use_rmsnorm else nn.LayerNorm(config.d_model)
        self.decoder_norm = RMSNorm(config.d_model, eps=getattr(config, 'layer_norm_eps', 1e-6)) if use_rmsnorm else nn.LayerNorm(config.d_model)
        
        # 输出投影
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # 权重共享(可选)
        self.output_projection.weight = self.token_embedding.weight
        
        # 初始化DBC-DAC优化器
        self.dbc_dac_optimizer = DBCDAC_Optimizer(
            rank_ratio_proj=rank_ratio_proj,
            rank_ratio_res=rank_ratio_res,
            threshold=dbc_threshold,
            iterations=dbc_iterations,
            apply_to_gradients=True
        ) if use_dbc_dac else None
        
        # 添加梯度稳定钩子
        self.gradient_hooks = add_gradient_hooks_to_model(self, self.dbc_dac_optimizer) if self.dbc_dac_optimizer else []
            
        # 初始化参数
        self._reset_parameters()
    
    def _reset_parameters(self):
        """初始化模型参数"""
        # 初始化嵌入
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # 对于padding_idx，将嵌入向量置零
        if self.token_embedding.padding_idx is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.token_embedding.padding_idx].fill_(0)

    # ------------------------------------------------------------------
    # MoE aux_loss 聚合
    # ------------------------------------------------------------------

    def _gather_aux_loss(self, layers) -> torch.Tensor:
        """从所有层收集 MoE 辅助损失并求和。"""
        total = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in layers:
            layer_aux = getattr(layer, "_aux_loss", None)
            if layer_aux is not None and isinstance(layer_aux, torch.Tensor):
                total = total + layer_aux
        return total

    # ------------------------------------------------------------------
    # GPT-only 路径（Patch 1）
    # ------------------------------------------------------------------

    def _build_causal_mask(self, tgt_len: int, device) -> torch.Tensor:
        """构建 causal mask：bool 矩阵，True 表示「被遮掉（不可见）」。"""
        return torch.triu(
            torch.ones((tgt_len, tgt_len), device=device, dtype=torch.bool),
            diagonal=1
        )

    def forward_lm(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Decoder-only LM forward（纯 GPT 路径）。

        参数:
            input_ids: (B, S) — token id
            attention_mask: (B, S)，可选。
                - dtype=bool → True=keep（HF 风格），内部转为 True=mask
                - dtype=int/float → 1=keep, 0=pad
            return_hidden: 是否同时返回最后一层 hidden states

        返回:
            logits (B, S, vocab_size)，或 (logits, hidden) 当 return_hidden=True
            注: 当 use_moe=True 时，可通过 model.last_aux_loss 获取辅助损失
        """
        bsz, seqlen = input_ids.shape
        device = input_ids.device

        x = self.token_embedding(input_ids)
        x = self.positional_encoding(x)

        # causal mask (S, S)，True=mask
        causal_mask = self._build_causal_mask(seqlen, device=device)

        # key padding mask (B, S)，True=mask
        key_padding_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                key_padding_mask = ~attention_mask
            else:
                key_padding_mask = (attention_mask == 0)

        for layer in self.decoder_layers:
            x = layer(
                tgt=x,
                memory=None,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=key_padding_mask,
                memory_mask=None,
                memory_key_padding_mask=None,
            )

        x = self.decoder_norm(x)
        logits = self.output_projection(x)

        # 聚合 MoE 辅助损失（挂在 self 上供训练脚本使用）
        self.last_aux_loss = self._gather_aux_loss(self.decoder_layers) * self.moe_aux_weight

        if return_hidden:
            return logits, x
        return logits

    # ------------------------------------------------------------------
    # 保留 Encoder 相关方法（seq2seq 路径随时可切回）
    # ------------------------------------------------------------------

    def encode(
        self,
        src_tokens: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        编码器前向传播
    
        参数:
            src_tokens: 源序列token ID [batch_size, src_len]
            src_mask: 源序列掩码
            src_key_padding_mask: 源序列填充掩码
    
        返回:
            memory: 编码器输出
        """
        # 获取词嵌入
        src = self.token_embedding(src_tokens)
    
        # 添加位置编码
        src = self.positional_encoding(src)
    
        # 通过编码器层
        for layer in self.encoder_layers:
            src = layer(
                src=src,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask
            )
    
        # 最终层归一化
        memory = self.encoder_norm(src)
    
        return memory
    
    def decode(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        解码器前向传播
    
        参数:
            tgt_tokens: 目标序列token ID [batch_size, tgt_len]
            memory: 编码器输出
            tgt_mask: 目标序列掩码
            memory_mask: 记忆掩码
            tgt_key_padding_mask: 目标序列填充掩码
            memory_key_padding_mask: 记忆填充掩码
    
        返回:
            output: 解码器输出
        """
        # 获取词嵌入
        tgt = self.token_embedding(tgt_tokens)
    
        # 添加位置编码
        tgt = self.positional_encoding(tgt)
    
        # 如果没有提供目标掩码，创建自回归掩码（上三角矩阵）
        if tgt_mask is None and self.config.batch_first:
            tgt_len = tgt.size(1)
            device = tgt.device
            tgt_mask = torch.triu(
                torch.full((tgt_len, tgt_len), float('-inf'), device=device),
                diagonal=1
            )
    
        # 通过解码器层
        for layer in self.decoder_layers:
            tgt = layer(
                tgt=tgt,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
    
        # 最终层归一化
        output = self.decoder_norm(tgt)
    
        return output
    
    def forward(
        self,
        src_tokens: torch.Tensor = None,
        tgt_tokens: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        # 兼容AutopoieticAttention风格参数
        query: torch.Tensor = None,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """前向路由器：decoder_only=True 时走 GPT 路径，否则走 seq2seq 路径。"""
        if getattr(self, "decoder_only", True):
            # GPT-only 路径
            input_ids = src_tokens if src_tokens is not None else kwargs.get("input_ids")
            if input_ids is None and query is not None:
                input_ids = query
            attention_mask = src_key_padding_mask if src_key_padding_mask is not None else kwargs.get("attention_mask")
            return_hidden = kwargs.get("return_hidden", False)
            return self.forward_lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_hidden=return_hidden,
            )
        # seq2seq 路径（保留旧逻辑）
        return self.forward_seq2seq(
            src_tokens=src_tokens,
            tgt_tokens=tgt_tokens,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            return_dict=return_dict,
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            **kwargs,
        )

    def forward_seq2seq(
        self,
        src_tokens: torch.Tensor = None,
        tgt_tokens: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        # 兼容AutopoieticAttention风格参数
        query: torch.Tensor = None,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        
        # 将自注意力接口的参数映射到Transformer接口
        if src_tokens is None and query is not None:
            src_tokens = query
        if tgt_tokens is None:
            if key is not None:
                tgt_tokens = key
            else:
                tgt_tokens = src_tokens
        if src_mask is None and attn_mask is not None:
            src_mask = attn_mask
        if src_key_padding_mask is None and key_padding_mask is not None:
            src_key_padding_mask = key_padding_mask
        
        # **确保掩码是bool类型**（若原本是float或long，则转为bool）
        if src_mask is not None and src_mask.dtype != torch.bool:
            src_mask = src_mask.to(torch.bool)
        if tgt_mask is not None and tgt_mask.dtype != torch.bool:
            tgt_mask = tgt_mask.to(torch.bool)
        if memory_mask is not None and memory_mask.dtype != torch.bool:
            memory_mask = memory_mask.to(torch.bool)
        if src_key_padding_mask is not None and src_key_padding_mask.dtype != torch.bool:
            src_key_padding_mask = src_key_padding_mask.to(torch.bool)
        if tgt_key_padding_mask is not None and tgt_key_padding_mask.dtype != torch.bool:
            tgt_key_padding_mask = tgt_key_padding_mask.to(torch.bool)
        if memory_key_padding_mask is not None and memory_key_padding_mask.dtype != torch.bool:
            memory_key_padding_mask = memory_key_padding_mask.to(torch.bool)
    
        memory = self.encode(
            src_tokens=src_tokens,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        decoder_output = self.decode(
            tgt_tokens=tgt_tokens,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask if memory_key_padding_mask is not None else src_key_padding_mask
        )
        
        # 生成logits
        logits = self.output_projection(decoder_output)

        # 聚合 MoE 辅助损失
        aux_enc = self._gather_aux_loss(self.encoder_layers)
        aux_dec = self._gather_aux_loss(self.decoder_layers)
        self.last_aux_loss = (aux_enc + aux_dec) * self.moe_aux_weight

        # 根据return_dict参数决定返回形式
        if return_dict:
            return {
                "logits": logits,
                "encoder_output": memory,
                "decoder_output": decoder_output,
                "aux_loss": self.last_aux_loss,
            }
        else:
            # 默认直接返回logits
            return logits

    def generate(
        self,
        input_ids,
        max_length=50,
        temperature=1.0,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.0,
        do_sample=True,
        num_beams=1,
        eos_token_id=None,
        pad_token_id=None,
    ):
        """生成路由器：decoder_only=True 走 LM 路径，否则走 seq2seq 路径。"""
        if getattr(self, "decoder_only", True):
            return self.generate_lm(
                input_ids=input_ids,
                max_new_tokens=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )
        return self.generate_seq2seq(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            num_beams=num_beams,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

    @torch.no_grad()
    def generate_lm(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Decoder-only (GPT) 自回归生成。

        参数:
            input_ids: (B, S) — prompt token ids
            max_new_tokens: 最多额外生成的 token 数量
            其余参数同 generate()

        返回:
            generated_ids: (B, max_new_tokens) — 仅新生成的部分
        """
        if input_ids is None:
            raise ValueError("input_ids 不能为空")

        device = input_ids.device
        batch_size = input_ids.size(0)

        if eos_token_id is None:
            eos_token_id = getattr(self.config, "eos_token_id", 3)
        if pad_token_id is None:
            pad_token_id = getattr(self.config, "pad_token_id", 0)
        unk_token_id = getattr(self.config, "unk_token_id", None)

        # 当前序列从 prompt 开始，持续 append 新 token
        cur_ids = input_ids.clone()
        generated_ids = torch.empty((batch_size, 0), device=device, dtype=torch.long)

        was_training = self.training
        self.eval()

        try:
            for _ in range(max_new_tokens):
                # GPT forward：只有 input_ids，无 memory
                logits = self.forward_lm(cur_ids)          # (B, S, V)
                next_token_logits = logits[:, -1, :]       # (B, V)

                # 重复惩罚
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        history = set(cur_ids[i].tolist())
                        for tid in history:
                            if next_token_logits[i, tid] > 0:
                                next_token_logits[i, tid] /= repetition_penalty
                            else:
                                next_token_logits[i, tid] *= repetition_penalty

                # 温度
                next_token_logits = next_token_logits / max(float(temperature), 1e-5)

                # 屏蔽特殊符号
                if pad_token_id is not None and 0 <= pad_token_id < next_token_logits.size(-1):
                    next_token_logits[:, pad_token_id] = -float("inf")
                if unk_token_id is not None and 0 <= unk_token_id < next_token_logits.size(-1):
                    next_token_logits[:, unk_token_id] = -float("inf")

                # 采样 / 贪心
                if do_sample:
                    if top_k > 0:
                        v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                        next_token_logits[next_token_logits < v[:, [-1]]] = -float("inf")
                    if 0 < top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        sorted_probs = F.softmax(sorted_logits, dim=-1)
                        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        for i in range(batch_size):
                            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                            next_token_logits[i, indices_to_remove] = -float("inf")
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                cur_ids = torch.cat([cur_ids, next_token], dim=1)

                if (next_token == eos_token_id).all():
                    break
        finally:
            if was_training:
                self.train()

        return generated_ids

    def generate_seq2seq(
        self,
        input_ids,
        max_length=50,
        temperature=1.0,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.0,
        do_sample=True,
        num_beams=1,
        eos_token_id=None,
        pad_token_id=None,
    ):
        """
        ⭐ 修复后的文本生成方法 (Encoder-Decoder 逻辑修正版)

        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            max_length: 最大生成长度
            temperature: 采样温度
            top_p: nucleus采样参数
            top_k: top-k采样参数
            repetition_penalty: 重复惩罚
            do_sample: 是否采样(False则贪心解码)
            num_beams: beam search的beam数量
            eos_token_id: 结束标记ID
            pad_token_id: 填充标记ID

        Returns:
            生成的token IDs [batch_size, generated_length]
        """
        del num_beams  # 当前实现不支持beam search，避免未使用参数警告

        if input_ids is None:
            raise ValueError("input_ids 不能为空")

        device = input_ids.device
        batch_size = input_ids.size(0)

        # 1. 准备特殊 Token
        bos_token_id = getattr(self.config, "bos_token_id", 2)
        if eos_token_id is None:
            eos_token_id = getattr(self.config, "eos_token_id", 3)
        if pad_token_id is None:
            pad_token_id = getattr(self.config, "pad_token_id", 0)
        unk_token_id = getattr(self.config, "unk_token_id", None)

        # ------------------------------------------------------------------
        # 🚀 核心逻辑修复：从 GPT 模式切换回 Encoder-Decoder 模式
        # ------------------------------------------------------------------
        
        # 2. 编码阶段 (Encoder)
        # 一次性读懂 Prompt，获取记忆
        memory = self.encode(
            src_tokens=input_ids,
            src_key_padding_mask=(input_ids == pad_token_id)
        )

        # 3. 解码准备 (Decoder)
        # 给解码器一张白纸，只写一个 [BOS] 开头
        # 绝对不能把 input_ids 喂给解码器，否则它看到 EOS 就会停止！
        decoder_input = torch.full((batch_size, 1), bos_token_id, device=device, dtype=torch.long)
        
        # 用于保存生成结果 (不包含 BOS)
        generated_ids = torch.empty((batch_size, 0), device=device, dtype=torch.long)

        was_training = self.training
        self.eval()

        try:
            with torch.no_grad():
                # 循环生成 Response
                for step in range(max_length):
                    # 前向解码：传入 memory 和 当前已生成的 decoder_input
                    # 注意：我们使用 decode() 方法而不是 forward()
                    decoder_output = self.decode(
                        tgt_tokens=decoder_input,
                        memory=memory,
                        tgt_mask=None, # 内部会自动生成因果掩码
                        memory_mask=None,
                        tgt_key_padding_mask=None,
                        memory_key_padding_mask=(input_ids == pad_token_id)
                    )
                    
                    # 映射到词表
                    logits = self.output_projection(decoder_output)
                    next_token_logits = logits[:, -1, :] # 取最后一个时间步

                    # --- 重复惩罚逻辑 ---
                    if repetition_penalty != 1.0:
                        for i in range(batch_size):
                            # 注意：我们检查的是已经生成的 generated_ids (不含 prompt)
                            history = set(generated_ids[i].tolist())
                            if not history:
                                continue
                            
                            # 将 tensor 转为 list 以便索引，或者直接使用 scatter/gather 优化
                            # 这里为了兼容性保持循环写法，但加入了 logits 正负值的正确处理
                            for token_id in history:
                                if next_token_logits[i, token_id] > 0:
                                    next_token_logits[i, token_id] /= repetition_penalty
                                else:
                                    next_token_logits[i, token_id] *= repetition_penalty

                    # 温度调节
                    temperature = max(float(temperature), 1e-5)
                    next_token_logits = next_token_logits / temperature

                    # 屏蔽特殊符号 (PAD, UNK)
                    if pad_token_id is not None and 0 <= pad_token_id < next_token_logits.size(-1):
                        next_token_logits[:, pad_token_id] = -float('inf')
                    if unk_token_id is not None and 0 <= unk_token_id < next_token_logits.size(-1):
                        next_token_logits[:, unk_token_id] = -float('inf')

                    # 采样
                    if do_sample:
                        # Top-K
                        if top_k > 0:
                            v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                            next_token_logits[next_token_logits < v[:, [-1]]] = -float('inf')
                        
                        # Top-P
                        if 0 < top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                            sorted_probs = F.softmax(sorted_logits, dim=-1)
                            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            for i in range(batch_size):
                                indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                                next_token_logits[i, indices_to_remove] = -float('inf')

                        probs = F.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        # 贪婪搜索
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                    # 拼接到结果中
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)
                    decoder_input = torch.cat([decoder_input, next_token], dim=1)

                    # 检查 EOS
                    if (next_token == eos_token_id).all():
                        break
                        
        finally:
            if was_training:
                self.train()

        # 返回生成结果 (只返回生成的回复部分)
        return generated_ids


class APTLargeModel(APTModel):
    """APTLargeModel是APTModel的别名，用于兼容性目的"""
    def __init__(self, config):
        super().__init__(config)
        # 初始化Taylor参数，确保存在这些属性
        self.register_parameter(
            'taylor_epsilon', 
            nn.Parameter(torch.tensor(config.epsilon, dtype=torch.float))
        )
        self.register_parameter(
            'taylor_alpha', 
            nn.Parameter(torch.tensor(config.alpha, dtype=torch.float))
        )
    
    def update_dynamic_taylor_parameters(self, learning_rate):
        """更新动态Taylor展开参数"""
        try:
            # 如果使用LR调度器，需要根据当前学习率调整参数
            lr_factor = float(learning_rate) / float(self.config.base_lr)
            
            # 安全地更新参数
            if hasattr(self, 'taylor_epsilon'):
                self.taylor_epsilon.data = torch.clamp(
                    self.taylor_epsilon * (1.0 + self.config.alpha * lr_factor),
                    min=0.1, max=10.0
                )
            
            if hasattr(self, 'taylor_alpha'):
                self.taylor_alpha.data = torch.clamp(
                    self.taylor_alpha * (1.0 - self.config.beta * lr_factor),
                    min=0.001, max=0.1
                )
            
            # 更新所有注意力层的参数
            for name, module in self.named_modules():
                if hasattr(module, 'tau') and isinstance(module.tau, torch.nn.Parameter):
                    # 调整温度参数
                    module.tau.data = torch.clamp(
                        module.tau * (1.0 - 0.01 * lr_factor),
                        min=0.5, max=2.0
                    )
        except Exception as e:
            # 如果出现任何异常，记录但不中断训练
            print(f"警告: 动态参数更新失败: {e}")
            # 简单地通过，确保方法不会抛出异常
            pass