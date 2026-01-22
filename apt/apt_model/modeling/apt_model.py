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
from apt.apt_model.utils.fake_torch import get_torch
torch = get_torch()
from apt.apt_model.utils.fake_torch import get_torch
torch = get_torch()
nn = torch.nn
F = torch.nn.functional
import math
import warnings
import sys
from typing import Optional, Tuple, List, Dict, Union

# 导入左旋平滑模块
from apt.apt_model.modeling.left_spin_smooth import (
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
    """
    自生成注意力机制 - 论文完整实现版本
    实现自生成变换器(APT)的核心自生成注意力计算
    集成DBC-DAC稳定化技术
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        eps: float = 1e-6,  # 无穷倒数缩放因子
        alpha: float = 0.1,  # 泰勒展开系数
        init_tau: float = 1.0,  # 初始温度参数
        sr_ratio: int = 4,  # 自生成矩阵压缩比
        use_autopoietic: bool = True,  # 是否使用自生成机制
        batch_first: bool = True,
        # DBC-DAC相关参数（此处仅保留接口，不影响本类核心实现）
        use_dbc_dac: bool = True,
        debug_mode: bool = False,
        rank_ratio_proj: float = 0.1,
        rank_ratio_res: float = 0.05,
        dbc_threshold: float = 1e-6,
        dbc_iterations: int = 1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.eps = eps
        self.alpha = alpha
        self.init_tau = init_tau
        self.sr_ratio = sr_ratio
        self.use_autopoietic = use_autopoietic
        self.batch_first = batch_first
        self.res_scale = 1.0

        # 查询、键、值的线性变换
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # 自生成变换网络 - 使用卷积层处理注意力矩阵
        hidden_dim = max(16, embed_dim // sr_ratio)
        self.sr_conv1 = nn.Conv2d(1, hidden_dim, kernel_size=1)
        self.sr_layernorm = nn.LayerNorm([hidden_dim])
        self.sr_conv2 = nn.Conv2d(hidden_dim, 1, kernel_size=1)

        # 可学习的温度参数
        self.tau = nn.Parameter(torch.ones(1) * init_tau)

        # 创建dropout层
        self.dropout_layer = nn.Dropout(dropout)

        self._reset_parameters()
        self.res_scale = 1.0

        self.debug_mode = debug_mode # 保存状态

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.xavier_uniform_(self.sr_conv1.weight)
        nn.init.xavier_uniform_(self.sr_conv2.weight)
        nn.init.constant_(self.q_proj.bias, 0.)
        nn.init.constant_(self.k_proj.bias, 0.)
        nn.init.constant_(self.v_proj.bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)
        nn.init.constant_(self.sr_conv1.bias, 0.)
        nn.init.constant_(self.sr_conv2.bias, 0.)

    def log_debug(self, message: str):
            # 【修改点】如果不开启 debug 模式，直接跳过，绝不执行 IO
            if not getattr(self, 'debug_mode', False):
                return
            
            # 只有开启了才写文件
            try:
                with open(DEBUG_LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(message + "\n")
            except Exception:
                pass

    def autopoietic_transform(
        self, 
        attention_scores: torch.Tensor,
        attn_mask: torch.Tensor = None
    ) -> torch.Tensor:
        
        # 1. 给统计代码加上“阀门”
        if self.debug_mode:
            debug_lines = []
            debug_lines.append("...")
            min_val = attention_scores.min().item() # 只有开启debug才执行同步

        """
        自生成变换过程：对输入的注意力分数进行一系列变换。
        这里添加详细的统计信息打印，并写入DEBUG_LOG_FILE。
        """
        # 第一步：打印输入attention_scores统计
        debug_lines = []  # 初始化变量防止下面报错
        
        # 加上阀门！
        if getattr(self, 'debug_mode', False):
            # 第一步：打印输入attention_scores统计
            debug_lines.append("\n[autopoietic_transform] >>>>>>>>> ENTER FUNCTION <<<<<<<<")
            
            # 【重点】把下面这些会卡顿的代码统统缩进进来！
            min_val = attention_scores.min().item()
            max_val = attention_scores.max().item()
            mean_val = attention_scores.mean().item()
            std_val = attention_scores.std().item()
            has_nan = torch.isnan(attention_scores).any().item()
            has_inf = torch.isinf(attention_scores).any().item()

            debug_lines.append(
                f"[Input Stats] shape={list(attention_scores.shape)} "
                f"min={min_val:.4f}, max={max_val:.4f}, mean={mean_val:.4f}, std={std_val:.4f}, "
                f"NaN={has_nan}, Inf={has_inf}"
            )

        # 如果不使用自生成机制，直接返回
        if not self.use_autopoietic:
            debug_lines.append("[Info] use_autopoietic=False, skipping transform.")
            self.log_debug("\n".join(debug_lines))
            return attention_scores

        # 对输入进行 clamp / nan_to_num 以保证数值安全
        attention_scores = torch.nan_to_num(attention_scores, nan=0.0, posinf=10.0, neginf=-10.0)
        attention_scores = torch.clamp(attention_scores, min=-15.0, max=15.0)

        # 开始变换
        original_scores = attention_scores.clone()
        batch_size, num_heads, seq_len1, seq_len2 = attention_scores.shape
        transformed_batch_list = []

        for b in range(batch_size):
            batch_scores = attention_scores[b]  # shape: [num_heads, seq_len1, seq_len2]
            mean_attention = batch_scores.mean(dim=0)  # [seq_len1, seq_len2]

            # 记录一下batch_scores统计
            b_min = batch_scores.min().item()
            b_max = batch_scores.max().item()
            b_mean = batch_scores.mean().item()
            b_std = batch_scores.std().item()
            debug_lines.append(
                f"[Batch {b}] batch_scores stats: min={b_min:.4f}, max={b_max:.4f}, "
                f"mean={b_mean:.4f}, std={b_std:.4f}"
            )

            # eps处理
            eps_safe = torch.clamp(torch.tensor(self.eps, device=attention_scores.device), min=0.05, max=0.8)
            scaled_attention = torch.clamp(mean_attention, min=-8.0, max=8.0) * eps_safe
            scaled_attention = torch.clamp(scaled_attention, min=-10.0, max=10.0)

            # 卷积映射
            try:
                reshaped_attn = scaled_attention.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len1, seq_len2]
                hidden_attn = self.sr_conv1(reshaped_attn)
                hidden_attn = F.relu(torch.clamp(hidden_attn, min=-5.0, max=5.0))
                autopoietic_attn = self.sr_conv2(hidden_attn)
                autopoietic_attn = autopoietic_attn.squeeze(0).squeeze(0)  # [seq_len1, seq_len2]
                autopoietic_attn = torch.clamp(autopoietic_attn, min=-5.0, max=5.0)
            except Exception as e:
                debug_lines.append(
                    f"[Batch {b}] 卷积映射出错: {e}, 使用平滑替代"
                )
                kernel_size = min(5, min(seq_len1, seq_len2))
                if kernel_size % 2 == 0:
                    kernel_size -= 1
                if kernel_size >= 3:
                    try:
                        gaussian_kernel = torch.ones((1, 1, kernel_size, kernel_size), device=attention_scores.device)
                        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
                        smoothed = F.conv2d(
                            scaled_attention.unsqueeze(0).unsqueeze(0),
                            gaussian_kernel,
                            padding=kernel_size//2
                        )
                        autopoietic_attn = smoothed.squeeze(0).squeeze(0)
                        autopoietic_attn = torch.clamp(autopoietic_attn, min=-5.0, max=5.0)
                    except Exception:
                        autopoietic_attn = torch.tanh(scaled_attention * 0.5) * 2.0
                else:
                    autopoietic_attn = torch.tanh(scaled_attention * 0.5) * 2.0

            # 检查 NaN/Inf
                autopoietic_attn = torch.nan_to_num(autopoietic_attn, nan=0.0, posinf=2.0, neginf=-2.0)
                autopoietic_attn = torch.clamp(autopoietic_attn, min=-5.0, max=5.0)

            # 处理掩码
            mean_padding_mask = None
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    mean_padding_mask = attn_mask.clone()
                elif attn_mask.dim() == 3 and attn_mask.size(0) == batch_size:
                    mean_padding_mask = attn_mask[b]
                elif attn_mask.dim() == 4:
                    mean_padding_mask = attn_mask[b, 0]
                if mean_padding_mask is not None and mean_padding_mask.dtype != torch.bool:
                    mean_padding_mask = mean_padding_mask.to(torch.bool)


            # 🚀 左旋平滑替换泰勒展开
            # 传统: taylor = 1.0 + α·Δ  (遇尖点会炸)
            # 左旋: taylor = 1.0 + g(φ)·Δ  (遇尖点自动缩小步长)

            # 计算尖点强度（基于二范数）
            base_value = torch.ones_like(autopoietic_attn)
            delta_attn = autopoietic_attn  # 增量部分

            # 计算相对变化强度
            norm_base = torch.norm(base_value, p=2, dim=-1, keepdim=True) + 1e-8
            norm_delta = torch.norm(delta_attn, p=2, dim=-1, keepdim=True)
            spike_strength = norm_delta / norm_base

            # 缓冲角: φ = α·softplus(s - τ)
            left_spin_alpha = 0.5
            left_spin_tau = 0.3
            phi = left_spin_alpha * F.softplus(spike_strength - left_spin_tau)

            # 门控函数: g(φ) = 1/√(1+φ²)
            gate = 1.0 / torch.sqrt(1.0 + phi ** 2)

            # 应用左旋平滑
            scale_factor = 50.0
            scaled_attn_2 = autopoietic_attn * scale_factor * gate  # 🔥 关键替换
            alpha_safe = 0.05
            taylor_expanded = 1.0 + alpha_safe * scaled_attn_2
            taylor_expanded = torch.clamp(taylor_expanded, min=0.5, max=1.5)

            if mean_padding_mask is not None:
                taylor_expanded = torch.where(
                    mean_padding_mask,
                    torch.ones_like(taylor_expanded),
                    taylor_expanded
                )

            # Sigmoid平滑
            sigmoid_smoothed = torch.sigmoid(taylor_expanded)

            # 模糊概率
            try:
                safe_mean = torch.clamp(mean_attention, min=-10.0, max=10.0)
                attn_probs = F.softmax(safe_mean, dim=-1)
                epsilon = 1e-6
                H = -attn_probs * torch.log(attn_probs + epsilon)
                lambda_param = torch.tensor(3.0, device=attention_scores.device)
                F_matrix = F.softmax(lambda_param * H, dim=-1)
                F_matrix = torch.nan_to_num(F_matrix, nan=1.0/seq_len2)
            except Exception as e:
                debug_lines.append(f"[Batch {b}] 模糊概率计算出错: {e}")
                F_matrix = torch.ones_like(mean_attention) / seq_len2

            transformed = sigmoid_smoothed * F_matrix

            # 能量平衡
            try:
                energy_original = torch.norm(mean_attention, p='fro') + 1e-4
                energy_transformed = torch.norm(transformed, p='fro') + 1e-4
                gamma = torch.clamp(energy_original / energy_transformed, min=0.8, max=1.2)
                transformed = gamma * transformed
            except Exception as e:
                debug_lines.append(f"[Batch {b}] 能量平衡计算出错: {e}")

            # 动态标准差调整
            try:
                t_mean = transformed.mean()
                o_mean = mean_attention.mean()
                min_var = 1e-2
                t_var = torch.clamp(((transformed - t_mean) ** 2).mean(), min=min_var)
                o_var = torch.clamp(((mean_attention - o_mean) ** 2).mean(), min=min_var)
                t_std = torch.sqrt(t_var)
                o_std = torch.sqrt(o_var)
                gamma_dyn = torch.clamp(o_std / t_std, min=0.8, max=1.2)
                available_range = torch.clamp(torch.abs(mean_attention).max(), min=1.0, max=10.0)
                std_multiplier = 0.3 * (1.0 / torch.log(1.0 + available_range))
                std_multiplier = torch.clamp(std_multiplier, min=0.1, max=0.5)
                centered = transformed - t_mean
                scaled = std_multiplier * gamma_dyn * centered
                scaled_transform = t_mean + scaled
                entropy = torch.mean(H)
                max_entropy = -torch.log(torch.tensor(1.0/seq_len2, device=attention_scores.device))
                normalized_entropy = entropy / max_entropy
                base_ratio = 0.4
                entropy_factor = torch.clamp(normalized_entropy, min=0.0, max=0.4)
                residual_ratio = base_ratio + entropy_factor
                final_scores = (1 - residual_ratio) * scaled_transform + residual_ratio * mean_attention
            except Exception as e:
                debug_lines.append(f"[Batch {b}] 标准差调整出错: {e}")
                final_scores = 0.5 * transformed + 0.5 * mean_attention

            # 自适应温度调节
            try:
                base_tau = torch.clamp(self.tau, min=1.0, max=1.5)
                values = final_scores.reshape(-1)
                q_75 = torch.quantile(values, 0.75)
                q_25 = torch.quantile(values, 0.25)
                score_range = torch.clamp(q_75 - q_25, min=0.5, max=5.0)
                adaptive_tau = base_tau * (1.0 + 0.1 * torch.log1p(score_range))
                adaptive_tau = torch.clamp(adaptive_tau, min=1.0, max=2.0)
                final_scores = final_scores / adaptive_tau
                final_scores = torch.clamp(final_scores, min=-15.0, max=15.0)
            except Exception as e:
                debug_lines.append(f"[Batch {b}] 温度调节出错: {e}")
                final_scores = torch.clamp(final_scores / 1.0, min=-10.0, max=10.0)

            # 检查异常值比例
            abnormal_mask = torch.isnan(final_scores) | torch.isinf(final_scores)
            abnormal_ratio = abnormal_mask.float().mean().item()
            if abnormal_ratio > 0.2:
                debug_lines.append(f"[Batch {b}] 警告: 异常比例过高({abnormal_ratio*100:.2f}%), 使用安全回退 -> mean_attention")
                final_scores = torch.clamp(mean_attention, min=-10.0, max=10.0)
            else:
                final_scores = torch.nan_to_num(final_scores, nan=0.0)
                final_scores = torch.clamp(final_scores, min=-10.0, max=10.0)

            batch_transform = []
            for h in range(batch_scores.size(0)):
                head_base = final_scores.clone()
                head_delta = batch_scores[h] - mean_attention
                delta_scale = 0.2
                head_specific = head_base + delta_scale * head_delta
                head_specific = torch.clamp(head_specific, min=-15.0, max=15.0)
                batch_transform.append(head_specific)
            if len(batch_transform) > 0:
                batch_transform = torch.stack(batch_transform)
            else:
                batch_transform = final_scores.unsqueeze(0)
            transformed_batch_list.append(batch_transform)

        transform_scores = torch.stack(transformed_batch_list)

        # 处理全局 attn_mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            transform_scores = transform_scores + self.res_scale * attn_mask

        transform_scores = torch.nan_to_num(transform_scores, nan=0.0)
        transform_scores = torch.clamp(transform_scores, min=-30.0, max=30.0)

        # 打印 transform_scores 统计
        final_min = transform_scores.min().item()
        final_max = transform_scores.max().item()
        final_mean = transform_scores.mean().item()
        final_std = transform_scores.std().item()
        final_has_nan = torch.isnan(transform_scores).any().item()
        final_has_inf = torch.isinf(transform_scores).any().item()
        debug_lines.append(
            f"[Output Stats] transform_scores shape={list(transform_scores.shape)} "
            f"min={final_min:.4f}, max={final_max:.4f}, mean={final_mean:.4f}, std={final_std:.4f}, "
            f"NaN={final_has_nan}, Inf={final_has_inf}"
        )
        debug_lines.append("[autopoietic_transform] >>>>>>>>> EXIT FUNCTION <<<<<<<<")

        # 将所有调试信息写入日志
        self.log_debug("\n".join(debug_lines))
        return transform_scores

    def forward(
        self, query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        need_weights: bool = True
        ) -> tuple:
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        batch_size, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = q * self.scaling
        attn_weights = torch.matmul(q, k.transpose(-2, -1))

        if key_padding_mask is not None:
            key_padding_mask_expanded = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(key_padding_mask_expanded, float('-inf'))

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_weights = attn_weights + self.res_scale * attn_mask

        attn_weights = self.autopoietic_transform(attn_weights, attn_mask)
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout_layer(attn_probs)

        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        if not self.batch_first:
            attn_output = attn_output.transpose(0, 1)

        if need_weights:
            avg_weights = attn_probs.mean(dim=1)
            return attn_output, avg_weights
        return attn_output, None


class APTEncoderLayer(nn.Module):
    """
    APT编码器层
    集成自生成注意力机制 + 左旋平滑残差连接
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
        left_spin_beta: float = 0.7
    ):
        super().__init__()

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

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # 激活函数
        self.activation = F.gelu if activation == "gelu" else F.relu

        # 配置
        self.batch_first = batch_first
        self.use_left_spin = use_left_spin

        # 🚀 左旋平滑残差连接（替换传统泰勒展开）
        if use_left_spin:
            self.left_spin_attn = LeftSpinResidual(
                alpha=left_spin_alpha,
                tau=left_spin_tau,
                beta=left_spin_beta,
                gate_type='normalized',
                adaptive=True
            )
            self.left_spin_ffn = LeftSpinResidual(
                alpha=left_spin_alpha,
                tau=left_spin_tau,
                beta=left_spin_beta,
                gate_type='normalized',
                adaptive=True
            )
        else:
            self.left_spin_attn = None
            self.left_spin_ffn = None

        # 【新增这行】默认关闭 debug，防止拖慢速度
        self.debug_mode = getattr(config, 'debug_mode', False) if 'config' in locals() else False
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        编码器层前向传播（集成左旋平滑）

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

        # 替换: src = src + src2  →  src = LeftSpin(src, src2)
        if self.use_left_spin and self.left_spin_attn is not None:
            src = self.left_spin_attn(src, src2_dropout)
        else:
            # 降级为标准残差
            src = src + src2_dropout

        src = self.norm1(src)

        # 🚀 前馈网络子层（左旋平滑残差连接）
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2_dropout = self.dropout2(src2)

        # 替换: src = src + src2  →  src = LeftSpin(src, src2)
        if self.use_left_spin and self.left_spin_ffn is not None:
            src = self.left_spin_ffn(src, src2_dropout)
        else:
            # 降级为标准残差
            src = src + src2_dropout

        src = self.norm2(src)

        return src


class APTDecoderLayer(nn.Module):
    """
    APT解码器层
    集成自生成注意力机制 + 左旋平滑残差连接
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
        left_spin_beta: float = 0.7
    ):
        super().__init__()

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

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # 激活函数
        self.activation = F.gelu if activation == "gelu" else F.relu

        # 配置
        self.batch_first = batch_first
        self.use_left_spin = use_left_spin

        # 🚀 左旋平滑残差连接（3个子层）
        if use_left_spin:
            self.left_spin_self_attn = LeftSpinResidual(
                alpha=left_spin_alpha,
                tau=left_spin_tau,
                beta=left_spin_beta,
                gate_type='normalized',
                adaptive=True
            )
            self.left_spin_cross_attn = LeftSpinResidual(
                alpha=left_spin_alpha,
                tau=left_spin_tau,
                beta=left_spin_beta,
                gate_type='normalized',
                adaptive=True
            )
            self.left_spin_ffn = LeftSpinResidual(
                alpha=left_spin_alpha,
                tau=left_spin_tau,
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
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        解码器层前向传播（集成左旋平滑）

        参数:
            tgt: 目标序列 [seq_len, batch_size, d_model] 或 [batch_size, seq_len, d_model]
            memory: 编码器输出 同上
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

        # 🚀 编码器-解码器注意力子层（左旋平滑残差连接）
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

        # 🚀 前馈网络子层（左旋平滑残差连接）
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt2_dropout = self.dropout3(tgt2)

        if self.use_left_spin and self.left_spin_ffn is not None:
            tgt = self.left_spin_ffn(tgt, tgt2_dropout)
        else:
            tgt = tgt + tgt2_dropout

        tgt = self.norm3(tgt)

        return tgt


class APTModelConfiguration:
    """APT模型配置类（集成左旋平滑）"""
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

        # 创建编码器层
        encoder_layers = []
        for _ in range(config.num_encoder_layers):
            encoder_layers.append(
                APTEncoderLayer(
                    d_model=config.d_model,
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
                    left_spin_beta=left_spin_beta
                )
            )

        # 创建解码器层
        decoder_layers = []
        for _ in range(config.num_decoder_layers):
            decoder_layers.append(
                APTDecoderLayer(
                    d_model=config.d_model,
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
                    left_spin_beta=left_spin_beta
                )
            )
        
        # 编码器和解码器
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.decoder_layers = nn.ModuleList(decoder_layers)
        
        # 最终层归一化
        self.encoder_norm = nn.LayerNorm(config.d_model)
        self.decoder_norm = nn.LayerNorm(config.d_model)
        
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
        
        # 根据return_dict参数决定返回形式
        if return_dict:
            return {
                "logits": logits,
                "encoder_output": memory,
                "decoder_output": decoder_output
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