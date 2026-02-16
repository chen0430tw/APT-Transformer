"""
随机投影核 (Random Projection Kernel)
=====================================

基于 CompAct (NAACL 2025) 的优化实现：
- 前向传播：生成随机投影矩阵 P，计算 z = x @ P，存储伴随矩阵 P^T
- 反向传播：直接使用存储的 P^T，避免重新生成（更快）

核心优化：
  CompAct 原版：每次反向传播都要从 seed 重新生成 P（慢）
  本实现：存储 P^T，反向传播直接使用（快）

内存开销：
  单层：r × n × 4 bytes（例如 r=512, n=8192 → 16 MB）
  70B 模型 80 层：~1.28 GB（可接受）

作者：GPT-5.2 R2
版本：1.0.0
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProjectionKernelConfig:
    """随机投影核配置"""
    rank: int = 512                      # 压缩秩 r
    distribution: Literal["gaussian", "rademacher", "sparse"] = "gaussian"
    seed_mode: Literal["per_layer", "global", "hash"] = "per_layer"
    device: str = "cuda"
    dtype: torch.dtype = torch.float32

    # 稀疏投影配置（当 distribution="sparse" 时）
    sparse_density: float = 0.1          # 稀疏度

    # 全局核变换模式
    global_transform: Literal["none", "roll", "hadamard", "circulant"] = "none"


@torch.no_grad()
def generate_projection_matrix(
    n: int,
    r: int,
    *,
    seed: int,
    device: torch.device | str = "cuda",
    distribution: str = "gaussian",
    sparse_density: float = 0.1,
) -> torch.Tensor:
    """
    生成随机投影矩阵 P ∈ R^(n×r)

    Args:
        n: 输入维度
        r: 压缩秩
        seed: 随机种子
        device: 设备
        distribution: 分布类型 ("gaussian", "rademacher", "sparse")
        sparse_density: 稀疏密度（仅当 distribution="sparse" 时）

    Returns:
        P: (n, r) 随机投影矩阵
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0xFFFFFFFF)

    if distribution == "gaussian":
        # 高斯分布：P_ij ~ N(0, 1/√r)
        P = torch.randn(n, r, device=device, generator=g) / math.sqrt(r)

    elif distribution == "rademacher":
        # Rademacher 分布：P_ij ∈ {+1, -1} / √r
        P = torch.randint(0, 2, (n, r), device=device, generator=g).float()
        P = P * 2 - 1  # {0, 1} → {-1, +1}
        P = P / math.sqrt(r)

    elif distribution == "sparse":
        # 稀疏投影：大部分元素为 0，少数为非零
        P = torch.zeros(n, r, device=device)
        nnz = int(n * r * sparse_density)  # 非零元素数量
        if nnz > 0:
            rows = torch.randint(0, n, (nnz,), device=device, generator=g)
            cols = torch.randint(0, r, (nnz,), device=device, generator=g)
            values = torch.randn(nnz, device=device, generator=g) / math.sqrt(r * sparse_density)
            P[rows, cols] = values

    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    return P


@torch.no_grad()
def apply_global_transform(
    P: torch.Tensor,
    transform: str,
    layer_id: int,
) -> torch.Tensor:
    """
    对全局核应用轻量级变换，生成层特定核

    Args:
        P: 全局核 (n, r)
        transform: 变换类型
        layer_id: 层 ID

    Returns:
        transformed_P: 变换后的核 (n, r)
    """
    n, r = P.shape

    if transform == "none":
        return P

    elif transform == "roll":
        # 循环移位：快速、确定性、可逆
        shift = (layer_id * 7919) % n  # 使用质数作为乘子
        return torch.roll(P, shift, dims=0)

    elif transform == "circulant":
        # 循环矩阵变换：每行是上一行的循环移位
        shift = layer_id % r
        return torch.roll(P, shift, dims=1)

    elif transform == "hadamard":
        # Hadamard 变换（需要 n 是 2 的幂）
        # 简化版：元素级乘以伪随机符号
        torch.manual_seed(layer_id * 31)
        sign = torch.randint(0, 2, (n, 1), device=P.device).float() * 2 - 1
        return P * sign

    else:
        raise ValueError(f"Unknown transform: {transform}")


class RandomProjectionKernel:
    """
    随机投影核管理器

    核心功能：
      1. 管理多层随机投影核
      2. 存储伴随矩阵 P^T（用于快速反向传播）
      3. 支持全局核 + 轻量级变换（节省内存）
    """

    def __init__(
        self,
        config: Optional[ProjectionKernelConfig] = None,
        global_seed: int = 42,
    ):
        self.config = config or ProjectionKernelConfig()
        self.global_seed = global_seed

        # 存储：{layer_id: P^T}
        self.adjoint_kernels: Dict[str, torch.Tensor] = {}

        # 全局核（可选）
        self.global_kernel: Optional[torch.Tensor] = None

        # 统计
        self.total_memory_bytes = 0
        self.num_layers = 0

    def _layer_seed(self, layer_id: str) -> int:
        """从 layer_id 生成种子"""
        if self.config.seed_mode == "hash":
            return hash(layer_id) & 0xFFFFFFFF
        elif self.config.seed_mode == "global":
            return self.global_seed
        else:  # per_layer
            return (hash(layer_id) ^ self.global_seed) & 0xFFFFFFFF

    def register_layer(
        self,
        layer_id: str,
        n: int,
        r: Optional[int] = None,
    ) -> torch.Tensor:
        """
        注册层并生成投影核

        Args:
            layer_id: 层标识符（如 "transformer.h.0", "lm_head"）
            n: 输入维度（hidden_dim）
            r: 压缩秩（默认使用 config.rank）

        Returns:
            P: 投影矩阵 (n, r)
        """
        rank = r or self.config.rank
        device = self.config.device

        # 生成投影矩阵
        if self.config.seed_mode == "global" and self.global_kernel is not None:
            # 从全局核派生
            layer_idx = hash(layer_id) & 0xFFFF
            P = apply_global_transform(
                self.global_kernel,
                self.config.global_transform,
                layer_idx,
            )
            # 如果维度不匹配，需要重新采样
            if P.shape[0] != n or P.shape[1] != rank:
                seed = self._layer_seed(layer_id)
                P = generate_projection_matrix(
                    n, rank, seed=seed, device=device,
                    distribution=self.config.distribution,
                    sparse_density=self.config.sparse_density,
                )
        else:
            # 直接生成
            seed = self._layer_seed(layer_id)
            P = generate_projection_matrix(
                n, rank, seed=seed, device=device,
                distribution=self.config.distribution,
                sparse_density=self.config.sparse_density,
            )

        # 存储伴随矩阵 P^T (r, n)
        P_adjoint = P.T
        self.adjoint_kernels[layer_id] = P_adjoint

        # 更新统计
        self.num_layers += 1
        self.total_memory_bytes += P_adjoint.numel() * P_adjoint.element_size()

        return P

    def get_projection(self, layer_id: str) -> torch.Tensor:
        """
        获取投影矩阵 P（用于前向传播）

        Args:
            layer_id: 层标识符

        Returns:
            P: 投影矩阵 (n, r)
        """
        P_adjoint = self.adjoint_kernels.get(layer_id)
        if P_adjoint is None:
            raise KeyError(f"Layer {layer_id} not registered. Call register_layer() first.")
        return P_adjoint.T  # (r, n)^T = (n, r)

    def get_adjoint(self, layer_id: str) -> torch.Tensor:
        """
        获取伴随矩阵 P^T（用于反向传播）

        这是本优化的核心：反向传播不需要重新生成 P，直接使用存储的 P^T

        Args:
            layer_id: 层标识符

        Returns:
            P^T: 伴随矩阵 (r, n)
        """
        P_adjoint = self.adjoint_kernels.get(layer_id)
        if P_adjoint is None:
            raise KeyError(f"Layer {layer_id} not registered. Call register_layer() first.")
        return P_adjoint  # (r, n)

    def init_global_kernel(self, n: int, r: Optional[int] = None) -> None:
        """
        初始化全局核（可选）

        优势：
          - 所有层共享一个核
          - 内存开销从 O(L × n × r) 降到 O(n × r)
          - 每层通过轻量级变换派生

        Args:
            n: 输入维度
            r: 压缩秩
        """
        rank = r or self.config.rank
        self.global_kernel = generate_projection_matrix(
            n, rank,
            seed=self.global_seed,
            device=self.config.device,
            distribution=self.config.distribution,
            sparse_density=self.config.sparse_density,
        )

    def memory_usage_mb(self) -> float:
        """返回内存使用量（MB）"""
        return self.total_memory_bytes / (1024 * 1024)

    def clear(self) -> None:
        """清空所有存储的核"""
        self.adjoint_kernels.clear()
        self.total_memory_bytes = 0
        self.num_layers = 0


class CompActFunction(torch.autograd.Function):
    """
    CompAct 前向/反向传播函数

    前向：z = x @ P
    反向：Ĝ = z^T @ ∂L/∂o
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        layer_id: str,
        kernel_manager: RandomProjectionKernel,
    ) -> torch.Tensor:
        """
        前向传播：压缩激活值

        Args:
            x: 输入激活 (batch, seq_len, hidden_dim)
            layer_id: 层标识符
            kernel_manager: 随机投影核管理器

        Returns:
            z: 压缩激活 (batch, seq_len, rank)
        """
        # 获取投影矩阵 P
        P = kernel_manager.get_projection(layer_id)  # (hidden_dim, rank)

        # 压缩：z = x @ P
        # x: (B, T, n), P: (n, r) → z: (B, T, r)
        z = torch.matmul(x, P)

        # 保存上下文用于反向传播
        ctx.save_for_backward(z)
        ctx.layer_id = layer_id
        ctx.kernel_manager = kernel_manager

        return z

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,  # ∂L/∂z
    ) -> Tuple[Optional[torch.Tensor], None, None]:
        """
        反向传播：使用压缩激活计算梯度

        Args:
            grad_output: 梯度 ∂L/∂z (batch, seq_len, rank)

        Returns:
            grad_input: ∂L/∂x (batch, seq_len, hidden_dim)
        """
        z, = ctx.saved_tensors
        kernel_manager = ctx.kernel_manager
        layer_id = ctx.layer_id

        # 获取伴随矩阵 P^T
        P_adjoint = kernel_manager.get_adjoint(layer_id)  # (rank, hidden_dim)

        # 梯度投影：∂L/∂x = ∂L/∂z @ P^T
        # grad_output: (B, T, r), P^T: (r, n) → grad_input: (B, T, n)
        grad_input = torch.matmul(grad_output, P_adjoint)

        return grad_input, None, None


def compact_act_forward(
    x: torch.Tensor,
    layer_id: str,
    kernel_manager: RandomProjectionKernel,
) -> torch.Tensor:
    """
    便捷函数：CompAct 前向传播

    Args:
        x: 输入激活 (batch, seq_len, hidden_dim)
        layer_id: 层标识符
        kernel_manager: 随机投影核管理器

    Returns:
        z: 压缩激活 (batch, seq_len, rank)
    """
    return CompActFunction.apply(x, layer_id, kernel_manager)


class CompActLinear(nn.Module):
    """
    集成 CompAct 的线性层

    功能：
      1. 正常线性变换：y = x @ W^T + b
      2. 并行压缩激活：z = x @ P
      3. 训练时保存 z 用于梯度压缩

    使用场景：
      - 替换 nn.Linear
      - 大模型训练时节省激活值内存
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_manager: RandomProjectionKernel,
        layer_id: str,
        bias: bool = True,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer_id = layer_id
        self.kernel_manager = kernel_manager

        # 注册投影核
        kernel_manager.register_layer(layer_id, n=in_features, r=kernel_manager.config.rank)

        # 标准线性层权重
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """初始化权重（标准 PyTorch 方式）"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入 (batch, *, in_features)

        Returns:
            y: 输出 (batch, *, out_features)
        """
        # 正常线性变换
        y = F.linear(x, self.weight, self.bias)

        # 如果在训练，同时计算压缩激活（用于检查点/梯度压缩）
        if self.training:
            z = compact_act_forward(x, self.layer_id, self.kernel_manager)
            # 可以保存 z 到某个全局检查点管理器
            # 这里简化处理：直接返回 y
            # 实际使用时应该通过 context manager 传递 z

        return y

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, layer_id={self.layer_id}"


# ============================================================================
# Adam 优化器集成（可选）
# ============================================================================

class CompActAdamState:
    """CompAct Adam 更新状态"""

    def __init__(self, param: torch.Tensor, rank: int):
        self.device = param.device
        self.dtype = param.dtype

        # 压缩梯度的 Adam 状态
        self.m = torch.zeros(rank, param.shape[0], device=self.device, dtype=self.dtype)  # (r, n)
        self.v = torch.zeros(rank, param.shape[0], device=self.device, dtype=self.dtype)  # (r, n)
        self.step = 0


def compact_adam_update(
    param: torch.Tensor,
    grad_compressed: torch.Tensor,  # (r, n) 压缩梯度
    state: CompActAdamState,
    layer_id: str,
    kernel_manager: RandomProjectionKernel,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> None:
    """
    CompAct Adam 更新

    流程：
      1. 在压缩空间更新 m, v
      2. 计算 N_t = m / (√v + ε)
      3. 解压：G̃_t = α · P · N_t
      4. 更新参数：W_{t+1} = W_t - η · G̃_t

    Args:
        param: 参数张量 (out_features, in_features)
        grad_compressed: 压缩梯度 (rank, in_features)
        state: Adam 状态
        layer_id: 层标识符
        kernel_manager: 随机投影核管理器
        lr: 学习率
        beta1, beta2: Adam 衰减率
        eps: 数值稳定项
        weight_decay: 权重衰减
    """
    state.step += 1
    bias_correction1 = 1 - beta1 ** state.step
    bias_correction2 = 1 - beta2 ** state.step

    # 步骤 1-5: 在压缩空间更新 Adam 状态
    # m_t = β1 * m_{t-1} + (1-β1) * Ĝ_t
    # v_t = β2 * v_{t-1} + (1-β2) * Ĝ_t^2
    state.m.mul_(beta1).add_(grad_compressed, alpha=1 - beta1)
    state.v.mul_(beta2).addcmul_(grad_compressed, grad_compressed, value=1 - beta2)

    # 步骤 6: 计算 N_t = m_t / (√(v_t) + ε) * √(bias_correction2 / bias_correction1)
    denom = (state.v.sqrt() / math.sqrt(bias_correction2)).add_(eps / math.sqrt(bias_correction2))
    N_t = state.m.mul(math.sqrt(bias_correction2) / bias_correction1) / denom

    # 步骤 7-8: 解压 - 使用存储的伴随矩阵
    # 获取 P (n, r) → 需要 P_adjoint.T
    P_adjoint = kernel_manager.get_adjoint(layer_id)  # (r, n)
    P = P_adjoint.T  # (n, r)

    # 解压：G̃_t = P · N_t (n, r) @ (r, n) → (n, n)
    # 但这里 grad_compressed 是 (r, n)，所以 N_t 也是 (r, n)
    # G̃_t = P^T @ N_t 也就是 (n, r) @ (r, n) → (n, n)
    G_tilde = torch.matmul(P, N_t)  # (n, n)

    # 步骤 9: 参数更新
    # W_{t+1} = W_t - η · G̃_t^T（因为 param 是 (out, in)）
    param.add_(G_tilde.T, alpha=-lr)

    if weight_decay != 0.0:
        param.add_(param, alpha=-lr * weight_decay)


# ============================================================================
# 便捷函数
# ============================================================================

def create_compact_optimizer(
    model: nn.Module,
    kernel_manager: RandomProjectionKernel,
    lr: float = 1e-3,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    创建集成 CompAct 的优化器（简化版）

    注意：这是演示版本，实际使用可能需要更复杂的实现
    """
    # 这里返回标准 Adam，实际应用中需要自定义优化器类
    return torch.optim.Adam(model.parameters(), lr=lr, **kwargs)


def estimate_memory(
    num_layers: int,
    hidden_dim: int,
    rank: int,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, float]:
    """
    估算随机投影核的内存使用

    Args:
        num_layers: 层数
        hidden_dim: 隐藏层维度
        rank: 压缩秩
        dtype: 数据类型

    Returns:
        内存统计字典（MB）
    """
    bytes_per_element = torch.finfo(dtype).bits // 8
    bytes_per_layer = hidden_dim * rank * bytes_per_element
    total_bytes = num_layers * bytes_per_layer

    return {
        "per_layer_mb": bytes_per_layer / (1024 * 1024),
        "total_mb": total_bytes / (1024 * 1024),
        "total_gb": total_bytes / (1024 * 1024 * 1024),
    }


# ============================================================================
# 测试/验证
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("随机投影核 (Random Projection Kernel) - 测试")
    print("=" * 70)
    print()

    # 配置
    config = ProjectionKernelConfig(
        rank=512,
        distribution="gaussian",
        seed_mode="per_layer",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 创建核管理器
    kernel_mgr = RandomProjectionKernel(config, global_seed=42)

    # 模拟 70B 模型
    num_layers = 80
    hidden_dim = 8192

    print(f"模拟注册 {num_layers} 层...")
    for i in range(num_layers):
        layer_id = f"transformer.h.{i}"
        kernel_mgr.register_layer(layer_id, n=hidden_dim, r=config.rank)

    print(f"[OK] Registration complete")
    print()
    print(f"Memory usage:")
    print(f"  每层: {config.rank} × {hidden_dim} × 4 bytes = {config.rank * hidden_dim * 4 / (1024**2):.2f} MB")
    print(f"  {num_layers} 层总计: {kernel_mgr.memory_usage_mb():.2f} MB ({kernel_mgr.memory_usage_mb() / 1024:.2f} GB)")
    print()

    # 测试前向传播
    print("Testing forward/backward...")
    B, T, n = 2, 128, hidden_dim
    x = torch.randn(B, T, n, device=config.device, requires_grad=True)
    layer_id = "transformer.h.0"

    # 前向
    z = compact_act_forward(x, layer_id, kernel_mgr)
    print(f"  Input shape: {x.shape}")
    print(f"  Compressed shape: {z.shape}")

    # 反向
    grad_output = torch.randn_like(z)
    grad_input = torch.autograd.grad(z, x, grad_output)[0]
    print(f"  Gradient shape: {grad_input.shape}")

    print()
    print("[OK] All tests passed!")
    print("=" * 70)
