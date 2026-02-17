"""
LECAC (Low-Entropy Compensation for Activated Computation)
=======================================================
实现 ActNN 风格的自定义 Linear 层，支持 INT8 激活值量化训练

核心思路：
1. Forward: 正常计算 + 量化存储激活值
2. Backward: 反量化恢复 + 手动计算梯度
3. 使用 Straight-Through Estimator (STE) 处理量化
"""
import os
import sys

_BASE = "D:/APT-Transformer"
os.environ["TORCH_HOME"] = f"{_BASE}/.torch_cache"
os.environ["TEMP"] = f"{_BASE}/.temp"
os.makedirs(_BASE + "/.torch_cache", exist_ok=True)
os.makedirs(_BASE + "/.temp", exist_ok=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================================
# 量化/反量化函数
# ============================================================================

def quantize_int8_symmetric(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对称 INT8 量化

    Args:
        x: 输入张量 (FP32)

    Returns:
        (x_int8, scale): INT8 张量和缩放因子
    """
    with torch.no_grad():
        # 对称量化: [-127, 127]
        amax = x.abs().max()
        scale = torch.clamp(amax / 127.0, min=1e-6)
        x_int8 = (x / scale).round().clamp(-127, 127).to(torch.int8)
        return x_int8, scale


def dequantize_int8_symmetric(x_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    对称 INT8 反量化

    Args:
        x_int8: INT8 张量
        scale: 缩放因子

    Returns:
        反量化后的 FP32 张量
    """
    return x_int8.float() * scale


# ============================================================================
# LECAC 自定义 Function
# ============================================================================

class LECACLinearFunction(torch.autograd.Function):
    """
    自定义 Linear Function，支持 INT8 激活值量化

    Forward: 正常计算 + 量化存储激活值
    Backward: 反量化恢复 + 手动计算梯度
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                use_int8: bool = True, alpha: float = 0.0):
        """
        Args:
            input: 输入张量 [M, K]
            weight: 权重张量 [N, K]
            bias: 偏置张量 [N]
            use_int8: 是否使用 INT8 量化
            alpha: LDBR 补偿强度
        """
        # 1. 正常计算输出
        output = torch.nn.functional.linear(input, weight, bias)

        # 2. 量化存储激活值（用于 backward）
        if use_int8:
            with torch.no_grad():
                input_int8, scale = quantize_int8_symmetric(input)
                # 计算 LDBR 补偿参数
                input_recon = dequantize_int8_symmetric(input_int8, scale)
                error_std = (input - input_recon).std()
                K = input.numel()

            # 保存用于 backward
            ctx.save_for_backward(input_int8, scale, weight, bias)
            ctx.use_int8 = True
            ctx.alpha = alpha
            ctx.error_std = error_std
            ctx.K = K
            ctx.input_shape = input.shape
        else:
            # 不使用 INT8，直接保存
            ctx.save_for_backward(input, weight, bias)
            ctx.use_int8 = False
            ctx.input_shape = input.shape

        ctx.save_input_shape = input.shape

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        手动计算梯度

        Args:
            grad_output: 输出梯度 [M, N]

        Returns:
            grad_input: 输入梯度 [M, K]
            grad_weight: 权重梯度 [N, K]
            grad_bias: 偏置梯度 [N]
        """
        if ctx.use_int8:
            # 恢复保存的张量
            input_int8, scale, weight, bias = ctx.saved_tensors

            # 反量化输入
            input_recon = dequantize_int8_symmetric(input_int8, scale)

            # LDBR 补偿（可选）
            if ctx.alpha > 0:
                with torch.no_grad():
                    dimension_balance = math.log(ctx.K + math.e)
                    noise = torch.randn_like(input_recon) * ctx.alpha
                    compensation = (ctx.error_std / dimension_balance) * noise
                input_recon = input_recon + compensation

            # 手动计算梯度
            # grad_input = grad_output @ weight
            grad_input = grad_output.mm(weight)

            # grad_weight = grad_output.T @ input_recon
            grad_weight = grad_output.t().mm(input_recon)

            # grad_bias = sum(grad_output)
            grad_bias = grad_output.sum(0) if bias is not None else None

        else:
            # 不使用 INT8，标准计算
            input, weight, bias = ctx.saved_tensors

            grad_input = grad_output.mm(weight)
            grad_weight = grad_output.t().mm(input)
            grad_bias = grad_output.sum(0) if bias is not None else None

        return grad_input, grad_weight, grad_bias


# ============================================================================
# 自定义 Linear 层
# ============================================================================

class LECACLinear(nn.Module):
    """
    支持 INT8 激活值量化的 Linear 层

    Args:
        in_features: 输入特征数
        out_features: 输出特征数
        bias: 是否使用偏置
        use_int8: 是否使用 INT8 量化激活值
        alpha: LDBR 补偿强度
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 use_int8: bool = True, alpha: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_int8 = use_int8
        self.alpha = alpha

        self.weight = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return LECACLinearFunction.apply(input, self.weight, self.bias, self.use_int8, self.alpha)

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'use_int8={self.use_int8}, alpha={self.alpha}'


# ============================================================================
# 测试
# ============================================================================

print("=" * 70)
print("LECAC INT8 量化 Linear 层测试")
print("=" * 70)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")

# 创建测试模型
in_features, out_features = 768, 768
batch_size = 4
seq_len = 128

# 1. Baseline (FP32)
print("[1] 创建 Baseline 模型 (FP32)")
baseline_linear = nn.Linear(in_features, out_features).to(device)
baseline_linear.train()

x = torch.randn(batch_size, seq_len, in_features, device=device, requires_grad=True)
y_base = baseline_linear(x)
loss_base = y_base.sum()
loss_base.backward()

print(f"  x.grad norm: {x.grad.norm().item():.6e}")
print(f"  weight.grad norm: {baseline_linear.weight.grad.norm().item():.6e}")

# 2. LECAC INT8 (无补偿)
print("\n[2] 创建 LECAC 模型 (INT8, alpha=0.0)")
lecac_linear = LECACLinear(in_features, out_features, use_int8=True, alpha=0.0).to(device)
# 使用相同的权重
lecac_linear.weight.data = baseline_linear.weight.data.clone()
if baseline_linear.bias is not None:
    lecac_linear.bias.data = baseline_linear.bias.data.clone()

x_int8 = torch.randn(batch_size, seq_len, in_features, device=device, requires_grad=True)
y_int8 = lecac_linear(x_int8)
loss_int8 = y_int8.sum()
loss_int8.backward()

print(f"  x.grad norm: {x_int8.grad.norm().item():.6e}")
print(f"  weight.grad norm: {lecac_linear.weight.grad.norm().item():.6e}")

# 计算梯度相似度
grad_input_sim = F.cosine_similarity(x.grad.flatten(), x_int8.grad.flatten(), dim=0).item()
grad_weight_sim = F.cosine_similarity(baseline_linear.weight.grad.flatten(),
                                     lecac_linear.weight.grad.flatten(), dim=0).item()

print(f"\n  梯度相似度:")
print(f"    input:  {grad_input_sim:.6f}")
print(f"    weight: {grad_weight_sim:.6f}")

# 3. LECAC INT8 (带 LDBR 补偿)
print("\n[3] 创建 LECAC 模型 (INT8, alpha=0.5)")
lecac_linear_ldbr = LECACLinear(in_features, out_features, use_int8=True, alpha=0.5).to(device)
lecac_linear_ldbr.weight.data = baseline_linear.weight.data.clone()
if baseline_linear.bias is not None:
    lecac_linear_ldbr.bias.data = baseline_linear.bias.data.clone()

x_ldbr = torch.randn(batch_size, seq_len, in_features, device=device, requires_grad=True)
y_ldbr = lecac_linear_ldbr(x_ldbr)
loss_ldbr = y_ldbr.sum()
loss_ldbr.backward()

print(f"  x.grad norm: {x_ldbr.grad.norm().item():.6e}")
print(f"  weight.grad norm: {lecac_linear_ldbr.weight.grad.norm().item():.6e}")

# 计算梯度相似度
grad_input_sim_ldbr = F.cosine_similarity(x.grad.flatten(), x_ldbr.grad.flatten(), dim=0).item()
grad_weight_sim_ldbr = F.cosine_similarity(baseline_linear.weight.grad.flatten(),
                                        lecac_linear_ldbr.weight.grad.flatten(), dim=0).item()

print(f"\n  梯度相似度:")
print(f"    input:  {grad_input_sim_ldbr:.6f}")
print(f"    weight: {grad_weight_sim_ldbr:.6f}")


print("\n" + "=" * 70)
print("结果汇总")
print("=" * 70)

print(f"\nBaseline (FP32):")
print(f"  weight.grad norm: {baseline_linear.weight.grad.norm().item():.6e}")

print(f"\nLECAC INT8 (无补偿, alpha=0.0):")
print(f"  weight.grad norm: {lecac_linear.weight.grad.norm().item():.6e}")
print(f"  梯度余弦相似度: {grad_weight_sim:.6f}")
if grad_weight_sim > 0.99:
    print(f"  状态: ✅ 优秀 (>0.99)")
elif grad_weight_sim > 0.95:
    print(f"  状态: ✅ 良好 (>0.95)")
elif grad_weight_sim > 0.90:
    print(f"  状态: ⚠️  可接受 (>0.90)")
else:
    print(f"  状态: ❌ 需要改进")

print(f"\nLECAC INT8 (LDBR补偿, alpha=0.5):")
print(f"  weight.grad norm: {lecac_linear_ldbr.weight.grad.norm().item():.6e}")
print(f"  梯度余弦相似度: {grad_weight_sim_ldbr:.6f}")
if grad_weight_sim_ldbr > 0.99:
    print(f"  状态: ✅ 优秀 (>0.99)")
elif grad_weight_sim_ldbr > 0.95:
    print(f"  状态: ✅ 良好 (>0.95)")
elif grad_weight_sim_ldbr > 0.90:
    print(f"  状态: ⚠️  可接受 (>0.90)")
else:
    print(f"  状态: ❌ 需要改进")

print(f"\nLDBR 补偿效果:")
improvement = grad_weight_sim_ldbr - grad_weight_sim
print(f"  相似度提升: {improvement:+.6f}")
if improvement > 0:
    print(f"  ✅ LDBR 补偿有效")
else:
    print(f"  ⚠️  LDBR 补偿效果不明显 (可能需要调整 alpha)")

print("\n" + "=" * 70)
