"""
LECAC (Low-Entropy Compensation for Activated Computation) - 修正版
=========================================================================
使用相同输入对比梯度
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


def quantize_int8_symmetric(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """对称 INT8 量化"""
    with torch.no_grad():
        amax = x.abs().max()
        scale = torch.clamp(amax / 127.0, min=1e-6)
        x_int8 = (x / scale).round().clamp(-127, 127).to(torch.int8)
        return x_int8, scale


def dequantize_int8_symmetric(x_int8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """对称 INT8 反量化"""
    return x_int8.float() * scale


class LECACLinearFunction(torch.autograd.Function):
    """自定义 Linear Function，支持 INT8 激活值量化"""

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                use_int8: bool = True, alpha: float = 0.0):
        output = torch.nn.functional.linear(input, weight, bias)

        if use_int8:
            with torch.no_grad():
                input_int8, scale = quantize_int8_symmetric(input)
                input_recon = dequantize_int8_symmetric(input_int8, scale)
                error_std = (input - input_recon).std()
                K = input.numel()

            ctx.save_for_backward(input_int8, scale, weight, bias)
            ctx.use_int8 = True
            ctx.alpha = alpha
            ctx.error_std = error_std
            ctx.K = K
            ctx.input_shape = input.shape
        else:
            ctx.save_for_backward(input, weight, bias)
            ctx.use_int8 = False
            ctx.input_shape = input.shape

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.use_int8:
            input_int8, scale, weight, bias = ctx.saved_tensors
            input_recon = dequantize_int8_symmetric(input_int8, scale)

            # LDBR 补偿
            if ctx.alpha > 0:
                with torch.no_grad():
                    dimension_balance = math.log(ctx.K + math.e)
                    noise = torch.randn_like(input_recon) * ctx.alpha
                    compensation = (ctx.error_std / dimension_balance) * noise
                input_recon = input_recon + compensation

            grad_input = grad_output.mm(weight)
            grad_weight = grad_output.t().mm(input_recon)
            grad_bias = grad_output.sum(0) if bias is not None else None

        else:
            input, weight, bias = ctx.saved_tensors
            grad_input = grad_output.mm(weight)
            grad_weight = grad_output.t().mm(input)
            grad_bias = grad_output.sum(0) if bias is not None else None

        return grad_input, grad_weight, grad_bias, None, None


class LECACLinear(nn.Module):
    """支持 INT8 激活值量化的 Linear 层"""
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


print("=" * 70)
print("LECAC INT8 量化 - 相同输入测试")
print("=" * 70)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")

in_features, out_features = 768, 768
batch_size = 512

# 固定输入
torch.manual_seed(42)
x = torch.randn(batch_size, in_features, device=device, requires_grad=True)

# 1. Baseline
print("[1] Baseline (FP32)")
baseline_linear = nn.Linear(in_features, out_features).to(device)
baseline_linear.train()

y_base = baseline_linear(x)
loss_base = y_base.sum()
loss_base.backward()

base_grad_norm = x.grad.norm().item()
weight_grad_norm = baseline_linear.weight.grad.norm().item()
print(f"  x.grad norm: {base_grad_norm:.6e}")
print(f"  weight.grad norm: {weight_grad_norm:.6e}")

# 2. LECAC INT8 (无补偿)
print("\n[2] LECAC INT8 (alpha=0.0)")
lecac_linear = LECACLinear(in_features, out_features, use_int8=True, alpha=0.0).to(device)
lecac_linear.weight.data = baseline_linear.weight.data.clone()
if baseline_linear.bias is not None:
    lecac_linear.bias.data = baseline_linear.bias.data.clone()

x.grad = None  # 清零
y_int8 = lecac_linear(x)
loss_int8 = y_int8.sum()
loss_int8.backward()

int8_grad_norm = x.grad.norm().item()
int8_weight_grad_norm = lecac_linear.weight.grad.norm().item()
print(f"  x.grad norm: {int8_grad_norm:.6e}")
print(f"  weight.grad norm: {int8_weight_grad_norm:.6e}")

# 梯度相似度
grad_input_sim = F.cosine_similarity(x.grad.flatten(), baseline_linear.weight.grad.mm(x).flatten(), dim=0).item()
grad_weight_sim = F.cosine_similarity(lecac_linear.weight.grad.flatten(), baseline_linear.weight.grad.flatten(), dim=0).item()

print(f"\n  梯度余弦相似度:")
print(f"    input:  {grad_input_sim:.8f}")
print(f"    weight: {grad_weight_sim:.8f}")

# 3. LECAC INT8 (带 LDBR)
print("\n[3] LECAC INT8 (alpha=0.5)")
lecac_linear_ldbr = LECACLinear(in_features, out_features, use_int8=True, alpha=0.5).to(device)
lecac_linear_ldbr.weight.data = baseline_linear.weight.data.clone()
if baseline_linear.bias is not None:
    lecac_linear_ldbr.bias.data = baseline_linear.bias.data.clone()

x.grad = None
y_ldbr = lecac_linear_ldbr(x)
loss_ldbr = y_ldbr.sum()
loss_ldbr.backward()

ldbr_grad_norm = x.grad.norm().item()
ldbr_weight_grad_norm = lecac_linear_ldbr.weight.grad.norm().item()
print(f"  x.grad norm: {ldbr_grad_norm:.6e}")
print(f"  weight.grad norm: {ldbr_weight_grad_norm:.6e}")

# 梯度相似度
grad_input_sim_ldbr = F.cosine_similarity(x.grad.flatten(), baseline_linear.weight.grad.mm(x).flatten(), dim=0).item()
grad_weight_sim_ldbr = F.cosine_similarity(lecac_linear_ldbr.weight.grad.flatten(), baseline_linear.weight.grad.flatten(), dim=0).item()

print(f"\n  梯度余弦相似度:")
print(f"    input:  {grad_input_sim_ldbr:.8f}")
print(f"    weight: {grad_weight_sim_ldbr:.8f}")


print("\n" + "=" * 70)
print("结果汇总")
print("=" * 70)

print(f"\nBaseline (FP32):")
print(f"  weight.grad norm: {weight_grad_norm:.6e}")

print(f"\nLECAC INT8 (无补偿, alpha=0.0):")
print(f"  weight.grad norm: {int8_weight_grad_norm:.6e}")
print(f"  梯度余弦相似度: {grad_weight_sim:.8f}")

print(f"\nLECAC INT8 (LDBR补偿, alpha=0.5):")
print(f"  weight.grad norm: {ldbr_weight_grad_norm:.6e}")
print(f"  梯度余弦相似度: {grad_weight_sim_ldbr:.8f}")

print(f"\n状态评估:")
if grad_weight_sim > 0.999:
    print(f"  无补偿: Excellent (>0.999)")
elif grad_weight_sim > 0.99:
    print(f"  无补偿: Good (>0.99)")
elif grad_weight_sim > 0.95:
    print(f"  无补偿: Fair (>0.95)")
else:
    print(f"  无补偿: Need improvement")

if grad_weight_sim_ldbr > 0.999:
    print(f"  LDBR: Excellent (>0.999)")
elif grad_weight_sim_ldbr > 0.99:
    print(f"  LDBR: Good (>0.99)")
elif grad_weight_sim_ldbr > 0.95:
    print(f"  LDBR: Fair (>0.95)")
else:
    print(f"  LDBR: Need improvement")

print(f"\nLDBR 效果:")
improvement = grad_weight_sim_ldbr - grad_weight_sim
print(f"  相似度提升: {improvement:+.8f}")
if improvement > 0:
    print(f"  有效: LDBR 补偿改善了梯度估计")
else:
    print(f"  注意: 补偿效果不明显，可能需要调整 alpha")

print("\n" + "=" * 70)
