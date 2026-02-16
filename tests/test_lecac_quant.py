"""
LECAC 量化测试 - 参数化融合版
========================================
整合所有 INT2/INT4 测试散件到一个脚本中

支持：
- --bits {2,4}: 量化比特数
- --mode {stats,training,warmup,orthogonal,alpha_sweep}
- --alpha: LECAC alpha 值（默认 4/e）
- --epochs: 训练轮数

使用示例：
  # INT2 基础统计
  python test_lecac_quant.py --bits 2 --mode stats

  # INT2 完整训练（alpha=4/e）
  python test_lecac_quant.py --bits 2 --mode training --alpha 4_over_e --epochs 10

  # INT2 热启动训练
  python test_lecac_quant.py --bits 2 --mode warmup --warmup_epochs 5 --epochs 10

  # INT2 正交投影补偿
  python test_lecac_quant.py --bits 2 --mode orthogonal --epochs 10

  # INT2 alpha 扫描
  python test_lecac_quant.py --bits 2 --mode alpha_sweep --epochs 5

  # INT4 训练
  python test_lecac_quant.py --bits 4 --mode training --epochs 10
"""
import argparse
import os
import sys

_BASE = "D:/APT-Transformer"
os.environ["TORCH_HOME"] = f"{_BASE}/.torch_cache"
os.environ["TEMP"] = f"{_BASE}/.temp"
for d in [f"{_BASE}/.torch_cache", f"{_BASE}/.temp"]:
    os.makedirs(d, exist_ok=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


# ============================================================================
# 量化/反量化函数
# ============================================================================

def quantize_int2_symmetric(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """对称 INT2 量化（范围: -2 到 1）"""
    with torch.no_grad():
        amax = x.abs().max()
        scale = torch.clamp(amax / 1.0, min=1e-6)
        x_int2 = (x / scale).round().clamp(-2, 1).to(torch.int8)
        return x_int2, scale


def dequantize_int2_symmetric(x_int2: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """对称 INT2 反量化"""
    return x_int2.float() * scale


def quantize_int4_symmetric(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """对称 INT4 量化（范围: -8 到 7）"""
    with torch.no_grad():
        amax = x.abs().max()
        scale = torch.clamp(amax / 7.0, min=1e-6)
        x_int4 = (x / scale).round().clamp(-8, 7).to(torch.int8)
        return x_int4, scale


def dequantize_int4_symmetric(x_int4: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """对称 INT4 反量化"""
    return x_int4.float() * scale


# ============================================================================
# LECAC Linear Functions
# ============================================================================

class LECACLinearFunction_INT2(torch.autograd.Function):
    """INT2 LECAC 线性层"""

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                use_int2: bool = True, alpha: float = 0.0):
        output = torch.nn.functional.linear(input, weight, bias)

        if use_int2:
            with torch.no_grad():
                input_int2, scale = quantize_int2_symmetric(input)
                input_recon = dequantize_int2_symmetric(input_int2, scale)
                error_std = (input - input_recon).std()
                K = input.numel()

            ctx.save_for_backward(input_int2, scale, weight, bias)
            ctx.use_int2 = True
            ctx.alpha = alpha
            ctx.error_std = error_std
            ctx.K = K
            ctx.input_shape = input.shape
        else:
            ctx.save_for_backward(input, weight, bias)
            ctx.use_int2 = False
            ctx.input_shape = input.shape

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.use_int2:
            input_int2, scale, weight, bias = ctx.saved_tensors
            input_recon = dequantize_int2_symmetric(input_int2, scale)

            # LECAC 补偿
            if ctx.alpha > 0:
                with torch.no_grad():
                    dimension_balance = math.log(ctx.K + math.e)
                    noise = torch.randn_like(input_recon) * ctx.alpha
                    compensation = (ctx.error_std / dimension_balance) * noise
                input_recon = input_recon + compensation

            grad_input = grad_output.mm(weight)
            grad_weight = grad_output.t().mm(input_recon)
            grad_bias = grad_output.sum(0) if bias is not None else None

            if ctx.input_shape != grad_input.shape:
                grad_input = grad_input.view(ctx.input_shape)
        else:
            input, weight, bias = ctx.saved_tensors
            grad_input = grad_output.mm(weight)
            grad_weight = grad_output.t().mm(input)
            grad_bias = grad_output.sum(0) if bias is not None else None

        return grad_input, grad_weight, grad_bias, None


class LECACLinearFunction_INT4(torch.autograd.Function):
    """INT4 LECAC 线性层"""

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                use_int4: bool = True, alpha: float = 0.0):
        output = torch.nn.functional.linear(input, weight, bias)

        if use_int4:
            with torch.no_grad():
                input_int4, scale = quantize_int4_symmetric(input)
                input_recon = dequantize_int4_symmetric(input_int4, scale)
                error_std = (input - input_recon).std()
                K = input.numel()

            ctx.save_for_backward(input_int4, scale, weight, bias)
            ctx.use_int4 = True
            ctx.alpha = alpha
            ctx.error_std = error_std
            ctx.K = K
            ctx.input_shape = input.shape
        else:
            ctx.save_for_backward(input, weight, bias)
            ctx.use_int4 = False
            ctx.input_shape = input.shape

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.use_int4:
            input_int4, scale, weight, bias = ctx.saved_tensors
            input_recon = dequantize_int4_symmetric(input_int4, scale)

            # LECAC 补偿
            if ctx.alpha > 0:
                with torch.no_grad():
                    dimension_balance = math.log(ctx.K + math.e)
                    noise = torch.randn_like(input_recon) * ctx.alpha
                    compensation = (ctx.error_std / dimension_balance) * noise
                input_recon = input_recon + compensation

            grad_input = grad_output.mm(weight)
            grad_weight = grad_output.t().mm(input_recon)
            grad_bias = grad_output.sum(0) if bias is not None else None

            if ctx.input_shape != grad_input.shape:
                grad_input = grad_input.view(ctx.input_shape)
        else:
            input, weight, bias = ctx.saved_tensors
            grad_input = grad_output.mm(weight)
            grad_weight = grad_output.t().mm(input)
            grad_bias = grad_output.sum(0) if bias is not None else None

        return grad_input, grad_weight, grad_bias, None


# ============================================================================
# 正交投影补偿版本
# ============================================================================

def orthogonal_projection(tensor: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """正交投影：将 tensor 投影到 direction 的正交补空间"""
    direction_norm = direction.norm()
    if direction_norm < 1e-8:
        return tensor
    direction_normalized = direction / direction_norm
    projection_length = (tensor * direction_normalized).sum()
    projection = projection_length * direction_normalized
    orthogonal_component = tensor - projection
    return orthogonal_component


class OrthogonalLECACLinearFunction(torch.autograd.Function):
    """正交投影 LECAC - 不破坏方向相似度"""

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                use_int2: bool = True, alpha: float = 0.0):
        output = torch.nn.functional.linear(input, weight, bias)

        if use_int2:
            with torch.no_grad():
                input_int2, scale = quantize_int2_symmetric(input)
                input_recon = dequantize_int2_symmetric(input_int2, scale)
                error_std = (input - input_recon).std()
                K = input.numel()

            ctx.save_for_backward(input_int2, scale, weight, bias, input_recon)
            ctx.use_int2 = True
            ctx.alpha = alpha
            ctx.error_std = error_std
            ctx.K = K
            ctx.input_shape = input.shape
        else:
            ctx.save_for_backward(input, weight, bias)
            ctx.use_int2 = False
            ctx.input_shape = input.shape

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.use_int2:
            input_int2, scale, weight, bias, input_recon = ctx.saved_tensors

            # LECAC 补偿
            if ctx.alpha > 0:
                with torch.no_grad():
                    dimension_balance = math.log(ctx.K + math.e)
                    noise = torch.randn_like(input_recon) * ctx.alpha
                    compensation = (ctx.error_std / dimension_balance) * noise
                input_recon = input_recon + compensation

            # 正交投影补偿
            if ctx.alpha > 0:
                with torch.no_grad():
                    # 对重构值进行正交投影
                    grad_output_orth = orthogonal_projection(input_recon, grad_output)
                    input_recon = input_recon + grad_output_orth * 0.5  # 混合 50%

            grad_input = grad_output.mm(weight)
            grad_weight = grad_output.t().mm(input_recon)
            grad_bias = grad_output.sum(0) if bias is not None else None

            if ctx.input_shape != grad_input.shape:
                grad_input = grad_input.view(ctx.input_shape)
        else:
            input, weight, bias = ctx.saved_tensors
            grad_input = grad_output.mm(weight)
            grad_weight = grad_output.t().mm(input)
            grad_bias = grad_output.sum(0) if bias is not None else None

        return grad_input, grad_weight, grad_bias, None


# ============================================================================
# 模型定义
# ============================================================================

class QuantizedLinear(nn.Module):
    """支持 LECAC 的量化线性层"""
    def __init__(self, in_features, out_features, bits=2, alpha=0.0, use_orthogonal=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.alpha = alpha
        self.use_orthogonal = use_orthogonal

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, input):
        if self.bits == 2:
            if self.use_orthogonal:
                return OrthogonalLECACLinearFunction.apply(input, self.weight, self.bias, True, self.alpha)
            else:
                return LECACLinearFunction_INT2.apply(input, self.weight, self.bias, True, self.alpha)
        elif self.bits == 4:
            return LECACLinearFunction_INT4.apply(input, self.weight, self.bias, True, self.alpha)
        else:
            return torch.nn.functional.linear(input, self.weight, self.bias)


class SimpleModel(nn.Module):
    """简单测试模型"""
    def __init__(self, bits=2, alpha=0.0, use_orthogonal=False):
        super().__init__()
        self.bits = bits
        self.alpha = alpha
        self.use_orthogonal = use_orthogonal

        self.fc1 = QuantizedLinear(128, 256, bits, alpha, use_orthogonal)
        self.fc2 = QuantizedLinear(256, 128, bits, alpha, use_orthogonal)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


# ============================================================================
# 测试函数
# ============================================================================

def get_alpha_value(alpha_str: str) -> float:
    """将 alpha 字符串转换为数值"""
    alpha_map = {
        "4_over_e": 4.0 / math.e,
        "4/e": 4.0 / math.e,
        "nec": 4.0 / math.e,  # Natural Equilibrium Constant
    }
    if alpha_str in alpha_map:
        return alpha_map[alpha_str]
    try:
        return float(alpha_str)
    except ValueError:
        raise ValueError(f"Invalid alpha value: {alpha_str}")


def run_stats_test(bits, alpha):
    """基础统计测试"""
    print(f"\n{'='*70}")
    print(f"Mode: STATS - {bits}-bit LECAC")
    print(f"Alpha: {alpha:.6f}")
    print(f"{'='*70}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleModel(bits=bits, alpha=alpha).to(device)

    # 测试前向传播
    x = torch.randn(32, 128).to(device)
    y = model(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print(f"输出均值: {y.mean().item():.6f}")
    print(f"输出标准差: {y.std().item():.6f}")

    # 测试反向传播
    loss = y.mean()
    loss.backward()

    print(f"Loss: {loss.item():.6f}")
    print(f"梯度计算成功")


def run_training_test(bits, alpha, epochs):
    """完整训练测试"""
    print(f"\n{'='*70}")
    print(f"Mode: TRAINING - {bits}-bit LECAC")
    print(f"Alpha: {alpha:.6f}")
    print(f"Epochs: {epochs}")
    print(f"{'='*70}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleModel(bits=bits, alpha=alpha).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    x = torch.randn(32, 128).to(device)
    target = torch.randn(32, 128).to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        output = model(x)
        loss = F.mse_loss(output, target)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}: Loss = {loss.item():.6f}")


def run_warmup_test(bits, alpha, warmup_epochs, epochs):
    """热启动训练测试"""
    print(f"\n{'='*70}")
    print(f"Mode: WARMUP - {bits}-bit LECAC")
    print(f"Alpha: {alpha:.6f}")
    print(f"Warmup: {warmup_epochs} epochs (FP32)")
    print(f"Training: {epochs} epochs (INT2)")
    print(f"{'='*70}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleModel(bits=bits, alpha=alpha).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    x = torch.randn(32, 128).to(device)
    target = torch.randn(32, 128).to(device)

    total_epochs = warmup_epochs + epochs

    for epoch in range(total_epochs):
        model.train()
        optimizer.zero_grad()

        # 前 warmup_epochs 使用 FP32
        use_int2 = (epoch >= warmup_epochs)
        current_alpha = alpha if use_int2 else 0.0

        # 临时关闭 INT2
        if not use_int2:
            output = torch.nn.functional.linear(model.fc1.weight, model.fc1.bias)
            output = torch.nn.functional.linear(model.fc2.weight, model.fc2.bias)
        else:
            output = model(x)

        loss = F.mse_loss(output, target)

        loss.backward()
        optimizer.step()

        mode_str = "FP32" if not use_int2 else f"INT2 (alpha={current_alpha:.3f})"
        if (epoch + 1) % max(1, total_epochs // 5) == 0 or epoch == total_epochs - 1:
            print(f"Epoch {epoch+1}/{total_epochs} [{mode_str}]: Loss = {loss.item():.6f}")


def run_orthogonal_test(bits, alpha, epochs):
    """正交投影补偿测试"""
    print(f"\n{'='*70}")
    print(f"Mode: ORTHOGONAL - {bits}-bit LECAC with 正交投影")
    print(f"Alpha: {alpha:.6f}")
    print(f"Epochs: {epochs}")
    print(f"{'='*70}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleModel(bits=bits, alpha=alpha, use_orthogonal=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    x = torch.randn(32, 128).to(device)
    target = torch.randn(32, 128).to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        output = model(x)
        loss = F.mse_loss(output, target)

        loss.backward()
        optimizer.step()

        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}: Loss = {loss.item():.6f}")


def run_alpha_sweep(bits, epochs):
    """Alpha 扫描测试"""
    print(f"\n{'='*70}")
    print(f"Mode: ALPHA SWEEP - {bits}-bit LECAC")
    print(f"Epochs: {epochs} per alpha value")
    print(f"{'='*70}")

    alphas = [0.0, 0.5, 1.0, 1.4715, 2.0]  # 1.4715 ≈ 4/e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(32, 128).to(device)
    target = torch.randn(32, 128).to(device)

    results = []

    for alpha in alphas:
        model = SimpleModel(bits=bits, alpha=alpha).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        epoch_loss = 0
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            output = model(x)
            loss = F.mse_loss(output, target)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / epochs
        results.append((alpha, avg_loss))
        print(f"Alpha = {alpha:.4f}: Avg Loss = {avg_loss:.6f}")

    print(f"\n最佳 Alpha: {min(results, key=lambda x: x[1])[0]:.4f} (Loss = {min(results, key=lambda x: x[1])[1]:.6f})")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LECAC 量化测试 - 参数化融合版",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--bits", type=int, choices=[2, 4], default=2,
                        help="量化比特数（默认: 2）")
    parser.add_argument("--mode", type=str,
                        choices=["stats", "training", "warmup", "orthogonal", "alpha_sweep"],
                        default="stats",
                        help="测试模式（默认: stats）")
    parser.add_argument("--alpha", type=str, default="4_over_e",
                        help="LECAC alpha 值（默认: 4_over_e，即 4/e≈1.4715）")
    parser.add_argument("--epochs", type=int, default=10,
                        help="训练轮数（默认: 10）")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="热启动轮数（仅在 warmup 模式使用，默认: 5）")

    args = parser.parse_args()

    # 转换 alpha 值
    alpha = get_alpha_value(args.alpha)

    # 打印配置
    print("=" * 70)
    print("LECAC 量化测试 - 参数化融合版")
    print("=" * 70)
    print(f"Bits: {args.bits}")
    print(f"Mode: {args.mode}")
    print(f"Alpha: {alpha:.6f}")
    print(f"Epochs: {args.epochs}")
    print("=" * 70)

    # 根据模式运行测试
    if args.mode == "stats":
        run_stats_test(args.bits, alpha)
    elif args.mode == "training":
        run_training_test(args.bits, alpha, args.epochs)
    elif args.mode == "warmup":
        run_warmup_test(args.bits, alpha, args.warmup_epochs, args.epochs)
    elif args.mode == "orthogonal":
        run_orthogonal_test(args.bits, alpha, args.epochs)
    elif args.mode == "alpha_sweep":
        run_alpha_sweep(args.bits, args.epochs)

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
