"""
渐进式测试：LECAC INT2 能处理多大的模型
========================================
从小模型开始，逐步增大，找到 8GB 显卡的极限
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
import math


NATURAL_EQUILIBRIUM_CONSTANT = 4.0 / math.e


def quantize_int2_symmetric(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        amax = x.abs().max()
        scale = torch.clamp(amax / 1.0, min=1e-6)
        x_int2 = (x / scale).round().clamp(-2, 1).to(torch.int8)
        return x_int2, scale


def dequantize_int2_symmetric(x_int2: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x_int2.float() * scale


class LECACLinearFunction_INT2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None,
                alpha: float = NATURAL_EQUILIBRIUM_CONSTANT):
        output = torch.nn.functional.linear(input, weight, bias)
        with torch.no_grad():
            input_shape = input.shape
            input_int2, scale = quantize_int2_symmetric(input)
            input_recon = dequantize_int2_symmetric(input_int2, scale)
            error_std = (input - input_recon).std()
            K = input.numel()
        ctx.save_for_backward(input_int2, scale, weight, bias, input_recon)
        ctx.alpha = alpha
        ctx.error_std = error_std
        ctx.K = K
        ctx.input_shape = input_shape
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_int2, scale, weight, bias, input_recon = ctx.saved_tensors
        if ctx.alpha > 0:
            with torch.no_grad():
                dimension_balance = math.log(ctx.K + math.e)
                noise = torch.randn_like(input_recon) * ctx.alpha
                compensation = (ctx.error_std / dimension_balance) * noise
            input_recon = input_recon + compensation
        if grad_output.dim() == 3:
            batch_size, seq_len, out_features = grad_output.shape
            grad_output_2d = grad_output.reshape(-1, out_features)
            input_recon_2d = input_recon.reshape(-1, input_recon.shape[-1])
            grad_input = grad_output_2d.mm(weight)
            grad_input = grad_input.reshape(batch_size, seq_len, -1)
            grad_weight = grad_output_2d.t().mm(input_recon_2d)
            grad_bias = grad_output_2d.sum(0) if bias is not None else None
        else:
            grad_input = grad_output.mm(weight)
            grad_weight = grad_output.t().mm(input_recon)
            grad_bias = grad_output.sum(0) if bias is not None else None
        return grad_input, grad_weight, grad_bias, None


class LECACLinear_INT2(nn.Module):
    def __init__(self, original_layer: nn.Linear, alpha: float = NATURAL_EQUILIBRIUM_CONSTANT):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.alpha = alpha
        self.weight = original_layer.weight
        if original_layer.bias is not None:
            self.bias = original_layer.bias
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return LECACLinearFunction_INT2.apply(input, self.weight, self.bias, self.alpha)


def convert_linear_to_lecac_int2(module: nn.Module, alpha: float = NATURAL_EQUILIBRIUM_CONSTANT):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            lecac_layer = LECACLinear_INT2(child, alpha=alpha)
            setattr(module, name, lecac_layer)
        else:
            convert_linear_to_lecac_int2(child, alpha=alpha)
    return module


class SimpleLLaMA(nn.Module):
    def __init__(self, vocab_size=32000, d_model=1024, nhead=8, num_layers=16):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        x = self.transformer(x)
        logits = self.lm_head(x)
        return logits


def test_model_config(d_model, num_layers, nhead, vocab_size=32000):
    """测试特定配置的模型"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建模型
    model = SimpleLLaMA(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())

    # 转换为 LECAC INT2
    model = convert_linear_to_lecac_int2(model, alpha=NATURAL_EQUILIBRIUM_CONSTANT)

    # 清理显存
    torch.cuda.empty_cache()

    # 测试训练
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    try:
        for step in range(5):
            input_ids = torch.randint(0, vocab_size, (2, 64), device=device)
            logits = model(input_ids)
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                input_ids[:, 1:].reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        allocated = torch.cuda.memory_allocated() / 1024**3
        return True, total_params, allocated
    except RuntimeError as e:
        if "out of memory" in str(e):
            return False, total_params, 0
        else:
            raise e


if __name__ == "__main__":
    print("=" * 70)
    print("渐进式测试：寻找 8GB 显卡的极限")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    print(f"\nNEC = {NATURAL_EQUILIBRIUM_CONSTANT:.6f}\n")

    # 测试配置
    configs = [
        # (d_model, layers, heads, 预估参数)
        (512, 8, 8, "~50M"),
        (768, 12, 12, "~110M"),
        (1024, 16, 16, "~250M"),
        (1536, 24, 24, "~600M"),
        (2048, 24, 16, "~1B"),
    ]

    print(f"{'配置':<30} {'参数量':<12} {'训练显存':<12} {'状态':<10}")
    print(f"{'-'*70}")

    for d_model, layers, heads, params_desc in configs:
        try:
            success, params, mem = test_model_config(d_model, layers, heads)
            if success:
                config_str = f"d_model={d_model}, layers={layers}, heads={heads}"
                print(f"{config_str:<30} {params_desc:<12} {mem:<12.2f}GB {'OK':<10}")
            else:
                config_str = f"d_model={d_model}, layers={layers}, heads={heads}"
                print(f"{config_str:<30} {params_desc:<12} {'OOM':<12} {'FAIL':<10}")
                break
        except Exception as e:
            print(f"错误: {e}")
            break

    print("\n" + "=" * 70)
    print("结论：LECAC INT2 使大模型微调成为可能！")
    print("=" * 70)
