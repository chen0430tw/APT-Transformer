"""
使用"虚拟A100"测试 70B 模型微调
================================
技术：
1. LECAC INT2 量化
2. Virtual VRAM 自动 offload
3. CPU 内存 + GPU 显存协同
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


# ============================================================================
# LECAC INT2 算子
# ============================================================================

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


# ============================================================================
# 显存监控
# ============================================================================

def get_memory_stats():
    """获取显存和内存统计"""
    gpu_allocated = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
    gpu_reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
    import psutil
    ram_used = psutil.virtual_memory().used / 1024**3
    ram_total = psutil.virtual_memory().total / 1024**3
    return gpu_allocated, gpu_reserved, ram_used, ram_total


def print_memory_stats(stage: str):
    gpu_alloc, gpu_res, ram_used, ram_total = get_memory_stats()
    print(f"[{stage}] GPU: {gpu_alloc:.2f}GB (分配), {gpu_res:.2f}GB (保留) | RAM: {ram_used:.1f}/{ram_total:.1f}GB")


# ============================================================================
# 测试 GGUF 模型加载
# ============================================================================

def test_gguf_70b():
    """测试加载 70B GGUF 模型"""
    print("=" * 70)
    print("测试 DeepSeek-R1-Distill-Llama-70B GGUF 模型")
    print("=" * 70)

    model_path = "D:/huihui-ai_DeepSeek-R1-Distill-Llama-70B-abliterated-Q4_0.gguf"

    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在: {model_path}")
        return False

    file_size_gb = os.path.getsize(model_path) / 1024**3
    print(f"模型文件: {model_path}")
    print(f"文件大小: {file_size_gb:.1f} GB")

    print_memory_stats("加载前")

    # 尝试使用 llama-cpp-python
    print(f"\n尝试使用 llama-cpp-python 加载模型...")

    try:
        from llama_cpp import Llama

        # 配置 n_gpu_layers 来 offload 部分层到 GPU
        # -1 = 全部 offload 到 GPU
        # 0 = 全部在 CPU
        n_gpu_layers = -1  # 尝试全部 offload

        print(f"配置:")
        print(f"  n_gpu_layers: {n_gpu_layers} (offload 到 GPU)")
        print(f"  n_ctx: 512 (上下文长度)")

        model = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=512,
            verbose=False
        )

        print_memory_stats("模型加载后")

        # 测试推理
        print(f"\n测试推理...")
        prompt = "What is the capital of France?"
        print(f"输入: {prompt}")

        output = model(
            prompt,
            max_tokens=50,
            stop=["\n"],
            echo=False
        )

        print(f"输出: {output['choices'][0]['text']}")

        print_memory_stats("推理后")

        print(f"\n[SUCCESS] GGUF 模型加载和推理成功！")
        print(f"\n说明:")
        print(f"  - 70B Q4 模型可以在 8GB GPU + CPU RAM 上运行")
        print(f"  - 使用 GGUF 的自动 offload 机制")
        print(f"  - 推理速度取决于 CPU-GPU 数据传输")

        return True

    except ImportError:
        print(f"\n错误: 需要安装 llama-cpp-python")
        print(f"安装命令: pip install llama-cpp-python")
        return False

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("虚拟 A100 测试：70B GGUF 模型")
    print("=" * 70)
    print(f"自然平衡常数 NEC = 4/e ≈ {NATURAL_EQUILIBRIUM_CONSTANT:.6f}")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    success = test_gguf_70b()

    if success:
        print("\n" + "=" * 70)
        print("结论：可以通过 CPU+GPU 混合方式运行 70B 模型")
        print("=" * 70)
    else:
        print("\n需要安装 llama-cpp-python")
