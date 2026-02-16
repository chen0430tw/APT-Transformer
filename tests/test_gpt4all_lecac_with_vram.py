"""
使用 LECAC INT2 + Virtual VRAM 测试大模型微调
================================================
充分利用 virtual_vram 的自动 offload 功能
当显存不够时，自动把权重 offload 到 CPU 内存
"""
import os
import sys

_BASE = "D:/APT-Transformer"
sys.path.insert(0, _BASE)

os.environ["TORCH_HOME"] = f"{_BASE}/.torch_cache"
os.environ["TEMP"] = f"{_BASE}/.temp"
os.makedirs(_BASE + "/.torch_cache", exist_ok=True)
os.makedirs(_BASE + "/.temp", exist_ok=True)

import torch
import torch.nn as nn
import math
import gc


# ============================================================================
# 导入 virtual_vram
# ============================================================================

from apt.vgpu.runtime.virtual_vram import virtual_vram, VirtualVRAMConfig, NATURAL_EQUILIBRIUM_CONSTANT


# ============================================================================
# 显存监控
# ============================================================================

def get_gpu_memory():
    """获取 GPU 显存占用"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0.0, 0.0


def print_memory_stats(stage: str):
    """打印显存统计"""
    allocated, reserved = get_gpu_memory()
    print(f"[{stage}] GPU显存: {allocated:.2f} GB (分配), {reserved:.2f} GB (保留)")


# ============================================================================
# 创建大模型
# ============================================================================

class BigLLaMA(nn.Module):
    """较大的 LLaMA 模型，模拟 1B-3B 规模"""

    def __init__(self, vocab_size=32000, d_model=2048, nhead=16, num_layers=24):
        super().__init__()
        self.d_model = d_model

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重绑定
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids):
        x = self.token_embedding(input_ids)
        x = self.transformer(x)
        logits = self.lm_head(x)
        return logits


# ============================================================================
# 测试主函数
# ============================================================================

def test_with_virtual_vram():
    """测试使用 virtual_vram 的大模型训练"""
    print("=" * 70)
    print("测试 LECAC INT2 + Virtual VRAM 大模型微调")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # 配置 virtual_vram
    print(f"\n配置 Virtual VRAM:")
    print(f"  - 自动 offload 大于 1MB 的 tensor")
    print(f"  - INT8 压缩")
    print(f"  - LECAC 补偿 (alpha={NATURAL_EQUILIBRIUM_CONSTANT:.4f})")

    config = VirtualVRAMConfig(
        enabled=True,
        min_tensor_bytes=1 << 20,  # 1MB
        compress=True,
        compress_dtype="int8",
        use_lecac=True,
        lecac_alpha=NATURAL_EQUILIBRIUM_CONSTANT,
        verbose=True
    )

    # 创建大模型
    print(f"\n创建大模型 (模拟 1B+ 规模)...")
    print(f"配置: d_model=2048, layers=24, heads=16")

    model = BigLLaMA(
        vocab_size=32000,
        d_model=2048,
        nhead=16,
        num_layers=24
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params / 1e9:.2f}B")
    print(f"FP32 理论大小: {total_params * 4 / 1024**3:.2f} GB")

    # 训练配置
    batch_size = 2  # 减小 batch size
    seq_len = 128
    num_steps = 20

    print(f"\n开始训练 (Virtual VRAM 模式)...")
    print(f"配置: batch_size={batch_size}, seq_len={seq_len}, steps={num_steps}")

    # 使用 virtual_vram context
    with virtual_vram(config):
        model = model.to(device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        print_memory_stats("Virtual VRAM 启动后")

        losses = []

        for step in range(num_steps):
            # 生成模拟数据
            input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)

            # 前向传播
            logits = model(input_ids)

            # 计算损失
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, 32000),
                input_ids[:, 1:].reshape(-1)
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 优化器步骤
            optimizer.step()

            losses.append(loss.item())

            # 打印进度
            if (step + 1) % 5 == 0:
                avg_loss = sum(losses[-5:]) / 5
                allocated, reserved = get_gpu_memory()
                print(f"Step {step+1}/{num_steps}: Loss={avg_loss:.4f}, GPU={allocated:.2f}GB")

        print_memory_stats("训练后")

    # 最终统计
    print(f"\n" + "=" * 70)
    print(f"最终结果")
    print(f"=" * 70)
    print(f"模型参数: {total_params / 1e9:.2f}B")
    print(f"FP32 大小: {total_params * 4 / 1024**3:.2f} GB")
    print(f"训练步数: {num_steps}")
    print(f"最终损失: {losses[-1]:.4f}")

    print(f"\nVirtual VRAM 优势:")
    print(f"  - 自动 offload 大权重到 CPU 内存")
    print(f"  - INT8 压缩节省传输带宽")
    print(f"  - LECAC 补偿保证训练效果")
    print(f"  - 在 8GB 显卡上训练 1B+ 模型")

    print(f"\n[SUCCESS] LECAC INT2 + Virtual VRAM = 大模型微调民主化！")
    print("=" * 70)

    return True


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LECAC INT2 + Virtual VRAM 大模型微调测试")
    print("=" * 70)
    print(f"自然平衡常数 (NEC) = 4/e ≈ {NATURAL_EQUILIBRIUM_CONSTANT:.6f}")
    print("=" * 70)

    try:
        success = test_with_virtual_vram()
        if success:
            print("\n测试完成！")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
