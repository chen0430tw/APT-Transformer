#!/usr/bin/env python3
"""
GPT-4o 架构性能测试脚本

测试 apt.model.architectures.gpt4o_model 的性能表现
"""
import os
import time
import argparse
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPT4oConfig:
    """GPT-4o 配置"""
    vocab_size: int = 50257
    seq_len: int = 1024
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    vein_rank: int = 64  # Vein subspace rank


class SimpleGPT4o(nn.Module):
    """简化的 GPT-4o 风格模型（用于测试）"""

    def __init__(self, config: GPT4oConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(0.1)

        # Transformer blocks with Vein-like attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)

        # Output projection
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, labels=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            labels: (batch_size, seq_len)

        Returns:
            loss, logits
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)

        # Layer norm
        x = self.ln_f(x)

        # LM head
        logits = self.lm_head(x)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-1
            )

        return loss, logits


def try_import_gpt4o():
    """尝试导入真实的 GPT-4o 模型"""
    try:
        from apt.model.architectures.gpt4o_model import GPT4oModel
        print("✅ 使用真实 GPT-4o 模型")
        return True, GPT4oModel
    except Exception as e:
        print(f"⚠️ 无法导入真实 GPT-4o: {e}")
        print("✅ 使用简化版 GPT-4o 风格模型")
        return False, SimpleGPT4o


def apply_virtual_blackwell_optimization(model):
    """应用 Virtual Blackwell 优化"""
    try:
        from apt.vgpu.runtime import vb_integration

        config = {
            "enabled": True,
            "pulse_interval": 20,
            "fake_int8": False,
            "gate_projection": True
        }

        model, vb_adapter = vb_integration.apply_virtual_blackwell_v64(model, config)
        print("✅ Virtual Blackwell 已启用")
        return model, vb_adapter
    except Exception as e:
        print(f"⚠️ Virtual Blackwell 不可用: {e}")
        return model, None


def apply_lecac_optimization(model, bits=2):
    """应用 LECaC 量化"""
    try:
        from apt.vgpu.runtime import lecac

        config = {
            "bits": bits,
            "alpha_warmup": True,
            "warmup_multiplier": 3.0
        }

        model = lecac.apply_lecak_to_model(model, config)
        print(f"✅ LECaC INT{bits} 已启用")
        return model
    except Exception as e:
        print(f"⚠️ LECaC 不可用: {e}")
        return model


def apply_virtual_vram_optimization(model):
    """应用 Virtual VRAM"""
    try:
        from apt.vgpu.runtime import virtual_vram

        config = {
            "enable_nested_v16": True,
            "enable_prefetch": True
        }

        model = virtual_vram.enable_virtual_vram(model, config)
        print("✅ Virtual VRAM 已启用")
        return model
    except Exception as e:
        print(f"⚠️ Virtual VRAM 不可用: {e}")
        return model


def generate_dummy_data(vocab_size, seq_len, batch_size, num_batches):
    """生成虚拟训练数据"""
    for _ in range(num_batches):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()
        yield input_ids, labels


def benchmark(model, dataloader, device, num_steps=100):
    """基准测试"""
    model.train()
    model.to(device)

    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95))

    print(f"\n{'='*60}")
    print(f"开始训练 ({num_steps} 步)")
    print(f"{'='*60}")

    for step, (input_ids, labels) in enumerate(dataloader):
        if step >= num_steps:
            break

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward
        loss, logits = model(input_ids, labels)

        # Backward
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        total_tokens += input_ids.numel()

        # 每 10 步打印一次
        if (step + 1) % 10 == 0 or step == 0:
            elapsed = time.time() - start_time
            tok_s = total_tokens / elapsed
            avg_loss = total_loss / (step + 1)
            print(f"Step {step+1:3d}/{num_steps} | Loss: {avg_loss:.4f} | Tok/s: {tok_s:,.0f}")

    total_time = time.time() - start_time
    avg_loss = total_loss / num_steps
    avg_tok_s = total_tokens / total_time

    print(f"{'='*60}")
    print(f"训练完成!")
    print(f"总时间: {total_time:.1f}秒")
    print(f"平均 Loss: {avg_loss:.4f}")
    print(f"平均速度: {avg_tok_s:,.0f} tok/s")
    print(f"{'='*60}\n")

    return {
        "total_time": total_time,
        "avg_loss": avg_loss,
        "avg_tok_s": avg_tok_s,
        "total_tokens": total_tokens
    }


def main():
    parser = argparse.ArgumentParser(description="GPT-4o 性能测试")
    parser.add_argument("--use-real-gpt4o", action="store_true",
                       help="使用真实的 GPT-4o 模型（如果可用）")
    parser.add_argument("--batch-size", type=int, default=4, help="批次大小")
    parser.add_argument("--seq-len", type=int, default=1024, help="序列长度")
    parser.add_argument("--num-steps", type=int, default=100, help="训练步数")
    parser.add_argument("--no-distributed", action="store_true", help="禁用分布式")

    # 优化选项
    parser.add_argument("--use-virtual-blackwell", action="store_true",
                       help="启用 Virtual Blackwell")
    parser.add_argument("--use-lecac", action="store_true",
                       help="启用 LECaC 量化")
    parser.add_argument("--lecac-bits", type=int, default=2,
                       help="LECaC 量化位数")
    parser.add_argument("--use-virtual-vram", action="store_true",
                       help="启用 Virtual VRAM")

    args = parser.parse_args()

    # 设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ 使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️ 使用 CPU")

    # 尝试导入真实模型
    use_real, ModelClass = try_import_gpt4o()

    # 模型配置
    config = GPT4oConfig(d_model=768, n_heads=12, n_layers=12, d_ff=3072)

    print(f"\n{'='*60}")
    print(f"GPT-4o 架构性能测试")
    print(f"{'='*60}")
    print(f"模型: {'真实 GPT-4o' if use_real else '简化版 GPT-4o'}")
    print(f"配置: d_model={config.d_model}, layers={config.n_layers}, heads={config.n_heads}")
    print(f"批次大小: {args.batch_size}")
    print(f"序列长度: {args.seq_len}")
    print(f"训练步数: {args.num_steps}")
    print(f"{'='*60}\n")

    # 创建模型
    if use_real and args.use_real_gpt4o:
        # 使用真实模型（需要适当的配置）
        try:
            model = ModelClass(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
                d_ff=config.d_ff,
                vein_rank=config.vein_rank
            )
        except Exception as e:
            print(f"⚠️ 真实模型初始化失败: {e}")
            print("✅ 回退到简化版模型")
            model = SimpleGPT4o(config)
    else:
        model = SimpleGPT4o(config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params / 1e6:.1f}M")

    # 应用优化
    print("\n优化配置:")
    if args.use_lecac:
        model = apply_lecac_optimization(model, bits=args.lecac_bits)

    if args.use_virtual_vram:
        model = apply_virtual_vram_optimization(model)

    if args.use_virtual_blackwell:
        model, _ = apply_virtual_blackwell_optimization(model)

    if not any([args.use_lecac, args.use_virtual_vram, args.use_virtual_blackwell]):
        print("无优化（纯训练）")

    print()

    # 生成数据
    dataloader = generate_dummy_data(
        vocab_size=config.vocab_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_batches=args.num_steps + 10
    )

    # 基准测试
    results = benchmark(model, dataloader, device, num_steps=args.num_steps)

    # 打印总结
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)


if __name__ == "__main__":
    main()
