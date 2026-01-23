#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
虚拟Blackwell快速测试脚本（简化版）

专门用于快速测试，使用较小的模型配置。

用法:
    python test_vb_quick.py                    # 默认配置
    python test_vb_quick.py --small            # 超小模型
    python test_vb_quick.py --iterations 20    # 自定义迭代次数
"""

import sys
import os
import time
import argparse
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 检查torch
try:
    import torch
    import torch.nn as nn
    import numpy as np
except ImportError:
    print("❌ 需要安装PyTorch: pip install torch numpy")
    sys.exit(1)

# 导入优化模块
from apt.perf.optimization import enable_vb_optimization, VB_TORCH_AVAILABLE

# 抑制警告
import warnings
warnings.filterwarnings('ignore')
os.environ['SUPPRESS_APT_WARNINGS'] = 'True'


class SimpleTransformerBlock(nn.Module):
    """简单的Transformer块用于快速测试"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        # Multi-head attention
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class SimpleTransformer(nn.Module):
    """简单的Transformer模型"""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len, d_model) 如果已经是embedding
        # 或 (batch, seq_len) 如果是token ids

        if x.dim() == 2:  # token ids
            seq_len = x.size(1)
            pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
            x = self.embedding(x) + self.pos_embedding(pos)
        elif x.dim() == 3:  # already embedded
            pass
        else:
            raise ValueError(f"Unexpected input dimension: {x.dim()}")

        for block in self.blocks:
            x = block(x)

        return self.output(x)


def benchmark_model(model, input_data, num_iterations=10, warmup=5):
    """基准测试"""
    model.eval()

    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_data)

    # 测试
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.time()
            _ = model(input_data)
            times.append(time.time() - start)

    times = np.array(times)
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'median': np.median(times),
        'min': np.min(times),
        'max': np.max(times)
    }


def count_linear_layers(model):
    """统计线性层数量"""
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description='虚拟Blackwell快速测试')
    parser.add_argument('--small', action='store_true',
                      help='使用超小模型（更快）')
    parser.add_argument('--iterations', type=int, default=10,
                      help='测试迭代次数')
    parser.add_argument('--batch-size', type=int, default=4,
                      help='批量大小')
    parser.add_argument('--seq-len', type=int, default=64,
                      help='序列长度')
    args = parser.parse_args()

    print("\n" + "="*70)
    print(" "*20 + "虚拟Blackwell快速测试")
    print("="*70)

    # 模型配置
    if args.small:
        config = {
            'vocab_size': 10000,
            'd_model': 256,
            'n_heads': 4,
            'n_layers': 4,
            'd_ff': 256 * 4,
            'max_len': 128
        }
        print("\n📐 配置: 超小模型")
    else:
        config = {
            'vocab_size': 30000,
            'd_model': 512,
            'n_heads': 8,
            'n_layers': 6,
            'd_ff': 512 * 4,
            'max_len': 256
        }
        print("\n📐 配置: 标准模型")

    print(f"   维度: {config['d_model']}")
    print(f"   层数: {config['n_layers']}")
    print(f"   注意力头: {config['n_heads']}")
    print(f"   FFN维度: {config['d_ff']}")

    print(f"\n⚙️  测试参数:")
    print(f"   Batch大小: {args.batch_size}")
    print(f"   序列长度: {args.seq_len}")
    print(f"   迭代次数: {args.iterations}")

    # 创建输入
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   设备: {device}")

    input_data = torch.randn(
        args.batch_size,
        args.seq_len,
        config['d_model'],
        device=device
    )

    # 测试1: 原始模型
    print("\n" + "─"*70)
    print("  测试1: 原始模型")
    print("─"*70)

    model_orig = SimpleTransformer(**config).to(device)
    total_params = sum(p.numel() for p in model_orig.parameters())
    n_linear = count_linear_layers(model_orig)

    print(f"\n📊 模型信息:")
    print(f"   总参数: {total_params:,}")
    print(f"   线性层数量: {n_linear}")
    print(f"   模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")

    print(f"\n⏱️  基准测试...")
    result_orig = benchmark_model(model_orig, input_data, args.iterations)

    print(f"\n📈 性能:")
    print(f"   平均时间: {result_orig['mean']*1000:.2f} ± {result_orig['std']*1000:.2f} ms")
    print(f"   中位数: {result_orig['median']*1000:.2f} ms")
    print(f"   范围: [{result_orig['min']*1000:.2f}, {result_orig['max']*1000:.2f}] ms")
    print(f"   吞吐量: {args.batch_size / result_orig['mean']:.2f} samples/sec")

    # 测试2: VB优化模型
    print("\n" + "─"*70)
    print("  测试2: 虚拟Blackwell优化模型")
    print("─"*70)

    if not VB_TORCH_AVAILABLE:
        print("\n⚠️  VB PyTorch集成不可用，跳过测试")
        return

    model_vb = SimpleTransformer(**config).to(device)

    print(f"\n🚀 应用虚拟Blackwell优化...")
    try:
        model_vb = enable_vb_optimization(
            model_vb,
            mode='training',
            enable_quantization=False,  # 禁用量化（太慢，每个8x8 block都要做SVD）
            replace_pattern='large'  # 只优化大型层
        )
    except Exception as e:
        print(f"❌ VB优化失败: {e}")
        return

    print(f"\n⏱️  基准测试...")
    result_vb = benchmark_model(model_vb, input_data, args.iterations)

    print(f"\n📈 性能:")
    print(f"   平均时间: {result_vb['mean']*1000:.2f} ± {result_vb['std']*1000:.2f} ms")
    print(f"   中位数: {result_vb['median']*1000:.2f} ms")
    print(f"   范围: [{result_vb['min']*1000:.2f}, {result_vb['max']*1000:.2f}] ms")
    print(f"   吞吐量: {args.batch_size / result_vb['mean']:.2f} samples/sec")

    # VB统计
    try:
        model_vb.print_all_stats()
    except:
        pass

    # 对比
    print("\n" + "="*70)
    print(" "*25 + "对比结果")
    print("="*70)

    speedup = result_orig['mean'] / result_vb['mean']

    print(f"\n原始模型:")
    print(f"   时间: {result_orig['mean']*1000:.2f} ms")
    print(f"   吞吐量: {args.batch_size / result_orig['mean']:.2f} samples/sec")

    print(f"\nVB优化模型:")
    print(f"   时间: {result_vb['mean']*1000:.2f} ms")
    print(f"   吞吐量: {args.batch_size / result_vb['mean']:.2f} samples/sec")

    print(f"\n🎯 加速比: {speedup:.2f}×")

    if speedup > 1.2:
        print("   ✅ 显著加速!")
        emoji = "🚀"
    elif speedup > 1.0:
        print("   ✓ 轻微加速")
        emoji = "✓"
    elif speedup > 0.8:
        print("   ≈ 性能相当")
        emoji = "≈"
    else:
        print("   ⚠️  性能下降")
        emoji = "⚠️"

        if device == 'cpu':
            print("\n   说明: 在CPU环境下，虚拟化开销可能超过收益")
            print("   建议: 在GPU环境测试以获得最佳性能")

    # 可视化
    print(f"\n{emoji} 性能对比:")
    bar_length = 50
    orig_bar = int(bar_length * result_orig['mean'] / max(result_orig['mean'], result_vb['mean']))
    vb_bar = int(bar_length * result_vb['mean'] / max(result_orig['mean'], result_vb['mean']))

    print(f"   原始:  {'█' * orig_bar}{' ' * (bar_length - orig_bar)} {result_orig['mean']*1000:.1f}ms")
    print(f"   VB优化: {'█' * vb_bar}{' ' * (bar_length - vb_bar)} {result_vb['mean']*1000:.1f}ms")

    print("\n" + "="*70)
    print("测试完成!")
    print("="*70)

    print("\n💡 提示:")
    print("   - 使用 --small 测试更小的模型（更快）")
    print("   - 使用 --iterations 20 增加测试次数（更准确）")
    print("   - 在GPU上运行可获得更好的加速效果")
    print("   - 完整测试请使用: python test_vb_models.py")


if __name__ == "__main__":
    main()
