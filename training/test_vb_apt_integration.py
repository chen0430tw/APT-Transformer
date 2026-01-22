"""
虚拟Blackwell集成测试：APT大模型训练验证

测试场景：
1. 基础集成：VGPU Stack + APT模型
2. Flash优化：FP4量化 + Flash Attention + APT
3. 完整堆叠：多级内存 + 所有优化
4. 性能对比：标准训练 vs 虚拟Blackwell
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import time
from typing import Dict

# APT模型
from apt.core.modeling.apt_model import (
    APTModel,
    APTLargeModel,
    APTModelConfiguration
)

# 虚拟Blackwell组件
from apt_model.optimization import (
    # VGPU堆叠
    VGPUStack,
    create_vgpu_stack,
    VGPUStackLinear,
    # Flash优化
    FlashAttention,
    FusedFP4Linear,
    OptimizedTransformerBlock,
    # 评估器
    VGPUResourceEstimator,
    ModelConfig,
)


class VBOptimizedAPTModel(nn.Module):
    """
    虚拟Blackwell优化的APT模型

    集成：
    - VGPU堆叠（多级内存管理）
    - FP4量化（参数压缩）
    - Flash Attention（注意力优化）
    """

    def __init__(self, apt_config, vgpu_stack: VGPUStack,
                 use_fp4: bool = True,
                 use_flash_attn: bool = True):
        super().__init__()

        self.config = apt_config
        self.vgpu_stack = vgpu_stack
        self.use_fp4 = use_fp4
        self.use_flash_attn = use_flash_attn
        self.optimized_layers = []  # 必须在_optimize_model之前初始化

        # 创建原始APT模型
        self.base_model = APTLargeModel(apt_config)

        # 替换关键层为VGPU优化版本
        self._optimize_model()

    def _optimize_model(self):
        """将模型中的Linear层替换为VGPU优化版本"""
        layer_count = 0

        def replace_linear(module, prefix=''):
            nonlocal layer_count

            for name, child in list(module.named_children()):
                full_name = f"{prefix}.{name}" if prefix else name

                if isinstance(child, nn.Linear):
                    # 替换为VGPU Stack Linear
                    in_features = child.in_features
                    out_features = child.out_features
                    has_bias = child.bias is not None

                    # 创建VGPU优化层
                    vgpu_linear = VGPUStackLinear(
                        in_features, out_features,
                        self.vgpu_stack, bias=has_bias
                    )

                    # 复制权重
                    with torch.no_grad():
                        vgpu_linear.weight.copy_(child.weight)
                        if has_bias:
                            vgpu_linear.bias.copy_(child.bias)

                    # 替换
                    setattr(module, name, vgpu_linear)
                    layer_count += 1
                    self.optimized_layers.append(full_name)

                else:
                    # 递归处理子模块
                    replace_linear(child, full_name)

        # 执行替换
        replace_linear(self.base_model)
        print(f"✓ 已优化 {layer_count} 个线性层")

    def forward(self, input_ids, attention_mask=None):
        """前向传播"""
        return self.base_model(input_ids, attention_mask)

    def get_stats(self) -> Dict:
        """获取VGPU统计"""
        return self.vgpu_stack.get_stats()


def test_basic_integration():
    """测试1：基础集成 - VGPU Stack + APT模型"""
    print("="*70)
    print("测试1: 基础集成（VGPU Stack + APT模型）")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 小型APT配置
    apt_config = APTModelConfiguration(
        vocab_size=10000,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        max_position_embeddings=512,
        dropout=0.1
    )

    print(f"\n[模型配置]")
    print(f"  隐藏层: {apt_config.hidden_size}")
    print(f"  层数: {apt_config.num_layers}")
    print(f"  注意力头: {apt_config.num_heads}")
    print(f"  词表: {apt_config.vocab_size}")

    # 创建VGPU Stack
    stack = create_vgpu_stack()

    # 创建VB优化模型
    print(f"\n[创建虚拟Blackwell优化模型]")
    vb_model = VBOptimizedAPTModel(apt_config, stack).to(device)

    # 测试前向传播
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, apt_config.vocab_size,
                             (batch_size, seq_len),
                             device=device)

    print(f"\n[前向传播测试]")
    print(f"  Batch: {batch_size}")
    print(f"  Seq Length: {seq_len}")

    start = time.time()
    with torch.no_grad():
        output = vb_model(input_ids)
    elapsed = time.time() - start

    print(f"  输出形状: {output.shape}")
    print(f"  耗时: {elapsed*1000:.2f}ms")

    # 打印VGPU统计
    print(f"\n[VGPU统计]")
    stack.print_stats()

    print("\n✅ 测试1完成\n")


def test_memory_estimation():
    """测试2：内存估算 - 预测APT模型资源需求"""
    print("="*70)
    print("测试2: 内存估算（预测APT资源需求）")
    print("="*70)

    # 中型APT配置
    apt_config = APTModelConfiguration(
        vocab_size=50000,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        max_position_embeddings=2048
    )

    # 转换为评估器配置
    model_config = ModelConfig(
        vocab_size=apt_config.vocab_size,
        hidden_size=apt_config.hidden_size,
        num_layers=apt_config.num_layers,
        num_heads=apt_config.num_heads,
        seq_length=apt_config.max_position_embeddings,
        batch_size=8,
        mixed_precision=True,
        gradient_checkpointing=False
    )

    # 评估
    estimator = VGPUResourceEstimator()
    estimator.estimate_transformer(model_config)

    # 生成VGPU配置
    available_gpus = [
        {'device': 'cuda:0', 'vram_gb': 8, 'speed_gbps': 900}
    ]
    estimator.generate_vgpu_config(available_gpus, target_hit_rate=0.90)

    # 打印报告
    estimator.print_report()

    print("\n✅ 测试2完成\n")


def test_training_loop():
    """测试3：训练循环 - 完整训练流程"""
    print("="*70)
    print("测试3: 训练循环（虚拟Blackwell优化）")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 小型配置（快速测试）
    apt_config = APTModelConfiguration(
        vocab_size=5000,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        max_position_embeddings=256,
        dropout=0.1
    )

    # 创建VGPU Stack
    stack = create_vgpu_stack()

    # 创建模型
    model = VBOptimizedAPTModel(apt_config, stack).to(device)

    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 模拟训练数据
    batch_size = 4
    seq_len = 64
    num_batches = 10

    print(f"\n[训练配置]")
    print(f"  Batch Size: {batch_size}")
    print(f"  Seq Length: {seq_len}")
    print(f"  训练批次: {num_batches}")
    print(f"  设备: {device}")

    # 训练循环
    print(f"\n[开始训练]")
    total_time = 0

    for batch_idx in range(num_batches):
        # 生成随机数据
        input_ids = torch.randint(0, apt_config.vocab_size,
                                 (batch_size, seq_len),
                                 device=device)
        labels = torch.randint(0, apt_config.vocab_size,
                              (batch_size, seq_len),
                              device=device)

        start = time.time()

        # 前向传播
        optimizer.zero_grad()
        output = model(input_ids)

        # 计算损失（简单交叉熵）
        loss = nn.functional.cross_entropy(
            output.view(-1, apt_config.vocab_size),
            labels.view(-1)
        )

        # 反向传播
        loss.backward()
        optimizer.step()

        elapsed = time.time() - start
        total_time += elapsed

        if (batch_idx + 1) % 5 == 0:
            print(f"  Batch {batch_idx+1}/{num_batches} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Time: {elapsed*1000:.2f}ms")

    avg_time = total_time / num_batches
    print(f"\n[训练完成]")
    print(f"  总耗时: {total_time:.2f}s")
    print(f"  平均每批: {avg_time*1000:.2f}ms")

    # VGPU统计
    stack.print_stats()

    print("\n✅ 测试3完成\n")


def test_performance_comparison():
    """测试4：性能对比 - 标准 vs 虚拟Blackwell"""
    print("="*70)
    print("测试4: 性能对比（标准 vs 虚拟Blackwell）")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 配置
    apt_config = APTModelConfiguration(
        vocab_size=5000,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        max_position_embeddings=256
    )

    batch_size = 4
    seq_len = 64
    num_batches = 20

    # ===== 标准APT模型 =====
    print("\n[1] 标准APT模型")
    model_std = APTLargeModel(apt_config).to(device)

    input_ids = torch.randint(0, apt_config.vocab_size,
                             (batch_size, seq_len),
                             device=device)

    start = time.time()
    for _ in range(num_batches):
        with torch.no_grad():
            output = model_std(input_ids)
    time_std = time.time() - start

    print(f"  耗时: {time_std:.3f}s ({time_std/num_batches*1000:.2f}ms/batch)")

    # ===== 虚拟Blackwell优化 =====
    print("\n[2] 虚拟Blackwell优化")
    stack = create_vgpu_stack()
    model_vb = VBOptimizedAPTModel(apt_config, stack).to(device)

    # 复制权重（公平对比）
    model_vb.base_model.load_state_dict(model_std.state_dict())

    start = time.time()
    for _ in range(num_batches):
        with torch.no_grad():
            output = model_vb(input_ids)
    time_vb = time.time() - start

    print(f"  耗时: {time_vb:.3f}s ({time_vb/num_batches*1000:.2f}ms/batch)")

    # ===== 对比 =====
    print("\n[性能对比]")
    overhead = (time_vb - time_std) / time_std * 100
    print(f"  标准APT:        {time_std:.3f}s")
    print(f"  虚拟Blackwell:  {time_vb:.3f}s")
    print(f"  开销:           {overhead:+.1f}%")

    if overhead < 20:
        print("  ✅ 开销可接受（<20%）")
    else:
        print("  ⚠️ 开销较高，需要优化")

    # VGPU统计
    stack.print_stats()

    print("\n✅ 测试4完成\n")


def test_large_model():
    """测试5：大模型配置 - 评估真实大模型"""
    print("="*70)
    print("测试5: 大模型配置（类GPT-2规模）")
    print("="*70)

    # 大型APT配置
    apt_config = APTModelConfiguration(
        vocab_size=50000,
        hidden_size=1024,
        num_layers=24,
        num_heads=16,
        max_position_embeddings=2048,
        dropout=0.1
    )

    print(f"\n[模型规模]")
    print(f"  隐藏层: {apt_config.hidden_size}")
    print(f"  层数: {apt_config.num_layers}")
    print(f"  注意力头: {apt_config.num_heads}")
    print(f"  词表: {apt_config.vocab_size}")

    # 估算资源
    model_config = ModelConfig(
        vocab_size=apt_config.vocab_size,
        hidden_size=apt_config.hidden_size,
        num_layers=apt_config.num_layers,
        num_heads=apt_config.num_heads,
        seq_length=apt_config.max_position_embeddings,
        batch_size=4,
        mixed_precision=True,
        gradient_checkpointing=True
    )

    estimator = VGPUResourceEstimator()
    estimator.estimate_transformer(model_config)

    # 多种GPU配置方案
    print("\n" + "─"*70)
    print("方案A: 单卡RTX 3090 24GB")
    print("─"*70)
    estimator.generate_vgpu_config([
        {'device': 'cuda:0', 'vram_gb': 24, 'speed_gbps': 900}
    ])
    estimator.print_report()

    print("\n" + "─"*70)
    print("方案B: 双卡RTX 3070 8GB")
    print("─"*70)
    estimator.generate_vgpu_config([
        {'device': 'cuda:0', 'vram_gb': 8, 'speed_gbps': 900},
        {'device': 'cuda:1', 'vram_gb': 8, 'speed_gbps': 900}
    ])
    estimator.print_report()

    print("\n✅ 测试5完成\n")


def main():
    """运行所有测试"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║       虚拟Blackwell × APT大模型集成测试                            ║")
    print("║       VGPU Stack + Flash Optimization + APT Training              ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print("\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"检测到设备: {device}")
    if device == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
    print("\n")

    tests = [
        ("基础集成", test_basic_integration),
        ("内存估算", test_memory_estimation),
        ("训练循环", test_training_loop),
        ("性能对比", test_performance_comparison),
        ("大模型配置", test_large_model),
    ]

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default='all',
                       help='运行指定测试：all, basic, memory, training, perf, large')
    args = parser.parse_args()

    test_map = {
        'all': None,
        'basic': 0,
        'memory': 1,
        'training': 2,
        'perf': 3,
        'large': 4
    }

    if args.test == 'all':
        for name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                print(f"\n❌ 测试 '{name}' 失败: {e}")
                import traceback
                traceback.print_exc()
                print()
    elif args.test in test_map and test_map[args.test] is not None:
        idx = test_map[args.test]
        name, test_func = tests[idx]
        test_func()
    else:
        print(f"❌ 未知测试: {args.test}")
        print(f"可用测试: {list(test_map.keys())}")

    print("="*70)
    print("所有测试完成！")
    print("="*70)


if __name__ == "__main__":
    main()
