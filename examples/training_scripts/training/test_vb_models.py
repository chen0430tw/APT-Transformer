#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
虚拟Blackwell模型测试脚本

测试APT-Transformer各种模型在虚拟Blackwell优化下的性能表现。

用法:
    python test_vb_models.py --model apt --batch-size 4 --seq-len 128
    python test_vb_models.py --model all --epochs 5
"""

import sys
import os
import time
import argparse
import warnings
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 检查依赖
try:
    import torch
    import torch.nn as nn
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    print("❌ 错误: 需要安装PyTorch")
    print("请运行: pip install torch")
    sys.exit(1)

from apt.perf.optimization import enable_vb_optimization, VB_TORCH_AVAILABLE

# 抑制警告
warnings.filterwarnings('ignore')
os.environ['SUPPRESS_APT_WARNINGS'] = 'True'


def print_header(title):
    """打印标题"""
    print("\n" + "="*70)
    print(f"{title:^70}")
    print("="*70)


def print_section(title):
    """打印章节"""
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")


class ModelTester:
    """模型测试器"""

    def __init__(self, device='cpu', batch_size=4, seq_len=128, d_model=512):
        self.device = device
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.d_model = d_model

        print(f"\n⚙️  测试配置:")
        print(f"   设备: {device}")
        print(f"   Batch大小: {batch_size}")
        print(f"   序列长度: {seq_len}")
        print(f"   模型维度: {d_model}")

    def create_dummy_input(self):
        """创建测试输入"""
        return torch.randn(
            self.batch_size,
            self.seq_len,
            self.d_model,
            device=self.device
        )

    def benchmark_model(self, model_fn, model_name, num_iterations=10, enable_vb=False):
        """基准测试单个模型"""
        print_section(f"测试 {model_name} {'(虚拟Blackwell)' if enable_vb else '(原始)'}")

        # 创建模型
        try:
            model = model_fn()
            model = model.to(self.device)
            model.eval()
        except Exception as e:
            print(f"❌ 模型创建失败: {e}")
            return None

        # 应用虚拟Blackwell优化
        if enable_vb:
            try:
                print("\n🚀 应用虚拟Blackwell优化...")
                model = enable_vb_optimization(
                    model,
                    mode='training',
                    enable_quantization=True,
                    replace_pattern='large'  # 只优化大型层
                )
            except Exception as e:
                print(f"⚠️  虚拟Blackwell优化失败: {e}")
                print("   继续使用原始模型...")

        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n📊 模型信息:")
        print(f"   总参数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        print(f"   模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")

        # 创建输入
        x = self.create_dummy_input()

        # 预热
        print(f"\n🔥 预热 {num_iterations//2} 次...")
        with torch.no_grad():
            for _ in range(num_iterations // 2):
                try:
                    _ = model(x)
                except Exception as e:
                    print(f"❌ 预热失败: {e}")
                    return None

        # 基准测试
        print(f"\n⏱️  运行 {num_iterations} 次迭代...")
        times = []

        with torch.no_grad():
            for i in range(num_iterations):
                if self.device == 'cuda':
                    torch.cuda.synchronize()

                start = time.time()
                try:
                    output = model(x)
                except Exception as e:
                    print(f"❌ 迭代 {i+1} 失败: {e}")
                    return None

                if self.device == 'cuda':
                    torch.cuda.synchronize()

                elapsed = time.time() - start
                times.append(elapsed)

                if (i + 1) % (num_iterations // 2) == 0:
                    print(f"   进度: {i+1}/{num_iterations}")

        # 统计结果
        times = np.array(times)
        mean_time = np.mean(times)
        std_time = np.std(times)
        median_time = np.median(times)

        result = {
            'model_name': model_name,
            'vb_enabled': enable_vb,
            'total_params': total_params,
            'mean_time': mean_time,
            'std_time': std_time,
            'median_time': median_time,
            'times': times,
            'output_shape': output.shape if output is not None else None
        }

        print(f"\n📈 性能指标:")
        print(f"   平均时间: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"   中位数时间: {median_time*1000:.2f} ms")
        print(f"   吞吐量: {self.batch_size / mean_time:.2f} samples/sec")
        if output is not None:
            print(f"   输出形状: {output.shape}")

        # 如果启用了VB，显示统计
        if enable_vb and hasattr(model, 'print_all_stats'):
            try:
                model.print_all_stats()
            except:
                pass

        return result

    def compare_models(self, model_fn, model_name, num_iterations=10):
        """对比原始模型和VB优化模型"""
        print_header(f"对比测试: {model_name}")

        # 测试原始模型
        result_orig = self.benchmark_model(
            model_fn, model_name, num_iterations, enable_vb=False
        )

        if result_orig is None:
            print(f"\n⚠️  {model_name} 原始模型测试失败，跳过VB测试")
            return None

        # 测试VB优化模型
        result_vb = self.benchmark_model(
            model_fn, model_name, num_iterations, enable_vb=True
        )

        if result_vb is None:
            print(f"\n⚠️  {model_name} VB优化测试失败")
            return {'original': result_orig, 'vb': None}

        # 计算加速比
        speedup = result_orig['mean_time'] / result_vb['mean_time']

        print_section("对比结果")
        print(f"\n原始模型:")
        print(f"   平均时间: {result_orig['mean_time']*1000:.2f} ms")
        print(f"   吞吐量: {self.batch_size / result_orig['mean_time']:.2f} samples/sec")

        print(f"\nVB优化模型:")
        print(f"   平均时间: {result_vb['mean_time']*1000:.2f} ms")
        print(f"   吞吐量: {self.batch_size / result_vb['mean_time']:.2f} samples/sec")

        print(f"\n🎯 加速比: {speedup:.2f}×")

        if speedup > 1.2:
            print(f"   ✅ 显著加速!")
        elif speedup > 1.0:
            print(f"   ✓ 轻微加速")
        elif speedup > 0.8:
            print(f"   ≈ 性能相当")
        else:
            print(f"   ⚠️  性能下降 (预期在CPU上)")

        return {
            'original': result_orig,
            'vb': result_vb,
            'speedup': speedup
        }


def test_apt_model(tester):
    """测试APT模型"""
    from apt.core.modeling.apt_model import APTLargeModel
    from apt.core.config.apt_config import APTConfig

    def create_model():
        config = APTConfig(
            d_model=tester.d_model,
            n_heads=8,
            n_layers=6,
            d_ff=tester.d_model * 4,
            vocab_size=30000,
            max_seq_length=tester.seq_len
        )
        return APTLargeModel(config)

    return tester.compare_models(create_model, "APT模型")


def test_gpt5_model(tester):
    """测试GPT-5模型"""
    try:
        from apt.core.modeling.gpt5_model import GPT5Model, GPT5Config

        def create_model():
            config = GPT5Config(
                d_model=tester.d_model,
                n_heads=8,
                n_layers=6,
                d_ff=tester.d_model * 4,
                vocab_size=30000,
                max_seq_length=tester.seq_len
            )
            return GPT5Model(config)

        return tester.compare_models(create_model, "GPT-5模型")
    except Exception as e:
        print(f"⚠️  GPT-5模型不可用: {e}")
        return None


def test_claude4_model(tester):
    """测试Claude4模型"""
    try:
        from apt.core.modeling.claude4_model import Claude4Model, Claude4Config

        def create_model():
            config = Claude4Config(
                d_model=tester.d_model,
                n_heads=8,
                n_layers=6,
                d_ff=tester.d_model * 4,
                vocab_size=30000,
                max_seq_length=tester.seq_len
            )
            return Claude4Model(config)

        return tester.compare_models(create_model, "Claude4模型")
    except Exception as e:
        print(f"⚠️  Claude4模型不可用: {e}")
        return None


def test_multimodal_model(tester):
    """测试多模态模型"""
    try:
        from apt.core.modeling.multimodal_model import MultimodalAPTModel
        from apt.core.config.multimodal_config import MultimodalConfig

        def create_model():
            config = MultimodalConfig(
                d_model=tester.d_model,
                n_heads=8,
                n_layers=6,
                d_ff=tester.d_model * 4,
                vocab_size=30000,
                max_seq_length=tester.seq_len
            )
            return MultimodalAPTModel(config)

        return tester.compare_models(create_model, "多模态模型")
    except Exception as e:
        print(f"⚠️  多模态模型不可用: {e}")
        return None


def generate_report(all_results):
    """生成测试报告"""
    print_header("测试报告汇总")

    successful_tests = {k: v for k, v in all_results.items() if v is not None}

    if not successful_tests:
        print("\n❌ 没有成功的测试")
        return

    print(f"\n✅ 成功测试: {len(successful_tests)}/{len(all_results)}")
    print(f"\n{'模型':<20} {'原始(ms)':<15} {'VB(ms)':<15} {'加速比':<10} {'状态'}")
    print("─" * 70)

    for model_name, result in successful_tests.items():
        if result is None or 'original' not in result:
            continue

        orig_time = result['original']['mean_time'] * 1000

        if result.get('vb') is not None:
            vb_time = result['vb']['mean_time'] * 1000
            speedup = result.get('speedup', 0)

            if speedup > 1.2:
                status = "✅ 显著加速"
            elif speedup > 1.0:
                status = "✓ 轻微加速"
            elif speedup > 0.8:
                status = "≈ 性能相当"
            else:
                status = "⚠️ 性能下降"

            print(f"{model_name:<20} {orig_time:<15.2f} {vb_time:<15.2f} {speedup:<10.2f} {status}")
        else:
            print(f"{model_name:<20} {orig_time:<15.2f} {'N/A':<15} {'N/A':<10} ❌ VB失败")

    print("\n" + "="*70)
    print("说明:")
    print("  - 在CPU环境下，VB优化可能因开销>收益而变慢")
    print("  - 在GPU环境下，预期有2-4×的加速效果")
    print("  - 建议在GPU上运行以获得最佳性能")
    print("="*70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='虚拟Blackwell模型性能测试')
    parser.add_argument('--model', type=str, default='apt',
                      choices=['apt', 'gpt5', 'claude4', 'multimodal', 'all'],
                      help='要测试的模型')
    parser.add_argument('--batch-size', type=int, default=4,
                      help='批量大小')
    parser.add_argument('--seq-len', type=int, default=128,
                      help='序列长度')
    parser.add_argument('--d-model', type=int, default=512,
                      help='模型维度')
    parser.add_argument('--iterations', type=int, default=10,
                      help='测试迭代次数')
    parser.add_argument('--device', type=str, default='auto',
                      choices=['auto', 'cpu', 'cuda'],
                      help='运行设备')

    args = parser.parse_args()

    # 确定设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，切换到CPU")
        device = 'cpu'

    # 打印欢迎信息
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + " "*15 + "虚拟Blackwell模型性能测试" + " "*17 + "║")
    print("╚" + "="*68 + "╝")

    if not VB_TORCH_AVAILABLE:
        print("\n⚠️  警告: PyTorch集成模块不可用")
        print("   将只显示模型信息，无法进行VB优化")

    # 创建测试器
    tester = ModelTester(
        device=device,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model
    )

    # 运行测试
    all_results = {}

    if args.model == 'all':
        models_to_test = ['apt', 'gpt5', 'claude4', 'multimodal']
    else:
        models_to_test = [args.model]

    for model_name in models_to_test:
        try:
            if model_name == 'apt':
                result = test_apt_model(tester)
            elif model_name == 'gpt5':
                result = test_gpt5_model(tester)
            elif model_name == 'claude4':
                result = test_claude4_model(tester)
            elif model_name == 'multimodal':
                result = test_multimodal_model(tester)
            else:
                result = None

            all_results[model_name] = result

        except Exception as e:
            print(f"\n❌ {model_name} 测试出错: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_name] = None

    # 生成报告
    generate_report(all_results)

    print("\n🎉 测试完成!")
    print("\n💡 提示:")
    print("   - 在GPU环境运行可获得显著加速")
    print("   - 使用 --device cuda 指定GPU设备")
    print("   - 使用 --model all 测试所有模型")
    print("   - 调整 --batch-size 和 --seq-len 测试不同场景")


if __name__ == "__main__":
    main()
