"""
测试虚拟Blackwell资源评估器

场景：
1. 评估标准模型（GPT-2系列）
2. 自定义模型评估
3. 多GPU配置
4. 导出配置文件
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apt_model.optimization.vgpu_estimator import (
    VGPUResourceEstimator,
    ModelConfig,
    quick_estimate
)


def test_standard_models():
    """测试1：标准模型评估"""
    print("="*70)
    print("测试1: 标准模型评估（GPT-2系列）")
    print("="*70)

    models = ['gpt2-small', 'gpt2-medium', 'gpt2-large']

    for model_name in models:
        print(f"\n{'─'*70}")
        print(f"模型: {model_name.upper()}")
        print('─'*70)
        quick_estimate(model_name)
        print("\n")


def test_custom_model():
    """测试2：自定义模型"""
    print("\n" + "="*70)
    print("测试2: 自定义模型（类似GPT-3 Small）")
    print("="*70)

    # 自定义配置
    config = ModelConfig(
        vocab_size=50000,
        hidden_size=2048,
        num_layers=24,
        num_heads=16,
        seq_length=2048,
        batch_size=8,
        gradient_checkpointing=True,  # 启用梯度检查点
        mixed_precision=True,         # 启用混合精度
        optimizer='adamw'
    )

    estimator = VGPUResourceEstimator()

    # 估算内存
    estimator.estimate_transformer(config)

    # 单GPU配置
    available_gpus = [
        {'device': 'cuda:0', 'vram_gb': 24, 'speed_gbps': 900}  # RTX 3090/4090
    ]

    estimator.generate_vgpu_config(available_gpus, target_hit_rate=0.85)
    estimator.print_report()

    # 推荐批次大小
    print("\n[5] 批次大小推荐 (24GB GPU):")
    batch_sizes = estimator.recommend_batch_size(24)
    print(f"  推荐批次大小: {batch_sizes}")


def test_multi_gpu():
    """测试3：多GPU配置"""
    print("\n" + "="*70)
    print("测试3: 多GPU配置（4×A100 80GB）")
    print("="*70)

    # 大模型配置（类似LLaMA-13B）
    config = ModelConfig(
        vocab_size=32000,
        hidden_size=5120,
        num_layers=40,
        num_heads=40,
        seq_length=2048,
        batch_size=4,
        gradient_checkpointing=True,
        mixed_precision=True,
        optimizer='adamw'
    )

    estimator = VGPUResourceEstimator()
    estimator.estimate_transformer(config)

    # 4张A100配置
    available_gpus = [
        {'device': 'cuda:0', 'vram_gb': 80, 'speed_gbps': 2000},  # NVLink
        {'device': 'cuda:1', 'vram_gb': 80, 'speed_gbps': 2000},
        {'device': 'cuda:2', 'vram_gb': 80, 'speed_gbps': 2000},
        {'device': 'cuda:3', 'vram_gb': 80, 'speed_gbps': 2000},
    ]

    estimator.generate_vgpu_config(available_gpus, target_hit_rate=0.95)
    estimator.print_report()


def test_consumer_gpu():
    """测试4：消费级GPU配置"""
    print("\n" + "="*70)
    print("测试4: 消费级GPU（RTX 3070 8GB）")
    print("="*70)

    # 小模型配置
    config = ModelConfig(
        vocab_size=50000,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        seq_length=512,
        batch_size=16,
        gradient_checkpointing=False,
        mixed_precision=True,
        optimizer='adamw'
    )

    estimator = VGPUResourceEstimator()
    estimator.estimate_transformer(config)

    # RTX 3070配置
    available_gpus = [
        {'device': 'cuda:0', 'vram_gb': 8, 'speed_gbps': 900}
    ]

    estimator.generate_vgpu_config(available_gpus, target_hit_rate=0.90)
    estimator.print_report()

    print("\n[5] 批次大小推荐 (8GB GPU):")
    batch_sizes = estimator.recommend_batch_size(8)
    print(f"  推荐批次大小: {batch_sizes}")


def test_extreme_model():
    """测试5：超大模型（类似GPT-3 175B）"""
    print("\n" + "="*70)
    print("测试5: 超大模型（类似GPT-3 175B）")
    print("="*70)

    # 175B参数配置
    config = ModelConfig(
        vocab_size=50000,
        hidden_size=12288,
        num_layers=96,
        num_heads=96,
        seq_length=2048,
        batch_size=1,
        gradient_checkpointing=True,
        mixed_precision=True,
        optimizer='adamw'
    )

    estimator = VGPUResourceEstimator()
    estimator.estimate_transformer(config)

    # 8×A100 80GB集群
    available_gpus = [
        {'device': f'cuda:{i}', 'vram_gb': 80, 'speed_gbps': 2000}
        for i in range(8)
    ]

    estimator.generate_vgpu_config(available_gpus, target_hit_rate=0.80)
    estimator.print_report()

    print("\n💡 超大模型需要额外优化:")
    print("  - 张量并行（Tensor Parallelism）")
    print("  - 流水线并行（Pipeline Parallelism）")
    print("  - ZeRO-3优化器状态分片")
    print("  - CPU/NVMe卸载")


def test_export_config():
    """测试6：导出配置"""
    print("\n" + "="*70)
    print("测试6: 导出配置文件")
    print("="*70)

    config = ModelConfig(
        vocab_size=50000,
        hidden_size=1024,
        num_layers=24,
        num_heads=16,
        seq_length=1024,
        batch_size=16,
        mixed_precision=True
    )

    estimator = VGPUResourceEstimator()
    estimator.estimate_transformer(config)

    available_gpus = [
        {'device': 'cuda:0', 'vram_gb': 24, 'speed_gbps': 900}
    ]

    estimator.generate_vgpu_config(available_gpus)

    # 导出
    output_file = "vgpu_config_gpt2_medium.json"
    estimator.save_config(output_file)
    print(f"\n配置已导出到: {output_file}")


def test_comparison_table():
    """测试7：对比表格"""
    print("\n" + "="*70)
    print("测试7: 模型对比表（训练内存需求）")
    print("="*70)

    models = {
        'GPT-2 Small': ModelConfig(
            vocab_size=50000, hidden_size=768, num_layers=12,
            num_heads=12, seq_length=1024, batch_size=32, mixed_precision=True
        ),
        'GPT-2 Medium': ModelConfig(
            vocab_size=50000, hidden_size=1024, num_layers=24,
            num_heads=16, seq_length=1024, batch_size=16, mixed_precision=True
        ),
        'GPT-2 Large': ModelConfig(
            vocab_size=50000, hidden_size=1280, num_layers=36,
            num_heads=20, seq_length=1024, batch_size=8, mixed_precision=True
        ),
        'LLaMA-7B': ModelConfig(
            vocab_size=32000, hidden_size=4096, num_layers=32,
            num_heads=32, seq_length=2048, batch_size=4, mixed_precision=True,
            gradient_checkpointing=True
        ),
    }

    print("\n{:<15} {:>12} {:>15} {:>15} {:>15}".format(
        "模型", "参数量", "训练内存", "推理内存", "推荐GPU"
    ))
    print("-" * 70)

    for name, config in models.items():
        estimator = VGPUResourceEstimator()
        estimator.estimate_transformer(config)

        params = estimator._count_transformer_params(config)
        train_mem = estimator.memory_estimate.total_train / (1024**3)
        infer_mem = estimator.memory_estimate.total_inference / (1024**3)

        # 推荐GPU
        if train_mem < 12:
            gpu = "RTX 3060 12GB"
        elif train_mem < 24:
            gpu = "RTX 3090 24GB"
        elif train_mem < 48:
            gpu = "A100 40GB"
        else:
            gpu = "A100 80GB ×2+"

        print("{:<15} {:>10.1f}M {:>13.1f}GB {:>13.1f}GB  {:<15}".format(
            name, params/1e6, train_mem, infer_mem, gpu
        ))


def main():
    """运行所有测试"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║          虚拟Blackwell资源评估器测试套件                           ║")
    print("║          VGPU Resource Estimator - Planning Tool                  ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print("\n")

    tests = [
        ("标准模型", test_standard_models),
        ("自定义模型", test_custom_model),
        ("多GPU配置", test_multi_gpu),
        ("消费级GPU", test_consumer_gpu),
        ("超大模型", test_extreme_model),
        ("导出配置", test_export_config),
        ("对比表格", test_comparison_table),
    ]

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=str, default='all',
                       help='运行指定测试：all, standard, custom, multi, consumer, extreme, export, table')
    args = parser.parse_args()

    test_map = {
        'all': None,
        'standard': 0,
        'custom': 1,
        'multi': 2,
        'consumer': 3,
        'extreme': 4,
        'export': 5,
        'table': 6
    }

    if args.test == 'all':
        for name, test_func in tests:
            try:
                test_func()
            except Exception as e:
                print(f"\n❌ 测试 '{name}' 失败: {e}")
                import traceback
                traceback.print_exc()
    elif args.test in test_map and test_map[args.test] is not None:
        idx = test_map[args.test]
        name, test_func = tests[idx]
        test_func()
    else:
        print(f"❌ 未知测试: {args.test}")
        print(f"可用测试: {list(test_map.keys())}")

    print("\n" + "="*70)
    print("测试完成！")
    print("="*70)


if __name__ == "__main__":
    main()
