#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Profile配置系统使用示例

展示如何使用APT 2.0的profile配置系统
"""

from apt.core.config import load_profile, list_profiles


def example_list_profiles():
    """示例1: 列出所有可用的profile"""
    print("=" * 60)
    print("示例1: 列出所有可用的profile")
    print("=" * 60)

    profiles = list_profiles()
    print(f"可用的profiles: {profiles}\n")


def example_load_profile():
    """示例2: 加载和使用profile"""
    print("=" * 60)
    print("示例2: 加载standard profile")
    print("=" * 60)

    config = load_profile('standard')

    print(f"Profile名称: {config.profile.name}")
    print(f"描述: {config.profile.description}")
    print(f"版本: {config.profile.version}\n")

    print("模型配置:")
    print(f"  架构: {config.model.architecture}")
    print(f"  隐藏层大小: {config.model.hidden_size}")
    print(f"  层数: {config.model.num_layers}")
    print(f"  注意力头: {config.model.num_attention_heads}\n")

    print("训练配置:")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  混合精度: {config.training.mixed_precision}")
    print(f"  优化器: {config.training.optimizer}\n")

    print("分布式配置:")
    print(f"  启用: {config.training.distributed.enabled}")
    print(f"  Backend: {config.training.distributed.backend}")
    print(f"  World size: {config.training.distributed.world_size}\n")

    print("VGPU配置:")
    print(f"  启用: {config.vgpu.enabled}")
    print(f"  最大虚拟GPU数: {config.vgpu.max_virtual_gpus}")
    print(f"  调度策略: {config.vgpu.scheduling}\n")


def example_compare_profiles():
    """示例3: 比较不同profile的配置"""
    print("=" * 60)
    print("示例3: 比较lite vs pro配置")
    print("=" * 60)

    lite = load_profile('lite')
    pro = load_profile('pro')

    print("配置对比:")
    print(f"{'项目':<30} {'lite':<20} {'pro':<20}")
    print("-" * 70)
    print(f"{'Hidden size':<30} {lite.model.hidden_size:<20} {pro.model.hidden_size:<20}")
    print(f"{'Num layers':<30} {lite.model.num_layers:<20} {pro.model.num_layers:<20}")
    print(f"{'Batch size':<30} {lite.training.batch_size:<20} {pro.training.batch_size:<20}")
    print(f"{'Learning rate':<30} {lite.training.learning_rate:<20} {pro.training.learning_rate:<20}")
    print(f"{'Distributed':<30} {str(lite.training.distributed.enabled):<20} {str(pro.training.distributed.enabled):<20}")
    print(f"{'VGPU enabled':<30} {str(lite.vgpu.enabled):<20} {str(pro.vgpu.enabled):<20}")
    print(f"{'Max vGPUs':<30} {lite.vgpu.max_virtual_gpus:<20} {pro.vgpu.max_virtual_gpus:<20}")
    print()


def example_use_in_training():
    """示例4: 在训练中使用profile"""
    print("=" * 60)
    print("示例4: 在训练中使用profile配置")
    print("=" * 60)

    # 加载配置
    config = load_profile('standard')

    print("模拟训练流程:")
    print(f"1. 加载profile: {config.profile.name}")
    print(f"2. 创建模型: {config.model.architecture}")
    print(f"   - hidden_size={config.model.hidden_size}")
    print(f"   - num_layers={config.model.num_layers}")
    print(f"3. 设置训练参数:")
    print(f"   - batch_size={config.training.batch_size}")
    print(f"   - learning_rate={config.training.learning_rate}")
    print(f"   - optimizer={config.training.optimizer}")
    print(f"4. 配置分布式训练:")
    print(f"   - enabled={config.training.distributed.enabled}")
    print(f"   - world_size={config.training.distributed.world_size}")
    print(f"5. 启用监控:")
    print(f"   - tensorboard={config.monitoring.tensorboard}")
    print(f"   - wandb={config.monitoring.wandb}")
    print(f"6. 配置检查点:")
    print(f"   - save_interval={config.checkpoints.save_interval}")
    print(f"   - keep_last_n={config.checkpoints.keep_last_n}")

    print("\n训练伪代码:")
    print("""
    from apt.model.architectures import APTLargeModel
    from apt.trainops.engine import Trainer

    # 使用profile配置创建模型
    model = APTLargeModel(
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        num_attention_heads=config.model.num_attention_heads,
    )

    # 使用profile配置创建训练器
    trainer = Trainer(
        model=model,
        batch_size=config.training.batch_size,
        learning_rate=config.training.learning_rate,
        optimizer=config.training.optimizer,
    )

    # 开始训练
    trainer.train()
    """)


def example_access_raw_config():
    """示例5: 访问原始YAML配置"""
    print("=" * 60)
    print("示例5: 访问原始YAML配置")
    print("=" * 60)

    config = load_profile('standard')

    # 访问未被dataclass覆盖的原始配置
    print("扩展配置 (RAG):")
    if config.extensions.rag.get('enabled'):
        print(f"  启用: {config.extensions.rag['enabled']}")
        print(f"  索引类型: {config.extensions.rag.get('index_type', 'N/A')}")
        print(f"  Embedding维度: {config.extensions.rag.get('embedding_dim', 'N/A')}")

    print("\n原始YAML数据结构:")
    print(f"  顶层keys: {list(config._raw.keys())}")
    print(f"  Model keys: {list(config._raw.get('model', {}).keys())}")
    print()


def main():
    """运行所有示例"""
    try:
        example_list_profiles()
        example_load_profile()
        example_compare_profiles()
        example_use_in_training()
        example_access_raw_config()

        print("=" * 60)
        print("✓ 所有示例运行完成")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
