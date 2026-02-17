#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT 2.0 快速开始脚本

最简单的方式启动 APT 训练：
    python quickstart.py

或指定 profile：
    python quickstart.py --profile standard
    python quickstart.py --profile pro
"""

import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='APT 2.0 快速开始 - 一键启动训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python quickstart.py                    # 使用 lite profile (最快)
  python quickstart.py --profile standard # 使用 standard profile
  python quickstart.py --profile pro      # 使用 pro profile (分布式)
  python quickstart.py --profile full     # 使用 full profile (Virtual Blackwell)
  python quickstart.py --list-profiles    # 列出所有可用 profiles
  python quickstart.py --demo             # 仅查看配置，不训练
        """
    )

    parser.add_argument(
        '--profile',
        type=str,
        default='lite',
        choices=['lite', 'standard', 'pro', 'full'],
        help='选择配置 profile (默认: lite)'
    )

    parser.add_argument(
        '--list-profiles',
        action='store_true',
        help='列出所有可用的 profiles'
    )

    parser.add_argument(
        '--demo',
        action='store_true',
        help='仅查看配置信息，不实际训练'
    )

    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='训练数据路径 (可选)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='训练轮数 (默认: 10)'
    )

    args = parser.parse_args()

    # 导入必要模块
    try:
        from apt.core.config import load_profile, list_profiles
    except ImportError as e:
        print(f"❌ 错误: 无法导入 APT 模块")
        print(f"   请确保已安装: pip install -e .")
        print(f"   详细错误: {e}")
        return 1

    # 列出 profiles
    if args.list_profiles:
        print("=" * 60)
        print("可用的 APT 2.0 Profiles")
        print("=" * 60)
        profiles = list_profiles()
        for p in profiles:
            print(f"  • {p}")
        print()
        print("使用方法: python quickstart.py --profile <name>")
        print("=" * 60)
        return 0

    # 加载配置
    print(f"🚀 APT 2.0 快速开始")
    print(f"📋 正在加载 profile: {args.profile}")
    print()

    try:
        config = load_profile(args.profile)
    except Exception as e:
        print(f"❌ 错误: 无法加载 profile '{args.profile}'")
        print(f"   详细错误: {e}")
        return 1

    # 显示配置信息
    print("=" * 60)
    print(f"Profile: {config.profile.name}")
    print(f"描述: {config.profile.description}")
    print(f"版本: {config.profile.version}")
    print("=" * 60)
    print()

    print("📊 配置详情:")
    print(f"  模型架构: {config.model.architecture}")
    print(f"  隐藏层大小: {config.model.hidden_size}")
    print(f"  层数: {config.model.num_layers}")
    print(f"  注意力头: {config.model.num_attention_heads}")
    print()

    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  优化器: {config.training.optimizer}")
    print(f"  混合精度: {config.training.mixed_precision}")
    print()

    if config.training.distributed.enabled:
        print(f"  分布式训练: ✅ 启用")
        print(f"  Backend: {config.training.distributed.backend}")
        print(f"  World size: {config.training.distributed.world_size}")
    else:
        print(f"  分布式训练: ❌ 未启用")
    print()

    if config.vgpu.enabled:
        print(f"  Virtual Blackwell: ✅ 启用")
        print(f"  最大虚拟GPU: {config.vgpu.max_virtual_gpus}")
        print(f"  调度策略: {config.vgpu.scheduling}")
    else:
        print(f"  Virtual Blackwell: ❌ 未启用")
    print()

    # Demo 模式：只显示配置
    if args.demo:
        print("=" * 60)
        print("✅ Demo 模式 - 配置已加载成功")
        print("   要开始训练，请移除 --demo 参数")
        print("=" * 60)
        return 0

    # 开始训练
    print("=" * 60)
    print("🎯 开始训练")
    print("=" * 60)
    print()

    # 这里是实际训练逻辑
    # 注意：这是一个简化的示例，实际训练需要更多配置
    print("⚠️  注意: 完整的训练功能正在实现中")
    print()
    print("如需完整训练，请使用以下方式:")
    print()
    print("方式 1: Python 代码")
    print("```python")
    print(f"from apt.core.config import load_profile")
    print(f"from apt.trainops.engine import Trainer")
    print()
    print(f"config = load_profile('{args.profile}')")
    print(f"trainer = Trainer(config)")
    print(f"trainer.train(num_epochs={args.epochs})")
    print("```")
    print()

    print("方式 2: 运行示例脚本")
    print(f"    python examples/use_profiles.py")
    print()

    print("方式 3: 查看所有训练脚本")
    print(f"    ls examples/training_scripts/")
    print()

    print("=" * 60)
    print("💡 提示: 查看 docs/README.md 获取完整文档")
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
