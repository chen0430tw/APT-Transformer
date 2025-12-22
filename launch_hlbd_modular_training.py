#!/usr/bin/env python3
"""
HLBD模块化训练启动器
自动组合多个HLBD数据集进行联合训练

特性:
- 🔗 自动加载HLBD Full V2 (5000样本) + HLBD Hardcore V2 (5042样本)
- 📊 总计约10,000个训练样本
- 🎲 自动混合打散，防止模式坍缩
- 📈 统一训练流程
"""

import os
import sys
import subprocess
from pathlib import Path


def check_datasets():
    """检查数据集文件是否存在"""
    datasets = [
        'data/HLBD_Full_V2.json',
        'data/HLBD_Hardcore_Full_V2.json'
    ]

    print("=" * 60)
    print("检查数据集文件...")
    print("=" * 60)

    missing = []
    for dataset in datasets:
        if Path(dataset).exists():
            size = Path(dataset).stat().st_size / (1024 * 1024)  # MB
            print(f"✓ {dataset} ({size:.1f} MB)")
        else:
            print(f"✗ {dataset} (不存在)")
            missing.append(dataset)

    if missing:
        print("\n❌ 缺少数据集文件:")
        for m in missing:
            print(f"   - {m}")
        print("\n请先生成数据集:")
        print("   python3 tools/generate_hlbd_full_v2.py")
        print("   python3 tools/generate_hlbd_hardcore_v2.py")
        return False

    return True


def check_dependencies():
    """检查Python依赖"""
    print("\n" + "=" * 60)
    print("检查Python依赖...")
    print("=" * 60)

    dependencies = {
        'torch': 'PyTorch',
        'numpy': 'NumPy'
    }

    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name}")
            missing.append(name)

    if missing:
        print("\n❌ 缺少依赖:")
        for m in missing:
            print(f"   - {m}")
        print("\n安装命令:")
        print("   pip install torch numpy")
        return False

    return True


def main():
    """主启动流程"""
    print("=" * 60)
    print("🔗 HLBD模块化训练启动器")
    print("=" * 60)
    print()

    # 检查数据集
    if not check_datasets():
        return 1

    # 检查依赖
    if not check_dependencies():
        return 1

    # 训练配置
    print("\n" + "=" * 60)
    print("训练配置")
    print("=" * 60)
    print("数据集: HLBD Full V2 + HLBD Hardcore V2")
    print("总样本: ~10,000")
    print("训练轮数: 50")
    print("批次大小: 16 (梯度累积x2)")
    print("保存目录: hlbd_modular")
    print("=" * 60)
    print()

    # 构建训练命令
    cmd = [
        sys.executable,
        'training/train_hlbd_playground.py',
        '--datasets',
        'data/HLBD_Full_V2.json',
        'data/HLBD_Hardcore_Full_V2.json',
        '--epochs', '50',
        '--save-dir', 'hlbd_modular',
        '--save-interval', '10'
    ]

    print("启动命令:")
    print(" ".join(cmd))
    print()
    print("=" * 60)
    print("🚀 开始训练...")
    print("=" * 60)
    print()

    # 启动训练
    try:
        subprocess.run(cmd, check=True)

        print("\n" + "=" * 60)
        print("✅ 训练完成！")
        print("=" * 60)
        print()
        print("模型保存位置: hlbd_modular/")
        print("查看训练进度: hlbd_modular/experiment_report.json")
        return 0

    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        return 130


if __name__ == "__main__":
    sys.exit(main())
