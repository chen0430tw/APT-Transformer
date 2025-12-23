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


def find_latest_checkpoint(save_dir='hlbd_modular'):
    """查找最新的checkpoint"""
    save_path = Path(save_dir)
    if not save_path.exists():
        return None

    # 查找所有checkpoint文件
    checkpoints = list(save_path.glob('checkpoint_epoch_*.pt'))
    if not checkpoints:
        return None

    # 提取epoch数字并排序
    checkpoint_epochs = []
    for ckpt in checkpoints:
        try:
            # 文件名格式: checkpoint_epoch_10.pt
            epoch_num = int(ckpt.stem.split('_')[-1])
            checkpoint_epochs.append((epoch_num, ckpt))
        except ValueError:
            continue

    if not checkpoint_epochs:
        return None

    # 返回最新的checkpoint
    checkpoint_epochs.sort(key=lambda x: x[0], reverse=True)
    latest_epoch, latest_ckpt = checkpoint_epochs[0]

    return {
        'path': latest_ckpt,
        'epoch': latest_epoch,
        'size_mb': latest_ckpt.stat().st_size / (1024 * 1024)
    }


def ask_resume_training(checkpoint_info):
    """询问是否恢复训练"""
    print("\n" + "=" * 60)
    print("🔍 发现已有checkpoint")
    print("=" * 60)
    print(f"文件: {checkpoint_info['path']}")
    print(f"Epoch: {checkpoint_info['epoch']}")
    print(f"大小: {checkpoint_info['size_mb']:.1f} MB")
    print()

    while True:
        response = input("是否从此checkpoint恢复训练? [Y/n]: ").strip().lower()
        if response in ['', 'y', 'yes']:
            return True
        elif response in ['n', 'no']:
            print("\n⚠️  将从epoch 1重新开始训练（已有checkpoint将被覆盖）")
            confirm = input("确认重新开始? [y/N]: ").strip().lower()
            if confirm in ['y', 'yes']:
                return False
            # 否则继续循环询问
        else:
            print("请输入 y 或 n")


def main():
    """主启动流程"""
    # 确保在项目根目录运行
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    os.chdir(project_root)

    print("=" * 60)
    print("🔗 HLBD模块化训练启动器")
    print("=" * 60)
    print(f"项目目录: {project_root}")
    print()

    # 检查数据集
    if not check_datasets():
        return 1

    # 检查依赖
    if not check_dependencies():
        return 1

    # 检查是否有已存在的checkpoint
    checkpoint_info = find_latest_checkpoint('hlbd_modular')
    resume_training = False

    if checkpoint_info:
        resume_training = ask_resume_training(checkpoint_info)

    # 训练配置
    print("\n" + "=" * 60)
    print("训练配置")
    print("=" * 60)
    print("数据集: HLBD Full V2 + HLBD Hardcore V2")
    print("总样本: ~10,000")
    print("训练轮数: 50")
    print("批次大小: 16 (梯度累积x2)")
    print("保存目录: hlbd_modular")
    if resume_training:
        print(f"恢复模式: 从Epoch {checkpoint_info['epoch']} 继续")
    else:
        print("训练模式: 从头开始")
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

    # 添加恢复参数
    if resume_training:
        cmd.extend(['--resume', str(checkpoint_info['path'])])

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
