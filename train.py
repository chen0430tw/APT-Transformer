#!/usr/bin/env python3
"""
APT统一训练启动器
支持多种训练后端的统一入口

支持的后端:
- local: 本地单卡训练（RTX 3070优化）
- playground: Playground Theory训练（推荐用于HLBD）
- deepspeed: 分布式训练（ZeRO-2/3优化）
- azure: Azure ML云端训练（MLflow跟踪）
- huggingface: HuggingFace Trainer（生态系统集成）

使用示例:
    # 本地Playground训练
    python train.py --backend playground --dataset HLBD_Hardcore_Full.json --epochs 100

    # DeepSpeed分布式训练
    python train.py --backend deepspeed --zero-stage 2 --epochs 100

    # Azure ML云端训练
    python train.py --backend azure --subscription-id <ID> --resource-group <RG> --workspace-name <WS>

    # HuggingFace Trainer（支持W&B）
    python train.py --backend huggingface --wandb --epochs 100
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional


# ============================================================================
# 后端检查
# ============================================================================

def check_backend_available(backend: str) -> tuple[bool, str]:
    """
    检查后端是否可用

    Returns:
        (is_available, message)
    """
    if backend == "local":
        return True, "本地训练始终可用"

    elif backend == "playground":
        script = Path("train_hlbd_playground.py")
        if not script.exists():
            return False, f"缺少训练脚本: {script}"
        return True, "Playground训练可用"

    elif backend == "deepspeed":
        try:
            import deepspeed
            script = Path("train_deepspeed.py")
            if not script.exists():
                return False, f"缺少训练脚本: {script}"
            return True, f"DeepSpeed {deepspeed.__version__} 可用"
        except ImportError:
            return False, "DeepSpeed未安装 (pip install deepspeed)"

    elif backend == "azure":
        try:
            from azure.ai.ml import MLClient
            script = Path("train_azure_ml.py")
            if not script.exists():
                return False, f"缺少训练脚本: {script}"
            return True, "Azure ML SDK可用"
        except ImportError:
            return False, "Azure ML SDK未安装 (pip install azure-ai-ml mlflow azureml-mlflow)"

    elif backend == "huggingface":
        try:
            import transformers
            script = Path("train_hf_trainer.py")
            if not script.exists():
                return False, f"缺少训练脚本: {script}"
            return True, f"HuggingFace Transformers {transformers.__version__} 可用"
        except ImportError:
            return False, "Transformers未安装 (pip install transformers datasets accelerate)"

    else:
        return False, f"未知后端: {backend}"


def list_backends():
    """列出所有可用后端"""
    print("\n" + "=" * 60)
    print("📋 可用训练后端")
    print("=" * 60)

    backends = ["local", "playground", "deepspeed", "azure", "huggingface"]

    for backend in backends:
        available, message = check_backend_available(backend)
        status = "✅" if available else "❌"

        # 描述
        descriptions = {
            "local": "本地单卡训练（基础）",
            "playground": "Playground Theory训练（推荐用于HLBD）",
            "deepspeed": "分布式训练（ZeRO优化，支持超大模型）",
            "azure": "Azure ML云端训练（MLflow跟踪）",
            "huggingface": "HuggingFace Trainer（W&B、TensorBoard集成）"
        }

        print(f"\n{status} {backend:15} - {descriptions[backend]}")
        print(f"   {message}")

    print("\n" + "=" * 60)


# ============================================================================
# 后端启动器
# ============================================================================

def launch_playground(args: argparse.Namespace):
    """启动Playground训练"""
    print("\n🎮 启动Playground训练...")

    cmd = [
        "python", "train_hlbd_playground.py",
        "--dataset", args.dataset,
        "--epochs", str(args.epochs),
        "--save-dir", args.output_dir,
        "--save-interval", str(args.save_interval)
    ]

    print(f"命令: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def launch_deepspeed(args: argparse.Namespace):
    """启动DeepSpeed训练"""
    print("\n🚀 启动DeepSpeed训练...")

    # 构建命令
    cmd = [
        "deepspeed",
        "--num_gpus", str(args.num_gpus),
        "train_deepspeed.py",
        "--dataset", args.dataset,
        "--epochs", str(args.epochs),
        "--save-dir", args.output_dir,
        "--zero-stage", str(args.zero_stage),
        "--train-batch-size", str(args.batch_size * args.num_gpus),
        "--gradient-accumulation", str(args.gradient_accumulation),
    ]

    # 模型参数
    cmd.extend([
        "--d-model", str(args.d_model),
        "--n-layers", str(args.n_layers),
        "--n-heads", str(args.n_heads)
    ])

    # 混合精度
    if args.fp16:
        cmd.append("--fp16")
    if args.bf16:
        cmd.append("--bf16")

    # CPU卸载
    if args.cpu_offload:
        cmd.append("--cpu-offload")

    print(f"命令: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def launch_azure(args: argparse.Namespace):
    """启动Azure ML训练"""
    print("\n☁️  启动Azure ML训练...")

    cmd = [
        "python", "train_azure_ml.py",
        "--subscription-id", args.azure_subscription_id,
        "--resource-group", args.azure_resource_group,
        "--workspace-name", args.azure_workspace_name,
        "--dataset", args.dataset,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--compute-name", args.azure_compute_name,
        "--experiment-name", args.azure_experiment_name,
        "--run-name", args.azure_run_name,
    ]

    # 模型参数
    cmd.extend([
        "--d-model", str(args.d_model),
        "--n-layers", str(args.n_layers),
        "--n-heads", str(args.n_heads)
    ])

    # 超参数扫描
    if args.azure_sweep:
        cmd.append("--sweep")

    print(f"命令: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def launch_huggingface(args: argparse.Namespace):
    """启动HuggingFace Trainer训练"""
    print("\n🤗 启动HuggingFace Trainer训练...")

    cmd = [
        "python", "train_hf_trainer.py",
        "--dataset", args.dataset,
        "--output-dir", args.output_dir,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--gradient-accumulation-steps", str(args.gradient_accumulation),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", str(args.weight_decay),
    ]

    # 模型参数
    cmd.extend([
        "--d-model", str(args.d_model),
        "--n-layers", str(args.n_layers),
        "--n-heads", str(args.n_heads)
    ])

    # 混合精度
    if args.fp16:
        cmd.append("--fp16")
    if args.bf16:
        cmd.append("--bf16")

    # Weights & Biases
    if args.wandb:
        cmd.append("--wandb")
        if args.wandb_project:
            cmd.extend(["--wandb-project", args.wandb_project])

    # 早停
    if args.early_stopping:
        cmd.append("--early-stopping")
        cmd.extend(["--early-stopping-patience", str(args.early_stopping_patience)])

    # DeepSpeed配置
    if args.hf_deepspeed_config:
        cmd.extend(["--deepspeed", args.hf_deepspeed_config])

    print(f"命令: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='APT统一训练启动器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看所有可用后端
  python train.py --list-backends

  # Playground训练（推荐用于HLBD）
  python train.py --backend playground --epochs 100

  # DeepSpeed分布式训练
  python train.py --backend deepspeed --zero-stage 2 --num-gpus 2

  # Azure ML云端训练
  python train.py --backend azure \\
      --azure-subscription-id <ID> \\
      --azure-resource-group <RG> \\
      --azure-workspace-name <WS>

  # HuggingFace + Weights & Biases
  python train.py --backend huggingface --wandb --wandb-project my-apt
        """
    )

    # 主要参数
    parser.add_argument('--backend', type=str,
                       choices=['playground', 'deepspeed', 'azure', 'huggingface'],
                       help='训练后端选择')
    parser.add_argument('--list-backends', action='store_true',
                       help='列出所有可用后端')

    # 通用训练参数
    parser.add_argument('--dataset', type=str, default='HLBD_Hardcore_Full.json',
                       help='数据集路径')
    parser.add_argument('--output-dir', type=str, default='training_output',
                       help='输出目录')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch大小')
    parser.add_argument('--save-interval', type=int, default=25,
                       help='保存间隔（epochs）')

    # 模型参数
    parser.add_argument('--d-model', type=int, default=256,
                       help='模型维度')
    parser.add_argument('--n-layers', type=int, default=6,
                       help='层数')
    parser.add_argument('--n-heads', type=int, default=8,
                       help='注意力头数')

    # 优化器参数
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='权重衰减')
    parser.add_argument('--gradient-accumulation', type=int, default=2,
                       help='梯度累积步数')

    # 混合精度
    parser.add_argument('--fp16', action='store_true',
                       help='启用FP16混合精度')
    parser.add_argument('--bf16', action='store_true',
                       help='启用BF16混合精度')

    # DeepSpeed特定参数
    deepspeed_group = parser.add_argument_group('DeepSpeed参数')
    deepspeed_group.add_argument('--num-gpus', type=int, default=1,
                                 help='GPU数量')
    deepspeed_group.add_argument('--zero-stage', type=int, default=2, choices=[1, 2, 3],
                                 help='ZeRO优化阶段')
    deepspeed_group.add_argument('--cpu-offload', action='store_true',
                                 help='启用CPU卸载')

    # Azure ML特定参数
    azure_group = parser.add_argument_group('Azure ML参数')
    azure_group.add_argument('--azure-subscription-id', type=str,
                            help='Azure订阅ID')
    azure_group.add_argument('--azure-resource-group', type=str,
                            help='资源组名称')
    azure_group.add_argument('--azure-workspace-name', type=str,
                            help='工作区名称')
    azure_group.add_argument('--azure-compute-name', type=str, default='gpu-cluster',
                            help='计算集群名称')
    azure_group.add_argument('--azure-experiment-name', type=str, default='apt-training',
                            help='实验名称')
    azure_group.add_argument('--azure-run-name', type=str, default='hlbd-playground',
                            help='运行名称')
    azure_group.add_argument('--azure-sweep', action='store_true',
                            help='启用超参数扫描')

    # HuggingFace特定参数
    hf_group = parser.add_argument_group('HuggingFace参数')
    hf_group.add_argument('--wandb', action='store_true',
                         help='启用Weights & Biases')
    hf_group.add_argument('--wandb-project', type=str, default='apt-training',
                         help='W&B项目名称')
    hf_group.add_argument('--early-stopping', action='store_true',
                         help='启用早停')
    hf_group.add_argument('--early-stopping-patience', type=int, default=5,
                         help='早停patience')
    hf_group.add_argument('--hf-deepspeed-config', type=str,
                         help='HuggingFace DeepSpeed配置文件')

    args = parser.parse_args()

    # 列出后端
    if args.list_backends:
        list_backends()
        return

    # 检查后端参数
    if not args.backend:
        print("❌ 错误: 请指定 --backend 参数")
        print("\n可用后端: playground, deepspeed, azure, huggingface")
        print("使用 --list-backends 查看详细信息")
        sys.exit(1)

    # 检查后端可用性
    available, message = check_backend_available(args.backend)
    if not available:
        print(f"❌ 后端不可用: {args.backend}")
        print(f"   原因: {message}")
        sys.exit(1)

    # 打印训练信息
    print("\n" + "=" * 60)
    print("🚀 APT统一训练启动器")
    print("=" * 60)
    print(f"后端: {args.backend}")
    print(f"数据集: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch大小: {args.batch_size}")
    print(f"模型: d_model={args.d_model}, n_layers={args.n_layers}, n_heads={args.n_heads}")
    print("=" * 60)

    # 启动对应后端
    try:
        if args.backend == "playground":
            launch_playground(args)
        elif args.backend == "deepspeed":
            launch_deepspeed(args)
        elif args.backend == "azure":
            # 检查必需参数
            if not all([args.azure_subscription_id, args.azure_resource_group, args.azure_workspace_name]):
                print("❌ 错误: Azure ML需要以下参数:")
                print("   --azure-subscription-id")
                print("   --azure-resource-group")
                print("   --azure-workspace-name")
                sys.exit(1)
            launch_azure(args)
        elif args.backend == "huggingface":
            launch_huggingface(args)

        print("\n" + "=" * 60)
        print("✨ 训练启动成功！")
        print("=" * 60)

    except subprocess.CalledProcessError as e:
        print(f"\n❌ 训练失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        sys.exit(0)


if __name__ == "__main__":
    main()
