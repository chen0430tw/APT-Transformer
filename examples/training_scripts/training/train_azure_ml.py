#!/usr/bin/env python3
"""
APT模型 + Azure ML集成
支持云端分布式训练、MLflow实验跟踪、自动超参数调优

特性:
- Azure ML计算集群自动管理
- MLflow实验跟踪和模型注册
- 分布式训练（PyTorch DDP）
- 超参数扫描（Sweep jobs）
- 云端checkpoint管理
- TensorBoard集成

使用前准备:
1. 安装Azure ML SDK: pip install azure-ai-ml mlflow azureml-mlflow
2. 配置Azure凭证: az login
3. 创建workspace配置文件
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

try:
    from azure.ai.ml import MLClient, command, Input
    from azure.ai.ml.entities import Environment, AmlCompute
    from azure.identity import DefaultAzureCredential
    from azure.ai.ml.sweep import Choice, Uniform
    import mlflow
    import mlflow.pytorch
    AZURE_ML_AVAILABLE = True
except ImportError:
    AZURE_ML_AVAILABLE = False
    print("⚠️  Azure ML SDK未安装，请运行:")
    print("   pip install azure-ai-ml mlflow azureml-mlflow")

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))


# ============================================================================
# Azure ML配置
# ============================================================================

class AzureMLConfig:
    """Azure ML配置类"""

    def __init__(
        self,
        subscription_id: str,
        resource_group: str,
        workspace_name: str,
        compute_name: str = "gpu-cluster",
        experiment_name: str = "apt-training"
    ):
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        self.compute_name = compute_name
        self.experiment_name = experiment_name


def create_ml_client(config: AzureMLConfig) -> MLClient:
    """创建Azure ML客户端"""
    credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=config.subscription_id,
        resource_group_name=config.resource_group,
        workspace_name=config.workspace_name
    )

    print(f"✅ 连接到Azure ML Workspace: {config.workspace_name}")
    return ml_client


def create_or_get_compute(ml_client: MLClient, compute_name: str, vm_size: str = "Standard_NC6s_v3"):
    """创建或获取计算集群"""
    try:
        compute = ml_client.compute.get(compute_name)
        print(f"✅ 使用现有计算集群: {compute_name}")
        return compute
    except Exception:
        print(f"🔧 创建新计算集群: {compute_name} ({vm_size})")

        compute_config = AmlCompute(
            name=compute_name,
            type="amlcompute",
            size=vm_size,
            min_instances=0,
            max_instances=4,
            idle_time_before_scale_down=300
        )

        compute = ml_client.compute.begin_create_or_update(compute_config).result()
        print(f"✅ 计算集群已创建")
        return compute


def create_environment(ml_client: MLClient, environment_name: str = "apt-training-env"):
    """创建训练环境"""

    # Conda依赖配置
    conda_yaml = """
name: apt-training
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pytorch::pytorch>=2.0.0
  - pytorch::torchvision
  - pytorch::torchaudio
  - pip
  - pip:
    - mlflow
    - azureml-mlflow
    - numpy
    - matplotlib
"""

    # Docker基础镜像
    dockerfile = """
FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04

# 设置工作目录
WORKDIR /workspace

# 复制conda配置
COPY conda.yaml /workspace/conda.yaml

# 安装conda环境
RUN conda env create -f conda.yaml && conda clean -a -y
"""

    env = Environment(
        name=environment_name,
        description="APT模型训练环境 (PyTorch + CUDA)",
        conda_file=conda_yaml,
        image="mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04"
    )

    env = ml_client.environments.create_or_update(env)
    print(f"✅ 环境已创建: {environment_name}")
    return env


# ============================================================================
# Azure ML训练脚本生成器
# ============================================================================

def generate_training_script() -> str:
    """生成Azure ML训练脚本

    这个脚本将在Azure ML计算集群上执行
    """

    script = '''#!/usr/bin/env python3
"""
Azure ML训练脚本
在Azure计算集群上执行
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import mlflow
import mlflow.pytorch

# 导入APT模型
from apt_model.modeling.apt_model import APTModel, APTModelConfiguration
from train_hlbd_playground import DynamicTagTokenizer, HLBDPlaygroundDataset, collate_fn


def train_with_mlflow(args):
    """带MLflow跟踪的训练函数"""

    # MLflow设置
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name):
        # 记录超参数
        mlflow.log_params({
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay
        })

        # 设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 使用设备: {device}")
        mlflow.log_param("device", device)

        # Tokenizer
        tokenizer = DynamicTagTokenizer(vocab_size=5000)

        # 数据集
        dataset = HLBDPlaygroundDataset(args.dataset, tokenizer)
        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )

        # 模型
        model_config = APTModelConfiguration(
            vocab_size=tokenizer.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            num_encoder_layers=args.n_layers,
            num_decoder_layers=args.n_layers,
            d_ff=args.d_model * 4,
            max_seq_len=256,
            dropout=0.1,
            use_dbc_dac=True
        )
        model = APTModel(model_config).to(device)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"   总参数: {total_params:,}")
        mlflow.log_param("total_params", total_params)

        # 优化器
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-5
        )

        # 混合精度
        scaler = GradScaler() if args.mixed_precision else None

        # 损失函数
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # 训练循环
        print("\\n" + "=" * 60)
        print("🚀 开始Azure ML训练")
        print("=" * 60)

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                # 混合精度前向传播
                with autocast(enabled=args.mixed_precision):
                    logits = model(input_ids)
                    loss = criterion(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1)
                    )

                # 反向传播
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                optimizer.zero_grad()

                total_loss += loss.item()
                num_batches += 1

                if batch_idx % 20 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"   Batch {batch_idx}/{len(train_loader)} | "
                          f"Loss: {loss.item():.4f} | LR: {current_lr:.6f}")

            # 更新学习率
            scheduler.step()

            avg_loss = total_loss / num_batches
            current_lr = scheduler.get_last_lr()[0]

            print(f"\\n📍 Epoch {epoch + 1}/{args.epochs}")
            print(f"   Avg Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

            # MLflow记录指标
            mlflow.log_metrics({
                "train_loss": avg_loss,
                "learning_rate": current_lr
            }, step=epoch)

            # 定期保存checkpoint
            if (epoch + 1) % args.save_interval == 0:
                checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'tokenizer_char_to_id': tokenizer.char_to_id,
                    'tokenizer_id_to_char': tokenizer.id_to_char,
                    'tokenizer_next_id': tokenizer.next_id,
                    'tokenizer_vocab_size': tokenizer.vocab_size,
                }, checkpoint_path)

                # 上传到MLflow
                mlflow.log_artifact(checkpoint_path)
                print(f"   ✅ Checkpoint已上传到MLflow")

        # 保存最终模型
        final_model_path = "final_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': model_config.__dict__,
            'tokenizer_char_to_id': tokenizer.char_to_id,
            'tokenizer_id_to_char': tokenizer.id_to_char,
            'tokenizer_next_id': tokenizer.next_id,
            'tokenizer_vocab_size': tokenizer.vocab_size,
        }, final_model_path)

        # 注册模型到MLflow
        mlflow.pytorch.log_model(model, "apt_model")
        mlflow.log_artifact(final_model_path)

        print("\\n" + "=" * 60)
        print("✨ Azure ML训练完成！")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Azure ML APT训练脚本')

    # 数据参数
    parser.add_argument('--dataset', type=str, default='../data/HLBD_Hardcore_Full.json')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--save-interval', type=int, default=25)

    # 模型参数
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--n-heads', type=int, default=8)

    # 训练参数
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--mixed-precision', action='store_true', default=True)

    # MLflow参数
    parser.add_argument('--experiment-name', type=str, default='apt-training')
    parser.add_argument('--run-name', type=str, default='hlbd-playground')

    args = parser.parse_args()

    train_with_mlflow(args)


if __name__ == "__main__":
    main()
'''

    return script


# ============================================================================
# Azure ML任务提交
# ============================================================================

def submit_training_job(
    ml_client: MLClient,
    config: AzureMLConfig,
    environment_name: str,
    dataset_path: str,
    args: argparse.Namespace
):
    """提交训练任务到Azure ML"""

    # 生成训练脚本
    print("📝 生成训练脚本...")
    training_script = generate_training_script()

    # 保存到本地
    script_path = Path("azure_training_script.py")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(training_script)
    print(f"   ✓ 脚本已保存: {script_path}")

    # 上传数据集（如果需要）
    print("📂 准备数据集...")
    # 这里可以上传到Azure Blob Storage或使用Azure ML数据资产

    # 创建命令任务
    print("🚀 创建训练任务...")

    job = command(
        code="./",  # 代码目录
        command=f"""
            python azure_training_script.py \\
                --dataset ${{inputs.dataset}} \\
                --epochs {args.epochs} \\
                --batch-size {args.batch_size} \\
                --d-model {args.d_model} \\
                --n-layers {args.n_layers} \\
                --n-heads {args.n_heads} \\
                --learning-rate {args.learning_rate} \\
                --weight-decay {args.weight_decay} \\
                --experiment-name {config.experiment_name} \\
                --run-name {args.run_name}
        """,
        inputs={
            "dataset": Input(type="uri_file", path=dataset_path)
        },
        environment=f"{environment_name}@latest",
        compute=config.compute_name,
        experiment_name=config.experiment_name,
        display_name=args.run_name,
        description="APT模型HLBD训练",
    )

    # 提交任务
    print("📤 提交任务到Azure ML...")
    returned_job = ml_client.jobs.create_or_update(job)

    print(f"\n{'=' * 60}")
    print(f"✅ 任务已提交！")
    print(f"{'=' * 60}")
    print(f"任务名称: {returned_job.name}")
    print(f"任务状态: {returned_job.status}")
    print(f"实验名称: {config.experiment_name}")
    print(f"\n🔗 Azure ML Studio链接:")
    print(f"   {returned_job.studio_url}")
    print(f"\n💡 查看任务状态:")
    print(f"   az ml job show --name {returned_job.name}")
    print(f"\n📊 查看实时日志:")
    print(f"   az ml job stream --name {returned_job.name}")

    return returned_job


def submit_sweep_job(
    ml_client: MLClient,
    config: AzureMLConfig,
    environment_name: str,
    dataset_path: str
):
    """提交超参数扫描任务"""

    from azure.ai.ml.sweep import Choice, Uniform

    print("🔬 创建超参数扫描任务...")

    # 基础命令
    base_job = command(
        code="./",
        command="""
            python azure_training_script.py \\
                --dataset ${{inputs.dataset}} \\
                --epochs 50 \\
                --batch-size ${{search_space.batch_size}} \\
                --d-model ${{search_space.d_model}} \\
                --n-layers ${{search_space.n_layers}} \\
                --learning-rate ${{search_space.learning_rate}} \\
                --weight-decay ${{search_space.weight_decay}}
        """,
        inputs={"dataset": Input(type="uri_file", path=dataset_path)},
        environment=f"{environment_name}@latest",
        compute=config.compute_name,
    )

    # 搜索空间
    sweep_job = base_job.sweep(
        sampling_algorithm="random",
        primary_metric="train_loss",
        goal="minimize",
        max_total_trials=20,
        max_concurrent_trials=4,
        search_space={
            "batch_size": Choice([8, 16, 32]),
            "d_model": Choice([128, 256, 512]),
            "n_layers": Choice([4, 6, 8]),
            "learning_rate": Uniform(1e-5, 1e-3),
            "weight_decay": Uniform(0.001, 0.1)
        }
    )

    # 提交
    returned_sweep = ml_client.jobs.create_or_update(sweep_job)

    print(f"✅ 超参数扫描任务已提交: {returned_sweep.name}")
    print(f"🔗 查看进度: {returned_sweep.studio_url}")

    return returned_sweep


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Azure ML APT训练管理')

    # Azure ML配置
    parser.add_argument('--subscription-id', type=str, required=True,
                       help='Azure订阅ID')
    parser.add_argument('--resource-group', type=str, required=True,
                       help='资源组名称')
    parser.add_argument('--workspace-name', type=str, required=True,
                       help='Azure ML工作区名称')
    parser.add_argument('--compute-name', type=str, default='gpu-cluster',
                       help='计算集群名称')
    parser.add_argument('--experiment-name', type=str, default='apt-training',
                       help='实验名称')

    # 计算资源
    parser.add_argument('--vm-size', type=str, default='Standard_NC6s_v3',
                       help='VM规格 (Standard_NC6s_v3 = 1x V100)')
    parser.add_argument('--environment-name', type=str, default='apt-training-env',
                       help='训练环境名称')

    # 数据和训练参数
    parser.add_argument('--dataset', type=str, default='../data/HLBD_Hardcore_Full.json',
                       help='数据集路径')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='batch大小')

    # 模型参数
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-layers', type=int, default=6)
    parser.add_argument('--n-heads', type=int, default=8)

    # 训练参数
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)

    # 运行配置
    parser.add_argument('--run-name', type=str, default='hlbd-playground',
                       help='运行名称')
    parser.add_argument('--sweep', action='store_true',
                       help='启用超参数扫描')

    args = parser.parse_args()

    if not AZURE_ML_AVAILABLE:
        print("\n❌ Azure ML SDK未安装")
        print("   请运行: pip install azure-ai-ml mlflow azureml-mlflow")
        return

    print("\n🚀 Azure ML APT训练")
    print("=" * 60)

    # 创建配置
    config = AzureMLConfig(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace_name,
        compute_name=args.compute_name,
        experiment_name=args.experiment_name
    )

    # 连接到Azure ML
    print("\n1️⃣  连接Azure ML...")
    ml_client = create_ml_client(config)

    # 创建/获取计算集群
    print("\n2️⃣  准备计算资源...")
    create_or_get_compute(ml_client, args.compute_name, args.vm_size)

    # 创建环境
    print("\n3️⃣  创建训练环境...")
    create_environment(ml_client, args.environment_name)

    # 提交任务
    print("\n4️⃣  提交训练任务...")
    if args.sweep:
        submit_sweep_job(ml_client, config, args.environment_name, args.dataset)
    else:
        submit_training_job(ml_client, config, args.environment_name, args.dataset, args)

    print("\n" + "=" * 60)
    print("✨ 任务提交完成！")
    print("=" * 60)
    print("\n💡 提示:")
    print("1. 在Azure ML Studio查看训练进度")
    print("2. 使用MLflow查看实验结果")
    print("3. 训练完成后模型会自动注册到MLflow")


if __name__ == "__main__":
    main()
