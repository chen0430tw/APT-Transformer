#!/usr/bin/env python3
"""
APT模型 + DeepSpeed集成
支持ZeRO-2和ZeRO-3优化，实现超大规模模型训练

特性:
- ZeRO-2: 分布式优化器和梯度 (适合多GPU)
- ZeRO-3: 分布式参数 (适合100B+模型)
- CPU卸载优化
- 混合精度训练 (FP16/BF16)
- 梯度累积
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("⚠️  DeepSpeed未安装，请运行: pip install deepspeed")

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from apt.apt_model.modeling.apt_model import APTModel, APTModelConfiguration
from train_hlbd_playground import DynamicTagTokenizer, HLBDPlaygroundDataset, collate_fn


# ============================================================================
# DeepSpeed配置生成
# ============================================================================

def create_deepspeed_config(
    stage: int = 2,
    train_batch_size: int = 64,
    gradient_accumulation_steps: int = 4,
    enable_fp16: bool = True,
    enable_bf16: bool = False,
    cpu_offload: bool = False,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01
):
    """
    生成DeepSpeed配置

    Args:
        stage: ZeRO优化阶段 (1, 2, 3)
        train_batch_size: 总batch大小
        gradient_accumulation_steps: 梯度累积步数
        enable_fp16: 启用FP16混合精度
        enable_bf16: 启用BF16混合精度 (需要Ampere+ GPU)
        cpu_offload: 启用CPU卸载
        learning_rate: 学习率
        weight_decay: 权重衰减
    """

    config = {
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": 1.0,

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": learning_rate,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": weight_decay
            }
        },

        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": learning_rate,
                "warmup_num_steps": 500,
                "total_num_steps": 10000
            }
        },

        "zero_optimization": {
            "stage": stage,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "allgather_bucket_size": 2e8
        },

        "steps_per_print": 100,
        "wall_clock_breakdown": False
    }

    # ZeRO-2/3 优化器卸载
    if stage >= 2 and cpu_offload:
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True
        }

    # ZeRO-3 参数卸载
    if stage == 3:
        config["zero_optimization"]["stage3_prefetch_bucket_size"] = 5e7
        config["zero_optimization"]["stage3_param_persistence_threshold"] = 1e6

        if cpu_offload:
            config["zero_optimization"]["offload_param"] = {
                "device": "cpu",
                "pin_memory": True
            }

    # 混合精度配置
    if enable_fp16:
        config["fp16"] = {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1,
            "initial_scale_power": 16
        }
    elif enable_bf16:
        config["bf16"] = {
            "enabled": True
        }

    return config


# ============================================================================
# DeepSpeed训练器
# ============================================================================

class DeepSpeedAPTTrainer:
    """APT模型DeepSpeed训练器"""

    def __init__(
        self,
        model: APTModel,
        train_loader: DataLoader,
        tokenizer: DynamicTagTokenizer,
        ds_config: dict,
        device: str = 'cuda'
    ):
        """
        Args:
            model: APT模型
            train_loader: 训练数据加载器
            tokenizer: Tokenizer
            ds_config: DeepSpeed配置字典
            device: 设备（DeepSpeed会自动处理）
        """
        if not DEEPSPEED_AVAILABLE:
            raise ImportError("DeepSpeed未安装")

        self.train_loader = train_loader
        self.tokenizer = tokenizer
        self.device = device

        # 使用DeepSpeed初始化模型
        self.model_engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=model,
            config=ds_config
        )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # 统计
        self.losses = []
        self.global_step = 0

    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model_engine.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # DeepSpeed自动处理设备移动
            input_ids = batch['input_ids'].to(self.model_engine.device)
            labels = batch['labels'].to(self.model_engine.device)

            # Forward
            outputs = self.model_engine(input_ids)

            # 计算损失
            loss = self.criterion(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1)
            )

            # DeepSpeed backward
            self.model_engine.backward(loss)

            # DeepSpeed step (自动处理梯度累积)
            self.model_engine.step()

            self.global_step += 1
            total_loss += loss.item()
            num_batches += 1

            # 打印进度
            if batch_idx % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"   Batch {batch_idx}/{len(self.train_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"LR: {current_lr:.6f} | "
                      f"Step: {self.global_step}")

        avg_loss = total_loss / num_batches
        self.losses.append(avg_loss)

        return avg_loss

    def save_checkpoint(self, save_dir: str, epoch: int):
        """保存DeepSpeed checkpoint"""
        save_path = Path(save_dir) / f"checkpoint_epoch_{epoch}"

        # DeepSpeed保存
        self.model_engine.save_checkpoint(str(save_path))

        # 额外保存tokenizer
        tokenizer_path = save_path / "tokenizer_state.json"
        with open(tokenizer_path, 'w') as f:
            json.dump({
                'char_to_id': self.tokenizer.char_to_id,
                'id_to_char': {str(k): v for k, v in self.tokenizer.id_to_char.items()},
                'next_id': self.tokenizer.next_id,
                'vocab_size': self.tokenizer.vocab_size
            }, f, ensure_ascii=False, indent=2)

        print(f"✅ DeepSpeed checkpoint已保存: {save_path}")

    def load_checkpoint(self, load_dir: str):
        """加载DeepSpeed checkpoint"""
        load_path = Path(load_dir)

        # DeepSpeed加载
        _, client_state = self.model_engine.load_checkpoint(str(load_path))

        # 加载tokenizer
        tokenizer_path = load_path / "tokenizer_state.json"
        if tokenizer_path.exists():
            with open(tokenizer_path) as f:
                state = json.load(f)
                self.tokenizer.char_to_id = state['char_to_id']
                self.tokenizer.id_to_char = {int(k): v for k, v in state['id_to_char'].items()}
                self.tokenizer.next_id = state['next_id']
                self.tokenizer.vocab_size = state['vocab_size']

        print(f"✅ DeepSpeed checkpoint已加载: {load_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='APT + DeepSpeed训练')

    # 数据和模型参数
    parser.add_argument('--dataset', type=str, default='../data/HLBD_Hardcore_Full.json',
                       help='HLBD数据集路径')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--save-dir', type=str, default='deepspeed_output',
                       help='保存目录')

    # DeepSpeed参数
    parser.add_argument('--deepspeed-config', type=str, default=None,
                       help='DeepSpeed配置文件路径 (如果提供，忽略其他DS参数)')
    parser.add_argument('--zero-stage', type=int, default=2, choices=[1, 2, 3],
                       help='ZeRO优化阶段')
    parser.add_argument('--train-batch-size', type=int, default=64,
                       help='总训练batch大小')
    parser.add_argument('--gradient-accumulation', type=int, default=4,
                       help='梯度累积步数')
    parser.add_argument('--cpu-offload', action='store_true',
                       help='启用CPU卸载')
    parser.add_argument('--fp16', action='store_true',
                       help='启用FP16混合精度')
    parser.add_argument('--bf16', action='store_true',
                       help='启用BF16混合精度')

    # 模型参数
    parser.add_argument('--d-model', type=int, default=256,
                       help='模型维度')
    parser.add_argument('--n-layers', type=int, default=6,
                       help='层数')
    parser.add_argument('--n-heads', type=int, default=8,
                       help='注意力头数')

    # DeepSpeed需要的参数
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='DeepSpeed分布式训练local rank')

    args = parser.parse_args()

    # DeepSpeed初始化
    deepspeed.init_distributed()

    # 获取或创建DeepSpeed配置
    if args.deepspeed_config and Path(args.deepspeed_config).exists():
        print(f"📄 加载DeepSpeed配置: {args.deepspeed_config}")
        with open(args.deepspeed_config) as f:
            ds_config = json.load(f)
    else:
        print(f"🔧 生成DeepSpeed配置 (ZeRO-{args.zero_stage})")
        ds_config = create_deepspeed_config(
            stage=args.zero_stage,
            train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            enable_fp16=args.fp16,
            enable_bf16=args.bf16,
            cpu_offload=args.cpu_offload
        )

        # 保存配置供参考
        config_path = Path(args.save_dir) / "deepspeed_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(ds_config, f, indent=2)
        print(f"   配置已保存: {config_path}")

    # 本地rank
    local_rank = args.local_rank if args.local_rank != -1 else 0

    if local_rank == 0:
        print(f"\n🚀 APT + DeepSpeed训练")
        print(f"   ZeRO阶段: {args.zero_stage}")
        print(f"   Batch大小: {args.train_batch_size}")
        print(f"   梯度累积: {args.gradient_accumulation}")
        print(f"   混合精度: {'FP16' if args.fp16 else 'BF16' if args.bf16 else 'FP32'}")
        print(f"   CPU卸载: {args.cpu_offload}")

    # Tokenizer
    if local_rank == 0:
        print(f"\n🔤 初始化Tokenizer...")
    tokenizer = DynamicTagTokenizer(vocab_size=5000)

    # 数据集
    dataset = HLBDPlaygroundDataset(args.dataset, tokenizer)

    # DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size // args.gradient_accumulation,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    # 模型
    if local_rank == 0:
        print(f"\n🏗️  构建APT模型...")

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

    model = APTModel(model_config)

    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   总参数: {total_params:,}")

    # 训练器
    trainer = DeepSpeedAPTTrainer(
        model=model,
        train_loader=train_loader,
        tokenizer=tokenizer,
        ds_config=ds_config
    )

    # 训练循环
    if local_rank == 0:
        print("\n" + "=" * 60)
        print("🚀 开始DeepSpeed训练")
        print("=" * 60)

    for epoch in range(args.epochs):
        if local_rank == 0:
            print(f"\n📍 Epoch {epoch + 1}/{args.epochs}")

        # 训练
        avg_loss = trainer.train_epoch(epoch)

        if local_rank == 0:
            print(f"   平均Loss: {avg_loss:.4f}")

            # 定期保存
            if (epoch + 1) % 25 == 0:
                trainer.save_checkpoint(args.save_dir, epoch + 1)

    # 保存最终模型
    if local_rank == 0:
        trainer.save_checkpoint(args.save_dir, args.epochs)
        print("\n" + "=" * 60)
        print("✨ DeepSpeed训练完成！")
        print("=" * 60)


if __name__ == "__main__":
    main()
