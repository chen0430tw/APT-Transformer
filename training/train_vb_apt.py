"""
虚拟Blackwell训练启动脚本

完整的APT大模型训练流程，集成所有虚拟Blackwell优化：
- VGPU堆叠（多级内存管理）
- FP4量化（参数压缩）
- Flash Attention（注意力优化）
- 自动资源评估和配置

使用示例：
    # 小模型快速测试
    python train_vb_apt.py --config small --epochs 10

    # 中型模型训练
    python train_vb_apt.py --config medium --batch-size 16

    # 大模型训练（自动VGPU配置）
    python train_vb_apt.py --config large --vgpu-auto

    # 自定义配置
    python train_vb_apt.py --hidden-size 1024 --num-layers 24 --batch-size 8
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import argparse
import json
import time
from datetime import datetime
from typing import Dict, Optional
import os

# APT模型
from apt.core.modeling.apt_model import (
    APTLargeModel,
    APTModelConfiguration
)

# 虚拟Blackwell
from apt_model.optimization import (
    VGPUStack,
    create_vgpu_stack,
    VGPUResourceEstimator,
    ModelConfig,
)

# 集成模块
from test_vb_apt_integration import VBOptimizedAPTModel


# ============================================================================
# 配置预设
# ============================================================================

PRESETS = {
    'tiny': {
        'vocab_size': 5000,
        'hidden_size': 128,
        'num_layers': 2,
        'num_heads': 4,
        'max_position_embeddings': 256,
        'batch_size': 32,
    },
    'small': {
        'vocab_size': 10000,
        'hidden_size': 256,
        'num_layers': 4,
        'num_heads': 4,
        'max_position_embeddings': 512,
        'batch_size': 16,
    },
    'medium': {
        'vocab_size': 30000,
        'hidden_size': 768,
        'num_layers': 12,
        'num_heads': 12,
        'max_position_embeddings': 1024,
        'batch_size': 8,
    },
    'large': {
        'vocab_size': 50000,
        'hidden_size': 1024,
        'num_layers': 24,
        'num_heads': 16,
        'max_position_embeddings': 2048,
        'batch_size': 4,
    },
}


# ============================================================================
# 简单数据集（用于演示）
# ============================================================================

class DummyTextDataset(Dataset):
    """虚拟文本数据集（用于测试）"""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 1000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成随机序列
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        # 标签是输入右移一位
        labels = torch.roll(input_ids, -1)
        labels[-1] = 0  # 最后一个token用padding

        return {
            'input_ids': input_ids,
            'labels': labels
        }


# ============================================================================
# 训练器
# ============================================================================

class VBTrainer:
    """虚拟Blackwell训练器"""

    def __init__(self, args):
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 创建输出目录
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"training_{timestamp}.log"

        # 初始化
        self.model = None
        self.vgpu_stack = None
        self.optimizer = None
        self.train_loader = None

        self.log(f"虚拟Blackwell训练器初始化")
        self.log(f"设备: {self.device}")
        self.log(f"输出目录: {self.output_dir}")

    def log(self, message: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + '\n')

    def setup_model(self):
        """设置模型"""
        self.log("\n" + "="*70)
        self.log("步骤1: 模型配置")
        self.log("="*70)

        # 获取配置
        if self.args.config:
            preset = PRESETS[self.args.config]
            self.log(f"使用预设配置: {self.args.config}")
        else:
            preset = {}
            self.log("使用自定义配置")

        # 覆盖配置
        config_dict = {
            'vocab_size': self.args.vocab_size or preset.get('vocab_size', 10000),
            'hidden_size': self.args.hidden_size or preset.get('hidden_size', 256),
            'num_layers': self.args.num_layers or preset.get('num_layers', 4),
            'num_heads': self.args.num_heads or preset.get('num_heads', 4),
            'max_position_embeddings': self.args.seq_length or preset.get('max_position_embeddings', 512),
            'dropout': self.args.dropout,
        }

        self.batch_size = self.args.batch_size or preset.get('batch_size', 16)

        # 打印配置
        self.log("\n模型配置:")
        for key, value in config_dict.items():
            self.log(f"  {key}: {value}")
        self.log(f"  batch_size: {self.batch_size}")

        # 创建APT配置
        apt_config = APTModelConfiguration(**config_dict)

        # 资源评估
        if self.args.estimate_resources:
            self.log("\n进行资源评估...")
            self._estimate_resources(config_dict)

        # 创建VGPU Stack
        self.log("\n创建VGPU堆叠...")
        if self.args.vgpu_auto:
            self.vgpu_stack = self._create_auto_vgpu_stack(config_dict)
        else:
            self.vgpu_stack = create_vgpu_stack()

        # 创建模型
        self.log("\n创建虚拟Blackwell优化模型...")
        self.model = VBOptimizedAPTModel(
            apt_config,
            self.vgpu_stack,
            use_fp4=self.args.use_fp4,
            use_flash_attn=self.args.use_flash_attn
        ).to(self.device)

        # 统计参数
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.log(f"\n模型参数:")
        self.log(f"  总参数: {total_params:,} ({total_params/1e6:.1f}M)")
        self.log(f"  可训练: {trainable_params:,} ({trainable_params/1e6:.1f}M)")

    def _estimate_resources(self, config_dict: Dict):
        """估算资源需求"""
        model_config = ModelConfig(
            vocab_size=config_dict['vocab_size'],
            hidden_size=config_dict['hidden_size'],
            num_layers=config_dict['num_layers'],
            num_heads=config_dict['num_heads'],
            seq_length=config_dict['max_position_embeddings'],
            batch_size=self.batch_size,
            mixed_precision=self.args.mixed_precision,
            gradient_checkpointing=self.args.gradient_checkpointing
        )

        estimator = VGPUResourceEstimator()
        estimator.estimate_transformer(model_config)

        # 获取可用GPU
        available_gpus = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                available_gpus.append({
                    'device': f'cuda:{i}',
                    'vram_gb': props.total_memory / (1024**3),
                    'speed_gbps': 900  # 假设值
                })

        estimator.generate_vgpu_config(available_gpus or [{'device': 'cpu', 'vram_gb': 16, 'speed_gbps': 50}])
        estimator.print_report()

    def _create_auto_vgpu_stack(self, config_dict: Dict) -> VGPUStack:
        """自动创建VGPU堆叠配置"""
        # 简单估算所需容量
        param_count = (config_dict['vocab_size'] * config_dict['hidden_size'] +
                      config_dict['num_layers'] * config_dict['hidden_size'] * config_dict['hidden_size'] * 4)
        param_gb = param_count * 4 / (1024**3)  # float32

        self.log(f"  估算参数内存: {param_gb:.2f} GB")

        # 获取GPU容量
        if torch.cuda.is_available():
            gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.log(f"  GPU显存: {gpu_gb:.1f} GB")

            config = {
                'levels': [
                    {'capacity_mb': int(gpu_gb * 0.7 * 1024), 'device': 'cuda:0', 'speed_gbps': 900},
                    {'capacity_mb': int(param_gb * 2 * 1024), 'device': 'cpu', 'speed_gbps': 50},
                    {'capacity_mb': int(param_gb * 4 * 1024), 'device': 'ssd', 'speed_gbps': 7},
                ]
            }
        else:
            config = None

        from apt_model.optimization.vgpu_stack import VGPUStack
        return VGPUStack(config)

    def setup_data(self):
        """设置数据"""
        self.log("\n" + "="*70)
        self.log("步骤2: 数据加载")
        self.log("="*70)

        # 创建数据集
        train_dataset = DummyTextDataset(
            vocab_size=self.model.config.vocab_size,
            seq_len=self.model.config.max_position_embeddings,
            num_samples=self.args.num_samples
        )

        # 创建DataLoader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Windows上设为0
        )

        self.log(f"数据集大小: {len(train_dataset)}")
        self.log(f"批次数: {len(self.train_loader)}")
        self.log(f"批次大小: {self.batch_size}")

    def setup_optimizer(self):
        """设置优化器"""
        self.log("\n" + "="*70)
        self.log("步骤3: 优化器配置")
        self.log("="*70)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )

        self.log(f"优化器: AdamW")
        self.log(f"学习率: {self.args.learning_rate}")
        self.log(f"权重衰减: {self.args.weight_decay}")

    def train(self):
        """训练循环"""
        self.log("\n" + "="*70)
        self.log("步骤4: 开始训练")
        self.log("="*70)

        best_loss = float('inf')
        global_step = 0

        for epoch in range(self.args.epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            epoch_batches = 0

            self.model.train()

            for batch_idx, batch in enumerate(self.train_loader):
                step_start = time.time()

                # 数据移到设备
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                # 前向传播
                self.optimizer.zero_grad()
                output = self.model(input_ids)

                # 计算损失
                loss = nn.functional.cross_entropy(
                    output.view(-1, self.model.config.vocab_size),
                    labels.view(-1)
                )

                # 反向传播
                loss.backward()
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.args.max_grad_norm
                    )
                self.optimizer.step()

                # 统计
                step_time = time.time() - step_start
                epoch_loss += loss.item()
                epoch_batches += 1
                global_step += 1

                # 日志
                if (batch_idx + 1) % self.args.log_interval == 0:
                    avg_loss = epoch_loss / epoch_batches
                    self.log(f"Epoch {epoch+1}/{self.args.epochs} | "
                           f"Batch {batch_idx+1}/{len(self.train_loader)} | "
                           f"Loss: {loss.item():.4f} | "
                           f"Avg Loss: {avg_loss:.4f} | "
                           f"Time: {step_time*1000:.0f}ms")

                # 保存检查点
                if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                    self._save_checkpoint(epoch, global_step, loss.item())

            # Epoch结束
            epoch_time = time.time() - epoch_start
            avg_loss = epoch_loss / epoch_batches

            self.log(f"\n{'='*70}")
            self.log(f"Epoch {epoch+1} 完成")
            self.log(f"  平均损失: {avg_loss:.4f}")
            self.log(f"  耗时: {epoch_time:.2f}s")
            self.log(f"  吞吐量: {len(self.train_loader)/epoch_time:.2f} batch/s")
            self.log(f"{'='*70}\n")

            # VGPU统计
            if (epoch + 1) % self.args.stat_interval == 0:
                self.log("\nVGPU堆叠统计:")
                self.vgpu_stack.print_stats()

            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save_checkpoint(epoch, global_step, avg_loss, is_best=True)

        self.log("\n" + "="*70)
        self.log("训练完成！")
        self.log("="*70)
        self.log(f"最佳损失: {best_loss:.4f}")

    def _save_checkpoint(self, epoch: int, step: int, loss: float, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'global_step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': vars(self.args),
        }

        if is_best:
            filename = self.output_dir / 'best_model.pt'
            self.log(f"💾 保存最佳模型: {filename}")
        else:
            filename = self.output_dir / f'checkpoint_step_{step}.pt'
            self.log(f"💾 保存检查点: {filename}")

        torch.save(checkpoint, filename)


# ============================================================================
# 主函数
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='虚拟Blackwell APT训练')

    # 模型配置
    parser.add_argument('--config', type=str, choices=list(PRESETS.keys()),
                       help='使用预设配置 (tiny/small/medium/large)')
    parser.add_argument('--vocab-size', type=int, help='词表大小')
    parser.add_argument('--hidden-size', type=int, help='隐藏层大小')
    parser.add_argument('--num-layers', type=int, help='层数')
    parser.add_argument('--num-heads', type=int, help='注意力头数')
    parser.add_argument('--seq-length', type=int, help='序列长度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')

    # 训练配置
    parser.add_argument('--batch-size', type=int, help='批次大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='梯度裁剪')
    parser.add_argument('--num-samples', type=int, default=1000, help='训练样本数')

    # VGPU配置
    parser.add_argument('--vgpu-auto', action='store_true',
                       help='自动配置VGPU堆叠')
    parser.add_argument('--use-fp4', action='store_true',
                       help='启用FP4量化')
    parser.add_argument('--use-flash-attn', action='store_true',
                       help='启用Flash Attention')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='启用混合精度')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                       help='启用梯度检查点')

    # 其他
    parser.add_argument('--output-dir', type=str, default='./output',
                       help='输出目录')
    parser.add_argument('--log-interval', type=int, default=10,
                       help='日志间隔（批次）')
    parser.add_argument('--stat-interval', type=int, default=5,
                       help='统计间隔（轮数）')
    parser.add_argument('--save-steps', type=int, default=0,
                       help='保存间隔（步数，0=不保存）')
    parser.add_argument('--estimate-resources', action='store_true',
                       help='训练前估算资源需求')

    return parser.parse_args()


def main():
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║              虚拟Blackwell APT训练系统                              ║")
    print("║              VGPU Stack + Flash Optimization                       ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print("\n")

    # 解析参数
    args = parse_args()

    # 创建训练器
    trainer = VBTrainer(args)

    # 设置
    trainer.setup_model()
    trainer.setup_data()
    trainer.setup_optimizer()

    # 训练
    trainer.train()

    print("\n✅ 训练完成！")
    print(f"输出目录: {trainer.output_dir}")


if __name__ == "__main__":
    main()
