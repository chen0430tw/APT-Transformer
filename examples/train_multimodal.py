#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多模态APT模型训练脚本
Training script for multimodal APT model
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from apt.core.config.config import APTConfig
from apt.core.config.multimodal_config import MultimodalConfig
from apt.core.modeling.multimodal_model import MultimodalAPTModel, create_multimodal_model
from apt.core.data.multimodal_dataset import create_multimodal_dataloader
from apt.core.training.trainer import APTTrainer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultimodalTrainer:
    """多模态模型训练器"""

    def __init__(
        self,
        model: MultimodalAPTModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        save_dir: str = './checkpoints',
        log_interval: int = 100,
        eval_interval: int = 1000,
        save_interval: int = 1000
    ):
        """
        Args:
            model: 多模态模型
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 设备
            save_dir: 保存目录
            log_interval: 日志间隔
            eval_interval: 评估间隔
            save_interval: 保存间隔
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 优化器
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-4,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # 间隔设置
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval

        # 训练统计
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_dataloader):
            # 将数据移到设备
            batch = self._move_to_device(batch)

            # 前向传播
            outputs = self.model(
                input_ids=batch.get('text_input_ids'),
                attention_mask=batch.get('text_attention_mask'),
                pixel_values=batch.get('pixel_values'),
                audio_values=batch.get('audio_values'),
                labels=batch.get('labels'),
                return_dict=True
            )

            loss = outputs['loss']

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # 记录
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # 日志
            if self.global_step % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {self.epoch} | Step {self.global_step} | "
                    f"Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | LR: {lr:.2e}"
                )

            # 评估
            if self.val_dataloader is not None and self.global_step % self.eval_interval == 0:
                val_loss = self.evaluate()
                logger.info(f"Validation Loss: {val_loss:.4f}")

                # 保存最佳模型
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt')
                    logger.info(f"Saved best model with val_loss={val_loss:.4f}")

                self.model.train()

            # 定期保存
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')

        avg_epoch_loss = total_loss / num_batches
        self.train_losses.append(avg_epoch_loss)

        return avg_epoch_loss

    @torch.no_grad()
    def evaluate(self) -> float:
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_dataloader:
            batch = self._move_to_device(batch)

            outputs = self.model(
                input_ids=batch.get('text_input_ids'),
                attention_mask=batch.get('text_attention_mask'),
                pixel_values=batch.get('pixel_values'),
                audio_values=batch.get('audio_values'),
                labels=batch.get('labels'),
                return_dict=True
            )

            loss = outputs['loss']
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)

        return avg_loss

    def train(self, num_epochs: int):
        """训练多个epoch"""
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(num_epochs):
            self.epoch = epoch
            logger.info(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

            avg_loss = self.train_epoch()
            logger.info(f"Epoch {epoch + 1} completed | Avg Loss: {avg_loss:.4f}")

            # Epoch结束后评估
            if self.val_dataloader is not None:
                val_loss = self.evaluate()
                logger.info(f"Validation Loss: {val_loss:.4f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt')

            # 保存epoch checkpoint
            self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')

        logger.info("\nTraining completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

        # 保存训练历史
        self.save_training_history()

    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint_path = self.save_dir / filename

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def save_training_history(self):
        """保存训练历史"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'total_steps': self.global_step,
            'total_epochs': self.epoch + 1
        }

        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        logger.info(f"Saved training history to {history_path}")

    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """将batch移到设备"""
        result = {}
        for key, value in batch.items():
            if key == 'metadata':
                result[key] = value
            elif isinstance(value, torch.Tensor):
                result[key] = value.to(self.device)
            else:
                result[key] = value
        return result


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train Multimodal APT Model')

    # 数据参数
    parser.add_argument('--train_data', type=str, required=True, help='Training data path')
    parser.add_argument('--val_data', type=str, default=None, help='Validation data path')
    parser.add_argument('--modalities', nargs='+', default=['text', 'vision', 'audio'],
                        help='Modalities to use')

    # 模型参数
    parser.add_argument('--d_model', type=int, default=768, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--vision_encoder', type=str, default='simple',
                        choices=['simple', 'clip', 'vit', 'resnet'], help='Vision encoder type')
    parser.add_argument('--audio_encoder', type=str, default='simple',
                        choices=['simple', 'wav2vec2', 'hubert', 'whisper'], help='Audio encoder type')
    parser.add_argument('--fusion_method', type=str, default='cross_attention',
                        choices=['cross_attention', 'tri_modal', 'concatenate', 'add', 'gated'],
                        help='Fusion method')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')

    # 其他参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--log_interval', type=int, default=100, help='Log interval')
    parser.add_argument('--eval_interval', type=int, default=1000, help='Eval interval')
    parser.add_argument('--save_interval', type=int, default=1000, help='Save interval')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 创建配置
    config = APTConfig(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        vocab_size=50000  # 根据实际情况调整
    )

    multimodal_config = MultimodalConfig(
        enable_image='vision' in args.modalities,
        enable_audio='audio' in args.modalities
    )

    # 创建模型
    logger.info("Creating multimodal model...")
    model = create_multimodal_model(
        config=config,
        multimodal_config=multimodal_config,
        vision_encoder=args.vision_encoder,
        audio_encoder=args.audio_encoder,
        fusion_method=args.fusion_method
    )

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # 创建数据加载器
    logger.info("Loading training data...")

    # 这里需要根据实际情况创建tokenizer和processor
    # 示例使用None，实际使用时需要初始化
    tokenizer = None  # TODO: 初始化tokenizer
    vision_processor = None  # TODO: 初始化vision processor
    audio_processor = None  # TODO: 初始化audio processor

    train_dataloader = create_multimodal_dataloader(
        data_path=args.train_data,
        tokenizer=tokenizer,
        vision_processor=vision_processor,
        audio_processor=audio_processor,
        modalities=args.modalities,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_dataloader = None
    if args.val_data:
        logger.info("Loading validation data...")
        val_dataloader = create_multimodal_dataloader(
            data_path=args.val_data,
            tokenizer=tokenizer,
            vision_processor=vision_processor,
            audio_processor=audio_processor,
            modalities=args.modalities,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )

    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 创建学习率调度器
    from torch.optim.lr_scheduler import OneCycleLR
    total_steps = len(train_dataloader) * args.num_epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=args.warmup_steps / total_steps
    )

    # 创建训练器
    trainer = MultimodalTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        save_dir=args.save_dir,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval
    )

    # 开始训练
    trainer.train(num_epochs=args.num_epochs)


if __name__ == '__main__':
    main()
