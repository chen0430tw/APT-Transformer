#!/usr/bin/env python3
"""
APT Playground Training Script
使用HuggingFace数据集训练APT模型

特性：
1. HuggingFace数据集集成 (wikitext-103-v1流式加载)
2. "Playground Theory" - CosineAnnealingWarmRestarts动态学习率
3. RTX 3070优化 (混合精度 + 梯度累积)
4. DBC-DAC梯度稳定
5. 支持HLBD Legacy模式调试
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# HuggingFace datasets
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  警告: datasets库未安装，HuggingFace功能不可用")
    print("   安装: pip install datasets")

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from apt_model.modeling.apt_model import APTModel, APTModelConfiguration
from apt_model.tokenization.char_tokenizer import CharacterTokenizer


# ============================================================================
# 配置类：Project Swift (RTX 3070优化)
# ============================================================================

class ProjectSwiftConfig:
    """
    RTX 3070 Laptop优化配置

    硬件规格：
    - VRAM: 8GB
    - CUDA Cores: 5120
    - Memory Bus: 256-bit
    - TDP: 125W

    优化策略：
    - 混合精度训练 (FP16)
    - 梯度累积 (减少batch size内存压力)
    - 模型大小适中 (d_model=512, n_layers=12)
    """

    # 模型配置
    d_model = 512
    n_layers = 12
    n_heads = 8
    d_ff = 2048
    max_seq_len = 512
    dropout = 0.1

    # 训练配置
    batch_size = 8           # 小batch适配8GB VRAM
    gradient_accumulation_steps = 4  # 有效batch=32
    mixed_precision = True   # FP16混合精度
    max_grad_norm = 1.0      # 梯度裁剪

    # Playground Theory: 动态学习率
    base_lr = 3e-4
    min_lr = 1e-5
    T_0 = 10                 # 重启周期 (epochs)
    T_mult = 2               # 周期倍增

    # DBC-DAC配置
    use_dbc_dac = True
    rank_ratio_proj = 0.95
    threshold = 1e-6


# ============================================================================
# HuggingFace数据集适配器
# ============================================================================

class HuggingFaceDataset(Dataset):
    """
    HuggingFace数据集适配器
    支持流式加载大型数据集 (如wikitext-103-v1)
    """

    def __init__(
        self,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-103-v1",
        split: str = "train",
        tokenizer: Optional[CharacterTokenizer] = None,
        max_length: int = 512,
        streaming: bool = True,
        max_samples: Optional[int] = None
    ):
        """
        Args:
            dataset_name: HF数据集名称
            dataset_config: 数据集配置
            split: 数据集划分 (train/validation/test)
            tokenizer: 字符tokenizer
            max_length: 最大序列长度
            streaming: 是否流式加载
            max_samples: 最大样本数 (用于快速测试)
        """
        if not HF_AVAILABLE:
            raise ImportError("请安装datasets库: pip install datasets")

        self.tokenizer = tokenizer or CharacterTokenizer()
        self.max_length = max_length

        print(f"📦 加载HuggingFace数据集: {dataset_name}/{dataset_config} ({split})")
        print(f"   流式加载: {streaming}")

        # 加载数据集
        self.dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=split,
            streaming=streaming
        )

        # 如果是流式，转换为列表（限制样本数）
        if streaming:
            print(f"   转换流式数据...")
            samples = []
            for i, example in enumerate(self.dataset):
                if max_samples and i >= max_samples:
                    break
                if example['text'].strip():  # 过滤空文本
                    samples.append(example)
            self.samples = samples
            print(f"   ✓ 加载 {len(self.samples)} 个样本")
        else:
            self.samples = [ex for ex in self.dataset if ex['text'].strip()]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]['text']

        # 字符级tokenization
        tokens = self.tokenizer.encode(text)

        # 截断或填充
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        # 转换为tensor
        input_ids = torch.tensor(tokens, dtype=torch.long)

        # 自回归任务: 输入和目标错位
        # input: [BOS, t1, t2, ..., tn-1]
        # target: [t1, t2, t3, ..., tn]
        return {
            'input_ids': input_ids[:-1] if len(input_ids) > 1 else input_ids,
            'labels': input_ids[1:] if len(input_ids) > 1 else input_ids
        }


# ============================================================================
# HLBD Legacy数据集 (用于调试对比)
# ============================================================================

class HLBDLegacyDataset(Dataset):
    """HLBD Hardcore数据集加载器"""

    def __init__(self, json_path: str, tokenizer: CharacterTokenizer):
        self.tokenizer = tokenizer

        print(f"📂 加载HLBD数据集: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 合并所有模块的数据
        self.pairs = []
        for module_name, module_data in data['data'].items():
            for item in module_data:
                self.pairs.append((item['input'], item['output']))

        print(f"   ✓ 加载 {len(self.pairs)} 个训练对")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]

        # Encode
        src_ids = self.tokenizer.encode(src)
        tgt_ids = self.tokenizer.encode(tgt)

        # 拼接: [src, SEP, tgt]
        # 假设vocab[1]是SEP token
        input_ids = src_ids + [1] + tgt_ids

        return {
            'input_ids': torch.tensor(input_ids[:-1], dtype=torch.long),
            'labels': torch.tensor(input_ids[1:], dtype=torch.long)
        }


# ============================================================================
# Collate函数：动态padding
# ============================================================================

def collate_fn(batch):
    """
    动态padding批次数据
    """
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    # 找到最大长度
    max_len = max(len(ids) for ids in input_ids)

    # Padding
    padded_input = []
    padded_labels = []

    for inp, lab in zip(input_ids, labels):
        pad_len = max_len - len(inp)
        padded_input.append(torch.cat([inp, torch.zeros(pad_len, dtype=torch.long)]))
        padded_labels.append(torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)]))

    return {
        'input_ids': torch.stack(padded_input),
        'labels': torch.stack(padded_labels)
    }


# ============================================================================
# 训练器
# ============================================================================

class APTPlaygroundTrainer:
    """
    APT Playground训练器
    实现"Playground Theory"和RTX 3070优化
    """

    def __init__(
        self,
        config: ProjectSwiftConfig,
        model: APTModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda'
    ):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.base_lr,
            weight_decay=0.01
        )

        # Playground Theory: CosineAnnealingWarmRestarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.T_0,
            T_mult=config.T_mult,
            eta_min=config.min_lr
        )

        # 混合精度
        self.scaler = GradScaler() if config.mixed_precision else None

        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # 统计
        self.global_step = 0
        self.epoch = 0

    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        epoch_start = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 混合精度上下文
            with autocast(enabled=self.config.mixed_precision):
                # Forward
                logits = self.model(input_ids)

                # 计算损失
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )

                # 梯度累积
                loss = loss / self.config.gradient_accumulation_steps

            # Backward (混合精度)
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # 梯度累积步骤
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                # 优化器步骤
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            # 每100步打印
            if batch_idx % 100 == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"   Step {batch_idx}/{len(self.train_loader)} | "
                      f"Loss: {loss.item():.4f} | LR: {current_lr:.6f}")

        # Epoch结束后更新学习率
        self.scheduler.step()

        avg_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start

        return avg_loss, epoch_time

    @torch.no_grad()
    def validate(self):
        """验证"""
        if not self.val_loader:
            return None

        self.model.eval()
        total_loss = 0
        num_batches = 0

        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            logits = self.model(input_ids)
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, save_path: str, **kwargs):
        """保存checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.__dict__,
            **kwargs
        }

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, save_path)
        print(f"✅ Checkpoint已保存: {save_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="APT Playground Training")

    # 数据集选择
    parser.add_argument('--dataset', type=str, default='wikitext',
                        choices=['wikitext', 'hlbd'],
                        help='数据集类型')
    parser.add_argument('--hlbd-path', type=str, default='HLBD_Hardcore_Full.json',
                        help='HLBD数据集路径')
    parser.add_argument('--max-samples', type=int, default=10000,
                        help='最大训练样本数 (用于快速测试)')

    # 训练配置
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--save-dir', type=str, default='playground_checkpoints',
                        help='Checkpoint保存目录')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='保存间隔 (epochs)')

    # DBC-DAC
    parser.add_argument('--no-dbc-dac', action='store_true',
                        help='禁用DBC-DAC')

    args = parser.parse_args()

    # 设备检查
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 使用设备: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 配置
    config = ProjectSwiftConfig()
    if args.no_dbc_dac:
        config.use_dbc_dac = False

    print("\n" + "=" * 60)
    print("APT Playground Training")
    print("Playground Theory: CosineAnnealingWarmRestarts")
    print("=" * 60)
    print(f"\n📋 配置:")
    print(f"   模型: d_model={config.d_model}, n_layers={config.n_layers}, n_heads={config.n_heads}")
    print(f"   Batch Size: {config.batch_size} × {config.gradient_accumulation_steps} (累积)")
    print(f"   学习率: {config.base_lr} (Cosine Annealing, T_0={config.T_0})")
    print(f"   混合精度: {config.mixed_precision}")
    print(f"   DBC-DAC: {config.use_dbc_dac}")

    # Tokenizer
    print(f"\n🔤 初始化Tokenizer...")
    tokenizer = CharacterTokenizer(vocab_size=2000)

    # 数据集
    print(f"\n📦 准备数据集: {args.dataset}")
    if args.dataset == 'wikitext':
        train_dataset = HuggingFaceDataset(
            dataset_name='wikitext',
            dataset_config='wikitext-103-v1',
            split='train',
            tokenizer=tokenizer,
            max_length=config.max_seq_len,
            max_samples=args.max_samples
        )
        val_dataset = HuggingFaceDataset(
            dataset_name='wikitext',
            dataset_config='wikitext-103-v1',
            split='validation',
            tokenizer=tokenizer,
            max_length=config.max_seq_len,
            max_samples=args.max_samples // 10
        )
    else:  # HLBD
        train_dataset = HLBDLegacyDataset(args.hlbd_path, tokenizer)
        val_dataset = None

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # 避免多进程陷阱
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

    # 模型
    print(f"\n🏗️  构建APT模型...")
    model_config = APTModelConfiguration(
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        use_dbc_dac=config.use_dbc_dac,
        rank_ratio_proj=config.rank_ratio_proj,
        threshold=config.threshold
    )
    model = APTModel(model_config)

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   总参数: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")

    # 训练器
    trainer = APTPlaygroundTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 训练循环
    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    for epoch in range(args.epochs):
        trainer.epoch = epoch
        print(f"\n📍 Epoch {epoch + 1}/{args.epochs}")

        # 训练
        train_loss, epoch_time = trainer.train_epoch()
        print(f"   训练损失: {train_loss:.4f}")
        print(f"   用时: {epoch_time:.2f}s")

        # 验证
        if val_loader:
            val_loss = trainer.validate()
            print(f"   验证损失: {val_loss:.4f}")

        # 保存checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_path = save_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            trainer.save_checkpoint(
                str(save_path),
                train_loss=train_loss,
                val_loss=val_loss if val_loader else None
            )

    # 保存最终模型
    final_path = save_dir / "final_model.pt"
    trainer.save_checkpoint(str(final_path))

    print("\n" + "=" * 60)
    print("✨ 训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
