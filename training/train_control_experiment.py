#!/usr/bin/env python3
"""
APT对照实验训练脚本
并行训练自生成和非自生成APT模型，用于对比分析

实验设计：
1. 对照组：标准Transformer (use_autopoietic=False)
2. 实验组：自生成Transformer (use_autopoietic=True)
3. 同一数据集、同一超参数、同一训练流程
4. 对比HLBD任务的学习效果

目标：
- 验证自生成机制是否提升HLBD学习能力
- 分析是否能减少"shortcut learning"（模型偷懒）
- 评估参数开销vs性能提升的权衡
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from apt_model.modeling.apt_control import (
    create_control_model,
    create_autopoietic_model,
    compare_model_architectures
)
from apt_model.tokenization.char_tokenizer import CharacterTokenizer


# ============================================================================
# HLBD数据集
# ============================================================================

class HLBDDataset(Dataset):
    """HLBD Hardcore数据集"""

    def __init__(self, json_path: str, tokenizer: CharacterTokenizer, max_len: int = 128):
        self.tokenizer = tokenizer
        self.max_len = max_len

        print(f"📂 加载HLBD数据集: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 合并所有模块
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
        input_ids = src_ids + [1] + tgt_ids

        # 截断
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]

        return {
            'input_ids': torch.tensor(input_ids[:-1], dtype=torch.long),
            'labels': torch.tensor(input_ids[1:], dtype=torch.long)
        }


def collate_fn(batch):
    """动态padding"""
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    max_len = max(len(ids) for ids in input_ids)

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

class ControlExperimentTrainer:
    """对照实验训练器"""

    def __init__(
        self,
        control_model: nn.Module,
        autopoietic_model: nn.Module,
        train_loader: DataLoader,
        device: str = 'cuda',
        lr: float = 3e-4
    ):
        self.control_model = control_model.to(device)
        self.autopoietic_model = autopoietic_model.to(device)
        self.train_loader = train_loader
        self.device = device

        # 优化器
        self.control_optimizer = torch.optim.AdamW(
            control_model.parameters(),
            lr=lr,
            weight_decay=0.01
        )
        self.autopoietic_optimizer = torch.optim.AdamW(
            autopoietic_model.parameters(),
            lr=lr,
            weight_decay=0.01
        )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # 统计
        self.control_losses = []
        self.autopoietic_losses = []

    def train_epoch(self, epoch: int):
        """训练一个epoch，同时训练两个模型"""
        self.control_model.train()
        self.autopoietic_model.train()

        control_total_loss = 0
        autopoietic_total_loss = 0
        num_batches = 0

        epoch_start = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # ============= 训练对照组模型 =============
            self.control_optimizer.zero_grad()
            control_logits = self.control_model(input_ids)
            control_loss = self.criterion(
                control_logits.view(-1, control_logits.size(-1)),
                labels.view(-1)
            )
            control_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.control_model.parameters(), 1.0)
            self.control_optimizer.step()

            # ============= 训练实验组模型 =============
            self.autopoietic_optimizer.zero_grad()
            autopoietic_logits = self.autopoietic_model(input_ids)
            autopoietic_loss = self.criterion(
                autopoietic_logits.view(-1, autopoietic_logits.size(-1)),
                labels.view(-1)
            )
            autopoietic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.autopoietic_model.parameters(), 1.0)
            self.autopoietic_optimizer.step()

            # 统计
            control_total_loss += control_loss.item()
            autopoietic_total_loss += autopoietic_loss.item()
            num_batches += 1

            # 打印进度
            if batch_idx % 20 == 0:
                print(f"   Batch {batch_idx}/{len(self.train_loader)} | "
                      f"Control Loss: {control_loss.item():.4f} | "
                      f"Autopoietic Loss: {autopoietic_loss.item():.4f}")

        # Epoch统计
        control_avg_loss = control_total_loss / num_batches
        autopoietic_avg_loss = autopoietic_total_loss / num_batches
        epoch_time = time.time() - epoch_start

        self.control_losses.append(control_avg_loss)
        self.autopoietic_losses.append(autopoietic_avg_loss)

        return {
            'control_loss': control_avg_loss,
            'autopoietic_loss': autopoietic_avg_loss,
            'epoch_time': epoch_time
        }

    @torch.no_grad()
    def evaluate(self, eval_pairs: List[Tuple[str, str]], tokenizer: CharacterTokenizer):
        """评估两个模型的生成质量"""
        self.control_model.eval()
        self.autopoietic_model.eval()

        results = {
            'control': [],
            'autopoietic': []
        }

        print("\n" + "=" * 60)
        print("生成质量评估")
        print("=" * 60)

        for src, expected in eval_pairs:
            src_ids = tokenizer.encode(src)
            input_tensor = torch.tensor([src_ids], dtype=torch.long).to(self.device)

            # 对照组生成
            control_output = self.control_model(input_tensor)
            control_pred = control_output.argmax(dim=-1)[0].tolist()
            control_text = tokenizer.decode(control_pred)

            # 实验组生成
            autopoietic_output = self.autopoietic_model(input_tensor)
            autopoietic_pred = autopoietic_output.argmax(dim=-1)[0].tolist()
            autopoietic_text = tokenizer.decode(autopoietic_pred)

            results['control'].append({
                'input': src,
                'expected': expected,
                'generated': control_text
            })
            results['autopoietic'].append({
                'input': src,
                'expected': expected,
                'generated': autopoietic_text
            })

            print(f"\n输入: {src}")
            print(f"期望: {expected}")
            print(f"对照组: {control_text}")
            print(f"实验组: {autopoietic_text}")

        return results

    def save_checkpoint(self, save_dir: str, epoch: int):
        """保存checkpoint"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # 保存对照组
        control_ckpt = save_path / f"control_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.control_model.state_dict(),
            'optimizer_state_dict': self.control_optimizer.state_dict(),
            'losses': self.control_losses,
            'config': self.control_model.config.to_dict()
        }, control_ckpt)

        # 保存实验组
        autopoietic_ckpt = save_path / f"autopoietic_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.autopoietic_model.state_dict(),
            'optimizer_state_dict': self.autopoietic_optimizer.state_dict(),
            'losses': self.autopoietic_losses,
            'config': self.autopoietic_model.config.to_dict()
        }, autopoietic_ckpt)

        print(f"\n✅ Checkpoint已保存 (Epoch {epoch})")
        print(f"   对照组: {control_ckpt}")
        print(f"   实验组: {autopoietic_ckpt}")

    def save_progress_report(self, save_dir: str):
        """保存实时进度报告（用于可视化）"""
        report_path = Path(save_dir) / "experiment_report.json"
        report = {
            'control_losses': list(self.control_losses),
            'autopoietic_losses': list(self.autopoietic_losses),
            'current_epoch': len(self.control_losses),
            'timestamp': time.time()
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="APT对照实验训练")

    parser.add_argument('--dataset', type=str, default='../data/HLBD_Hardcore_Full.json',
                        help='HLBD数据集路径')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch大小')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='学习率')
    parser.add_argument('--d-model', type=int, default=256,
                        help='模型维度')
    parser.add_argument('--n-layers', type=int, default=6,
                        help='层数')
    parser.add_argument('--n-heads', type=int, default=8,
                        help='注意力头数')
    parser.add_argument('--save-dir', type=str, default='control_experiments',
                        help='保存目录')
    parser.add_argument('--save-interval', type=int, default=25,
                        help='保存间隔')

    args = parser.parse_args()

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 使用设备: {device}")

    # Tokenizer
    print(f"\n🔤 初始化Tokenizer...")
    tokenizer = CharacterTokenizer(vocab_size=2000)

    # 数据集
    dataset = HLBDDataset(args.dataset, tokenizer)

    # 预填充词汇表
    print(f"\n📝 预填充词汇表...")
    for src, tgt in dataset.pairs:
        tokenizer.encode(src)
        tokenizer.encode(tgt)
    print(f"   ✓ 词汇表大小: {len(tokenizer.char_to_id)}")

    # DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    # 创建模型对
    print(f"\n🏗️  创建对照实验模型对...")
    control_model = create_control_model(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        use_dbc_dac=True
    )

    print()

    autopoietic_model = create_autopoietic_model(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        use_dbc_dac=True
    )

    # 比较架构
    compare_model_architectures(control_model, autopoietic_model)

    # 训练器
    trainer = ControlExperimentTrainer(
        control_model=control_model,
        autopoietic_model=autopoietic_model,
        train_loader=train_loader,
        device=device,
        lr=args.lr
    )

    # 准备评估样本
    eval_pairs = [
        ("12 + 34 = ?", "46"),
        ("三角形有几条边？", "3"),
        ("鼠后面是什么生肖？", "牛"),
        ("水在标准大气压下的沸点是多少摄氏度？", "100摄氏度"),
    ]

    # 训练循环
    print("\n" + "=" * 60)
    print("开始对照实验训练")
    print("=" * 60)

    for epoch in range(args.epochs):
        print(f"\n📍 Epoch {epoch + 1}/{args.epochs}")

        # 训练
        results = trainer.train_epoch(epoch)

        print(f"\n   对照组损失: {results['control_loss']:.4f}")
        print(f"   实验组损失: {results['autopoietic_loss']:.4f}")
        print(f"   用时: {results['epoch_time']:.2f}s")

        # 🎨 实时保存进度（用于可视化）
        trainer.save_progress_report(args.save_dir)

        # 定期评估
        if (epoch + 1) % args.save_interval == 0:
            trainer.evaluate(eval_pairs, tokenizer)
            trainer.save_checkpoint(args.save_dir, epoch + 1)

    # 最终评估
    print("\n" + "=" * 60)
    print("最终评估")
    print("=" * 60)
    final_results = trainer.evaluate(eval_pairs, tokenizer)

    # 保存最终模型
    trainer.save_checkpoint(args.save_dir, args.epochs)

    # 生成对比报告
    report_path = Path(args.save_dir) / "experiment_report.json"
    report = {
        'config': vars(args),
        'control_losses': trainer.control_losses,
        'autopoietic_losses': trainer.autopoietic_losses,
        'final_evaluation': final_results
    }

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 实验报告已保存: {report_path}")

    print("\n" + "=" * 60)
    print("✨ 对照实验完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
