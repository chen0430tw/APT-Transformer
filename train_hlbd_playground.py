#!/usr/bin/env python3
"""
HLBD Playground训练脚本
使用APT Playground架构训练HLBD Hardcore数据集

特性:
- 🎢 Playground Theory (CosineAnnealingWarmRestarts)
- 🚀 RTX 3070优化 (混合精度 + 梯度累积)
- 🏷️  支持动态标签 ([EMOJI], [EN], [PY], [JP], [KR])
- 🔧 DBC-DAC梯度稳定
- 📊 实时可视化支持
"""

import os
import sys
import json
import time
import re
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from apt_model.modeling.apt_model import APTModel, APTModelConfiguration


# ============================================================================
# 动态标签Tokenizer (从HLBD脚本移植)
# ============================================================================

class DynamicTagTokenizer:
    """支持动态标签的字符级tokenizer"""

    def __init__(self, vocab_size=5000):
        # 基础特殊token
        self.vocab = {
            '[PAD]': 0, '[UNK]': 1, '[BOS]': 2, '[EOS]': 3,
            '[EMOJI]': 4, '[PHRASE]': 5, '[EN]': 6, '[PY]': 7, '[JP]': 8, '[KR]': 9,
        }
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        self.vocab_size = vocab_size

        self.char_to_id = self.vocab.copy()
        self.id_to_char = {v: k for k, v in self.vocab.items()}
        self.next_id = 10

        # 预编译标签正则
        self.tag_pattern = re.compile(r'(\[EMOJI\]|\[PHRASE\]|\[EN\]|\[PY\]|\[JP\]|\[KR\])')

    def _tokenize_text(self, text):
        """先切标签，再切字符"""
        tokens = []
        parts = self.tag_pattern.split(text)

        for part in parts:
            if part in self.vocab:
                # 标签 - 直接添加ID
                tokens.append(self.vocab[part])
            else:
                # 普通文本 - 逐字符
                for char in part:
                    if char.strip():
                        tokens.append(self._get_or_add_char(char))
                    elif char == ' ':
                        tokens.append(self._get_or_add_char(char))

        return tokens

    def _get_or_add_char(self, char):
        """动态添加字符"""
        if char not in self.char_to_id:
            if self.next_id < self.vocab_size:
                self.char_to_id[char] = self.next_id
                self.id_to_char[self.next_id] = char
                self.next_id += 1
            else:
                return self.unk_token_id
        return self.char_to_id[char]

    def encode(self, text):
        """编码文本"""
        ids = [self.bos_token_id]
        ids.extend(self._tokenize_text(text))
        ids.append(self.eos_token_id)
        return ids

    def decode(self, ids, skip_special_tokens=True):
        """解码ID序列"""
        chars = []
        for id in ids:
            if isinstance(id, torch.Tensor):
                id = id.item()

            if skip_special_tokens and id in [self.pad_token_id, self.bos_token_id,
                                               self.eos_token_id, self.unk_token_id]:
                continue

            char = self.id_to_char.get(id, '[UNK]')
            chars.append(char)

        return ''.join(chars)


# ============================================================================
# HLBD数据集
# ============================================================================

class HLBDPlaygroundDataset(Dataset):
    """HLBD Hardcore数据集"""

    def __init__(self, json_path: str, tokenizer: DynamicTagTokenizer, max_len: int = 128):
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

        # 预填充词汇表（避免多进程陷阱）
        print(f"   预填充词汇表...")
        for src, tgt in self.pairs:
            self.tokenizer.encode(src)
            self.tokenizer.encode(tgt)
        print(f"   ✓ 词汇表大小: {len(self.tokenizer.char_to_id)}")

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
# Playground配置
# ============================================================================

class PlaygroundConfig:
    """RTX 3070优化配置"""

    # 模型配置
    d_model = 256
    n_layers = 6
    n_heads = 8
    d_ff = 1024
    max_seq_len = 256
    dropout = 0.1

    # 训练配置
    batch_size = 16
    gradient_accumulation_steps = 2
    mixed_precision = True
    max_grad_norm = 1.0

    # Playground Theory
    base_lr = 3e-4
    min_lr = 1e-5
    T_0 = 10  # Cosine重启周期
    T_mult = 2

    # DBC-DAC
    use_dbc_dac = True


# ============================================================================
# 训练器
# ============================================================================

class HLBDPlaygroundTrainer:
    """HLBD Playground训练器"""

    def __init__(
        self,
        config: PlaygroundConfig,
        model: APTModel,
        train_loader: DataLoader,
        tokenizer: DynamicTagTokenizer,
        device: str = 'cuda'
    ):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.tokenizer = tokenizer
        self.device = device

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.base_lr,
            weight_decay=0.01  # ✅ Weight Decay
        )

        # Playground Theory调度器
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
        self.losses = []

    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        epoch_start = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 混合精度
            with autocast(enabled=self.config.mixed_precision):
                logits = self.model(input_ids)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                loss = loss / self.config.gradient_accumulation_steps

            # Backward
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # 梯度累积
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            if batch_idx % 20 == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"   Batch {batch_idx}/{len(self.train_loader)} | "
                      f"Loss: {loss.item():.4f} | LR: {current_lr:.6f}")

        # 更新学习率
        self.scheduler.step()

        avg_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start
        self.losses.append(avg_loss)

        return avg_loss, epoch_time

    def save_checkpoint(self, save_path: str, epoch: int):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'tokenizer_char_to_id': self.tokenizer.char_to_id,
            'tokenizer_id_to_char': self.tokenizer.id_to_char,
            'tokenizer_next_id': self.tokenizer.next_id,
            'tokenizer_vocab_size': self.tokenizer.vocab_size,
            'config': self.config.__dict__,
            'losses': self.losses
        }

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, save_path)
        print(f"✅ Checkpoint已保存: {save_path}")

    def save_progress_report(self, save_dir: str):
        """保存进度报告（用于可视化）"""
        report_path = Path(save_dir) / "experiment_report.json"
        report = {
            'control_losses': self.losses,  # 兼容可视化脚本
            'current_epoch': len(self.losses),
            'timestamp': time.time()
        }

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='HLBD Playground训练')

    parser.add_argument('--dataset', type=str, default='HLBD_Hardcore_Full.json',
                       help='HLBD数据集路径')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--save-dir', type=str, default='hlbd_playground',
                       help='保存目录')
    parser.add_argument('--save-interval', type=int, default=25,
                       help='保存间隔')

    args = parser.parse_args()

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 使用设备: {device}")

    # 配置
    config = PlaygroundConfig()

    # Tokenizer
    print(f"\n🔤 初始化Tokenizer（支持动态标签）...")
    tokenizer = DynamicTagTokenizer(vocab_size=5000)

    # 数据集
    dataset = HLBDPlaygroundDataset(args.dataset, tokenizer)

    # DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # 避免多进程陷阱
    )

    # 模型
    print(f"\n🏗️  构建APT模型...")
    model_config = APTModelConfiguration(
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        num_encoder_layers=config.n_layers,
        num_decoder_layers=config.n_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        use_dbc_dac=config.use_dbc_dac
    )
    model = APTModel(model_config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   总参数: {total_params:,}")

    # 训练器
    trainer = HLBDPlaygroundTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        tokenizer=tokenizer,
        device=device
    )

    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 训练循环
    print("\n" + "=" * 60)
    print("🎮 HLBD Playground训练开始")
    print("=" * 60)

    for epoch in range(args.epochs):
        print(f"\n📍 Epoch {epoch + 1}/{args.epochs}")

        # 训练
        loss, epoch_time = trainer.train_epoch(epoch)
        print(f"   Loss: {loss:.4f} | 用时: {epoch_time:.2f}s")

        # 保存进度（用于可视化）
        trainer.save_progress_report(args.save_dir)

        # 定期保存
        if (epoch + 1) % args.save_interval == 0:
            save_path = save_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            trainer.save_checkpoint(str(save_path), epoch + 1)

    # 保存最终模型
    final_path = save_dir / "final_model.pt"
    trainer.save_checkpoint(str(final_path), args.epochs)

    print("\n" + "=" * 60)
    print("✨ 训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
