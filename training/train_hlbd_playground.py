#!/usr/bin/env python3
"""
HLBD Playground训练脚本
使用APT Playground架构训练HLBD数据集（支持模块化多数据集训练）

特性:
- 🔗 模块化训练 - 支持同时加载多个HLBD数据集
- 📊 自动格式识别 - HLBD Full (8层) + HLBD Hardcore (模块化)
- 🎢 Playground Theory (CosineAnnealingWarmRestarts)
- 🚀 RTX 3070优化 (混合精度 + 梯度累积)
- 🏷️  支持动态标签 ([EMOJI], [EN], [PY], [JP], [KR], [PHRASE])
- 🔧 DBC-DAC梯度稳定
- 📊 实时可视化支持
- 🎲 数据稀释学 - 自动打散混合，防止模式坍缩

数据集格式支持:
1. HLBD Full: 8层分层语言结构 (包含Level 3句法层)
2. HLBD Hardcore: 严格逻辑问答 (几何、算术、生肖、物理、英文)

使用示例:
  # 单数据集
  python train_hlbd_playground.py --dataset data/HLBD_Hardcore_Full_V2.json

  # 多数据集联合训练 (~10,000样本)
  python train_hlbd_playground.py --datasets data/HLBD_Full_V2.json data/HLBD_Hardcore_Full_V2.json --epochs 50
"""

import os
import sys
import json
import time
import math
import re
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# ============================================================================
# 修复路径问题：添加项目根目录到sys.path
# ============================================================================
# 获取当前脚本所在目录的父目录（即项目根目录）
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

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
    """HLBD模块化数据集 - 支持多数据集和多格式"""

    def __init__(self, json_paths, tokenizer: DynamicTagTokenizer, max_len: int = 128):
        """
        初始化HLBD数据集

        Args:
            json_paths: 单个路径(str)或多个路径(list)
            tokenizer: DynamicTagTokenizer实例
            max_len: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_len = max_len

        # 统一处理为列表
        if isinstance(json_paths, str):
            json_paths = [json_paths]

        self.pairs = []
        self.dataset_stats = {}

        print(f"\n📚 模块化HLBD数据集加载器")
        print(f"   数据集数量: {len(json_paths)}")
        print("=" * 60)

        # 加载所有数据集
        for i, json_path in enumerate(json_paths, 1):
            print(f"\n📂 [{i}/{len(json_paths)}] 加载数据集: {json_path}")
            dataset_pairs = self._load_single_dataset(json_path)

            if dataset_pairs:
                self.pairs.extend(dataset_pairs)
                dataset_name = Path(json_path).stem
                self.dataset_stats[dataset_name] = len(dataset_pairs)
                print(f"   ✓ 成功加载 {len(dataset_pairs)} 个训练对")
            else:
                print(f"   ⚠️  数据集为空或加载失败")

        # 打散混合
        random.shuffle(self.pairs)

        print(f"\n📊 数据集统计:")
        for name, count in self.dataset_stats.items():
            percentage = count / len(self.pairs) * 100 if self.pairs else 0
            print(f"   {name}: {count} 对 ({percentage:.1f}%)")
        print(f"   总计: {len(self.pairs)} 个训练对")
        print(f"   ✓ 已混合打散")

        # 预填充词汇表（避免多进程陷阱）
        print(f"\n🔤 预填充词汇表...")
        for src, tgt in self.pairs:
            self.tokenizer.encode(src)
            self.tokenizer.encode(tgt)
        print(f"   ✓ 词汇表大小: {len(self.tokenizer.char_to_id)}")
        print("=" * 60)

    def _load_single_dataset(self, json_path: str):
        """加载单个数据集并转换为统一格式"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 检测数据集类型
            if 'samples' in data:
                # HLBD Full格式（8层结构）
                print(f"   格式: HLBD Full (8层结构)")
                return self._process_hlbd_full(data['samples'])
            elif 'data' in data:
                # HLBD Hardcore格式（模块化）
                print(f"   格式: HLBD Hardcore (模块化)")
                return self._process_hlbd_hardcore(data['data'])
            else:
                print(f"   ⚠️  未知的数据集格式")
                return []

        except Exception as e:
            print(f"   ❌ 加载失败: {e}")
            return []

    def _process_hlbd_hardcore(self, data):
        """处理HLBD Hardcore格式（模块化Q&A）"""
        pairs = []
        for module_name, module_data in data.items():
            for item in module_data:
                src = item['input']
                tgt = item['output']
                pairs.append((src, tgt))
        return pairs

    def _process_hlbd_full(self, samples):
        """处理HLBD Full格式（8层结构）"""
        pairs = []

        for sample in samples:
            concept = sample.get('concept', '')

            # 构建输入：概念 + 关键层级信息
            input_parts = [f"概念: {concept}"]

            # Level 1: 字卡 + Emoji
            if 'level_1' in sample:
                char_card = sample['level_1'].get('字卡', '')
                emoji = sample['level_1'].get('emoji', '')
                input_parts.append(f"[EMOJI] {char_card} {emoji}")

            # Level 2: 短语
            if 'level_2' in sample:
                phrase = sample['level_2'].get('短语', '')
                input_parts.append(f"[PHRASE] {phrase}")

            # Level 3: 数学（句法结构）← 重要！
            if 'level_3' in sample:
                math_expr = sample['level_3'].get('数学', '')
                input_parts.append(f"句法结构: {math_expr}")

            input_text = "\n".join(input_parts)

            # 构建输出：多语言翻译
            output_parts = []

            # Level 4: 拼音
            if 'level_4' in sample:
                pinyin = sample['level_4'].get('拼音', '')
                output_parts.append(f"[PY] {pinyin}")

            # Level 5: 英文
            if 'level_5' in sample:
                english = sample['level_5'].get('英文', '')
                output_parts.append(f"[EN] {english}")

            # Level 6: 中文
            if 'level_6' in sample:
                chinese = sample['level_6'].get('中文', '')
                output_parts.append(f"{chinese}")

            # Level 7: 日文
            if 'level_7' in sample:
                japanese = sample['level_7'].get('日文', '')
                if japanese:
                    output_parts.append(f"[JP] {japanese}")

            # Level 8: 韩文
            if 'level_8' in sample:
                korean = sample['level_8'].get('韩文', '')
                if korean:
                    output_parts.append(f"[KR] {korean}")

            output_text = "\n".join(output_parts)

            pairs.append((input_text, output_text))

        return pairs

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
    num_heads = 8  # ✅ 修复：统一使用num_heads（256/8=32可整除）
    d_ff = 1024
    max_seq_len = 256
    dropout = 0.1

    # 训练配置
    batch_size = 16
    gradient_accumulation_steps = 2
    mixed_precision = True
    max_grad_norm = 1.0
    epochs = 100  # 添加默认epochs

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
    """HLBD Playground训练器（全能仪表盘版）"""

    def __init__(
        self,
        config: PlaygroundConfig,
        model: APTModel,
        train_loader: DataLoader,
        tokenizer: DynamicTagTokenizer,
        dataset_stats: dict = None,
        save_dir: str = 'hlbd_playground',
        device: str = 'cuda'
    ):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.tokenizer = tokenizer
        self.dataset_stats = dataset_stats or {}
        self.save_dir = save_dir
        self.device = device

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.base_lr,
            weight_decay=0.01
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

        # 统计 - 按epoch聚合
        self.losses = []
        self.batch_losses = []  # 实时batch loss用于可视化

    def train_epoch(self, epoch: int):
        """训练一个epoch（全能仪表盘版）"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        epoch_start = time.time()

        # 创建进度条
        pbar = tqdm(
            self.train_loader,
            desc=f"📍 Epoch {epoch + 1}",
            unit="batch",
            ncols=120
        )

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            # --- ⏱️ 计时：前向传播 ---
            t0 = time.time()

            # 混合精度前向
            with autocast(enabled=self.config.mixed_precision):
                logits = self.model(input_ids)
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                # 记录真实Loss（未除以accumulation_steps前）
                real_loss_val = loss.item()

                # 为梯度累积做除法
                loss = loss / self.config.gradient_accumulation_steps

            t1 = time.time()
            fw_ms = (t1 - t0) * 1000

            # --- ⏱️ 计时：反向传播 ---
            t2 = time.time()

            # Backward
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # 梯度累积更新
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

            t3 = time.time()
            bw_ms = (t3 - t2) * 1000

            # --- 📊 计算指标（Acc & PPL）---
            with torch.no_grad():
                # PPL (Perplexity) = exp(Loss)
                try:
                    ppl_val = math.exp(min(real_loss_val, 20))  # 限制最大值防止溢出
                except OverflowError:
                    ppl_val = float('inf')

                # Accuracy (准确率)
                preds = logits.argmax(dim=-1)
                mask = labels != -100
                correct = (preds == labels) & mask
                accuracy = correct.sum().float() / mask.sum().float() if mask.sum() > 0 else torch.tensor(0.0)
                acc_val = accuracy.item() * 100

            total_loss += real_loss_val
            num_batches += 1

            # 保存batch loss用于可视化
            self.batch_losses.append({
                'epoch': epoch + 1,
                'batch': batch_idx,
                'loss': real_loss_val,
                'ppl': ppl_val,
                'acc': acc_val
            })

            # --- 🚀 实时更新进度条 ---
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({
                "Loss": f"{real_loss_val:.4f}",
                "PPL": f"{ppl_val:.1f}",
                "Acc": f"{acc_val:.1f}%",
                "LR": f"{current_lr:.6f}",
                "FW": f"{fw_ms:.0f}ms",
                "BW": f"{bw_ms:.0f}ms"
            })

            # --- 📡 实时更新可视化（每10个batch）---
            if batch_idx % 10 == 0:
                self._save_batch_progress()

        # Epoch结束更新学习率
        self.scheduler.step()

        avg_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start
        self.losses.append(avg_loss)

        return avg_loss, epoch_time

    def _save_batch_progress(self):
        """保存batch级别的进度（cluster存储，不每秒一个文件）"""
        report_path = Path(self.save_dir) / "experiment_report.json"

        # 确保目录存在
        report_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Cluster存储：按epoch聚合，每个epoch最多保留100个点
            epoch_clusters = {}
            for item in self.batch_losses:
                epoch_num = item['epoch']
                if epoch_num not in epoch_clusters:
                    epoch_clusters[epoch_num] = []
                epoch_clusters[epoch_num].append(item['loss'])

            # 压缩：每个epoch均匀采样最多100个点
            clustered_losses = []
            for epoch_num in sorted(epoch_clusters.keys()):
                losses = epoch_clusters[epoch_num]
                if len(losses) <= 100:
                    clustered_losses.extend(losses)
                else:
                    # 均匀采样
                    step = len(losses) / 100
                    sampled = [losses[int(i * step)] for i in range(100)]
                    clustered_losses.extend(sampled)

            report = {
                'control_losses': self.losses,  # Epoch平均loss
                'batch_losses': clustered_losses,  # Batch实时loss（cluster压缩）
                'current_epoch': len(self.losses),
                'total_batches': len(self.batch_losses),
                'dataset_stats': self.dataset_stats,
                'timestamp': time.time()
            }

            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception as e:
            # 静默失败，不中断训练
            pass

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
            'losses': self.losses,
            'dataset_stats': self.dataset_stats
        }

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, save_path)
        print(f"✅ Checkpoint已保存: {save_path}")

        # 如果是多数据集训练，显示统计信息
        if len(self.dataset_stats) > 1:
            print(f"   数据集来源:")
            for name, count in self.dataset_stats.items():
                print(f"     - {name}: {count} 样本")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='HLBD Playground训练 - 支持单数据集和多数据集模块化训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 单数据集训练（向后兼容）
  python train_hlbd_playground.py --dataset data/HLBD_Hardcore_Full_V2.json

  # 多数据集模块化训练
  python train_hlbd_playground.py --datasets data/HLBD_Full_V2.json data/HLBD_Hardcore_Full_V2.json

  # 自定义训练参数
  python train_hlbd_playground.py --datasets data/*.json --epochs 50 --save-dir models/hlbd_combined
        """
    )

    parser.add_argument('--dataset', type=str, default=None,
                       help='单个HLBD数据集路径（向后兼容）')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='多个HLBD数据集路径（模块化训练）')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--save-dir', type=str, default='hlbd_playground',
                       help='保存目录')
    parser.add_argument('--save-interval', type=int, default=25,
                       help='保存间隔')

    args = parser.parse_args()

    # 处理数据集参数
    if args.datasets:
        dataset_paths = args.datasets
    elif args.dataset:
        dataset_paths = args.dataset
    else:
        dataset_paths = str(PROJECT_ROOT / 'data' / 'HLBD_Hardcore_Full.json')
        print(f"⚠️  未指定数据集，使用默认: {dataset_paths}\n")

    # 设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 使用设备: {device}")

    # 配置
    config = PlaygroundConfig()
    config.epochs = args.epochs  # 更新epochs

    # Tokenizer
    print(f"\n🔤 初始化Tokenizer（支持动态标签）...")
    tokenizer = DynamicTagTokenizer(vocab_size=5000)

    # 数据集
    dataset = HLBDPlaygroundDataset(dataset_paths, tokenizer)

    # DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    # 模型配置（✅ 修复：统一使用num_heads）
    print(f"\n🏗️  构建APT模型...")
    model_config = APTModelConfiguration(
        vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,  # ✅ 左边必须是num_heads
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
    print(f"   d_model={config.d_model}, num_heads={config.num_heads} (每头维度={config.d_model//config.num_heads})")

    # 训练器
    trainer = HLBDPlaygroundTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        tokenizer=tokenizer,
        dataset_stats=dataset.dataset_stats,
        save_dir=args.save_dir,
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
        # 训练
        loss, epoch_time = trainer.train_epoch(epoch)

        print(f"\n📊 Epoch {epoch + 1}/{args.epochs} 完成:")
        print(f"   平均Loss: {loss:.4f}")
        print(f"   PPL: {math.exp(min(loss, 20)):.2f}")
        print(f"   用时: {epoch_time:.2f}s")

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
    print(f"\n📁 模型保存位置: {save_dir}")
    print(f"📈 可视化数据: {save_dir / 'experiment_report.json'}")


if __name__ == "__main__":
    main()
