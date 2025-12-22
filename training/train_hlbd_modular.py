#!/usr/bin/env python3
"""
HLBD模块化训练脚本
支持同时加载多个HLBD数据集进行联合训练

特性:
- 🔗 多数据集合并训练
- 📊 自动格式兼容处理
- 🎲 数据集混合打散
- 📈 统一训练流程
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

# 检查依赖
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from torch.cuda.amp import autocast, GradScaler
except ImportError:
    print("❌ PyTorch未安装，请先安装:")
    print("   pip install torch")
    sys.exit(1)


class ModularHLBDDataset:
    """模块化HLBD数据集加载器"""

    def __init__(self, dataset_paths: List[str]):
        self.dataset_paths = dataset_paths
        self.all_samples = []
        self.dataset_stats = {}

    def load_datasets(self):
        """加载并合并多个数据集"""
        print("=" * 60)
        print("模块化HLBD数据集加载器")
        print("=" * 60)
        print()

        for i, path in enumerate(self.dataset_paths, 1):
            print(f"📂 加载数据集 {i}/{len(self.dataset_paths)}: {path}")
            samples = self._load_single_dataset(path)

            if samples:
                self.all_samples.extend(samples)
                dataset_name = Path(path).stem
                self.dataset_stats[dataset_name] = len(samples)
                print(f"   ✓ 成功加载 {len(samples)} 条样本")
            else:
                print(f"   ⚠️  数据集为空或加载失败")
            print()

        # 打散混合
        import random
        random.shuffle(self.all_samples)

        print(f"📊 数据集统计:")
        for name, count in self.dataset_stats.items():
            percentage = count / len(self.all_samples) * 100
            print(f"   {name}: {count} 条 ({percentage:.1f}%)")
        print(f"   总计: {len(self.all_samples)} 条样本")
        print(f"   ✓ 已混合打散\n")

        return self.all_samples

    def _load_single_dataset(self, path: str) -> List[Dict]:
        """加载单个数据集并转换为统一格式"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 检测数据集类型
            if 'samples' in data:
                # HLBD Full格式（8层结构）
                return self._process_hlbd_full(data['samples'])
            elif 'data' in data:
                # HLBD Hardcore格式（模块化）
                return self._process_hlbd_hardcore(data['data'])
            else:
                print(f"   ⚠️  未知的数据集格式")
                return []

        except Exception as e:
            print(f"   ❌ 加载失败: {e}")
            return []

    def _process_hlbd_full(self, samples: List[Dict]) -> List[Dict]:
        """处理HLBD Full格式（8层结构）"""
        processed = []

        for sample in samples:
            # 提取所有层级的文本
            layers = []
            concept = sample.get('concept', '')

            # Level 1: 字卡
            if 'level_1' in sample:
                char_card = sample['level_1'].get('字卡', '')
                emoji = sample['level_1'].get('emoji', '')
                layers.append(f"字卡: {char_card} {emoji}")

            # Level 2: 短语
            if 'level_2' in sample:
                phrase = sample['level_2'].get('短语', '')
                layers.append(f"短语: {phrase}")

            # Level 3: 数学（句法结构）← 重要！
            if 'level_3' in sample:
                math_expr = sample['level_3'].get('数学', '')
                layers.append(f"句法: {math_expr}")

            # Level 4: 拼音
            if 'level_4' in sample:
                pinyin = sample['level_4'].get('拼音', '')
                layers.append(f"拼音: {pinyin}")

            # Level 5: 英文
            if 'level_5' in sample:
                english = sample['level_5'].get('英文', '')
                layers.append(f"英文: {english}")

            # Level 6: 中文
            if 'level_6' in sample:
                chinese = sample['level_6'].get('中文', '')
                layers.append(f"中文: {chinese}")

            # Level 7-8: 日文、韩文（可选）
            if 'level_7' in sample:
                japanese = sample['level_7'].get('日文', '')
                if japanese:
                    layers.append(f"日文: {japanese}")

            if 'level_8' in sample:
                korean = sample['level_8'].get('韩文', '')
                if korean:
                    layers.append(f"韩文: {korean}")

            # 组合成训练文本
            text = f"概念: {concept}\n" + "\n".join(layers)

            processed.append({
                'text': text,
                'source': 'HLBD_Full',
                'concept': concept
            })

        return processed

    def _process_hlbd_hardcore(self, data: Dict) -> List[Dict]:
        """处理HLBD Hardcore格式（模块化）"""
        processed = []

        for module, samples in data.items():
            for sample in samples:
                input_text = sample.get('input', '')
                output_text = sample.get('output', '')

                # 转换为问答格式
                text = f"问题: {input_text}\n答案: {output_text}"

                processed.append({
                    'text': text,
                    'source': f'HLBD_Hardcore_{module}',
                    'module': module
                })

        return processed


class HLBDTrainingDataset(Dataset):
    """PyTorch数据集包装器"""

    def __init__(self, samples: List[Dict], tokenizer, max_length: int = 512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['text']

        # 简单字符级tokenization（实际训练时需要proper tokenizer）
        # 这里仅作示例
        tokens = [ord(c) % 5000 for c in text[:self.max_length]]

        # Padding
        if len(tokens) < self.max_length:
            tokens += [0] * (self.max_length - len(tokens))

        input_ids = torch.tensor(tokens[:self.max_length], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': input_ids.clone()  # 自回归训练
        }


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="HLBD模块化训练 - 支持多数据集合并训练"
    )

    # 支持多个数据集
    parser.add_argument(
        '--datasets',
        nargs='+',
        required=True,
        help='数据集路径列表（支持多个），例如: data/HLBD_Full_V2.json data/HLBD_Hardcore_Full_V2.json'
    )

    # 训练参数
    parser.add_argument('--output-dir', type=str, default='models/hlbd_modular',
                       help='模型输出目录')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                       help='学习率')
    parser.add_argument('--max-length', type=int, default=512,
                       help='最大序列长度')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='训练设备')

    return parser.parse_args()


def main():
    """主训练流程"""
    args = parse_args()

    print("=" * 60)
    print("🔗 HLBD模块化训练系统")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  数据集数量: {len(args.datasets)}")
    for i, ds in enumerate(args.datasets, 1):
        print(f"    {i}. {ds}")
    print(f"  输出目录: {args.output_dir}")
    print(f"  训练轮数: {args.epochs}")
    print(f"  批次大小: {args.batch_size}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  设备: {args.device}")
    print()

    # 加载数据集
    loader = ModularHLBDDataset(args.datasets)
    all_samples = loader.load_datasets()

    if len(all_samples) == 0:
        print("❌ 没有有效的训练样本")
        return 1

    # 创建PyTorch数据集（这里需要实际的tokenizer）
    print("📦 准备训练数据...")
    # 注意：这里使用简化的tokenizer，实际训练需要proper tokenizer
    train_dataset = HLBDTrainingDataset(
        all_samples,
        tokenizer=None,  # 实际需要传入tokenizer
        max_length=args.max_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    print(f"   ✓ 训练样本: {len(train_dataset)}")
    print(f"   ✓ 批次数: {len(train_loader)}")
    print()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存数据集统计
    stats = {
        'total_samples': len(all_samples),
        'datasets': loader.dataset_stats,
        'training_config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'max_length': args.max_length
        }
    }

    with open(output_dir / 'dataset_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("💡 提示:")
    print("   此脚本展示了模块化训练的框架")
    print("   实际训练需要:")
    print("   1. 加载完整的APT模型")
    print("   2. 配置proper tokenizer")
    print("   3. 实现训练循环")
    print("   4. 添加评估和保存逻辑")
    print()
    print(f"📊 数据集统计已保存到: {output_dir / 'dataset_stats.json'}")
    print()
    print("=" * 60)
    print("✅ 模块化数据加载完成！")
    print("=" * 60)
    print()
    print("🎯 下一步: 集成到完整的training/train_hlbd_playground.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
