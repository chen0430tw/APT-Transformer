#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HLBD (Hierarchical Language Bootstrapping Dataset) 训练和评估模块

这个模块提供了使用分层语言启蒙数据集训练APT模型的命令行接口。

用法:
    # 训练模式
    python -m apt_model.hlbd --hlbd-path 分层语言启蒙数据集.txt --output-dir apt_hlbd_model --epochs 20

    # 评估模式
    python -m apt_model.hlbd --hlbd-path 分层语言启蒙数据集.txt --output-dir apt_hlbd_model --evaluate-only
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 导入HLBD适配器
from apt_model.data.hlbd.hlbd_adapter import (
    HLBDDataProcessor,
    HLBDDataset,
    HLBDModelEvaluator,
    prepare_hlbd_tokenizer,
    create_hlbd_apt_config,
    prepare_hlbd_datasets
)

# 导入APT模型和训练器
from apt_model.modeling.apt_model import APTModel
from apt_model.training.trainer import train_model
from apt_model.training.checkpoint import CheckpointManager
from apt_model.utils import setup_logging, set_seed, get_device

# 设置日志
logger = logging.getLogger("apt_model.hlbd")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="HLBD (分层语言启蒙数据集) 训练和评估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 训练20个epoch
  python -m apt_model.hlbd --hlbd-path 分层语言启蒙数据集.txt --output-dir apt_hlbd_model --epochs 20

  # 仅评估已训练模型
  python -m apt_model.hlbd --hlbd-path 分层语言启蒙数据集.txt --output-dir apt_hlbd_model --evaluate-only

  # 使用GPU训练
  python -m apt_model.hlbd --hlbd-path 分层语言启蒙数据集.txt --output-dir apt_hlbd_model --epochs 20 --device cuda

  # 自定义批次大小和学习率
  python -m apt_model.hlbd --hlbd-path 分层语言启蒙数据集.txt --output-dir apt_hlbd_model --epochs 20 --batch-size 16 --lr 5e-5
        """
    )

    # 必需参数
    parser.add_argument(
        "--hlbd-path",
        type=str,
        required=True,
        help="HLBD数据集文件路径（例如：分层语言启蒙数据集.txt）"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="模型输出目录"
    )

    # 训练参数
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="训练轮数（默认：20）"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="批次大小（默认：8）"
    )

    parser.add_argument(
        "--lr", "--learning-rate",
        type=float,
        default=3e-5,
        dest="learning_rate",
        help="学习率（默认：3e-5）"
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="最大序列长度（默认：512）"
    )

    # 模型参数
    parser.add_argument(
        "--d-model",
        type=int,
        default=768,
        help="模型维度（默认：768）"
    )

    parser.add_argument(
        "--num-heads",
        type=int,
        default=12,
        help="注意力头数（默认：12）"
    )

    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="层数（默认：6）"
    )

    # 数据处理参数
    parser.add_argument(
        "--include-multilingual",
        action="store_true",
        default=True,
        help="包含多语言文本（默认：True）"
    )

    parser.add_argument(
        "--include-separate-levels",
        action="store_true",
        default=True,
        help="包含各层级单独的文本（默认：True）"
    )

    # 设备参数
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="计算设备（默认：auto - 自动检测）"
    )

    # 模式选择
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="仅评估模式，不进行训练"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从检查点恢复训练（提供检查点文件路径）"
    )

    # 其他参数
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认：42）"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细输出模式"
    )

    return parser.parse_args()


def setup_environment(args):
    """设置环境"""
    # 设置日志级别
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    # 设置随机种子
    set_seed(args.seed)

    # 确定设备
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)

    logger.info(f"使用设备: {device}")
    logger.info(f"随机种子: {args.seed}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"输出目录: {args.output_dir}")

    return device


def load_hlbd_data(args):
    """加载HLBD数据"""
    logger.info(f"加载HLBD数据集: {args.hlbd_path}")

    if not os.path.exists(args.hlbd_path):
        logger.error(f"数据集文件不存在: {args.hlbd_path}")
        sys.exit(1)

    # 创建数据处理器
    processor = HLBDDataProcessor(data_path=args.hlbd_path)

    # 处理数据
    processor.process_data(
        include_multilingual=args.include_multilingual,
        include_separate_levels=args.include_separate_levels
    )

    training_texts = processor.get_training_texts()
    logger.info(f"成功加载 {len(training_texts)} 个训练样本")

    # 显示样本示例
    if args.verbose and training_texts:
        logger.debug("=== 训练样本示例 ===")
        for i, text in enumerate(training_texts[:3]):
            logger.debug(f"样本 {i+1}:\n{text}\n")

    return processor


def prepare_model_and_tokenizer(args, processor):
    """准备模型和分词器"""
    logger.info("准备分词器...")

    # 准备HLBD分词器
    tokenizer = prepare_hlbd_tokenizer(
        hlbd_samples_or_path=processor.raw_samples,
        vocab_size=50000
    )

    logger.info(f"分词器词汇表大小: {tokenizer.vocab_size}")

    # 创建模型配置
    logger.info("创建模型配置...")
    config = create_hlbd_apt_config(vocab_size=tokenizer.vocab_size)

    # 可选：覆盖配置参数
    config.d_model = args.d_model
    config.num_heads = args.num_heads
    config.num_encoder_layers = args.num_layers
    config.num_decoder_layers = args.num_layers
    config.max_seq_len = args.max_length

    logger.info(f"模型配置: d_model={config.d_model}, num_heads={config.num_heads}, "
                f"num_layers={config.num_encoder_layers}, vocab_size={config.vocab_size}")

    # 创建模型
    logger.info("初始化APT模型...")
    model = APTModel(config)

    # 如果有检查点，加载它
    if args.resume:
        logger.info(f"从检查点恢复: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])

    return model, tokenizer, config


def train_hlbd_model(args, model, tokenizer, processor, device):
    """训练HLBD模型"""
    logger.info("="*60)
    logger.info("开始HLBD模型训练")
    logger.info("="*60)

    # 准备数据集
    logger.info("准备训练数据集...")
    train_loader, val_loader = prepare_hlbd_datasets(
        processor=processor,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size
    )

    logger.info(f"训练批次数: {len(train_loader)}")
    logger.info(f"验证批次数: {len(val_loader)}")

    # 移动模型到设备
    model = model.to(device)

    # 设置优化器
    from apt_model.training.optimizer import create_optimizer_and_scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model,
        learning_rate=args.learning_rate,
        warmup_steps=1000,
        total_steps=args.epochs * len(train_loader)
    )

    # 创建检查点管理器
    checkpoint_manager = CheckpointManager(
        save_dir=args.output_dir,
        max_checkpoints=3
    )

    # 训练循环
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        logger.info(f"{'='*60}")

        # 训练阶段
        model.train()
        train_loss = 0.0

        for batch_idx, (src_ids, tgt_ids) in enumerate(train_loader):
            src_ids = src_ids.to(device)
            tgt_ids = tgt_ids.to(device)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(src_ids, tgt_ids)
            loss = outputs['loss']

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                avg_loss = train_loss / (batch_idx + 1)
                lr = scheduler.get_last_lr()[0]
                logger.info(f"  Batch {batch_idx + 1}/{len(train_loader)} - "
                           f"Loss: {avg_loss:.4f}, LR: {lr:.6f}")

        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"平均训练损失: {avg_train_loss:.4f}")

        # 验证阶段
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for src_ids, tgt_ids in val_loader:
                src_ids = src_ids.to(device)
                tgt_ids = tgt_ids.to(device)

                outputs = model(src_ids, tgt_ids)
                loss = outputs['loss']
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        logger.info(f"平均验证损失: {avg_val_loss:.4f}")

        # 保存检查点
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            logger.info(f"新的最佳模型！验证损失: {best_val_loss:.4f}")

        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            metrics={'train_loss': avg_train_loss, 'val_loss': avg_val_loss},
            is_best=is_best
        )

    logger.info("\n" + "="*60)
    logger.info("训练完成！")
    logger.info(f"最佳验证损失: {best_val_loss:.4f}")
    logger.info(f"模型保存在: {args.output_dir}")
    logger.info("="*60)

    return model


def evaluate_hlbd_model(args, model, tokenizer, processor, device):
    """评估HLBD模型"""
    logger.info("="*60)
    logger.info("开始HLBD模型评估")
    logger.info("="*60)

    # 移动模型到设备
    model = model.to(device)
    model.eval()

    # 创建评估器
    evaluator = HLBDModelEvaluator(
        model=model,
        tokenizer=tokenizer,
        processor=processor
    )

    # 评估所有语言对
    logger.info("\n评估所有语言对的翻译能力...")
    results = evaluator.evaluate_all_language_pairs(num_samples=3)

    logger.info(f"\n总体平均相似度: {results['overall_avg_similarity']:.4f}")

    logger.info("\n各语言对详细结果:")
    for pair_name, pair_result in results['language_pairs'].items():
        if 'avg_similarity' in pair_result:
            logger.info(f"  {pair_name}: {pair_result['avg_similarity']:.4f}")

    # 评估概念完成能力
    logger.info("\n评估概念完成能力...")
    concept_results = evaluator.evaluate_concept_completion(num_samples=5)

    if 'avg_similarity' in concept_results:
        logger.info(f"概念完成平均相似度: {concept_results['avg_similarity']:.4f}")

    # 保存评估结果
    import json
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'language_pairs': results,
            'concept_completion': concept_results
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"\n评估结果已保存到: {results_file}")
    logger.info("="*60)

    return results


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 设置环境
    device = setup_environment(args)

    # 加载HLBD数据
    processor = load_hlbd_data(args)

    # 准备模型和分词器
    model, tokenizer, config = prepare_model_and_tokenizer(args, processor)

    if args.evaluate_only:
        # 仅评估模式
        if not args.resume:
            logger.warning("评估模式需要提供 --resume 参数指定模型检查点")
            logger.info("尝试从输出目录加载最新检查点...")

            # 尝试找到最新的检查点
            checkpoint_files = list(Path(args.output_dir).glob("checkpoint_*.pt"))
            if checkpoint_files:
                latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
                logger.info(f"找到检查点: {latest_checkpoint}")
                checkpoint = torch.load(latest_checkpoint)
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                logger.error(f"在 {args.output_dir} 中未找到检查点文件")
                sys.exit(1)

        # 评估模型
        evaluate_hlbd_model(args, model, tokenizer, processor, device)
    else:
        # 训练模式
        model = train_hlbd_model(args, model, tokenizer, processor, device)

        # 训练后自动评估
        logger.info("\n训练完成，开始评估...")
        evaluate_hlbd_model(args, model, tokenizer, processor, device)


if __name__ == "__main__":
    main()
