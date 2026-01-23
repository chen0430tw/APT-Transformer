#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT模型微调（Fine-tuning）模块

复用现有的模块化组件：
- checkpoint.load_model() - 加载预训练模型
- trainer 中的训练循环
- data loading 功能
- evaluator 功能

支持功能：
1. 基础微调
2. 冻结层（Frozen Layers）
3. 学习率衰减
4. 早停机制（Early Stopping）
5. 参数高效微调（LoRA - 可选）
"""

import os
from apt.apt_model.utils.fake_torch import get_torch
torch = get_torch()
from apt.apt_model.utils.fake_torch import get_torch
torch = get_torch()
nn = torch.nn
from tqdm import tqdm
from datetime import datetime
from typing import Optional, List, Dict, Tuple

# 复用现有模块
from apt.trainops.checkpoints.checkpoint import load_model, save_model
from apt.core.data.external_data import load_external_data
from apt.core.generation.generator import generate_natural_text
from apt.core.generation.evaluator import evaluate_text_quality
from apt.apt_model.utils import get_device, set_seed
from apt.core.config.settings_manager import settings


class FineTuner:
    """APT模型微调器 - 复用模块化组件"""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        logger: Optional[object] = None
    ):
        """
        初始化微调器

        参数:
            model_path: 预训练模型路径
            device: 计算设备
            logger: 日志记录器
        """
        self.device = device or get_device()
        self.logger = logger

        # 加载预训练模型（复用checkpoint模块）
        self.log_info(f"📂 加载预训练模型: {model_path}")
        self.model, self.tokenizer, self.config = load_model(model_path, self.device)
        self.log_info(f"✓ 模型加载成功")

        # 记录原始参数
        self.original_params = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def log_info(self, message):
        """日志输出"""
        if self.logger:
            self.logger.info(message)
        print(message)

    def freeze_layers(
        self,
        freeze_embeddings: bool = True,
        freeze_encoder_layers: Optional[int] = None,
        freeze_decoder_layers: Optional[int] = None
    ):
        """
        冻结指定层的参数

        参数:
            freeze_embeddings: 是否冻结embedding层
            freeze_encoder_layers: 冻结前N层encoder（None=不冻结）
            freeze_decoder_layers: 冻结前N层decoder（None=不冻结）
        """
        self.log_info("\n🔒 冻结层设置:")

        # 冻结embeddings
        if freeze_embeddings and hasattr(self.model, 'embeddings'):
            for param in self.model.embeddings.parameters():
                param.requires_grad = False
            self.log_info("  ✓ Embeddings已冻结")

        # 冻结encoder层
        if freeze_encoder_layers and hasattr(self.model, 'encoder'):
            if hasattr(self.model.encoder, 'layers'):
                for i, layer in enumerate(self.model.encoder.layers[:freeze_encoder_layers]):
                    for param in layer.parameters():
                        param.requires_grad = False
                self.log_info(f"  ✓ 前{freeze_encoder_layers}层Encoder已冻结")

        # 冻结decoder层
        if freeze_decoder_layers and hasattr(self.model, 'decoder'):
            if hasattr(self.model.decoder, 'layers'):
                for i, layer in enumerate(self.model.decoder.layers[:freeze_decoder_layers]):
                    for param in layer.parameters():
                        param.requires_grad = False
                self.log_info(f"  ✓ 前{freeze_decoder_layers}层Decoder已冻结")

        # 更新可训练参数统计
        self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_ratio = 100 * (1 - self.trainable_params / self.original_params)

        self.log_info(f"\n📊 参数统计:")
        self.log_info(f"  总参数: {self.original_params:,}")
        self.log_info(f"  可训练: {self.trainable_params:,}")
        self.log_info(f"  已冻结: {frozen_ratio:.1f}%")

    def fine_tune(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        epochs: int = 5,
        batch_size: int = 8,
        learning_rate: float = 1e-5,  # 微调使用较小的学习率
        warmup_steps: int = 100,
        max_samples: Optional[int] = None,
        save_path: str = "apt_model_finetuned",
        early_stopping_patience: int = 3,
        eval_steps: int = 100,
        save_steps: int = 500,
    ) -> Tuple[nn.Module, object, object]:
        """
        执行微调训练（复用trainer的训练逻辑）

        参数:
            train_data_path: 训练数据路径
            val_data_path: 验证数据路径（可选）
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率（微调建议1e-5到5e-5）
            warmup_steps: 预热步数
            max_samples: 最大样本数
            save_path: 保存路径
            early_stopping_patience: 早停耐心值
            eval_steps: 评估间隔
            save_steps: 保存间隔

        返回:
            (model, tokenizer, config)
        """
        self.log_info("\n" + "="*60)
        self.log_info("🎯 开始微调训练")
        self.log_info("="*60)

        # 加载微调数据（复用data模块）
        self.log_info(f"\n📊 加载微调数据: {train_data_path}")
        train_texts = load_external_data(train_data_path, max_samples=max_samples)
        self.log_info(f"  训练样本数: {len(train_texts)}")

        val_texts = None
        if val_data_path:
            val_texts = load_external_data(val_data_path, max_samples=max_samples)
            self.log_info(f"  验证样本数: {len(val_texts)}")

        # 准备数据（复用trainer的DataLoader逻辑）
        Dataset = torch.utils.data.Dataset
        DataLoader = torch.utils.data.DataLoader

        class TextDataset(Dataset):
            def __init__(self, texts, tokenizer, max_length=512):
                self.texts = texts
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = self.texts[idx]
                encoding = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                return {
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze()
                }

        train_dataset = TextDataset(train_texts, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_texts:
            val_dataset = TextDataset(val_texts, self.tokenizer)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # 设置优化器（微调使用较小学习率）
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=self.config.weight_decay if hasattr(self.config, 'weight_decay') else 0.01
        )

        # 学习率调度器
        total_steps = len(train_loader) * epochs
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        self.log_info(f"\n⚙️  训练配置:")
        self.log_info(f"  Epochs: {epochs}")
        self.log_info(f"  Batch Size: {batch_size}")
        self.log_info(f"  Learning Rate: {learning_rate}")
        self.log_info(f"  Warmup Steps: {warmup_steps}")
        self.log_info(f"  Total Steps: {total_steps}")

        # 训练循环（复用trainer的训练逻辑）
        self.model.train()
        best_val_loss = float('inf')
        patience_counter = 0
        global_step = 0

        for epoch in range(epochs):
            self.log_info(f"\n{'='*60}")
            self.log_info(f"Epoch {epoch + 1}/{epochs}")
            self.log_info(f"{'='*60}")

            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

            for batch_idx, batch in enumerate(progress_bar):
                # 准备输入
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                # 前向传播
                outputs = self.model(
                    src_ids=input_ids,
                    tgt_ids=input_ids,  # 自回归任务
                    src_mask=attention_mask
                )

                # 计算损失
                logits = outputs.view(-1, outputs.size(-1))
                targets = input_ids.view(-1)
                loss = nn.functional.cross_entropy(
                    logits,
                    targets,
                    ignore_index=self.tokenizer.pad_token_id
                )

                # 反向传播
                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪
                if hasattr(self.config, 'gradient_clip'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )

                optimizer.step()

                if global_step < warmup_steps:
                    scheduler.step()

                epoch_loss += loss.item()
                global_step += 1

                # 更新进度条
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

                # 定期评估
                if val_loader and global_step % eval_steps == 0:
                    val_loss = self._evaluate(val_loader)
                    self.log_info(f"\n  Step {global_step} | Val Loss: {val_loss:.4f}")

                    # 早停检查
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # 保存最佳模型
                        self._save_checkpoint(save_path, "best")
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            self.log_info(f"\n⚠️  Early stopping triggered at step {global_step}")
                            break

                    self.model.train()

                # 定期保存
                if global_step % save_steps == 0:
                    self._save_checkpoint(save_path, f"step_{global_step}")

            # Epoch结束统计
            avg_loss = epoch_loss / len(train_loader)
            self.log_info(f"\n📊 Epoch {epoch+1} 平均损失: {avg_loss:.4f}")

            # 生成样本测试（复用generator模块）
            self._generate_sample()

            if patience_counter >= early_stopping_patience:
                break

        # 保存最终模型
        self.log_info(f"\n💾 保存最终模型到: {save_path}")
        save_model(self.model, self.tokenizer, save_path, self.config)

        self.log_info("\n" + "="*60)
        self.log_info("✅ 微调完成！")
        self.log_info("="*60)

        return self.model, self.tokenizer, self.config

    def _evaluate(self, val_loader) -> float:
        """评估模型（复用evaluator逻辑）"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(
                    src_ids=input_ids,
                    tgt_ids=input_ids,
                    src_mask=attention_mask
                )

                logits = outputs.view(-1, outputs.size(-1))
                targets = input_ids.view(-1)
                loss = nn.functional.cross_entropy(
                    logits,
                    targets,
                    ignore_index=self.tokenizer.pad_token_id
                )

                total_loss += loss.item()

        return total_loss / len(val_loader)

    def _generate_sample(self):
        """生成样本文本（复用generator模块）"""
        self.model.eval()

        try:
            sample_text = generate_natural_text(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt="人工智能",
                max_length=50,
                temperature=0.8,
                device=self.device
            )

            # 评估质量（复用evaluator模块）
            quality_score = evaluate_text_quality(sample_text)

            self.log_info(f"\n  📝 生成样本: {sample_text}")
            self.log_info(f"  📊 质量评分: {quality_score:.2f}/100")
        except Exception as e:
            self.log_info(f"\n  ⚠️  生成样本失败: {e}")

        self.model.train()

    def _save_checkpoint(self, save_path: str, suffix: str):
        """保存检查点"""
        checkpoint_path = f"{save_path}_{suffix}"
        save_model(self.model, self.tokenizer, checkpoint_path, self.config)
        self.log_info(f"  💾 保存检查点: {checkpoint_path}")


def fine_tune_model(
    pretrained_model_path: str,
    train_data_path: str,
    val_data_path: Optional[str] = None,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    freeze_embeddings: bool = False,
    freeze_encoder_layers: Optional[int] = None,
    freeze_decoder_layers: Optional[int] = None,
    save_path: str = "apt_model_finetuned",
    logger: Optional[object] = None,
    **kwargs
) -> Tuple[nn.Module, object, object]:
    """
    微调APT模型的便捷函数

    参数:
        pretrained_model_path: 预训练模型路径
        train_data_path: 训练数据路径
        val_data_path: 验证数据路径
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        freeze_embeddings: 是否冻结embedding层
        freeze_encoder_layers: 冻结前N层encoder
        freeze_decoder_layers: 冻结前N层decoder
        save_path: 保存路径
        logger: 日志记录器
        **kwargs: 其他参数传递给fine_tune方法

    返回:
        (model, tokenizer, config)
    """
    # 创建微调器
    finetuner = FineTuner(pretrained_model_path, logger=logger)

    # 冻结层（如果需要）
    if freeze_embeddings or freeze_encoder_layers or freeze_decoder_layers:
        finetuner.freeze_layers(
            freeze_embeddings=freeze_embeddings,
            freeze_encoder_layers=freeze_encoder_layers,
            freeze_decoder_layers=freeze_decoder_layers
        )

    # 执行微调
    return finetuner.fine_tune(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_path=save_path,
        **kwargs
    )
