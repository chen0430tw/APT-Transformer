#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT 训练模块

包含所有APT模型的训练相关功能：
- 训练器（Trainer）
- 微调器（Finetuner）
- 各种训练回调和监控
"""

# 主训练函数
from apt_model.training.trainer import train_model, Trainer

# 微调
from apt_model.training.finetuner import Finetuner

# 特定训练器
from apt_model.training.claude_trainer import ClaudeTrainer
from apt_model.training.gpt_trainer import GPTTrainer
from apt_model.training.vft_tva_trainer import VFTTVATrainer

# 训练组件
from apt_model.training.callbacks import (
    TrainingCallback,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
)
from apt_model.training.checkpoint import CheckpointManager
from apt_model.training.gradient_monitor import GradientMonitor
from apt_model.training.training_events import TrainingEventManager
from apt_model.training.training_guard import TrainingGuard
from apt_model.training.training_monitor import TrainingMonitor

# 优化器和混合精度
from apt_model.training.optimizer import APTOptimizer
from apt_model.training.mixed_precision import MixedPrecisionTrainer

# 特殊训练
from apt_model.training.train_reasoning import train_reasoning
from apt_model.training.apt_integration import APTIntegration
from apt_model.training.sosa_core import SOSACore

__all__ = [
    # 主训练
    'train_model',
    'Trainer',
    'Finetuner',

    # 特定训练器
    'ClaudeTrainer',
    'GPTTrainer',
    'VFTTVATrainer',

    # 训练组件
    'TrainingCallback',
    'EarlyStopping',
    'ModelCheckpoint',
    'LearningRateScheduler',
    'CheckpointManager',
    'GradientMonitor',
    'TrainingEventManager',
    'TrainingGuard',
    'TrainingMonitor',

    # 优化器和混合精度
    'APTOptimizer',
    'MixedPrecisionTrainer',

    # 特殊训练
    'train_reasoning',
    'APTIntegration',
    'SOSACore',
]
