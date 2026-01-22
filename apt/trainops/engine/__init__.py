#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training Engine

训练引擎实现：
- Trainer: 主训练器
- Finetuner: 微调器
- Custom trainers (Claude, GPT5, VFT-TVA等)
- Training callbacks and hooks
- Mixed precision training
- Optimizer wrappers
"""

# 主训练器
from apt.trainops.engine.trainer import Trainer, train_model
from apt.trainops.engine.finetuner import Finetuner

# 特定训练器
from apt.trainops.engine.claude_trainer import ClaudeTrainer
from apt.trainops.engine.gpt_trainer import GPTTrainer
from apt.trainops.engine.vft_tva_trainer import VFTTVATrainer
from apt.trainops.engine.train_reasoning import ReasoningTrainer

# 训练支持
from apt.trainops.engine.callbacks import (
    TrainingCallback,
    EarlyStoppingCallback,
    LearningRateSchedulerCallback,
)
from apt.trainops.engine.hooks import TrainingHook
from apt.trainops.engine.training_events import TrainingEventManager
from apt.trainops.engine.mixed_precision import MixedPrecisionManager
from apt.trainops.engine.optimizer import get_optimizer
from apt.trainops.engine.apt_integration import APTIntegration
from apt.trainops.engine.sosa_core import SOSACore

__all__ = [
    # Main trainers
    'Trainer',
    'train_model',
    'Finetuner',
    # Specific trainers
    'ClaudeTrainer',
    'GPTTrainer',
    'VFTTVATrainer',
    'ReasoningTrainer',
    # Support
    'TrainingCallback',
    'EarlyStoppingCallback',
    'LearningRateSchedulerCallback',
    'TrainingHook',
    'TrainingEventManager',
    'MixedPrecisionManager',
    'get_optimizer',
    'APTIntegration',
    'SOSACore',
]
