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
try:
    from apt.trainops.engine.trainer import Trainer, train_model
except ImportError:
    Trainer = None
    train_model = None
try:
    from apt.trainops.engine.finetuner import Finetuner
except ImportError:
    Finetuner = None

# 特定训练器
try:
    from apt.trainops.engine.claude_trainer import ClaudeTrainer
except ImportError:
    ClaudeTrainer = None
try:
    from apt.trainops.engine.gpt_trainer import GPTTrainer
except ImportError:
    GPTTrainer = None
try:
    from apt.trainops.engine.vft_tva_trainer import VFTTVATrainer
except ImportError:
    VFTTVATrainer = None
try:
    from apt.trainops.engine.train_reasoning import ReasoningTrainer
except ImportError:
    ReasoningTrainer = None

# 训练支持
try:
    from apt.trainops.engine.callbacks import (
        TrainingCallback,
        EarlyStoppingCallback,
        LearningRateSchedulerCallback,
    )
except ImportError:
    TrainingCallback = None
    EarlyStoppingCallback = None
    LearningRateSchedulerCallback = None
try:
    from apt.trainops.engine.hooks import TrainingHook
except ImportError:
    TrainingHook = None
try:
    from apt.trainops.engine.training_events import TrainingEventManager
except ImportError:
    TrainingEventManager = None
try:
    from apt.trainops.engine.mixed_precision import MixedPrecisionManager
except ImportError:
    MixedPrecisionManager = None
try:
    from apt.trainops.engine.optimizer import get_optimizer
except ImportError:
    get_optimizer = None
try:
    from apt.trainops.engine.apt_integration import APTIntegration
except ImportError:
    APTIntegration = None
try:
    from apt.trainops.engine.sosa_core import SOSACore
except ImportError:
    SOSACore = None

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
