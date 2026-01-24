#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT TrainOps Domain

训练运营域：训练编排和生命周期管理

子模块：
- engine: 训练引擎（Trainer, Finetuner等）
- distributed: 分布式训练（DDP, FSDP, DeepSpeed等）
- data: 数据加载和预处理
- checkpoints: 检查点管理
- eval: 评估和验证
- artifacts: 训练产物管理（模型、日志、指标等）

使用示例：
    try:
        from apt.trainops.engine import Trainer
    except ImportError:
        Trainer = None
    try:
        from apt.trainops.distributed import setup_ddp
    except ImportError:
        setup_ddp = None
    try:
        from apt.trainops.data import APTDataLoader
    except ImportError:
        APTDataLoader = None
"""

__version__ = '2.0.0-alpha'

# 主要模块导出
# Note: Using try-except to handle potential import errors
try:
    try:
        from apt.trainops.engine import Trainer, Finetuner, train_model
    except ImportError:
        Trainer = None
        Finetuner = None
        train_model = None
except ImportError:
    Trainer = None
    Finetuner = None
    train_model = None

try:
    try:
        from apt.trainops.data import create_dataloader, APTDataLoader
    except ImportError:
        create_dataloader = None
        APTDataLoader = None
except ImportError:
    create_dataloader = None
    APTDataLoader = None

try:
    try:
        from apt.trainops.checkpoints import CheckpointManager, save_checkpoint, load_checkpoint
    except ImportError:
        CheckpointManager = None
        save_checkpoint = None
        load_checkpoint = None
except ImportError:
    CheckpointManager = None
    save_checkpoint = None
    load_checkpoint = None

try:
    try:
        from apt.trainops.eval import TrainingMonitor, TrainingGuard
    except ImportError:
        TrainingMonitor = None
        TrainingGuard = None
except ImportError:
    TrainingMonitor = None
    TrainingGuard = None

__all__ = [
    # Engine
    'Trainer',
    'Finetuner',
    'train_model',
    # Data
    'create_dataloader',
    'APTDataLoader',
    # Checkpoints
    'CheckpointManager',
    'save_checkpoint',
    'load_checkpoint',
    # Eval
    'TrainingMonitor',
    'TrainingGuard',
]
