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
    from apt.trainops.engine import Trainer
    from apt.trainops.distributed import setup_ddp
    from apt.trainops.data import APTDataLoader
"""

__version__ = '2.0.0-alpha'

# 此模块将在PR-3中从apt.apt_model.training迁移内容
__all__ = []
