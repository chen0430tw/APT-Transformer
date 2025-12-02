#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
强化学习模块

提供各种RL算法实现

作者: chen0430tw
"""

from .reward_model import (
    RewardModel,
    RewardModelTrainer,
    create_reward_model
)

from .rlhf_trainer import (
    RLHFTrainer,
    RLHFConfig,
    create_rlhf_trainer
)

from .dpo_trainer import (
    DPOTrainer,
    DPOConfig,
    create_dpo_trainer
)

from .grpo_trainer import (
    GRPOTrainer,
    GRPOConfig,
    create_grpo_trainer
)

__all__ = [
    # Reward Model
    'RewardModel',
    'RewardModelTrainer',
    'create_reward_model',

    # RLHF
    'RLHFTrainer',
    'RLHFConfig',
    'create_rlhf_trainer',

    # DPO
    'DPOTrainer',
    'DPOConfig',
    'create_dpo_trainer',

    # GRPO
    'GRPOTrainer',
    'GRPOConfig',
    'create_grpo_trainer',
]
