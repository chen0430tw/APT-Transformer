#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Optimizer and learning rate scheduler for APT model"""

from apt.core.fake_torch import get_torch
torch = get_torch()
from transformers import get_cosine_schedule_with_warmup

def create_optimizer_and_scheduler(model, learning_rate, steps_per_epoch, epochs):
    """
    创建优化器和学习率调度器
    
    参数:
        model: 训练模型
        learning_rate: 基础学习率
        steps_per_epoch: 每轮的步数
        epochs: 训练轮数
        
    返回:
        tuple: (optimizer, scheduler)
    """
    # 将参数分为有权重衰减和无权重衰减两组
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    # 创建AdamW优化器
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
    
    # 计算总步数
    total_steps = steps_per_epoch * epochs
    
    # 创建带有预热和余弦衰减的学习率调度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),  # 10%的步数用于预热
        num_training_steps=total_steps
    )
    
    return optimizer, scheduler

def get_learning_rate(optimizer):
    """获取当前学习率"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_optimizer(
    model,
    optimizer_type: str = 'adamw',
    learning_rate: float = 3e-5,
    weight_decay: float = 0.01,
    **kwargs
):
    """
    Get optimizer for model training

    Args:
        model: The model to optimize
        optimizer_type: Type of optimizer ('adamw', 'adam', 'sgd')
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        **kwargs: Additional optimizer arguments

    Returns:
        optimizer: Configured optimizer
    """
    if optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
            **{k: v for k, v in kwargs.items() if k != 'momentum'}
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")