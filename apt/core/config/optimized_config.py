#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""临时APT配置 - Trial 0"""

class APTConfig:
    """APT模型配置类 - Optuna试验"""
    
    def __init__(self, vocab_size=50257, d_model=768, d_ff=2048, num_heads=12, 
                 num_encoder_layers=6, num_decoder_layers=6, dropout=0.12340279606636548, 
                 max_seq_len=512, epsilon=0.0754520520701671, alpha=0.0018679147494991151, beta=0.004582013868952864, base_lr=4.5862562590959464e-05,
                 pad_token_id=0, bos_token_id=1, eos_token_id=2, 
                 activation="gelu", use_autopoietic=True, sr_ratio=4, init_tau=1.692940916619948, 
                 batch_first=True, warmup_steps=1500, weight_decay=0.022022300234864175,
                 attention_dropout=0.12339917805043041, layer_norm_eps=1e-5, gradient_clip=0.9956508044572319):
        """初始化模型配置"""
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.base_lr = base_lr
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.activation = activation
        self.use_autopoietic = use_autopoietic
        self.sr_ratio = sr_ratio
        self.init_tau = init_tau
        self.batch_first = batch_first
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.gradient_clip = gradient_clip
        
    def to_dict(self):
        """将配置转换为字典"""
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'num_heads': self.num_heads,
            'num_encoder_layers': self.num_encoder_layers,
            'num_decoder_layers': self.num_decoder_layers,
            'dropout': self.dropout,
            'max_seq_len': self.max_seq_len,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
            'beta': self.beta,
            'base_lr': self.base_lr,
            'pad_token_id': self.pad_token_id,
            'bos_token_id': self.bos_token_id,
            'eos_token_id': self.eos_token_id,
            'activation': self.activation,
            'use_autopoietic': self.use_autopoietic,
            'sr_ratio': self.sr_ratio,
            'init_tau': self.init_tau,
            'batch_first': self.batch_first,
            'warmup_steps': self.warmup_steps,
            'weight_decay': self.weight_decay,
            'attention_dropout': self.attention_dropout,
            'layer_norm_eps': self.layer_norm_eps,
            'gradient_clip': self.gradient_clip
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """从字典创建配置"""
        return cls(**config_dict)