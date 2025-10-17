#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT模型最佳配置 - 由Optuna优化生成
生成时间: 2025-03-10 16:27:05
优化分数: 74.67/100
"""

class APTConfig:
    """APT模型配置类 - 由Optuna优化"""
    
    def __init__(self, vocab_size=50257, d_model=768, d_ff=2048, num_heads=12, 
                 num_encoder_layers=6, num_decoder_layers=6, dropout=0.15023625280036187, 
                 max_seq_len=512, epsilon=0.08404568130694745, alpha=0.0010648461814669063, beta=0.0022270401510050193, base_lr=2.418394206230888e-05,
                 pad_token_id=0, bos_token_id=1, eos_token_id=2, 
                 activation="gelu", use_autopoietic=True, sr_ratio=6, init_tau=1.3482871331894277, 
                 batch_first=True, warmup_steps=1500, weight_decay=0.01284451402180246,
                 attention_dropout=0.19455075657434573, layer_norm_eps=1e-5, gradient_clip=0.9455654465356627):
        """初始化模型配置"""
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.d_ff = d_ff
            self.num_heads = num_heads
            self.num_encoder_layers = num_encoder_layers
            self.num_decoder_layers = num_decoder_layers
            self.dropout = dropout
            self.max_seq_len = max_seq_len
            
            # 动态Taylor展开系数
            self.epsilon = epsilon        
            self.alpha = alpha            
            self.beta = beta              
            self.base_lr = base_lr        
            
            # 特殊标记ID
            self.pad_token_id = pad_token_id
            self.bos_token_id = bos_token_id
            self.eos_token_id = eos_token_id
            
            # 注意力机制参数
            self.activation = activation  
            self.use_autopoietic = use_autopoietic  
            self.sr_ratio = sr_ratio      
            self.init_tau = init_tau      
            self.batch_first = batch_first 
            
            # 训练稳定性参数
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