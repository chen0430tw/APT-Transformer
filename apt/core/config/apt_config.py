#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT 模型配置模块
定义了APT (自生成变换器) 模型的各种配置类
"""

import os
import json
from typing import Dict, Any, Optional

class APTConfig:
    """APT模型配置类"""
    
    def __init__(
        self, 
        vocab_size=50257,  # 词汇表大小
        d_model=768,  # 模型维度
        d_ff=2048,  # 前馈网络维度
        num_heads=12,  # 注意力头数
        num_encoder_layers=6,  # 编码器层数
        num_decoder_layers=6,  # 解码器层数
        dropout=0.15,  # Dropout比率
        max_seq_len=512,  # 最大序列长度
        # 自生成变换器的特定参数
        epsilon=0.08,  # 无穷倒数缩放因子
        alpha=0.0008,  # 泰勒展开系数
        beta=0.003,  # 动态调节系数
        base_lr=4e-5,  # 基准学习率
        # 特殊标记ID
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        # 注意力机制参数
        activation="gelu",  # 激活函数
        use_autopoietic=True,  # 是否使用自生成机制
        sr_ratio=6,  # 自生成矩阵压缩比
        init_tau=1.3,  # 初始温度参数
        batch_first=True,  # 是否使用batch_first格式
        # 训练稳定性参数
        warmup_steps=1500,  # 预热步数
        weight_decay=0.015,  # 权重衰减
        attention_dropout=0.15,  # 注意力Dropout比率
        layer_norm_eps=1e-5,  # 层归一化epsilon
        gradient_clip=0.8,  # 梯度裁剪
        # DBC-DAC相关参数
        use_dbc_dac=False,  # 是否使用DBC-DAC稳定
        rank_ratio_proj=0.1,  # DBC投影比例
        rank_ratio_res=0.05,  # DAC残差比例
        dbc_threshold=1e-6,  # DBC阈值
        dbc_iterations=1,  # DAC迭代次数
        # 其他参数
        tokenizer_type=None,  # 分词器类型
        language=None,  # 语言类型
        **kwargs
    ):
        """初始化模型配置"""
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        
        # 动态Taylor展开系数 - 调整为更稳定且灵活的值
        self.epsilon = epsilon        # 从0.1调整到0.08，轻微减少缩放强度
        self.alpha = alpha            # 从0.001调整到0.0008，适度减小泰勒展开系数
        self.beta = beta              # 从0.005调整到0.003，适度调整动态系数
        self.base_lr = base_lr        # 从5e-5调整到4e-5，略微保守的学习率
        
        # 特殊标记ID
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # 注意力机制参数 - 平衡自生成机制
        self.activation = activation  # 激活函数
        self.use_autopoietic = use_autopoietic  # 是否使用自生成机制
        self.sr_ratio = sr_ratio      # 从8减小到6，平衡信息保留
        self.init_tau = init_tau      # 从1.5调整到1.3，使注意力分布适度平滑
        self.batch_first = batch_first # 是否使用batch_first格式
        
        # 训练稳定性参数 - 适度调整
        self.warmup_steps = warmup_steps      # 从1000增加到1500，适度延长预热
        self.weight_decay = weight_decay      # 从0.01增加到0.015，适度增强正则化
        self.attention_dropout = attention_dropout  # 设置为0.15，增强注意力机制稳定性
        self.layer_norm_eps = layer_norm_eps  # 保持原值1e-5
        self.gradient_clip = gradient_clip    # 从1.0调整到0.8，适度梯度裁剪
        
        # DBC-DAC相关参数
        self.use_dbc_dac = use_dbc_dac
        self.rank_ratio_proj = rank_ratio_proj
        self.rank_ratio_res = rank_ratio_res
        self.dbc_threshold = dbc_threshold
        self.dbc_iterations = dbc_iterations
        
        # 分词器和语言参数
        self.tokenizer_type = tokenizer_type
        self.language = language
        
        # 添加任何额外参数
        for key, value in kwargs.items():
            setattr(self, key, value)
    
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
            'gradient_clip': self.gradient_clip,
            'use_dbc_dac': self.use_dbc_dac,
            'rank_ratio_proj': self.rank_ratio_proj,
            'rank_ratio_res': self.rank_ratio_res,
            'dbc_threshold': self.dbc_threshold,
            'dbc_iterations': self.dbc_iterations,
            'tokenizer_type': self.tokenizer_type,
            'language': self.language
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """从字典创建配置"""
        return cls(**config_dict)
    
    def save_pretrained(self, save_directory):
        """保存配置到指定目录"""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        config_file = os.path.join(save_directory, "config.json")
        
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_path):
        """从预训练目录加载配置"""
        import os
        import json
        
        config_file = os.path.join(model_path, "config.json")
        
        if not os.path.exists(config_file):
            raise ValueError(f"在 {model_path} 中找不到config.json")
        
        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)


class MultimodalConfig:
    """多模态配置类"""
    
    def __init__(
        self,
        enable_image: bool = False,  # 是否启用图像模态
        enable_audio: bool = False,  # 是否启用音频模态
        image_size: int = 224,       # 图像大小
        patch_size: int = 16,        # 图像patch大小
        audio_sample_rate: int = 16000,  # 音频采样率
        max_audio_length: int = 10,  # 最大音频长度(秒)
        modality_dropout: float = 0.1,  # 模态Dropout率
        **kwargs
    ):
        """初始化多模态配置"""
        self.enable_image = enable_image
        self.enable_audio = enable_audio
        self.image_size = image_size
        self.patch_size = patch_size
        self.audio_sample_rate = audio_sample_rate
        self.max_audio_length = max_audio_length
        self.modality_dropout = modality_dropout
        
        # 添加其他参数
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # 后处理
        self.__post_init__()
    
    def __post_init__(self):
        """初始化后处理"""
        # 检查是否至少启用了一个额外模态
        if not (self.enable_image or self.enable_audio):
            print("警告: 未启用任何额外模态，将使用纯文本模式")
        
        # 计算图像块数
        if self.enable_image:
            self.num_patches = (self.image_size // self.patch_size) ** 2
        else:
            self.num_patches = 0
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'enable_image': self.enable_image,
            'enable_audio': self.enable_audio,
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'audio_sample_rate': self.audio_sample_rate,
            'max_audio_length': self.max_audio_length,
            'modality_dropout': self.modality_dropout,
            'num_patches': getattr(self, 'num_patches', 0)
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """从字典创建配置"""
        return cls(**{k: v for k, v in config_dict.items() if k != 'num_patches'})
    
    def get_enabled_modalities(self):
        """获取已启用的模态列表"""
        modalities = ['text']  # 文本模态始终启用
        if self.enable_image:
            modalities.append('image')
        if self.enable_audio:
            modalities.append('audio')
        return modalities


class HardwareProfile:
    """硬件配置信息类"""
    
    def __init__(
        self,
        gpu_count: int = 0,
        gpu_type: str = "",
        gpu_memory: int = 0,  # 单位MB
        cpu_count: int = 0,
        ram_size: int = 0,    # 单位MB
        disk_space: int = 0,  # 单位MB
        **kwargs
    ):
        """初始化硬件配置信息"""
        self.gpu_count = gpu_count
        self.gpu_type = gpu_type
        self.gpu_memory = gpu_memory
        self.cpu_count = cpu_count
        self.ram_size = ram_size
        self.disk_space = disk_space
        
        # 添加其他参数
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def detect_hardware(cls):
        """检测当前系统的硬件配置"""
        import os
        import platform
        import psutil
        
        # 检测CPU信息
        cpu_count = os.cpu_count()
        
        # 检测内存信息
        ram_info = psutil.virtual_memory()
        ram_size = ram_info.total // (1024 * 1024)  # 转换为MB
        
        # 检测磁盘信息
        disk_info = psutil.disk_usage('/')
        disk_space = disk_info.total // (1024 * 1024)  # 转换为MB
        
        # 检测GPU信息
        gpu_count = 0
        gpu_type = ""
        gpu_memory = 0
        
        try:
            from apt_model.utils.fake_torch import get_torch
            torch = get_torch()
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    gpu_type = torch.cuda.get_device_name(0)
                    # 注意: 这里的GPU内存检测可能不准确
                    try:
                        import subprocess
                        result = subprocess.check_output(
                            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,nounits,noheader'],
                            encoding='utf-8'
                        )
                        gpu_memory = int(result.strip().split('\n')[0]) // 1024  # 转换为MB
                    except:
                        # 如果nvidia-smi不可用，使用估计值
                        gpu_memory = 8 * 1024  # 假设8GB
        except:
            pass
        
        return cls(
            gpu_count=gpu_count,
            gpu_type=gpu_type,
            gpu_memory=gpu_memory,
            cpu_count=cpu_count,
            ram_size=ram_size,
            disk_space=disk_space
        )
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'gpu_count': self.gpu_count,
            'gpu_type': self.gpu_type,
            'gpu_memory': self.gpu_memory,
            'cpu_count': self.cpu_count,
            'ram_size': self.ram_size,
            'disk_space': self.disk_space
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """从字典创建配置"""
        return cls(**config_dict)
    
    def is_compatible_with(self, model_config: APTConfig) -> bool:
        """检查硬件是否与模型配置兼容"""
        # 估算模型大小（参数量 * 4字节）
        model_size_mb = self._estimate_model_size_mb(model_config)
        
        # 检查GPU内存是否足够（若使用GPU）
        if self.gpu_count > 0:
            # 估计模型训练所需的GPU内存（模型大小的3倍作为安全边界）
            required_gpu_mb = model_size_mb * 3
            if required_gpu_mb > self.gpu_memory:
                return False
        else:
            # 检查CPU训练时RAM是否足够
            required_ram_mb = model_size_mb * 2
            if required_ram_mb > self.ram_size:
                return False
        
        return True
    
    def _estimate_model_size_mb(self, config: APTConfig) -> float:
        """估算模型大小（MB）"""
        # 主要参数来源:
        # 1. 词嵌入: vocab_size * d_model
        # 2. 编码器: num_encoder_layers * (注意力参数 + 前馈参数)
        # 3. 解码器: num_decoder_layers * (注意力参数 + 交叉注意力参数 + 前馈参数)
        # 其中注意力参数 = 4 * d_model * d_model
        # 前馈参数 = 2 * d_model * d_ff
        
        embed_params = config.vocab_size * config.d_model
        
        attn_params = 4 * config.d_model * config.d_model  # Q,K,V,O投影
        ff_params = 2 * config.d_model * config.d_ff       # 两个线性层
        
        encoder_layer_params = attn_params + ff_params
        decoder_layer_params = 2 * attn_params + ff_params  # 自注意力+交叉注意力
        
        total_params = (
            embed_params + 
            config.num_encoder_layers * encoder_layer_params +
            config.num_decoder_layers * decoder_layer_params
        )
        
        # 每个参数4字节(float32)，转换为MB
        model_size_mb = total_params * 4 / (1024 * 1024)
        return model_size_mb


def create_optimized_config(
    size: str = "base",
    language: str = "en",
    use_autopoietic: bool = True,
    multimodal: bool = False
) -> APTConfig:
    """
    创建优化的模型配置
    
    参数:
        size: 模型大小，可选 "tiny", "small", "base", "large"
        language: 语言，可选 "en"(英文), "zh"(中文), "multilingual"(多语言)
        use_autopoietic: 是否使用自生成机制
        multimodal: 是否使用多模态
        
    返回:
        APTConfig: 优化的配置
    """
    # 基础配置 - 根据大小调整
    if size == "tiny":
        config = APTConfig(
            vocab_size=30522,
            d_model=256,
            d_ff=1024,
            num_heads=4,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dropout=0.1,
            max_seq_len=512
        )
    elif size == "small":
        config = APTConfig(
            vocab_size=30522,
            d_model=512,
            d_ff=2048,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dropout=0.1,
            max_seq_len=512
        )
    elif size == "base":
        config = APTConfig(
            vocab_size=50257,
            d_model=768,
            d_ff=3072,
            num_heads=12,
            num_encoder_layers=12,
            num_decoder_layers=12,
            dropout=0.1,
            max_seq_len=2048
        )
    elif size == "large":
        config = APTConfig(
            vocab_size=50257,
            d_model=1024,
            d_ff=4096,
            num_heads=16,
            num_encoder_layers=24,
            num_decoder_layers=24,
            dropout=0.15,
            max_seq_len=2048
        )
    else:
        raise ValueError(f"不支持的模型大小: {size}")
    
    # 语言特定配置
    if language == "zh":
        config.vocab_size = 21128  # 中文词汇表大小估计值
        config.tokenizer_type = "chinese-char"
        config.language = "zh"
    elif language == "multilingual":
        config.vocab_size = 250000  # 多语言词汇表大小估计值
        config.d_model = min(config.d_model * 2, 2048)  # 增大模型维度
        config.language = "multilingual"
    else:  # 默认英文
        config.language = "en"
    
    # 自生成参数配置
    config.use_autopoietic = use_autopoietic
    
    # 多模态配置
    if multimodal:
        # 增加模型容量来处理多模态
        config.d_model = min(config.d_model * 2, 2048)
        config.d_ff = min(config.d_ff * 2, 8192)
    
    return config


# 为了向后兼容性，保留原始名称
APTModelConfig = APTConfig