#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Profile Configuration Loader

加载和管理APT配置文件（profiles/*.yaml）

支持的profile:
- lite.yaml - 轻量级配置
- standard.yaml - 标准配置
- pro.yaml - 专业配置
- full.yaml - 完整配置

使用示例:
    from apt.core.config import load_profile

    config = load_profile('standard')
    print(f"Batch size: {config.training.batch_size}")
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProfileMetadata:
    """Profile元数据"""
    name: str
    description: str
    version: str


@dataclass
class ModelConfig:
    """模型配置"""
    architecture: str
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    intermediate_size: int
    max_position_embeddings: int
    vocab_size: int
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DistributedConfig:
    """分布式配置"""
    enabled: bool = False
    backend: Optional[str] = None
    world_size: int = 1
    pipeline_parallel: int = 1
    tensor_parallel: int = 1
    data_parallel: int = 1


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-5
    warmup_steps: int = 1000
    max_steps: int = 100000
    mixed_precision: str = 'bf16'
    optimizer: str = 'adamw'
    weight_decay: float = 0.01
    distributed: DistributedConfig = field(default_factory=DistributedConfig)


@dataclass
class VGPUConfig:
    """VGPU配置"""
    enabled: bool = False
    max_virtual_gpus: int = 1
    scheduling: str = 'fair'
    isolation: bool = False
    memory_pooling: bool = False


@dataclass
class ExtensionsConfig:
    """扩展配置"""
    rag: Dict[str, Any] = field(default_factory=dict)
    knowledge_graph: Dict[str, Any] = field(default_factory=dict)
    mcp: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringConfig:
    """监控配置"""
    tensorboard: bool = True
    wandb: bool = False
    mlflow: bool = False
    metrics_interval: int = 100


@dataclass
class CheckpointsConfig:
    """检查点配置"""
    save_interval: int = 500
    keep_last_n: int = 5
    save_optimizer: bool = True
    async_save: bool = False


@dataclass
class APTProfile:
    """APT完整配置"""
    profile: ProfileMetadata
    model: ModelConfig
    training: TrainingConfig
    vgpu: VGPUConfig = field(default_factory=VGPUConfig)
    extensions: ExtensionsConfig = field(default_factory=ExtensionsConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    checkpoints: CheckpointsConfig = field(default_factory=CheckpointsConfig)

    # 原始YAML数据
    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)


class ProfileLoader:
    """Profile配置加载器"""

    def __init__(self, profiles_dir: Optional[Union[str, Path]] = None):
        """
        初始化ProfileLoader

        Args:
            profiles_dir: profiles目录路径，默认为项目根目录下的profiles/
        """
        if profiles_dir is None:
            # 尝试找到项目根目录
            current = Path(__file__).resolve()
            # 向上查找直到找到profiles目录
            for parent in [current.parent.parent.parent, current.parent.parent.parent.parent]:
                potential = parent / 'profiles'
                if potential.exists() and potential.is_dir():
                    profiles_dir = potential
                    break

            if profiles_dir is None:
                profiles_dir = Path.cwd() / 'profiles'

        self.profiles_dir = Path(profiles_dir)

        if not self.profiles_dir.exists():
            logger.warning(f"Profiles directory not found: {self.profiles_dir}")

    def list_profiles(self) -> list:
        """列出所有可用的profile"""
        if not self.profiles_dir.exists():
            return []

        profiles = []
        for yaml_file in self.profiles_dir.glob('*.yaml'):
            if yaml_file.stem not in ['README']:
                profiles.append(yaml_file.stem)

        return sorted(profiles)

    def load(self, profile_name: str) -> APTProfile:
        """
        加载指定的profile

        Args:
            profile_name: profile名称（不含.yaml扩展名）

        Returns:
            APTProfile对象

        Raises:
            FileNotFoundError: 如果profile文件不存在
            ValueError: 如果YAML格式错误
        """
        profile_path = self.profiles_dir / f"{profile_name}.yaml"

        if not profile_path.exists():
            available = self.list_profiles()
            raise FileNotFoundError(
                f"Profile '{profile_name}' not found at {profile_path}\n"
                f"Available profiles: {', '.join(available)}"
            )

        try:
            with open(profile_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to load profile '{profile_name}': {e}")

        return self._parse_profile(data, profile_name)

    def _parse_profile(self, data: Dict[str, Any], profile_name: str) -> APTProfile:
        """解析YAML数据为APTProfile对象"""

        # Profile metadata
        profile_meta = ProfileMetadata(
            name=data.get('profile', {}).get('name', profile_name),
            description=data.get('profile', {}).get('description', ''),
            version=data.get('profile', {}).get('version', '2.0.0'),
        )

        # Model config
        model_data = data.get('model', {})
        model_config = ModelConfig(
            architecture=model_data.get('architecture', 'apt_base'),
            hidden_size=model_data.get('hidden_size', 1024),
            num_layers=model_data.get('num_layers', 24),
            num_attention_heads=model_data.get('num_attention_heads', 16),
            intermediate_size=model_data.get('intermediate_size', 4096),
            max_position_embeddings=model_data.get('max_position_embeddings', 2048),
            vocab_size=model_data.get('vocab_size', 50000),
            features=model_data.get('features', {}),
        )

        # Distributed config
        training_data = data.get('training', {})
        dist_data = training_data.get('distributed', {})
        distributed_config = DistributedConfig(
            enabled=dist_data.get('enabled', False),
            backend=dist_data.get('backend'),
            world_size=dist_data.get('world_size', 1),
            pipeline_parallel=dist_data.get('pipeline_parallel', 1),
            tensor_parallel=dist_data.get('tensor_parallel', 1),
            data_parallel=dist_data.get('data_parallel', 1),
        )

        # Training config
        training_config = TrainingConfig(
            batch_size=training_data.get('batch_size', 32),
            gradient_accumulation_steps=training_data.get('gradient_accumulation_steps', 1),
            learning_rate=training_data.get('learning_rate', 3e-5),
            warmup_steps=training_data.get('warmup_steps', 1000),
            max_steps=training_data.get('max_steps', 100000),
            mixed_precision=training_data.get('mixed_precision', 'bf16'),
            optimizer=training_data.get('optimizer', 'adamw'),
            weight_decay=training_data.get('weight_decay', 0.01),
            distributed=distributed_config,
        )

        # VGPU config
        vgpu_data = data.get('vgpu', {})
        vgpu_config = VGPUConfig(
            enabled=vgpu_data.get('enabled', False),
            max_virtual_gpus=vgpu_data.get('max_virtual_gpus', 1),
            scheduling=vgpu_data.get('scheduling', 'fair'),
            isolation=vgpu_data.get('isolation', False),
            memory_pooling=vgpu_data.get('memory_pooling', False),
        )

        # Extensions config
        ext_data = data.get('extensions', {})
        extensions_config = ExtensionsConfig(
            rag=ext_data.get('rag', {}),
            knowledge_graph=ext_data.get('knowledge_graph', {}),
            mcp=ext_data.get('mcp', {}),
        )

        # Monitoring config
        mon_data = data.get('monitoring', {})
        monitoring_config = MonitoringConfig(
            tensorboard=mon_data.get('tensorboard', True),
            wandb=mon_data.get('wandb', False),
            mlflow=mon_data.get('mlflow', False),
            metrics_interval=mon_data.get('metrics_interval', 100),
        )

        # Checkpoints config
        ckpt_data = data.get('checkpoints', {})
        checkpoints_config = CheckpointsConfig(
            save_interval=ckpt_data.get('save_interval', 500),
            keep_last_n=ckpt_data.get('keep_last_n', 5),
            save_optimizer=ckpt_data.get('save_optimizer', True),
            async_save=ckpt_data.get('async_save', False),
        )

        return APTProfile(
            profile=profile_meta,
            model=model_config,
            training=training_config,
            vgpu=vgpu_config,
            extensions=extensions_config,
            monitoring=monitoring_config,
            checkpoints=checkpoints_config,
            _raw=data,
        )


# 全局loader实例
_loader = None


def get_loader() -> ProfileLoader:
    """获取全局ProfileLoader实例"""
    global _loader
    if _loader is None:
        _loader = ProfileLoader()
    return _loader


def load_profile(profile_name: str) -> APTProfile:
    """
    加载指定的profile配置

    Args:
        profile_name: profile名称（lite, standard, pro, full）

    Returns:
        APTProfile对象

    Example:
        >>> config = load_profile('standard')
        >>> print(config.training.batch_size)
        32
    """
    return get_loader().load(profile_name)


def list_profiles() -> list:
    """
    列出所有可用的profile

    Returns:
        profile名称列表

    Example:
        >>> profiles = list_profiles()
        >>> print(profiles)
        ['full', 'lite', 'pro', 'standard']
    """
    return get_loader().list_profiles()


__all__ = [
    'ProfileLoader',
    'APTProfile',
    'ModelConfig',
    'TrainingConfig',
    'DistributedConfig',
    'VGPUConfig',
    'ExtensionsConfig',
    'MonitoringConfig',
    'CheckpointsConfig',
    'load_profile',
    'list_profiles',
    'get_loader',
]
