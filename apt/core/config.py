#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Core Configuration Module

Defines configuration classes for the APT (Autopoietic Transformer) framework.
Supports both programmatic configuration and YAML-based profiles.

Key features:
- YAML profile loading
- JSON serialization/deserialization
- Hardware compatibility checking
- Multimodal configuration
- Provider-based configuration
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Main Model Configuration
# ============================================================================

@dataclass
class APTConfig:
    """
    APT model configuration.

    This configuration class defines all parameters for the APT model,
    including architecture, training, and provider settings.
    """

    # ========== Model Architecture ==========
    vocab_size: int = 50257
    d_model: int = 768
    d_ff: int = 2048
    num_heads: int = 12
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dropout: float = 0.15
    max_seq_len: int = 512

    # ========== Autopoietic Attention Parameters ==========
    epsilon: float = 0.08  # Infinity inverse scaling factor
    alpha: float = 0.0008  # Taylor expansion coefficient
    beta: float = 0.003  # Dynamic adjustment coefficient
    use_autopoietic: bool = True  # Enable autopoietic mechanism
    sr_ratio: int = 6  # Self-generation matrix compression ratio
    init_tau: float = 1.3  # Initial temperature parameter

    # ========== Training Parameters ==========
    base_lr: float = 4e-5  # Base learning rate
    warmup_steps: int = 1500  # Warmup steps
    weight_decay: float = 0.015  # Weight decay
    attention_dropout: float = 0.15  # Attention dropout
    gradient_clip: float = 0.8  # Gradient clipping threshold

    # ========== Special Token IDs ==========
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # ========== Other Parameters ==========
    activation: str = "gelu"
    batch_first: bool = True
    layer_norm_eps: float = 1e-5

    # ========== DBC-DAC Parameters ==========
    use_dbc_dac: bool = False  # Enable DBC-DAC stabilization
    rank_ratio_proj: float = 0.1  # DBC projection ratio
    rank_ratio_res: float = 0.05  # DAC residual ratio
    dbc_threshold: float = 1e-6  # DBC threshold
    dbc_iterations: int = 1  # DAC iterations

    # ========== Language & Tokenizer ==========
    tokenizer_type: Optional[str] = None
    language: Optional[str] = None

    # ========== Provider Configuration ==========
    attention_name: str = "tva_default"  # Attention provider name
    ffn_name: str = "default"  # FFN provider name
    router_name: str = "topk_default"  # Router provider name (for MoE)
    align_name: str = "bistate_default"  # Alignment provider name
    retrieval_name: str = "none"  # Retrieval provider name

    # ========== Plugin Configuration ==========
    plugins: List[str] = field(default_factory=list)  # Enabled plugins

    # ========== Schedule Configuration ==========
    schedules: Dict[str, Any] = field(default_factory=dict)  # Curriculum schedules

    # ========== Additional Parameters ==========
    extra: Dict[str, Any] = field(default_factory=dict)  # Extra parameters

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'APTConfig':
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            APTConfig instance
        """
        # Filter out unknown keys
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}

        # Store extra parameters
        extra = {k: v for k, v in config_dict.items() if k not in valid_keys}
        if extra:
            filtered['extra'] = extra

        return cls(**filtered)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'APTConfig':
        """
        Load configuration from YAML profile.

        Args:
            yaml_path: Path to YAML file

        Returns:
            APTConfig instance

        Example:
            config = APTConfig.from_yaml('profiles/gpt5_moe_reasoning.yaml')
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_dict = yaml.safe_load(f)

        # Flatten nested structure
        config_dict = cls._flatten_yaml(yaml_dict)

        logger.info(f"Loaded configuration from {yaml_path}")
        return cls.from_dict(config_dict)

    @classmethod
    def _flatten_yaml(cls, yaml_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten nested YAML structure into flat config dict.

        YAML structure:
            model:
              d_model: 768
              attention_name: tva_default
            training:
              batch_size: 32

        Becomes:
            {
              'd_model': 768,
              'attention_name': 'tva_default',
              'batch_size': 32,  # goes to extra
            }
        """
        flat = {}

        # Top-level keys
        for key in ['name', 'version', 'description', 'plugins', 'schedules']:
            if key in yaml_dict:
                flat[key] = yaml_dict[key]

        # Model section
        if 'model' in yaml_dict:
            for k, v in yaml_dict['model'].items():
                # Handle nested dicts (e.g., model.tva, model.moe)
                if isinstance(v, dict):
                    flat['extra'] = flat.get('extra', {})
                    flat['extra'][k] = v
                else:
                    flat[k] = v

        # Training, data, hardware, etc. -> extra
        for section in ['training', 'data', 'evaluation', 'monitoring', 'hardware', 'debug']:
            if section in yaml_dict:
                flat['extra'] = flat.get('extra', {})
                flat['extra'][section] = yaml_dict[section]

        return flat

    def save_pretrained(self, save_directory: str):
        """
        Save configuration to directory.

        Args:
            save_directory: Directory path
        """
        os.makedirs(save_directory, exist_ok=True)
        config_file = os.path.join(save_directory, "config.json")

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Configuration saved to {config_file}")

    @classmethod
    def from_pretrained(cls, model_path: str) -> 'APTConfig':
        """
        Load configuration from pretrained directory.

        Args:
            model_path: Model directory path

        Returns:
            APTConfig instance
        """
        config_file = os.path.join(model_path, "config.json")

        if not os.path.exists(config_file):
            raise FileNotFoundError(f"config.json not found in {model_path}")

        with open(config_file, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    def get_provider_config(self, provider_type: str) -> Dict[str, Any]:
        """
        Get provider-specific configuration.

        Args:
            provider_type: Provider type ('attention', 'ffn', 'router', etc.)

        Returns:
            Configuration dictionary for the provider
        """
        # Build config from main params
        config = {
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'd_ff': self.d_ff,
            'dropout': self.dropout,
        }

        # Add provider-specific params from extra
        if provider_type in self.extra:
            config.update(self.extra[provider_type])

        return config

    def __repr__(self):
        return (
            f"APTConfig("
            f"d_model={self.d_model}, "
            f"num_heads={self.num_heads}, "
            f"num_layers={self.num_encoder_layers}, "
            f"plugins={self.plugins})"
        )


# ============================================================================
# Multimodal Configuration
# ============================================================================

@dataclass
class MultimodalConfig:
    """Multimodal configuration for image/audio processing."""

    enable_image: bool = False
    enable_audio: bool = False
    image_size: int = 224
    patch_size: int = 16
    audio_sample_rate: int = 16000
    max_audio_length: int = 10
    modality_dropout: float = 0.1

    def __post_init__(self):
        """Post-initialization processing."""
        if not (self.enable_image or self.enable_audio):
            logger.warning("No modalities enabled, using text-only mode")

        if self.enable_image:
            self.num_patches = (self.image_size // self.patch_size) ** 2
        else:
            self.num_patches = 0

    def get_enabled_modalities(self) -> List[str]:
        """Get list of enabled modalities."""
        modalities = ['text']
        if self.enable_image:
            modalities.append('image')
        if self.enable_audio:
            modalities.append('audio')
        return modalities

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MultimodalConfig':
        """Create from dictionary."""
        filtered = {k: v for k, v in config_dict.items() if k != 'num_patches'}
        return cls(**filtered)


# ============================================================================
# Hardware Profile
# ============================================================================

@dataclass
class HardwareProfile:
    """Hardware configuration and compatibility checking."""

    gpu_count: int = 0
    gpu_type: str = ""
    gpu_memory: int = 0  # MB
    cpu_count: int = 0
    ram_size: int = 0  # MB
    disk_space: int = 0  # MB

    @classmethod
    def detect_hardware(cls) -> 'HardwareProfile':
        """
        Detect current system hardware.

        Returns:
            HardwareProfile with detected specs
        """
        import platform
        import psutil

        # CPU
        cpu_count = os.cpu_count() or 0

        # RAM
        ram_info = psutil.virtual_memory()
        ram_size = ram_info.total // (1024 * 1024)

        # Disk
        disk_info = psutil.disk_usage('/')
        disk_space = disk_info.total // (1024 * 1024)

        # GPU
        gpu_count = 0
        gpu_type = ""
        gpu_memory = 0

        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    gpu_type = torch.cuda.get_device_name(0)
                    # Try to get actual GPU memory
                    try:
                        import subprocess
                        result = subprocess.check_output(
                            ['nvidia-smi', '--query-gpu=memory.total',
                             '--format=csv,nounits,noheader'],
                            encoding='utf-8'
                        )
                        gpu_memory = int(result.strip().split('\n')[0])
                    except:
                        gpu_memory = 8 * 1024  # Default 8GB
        except ImportError:
            pass

        logger.info(
            f"Detected hardware: {gpu_count} GPUs ({gpu_type}), "
            f"{cpu_count} CPUs, {ram_size}MB RAM"
        )

        return cls(
            gpu_count=gpu_count,
            gpu_type=gpu_type,
            gpu_memory=gpu_memory,
            cpu_count=cpu_count,
            ram_size=ram_size,
            disk_space=disk_space
        )

    def is_compatible_with(self, model_config: APTConfig) -> bool:
        """
        Check hardware compatibility with model config.

        Args:
            model_config: APT model configuration

        Returns:
            True if hardware is sufficient
        """
        model_size_mb = self._estimate_model_size_mb(model_config)

        if self.gpu_count > 0:
            # GPU training: need 3x model size
            required_gpu_mb = model_size_mb * 3
            return required_gpu_mb <= self.gpu_memory
        else:
            # CPU training: need 2x model size
            required_ram_mb = model_size_mb * 2
            return required_ram_mb <= self.ram_size

    def _estimate_model_size_mb(self, config: APTConfig) -> float:
        """Estimate model size in MB."""
        # Embeddings
        embed_params = config.vocab_size * config.d_model

        # Per-layer parameters
        attn_params = 4 * config.d_model * config.d_model
        ff_params = 2 * config.d_model * config.d_ff

        # Total
        encoder_layer_params = attn_params + ff_params
        decoder_layer_params = 2 * attn_params + ff_params

        total_params = (
            embed_params +
            config.num_encoder_layers * encoder_layer_params +
            config.num_decoder_layers * decoder_layer_params
        )

        # float32: 4 bytes per parameter
        model_size_mb = total_params * 4 / (1024 * 1024)
        return model_size_mb

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HardwareProfile':
        """Create from dictionary."""
        return cls(**config_dict)


# ============================================================================
# Helper Functions
# ============================================================================

def create_optimized_config(
    size: str = "base",
    language: str = "en",
    use_autopoietic: bool = True,
    multimodal: bool = False
) -> APTConfig:
    """
    Create optimized model configuration.

    Args:
        size: Model size ('tiny', 'small', 'base', 'large')
        language: Language ('en', 'zh', 'multilingual')
        use_autopoietic: Enable autopoietic mechanism
        multimodal: Enable multimodal

    Returns:
        Optimized APTConfig

    Example:
        config = create_optimized_config(size='base', language='zh')
    """
    # Base configurations by size
    size_configs = {
        'tiny': {
            'vocab_size': 30522,
            'd_model': 256,
            'd_ff': 1024,
            'num_heads': 4,
            'num_encoder_layers': 3,
            'num_decoder_layers': 3,
            'dropout': 0.1,
            'max_seq_len': 512
        },
        'small': {
            'vocab_size': 30522,
            'd_model': 512,
            'd_ff': 2048,
            'num_heads': 8,
            'num_encoder_layers': 6,
            'num_decoder_layers': 6,
            'dropout': 0.1,
            'max_seq_len': 512
        },
        'base': {
            'vocab_size': 50257,
            'd_model': 768,
            'd_ff': 3072,
            'num_heads': 12,
            'num_encoder_layers': 12,
            'num_decoder_layers': 12,
            'dropout': 0.1,
            'max_seq_len': 2048
        },
        'large': {
            'vocab_size': 50257,
            'd_model': 1024,
            'd_ff': 4096,
            'num_heads': 16,
            'num_encoder_layers': 24,
            'num_decoder_layers': 24,
            'dropout': 0.15,
            'max_seq_len': 2048
        }
    }

    if size not in size_configs:
        raise ValueError(f"Unsupported model size: {size}")

    config_dict = size_configs[size].copy()

    # Language-specific adjustments
    if language == "zh":
        config_dict['vocab_size'] = 21128
        config_dict['tokenizer_type'] = "chinese-char"
        config_dict['language'] = "zh"
    elif language == "multilingual":
        config_dict['vocab_size'] = 250000
        config_dict['d_model'] = min(config_dict['d_model'] * 2, 2048)
        config_dict['language'] = "multilingual"
    else:
        config_dict['language'] = "en"

    # Autopoietic mechanism
    config_dict['use_autopoietic'] = use_autopoietic

    # Multimodal expansion
    if multimodal:
        config_dict['d_model'] = min(config_dict['d_model'] * 2, 2048)
        config_dict['d_ff'] = min(config_dict['d_ff'] * 2, 8192)

    return APTConfig.from_dict(config_dict)


# Backward compatibility aliases
APTModelConfig = APTConfig
