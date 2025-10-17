#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model (自生成变换器) Utils Module
提供各种辅助功能的工具模块
"""

# Import utility functions and classes for easier access from the package level
from .logging_utils import setup_logging
from .resource_monitor import ResourceMonitor
from .error_handler import EnhancedErrorHandler
from .cache_manager import CacheManager
from .language_manager import LanguageManager
from .hardware_check import check_hardware_compatibility
try:
    from .visualization import ModelVisualizer
except ModuleNotFoundError as exc:
    if exc.name not in {"matplotlib", "apt_model.utils.visualization"}:
        raise
    ModelVisualizer = None  # Optional dependency (matplotlib) is not available.
from .time_estimator import TrainingTimeEstimator

# Set up common devices and seed utilities
# These are sometimes imported from the root but defined here
import torch
import random
import numpy as np

def set_seed(seed):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device(force_cpu=False):
    """获取计算设备"""
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def memory_cleanup():
    """清理内存"""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Set up default device
device = get_device()

# Define version
__version__ = '0.1.0'

__all__ = [
    'setup_logging',
    'ResourceMonitor',
    'EnhancedErrorHandler',
    'CacheManager',
    'LanguageManager',
    'check_hardware_compatibility',
    'ModelVisualizer',
    'TrainingTimeEstimator',
    'set_seed',
    'get_device',
    'memory_cleanup',
    'device'
]