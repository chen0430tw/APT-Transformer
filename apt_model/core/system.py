#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
核心系统模块 - 系统初始化和设备管理

整合功能：
- 设备管理（CPU/GPU选择）
- 随机种子设置
- 内存管理
- 系统初始化

整合自：
- apt_model/utils/__init__.py (set_seed, get_device, memory_cleanup)
- apt_model/utils/common.py (_initialize_common)
"""

import os
import gc
import random
from datetime import datetime
from typing import Tuple, Optional
import logging

import torch
import numpy as np


# ============================================================================
# 设备管理
# ============================================================================

def get_device(force_cpu: bool = False) -> torch.device:
    """
    获取计算设备

    参数:
        force_cpu: 是否强制使用CPU

    返回:
        torch.device: 计算设备
    """
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device_info() -> dict:
    """
    获取设备详细信息

    返回:
        dict: 设备信息字典
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "device_name": "CPU",
        "cuda_version": None,
    }

    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda

    return info


def set_device(device_id: int = 0) -> torch.device:
    """
    设置CUDA设备

    参数:
        device_id: GPU设备ID

    返回:
        torch.device: 设备对象
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


# ============================================================================
# 随机种子管理
# ============================================================================

def set_seed(seed: int = 42) -> None:
    """
    设置随机种子以确保可重现性

    参数:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 确保CUDA操作的确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# 内存管理
# ============================================================================

def memory_cleanup() -> None:
    """清理内存和缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_memory_info() -> dict:
    """
    获取内存使用信息

    返回:
        dict: 内存信息字典
    """
    info = {"ram": {}, "vram": {}}

    # RAM信息
    try:
        import psutil
        memory = psutil.virtual_memory()
        info["ram"] = {
            "total_gb": memory.total / (1024**3),
            "used_gb": memory.used / (1024**3),
            "available_gb": memory.available / (1024**3),
            "percent": memory.percent,
        }
    except ImportError:
        info["ram"] = {"error": "psutil not installed"}

    # VRAM信息
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            info["vram"][f"gpu_{i}"] = {
                "allocated_gb": torch.cuda.memory_allocated(i) / (1024**3),
                "reserved_gb": torch.cuda.memory_reserved(i) / (1024**3),
                "max_allocated_gb": torch.cuda.max_memory_allocated(i) / (1024**3),
            }

    return info


# ============================================================================
# 系统初始化
# ============================================================================

class SystemInitializer:
    """
    系统初始化器

    集成系统初始化的所有步骤，包括：
    - 设备设置
    - 随机种子设置
    - 日志初始化
    - 语言管理器初始化
    """

    @staticmethod
    def initialize(args, init_logging: bool = True, init_language: bool = True):
        """
        初始化系统组件

        参数:
            args: 命令行参数
            init_logging: 是否初始化日志系统
            init_language: 是否初始化语言管理器

        返回:
            tuple: (logger, language_manager, device)
        """
        # 设置随机种子
        set_seed(args.seed)

        # 设置设备
        device = get_device(args.force_cpu)

        # 初始化日志
        logger = None
        if init_logging:
            from apt_model.infrastructure.logging import setup_logging
            log_level = "DEBUG" if getattr(args, 'verbose', False) else "INFO"
            log_file = f"apt_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            logger = setup_logging(log_file=log_file, level=log_level)

        # 初始化语言管理器
        lang_manager = None
        if init_language:
            from apt_model.utils.language_manager import LanguageManager
            lang_manager = LanguageManager(
                getattr(args, 'language', 'zh_CN'),
                getattr(args, 'language_file', None)
            )

            # 记录基本信息
            if logger and lang_manager:
                logger.info(lang_manager.get("welcome"))
                logger.info(lang_manager.get("language") + f": {args.language}")
                logger.info(lang_manager.get("training.device").format(device))

        return logger, lang_manager, device


# ============================================================================
# 便捷函数（向后兼容）
# ============================================================================

def _initialize_common(args):
    """
    初始化通用组件（向后兼容）

    参数:
        args: 命令行参数

    返回:
        tuple: (logger, language_manager, device)
    """
    return SystemInitializer.initialize(args)


# ============================================================================
# 模块级变量
# ============================================================================

# 默认设备（用于向后兼容）
device = get_device()


# ============================================================================
# 公共API
# ============================================================================

__all__ = [
    # 设备管理
    'get_device',
    'get_device_info',
    'set_device',
    'device',
    # 随机种子
    'set_seed',
    # 内存管理
    'memory_cleanup',
    'get_memory_info',
    # 系统初始化
    'SystemInitializer',
    '_initialize_common',  # 向后兼容
]
