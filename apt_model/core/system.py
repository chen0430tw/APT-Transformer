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

from apt_model.utils.fake_torch import get_torch
torch = get_torch()
import numpy as np


# ============================================================================
# 设备管理
# ============================================================================

def get_device(force_cpu: bool = False,
               prefer_npu: bool = False,
               prefer_hpu: bool = False,
               prefer_xpu: bool = False) -> torch.device:
    """
    获取计算设备（支持多厂商加速器）

    参数:
        force_cpu: 是否强制使用CPU
        prefer_npu: 是否优先使用华为昇腾NPU
        prefer_hpu: 是否优先使用Intel Habana Gaudi HPU
        prefer_xpu: 是否优先使用Intel XPU (包括Ultra NPU)

    返回:
        torch.device: 计算设备

    优先级（无prefer参数时）: CUDA > HPU > NPU > XPU > CPU
    """
    if force_cpu:
        return torch.device("cpu")

    # 检测各类加速器
    cuda_available = torch.cuda.is_available()

    npu_available = False
    try:
        import torch_npu
        npu_available = torch_npu.npu.is_available()
    except ImportError:
        pass

    hpu_available = False
    try:
        import habana_frameworks.torch as habana_torch
        hpu_available = hasattr(habana_torch, 'hpu') and habana_torch.hpu.is_available()
    except ImportError:
        pass

    xpu_available = False
    try:
        import intel_extension_for_pytorch as ipex
        xpu_available = hasattr(ipex, 'xpu') and ipex.xpu.is_available()
    except ImportError:
        pass

    # 用户指定优先级
    if prefer_hpu and hpu_available:
        return torch.device("hpu:0")
    if prefer_npu and npu_available:
        return torch.device("npu:0")
    if prefer_xpu and xpu_available:
        return torch.device("xpu:0")

    # 默认优先级: CUDA > HPU > NPU > XPU > CPU
    if cuda_available:
        return torch.device("cuda")
    elif hpu_available:
        return torch.device("hpu:0")
    elif npu_available:
        return torch.device("npu:0")
    elif xpu_available:
        return torch.device("xpu:0")
    else:
        return torch.device("cpu")


def get_device_info() -> dict:
    """
    获取设备详细信息（支持多厂商加速器）

    返回:
        dict: 设备信息字典
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "npu_available": False,
        "hpu_available": False,
        "xpu_available": False,
        "device_count": 0,
        "device_name": "CPU",
        "cuda_version": None,
        "npu_version": None,
        "hpu_version": None,
        "xpu_version": None,
        "device_type": "cpu",
    }

    # 检测CUDA (优先级最高)
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["device_type"] = "cuda"
        return info

    # 检测Intel Habana Gaudi HPU
    try:
        import habana_frameworks.torch as habana_torch
        if hasattr(habana_torch, 'hpu') and habana_torch.hpu.is_available():
            info["hpu_available"] = True
            info["device_count"] = habana_torch.hpu.device_count()
            info["device_name"] = f"Intel Habana Gaudi HPU"
            info["hpu_version"] = getattr(habana_torch, '__version__', 'unknown')
            info["device_type"] = "hpu"
            return info
    except ImportError:
        pass

    # 检测华为昇腾NPU
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            info["npu_available"] = True
            info["device_count"] = torch_npu.npu.device_count()
            info["device_name"] = f"Huawei Ascend NPU {torch_npu.npu.get_device_name(0)}"
            info["npu_version"] = getattr(torch_npu, '__version__', 'unknown')
            info["device_type"] = "npu"
            return info
    except ImportError:
        pass

    # 检测Intel XPU
    try:
        import intel_extension_for_pytorch as ipex
        if hasattr(ipex, 'xpu') and ipex.xpu.is_available():
            info["xpu_available"] = True
            info["device_count"] = ipex.xpu.device_count()
            info["device_name"] = f"Intel XPU {ipex.xpu.get_device_name(0)}"
            info["xpu_version"] = getattr(ipex, '__version__', 'unknown')
            info["device_type"] = "xpu"
            return info
    except ImportError:
        pass

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
    设置随机种子以确保可重现性（支持多厂商加速器）

    参数:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CUDA种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 华为昇腾NPU种子
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            torch_npu.npu.manual_seed_all(seed)
    except ImportError:
        pass

    # Intel Habana Gaudi HPU种子
    try:
        import habana_frameworks.torch as habana_torch
        if hasattr(habana_torch, 'hpu') and habana_torch.hpu.is_available():
            habana_torch.hpu.manual_seed_all(seed)
    except (ImportError, AttributeError):
        pass

    # Intel XPU种子
    try:
        import intel_extension_for_pytorch as ipex
        if hasattr(ipex, 'xpu') and ipex.xpu.is_available():
            ipex.xpu.manual_seed_all(seed)
    except (ImportError, AttributeError):
        pass


# ============================================================================
# 内存管理
# ============================================================================

def memory_cleanup() -> None:
    """清理内存和缓存（支持多厂商加速器）"""
    gc.collect()

    # 清理CUDA缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 清理华为昇腾NPU缓存
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            torch_npu.npu.empty_cache()
    except ImportError:
        pass

    # 清理Intel Habana Gaudi HPU缓存
    try:
        import habana_frameworks.torch as habana_torch
        if hasattr(habana_torch, 'hpu') and habana_torch.hpu.is_available():
            habana_torch.hpu.empty_cache()
    except (ImportError, AttributeError):
        pass

    # 清理Intel XPU缓存
    try:
        import intel_extension_for_pytorch as ipex
        if hasattr(ipex, 'xpu') and ipex.xpu.is_available():
            ipex.xpu.empty_cache()
    except (ImportError, AttributeError):
        pass


def get_memory_info() -> dict:
    """
    获取内存使用信息（支持多厂商加速器）

    返回:
        dict: 内存信息字典
    """
    info = {"ram": {}, "vram": {}, "npu_memory": {}, "hpu_memory": {}, "xpu_memory": {}}

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

    # VRAM信息（GPU）
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            info["vram"][f"gpu_{i}"] = {
                "allocated_gb": torch.cuda.memory_allocated(i) / (1024**3),
                "reserved_gb": torch.cuda.memory_reserved(i) / (1024**3),
                "max_allocated_gb": torch.cuda.max_memory_allocated(i) / (1024**3),
            }

    # 华为昇腾NPU内存信息
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            for i in range(torch_npu.npu.device_count()):
                info["npu_memory"][f"npu_{i}"] = {
                    "allocated_gb": torch_npu.npu.memory_allocated(i) / (1024**3),
                    "reserved_gb": torch_npu.npu.memory_reserved(i) / (1024**3),
                    "max_allocated_gb": torch_npu.npu.max_memory_allocated(i) / (1024**3),
                }
    except (ImportError, AttributeError):
        pass

    # Intel Habana Gaudi HPU内存信息
    try:
        import habana_frameworks.torch as habana_torch
        if hasattr(habana_torch, 'hpu') and habana_torch.hpu.is_available():
            for i in range(habana_torch.hpu.device_count()):
                try:
                    info["hpu_memory"][f"hpu_{i}"] = {
                        "allocated_gb": habana_torch.hpu.memory_allocated(i) / (1024**3),
                        "reserved_gb": habana_torch.hpu.memory_reserved(i) / (1024**3),
                        "max_allocated_gb": habana_torch.hpu.max_memory_allocated(i) / (1024**3),
                    }
                except AttributeError:
                    pass
    except ImportError:
        pass

    # Intel XPU内存信息
    try:
        import intel_extension_for_pytorch as ipex
        if hasattr(ipex, 'xpu') and ipex.xpu.is_available():
            for i in range(ipex.xpu.device_count()):
                try:
                    info["xpu_memory"][f"xpu_{i}"] = {
                        "allocated_gb": ipex.xpu.memory_allocated(i) / (1024**3),
                        "reserved_gb": ipex.xpu.memory_reserved(i) / (1024**3),
                        "max_allocated_gb": ipex.xpu.max_memory_allocated(i) / (1024**3),
                    }
                except AttributeError:
                    pass
    except ImportError:
        pass

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
