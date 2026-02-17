#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
核心硬件模块 - 硬件检测和性能评估

整合功能：
- GPU性能评估
- 硬件信息收集
- 硬件兼容性检查
- 训练设置推荐

整合自：
- apt_model/utils/hardware_check.py (部分核心功能)
"""

import platform
import logging
from typing import Dict, List, Any, Optional, Union

import torch

# 可选依赖
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import pynvml as nvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False


# ============================================================================
# GPU性能数据库
# ============================================================================

GPU_PERFORMANCE_MAP = {
    # RTX 40 Series
    "RTX 4090": 82.58,
    "RTX 4080": 48.74,
    "RTX 4070 Ti": 40.09,
    "RTX 4070": 29.15,
    "RTX 4060 Ti": 22.06,
    "RTX 4060": 15.09,

    # RTX 30 Series
    "RTX 3090": 35.58,
    "RTX 3080 Ti": 34.1,
    "RTX 3080": 29.77,
    "RTX 3070 Ti": 21.7,
    "RTX 3070": 20.31,
    "RTX 3060 Ti": 16.2,
    "RTX 3060": 12.74,

    # RTX 20 Series
    "RTX 2080 Ti": 13.45,
    "RTX 2080": 10.07,
    "RTX 2070": 7.46,
    "RTX 2060": 6.45,

    # Data Center - H100/A100
    "H100": 51.0,
    "H100 SXM5": 51.0,
    "H800": 51.0,
    "A100": 19.5,
    "A100 SXM4": 19.5,
    "A800": 19.5,
    "A10": 31.2,
    "A40": 37.4,
    "L40": 90.5,

    # Data Center - V100/T4
    "V100": 14.0,
    "V100 SXM2": 15.7,
    "T4": 8.1,
    "P100": 10.6,

    # Professional
    "RTX A6000": 38.71,
    "RTX A5000": 27.8,
    "RTX A4000": 19.17,
    "Quadro RTX 8000": 16.3,
    "Quadro RTX 6000": 16.3,

    # Titan
    "Titan RTX": 16.31,
    "Titan V": 14.9,
}


# ============================================================================
# GPU性能评估
# ============================================================================

def estimate_gpu_performance(gpu_name: str) -> float:
    """
    估算GPU性能（TFLOPS）

    参数:
        gpu_name: GPU名称

    返回:
        float: 估算的TFLOPS值（FP32）
    """
    for model, tflops in GPU_PERFORMANCE_MAP.items():
        if model in gpu_name:
            return tflops

    # 未知GPU，返回默认值
    return 10.0


def get_gpu_properties(device_index: int = 0) -> Dict[str, Any]:
    """
    获取GPU属性

    参数:
        device_index: GPU设备索引

    返回:
        dict: GPU属性字典
    """
    if not torch.cuda.is_available():
        return {}

    props = torch.cuda.get_device_properties(device_index)
    name = torch.cuda.get_device_name(device_index)

    return {
        'index': device_index,
        'name': name,
        'total_memory_gb': props.total_memory / (1024**3),
        'compute_capability': f"{props.major}.{props.minor}",
        'multi_processor_count': getattr(props, 'multi_processor_count', 0),
        'max_threads_per_block': getattr(props, 'max_threads_per_block', 1024),
        'clock_rate_mhz': getattr(props, 'clock_rate', 0) / 1000,
        'estimated_tflops': estimate_gpu_performance(name),
    }


def get_all_gpu_properties() -> List[Dict[str, Any]]:
    """
    获取所有GPU的属性

    返回:
        list: GPU属性字典列表
    """
    if not torch.cuda.is_available():
        return []

    return [get_gpu_properties(i) for i in range(torch.cuda.device_count())]


# ============================================================================
# 硬件信息收集
# ============================================================================

class HardwareProfiler:
    """硬件信息收集器"""

    @staticmethod
    def get_cpu_info() -> Dict[str, Any]:
        """获取CPU信息"""
        info = {
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'cores': 0,
            'threads': 0,
            'frequency_mhz': 0.0,
        }

        if HAS_PSUTIL:
            info['cores'] = psutil.cpu_count(logical=False) or 0
            info['threads'] = psutil.cpu_count(logical=True) or 0
            try:
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    info['frequency_mhz'] = cpu_freq.current
            except Exception:
                pass

        return info

    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """获取内存信息（GB）"""
        info = {
            'total_gb': 0.0,
            'available_gb': 0.0,
            'used_gb': 0.0,
            'percent': 0.0,
        }

        if HAS_PSUTIL:
            mem = psutil.virtual_memory()
            info['total_gb'] = mem.total / (1024**3)
            info['available_gb'] = mem.available / (1024**3)
            info['used_gb'] = mem.used / (1024**3)
            info['percent'] = mem.percent

        return info

    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """获取GPU信息"""
        info = {
            'available': torch.cuda.is_available(),
            'count': 0,
            'devices': [],
            'total_vram_gb': 0.0,
            'estimated_total_tflops': 0.0,
        }

        if torch.cuda.is_available():
            info['count'] = torch.cuda.device_count()
            info['devices'] = get_all_gpu_properties()
            info['total_vram_gb'] = sum(d['total_memory_gb'] for d in info['devices'])
            info['estimated_total_tflops'] = sum(d['estimated_tflops'] for d in info['devices'])

        return info

    @staticmethod
    def get_disk_info() -> Dict[str, float]:
        """获取磁盘信息（GB）"""
        info = {
            'total_gb': 0.0,
            'used_gb': 0.0,
            'free_gb': 0.0,
        }

        if HAS_PSUTIL:
            try:
                disk = psutil.disk_usage('/')
                info['total_gb'] = disk.total / (1024**3)
                info['used_gb'] = disk.used / (1024**3)
                info['free_gb'] = disk.free / (1024**3)
            except Exception:
                pass

        return info

    @staticmethod
    def get_full_profile() -> Dict[str, Any]:
        """
        获取完整的硬件配置信息

        返回:
            dict: 包含CPU、内存、GPU、磁盘等信息的字典
        """
        return {
            'cpu': HardwareProfiler.get_cpu_info(),
            'memory': HardwareProfiler.get_memory_info(),
            'gpu': HardwareProfiler.get_gpu_info(),
            'disk': HardwareProfiler.get_disk_info(),
            'platform': {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'python_version': platform.python_version(),
            }
        }


# ============================================================================
# 内存需求估算
# ============================================================================

def estimate_model_memory(
    num_parameters: int,
    batch_size: int = 1,
    seq_length: int = 512,
    precision: str = 'fp32'
) -> Dict[str, float]:
    """
    估算模型内存需求

    参数:
        num_parameters: 模型参数数量
        batch_size: 批次大小
        seq_length: 序列长度
        precision: 精度（'fp32', 'fp16', 'int8'）

    返回:
        dict: 内存需求（GB）
    """
    # 字节数
    bytes_per_param = {
        'fp32': 4,
        'fp16': 2,
        'bf16': 2,
        'int8': 1,
    }.get(precision, 4)

    # 模型参数
    model_memory = num_parameters * bytes_per_param

    # 优化器状态（Adam需要2倍参数）
    optimizer_memory = num_parameters * bytes_per_param * 2

    # 梯度
    gradient_memory = num_parameters * bytes_per_param

    # 激活值（粗略估计）
    activation_memory = batch_size * seq_length * num_parameters * 0.1 * bytes_per_param

    # 总计
    total_memory = model_memory + optimizer_memory + gradient_memory + activation_memory

    # 添加20%的缓冲
    total_memory_with_buffer = total_memory * 1.2

    return {
        'model_gb': model_memory / (1024**3),
        'optimizer_gb': optimizer_memory / (1024**3),
        'gradient_gb': gradient_memory / (1024**3),
        'activation_gb': activation_memory / (1024**3),
        'total_gb': total_memory / (1024**3),
        'total_with_buffer_gb': total_memory_with_buffer / (1024**3),
    }


# ============================================================================
# 硬件兼容性检查
# ============================================================================

def check_hardware_compatibility(
    model_config,
    batch_size: int = 8,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    检查硬件是否满足模型训练要求

    参数:
        model_config: 模型配置对象（需要有vocab_size, hidden_size等属性）
        batch_size: 批次大小
        logger: 日志记录器

    返回:
        bool: 硬件是否兼容
    """
    hardware = HardwareProfiler.get_full_profile()

    # 检查GPU
    if not hardware['gpu']['available']:
        msg = "未检测到GPU，将使用CPU训练（速度可能很慢）"
        if logger:
            logger.warning(msg)
        else:
            print(f"警告: {msg}")
        return False

    # 估算内存需求
    try:
        num_params = getattr(model_config, 'num_parameters', None)
        if num_params is None:
            # 粗略估计
            vocab_size = getattr(model_config, 'vocab_size', 50257)
            hidden_size = getattr(model_config, 'hidden_size', 768)
            num_layers = getattr(model_config, 'num_hidden_layers', 12)
            num_params = vocab_size * hidden_size + num_layers * hidden_size * hidden_size * 12

        memory_req = estimate_model_memory(num_params, batch_size)
        required_gb = memory_req['total_with_buffer_gb']
    except Exception as e:
        if logger:
            logger.warning(f"无法估算内存需求: {e}")
        required_gb = 4.0  # 默认估计

    # 检查显存
    devices = hardware['gpu']['devices']
    if not devices:
        if logger:
            logger.warning("未检测到GPU设备信息")
        return False

    max_gpu_memory = max(d['total_memory_gb'] for d in devices)

    if required_gb > max_gpu_memory:
        msg = (
            f"显存不足：模型需要约 {required_gb:.1f}GB 显存，"
            f"但可用显存仅有 {max_gpu_memory:.1f}GB。"
            f"建议减小batch_size或使用更小的模型。"
        )
        if logger:
            logger.warning(msg)
        else:
            print(f"警告: {msg}")
        return False

    # 显示GPU信息
    gpu_info = ", ".join([
        f"{d['name']} ({d['total_memory_gb']:.1f}GB)"
        for d in devices
    ])
    msg = f"检测到 {hardware['gpu']['count']} 个GPU: {gpu_info}"
    if logger:
        logger.info(msg)
    else:
        print(msg)

    return True


def get_recommended_batch_size(
    model_config,
    target_memory_usage: float = 0.7,
    logger: Optional[logging.Logger] = None
) -> int:
    """
    根据硬件推荐批次大小

    参数:
        model_config: 模型配置
        target_memory_usage: 目标显存使用率（0-1）
        logger: 日志记录器

    返回:
        int: 推荐的批次大小
    """
    hardware = HardwareProfiler.get_full_profile()

    if not hardware['gpu']['available'] or not hardware['gpu']['devices']:
        return 1  # CPU模式，使用小批次

    # 获取最大GPU显存
    max_gpu_memory = max(d['total_memory_gb'] for d in hardware['gpu']['devices'])
    target_memory = max_gpu_memory * target_memory_usage

    # 尝试不同的批次大小
    try:
        num_params = getattr(model_config, 'num_parameters', None)
        if num_params is None:
            vocab_size = getattr(model_config, 'vocab_size', 50257)
            hidden_size = getattr(model_config, 'hidden_size', 768)
            num_layers = getattr(model_config, 'num_hidden_layers', 12)
            num_params = vocab_size * hidden_size + num_layers * hidden_size * hidden_size * 12

        for batch_size in [32, 16, 8, 4, 2, 1]:
            mem_req = estimate_model_memory(num_params, batch_size)
            if mem_req['total_with_buffer_gb'] <= target_memory:
                if logger:
                    logger.info(f"推荐批次大小: {batch_size} (预计显存使用: {mem_req['total_with_buffer_gb']:.1f}GB)")
                return batch_size
    except Exception as e:
        if logger:
            logger.warning(f"无法计算推荐批次大小: {e}")

    return 8  # 默认值


# ============================================================================
# 便捷函数（向后兼容）
# ============================================================================

def get_hardware_profile() -> Dict[str, Any]:
    """获取硬件配置信息（向后兼容）"""
    return HardwareProfiler.get_full_profile()


# ============================================================================
# 公共API
# ============================================================================

__all__ = [
    # GPU性能
    'estimate_gpu_performance',
    'get_gpu_properties',
    'get_all_gpu_properties',
    # 硬件信息
    'HardwareProfiler',
    'get_hardware_profile',
    # 内存估算
    'estimate_model_memory',
    # 兼容性检查
    'check_hardware_compatibility',
    'get_recommended_batch_size',
]
