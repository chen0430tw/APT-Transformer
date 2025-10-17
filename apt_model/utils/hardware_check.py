#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model training time estimation utilities.
Provides tools to estimate training time based on model configuration, dataset size, and hardware.
"""

import os
import math
import time
import platform
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

import torch
import numpy as np

# 尝试导入 psutil (CPU/内存检测)
try:
    import psutil
except ImportError:
    psutil = None

# 尝试导入 pynvml (更详细的 GPU 信息)
try:
    import pynvml as nvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

# 尝试导入 GPUtil (获取 GPU 负载、显存使用等)
try:
    import GPUtil
    gputil = GPUtil
except ImportError:
    gputil = None


# ---------------------------------------------------------------------
# 1. GPU 性能估计
# ---------------------------------------------------------------------
def estimate_gpu_performance(gpu_name: str) -> float:
    """
    Estimate GPU performance in TFLOPS based on common GPU models.
    
    Args:
        gpu_name (str): Name of the GPU
        
    Returns:
        float: Estimated TFLOPs, or 10.0 if unknown
    """
    # Performance map for common GPUs (approximate FP32 TFLOPS)
    performance_map = {
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
        
        # GTX 10 Series
        "GTX 1080 Ti": 11.34,
        "GTX 1080": 8.87,
        "GTX 1070": 6.46,
        
        # RTX 40 Series
        "RTX 4090": 82.58,
        "RTX 4080": 48.74,
        "RTX 4070 Ti": 40.09,
        "RTX 4070": 29.15,
        "RTX 4060 Ti": 22.06,
        "RTX 4060": 15.09,
        
        # 专业卡 / Data Center
        "RTX A6000": 38.71,
        "RTX A5000": 27.8,
        "RTX A4000": 19.17,
        "Quadro RTX 8000": 16.3,
        "Quadro RTX 6000": 16.3,
        "Quadro RTX 5000": 11.2,
        "Quadro RTX 4000": 7.1,
        
        # A100 / A800
        "A100": 19.5,    # FP32
        "A100 SXM4": 19.5,
        "A100 PCIe": 19.5,
        "A800": 19.5,    # China-specific version of A100
        "A10": 31.2,
        "A40": 37.4,
        "A30": 24.1,
        "A10G": 31.2,
        
        # H100 / H800
        "H100": 51.0,    # FP32
        "H100 SXM5": 51.0,
        "H100 PCIe": 51.0,
        "H800": 51.0,    # China-specific version of H100
        
        # 其他 Data Center
        "L4": 30.3,
        "L40": 90.5,
        
        # Older Tesla
        "V100": 14.0,
        "V100 SXM2": 15.7,
        "V100 PCIe": 14.0,
        "T4": 8.1,
        "P100": 10.6,
        "K80": 8.7,
        "Tesla K80": 8.73,
        "Tesla P4": 5.5,
        "Tesla M40": 7.0,
        "Tesla M60": 9.7,
        
        # Titan
        "Titan RTX": 16.31,
        "Titan V": 14.9,
        "Titan X": 11.0,
    }
    
    for model, tflops in performance_map.items():
        if model in gpu_name:
            return tflops
    
    # Fallback to default value if unknown
    return 10.0


# ---------------------------------------------------------------------
# 2. CUDA 设备信息
# ---------------------------------------------------------------------
def get_cuda_device_properties():
    """
    Get CUDA device properties for all available GPUs.
    
    Returns:
        List of dicts containing GPU properties
    """
    import torch
    if not torch.cuda.is_available():
        return []
    
    devices = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        try:
            clock_rate = props.clock_rate / 1000  # Convert to MHz
        except AttributeError:
            clock_rate = 0  # 默认值
        devices.append({
            'index': i,
            'name': torch.cuda.get_device_name(i),
            'total_memory': props.total_memory / (1024**3),  # Convert to GB
            'compute_capability': f"{props.major}.{props.minor}",
            'multi_processor_count': getattr(props, 'multi_processor_count', 0),
            'max_threads_per_block': getattr(props, 'max_threads_per_block', 1024),
            'clock_rate': clock_rate
        })
    return devices


# ---------------------------------------------------------------------
# 3. 硬件信息收集
# ---------------------------------------------------------------------
def get_hardware_profile() -> Dict[str, Union[str, int, float, List]]:
    """
    Get detailed hardware profile information.
    
    Returns:
        Dict: Hardware profile information
    """
    profile = {
        'cpu': {
            'name': 'Unknown',
            'cores': 0,
            'threads': 0,
            'frequency': 0.0,
            'architecture': platform.machine(),
        },
        'memory': {
            'total_gb': 0.0,
            'available_gb': 0.0,
        },
        'gpu': {
            'available': torch.cuda.is_available(),
            'count': 0,
            'devices': [],
            'total_vram_gb': 0.0,
            'estimated_tflops': 0.0,
        },
        'disk': {
            'total_gb': 0.0,
            'available_gb': 0.0,
        },
        'system': {
            'os': platform.system(),
            'os_version': platform.version(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        }
    }
    
    # Detect CPU information
    try:
        if psutil:
            profile['cpu']['cores'] = psutil.cpu_count(logical=False)
            profile['cpu']['threads'] = psutil.cpu_count(logical=True)
            profile['cpu']['frequency'] = psutil.cpu_freq().current if psutil.cpu_freq() else 0.0
            
            # Try to get CPU name on different platforms
            if platform.system() == "Windows":
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                profile['cpu']['name'] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
            elif platform.system() == "Darwin":  # macOS
                import subprocess
                cmd = ['sysctl', '-n', 'machdep.cpu.brand_string']
                profile['cpu']['name'] = subprocess.check_output(cmd).decode().strip()
            elif platform.system() == "Linux":
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if line.startswith('model name'):
                            profile['cpu']['name'] = line.split(':')[1].strip()
                            break
    except Exception as e:
        logging.warning(f"Error detecting CPU information: {e}")
    
    # Detect memory information
    try:
        if psutil:
            mem = psutil.virtual_memory()
            profile['memory']['total_gb'] = mem.total / (1024**3)
            profile['memory']['available_gb'] = mem.available / (1024**3)
    except Exception as e:
        logging.warning(f"Error detecting memory information: {e}")
    
    # Detect disk information
    try:
        if psutil:
            disk = psutil.disk_usage('/')
            profile['disk']['total_gb'] = disk.total / (1024**3)
            profile['disk']['available_gb'] = disk.free / (1024**3)
    except Exception as e:
        logging.warning(f"Error detecting disk information: {e}")
    
    # Detect GPU information
    if torch.cuda.is_available():
        try:
            gpu_count = torch.cuda.device_count()
            profile['gpu']['count'] = gpu_count
            
            total_vram = 0.0
            total_tflops = 0.0
            
            gpu_properties = get_cuda_device_properties()
            profile['gpu']['devices'] = gpu_properties
            
            for gpu in gpu_properties:
                total_vram += gpu['total_memory']
                gpu_tflops = estimate_gpu_performance(gpu['name'])
                total_tflops += gpu_tflops
                
                # Add estimated TFLOPS to each GPU in the list
                for device in profile['gpu']['devices']:
                    if device['index'] == gpu['index']:
                        device['estimated_tflops'] = gpu_tflops
                        
                        # Special handling for A800 and H800 (China-specific versions)
                        if "A800" in gpu['name']:
                            device['notes'] = "A800是A100的中国特供版，计算性能相同但NVLink带宽限制为400GB/s"
                        elif "H800" in gpu['name']:
                            device['notes'] = "H800是H100的中国特供版，计算性能相近但NVLink带宽有限制"
                        # Add note for unknown GPUs
                        elif estimate_gpu_performance(gpu['name']) == 10.0 and "RTX" not in gpu['name'] and "GTX" not in gpu['name']:
                            device['notes'] = "未知GPU型号，性能估计可能不准确"
            
            profile['gpu']['total_vram_gb'] = total_vram
            profile['gpu']['estimated_tflops'] = total_tflops
            
            # Try to get more detailed GPU info using NVML if available
            if NVML_AVAILABLE:
                try:
                    nvml.nvmlInit()
                    for i in range(gpu_count):
                        handle = nvml.nvmlDeviceGetHandleByIndex(i)
                        util = nvml.nvmlDeviceGetUtilizationRates(handle)
                        temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                        
                        for device in profile['gpu']['devices']:
                            if device['index'] == i:
                                device['utilization'] = util.gpu
                                device['temperature'] = temp
                    nvml.nvmlShutdown()
                except Exception as e:
                    logging.warning(f"Error getting detailed GPU info via NVML: {e}")
            
            # Try to get additional GPU info using GPUtil
            if gputil:
                try:
                    gpus = gputil.getGPUs()
                    for gpu in gpus:
                        for device in profile['gpu']['devices']:
                            if device['index'] == gpu.id:
                                device['load'] = gpu.load * 100
                                device['memory_used'] = gpu.memoryUsed / 1024  # GB
                                device['memory_free'] = gpu.memoryFree / 1024  # GB
                except Exception as e:
                    logging.warning(f"Error getting GPU info via GPUtil: {e}")
                
        except Exception as e:
            logging.warning(f"Error detecting GPU information: {e}")
    
    return profile


# ---------------------------------------------------------------------
# 4. 模型内存需求估计
# ---------------------------------------------------------------------
def estimate_model_memory_requirements(model_config, batch_size) -> Dict[str, float]:
    """
    Estimate model memory requirements for different precision types.
    
    Args:
        model_config: Model configuration
        batch_size: Batch size for training
        
    Returns:
        Dict with memory estimates in GB
    """
    d_model = getattr(model_config, 'd_model', 768)
    num_encoder_layers = getattr(model_config, 'num_encoder_layers', 6)
    num_decoder_layers = getattr(model_config, 'num_decoder_layers', 6)
    vocab_size = getattr(model_config, 'vocab_size', 50257)
    max_seq_len = getattr(model_config, 'max_seq_len', 512)
    
    # Embedding layer
    embedding_params = vocab_size * d_model
    
    # Encoder layers
    encoder_params_per_layer = 4 * d_model * d_model  # Self-attn
    encoder_params_per_layer += 2 * d_model * (d_model * 4)  # FFN
    encoder_params_per_layer += 4 * d_model  # LN
    
    # Decoder layers
    decoder_params_per_layer = 4 * d_model * d_model  # Self-attn
    decoder_params_per_layer += 4 * d_model * d_model  # Cross-attn
    decoder_params_per_layer += 2 * d_model * (d_model * 4)  # FFN
    decoder_params_per_layer += 6 * d_model  # LN
    
    # Output projection
    output_params = d_model * vocab_size
    
    # Total params
    total_params = (
        embedding_params +
        (encoder_params_per_layer * num_encoder_layers) +
        (decoder_params_per_layer * num_decoder_layers) +
        output_params
    )
    
    # Bytes per param
    bytes_per_param_fp32 = 4
    bytes_per_param_fp16 = 2
    bytes_per_param_int8 = 1
    
    # Activation memory (approx)
    activation_size = batch_size * max_seq_len * d_model
    activation_memory_fp32 = (
        activation_size *
        (num_encoder_layers + num_decoder_layers) *
        bytes_per_param_fp32
    )
    
    # Optimizer states (Adam ~ 8 bytes per param)
    optimizer_memory = total_params * 8
    
    # Grad memory (fp32)
    gradient_memory_fp32 = total_params * bytes_per_param_fp32
    
    # total memory (fp32)
    total_memory_fp32 = (
        (total_params * bytes_per_param_fp32) +
        optimizer_memory +
        gradient_memory_fp32 +
        activation_memory_fp32
    )
    
    # total memory (fp16)
    total_memory_fp16 = (
        (total_params * bytes_per_param_fp16) +
        optimizer_memory +                      # Optimizer (still fp32)
        (total_params * bytes_per_param_fp16) + # Grad
        (activation_memory_fp32 / 2)            # Act in fp16
    )
    
    # total memory (int8)
    total_memory_int8 = (
        (total_params * bytes_per_param_int8) +
        optimizer_memory +
        gradient_memory_fp32 +
        activation_memory_fp32
    )
    
    gb_conversion = 1024**3
    buffer_factor = 1.2  # 20% overhead
    
    return {
        'total_params': total_params,
        'model_size_gb': (total_params * bytes_per_param_fp32) / gb_conversion,
        'total_memory_gb_fp32': (total_memory_fp32 / gb_conversion) * buffer_factor,
        'total_memory_gb_fp16': (total_memory_fp16 / gb_conversion) * buffer_factor,
        'total_memory_gb_int8': (total_memory_int8 / gb_conversion) * buffer_factor,
        'total_memory_gb_recommended': max((total_memory_fp16 / gb_conversion) * buffer_factor, 1.0)
    }


# ---------------------------------------------------------------------
# 5. 硬件摘要打印
# ---------------------------------------------------------------------
def print_hardware_summary(logger=None):
    """
    Print a summary of the available hardware.
    
    Args:
        logger: Optional logger for logging messages
    """
    hardware = get_hardware_profile()
    
    summary = "=" * 60 + "\n"
    summary += "硬件信息摘要\n"
    summary += "=" * 60 + "\n\n"
    
    # System info
    summary += "系统信息:\n"
    summary += f"  操作系统: {hardware['system']['os']} {hardware['system']['os_version']}\n"
    summary += f"  Python版本: {hardware['system']['python_version']}\n"
    summary += f"  PyTorch版本: {hardware['system']['torch_version']}\n"
    if hardware['system']['cuda_version']:
        summary += f"  CUDA版本: {hardware['system']['cuda_version']}\n"
    summary += "\n"
    
    # CPU info
    summary += "CPU信息:\n"
    summary += f"  处理器: {hardware['cpu']['name']}\n"
    summary += f"  物理核心数: {hardware['cpu']['cores']}\n"
    summary += f"  逻辑处理器数: {hardware['cpu']['threads']}\n"
    if hardware['cpu']['frequency'] > 0:
        summary += f"  频率: {hardware['cpu']['frequency'] / 1000:.2f} GHz\n"
    summary += "\n"
    
    # Memory info
    summary += "内存信息:\n"
    summary += f"  总内存: {hardware['memory']['total_gb']:.1f} GB\n"
    summary += f"  可用内存: {hardware['memory']['available_gb']:.1f} GB\n"
    summary += "\n"
    
    # GPU info
    summary += "GPU信息:\n"
    if hardware['gpu']['available']:
        summary += f"  GPU数量: {hardware['gpu']['count']}\n"
        for i, device in enumerate(hardware['gpu']['devices']):
            summary += f"  GPU {i + 1}: {device['name']}\n"
            summary += f"    显存: {device['total_memory']:.1f} GB\n"
            summary += f"    计算能力: {device['compute_capability']}\n"
            if 'estimated_tflops' in device:
                summary += f"    估计性能: {device['estimated_tflops']:.1f} TFLOPS\n"
            if 'notes' in device:
                summary += f"    备注: {device['notes']}\n"
            if 'utilization' in device:
                summary += f"    使用率: {device['utilization']}%\n"
            if 'temperature' in device:
                summary += f"    温度: {device['temperature']}°C\n"
            if 'memory_used' in device and 'memory_free' in device:
                summary += f"    已用显存: {device['memory_used']:.1f} GB\n"
                summary += f"    空闲显存: {device['memory_free']:.1f} GB\n"
        summary += f"  总显存: {hardware['gpu']['total_vram_gb']:.1f} GB\n"
        summary += f"  总性能: {hardware['gpu']['estimated_tflops']:.1f} TFLOPS\n"
    else:
        summary += "  未检测到可用的GPU\n"
    summary += "\n"
    
    # Disk info
    summary += "磁盘信息:\n"
    summary += f"  总空间: {hardware['disk']['total_gb']:.1f} GB\n"
    summary += f"  可用空间: {hardware['disk']['available_gb']:.1f} GB\n"
    summary += "\n"
    
    # Print or log the summary
    if logger:
        for line in summary.split('\n'):
            logger.info(line)
    else:
        print(summary)
    
    return summary


# ---------------------------------------------------------------------
# 6. 硬件兼容性检查
# ---------------------------------------------------------------------
def check_hardware_compatibility(model_config, logger=None) -> bool:
    """
    Check if the hardware is compatible with the model requirements.
    
    Args:
        model_config: Model configuration object
        logger: Optional logger for logging messages
        
    Returns:
        bool: True if hardware is compatible, False otherwise
    """
    hardware = get_hardware_profile()
    has_gpu = hardware['gpu']['available']
    
    if not has_gpu:
        message = "未检测到GPU，将使用CPU训练（速度可能很慢）"
        if logger:
            logger.warning(message)
        else:
            print(f"警告: {message}")
        return False
    
    # 基于默认 batch_size=8 估计
    memory_requirements = estimate_model_memory_requirements(model_config, batch_size=8)
    gpu_count = hardware['gpu']['count']
    gpu_names = [device['name'] for device in hardware['gpu']['devices']]
    gpu_memory = [device['total_memory'] for device in hardware['gpu']['devices']]
    
    model_size_gb = memory_requirements['total_memory_gb_recommended']
    
    # 这里保留“杂—鱼~”等调侃提示
    if not gpu_memory:
        # 如果 gpu_memory 为空，跳过 GPU 内存检查
        logger.warning("未检测到 GPU 内存信息，跳过 GPU 内存检查")
    elif model_size_gb > max(gpu_memory):
        message = (
            f"杂—鱼~ 杂—鱼~，你的显卡看起来跟你一样什么都不行呢，果然是个垃圾♡\n"
            f"模型需要至少 {model_size_gb:.1f}GB 显存，但你的显卡只有 {max(gpu_memory):.1f}GB\n"
            f"尝试减小模型大小、批次大小或使用更少的参数"
        )
        if logger:
            logger.warning(message)
        else:
            print(f"警告: {message}")
        return False
    
    if gpu_count > 0 and len(gpu_names) == gpu_count and len(gpu_memory) == gpu_count:
        info_message = f"检测到 {gpu_count} 个GPU: " + ", ".join([f"{gpu_names[i]} ({gpu_memory[i]:.1f}GB)" for i in range(gpu_count)])
        if logger:
            logger.info(info_message)
        else:
            print(info_message)
    else:
        # 如果列表长度不匹配，给出一个一般性消息
        info_message = f"检测到 {gpu_count} 个GPU"
        if gpu_count > 0:
            if logger:
                logger.info(info_message)
                logger.warning("未能完全获取GPU详细信息")
            else:
                print(info_message)
                print("警告: 未能完全获取GPU详细信息")
    
    return True


# ---------------------------------------------------------------------
# 7. 推荐训练设置
# ---------------------------------------------------------------------
def get_recommended_settings(model_config, logger=None):
    """
    Get recommended training settings based on hardware profile.
    
    Args:
        model_config: Model configuration object
        logger: Optional logger for logging messages
        
    Returns:
        Dict: Recommended settings
    """
    hardware = get_hardware_profile()
    recommendations = {
        'batch_size': 8,
        'learning_rate': 3e-5,
        'precision': 'float32',
        'gradient_accumulation': 1,
        'use_amp': False,
    }
    
    if hardware['gpu']['available']:
        gpu_memory = max([device['total_memory'] for device in hardware['gpu']['devices']])
        gpu_names = [device['name'] for device in hardware['gpu']['devices']]
        
        memory_req = estimate_model_memory_requirements(model_config, recommendations['batch_size'])
        
        # 如果所需内存超过可用显存70%，则尝试降低 batch_size
        if memory_req['total_memory_gb_fp32'] > gpu_memory * 0.7:
            for bs_candidate in [4, 2, 1]:
                new_req = estimate_model_memory_requirements(model_config, bs_candidate)
                if new_req['total_memory_gb_fp32'] < gpu_memory * 0.7:
                    recommendations['batch_size'] = bs_candidate
                    break
        elif gpu_memory > 16:
            recommendations['batch_size'] = 16
        
        # 如果模型较大或显存较小，启用混合精度
        if (
            memory_req['total_memory_gb_fp32'] > gpu_memory * 0.5 or
            getattr(model_config, 'd_model', 768) >= 1024 or
            gpu_memory < 12
        ):
            recommendations['precision'] = 'float16'
            recommendations['use_amp'] = True
        
        # 如果仍然超过显存80%，则启用梯度累积
        if memory_req['total_memory_gb_fp32'] > gpu_memory * 0.8:
            recommendations['gradient_accumulation'] = 4
        
        # 学习率随 batch_size (√scale)
        base_batch_size = 8
        if recommendations['batch_size'] != base_batch_size:
            lr_scale_factor = (recommendations['batch_size'] / base_batch_size) ** 0.5
            recommendations['learning_rate'] = 3e-5 * lr_scale_factor
        
        # 特殊处理 H100/H800
        if any("H100" in name or "H800" in name for name in gpu_names):
            # FP8 / BF16
            recommendations['precision'] = 'bfloat16'
            recommendations['use_amp'] = True
        
        # 特殊处理 A100/A800
        if any("A100" in name or "A800" in name for name in gpu_names):
            recommendations['precision'] = 'bfloat16'
            recommendations['use_amp'] = True
        
        # 对未知 GPU 保守处理
        if any(estimate_gpu_performance(name) == 10.0 and "RTX" not in name and "GTX" not in name for name in gpu_names):
            recommendations['batch_size'] = max(1, recommendations['batch_size'] // 2)
            recommendations['precision'] = 'float16'
            recommendations['use_amp'] = True
            recommendations['gradient_accumulation'] = max(2, recommendations['gradient_accumulation'])
            recommendations['unknown_gpu'] = True
    else:
        # CPU-only
        recommendations['batch_size'] = 1
        recommendations['learning_rate'] = 1e-5
        recommendations['precision'] = 'float32'
        recommendations['use_amp'] = False
        recommendations['cpu_only'] = True
    
    if logger:
        logger.info("基于硬件配置的训练设置建议:")
        logger.info(f"  批次大小: {recommendations['batch_size']}")
        logger.info(f"  学习率: {recommendations['learning_rate']}")
        logger.info(f"  精度: {recommendations['precision']}")
        logger.info(f"  梯度累积: {recommendations['gradient_accumulation']}")
        logger.info(f"  自动混合精度: {recommendations['use_amp']}")
    
    return recommendations


# ---------------------------------------------------------------------
# 8. TrainingTimeEstimator 类 (主要)
# ---------------------------------------------------------------------
class TrainingTimeEstimator:
    """
    Training time estimation tool for APT models.
    Provides accurate time estimates based on model configuration,
    dataset properties, and available hardware.
    """
    
    def __init__(self, model_config, dataset_size, batch_size, epochs, logger=None):
        """
        Initialize the training time estimator.
        
        Args:
            model_config: Model configuration object
            dataset_size (int): Number of samples in the dataset
            batch_size (int): Training batch size
            epochs (int): Number of training epochs
            logger (optional): Logger instance for logging messages
        """
        self.logger = logger
        self.model_config = model_config
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Detect hardware configuration
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0
        self.gpu_name = torch.cuda.get_device_name(0) if self.gpu_available and self.gpu_count > 0 else "Unknown"
        self.gpu_memory = (
            torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if self.gpu_available and self.gpu_count > 0 else 0
        )
        
        # Estimate GPU performance (TFLOPS)
        self.estimated_tflops = self._estimate_gpu_performance()
    
    def _estimate_gpu_performance(self) -> float:
        """
        Estimate GPU computational performance based on the detected GPU model.
        
        Returns:
            float: Estimated performance in TFLOPS
        """
        if not self.gpu_available or self.gpu_count == 0:
            return 0.0
        
        return estimate_gpu_performance(self.gpu_name)
    
    def estimate_training_time(self) -> Dict[str, Any]:
        """
        Predict training time based on model configuration, dataset properties, and hardware.
        
        Returns:
            dict: Training time estimates and related information
        """
        # If no GPU, do a rough CPU estimate
        if not self.gpu_available:
            total_hours = self.dataset_size * self.epochs / 10.0  # extremely rough
            return {
                "total_hours": total_hours,
                "formatted_time": self._format_time(total_hours * 3600),
                "note": "CPU-only training. Actual speed might be very slow."
            }
        
        # 估算模型复杂度
        model_complexity = self._calculate_model_complexity()  # FLOPs
        # 估算每个样本 FLOP -> time
        time_per_sample = self._calculate_compute_per_sample(model_complexity)
        
        # batch_size 效率 + GPU TFLOPS
        batch_efficiency = min(1.5, 0.7 + (0.3 * math.log2(max(self.batch_size, 1)) / math.log2(32)))
        gpu_factor = 10.0 / max(self.estimated_tflops, 0.1)
        
        # 多 GPU 效率
        multi_gpu_efficiency = 0.8
        if self.gpu_count > 1:
            gpu_scaling = 1.0 + (self.gpu_count - 1) * multi_gpu_efficiency
            gpu_factor /= gpu_scaling
        
        # 计算最终单样本时间
        estimated_time_per_sample = time_per_sample * gpu_factor / batch_efficiency
        
        steps_per_epoch = math.ceil(self.dataset_size / self.batch_size)
        total_steps = steps_per_epoch * self.epochs
        
        compute_seconds = total_steps * self.batch_size * estimated_time_per_sample
        overhead_factor = 1.2
        total_seconds = compute_seconds * overhead_factor
        
        total_hours = total_seconds / 3600
        epoch_hours = total_hours / self.epochs
        
        return {
            "total_seconds": total_seconds,
            "total_hours": total_hours,
            "epoch_hours": epoch_hours,
            "formatted_total_time": self._format_time(total_seconds),
            "formatted_epoch_time": self._format_time(epoch_hours * 3600),
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "time_per_sample": estimated_time_per_sample,
            "model_complexity_gflops": model_complexity / 1e9,
            "estimated_tflops": self.estimated_tflops,
            "gpu_name": self.gpu_name,
            "gpu_count": self.gpu_count,
            "gpu_memory_gb": self.gpu_memory,
            "batch_size": self.batch_size,
            "dataset_size": self.dataset_size,
            "device": "gpu",
        }
    
    def _calculate_model_complexity(self) -> float:
        """
        计算模型 FLOPs（大致）用于训练时间预估
        """
        config = self.model_config
        d_model = config.d_model
        num_heads = getattr(config, 'num_heads', 8)
        d_ff = config.d_ff
        vocab_size = config.vocab_size
        num_encoder_layers = config.num_encoder_layers
        num_decoder_layers = config.num_decoder_layers
        seq_len = getattr(config, 'max_seq_len', 512)
        
        # 自注意力 FLOPs
        encoder_self_attn = 4 * seq_len * seq_len * d_model * num_encoder_layers
        decoder_self_attn = 4 * seq_len * seq_len * d_model * num_decoder_layers
        decoder_cross_attn = 4 * seq_len * seq_len * d_model * num_decoder_layers
        
        # FFN FLOPs
        encoder_ffn = 2 * seq_len * d_model * d_ff * num_encoder_layers
        decoder_ffn = 2 * seq_len * d_model * d_ff * num_decoder_layers
        
        # embedding & output
        embedding_flops = seq_len * d_model * vocab_size
        output_flops = seq_len * d_model * vocab_size
        
        total_flops = (
            encoder_self_attn + decoder_self_attn + decoder_cross_attn +
            encoder_ffn + decoder_ffn + embedding_flops + output_flops
        )
        
        # backward pass ~ 2.5x
        return total_flops * 2.5
    
    def _calculate_compute_per_sample(self, model_complexity: float) -> float:
        """
        Estimate compute time per sample based on model complexity.
        """
        base_flop_time = 1e-13  # ~ 10 TFLOPS baseline
        size_overhead = 1.0 + 0.1 * math.log2(max(1, model_complexity / 1e9))
        return model_complexity * base_flop_time * size_overhead
    
    def _format_time(self, seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}秒"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}分钟"
        elif seconds < 86400:
            hours = seconds / 3600
            return f"{hours:.1f}小时"
        else:
            days = seconds / 86400
            return f"{days:.1f}天"
    
    def print_estimation(self) -> Dict[str, Any]:
        estimation = self.estimate_training_time()
        print("\n" + "="*50)
        print("训练时间预估")
        print("="*50)
        
        # 模型信息
        print(f"\n模型信息:")
        print(f"  隐藏维度(d_model): {self.model_config.d_model}")
        print(f"  编码器层数: {self.model_config.num_encoder_layers}")
        print(f"  解码器层数: {self.model_config.num_decoder_layers}")
        print(f"  计算复杂度(约): {estimation['model_complexity_gflops']:.2f} GFLOPs")
        
        # 数据集信息
        print(f"\n数据集信息:")
        print(f"  样本数: {self.dataset_size}")
        print(f"  批次大小: {self.batch_size}")
        print(f"  训练轮数: {self.epochs}")
        print(f"  每轮步数: {estimation['steps_per_epoch']}")
        print(f"  总步数: {estimation['total_steps']}")
        
        # 硬件信息
        print(f"\n硬件信息:")
        if self.gpu_available:
            print(f"  GPU: {self.gpu_name} x{self.gpu_count}")
            print(f"  显存: {self.gpu_memory:.1f}GB")
            print(f"  估计性能: {self.estimated_tflops:.1f} TFLOPS")
        else:
            print(f"  使用CPU训练")
        
        # 预估训练时间
        print(f"\n预估训练时间:")
        print(f"  总时间: {estimation['formatted_total_time']}")
        if 'formatted_epoch_time' in estimation:
            print(f"  每轮时间: {estimation['formatted_epoch_time']}")
        
        # 其他提示
        if not self.gpu_available:
            print("\n警告: 使用CPU训练，速度可能极慢")
        elif self.gpu_memory < 8:
            print(f"\n警告: GPU显存较小 ({self.gpu_memory:.1f}GB)，可能影响训练速度或导致内存不足")
        
        print("="*50)
        return estimation


# ---------------------------------------------------------------------
# 9. 入口函数 & CLI
# ---------------------------------------------------------------------
def main_cli():
    import argparse
    
    parser = argparse.ArgumentParser(description='APT Model Training Time Estimator')
    parser.add_argument('--d-model', type=int, default=768, help='Model dimension')
    parser.add_argument('--num-encoder-layers', type=int, default=4, help='Number of encoder layers')
    parser.add_argument('--num-decoder-layers', type=int, default=4, help='Number of decoder layers')
    parser.add_argument('--dataset-size', type=int, default=10000, help='Dataset size')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--calibrate', action='store_true', help='Run calibration')
    
    args = parser.parse_args()
    
    # Create a small config
    from dataclasses import dataclass
    
    @dataclass
    class TempConfig:
        d_model: int
        num_encoder_layers: int
        num_decoder_layers: int
        vocab_size: int = 50257
        d_ff: int = 2048
        num_heads: int = 8
        dropout: float = 0.1
        max_seq_len: int = 512
        epsilon: float = 1.0
        alpha: float = 0.01
        beta: float = 0.01
        base_lr: float = 1e-4
    
    config = TempConfig(
        d_model=args.d_model,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers
    )
    
    estimator = TrainingTimeEstimator(
        model_config=config,
        dataset_size=args.dataset_size,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    if args.calibrate:
        print("当前示例中尚未实现完整 calibration 流程，仅演示。")
        # 如果需要，可自行实现 calibrate_estimation()
    
    estimation = estimator.print_estimation()
    print(f"\n详细预估信息: {estimation}")


if __name__ == "__main__":
    main_cli()