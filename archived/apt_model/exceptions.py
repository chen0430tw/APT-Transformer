#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT 统一异常处理系统

提供统一的错误码、错误信息和异常类，便于调试和错误追踪。
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class APTError(Exception):
    """APT 基础异常类

    所有 APT 异常的基类，包含错误码和详细信息。
    """

    error_code: str = "E000"
    error_category: str = "UNKNOWN"

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ):
        """
        Args:
            message: 错误信息
            details: 错误详情（字典）
            suggestion: 解决建议
        """
        self.message = message
        self.details = details or {}
        self.suggestion = suggestion
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """格式化错误信息"""
        msg = f"[{self.error_code}] {self.error_category}: {self.message}"
        if self.details:
            msg += f"\nDetails: {self.details}"
        if self.suggestion:
            msg += f"\n💡 Suggestion: {self.suggestion}"
        return msg

    def log(self, level: int = logging.ERROR):
        """记录错误到日志"""
        logger.log(level, self._format_message(), exc_info=True)


# ============================================================================
# 模型相关异常 (E1xx)
# ============================================================================

class ModelError(APTError):
    """模型相关错误基类"""
    error_code = "E100"
    error_category = "MODEL"


class ModelLoadError(ModelError):
    """模型加载失败"""
    error_code = "E101"

    def __init__(self, model_path: str, reason: str):
        super().__init__(
            f"Failed to load model from '{model_path}': {reason}",
            details={"model_path": model_path, "reason": reason},
            suggestion="Check if the model file exists and is not corrupted. "
                      "Try using weights_only=True for security."
        )


class ModelArchitectureError(ModelError):
    """模型架构不匹配"""
    error_code = "E102"

    def __init__(self, expected: str, actual: str):
        super().__init__(
            f"Model architecture mismatch: expected {expected}, got {actual}",
            details={"expected": expected, "actual": actual},
            suggestion="Make sure you're loading a checkpoint compatible with this model class."
        )


class ModelInitializationError(ModelError):
    """模型初始化失败"""
    error_code = "E103"

    def __init__(self, model_name: str, reason: str):
        super().__init__(
            f"Failed to initialize model '{model_name}': {reason}",
            details={"model_name": model_name, "reason": reason}
        )


# ============================================================================
# 训练相关异常 (E2xx)
# ============================================================================

class TrainingError(APTError):
    """训练相关错误基类"""
    error_code = "E200"
    error_category = "TRAINING"


class DataLoadError(TrainingError):
    """数据加载失败"""
    error_code = "E201"

    def __init__(self, data_path: str, reason: str):
        super().__init__(
            f"Failed to load training data from '{data_path}': {reason}",
            details={"data_path": data_path, "reason": reason},
            suggestion="Check if the data file exists and is in the correct format."
        )


class TrainingConfigError(TrainingError):
    """训练配置错误"""
    error_code = "E202"

    def __init__(self, config_key: str, issue: str):
        super().__init__(
            f"Invalid training configuration for '{config_key}': {issue}",
            details={"config_key": config_key, "issue": issue},
            suggestion="Check the training configuration documentation."
        )


class CheckpointSaveError(TrainingError):
    """检查点保存失败"""
    error_code = "E203"

    def __init__(self, checkpoint_path: str, reason: str):
        super().__init__(
            f"Failed to save checkpoint to '{checkpoint_path}': {reason}",
            details={"checkpoint_path": checkpoint_path, "reason": reason},
            suggestion="Check disk space and write permissions."
        )


class OutOfMemoryError(TrainingError):
    """显存不足"""
    error_code = "E204"

    def __init__(self, required_mb: Optional[int] = None, available_mb: Optional[int] = None):
        details = {}
        if required_mb:
            details["required_mb"] = required_mb
        if available_mb:
            details["available_mb"] = available_mb

        super().__init__(
            "GPU out of memory during training",
            details=details,
            suggestion="Try reducing batch_size, max_length, or use gradient_accumulation_steps. "
                      "Consider enabling mixed precision training (fp16/bf16)."
        )


# ============================================================================
# 数据相关异常 (E3xx)
# ============================================================================

class DataError(APTError):
    """数据相关错误基类"""
    error_code = "E300"
    error_category = "DATA"


class DataFormatError(DataError):
    """数据格式错误"""
    error_code = "E301"

    def __init__(self, expected_format: str, actual_format: str, file_path: str):
        super().__init__(
            f"Data format mismatch in '{file_path}': expected {expected_format}, got {actual_format}",
            details={"expected": expected_format, "actual": actual_format, "file": file_path},
            suggestion="Check the data file format. Supported formats: txt, json, csv, jsonl."
        )


class DataValidationError(DataError):
    """数据验证失败"""
    error_code = "E302"

    def __init__(self, field: str, reason: str):
        super().__init__(
            f"Data validation failed for field '{field}': {reason}",
            details={"field": field, "reason": reason}
        )


class EmptyDatasetError(DataError):
    """数据集为空"""
    error_code = "E303"

    def __init__(self, dataset_path: str):
        super().__init__(
            f"Dataset is empty: '{dataset_path}'",
            details={"dataset_path": dataset_path},
            suggestion="Make sure the dataset file contains valid data."
        )


# ============================================================================
# 配置相关异常 (E4xx)
# ============================================================================

class ConfigError(APTError):
    """配置相关错误基类"""
    error_code = "E400"
    error_category = "CONFIG"


class ConfigFileError(ConfigError):
    """配置文件错误"""
    error_code = "E401"

    def __init__(self, config_path: str, reason: str):
        super().__init__(
            f"Failed to load config from '{config_path}': {reason}",
            details={"config_path": config_path, "reason": reason},
            suggestion="Check if the config file exists and is valid YAML/JSON."
        )


class MissingConfigError(ConfigError):
    """缺少必需配置"""
    error_code = "E402"

    def __init__(self, config_key: str):
        super().__init__(
            f"Required configuration key is missing: '{config_key}'",
            details={"config_key": config_key},
            suggestion=f"Add '{config_key}' to your configuration file or pass it as an argument."
        )


class InvalidConfigValueError(ConfigError):
    """配置值无效"""
    error_code = "E403"

    def __init__(self, config_key: str, value: Any, reason: str):
        super().__init__(
            f"Invalid value for config '{config_key}': {value} - {reason}",
            details={"config_key": config_key, "value": str(value), "reason": reason}
        )


# ============================================================================
# 插件相关异常 (E5xx)
# ============================================================================

class PluginError(APTError):
    """插件相关错误基类"""
    error_code = "E500"
    error_category = "PLUGIN"


class PluginLoadError(PluginError):
    """插件加载失败"""
    error_code = "E501"

    def __init__(self, plugin_name: str, reason: str):
        super().__init__(
            f"Failed to load plugin '{plugin_name}': {reason}",
            details={"plugin_name": plugin_name, "reason": reason},
            suggestion="Check if the plugin is installed and its dependencies are met."
        )


class PluginNotFoundError(PluginError):
    """插件不存在"""
    error_code = "E502"

    def __init__(self, plugin_name: str):
        super().__init__(
            f"Plugin not found: '{plugin_name}'",
            details={"plugin_name": plugin_name},
            suggestion="Check available plugins with: apt-cli list-plugins"
        )


class PluginDependencyError(PluginError):
    """插件依赖缺失"""
    error_code = "E503"

    def __init__(self, plugin_name: str, missing_dependency: str):
        super().__init__(
            f"Plugin '{plugin_name}' requires missing dependency: {missing_dependency}",
            details={"plugin_name": plugin_name, "dependency": missing_dependency},
            suggestion=f"Install the dependency: pip install {missing_dependency}"
        )


# ============================================================================
# API相关异常 (E6xx)
# ============================================================================

class APIError(APTError):
    """API相关错误基类"""
    error_code = "E600"
    error_category = "API"


class APIKeyError(APIError):
    """API密钥错误"""
    error_code = "E601"

    def __init__(self, service_name: str):
        super().__init__(
            f"API key not found for service: {service_name}",
            details={"service": service_name},
            suggestion=f"Set the API key via environment variable or config file. "
                      f"Example: export {service_name.upper()}_API_KEY=your_key"
        )


class APIRequestError(APIError):
    """API请求失败"""
    error_code = "E602"

    def __init__(self, service_name: str, status_code: int, reason: str):
        super().__init__(
            f"API request to {service_name} failed: {status_code} - {reason}",
            details={"service": service_name, "status_code": status_code, "reason": reason}
        )


class APIRateLimitError(APIError):
    """API速率限制"""
    error_code = "E603"

    def __init__(self, service_name: str, retry_after: Optional[int] = None):
        details = {"service": service_name}
        if retry_after:
            details["retry_after_seconds"] = retry_after

        super().__init__(
            f"API rate limit exceeded for {service_name}",
            details=details,
            suggestion=f"Wait {retry_after} seconds before retrying" if retry_after else "Reduce request frequency"
        )


# ============================================================================
# 硬件相关异常 (E7xx)
# ============================================================================

class HardwareError(APTError):
    """硬件相关错误基类"""
    error_code = "E700"
    error_category = "HARDWARE"


class GPUNotFoundError(HardwareError):
    """GPU不可用"""
    error_code = "E701"

    def __init__(self):
        super().__init__(
            "No GPU detected, but GPU is required for this operation",
            suggestion="Install CUDA and PyTorch with GPU support, or use --device cpu"
        )


class CUDAError(HardwareError):
    """CUDA错误"""
    error_code = "E702"

    def __init__(self, cuda_error: str):
        super().__init__(
            f"CUDA error: {cuda_error}",
            details={"cuda_error": cuda_error},
            suggestion="Check CUDA installation and GPU driver version"
        )


# ============================================================================
# 安全相关异常 (E8xx)
# ============================================================================

class SecurityError(APTError):
    """安全相关错误基类"""
    error_code = "E800"
    error_category = "SECURITY"


class PathTraversalError(SecurityError):
    """路径遍历攻击"""
    error_code = "E801"

    def __init__(self, attempted_path: str):
        super().__init__(
            f"Path traversal attempt detected: '{attempted_path}'",
            details={"attempted_path": attempted_path},
            suggestion="Only use relative paths within the project directory"
        )


class UnsafeFileOperationError(SecurityError):
    """不安全的文件操作"""
    error_code = "E802"

    def __init__(self, operation: str, file_path: str, reason: str):
        super().__init__(
            f"Unsafe file operation '{operation}' on '{file_path}': {reason}",
            details={"operation": operation, "file_path": file_path, "reason": reason}
        )


class InputValidationError(SecurityError):
    """输入验证失败"""
    error_code = "E803"

    def __init__(self, field: str, value: str, reason: str):
        super().__init__(
            f"Input validation failed for '{field}': {reason}",
            details={"field": field, "value": value[:100], "reason": reason},
            suggestion="Check input format and constraints"
        )


# ============================================================================
# 工具函数
# ============================================================================

def handle_exception(
    error: Exception,
    logger: logging.Logger,
    reraise: bool = True,
    level: int = logging.ERROR
) -> None:
    """统一的异常处理函数

    Args:
        error: 捕获的异常
        logger: 日志记录器
        reraise: 是否重新抛出异常
        level: 日志级别
    """
    if isinstance(error, APTError):
        error.log(level)
    else:
        logger.log(level, f"Unexpected error: {error}", exc_info=True)

    if reraise:
        raise error


# 错误码参考表（用于文档）
ERROR_CODE_REFERENCE = {
    "E1xx": "模型相关错误",
    "E2xx": "训练相关错误",
    "E3xx": "数据相关错误",
    "E4xx": "配置相关错误",
    "E5xx": "插件相关错误",
    "E6xx": "API相关错误",
    "E7xx": "硬件相关错误",
    "E8xx": "安全相关错误",
}
