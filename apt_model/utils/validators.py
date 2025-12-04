#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
输入验证和安全检查工具

提供统一的输入验证、路径检查、API key 管理等安全功能。
"""

import os
import re
from pathlib import Path
from typing import Optional, Union, List
import logging

from apt_model.exceptions import (
    PathTraversalError,
    InputValidationError,
    APIKeyError,
    InvalidConfigValueError
)

logger = logging.getLogger(__name__)


# ============================================================================
# 路径验证
# ============================================================================

def validate_file_path(
    file_path: Union[str, Path],
    must_exist: bool = False,
    allowed_extensions: Optional[List[str]] = None,
    base_dir: Optional[Path] = None
) -> Path:
    """验证文件路径安全性

    Args:
        file_path: 文件路径
        must_exist: 是否必须存在
        allowed_extensions: 允许的文件扩展名列表（如 ['.txt', '.json']）
        base_dir: 基础目录，如果指定则只允许该目录下的路径

    Returns:
        Path: 验证后的路径对象

    Raises:
        PathTraversalError: 检测到路径遍历攻击
        FileNotFoundError: 文件不存在（当 must_exist=True 时）
        InputValidationError: 文件扩展名不允许
    """
    path = Path(file_path).resolve()

    # 检查路径遍历
    if base_dir:
        base_dir = Path(base_dir).resolve()
        try:
            path.relative_to(base_dir)
        except ValueError:
            raise PathTraversalError(str(file_path))

    # 检查文件是否存在
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # 检查扩展名
    if allowed_extensions:
        if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
            raise InputValidationError(
                field="file_path",
                value=str(file_path),
                reason=f"File extension must be one of {allowed_extensions}, got {path.suffix}"
            )

    return path


def validate_directory(
    dir_path: Union[str, Path],
    must_exist: bool = False,
    create_if_missing: bool = False
) -> Path:
    """验证目录路径

    Args:
        dir_path: 目录路径
        must_exist: 是否必须存在
        create_if_missing: 如果不存在是否创建

    Returns:
        Path: 验证后的目录路径

    Raises:
        NotADirectoryError: 路径不是目录
    """
    path = Path(dir_path).resolve()

    if path.exists() and not path.is_dir():
        raise NotADirectoryError(f"Path exists but is not a directory: {path}")

    if must_exist and not path.exists():
        if create_if_missing:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")
        else:
            raise FileNotFoundError(f"Directory not found: {path}")

    return path


# ============================================================================
# 数值验证
# ============================================================================

def validate_range(
    value: Union[int, float],
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    param_name: str = "value"
) -> Union[int, float]:
    """验证数值范围

    Args:
        value: 要验证的值
        min_value: 最小值（包含）
        max_value: 最大值（包含）
        param_name: 参数名称（用于错误信息）

    Returns:
        验证后的值

    Raises:
        InvalidConfigValueError: 值超出范围
    """
    if min_value is not None and value < min_value:
        raise InvalidConfigValueError(
            config_key=param_name,
            value=value,
            reason=f"Value must be >= {min_value}"
        )

    if max_value is not None and value > max_value:
        raise InvalidConfigValueError(
            config_key=param_name,
            value=value,
            reason=f"Value must be <= {max_value}"
        )

    return value


def validate_positive(value: Union[int, float], param_name: str = "value") -> Union[int, float]:
    """验证正数

    Args:
        value: 要验证的值
        param_name: 参数名称

    Returns:
        验证后的值

    Raises:
        InvalidConfigValueError: 不是正数
    """
    if value <= 0:
        raise InvalidConfigValueError(
            config_key=param_name,
            value=value,
            reason="Value must be positive (> 0)"
        )
    return value


# ============================================================================
# 字符串验证
# ============================================================================

def validate_string_length(
    text: str,
    min_length: int = 0,
    max_length: Optional[int] = None,
    param_name: str = "text"
) -> str:
    """验证字符串长度

    Args:
        text: 要验证的字符串
        min_length: 最小长度
        max_length: 最大长度
        param_name: 参数名称

    Returns:
        验证后的字符串

    Raises:
        InputValidationError: 长度超出范围
    """
    length = len(text)

    if length < min_length:
        raise InputValidationError(
            field=param_name,
            value=text[:100],
            reason=f"String length must be >= {min_length}, got {length}"
        )

    if max_length and length > max_length:
        raise InputValidationError(
            field=param_name,
            value=text[:100],
            reason=f"String length must be <= {max_length}, got {length}"
        )

    return text


def sanitize_filename(filename: str) -> str:
    """清理文件名，移除不安全字符

    Args:
        filename: 原始文件名

    Returns:
        清理后的文件名
    """
    # 移除路径分隔符和特殊字符
    filename = re.sub(r'[/\\:*?"<>|]', '_', filename)
    # 移除前后空白
    filename = filename.strip()
    # 限制长度
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255 - len(ext)] + ext

    return filename


# ============================================================================
# API Key 管理
# ============================================================================

def get_api_key(
    service_name: str,
    env_var_name: Optional[str] = None,
    config_value: Optional[str] = None,
    required: bool = True
) -> Optional[str]:
    """获取 API Key（优先使用环境变量）

    Args:
        service_name: 服务名称（如 'openai', 'anthropic'）
        env_var_name: 环境变量名（如果不指定，自动生成为 {SERVICE}_API_KEY）
        config_value: 配置文件中的值（作为备选）
        required: 是否必需

    Returns:
        API Key 或 None

    Raises:
        APIKeyError: 必需但未找到 API Key
    """
    if not env_var_name:
        env_var_name = f"{service_name.upper()}_API_KEY"

    # 优先从环境变量获取
    api_key = os.getenv(env_var_name)

    # 如果环境变量没有，使用配置值
    if not api_key and config_value:
        api_key = config_value
        logger.warning(
            f"Using API key from config for {service_name}. "
            f"Consider using environment variable {env_var_name} instead for better security."
        )

    # 如果必需但没有找到
    if required and not api_key:
        raise APIKeyError(service_name)

    return api_key


# ============================================================================
# 模型文件验证
# ============================================================================

def validate_model_file(
    model_path: Union[str, Path],
    check_size: bool = True,
    max_size_gb: float = 50.0
) -> Path:
    """验证模型文件

    Args:
        model_path: 模型文件路径
        check_size: 是否检查文件大小
        max_size_gb: 最大文件大小（GB）

    Returns:
        验证后的路径

    Raises:
        FileNotFoundError: 文件不存在
        InputValidationError: 文件过大或格式错误
    """
    path = Path(model_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    # 检查扩展名
    allowed_extensions = ['.pt', '.pth', '.bin', '.safetensors', '.ckpt']
    if path.suffix.lower() not in allowed_extensions:
        raise InputValidationError(
            field="model_path",
            value=str(model_path),
            reason=f"Model file must be one of {allowed_extensions}, got {path.suffix}"
        )

    # 检查文件大小
    if check_size:
        size_gb = path.stat().st_size / (1024 ** 3)
        if size_gb > max_size_gb:
            raise InputValidationError(
                field="model_path",
                value=str(model_path),
                reason=f"Model file too large: {size_gb:.2f}GB (max: {max_size_gb}GB)"
            )

    return path


# ============================================================================
# 批量大小验证
# ============================================================================

def validate_batch_size(
    batch_size: int,
    available_memory_mb: Optional[int] = None,
    model_size_mb: Optional[int] = None
) -> int:
    """验证批量大小

    Args:
        batch_size: 批量大小
        available_memory_mb: 可用内存（MB）
        model_size_mb: 模型大小（MB）

    Returns:
        验证后的批量大小

    Raises:
        InvalidConfigValueError: 批量大小无效
    """
    if batch_size <= 0:
        raise InvalidConfigValueError(
            config_key="batch_size",
            value=batch_size,
            reason="Batch size must be positive"
        )

    # 粗略估计内存需求
    if available_memory_mb and model_size_mb:
        # 估算：模型 + 梯度 + 优化器状态 ≈ 4x 模型大小
        # 每个样本额外需要约 batch_size * 100MB（粗略估计）
        estimated_mb = model_size_mb * 4 + batch_size * 100

        if estimated_mb > available_memory_mb:
            suggested_batch_size = max(1, int((available_memory_mb - model_size_mb * 4) / 100))
            logger.warning(
                f"Batch size {batch_size} may cause OOM. "
                f"Estimated memory: {estimated_mb}MB, Available: {available_memory_mb}MB. "
                f"Suggested batch size: {suggested_batch_size}"
            )

    return batch_size


# ============================================================================
# URL 验证
# ============================================================================

def validate_url(url: str, allowed_schemes: Optional[List[str]] = None) -> str:
    """验证 URL 格式

    Args:
        url: URL 字符串
        allowed_schemes: 允许的协议（如 ['http', 'https']）

    Returns:
        验证后的 URL

    Raises:
        InputValidationError: URL 格式错误
    """
    # 简单的 URL 格式检查
    url_pattern = re.compile(
        r'^(?P<scheme>https?|ftp)://'  # 协议
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # 域名
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP
        r'(?::\d+)?'  # 端口
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )

    match = url_pattern.match(url)
    if not match:
        raise InputValidationError(
            field="url",
            value=url,
            reason="Invalid URL format"
        )

    # 检查协议
    if allowed_schemes:
        scheme = match.group('scheme').lower()
        if scheme not in allowed_schemes:
            raise InputValidationError(
                field="url",
                value=url,
                reason=f"URL scheme must be one of {allowed_schemes}, got {scheme}"
            )

    return url
