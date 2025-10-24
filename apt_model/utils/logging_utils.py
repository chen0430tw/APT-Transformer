#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Logging utilities for the APT Model training system.
"""

import os
import logging
from typing import Optional
import sys

def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    设置 APT 模型系统的日志配置

    参数:
        log_file: 日志文件路径。如果没有提供或不包含目录，
                  日志文件将保存在 apt_model/log 文件夹中。
        level: 日志级别（默认为 INFO）

    返回:
        logger: 配置后的日志记录器实例
    """
    logger = logging.getLogger('apt_model')
    logger.setLevel(level)
    logger.handlers = []  # 清除现有的处理器

    # 检查是否设置了APT_NO_STDOUT_ENCODING环境变量
    no_encoding = os.environ.get("APT_NO_STDOUT_ENCODING", "0") == "1"
    
    # 创建控制台处理器（根据环境变量决定是否指定UTF-8编码）
    if no_encoding:
        # 不指定编码，避免Windows中文编码问题
        console_handler = logging.StreamHandler(sys.stdout)
    else:
        # 使用UTF-8编码
        console_handler = logging.StreamHandler(
            open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
        )
    
    console_handler.setLevel(level)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 如果指定了 log_file，则创建文件处理器
    if log_file:
        # 如果 log_file 没有包含目录，则默认放在 apt_model/log 文件夹下
        if not os.path.dirname(log_file):
            # 获取当前模块所在目录（假设当前模块在 apt_model/utils 下）
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            log_dir = os.path.join(base_dir, "log")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, log_file)
        else:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 打印日志文件的绝对路径，便于调试
        print(f"日志文件保存路径：{os.path.abspath(log_file)}")
    
    return logger


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter to add colors to console output based on log level.
    """
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[41m',  # Red background
        'RESET': '\033[0m',      # Reset to default
    }
    
    def format(self, record):
        """Format log record with colors."""
        log_message = super().format(record)
        if record.levelname in self.COLORS:
            return f"{self.COLORS[record.levelname]}{log_message}{self.COLORS['RESET']}"
        return log_message


def setup_colored_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging with colored output to console.
    
    Args:
        log_file: Path to the log file (no colors in file)
        level: Logging level (default: INFO)
        
    Returns:
        logger: Configured logger instance with colored console output
    """
    logger = logging.getLogger('apt_model')
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

    # 检查是否设置了APT_NO_STDOUT_ENCODING环境变量
    no_encoding = os.environ.get("APT_NO_STDOUT_ENCODING", "0") == "1"
    
    # 创建控制台处理器（根据环境变量决定是否指定UTF-8编码）
    if no_encoding:
        # 不指定编码，避免Windows中文编码问题
        console_handler = logging.StreamHandler(sys.stdout)
    else:
        # 使用UTF-8编码
        console_handler = logging.StreamHandler(
            open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
        )
        
    console_handler.setLevel(level)
    
    colored_formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(colored_formatter)
    
    # Add console handler
    logger.addHandler(console_handler)
    
    # Add file handler if log_file is provided (without colors)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        logger.addHandler(file_handler)
    
    return logger


def get_progress_logger(name: str = 'progress', log_file: Optional[str] = None) -> logging.Logger:
    """
    Create a specialized logger for progress reporting (e.g., for tqdm integration).
    
    Args:
        name: Name of the logger
        log_file: Path to the log file
        
    Returns:
        logger: Configured progress logger
    """
    logger = logging.getLogger(f'apt_model.{name}')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers

    # 检查是否设置了APT_NO_STDOUT_ENCODING环境变量
    no_encoding = os.environ.get("APT_NO_STDOUT_ENCODING", "0") == "1"
    
    # Simple formatter for progress logs
    formatter = logging.Formatter('%(message)s')
    
    # 创建控制台处理器（根据环境变量决定是否指定UTF-8编码）
    if no_encoding:
        # 不指定编码，避免Windows中文编码问题
        console_handler = logging.StreamHandler(sys.stdout)
    else:
        # 使用UTF-8编码
        console_handler = logging.StreamHandler(
            open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
        )
        
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger