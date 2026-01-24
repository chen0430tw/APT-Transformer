#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基础设施 - 日志系统

整合功能：
- 标准日志设置
- 彩色日志输出
- 进度日志记录
- 日志文件管理

整合自：
- apt_model/utils/logging_utils.py
"""

import os
import sys
import logging
from typing import Optional
from datetime import datetime


# ============================================================================
# 彩色日志格式化器
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """
    自定义格式化器，根据日志级别添加颜色

    使用ANSI颜色代码为控制台输出添加颜色
    """

    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[41m',  # Red background
        'RESET': '\033[0m',      # Reset to default
    }

    def format(self, record):
        """格式化日志记录并添加颜色"""
        log_message = super().format(record)
        if record.levelname in self.COLORS:
            return f"{self.COLORS[record.levelname]}{log_message}{self.COLORS['RESET']}"
        return log_message


# ============================================================================
# 日志设置函数
# ============================================================================

def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    use_colors: bool = False
) -> logging.Logger:
    """
    设置APT模型系统的日志配置

    参数:
        log_file: 日志文件路径。如果没有提供或不包含目录，
                  日志文件将保存在 apt_model/log 文件夹中
        level: 日志级别（默认为INFO）
        use_colors: 是否使用彩色输出

    返回:
        logger: 配置后的日志记录器实例
    """
    logger = logging.getLogger('apt_model')
    logger.setLevel(level)
    logger.handlers = []  # 清除现有的处理器

    # 检查是否设置了APT_NO_STDOUT_ENCODING环境变量
    no_encoding = os.environ.get("APT_NO_STDOUT_ENCODING", "0") == "1"

    # 创建控制台处理器
    if no_encoding:
        console_handler = logging.StreamHandler(sys.stdout)
    else:
        try:
            console_handler = logging.StreamHandler(
                open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
            )
        except (OSError, ValueError):
            # 如果stdout文件描述符无效，使用简单的StreamHandler
            console_handler = logging.StreamHandler(sys.stdout)

    console_handler.setLevel(level)

    # 创建格式器
    if use_colors:
        formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 如果指定了log_file，则创建文件处理器
    if log_file:
        # 如果log_file没有包含目录，则默认放在apt_model/log文件夹下
        if not os.path.dirname(log_file):
            # 获取项目根目录
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            log_dir = os.path.join(base_dir, "log")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, log_file)
        else:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        # 文件日志不使用颜色
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # 打印日志文件的绝对路径
        print(f"日志文件保存路径：{os.path.abspath(log_file)}")

    return logger


def setup_colored_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    设置带有彩色输出的日志系统

    参数:
        log_file: 日志文件路径（文件中不使用颜色）
        level: 日志级别（默认: INFO）

    返回:
        logger: 配置后的日志记录器实例
    """
    return setup_logging(log_file=log_file, level=level, use_colors=True)


def get_progress_logger(
    name: str = 'progress',
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    创建专用的进度报告日志记录器（例如用于tqdm集成）

    参数:
        name: 日志记录器名称
        log_file: 日志文件路径

    返回:
        logger: 配置后的进度日志记录器
    """
    logger = logging.getLogger(f'apt_model.{name}')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # 清除现有处理器

    # 检查是否设置了APT_NO_STDOUT_ENCODING环境变量
    no_encoding = os.environ.get("APT_NO_STDOUT_ENCODING", "0") == "1"

    # 简单的格式器用于进度日志
    formatter = logging.Formatter('%(message)s')

    # 创建控制台处理器
    if no_encoding:
        console_handler = logging.StreamHandler(sys.stdout)
    else:
        try:
            console_handler = logging.StreamHandler(
                open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
            )
        except (OSError, ValueError):
            # 如果stdout文件描述符无效，使用简单的StreamHandler
            console_handler = logging.StreamHandler(sys.stdout)

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 添加文件处理器（如果指定）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ============================================================================
# 日志管理器类
# ============================================================================

class LogManager:
    """
    日志管理器

    提供统一的日志管理接口，支持多个日志记录器和日志文件
    """

    def __init__(self, base_dir: Optional[str] = None):
        """
        初始化日志管理器

        参数:
            base_dir: 日志文件基础目录
        """
        if base_dir is None:
            # 默认使用项目根目录下的log文件夹
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_dir = os.path.join(project_root, "log")

        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        self.loggers = {}
        self.log_files = {}

    def create_logger(
        self,
        name: str,
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        use_colors: bool = False
    ) -> logging.Logger:
        """
        创建新的日志记录器

        参数:
            name: 日志记录器名称
            log_file: 日志文件名（可选）
            level: 日志级别
            use_colors: 是否使用彩色输出

        返回:
            logger: 日志记录器实例
        """
        if log_file:
            log_file_path = os.path.join(self.base_dir, log_file)
        else:
            log_file_path = None

        logger = setup_logging(log_file=log_file_path, level=level, use_colors=use_colors)
        self.loggers[name] = logger

        if log_file:
            self.log_files[name] = log_file_path

        return logger

    def get_logger(self, name: str) -> Optional[logging.Logger]:
        """
        获取已创建的日志记录器

        参数:
            name: 日志记录器名称

        返回:
            logger: 日志记录器实例，如果不存在则返回None
        """
        return self.loggers.get(name)

    def create_session_logger(
        self,
        prefix: str = "apt_model",
        level: int = logging.INFO,
        use_colors: bool = False
    ) -> logging.Logger:
        """
        创建带有时间戳的会话日志记录器

        参数:
            prefix: 日志文件名前缀
            level: 日志级别
            use_colors: 是否使用彩色输出

        返回:
            logger: 日志记录器实例
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"{prefix}_{timestamp}.log"
        return self.create_logger(
            name=f"session_{timestamp}",
            log_file=log_file,
            level=level,
            use_colors=use_colors
        )

    def list_log_files(self) -> list:
        """
        列出所有管理的日志文件

        返回:
            list: 日志文件路径列表
        """
        return list(self.log_files.values())

    def cleanup_old_logs(self, days: int = 30) -> int:
        """
        清理旧的日志文件

        参数:
            days: 保留最近多少天的日志

        返回:
            int: 删除的文件数量
        """
        import time
        from pathlib import Path

        cutoff_time = time.time() - (days * 86400)
        deleted_count = 0

        for log_file in Path(self.base_dir).glob("*.log"):
            if log_file.stat().st_mtime < cutoff_time:
                try:
                    log_file.unlink()
                    deleted_count += 1
                except Exception as e:
                    logging.warning(f"Failed to delete log file {log_file}: {e}")

        return deleted_count


# ============================================================================
# 公共API
# ============================================================================

__all__ = [
    # 日志设置
    'setup_logging',
    'setup_colored_logging',
    'get_progress_logger',
    # 日志管理
    'LogManager',
    # 格式化器
    'ColoredFormatter',
]
