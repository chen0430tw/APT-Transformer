#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT 统一日志配置

提供统一的日志格式和配置，支持终端彩色输出和文件保存。
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import os


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""

    # ANSI 颜色码
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'
    }

    def format(self, record):
        # 添加颜色
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"

        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    enable_color: bool = True
) -> logging.Logger:
    """设置全局日志配置

    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径（None 则不保存文件）
        format_string: 自定义格式字符串
        enable_color: 是否启用彩色输出

    Returns:
        配置好的 root logger
    """
    # 从环境变量读取配置
    level = os.getenv('APT_LOG_LEVEL', level).upper()
    log_file = os.getenv('APT_LOG_FILE', log_file)
    enable_color = os.getenv('APT_LOG_COLOR', str(enable_color)).lower() == 'true'

    # 默认格式
    if format_string is None:
        format_string = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'

    # 时间格式
    date_format = '%Y-%m-%d %H:%M:%S'

    # 获取 root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level))

    # 清除已有的 handlers
    logger.handlers.clear()

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)

    if enable_color:
        console_formatter = ColoredFormatter(format_string, datefmt=date_format)
    else:
        console_formatter = logging.Formatter(format_string, datefmt=date_format)

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 文件 handler（如果指定）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # 文件不使用颜色
        file_formatter = logging.Formatter(format_string, datefmt=date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """获取指定名称的 logger

    Args:
        name: logger 名称（通常使用 __name__）

    Returns:
        Logger 实例
    """
    return logging.getLogger(name)


# 便捷的日志函数（用于替代简单的 print）
def log_info(message: str, logger: Optional[logging.Logger] = None):
    """记录 INFO 级别日志"""
    if logger is None:
        logger = logging.getLogger()
    logger.info(message)


def log_warning(message: str, logger: Optional[logging.Logger] = None):
    """记录 WARNING 级别日志"""
    if logger is None:
        logger = logging.getLogger()
    logger.warning(message)


def log_error(message: str, exc_info: bool = False, logger: Optional[logging.Logger] = None):
    """记录 ERROR 级别日志"""
    if logger is None:
        logger = logging.getLogger()
    logger.error(message, exc_info=exc_info)


def log_debug(message: str, logger: Optional[logging.Logger] = None):
    """记录 DEBUG 级别日志"""
    if logger is None:
        logger = logging.getLogger()
    logger.debug(message)


# 初始化默认配置（导入时自动执行）
_initialized = False

def init_default_logging():
    """初始化默认日志配置（仅执行一次）"""
    global _initialized
    if not _initialized:
        setup_logging(
            level=os.getenv('APT_LOG_LEVEL', 'INFO'),
            log_file=os.getenv('APT_LOG_FILE'),
            enable_color=os.getenv('APT_LOG_COLOR', 'true').lower() == 'true'
        )
        _initialized = True


# 自动初始化
init_default_logging()


# 用于兼容性的别名（保持与 infrastructure.logging 的接口兼容）
def info_print(message: str):
    """兼容接口：用于用户友好的信息输出

    这个函数会同时调用 print() 和 logging，
    确保用户看到输出的同时也保存到日志。
    """
    print(message)
    logging.getLogger('apt_model').info(message.strip())


def error_print(message: str):
    """兼容接口：用于错误信息输出"""
    print(message, file=sys.stderr)
    logging.getLogger('apt_model').error(message.strip())


def warning_print(message: str):
    """兼容接口：用于警告信息输出"""
    print(message)
    logging.getLogger('apt_model').warning(message.strip())


# 示例使用
if __name__ == "__main__":
    # 演示不同日志级别
    logger = get_logger(__name__)

    logger.debug("这是调试信息")
    logger.info("这是一般信息")
    logger.warning("这是警告信息")
    logger.error("这是错误信息")
    logger.critical("这是严重错误信息")

    # 演示便捷函数
    log_info("使用便捷函数记录日志")
    log_warning("这是一个警告")

    # 演示异常记录
    try:
        1 / 0
    except Exception as e:
        log_error("捕获到异常", exc_info=True)
