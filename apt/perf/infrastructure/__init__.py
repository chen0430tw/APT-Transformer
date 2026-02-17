#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基础设施模块

提供日志、错误处理、监控等基础设施功能
"""

from .logging import (
    setup_logging,
    setup_colored_logging,
    get_progress_logger,
    LogManager,
    ColoredFormatter,
)

from .errors import (
    ErrorHandler,
    memory_cleanup,
    safe_execute,
    with_error_handling,
    ErrorContext,
)

__all__ = [
    # 日志
    'setup_logging',
    'setup_colored_logging',
    'get_progress_logger',
    'LogManager',
    'ColoredFormatter',
    # 错误处理
    'ErrorHandler',
    'memory_cleanup',
    'safe_execute',
    'with_error_handling',
    'ErrorContext',
]
