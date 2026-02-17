#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基础设施 - 错误处理和恢复

整合功能：
- 错误捕获和记录
- 自动恢复机制
- 错误统计
- 内存清理

整合自：
- apt_model/utils/error_handler.py
"""

import traceback
import logging
import time
from typing import Optional, Dict, Any, Callable
from functools import wraps


# ============================================================================
# 内存清理
# ============================================================================

def memory_cleanup() -> None:
    """清理内存，包括Python垃圾回收和CUDA缓存"""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ============================================================================
# 错误处理器类
# ============================================================================

class ErrorHandler:
    """
    增强的错误处理器

    提供详细的错误信息和自动恢复功能
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        checkpoint_manager: Any = None,
        max_recovery_attempts: int = 3
    ):
        """
        初始化错误处理器

        参数:
            logger: 日志记录器
            checkpoint_manager: 检查点管理器（用于自动保存）
            max_recovery_attempts: 最大恢复尝试次数
        """
        self.logger = logger or logging.getLogger(__name__)
        self.checkpoint_manager = checkpoint_manager
        self.max_recovery_attempts = max_recovery_attempts

        self.error_counts: Dict[str, int] = {}
        self.recovery_handlers: Dict[str, Callable] = {}

        # 注册默认恢复处理器
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """注册默认的错误恢复处理器"""
        # 内存相关错误
        self.register_handler("CUDA out of memory", self._handle_memory_error)
        self.register_handler("MemoryError", self._handle_memory_error)

        # 连接相关错误
        self.register_handler("ConnectionError", self._handle_temporary_error)
        self.register_handler("TimeoutError", self._handle_temporary_error)
        self.register_handler("IOError", self._handle_temporary_error)

    def register_handler(self, error_pattern: str, handler: Callable) -> None:
        """
        注册自定义错误恢复处理器

        参数:
            error_pattern: 错误消息模式
            handler: 处理器函数
        """
        self.recovery_handlers[error_pattern] = handler

    def handle_exception(
        self,
        exception: Exception,
        context: str = ""
    ) -> bool:
        """
        处理异常，记录日志并尝试恢复

        参数:
            exception: 异常对象
            context: 上下文信息

        返回:
            bool: 是否应该继续执行（恢复成功）
        """
        error_type = type(exception).__name__
        error_msg = str(exception)

        # 跟踪错误发生次数
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1

        # 记录错误
        self.logger.error(f"{context} Error: {error_type}: {error_msg}")
        self.logger.error(f"Detailed traceback:\n{traceback.format_exc()}")

        # 尝试恢复
        should_continue = self._attempt_recovery(error_type, error_msg)

        # 检查错误次数是否过多
        if self.error_counts[error_type] > self.max_recovery_attempts:
            self.logger.warning(
                f"{error_type} error occurred {self.error_counts[error_type]} times, "
                f"exceeding maximum recovery attempts"
            )
            return False

        return should_continue

    def _attempt_recovery(self, error_type: str, error_msg: str) -> bool:
        """
        尝试从错误中恢复

        参数:
            error_type: 错误类型
            error_msg: 错误消息

        返回:
            bool: 恢复是否成功
        """
        # 检查是否有匹配的恢复处理器
        for pattern, handler in self.recovery_handlers.items():
            if pattern in error_msg or pattern == error_type:
                self.logger.info(f"Attempting recovery using handler for: {pattern}")
                try:
                    return handler(error_type, error_msg)
                except Exception as e:
                    self.logger.error(f"Recovery handler failed: {e}")
                    return False

        # 没有找到合适的处理器
        self.logger.warning(f"No recovery handler found for: {error_type}")
        return False

    def _handle_memory_error(self, error_type: str, error_msg: str) -> bool:
        """处理内存错误"""
        self.logger.info("Handling memory error: cleaning up memory")
        memory_cleanup()
        time.sleep(1)  # 等待内存清理完成
        return True  # 尝试继续

    def _handle_temporary_error(self, error_type: str, error_msg: str) -> bool:
        """处理临时错误（连接、超时等）"""
        self.logger.info(f"Handling temporary error: {error_type}. Retrying...")
        time.sleep(2)  # 等待后重试
        return True

    def get_error_summary(self) -> Dict[str, int]:
        """
        获取错误统计摘要

        返回:
            dict: 错误类型和发生次数的字典
        """
        return self.error_counts.copy()

    def reset_error_counts(self) -> None:
        """重置错误计数"""
        self.error_counts.clear()


# ============================================================================
# 装饰器
# ============================================================================

def with_error_handling(
    logger: Optional[logging.Logger] = None,
    retry_on_error: bool = True,
    max_retries: int = 3,
    cleanup_on_error: bool = True
):
    """
    错误处理装饰器

    参数:
        logger: 日志记录器
        retry_on_error: 是否在错误时重试
        max_retries: 最大重试次数
        cleanup_on_error: 错误时是否清理内存

    使用示例:
        @with_error_handling(logger=my_logger, retry_on_error=True)
        def my_function():
            # 可能抛出错误的代码
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or logging.getLogger(__name__)
            attempt = 0

            while attempt < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    _logger.error(
                        f"Error in {func.__name__} (attempt {attempt}/{max_retries}): {e}"
                    )
                    _logger.error(traceback.format_exc())

                    if cleanup_on_error:
                        memory_cleanup()

                    if not retry_on_error or attempt >= max_retries:
                        raise

                    # 指数退避
                    wait_time = 2 ** attempt
                    _logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

            raise RuntimeError(f"Function {func.__name__} failed after {max_retries} attempts")

        return wrapper
    return decorator


def safe_execute(func: Callable, *args, **kwargs) -> tuple:
    """
    安全执行函数，捕获所有异常

    参数:
        func: 要执行的函数
        *args: 位置参数
        **kwargs: 关键字参数

    返回:
        tuple: (success: bool, result: Any, error: Optional[Exception])
    """
    try:
        result = func(*args, **kwargs)
        return True, result, None
    except Exception as e:
        return False, None, e


# ============================================================================
# 上下文管理器
# ============================================================================

class ErrorContext:
    """
    错误处理上下文管理器

    使用示例:
        with ErrorContext(logger, "Training step"):
            # 可能抛出错误的代码
            pass
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        context: str = "",
        suppress: bool = False,
        cleanup_on_error: bool = True
    ):
        """
        初始化错误上下文

        参数:
            logger: 日志记录器
            context: 上下文描述
            suppress: 是否抑制错误（不重新抛出）
            cleanup_on_error: 错误时是否清理内存
        """
        self.logger = logger or logging.getLogger(__name__)
        self.context = context
        self.suppress = suppress
        self.cleanup_on_error = cleanup_on_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.logger.error(f"Error in {self.context}: {exc_type.__name__}: {exc_val}")
            self.logger.error(traceback.format_exc())

            if self.cleanup_on_error:
                memory_cleanup()

            return self.suppress  # 如果suppress=True，抑制异常
        return False


# ============================================================================
# 公共API
# ============================================================================

__all__ = [
    # 错误处理类
    'ErrorHandler',
    # 工具函数
    'memory_cleanup',
    'safe_execute',
    # 装饰器
    'with_error_handling',
    # 上下文管理器
    'ErrorContext',
]
