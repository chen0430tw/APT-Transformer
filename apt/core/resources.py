#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
核心资源管理模块 - 资源监控和缓存管理

整合功能：
- 资源监控（CPU、内存、GPU）
- 缓存管理
- 资源统计
- 资源清理

整合自：
- apt_model/utils/resource_monitor.py
- apt_model/utils/cache_manager.py（部分）
"""

import os
import time
import shutil
import logging
from typing import Optional, Dict, Any, List
from collections import defaultdict
from pathlib import Path

from apt_model.utils.fake_torch import get_torch
torch = get_torch()

# 可选依赖
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ============================================================================
# 资源监控
# ============================================================================

class ResourceMonitor:
    """
    资源使用监控器

    跟踪CPU、内存和GPU的使用情况
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        log_interval: int = 10
    ):
        """
        初始化资源监控器

        参数:
            logger: 日志记录器
            log_interval: 日志记录间隔（秒）
        """
        self.logger = logger or logging.getLogger(__name__)
        self.log_interval = log_interval
        self.last_log_time = 0
        self.running = False

        self.stats_history = {
            'cpu': [],
            'memory': [],
            'gpu': []
        }

        # 检查GPU是否可用
        self.has_gpu = torch.cuda.is_available()
        if self.has_gpu:
            self.num_gpus = torch.cuda.device_count()

    def start(self) -> None:
        """启动监控"""
        self.running = True
        self.last_log_time = time.time()
        self.logger.info("Resource monitoring started")

    def stop(self) -> None:
        """停止监控并打印摘要"""
        self.running = False
        self.logger.info("Resource monitoring stopped")
        self._print_summary()

    def check_resources(self, force_log: bool = False) -> Optional[Dict[str, Any]]:
        """
        检查资源使用情况

        参数:
            force_log: 是否强制记录日志

        返回:
            dict: 资源统计信息
        """
        if not self.running:
            return None

        current_time = time.time()
        stats = {}

        try:
            # 获取CPU和内存使用情况
            if HAS_PSUTIL:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()

                stats['cpu'] = cpu_percent
                stats['memory_used_gb'] = memory.used / (1024**3)
                stats['memory_total_gb'] = memory.total / (1024**3)
                stats['memory_percent'] = memory.percent

                self.stats_history['cpu'].append(cpu_percent)
                self.stats_history['memory'].append(memory.percent)

            # 获取GPU使用情况
            if self.has_gpu:
                gpu_stats = []
                for i in range(self.num_gpus):
                    gpu_memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    gpu_memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)

                    gpu_stats.append({
                        'device': i,
                        'memory_allocated_gb': gpu_memory_allocated,
                        'memory_reserved_gb': gpu_memory_reserved,
                    })

                stats['gpu'] = gpu_stats
                self.stats_history['gpu'].append(gpu_stats)

            # 记录日志
            if force_log or (current_time - self.last_log_time >= self.log_interval):
                self._log_stats(stats)
                self.last_log_time = current_time

            return stats

        except Exception as e:
            self.logger.error(f"Error checking resources: {e}")
            return None

    def _log_stats(self, stats: Dict[str, Any]) -> None:
        """记录资源统计信息"""
        if 'cpu' in stats:
            self.logger.info(
                f"CPU: {stats['cpu']:.1f}%, "
                f"Memory: {stats['memory_used_gb']:.1f}/{stats['memory_total_gb']:.1f}GB "
                f"({stats['memory_percent']:.1f}%)"
            )

        if 'gpu' in stats:
            for gpu in stats['gpu']:
                self.logger.info(
                    f"GPU {gpu['device']}: "
                    f"Allocated: {gpu['memory_allocated_gb']:.2f}GB, "
                    f"Reserved: {gpu['memory_reserved_gb']:.2f}GB"
                )

    def _print_summary(self) -> None:
        """打印资源使用摘要"""
        if self.stats_history['cpu']:
            avg_cpu = sum(self.stats_history['cpu']) / len(self.stats_history['cpu'])
            max_cpu = max(self.stats_history['cpu'])
            self.logger.info(f"CPU Usage - Avg: {avg_cpu:.1f}%, Max: {max_cpu:.1f}%")

        if self.stats_history['memory']:
            avg_mem = sum(self.stats_history['memory']) / len(self.stats_history['memory'])
            max_mem = max(self.stats_history['memory'])
            self.logger.info(f"Memory Usage - Avg: {avg_mem:.1f}%, Max: {max_mem:.1f}%")

    def get_current_stats(self) -> Dict[str, Any]:
        """
        获取当前资源使用情况（不记录历史）

        返回:
            dict: 当前资源统计
        """
        stats = {}

        if HAS_PSUTIL:
            stats['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            stats['memory'] = {
                'used_gb': memory.used / (1024**3),
                'total_gb': memory.total / (1024**3),
                'percent': memory.percent,
            }

        if self.has_gpu:
            gpu_stats = []
            for i in range(self.num_gpus):
                gpu_stats.append({
                    'device': i,
                    'allocated_gb': torch.cuda.memory_allocated(i) / (1024**3),
                    'reserved_gb': torch.cuda.memory_reserved(i) / (1024**3),
                    'max_allocated_gb': torch.cuda.max_memory_allocated(i) / (1024**3),
                })
            stats['gpu'] = gpu_stats

        return stats


# ============================================================================
# 缓存管理
# ============================================================================

class CacheManager:
    """
    缓存管理器

    管理各种缓存文件和目录
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化缓存管理器

        参数:
            cache_dir: 缓存目录路径（默认: ~/.apt_cache）
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)

        # 设置缓存目录
        if cache_dir is None:
            self.cache_dir = os.path.expanduser("~/.apt_cache")
        else:
            self.cache_dir = os.path.abspath(cache_dir)

        os.makedirs(self.cache_dir, exist_ok=True)

        # 定义子目录
        project_root = Path(__file__).parent.parent.parent
        self.subdirs = {
            "models": os.path.join(self.cache_dir, "models"),
            "datasets": os.path.join(self.cache_dir, "datasets"),
            "tokenizers": os.path.join(self.cache_dir, "tokenizers"),
            "checkpoints": os.path.join(self.cache_dir, "checkpoints"),
            "logs": os.path.join(self.cache_dir, "logs"),
            "visualizations": os.path.join(project_root, "report"),
            "temp": os.path.join(self.cache_dir, "temp"),
        }

        # 创建子目录
        for subdir in self.subdirs.values():
            os.makedirs(subdir, exist_ok=True)

        self.logger.debug(f"Cache manager initialized: {self.cache_dir}")

    def get_cache_path(self, cache_type: str, filename: str = "") -> str:
        """
        获取缓存路径

        参数:
            cache_type: 缓存类型（models, datasets等）
            filename: 文件名（可选）

        返回:
            str: 缓存路径
        """
        if cache_type not in self.subdirs:
            raise ValueError(f"Unknown cache type: {cache_type}")

        base_path = self.subdirs[cache_type]
        if filename:
            return os.path.join(base_path, filename)
        return base_path

    def clean_cache(
        self,
        cache_type: Optional[str] = None,
        days: int = 30,
        exclude: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        清理缓存文件

        参数:
            cache_type: 缓存类型（None表示清理所有）
            days: 清理多少天前的文件（0表示全部）
            exclude: 排除的文件/目录模式列表

        返回:
            dict: 清理结果
        """
        result = {
            'cleaned_files': 0,
            'cleaned_dirs': 0,
            'errors': [],
            'skipped': 0
        }

        exclude = exclude or []

        # 确定要清理的目录
        if cache_type is None:
            dirs_to_clean = list(self.subdirs.values())
        elif cache_type in self.subdirs:
            dirs_to_clean = [self.subdirs[cache_type]]
        else:
            result['errors'].append(f"Unknown cache type: {cache_type}")
            return result

        cutoff_time = time.time() - (days * 86400) if days > 0 else float('inf')

        for directory in dirs_to_clean:
            if not os.path.exists(directory):
                continue

            try:
                for root, dirs, files in os.walk(directory, topdown=False):
                    # 清理文件
                    for file in files:
                        file_path = os.path.join(root, file)

                        # 检查是否在排除列表中
                        if any(pattern in file_path for pattern in exclude):
                            result['skipped'] += 1
                            continue

                        # 检查文件年龄
                        if os.path.getmtime(file_path) < cutoff_time:
                            try:
                                os.remove(file_path)
                                result['cleaned_files'] += 1
                            except Exception as e:
                                result['errors'].append(f"Error removing {file_path}: {e}")

                    # 清理空目录
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        if not os.listdir(dir_path):
                            try:
                                os.rmdir(dir_path)
                                result['cleaned_dirs'] += 1
                            except Exception as e:
                                result['errors'].append(f"Error removing {dir_path}: {e}")

            except Exception as e:
                result['errors'].append(f"Error cleaning {directory}: {e}")

        self.logger.info(
            f"Cache cleanup complete: {result['cleaned_files']} files, "
            f"{result['cleaned_dirs']} directories removed"
        )

        return result

    def get_cache_size(self, cache_type: Optional[str] = None) -> Dict[str, float]:
        """
        获取缓存大小

        参数:
            cache_type: 缓存类型（None表示所有）

        返回:
            dict: 缓存大小（GB）
        """
        sizes = {}

        if cache_type is None:
            dirs_to_check = self.subdirs.items()
        elif cache_type in self.subdirs:
            dirs_to_check = [(cache_type, self.subdirs[cache_type])]
        else:
            return {}

        for name, directory in dirs_to_check:
            total_size = 0
            try:
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            total_size += os.path.getsize(file_path)
                        except Exception:
                            pass
                sizes[name] = total_size / (1024**3)  # Convert to GB
            except Exception as e:
                self.logger.error(f"Error calculating size for {name}: {e}")
                sizes[name] = 0.0

        return sizes

    def clear_cache_type(self, cache_type: str) -> bool:
        """
        清空特定类型的缓存

        参数:
            cache_type: 缓存类型

        返回:
            bool: 是否成功
        """
        if cache_type not in self.subdirs:
            self.logger.error(f"Unknown cache type: {cache_type}")
            return False

        directory = self.subdirs[cache_type]
        try:
            shutil.rmtree(directory)
            os.makedirs(directory, exist_ok=True)
            self.logger.info(f"Cache cleared: {cache_type}")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing cache {cache_type}: {e}")
            return False


# ============================================================================
# 公共API
# ============================================================================

__all__ = [
    'ResourceMonitor',
    'CacheManager',
]
