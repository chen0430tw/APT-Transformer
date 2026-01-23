#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Resource monitoring utilities for the APT Model training system.
Tracks CPU, memory, and GPU utilization during training.
"""

import time
from typing import Optional, Dict, Any, List
import logging
from collections import defaultdict


class ResourceMonitor:
    """
    Resource usage monitoring class for tracking CPU, memory, and GPU usage.
    """
    def __init__(self, logger: Optional[logging.Logger] = None, log_interval: int = 10):
        """
        Initialize resource monitor.
        
        Args:
            logger: Logger for recording resource usage
            log_interval: Logging interval in seconds
        """
        self.logger = logger
        self.log_interval = log_interval
        self.last_log_time = 0
        self.running = False
        self.stats_history = {
            'cpu': [],
            'memory': [],
            'gpu': []
        }
        
        # Check if GPU is available
        self.has_gpu = False
        try:
            import torch
            self.has_gpu = torch.cuda.is_available()
            if self.has_gpu:
                self.num_gpus = torch.cuda.device_count()
        except (ImportError, AttributeError):
            pass
        
        # Check for optional dependencies
        self.has_psutil = False
        try:
            import psutil
            self.has_psutil = True
        except ImportError:
            if self.logger:
                self.logger.warning("psutil not installed; CPU and memory monitoring will be limited")
            else:
                print("Warning: psutil not installed; CPU and memory monitoring will be limited")
    
    def start(self) -> None:
        """Start monitoring resources."""
        self.running = True
        self.last_log_time = time.time()
        if self.logger:
            self.logger.info("Resource monitoring started")
    
    def stop(self) -> None:
        """Stop monitoring and print summary statistics."""
        self.running = False
        if self.logger:
            self.logger.info("Resource monitoring stopped")
            
        # Print resource usage summary
        self._print_stats_summary()
    
    def check_resources(self, force_log: bool = False) -> Optional[Dict[str, Any]]:
        """
        Check resource usage (CPU, memory, GPU).
        
        Args:
            force_log: Whether to force logging regardless of interval
            
        Returns:
            Dictionary containing resource statistics
        """
        if not self.running:
            return None
        
        current_time = time.time()
        stats = {}
        
        try:
            # Get CPU usage
            if self.has_psutil:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=0.1)
                stats['cpu'] = cpu_percent
                self.stats_history['cpu'].append(cpu_percent)
                
                # Get memory usage
                memory = psutil.virtual_memory()
                memory_used_gb = memory.used / (1024 ** 3)
                memory_total_gb = memory.total / (1024 ** 3)
                memory_percent = memory.percent
                stats['memory'] = {
                    'used_gb': memory_used_gb,
                    'total_gb': memory_total_gb,
                    'percent': memory_percent
                }
                self.stats_history['memory'].append(memory_percent)
            
            # Get GPU usage
            gpu_stats = []
            if self.has_gpu:
                try:
                    import torch
                    for i in range(self.num_gpus):
                        gpu_used_gb = torch.cuda.memory_allocated(i) / (1024 ** 3)
                        gpu_total_gb = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                        gpu_percent = (gpu_used_gb / gpu_total_gb) * 100
                        gpu_stats.append({
                            'id': i,
                            'used_gb': gpu_used_gb,
                            'total_gb': gpu_total_gb,
                            'percent': gpu_percent
                        })
                        self.stats_history['gpu'].append(gpu_percent)
                    stats['gpu'] = gpu_stats
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Error getting GPU information: {e}")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error getting resource information: {e}")
        
        # Log resource usage
        if force_log or (current_time - self.last_log_time >= self.log_interval):
            self._log_resources(stats)
            self.last_log_time = current_time
        
        return stats
    
    def _log_resources(self, stats: Dict[str, Any]) -> None:
        """
        Log resource usage information.
        
        Args:
            stats: Dictionary containing resource statistics
        """
        if not self.logger:
            return
        
        if 'cpu' in stats:
            self.logger.info(f"CPU usage: {stats['cpu']}%")
        
        if 'memory' in stats:
            mem = stats['memory']
            self.logger.info(f"Memory usage: {mem['used_gb']:.1f}/{mem['total_gb']:.1f}GB ({mem['percent']}%)")
        
        if 'gpu' in stats:
            for gpu in stats['gpu']:
                self.logger.info(f"GPU {gpu['id']} usage: {gpu['used_gb']:.1f}/{gpu['total_gb']:.1f}GB ({gpu['percent']:.1f}%)")
    
    def _print_stats_summary(self) -> None:
        """Print a summary of resource usage statistics."""
        if not self.stats_history['cpu'] and not self.stats_history['gpu']:
            return
        
        summary = "Resource Usage Statistics:\n"
        
        # CPU statistics
        if self.stats_history['cpu']:
            cpu_avg = sum(self.stats_history['cpu']) / len(self.stats_history['cpu'])
            cpu_max = max(self.stats_history['cpu'])
            summary += f"- CPU: Avg {cpu_avg:.1f}%, Max {cpu_max:.1f}%\n"
        
        # Memory statistics
        if self.stats_history['memory']:
            memory_avg = sum(self.stats_history['memory']) / len(self.stats_history['memory'])
            memory_max = max(self.stats_history['memory'])
            summary += f"- Memory: Avg {memory_avg:.1f}%, Max {memory_max:.1f}%\n"
        
        # GPU statistics
        if self.has_gpu and self.stats_history['gpu']:
            gpu_data = defaultdict(list)
            
            # Group GPU stats by device ID
            for i, stat in enumerate(self.stats_history['gpu']):
                gpu_id = i % self.num_gpus
                gpu_data[gpu_id].append(stat)
            
            for gpu_id, stats in gpu_data.items():
                gpu_avg = sum(stats) / len(stats)
                gpu_max = max(stats)
                summary += f"- GPU {gpu_id}: Avg {gpu_avg:.1f}%, Max {gpu_max:.1f}%\n"
        
        if self.logger:
            self.logger.info(summary)
        else:
            print(summary)
    
    def get_peak_memory_usage(self) -> Dict[str, float]:
        """
        Get peak memory usage statistics.
        
        Returns:
            Dictionary with peak memory usage for CPU and each GPU
        """
        result = {}
        
        # CPU memory peak
        if self.stats_history['memory']:
            result['cpu_memory_percent'] = max(self.stats_history['memory'])
        
        # GPU memory peak
        if self.has_gpu and self.stats_history['gpu']:
            gpu_data = defaultdict(list)
            
            # Group GPU stats by device ID
            for i, stat in enumerate(self.stats_history['gpu']):
                gpu_id = i % self.num_gpus
                gpu_data[gpu_id].append(stat)
            
            for gpu_id, stats in gpu_data.items():
                result[f'gpu{gpu_id}_memory_percent'] = max(stats)
        
        return result

    def get_utilization_history(self) -> Dict[str, List[float]]:
        """
        Get the resource utilization history for plotting.
        
        Returns:
            Dictionary with utilization history for CPU, memory, and GPUs
        """
        return self.stats_history.copy()


# Function to easily create and start a resource monitor
def create_resource_monitor(logger: Optional[logging.Logger] = None, 
                           log_interval: int = 10) -> ResourceMonitor:
    """
    Create and start a resource monitor.
    
    Args:
        logger: Logger for recording resource usage
        log_interval: Logging interval in seconds
        
    Returns:
        Started ResourceMonitor instance
    """
    monitor = ResourceMonitor(logger=logger, log_interval=log_interval)
    monitor.start()
    return monitor