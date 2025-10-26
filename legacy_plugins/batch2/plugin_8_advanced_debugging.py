"""
Advanced Debugging Plugin for APT
高级调试插件 - 提供全方位的训练调试和诊断工具
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Callable
import time
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import traceback
import sys
from contextlib import contextmanager


class AdvancedDebuggingPlugin:
    """
    高级调试插件
    
    提供功能:
    1. 梯度监控和分析
    2. 激活值统计
    3. 内存使用追踪
    4. 性能分析 (Profiling)
    5. 异常检测和诊断
    6. 训练过程可视化
    7. 参数变化追踪
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.name = "advanced-debugging"
        self.version = "1.0.0"
        self.config = config
        
        # 调试模式
        self.debug_level = config.get('debug_level', 'normal')  # minimal/normal/verbose
        self.log_interval = config.get('log_interval', 100)
        
        # 监控选项
        self.monitor_gradients = config.get('monitor_gradients', True)
        self.monitor_activations = config.get('monitor_activations', True)
        self.monitor_memory = config.get('monitor_memory', True)
        self.monitor_performance = config.get('monitor_performance', True)
        
        # 存储路径
        self.output_dir = config.get('output_dir', './debug_output')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 监控数据
        self.gradient_stats = defaultdict(list)
        self.activation_stats = defaultdict(list)
        self.memory_stats = []
        self.performance_stats = []
        self.parameter_history = defaultdict(list)
        
        # 异常记录
        self.anomalies = []
        
        # 钩子句柄
        self.hooks = []
        
        print(f"🐛 高级调试插件初始化完成 (调试级别: {self.debug_level})")
    
    # ==================== 梯度监控 ====================
    
    def register_gradient_hooks(self, model: nn.Module):
        """注册梯度监控钩子"""
        if not self.monitor_gradients:
            return
        
        print("📊 注册梯度监控钩子...")
        
        def gradient_hook(module, grad_input, grad_output):
            """梯度钩子函数"""
            module_name = self._get_module_name(model, module)
            
            if grad_output[0] is not None:
                grad = grad_output[0]
                
                stats = {
                    'mean': grad.abs().mean().item(),
                    'std': grad.std().item(),
                    'max': grad.abs().max().item(),
                    'min': grad.abs().min().item(),
                    'norm': grad.norm().item(),
                }
                
                self.gradient_stats[module_name].append(stats)
                
                # 检测梯度异常
                self._check_gradient_anomaly(module_name, stats)
        
        # 为所有模块注册钩子
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 只监控叶子模块
                hook = module.register_full_backward_hook(gradient_hook)
                self.hooks.append(hook)
        
        print(f"✅ 已注册 {len(self.hooks)} 个梯度监控钩子")
    
    def _check_gradient_anomaly(self, module_name: str, stats: Dict[str, float]):
        """检测梯度异常"""
        threshold = self.config.get('gradient_threshold', 10.0)
        
        # 检测梯度爆炸
        if stats['max'] > threshold:
            anomaly = {
                'type': 'gradient_explosion',
                'module': module_name,
                'value': stats['max'],
                'threshold': threshold,
                'timestamp': time.time()
            }
            self.anomalies.append(anomaly)
            print(f"⚠️ 梯度爆炸警告: {module_name} (max={stats['max']:.4f})")
        
        # 检测梯度消失
        if stats['mean'] < 1e-7:
            anomaly = {
                'type': 'gradient_vanishing',
                'module': module_name,
                'value': stats['mean'],
                'timestamp': time.time()
            }
            self.anomalies.append(anomaly)
            print(f"⚠️ 梯度消失警告: {module_name} (mean={stats['mean']:.2e})")
    
    def get_gradient_report(self) -> Dict[str, Any]:
        """生成梯度分析报告"""
        report = {}
        
        for module_name, stats_list in self.gradient_stats.items():
            if not stats_list:
                continue
            
            # 计算统计量
            means = [s['mean'] for s in stats_list]
            maxs = [s['max'] for s in stats_list]
            norms = [s['norm'] for s in stats_list]
            
            report[module_name] = {
                'avg_mean': np.mean(means),
                'avg_max': np.mean(maxs),
                'avg_norm': np.mean(norms),
                'trend': 'increasing' if means[-1] > means[0] else 'decreasing',
                'num_samples': len(stats_list)
            }
        
        return report
    
    # ==================== 激活值监控 ====================
    
    def register_activation_hooks(self, model: nn.Module):
        """注册激活值监控钩子"""
        if not self.monitor_activations:
            return
        
        print("📊 注册激活值监控钩子...")
        
        def activation_hook(module, input, output):
            """激活值钩子函数"""
            module_name = self._get_module_name(model, module)
            
            if isinstance(output, torch.Tensor):
                stats = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'max': output.max().item(),
                    'min': output.min().item(),
                    'sparsity': (output == 0).float().mean().item(),
                }
                
                self.activation_stats[module_name].append(stats)
                
                # 检测激活异常
                self._check_activation_anomaly(module_name, stats)
        
        # 为所有模块注册钩子
        hook_count = 0
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 只监控叶子模块
                hook = module.register_forward_hook(activation_hook)
                self.hooks.append(hook)
                hook_count += 1
        
        print(f"✅ 已注册 {hook_count} 个激活值监控钩子")
    
    def _check_activation_anomaly(self, module_name: str, stats: Dict[str, float]):
        """检测激活值异常"""
        # 检测死神经元 (过高的稀疏度)
        if stats['sparsity'] > 0.9:
            anomaly = {
                'type': 'dead_neurons',
                'module': module_name,
                'sparsity': stats['sparsity'],
                'timestamp': time.time()
            }
            self.anomalies.append(anomaly)
            
            if self.debug_level == 'verbose':
                print(f"⚠️ 死神经元警告: {module_name} (稀疏度={stats['sparsity']:.2%})")
        
        # 检测激活值饱和
        if abs(stats['mean']) > 10:
            anomaly = {
                'type': 'activation_saturation',
                'module': module_name,
                'mean': stats['mean'],
                'timestamp': time.time()
            }
            self.anomalies.append(anomaly)
            
            if self.debug_level == 'verbose':
                print(f"⚠️ 激活饱和警告: {module_name} (mean={stats['mean']:.2f})")
    
    # ==================== 内存监控 ====================
    
    def track_memory(self, step: int):
        """追踪内存使用"""
        if not self.monitor_memory or not torch.cuda.is_available():
            return
        
        memory_stats = {
            'step': step,
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved() / 1024**3,  # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
            'timestamp': time.time()
        }
        
        self.memory_stats.append(memory_stats)
        
        # 检测内存泄漏
        if len(self.memory_stats) > 10:
            recent_allocated = [s['allocated'] for s in self.memory_stats[-10:]]
            if recent_allocated[-1] > recent_allocated[0] * 1.5:
                print(f"⚠️ 可能的内存泄漏: {recent_allocated[0]:.2f}GB -> {recent_allocated[-1]:.2f}GB")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """生成内存使用报告"""
        if not self.memory_stats:
            return {}
        
        allocated = [s['allocated'] for s in self.memory_stats]
        
        return {
            'peak_memory_gb': max(allocated),
            'avg_memory_gb': np.mean(allocated),
            'current_memory_gb': allocated[-1],
            'memory_trend': 'increasing' if allocated[-1] > allocated[0] else 'stable',
        }
    
    # ==================== 性能分析 ====================
    
    @contextmanager
    def profile_section(self, name: str):
        """性能分析上下文管理器"""
        if not self.monitor_performance:
            yield
            return
        
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            stats = {
                'name': name,
                'duration': end_time - start_time,
                'memory_delta': (end_memory - start_memory) / 1024**2,  # MB
                'timestamp': time.time()
            }
            
            self.performance_stats.append(stats)
            
            if self.debug_level == 'verbose':
                print(f"⏱️ {name}: {stats['duration']:.3f}s, "
                      f"Memory: {stats['memory_delta']:+.1f}MB")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """生成性能分析报告"""
        if not self.performance_stats:
            return {}
        
        # 按名称分组
        grouped = defaultdict(list)
        for stat in self.performance_stats:
            grouped[stat['name']].append(stat)
        
        report = {}
        for name, stats in grouped.items():
            durations = [s['duration'] for s in stats]
            report[name] = {
                'total_time': sum(durations),
                'avg_time': np.mean(durations),
                'min_time': min(durations),
                'max_time': max(durations),
                'call_count': len(durations)
            }
        
        return report
    
    # ==================== 参数追踪 ====================
    
    def track_parameters(self, model: nn.Module, step: int):
        """追踪模型参数变化"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                stats = {
                    'step': step,
                    'mean': param.data.mean().item(),
                    'std': param.data.std().item(),
                    'norm': param.data.norm().item(),
                    'grad_norm': param.grad.norm().item() if param.grad is not None else 0,
                }
                
                self.parameter_history[name].append(stats)
    
    def get_parameter_report(self) -> Dict[str, Any]:
        """生成参数变化报告"""
        report = {}
        
        for param_name, history in self.parameter_history.items():
            if len(history) < 2:
                continue
            
            norms = [h['norm'] for h in history]
            grad_norms = [h['grad_norm'] for h in history]
            
            report[param_name] = {
                'initial_norm': norms[0],
                'final_norm': norms[-1],
                'norm_change': norms[-1] - norms[0],
                'avg_grad_norm': np.mean(grad_norms),
                'max_grad_norm': max(grad_norms),
            }
        
        return report
    
    # ==================== 异常诊断 ====================
    
    def diagnose_training(self, loss_history: List[float]) -> Dict[str, Any]:
        """诊断训练问题"""
        diagnosis = {
            'status': 'healthy',
            'issues': [],
            'recommendations': []
        }
        
        if not loss_history:
            return diagnosis
        
        # 检查损失是否为NaN或Inf
        if any(np.isnan(loss) or np.isinf(loss) for loss in loss_history):
            diagnosis['status'] = 'critical'
            diagnosis['issues'].append('Loss contains NaN or Inf values')
            diagnosis['recommendations'].append('Reduce learning rate or check data quality')
        
        # 检查损失是否不下降
        if len(loss_history) > 100:
            recent_losses = loss_history[-100:]
            if max(recent_losses) - min(recent_losses) < 0.01:
                diagnosis['status'] = 'warning'
                diagnosis['issues'].append('Loss not decreasing')
                diagnosis['recommendations'].append('Increase learning rate or check model capacity')
        
        # 检查损失是否震荡
        if len(loss_history) > 10:
            recent_losses = loss_history[-10:]
            variance = np.var(recent_losses)
            if variance > np.mean(recent_losses):
                diagnosis['status'] = 'warning'
                diagnosis['issues'].append('Loss is oscillating')
                diagnosis['recommendations'].append('Reduce learning rate or use gradient clipping')
        
        # 检查梯度问题
        gradient_report = self.get_gradient_report()
        for module_name, stats in gradient_report.items():
            if stats['avg_max'] > 10:
                diagnosis['issues'].append(f'Gradient explosion in {module_name}')
                diagnosis['recommendations'].append('Use gradient clipping')
            elif stats['avg_mean'] < 1e-7:
                diagnosis['issues'].append(f'Gradient vanishing in {module_name}')
                diagnosis['recommendations'].append('Use residual connections or different activation')
        
        return diagnosis
    
    # ==================== 可视化 ====================
    
    def visualize_gradients(self, save_path: Optional[str] = None):
        """可视化梯度统计"""
        if not self.gradient_stats:
            print("⚠️ 没有梯度数据可视化")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Gradient Statistics', fontsize=16)
        
        # 选择几个代表性的层
        sample_modules = list(self.gradient_stats.keys())[:4]
        
        for idx, module_name in enumerate(sample_modules):
            if idx >= 4:
                break
            
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            
            stats_list = self.gradient_stats[module_name]
            means = [s['mean'] for s in stats_list]
            
            ax.plot(means, label='Mean Gradient')
            ax.set_title(f'{module_name}')
            ax.set_xlabel('Step')
            ax.set_ylabel('Gradient Magnitude')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"✅ 梯度可视化已保存到: {save_path}")
        else:
            plt.savefig(os.path.join(self.output_dir, 'gradients.png'))
            print(f"✅ 梯度可视化已保存到: {self.output_dir}/gradients.png")
        
        plt.close()
    
    def visualize_memory(self, save_path: Optional[str] = None):
        """可视化内存使用"""
        if not self.memory_stats:
            print("⚠️ 没有内存数据可视化")
            return
        
        steps = [s['step'] for s in self.memory_stats]
        allocated = [s['allocated'] for s in self.memory_stats]
        reserved = [s['reserved'] for s in self.memory_stats]
        
        plt.figure(figsize=(12, 6))
        plt.plot(steps, allocated, label='Allocated', marker='o', markersize=3)
        plt.plot(steps, reserved, label='Reserved', marker='s', markersize=3)
        plt.xlabel('Training Step')
        plt.ylabel('Memory (GB)')
        plt.title('GPU Memory Usage')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            print(f"✅ 内存可视化已保存到: {save_path}")
        else:
            plt.savefig(os.path.join(self.output_dir, 'memory.png'))
            print(f"✅ 内存可视化已保存到: {self.output_dir}/memory.png")
        
        plt.close()
    
    # ==================== 报告生成 ====================
    
    def generate_full_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """生成完整的调试报告"""
        print("📋 生成完整调试报告...")
        
        report = {
            'timestamp': time.time(),
            'debug_level': self.debug_level,
            'gradient_report': self.get_gradient_report(),
            'memory_report': self.get_memory_report(),
            'performance_report': self.get_performance_report(),
            'parameter_report': self.get_parameter_report(),
            'anomalies': self.anomalies,
            'anomaly_summary': {
                'total': len(self.anomalies),
                'by_type': {}
            }
        }
        
        # 统计异常类型
        for anomaly in self.anomalies:
            anomaly_type = anomaly['type']
            report['anomaly_summary']['by_type'][anomaly_type] = \
                report['anomaly_summary']['by_type'].get(anomaly_type, 0) + 1
        
        # 保存报告
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'debug_report.json')
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 调试报告已保存到: {save_path}")
        
        # 生成可视化
        self.visualize_gradients()
        if self.memory_stats:
            self.visualize_memory()
        
        return report
    
    # ==================== 辅助方法 ====================
    
    def _get_module_name(self, model: nn.Module, target_module: nn.Module) -> str:
        """获取模块名称"""
        for name, module in model.named_modules():
            if module is target_module:
                return name
        return "unknown"
    
    def cleanup(self):
        """清理钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        print("🧹 已清理所有钩子")
    
    # ==================== 插件钩子 ====================
    
    def on_training_start(self, context: Dict[str, Any]):
        """训练开始时的钩子"""
        model = context.get('model')
        
        if model:
            self.register_gradient_hooks(model)
            self.register_activation_hooks(model)
        
        print(f"🐛 调试插件已启动 (监控: 梯度={self.monitor_gradients}, "
              f"激活={self.monitor_activations}, 内存={self.monitor_memory})")
    
    def on_batch_end(self, context: Dict[str, Any]):
        """批次结束时的钩子"""
        step = context.get('step', 0)
        
        # 追踪内存
        if step % self.log_interval == 0:
            self.track_memory(step)
        
        # 追踪参数
        if self.config.get('track_parameters', False):
            model = context.get('model')
            if model and step % self.log_interval == 0:
                self.track_parameters(model, step)
    
    def on_training_end(self, context: Dict[str, Any]):
        """训练结束时的钩子"""
        print("\n" + "=" * 60)
        print("📊 生成最终调试报告...")
        
        # 生成完整报告
        report = self.generate_full_report()
        
        # 打印摘要
        print("\n📋 调试摘要:")
        print(f"  总异常数: {len(self.anomalies)}")
        
        if self.anomalies:
            print("  异常类型分布:")
            anomaly_types = {}
            for anomaly in self.anomalies:
                anomaly_type = anomaly['type']
                anomaly_types[anomaly_type] = anomaly_types.get(anomaly_type, 0) + 1
            
            for anomaly_type, count in anomaly_types.items():
                print(f"    - {anomaly_type}: {count}")
        
        # 清理
        self.cleanup()
        
        print("=" * 60)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("🐛 高级调试插件 (Advanced Debugging Plugin)")
    print("=" * 60)
    
    # 配置
    config = {
        'debug_level': 'verbose',
        'log_interval': 10,
        'monitor_gradients': True,
        'monitor_activations': True,
        'monitor_memory': True,
        'monitor_performance': True,
        'track_parameters': True,
        'gradient_threshold': 10.0,
        'output_dir': './debug_output'
    }
    
    plugin = AdvancedDebuggingPlugin(config)
    
    print("\n💡 插件功能:")
    print("1. 📊 梯度监控和异常检测")
    print("2. 🔍 激活值统计和分析")
    print("3. 💾 内存使用追踪")
    print("4. ⏱️ 性能分析和profiling")
    print("5. 🚨 异常检测和诊断")
    print("6. 📈 训练过程可视化")
    print("7. 📝 完整的调试报告")
    
    print("\n📝 使用示例:")
    print("""
    # 在训练开始时注册钩子
    plugin.on_training_start({'model': model})
    
    # 在训练循环中
    for step, batch in enumerate(dataloader):
        # 性能分析
        with plugin.profile_section('forward_pass'):
            outputs = model(batch)
        
        with plugin.profile_section('backward_pass'):
            loss.backward()
        
        # 批次结束
        plugin.on_batch_end({'step': step, 'model': model})
    
    # 训练结束时生成报告
    plugin.on_training_end({})
    """)
    
    # 演示诊断功能
    print("\n🔍 演示诊断功能:")
    
    # 模拟一些损失值
    loss_history = [2.5, 2.3, 2.1, float('nan'), 1.8]
    diagnosis = plugin.diagnose_training(loss_history)
    
    print(f"\n诊断结果: {diagnosis['status']}")
    if diagnosis['issues']:
        print("发现的问题:")
        for issue in diagnosis['issues']:
            print(f"  - {issue}")
        print("建议:")
        for rec in diagnosis['recommendations']:
            print(f"  - {rec}")
    
    print("\n" + "=" * 60)
