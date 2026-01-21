#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
梯度监控工具

功能：
1. 检测梯度消失/爆炸
2. 记录梯度范数历史
3. 可视化梯度流
4. 梯度异常检测（NaN, Inf）
5. 🔮 导出JSON数据（供WebUI/API使用）
6. 🔮 支持分布式训练的梯度同步
"""

import torch
import numpy as np
import json
from collections import defaultdict
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # 非GUI backend
import matplotlib.pyplot as plt


class GradientMonitor:
    """梯度监控工具"""

    def __init__(self, model, logger: Optional[logging.Logger] = None,
                 export_dir: Optional[str] = None):
        """
        初始化梯度监控器

        Args:
            model: 要监控的模型
            logger: 日志记录器
            export_dir: 导出目录（用于WebUI数据）
        """
        self.model = model
        self.logger = logger or logging.getLogger(__name__)
        self.export_dir = Path(export_dir) if export_dir else Path(".cache/gradient_monitor")
        self.export_dir.mkdir(parents=True, exist_ok=True)

        # 梯度历史
        self.gradient_history = defaultdict(list)  # {layer_name: [norm1, norm2, ...]}
        self.gradient_norms = []  # [(step, total_norm), ...]

        # 异常统计
        self.anomaly_counts = {
            'nan': 0,
            'inf': 0,
            'vanishing': 0,  # < 1e-7
            'exploding': 0   # > 1e3
        }

        # 🔮 分布式训练支持
        self.distributed = torch.distributed.is_initialized() if hasattr(torch.distributed, 'is_initialized') else False
        self.rank = torch.distributed.get_rank() if self.distributed else 0

    def check_gradient_flow(self) -> Tuple[Dict[str, float], List[str]]:
        """
        检查梯度流，识别梯度消失/爆炸

        Returns:
            gradients: {layer_name: grad_norm}
            issues: 问题列表
        """
        gradients = {}
        issues = []

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradients[name] = grad_norm
                self.gradient_history[name].append(grad_norm)

                # 检测异常
                if grad_norm < 1e-7:
                    issues.append(f"⚠️  梯度消失: {name} (norm={grad_norm:.2e})")
                    self.anomaly_counts['vanishing'] += 1

                elif grad_norm > 1e3:
                    issues.append(f"⚠️  梯度爆炸: {name} (norm={grad_norm:.2e})")
                    self.anomaly_counts['exploding'] += 1

                elif torch.isnan(torch.tensor(grad_norm)):
                    issues.append(f"❌ NaN梯度: {name}")
                    self.anomaly_counts['nan'] += 1

                elif torch.isinf(torch.tensor(grad_norm)):
                    issues.append(f"❌ Inf梯度: {name}")
                    self.anomaly_counts['inf'] += 1

        # 记录问题
        if issues and self.logger:
            for issue in issues:
                self.logger.warning(issue)

        return gradients, issues

    def log_gradient_norms(self, step: int) -> float:
        """
        记录梯度范数

        Args:
            step: 当前步数

        Returns:
            total_norm: 总梯度范数
        """
        total_norm = 0.0
        param_count = 0

        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                param_count += 1

        if param_count > 0:
            total_norm = total_norm ** 0.5
        else:
            total_norm = 0.0

        self.gradient_norms.append((step, total_norm))

        if self.logger:
            self.logger.debug(f"Step {step}: Total gradient norm = {total_norm:.4f}")

        return total_norm

    def detect_gradient_anomalies(self) -> List[str]:
        """
        检测梯度异常（NaN, Inf等）

        Returns:
            anomalies: 异常列表
        """
        anomalies = []

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    anomalies.append(f"NaN in {name}")

                if torch.isinf(param.grad).any():
                    anomalies.append(f"Inf in {name}")

        return anomalies

    def plot_gradient_flow(self, save_path: Optional[str] = None) -> str:
        """
        可视化梯度流

        Args:
            save_path: 保存路径

        Returns:
            实际保存路径
        """
        if save_path is None:
            save_path = self.export_dir / "gradient_flow.png"
        else:
            save_path = Path(save_path)

        fig, ax = plt.subplots(figsize=(15, 6))

        # 提取数据
        layers = []
        avg_grads = []

        for name, grad_list in self.gradient_history.items():
            if len(grad_list) > 0:
                layers.append(name.replace('.', '\n'))  # 换行显示
                avg_grads.append(np.mean(grad_list))

        # 绘制条形图
        colors = ['green' if g > 1e-7 and g < 1e3 else 'red' for g in avg_grads]

        ax.bar(range(len(layers)), avg_grads, alpha=0.7, color=colors)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=90, ha='right', fontsize=8)
        ax.set_ylabel('Average Gradient Norm')
        ax.set_title('Gradient Flow Across Layers')
        ax.set_yscale('log')
        ax.axhline(y=1e-7, color='orange', linestyle='--', label='Vanishing threshold')
        ax.axhline(y=1e3, color='red', linestyle='--', label='Exploding threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        self.logger.info(f"梯度流图已保存: {save_path}")

        return str(save_path)

    def plot_gradient_norms_timeline(self, save_path: Optional[str] = None) -> str:
        """
        绘制梯度范数时间线

        Args:
            save_path: 保存路径

        Returns:
            实际保存路径
        """
        if save_path is None:
            save_path = self.export_dir / "gradient_norms_timeline.png"
        else:
            save_path = Path(save_path)

        if len(self.gradient_norms) == 0:
            self.logger.warning("No gradient norms recorded")
            return ""

        fig, ax = plt.subplots(figsize=(12, 6))

        steps, norms = zip(*self.gradient_norms)

        ax.plot(steps, norms, linewidth=2, color='blue', alpha=0.7)
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Total Gradient Norm')
        ax.set_title('Gradient Norm Over Time')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        self.logger.info(f"梯度范数时间线已保存: {save_path}")

        return str(save_path)

    def get_gradient_stats(self) -> Dict[str, Dict[str, float]]:
        """
        获取梯度统计信息

        Returns:
            stats: {layer_name: {mean, std, min, max}}
        """
        stats = {}

        for name, grad_list in self.gradient_history.items():
            if len(grad_list) > 0:
                stats[name] = {
                    'mean': float(np.mean(grad_list)),
                    'std': float(np.std(grad_list)),
                    'min': float(np.min(grad_list)),
                    'max': float(np.max(grad_list)),
                    'count': len(grad_list)
                }

        return stats

    # =========================================================================
    # 🔮 WebUI/API数据导出接口
    # =========================================================================

    def export_for_webui(self) -> Dict:
        """
        导出数据供WebUI/API使用

        Returns:
            data: {
                'gradient_stats': {...},
                'gradient_timeline': [...],
                'anomaly_counts': {...},
                'latest_issues': [...]
            }

        WebUI可以通过API获取：
        GET /api/training/gradients
        """
        # 导出梯度统计
        gradient_stats = self.get_gradient_stats()

        # 导出梯度时间线
        gradient_timeline = [
            {'step': int(step), 'norm': float(norm)}
            for step, norm in self.gradient_norms
        ]

        # 导出异常计数
        anomaly_counts = self.anomaly_counts.copy()

        # 最近的问题
        _, latest_issues = self.check_gradient_flow()

        data = {
            'gradient_stats': gradient_stats,
            'gradient_timeline': gradient_timeline,
            'anomaly_counts': anomaly_counts,
            'latest_issues': latest_issues,
            'total_monitored_layers': len(self.gradient_history),
            'total_steps': len(self.gradient_norms)
        }

        # 保存为JSON（供WebUI/API读取）
        json_path = self.export_dir / "gradient_data.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"梯度数据已导出: {json_path}")

        return data

    def export_summary_report(self, save_path: Optional[str] = None) -> str:
        """
        导出汇总报告

        Args:
            save_path: 保存路径

        Returns:
            报告路径
        """
        if save_path is None:
            save_path = self.export_dir / "gradient_summary.md"
        else:
            save_path = Path(save_path)

        stats = self.get_gradient_stats()

        # 生成Markdown报告
        report = "# 梯度监控报告\n\n"

        # 异常统计
        report += "## 异常统计\n\n"
        report += f"- NaN梯度: {self.anomaly_counts['nan']}\n"
        report += f"- Inf梯度: {self.anomaly_counts['inf']}\n"
        report += f"- 梯度消失: {self.anomaly_counts['vanishing']}\n"
        report += f"- 梯度爆炸: {self.anomaly_counts['exploding']}\n\n"

        # 层级统计
        report += "## 各层梯度统计\n\n"
        report += "| 层名 | 平均值 | 标准差 | 最小值 | 最大值 |\n"
        report += "|------|--------|--------|--------|--------|\n"

        for name, stat in sorted(stats.items()):
            report += f"| {name} | {stat['mean']:.2e} | {stat['std']:.2e} | {stat['min']:.2e} | {stat['max']:.2e} |\n"

        # 保存报告
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)

        self.logger.info(f"汇总报告已保存: {save_path}")

        return str(save_path)

    # =========================================================================
    # 🔮 分布式训练支持
    # =========================================================================

    def sync_gradients_distributed(self):
        """
        在分布式训练中同步梯度信息

        未来DDP训练时调用，确保所有rank的梯度监控一致
        """
        if not self.distributed:
            return

        # 🔮 分布式伏笔：同步梯度范数
        # 在DDP训练中，每个rank的梯度可能略有不同
        # 这里可以all_reduce梯度范数，获取全局平均值

        if len(self.gradient_norms) > 0:
            last_step, last_norm = self.gradient_norms[-1]

            # 将标量转为tensor
            norm_tensor = torch.tensor([last_norm], dtype=torch.float32)

            # 未来可以添加：
            # torch.distributed.all_reduce(norm_tensor, op=torch.distributed.ReduceOp.AVG)

            # 更新为全局平均值
            # self.gradient_norms[-1] = (last_step, norm_tensor.item())

            if self.rank == 0:
                self.logger.debug(f"[Rank {self.rank}] 梯度范数已同步")

    def aggregate_anomalies_distributed(self):
        """
        在分布式训练中聚合异常统计

        未来DDP训练时调用，汇总所有rank的异常
        """
        if not self.distributed:
            return

        # 🔮 分布式伏笔：聚合异常计数
        # 将所有rank的异常计数加总

        # 未来可以添加：
        # for key in self.anomaly_counts:
        #     count_tensor = torch.tensor([self.anomaly_counts[key]], dtype=torch.int64)
        #     torch.distributed.all_reduce(count_tensor, op=torch.distributed.ReduceOp.SUM)
        #     self.anomaly_counts[key] = count_tensor.item()

        if self.rank == 0:
            self.logger.info(f"[Rank {self.rank}] 异常统计已聚合")

    # =========================================================================
    # 便捷方法
    # =========================================================================

    def generate_all_reports(self):
        """
        生成所有报告（图表 + JSON + Markdown）

        供训练结束后调用
        """
        self.logger.info("正在生成梯度监控报告...")

        # 生成图表
        self.plot_gradient_flow()
        self.plot_gradient_norms_timeline()

        # 导出数据
        self.export_for_webui()
        self.export_summary_report()

        self.logger.info(f"所有报告已生成在: {self.export_dir}")

        return {
            'export_dir': str(self.export_dir),
            'gradient_flow_plot': str(self.export_dir / "gradient_flow.png"),
            'gradient_timeline_plot': str(self.export_dir / "gradient_norms_timeline.png"),
            'json_data': str(self.export_dir / "gradient_data.json"),
            'summary_report': str(self.export_dir / "gradient_summary.md")
        }


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 示例：如何在训练循环中使用
    import torch.nn as nn

    # 创建示例模型
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )

    # 初始化监控器
    monitor = GradientMonitor(model, export_dir=".cache/gradient_monitor")

    # 模拟训练
    optimizer = torch.optim.Adam(model.parameters())

    for step in range(100):
        # 前向传播
        x = torch.randn(5, 10)
        output = model(x)
        loss = output.sum()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度监控
        gradients, issues = monitor.check_gradient_flow()
        total_norm = monitor.log_gradient_norms(step)

        # 检测异常
        anomalies = monitor.detect_gradient_anomalies()
        if anomalies:
            print(f"Step {step}: 检测到异常: {anomalies}")

        optimizer.step()

    # 生成报告
    reports = monitor.generate_all_reports()
    print(f"报告已生成: {reports}")

    # 导出WebUI数据
    webui_data = monitor.export_for_webui()
    print(f"WebUI数据: {webui_data['total_steps']} 步")
