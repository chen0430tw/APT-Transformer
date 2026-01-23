#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization module for APT Model
Provides visualization tools for model training and evaluation results
"""

import os
import time
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch

# 为确保中文显示，设置 matplotlib 默认字体（此设置仅在创建图表前生效）。
# matplotlib 是可选依赖，因此在缺失时不要让导入失败。
try:  # pragma: no cover - executed only when matplotlib is present
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
except ImportError:  # pragma: no cover - optional dependency path
    matplotlib = None

class ModelVisualizer:
    """
    Model visualization tool
    Generates various evaluation charts and report.
    
    图表输出路径会默认存放在项目的 [apt_model/report] 目录下，
    如使用 cache_manager 可进一步管理。
    """
    def __init__(self, cache_manager=None, logger=None):
        """
        Initialize the visualization tool
        
        Args:
            cache_manager: Cache manager，用于生成默认输出路径
            logger: Logger实例
        """
        self.logger = logger
        self.cache_manager = cache_manager

        # 检查可视化依赖库
        self.has_matplotlib = self._check_matplotlib()
        self.has_plotly = self._check_plotly()
        self.has_seaborn = self._check_seaborn()

        if not any([self.has_matplotlib, self.has_plotly, self.has_seaborn]):
            warning_msg = ("No visualization libraries are installed. "
                           "Visualization features are not available. Please install matplotlib, plotly, or seaborn.")
            if self.logger:
                self.logger.warning(warning_msg)
            print(f"Warning: {warning_msg}")
        else:
            if self.logger:
                self.logger.info("Visualization dependencies check passed.")
            print("Visualization dependencies check passed.")

    def _check_matplotlib(self) -> bool:
        """Check if matplotlib is available"""
        try:
            import matplotlib.pyplot as plt
            return True
        except ImportError:
            return False

    def _check_plotly(self) -> bool:
        """Check if plotly is available"""
        try:
            import plotly.graph_objects as go
            return True
        except ImportError:
            return False

    def _check_seaborn(self) -> bool:
        """Check if seaborn is available"""
        try:
            import seaborn as sns
            return True
        except ImportError:
            return False

    def _get_default_visualization_dir(self) -> str:
        """
        返回默认的图表输出目录，即项目的 [apt_model/report] 目录
        """
        # 当前文件位于 apt_model/utils，返回上一级目录（apt_model）
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_dir = os.path.join(base_dir, "report")
        os.makedirs(default_dir, exist_ok=True)
        return default_dir

    def create_training_history_plot(self, history: Dict[str, List[float]], 
                                     output_path: Optional[str] = None, 
                                     title: str = "Training History") -> Optional[str]:
        """
        Create a training history plot

        Args:
            history: Training history data (例如：{'loss': [...], 'val_loss': [...], 'learning_rate': [...]})
            output_path: 输出文件路径，如果为 None，则自动保存到默认目录
            title: 图表标题

        Returns:
            str: 图表保存的完整路径，或 None（若出错）
        """
        print("开始创建训练历史图表...")
        if not self.has_matplotlib:
            msg = "Error: matplotlib is not installed. Cannot create training history plot."
            if self.logger:
                self.logger.warning(msg)
            print(msg)
            return None

        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

            # 绘制损失曲线
            if 'loss' in history:
                axes[0].plot(history['loss'], label='Training Loss')
            if 'val_loss' in history:
                axes[0].plot(history['val_loss'], label='Validation Loss')

            axes[0].set_ylabel('Loss')
            axes[0].set_title(f'{title} - Loss')
            axes[0].legend()
            axes[0].grid(True)

            # 绘制学习率变化曲线
            if 'learning_rate' in history:
                axes[1].plot(history['learning_rate'], label='Learning Rate')
                axes[1].set_xlabel('Steps')
                axes[1].set_ylabel('Learning Rate')
                axes[1].set_title('Learning Rate Change')
                axes[1].legend()
                axes[1].grid(True)

            plt.tight_layout()

            if output_path:
                out_path = os.path.abspath(output_path)
            elif self.cache_manager:
                timestamp = int(time.time())
                out_path = self.cache_manager.get_cache_path("report", f"training_history_{timestamp}.png")
            else:
                default_dir = self._get_default_visualization_dir()
                out_path = os.path.join(default_dir, f"training_history_{int(time.time())}.png")
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path)
            plt.close()
            if self.logger:
                self.logger.info(f"Training history chart saved to: {out_path}")
            print(f"训练历史图表已保存到: {out_path}")
            return out_path
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating training history chart: {e}")
            import traceback
            print(f"创建训练历史图表时出错: {e}")
            print(traceback.format_exc())
            return None

    def create_evaluation_radar_chart(self, metrics: Dict[str, float], 
                                      output_path: Optional[str] = None, 
                                      title: str = "Model Evaluation") -> Optional[str]:
        """
        Create an evaluation radar chart

        Args:
            metrics: Evaluation metrics字典，例如 {"Generation Quality": 85, "Reasoning Ability": 72, ...}
            output_path: 输出路径，如果为 None，则自动保存到默认目录
            title: 图表标题

        Returns:
            str: 图表保存的路径，或 None（若出错）
        """
        print("开始创建评估雷达图...")
        if not self.has_plotly:
            msg = "Error: plotly is not installed. Cannot create evaluation radar chart."
            if self.logger:
                self.logger.warning(msg)
            print(msg)
            return None

        try:
            import plotly.graph_objects as go

            categories = list(metrics.keys())
            values = [metrics[cat] for cat in categories]

            fig = go.Figure()

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=title
            ))

            fig.update_layout(
                title=title,
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                )
            )

            if output_path:
                out_path = os.path.abspath(output_path)
            elif self.cache_manager:
                timestamp = int(time.time())
                out_path = self.cache_manager.get_cache_path("report", f"evaluation_radar_{timestamp}.png")
            else:
                default_dir = self._get_default_visualization_dir()
                out_path = os.path.join(default_dir, f"evaluation_radar_{int(time.time())}.html")
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            try:
                fig.write_image(out_path)
            except Exception as e:
                print(f"使用默认格式保存图表失败: {e}")
                if self.logger:
                    self.logger.warning(f"Failed to save chart with default format: {e}")
                # 尝试使用HTML格式
                html_path = out_path.rsplit('.', 1)[0] + '.html'
                print(f"尝试保存为HTML格式: {html_path}")
                fig.write_html(html_path)
                out_path = html_path

            if self.logger:
                self.logger.info(f"Evaluation radar chart saved to: {out_path}")
            print(f"评估雷达图已保存到: {out_path}")
            return out_path
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating evaluation radar chart: {e}")
            import traceback
            print(f"创建评估雷达图时出错: {e}")
            print(traceback.format_exc())
            return None

    def create_comparison_bar_chart(self, models_data: Dict[str, Dict[str, float]], 
                                    metric: str, 
                                    output_path: Optional[str] = None, 
                                    title: str = "Model Comparison") -> Optional[str]:
        """
        Create a model comparison bar chart

        Args:
            models_data: 模型数据字典，格式为 {"model_name": {metric: value, ...}, ...}
            metric: 要对比的指标
            output_path: 输出路径，若为 None 则自动生成路径
            title: 图表标题

        Returns:
            str: 图表保存的路径，或 None（若出错）
        """
        print("开始创建模型对比图...")
        if not self.has_matplotlib:
            msg = "Error: matplotlib is not installed. Cannot create model comparison bar chart."
            if self.logger:
                self.logger.warning(msg)
            print(msg)
            return None

        try:
            import matplotlib.pyplot as plt

            model_names = list(models_data.keys())
            values = [models_data[model].get(metric, 0) for model in model_names]

            plt.figure(figsize=(10, 6))
            bars = plt.bar(model_names, values, color='skyblue')
            plt.xlabel("Model")
            plt.ylabel(metric)
            plt.title(f'{title} - {metric}')

            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom')

            plt.tight_layout()

            if output_path:
                out_path = os.path.abspath(output_path)
            elif self.cache_manager:
                timestamp = int(time.time())
                out_path = self.cache_manager.get_cache_path("report", f"model_comparison_{metric}_{timestamp}.png")
            else:
                default_dir = self._get_default_visualization_dir()
                out_path = os.path.join(default_dir, f"model_comparison_{metric}_{int(time.time())}.png")
            
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path)
            plt.close()
            print(f"模型对比图已保存到: {out_path}")
            if self.logger:
                self.logger.info(f"Model comparison bar chart saved to: {out_path}")
            return out_path
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating model comparison bar chart: {e}")
            import traceback
            print(f"创建模型对比图时出错: {e}")
            print(traceback.format_exc())
            return None

    def create_attention_heatmap(self, attention_weights: Union[np.ndarray, torch.Tensor], 
                                 tokens: List[str], 
                                 output_path: Optional[str] = None, 
                                 title: str = "Attention Weights") -> Optional[str]:
        """
        Create an attention heatmap

        Args:
            attention_weights: 注意力矩阵（numpy数组或Tensor）
            tokens: 对应的token列表
            output_path: 输出路径，若为 None 则自动生成路径
            title: 图表标题

        Returns:
            str: 图表保存的路径，或 None（若出错）
        """
        print("开始创建注意力热图...")
        if not self.has_seaborn:
            msg = "Error: seaborn is not installed. Cannot create attention heatmap."
            if self.logger:
                self.logger.warning(msg)
            print(msg)
            return None

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # 如果注意力权重是Tensor，则转换为numpy数组
            if isinstance(attention_weights, torch.Tensor):
                attention_weights = attention_weights.cpu().detach().numpy()

            # 检查 tokens 和 attention_weights 的维度匹配
            if len(tokens) != attention_weights.shape[0] or len(tokens) != attention_weights.shape[1]:
                msg = f"错误: Token数量 ({len(tokens)}) 与注意力矩阵维度 ({attention_weights.shape}) 不匹配"
                if self.logger:
                    self.logger.error(msg)
                print(msg)
                return None

            plt.figure(figsize=(12, 10))
            ax = sns.heatmap(attention_weights, xticklabels=tokens, yticklabels=tokens,
                             cmap="YlGnBu", vmin=0, vmax=np.max(attention_weights),
                             cbar_kws={'label': 'Attention Weight'})
            plt.title(title)
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            plt.tight_layout()

            if output_path:
                out_path = os.path.abspath(output_path)
            elif self.cache_manager:
                timestamp = int(time.time())
                out_path = self.cache_manager.get_cache_path("report", f"attention_heatmap_{timestamp}.png")
            else:
                default_dir = self._get_default_visualization_dir()
                out_path = os.path.join(default_dir, f"attention_heatmap_{int(time.time())}.png")
            
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"注意力热图已保存到: {out_path}")
            if self.logger:
                self.logger.info(f"Attention heatmap saved to: {out_path}")
            return out_path
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating attention heatmap: {e}")
            import traceback
            print(f"创建注意力热图时出错: {e}")
            print(traceback.format_exc())
            return None

    def create_quality_trend_chart(self, quality_scores: List[Tuple[int, float]], 
                                   output_path: Optional[str] = None, 
                                   title: str = "Generation Quality Trend") -> Optional[str]:
        """
        Create a generation quality trend chart

        Args:
            quality_scores: List of (epoch, score) tuples
            output_path: 输出路径，若为 None 则自动生成路径
            title: 图表标题

        Returns:
            str: 图表保存的路径，或 None（若出错）
        """
        print("开始创建质量趋势图...")
        if not self.has_matplotlib:
            msg = "Error: matplotlib is not installed. Cannot create quality trend chart."
            if self.logger:
                self.logger.warning(msg)
            print(msg)
            return None

        try:
            import matplotlib.pyplot as plt

            epochs = [item[0] for item in quality_scores]
            scores = [item[1] for item in quality_scores]

            plt.figure(figsize=(10, 6))
            plt.plot(epochs, scores, 'b-o', label='Generation Quality Score')
            # 添加趋势线
            z = np.polyfit(epochs, scores, 1)
            p = np.poly1d(z)
            plt.plot(epochs, p(epochs), "r--", label='Trend Line')

            plt.title(title)
            plt.xlabel('Training Epoch')
            plt.ylabel('Quality Score')
            plt.ylim(0, 100)
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            if output_path:
                out_path = os.path.abspath(output_path)
            elif self.cache_manager:
                timestamp = int(time.time())
                out_path = self.cache_manager.get_cache_path("report", f"quality_trend_{timestamp}.png")
            else:
                default_dir = self._get_default_visualization_dir()
                out_path = os.path.join(default_dir, f"quality_trend_{int(time.time())}.png")

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path)
            plt.close()
            print(f"质量趋势图已保存到: {out_path}")
            if self.logger:
                self.logger.info(f"Quality trend chart saved to: {out_path}")
            return out_path
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating quality trend chart: {e}")
            import traceback
            print(f"创建质量趋势图时出错: {e}")
            print(traceback.format_exc())
            return None

    def create_evaluation_report(self, model_name: str, 
                                 evaluation_results: Dict[str, Any], 
                                 output_path: Optional[str] = None) -> Optional[str]:
        """
        Create an evaluation report

        Args:
            model_name: 模型名称
            evaluation_results: 评估结果数据字典
            output_path: 输出路径，若为 None 则自动生成路径

        Returns:
            str: 报告保存的路径，或 None（若出错）
        """
        print("开始生成评估报告...")
        try:
            report = f"# {model_name} Evaluation Report\n\n"
            report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

            if 'overall' in evaluation_results:
                report += "## Overall Performance\n\n"
                overall = evaluation_results['overall']
                for metric, value in overall.items():
                    if isinstance(value, (int, float)):
                        report += f"- **{metric}**: {value:.2f}\n"
                    else:
                        report += f"- **{metric}**: {value}\n"
                report += "\n"

            if 'categories' in evaluation_results:
                report += "## Category Performance\n\n"
                categories = evaluation_results['categories']
                report += "| Category | Score | Sample Count |\n"
                report += "|----------|-------|---------------|\n"
                for category, data in categories.items():
                    score = data.get('score', 0)
                    count = data.get('count', 0)
                    report += f"| {category} | {score:.2f} | {count} |\n"
                report += "\n"

            if 'samples' in evaluation_results:
                report += "## Sample Evaluation\n\n"
                samples = evaluation_results['samples']
                for i, sample in enumerate(samples[:10]):
                    report += f"### Sample {i+1}\n\n"
                    report += f"**Prompt**: {sample.get('prompt', '')}\n\n"
                    report += f"**Model Response**: {sample.get('response', '')}\n\n"
                    report += f"**Reference Response**: {sample.get('reference', '')}\n\n"
                    report += f"**Score**: {sample.get('score', 0):.2f}\n\n"
                    report += f"**Feedback**: {sample.get('feedback', '')}\n\n"
                    report += "---\n\n"
                report += f"*Note: Only showing first 10 samples, total {len(samples)} samples*\n\n"

            report += "## Evaluation Charts\n\n"
            report += "*Charts are provided in separate files*\n\n"

            if output_path:
                out_path = os.path.abspath(output_path)
            elif self.cache_manager:
                timestamp = int(time.time())
                out_path = self.cache_manager.get_cache_path("report", f"{model_name}_evaluation_report_{timestamp}.md")
            else:
                default_dir = self._get_default_visualization_dir()
                out_path = os.path.join(default_dir, f"{model_name}_evaluation_report_{int(time.time())}.md")

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"评估报告已保存到: {out_path}")
            if self.logger:
                self.logger.info(f"Evaluation report saved to: {out_path}")
            return out_path
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating evaluation report: {e}")
            import traceback
            print(f"创建评估报告时出错: {e}")
            print(traceback.format_exc())
            return None

    def get_cache_path(self, category: str, filename: str) -> str:
        """
        Get path for a cached file

        Args:
            category: 缓存类别
            filename: 文件名

        Returns:
            str: 缓存文件的完整路径
        """
        if self.cache_manager:
            return self.cache_manager.get_cache_path(category, filename)
        else:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", category)
            os.makedirs(cache_dir, exist_ok=True)
            return os.path.join(cache_dir, filename)

    def plot_model_comparison(self, model_scores: dict) -> Optional[str]:
        """
        绘制模型得分对比图

        参数:
            model_scores (dict): 模型得分字典，例如 {"model_name": score, ...}

        返回:
            str: 图表保存的路径，或 None（若出错）
        """
        print("开始绘制模型对比图...")
        if not self.has_matplotlib:
            msg = "Error: matplotlib is not installed. Cannot create model comparison plot."
            if self.logger:
                self.logger.warning(msg)
            print(msg)
            return None

        try:
            import matplotlib.pyplot as plt

            models = list(model_scores.keys())
            scores = [model_scores[model] for model in models]

            plt.figure(figsize=(10, 6))
            bars = plt.bar(models, scores, color='skyblue')
            plt.xlabel("模型")
            plt.ylabel("得分")
            plt.title("模型性能对比")
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.1f}', ha='center', va='bottom')
            plt.tight_layout()

            timestamp = int(time.time())
            if self.cache_manager:
                out_path = self.cache_manager.get_cache_path("report", f"model_comparison_{timestamp}.png")
            else:
                default_dir = self._get_default_visualization_dir()
                out_path = os.path.join(default_dir, f"model_comparison_{timestamp}.png")

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path)
            plt.close()
            print(f"模型对比图已保存到: {out_path}")
            if self.logger:
                self.logger.info(f"Model comparison plot saved to: {out_path}")
            return out_path
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating model comparison plot: {e}")
            import traceback
            print(f"创建模型对比图时出错: {e}")
            print(traceback.format_exc())
            return None

    def plot_category_comparison(self, model_name: str, category_scores: dict, output_path: str = None) -> Optional[str]:
        """
        绘制模型各类别得分对比条形图。

        参数:
            model_name (str): 模型名称，用于图表标题。
            category_scores (dict): 每个类别的得分字典，例如 {"factual": 50, "logical": 60, ...}
            output_path (str, optional): 指定保存路径，若为 None，则自动生成。

        返回:
            str: 图表保存的文件路径，或 None（若出错）
        """
        print("开始绘制类别对比图...")
        if not self.has_matplotlib:
            msg = "Error: matplotlib is not installed. Cannot create category comparison chart."
            if self.logger:
                self.logger.warning(msg)
            print(msg)
            return None

        try:
            import matplotlib.pyplot as plt

            # 如果需要中文显示
            try:
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to set font for Chinese display: {e}")
                print(f"Warning: Failed to set font for Chinese display: {e}")

            if output_path is None:
                if self.cache_manager:
                    timestamp = int(time.time())
                    output_path = self.cache_manager.get_cache_path("report", f"{model_name}_category_comparison_{timestamp}.png")
                else:
                    default_dir = self._get_default_visualization_dir()
                    timestamp = int(time.time())
                    output_path = os.path.join(default_dir, f"{model_name}_category_comparison_{timestamp}.png")

            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            categories = list(category_scores.keys())
            scores = [category_scores[cat] for cat in categories]

            plt.figure(figsize=(10, 6))
            bars = plt.bar(categories, scores, color='skyblue')
            plt.xlabel("类别")
            plt.ylabel("得分")
            plt.ylim(0, 100)
            plt.title(f"{model_name} 各类别得分对比")
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.1f}', ha='center', va='bottom')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            print(f"类别对比图已保存到: {output_path}")
            if self.logger:
                self.logger.info(f"Category comparison plot saved to: {output_path}")
            return output_path
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating category comparison chart: {e}")
            import traceback
            print(f"创建类别对比图时出错: {e}")
            print(traceback.format_exc())
            return None

    def save_report(self, model_name: str, report_content: str, output_path: str = None) -> Optional[str]:
        """
        保存评估报告到文件

        参数:
            model_name (str): 模型名称，用于生成文件名
            report_content (str): 报告内容
            output_path (str, optional): 指定保存路径。如果为 None，则自动生成

        返回:
            str: 报告保存的文件路径，或 None（若出错）
        """
        print("开始保存评估报告...")
        try:
            if output_path is None:
                if self.cache_manager:
                    timestamp = int(time.time())
                    output_path = self.cache_manager.get_cache_path("report", f"{model_name}_report_{timestamp}.md")
                else:
                    default_dir = self._get_default_visualization_dir()
                    output_path = os.path.join(default_dir, f"{model_name}_report_{int(time.time())}.md")

            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            print(f"报告已保存到: {output_path}")
            if self.logger:
                self.logger.info(f"Report saved to: {output_path}")
            return output_path
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error saving report: {e}")
            import traceback
            print(f"保存报告时出错: {e}")
            print(traceback.format_exc())
            return None

    def get_cache_path(self, category: str, filename: str) -> str:
        """
        Get path for a cached file

        Args:
            category: 缓存类别
            filename: 文件名

        Returns:
            str: 缓存文件的完整路径
        """
        if self.cache_manager:
            return self.cache_manager.get_cache_path(category, filename)
        else:
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", category)
            os.makedirs(cache_dir, exist_ok=True)
            return os.path.join(cache_dir, filename)

    def plot_model_comparison(self, model_scores: dict) -> Optional[str]:
        """
        绘制模型得分对比图

        参数:
            model_scores (dict): 模型得分字典，例如 {"model_name": score, ...}

        返回:
            str: 图表保存的路径，或 None（若出错）
        """
        print("开始绘制模型对比图...")
        if not self.has_matplotlib:
            msg = "Error: matplotlib is not installed. Cannot create model comparison plot."
            if self.logger:
                self.logger.warning(msg)
            print(msg)
            return None

        try:
            import matplotlib.pyplot as plt

            models = list(model_scores.keys())
            scores = [model_scores[model] for model in models]

            plt.figure(figsize=(10, 6))
            bars = plt.bar(models, scores, color='skyblue')
            plt.xlabel("模型")
            plt.ylabel("得分")
            plt.title("模型性能对比")
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.1f}', ha='center', va='bottom')
            plt.tight_layout()

            timestamp = int(time.time())
            if self.cache_manager:
                out_path = self.cache_manager.get_cache_path("report", f"model_comparison_{timestamp}.png")
            else:
                default_dir = self._get_default_visualization_dir()
                out_path = os.path.join(default_dir, f"model_comparison_{timestamp}.png")

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path)
            plt.close()
            print(f"模型对比图已保存到: {out_path}")
            if self.logger:
                self.logger.info(f"Model comparison plot saved to: {out_path}")
            return out_path
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating model comparison plot: {e}")
            import traceback
            print(f"创建模型对比图时出错: {e}")
            print(traceback.format_exc())
            return None

    def plot_category_comparison(self, model_name: str, category_scores: dict, output_path: str = None) -> Optional[str]:
        """
        绘制模型各类别得分对比条形图。

        参数:
            model_name (str): 模型名称，用于图表标题。
            category_scores (dict): 每个类别的得分字典，例如 {"factual": 50, "logical": 60, ...}
            output_path (str, optional): 指定保存路径，若为 None，则自动生成。

        返回:
            str: 图表保存的文件路径，或 None（若出错）
        """
        print("开始绘制类别对比图...")
        if not self.has_matplotlib:
            msg = "Error: matplotlib is not installed. Cannot create category comparison chart."
            if self.logger:
                self.logger.warning(msg)
            print(msg)
            return None

        try:
            import matplotlib.pyplot as plt

            # 如果需要中文显示
            try:
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to set font for Chinese display: {e}")
                print(f"Warning: Failed to set font for Chinese display: {e}")

            if output_path is None:
                if self.cache_manager:
                    timestamp = int(time.time())
                    output_path = self.cache_manager.get_cache_path("report", f"{model_name}_category_comparison_{timestamp}.png")
                else:
                    default_dir = self._get_default_visualization_dir()
                    timestamp = int(time.time())
                    output_path = os.path.join(default_dir, f"{model_name}_category_comparison_{timestamp}.png")

            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            categories = list(category_scores.keys())
            scores = [category_scores[cat] for cat in categories]

            plt.figure(figsize=(10, 6))
            bars = plt.bar(categories, scores, color='skyblue')
            plt.xlabel("类别")
            plt.ylabel("得分")
            plt.ylim(0, 100)
            plt.title(f"{model_name} 各类别得分对比")
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.1f}', ha='center', va='bottom')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            print(f"类别对比图已保存到: {output_path}")
            if self.logger:
                self.logger.info(f"Category comparison plot saved to: {output_path}")
            return output_path
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error creating category comparison chart: {e}")
            import traceback
            print(f"创建类别对比图时出错: {e}")
            print(traceback.format_exc())
            return None