#!/usr/bin/env python3
"""
APT训练实时可视化系统 - 科幻风格
包含Loss地形图、动态曲线、梯度流可视化

特性:
- 🌌 3D Loss地形图实时更新
- 📈 科幻风格动态曲线（发光效果）
- 🎨 渐变色彩方案
- 🔄 实时刷新（训练进行时）
- 💫 粒子效果优化轨迹可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import json
import time
from pathlib import Path
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# 设置科幻风格主题
plt.style.use('dark_background')

# 自定义颜色方案 - 赛博朋克风格
CYBER_COLORS = {
    'primary': '#00F0FF',      # 霓虹蓝
    'secondary': '#FF00FF',    # 霓虹粉
    'success': '#00FF41',      # 矩阵绿
    'warning': '#FFD700',      # 金色
    'danger': '#FF1744',       # 红色警报
    'bg': '#0a0e27',          # 深空背景
    'grid': '#1e3a5f',        # 网格线
}


class SciFiVisualizer:
    """科幻风格训练可视化器"""

    def __init__(self, log_dir='./training_logs', refresh_rate=2.0):
        """
        Args:
            log_dir: 训练日志目录
            refresh_rate: 刷新频率（秒）
        """
        self.log_dir = Path(log_dir)
        self.refresh_rate = refresh_rate

        # 数据缓存
        self.control_losses = deque(maxlen=1000)
        self.autopoietic_losses = deque(maxlen=1000)
        self.grad_norms = deque(maxlen=1000)
        self.learning_rates = deque(maxlen=1000)
        self.epochs = deque(maxlen=1000)

        # Loss地形数据
        self.loss_landscape_history = []

        # 训练状态追踪
        self.last_update_time = None
        self.training_active = True
        self.no_update_timeout = 30  # 30秒无更新则认为训练已停止

        # 创建图形界面
        self.setup_figure()

    def setup_figure(self):
        """创建科幻风格的图形界面"""
        # 创建主窗口
        self.fig = plt.figure(figsize=(20, 12), facecolor=CYBER_COLORS['bg'])
        self.title_text = self.fig.suptitle('🚀 APT Training Visualization - Sci-Fi Edition',
                         fontsize=24, color=CYBER_COLORS['primary'],
                         weight='bold', y=0.98)

        # 创建子图网格
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(3, 3, figure=self.fig, hspace=0.3, wspace=0.3)

        # 1. 3D Loss地形图 (左上，2x2)
        self.ax_landscape = self.fig.add_subplot(gs[0:2, 0:2], projection='3d')
        self.setup_landscape_plot()

        # 2. Loss曲线对比 (右上)
        self.ax_loss = self.fig.add_subplot(gs[0, 2])
        self.setup_loss_curve()

        # 3. 梯度范数 (右中)
        self.ax_grad = self.fig.add_subplot(gs[1, 2])
        self.setup_gradient_plot()

        # 4. 学习率变化 (左下)
        self.ax_lr = self.fig.add_subplot(gs[2, 0])
        self.setup_lr_plot()

        # 5. 优化轨迹投影 (中下)
        self.ax_trajectory = self.fig.add_subplot(gs[2, 1])
        self.setup_trajectory_plot()

        # 6. 实时统计 (右下)
        self.ax_stats = self.fig.add_subplot(gs[2, 2])
        self.setup_stats_display()

        plt.tight_layout()

    def setup_landscape_plot(self):
        """设置3D Loss地形图"""
        self.ax_landscape.set_facecolor(CYBER_COLORS['bg'])
        self.ax_landscape.set_title('🌌 Loss Landscape (3D)',
                                   color=CYBER_COLORS['primary'],
                                   fontsize=14, weight='bold', pad=20)
        self.ax_landscape.set_xlabel('Parameter θ₁', color=CYBER_COLORS['success'])
        self.ax_landscape.set_ylabel('Parameter θ₂', color=CYBER_COLORS['success'])
        self.ax_landscape.set_zlabel('Loss', color=CYBER_COLORS['warning'])

        # 设置视角
        self.ax_landscape.view_init(elev=30, azim=45)

    def setup_loss_curve(self):
        """设置Loss曲线图"""
        self.ax_loss.set_facecolor(CYBER_COLORS['bg'])
        self.ax_loss.set_title('📉 Loss Curves - Dual Model',
                              color=CYBER_COLORS['primary'],
                              fontsize=12, weight='bold')
        self.ax_loss.set_xlabel('Epoch', color='white')
        self.ax_loss.set_ylabel('Loss', color='white')
        self.ax_loss.grid(True, alpha=0.2, color=CYBER_COLORS['grid'])

    def setup_gradient_plot(self):
        """设置梯度范数图"""
        self.ax_grad.set_facecolor(CYBER_COLORS['bg'])
        self.ax_grad.set_title('⚡ Gradient Norm Flow',
                              color=CYBER_COLORS['secondary'],
                              fontsize=12, weight='bold')
        self.ax_grad.set_xlabel('Epoch', color='white')
        self.ax_grad.set_ylabel('||∇L||', color='white')
        self.ax_grad.grid(True, alpha=0.2, color=CYBER_COLORS['grid'])

    def setup_lr_plot(self):
        """设置学习率图"""
        self.ax_lr.set_facecolor(CYBER_COLORS['bg'])
        self.ax_lr.set_title('🎢 Learning Rate Schedule',
                            color=CYBER_COLORS['warning'],
                            fontsize=12, weight='bold')
        self.ax_lr.set_xlabel('Epoch', color='white')
        self.ax_lr.set_ylabel('Learning Rate', color='white')
        self.ax_lr.grid(True, alpha=0.2, color=CYBER_COLORS['grid'])

    def setup_trajectory_plot(self):
        """设置优化轨迹图"""
        self.ax_trajectory.set_facecolor(CYBER_COLORS['bg'])
        self.ax_trajectory.set_title('🛸 Optimization Trajectory (2D Projection)',
                                    color=CYBER_COLORS['success'],
                                    fontsize=12, weight='bold')
        self.ax_trajectory.set_xlabel('PC1', color='white')
        self.ax_trajectory.set_ylabel('PC2', color='white')
        self.ax_trajectory.grid(True, alpha=0.2, color=CYBER_COLORS['grid'])

    def setup_stats_display(self):
        """设置统计信息显示"""
        self.ax_stats.axis('off')
        self.ax_stats.set_facecolor(CYBER_COLORS['bg'])

    def generate_loss_landscape(self, current_epoch, current_loss):
        """
        生成Loss地形图数据

        使用简化的损失函数近似：
        L(θ₁, θ₂) = L_current + perturbation_field(θ₁, θ₂)
        """
        # 创建参数网格
        theta1 = np.linspace(-2, 2, 50)
        theta2 = np.linspace(-2, 2, 50)
        Theta1, Theta2 = np.meshgrid(theta1, theta2)

        # 模拟损失函数地形（基于当前loss添加扰动）
        # 使用多个高斯函数模拟复杂地形
        Z = current_loss * np.ones_like(Theta1)

        # 添加多个局部极小值
        Z += 0.5 * np.exp(-((Theta1-0.5)**2 + (Theta2-0.5)**2) / 0.3)
        Z += 0.3 * np.exp(-((Theta1+0.8)**2 + (Theta2-0.3)**2) / 0.5)
        Z += 0.4 * np.exp(-((Theta1-0.2)**2 + (Theta2+0.7)**2) / 0.4)

        # 添加噪声使其更真实
        Z += 0.1 * np.random.randn(*Theta1.shape) * (1.0 / (current_epoch + 1))

        # 当前优化点位置（随epoch移动）
        current_pos = (
            np.sin(current_epoch * 0.1) * 0.8,
            np.cos(current_epoch * 0.1) * 0.8
        )

        return Theta1, Theta2, Z, current_pos

    def update_landscape(self, epoch, loss):
        """更新Loss地形图"""
        self.ax_landscape.clear()
        self.setup_landscape_plot()

        # 生成地形数据
        X, Y, Z, current_pos = self.generate_loss_landscape(epoch, loss)

        # 绘制地形表面（使用渐变色）
        surf = self.ax_landscape.plot_surface(
            X, Y, Z,
            cmap='twilight',  # 科幻风格配色
            alpha=0.8,
            edgecolor='none',
            antialiased=True,
            shade=True
        )

        # 添加等高线投影
        self.ax_landscape.contour(X, Y, Z,
                                 levels=10,
                                 colors=CYBER_COLORS['primary'],
                                 alpha=0.3,
                                 linewidths=1,
                                 offset=Z.min())

        # 标记当前优化点（发光球体）
        self.ax_landscape.scatter(
            [current_pos[0]], [current_pos[1]], [loss],
            color=CYBER_COLORS['success'],
            s=200,
            marker='o',
            edgecolors=CYBER_COLORS['warning'],
            linewidths=3,
            alpha=1.0,
            label='Current Position'
        )

        # 添加粒子轨迹效果
        if len(self.loss_landscape_history) > 0:
            trail_epochs = [h[0] for h in self.loss_landscape_history[-20:]]
            trail_x = [np.sin(e * 0.1) * 0.8 for e in trail_epochs]
            trail_y = [np.cos(e * 0.1) * 0.8 for e in trail_epochs]
            trail_z = [h[1] for h in self.loss_landscape_history[-20:]]

            # 绘制轨迹线（渐变透明度）
            for i in range(len(trail_x) - 1):
                alpha = (i + 1) / len(trail_x)
                self.ax_landscape.plot(
                    trail_x[i:i+2], trail_y[i:i+2], trail_z[i:i+2],
                    color=CYBER_COLORS['secondary'],
                    linewidth=2,
                    alpha=alpha
                )

        # 旋转动画效果
        self.ax_landscape.view_init(elev=30, azim=45 + epoch * 2)

        # 保存历史
        self.loss_landscape_history.append((epoch, loss))

    def update_loss_curve(self):
        """更新Loss曲线（发光效果）"""
        self.ax_loss.clear()
        self.setup_loss_curve()

        if len(self.epochs) > 0:
            epochs = list(self.epochs)

            # 绘制对照组曲线（霓虹蓝）
            if len(self.control_losses) > 0:
                # 主线
                self.ax_loss.plot(epochs, list(self.control_losses),
                                 color=CYBER_COLORS['primary'],
                                 linewidth=2,
                                 label='Control (No Autopoietic)',
                                 marker='o',
                                 markersize=4,
                                 markevery=max(1, len(epochs)//20))

                # 发光效果（多层半透明）
                for i, alpha in enumerate([0.3, 0.2, 0.1]):
                    self.ax_loss.plot(epochs, list(self.control_losses),
                                     color=CYBER_COLORS['primary'],
                                     linewidth=4 + i*2,
                                     alpha=alpha)

            # 绘制实验组曲线（霓虹粉）
            if len(self.autopoietic_losses) > 0:
                # 主线
                self.ax_loss.plot(epochs, list(self.autopoietic_losses),
                                 color=CYBER_COLORS['secondary'],
                                 linewidth=2,
                                 label='Autopoietic (APT)',
                                 marker='s',
                                 markersize=4,
                                 markevery=max(1, len(epochs)//20))

                # 发光效果
                for i, alpha in enumerate([0.3, 0.2, 0.1]):
                    self.ax_loss.plot(epochs, list(self.autopoietic_losses),
                                     color=CYBER_COLORS['secondary'],
                                     linewidth=4 + i*2,
                                     alpha=alpha)

        self.ax_loss.legend(loc='upper right',
                           framealpha=0.3,
                           edgecolor=CYBER_COLORS['grid'])

    def update_gradient_plot(self):
        """更新梯度范数图（能量波效果）"""
        self.ax_grad.clear()
        self.setup_gradient_plot()

        if len(self.epochs) > 0 and len(self.grad_norms) > 0:
            epochs = list(self.epochs)
            grads = list(self.grad_norms)

            # 填充区域（能量场）
            self.ax_grad.fill_between(epochs, 0, grads,
                                      color=CYBER_COLORS['warning'],
                                      alpha=0.3)

            # 主线
            self.ax_grad.plot(epochs, grads,
                             color=CYBER_COLORS['warning'],
                             linewidth=2,
                             marker='*',
                             markersize=6,
                             markevery=max(1, len(epochs)//20))

            # 发光效果
            for i, alpha in enumerate([0.2, 0.1]):
                self.ax_grad.plot(epochs, grads,
                                 color=CYBER_COLORS['warning'],
                                 linewidth=3 + i*2,
                                 alpha=alpha)

    def update_lr_plot(self):
        """更新学习率图（量子波动效果）"""
        self.ax_lr.clear()
        self.setup_lr_plot()

        if len(self.epochs) > 0 and len(self.learning_rates) > 0:
            epochs = list(self.epochs)
            lrs = list(self.learning_rates)

            # 主线
            self.ax_lr.plot(epochs, lrs,
                           color=CYBER_COLORS['success'],
                           linewidth=2)

            # 发光效果
            for i, alpha in enumerate([0.3, 0.2, 0.1]):
                self.ax_lr.plot(epochs, lrs,
                               color=CYBER_COLORS['success'],
                               linewidth=4 + i*2,
                               alpha=alpha)

            # 标记重启点（CosineAnnealingWarmRestarts）
            if len(lrs) > 1:
                # 检测学习率跳跃（重启点）
                for i in range(1, len(lrs)):
                    if lrs[i] > lrs[i-1] * 1.5:
                        self.ax_lr.axvline(x=epochs[i],
                                          color=CYBER_COLORS['danger'],
                                          linestyle='--',
                                          alpha=0.5,
                                          linewidth=1)

        self.ax_lr.set_yscale('log')

    def update_trajectory_plot(self):
        """更新优化轨迹图（星际航线）"""
        self.ax_trajectory.clear()
        self.setup_trajectory_plot()

        if len(self.epochs) > 1:
            # 使用PCA投影到2D（简化版本，用loss和grad近似）
            if len(self.control_losses) > 0 and len(self.autopoietic_losses) > 0:
                control_x = list(self.control_losses)
                control_y = list(self.grad_norms) if len(self.grad_norms) > 0 else [0] * len(control_x)

                # 绘制轨迹
                self.ax_trajectory.plot(control_x, control_y,
                                       color=CYBER_COLORS['primary'],
                                       linewidth=2,
                                       alpha=0.7,
                                       label='Control Path')

                # 起点
                self.ax_trajectory.scatter([control_x[0]], [control_y[0]],
                                          color=CYBER_COLORS['success'],
                                          s=200,
                                          marker='o',
                                          edgecolors='white',
                                          linewidths=2,
                                          label='Start',
                                          zorder=10)

                # 当前点
                self.ax_trajectory.scatter([control_x[-1]], [control_y[-1]],
                                          color=CYBER_COLORS['danger'],
                                          s=300,
                                          marker='*',
                                          edgecolors='white',
                                          linewidths=2,
                                          label='Current',
                                          zorder=10)

        self.ax_trajectory.legend(loc='upper right',
                                 framealpha=0.3,
                                 edgecolor=CYBER_COLORS['grid'])

    def update_stats_display(self):
        """更新统计信息（全息显示）"""
        self.ax_stats.clear()
        self.ax_stats.axis('off')

        # 构建统计文本
        stats_text = "┌─────────────────────────┐\n"
        stats_text += "│   TRAINING STATS        │\n"
        stats_text += "├─────────────────────────┤\n"

        if len(self.epochs) > 0:
            stats_text += f"│ Epoch: {self.epochs[-1]:>15} │\n"

            if len(self.control_losses) > 0:
                stats_text += f"│ Control Loss: {self.control_losses[-1]:>9.4f} │\n"

            if len(self.autopoietic_losses) > 0:
                stats_text += f"│ APT Loss: {self.autopoietic_losses[-1]:>13.4f} │\n"

            if len(self.grad_norms) > 0:
                stats_text += f"│ Grad Norm: {self.grad_norms[-1]:>12.4f} │\n"

            if len(self.learning_rates) > 0:
                stats_text += f"│ LR: {self.learning_rates[-1]:>19.6f} │\n"

            # 计算改进率
            if len(self.control_losses) > 1:
                improvement = (self.control_losses[0] - self.control_losses[-1]) / self.control_losses[0] * 100
                stats_text += f"│ Improvement: {improvement:>10.2f}% │\n"

        stats_text += "└─────────────────────────┘"

        # 显示文本（霓虹绿，矩阵风格）
        self.ax_stats.text(0.5, 0.5, stats_text,
                          fontsize=10,
                          color=CYBER_COLORS['success'],
                          family='monospace',
                          ha='center',
                          va='center',
                          weight='bold')

    def load_latest_data(self):
        """加载最新训练数据"""
        # 查找最新的实验报告
        report_files = list(self.log_dir.glob('experiment_report*.json'))

        if not report_files:
            return False

        latest_report = max(report_files, key=lambda p: p.stat().st_mtime)

        # 检查文件修改时间
        file_mtime = latest_report.stat().st_mtime
        current_time = time.time()

        # 如果文件超过timeout时间未更新，认为训练已停止
        if self.last_update_time is not None:
            time_since_update = current_time - file_mtime
            if time_since_update > self.no_update_timeout and self.training_active:
                self.training_active = False
                # 更新标题为"训练完成"
                self.title_text.set_text('✅ APT Training Complete - Final Results')
                self.title_text.set_color(CYBER_COLORS['success'])
                print(f"\n✅ 训练已完成（{self.no_update_timeout}秒无数据更新）")
                print("📊 可视化显示最终结果，可以关闭窗口退出")

        self.last_update_time = file_mtime

        try:
            with open(latest_report) as f:
                data = json.load(f)

            # 更新数据
            # 优先使用control_losses（epoch级别），如果为空则使用batch_losses（实时）
            if 'control_losses' in data and len(data['control_losses']) > 0:
                for i, loss in enumerate(data['control_losses']):
                    if i >= len(self.epochs):
                        self.epochs.append(i + 1)
                        self.control_losses.append(loss)
            elif 'batch_losses' in data and len(data['batch_losses']) > 0:
                # 使用batch_losses进行实时可视化
                # 每10个batch取一个点，避免数据过密
                batch_losses = data['batch_losses']
                step = max(1, len(batch_losses) // 100)  # 最多100个点
                for i in range(0, len(batch_losses), step):
                    if i >= len(self.control_losses):
                        epoch_progress = i / len(batch_losses)  # 当前epoch的进度
                        self.epochs.append(epoch_progress)
                        self.control_losses.append(batch_losses[i])

            if 'autopoietic_losses' in data:
                for i, loss in enumerate(data['autopoietic_losses']):
                    if i < len(self.epochs):
                        if i >= len(self.autopoietic_losses):
                            self.autopoietic_losses.append(loss)

            # 读取真实梯度范数数据
            if 'grad_norms' in data:
                for i, grad_norm in enumerate(data['grad_norms']):
                    if i >= len(self.grad_norms):
                        self.grad_norms.append(grad_norm)

            # 读取真实学习率数据
            if 'learning_rates' in data:
                for i, lr in enumerate(data['learning_rates']):
                    if i >= len(self.learning_rates):
                        self.learning_rates.append(lr)

            return True

        except Exception as e:
            print(f"⚠️  加载数据失败: {e}")
            return False

    def update_all_plots(self):
        """更新所有图表"""
        # 如果训练已停止，不再加载新数据，只保持显示
        if not self.training_active:
            return

        # 加载最新数据
        data_updated = self.load_latest_data()

        if not data_updated or len(self.epochs) == 0:
            return

        # 更新所有子图
        current_epoch = self.epochs[-1]
        current_loss = self.control_losses[-1] if len(self.control_losses) > 0 else 0

        self.update_landscape(current_epoch, current_loss)
        self.update_loss_curve()
        self.update_gradient_plot()
        self.update_lr_plot()
        self.update_trajectory_plot()
        self.update_stats_display()

        # 添加时间戳和状态
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        status = '🟢 Training Active' if self.training_active else '🔴 Training Stopped'
        self.fig.text(0.99, 0.01, f'{status} | Last Update: {timestamp}',
                     ha='right', va='bottom',
                     fontsize=8, color=CYBER_COLORS['success'] if self.training_active else CYBER_COLORS['danger'],
                     style='italic')

        plt.draw()

    def run(self):
        """启动实时可视化"""
        print("🚀 启动科幻风格训练可视化...")
        print(f"   日志目录: {self.log_dir}")
        print(f"   刷新频率: {self.refresh_rate}秒")
        print(f"   自动停止: {self.no_update_timeout}秒无更新时停止刷新")
        print("\n💡 提示:")
        print("   - 可视化会自动检测训练结束")
        print("   - 训练停止后显示最终结果，可直接关闭窗口")
        print("   - 或按 Ctrl+C 手动退出\n")

        # 初始化数据
        self.load_latest_data()

        # 创建动画
        def animate(frame):
            self.update_all_plots()
            return []

        # 使用FuncAnimation进行实时更新
        anim = FuncAnimation(
            self.fig,
            animate,
            interval=int(self.refresh_rate * 1000),
            blit=False,
            cache_frame_data=False
        )

        plt.show()


# ============================================================================
# 命令行接口
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='APT训练实时可视化 - 科幻风格',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认监控control_experiments目录
  python visualize_training.py

  # 指定日志目录和刷新频率
  python visualize_training.py --log-dir ./playground_checkpoints --refresh 1.0

  # 离线模式（查看已完成的训练）
  python visualize_training.py --offline
        """
    )

    parser.add_argument('--log-dir', type=str, default='control_experiments',
                       help='训练日志目录 (default: control_experiments)')
    parser.add_argument('--refresh', type=float, default=2.0,
                       help='刷新频率（秒） (default: 2.0)')
    parser.add_argument('--offline', action='store_true',
                       help='离线模式（不自动刷新）')

    args = parser.parse_args()

    # 创建可视化器
    viz = SciFiVisualizer(
        log_dir=args.log_dir,
        refresh_rate=args.refresh
    )

    if args.offline:
        # 离线模式：加载一次数据并显示
        print("📊 离线模式：加载训练数据...")
        viz.load_latest_data()
        viz.update_all_plots()
        plt.show()
    else:
        # 实时模式
        viz.run()
