#!/usr/bin/env python3
"""
多训练监控面板
自动发现并监控所有正在进行的训练

特性:
- 🔍 自动扫描训练目录
- 📊 统一界面显示所有训练
- 🎯 实时对比多个实验
- 🚦 活跃状态指示器
"""

import json
import time
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation

# 科幻配色
CYBER_COLORS = {
    'primary': '#00F0FF',
    'secondary': '#FF00FF',
    'success': '#00FF41',
    'warning': '#FFD700',
    'danger': '#FF1744',
    'bg': '#0a0e27',
    'grid': '#1e3a5f',
}

plt.style.use('dark_background')


class MultiTrainingMonitor:
    """多训练监控器"""

    def __init__(self, search_dirs=None, refresh_rate=3.0):
        """
        Args:
            search_dirs: 搜索目录列表（None表示自动搜索）
            refresh_rate: 刷新频率（秒）
        """
        self.search_dirs = search_dirs or [
            'control_experiments',
            'playground_checkpoints',
            'tests/saved_models',
            'demo_visualization',
        ]
        self.refresh_rate = refresh_rate
        self.trainings = {}

        # 创建界面
        self.setup_figure()

    def setup_figure(self):
        """创建多训练监控界面"""
        self.fig = plt.figure(figsize=(20, 12), facecolor=CYBER_COLORS['bg'])
        self.fig.suptitle('🎯 Multi-Training Monitor - APT Experiments',
                         fontsize=22, color=CYBER_COLORS['primary'],
                         weight='bold', y=0.98)

        gs = GridSpec(2, 2, figure=self.fig, hspace=0.25, wspace=0.25)

        # 1. 训练列表和状态
        self.ax_list = self.fig.add_subplot(gs[0, 0])
        self.setup_training_list()

        # 2. Loss对比图
        self.ax_loss = self.fig.add_subplot(gs[0, 1])
        self.setup_loss_comparison()

        # 3. 活跃度时间线
        self.ax_timeline = self.fig.add_subplot(gs[1, 0])
        self.setup_activity_timeline()

        # 4. 最佳模型排行榜
        self.ax_leaderboard = self.fig.add_subplot(gs[1, 1])
        self.setup_leaderboard()

    def setup_training_list(self):
        """设置训练列表"""
        self.ax_list.axis('off')
        self.ax_list.set_facecolor(CYBER_COLORS['bg'])

    def setup_loss_comparison(self):
        """设置Loss对比图"""
        self.ax_loss.set_facecolor(CYBER_COLORS['bg'])
        self.ax_loss.set_title('📊 Loss Comparison - All Trainings',
                              color=CYBER_COLORS['primary'],
                              fontsize=14, weight='bold')
        self.ax_loss.set_xlabel('Epoch', color='white')
        self.ax_loss.set_ylabel('Loss', color='white')
        self.ax_loss.grid(True, alpha=0.2, color=CYBER_COLORS['grid'])

    def setup_activity_timeline(self):
        """设置活跃度时间线"""
        self.ax_timeline.set_facecolor(CYBER_COLORS['bg'])
        self.ax_timeline.set_title('⏱️  Training Activity Timeline',
                                  color=CYBER_COLORS['secondary'],
                                  fontsize=14, weight='bold')
        self.ax_timeline.set_xlabel('Time (minutes ago)', color='white')
        self.ax_timeline.grid(True, alpha=0.2, color=CYBER_COLORS['grid'])

    def setup_leaderboard(self):
        """设置排行榜"""
        self.ax_leaderboard.axis('off')
        self.ax_leaderboard.set_facecolor(CYBER_COLORS['bg'])

    def discover_trainings(self):
        """自动发现训练"""
        current_time = time.time()
        discovered = {}

        for search_dir in self.search_dirs:
            dir_path = Path(search_dir)
            if not dir_path.exists():
                continue

            # 查找experiment_report.json
            report_files = list(dir_path.glob('**/experiment_report.json'))

            for report_file in report_files:
                try:
                    with open(report_file) as f:
                        data = json.load(f)

                    # 获取训练信息
                    training_name = report_file.parent.name
                    last_update = report_file.stat().st_mtime
                    age_minutes = (current_time - last_update) / 60

                    # 判断活跃状态
                    if age_minutes < 1:
                        status = '🟢 Active'
                        status_color = CYBER_COLORS['success']
                    elif age_minutes < 5:
                        status = '🟡 Recent'
                        status_color = CYBER_COLORS['warning']
                    else:
                        status = '⚪ Idle'
                        status_color = CYBER_COLORS['grid']

                    discovered[training_name] = {
                        'data': data,
                        'status': status,
                        'status_color': status_color,
                        'last_update': last_update,
                        'age_minutes': age_minutes,
                        'path': str(report_file.parent)
                    }

                except Exception as e:
                    print(f"⚠️  读取 {report_file} 失败: {e}")

        self.trainings = discovered
        return discovered

    def update_training_list(self):
        """更新训练列表"""
        self.ax_list.clear()
        self.ax_list.axis('off')
        self.ax_list.set_facecolor(CYBER_COLORS['bg'])

        # 标题
        title = "🔍 Discovered Trainings\n" + "─" * 40 + "\n"

        if not self.trainings:
            text = title + "\n❌ No active trainings found\n\n"
            text += "Searching in:\n"
            for d in self.search_dirs:
                text += f"  • {d}\n"
        else:
            text = title + f"\nFound {len(self.trainings)} training(s):\n\n"

            for i, (name, info) in enumerate(sorted(self.trainings.items()), 1):
                data = info['data']
                current_epoch = data.get('current_epoch', 0)

                # 获取最新loss
                if 'control_losses' in data and data['control_losses']:
                    latest_loss = data['control_losses'][-1]
                elif 'autopoietic_losses' in data and data['autopoietic_losses']:
                    latest_loss = data['autopoietic_losses'][-1]
                else:
                    latest_loss = 0.0

                age = f"{info['age_minutes']:.1f}m ago" if info['age_minutes'] < 60 else f"{info['age_minutes']/60:.1f}h ago"

                text += f"{i}. {name}\n"
                text += f"   {info['status']} | Epoch {current_epoch} | Loss: {latest_loss:.4f}\n"
                text += f"   Updated: {age}\n\n"

        self.ax_list.text(0.05, 0.95, text,
                         fontsize=10,
                         color=CYBER_COLORS['success'],
                         family='monospace',
                         ha='left', va='top',
                         transform=self.ax_list.transAxes)

    def update_loss_comparison(self):
        """更新Loss对比图"""
        self.ax_loss.clear()
        self.setup_loss_comparison()

        if not self.trainings:
            return

        # 配色方案
        colors = [
            CYBER_COLORS['primary'],
            CYBER_COLORS['secondary'],
            CYBER_COLORS['success'],
            CYBER_COLORS['warning'],
            '#FF6B9D',  # 粉红
            '#00D9FF',  # 青色
        ]

        for idx, (name, info) in enumerate(self.trainings.items()):
            data = info['data']
            color = colors[idx % len(colors)]

            # 绘制对照组
            if 'control_losses' in data and data['control_losses']:
                epochs = list(range(1, len(data['control_losses']) + 1))
                self.ax_loss.plot(epochs, data['control_losses'],
                                 color=color,
                                 linewidth=2,
                                 label=f"{name} (Control)",
                                 alpha=0.8,
                                 marker='o',
                                 markersize=3,
                                 markevery=max(1, len(epochs)//20))

            # 绘制实验组
            if 'autopoietic_losses' in data and data['autopoietic_losses']:
                epochs = list(range(1, len(data['autopoietic_losses']) + 1))
                self.ax_loss.plot(epochs, data['autopoietic_losses'],
                                 color=color,
                                 linewidth=2,
                                 linestyle='--',
                                 label=f"{name} (APT)",
                                 alpha=0.8,
                                 marker='s',
                                 markersize=3,
                                 markevery=max(1, len(epochs)//20))

        self.ax_loss.legend(loc='upper right',
                           framealpha=0.3,
                           edgecolor=CYBER_COLORS['grid'],
                           fontsize=8)

    def update_activity_timeline(self):
        """更新活跃度时间线"""
        self.ax_timeline.clear()
        self.setup_activity_timeline()

        if not self.trainings:
            return

        # 按最后更新时间排序
        sorted_trainings = sorted(self.trainings.items(),
                                 key=lambda x: x[1]['last_update'],
                                 reverse=True)

        names = []
        ages = []
        colors = []

        for name, info in sorted_trainings:
            names.append(name)
            ages.append(info['age_minutes'])
            colors.append(info['status_color'])

        # 横向条形图
        self.ax_timeline.barh(names, ages, color=colors, alpha=0.7)
        self.ax_timeline.set_xlabel('Time since last update (minutes)', color='white')

        # 标记活跃阈值
        self.ax_timeline.axvline(x=1, color=CYBER_COLORS['success'],
                                linestyle='--', alpha=0.3, label='Active threshold')
        self.ax_timeline.axvline(x=5, color=CYBER_COLORS['warning'],
                                linestyle='--', alpha=0.3, label='Recent threshold')

        self.ax_timeline.legend(loc='upper right', fontsize=8, framealpha=0.3)

    def update_leaderboard(self):
        """更新排行榜"""
        self.ax_leaderboard.clear()
        self.ax_leaderboard.axis('off')
        self.ax_leaderboard.set_facecolor(CYBER_COLORS['bg'])

        if not self.trainings:
            return

        # 计算每个训练的最佳loss
        rankings = []
        for name, info in self.trainings.items():
            data = info['data']

            best_loss = float('inf')

            if 'control_losses' in data and data['control_losses']:
                best_loss = min(best_loss, min(data['control_losses']))

            if 'autopoietic_losses' in data and data['autopoietic_losses']:
                best_loss = min(best_loss, min(data['autopoietic_losses']))

            if best_loss < float('inf'):
                rankings.append((name, best_loss, info['data'].get('current_epoch', 0)))

        # 排序
        rankings.sort(key=lambda x: x[1])

        # 构建排行榜文本
        text = "┌" + "─" * 38 + "┐\n"
        text += "│    🏆 LEADERBOARD - Best Loss    │\n"
        text += "├" + "─" * 38 + "┤\n"

        for i, (name, loss, epoch) in enumerate(rankings[:5], 1):
            # 奖牌
            if i == 1:
                medal = "🥇"
            elif i == 2:
                medal = "🥈"
            elif i == 3:
                medal = "🥉"
            else:
                medal = f" {i}."

            # 截断名称
            display_name = name[:20] if len(name) <= 20 else name[:17] + "..."

            text += f"│ {medal} {display_name:20} {loss:6.4f} │\n"

        text += "└" + "─" * 38 + "┘"

        self.ax_leaderboard.text(0.5, 0.5, text,
                                fontsize=11,
                                color=CYBER_COLORS['warning'],
                                family='monospace',
                                ha='center', va='center')

    def update_all(self):
        """更新所有面板"""
        # 发现训练
        self.discover_trainings()

        # 更新各个面板
        self.update_training_list()
        self.update_loss_comparison()
        self.update_activity_timeline()
        self.update_leaderboard()

        # 时间戳
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        self.fig.text(0.99, 0.01, f'Last Scan: {timestamp}',
                     ha='right', va='bottom',
                     fontsize=8, color=CYBER_COLORS['grid'],
                     style='italic')

        plt.draw()

    def run(self):
        """启动监控"""
        print("🎯 启动多训练监控面板...")
        print(f"   搜索目录: {', '.join(self.search_dirs)}")
        print(f"   刷新频率: {self.refresh_rate}秒")
        print("\n按 Ctrl+C 停止监控\n")

        # 初始扫描
        self.discover_trainings()
        print(f"🔍 发现 {len(self.trainings)} 个训练\n")

        for name, info in self.trainings.items():
            print(f"   {info['status']} {name}")

        # 创建动画
        def animate(frame):
            self.update_all()
            return []

        anim = FuncAnimation(
            self.fig,
            animate,
            interval=int(self.refresh_rate * 1000),
            blit=False,
            cache_frame_data=False
        )

        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='多训练监控面板')
    parser.add_argument('--dirs', nargs='+',
                       help='搜索目录列表')
    parser.add_argument('--refresh', type=float, default=3.0,
                       help='刷新频率（秒）')

    args = parser.parse_args()

    monitor = MultiTrainingMonitor(
        search_dirs=args.dirs,
        refresh_rate=args.refresh
    )

    monitor.run()
