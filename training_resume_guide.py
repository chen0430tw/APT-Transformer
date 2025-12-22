#!/usr/bin/env python3
"""
为训练脚本添加恢复功能的补丁

用法:
1. 在train_control_experiment.py中添加--resume参数
2. 在test_hlbd_quick_learning.py中添加--resume参数
"""

# ============================================================================
# 对照实验训练恢复功能
# ============================================================================

def add_resume_to_control_experiment():
    """
    为train_control_experiment.py添加恢复功能

    使用方法:
    1. 训练中断后，找到最新的checkpoint
       ls -lt control_experiments/*.pt | head -1

    2. 恢复训练
       python train_control_experiment.py --resume control_experiments/control_epoch_25.pt
    """

    example_code = """
# 在ControlExperimentTrainer类中添加:

def load_checkpoint(self, checkpoint_path: str):
    \"\"\"从checkpoint恢复训练\"\"\"
    print(f"\\n📦 从checkpoint恢复训练: {checkpoint_path}")

    # 判断是对照组还是实验组
    is_control = 'control' in checkpoint_path

    ckpt = torch.load(checkpoint_path)

    if is_control:
        self.control_model.load_state_dict(ckpt['model_state_dict'])
        self.control_optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.control_losses = ckpt['losses']
        start_epoch = ckpt['epoch']
        print(f"   ✓ 对照组模型已恢复到epoch {start_epoch}")
    else:
        self.autopoietic_model.load_state_dict(ckpt['model_state_dict'])
        self.autopoietic_optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.autopoietic_losses = ckpt['losses']
        start_epoch = ckpt['epoch']
        print(f"   ✓ 实验组模型已恢复到epoch {start_epoch}")

    return start_epoch


# 在main()函数中添加参数:

parser.add_argument('--resume', type=str, default=None,
                    help='从checkpoint恢复训练 (路径到.pt文件)')

# 在训练循环前添加:

start_epoch = 0
if args.resume:
    # 恢复对照组
    control_resume = args.resume.replace('autopoietic', 'control')
    if Path(control_resume).exists():
        start_epoch = trainer.load_checkpoint(control_resume)

    # 恢复实验组
    autopoietic_resume = args.resume.replace('control', 'autopoietic')
    if Path(autopoietic_resume).exists():
        trainer.load_checkpoint(autopoietic_resume)

# 修改训练循环:

for epoch in range(start_epoch, args.epochs):  # 从start_epoch开始
    print(f"\\n📍 Epoch {epoch + 1}/{args.epochs}")
    ...
"""

    return example_code


# ============================================================================
# 多训练监控方案
# ============================================================================

def multi_training_monitor_design():
    """
    设计多训练监控系统

    方案1: 多窗口模式（当前）
    - 优点: 简单，每个窗口独立
    - 缺点: 需要手动开多个窗口

    方案2: 单窗口多面板
    - 优点: 统一界面，方便对比
    - 缺点: 屏幕空间有限

    方案3: 自动发现模式
    - 优点: 自动监控所有训练
    - 缺点: 实现复杂
    """

    # 方案1示例（当前实现）
    usage_current = """
# 同时监控3个实验
python visualize_training.py --log-dir exp1_baseline &
python visualize_training.py --log-dir exp2_large_lr &
python visualize_training.py --log-dir exp3_small_model &

# 每个窗口独立显示
"""

    # 方案2示例（单窗口多面板）
    usage_multi_panel = """
# 一个窗口显示多个实验
python visualize_training.py \\
    --experiments exp1_baseline exp2_large_lr exp3_small_model \\
    --mode compare

# 界面布局:
┌────────────────────────────────────────┐
│  Experiment Comparison                 │
├──────────┬──────────┬──────────────────┤
│  Exp 1   │  Exp 2   │  Combined Loss   │
│  Loss    │  Loss    │  Curves          │
├──────────┼──────────┼──────────────────┤
│  Exp 3   │  Stats   │  Best Model      │
│  Loss    │  Table   │  Highlight       │
└──────────┴──────────┴──────────────────┘
"""

    # 方案3示例（自动发现）
    usage_auto_discover = """
# 自动监控当前目录下所有训练
python visualize_training.py --auto-discover

# 会扫描:
# - control_experiments/
# - playground_*/
# - tests/saved_models/
#
# 并显示活跃的训练列表:
🔍 发现3个活跃训练:
  1. control_experiments (Epoch 42/100, 活跃)
  2. playground_wikitext (Epoch 15/50, 活跃)
  3. exp_custom (Epoch 8/20, 暂停)

选择要监控的训练: [1-3] 或 'all':
"""

    return {
        'current': usage_current,
        'multi_panel': usage_multi_panel,
        'auto_discover': usage_auto_discover
    }


# ============================================================================
# 快速解决方案：使用tmux多窗格
# ============================================================================

def tmux_multi_monitor_setup():
    """
    使用tmux实现多训练监控

    优点:
    - 不需要修改代码
    - 可以同时看到多个可视化
    - 可以远程连接
    """

    tmux_script = """#!/bin/bash
# 创建tmux会话用于多训练监控

# 创建新会话
tmux new-session -d -s apt_training

# 分割窗格
tmux split-window -h
tmux split-window -v
tmux select-pane -t 0
tmux split-window -v

# 窗格布局:
# ┌─────────┬─────────┐
# │ Train 1 │ Visual1 │
# ├─────────┼─────────┤
# │ Train 2 │ Visual2 │
# └─────────┴─────────┘

# 启动训练和可视化
tmux select-pane -t 0
tmux send-keys "python train_control_experiment.py --epochs 100 --save-dir exp1" C-m

tmux select-pane -t 1
tmux send-keys "sleep 5 && python visualize_training.py --log-dir exp1 --refresh 2" C-m

tmux select-pane -t 2
tmux send-keys "python train_control_experiment.py --epochs 100 --save-dir exp2 --lr 1e-3" C-m

tmux select-pane -t 3
tmux send-keys "sleep 5 && python visualize_training.py --log-dir exp2 --refresh 2" C-m

# 连接到会话
tmux attach-session -t apt_training

# 操作说明:
# Ctrl+B + 方向键: 切换窗格
# Ctrl+B + d: 分离会话（后台运行）
# tmux attach -t apt_training: 重新连接
"""

    return tmux_script


if __name__ == "__main__":
    print("训练恢复和多监控解决方案")
    print("=" * 60)

    print("\n1️⃣  训练恢复功能:")
    print(add_resume_to_control_experiment())

    print("\n2️⃣  多训练监控方案:")
    designs = multi_training_monitor_design()
    print("\n当前方案（多窗口）:")
    print(designs['current'])

    print("\n3️⃣  快速解决方案（tmux）:")
    print(tmux_multi_monitor_setup())
