#!/usr/bin/env python3
"""
🚀 APT对齐训练启动器
快速启动APT推理与对齐训练

预设配置:
1. 标准对齐 (SFT → GRPO)
2. 忠诚度训练 (Loyalty)
3. 暴风雨训练 (Storm - 动态推理)
4. 完整流程 (All stages)

作者: chen0430tw
日期: 2024-12-23
"""

import os
import sys
import subprocess
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
os.chdir(PROJECT_ROOT)

# ANSI颜色
CYAN = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_header(text):
    """打印标题"""
    print(f"\n{CYAN}{BOLD}{'='*60}")
    print(f"{text}")
    print(f"{'='*60}{RESET}\n")


def print_success(text):
    """打印成功信息"""
    print(f"{GREEN}✓{RESET} {text}")


def print_warning(text):
    """打印警告"""
    print(f"{YELLOW}⚠{RESET}  {text}")


def print_error(text):
    """打印错误"""
    print(f"{RED}✗{RESET} {text}")


def check_dependencies():
    """检查Python依赖"""
    print_header("检查Python依赖...")

    try:
        import torch
        print_success(f"PyTorch {torch.__version__}")
    except ImportError:
        print_error("PyTorch 未安装")
        return False

    try:
        import numpy
        print_success(f"NumPy {numpy.__version__}")
    except ImportError:
        print_error("NumPy 未安装")
        return False

    return True


def get_training_mode():
    """获取训练模式"""
    print_header("选择训练模式")

    modes = {
        '1': {
            'name': '标准对齐 (SFT → GRPO)',
            'skip': 'dpo,loyalty,storm',
            'desc': '基础指令微调 + 策略优化'
        },
        '2': {
            'name': '忠诚度训练 (Loyalty)',
            'skip': 'sft,dpo,grpo,storm',
            'desc': '区分主人 vs 大众响应'
        },
        '3': {
            'name': '暴风雨训练 (Storm)',
            'skip': 'sft,dpo,grpo,loyalty',
            'desc': '动态推理 + 内化CoT'
        },
        '4': {
            'name': '完整流程 (All Stages)',
            'skip': '',
            'desc': 'SFT → GRPO → Loyalty → Storm'
        }
    }

    print("训练模式:")
    for key, mode in modes.items():
        print(f"  {CYAN}{key}{RESET}. {BOLD}{mode['name']}{RESET}")
        print(f"     {mode['desc']}")

    choice = input(f"\n{CYAN}选择模式 [1-4]:{RESET} ").strip()

    if choice not in modes:
        print_error("无效选择，使用默认: 标准对齐")
        choice = '1'

    return modes[choice]


def build_command(mode):
    """构建训练命令"""
    print_header("训练配置")

    cmd = [
        sys.executable,
        "training/train_apt_alignment.py"
    ]

    # 数据集路径 (示例)
    data_dir = PROJECT_ROOT / "data"

    # 根据模式添加参数
    if '标准对齐' in mode['name']:
        print(f"模式: {BOLD}标准对齐{RESET}")
        print(f"  → SFT数据集: data/instructions.json (示例)")
        print(f"  → GRPO prompts: data/prompts.json (示例)")

        cmd.extend([
            '--sft-data', 'data/instructions.json',
            '--prompts', 'data/prompts.json'
        ])

    elif '忠诚度' in mode['name']:
        print(f"模式: {BOLD}忠诚度训练{RESET}")
        print(f"  → 主人数据: data/owner_prompts.json (示例)")
        print(f"  → 公众数据: data/public_prompts.json (示例)")
        print(f"  → 奖励加成: +2.0")

        cmd.extend([
            '--owner-data', 'data/owner_prompts.json',
            '--public-data', 'data/public_prompts.json',
            '--owner-bonus', '2.0'
        ])

    elif '暴风雨' in mode['name']:
        print(f"模式: {BOLD}暴风雨训练{RESET}")
        print(f"  → 推理数据: data/cot_examples.json (示例)")
        print(f"  → 噪音比例: 0.3")
        print(f"  → 噪音策略: cosine")
        print(f"  → 内化CoT: 是")

        cmd.extend([
            '--reasoning-data', 'data/cot_examples.json',
            '--noise-ratio', '0.3',
            '--noise-schedule', 'cosine',
            '--internalize-cot'
        ])

    elif '完整流程' in mode['name']:
        print(f"模式: {BOLD}完整流程{RESET}")
        print(f"  → 包含所有阶段 (SFT → GRPO → Loyalty → Storm)")

        cmd.extend([
            '--sft-data', 'data/instructions.json',
            '--prompts', 'data/prompts.json',
            '--owner-data', 'data/owner_prompts.json',
            '--public-data', 'data/public_prompts.json',
            '--reasoning-data', 'data/cot_examples.json'
        ])

    # 通用参数
    cmd.extend([
        '--output-dir', './apt_aligned_models',
        '--device', 'cuda'
    ])

    # 跳过阶段
    if mode['skip']:
        cmd.extend(['--skip', mode['skip']])

    return cmd


def main():
    print_header("🚀 APT对齐训练启动器")

    print(f"项目目录: {PROJECT_ROOT}\n")

    # 检查依赖
    if not check_dependencies():
        print_error("依赖检查失败，请安装缺失的包")
        sys.exit(1)

    # 选择模式
    mode = get_training_mode()

    # 构建命令
    cmd = build_command(mode)

    # 显示命令
    print_header("启动命令")
    print(f"{CYAN}{' '.join(cmd)}{RESET}\n")

    # 确认启动
    confirm = input(f"{YELLOW}是否开始训练? [y/N]:{RESET} ").strip().lower()

    if confirm not in ['y', 'yes']:
        print_warning("训练已取消")
        sys.exit(0)

    # 启动训练
    print_header("🚀 开始训练...")
    try:
        subprocess.run(cmd, check=True)
        print_success("训练完成！")
    except subprocess.CalledProcessError as e:
        print_error(f"训练失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print_warning("\n训练被中断")
        sys.exit(1)


if __name__ == "__main__":
    main()
