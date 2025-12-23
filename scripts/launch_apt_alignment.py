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


def get_main_action():
    """获取主要操作"""
    print_header("选择操作")

    actions = {
        '1': {
            'name': '📦 准备数据集',
            'type': 'prepare_data',
            'desc': '下载和预处理HuggingFace数据集'
        },
        '2': {
            'name': '🚀 开始训练',
            'type': 'train',
            'desc': '启动APT对齐训练流程'
        },
        '3': {
            'name': '📊 查看数据集信息',
            'type': 'show_datasets',
            'desc': '显示已准备的数据集统计'
        }
    }

    print("操作:")
    for key, action in actions.items():
        print(f"  {CYAN}{key}{RESET}. {action['name']}")
        print(f"     {action['desc']}")

    choice = input(f"\n{CYAN}选择操作 [1-3]:{RESET} ").strip()

    if choice not in actions:
        print_error("无效选择，使用默认: 开始训练")
        choice = '2'

    return actions[choice]


def get_training_mode():
    """获取训练模式"""
    print_header("选择训练模式")

    modes = {
        '1': {
            'name': '标准对齐 (SFT → GRPO)',
            'skip': 'dpo,loyalty,storm',
            'desc': '基础指令微调 + 策略优化',
            'datasets': ['coig-cqia', 'hh-rlhf']
        },
        '2': {
            'name': '忠诚度训练 (Loyalty)',
            'skip': 'sft,dpo,grpo,storm',
            'desc': '区分主人 vs 大众响应',
            'datasets': ['loyalty_template']
        },
        '3': {
            'name': '暴风雨训练 (Storm)',
            'skip': 'sft,dpo,grpo,loyalty',
            'desc': '动态推理 + 内化CoT',
            'datasets': ['s1k']
        },
        '4': {
            'name': '完整流程 (All Stages)',
            'skip': '',
            'desc': 'SFT → GRPO → Loyalty → Storm',
            'datasets': ['coig-cqia', 'hh-rlhf', 's1k', 'loyalty_template']
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


def prepare_datasets():
    """数据集准备交互式界面"""
    print_header("📦 数据集准备")

    print("推荐数据集:")
    print(f"  {CYAN}1{RESET}. COIG-CQIA (48K中文指令) - {BOLD}SFT阶段{RESET}")
    print(f"  {CYAN}2{RESET}. simplescaling/s1K (1K推理traces) - {BOLD}Storm阶段{RESET}")
    print(f"  {CYAN}3{RESET}. HH-RLHF (160K偏好数据) - {BOLD}GRPO阶段{RESET}")
    print(f"  {CYAN}4{RESET}. 弱智吧子集 (从COIG-CQIA提取) - {BOLD}提升推理{RESET}")
    print(f"  {CYAN}5{RESET}. 忠诚度模板 (基于HH-RLHF) - {BOLD}Loyalty阶段{RESET}")
    print(f"  {CYAN}6{RESET}. 下载全部推荐数据集")

    choice = input(f"\n{CYAN}选择要准备的数据集 [1-6]:{RESET} ").strip()

    # 构建数据准备命令
    prepare_script = PROJECT_ROOT / "scripts" / "prepare_apt_datasets.py"

    if choice == '1':
        cmd = [sys.executable, str(prepare_script), '--sft']
    elif choice == '2':
        cmd = [sys.executable, str(prepare_script), '--cot']
    elif choice == '3':
        cmd = [sys.executable, str(prepare_script), '--dpo']
    elif choice == '4':
        cmd = [sys.executable, str(prepare_script), '--ruozhiba']
    elif choice == '5':
        cmd = [sys.executable, str(prepare_script), '--loyalty-template']
    elif choice == '6':
        cmd = [sys.executable, str(prepare_script), '--all', '--ruozhiba', '--loyalty-template']
    else:
        print_error("无效选择")
        return

    # 显示命令
    print_header("执行命令")
    print(f"{CYAN}{' '.join(cmd)}{RESET}\n")

    # 执行
    try:
        subprocess.run(cmd, check=True)
        print_success("\n数据集准备完成！")
    except subprocess.CalledProcessError as e:
        print_error(f"数据集准备失败: {e}")
    except KeyboardInterrupt:
        print_warning("\n操作被中断")


def show_dataset_info():
    """显示数据集信息"""
    print_header("📊 数据集信息")

    data_dir = PROJECT_ROOT / "data" / "apt_datasets"

    if not data_dir.exists():
        print_warning(f"数据目录不存在: {data_dir}")
        print_info("请先使用 '准备数据集' 功能下载数据")
        return

    import json

    # 扫描数据文件
    datasets = []
    for file_path in data_dir.glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                datasets.append({
                    'name': file_path.stem,
                    'path': file_path,
                    'size': len(data),
                    'file_size': file_path.stat().st_size / (1024 * 1024)  # MB
                })
        except Exception as e:
            print_warning(f"无法读取 {file_path.name}: {e}")

    if not datasets:
        print_warning("未找到任何数据集")
        print_info("请先使用 '准备数据集' 功能下载数据")
        return

    # 显示统计
    print(f"数据目录: {CYAN}{data_dir}{RESET}\n")
    print(f"{'数据集名称':<30} {'样本数':>10} {'文件大小':>12}")
    print("-" * 55)

    total_samples = 0
    total_size = 0

    for ds in sorted(datasets, key=lambda x: x['size'], reverse=True):
        print(f"{ds['name']:<30} {ds['size']:>10,} {ds['file_size']:>10.2f} MB")
        total_samples += ds['size']
        total_size += ds['file_size']

    print("-" * 55)
    print(f"{'总计':<30} {total_samples:>10,} {total_size:>10.2f} MB")


def check_required_datasets(mode):
    """检查所需数据集是否已准备"""
    data_dir = PROJECT_ROOT / "data" / "apt_datasets"

    if not data_dir.exists():
        print_warning("\n数据集目录不存在，建议先准备数据集")
        prepare = input(f"{CYAN}是否现在准备数据集? [y/N]:{RESET} ").strip().lower()
        if prepare in ['y', 'yes']:
            prepare_datasets()
            return True
        return False

    # 检查所需文件
    required = mode.get('datasets', [])
    missing = []

    for dataset_name in required:
        file_path = data_dir / f"{dataset_name}_train.json"
        if not file_path.exists():
            missing.append(dataset_name)

    if missing:
        print_warning(f"\n缺少数据集: {', '.join(missing)}")
        prepare = input(f"{CYAN}是否现在准备缺失的数据集? [y/N]:{RESET} ").strip().lower()
        if prepare in ['y', 'yes']:
            prepare_datasets()
            return True

    return True


def build_command(mode):
    """构建训练命令"""
    print_header("训练配置")

    cmd = [
        sys.executable,
        "training/train_apt_alignment.py"
    ]

    # 数据集目录
    data_dir = PROJECT_ROOT / "data" / "apt_datasets"

    # 根据模式添加参数
    if '标准对齐' in mode['name']:
        print(f"模式: {BOLD}标准对齐{RESET}")
        print(f"  → SFT数据集: coig-cqia_train.json")
        print(f"  → GRPO prompts: hh-rlhf_train.json")

        cmd.extend([
            '--sft-data', str(data_dir / 'coig-cqia_train.json'),
            '--prompts', str(data_dir / 'hh-rlhf_train.json')
        ])

    elif '忠诚度' in mode['name']:
        print(f"模式: {BOLD}忠诚度训练{RESET}")
        print(f"  → 忠诚度模板: loyalty_template.json")
        print(f"  → 奖励加成: +2.0")

        cmd.extend([
            '--loyalty-data', str(data_dir / 'loyalty_template.json'),
            '--owner-bonus', '2.0'
        ])

    elif '暴风雨' in mode['name']:
        print(f"模式: {BOLD}暴风雨训练{RESET}")
        print(f"  → 推理数据: s1k_train.json")
        print(f"  → 噪音比例: 0.3")
        print(f"  → 噪音策略: cosine")
        print(f"  → 内化CoT: 是")

        cmd.extend([
            '--reasoning-data', str(data_dir / 's1k_train.json'),
            '--noise-ratio', '0.3',
            '--noise-schedule', 'cosine',
            '--internalize-cot'
        ])

    elif '完整流程' in mode['name']:
        print(f"模式: {BOLD}完整流程{RESET}")
        print(f"  → 包含所有阶段 (SFT → GRPO → Loyalty → Storm)")

        cmd.extend([
            '--sft-data', str(data_dir / 'coig-cqia_train.json'),
            '--prompts', str(data_dir / 'hh-rlhf_train.json'),
            '--loyalty-data', str(data_dir / 'loyalty_template.json'),
            '--reasoning-data', str(data_dir / 's1k_train.json')
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

    # 选择主要操作
    action = get_main_action()

    if action['type'] == 'prepare_data':
        # 数据集准备
        prepare_datasets()

    elif action['type'] == 'show_datasets':
        # 显示数据集信息
        show_dataset_info()

    elif action['type'] == 'train':
        # 训练流程
        mode = get_training_mode()

        # 检查所需数据集
        if not check_required_datasets(mode):
            print_warning("数据集检查未通过，训练已取消")
            sys.exit(0)

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
