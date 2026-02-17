#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model (自生成变换器) Launcher (启动器)

统一的启动器入口：
- 初始化控制台核心
- 加载所有核心模块
- 分发命令到相应处理器
- 提供交互式和批处理模式
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')  # 设置标准输出编码为 UTF-8
sys.stderr.reconfigure(encoding='utf-8')
import traceback
from datetime import datetime

# 启用警告过滤器 - 美化错误消息
from apt_model.utils.warning_filter import enable_clean_warnings
enable_clean_warnings()

from apt_model.cli.parser import parse_arguments
from apt_model.utils.logging_utils import setup_logging
from apt_model.utils.language_manager import LanguageManager
from apt_model.utils import set_seed, get_device
from apt_model.console.core import initialize_console, get_console


def print_launcher_banner():
    """显示启动器横幅"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║          APT Launcher - Autopoietic Transformer Launcher          ║
    ║                    自生成变换器启动器                              ║
    ║                                                                   ║
    ║  Unified console system for APT model management                  ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def initialize_launcher(args):
    """
    初始化启动器

    参数:
        args: 命令行参数

    返回:
        tuple: (logger, console, device)
    """
    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    device = get_device(args.force_cpu)

    # 初始化日志记录
    log_file = f"apt_launcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_file=log_file)
    logger.info("APT Launcher starting...")
    logger.info(f"Device: {device}")
    logger.info(f"Language: {args.language}")

    # 构建控制台配置
    console_config = {
        'device': str(device),
        'language': args.language,
        'seed': args.seed,
        'cache_dir': getattr(args, 'cache_dir', '.cache'),
    }

    # 初始化控制台核心
    console = initialize_console(config=console_config, auto_start=True)
    logger.info("Console Core initialized")

    return logger, console, device


def show_default_help(console):
    """
    显示默认帮助信息（当没有指定命令时）

    参数:
        console: 控制台核心实例
    """
    print_launcher_banner()
    print("\n欢迎使用 APT Launcher！")
    print("\n这是一个统一的启动器，用于管理和启动所有 APT 功能。\n")

    # 从命令注册中心获取所有命令
    commands_by_category = console.command_registry.get_commands_by_category(include_placeholders=False)

    print("可用命令类别:")
    for category in sorted(commands_by_category.keys()):
        count = len(commands_by_category[category])
        print(f"  - {category.upper():<15} ({count} 个命令)")

    print("\n常用命令:")
    print("  console-status                - 显示控制台状态")
    print("  modules-list                  - 列出所有模块")
    print("  train                         - 训练模型")
    print("  chat                          - 交互式对话")
    print("  console-help                  - 显示帮助")

    print("\n获取详细信息:")
    print("  python -m apt_model console-commands      - 列出所有命令")
    print("  python -m apt_model console-help <cmd>    - 查看命令帮助")

    print("\n快速开始:")
    print("  python -m apt_model console-status        - 查看系统状态")
    print("  python -m apt_model modules-list          - 查看模块列表")
    print("  python -m apt_model train                 - 开始训练")
    print("  python -m apt_model debug                 - Debug诊断")
    print("  python -m apt_model config                - 配置管理")


def main():
    """
    Launcher 主函数

    启动器流程：
    1. 解析命令行参数
    2. 快速命令直接执行（help等）
    3. 其他命令初始化控制台核心
    4. 分发命令
    """
    try:
        # 解析命令行参数
        args = parse_arguments()

        # 快速命令 - 不需要初始化控制台（提升性能）
        FAST_COMMANDS = {'help', 'version', '--help', '-h', '--version', '-v'}

        if args.action in FAST_COMMANDS or args.action == 'help':
            # 直接显示帮助，不初始化控制台
            from apt_model.cli.commands import show_help
            show_help(args)
            sys.exit(0)

        if not args.action:
            # 没有命令，显示简化的帮助
            from apt_model.cli.commands import show_help
            show_help(args)
            sys.exit(0)

        # 初始化启动器（只有真正需要执行命令时才初始化）
        logger, console, device = initialize_launcher(args)

        # 根据动作选择功能
        if args.action:
            # 检查命令是否存在
            if not console.command_registry.has_command(args.action):
                logger.warning(f"未知的命令: {args.action}")
                print(f"错误: 未知命令 '{args.action}'")
                print("使用 'python -m apt_model console-help' 查看可用命令")
                sys.exit(1)

            # 执行命令
            logger.info(f"执行命令: {args.action}")
            try:
                exit_code = console.execute_command(args.action, args)
                logger.info(f"命令执行完成，退出码: {exit_code}")
                sys.exit(exit_code)
            except Exception as e:
                logger.error(f"执行命令 '{args.action}' 时出错: {e}")
                logger.error(traceback.format_exc())
                print(f"错误: {e}")
                sys.exit(1)
        else:
            # 显示默认帮助信息
            show_default_help(console)

    except KeyboardInterrupt:
        print("\n\n用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"启动器错误: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
