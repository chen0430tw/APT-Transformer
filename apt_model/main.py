#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model (自生成变换器) 训练工具
一个功能丰富的模型训练和评估工具

重构后的主入口：
- 使用命令注册系统
- 支持插件动态添加命令
- 简化的命令分发逻辑
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')  # 设置标准输出编码为 UTF-8
sys.stderr.reconfigure(encoding='utf-8')
import traceback
from datetime import datetime

from apt_model.cli.parser import parse_arguments
from apt_model.utils.logging_utils import setup_logging
from apt_model.utils.resource_monitor import ResourceMonitor
from apt_model.utils.language_manager import LanguageManager
from apt_model.utils.hardware_check import check_hardware_compatibility
from apt_model.utils.cache_manager import CacheManager
from apt_model.config.apt_config import APTConfig
from apt_model.utils import set_seed, get_device
from apt_model.cli.command_registry import command_registry, execute_command


def print_banner(lang_manager):
    """显示欢迎横幅"""
    _ = lambda key: lang_manager.get(key)
    banner = f"""
    ╔═══════════════════════════════════════════════╗
    ║          {_("app_name")}          ║
    ╠═══════════════════════════════════════════════╣
    ║ {_("amber.training")}                          ║
    ╚═══════════════════════════════════════════════╝
    """
    print(banner)


def initialize_system(args):
    """
    初始化系统配置

    参数:
        args: 命令行参数

    返回:
        tuple: (logger, lang_manager, device)
    """
    # 设置随机种子
    set_seed(args.seed)

    # 设置设备
    device = get_device(args.force_cpu)

    # 初始化语言管理器
    lang_manager = LanguageManager(args.language, args.language_file)
    _ = lambda key, *params: lang_manager.get(key).format(*params) if params else lang_manager.get(key)

    # 初始化日志记录
    log_file = f"apt_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_file=log_file)
    logger.info(_("welcome"))
    logger.info(_("language") + f": {args.language}")
    logger.info(_("training.device").format(device))

    # 显示欢迎横幅
    print_banner(lang_manager)

    # 初始化缓存管理器
    cache_manager = CacheManager(cache_dir=args.cache_dir, logger=logger)

    # 创建资源监控器
    resource_monitor = ResourceMonitor(logger=logger)

    # 检查硬件兼容性（仅在需要模型的命令时）
    if args.action in ['train', 'train-custom', 'chat', 'evaluate']:
        model_config = APTConfig()
        check_hardware_compatibility(model_config, logger)

    return logger, lang_manager, device


def show_default_help():
    """显示默认帮助信息（当没有指定命令时）"""
    print("欢迎使用APT模型！")
    print("\n可用命令:")

    # 从命令注册中心获取所有命令
    commands_by_category = command_registry.get_commands_by_category(include_placeholders=False)

    for category in sorted(commands_by_category.keys()):
        print(f"\n{category.upper()}:")
        for metadata in commands_by_category[category]:
            help_text = metadata.help_text or "无说明"
            print(f"  {metadata.name:<20} - {help_text}")
            if metadata.aliases:
                print(f"    {'':18}别名: {', '.join(metadata.aliases)}")

    print("\n获取详细帮助:")
    print("  python -m apt_model help")
    print("\n示例:")
    print("  python -m apt_model train")
    print("  python -m apt_model chat")
    print("  python -m apt_model evaluate --model-path my_model")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()

    # 初始化系统
    logger, lang_manager, device = initialize_system(args)

    # 根据动作选择功能
    if args.action:
        # 检查命令是否存在
        if not command_registry.has_command(args.action):
            logger.warning(f"未知的动作: {args.action}")
            print(f"错误: 未知命令 '{args.action}'")
            print("使用 'python -m apt_model help' 查看可用命令")
            sys.exit(1)

        # 执行命令
        try:
            exit_code = execute_command(args.action, args)
            sys.exit(exit_code)
        except Exception as e:
            logger.error(f"执行命令 '{args.action}' 时出错: {e}")
            logger.error(traceback.format_exc())
            print(f"错误: {e}")
            sys.exit(1)
    else:
        # 显示默认帮助信息
        show_default_help()


if __name__ == "__main__":
    main()
