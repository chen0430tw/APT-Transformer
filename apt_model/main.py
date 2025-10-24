#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model (自生成变换器) 训练工具
一个功能丰富的模型训练和评估工具
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
from apt_model.training.trainer import train_model
from apt_model.data.external_data import train_with_external_data
from apt_model.interactive.chat import chat_with_model
from apt_model.evaluation.model_evaluator import evaluate_model
from apt_model.utils.visualization import ModelVisualizer
from apt_model.utils.time_estimator import TrainingTimeEstimator
from apt_model.modeling.chinese_tokenizer_integration import (
    get_appropriate_tokenizer, 
    detect_language,
    is_chinese_text
)
from apt_model.cli.commands import (
    run_visualize_command,
    run_estimate_command,
    run_clean_cache_command,
    run_info_command,
    run_list_command,
    run_prune_command,
    run_size_command,
    run_test_command,
    run_compare_command,
    run_train_hf_command,
    run_distill_command,
    run_train_reasoning_command,
    run_process_data_command,
    run_backup_command,
    run_upload_command,
    run_export_ollama_command,
    run_train_command,
    run_train_custom_command,
    run_chat_command,
    run_evaluate_command,
    show_help
)
from apt_model.utils.common import _initialize_common

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置随机种子
    from apt_model.utils import set_seed
    set_seed(args.seed)
    
    # 设置设备
    from apt_model.utils import get_device
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
    banner = f"""
    ╔═══════════════════════════════════════════════╗
    ║          {_("app_name")}          ║
    ╠═══════════════════════════════════════════════╣
    ║ {_("amber.training")}                          ║
    ╚═══════════════════════════════════════════════╝
    """
    print(banner)
    
    # 初始化缓存管理器
    cache_manager = CacheManager(cache_dir=args.cache_dir, logger=logger)
    
    # 创建资源监控器
    resource_monitor = ResourceMonitor(logger=logger)
    
    # 检查硬件兼容性
    model_config = APTConfig()
    check_hardware_compatibility(model_config, logger)

    # 根据动作选择功能
    if args.action:
        # 训练相关命令
        if args.action == "train":
            # 调用更新后的命令执行函数
            exit_code = run_train_command(args)
            sys.exit(exit_code)
                
        elif args.action == "train-custom":
            # 调用更新后的自定义训练命令
            exit_code = run_train_custom_command(args)
            sys.exit(exit_code)
            
        elif args.action == "chat":
            # 调用更新后的聊天命令
            exit_code = run_chat_command(args)
            sys.exit(exit_code)
            
        elif args.action == "evaluate" or args.action == "eval":
            # 调用更新后的评估命令
            exit_code = run_evaluate_command(args)
            sys.exit(exit_code)

        elif args.action == "visualize":
            logger.info("生成模型评估可视化图表...")
            exit_code = run_visualize_command(args)
            sys.exit(exit_code)

        elif args.action == "estimate":
            logger.info("开始训练时间估算...")
            exit_code = run_estimate_command(args)
            sys.exit(exit_code)

        elif args.action == "clean-cache":
            logger.info("开始清理缓存...")
            exit_code = run_clean_cache_command(args)
            sys.exit(exit_code)

        elif args.action == "info":
            logger.info("显示模型/数据详细信息...")
            exit_code = run_info_command(args)
            sys.exit(exit_code)

        elif args.action == "list":
            logger.info("列出可用资源...")
            exit_code = run_list_command(args)
            sys.exit(exit_code)

        elif args.action == "prune":
            logger.info("删除旧模型或数据...")
            exit_code = run_prune_command(args)
            sys.exit(exit_code)

        elif args.action == "size":
            logger.info("计算数据或模型大小...")
            exit_code = run_size_command(args)
            sys.exit(exit_code)

        elif args.action == "test":
            logger.info("开始测试模型...")
            exit_code = run_test_command(args)
            sys.exit(exit_code)

        elif args.action == "compare":
            logger.info("比较模型性能...")
            exit_code = run_compare_command(args)
            sys.exit(exit_code)

        elif args.action == "train-hf":
            logger.info("训练 Hugging Face 兼容模型...")
            exit_code = run_train_hf_command(args)
            sys.exit(exit_code)

        elif args.action == "distill":
            logger.info("蒸馏模型...")
            exit_code = run_distill_command(args)
            sys.exit(exit_code)

        elif args.action == "train-reasoning":
            logger.info("训练逻辑推理能力模型...")
            exit_code = run_train_reasoning_command(args)
            sys.exit(exit_code)

        elif args.action == "process-data":
            logger.info("处理数据集...")
            exit_code = run_process_data_command(args)
            sys.exit(exit_code)

        elif args.action == "backup":
            logger.info("备份模型或数据...")
            exit_code = run_backup_command(args)
            sys.exit(exit_code)

        elif args.action == "upload":
            logger.info("上传模型或数据...")
            exit_code = run_upload_command(args)
            sys.exit(exit_code)

        elif args.action == "export-ollama":
            logger.info("导出模型到 Ollama 格式...")
            exit_code = run_export_ollama_command(args)
            sys.exit(exit_code)

        elif args.action == "config":
            # 配置管理命令
            from apt_model.cli.commands import run_config_command
            exit_code = run_config_command(args)
            sys.exit(exit_code)

        elif args.action == "debug":
            # Debug诊断命令
            from apt_model.cli.commands import run_debug_command
            exit_code = run_debug_command(args)
            sys.exit(exit_code)

        elif args.action == "help":
            # 显示帮助信息
            exit_code = show_help(args)
            sys.exit(exit_code)

        else:
            logger.warning(f"未知的动作: {args.action}")
            print("未知的动作，请使用 help 获取帮助信息。")
            sys.exit(1)
    else:
        # 显示帮助信息
        print("欢迎使用APT模型！")
        print("\n可用命令:")
        print("  train         - 训练模型")
        print("    --tokenizer-type chinese-char   - 使用字符级中文分词器")
        print("    --tokenizer-type chinese-word   - 使用词级中文分词器")
        print("    --model-language zh            - 明确指定中文训练")
        print("  train-custom  - 使用自定义数据训练模型")
        print("  chat          - 与模型交互对话")
        print("  evaluate      - 评估模型性能")
        print("  visualize     - 生成模型评估可视化图表")
        print("  estimate      - 估算训练时间")
        print("  clean-cache   - 清理缓存文件")
        print("  info          - 显示模型/数据详细信息")
        print("  list          - 列出可用资源")
        print("  prune         - 删除旧模型或数据")
        print("  size          - 计算数据或模型大小")
        print("  test          - 测试模型")
        print("  compare       - 比较模型性能")
        print("  train-hf      - 训练 Hugging Face 兼容模型")
        print("  distill       - 蒸馏模型")
        print("  train-reasoning - 训练逻辑推理能力模型")
        print("  process-data  - 处理数据集")
        print("  backup        - 备份模型或数据")
        print("  upload        - 上传模型或数据")
        print("  export-ollama - 导出模型到 Ollama 格式")
        print("  help          - 显示更多帮助信息")
        print("\n示例:")
        print("  python -m apt_model train --tokenizer-type chinese-char")
        print("  python -m apt_model train-custom --data-path my_chinese_data.txt --model-language zh")
        print("  python -m apt_model chat --model-path my_chinese_model")
        print("  python -m apt_model evaluate --model-path my_model")
    
if __name__ == "__main__":
    main()