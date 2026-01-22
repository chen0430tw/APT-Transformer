# 文件: apt_model/utils/common.py
import os
from datetime import datetime
from apt_model.utils import set_seed, get_device
from apt_model.utils.logging_utils import setup_logging
from apt_model.utils.language_manager import LanguageManager
from apt.core.config.settings_manager import settings

def _initialize_common(args):
    """
    Initialize common components needed by multiple commands

    自动读取全局配置文件中的debug设置
    优先级: 命令行参数 > 全局配置 > 默认值

    Parameters:
        args: Command line arguments

    Returns:
        tuple: (logger, language_manager, device)
    """
    # Set random seed
    set_seed(args.seed)

    # Set device
    device = get_device(args.force_cpu)

    # Initialize language manager
    lang_manager = LanguageManager(args.language, args.language_file)

    # Initialize logging
    # 优先级: 命令行 --verbose > 全局配置 debug.enabled > 默认INFO
    if hasattr(args, 'verbose') and args.verbose:
        # 命令行明确指定 --verbose
        log_level = "DEBUG"
    else:
        # 读取全局配置中的debug设置
        log_level = settings.get_log_level()

    # 如果全局配置启用了debug，显示提示信息
    if settings.get_debug_enabled() and not (hasattr(args, 'verbose') and args.verbose):
        print(f"🐛 Debug模式已启用 (配置文件: {settings.CONFIG_FILE})")
        print(f"   日志级别: {log_level}")
        print(f"   使用 'python -m apt_model config --set-debug off' 可以关闭")
        print()

    log_file = f"apt_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_file=log_file, level=log_level)

    # Log basic information
    logger.info(lang_manager.get("welcome"))
    logger.info(lang_manager.get("language") + f": {args.language}")
    logger.info(lang_manager.get("training.device").format(device))
    logger.debug(f"Debug mode: {log_level == 'DEBUG'}")
    logger.debug(f"Settings file: {settings.CONFIG_FILE}")

    return logger, lang_manager, device