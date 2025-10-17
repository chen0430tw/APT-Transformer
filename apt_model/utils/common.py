# 文件: apt_model/utils/common.py
import os
from datetime import datetime
from apt_model.utils import set_seed, get_device
from apt_model.utils.logging_utils import setup_logging
from apt_model.utils.language_manager import LanguageManager

def _initialize_common(args):
    """
    Initialize common components needed by multiple commands

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
    log_level = "DEBUG" if args.verbose else "INFO"
    log_file = f"apt_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_file=log_file, level=log_level)
    
    # Log basic information
    logger.info(lang_manager.get("welcome"))
    logger.info(lang_manager.get("language") + f": {args.language}")
    logger.info(lang_manager.get("training.device").format(device))
    
    return logger, lang_manager, device