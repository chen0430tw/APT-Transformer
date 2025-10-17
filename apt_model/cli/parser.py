#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Command-line argument parser for APT Model tool.
Handles all command-line options and arguments for the various functionalities.
"""

import argparse
import os

def parse_arguments():
    """
    Parse command-line arguments for APT Model tool.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='APT Model (自生成变换器) - Training and Evaluation Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m apt_model train                    - Train model with default parameters
  python -m apt_model train --epochs 10        - Train for 10 epochs
  python -m apt_model train --force-cpu        - Force CPU training
  python -m apt_model train --language zh      - Train with Chinese language
  python -m apt_model train --tokenizer-type chinese-char  - Use Chinese character tokenizer
  python -m apt_model test                     - Test model
  python -m apt_model chat                     - Interactive chat with model
  python -m apt_model chat --temperature 0.9   - Chat with higher temperature
  python -m apt_model train --language en_US   - Use English interface
  python -m apt_model evaluate                 - Evaluate model performance
"""
    )

    # Language related arguments
    parser.add_argument('--language', type=str, default="zh_CN",
                        choices=["zh_CN", "en_US"],
                        help='Interface language (default: zh_CN)')
    
    parser.add_argument('--language-file', type=str, default=None,
                        help='Custom language file path')
    
    # Action argument
    parser.add_argument('action', nargs='?', default=None, 
                        choices=['train', 'test', 'eval', 'evaluate', 'compare', 'chat', 
                                 'train-custom', 'train-hf', 'distill', 'train-reasoning', 
                                 'process-data', 'backup', 'upload', 'export-ollama', 
                                 'clean-cache', 'visualize', 'estimate'],
                        help='Action to perform')

    # ===============================
    #  Training related arguments
    # ===============================
    training_group = parser.add_argument_group('Training Options')
    training_group.add_argument('--epochs', type=int, default=20, 
                               help='Number of training epochs (default: 20)')

    training_group.add_argument('--batch-size', type=int, default=8, 
                               help='Training batch size (default: 8)')

    training_group.add_argument('--learning-rate', type=float, default=3e-5, 
                               help='Learning rate (default: 3e-5)')

    training_group.add_argument('--save-path', type=str, default="apt_model", 
                               help='Model save path (default: "apt_model")')

    # 重点：将原本的 model-path 改为可接受多个路径
    training_group.add_argument('--model-path', type=str, nargs='+', default=["apt_model"],
                               help='One or more model load paths (default: ["apt_model"])')

    training_group.add_argument('--force-cpu', action='store_true', 
                               help='Force CPU computation even if GPU is available')
    
    training_group.add_argument('--checkpoint-freq', type=int, default=1,
                               help='Checkpoint saving frequency in epochs (default: 1)')
    
    # ===============================
    #  Tokenizer related arguments (新增)
    # ===============================
    tokenizer_group = parser.add_argument_group('Tokenizer Options')
    tokenizer_group.add_argument('--tokenizer-type', type=str, default=None,
                                choices=['gpt2', 'chinese-char', 'chinese-word'],
                                help='Tokenizer type to use (default: automatic detection)')
    
    tokenizer_group.add_argument('--model-language', type=str, default=None,
                                choices=['en', 'zh'],
                                help='Language for model training (default: automatic detection)')
    
    tokenizer_group.add_argument('--vocab-size', type=int, default=None,
                                help='Tokenizer vocabulary size (default: based on tokenizer type)')
    
    # ===============================
    #  Generation related arguments
    # ===============================
    generation_group = parser.add_argument_group('Generation Options')
    generation_group.add_argument('--temperature', type=float, default=0.7, 
                                 help='Generation temperature parameter (default: 0.7)')
    generation_group.add_argument('--top-p', type=float, default=0.9, 
                                 help='Generation top-p parameter (default: 0.9)')
    generation_group.add_argument('--max-length', type=int, default=50, 
                                 help='Maximum generation length (default: 50)')
    
    # ===============================
    #  Data related arguments
    # ===============================
    data_group = parser.add_argument_group('Data Options')
    data_group.add_argument('--data-path', type=str, default=None,
                           help='External training data file path')
    data_group.add_argument('--max-samples', type=int, default=None,
                           help='Maximum number of samples to use from external data (default: all)')
    
    # ===============================
    #  Cache related arguments
    # ===============================
    cache_group = parser.add_argument_group('Cache Options')
    cache_group.add_argument('--cache-dir', type=str, default=None,
                            help='Cache directory path (default: ~/.apt_cache)')
    cache_group.add_argument('--clean-days', type=int, default=30,
                            help='Clean cache files older than specified days (default: 30)')
    
    # ===============================
    #  Evaluation related arguments
    # ===============================
    eval_group = parser.add_argument_group('Evaluation Options')
    eval_group.add_argument('--eval-sets', type=str, nargs='+', default=None,
                           help='Evaluation set names (default: all)')
    eval_group.add_argument('--num-eval-samples', type=int, default=None,
                           help='Number of samples per evaluation set (default: all)')
    eval_group.add_argument('--output-dir', type=str, default=None,
                           help='Evaluation output directory')
    eval_group.add_argument('--visualize-attention', action='store_true',
                           help='Generate attention heatmaps')
    
    # ===============================
    #  Multimodal related arguments
    # ===============================
    multimodal_group = parser.add_argument_group('Multimodal Options')
    multimodal_group.add_argument('--enable-image', action='store_true',
                                 help='Enable image modality')
    multimodal_group.add_argument('--enable-audio', action='store_true',
                                 help='Enable audio modality')
    
    # ===============================
    #  Resource monitoring arguments
    # ===============================
    monitor_group = parser.add_argument_group('Monitoring Options')
    monitor_group.add_argument('--monitor-resources', action='store_true',
                              help='Enable resource monitoring')
    monitor_group.add_argument('--log-interval', type=int, default=10,
                              help='Resource monitoring log interval in seconds (default: 10)')
    
    # ===============================
    #  Visualization related arguments
    # ===============================
    visual_group = parser.add_argument_group('Visualization Options')
    visual_group.add_argument('--create-plots', action='store_true',
                             help='Create training/evaluation plots')
    visual_group.add_argument('--plot-format', type=str, default='png',
                             choices=['png', 'pdf', 'svg', 'jpg'],
                             help='Plot format (default: png)')
    
    # ===============================
    #  Other arguments
    # ===============================
    other_group = parser.add_argument_group('Other Options')
    other_group.add_argument('--verbose', action='store_true',
                            help='Show detailed log information')
    other_group.add_argument('--seed', type=int, default=42,
                            help='Random seed (default: 42)')
    other_group.add_argument('--version', action='store_true',
                            help='Show version information')

    # 用于训练时间估算的 dataset-size 参数
    parser.add_argument('--dataset-size', type=int, default=1000,
                        help='Pseudo dataset size for training time estimation (default: 1000)')

    return parser.parse_args()


def get_config_from_args(args):
    """
    Create a configuration dictionary from parsed arguments.
    
    Args:
        args (argparse.Namespace): Parsed arguments
        
    Returns:
        dict: Configuration dictionary
    """
    # Convert arguments to a dictionary
    config = vars(args).copy()
    
    # Add additional configurations based on arguments
    config['device'] = 'cpu' if args.force_cpu else 'auto'
    
    # Set paths relative to the script location if needed
    if args.save_path and not os.path.isabs(args.save_path):
        config['save_path_abs'] = os.path.abspath(args.save_path)
    
    #
    # 注意，这里 model_path 是一个列表
    # 我们可以把第一个值当做默认值，也可以不做处理
    #
    if args.model_path:
        config['model_path_abs'] = [os.path.abspath(mp) for mp in args.model_path]
    
    return config

if __name__ == "__main__":
    # Test parser
    args = parse_arguments()
    config = get_config_from_args(args)
    print("Parsed arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    print("\nConfiguration:")
    for key, value in config.items():
        if key not in vars(args):
            print(f"  {key}: {value}")
