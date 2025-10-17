#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model (自生成变换器) CLI Module
Command-line interface for APT model training and evaluation tool
"""

from apt_model.cli.parser import parse_arguments
from apt_model.cli.commands import (
    run_train_command,
    run_train_custom_command,
    run_chat_command,
    run_evaluate_command,
    run_visualize_command,
    run_clean_cache_command,
    run_estimate_command,
    show_help
)

__all__ = [
    'parse_arguments',
    'run_train_command',
    'run_train_custom_command',
    'run_chat_command',
    'run_evaluate_command',
    'run_visualize_command',
    'run_clean_cache_command',
    'run_estimate_command',
    'show_help',
    'execute_command'
]

def execute_command(args):
    """
    Execute the appropriate command based on arguments
    
    Parameters:
        args: Command line arguments from argparse
    
    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    # Map actions to command functions
    command_map = {
        'train': run_train_command,
        'train-custom': run_train_custom_command,
        'chat': run_chat_command,
        'eval': run_evaluate_command,
        'evaluate': run_evaluate_command,
        'visualize': run_visualize_command,
        'clean-cache': run_clean_cache_command,
        'estimate': run_estimate_command
    }
    
    # Get the appropriate command function or default to help
    command_func = command_map.get(args.action, show_help)
    
    # Execute the command
    try:
        return command_func(args)
    except Exception as e:
        import traceback
        print(f"Error executing command '{args.action}': {e}")
        print(traceback.format_exc())
        return 1