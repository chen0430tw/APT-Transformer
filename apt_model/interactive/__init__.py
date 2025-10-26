#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive module for the APT Model (自生成变换器)
Provides functionality for interacting with trained models through chat interfaces.
"""

from .chat import chat_with_model

# Try to import Admin Mode (optional feature)
try:
    from .admin_mode import APTAdminMode
    _admin_mode_available = True
except ImportError:
    _admin_mode_available = False
    APTAdminMode = None

__all__ = ['chat_with_model']
if _admin_mode_available:
    __all__.append('APTAdminMode')

# Version of the interactive module
__version__ = '0.1.0'

def get_available_interactive_modes():
    """
    Returns a list of available interactive modes
    
    Returns:
        list: Available interactive modes
    """
    return ['chat', 'prompt_completion']

def initialize_interactive_session(model=None, tokenizer=None, config=None, logger=None):
    """
    Initialize an interactive session with the specified components.
    
    Args:
        model: The APT model to interact with
        tokenizer: The tokenizer for the model
        config: Configuration for the interactive session
        logger: Logger for the session
        
    Returns:
        dict: Session context with model, tokenizer, and settings
    """
    from datetime import datetime
    
    session = {
        'model': model,
        'tokenizer': tokenizer,
        'config': config,
        'logger': logger,
        'history': [],
        'session_id': f"apt_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'start_time': datetime.now(),
        'settings': {
            'temperature': 0.7,
            'top_p': 0.9,
            'max_length': 50,
            'keep_history': True,
            'max_history_length': 6
        }
    }
    
    if logger:
        logger.info(f"Interactive session initialized: {session['session_id']}")
    
    return session