#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced error handling utilities for the APT Model training system.
"""

import traceback
import logging
from typing import Optional, Dict, Any, Callable, Type


def memory_cleanup():
    """Clean up memory by clearing cache and running garbage collector."""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class EnhancedErrorHandler:
    """
    Enhanced error handler that provides detailed error information 
    and automatic recovery capabilities.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, 
                checkpoint_manager: Any = None):
        """
        Initialize error handler.
        
        Args:
            logger: Logger for recording errors
            checkpoint_manager: Optional checkpoint manager for automatic checkpointing
        """
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.error_counts: Dict[str, int] = {}
        self.recovery_handlers: Dict[str, Callable] = {}
        self.max_recovery_attempts = 3
        
        # Register default recovery handlers
        self._register_default_handlers()
    
    def _register_default_handlers(self) -> None:
        """Register default error recovery handlers."""
        # Memory-related errors
        self.register_handler("CUDA out of memory", self._handle_memory_error)
        self.register_handler("MemoryError", self._handle_memory_error)
        
        # Connection-related errors
        self.register_handler("ConnectionError", self._handle_temporary_error)
        self.register_handler("TimeoutError", self._handle_temporary_error)
        self.register_handler("IOError", self._handle_temporary_error)
    
    def register_handler(self, error_pattern: str, handler: Callable) -> None:
        """
        Register a custom error recovery handler.
        
        Args:
            error_pattern: Error message pattern to match
            handler: Handler function to call for recovery
        """
        self.recovery_handlers[error_pattern] = handler
    
    def handle_exception(self, exception: Exception, context: str = "") -> bool:
        """
        Handle an exception, log it, and attempt recovery if possible.
        
        Args:
            exception: The exception object
            context: Context information for the error
            
        Returns:
            bool: Whether execution should continue (recovery was successful)
        """
        error_type = type(exception).__name__
        error_msg = str(exception)
        
        # Track error occurrences
        if error_type not in self.error_counts:
            self.error_counts[error_type] = 0
        self.error_counts[error_type] += 1
        
        # Log the error
        if self.logger:
            self.logger.error(f"{context} Error: {error_type}: {error_msg}")
            self.logger.error(f"Detailed traceback:\n{traceback.format_exc()}")
        
        # Attempt recovery
        should_continue = self._attempt_recovery(error_type, error_msg)
        
        # Check if error count is too high for this type
        if self.error_counts[error_type] > self.max_recovery_attempts:
            if self.logger:
                self.logger.warning(
                    f"{error_type} error occurred {self.error_counts[error_type]} times, "
                    f"exceeding maximum recovery attempts"
                )
            should_continue = False
        
        return should_continue
    
    def _attempt_recovery(self, error_type: str, error_msg: str) -> bool:
        """
        Attempt to recover from an error.
        
        Args:
            error_type: The type of error
            error_msg: The error message
            
        Returns:
            bool: Whether recovery was successful and execution should continue
        """
        # Try pattern-based handlers first
        for pattern, handler in self.recovery_handlers.items():
            if pattern in error_type or pattern in error_msg:
                return handler(error_type, error_msg)
        
        # Default: no recovery
        return False
    
    def _handle_memory_error(self, error_type: str, error_msg: str) -> bool:
        """
        Handle memory-related errors by cleaning up memory.
        
        Args:
            error_type: The type of error
            error_msg: The error message
            
        Returns:
            bool: Whether to continue execution
        """
        if self.logger:
            self.logger.info("Attempting to recover from memory error by cleaning up memory...")
        
        memory_cleanup()
        
        # If checkpoint manager is available, try to save a checkpoint
        if self.checkpoint_manager:
            try:
                self.checkpoint_manager.save_emergency_checkpoint()
                if self.logger:
                    self.logger.info("Emergency checkpoint saved successfully")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Failed to save emergency checkpoint: {e}")
        
        return True
    
    def _handle_temporary_error(self, error_type: str, error_msg: str) -> bool:
        """
        Handle temporary errors like network issues.
        
        Args:
            error_type: The type of error
            error_msg: The error message
            
        Returns:
            bool: Whether to continue execution
        """
        if self.logger:
            self.logger.info(f"This appears to be a temporary error ({error_type}), will attempt to continue...")
        
        # If we have too many of these errors in a row, we might not want to continue
        error_count = self.error_counts.get(error_type, 0)
        return error_count <= self.max_recovery_attempts
    
    def reset_error_counts(self) -> None:
        """Reset error counters. Useful after successful recovery."""
        self.error_counts = {}


def safe_execute(func: Callable, 
                error_handler: Optional[EnhancedErrorHandler] = None,
                logger: Optional[logging.Logger] = None, 
                context: str = "",
                default_return: Any = None) -> Any:
    """
    Execute a function with error handling.
    
    Args:
        func: Function to execute
        error_handler: Enhanced error handler instance
        logger: Logger for errors if no error_handler provided
        context: Context information for error messages
        default_return: Default return value if execution fails
        
    Returns:
        Function result or default_return on error
    """
    try:
        return func()
    except Exception as e:
        # Use provided error handler if available
        if error_handler:
            should_continue = error_handler.handle_exception(e, context)
            if should_continue:
                # Try again once
                try:
                    return func()
                except Exception as retry_e:
                    if logger:
                        logger.error(f"Retry failed: {retry_e}")
        elif logger:
            # Basic error logging
            logger.error(f"{context} Error: {type(e).__name__}: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")
        
        return default_return


class ErrorContext:
    """
    Context manager for error handling.
    
    Example:
        with ErrorContext(error_handler, "Loading data"):
            data = load_data()
    """
    
    def __init__(self, 
                error_handler: Optional[EnhancedErrorHandler], 
                context: str,
                logger: Optional[logging.Logger] = None,
                default_return: Any = None):
        """
        Initialize error context.
        
        Args:
            error_handler: Enhanced error handler instance
            context: Context information for error messages
            logger: Logger for errors if no error_handler provided
            default_return: Default return value if execution fails
        """
        self.error_handler = error_handler
        self.context = context
        self.logger = logger
        self.default_return = default_return
        self.result = default_return
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            if self.error_handler:
                should_continue = self.error_handler.handle_exception(exc_val, self.context)
                return should_continue  # Suppress exception if recovery successful
            elif self.logger:
                self.logger.error(f"{self.context} Error: {exc_type.__name__}: {exc_val}")
                self.logger.error(f"Traceback:\n{''.join(traceback.format_tb(exc_tb))}")
            return False  # Re-raise exception
        
        return True  # No exception to handle