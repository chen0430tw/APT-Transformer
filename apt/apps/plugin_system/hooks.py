#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Hook System

Implements an event-driven hook system for plugins to integrate with
training lifecycle events.

Key features:
- Register callbacks for lifecycle events
- Priority-based hook execution
- Conditional hooks (execute only if condition met)
- Hook middleware (modify arguments/results)
"""

from typing import Dict, List, Callable, Any, Optional
from collections import defaultdict
import logging
import inspect

logger = logging.getLogger(__name__)


class Hook:
    """
    A single hook callback with metadata.

    Attributes:
        callback: The function to call
        priority: Execution priority (higher = earlier)
        condition: Optional condition function
        name: Optional name for this hook
    """

    def __init__(
        self,
        callback: Callable,
        priority: int = 0,
        condition: Optional[Callable] = None,
        name: Optional[str] = None
    ):
        """
        Initialize hook.

        Args:
            callback: Function to call when hook is triggered
            priority: Execution priority (higher = earlier, default: 0)
            condition: Optional condition function (returns bool)
            name: Optional name for debugging
        """
        self.callback = callback
        self.priority = priority
        self.condition = condition
        self.name = name or callback.__name__

        # Extract function signature for validation
        self.signature = inspect.signature(callback)

    def should_execute(self, **kwargs) -> bool:
        """
        Check if this hook should execute.

        Args:
            **kwargs: Hook context

        Returns:
            True if hook should execute
        """
        if self.condition is None:
            return True

        try:
            return self.condition(**kwargs)
        except Exception as e:
            logger.warning(f"Hook condition failed for {self.name}: {e}")
            return False

    def execute(self, **kwargs) -> Any:
        """
        Execute this hook.

        Args:
            **kwargs: Hook arguments

        Returns:
            Hook callback return value
        """
        try:
            return self.callback(**kwargs)
        except Exception as e:
            logger.error(f"Hook {self.name} failed: {e}", exc_info=True)
            return None

    def __repr__(self):
        return f"Hook({self.name}, priority={self.priority})"


class HookManager:
    """
    Global hook manager for APT plugins.

    The HookManager allows plugins to register callbacks for lifecycle events
    like on_epoch_start, on_step_end, etc. Hooks are executed in priority order.

    Usage:
        hook_manager = HookManager()

        # Register a hook
        def my_callback(epoch, **kwargs):
            print(f"Epoch {epoch} started!")

        hook_manager.register('on_epoch_start', my_callback, priority=10)

        # Trigger hooks
        hook_manager.trigger('on_epoch_start', epoch=5)
    """

    def __init__(self):
        """Initialize hook manager."""
        # Storage: {event_name: [Hook, Hook, ...]}
        self._hooks: Dict[str, List[Hook]] = defaultdict(list)

        # Middleware: {event_name: [middleware_fn, ...]}
        self._middleware: Dict[str, List[Callable]] = defaultdict(list)

        # Statistics
        self._stats: Dict[str, int] = defaultdict(int)

        logger.info("HookManager initialized")

    def register(
        self,
        event: str,
        callback: Callable,
        priority: int = 0,
        condition: Optional[Callable] = None,
        name: Optional[str] = None
    ) -> None:
        """
        Register a hook for an event.

        Args:
            event: Event name (e.g., 'on_epoch_start')
            callback: Function to call when event is triggered
            priority: Execution priority (higher = earlier)
            condition: Optional condition function
            name: Optional name for debugging

        Example:
            def my_hook(epoch, **kwargs):
                print(f"Epoch {epoch}")

            hook_manager.register('on_epoch_start', my_hook, priority=10)
        """
        hook = Hook(callback, priority, condition, name)
        self._hooks[event].append(hook)

        # Sort by priority (descending)
        self._hooks[event].sort(key=lambda h: h.priority, reverse=True)

        logger.debug(f"Registered hook '{hook.name}' for event '{event}' (priority={priority})")

    def unregister(self, event: str, callback: Callable) -> bool:
        """
        Unregister a hook.

        Args:
            event: Event name
            callback: The callback function to unregister

        Returns:
            True if hook was found and removed
        """
        if event not in self._hooks:
            return False

        original_count = len(self._hooks[event])
        self._hooks[event] = [h for h in self._hooks[event] if h.callback != callback]
        removed = original_count - len(self._hooks[event])

        if removed > 0:
            logger.debug(f"Unregistered {removed} hook(s) for event '{event}'")
            return True

        return False

    def trigger(self, event: str, **kwargs) -> List[Any]:
        """
        Trigger all hooks for an event.

        Args:
            event: Event name
            **kwargs: Arguments to pass to hooks

        Returns:
            List of hook return values

        Example:
            results = hook_manager.trigger('on_epoch_start', epoch=5, step=1000)
        """
        if event not in self._hooks:
            return []

        # Apply middleware to modify arguments
        if event in self._middleware:
            for middleware in self._middleware[event]:
                try:
                    kwargs = middleware(kwargs) or kwargs
                except Exception as e:
                    logger.warning(f"Middleware failed for event '{event}': {e}")

        results = []
        executed = 0

        for hook in self._hooks[event]:
            if hook.should_execute(**kwargs):
                result = hook.execute(**kwargs)
                results.append(result)
                executed += 1

        # Update statistics
        self._stats[event] += executed

        if executed > 0:
            logger.debug(f"Triggered {executed} hook(s) for event '{event}'")

        return results

    def register_middleware(self, event: str, middleware: Callable) -> None:
        """
        Register middleware to modify hook arguments.

        Middleware functions receive the kwargs dict and can modify it
        before hooks are executed.

        Args:
            event: Event name
            middleware: Function that takes and returns kwargs dict

        Example:
            def add_timestamp(kwargs):
                kwargs['timestamp'] = time.time()
                return kwargs

            hook_manager.register_middleware('on_step_end', add_timestamp)
        """
        self._middleware[event].append(middleware)
        logger.debug(f"Registered middleware for event '{event}'")

    def has_hooks(self, event: str) -> bool:
        """
        Check if any hooks are registered for an event.

        Args:
            event: Event name

        Returns:
            True if hooks exist for this event
        """
        return event in self._hooks and len(self._hooks[event]) > 0

    def list_hooks(self, event: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List all registered hooks.

        Args:
            event: If specified, only list hooks for this event

        Returns:
            Dictionary mapping event names to list of hook names
        """
        if event:
            return {event: [h.name for h in self._hooks.get(event, [])]}
        else:
            return {e: [h.name for h in hooks] for e, hooks in self._hooks.items()}

    def get_statistics(self) -> Dict[str, int]:
        """
        Get hook execution statistics.

        Returns:
            Dictionary mapping event names to execution counts
        """
        return dict(self._stats)

    def clear_hooks(self, event: Optional[str] = None) -> None:
        """
        Clear hooks.

        Args:
            event: If specified, only clear hooks for this event
        """
        if event:
            if event in self._hooks:
                del self._hooks[event]
                logger.info(f"Cleared hooks for event '{event}'")
        else:
            self._hooks.clear()
            logger.info("Cleared all hooks")

    def __repr__(self):
        total_hooks = sum(len(hooks) for hooks in self._hooks.values())
        return f"HookManager({len(self._hooks)} events, {total_hooks} hooks)"


# ============================================================================
# Global Hook Manager Singleton
# ============================================================================

# Global hook manager instance - import this in other modules
hook_manager = HookManager()


# ============================================================================
# Decorator for Easy Hook Registration
# ============================================================================

def hook(event: str, priority: int = 0, condition: Optional[Callable] = None):
    """
    Decorator to register a function as a hook.

    Usage:
        @hook('on_epoch_start', priority=10)
        def my_hook(epoch, **kwargs):
            print(f"Epoch {epoch}")

    Args:
        event: Event name
        priority: Execution priority
        condition: Optional condition function
    """
    def decorator(func):
        hook_manager.register(event, func, priority=priority, condition=condition)
        return func
    return decorator


# ============================================================================
# Standard Hook Events
# ============================================================================

class HookEvents:
    """
    Standard hook event names.

    This class defines the standard lifecycle events that plugins can hook into.
    """

    # Model lifecycle
    MODEL_INIT = 'on_model_init'

    # Training lifecycle
    TRAIN_BEGIN = 'on_train_begin'
    TRAIN_END = 'on_train_end'

    # Epoch lifecycle
    EPOCH_BEGIN = 'on_epoch_begin'
    EPOCH_END = 'on_epoch_end'

    # Step lifecycle
    STEP_BEGIN = 'on_step_begin'
    STEP_END = 'on_step_end'

    # Batch lifecycle
    BATCH_BEGIN = 'on_batch_begin'
    BATCH_END = 'on_batch_end'

    # Backward pass
    BACKWARD_BEGIN = 'on_backward_begin'
    BACKWARD_END = 'on_backward_end'

    # Optimizer
    OPTIMIZER_STEP = 'on_optimizer_step'

    # Validation
    VALIDATION_BEGIN = 'on_validation_begin'
    VALIDATION_END = 'on_validation_end'

    # Checkpointing
    CHECKPOINT_SAVE = 'on_checkpoint_save'
    CHECKPOINT_LOAD = 'on_checkpoint_load'

    @classmethod
    def all_events(cls) -> List[str]:
        """Get list of all standard event names."""
        return [
            value for name, value in cls.__dict__.items()
            if not name.startswith('_') and isinstance(value, str)
        ]


# ============================================================================
# Condition Helpers
# ============================================================================

def every_n_epochs(n: int) -> Callable:
    """
    Create a condition that triggers every N epochs.

    Args:
        n: Epoch interval

    Returns:
        Condition function

    Example:
        hook_manager.register(
            'on_epoch_end',
            my_callback,
            condition=every_n_epochs(5)  # Execute every 5 epochs
        )
    """
    def condition(epoch: int, **kwargs) -> bool:
        return epoch % n == 0
    return condition


def every_n_steps(n: int) -> Callable:
    """
    Create a condition that triggers every N steps.

    Args:
        n: Step interval

    Returns:
        Condition function
    """
    def condition(step: int, **kwargs) -> bool:
        return step % n == 0
    return condition


def after_epoch(threshold: int) -> Callable:
    """
    Create a condition that triggers after a certain epoch.

    Args:
        threshold: Minimum epoch number

    Returns:
        Condition function
    """
    def condition(epoch: int, **kwargs) -> bool:
        return epoch >= threshold
    return condition


def before_epoch(threshold: int) -> Callable:
    """
    Create a condition that triggers before a certain epoch.

    Args:
        threshold: Maximum epoch number

    Returns:
        Condition function
    """
    def condition(epoch: int, **kwargs) -> bool:
        return epoch < threshold
    return condition
