#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Training Callbacks and Schedulers

Implements curriculum scheduling and dynamic module control for APT models.
Enables configuration-driven training with:
- Dynamic module enabling (MoE, Align, Voter, etc.)
- Parameter annealing (route_temp, moe_capacity, etc.)
- Multi-criteria curriculum learning
- Callback hooks for training events
"""

import math
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ============================================================================
# Schedule Utilities
# ============================================================================

def lerp(start: float, end: float, t: float) -> float:
    """Linear interpolation."""
    return start + (end - start) * t


def cosine_anneal(start: float, end: float, t: float) -> float:
    """Cosine annealing."""
    return end + (start - end) * (1 + math.cos(math.pi * t)) / 2


def get_interpolator(schedule_type: str = "linear") -> Callable:
    """Get interpolation function by type."""
    interpolators = {
        "linear": lerp,
        "cosine": cosine_anneal,
    }
    return interpolators.get(schedule_type, lerp)


# ============================================================================
# Schedule Executor
# ============================================================================

class ScheduleExecutor:
    """
    Executes curriculum schedules defined in config.

    Example config.schedules:
        {
            "enable_moe_at_epoch": 2,
            "enable_align_at_epoch": 3,
            "route_temp": {
                "start": 1.5,
                "end": 0.8,
                "by": "epoch"
            }
        }
    """

    def __init__(self, config, modules: Dict[str, Any], total_epochs: int, total_steps: int):
        """
        Initialize schedule executor.

        Args:
            config: APTConfig with schedules field
            modules: Dict of modules (e.g., {'moe': moe_layer, 'align': align_layer})
            total_epochs: Total number of training epochs
            total_steps: Total number of training steps
        """
        self.config = config
        self.schedules = config.schedules if hasattr(config, 'schedules') else {}
        self.modules = modules
        self.total_epochs = total_epochs
        self.total_steps = total_steps

        logger.info(f"ScheduleExecutor initialized with {len(self.schedules)} schedules")

    def on_epoch_start(self, epoch: int):
        """Execute schedules at epoch start."""
        # Enable modules at specific epochs
        for key, value in self.schedules.items():
            if key.startswith("enable_") and key.endswith("_at_epoch"):
                if epoch == value:
                    module_name = key.replace("enable_", "").replace("_at_epoch", "")
                    if module_name in self.modules:
                        module = self.modules[module_name]
                        if hasattr(module, 'enable'):
                            module.enable(True)
                            logger.info(f"[Epoch {epoch}] Enabled module: {module_name}")
                        else:
                            logger.warning(f"Module {module_name} has no enable() method")

        # Update parameters with epoch-based schedules
        t = epoch / max(self.total_epochs, 1)
        self._update_parameters(t, by="epoch", current=epoch)

    def on_step(self, step: int):
        """Execute schedules at training step."""
        t = step / max(self.total_steps, 1)
        self._update_parameters(t, by="step", current=step)

    def _update_parameters(self, t: float, by: str, current: int):
        """Update parameters based on schedule."""
        for key, schedule in self.schedules.items():
            if not isinstance(schedule, dict):
                continue

            if schedule.get("by") != by:
                continue

            # Get start, end values
            start = schedule.get("start")
            end = schedule.get("end")
            if start is None or end is None:
                continue

            # Handle warmup
            warmup = schedule.get("warmup", 0)
            if by == "step" and warmup > 0:
                total_steps = self.total_steps
                if current < warmup:
                    # Warmup phase: 0 -> start
                    value = lerp(0, start, current / warmup)
                else:
                    # Main phase: start -> end
                    t_main = (current - warmup) / (total_steps - warmup)
                    interpolator = get_interpolator(schedule.get("type", "linear"))
                    value = interpolator(start, end, t_main)
            else:
                # No warmup: start -> end
                interpolator = get_interpolator(schedule.get("type", "linear"))
                value = interpolator(start, end, t)

            # Apply value to module
            self._apply_value(key, value)

    def _apply_value(self, param_name: str, value: float):
        """Apply scheduled value to module parameter."""
        # Parse param_name (e.g., "route_temp" -> module=router, attr=temperature)
        param_mapping = {
            "route_temp": ("router", "temperature"),
            "moe_capacity": ("moe", "capacity_factor"),
            "align_weight": ("align", "loss_weight"),
            "vote_threshold": ("voter", "entropy_threshold"),
        }

        if param_name in param_mapping:
            module_name, attr_name = param_mapping[param_name]
            if module_name in self.modules:
                module = self.modules[module_name]
                # Try different setter patterns
                if hasattr(module, f"set_{attr_name}"):
                    getattr(module, f"set_{attr_name}")(value)
                    logger.debug(f"Updated {module_name}.{attr_name} = {value:.4f}")
                elif hasattr(module, attr_name):
                    setattr(module, attr_name, value)
                    logger.debug(f"Updated {module_name}.{attr_name} = {value:.4f}")
                else:
                    logger.warning(f"Module {module_name} has no attribute {attr_name}")


# ============================================================================
# Callback Interface
# ============================================================================

class TrainingCallback:
    """Base callback class."""

    def on_train_begin(self, **kwargs):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, **kwargs):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, **kwargs):
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict, **kwargs):
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, batch_idx: int, **kwargs):
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, batch_idx: int, loss: float, **kwargs):
        """Called at the end of each batch."""
        pass

    def on_step(self, step: int, **kwargs):
        """Called after each optimization step."""
        pass


class ScheduleCallback(TrainingCallback):
    """Callback that executes curriculum schedules."""

    def __init__(self, schedule_executor: ScheduleExecutor):
        self.executor = schedule_executor

    def on_epoch_begin(self, epoch: int, **kwargs):
        self.executor.on_epoch_start(epoch)

    def on_step(self, step: int, **kwargs):
        self.executor.on_step(step)


class EntropyBasedVotingCallback(TrainingCallback):
    """Callback that enables voting based on batch entropy."""

    def __init__(self, modules: Dict, threshold: float = 2.2):
        self.modules = modules
        self.threshold = threshold

    def on_batch_end(self, batch_idx: int, loss: float, entropy: Optional[float] = None, **kwargs):
        if entropy is not None and 'voter' in self.modules:
            voter = self.modules['voter']
            if hasattr(voter, 'enable'):
                if entropy > self.threshold:
                    voter.enable(True)
                else:
                    voter.enable(False)


class LoggingCallback(TrainingCallback):
    """Callback for logging training progress."""

    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval

    def on_batch_end(self, batch_idx: int, loss: float, **kwargs):
        if (batch_idx + 1) % self.log_interval == 0:
            logger.info(f"Batch {batch_idx + 1}, Loss: {loss:.4f}")

    def on_epoch_end(self, epoch: int, metrics: Dict, **kwargs):
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Epoch {epoch} completed - {metric_str}")


# ============================================================================
# Callback Manager
# ============================================================================

class CallbackManager:
    """Manages multiple callbacks."""

    def __init__(self, callbacks: List[TrainingCallback]):
        self.callbacks = callbacks

    def trigger(self, event: str, **kwargs):
        """Trigger an event on all callbacks."""
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method:
                try:
                    method(**kwargs)
                except Exception as e:
                    logger.error(f"Error in callback {callback.__class__.__name__}.{event}: {e}")


# ============================================================================
# Convenience Functions
# ============================================================================

def create_default_callbacks(config, modules: Dict, total_epochs: int, total_steps: int) -> List[TrainingCallback]:
    """
    Create default set of callbacks for APT training.

    Args:
        config: APTConfig with schedules
        modules: Dict of model modules
        total_epochs: Total training epochs
        total_steps: Total training steps

    Returns:
        List of TrainingCallback instances
    """
    callbacks = []

    # Schedule callback (curriculum learning)
    if hasattr(config, 'schedules') and config.schedules:
        schedule_executor = ScheduleExecutor(config, modules, total_epochs, total_steps)
        callbacks.append(ScheduleCallback(schedule_executor))
        logger.info("Added ScheduleCallback")

    # Entropy-based voting callback
    if 'voter' in modules:
        callbacks.append(EntropyBasedVotingCallback(modules, threshold=2.2))
        logger.info("Added EntropyBasedVotingCallback")

    # Logging callback
    callbacks.append(LoggingCallback(log_interval=10))

    return callbacks
