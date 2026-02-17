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


class ProgressCallback(TrainingCallback):
    """
    Enhanced progress bar callback with rich display.

    Provides LPMM-style progress display with:
    - Spinner animation
    - Fancy progress bar
    - Percentage, count, elapsed/remaining time
    - Training metrics (loss, lr, gpu usage)
    """

    def __init__(self, use_rich: bool = False):
        """
        Initialize progress callback.

        Args:
            use_rich: Use rich library for advanced display (requires: pip install rich)
        """
        self.use_rich = use_rich
        self.progress_bar = None
        self.batch_task = None
        self.epoch_task = None

        if use_rich:
            try:
                from rich.progress import (
                    Progress, SpinnerColumn, TextColumn, BarColumn,
                    TaskProgressColumn, TimeRemainingColumn, TimeElapsedColumn,
                    MofNCompleteColumn
                )
                self.Progress = Progress
                self.progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(bar_width=50),
                    TaskProgressColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    TextColumn("剩余:"),
                    TimeRemainingColumn(),
                    refresh_per_second=10
                )
            except ImportError:
                logger.warning("Rich library not found, falling back to tqdm")
                self.use_rich = False

        if not self.use_rich:
            from tqdm import tqdm
            self.tqdm = tqdm

    def on_train_begin(self, total_epochs: int = None, **kwargs):
        """Start progress tracking at training begin."""
        if self.use_rich:
            self.progress.start()
            if total_epochs:
                self.epoch_task = self.progress.add_task(
                    "🎯 训练进度",
                    total=total_epochs
                )

    def on_epoch_begin(self, epoch: int, dataloader=None, **kwargs):
        """Create progress bar for new epoch."""
        if self.use_rich:
            if dataloader:
                self.batch_task = self.progress.add_task(
                    f"  📊 Epoch {epoch+1}",
                    total=len(dataloader)
                )
        else:
            # Enhanced tqdm progress bar
            if dataloader:
                self.progress_bar = self.tqdm(
                    total=len(dataloader),
                    desc=f"📊 Epoch {epoch+1}",
                    ncols=120,
                    bar_format=(
                        "{desc}: {percentage:3.0f}%|{bar:50}| "
                        "{n_fmt}/{total_fmt} "
                        "[{elapsed}<{remaining}, {rate_fmt}] "
                        "{postfix}"
                    ),
                    ascii=False,
                    colour='green',
                    leave=True
                )

    def on_batch_end(self, batch_idx: int, loss: float, lr: float = None,
                     gpu_usage: float = None, **kwargs):
        """Update progress bar after each batch."""
        if self.use_rich and self.batch_task:
            desc = f"  📊 Epoch {kwargs.get('epoch', 0)+1} | 损失: {loss:.4f}"
            if lr:
                desc += f" | 学习率: {lr:.2e}"
            if gpu_usage:
                desc += f" | GPU: {gpu_usage:.0f}%"

            self.progress.update(
                self.batch_task,
                advance=1,
                description=desc
            )

        elif self.progress_bar:
            postfix = {"损失": f"{loss:.4f}"}
            if lr:
                postfix["学习率"] = f"{lr:.2e}"
            if gpu_usage:
                postfix["GPU"] = f"{gpu_usage:.0f}%"

            self.progress_bar.update(1)
            self.progress_bar.set_postfix(postfix)

    def on_epoch_end(self, epoch: int, metrics: Dict = None, **kwargs):
        """Clean up epoch progress bar."""
        if self.use_rich:
            if self.batch_task:
                self.progress.remove_task(self.batch_task)
                self.batch_task = None
            if self.epoch_task:
                self.progress.update(self.epoch_task, advance=1)
        elif self.progress_bar:
            self.progress_bar.close()
            self.progress_bar = None

    def on_train_end(self, **kwargs):
        """Stop progress tracking."""
        if self.use_rich:
            self.progress.stop()


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

def create_default_callbacks(config, modules: Dict, total_epochs: int, total_steps: int,
                             use_rich_progress: bool = False) -> List[TrainingCallback]:
    """
    Create default set of callbacks for APT training.

    Args:
        config: APTConfig with schedules
        modules: Dict of model modules
        total_epochs: Total training epochs
        total_steps: Total training steps
        use_rich_progress: Use rich library for enhanced progress display

    Returns:
        List of TrainingCallback instances
    """
    callbacks = []

    # Progress callback (enhanced progress bar)
    callbacks.append(ProgressCallback(use_rich=use_rich_progress))
    logger.info(f"Added ProgressCallback (rich={use_rich_progress})")

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


# ==============================================================================
# Additional Callbacks
# ==============================================================================

class EarlyStoppingCallback(TrainingCallback):
    """
    Early stopping callback to stop training when validation loss stops improving.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def on_validation_end(self, metrics: dict):
        """Check if training should stop"""
        val_loss = metrics.get('val_loss', float('inf'))

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


class LearningRateSchedulerCallback(TrainingCallback):
    """
    Learning rate scheduler callback.
    """

    def __init__(self, scheduler, step_on: str = 'epoch'):
        """
        Args:
            scheduler: Learning rate scheduler (e.g., torch.optim.lr_scheduler)
            step_on: When to step the scheduler ('epoch' or 'batch')
        """
        super().__init__()
        self.scheduler = scheduler
        self.step_on = step_on

    def on_epoch_end(self, epoch: int, metrics: dict):
        """Step scheduler after epoch if configured"""
        if self.step_on == 'epoch':
            self.scheduler.step()

    def on_batch_end(self, batch_idx: int, loss: float):
        """Step scheduler after batch if configured"""
        if self.step_on == 'batch':
            self.scheduler.step()
