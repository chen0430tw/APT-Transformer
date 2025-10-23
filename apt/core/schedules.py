#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Schedule System

Implements curriculum-based training schedules that control when plugins
are enabled and how parameters change over training.

Key features:
- Epoch-based and step-based scheduling
- Parameter annealing (linear, exponential, cosine)
- Plugin activation scheduling
- Warmup and cooldown phases
"""

from typing import Dict, Any, Optional, Union, Callable
import math
import logging

logger = logging.getLogger(__name__)


class Schedule:
    """
    Curriculum schedule manager for APT training.

    Schedules control:
    1. When plugins are enabled (e.g., enable MoE at epoch 2)
    2. How parameters change over time (e.g., temperature annealing)
    3. Training phase transitions

    Configuration example:
        schedules:
          enable_moe_at_epoch: 2
          enable_align_at_epoch: 3

          route_temp:
            start: 1.5
            end: 0.8
            by: "epoch"

          moe_capacity:
            start: 1.5
            end: 1.1
            by: "step"
            warmup: 1000
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize schedule from configuration.

        Args:
            config: Schedule configuration dictionary
        """
        self.config = config
        self._enabled_plugins = set()
        self._param_schedules = {}

        # Parse configuration
        self._parse_config()

        logger.info(f"Schedule initialized with {len(self._param_schedules)} parameter schedules")

    def _parse_config(self):
        """Parse schedule configuration and build internal structures."""
        for key, value in self.config.items():
            if key.startswith('enable_') and key.endswith('_at_epoch'):
                # Plugin enablement schedule
                plugin_name = key[7:-9]  # Remove 'enable_' and '_at_epoch'
                self._enabled_plugins.add((plugin_name, 'epoch', value))

            elif key.startswith('enable_') and key.endswith('_at_step'):
                # Step-based enablement
                plugin_name = key[7:-8]  # Remove 'enable_' and '_at_step'
                self._enabled_plugins.add((plugin_name, 'step', value))

            elif isinstance(value, dict) and ('start' in value or 'end' in value):
                # Parameter schedule
                self._param_schedules[key] = self._create_param_schedule(key, value)

    def _create_param_schedule(
        self,
        name: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create a parameter schedule from configuration.

        Args:
            name: Parameter name
            config: Schedule config with start/end/by/type/warmup

        Returns:
            Internal schedule representation
        """
        schedule = {
            'name': name,
            'start': config.get('start', 0.0),
            'end': config.get('end', 0.0),
            'by': config.get('by', 'epoch'),  # 'epoch' or 'step'
            'type': config.get('type', 'linear'),  # 'linear', 'exp', 'cosine'
            'warmup': config.get('warmup', 0),
            'cooldown': config.get('cooldown', 0),
        }

        logger.debug(f"Created schedule for '{name}': {schedule}")
        return schedule

    def should_enable_plugin(
        self,
        plugin_name: str,
        epoch: Optional[int] = None,
        step: Optional[int] = None
    ) -> bool:
        """
        Check if a plugin should be enabled at the current training point.

        Args:
            plugin_name: Name of the plugin
            epoch: Current epoch (required for epoch-based schedules)
            step: Current step (required for step-based schedules)

        Returns:
            True if plugin should be enabled

        Example:
            if schedule.should_enable_plugin('moe', epoch=5):
                enable_moe()
        """
        for plugin, by, threshold in self._enabled_plugins:
            if plugin == plugin_name:
                if by == 'epoch' and epoch is not None:
                    return epoch >= threshold
                elif by == 'step' and step is not None:
                    return step >= threshold

        # If no schedule found, assume always enabled
        return True

    def get_param(
        self,
        param_name: str,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        max_epochs: Optional[int] = None,
        max_steps: Optional[int] = None
    ) -> float:
        """
        Get the current value of a scheduled parameter.

        Args:
            param_name: Parameter name
            epoch: Current epoch
            step: Current step
            max_epochs: Total epochs (required for epoch-based schedules)
            max_steps: Total steps (required for step-based schedules)

        Returns:
            Current parameter value

        Example:
            capacity = schedule.get_param('moe_capacity', epoch=10, max_epochs=50)
        """
        if param_name not in self._param_schedules:
            # No schedule defined, return 0 or raise error
            logger.warning(f"No schedule defined for '{param_name}'")
            return 0.0

        sched = self._param_schedules[param_name]

        # Determine current progress
        if sched['by'] == 'epoch':
            if epoch is None or max_epochs is None:
                raise ValueError(
                    f"Epoch-based schedule '{param_name}' requires epoch and max_epochs"
                )
            current = epoch
            maximum = max_epochs
        elif sched['by'] == 'step':
            if step is None or max_steps is None:
                raise ValueError(
                    f"Step-based schedule '{param_name}' requires step and max_steps"
                )
            current = step
            maximum = max_steps
        else:
            raise ValueError(f"Unknown schedule type: {sched['by']}")

        # Apply warmup
        if current < sched['warmup']:
            return sched['start']

        # Apply cooldown
        if sched['cooldown'] > 0 and current >= maximum - sched['cooldown']:
            return sched['end']

        # Calculate progress (0 to 1)
        effective_start = sched['warmup']
        effective_end = maximum - sched['cooldown']
        progress = (current - effective_start) / max(1, effective_end - effective_start)
        progress = max(0.0, min(1.0, progress))

        # Apply schedule function
        value = self._interpolate(
            sched['start'],
            sched['end'],
            progress,
            sched['type']
        )

        return value

    def _interpolate(
        self,
        start: float,
        end: float,
        progress: float,
        schedule_type: str
    ) -> float:
        """
        Interpolate between start and end values.

        Args:
            start: Starting value
            end: Ending value
            progress: Progress in [0, 1]
            schedule_type: Type of schedule ('linear', 'exp', 'cosine')

        Returns:
            Interpolated value
        """
        if schedule_type == 'linear':
            return start + (end - start) * progress

        elif schedule_type == 'exp':
            # Exponential decay/growth
            if start == 0:
                return end * progress
            ratio = end / start
            return start * (ratio ** progress)

        elif schedule_type == 'cosine':
            # Cosine annealing
            return end + (start - end) * (1 + math.cos(math.pi * progress)) / 2

        elif schedule_type == 'sqrt':
            # Square root schedule
            return start + (end - start) * math.sqrt(progress)

        elif schedule_type == 'inverse_sqrt':
            # Inverse square root (common for learning rate)
            return start + (end - start) * (1.0 - 1.0 / math.sqrt(1 + progress * 10))

        else:
            logger.warning(f"Unknown schedule type '{schedule_type}', using linear")
            return start + (end - start) * progress

    def get_all_params(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        max_epochs: Optional[int] = None,
        max_steps: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Get all scheduled parameters at once.

        Args:
            epoch: Current epoch
            step: Current step
            max_epochs: Total epochs
            max_steps: Total steps

        Returns:
            Dictionary of parameter names to current values
        """
        params = {}
        for param_name in self._param_schedules.keys():
            try:
                params[param_name] = self.get_param(
                    param_name,
                    epoch=epoch,
                    step=step,
                    max_epochs=max_epochs,
                    max_steps=max_steps
                )
            except ValueError as e:
                logger.warning(f"Failed to get param '{param_name}': {e}")

        return params

    def list_enabled_plugins(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None
    ) -> list:
        """
        List all plugins that should be enabled at the current point.

        Args:
            epoch: Current epoch
            step: Current step

        Returns:
            List of plugin names
        """
        enabled = []
        seen = set()

        for plugin, by, threshold in self._enabled_plugins:
            if plugin in seen:
                continue

            should_enable = False
            if by == 'epoch' and epoch is not None and epoch >= threshold:
                should_enable = True
            elif by == 'step' and step is not None and step >= threshold:
                should_enable = True

            if should_enable:
                enabled.append(plugin)
                seen.add(plugin)

        return enabled

    def add_param_schedule(
        self,
        name: str,
        start: float,
        end: float,
        by: str = 'epoch',
        schedule_type: str = 'linear',
        warmup: int = 0,
        cooldown: int = 0
    ):
        """
        Dynamically add a parameter schedule.

        Args:
            name: Parameter name
            start: Starting value
            end: Ending value
            by: 'epoch' or 'step'
            schedule_type: 'linear', 'exp', or 'cosine'
            warmup: Warmup period
            cooldown: Cooldown period
        """
        self._param_schedules[name] = {
            'name': name,
            'start': start,
            'end': end,
            'by': by,
            'type': schedule_type,
            'warmup': warmup,
            'cooldown': cooldown,
        }
        logger.info(f"Added parameter schedule: {name}")

    def __repr__(self):
        return (
            f"Schedule("
            f"{len(self._enabled_plugins)} plugin schedules, "
            f"{len(self._param_schedules)} param schedules)"
        )


# ============================================================================
# Predefined Schedule Functions
# ============================================================================

def linear_schedule(start: float, end: float, progress: float) -> float:
    """Linear interpolation schedule."""
    return start + (end - start) * progress


def cosine_schedule(start: float, end: float, progress: float) -> float:
    """Cosine annealing schedule."""
    return end + (start - end) * (1 + math.cos(math.pi * progress)) / 2


def exponential_schedule(start: float, end: float, progress: float) -> float:
    """Exponential schedule."""
    if start == 0:
        return end * progress
    ratio = end / start
    return start * (ratio ** progress)


def warmup_linear_schedule(
    start: float,
    end: float,
    current: int,
    total: int,
    warmup: int
) -> float:
    """Linear schedule with warmup."""
    if current < warmup:
        return start
    progress = (current - warmup) / max(1, total - warmup)
    return start + (end - start) * min(1.0, progress)
