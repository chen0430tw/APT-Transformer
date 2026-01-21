#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Plugin Base Class

Defines the base interface for all APT plugins. Plugins extend the core
system with optional functionality like MoE, alignment, retrieval, etc.

Key concepts:
- Plugins are optional and can be enabled/disabled
- Plugins can register providers with the registry
- Plugins can hook into training events
- Plugins have lifecycle methods (setup, teardown)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
import logging

logger = logging.getLogger(__name__)


class Plugin(ABC):
    """
    Base class for all APT plugins.

    Plugins extend the core APT system with optional functionality. They can:
    1. Register providers (attention, FFN, router, etc.)
    2. Hook into training events (on_epoch_start, on_step_end, etc.)
    3. Add new configuration options
    4. Modify model behavior dynamically

    Lifecycle:
        1. __init__(config) - Plugin created with configuration
        2. setup() - Initialize plugin (register providers, add hooks)
        3. on_* hooks - Called during training
        4. teardown() - Cleanup plugin resources

    Example plugin:
        class MyPlugin(Plugin):
            def get_name(self):
                return "my_plugin"

            def get_version(self):
                return "1.0.0"

            def setup(self, registry, hook_manager):
                # Register providers
                registry.register('attention', 'my_attention', MyAttention)

                # Add hooks
                hook_manager.register('on_epoch_start', self.on_epoch_start)

            def on_epoch_start(self, epoch, **kwargs):
                print(f"Epoch {epoch} started!")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize plugin with configuration.

        Args:
            config: Plugin-specific configuration dictionary
        """
        self.config = config or {}
        self.enabled = True
        self._is_setup = False

        logger.debug(f"Initialized plugin: {self.get_name()}")

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the unique name of this plugin.

        Returns:
            Plugin name (e.g., 'moe', 'align', 'retrieval')
        """
        pass

    @abstractmethod
    def get_version(self) -> str:
        """
        Get the semantic version of this plugin.

        Returns:
            Version string (e.g., '1.0.0', '2.3.1-beta')
        """
        pass

    def get_description(self) -> str:
        """
        Get a human-readable description of this plugin.

        Returns:
            Plugin description
        """
        return ""

    def get_dependencies(self) -> List[str]:
        """
        Get list of required dependencies (other plugins or packages).

        Returns:
            List of dependency strings (e.g., ['torch>=1.9.0', 'plugin:moe'])
        """
        return []

    def get_conflicts(self) -> List[str]:
        """
        Get list of conflicting plugins that cannot be used together.

        Returns:
            List of plugin names that conflict with this plugin
        """
        return []

    def get_priority(self) -> int:
        """
        Get plugin priority for initialization order (higher = earlier).

        Returns:
            Priority integer (default: 0)
        """
        return 0

    def setup(self, registry, hook_manager) -> None:
        """
        Setup plugin: register providers, add hooks, etc.

        This is called once when the plugin is loaded. Use this method to:
        - Register providers with the registry
        - Add hooks to the hook manager
        - Initialize plugin resources

        Args:
            registry: Global provider registry
            hook_manager: Global hook manager

        Example:
            def setup(self, registry, hook_manager):
                # Register provider
                registry.register('router', 'topk', TopKRouter, default=True)

                # Add hooks
                hook_manager.register('on_step_end', self.on_step_end)
        """
        self._is_setup = True
        logger.info(f"Plugin '{self.get_name()}' v{self.get_version()} setup complete")

    def teardown(self) -> None:
        """
        Teardown plugin: cleanup resources, unregister hooks, etc.

        This is called when the plugin is disabled or the system shuts down.
        Use this method to clean up any resources.
        """
        self._is_setup = False
        logger.info(f"Plugin '{self.get_name()}' teardown complete")

    def validate_config(self) -> bool:
        """
        Validate plugin configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        return True

    def is_enabled(self) -> bool:
        """
        Check if plugin is enabled.

        Returns:
            True if plugin is enabled
        """
        return self.enabled

    def enable(self) -> None:
        """Enable this plugin."""
        if not self.enabled:
            self.enabled = True
            logger.info(f"Enabled plugin: {self.get_name()}")

    def disable(self) -> None:
        """Disable this plugin."""
        if self.enabled:
            self.enabled = False
            logger.info(f"Disabled plugin: {self.get_name()}")

    def is_setup(self) -> bool:
        """
        Check if plugin has been setup.

        Returns:
            True if setup() has been called
        """
        return self._is_setup

    # ========== Lifecycle Hooks ==========
    # Subclasses can override these methods to hook into training lifecycle

    def on_model_init(self, model, **kwargs) -> None:
        """
        Called after model initialization.

        Args:
            model: The initialized model
            **kwargs: Additional arguments
        """
        pass

    def on_train_begin(self, **kwargs) -> None:
        """
        Called at the beginning of training.

        Args:
            **kwargs: Training context (trainer, config, etc.)
        """
        pass

    def on_train_end(self, **kwargs) -> None:
        """
        Called at the end of training.

        Args:
            **kwargs: Training context
        """
        pass

    def on_epoch_begin(self, epoch: int, **kwargs) -> None:
        """
        Called at the beginning of each epoch.

        Args:
            epoch: Current epoch number
            **kwargs: Additional arguments
        """
        pass

    def on_epoch_end(self, epoch: int, metrics: Optional[Dict[str, float]] = None, **kwargs) -> None:
        """
        Called at the end of each epoch.

        Args:
            epoch: Current epoch number
            metrics: Training metrics for this epoch
            **kwargs: Additional arguments
        """
        pass

    def on_step_begin(self, step: int, **kwargs) -> None:
        """
        Called at the beginning of each training step.

        Args:
            step: Current step number
            **kwargs: Additional arguments (batch, etc.)
        """
        pass

    def on_step_end(self, step: int, loss: Optional[float] = None, **kwargs) -> None:
        """
        Called at the end of each training step.

        Args:
            step: Current step number
            loss: Training loss for this step
            **kwargs: Additional arguments
        """
        pass

    def on_batch_begin(self, batch, **kwargs) -> None:
        """
        Called before processing a batch.

        Args:
            batch: The input batch
            **kwargs: Additional arguments
        """
        pass

    def on_batch_end(self, batch, outputs, **kwargs) -> None:
        """
        Called after processing a batch.

        Args:
            batch: The input batch
            outputs: Model outputs
            **kwargs: Additional arguments
        """
        pass

    def on_backward_begin(self, loss, **kwargs) -> None:
        """
        Called before backward pass.

        Args:
            loss: Loss tensor
            **kwargs: Additional arguments
        """
        pass

    def on_backward_end(self, **kwargs) -> None:
        """
        Called after backward pass.

        Args:
            **kwargs: Additional arguments
        """
        pass

    def on_optimizer_step(self, optimizer, **kwargs) -> None:
        """
        Called after optimizer step.

        Args:
            optimizer: The optimizer
            **kwargs: Additional arguments
        """
        pass

    def on_validation_begin(self, **kwargs) -> None:
        """
        Called at the beginning of validation.

        Args:
            **kwargs: Validation context
        """
        pass

    def on_validation_end(self, metrics: Optional[Dict[str, float]] = None, **kwargs) -> None:
        """
        Called at the end of validation.

        Args:
            metrics: Validation metrics
            **kwargs: Additional arguments
        """
        pass

    def on_checkpoint_save(self, checkpoint_path: str, **kwargs) -> None:
        """
        Called when saving a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            **kwargs: Additional arguments
        """
        pass

    def on_checkpoint_load(self, checkpoint_path: str, **kwargs) -> None:
        """
        Called when loading a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            **kwargs: Additional arguments
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.get_name()}, v{self.get_version()}, enabled={self.enabled})"


class PluginMetadata:
    """
    Metadata for a plugin (name, version, description, etc.).

    This is useful for plugin discovery and documentation.
    """

    def __init__(
        self,
        name: str,
        version: str,
        description: str = "",
        author: str = "",
        license: str = "",
        dependencies: Optional[List[str]] = None,
        conflicts: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ):
        """
        Initialize plugin metadata.

        Args:
            name: Plugin name
            version: Plugin version
            description: Plugin description
            author: Plugin author
            license: Plugin license
            dependencies: List of dependencies
            conflicts: List of conflicts
            tags: List of tags (e.g., ['moe', 'routing', 'experimental'])
        """
        self.name = name
        self.version = version
        self.description = description
        self.author = author
        self.license = license
        self.dependencies = dependencies or []
        self.conflicts = conflicts or []
        self.tags = tags or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'license': self.license,
            'dependencies': self.dependencies,
            'conflicts': self.conflicts,
            'tags': self.tags
        }

    @classmethod
    def from_plugin(cls, plugin: Plugin) -> 'PluginMetadata':
        """
        Create metadata from a plugin instance.

        Args:
            plugin: Plugin instance

        Returns:
            PluginMetadata instance
        """
        return cls(
            name=plugin.get_name(),
            version=plugin.get_version(),
            description=plugin.get_description(),
            dependencies=plugin.get_dependencies(),
            conflicts=plugin.get_conflicts()
        )

    def __repr__(self):
        return f"PluginMetadata({self.name}, v{self.version})"
