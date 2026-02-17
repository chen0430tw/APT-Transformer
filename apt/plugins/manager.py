#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Plugin Manager

Manages plugin lifecycle: discovery, loading, initialization, and execution.

Key features:
- Plugin discovery (from directories or configuration)
- Dependency resolution
- Conflict detection
- Dynamic loading/unloading
- Plugin state management
"""

from typing import Dict, List, Optional, Type, Any
from pathlib import Path
import importlib
import importlib.util
import sys
import logging

from apt.plugins.base import Plugin, PluginMetadata
from apt.plugins.hooks import HookManager, hook_manager as global_hook_manager
from apt.core.registry import registry

logger = logging.getLogger(__name__)


class PluginManager:
    """
    Plugin manager for APT system.

    The PluginManager handles the complete plugin lifecycle:
    1. Discovery: Find available plugins
    2. Loading: Load plugin classes
    3. Initialization: Create plugin instances
    4. Setup: Register providers and hooks
    5. Execution: Trigger lifecycle events
    6. Teardown: Cleanup plugin resources

    Usage:
        manager = PluginManager()

        # Register plugins
        manager.register_plugin('moe', MoEPlugin)

        # Load plugins from config
        manager.load_from_config(config)

        # Setup all plugins
        manager.setup_all()

        # Trigger events
        manager.trigger_event('on_epoch_start', epoch=5)
    """

    def __init__(self, hook_manager: Optional[HookManager] = None):
        """
        Initialize plugin manager.

        Args:
            hook_manager: Hook manager instance (uses global if None)
        """
        # Plugin registry: {name: plugin_class}
        self._plugin_classes: Dict[str, Type[Plugin]] = {}

        # Active plugins: {name: plugin_instance}
        self._plugins: Dict[str, Plugin] = {}

        # Plugin metadata cache
        self._metadata: Dict[str, PluginMetadata] = {}

        # Hook manager
        self.hook_manager = hook_manager or global_hook_manager

        # Plugin load order (for dependency resolution)
        self._load_order: List[str] = []

        logger.info("PluginManager initialized")

    def register_plugin(
        self,
        name: str,
        plugin_class: Type[Plugin],
        override: bool = False
    ) -> None:
        """
        Register a plugin class.

        Args:
            name: Plugin name (must match plugin.get_name())
            plugin_class: Plugin class (must inherit from Plugin)
            override: Allow overriding existing registration

        Raises:
            TypeError: If plugin_class doesn't inherit from Plugin
            ValueError: If plugin already registered and override=False
        """
        if not issubclass(plugin_class, Plugin):
            raise TypeError(f"{plugin_class.__name__} must inherit from Plugin")

        if name in self._plugin_classes and not override:
            raise ValueError(
                f"Plugin '{name}' already registered. Use override=True to replace."
            )

        self._plugin_classes[name] = plugin_class
        logger.info(f"Registered plugin class: {name}")

    def load_plugin(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Plugin:
        """
        Load and initialize a plugin.

        Args:
            name: Plugin name
            config: Plugin configuration

        Returns:
            Initialized plugin instance

        Raises:
            ValueError: If plugin not registered
        """
        if name not in self._plugin_classes:
            raise ValueError(f"Plugin '{name}' not registered")

        if name in self._plugins:
            logger.warning(f"Plugin '{name}' already loaded")
            return self._plugins[name]

        # Create plugin instance
        plugin_class = self._plugin_classes[name]
        plugin = plugin_class(config)

        # Validate configuration
        if not plugin.validate_config():
            raise ValueError(f"Invalid configuration for plugin '{name}'")

        # Store plugin
        self._plugins[name] = plugin

        # Cache metadata
        self._metadata[name] = PluginMetadata.from_plugin(plugin)

        logger.info(
            f"Loaded plugin: {name} v{plugin.get_version()} "
            f"(enabled={plugin.is_enabled()})"
        )

        return plugin

    def load_from_config(self, config) -> List[str]:
        """
        Load plugins from APTConfig.

        Args:
            config: APTConfig instance with 'plugins' list

        Returns:
            List of loaded plugin names

        Example:
            config = APTConfig(plugins=['moe', 'align'])
            manager.load_from_config(config)
        """
        plugin_names = getattr(config, 'plugins', [])

        if not plugin_names:
            logger.info("No plugins specified in config")
            return []

        loaded = []

        for name in plugin_names:
            try:
                # Get plugin-specific config from config.extra
                plugin_config = config.extra.get(name, {})
                self.load_plugin(name, plugin_config)
                loaded.append(name)
            except Exception as e:
                logger.error(f"Failed to load plugin '{name}': {e}")

        return loaded

    def setup_plugin(self, name: str) -> None:
        """
        Setup a plugin (register providers, add hooks).

        Args:
            name: Plugin name

        Raises:
            ValueError: If plugin not loaded
        """
        if name not in self._plugins:
            raise ValueError(f"Plugin '{name}' not loaded")

        plugin = self._plugins[name]

        if plugin.is_setup():
            logger.warning(f"Plugin '{name}' already setup")
            return

        # Check dependencies
        self._check_dependencies(plugin)

        # Setup plugin
        plugin.setup(registry, self.hook_manager)

        # Register plugin lifecycle hooks
        self._register_plugin_hooks(plugin)

        logger.info(f"Setup complete for plugin: {name}")

    def setup_all(self) -> None:
        """
        Setup all loaded plugins in dependency order.

        This resolves dependencies and sets up plugins in the correct order.
        """
        if not self._plugins:
            logger.info("No plugins to setup")
            return

        # Resolve load order
        self._resolve_load_order()

        # Setup in order
        for name in self._load_order:
            if name in self._plugins:
                try:
                    self.setup_plugin(name)
                except Exception as e:
                    logger.error(f"Failed to setup plugin '{name}': {e}", exc_info=True)

        logger.info(f"Setup {len(self._load_order)} plugin(s)")

    def teardown_plugin(self, name: str) -> None:
        """
        Teardown a plugin.

        Args:
            name: Plugin name
        """
        if name not in self._plugins:
            logger.warning(f"Plugin '{name}' not loaded")
            return

        plugin = self._plugins[name]

        if not plugin.is_setup():
            logger.warning(f"Plugin '{name}' not setup")
            return

        # Teardown plugin
        plugin.teardown()

        logger.info(f"Teardown complete for plugin: {name}")

    def teardown_all(self) -> None:
        """Teardown all plugins."""
        for name in reversed(self._load_order):
            if name in self._plugins:
                try:
                    self.teardown_plugin(name)
                except Exception as e:
                    logger.error(f"Failed to teardown plugin '{name}': {e}")

    def unload_plugin(self, name: str) -> None:
        """
        Unload a plugin.

        Args:
            name: Plugin name
        """
        if name in self._plugins:
            # Teardown first
            if self._plugins[name].is_setup():
                self.teardown_plugin(name)

            # Remove from active plugins
            del self._plugins[name]
            logger.info(f"Unloaded plugin: {name}")

    def get_plugin(self, name: str) -> Optional[Plugin]:
        """
        Get a loaded plugin instance.

        Args:
            name: Plugin name

        Returns:
            Plugin instance or None
        """
        return self._plugins.get(name)

    def has_plugin(self, name: str) -> bool:
        """
        Check if a plugin is loaded.

        Args:
            name: Plugin name

        Returns:
            True if plugin is loaded
        """
        return name in self._plugins

    def list_plugins(self, loaded_only: bool = False) -> List[str]:
        """
        List all plugins.

        Args:
            loaded_only: If True, only list loaded plugins

        Returns:
            List of plugin names
        """
        if loaded_only:
            return list(self._plugins.keys())
        else:
            return list(self._plugin_classes.keys())

    def get_metadata(self, name: str) -> Optional[PluginMetadata]:
        """
        Get plugin metadata.

        Args:
            name: Plugin name

        Returns:
            PluginMetadata or None
        """
        return self._metadata.get(name)

    def list_metadata(self) -> List[PluginMetadata]:
        """
        Get metadata for all loaded plugins.

        Returns:
            List of PluginMetadata
        """
        return list(self._metadata.values())

    def trigger_event(self, event: str, **kwargs) -> None:
        """
        Trigger a lifecycle event on all plugins.

        Args:
            event: Event name (e.g., 'on_epoch_start')
            **kwargs: Event arguments

        Example:
            manager.trigger_event('on_epoch_start', epoch=5, step=1000)

        Note:
            Plugin lifecycle methods are automatically registered as hooks
            during setup, so we only need to trigger via hook_manager.
        """
        # Trigger via hook manager (plugin lifecycle methods are registered as hooks)
        self.hook_manager.trigger(event, **kwargs)

    def enable_plugin(self, name: str) -> None:
        """
        Enable a plugin.

        Args:
            name: Plugin name
        """
        if name in self._plugins:
            self._plugins[name].enable()
        else:
            logger.warning(f"Plugin '{name}' not loaded")

    def disable_plugin(self, name: str) -> None:
        """
        Disable a plugin.

        Args:
            name: Plugin name
        """
        if name in self._plugins:
            self._plugins[name].disable()
        else:
            logger.warning(f"Plugin '{name}' not loaded")

    def _check_dependencies(self, plugin: Plugin) -> None:
        """
        Check if plugin dependencies are satisfied.

        Args:
            plugin: Plugin instance

        Raises:
            ValueError: If dependencies not satisfied
        """
        dependencies = plugin.get_dependencies()

        for dep in dependencies:
            if dep.startswith('plugin:'):
                # Plugin dependency
                dep_name = dep[7:]
                if dep_name not in self._plugins:
                    raise ValueError(
                        f"Plugin '{plugin.get_name()}' requires plugin '{dep_name}'"
                    )
            else:
                # Package dependency (just log warning)
                logger.debug(f"Plugin '{plugin.get_name()}' requires: {dep}")

    def _check_conflicts(self) -> None:
        """
        Check for plugin conflicts.

        Raises:
            ValueError: If conflicts detected
        """
        for name, plugin in self._plugins.items():
            conflicts = plugin.get_conflicts()
            for conflict in conflicts:
                if conflict in self._plugins:
                    raise ValueError(
                        f"Plugin '{name}' conflicts with '{conflict}'"
                    )

    def _resolve_load_order(self) -> None:
        """
        Resolve plugin load order based on dependencies and priorities.

        Uses topological sort with priority as tiebreaker.
        """
        # Simple priority-based sort for now
        # TODO: Implement proper dependency resolution (topological sort)

        plugins_with_priority = [
            (name, plugin.get_priority())
            for name, plugin in self._plugins.items()
        ]

        # Sort by priority (descending)
        plugins_with_priority.sort(key=lambda x: x[1], reverse=True)

        self._load_order = [name for name, _ in plugins_with_priority]

        logger.debug(f"Plugin load order: {self._load_order}")

    def _register_plugin_hooks(self, plugin: Plugin) -> None:
        """
        Register plugin lifecycle methods as hooks.

        Args:
            plugin: Plugin instance
        """
        # Map of event names to plugin methods
        hook_methods = [
            'on_model_init',
            'on_train_begin',
            'on_train_end',
            'on_epoch_begin',
            'on_epoch_end',
            'on_step_begin',
            'on_step_end',
            'on_batch_begin',
            'on_batch_end',
            'on_backward_begin',
            'on_backward_end',
            'on_optimizer_step',
            'on_validation_begin',
            'on_validation_end',
            'on_checkpoint_save',
            'on_checkpoint_load',
        ]

        priority = plugin.get_priority()

        for method_name in hook_methods:
            method = getattr(plugin, method_name, None)
            if method and callable(method):
                self.hook_manager.register(
                    method_name,
                    method,
                    priority=priority,
                    name=f"{plugin.get_name()}.{method_name}"
                )

    def discover_plugins(self, plugin_dir: Path) -> List[str]:
        """
        Discover plugins in a directory.

        Args:
            plugin_dir: Directory containing plugin modules

        Returns:
            List of discovered plugin names

        Note:
            Plugins must be Python modules with a class inheriting from Plugin.
        """
        if not plugin_dir.exists():
            logger.warning(f"Plugin directory not found: {plugin_dir}")
            return []

        discovered = []

        for file_path in plugin_dir.glob("*.py"):
            if file_path.name.startswith("_"):
                continue

            try:
                # Load module
                module_name = file_path.stem
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Find Plugin subclasses
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and
                            issubclass(attr, Plugin) and
                            attr is not Plugin):

                            # Create temporary instance to get name
                            temp = attr({})
                            plugin_name = temp.get_name()

                            # Register
                            self.register_plugin(plugin_name, attr)
                            discovered.append(plugin_name)

                            logger.info(
                                f"Discovered plugin: {plugin_name} "
                                f"(from {file_path.name})"
                            )

            except Exception as e:
                logger.error(f"Failed to load plugin from {file_path}: {e}")

        return discovered

    def __repr__(self):
        loaded = len(self._plugins)
        total = len(self._plugin_classes)
        return f"PluginManager({loaded}/{total} plugins loaded)"


# ============================================================================
# Global Plugin Manager
# ============================================================================

# Global plugin manager instance
plugin_manager = PluginManager()
