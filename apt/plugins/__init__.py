#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Plugin System

A flexible plugin architecture for extending APT with optional functionality.

Key components:
- Plugin: Base class for all plugins
- PluginManager: Manages plugin lifecycle
- HookManager: Event-driven hook system
- Built-in plugins: MoE, Alignment, Retrieval, etc.

Usage:
    from apt.plugins import plugin_manager, Plugin

    # Define a plugin
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

    # Register and load plugin
    plugin_manager.register_plugin('my_plugin', MyPlugin)
    plugin_manager.load_plugin('my_plugin')
    plugin_manager.setup_all()

    # Trigger events
    plugin_manager.trigger_event('on_epoch_start', epoch=5)
"""

from apt.plugins.base import Plugin, PluginMetadata
from apt.plugins.manager import PluginManager, plugin_manager
from apt.plugins.hooks import (
    HookManager,
    hook_manager,
    Hook,
    HookEvents,
    hook,
    every_n_epochs,
    every_n_steps,
    after_epoch,
    before_epoch
)

__all__ = [
    # Base classes
    'Plugin',
    'PluginMetadata',

    # Managers
    'PluginManager',
    'plugin_manager',
    'HookManager',
    'hook_manager',

    # Hook utilities
    'Hook',
    'HookEvents',
    'hook',

    # Condition helpers
    'every_n_epochs',
    'every_n_steps',
    'after_epoch',
    'before_epoch',
]
