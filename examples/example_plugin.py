#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example Plugin for APT

This is a complete example showing how to create a custom plugin for APT.

The plugin demonstrates:
1. Basic plugin structure
2. Provider registration
3. Hook registration
4. Configuration handling
5. Lifecycle management
"""

from apt.plugins import Plugin
from apt.core import Provider
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Example Provider
# ============================================================================

class ExampleAttentionProvider(Provider):
    """
    Example attention provider.

    This is a minimal provider implementation for demonstration purposes.
    """

    def __init__(self, config):
        self.config = config

    def get_name(self) -> str:
        return "example_attention"

    def get_version(self) -> str:
        return "1.0.0"

    def create_layer(self, d_model, num_heads, dropout=0.0, **kwargs):
        """
        Create an attention layer.

        In a real implementation, this would return an nn.Module.
        For this example, we just return a placeholder.
        """
        logger.info(
            f"Creating example attention layer: "
            f"d_model={d_model}, num_heads={num_heads}"
        )

        # In real implementation:
        # return ExampleAttentionLayer(d_model, num_heads, dropout)

        # For now, return a mock
        class MockAttention:
            def __init__(self):
                self.d_model = d_model
                self.num_heads = num_heads

            def __repr__(self):
                return f"ExampleAttention(d_model={self.d_model}, heads={self.num_heads})"

        return MockAttention()


# ============================================================================
# Example Plugin
# ============================================================================

class ExamplePlugin(Plugin):
    """
    Example plugin demonstrating all plugin features.

    This plugin:
    - Registers a custom attention provider
    - Adds hooks for training events
    - Tracks training statistics
    - Responds to configuration
    """

    def __init__(self, config=None):
        super().__init__(config)

        # Plugin state
        self.epoch_count = 0
        self.step_count = 0
        self.enabled_at_epoch = self.config.get('enable_at_epoch', 0)

        logger.info(f"ExamplePlugin initialized (enable_at_epoch={self.enabled_at_epoch})")

    def get_name(self) -> str:
        return "example"

    def get_version(self) -> str:
        return "1.0.0"

    def get_description(self) -> str:
        return "Example plugin demonstrating APT plugin system"

    def get_priority(self) -> int:
        # Higher priority means earlier execution
        return 10

    def setup(self, registry, hook_manager):
        """Setup plugin: register providers and hooks."""
        super().setup(registry, hook_manager)

        logger.info("Setting up ExamplePlugin...")

        # Register provider
        logger.info("Registering ExampleAttentionProvider...")
        registry.register(
            'attention',
            'example',
            ExampleAttentionProvider,
            default=False  # Don't set as default
        )

        # Add custom hooks
        logger.info("Registering hooks...")
        hook_manager.register(
            'on_epoch_start',
            self.custom_epoch_hook,
            priority=self.get_priority(),
            name='example.custom_epoch_hook'
        )

        logger.info("ExamplePlugin setup complete!")

    def teardown(self):
        """Cleanup plugin resources."""
        super().teardown()
        logger.info(
            f"ExamplePlugin teardown: "
            f"processed {self.epoch_count} epochs, {self.step_count} steps"
        )

    def validate_config(self) -> bool:
        """Validate plugin configuration."""
        if 'enable_at_epoch' in self.config:
            if not isinstance(self.config['enable_at_epoch'], int):
                logger.error("enable_at_epoch must be an integer")
                return False
            if self.config['enable_at_epoch'] < 0:
                logger.error("enable_at_epoch must be non-negative")
                return False

        return True

    # ========== Lifecycle Hooks ==========

    def on_train_begin(self, **kwargs):
        """Called at the beginning of training."""
        logger.info("ExamplePlugin: Training started!")
        logger.info(f"  Training config: {kwargs.get('config', 'N/A')}")

    def on_train_end(self, **kwargs):
        """Called at the end of training."""
        logger.info("ExamplePlugin: Training ended!")
        logger.info(f"  Total epochs: {self.epoch_count}")
        logger.info(f"  Total steps: {self.step_count}")

    def on_epoch_begin(self, epoch, **kwargs):
        """Called at the beginning of each epoch."""
        self.epoch_count += 1

        # Check if we should enable the plugin at this epoch
        if epoch == self.enabled_at_epoch:
            logger.info(f"ExamplePlugin: Enabling at epoch {epoch}")
            self.enable()

        if self.is_enabled():
            logger.info(f"ExamplePlugin: Epoch {epoch} started")

    def on_epoch_end(self, epoch, metrics=None, **kwargs):
        """Called at the end of each epoch."""
        if self.is_enabled():
            logger.info(f"ExamplePlugin: Epoch {epoch} ended")
            if metrics:
                logger.info(f"  Metrics: {metrics}")

    def on_step_end(self, step, loss=None, **kwargs):
        """Called at the end of each training step."""
        self.step_count += 1

        if self.is_enabled() and self.step_count % 100 == 0:
            logger.info(
                f"ExamplePlugin: Step {step} (loss={loss:.4f if loss else 'N/A'})"
            )

    def custom_epoch_hook(self, epoch, **kwargs):
        """
        Custom hook for demonstration.

        This shows how plugins can add additional hooks beyond
        the standard lifecycle methods.
        """
        if self.is_enabled():
            logger.info(f"ExamplePlugin custom hook: Processing epoch {epoch}")


# ============================================================================
# Usage Example
# ============================================================================

def main():
    """
    Demonstrate plugin system usage.
    """
    print("=" * 60)
    print("APT Plugin System Demo")
    print("=" * 60)

    # Import managers
    from apt.plugins import plugin_manager, hook_manager
    from apt.core import registry

    # 1. Register plugin
    print("\n1. Registering plugin...")
    plugin_manager.register_plugin('example', ExamplePlugin)
    print("   ✓ Plugin registered")

    # 2. Load plugin with config
    print("\n2. Loading plugin...")
    config = {'enable_at_epoch': 2}
    plugin = plugin_manager.load_plugin('example', config)
    print(f"   ✓ Plugin loaded: {plugin}")

    # 3. Setup plugin
    print("\n3. Setting up plugin...")
    plugin_manager.setup_plugin('example')
    print("   ✓ Plugin setup complete")

    # 4. Check registered providers
    print("\n4. Checking providers...")
    providers = registry.list_providers()
    print(f"   Registered providers: {providers}")

    # 5. Use provider
    print("\n5. Testing provider...")
    try:
        provider = registry.get('attention', 'example')
        layer = provider.create_layer(d_model=512, num_heads=8)
        print(f"   ✓ Created layer: {layer}")
    except Exception as e:
        print(f"   ✗ Provider test failed: {e}")

    # 6. Trigger lifecycle events
    print("\n6. Simulating training lifecycle...")

    print("\n   Training begin:")
    plugin_manager.trigger_event('on_train_begin', config={'lr': 0.001})

    for epoch in range(5):
        print(f"\n   Epoch {epoch}:")
        plugin_manager.trigger_event('on_epoch_begin', epoch=epoch)

        for step in range(100, 301, 100):
            plugin_manager.trigger_event('on_step_end', step=step, loss=0.5)

        plugin_manager.trigger_event('on_epoch_end', epoch=epoch, metrics={'acc': 0.95})

    print("\n   Training end:")
    plugin_manager.trigger_event('on_train_end')

    # 7. Check statistics
    print("\n7. Hook statistics:")
    stats = hook_manager.get_statistics()
    for event, count in sorted(stats.items()):
        print(f"   {event}: {count} executions")

    # 8. Teardown
    print("\n8. Teardown:")
    plugin_manager.teardown_all()
    print("   ✓ All plugins cleaned up")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(name)s - %(message)s'
    )

    main()
