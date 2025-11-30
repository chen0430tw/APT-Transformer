#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify APT plugin system works correctly.

This script validates:
1. Plugin base class
2. PluginManager functionality
3. HookManager functionality
4. Plugin lifecycle
5. Provider registration via plugins
"""

import sys
import traceback


def test_imports():
    """Test that all plugin modules can be imported."""
    print("=" * 60)
    print("Test 1: Plugin Module Imports")
    print("=" * 60)

    try:
        from apt.plugins import (
            Plugin,
            PluginManager,
            plugin_manager,
            HookManager,
            hook_manager,
            Hook,
            HookEvents,
            hook
        )
        print("✅ All plugin modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_hook_manager():
    """Test HookManager functionality."""
    print("\n" + "=" * 60)
    print("Test 2: HookManager")
    print("=" * 60)

    try:
        from apt.plugins import HookManager

        # Create manager
        manager = HookManager()
        print("✅ HookManager created")

        # Register a hook
        call_count = [0]  # Use list for mutable closure

        def test_hook(epoch, **kwargs):
            call_count[0] += 1

        manager.register('on_epoch_start', test_hook, priority=10)
        print("✅ Hook registered")

        # Trigger hook
        manager.trigger('on_epoch_start', epoch=1)
        assert call_count[0] == 1, "Hook should have been called once"
        print(f"✅ Hook triggered ({call_count[0]} call)")

        # Trigger again
        manager.trigger('on_epoch_start', epoch=2)
        assert call_count[0] == 2, "Hook should have been called twice"
        print(f"✅ Hook triggered again ({call_count[0]} calls total)")

        # Check statistics
        stats = manager.get_statistics()
        assert stats['on_epoch_start'] == 2
        print(f"✅ Hook statistics: {stats}")

        return True

    except Exception as e:
        print(f"❌ HookManager test failed: {e}")
        traceback.print_exc()
        return False


def test_conditional_hooks():
    """Test conditional hook execution."""
    print("\n" + "=" * 60)
    print("Test 3: Conditional Hooks")
    print("=" * 60)

    try:
        from apt.plugins import HookManager, every_n_epochs

        manager = HookManager()

        call_count = [0]

        def periodic_hook(epoch, **kwargs):
            call_count[0] += 1

        # Register hook that runs every 5 epochs
        manager.register(
            'on_epoch_start',
            periodic_hook,
            condition=every_n_epochs(5)
        )

        # Trigger for epochs 0-10
        for epoch in range(11):
            manager.trigger('on_epoch_start', epoch=epoch)

        # Should be called at epochs 0, 5, 10 = 3 times
        expected = 3
        assert call_count[0] == expected, f"Expected {expected} calls, got {call_count[0]}"
        print(f"✅ Conditional hook called {call_count[0]} times (expected {expected})")

        return True

    except Exception as e:
        print(f"❌ Conditional hooks test failed: {e}")
        traceback.print_exc()
        return False


def test_plugin_base():
    """Test Plugin base class."""
    print("\n" + "=" * 60)
    print("Test 4: Plugin Base Class")
    print("=" * 60)

    try:
        from apt.plugins import Plugin

        # Create a test plugin
        class TestPlugin(Plugin):
            def get_name(self):
                return "test"

            def get_version(self):
                return "1.0.0"

        # Instantiate
        plugin = TestPlugin({'key': 'value'})
        print(f"✅ Plugin created: {plugin}")

        # Check properties
        assert plugin.get_name() == "test"
        assert plugin.get_version() == "1.0.0"
        assert plugin.is_enabled() == True
        assert plugin.is_setup() == False
        print("✅ Plugin properties verified")

        # Test enable/disable
        plugin.disable()
        assert plugin.is_enabled() == False
        print("✅ Plugin disabled")

        plugin.enable()
        assert plugin.is_enabled() == True
        print("✅ Plugin enabled")

        return True

    except Exception as e:
        print(f"❌ Plugin base test failed: {e}")
        traceback.print_exc()
        return False


def test_plugin_manager():
    """Test PluginManager functionality."""
    print("\n" + "=" * 60)
    print("Test 5: PluginManager")
    print("=" * 60)

    try:
        from apt.plugins import Plugin, PluginManager

        # Create test plugin
        class TestPlugin(Plugin):
            def __init__(self, config=None):
                super().__init__(config)
                self.setup_called = False

            def get_name(self):
                return "test"

            def get_version(self):
                return "1.0.0"

            def setup(self, registry, hook_manager):
                super().setup(registry, hook_manager)
                self.setup_called = True

        # Create manager
        manager = PluginManager()
        print("✅ PluginManager created")

        # Register plugin
        manager.register_plugin('test', TestPlugin)
        print("✅ Plugin registered")

        # Check registration
        assert manager.has_plugin('test') == False  # Not loaded yet
        print("✅ Plugin not yet loaded (correct)")

        # Load plugin
        plugin = manager.load_plugin('test', {'config_key': 'value'})
        assert plugin is not None
        assert manager.has_plugin('test') == True
        print(f"✅ Plugin loaded: {plugin}")

        # Setup plugin
        manager.setup_plugin('test')
        assert plugin.setup_called == True
        assert plugin.is_setup() == True
        print("✅ Plugin setup called")

        # Test event triggering
        manager.trigger_event('on_epoch_start', epoch=1)
        print("✅ Event triggered")

        # Teardown
        manager.teardown_plugin('test')
        assert plugin.is_setup() == False
        print("✅ Plugin teardown")

        return True

    except Exception as e:
        print(f"❌ PluginManager test failed: {e}")
        traceback.print_exc()
        return False


def test_plugin_with_provider():
    """Test plugin that registers a provider."""
    print("\n" + "=" * 60)
    print("Test 6: Plugin with Provider Registration")
    print("=" * 60)

    try:
        from apt.plugins import Plugin, PluginManager
        from apt.core import Provider, registry

        # Create test provider
        class TestProvider(Provider):
            def __init__(self, config):
                self.config = config

            def get_name(self):
                return "test_provider"

            def get_version(self):
                return "1.0.0"

        # Create plugin that registers provider
        class TestPlugin(Plugin):
            def get_name(self):
                return "test_with_provider"

            def get_version(self):
                return "1.0.0"

            def setup(self, registry, hook_manager):
                super().setup(registry, hook_manager)
                registry.register('test_kind', 'test_provider', TestProvider)

        # Register and load plugin
        manager = PluginManager()
        manager.register_plugin('test_with_provider', TestPlugin)
        manager.load_plugin('test_with_provider')
        manager.setup_plugin('test_with_provider')
        print("✅ Plugin registered and setup")

        # Check provider is registered
        providers = registry.list_providers('test_kind')
        assert 'test_kind' in providers
        assert 'test_provider' in providers['test_kind']
        print(f"✅ Provider registered: {providers}")

        # Get provider
        provider = registry.get('test_kind', 'test_provider')
        assert provider is not None
        assert provider.get_name() == "test_provider"
        print(f"✅ Provider retrieved: {provider}")

        return True

    except Exception as e:
        print(f"❌ Plugin with provider test failed: {e}")
        traceback.print_exc()
        return False


def test_plugin_lifecycle():
    """Test complete plugin lifecycle."""
    print("\n" + "=" * 60)
    print("Test 7: Complete Plugin Lifecycle")
    print("=" * 60)

    try:
        from apt.plugins import Plugin, PluginManager

        # Track lifecycle calls
        lifecycle = []

        class LifecyclePlugin(Plugin):
            def get_name(self):
                return "lifecycle"

            def get_version(self):
                return "1.0.0"

            def setup(self, registry, hook_manager):
                super().setup(registry, hook_manager)
                lifecycle.append('setup')

            def on_train_begin(self, **kwargs):
                lifecycle.append('train_begin')

            def on_epoch_begin(self, epoch, **kwargs):
                lifecycle.append(f'epoch_{epoch}_begin')

            def on_epoch_end(self, epoch, **kwargs):
                lifecycle.append(f'epoch_{epoch}_end')

            def on_train_end(self, **kwargs):
                lifecycle.append('train_end')

            def teardown(self):
                lifecycle.append('teardown')
                super().teardown()

        # Create and setup
        manager = PluginManager()
        manager.register_plugin('lifecycle', LifecyclePlugin)
        manager.load_plugin('lifecycle')
        manager.setup_all()

        # Simulate training
        manager.trigger_event('on_train_begin')
        for epoch in range(3):
            manager.trigger_event('on_epoch_begin', epoch=epoch)
            manager.trigger_event('on_epoch_end', epoch=epoch)
        manager.trigger_event('on_train_end')

        # Teardown
        manager.teardown_all()

        # Check lifecycle
        expected = [
            'setup',
            'train_begin',
            'epoch_0_begin', 'epoch_0_end',
            'epoch_1_begin', 'epoch_1_end',
            'epoch_2_begin', 'epoch_2_end',
            'train_end',
            'teardown'
        ]

        print(f"Lifecycle calls: {lifecycle}")
        assert lifecycle == expected, f"Expected {expected}, got {lifecycle}"
        print("✅ Complete lifecycle verified")

        return True

    except Exception as e:
        print(f"❌ Plugin lifecycle test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("APT Plugin System Tests")
    print("=" * 60 + "\n")

    tests = [
        ("Imports", test_imports),
        ("HookManager", test_hook_manager),
        ("Conditional Hooks", test_conditional_hooks),
        ("Plugin Base", test_plugin_base),
        ("PluginManager", test_plugin_manager),
        ("Plugin with Provider", test_plugin_with_provider),
        ("Plugin Lifecycle", test_plugin_lifecycle),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    print("\n" + "=" * 60)
    if passed_count == total_count:
        print(f"✅ All {total_count} tests PASSED")
        print("=" * 60)
        return 0
    else:
        print(f"❌ {total_count - passed_count}/{total_count} tests FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
