#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify APT core modules can be imported successfully.

This script validates:
1. All core modules import without errors
2. Registry system works correctly
3. Configuration loading works
4. ModelBuilder can be instantiated
"""

import sys
import traceback

import pytest


MODULE_IMPORTS = [
    ("apt.core.registry", ["Provider", "Registry", "registry"]),
    ("apt.core.config", ["APTConfig", "MultimodalConfig", "HardwareProfile"]),
    ("apt.core.schedules", ["Schedule"]),
    ("apt.core.providers.attention", ["AttentionProvider"]),
    ("apt.core.providers.ffn", ["FFNProvider"]),
    ("apt.core.providers.router", ["RouterProvider"]),
    ("apt.core.providers.align", ["AlignProvider"]),
    ("apt.core.providers.retrieval", ["RetrievalProvider"]),
    ("apt.modeling.compose", ["ModelBuilder"]),
    ("apt", ["registry", "Provider"]),
]


@pytest.mark.parametrize("module_name,components", MODULE_IMPORTS)
def test_core_module_imports(module_name, components):
    """Pytest wrapper to ensure modules import correctly."""
    assert _import_module(module_name, components)


def _import_module(module_name, components=None):
    """Import a module and optionally specific components."""
    try:
        if components:
            exec(f"from {module_name} import {', '.join(components)}")
            print(f"✅ {module_name}: {', '.join(components)}")
        else:
            exec(f"import {module_name}")
            print(f"✅ {module_name}")
        return True
    except Exception as e:
        print(f"❌ {module_name}: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all import tests."""
    print("=" * 60)
    print("APT Core Module Import Tests")
    print("=" * 60)

    all_passed = True

    # Test core imports
    print("\n1. Testing core modules...")
    for module_name, components in MODULE_IMPORTS[:8]:
        if not _import_module(module_name, components):
            all_passed = False

    # Test modeling imports
    print("\n2. Testing modeling modules...")
    if not _import_module("apt.modeling.compose", ["ModelBuilder"]):
        all_passed = False

    # Test top-level imports
    print("\n3. Testing top-level imports...")
    if not _import_module("apt", ["registry", "Provider"]):
        all_passed = False

    # Test basic functionality
    print("\n4. Testing basic functionality...")
    try:
        from apt.core.registry import registry, Provider
        from apt.core.config import APTConfig
        from apt.modeling.compose import ModelBuilder

        # Test registry
        print("   Testing registry...")
        providers = registry.list_providers()
        print(f"   Registered providers: {providers}")

        # Test config
        print("   Testing config...")
        config = APTConfig(d_model=512, num_heads=8)
        print(f"   Created config: {config}")

        # Test config to/from dict
        config_dict = config.to_dict()
        config2 = APTConfig.from_dict(config_dict)
        print(f"   Config serialization: OK")

        # Test ModelBuilder
        print("   Testing ModelBuilder...")
        builder = ModelBuilder(config)
        components = builder.list_components()
        print(f"   ModelBuilder components: {components}")

        print("✅ Basic functionality tests passed")

    except Exception as e:
        print(f"❌ Basic functionality tests failed: {e}")
        traceback.print_exc()
        all_passed = False

    # Test Schedule
    print("\n5. Testing Schedule system...")
    try:
        from apt.core.schedules import Schedule

        schedule_config = {
            'enable_moe_at_epoch': 2,
            'enable_align_at_epoch': 3,
            'route_temp': {
                'start': 1.5,
                'end': 0.8,
                'by': 'epoch'
            }
        }

        schedule = Schedule(schedule_config)
        print(f"   Created schedule: {schedule}")

        # Test plugin enablement
        should_enable = schedule.should_enable_plugin('moe', epoch=5)
        print(f"   Should enable MoE at epoch 5: {should_enable}")

        # Test parameter schedule
        temp = schedule.get_param('route_temp', epoch=10, max_epochs=50)
        print(f"   Route temp at epoch 10: {temp:.2f}")

        print("✅ Schedule tests passed")

    except Exception as e:
        print(f"❌ Schedule tests failed: {e}")
        traceback.print_exc()
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All import tests PASSED")
        print("=" * 60)
        return 0
    else:
        print("❌ Some import tests FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
