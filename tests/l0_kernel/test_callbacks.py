#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for APT Training Callbacks system

Tests:
- ScheduleExecutor parameter interpolation
- Callback triggering mechanism
- Module enabling at specific epochs
"""

import sys
import math
sys.path.insert(0, '/home/user/APT-Transformer')

from apt_model.training.callbacks import (
    lerp,
    cosine_anneal,
    ScheduleExecutor,
    TrainingCallback,
    CallbackManager,
    create_default_callbacks,
)


# ============================================================================
# Test 1: Interpolation Functions
# ============================================================================

def test_interpolation():
    """Test linear and cosine interpolation."""
    print("=" * 60)
    print("Test 1: Interpolation Functions")
    print("=" * 60)

    # Linear interpolation
    assert abs(lerp(1.0, 2.0, 0.0) - 1.0) < 1e-6, "lerp(0.0) failed"
    assert abs(lerp(1.0, 2.0, 1.0) - 2.0) < 1e-6, "lerp(1.0) failed"
    assert abs(lerp(1.0, 2.0, 0.5) - 1.5) < 1e-6, "lerp(0.5) failed"
    print("✓ Linear interpolation works correctly")

    # Cosine annealing
    assert abs(cosine_anneal(1.0, 2.0, 0.0) - 1.0) < 1e-6, "cosine_anneal(0.0) failed"
    assert abs(cosine_anneal(1.0, 2.0, 1.0) - 2.0) < 1e-6, "cosine_anneal(1.0) failed"
    mid_value = cosine_anneal(1.0, 2.0, 0.5)
    print(f"  Cosine anneal at t=0.5: {mid_value:.4f}")
    print("✓ Cosine annealing works correctly")


# ============================================================================
# Test 2: Schedule Executor
# ============================================================================

class MockModule:
    """Mock module for testing."""
    def __init__(self, name):
        self.name = name
        self.enabled = False
        self.temperature = 1.0
        self.capacity_factor = 1.0

    def enable(self, value: bool):
        self.enabled = value
        print(f"  [{self.name}] enabled = {value}")

    def set_temperature(self, value: float):
        self.temperature = value
        print(f"  [{self.name}] temperature = {value:.4f}")

    def set_capacity_factor(self, value: float):
        self.capacity_factor = value
        print(f"  [{self.name}] capacity_factor = {value:.4f}")


class MockConfig:
    """Mock config for testing."""
    def __init__(self):
        self.schedules = {
            "enable_moe_at_epoch": 2,
            "enable_align_at_epoch": 3,
            "route_temp": {
                "start": 1.5,
                "end": 0.8,
                "by": "epoch",
                "type": "linear"
            },
            "moe_capacity": {
                "start": 1.5,
                "end": 1.1,
                "by": "epoch",
                "type": "cosine"
            }
        }


def test_schedule_executor():
    """Test ScheduleExecutor with mock modules."""
    print("\n" + "=" * 60)
    print("Test 2: Schedule Executor")
    print("=" * 60)

    # Create mock modules
    modules = {
        'moe': MockModule('MoE'),
        'align': MockModule('Align'),
        'router': MockModule('Router'),
    }

    config = MockConfig()
    total_epochs = 10
    total_steps = 1000

    executor = ScheduleExecutor(config, modules, total_epochs, total_steps)

    # Test epoch 2: MoE should be enabled
    print("\nEpoch 2:")
    executor.on_epoch_start(epoch=2)
    assert modules['moe'].enabled == True, "MoE should be enabled at epoch 2"
    print("✓ MoE enabled at epoch 2")

    # Test epoch 3: Align should be enabled
    print("\nEpoch 3:")
    executor.on_epoch_start(epoch=3)
    assert modules['align'].enabled == True, "Align should be enabled at epoch 3"
    print("✓ Align enabled at epoch 3")

    # Test parameter scheduling
    print("\nEpoch 0 (start):")
    executor.on_epoch_start(epoch=0)
    assert abs(modules['router'].temperature - 1.5) < 0.1, f"Router temp should be ~1.5, got {modules['router'].temperature}"

    print("\nEpoch 5 (middle):")
    executor.on_epoch_start(epoch=5)
    mid_temp = modules['router'].temperature
    assert 0.8 < mid_temp < 1.5, f"Router temp should be between 0.8 and 1.5, got {mid_temp}"

    print("\nEpoch 10 (end):")
    executor.on_epoch_start(epoch=10)
    # Note: epoch 10 > total_epochs, so t will be clamped
    print(f"  Router temperature: {modules['router'].temperature:.4f}")
    print(f"  MoE capacity: {modules['moe'].capacity_factor:.4f}")

    print("\n✓ Schedule executor works correctly")


# ============================================================================
# Test 3: Callback System
# ============================================================================

class TestCallback(TrainingCallback):
    """Test callback to track events."""
    def __init__(self):
        self.events = []

    def on_train_begin(self, **kwargs):
        self.events.append('train_begin')

    def on_epoch_begin(self, epoch, **kwargs):
        self.events.append(f'epoch_begin_{epoch}')

    def on_batch_end(self, batch_idx, loss, **kwargs):
        self.events.append(f'batch_end_{batch_idx}')

    def on_epoch_end(self, epoch, metrics, **kwargs):
        self.events.append(f'epoch_end_{epoch}')

    def on_train_end(self, **kwargs):
        self.events.append('train_end')


def test_callback_manager():
    """Test CallbackManager triggering."""
    print("\n" + "=" * 60)
    print("Test 3: Callback Manager")
    print("=" * 60)

    callback = TestCallback()
    manager = CallbackManager([callback])

    # Trigger events
    manager.trigger('on_train_begin')
    manager.trigger('on_epoch_begin', epoch=0)
    manager.trigger('on_batch_end', batch_idx=0, loss=1.5)
    manager.trigger('on_batch_end', batch_idx=1, loss=1.4)
    manager.trigger('on_epoch_end', epoch=0, metrics={'loss': 1.45})
    manager.trigger('on_train_end')

    # Verify events
    expected = [
        'train_begin',
        'epoch_begin_0',
        'batch_end_0',
        'batch_end_1',
        'epoch_end_0',
        'train_end'
    ]

    assert callback.events == expected, f"Events mismatch: {callback.events} != {expected}"
    print(f"✓ Callback manager triggered {len(callback.events)} events correctly")
    for event in callback.events:
        print(f"  - {event}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    try:
        test_interpolation()
        test_schedule_executor()
        test_callback_manager()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
