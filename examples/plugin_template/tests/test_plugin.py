"""
Unit tests for example plugin
"""

import pytest
from pathlib import Path
import sys

# Add plugin directory to path
plugin_dir = Path(__file__).parent.parent / "plugin"
sys.path.insert(0, str(plugin_dir))

from plugin import Plugin


def test_plugin_initialization():
    """Test plugin can be initialized"""
    plugin = Plugin()
    assert plugin is not None
    assert plugin.counter == 0


def test_plugin_manifest():
    """Test plugin manifest"""
    plugin = Plugin()
    manifest = plugin.get_manifest()

    assert manifest.name == "example_plugin"
    assert manifest.priority == 350
    assert manifest.category == "training"
    assert "on_init" in manifest.events
    assert "on_batch_start" in manifest.events
    assert "on_batch_end" in manifest.events


def test_plugin_on_init():
    """Test on_init event handler"""
    plugin = Plugin()
    context = {"model": "test_model", "config": {}}

    # Should not raise exception
    plugin.on_init(context)


def test_plugin_on_batch_start():
    """Test on_batch_start event handler"""
    plugin = Plugin()
    context = {"batch_idx": 0}

    plugin.on_batch_start(context)

    assert plugin.counter == 1
    assert "example_plugin_data" in context
    assert context["example_plugin_data"]["counter"] == 1


def test_plugin_on_batch_end():
    """Test on_batch_end event handler"""
    plugin = Plugin()
    context = {"batch_idx": 0}

    # Should not raise exception
    plugin.on_batch_end(context)


def test_plugin_multiple_batches():
    """Test plugin processes multiple batches"""
    plugin = Plugin()

    for i in range(5):
        context = {"batch_idx": i}
        plugin.on_batch_start(context)
        plugin.on_batch_end(context)

    assert plugin.counter == 5


def test_plugin_cleanup():
    """Test plugin cleanup"""
    plugin = Plugin()
    plugin.counter = 10

    plugin.cleanup()

    assert plugin.counter == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
