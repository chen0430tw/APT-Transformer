# APT Plugin Development Guide

## Overview

This guide explains how to develop plugins for the APT (Autopoietic Transformer) framework. The APT plugin system enables you to extend the core functionality with custom components, training strategies, and integrations.

**Table of Contents:**
- [What is a Plugin?](#what-is-a-plugin)
- [Plugin Architecture](#plugin-architecture)
- [Creating Your First Plugin](#creating-your-first-plugin)
- [Provider Registration](#provider-registration)
- [Hook System](#hook-system)
- [Configuration](#configuration)
- [Best Practices](#best-practices)
- [Examples](#examples)

---

## What is a Plugin?

A **plugin** is a modular extension that adds optional functionality to APT without modifying the core system. Plugins can:

- ✅ Register custom **Providers** (attention, FFN, router, etc.)
- ✅ Hook into **training lifecycle events** (epoch start, step end, etc.)
- ✅ Add **new configuration options**
- ✅ Modify **model behavior dynamically**
- ✅ Integrate **external libraries or services**

**Key benefits:**
- **Optional:** Plugins can be enabled/disabled via configuration
- **Isolated:** Plugins don't affect core system stability
- **Composable:** Multiple plugins can work together
- **Schedule-aware:** Plugins can be enabled at specific epochs

---

## Plugin Architecture

### Component Hierarchy

```
Plugin System
├── Plugin (base class)
│   ├── Lifecycle methods (setup, teardown)
│   ├── Hook methods (on_epoch_start, on_step_end, etc.)
│   └── Configuration validation
│
├── PluginManager
│   ├── Plugin registration
│   ├── Dependency resolution
│   ├── Lifecycle management
│   └── Event triggering
│
├── HookManager
│   ├── Event registration
│   ├── Priority-based execution
│   └── Conditional hooks
│
└── Providers
    ├── AttentionProvider
    ├── FFNProvider
    ├── RouterProvider
    └── ... (registered by plugins)
```

### Plugin Lifecycle

```
1. Registration  → Plugin class registered with PluginManager
2. Loading       → Plugin instance created with config
3. Setup         → Providers registered, hooks added
4. Execution     → Hooks triggered during training
5. Teardown      → Cleanup resources
6. Unloading     → Plugin removed from system
```

---

## Creating Your First Plugin

### Step 1: Define Plugin Class

```python
from apt.plugins import Plugin

class MyPlugin(Plugin):
    """My custom plugin."""

    def __init__(self, config=None):
        super().__init__(config)
        # Initialize plugin state
        self.counter = 0

    def get_name(self) -> str:
        """Return unique plugin name."""
        return "my_plugin"

    def get_version(self) -> str:
        """Return semantic version."""
        return "1.0.0"

    def get_description(self) -> str:
        """Return human-readable description."""
        return "My custom plugin for APT"
```

### Step 2: Implement Setup

```python
    def setup(self, registry, hook_manager):
        """Setup plugin: register providers and hooks."""
        super().setup(registry, hook_manager)

        # Register provider (if you have one)
        registry.register(
            'attention',
            'my_attention',
            MyAttentionProvider
        )

        # Add custom hooks
        hook_manager.register(
            'on_epoch_start',
            self.on_epoch_start,
            priority=10
        )
```

### Step 3: Implement Lifecycle Methods

```python
    def on_epoch_start(self, epoch, **kwargs):
        """Called at the start of each epoch."""
        self.counter += 1
        print(f"MyPlugin: Epoch {epoch} started (total: {self.counter})")

    def on_step_end(self, step, loss=None, **kwargs):
        """Called at the end of each step."""
        if step % 100 == 0:
            print(f"MyPlugin: Step {step}, loss={loss}")

    def teardown(self):
        """Cleanup plugin resources."""
        super().teardown()
        print(f"MyPlugin: Processed {self.counter} epochs")
```

### Step 4: Register and Use

```python
from apt.plugins import plugin_manager

# Register plugin class
plugin_manager.register_plugin('my_plugin', MyPlugin)

# Load plugin with config
config = {'enabled': True}
plugin_manager.load_plugin('my_plugin', config)

# Setup all plugins
plugin_manager.setup_all()

# Trigger events during training
plugin_manager.trigger_event('on_epoch_start', epoch=1)
```

---

## Provider Registration

Plugins can register **Providers** to add new implementations of core components.

### Creating a Provider

```python
from apt.core import Provider

class MyAttentionProvider(Provider):
    """Custom attention provider."""

    def __init__(self, config):
        self.config = config

    def get_name(self) -> str:
        return "my_attention"

    def get_version(self) -> str:
        return "1.0.0"

    def create_layer(self, d_model, num_heads, dropout=0.0, **kwargs):
        """Create attention layer."""
        return MyAttentionLayer(d_model, num_heads, dropout)

    def validate_config(self, config) -> bool:
        """Validate configuration."""
        return config.get('d_model', 0) > 0
```

### Registering in Plugin

```python
class MyPlugin(Plugin):
    def setup(self, registry, hook_manager):
        super().setup(registry, hook_manager)

        # Register provider
        registry.register(
            kind='attention',           # Provider type
            name='my_attention',        # Provider name
            provider_cls=MyAttentionProvider,
            default=False,              # Set as default?
            excludes=['flash_attn']     # Conflicts with these
        )
```

### Using the Provider

```python
from apt.modeling import ModelBuilder
from apt.core import APTConfig

# Configure to use your provider
config = APTConfig(
    d_model=768,
    attention_name='my_attention'  # Use your provider
)

# Build model
builder = ModelBuilder(config)
attention = builder.build_attention()  # Uses MyAttentionProvider
```

---

## Hook System

The hook system allows plugins to react to training lifecycle events.

### Standard Lifecycle Events

| Event | When | Arguments |
|-------|------|-----------|
| `on_model_init` | After model creation | `model` |
| `on_train_begin` | Start of training | `config`, `trainer` |
| `on_train_end` | End of training | `metrics` |
| `on_epoch_begin` | Start of epoch | `epoch` |
| `on_epoch_end` | End of epoch | `epoch`, `metrics` |
| `on_step_begin` | Start of step | `step`, `batch` |
| `on_step_end` | End of step | `step`, `loss` |
| `on_batch_begin` | Before batch | `batch` |
| `on_batch_end` | After batch | `batch`, `outputs` |
| `on_backward_begin` | Before backward | `loss` |
| `on_backward_end` | After backward | - |
| `on_optimizer_step` | After optimizer | `optimizer` |
| `on_validation_begin` | Start validation | - |
| `on_validation_end` | End validation | `metrics` |
| `on_checkpoint_save` | Saving checkpoint | `checkpoint_path` |
| `on_checkpoint_load` | Loading checkpoint | `checkpoint_path` |

### Hook Methods (Option 1)

Override lifecycle methods in your plugin:

```python
class MyPlugin(Plugin):
    def on_epoch_end(self, epoch, metrics=None, **kwargs):
        """Called at end of each epoch."""
        if metrics:
            print(f"Epoch {epoch}: accuracy={metrics.get('acc', 0):.2f}")
```

### Custom Hooks (Option 2)

Register custom hooks with the HookManager:

```python
def my_custom_hook(epoch, **kwargs):
    print(f"Custom hook for epoch {epoch}")

class MyPlugin(Plugin):
    def setup(self, registry, hook_manager):
        super().setup(registry, hook_manager)

        # Register custom hook
        hook_manager.register(
            'on_epoch_start',
            my_custom_hook,
            priority=10,     # Higher = earlier
            name='my_custom_hook'
        )
```

### Conditional Hooks

Execute hooks only when a condition is met:

```python
from apt.plugins import every_n_epochs, after_epoch

# Execute every 5 epochs
hook_manager.register(
    'on_epoch_end',
    save_checkpoint,
    condition=every_n_epochs(5)
)

# Execute only after epoch 10
hook_manager.register(
    'on_epoch_start',
    enable_advanced_features,
    condition=after_epoch(10)
)
```

### Hook Priorities

Hooks execute in priority order (higher = earlier):

```python
# This runs first (priority 100)
hook_manager.register('on_epoch_start', hook1, priority=100)

# This runs second (priority 50)
hook_manager.register('on_epoch_start', hook2, priority=50)

# This runs third (priority 0, default)
hook_manager.register('on_epoch_start', hook3)
```

---

## Configuration

### Plugin Configuration

Plugins receive configuration via `config` dict:

```python
class MyPlugin(Plugin):
    def __init__(self, config=None):
        super().__init__(config)

        # Read config values
        self.enable_at_epoch = config.get('enable_at_epoch', 0)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.verbose = config.get('verbose', False)

    def validate_config(self) -> bool:
        """Validate configuration."""
        if self.enable_at_epoch < 0:
            logger.error("enable_at_epoch must be non-negative")
            return False
        return True
```

### YAML Configuration

Plugins can be configured via YAML:

```yaml
# config.yaml
plugins:
  - my_plugin
  - other_plugin

my_plugin:
  enable_at_epoch: 2
  learning_rate: 0.001
  verbose: true

other_plugin:
  mode: "advanced"
```

Load config:

```python
from apt.core import APTConfig
from apt.plugins import plugin_manager

# Load config from YAML
config = APTConfig.from_yaml('config.yaml')

# Load plugins from config
plugin_manager.load_from_config(config)
plugin_manager.setup_all()
```

### Schedule Integration

Use the schedule system to enable plugins dynamically:

```yaml
plugins:
  - moe
  - align

schedules:
  enable_moe_at_epoch: 2      # Enable MoE at epoch 2
  enable_align_at_epoch: 3    # Enable alignment at epoch 3

  route_temp:
    start: 1.5
    end: 0.8
    by: "epoch"
```

Check in plugin:

```python
class MoEPlugin(Plugin):
    def on_epoch_begin(self, epoch, schedule=None, **kwargs):
        if schedule and schedule.should_enable_plugin('moe', epoch=epoch):
            self.enable()
            print(f"MoE enabled at epoch {epoch}")
```

---

## Best Practices

### 1. **Plugin Design**

✅ **DO:**
- Keep plugins focused on a single responsibility
- Use clear, descriptive names
- Provide comprehensive docstrings
- Validate configuration in `validate_config()`
- Handle errors gracefully

❌ **DON'T:**
- Modify core APT classes directly
- Assume other plugins are loaded
- Use global state
- Perform heavy computation in `__init__`

### 2. **Provider Registration**

✅ **DO:**
- Register providers in `setup()`, not `__init__()`
- Provide fallback implementations
- Document provider requirements
- Test providers independently

❌ **DON'T:**
- Override default providers without good reason
- Register the same provider multiple times
- Ignore provider validation errors

### 3. **Hook Management**

✅ **DO:**
- Use appropriate priorities
- Keep hook functions fast
- Use conditional hooks for periodic operations
- Log important events

❌ **DON'T:**
- Block training in hooks
- Raise exceptions in hooks (log instead)
- Add too many hooks per plugin
- Modify hook arguments

### 4. **Configuration**

✅ **DO:**
- Provide sensible defaults
- Validate all config values
- Document all config options
- Support YAML configuration

❌ **DON'T:**
- Require complex nested configs
- Use magic numbers
- Silently ignore invalid config
- Assume config keys exist

### 5. **Dependencies**

✅ **DO:**
- Declare dependencies in `get_dependencies()`
- Check dependencies in `setup()`
- Provide helpful error messages
- Document optional dependencies

❌ **DON'T:**
- Import heavy libraries in module scope
- Assume libraries are available
- Crash if optional dependencies missing

### 6. **Testing**

✅ **DO:**
- Test plugins in isolation
- Test with minimal config
- Test hook execution
- Test provider creation

❌ **DON'T:**
- Test with full training loops
- Depend on other plugins in tests
- Skip error cases

---

## Examples

### Example 1: Simple Logging Plugin

```python
from apt.plugins import Plugin

class LoggingPlugin(Plugin):
    """Log training events to file."""

    def __init__(self, config=None):
        super().__init__(config)
        self.log_file = config.get('log_file', 'training.log')
        self.file_handle = None

    def get_name(self) -> str:
        return "logging"

    def get_version(self) -> str:
        return "1.0.0"

    def setup(self, registry, hook_manager):
        super().setup(registry, hook_manager)
        self.file_handle = open(self.log_file, 'w')

    def teardown(self):
        if self.file_handle:
            self.file_handle.close()
        super().teardown()

    def on_epoch_end(self, epoch, metrics=None, **kwargs):
        if metrics:
            line = f"Epoch {epoch}: {metrics}\n"
            self.file_handle.write(line)
            self.file_handle.flush()
```

### Example 2: Early Stopping Plugin

```python
from apt.plugins import Plugin

class EarlyStoppingPlugin(Plugin):
    """Stop training if validation loss doesn't improve."""

    def __init__(self, config=None):
        super().__init__(config)
        self.patience = config.get('patience', 5)
        self.min_delta = config.get('min_delta', 0.001)

        self.best_loss = float('inf')
        self.wait = 0
        self.should_stop = False

    def get_name(self) -> str:
        return "early_stopping"

    def get_version(self) -> str:
        return "1.0.0"

    def on_validation_end(self, metrics=None, **kwargs):
        if not metrics or 'val_loss' not in metrics:
            return

        val_loss = metrics['val_loss']

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.should_stop = True
            print(f"Early stopping: no improvement for {self.patience} epochs")
```

### Example 3: Gradient Clipping Plugin

```python
from apt.plugins import Plugin

class GradientClippingPlugin(Plugin):
    """Clip gradients during training."""

    def __init__(self, config=None):
        super().__init__(config)
        self.max_norm = config.get('max_norm', 1.0)
        self.norm_type = config.get('norm_type', 2.0)

    def get_name(self) -> str:
        return "gradient_clipping"

    def get_version(self) -> str:
        return "1.0.0"

    def on_backward_end(self, model=None, **kwargs):
        if model is None:
            return

        import torch.nn as nn
        total_norm = nn.utils.clip_grad_norm_(
            model.parameters(),
            self.max_norm,
            norm_type=self.norm_type
        )

        if total_norm > self.max_norm:
            print(f"Gradient clipped: {total_norm:.2f} -> {self.max_norm}")
```

---

## Advanced Topics

### Plugin Dependencies

Declare dependencies on other plugins:

```python
class AdvancedPlugin(Plugin):
    def get_dependencies(self) -> List[str]:
        return [
            'plugin:moe',           # Requires MoE plugin
            'torch>=1.9.0',         # Requires PyTorch 1.9+
            'transformers>=4.0.0'   # Requires Transformers 4.0+
        ]
```

### Plugin Conflicts

Declare conflicting plugins:

```python
class FlashAttentionPlugin(Plugin):
    def get_conflicts(self) -> List[str]:
        return ['slow_attention', 'memory_attention']
```

### Dynamic Provider Selection

Select providers based on runtime conditions:

```python
class AdaptivePlugin(Plugin):
    def setup(self, registry, hook_manager):
        super().setup(registry, hook_manager)

        # Choose provider based on hardware
        if torch.cuda.is_available():
            registry.register('attention', 'fast', CudaAttention, default=True)
        else:
            registry.register('attention', 'cpu', CPUAttention, default=True)
```

---

## Complete Example

See `examples/example_plugin.py` for a complete, runnable example demonstrating all plugin features.

To run:

```bash
python examples/example_plugin.py
```

---

## Next Steps

1. **Study built-in plugins** in `apt/plugins/builtin/`
2. **Review the examples** in `examples/`
3. **Check the API reference** in `apt/plugins/base.py`
4. **Join the community** and share your plugins!

---

## Troubleshooting

### Plugin not loading

- Check plugin name matches `get_name()`
- Verify plugin class inherits from `Plugin`
- Check for syntax errors
- Enable debug logging

### Provider not found

- Ensure provider is registered in `setup()`
- Check provider name spelling
- Verify registry has the provider: `registry.list_providers()`

### Hooks not executing

- Check hook name spelling
- Verify plugin is enabled
- Check hook priority
- Verify events are being triggered

### Configuration errors

- Validate config in `validate_config()`
- Check for typos in YAML
- Verify config keys exist before accessing
- Use `config.get(key, default)` for optional values

---

## Resources

- **Core Documentation:** `REFACTOR_PLAN.md`, `MIGRATION_GUIDE.md`
- **API Reference:** Docstrings in source code
- **Examples:** `examples/` directory
- **Built-in Plugins:** `apt/plugins/builtin/`

---

## License

APT Plugin System is part of the APT project. See LICENSE for details.
