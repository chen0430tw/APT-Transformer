# APT-Transformer 配置与调度系统分析报告

## 📊 现状分析

### ✅ 已实现的功能

#### 1. **配置系统** (`apt/core/config.py`)
- ✅ YAML配置文件支持
- ✅ `APTConfig.from_yaml()` 加载配置
- ✅ `schedules` 字段定义（Dict[str, Any]）
- ✅ 插件列表配置
- ✅ 提供商配置（attention_name, ffn_name, router_name等）

#### 2. **YAML配置示例** (`examples/profiles/`)
**`gpt5_moe_reasoning.yaml` 已包含完整的课程调度**:
```yaml
schedules:
  # 插件启用时机
  enable_moe_at_epoch: 2        # epoch=2启用MoE
  enable_align_at_epoch: 3      # epoch=3启用对齐
  enable_voter_at_epoch: 5      # epoch=5启用投票

  # 参数退火
  route_temp:
    start: 1.5
    end: 0.8
    by: "epoch"

  moe_capacity:
    start: 1.5
    end: 1.1
    by: "epoch"

  align_weight:
    start: 0.0
    end: 0.3
    by: "step"
    warmup: 5000
```

#### 3. **训练器** (`apt_model/training/trainer.py`)
- ✅ 基本训练循环
- ✅ 学习率调度器（LR scheduler）
- ✅ 梯度累积
- ✅ 混合精度训练

---

### ❌ 缺失的关键功能

#### 1. **Callback/Hook 机制**
当前训练器**没有回调系统**来执行 `schedules` 配置：

```python
# 当前缺失：
def on_train_epoch_start(epoch, config, modules):
    """每个epoch开始时的钩子"""
    pass

def on_train_batch_end(batch_idx, stats, modules):
    """每个batch结束后的钩子"""
    pass

def on_step(step, config, modules):
    """每个优化步骤的钩子"""
    pass
```

#### 2. **Schedule 执行器**
没有代码读取 `config.schedules` 并执行相应操作：

```python
# 缺失的调度逻辑：
if epoch == config.schedules.get('enable_moe_at_epoch'):
    modules['moe'].enable(True)
```

#### 3. **动态模块启用/禁用**
模块（MoE、Align、Voter等）没有 `enable()`/`disable()` 接口：

```python
# 需要在各模块中实现：
class MoELayer:
    def enable(self, enabled: bool):
        self.enabled = enabled
```

#### 4. **参数退火调度器**
缺少通用的参数插值器（lerp/cosine等）：

```python
# 缺失：
def lerp(start, end, t):
    """线性插值"""
    return start + (end - start) * t

def cosine_anneal(start, end, t):
    """余弦退火"""
    return end + (start - end) * (1 + math.cos(math.pi * t)) / 2
```

---

## 🏗️ 主流做法对比

### DeepSpeed / Megatron-LM 架构
```
配置文件 (YAML/JSON)
    ↓
启动器 (deepspeed/torchrun)
    ↓
Trainer + Callbacks/Hooks
    ↓
on_epoch_start() → 读取schedule → 启用模块/调整参数
```

### HuggingFace Trainer
```python
class CustomCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        if state.epoch == 2:
            model.moe.enable(True)
```

### PyTorch Lightning
```python
class APTLightningModule(LightningModule):
    def on_train_epoch_start(self):
        if self.current_epoch == 2:
            self.model.moe.enable(True)
```

**APT当前架构缺少这一层！**

---

## 🚀 改进方案

### 方案1: 轻量级Callback系统（推荐快速实现）

#### 1.1 创建 `apt_model/training/callbacks.py`

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Training Callbacks and Schedulers

Implements curriculum scheduling and dynamic module control.
"""

import math
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ============================================================================
# Schedule Utilities
# ============================================================================

def lerp(start: float, end: float, t: float) -> float:
    """Linear interpolation."""
    return start + (end - start) * t


def cosine_anneal(start: float, end: float, t: float) -> float:
    """Cosine annealing."""
    return end + (start - end) * (1 + math.cos(math.pi * t)) / 2


def get_interpolator(schedule_type: str = "linear") -> Callable:
    """Get interpolation function by type."""
    interpolators = {
        "linear": lerp,
        "cosine": cosine_anneal,
    }
    return interpolators.get(schedule_type, lerp)


# ============================================================================
# Schedule Executor
# ============================================================================

class ScheduleExecutor:
    """
    Executes curriculum schedules defined in config.

    Example config.schedules:
        {
            "enable_moe_at_epoch": 2,
            "enable_align_at_epoch": 3,
            "route_temp": {
                "start": 1.5,
                "end": 0.8,
                "by": "epoch"
            }
        }
    """

    def __init__(self, config, modules: Dict[str, Any], total_epochs: int, total_steps: int):
        """
        Initialize schedule executor.

        Args:
            config: APTConfig with schedules field
            modules: Dict of modules (e.g., {'moe': moe_layer, 'align': align_layer})
            total_epochs: Total number of training epochs
            total_steps: Total number of training steps
        """
        self.config = config
        self.schedules = config.schedules if hasattr(config, 'schedules') else {}
        self.modules = modules
        self.total_epochs = total_epochs
        self.total_steps = total_steps

        logger.info(f"ScheduleExecutor initialized with {len(self.schedules)} schedules")

    def on_epoch_start(self, epoch: int):
        """Execute schedules at epoch start."""
        # Enable modules at specific epochs
        for key, value in self.schedules.items():
            if key.startswith("enable_") and key.endswith("_at_epoch"):
                if epoch == value:
                    module_name = key.replace("enable_", "").replace("_at_epoch", "")
                    if module_name in self.modules:
                        self.modules[module_name].enable(True)
                        logger.info(f"[Epoch {epoch}] Enabled module: {module_name}")

        # Update parameters with epoch-based schedules
        t = epoch / max(self.total_epochs, 1)
        self._update_parameters(t, by="epoch")

    def on_step(self, step: int):
        """Execute schedules at training step."""
        t = step / max(self.total_steps, 1)
        self._update_parameters(t, by="step")

    def _update_parameters(self, t: float, by: str):
        """Update parameters based on schedule."""
        for key, schedule in self.schedules.items():
            if not isinstance(schedule, dict):
                continue

            if schedule.get("by") != by:
                continue

            # Get start, end values
            start = schedule.get("start")
            end = schedule.get("end")
            if start is None or end is None:
                continue

            # Handle warmup
            warmup = schedule.get("warmup", 0)
            if by == "step" and warmup > 0:
                if t * self.total_steps < warmup:
                    # Warmup phase: 0 -> start
                    value = lerp(0, start, (t * self.total_steps) / warmup)
                else:
                    # Main phase: start -> end
                    t_main = (t * self.total_steps - warmup) / (self.total_steps - warmup)
                    interpolator = get_interpolator(schedule.get("type", "linear"))
                    value = interpolator(start, end, t_main)
            else:
                # No warmup: start -> end
                interpolator = get_interpolator(schedule.get("type", "linear"))
                value = interpolator(start, end, t)

            # Apply value to module
            self._apply_value(key, value)

    def _apply_value(self, param_name: str, value: float):
        """Apply scheduled value to module parameter."""
        # Parse param_name (e.g., "route_temp" -> module=router, attr=temperature)
        param_mapping = {
            "route_temp": ("router", "temperature"),
            "moe_capacity": ("moe", "capacity_factor"),
            "align_weight": ("align", "loss_weight"),
            "vote_threshold": ("voter", "entropy_threshold"),
        }

        if param_name in param_mapping:
            module_name, attr_name = param_mapping[param_name]
            if module_name in self.modules:
                module = self.modules[module_name]
                if hasattr(module, f"set_{attr_name}"):
                    getattr(module, f"set_{attr_name}")(value)
                    logger.debug(f"Updated {module_name}.{attr_name} = {value:.4f}")


# ============================================================================
# Callback Interface
# ============================================================================

class TrainingCallback:
    """Base callback class."""

    def on_train_begin(self, **kwargs):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, **kwargs):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, **kwargs):
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict, **kwargs):
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, batch_idx: int, **kwargs):
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, batch_idx: int, loss: float, **kwargs):
        """Called at the end of each batch."""
        pass

    def on_step(self, step: int, **kwargs):
        """Called after each optimization step."""
        pass


class ScheduleCallback(TrainingCallback):
    """Callback that executes curriculum schedules."""

    def __init__(self, schedule_executor: ScheduleExecutor):
        self.executor = schedule_executor

    def on_epoch_begin(self, epoch: int, **kwargs):
        self.executor.on_epoch_start(epoch)

    def on_step(self, step: int, **kwargs):
        self.executor.on_step(step)


class EntropyBasedVotingCallback(TrainingCallback):
    """Callback that enables voting based on batch entropy."""

    def __init__(self, modules: Dict, threshold: float = 2.2):
        self.modules = modules
        self.threshold = threshold

    def on_batch_end(self, batch_idx: int, loss: float, entropy: Optional[float] = None, **kwargs):
        if entropy is not None and 'voter' in self.modules:
            if entropy > self.threshold:
                self.modules['voter'].enable(True)
            else:
                self.modules['voter'].enable(False)


# ============================================================================
# Callback Manager
# ============================================================================

class CallbackManager:
    """Manages multiple callbacks."""

    def __init__(self, callbacks: List[TrainingCallback]):
        self.callbacks = callbacks

    def trigger(self, event: str, **kwargs):
        """Trigger an event on all callbacks."""
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method:
                method(**kwargs)
```

#### 1.2 修改 `trainer.py` 集成Callbacks

```python
from apt_model.training.callbacks import (
    CallbackManager,
    ScheduleExecutor,
    ScheduleCallback,
    EntropyBasedVotingCallback
)

def train_model(...):
    # ... 现有代码 ...

    # 创建模块字典
    modules = {
        'moe': model.moe_layer if hasattr(model, 'moe_layer') else None,
        'align': model.align_layer if hasattr(model, 'align_layer') else None,
        'voter': model.voter if hasattr(model, 'voter') else None,
        'router': model.router if hasattr(model, 'router') else None,
    }
    modules = {k: v for k, v in modules.items() if v is not None}

    # 创建调度执行器
    total_steps = epochs * len(dataloader)
    schedule_executor = ScheduleExecutor(config, modules, epochs, total_steps)

    # 创建回调
    callbacks = [
        ScheduleCallback(schedule_executor),
        EntropyBasedVotingCallback(modules, threshold=2.2)
    ]
    callback_manager = CallbackManager(callbacks)

    # 触发训练开始
    callback_manager.trigger('on_train_begin')

    # 训练循环
    for epoch in range(epochs):
        # 触发epoch开始
        callback_manager.trigger('on_epoch_begin', epoch=epoch)

        for i, (src_ids, tgt_ids) in enumerate(dataloader):
            # 触发batch开始
            callback_manager.trigger('on_batch_begin', batch_idx=i)

            # ... 现有训练代码 ...

            # 触发batch结束
            callback_manager.trigger('on_batch_end', batch_idx=i, loss=loss_value)

            # 触发step（优化步骤）
            if (i + 1) % accumulation_steps == 0:
                global_step += 1
                callback_manager.trigger('on_step', step=global_step)

        # 触发epoch结束
        callback_manager.trigger('on_epoch_end', epoch=epoch, metrics={})

    # 触发训练结束
    callback_manager.trigger('on_train_end')
```

### 方案2: PyTorch Lightning 重构（推荐长期）

迁移到 Lightning 可以获得：
- 完整的 Callback 生态
- 自动分布式训练
- Checkpoint 管理
- 日志集成（W&B, TensorBoard）

### 方案3: HuggingFace Trainer 适配

使用 HF Trainer + 自定义 Callback

---

## 📝 实施建议

### 立即实施（方案1）

1. **创建 callbacks.py** - 实现上述 ScheduleExecutor 和 CallbackManager
2. **修改 trainer.py** - 集成 callback 系统
3. **为模块添加接口** - 实现 `enable()`, `set_temperature()` 等方法
4. **测试** - 使用 `gpt5_moe_reasoning.yaml` 验证调度功能

### 长期优化（方案2/3）

- 评估迁移到 Lightning 或 HF Trainer 的收益
- 逐步重构训练流程
- 保持向后兼容

---

## 🔍 差距总结

| 功能 | 主流框架 | APT当前 | 缺口 |
|------|---------|---------|------|
| YAML配置 | ✅ | ✅ | 无 |
| Schedule定义 | ✅ | ✅ | 无 |
| Callback机制 | ✅ | ❌ | **关键** |
| Hook执行 | ✅ | ❌ | **关键** |
| 参数退火 | ✅ | ❌ | 重要 |
| 模块动态启用 | ✅ | ❌ | 重要 |
| 一条命令训练 | ✅ | ⚠️ | 可改进 |

---

## 🎯 结论

APT项目**已经有了良好的配置基础**（YAML + schedules），但**缺少执行层**（Callbacks/Hooks）。

**推荐行动**:
1. ✅ 实现 `callbacks.py` (ScheduleExecutor + CallbackManager)
2. ✅ 修改 `trainer.py` 集成 callback 触发点
3. ✅ 为模块添加 `enable()` 和参数设置接口
4. ✅ 测试完整的课程化训练流程

实施后，用户只需：
```bash
python train.py --config profiles/gpt5_moe_reasoning.yaml
```

训练器会自动：
- Epoch 2 启用 MoE
- Epoch 3 启用对齐
- Epoch 5 启用投票
- 路由温度从 1.5 退火到 0.8
- MoE容量从 1.5 收紧到 1.1

**完全符合主流大模型训练的"配置驱动+调度器+启动器"范式！**
