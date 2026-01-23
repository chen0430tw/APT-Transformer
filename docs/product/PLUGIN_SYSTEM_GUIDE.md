# APT插件系统完整指南

**最后更新**: 2024-12-22
**合并自**: PLUGIN_SYSTEM.md, PLUGINS_USAGE_GUIDE.md

<div align="center">

**从架构原理到实战应用的完整教程**

插件系统设计 | 26+ 生产级插件 | 开发指南 | 故障排查

</div>

---

## 📚 目录

### Part 1: 系统架构
1. [插件系统概览](#part-1-插件系统架构)
2. [核心架构设计](#核心架构设计)
3. [事件驱动机制](#事件驱动机制)
4. [优先级与资源管理](#优先级与资源管理)

### Part 2: 插件使用
5. [核心插件](#part-2-插件使用指南)
6. [部署插件](#部署插件)
7. [推理插件](#推理插件)
8. [插件开发](#插件开发)

### Part 3: 高级主题
9. [高级应用](#part-3-高级应用)
10. [故障排查](#故障排查)

---

# Part 1: 插件系统架构

## Overview

APT 插件系统是一个统一的事件驱动插件架构，基于 `memo.txt` 中定义的插件标准实现。它提供了：

- **优先级管理** - 10 级优先级系统（0-999）
- **事件派发** - 统一的生命周期事件
- **冲突检测** - 五层冲突防护机制
- **资源管理** - CPU/GPU/IO 预算控制
- **故障隔离** - 沙箱执行和降级
- **EQI 决策** - 可选的证据推理决策系统

## Architecture

```
Console Core
├── PluginBus (插件总线)
│   ├── 静态冲突检查
│   ├── 事件派发系统
│   ├── 优先级调度
│   ├── 资源管理
│   └── 故障隔离
├── EQI Manager (可选)
│   ├── 证据推理
│   ├── 净效用计算
│   ├── 软门控激活
│   └── 稳定性正则化
└── Plugins (插件)
    ├── GRPO Plugin
    ├── EQI Reporter Plugin
    ├── Route Optimizer Plugin
    └── ... (自定义插件)
```

## Plugin Priority System

插件优先级分为 10 个等级（基于 memo.txt 标准）：

| 优先级范围 | 类别 | 用途 | 示例 |
|-----------|------|------|------|
| 0-49 | Critical | Kill-switch、配置锁、权限校验 | PermissionPlugin |
| 50-149 | CoreRuntime | 推理控制器、解码策略、MoE负载均衡 | InferenceController |
| 150-249 | Performance | 梯度裁剪、显存调度、吞吐优化 | RouteOptimizer |
| 250-349 | Reasoning | Leaf-Vote、自洽重评分、推理链 | ReasoningChain |
| 350-449 | Training | GRPO/RLHF/DPO/ORPO | GRPOPlugin |
| 450-549 | Decision/EQI | EQI、资源优化、配额管理 | EQIManager |
| 550-649 | Admin/Audit | 审计、日志、合规 | AuditPlugin |
| 650-799 | Experimental | 试验性算子、研究功能 | ResearchFeature |
| 800-899 | Telemetry | 指标上报、追踪、监控 | EQIReporter |
| 900-999 | Post/Cleanup | 缓存清理、快照 | CacheCleanup |

**执行顺序**: 插件按优先级升序执行（Critical 最先，Cleanup 最后）

## Plugin Manifest

每个插件必须提供一个 `PluginManifest`，定义插件的元数据和行为：

```python
from apt_model.console.plugin_standards import PluginManifest, PluginPriority, PluginEvent

manifest = PluginManifest(
    # 基本信息
    name="my_plugin",
    version="1.0.0",
    description="My custom plugin",
    author="Your Name",

    # 优先级和行为
    priority=PluginPriority.TRAINING,  # 350-449
    blocking=True,  # 是否阻塞主线程

    # 事件订阅
    events=[
        PluginEvent.ON_BATCH_END,
        PluginEvent.ON_STEP_END
    ],

    # 依赖和冲突
    requires=["core:trainer"],  # 软依赖
    conflicts=["plugin:rlhf"],  # 硬冲突

    # 能力声明
    capabilities=["write_metrics", "read_state"],

    # 资源预算
    resources={
        "cpu_ms": 15.0,   # CPU 时间（毫秒）
        "gpu_ms": 5.0,    # GPU 时间（毫秒）
        "io_mb": 0.5      # I/O 占用（MB）
    },

    # 速率限制
    rate_limit={"steps": 1},  # 每步执行一次

    # 沙箱与容错
    sandbox=True,      # 失败时降级
    fail_limit=5,      # 连续失败 5 次后禁用

    # EQI 参数（可选）
    s_default=0.3,     # 默认净效用
    eta=1.2            # 证据调制参数
)
```

## Creating a Plugin

### 步骤 1: 继承 PluginBase

```python
from apt_model.console.plugin_standards import (
    PluginBase,
    PluginManifest,
    PluginPriority,
    PluginEvent
)

class MyPlugin(PluginBase):
    def __init__(self):
        super().__init__()
        # 初始化插件状态
        self.metrics = {}

    def get_manifest(self) -> PluginManifest:
        """返回插件清单"""
        return PluginManifest(
            name="my_plugin",
            version="1.0.0",
            priority=PluginPriority.TRAINING,
            events=[PluginEvent.ON_BATCH_END]
        )

    def on_batch_end(self, context: Dict[str, Any]):
        """处理 batch 结束事件"""
        step = context['step']
        data = context['data']

        # 处理事件逻辑
        loss = data.get('loss', 0.0)
        print(f"Batch ended at step {step}, loss={loss}")
```

### 步骤 2: 实现事件处理方法

可用的事件类型：

```python
# 训练生命周期
PluginEvent.ON_TRAIN_START
PluginEvent.ON_TRAIN_END

# Epoch 级别
PluginEvent.ON_EPOCH_START
PluginEvent.ON_EPOCH_END

# Batch 级别
PluginEvent.ON_BATCH_START
PluginEvent.ON_BATCH_END

# Step 级别
PluginEvent.ON_STEP_START
PluginEvent.ON_STEP_END
PluginEvent.ON_STEP_EVAL

# 评估
PluginEvent.ON_EVAL_START
PluginEvent.ON_EVAL_END

# 错误处理
PluginEvent.ON_FAIL
PluginEvent.ON_EXCEPTION

# 检查点
PluginEvent.ON_SAVE_CHECKPOINT
PluginEvent.ON_LOAD_CHECKPOINT

# 模型
PluginEvent.ON_MODEL_FORWARD
PluginEvent.ON_MODEL_BACKWARD
```

### 步骤 3: 使用插件私有命名空间

插件可以使用私有命名空间存储状态：

```python
def on_batch_end(self, context: Dict[str, Any]):
    # 存储私有数据
    self.set_context('last_loss', context['data'].get('loss'))

    # 读取私有数据
    last_loss = self.get_context('last_loss', default=0.0)
```

### 步骤 4: 写入公共数据（供其他插件读取）

```python
def on_step_end(self, context: Dict[str, Any]):
    data = context['data']

    # 写入到公共 metrics（其他插件可读）
    if 'metrics' not in data:
        data['metrics'] = {}
    data['metrics']['my_plugin_score'] = 0.95
```

## Using the Plugin System

### 基本用法

```python
from apt_model.console.core import ConsoleCore
from apt_model.console.plugin_standards import PluginEvent
from my_plugin import MyPlugin

# 1. 创建控制台
console = ConsoleCore(config={
    'plugins': {
        'enable_eqi': False,  # 可选启用 EQI
        'default_timeout_ms': 100.0
    }
})

# 2. 注册插件
console.register_plugin(MyPlugin())

# 3. 启动控制台（包括插件编译）
console.start(auto_load_plugins=True)

# 4. 派发事件
context = console.emit_event(
    PluginEvent.ON_BATCH_END,
    step=1,
    context_data={'loss': 0.35}
)

# 5. 获取插件统计
stats = console.get_plugin_statistics()
print(f"Total plugins: {stats['total_plugins']}")
print(f"Active plugins: {stats['active_plugins']}")
```

### 启用 EQI 决策

```python
console = ConsoleCore(config={
    'plugins': {
        'enable_eqi': True,
        'eqi': {
            'time_budget_ms': 20.0,
            'phi_gate': (2.0, 2.0, 1.0, 0.7),  # (a, b, c, d)
            'kappa_stability': 0.1
        }
    }
})
```

## 五层冲突防护机制

插件系统实现了五层冲突防护（基于 memo.txt）：

### 1. 加载期静态检查

编译时检查：
- **依赖检查**: `requires` 字段中的依赖是否满足
- **硬冲突检查**: `conflicts` 字段中的冲突插件是否同时加载
- **能力独占检查**: 独占能力（如 `route_override`）是否被多个插件声明

```python
# 编译时会自动执行
console.compile_plugins(fail_fast=False)
```

### 2. 事件域隔离

插件只能订阅特定事件，不同事件域互不干扰。

### 3. 合并策略

多个插件写入同一字段时的仲裁规则：
- **Last-writer-wins**: 最后写入的插件值生效
- **Accumulate**: 累加所有插件的值
- **Vote**: 投票选择最多的值
- **Override-by-priority**: 高优先级插件覆盖低优先级

### 4. 资源/时延防护

- **预算管理**: 每个插件声明 `cpu_ms`, `gpu_ms`, `io_mb` 预算
- **超时控制**: 阻塞插件有超时限制（基于优先级）
- **速率限制**: `rate_limit` 防止插件过度频繁执行

### 5. 故障隔离与降级

- **Sandbox**: 插件失败不影响主训练循环
- **Fail Limit**: 连续失败超过限制自动禁用
- **熔断**: 可以手动禁用插件

## Example Plugins

### GRPO Plugin (Training Tier)

Group Relative Policy Optimization 插件：

```python
# apt_model/console/plugins/grpo_plugin.py
class GRPOPlugin(PluginBase):
    """GRPO 训练插件"""

    def get_manifest(self):
        return PluginManifest(
            name="grpo",
            priority=PluginPriority.GRPO,  # 380
            events=[
                PluginEvent.ON_BATCH_END,
                PluginEvent.ON_STEP_END
            ],
            conflicts=["plugin:rlhf", "plugin:dpo"]
        )

    def on_batch_end(self, context):
        # 计算组内相对优势
        rewards = context['data'].get('rewards', [])
        # ... GRPO 逻辑
```

### EQI Reporter Plugin (Telemetry Tier)

EQI 指标上报插件：

```python
# apt_model/console/plugins/eqi_reporter_plugin.py
class EQIReporterPlugin(PluginBase):
    """EQI 上报插件"""

    def get_manifest(self):
        return PluginManifest(
            name="eqi_reporter",
            priority=PluginPriority.TRACING,  # 820
            blocking=False,  # 非阻塞
            events=[PluginEvent.ON_STEP_EVAL],
            rate_limit={"steps": 10}  # 每 10 步上报一次
        )

    def on_step_eval(self, context):
        # 收集并上报 EQI 证据
        evidence = context['data'].get('evidence', 1.0)
        # ... 上报逻辑
```

### Route Optimizer Plugin (Performance Tier)

MoE 路由优化插件：

```python
# apt_model/console/plugins/route_optimizer_plugin.py
class RouteOptimizerPlugin(PluginBase):
    """路由优化插件"""

    def get_manifest(self):
        return PluginManifest(
            name="route_optimizer",
            priority=PluginPriority.THROUGHPUT,  # 200
            events=[
                PluginEvent.ON_BATCH_START,
                PluginEvent.ON_STEP_END
            ],
            capabilities=["route_suggest", "read_metrics"]
        )

    def on_batch_start(self, context):
        # 提供路由建议
        suggestions = self._generate_routing_suggestions()
        context['data']['routing_suggestions'] = suggestions
```

## Plugin Capabilities

插件可以声明能力（capabilities），用于冲突检测：

### 独占能力（Exclusive）

只能有一个插件持有：

- `route_override` - 路由控制
- `decode_policy` - 解码策略
- `kill_switch` - 熔断开关

### 共享能力（Shared）

多个插件可以持有：

- `read_metrics` - 读取指标
- `write_metrics` - 写入指标
- `add_constraints` - 添加约束
- `route_suggest` - 路由建议
- `read_state` - 读取状态
- `write_state` - 写入状态

## Console Commands

插件系统提供了一系列 CLI 命令：

```bash
# 列出所有插件
plugins-list

# 显示插件信息
plugins-info <plugin_name>

# 启用/禁用插件
plugins-enable <plugin_name>
plugins-disable <plugin_name>

# 显示插件状态
plugins-status

# 显示插件统计
plugins-stats

# 重新编译插件
plugins-compile
```

## API Reference

### ConsoleCore

```python
class ConsoleCore:
    # 插件管理
    def register_plugin(self, plugin: PluginBase, manifest: Optional[PluginManifest] = None)
    def compile_plugins(self, fail_fast: bool = False)
    def emit_event(self, event: str, step: int, context_data: Optional[Dict[str, Any]] = None) -> EventContext

    # 插件控制
    def get_plugin(self, name: str) -> Optional[PluginBase]
    def enable_plugin(self, name: str)
    def disable_plugin(self, name: str, reason: str = "manual")

    # 统计信息
    def get_plugin_statistics() -> Dict[str, Any]
    def print_plugin_status()
```

### PluginBus

```python
class PluginBus:
    def __init__(self, enable_eqi: bool = False, default_timeout_ms: float = 100.0)

    # 插件注册
    def register(self, plugin: PluginBase, manifest: Optional[PluginManifest] = None)

    # 编译（静态冲突检查）
    def compile(self, fail_fast: bool = False)

    # 事件派发
    def emit(self, event: str, step: int, context_data: Optional[Dict[str, Any]] = None) -> EventContext

    # 插件管理
    def get_plugin(self, name: str) -> Optional[PluginBase]
    def enable_plugin(self, name: str)
    def disable_plugin(self, name: str, reason: str = "manual")

    # 统计
    def get_statistics() -> Dict[str, Any]
    def print_status()
```

### PluginBase

```python
class PluginBase:
    # 必须实现
    def get_manifest(self) -> PluginManifest

    # 可选实现
    def initialize(self, config: Optional[Dict[str, Any]] = None)
    def cleanup()

    # 私有命名空间
    def get_context(self, key: str, default: Any = None) -> Any
    def set_context(self, key: str, value: Any)

    # 事件处理方法（可选实现）
    def on_train_start(self, context: Dict[str, Any])
    def on_epoch_end(self, context: Dict[str, Any])
    def on_batch_end(self, context: Dict[str, Any])
    # ... 等
```

### EventContext

```python
@dataclass
class EventContext:
    event: str                  # 事件名称
    step: int                   # 当前步数
    epoch: Optional[int]        # 当前 epoch
    data: Dict[str, Any]        # 公共数据
    plugin_ns: Dict[str, Dict]  # 插件私有命名空间
    merged: Dict[str, Any]      # 合并后的结果

    # 方法
    def get(self, key: str, default: Any = None) -> Any
    def set(self, key: str, value: Any)
    def get_plugin_data(self, plugin_name: str, key: str, default: Any = None) -> Any
    def set_plugin_data(self, plugin_name: str, key: str, value: Any)
```

## Best Practices

### 1. 选择正确的优先级

根据插件的关键程度选择合适的优先级等级：
- 关键路径操作使用 Critical/CoreRuntime
- 性能优化使用 Performance
- 训练算法使用 Training
- 监控上报使用 Telemetry

### 2. 声明准确的资源预算

准确声明 `cpu_ms`, `gpu_ms`, `io_mb`，帮助系统做资源管理。

### 3. 使用速率限制

频繁执行的插件应该设置 `rate_limit` 避免性能影响。

### 4. 启用沙箱模式

除非绝对必要，应该设置 `sandbox=True` 确保插件失败不影响主流程。

### 5. 处理事件失败

```python
def on_batch_end(self, context):
    try:
        # 插件逻辑
        pass
    except Exception as e:
        logger.error(f"Plugin error: {e}")
        # 优雅降级
```

### 6. 文档化插件行为

在插件 docstring 中明确说明：
- 插件的功能
- 订阅的事件
- 读写的数据字段
- 对其他插件的影响

## Testing

运行插件系统测试：

```bash
# 完整测试（需要 torch）
python tests/test_plugin_system.py

# 独立测试（不需要 torch）
python tests/test_plugin_system_standalone.py
```

## Troubleshooting

### 插件未执行

1. 检查插件是否已注册：`console.get_plugin_statistics()`
2. 检查插件是否已编译：`console.compile_plugins()`
3. 检查插件是否被禁用：`plugins-info <name>`
4. 检查事件名称是否正确

### 插件冲突

如果插件被禁用因为冲突：
1. 检查 `conflicts` 字段
2. 检查 `requires` 依赖是否满足
3. 检查能力独占冲突

### 性能问题

1. 检查插件统计：`plugins-stats`
2. 查看平均耗时（avg_time_ms）
3. 考虑增加 `rate_limit`
4. 将 `blocking=False` 改为异步执行

## References

- `memo.txt` - 插件标准规范
- `apt_model/console/plugin_standards.py` - 插件标准实现
- `apt_model/console/plugin_bus.py` - 插件总线实现
- `apt_model/console/core.py` - Console Core 集成
- `apt_model/console/eqi_manager.py` - EQI Manager 实现

---

# Part 2: 插件使用指南


## 🎯 插件系统概览

### 什么是插件系统？

APT 插件系统是一个**事件驱动、优先级管理、资源可控**的统一插件架构，支持：

| 特性 | 说明 | 优势 |
|------|------|------|
| **事件驱动** | 15+ 生命周期事件（训练/推理/解码） | 灵活介入模型流程 |
| **优先级管理** | 10 级优先级（0-999） | 精确控制执行顺序 |
| **资源控制** | CPU/GPU/IO 预算管理 | 防止资源过载 |
| **冲突检测** | 5 层冲突防护机制 | 避免插件冲突 |
| **故障隔离** | 沙箱执行 + 降级策略 | 保证系统稳定性 |

### 插件分类

```
APT 插件生态系统
├── Critical (0-49) - Kill-switch、权限校验
├── CoreRuntime (50-149) - 推理控制、解码策略
├── Performance (150-249) - 路由优化、梯度裁剪
├── Reasoning (250-349) - Beam Search、自洽推理、程序辅助
├── Training (350-449) - GRPO、RLHF、DPO
├── Decision (450-549) - EQI 决策、资源优化
├── Admin (550-649) - 审计、日志、合规
├── Experimental (650-799) - 研究功能
└── Telemetry (800-899) - 指标上报、监控
```

### 快速开始

```python
from apt_model.console.plugin_bus import PluginBus
from apt_model.console.plugins.grpo_plugin import GRPOPlugin

# 1. 创建插件总线
bus = PluginBus()

# 2. 注册插件
grpo = GRPOPlugin()
bus.register(grpo)

# 3. 初始化插件
grpo.initialize({
    'group_size': 4,
    'learning_rate': 1e-5,
    'policy_model': policy_model,
    'reward_model': reward_model
})

# 4. 触发事件
bus.dispatch_event('on_batch_end', context={
    'step': 100,
    'data': {'rewards': [0.8, 0.9, 0.7, 0.85]}
})
```

---

## 🔧 核心插件

### 1. GRPO Plugin（强化学习训练）

#### 功能概述

**GRPO（Group Relative Policy Optimization）** 是一种组相对策略优化算法，通过**组内比较**来训练策略模型。

**核心思想**：
```
传统 RLHF:
每个响应独立计算奖励 → 训练策略模型

GRPO:
组内响应相互比较 → 计算相对优势 → 训练策略模型
```

**优势**：
- ✅ 更稳定的训练（组内归一化）
- ✅ 减少奖励模型偏差
- ✅ 更好的泛化能力

#### 使用方法

**1. 基础使用**

```python
from apt_model.console.plugins.grpo_plugin import GRPOPlugin
from apt_model.rl.grpo_trainer import GRPOTrainer, GRPOConfig

# 创建 GRPO 插件
grpo_plugin = GRPOPlugin()

# 配置
config = {
    'group_size': 4,  # 每组 4 个响应
    'learning_rate': 1e-5,
    'advantage_type': 'relative',  # 相对优势
    'policy_model': policy_model,
    'reward_model': reward_model,
    'device': 'cuda'
}

# 初始化
grpo_plugin.initialize(config)

# 注册到插件总线
bus.register(grpo_plugin)
```

**2. 训练循环集成**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
policy_model = AutoModelForCausalLM.from_pretrained('gpt2-medium')
reward_model = AutoModelForCausalLM.from_pretrained('gpt2-reward')
tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')

# 创建 GRPO 训练器
from apt_model.rl.grpo_trainer import GRPOTrainer, GRPOConfig

grpo_config = GRPOConfig(
    group_size=4,
    learning_rate=1e-5,
    advantage_type='relative',
    beta=0.01,  # KL 散度系数
    clip_range=0.2  # PPO 裁剪范围
)

trainer = GRPOTrainer(
    policy_model=policy_model,
    reward_model=reward_model,
    config=grpo_config,
    device='cuda'
)

# 训练循环
for batch in dataloader:
    # 1. 生成响应（每个 prompt 生成 4 个响应）
    responses = []
    for i in range(grpo_config.group_size):
        output = policy_model.generate(
            batch['input_ids'],
            max_length=512,
            do_sample=True,
            temperature=0.7 + i * 0.1  # 不同温度生成多样响应
        )
        responses.append(output)

    responses = torch.stack(responses, dim=1)  # [batch, group_size, seq_len]
    response_masks = (responses != tokenizer.pad_token_id).long()

    # 2. 计算奖励（可选）
    with torch.no_grad():
        rewards = reward_model(responses).logits[:, :, -1].mean(dim=-1)

    # 3. GRPO 训练步骤
    stats = trainer.train_step(
        responses=responses,
        response_masks=response_masks,
        rewards=rewards
    )

    print(f"Step {trainer.step}: "
          f"policy_loss={stats['policy_loss']:.4f}, "
          f"group_variance={stats['group_variance']:.4f}, "
          f"kl={stats['kl_divergence']:.4f}")
```

**3. 使用插件总线集成**

```python
# 创建插件总线
bus = PluginBus()

# 注册 GRPO 插件
grpo_plugin = GRPOPlugin()
grpo_plugin.initialize({
    'group_size': 4,
    'policy_model': policy_model,
    'reward_model': reward_model,
    'learning_rate': 1e-5
})
bus.register(grpo_plugin)

# 训练循环
for step, batch in enumerate(dataloader):
    # 生成响应
    responses = generate_group_responses(batch, group_size=4)
    response_masks = (responses != tokenizer.pad_token_id).long()

    # 分发事件 - 自动触发 GRPO 训练
    bus.dispatch_event('on_step_end', context={
        'step': step,
        'data': {
            'responses': responses,
            'response_masks': response_masks
        }
    })

    # 读取指标
    if step % 100 == 0:
        metrics = bus.get_plugin_metrics('grpo')
        print(f"Step {step}: GRPO metrics: {metrics}")
```

#### 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `group_size` | int | 4 | 每组响应数量 |
| `learning_rate` | float | 1e-5 | 策略模型学习率 |
| `advantage_type` | str | 'relative' | 优势类型（relative/normalized/rank） |
| `beta` | float | 0.01 | KL 散度系数 |
| `clip_range` | float | 0.2 | PPO 裁剪范围 |
| `policy_model` | nn.Module | - | 策略模型 |
| `reward_model` | nn.Module | - | 奖励模型 |

#### 输出指标

```python
# 插件输出的指标
metrics = {
    'grpo_variance': 0.042,        # 组内方差
    'grpo_updates': 1500,          # 策略更新次数
    'grpo_policy_loss': 0.312,     # 策略损失
    'grpo_kl': 0.008,              # KL 散度
}
```

---

### 2. Route Optimizer（MoE负载均衡）

#### 功能概述

**Route Optimizer** 用于优化 Mixture-of-Experts (MoE) 模型的**专家负载均衡**。

**核心问题**：
```
MoE 模型问题:
某些专家过载 → 其他专家闲置 → 计算效率低下

Route Optimizer 解决方案:
实时监控负载 → 检测过载 → 提供路由建议 → 动态调整
```

**优势**：
- ✅ 提升 MoE 模型效率
- ✅ 防止专家崩溃
- ✅ 负载可视化

#### 使用方法

**1. 基础使用**

```python
from apt_model.console.plugins.route_optimizer_plugin import RouteOptimizerPlugin

# 创建插件
route_opt = RouteOptimizerPlugin()

# 配置
route_opt.initialize({
    'num_experts': 8,           # 专家数量
    'load_threshold': 1.5,      # 过载阈值（平均值的 1.5 倍）
})

# 注册到插件总线
bus.register(route_opt)
```

**2. 与 MoE 模型集成**

```python
import torch
import torch.nn as nn
from apt_model.modeling.moe import MoELayer

class MoEModel(nn.Module):
    def __init__(self, d_model=512, num_experts=8):
        super().__init__()
        self.moe_layer = MoELayer(d_model, num_experts)

    def forward(self, x, routing_suggestions=None):
        # 使用路由建议（如果有）
        if routing_suggestions:
            expert_weights = self._adjust_routing(
                self.moe_layer.gate(x),
                routing_suggestions
            )
        else:
            expert_weights = self.moe_layer.gate(x)

        # MoE 前向传播
        output = self.moe_layer(x, expert_weights)
        return output, expert_weights

    def _adjust_routing(self, gate_logits, suggestions):
        """根据建议调整路由"""
        weights = torch.softmax(gate_logits, dim=-1)

        # 增强欠载专家的权重
        underloaded = suggestions['underloaded_expert']
        overloaded = suggestions['overloaded_expert']

        weights[:, underloaded] *= 1.2
        weights[:, overloaded] *= 0.8

        # 重新归一化
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return weights

# 创建模型
model = MoEModel(d_model=512, num_experts=8)

# 创建插件总线并注册 Route Optimizer
bus = PluginBus()
route_opt = RouteOptimizerPlugin()
route_opt.initialize({'num_experts': 8})
bus.register(route_opt)

# 训练循环
for step, batch in enumerate(dataloader):
    # Batch 开始事件
    bus.dispatch_event('on_batch_start', context={
        'step': step,
        'data': {}
    })

    # 获取路由建议
    suggestions = bus.get_context('route_optimizer', 'routing_suggestions')

    # 前向传播
    output, expert_weights = model(batch['input'], suggestions)

    # 记录路由信息
    expert_ids = expert_weights.argmax(dim=-1).cpu().numpy()

    # Step 结束事件
    bus.dispatch_event('on_step_end', context={
        'step': step,
        'data': {
            'routing': {
                'expert_ids': expert_ids.tolist()
            }
        }
    })

    # 读取指标
    if step % 100 == 0:
        metrics = bus.get_plugin_metrics('route_optimizer')
        print(f"Step {step}: Load variance={metrics['route_variance']:.4f}, "
              f"Efficiency={metrics['route_efficiency']:.4f}")
```

**3. 实时监控**

```python
import matplotlib.pyplot as plt

# 监控插件（每 10 步）
load_history = []

for step in range(1000):
    # ... 训练代码 ...

    if step % 10 == 0:
        # 获取负载历史
        history = route_opt.routing_history[-10:]
        if history:
            avg_loads = [sum(r['loads']) / len(r['loads']) for r in history]
            load_history.extend(avg_loads)

# 绘制负载曲线
plt.figure(figsize=(10, 5))
plt.plot(load_history)
plt.title('Expert Load Distribution Over Time')
plt.xlabel('Step (x10)')
plt.ylabel('Average Load')
plt.savefig('expert_load.png')
```

#### 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_experts` | int | 8 | MoE 专家数量 |
| `load_threshold` | float | 1.5 | 过载阈值（相对平均值） |

#### 输出指标

```python
metrics = {
    'route_variance': 0.123,       # 负载方差
    'route_efficiency': 0.89,      # 路由效率（0-1）
    'overload_events': 12,         # 过载事件次数
    'adjustments_made': 8          # 路由调整次数
}

# 路由建议格式
suggestions = {
    'underloaded_expert': 2,       # 欠载专家 ID
    'overloaded_expert': 5,        # 过载专家 ID
    'avg_loads': [1.2, 0.8, ...],  # 平均负载
    'recommendation': 'redirect'    # 建议（redirect/balanced）
}
```

---

### 3. EQI Reporter（指标上报）

#### 功能概述

**EQI Reporter** 用于收集和上报 **Evidence Qualitative Inference (EQI)** 指标。

**EQI 是什么？**
```
EQI = Evidence-based Qualitative Inference
证据驱动的定性推理决策系统

核心公式:
φ(s, E, κ) = sigmoid(κ(s + η·evidence(E)))

其中:
- s: 净效用 (net utility) = Latency - λ·Importance
- E: 证据（历史性能数据）
- η: 证据调制参数
- κ: 门控陡峭度
```

**用途**：
- ✅ 追踪插件激活证据
- ✅ 监控净效用趋势
- ✅ 可视化软门控激活
- ✅ 辅助插件调优

#### 使用方法

**1. 基础使用**

```python
from apt_model.console.plugins.eqi_reporter_plugin import EQIReporterPlugin

# 创建插件
eqi_reporter = EQIReporterPlugin()

# 配置
eqi_reporter.initialize({
    'report_interval': 100  # 每 100 步上报一次
})

# 注册到插件总线
bus.register(eqi_reporter)
```

**2. 与监控系统集成**

```python
import requests
import json

class EQIReporterWithAPI(EQIReporterPlugin):
    """扩展版 EQI Reporter：上报到监控 API"""

    def __init__(self, api_endpoint: str):
        super().__init__()
        self.api_endpoint = api_endpoint

    def _send_report(self, step: int, epoch: int = None):
        """重写上报方法：发送到 API"""
        report = {
            'step': step,
            'epoch': epoch,
            'timestamp': time.time(),
            'evidence_mean': self.metrics['evidence_mean'],
            'utility_mean': self.metrics['utility_mean'],
            'activations': self.metrics['activations'],
            'log_size': len(self.evidence_log)
        }

        try:
            # 发送到监控 API
            response = requests.post(
                f"{self.api_endpoint}/eqi-metrics",
                json=report,
                timeout=5
            )

            if response.status_code == 200:
                logger.info(f"[EQI Reporter] Sent report to API: {report}")
                self.metrics['reports_sent'] += 1
            else:
                logger.warning(f"[EQI Reporter] API returned {response.status_code}")

        except Exception as e:
            logger.error(f"[EQI Reporter] Failed to send report: {e}")

        # 存储到上下文
        self.set_context('last_report', report)

# 使用
eqi_reporter = EQIReporterWithAPI(api_endpoint="http://localhost:8080")
eqi_reporter.initialize({'report_interval': 50})
bus.register(eqi_reporter)
```

**3. 可视化 EQI 指标**

```python
import matplotlib.pyplot as plt
import numpy as np

# 收集数据
steps = []
evidence_means = []
utility_means = []

for step in range(1000):
    # ... 训练代码 ...

    # 触发评估事件
    bus.dispatch_event('on_step_eval', context={
        'step': step,
        'data': {
            'metrics': {'loss': 0.5},
            'evidence': 0.8 + np.random.randn() * 0.1,
            'utility': 0.6 + np.random.randn() * 0.05
        }
    })

    if step % 10 == 0:
        report = eqi_reporter.get_context('last_report')
        if report:
            steps.append(step)
            evidence_means.append(report['evidence_mean'])
            utility_means.append(report['utility_mean'])

# 绘制双轴图
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.set_xlabel('Training Step')
ax1.set_ylabel('Evidence Mean', color='tab:blue')
ax1.plot(steps, evidence_means, color='tab:blue', label='Evidence')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Utility Mean', color='tab:orange')
ax2.plot(steps, utility_means, color='tab:orange', label='Utility')
ax2.tick_params(axis='y', labelcolor='tab:orange')

plt.title('EQI Metrics Over Training')
plt.savefig('eqi_metrics.png')
```

#### 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `report_interval` | int | 100 | 上报间隔（步数） |

#### 输出指标

```python
metrics = {
    'evidence_mean': 0.82,         # 证据均值
    'utility_mean': 0.65,          # 净效用均值
    'activations': 1523,           # 激活次数
    'reports_sent': 15             # 上报次数
}

# 报告格式
report = {
    'step': 1000,
    'epoch': 5,
    'timestamp': 1701234567.89,
    'evidence_mean': 0.82,
    'utility_mean': 0.65,
    'activations': 1523,
    'log_size': 1000
}
```

---

## 🚀 部署插件

### 1. Ollama Export (本地部署)

#### 功能概述

**Ollama Export Plugin** 将 APT 模型导出为 **Ollama 格式**，支持本地部署和推理。

**核心功能**:
```
APT 模型 → GGUF格式 → Modelfile配置 → Ollama注册 → 本地运行
```

**支持的量化方式**:
- ✅ Q4_0 - 4位量化 (最小体积)
- ✅ Q4_K_M - 4位K-quants (推荐)
- ✅ Q5_K_M - 5位K-quants (平衡)
- ✅ Q8_0 - 8位量化 (高质量)
- ✅ F16 - 半精度浮点

**优势**:
- ✅ 本地部署无需云端
- ✅ 模型体积减小 70-80%
- ✅ 推理速度提升
- ✅ 隐私保护

#### 使用方法

**1. 基础使用**

```python
from apt_model.plugins.ollama_export_plugin import OllamaExportPlugin

# 创建插件
config = {
    'quantization': 'Q4_K_M',    # 量化类型
    'context_length': 2048,       # 上下文长度
    'temperature': 0.7,           # 采样温度
}

plugin = OllamaExportPlugin(config)

# 完整导出流程
results = plugin.export_complete(
    model_path="./trained_model",      # APT模型路径
    output_dir="./ollama_export",      # 输出目录
    model_name="apt-chinese",          # Ollama模型名称
    register=True                      # 自动注册到Ollama
)

print(f"✅ GGUF文件: {results['gguf_path']}")
print(f"✅ Modelfile: {results['modelfile_path']}")
print(f"✅ 已注册: {results['registered']}")
```

**2. 分步导出**

```python
# Step 1: 转换为GGUF格式
gguf_path = plugin.export_to_gguf(
    model_path="./trained_model",
    output_path="./apt-model.gguf",
    quantization="Q4_K_M"
)

# Step 2: 创建Modelfile
modelfile_path = plugin.create_modelfile(
    gguf_path=gguf_path,
    system_prompt="你是一个由APT模型驱动的AI助手。",
    template="""{{ if .System }}{{ .System }}{{ end }}
{{ if .Prompt }}用户: {{ .Prompt }}{{ end }}
助手: """
)

# Step 3: 注册到Ollama
success = plugin.register_to_ollama(
    modelfile_path=modelfile_path,
    model_name="apt-chinese:latest"
)

if success:
    print("✅ 模型已注册到Ollama!")
    print("运行: ollama run apt-chinese:latest")
```

**3. 训练后自动导出**

```python
# 配置自动导出
config = {
    'quantization': 'Q4_K_M',
    'auto_export': True,           # 训练结束自动导出
    'auto_register': True,         # 自动注册到Ollama
    'output_dir': './ollama_models'
}

plugin = OllamaExportPlugin(config)

# 在训练循环中注册插件
from apt_model.console.plugin_bus import PluginBus
bus = PluginBus()
bus.register(plugin)

# 训练结束后会自动触发导出
# bus.dispatch_event('on_training_end', context={
#     'checkpoint_path': './final_model'
# })
```

**4. 测试导出的模型**

```python
# 测试模型
response = plugin.test_model(
    model_name="apt-chinese:latest",
    prompt="你好！介绍一下你自己。"
)

print(f"模型响应: {response}")

# 或使用命令行
# $ ollama run apt-chinese:latest
# >>> 你好！介绍一下你自己。
# 你好！我是一个由APT模型驱动的AI助手...
```

**5. 不同量化方式对比**

```python
# 导出多个量化版本进行对比
quantizations = ['Q4_0', 'Q4_K_M', 'Q5_K_M', 'Q8_0']

for quant in quantizations:
    plugin = OllamaExportPlugin({'quantization': quant})

    results = plugin.export_complete(
        model_path="./trained_model",
        output_dir=f"./ollama_export_{quant}",
        model_name=f"apt-model-{quant.lower()}",
        register=True
    )

    # 检查文件大小
    import os
    size_mb = os.path.getsize(results['gguf_path']) / (1024 * 1024)
    print(f"{quant}: {size_mb:.2f} MB")

# 输出示例:
# Q4_0:   1250.32 MB  (最小)
# Q4_K_M: 1380.45 MB  (推荐)
# Q5_K_M: 1620.78 MB  (平衡)
# Q8_0:   2340.92 MB  (高质量)
```

#### 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `quantization` | str | 'Q4_K_M' | 量化类型 |
| `context_length` | int | 2048 | 上下文长度 |
| `temperature` | float | 0.7 | 采样温度 |
| `auto_export` | bool | False | 训练后自动导出 |
| `auto_register` | bool | False | 自动注册到Ollama |
| `output_dir` | str | './ollama_export' | 输出目录 |

#### 量化方式说明

| 量化类型 | 精度 | 体积 | 速度 | 适用场景 |
|---------|------|------|------|---------|
| **Q4_0** | ⭐⭐ | 最小 | 最快 | 资源受限环境 |
| **Q4_K_M** | ⭐⭐⭐ | 小 | 快 | **推荐用于生产** |
| **Q5_K_M** | ⭐⭐⭐⭐ | 中 | 中 | 质量要求高 |
| **Q8_0** | ⭐⭐⭐⭐⭐ | 大 | 慢 | 最高质量 |
| **F16** | ⭐⭐⭐⭐⭐ | 最大 | 最慢 | 研究/对比 |

#### Modelfile 自定义

```python
# 创建自定义Modelfile
custom_system_prompt = """你是一个专业的中文AI助手，专注于以下领域:
- 技术问答
- 代码生成
- 文档写作

请用简洁、专业的语言回答问题。"""

custom_template = """{{ if .System }}系统: {{ .System }}

{{ end }}{{ if .Prompt }}用户: {{ .Prompt }}
{{ end }}助手: """

modelfile_path = plugin.create_modelfile(
    gguf_path="./apt-model.gguf",
    output_path="./Modelfile",
    system_prompt=custom_system_prompt,
    template=custom_template
)
```

#### 命令行使用

导出后可以直接用Ollama命令行:

```bash
# 运行模型
ollama run apt-chinese:latest

# 交互式对话
>>> 你好！
你好！我是一个由APT模型驱动的AI助手...

>>> 请用Python写一个快速排序
当然，下面是Python实现的快速排序算法:
```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    ...
```

# 查看模型列表
ollama list

# 删除模型
ollama rm apt-chinese:latest

# 复制模型
ollama cp apt-chinese:latest apt-chinese:backup
```

#### 输出文件结构

```
ollama_export/
├── apt-chinese.gguf         # GGUF模型文件 (量化后)
├── Modelfile                 # Ollama配置文件
└── README.md                 # 使用说明 (可选)
```

#### 故障排查

**1. Ollama未安装**

```
❌ Ollama命令未找到
```

**解决方案**:
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# 访问 https://ollama.ai/download
```

**2. GGUF转换失败**

```
❌ GGUF转换失败: KeyError: 'model.embed_tokens.weight'
```

**解决方案**:
```python
# 确保模型路径正确，包含 pytorch_model.bin
import os
print(os.listdir("./trained_model"))
# 应该看到: ['pytorch_model.bin', 'config.json', ...]

# 或者使用HuggingFace格式
model = AutoModelForCausalLM.from_pretrained("./trained_model")
model.save_pretrained("./trained_model_fixed")
```

**3. 注册失败**

```
❌ 注册失败: Error: model already exists
```

**解决方案**:
```bash
# 删除旧模型
ollama rm apt-chinese:latest

# 或使用不同的标签
python export.py --model-name apt-chinese:v2
```

---

## 🧠 推理插件

### 1. Beam Search（多路径搜索）

#### 功能概述

**Beam Search** 是一种**多路径搜索算法**，维护 k 个候选推理路径，选择得分最高的路径作为最终答案。

**核心思想**：
```
贪婪搜索:
每步选择概率最高的 token → 可能陷入局部最优

Beam Search:
每步维护 k 个候选路径 → 综合考虑全局得分 → 选择最优路径
```

**适用场景**：
- ✅ 数学推理（需要精确步骤）
- ✅ 代码生成（需要正确语法）
- ✅ 逻辑推理（需要完整推理链）

#### 使用方法

**1. 基础使用**

```python
from apt_model.console.plugins.reasoning.beam_search_plugin import BeamSearchReasoningPlugin

# 创建插件
beam_search = BeamSearchReasoningPlugin(config={
    'beam_width': 4,           # Beam 宽度（候选数量）
    'length_penalty': 0.6,     # 长度惩罚参数
    'max_steps': 50,           # 最大推理步数
    'diversity_penalty': 0.5,  # 多样性惩罚
    'early_stopping': True     # 早停
})

# 注册到插件总线
bus.register(beam_search)
```

**2. 推理示例**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
model = AutoModelForCausalLM.from_pretrained('gpt2-medium')
tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')

# 创建插件总线
bus = PluginBus()
beam_search = BeamSearchReasoningPlugin(config={'beam_width': 5})
bus.register(beam_search)

# 推理
question = "What is 15% of 80?"
input_ids = tokenizer.encode(question, return_tensors='pt')

# 触发推理事件
bus.dispatch_event('on_inference_start', context={
    'data': {
        'use_beam_search': True,
        'model': model,
        'tokenizer': tokenizer,
        'input_ids': input_ids
    }
})

# 触发解码事件
bus.dispatch_event('on_decode', context={
    'step': 0,
    'data': {
        'model': model,
        'tokenizer': tokenizer,
        'input_ids': input_ids
    }
})

# 获取结果
result = bus.get_data('beam_search_result')
print(f"Best path: {result['path']}")
print(f"Score: {result['score']:.4f}")
print(f"Steps: {result['num_steps']}")
```

**3. 自定义评分函数**

```python
class CustomBeamSearch(BeamSearchReasoningPlugin):
    """自定义 Beam Search：添加推理正确性评分"""

    def __init__(self, config):
        super().__init__(config)
        self.correctness_weight = 0.3  # 正确性权重

    def _score_beam(self, beam, model, tokenizer):
        """自定义评分：语言模型得分 + 正确性得分"""
        # 原始得分（语言模型概率）
        lm_score = beam.normalized_score(self.length_penalty)

        # 正确性得分（检查推理链是否合理）
        correctness_score = self._check_reasoning(beam.tokens, tokenizer)

        # 综合得分
        final_score = (1 - self.correctness_weight) * lm_score + \
                      self.correctness_weight * correctness_score

        return final_score

    def _check_reasoning(self, tokens, tokenizer):
        """检查推理链正确性"""
        text = tokenizer.decode(tokens)

        # 简单启发式：检查是否包含推理关键词
        reasoning_keywords = ['because', 'therefore', 'so', '因此', '所以']
        score = sum(1 for kw in reasoning_keywords if kw in text.lower())

        return min(score / len(reasoning_keywords), 1.0)

# 使用
custom_beam = CustomBeamSearch(config={'beam_width': 4})
bus.register(custom_beam)
```

#### 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `beam_width` | int | 4 | Beam 宽度（候选路径数） |
| `length_penalty` | float | 0.6 | 长度惩罚（0=无惩罚，1=全惩罚） |
| `max_steps` | int | 50 | 最大推理步数 |
| `diversity_penalty` | float | 0.0 | 多样性惩罚（鼓励不同路径） |
| `early_stopping` | bool | True | 早停（所有路径完成时停止） |

#### 输出结果

```python
result = {
    'path': [101, 2054, 2003, ...],  # Token IDs
    'score': -4.23,                  # 归一化得分
    'num_steps': 12,                 # 实际步数
    'beam_width': 4                  # Beam 宽度
}
```

---

### 2. Self-Consistency（自洽推理）

#### 功能概述

**Self-Consistency** 通过生成**多条独立推理路径**，然后**投票选择最一致**的答案来提升推理可靠性。

**核心思想**：
```
单次生成:
prompt → 模型生成 → 答案（可能错误）

Self-Consistency:
prompt → 生成 N 条路径 → 提取答案 → 投票 → 最一致答案（更可靠）
```

**优势**：
- ✅ 提升推理准确性（尤其数学题）
- ✅ 提供置信度评分
- ✅ 捕获多样推理方式

#### 使用方法

**1. 基础使用**

```python
from apt_model.console.plugins.reasoning.self_consistency_plugin import SelfConsistencyPlugin

# 创建插件
sc_plugin = SelfConsistencyPlugin(config={
    'num_paths': 5,            # 生成 5 条推理路径
    'temperature': 0.7,        # 采样温度（多样性）
    'answer_patterns': [       # 答案提取模式
        r'[Aa]nswer:\s*(.+)',
        r'答案[:：]\s*(.+)',
        r'因此[:：]\s*(.+)',
    ]
})

# 注册到插件总线
bus.register(sc_plugin)
```

**2. 完整推理示例**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
model = AutoModelForCausalLM.from_pretrained('gpt2-medium')
tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')

# 创建插件总线
bus = PluginBus()
sc_plugin = SelfConsistencyPlugin(config={'num_paths': 10, 'temperature': 0.8})
bus.register(sc_plugin)

# 推理
question = "If a train travels at 60 mph for 2.5 hours, how far does it travel?"

# 触发推理事件
bus.dispatch_event('on_inference_start', context={
    'data': {
        'use_self_consistency': True
    }
})

# 触发解码事件
bus.dispatch_event('on_decode', context={
    'step': 0,
    'data': {
        'model': model,
        'tokenizer': tokenizer,
        'input_text': question
    }
})

# 获取结果
result = bus.get_data('self_consistency_result')

print(f"Selected Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Paths Generated: {result['paths_count']}")
print(f"Vote Distribution: {result['vote_distribution']}")

# 示例输出:
# Selected Answer: 150 miles
# Confidence: 80.00%
# Paths Generated: 10
# Vote Distribution: {'150 miles': 8, '15 miles': 1, '160 miles': 1}
```

**3. 自定义答案提取**

```python
class MathSelfConsistency(SelfConsistencyPlugin):
    """数学题专用 Self-Consistency：提取数值答案"""

    def _extract_answer(self, reasoning_path: str) -> str:
        """重写答案提取：专门提取数值"""
        import re

        # 优先匹配明确的答案标记
        for pattern in self.answer_patterns:
            match = re.search(pattern, reasoning_path, re.MULTILINE)
            if match:
                answer_text = match.group(1).strip()
                # 提取数值
                numbers = re.findall(r'[-+]?\d*\.?\d+', answer_text)
                if numbers:
                    return numbers[0]  # 返回第一个数值

        # 回退：提取路径中最后出现的数值
        numbers = re.findall(r'[-+]?\d*\.?\d+', reasoning_path)
        if numbers:
            return numbers[-1]

        return ""

    def _normalize_answer(self, answer: str) -> str:
        """归一化数值答案"""
        try:
            # 转换为浮点数再转回字符串（统一格式）
            num = float(answer)
            # 如果是整数，去掉小数点
            if num == int(num):
                return str(int(num))
            return f"{num:.2f}"  # 保留 2 位小数
        except:
            return answer.lower().strip()

# 使用
math_sc = MathSelfConsistency(config={'num_paths': 5})
bus.register(math_sc)
```

**4. 与 Chain-of-Thought 结合**

```python
def self_consistency_with_cot(model, tokenizer, question, num_paths=5):
    """Self-Consistency + Chain-of-Thought"""

    # CoT Prompt
    cot_prompt = f"""Let's solve this step by step:

Question: {question}

Step-by-step solution:"""

    paths = []
    answers = []

    for i in range(num_paths):
        # 生成推理路径（不同温度）
        input_ids = tokenizer.encode(cot_prompt, return_tensors='pt')
        output = model.generate(
            input_ids,
            max_length=200,
            do_sample=True,
            temperature=0.7 + i * 0.05,  # 递增温度
            top_p=0.95
        )

        path = tokenizer.decode(output[0], skip_special_tokens=True)
        paths.append(path)

        # 提取答案
        answer = extract_final_answer(path)
        answers.append(answer)

    # 投票
    from collections import Counter
    vote_counts = Counter(answers)
    best_answer = vote_counts.most_common(1)[0]

    return {
        'answer': best_answer[0],
        'confidence': best_answer[1] / num_paths,
        'paths': paths,
        'all_answers': answers
    }

# 使用
result = self_consistency_with_cot(
    model, tokenizer,
    question="What is 15% of 80?",
    num_paths=10
)
```

#### 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_paths` | int | 5 | 生成路径数量 |
| `temperature` | float | 0.7 | 采样温度（多样性） |
| `answer_patterns` | list | [...] | 答案提取正则模式 |

#### 输出结果

```python
result = {
    'answer': '12',                          # 选择的答案
    'confidence': 0.80,                      # 置信度（80%）
    'paths_count': 5,                        # 生成路径数
    'vote_distribution': {                   # 投票分布
        '12': 4,
        '11.5': 1
    }
}
```

---

### 3. Program-Aided（程序辅助推理）

#### 功能概述

**Program-Aided Reasoning (PAL)** 将自然语言问题转换为**可执行 Python 代码**，通过符号计算获得精确答案。

**核心思想**：
```
传统 LLM 推理:
"15% of 80 is..." → 模型猜测 → 可能出错

Program-Aided:
"15% of 80" → 生成代码 "0.15 * 80" → 执行 → 12.0（精确）
```

**优势**：
- ✅ 数学计算 100% 准确
- ✅ 支持复杂逻辑推理
- ✅ 可审计（代码可读）

#### 使用方法

**1. 基础使用**

```python
from apt_model.console.plugins.reasoning.program_aided_plugin import ProgramAidedReasoningPlugin

# 创建插件
pal_plugin = ProgramAidedReasoningPlugin(config={
    'timeout': 5.0,              # 代码执行超时（秒）
    'max_code_length': 1000,     # 最大代码长度
    'allowed_modules': [         # 允许的模块
        'math', 'statistics', 'datetime'
    ],
    'forbidden_keywords': [      # 禁止的关键词
        'import os', 'eval', 'exec', 'open('
    ]
})

# 注册到插件总线
bus.register(pal_plugin)
```

**2. 完整推理示例**

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载代码生成模型（如 CodeGen）
model = AutoModelForCausalLM.from_pretrained('Salesforce/codegen-350M-mono')
tokenizer = AutoTokenizer.from_pretrained('Salesforce/codegen-350M-mono')

# 创建插件总线
bus = PluginBus()
pal_plugin = ProgramAidedReasoningPlugin(config={'timeout': 10.0})
bus.register(pal_plugin)

# 推理
question = "A store has 120 apples. They sell 35% of them in the morning and 20% in the afternoon. How many apples are left?"

# 触发推理事件
bus.dispatch_event('on_inference_start', context={
    'data': {
        'use_program_aided': True
    }
})

# 触发解码事件
bus.dispatch_event('on_decode', context={
    'step': 0,
    'data': {
        'model': model,
        'tokenizer': tokenizer,
        'question': question
    }
})

# 获取结果
result = bus.get_data('program_aided_result')

if result['success']:
    print(f"Generated Code:\n{result['code']}")
    print(f"Execution Result: {result['result']}")
else:
    print(f"Error: {result['error']}")

# 示例输出:
# Generated Code:
# # Calculate remaining apples
# total_apples = 120
# morning_sold = total_apples * 0.35
# afternoon_sold = total_apples * 0.20
# remaining = total_apples - morning_sold - afternoon_sold
# print(remaining)
#
# Execution Result: 54.0
```

**3. 自定义代码生成提示**

```python
class MathPAL(ProgramAidedReasoningPlugin):
    """数学题专用 PAL：优化代码生成提示"""

    def __init__(self, config):
        super().__init__(config)

        # 自定义代码生成提示模板
        self.code_prompt_template = """# Solve this math problem using Python:
# Question: {question}
#
# Write clean, executable Python code to solve it.
# Use comments to explain your logic.
# Print the final answer.

import math

"""

    def _generate_code(self, model, tokenizer, question: str) -> str:
        """生成 Python 代码"""
        # 创建提示
        prompt = self.code_prompt_template.format(question=question)

        # 生成代码
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        output = model.generate(
            input_ids,
            max_length=len(input_ids[0]) + 200,
            temperature=0.2,  # 低温度（更确定）
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # 提取代码部分（去掉提示）
        code = generated_text[len(prompt):].strip()

        # 后处理：确保有 print 语句
        if 'print(' not in code:
            # 自动添加 print（假设最后一个变量是答案）
            lines = code.split('\n')
            if lines and '=' in lines[-1]:
                var_name = lines[-1].split('=')[0].strip()
                code += f"\nprint({var_name})"

        return code

# 使用
math_pal = MathPAL(config={'timeout': 10.0})
bus.register(math_pal)
```

**4. 安全执行沙箱**

```python
import ast
import sys
from io import StringIO

def safe_execute_code(code: str, timeout: float = 5.0) -> tuple:
    """
    安全执行 Python 代码

    Returns:
        (output, error)
    """
    # 1. 静态分析
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return None, f"Syntax error: {e}"

    # 2. 检查危险操作
    dangerous_nodes = []
    for node in ast.walk(tree):
        # 检查 import 语句
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names if isinstance(node, ast.Import) else [node]:
                module = alias.name if isinstance(alias, ast.alias) else node.module
                if module not in ['math', 'statistics', 'datetime']:
                    dangerous_nodes.append(f"Forbidden import: {module}")

        # 检查函数调用
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ['eval', 'exec', 'compile', '__import__']:
                    dangerous_nodes.append(f"Forbidden function: {node.func.id}")

    if dangerous_nodes:
        return None, "; ".join(dangerous_nodes)

    # 3. 执行代码（有限环境）
    stdout_capture = StringIO()
    safe_globals = {
        '__builtins__': {
            'print': print,
            'range': range,
            'len': len,
            'sum': sum,
            'max': max,
            'min': min,
            'abs': abs,
            'round': round,
        },
        'math': __import__('math'),
    }

    try:
        with redirect_stdout(stdout_capture):
            exec(code, safe_globals)

        output = stdout_capture.getvalue().strip()
        return output, None

    except Exception as e:
        return None, f"{type(e).__name__}: {str(e)}"
```

#### 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `timeout` | float | 5.0 | 代码执行超时（秒） |
| `max_code_length` | int | 1000 | 最大代码长度 |
| `allowed_modules` | list | [...] | 允许导入的模块 |
| `forbidden_keywords` | list | [...] | 禁止的关键词 |

#### 输出结果

```python
# 成功执行
result = {
    'success': True,
    'result': '54.0',            # 执行输出
    'code': '# Generated code...'  # 生成的代码
}

# 执行失败
result = {
    'success': False,
    'error': 'Validation failed: Forbidden keyword found: import os',
    'code': '# Generated code...'
}
```

---

## 🛠️ 插件开发

### 创建自定义插件

**1. 插件模板**

```python
from apt_model.console.plugin_standards import (
    PluginBase,
    PluginManifest,
    PluginPriority,
    PluginEvent,
    PluginCapability
)
import logging

logger = logging.getLogger(__name__)


class MyCustomPlugin(PluginBase):
    """
    自定义插件示例

    功能描述：[你的插件功能]
    """

    def __init__(self):
        """初始化插件"""
        super().__init__()
        self.config = {}
        self.metrics = {
            'counter': 0,
            'avg_value': 0.0,
        }

    def get_manifest(self) -> PluginManifest:
        """
        获取插件清单

        Returns:
            插件清单
        """
        return PluginManifest(
            name="my_custom_plugin",
            version="1.0.0",
            description="My custom plugin for doing X",
            author="Your Name",

            # 优先级（根据插件类型选择）
            priority=PluginPriority.EXPERIMENTAL,  # 650-799

            # 是否阻塞主线程
            blocking=False,

            # 监听的事件
            events=[
                PluginEvent.ON_BATCH_START,
                PluginEvent.ON_STEP_END,
            ],

            # 依赖项
            requires=[
                "core:trainer",
            ],

            # 冲突项
            conflicts=[],

            # 能力
            capabilities=[
                PluginCapability.READ_METRICS,
                PluginCapability.WRITE_METRICS,
            ],

            # 资源预算
            resources={
                "cpu_ms": 10.0,
                "gpu_ms": 5.0,
                "io_mb": 0.5
            },

            # 速率限制
            rate_limit={
                "steps": 10  # 每 10 步最多执行一次
            },

            # 沙箱模式
            sandbox=True,

            # 失败容忍度
            fail_limit=5,

            # EQI 参数
            s_default=0.5,  # 默认净效用
            eta=1.0         # 证据调制参数
        )

    def initialize(self, config: dict = None):
        """
        初始化插件

        Args:
            config: 配置字典
        """
        if config:
            self.config = config
            logger.info(f"[MyCustomPlugin] Initialized with config: {config}")

    def on_batch_start(self, context: dict):
        """
        Batch 开始事件处理器

        Args:
            context: 事件上下文
        """
        step = context.get('step', 0)
        data = context.get('data', {})

        # 你的逻辑
        self.metrics['counter'] += 1

        logger.debug(f"[MyCustomPlugin] on_batch_start at step {step}")

    def on_step_end(self, context: dict):
        """
        Step 结束事件处理器

        Args:
            context: 事件上下文
        """
        step = context.get('step', 0)
        data = context.get('data', {})

        # 读取指标
        metrics = data.get('metrics', {})

        # 你的逻辑
        value = metrics.get('loss', 0.0)
        self.metrics['avg_value'] = (
            (self.metrics['avg_value'] * (self.metrics['counter'] - 1) + value)
            / self.metrics['counter']
        )

        # 写入指标
        if 'metrics' not in data:
            data['metrics'] = {}
        data['metrics']['my_custom_metric'] = self.metrics['avg_value']

        logger.debug(f"[MyCustomPlugin] on_step_end at step {step}")

    def cleanup(self):
        """清理资源"""
        logger.info(f"[MyCustomPlugin] Cleanup: {self.metrics}")
```

**2. 注册和使用**

```python
# 创建插件
my_plugin = MyCustomPlugin()

# 初始化
my_plugin.initialize({
    'param1': 'value1',
    'param2': 42
})

# 注册到插件总线
bus = PluginBus()
bus.register(my_plugin)

# 训练循环
for step, batch in enumerate(dataloader):
    # Batch 开始
    bus.dispatch_event('on_batch_start', context={
        'step': step,
        'data': {}
    })

    # ... 训练代码 ...
    loss = train_step(batch)

    # Step 结束
    bus.dispatch_event('on_step_end', context={
        'step': step,
        'data': {
            'metrics': {'loss': loss}
        }
    })
```

### 插件开发最佳实践

**1. 选择合适的优先级**

```python
# 根据插件功能选择优先级
if plugin_type == 'kill_switch':
    priority = PluginPriority.KILLSWITCH  # 0-49
elif plugin_type == 'inference':
    priority = PluginPriority.INFERENCE_CTRL  # 50-149
elif plugin_type == 'optimization':
    priority = PluginPriority.THROUGHPUT  # 150-249
elif plugin_type == 'reasoning':
    priority = PluginPriority.BEAM_SEARCH  # 250-349
elif plugin_type == 'training':
    priority = PluginPriority.GRPO  # 350-449
```

**2. 异常处理**

```python
def on_step_end(self, context: dict):
    """Step 结束事件处理器（带异常处理）"""
    try:
        step = context.get('step', 0)
        data = context.get('data', {})

        # 你的逻辑
        result = self.process(data)

        # 写入结果
        data['my_result'] = result

    except Exception as e:
        logger.error(f"[MyPlugin] Error in on_step_end: {e}")
        # 记录失败（会触发 fail_limit 检查）
        self.record_failure()
```

**3. 资源管理**

```python
class ResourceAwarePlugin(PluginBase):
    """资源感知插件"""

    def __init__(self):
        super().__init__()
        self.gpu_available = torch.cuda.is_available()
        self.cache = {}

    def get_manifest(self) -> PluginManifest:
        # 根据可用资源调整预算
        gpu_ms = 50.0 if self.gpu_available else 0.0
        cpu_ms = 20.0 if not self.gpu_available else 5.0

        return PluginManifest(
            # ...
            resources={
                "cpu_ms": cpu_ms,
                "gpu_ms": gpu_ms,
                "io_mb": 2.0
            }
        )

    def cleanup(self):
        """清理资源"""
        # 清空缓存
        self.cache.clear()

        # 释放 GPU 内存
        if self.gpu_available:
            torch.cuda.empty_cache()

        logger.info("[ResourceAwarePlugin] Resources cleaned up")
```

---

## 🚀 高级应用

### 1. 多插件组合

```python
# 组合多个推理插件
bus = PluginBus()

# 1. Beam Search（探索多条路径）
beam_search = BeamSearchReasoningPlugin(config={
    'beam_width': 5,
    'max_steps': 30
})
bus.register(beam_search)

# 2. Self-Consistency（投票机制）
sc_plugin = SelfConsistencyPlugin(config={
    'num_paths': 10,
    'temperature': 0.8
})
# 注意：与 Beam Search 冲突，二选一
# bus.register(sc_plugin)

# 3. Program-Aided（精确计算）
pal_plugin = ProgramAidedReasoningPlugin(config={
    'timeout': 10.0
})
bus.register(pal_plugin)

# 4. EQI Reporter（监控）
eqi_reporter = EQIReporterPlugin()
eqi_reporter.initialize({'report_interval': 50})
bus.register(eqi_reporter)

# 推理：先尝试 PAL，失败则用 Beam Search
question = "Complex math problem..."

# 尝试 PAL
bus.dispatch_event('on_inference_start', context={
    'data': {'use_program_aided': True}
})
bus.dispatch_event('on_decode', context={
    'step': 0,
    'data': {'model': model, 'tokenizer': tokenizer, 'question': question}
})

pal_result = bus.get_data('program_aided_result')

if not pal_result or not pal_result.get('success'):
    # PAL 失败，使用 Beam Search
    logger.info("PAL failed, falling back to Beam Search")

    bus.dispatch_event('on_inference_start', context={
        'data': {'use_beam_search': True}
    })
    bus.dispatch_event('on_decode', context={
        'step': 0,
        'data': {'model': model, 'tokenizer': tokenizer, 'input_ids': input_ids}
    })

    beam_result = bus.get_data('beam_search_result')
    answer = tokenizer.decode(beam_result['path'])
else:
    answer = pal_result['result']

print(f"Final answer: {answer}")
```

### 2. 动态插件加载

```python
import importlib

class PluginLoader:
    """动态插件加载器"""

    def __init__(self, plugin_dir: str = "plugins"):
        self.plugin_dir = plugin_dir
        self.loaded_plugins = {}

    def load_plugin(self, plugin_name: str):
        """动态加载插件"""
        try:
            # 导入插件模块
            module_path = f"{self.plugin_dir}.{plugin_name}"
            module = importlib.import_module(module_path)

            # 查找插件类（约定：类名 = 插件名 + Plugin）
            plugin_class_name = ''.join(word.capitalize() for word in plugin_name.split('_')) + 'Plugin'
            plugin_class = getattr(module, plugin_class_name)

            # 实例化插件
            plugin = plugin_class()

            self.loaded_plugins[plugin_name] = plugin
            logger.info(f"Loaded plugin: {plugin_name}")

            return plugin

        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
            return None

    def load_all_plugins(self, bus: PluginBus):
        """加载所有插件"""
        import os
        import glob

        # 查找所有插件文件
        plugin_files = glob.glob(os.path.join(self.plugin_dir, "*_plugin.py"))

        for plugin_file in plugin_files:
            plugin_name = os.path.basename(plugin_file)[:-3]  # 去掉 .py
            plugin = self.load_plugin(plugin_name)

            if plugin:
                bus.register(plugin)

# 使用
loader = PluginLoader(plugin_dir="apt_model/console/plugins")
loader.load_all_plugins(bus)
```

### 3. 插件配置文件

```yaml
# plugins_config.yaml
plugins:
  - name: grpo
    enabled: true
    config:
      group_size: 4
      learning_rate: 0.00001
      advantage_type: relative

  - name: route_optimizer
    enabled: true
    config:
      num_experts: 8
      load_threshold: 1.5

  - name: beam_search
    enabled: false  # 禁用
    config:
      beam_width: 5
      length_penalty: 0.6

  - name: self_consistency
    enabled: true
    config:
      num_paths: 10
      temperature: 0.8
```

```python
import yaml

def load_plugins_from_config(config_file: str, bus: PluginBus):
    """从配置文件加载插件"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    plugin_registry = {
        'grpo': GRPOPlugin,
        'route_optimizer': RouteOptimizerPlugin,
        'beam_search': BeamSearchReasoningPlugin,
        'self_consistency': SelfConsistencyPlugin,
        'program_aided': ProgramAidedReasoningPlugin,
        'eqi_reporter': EQIReporterPlugin,
    }

    for plugin_spec in config['plugins']:
        if not plugin_spec.get('enabled', True):
            continue

        name = plugin_spec['name']
        plugin_class = plugin_registry.get(name)

        if not plugin_class:
            logger.warning(f"Unknown plugin: {name}")
            continue

        # 创建插件
        plugin = plugin_class()

        # 初始化（如果有配置）
        if 'config' in plugin_spec:
            plugin.initialize(plugin_spec['config'])

        # 注册
        bus.register(plugin)
        logger.info(f"Registered plugin from config: {name}")

# 使用
load_plugins_from_config('plugins_config.yaml', bus)
```

---

## 🐛 故障排查

### 常见问题

**1. 插件冲突**

```
错误: ConflictError: Plugin 'self_consistency' conflicts with 'beam_search'
```

**解决方案**：
```python
# 不要同时注册冲突插件
bus.register(beam_search)
# bus.register(sc_plugin)  # ❌ 会冲突

# 或者根据场景选择
if task_type == 'math':
    bus.register(pal_plugin)  # 数学题用 PAL
elif task_type == 'reasoning':
    bus.register(beam_search)  # 推理题用 Beam Search
```

**2. 资源超限**

```
错误: ResourceExceededError: CPU budget exceeded (500ms > 450ms)
```

**解决方案**：
```python
# 方法1: 调整插件配置（减少计算量）
grpo_plugin.initialize({
    'group_size': 2  # 减少到 2（默认 4）
})

# 方法2: 增加速率限制
manifest = grpo_plugin.get_manifest()
manifest.rate_limit['steps'] = 5  # 每 5 步执行一次（默认 1）

# 方法3: 使用异步模式
manifest.blocking = False  # 改为非阻塞
```

**3. 插件失败**

```
错误: Plugin 'my_plugin' reached fail_limit (5 failures)
```

**解决方案**：
```python
# 检查插件日志
logger.info(f"Plugin failures: {my_plugin.get_context('failures')}")

# 增加失败容忍度
manifest = my_plugin.get_manifest()
manifest.fail_limit = 10  # 增加到 10（默认 5）

# 添加异常处理
def on_step_end(self, context: dict):
    try:
        # 你的逻辑
        pass
    except Exception as e:
        logger.error(f"Error: {e}")
        # 降级处理
        self.use_fallback_logic()
```

### 调试技巧

**1. 启用详细日志**

```python
import logging

# 设置日志级别
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 插件日志
logger = logging.getLogger('apt_model.console.plugins')
logger.setLevel(logging.DEBUG)
```

**2. 插件性能分析**

```python
import time

class ProfilingPlugin(PluginBase):
    """性能分析插件"""

    def __init__(self):
        super().__init__()
        self.timings = []

    def on_step_end(self, context: dict):
        start_time = time.perf_counter()

        # 你的逻辑
        self.process(context)

        elapsed = time.perf_counter() - start_time
        self.timings.append(elapsed)

        if len(self.timings) % 100 == 0:
            avg_time = sum(self.timings) / len(self.timings)
            logger.info(f"[Profiling] Average time: {avg_time*1000:.2f}ms")
```

**3. 插件状态检查**

```python
# 获取所有已注册插件
registered_plugins = bus.list_plugins()
print(f"Registered plugins: {registered_plugins}")

# 获取插件状态
for plugin_name in registered_plugins:
    manifest = bus.get_plugin_manifest(plugin_name)
    print(f"\n{plugin_name}:")
    print(f"  Priority: {manifest.priority}")
    print(f"  Events: {manifest.events}")
    print(f"  Resources: {manifest.resources}")
    print(f"  Conflicts: {manifest.conflicts}")
```

---

## 📚 参考资源

### 学术论文

- [PAL: Program-aided Language Models](https://arxiv.org/abs/2211.10435) - Gao et al., 2022
- [Self-Consistency Improves Chain of Thought](https://arxiv.org/abs/2203.11171) - Wang et al., 2022
- [Beam Search Strategies for Neural Machine Translation](https://arxiv.org/abs/1702.01806) - Freitag & Al-Onaizan, 2017
- [Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300) - GRPO 论文

### APT 相关文档

- [插件系统架构](PLUGIN_SYSTEM.md) - 插件系统设计文档
- [DeepSeek 训练指南](DEEPSEEK_TRAINING_GUIDE.md) - MoE 训练教程
- [图脑训练教程](GRAPH_BRAIN_TRAINING_GUIDE.md) - 图推理架构
- [数据预处理指南](DATA_PREPROCESSING_GUIDE.md) - 数据清洗流程

### 代码示例

```bash
# 插件示例代码
apt_model/console/plugins/
├── grpo_plugin.py              # GRPO 训练插件
├── route_optimizer_plugin.py   # 路由优化插件
├── eqi_reporter_plugin.py      # EQI 上报插件
└── reasoning/
    ├── beam_search_plugin.py        # Beam Search
    ├── self_consistency_plugin.py   # Self-Consistency
    └── program_aided_plugin.py      # Program-Aided
```

---

## 📝 更新日志

- **v1.0.0** (2025-12) - 初始版本
  - ✅ 核心插件文档（GRPO、Route Optimizer、EQI Reporter）
  - ✅ 推理插件文档（Beam Search、Self-Consistency、PAL）
  - ✅ 插件开发指南
  - ✅ 完整代码示例
  - ✅ 故障排查指南

---

<div align="center">

**让插件系统为你的 AI 模型赋能！ 🚀**

26+ 生产级插件，开箱即用，灵活可扩展

如有问题，请提交 [Issue](https://github.com/chen0430tw/APT-Transformer/issues)

</div>
