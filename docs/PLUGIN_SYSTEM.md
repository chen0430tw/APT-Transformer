# APT Plugin System Documentation

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
