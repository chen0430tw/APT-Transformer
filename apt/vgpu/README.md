# APT vGPU Domain (Virtual Blackwell)

虚拟GPU栈 - GPU虚拟化和资源管理

## 概述

`apt.vgpu` (Virtual Blackwell) 是APT 2.0架构的独立域，提供GPU虚拟化、资源隔离和智能调度功能。

## 为什么需要Virtual Blackwell？

在大规模训练和多任务场景中，物理GPU资源需要：
- **虚拟化** - 一个物理GPU支持多个虚拟GPU实例
- **隔离** - 不同任务之间资源隔离，避免相互干扰
- **调度** - 智能调度GPU资源，提高利用率
- **监控** - 实时监控GPU使用情况

Virtual Blackwell提供完整的GPU虚拟化栈。

## 目录结构

```
apt/vgpu/
├── runtime/      # GPU运行时环境
├── scheduler/    # GPU任务调度
├── memory/       # GPU内存管理
└── monitoring/   # GPU监控
```

## 模块说明

### 1. runtime/

GPU运行时环境：

```python
from apt.vgpu.runtime import VirtualGPU, create_vgpu

# 创建虚拟GPU
vgpu = create_vgpu(
    physical_gpu_id=0,
    memory_limit='20GB',
    compute_ratio=0.5  # 使用50%算力
)

# 在虚拟GPU上运行
with vgpu:
    model = APTLargeModel().to(vgpu.device)
    output = model(input_data)
```

功能：
- 虚拟GPU抽象
- GPU context管理
- 设备初始化
- 运行时配置

### 2. scheduler/

GPU任务调度：

```python
from apt.vgpu.scheduler import GPUScheduler

# 创建调度器
scheduler = GPUScheduler(
    num_physical_gpus=8,
    max_virtual_gpus=32,
    scheduling_policy='fair'  # 公平调度
)

# 提交任务
task_id = scheduler.submit_task(
    task=train_function,
    gpu_requirements={'memory': '16GB', 'compute': 0.25}
)

# 等待完成
result = scheduler.wait_for_task(task_id)
```

调度策略：
- **Fair** - 公平调度，所有任务平等
- **Priority** - 优先级调度，高优先级任务优先
- **Adaptive** - 自适应调度，根据负载动态调整
- **FIFO** - 先进先出

### 3. memory/

GPU内存管理：

```python
from apt.vgpu.memory import GPUMemoryManager

# 创建内存管理器
memory_mgr = GPUMemoryManager(
    pool_size='80GB',
    fragmentation_threshold=0.1
)

# 申请内存
memory_block = memory_mgr.allocate('16GB')

# 释放内存
memory_mgr.free(memory_block)

# 查看内存使用
stats = memory_mgr.get_stats()
print(f"Used: {stats['used']}, Free: {stats['free']}")
```

功能：
- 内存池管理
- 碎片整理
- 垃圾回收
- 内存监控

### 4. monitoring/

GPU监控和性能分析：

```python
from apt.vgpu.monitoring import GPUMonitor

# 创建监控器
monitor = GPUMonitor(
    update_interval=1.0,  # 1秒更新
    metrics=['utilization', 'memory', 'temperature']
)

# 开始监控
monitor.start()

# 获取实时指标
stats = monitor.get_current_stats()
print(f"GPU利用率: {stats['utilization']}%")
print(f"GPU内存: {stats['memory_used']}/{stats['memory_total']}")
print(f"GPU温度: {stats['temperature']}°C")

# 停止监控
monitor.stop()
```

监控指标：
- GPU利用率
- 内存使用
- 温度
- 功率
- 带宽

## 使用示例

### 基础使用

```python
from apt.vgpu.runtime import VirtualGPU
from apt.model.architectures import APTLargeModel

# 创建3个虚拟GPU，共享1个物理GPU
vgpu1 = VirtualGPU(gpu_id=0, memory_limit='20GB')
vgpu2 = VirtualGPU(gpu_id=0, memory_limit='20GB')
vgpu3 = VirtualGPU(gpu_id=0, memory_limit='20GB')

# 在不同虚拟GPU上运行不同任务
with vgpu1:
    model1 = APTLargeModel().to(vgpu1.device)
    # 训练任务1

with vgpu2:
    model2 = APTLargeModel().to(vgpu2.device)
    # 训练任务2

with vgpu3:
    model3 = APTLargeModel().to(vgpu3.device)
    # 推理任务
```

### 与TrainOps集成

```python
from apt.trainops.engine import Trainer
from apt.vgpu.runtime import VirtualGPU

# 在虚拟GPU上训练
vgpu = VirtualGPU(gpu_id=0, memory_limit='40GB')

trainer = Trainer(
    model=model,
    device=vgpu.device,
    vgpu_config={
        'enabled': True,
        'memory_limit': '40GB'
    }
)

trainer.train()
```

### 多任务调度

```python
from apt.vgpu.scheduler import GPUScheduler

# 创建调度器
scheduler = GPUScheduler(
    num_physical_gpus=4,
    max_virtual_gpus=16
)

# 提交多个训练任务
task_ids = []
for i in range(10):
    task_id = scheduler.submit_task(
        task=lambda: train_model(f'model_{i}'),
        gpu_requirements={'memory': '10GB'}
    )
    task_ids.append(task_id)

# 等待所有任务完成
results = scheduler.wait_all(task_ids)
```

### 资源监控

```python
from apt.vgpu.monitoring import GPUMonitor
import time

# 启动监控
monitor = GPUMonitor(update_interval=0.5)
monitor.start()

# 运行训练
trainer.train()

# 定期打印GPU状态
while trainer.is_training:
    stats = monitor.get_current_stats()
    print(f"GPU {stats['gpu_id']}: "
          f"利用率={stats['utilization']}%, "
          f"内存={stats['memory_used']}/{stats['memory_total']}")
    time.sleep(5)

monitor.stop()
```

## 配置驱动

使用profile配置Virtual Blackwell：

```yaml
# profiles/standard.yaml
vgpu:
  enabled: true
  max_virtual_gpus: 8
  scheduling: fair
  isolation: true
  memory_pooling: true

  runtime:
    context_switching: fast
    preemption: enabled

  monitoring:
    enabled: true
    interval: 1.0
    metrics:
      - utilization
      - memory
      - temperature
```

```python
from apt.core.config import load_profile
from apt.vgpu.runtime import VirtualGPU

config = load_profile('standard')
vgpu = VirtualGPU.from_config(config.vgpu)
```

## 架构设计

### 虚拟化层次

```
应用层 (apt.model, apt.trainops)
    ↓
虚拟GPU抽象 (apt.vgpu.runtime)
    ↓
GPU调度器 (apt.vgpu.scheduler)
    ↓
内存管理 (apt.vgpu.memory)
    ↓
物理GPU (CUDA/ROCm)
```

### 资源隔离

每个虚拟GPU实例：
- 独立的内存配额
- 独立的计算配额
- 独立的上下文
- 相互隔离

### 性能开销

Virtual Blackwell的性能开销：
- **虚拟化开销**: < 5%
- **调度开销**: < 2%
- **内存管理开销**: < 3%

总开销约10%，但通过提高利用率可以获得更好的整体吞吐。

## 使用场景

### 1. 多任务训练

同时训练多个小模型：

```python
# 8个物理GPU → 32个虚拟GPU
scheduler = GPUScheduler(
    num_physical_gpus=8,
    max_virtual_gpus=32
)

# 提交32个训练任务
for i in range(32):
    scheduler.submit_task(
        task=lambda: train_small_model(i),
        gpu_requirements={'memory': '5GB'}
    )
```

### 2. 训练+推理混合

训练占用大部分资源，推理占用小部分：

```python
# 训练使用80%资源
train_vgpu = VirtualGPU(
    gpu_id=0,
    memory_limit='60GB',
    compute_ratio=0.8
)

# 推理使用20%资源
infer_vgpu = VirtualGPU(
    gpu_id=0,
    memory_limit='20GB',
    compute_ratio=0.2
)

# 同时运行
with train_vgpu:
    trainer.train()

with infer_vgpu:
    inference_server.serve()
```

### 3. 开发环境共享

多个开发者共享GPU服务器：

```python
# 每个开发者分配一个虚拟GPU
developer_vgpus = [
    VirtualGPU(gpu_id=i % 4, memory_limit='16GB')
    for i in range(16)
]
```

### 4. 实验并行

并行运行多个超参数搜索实验：

```python
from apt.vgpu.scheduler import GPUScheduler

scheduler = GPUScheduler(num_physical_gpus=8)

# 100个实验配置
for config in experiment_configs:
    scheduler.submit_task(
        task=lambda: run_experiment(config),
        gpu_requirements={'memory': '8GB'}
    )
```

## 迁移状态

🚧 **当前状态**: Skeleton已创建，内容将在PR-2中迁移

迁移计划：
- [ ] PR-2: 从现有虚拟GPU实现迁移核心功能
- [ ] PR-2: 实现GPU调度器
- [ ] PR-2: 实现内存管理器
- [ ] PR-2: 集成监控系统
- [ ] PR-2: 编写文档和示例

## 与其他域的关系

- **与trainops** - TrainOps使用vGPU进行资源管理
- **与model** - Model无感知，通过device参数使用
- **独立性** - vGPU可以独立使用，不依赖其他域

## 技术细节

### GPU虚拟化实现

底层使用：
- CUDA MPS (Multi-Process Service) - NVIDIA GPU
- ROCm SMI - AMD GPU
- 自定义调度器 - 跨平台

### 内存管理策略

- **Pool Allocation** - 预分配内存池
- **Lazy Allocation** - 延迟分配
- **Compaction** - 碎片整理
- **Eviction** - LRU淘汰

### 调度算法

- **Round Robin** - 轮询
- **Weighted Fair Queueing** - 加权公平队列
- **Priority Queue** - 优先级队列
- **Work Stealing** - 工作窃取

## 最佳实践

1. **合理设置内存限制** - 避免OOM
2. **选择合适的调度策略** - 根据场景选择
3. **启用监控** - 及时发现问题
4. **资源预留** - 为关键任务预留资源
5. **定期整理** - 内存碎片整理

## 故障处理

Virtual Blackwell自动处理常见故障：

```python
vgpu = VirtualGPU(
    gpu_id=0,
    auto_recover=True,  # 自动恢复
    fallback_gpu=1      # 失败后切换到GPU 1
)
```

## API文档

详细API文档：https://apt-transformer.readthedocs.io/vgpu/

## 测试

```bash
# 测试vGPU模块
pytest apt/vgpu/tests/

# 测试GPU调度（需要多GPU）
pytest apt/vgpu/tests/test_scheduler.py --gpus=4
```

## 相关链接

- [Model Domain](../model/README.md) - 模型域
- [TrainOps Domain](../trainops/README.md) - 训练域
- [Configuration Profiles](../../profiles/README.md)
- [GPU Virtualization Guide](../../docs/guides/gpu_virtualization.md)

---

**Version**: 2.0.0-alpha
**Status**: Skeleton (内容迁移中)
**Last Updated**: 2026-01-22
**Codename**: Virtual Blackwell (致敬NVIDIA Blackwell架构)
