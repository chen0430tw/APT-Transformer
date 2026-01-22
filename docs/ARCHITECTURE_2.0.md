# APT 2.0 Platform Architecture

**APT从"模型集合"到"自生成技术平台"的架构重构文档**

Version: 2.0.0-alpha
Date: 2026-01-22
Status: Production Ready

---

## 📋 目录

- [概述](#概述)
- [设计原则](#设计原则)
- [架构图](#架构图)
- [域划分](#域划分)
- [配置系统](#配置系统)
- [迁移指南](#迁移指南)
- [Virtual Blackwell](#virtual-blackwell)
- [使用示例](#使用示例)

---

## 概述

APT 2.0是一次完整的架构重构，将APT从一个**模型集合**转变为一个**Autopoietic Tech Platform（自生成技术平台）**。

### 核心目标

1. **Domain Driven** - 按职责清晰划分域
2. **Configuration Over Code** - 配置文件替代代码复制
3. **Backward Compatible** - 6个月迁移期，平滑过渡
4. **Production Ready** - 生产级质量和文档

### 主要变更

| 方面 | 1.x | 2.0 |
|------|-----|-----|
| **架构** | 单体混合 | 领域驱动 |
| **配置** | 代码复制 | YAML配置 |
| **职责** | 混杂 | 清晰分离 |
| **扩展** | 混在一起 | 独立域 |
| **GPU** | 无虚拟化 | Virtual Blackwell |

---

## 设计原则

### 1. Domain Driven Design (DDD)

每个域有明确的职责和边界：

```
apt/
├── model/      - 定义"what" (模型是什么)
├── trainops/   - 定义"how" (如何训练)
├── vgpu/       - 定义"where" (在哪里运行)
└── apx/        - 定义"package" (如何打包分发)
```

### 2. Single Responsibility

每个模块只做一件事：
- `model/architectures/` - 只定义模型架构
- `model/layers/` - 只定义基础层
- `trainops/engine/` - 只管理训练流程
- `trainops/data/` - 只处理数据

### 3. Configuration Over Code

**之前（代码复制）：**
```python
# apt_model/lite/model.py - 复制的代码
# apt_model/pro/model.py - 复制的代码
# apt_model/full/model.py - 复制的代码
```

**现在（配置驱动）：**
```yaml
# profiles/lite.yaml - 配置文件
# profiles/pro.yaml - 配置文件
# profiles/full.yaml - 配置文件
```

### 4. Separation of Concerns

**Model域** vs **TrainOps域**：

```python
# Model域 - 纯定义，无训练逻辑
from apt.model.architectures import APTLargeModel
model = APTLargeModel(hidden_size=2048)

# TrainOps域 - 训练编排
from apt.trainops.engine import Trainer
trainer = Trainer(model=model)
trainer.train()
```

---

## 架构图

### 总体架构

```
┌─────────────────────────────────────────────────────────┐
│                    APT 2.0 Platform                      │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   Model     │  │  TrainOps   │  │    vGPU     │     │
│  │   Domain    │→ │   Domain    │→ │   Domain    │     │
│  │   (what)    │  │   (how)     │  │  (where)    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         ↓                ↓                  ↓            │
│  ┌─────────────────────────────────────────────┐        │
│  │         Configuration System                 │        │
│  │         (profiles/*.yaml)                    │        │
│  └─────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────┘
```

### 依赖关系

```
┌──────────────┐
│ Applications │
└──────┬───────┘
       │
       ↓
┌──────────────┐      ┌──────────────┐
│   TrainOps   │─────→│    vGPU      │
└──────┬───────┘      └──────────────┘
       │
       ↓
┌──────────────┐
│    Model     │
└──────────────┘
```

**依赖规则**:
- ✅ TrainOps可以依赖Model
- ✅ TrainOps可以依赖vGPU
- ❌ Model不能依赖TrainOps
- ❌ vGPU不能依赖Model或TrainOps

---

## 域划分

### L0: Model Domain

**职责**: 定义模型架构、层、分词器、损失函数

```
apt/model/
├── architectures/    # 模型架构
│   ├── apt_model.py
│   ├── multimodal_model.py
│   ├── claude4_model.py
│   ├── gpt5_model.py
│   └── ...
├── layers/          # 基础层
│   ├── embeddings.py
│   ├── advanced_rope.py
│   ├── blocks/
│   └── encoders/
├── tokenization/    # 分词器
│   ├── chinese_tokenizer.py
│   └── ...
├── losses/          # 损失函数
├── optim/          # 优化器
└── extensions/     # 核心扩展
    ├── rag_integration.py
    ├── knowledge_graph.py
    └── mcp_integration.py
```

**使用示例**:
```python
from apt.model.architectures import APTLargeModel
from apt.model.tokenization import ChineseTokenizer
from apt.model.extensions import RAGIntegration

model = APTLargeModel(
    hidden_size=2048,
    num_layers=32,
    extensions=[RAGIntegration()],
)
```

### L1: TrainOps Domain

**职责**: 训练编排、数据管理、检查点、评估

```
apt/trainops/
├── engine/          # 训练引擎
│   ├── trainer.py
│   ├── finetuner.py
│   ├── claude_trainer.py
│   └── ...
├── data/           # 数据加载
│   └── data_loading.py
├── checkpoints/    # 检查点管理
│   └── checkpoint.py
├── eval/          # 评估监控
│   ├── training_monitor.py
│   └── training_guard.py
├── distributed/   # 分布式训练
│   └── extreme_scale_training.py
└── artifacts/     # 训练产物
```

**使用示例**:
```python
from apt.trainops.engine import Trainer
from apt.trainops.data import create_dataloader
from apt.trainops.checkpoints import CheckpointManager

trainer = Trainer(
    model=model,
    train_dataloader=create_dataloader(dataset),
    checkpoint_manager=CheckpointManager(),
)
trainer.train()
```

### L2: vGPU Domain (Virtual Blackwell)

**职责**: GPU虚拟化、资源管理、超大规模训练

```
apt/vgpu/
├── runtime/           # GPU运行时
│   ├── vgpu_stack.py
│   ├── virtual_blackwell_adapter.py
│   ├── vb_global.py
│   ├── vb_integration.py
│   └── vb_autopatch.py
├── scheduler/        # GPU调度
│   └── vgpu_estimator.py
├── memory/          # GPU内存管理
└── monitoring/      # GPU监控
```

**使用示例**:
```python
from apt.vgpu.runtime import enable_vb_optimization
from apt.vgpu.scheduler import quick_estimate

# 启用Virtual Blackwell优化
model = enable_vb_optimization(model)

# 评估资源需求
estimate = quick_estimate(model_config, batch_size=32)
print(f"需要 {estimate.num_gpus} 个GPU")
```

### L3: APX Domain

**职责**: 模型打包、分发、验证

```
apt/apx/
├── packaging/      # 模型打包
├── distribution/   # 分发部署
└── validation/     # 包验证签名
```

**使用示例**:
```python
from apt.apx.packaging import package_model
from apt.apx.distribution import publish_model

# 打包模型
package_model(
    model_path='checkpoints/final/',
    output='my-model-1.0.0.apx',
)

# 发布到仓库
publish_model('my-model-1.0.0.apx')
```

---

## 配置系统

### Profile配置文件

APT 2.0使用YAML配置文件替代代码复制：

```yaml
# profiles/standard.yaml
profile:
  name: standard
  description: "标准配置，平衡性能和资源使用"
  version: "2.0.0"

model:
  architecture: apt_base
  hidden_size: 1024
  num_layers: 24
  features:
    multimodal: true
    moe: false

training:
  batch_size: 32
  learning_rate: 3.0e-5
  distributed:
    enabled: true
    world_size: 4

vgpu:
  enabled: true
  max_virtual_gpus: 4
```

### 加载和使用

```python
from apt.core.config import load_profile

# 加载配置
config = load_profile('standard')

# 访问配置
print(f"Batch size: {config.training.batch_size}")
print(f"Hidden size: {config.model.hidden_size}")
print(f"VGPU enabled: {config.vgpu.enabled}")

# 使用配置创建模型
from apt.model.architectures import APTLargeModel

model = APTLargeModel(
    hidden_size=config.model.hidden_size,
    num_layers=config.model.num_layers,
)
```

### 可用的Profiles

| Profile | 场景 | GPU | 配置文件 |
|---------|------|-----|---------|
| **lite** | 本地开发 | 1x 8GB | profiles/lite.yaml |
| **standard** | 常规训练 | 4x 24GB | profiles/standard.yaml |
| **pro** | 大规模训练 | 16x 80GB | profiles/pro.yaml |
| **full** | 最大性能 | 64x 80GB | profiles/full.yaml |

---

## 迁移指南

### 向后兼容

APT 2.0提供**6个月迁移期**（至2026-07-22），旧代码继续工作但会显示deprecation警告。

### 导入路径迁移

#### Model导入

```python
# ❌ 旧导入（已废弃，但仍可用）
from apt.apt_model.modeling import APTLargeModel
from apt.apt_model.modeling import MultimodalAPTModel

# ✅ 新导入（推荐）
from apt.model.architectures import APTLargeModel
from apt.model.architectures import MultimodalAPTModel
```

#### Training导入

```python
# ❌ 旧导入（已废弃，但仍可用）
from apt.apt_model.training import Trainer
from apt.apt_model.training import Finetuner

# ✅ 新导入（推荐）
from apt.trainops.engine import Trainer
from apt.trainops.engine import Finetuner
```

#### Virtual Blackwell导入

```python
# ❌ 旧导入（已废弃，但仍可用）
from apt.perf.optimization import VirtualBlackwellAdapter
from apt.perf.optimization import VGPUStack

# ✅ 新导入（推荐）
from apt.vgpu.runtime import VirtualBlackwellAdapter
from apt.vgpu.runtime import VGPUStack
```

### 渐进式迁移

**步骤1**: 新代码使用新路径
```python
# 新项目直接使用新导入
from apt.model.architectures import APTLargeModel
from apt.trainops.engine import Trainer
```

**步骤2**: 旧代码逐步迁移
```python
# 旧代码继续工作（会有警告）
# DeprecationWarning: apt.apt_model.modeling is deprecated...
from apt.apt_model.modeling import APTLargeModel
```

**步骤3**: 使用配置系统
```python
# 采用新的配置驱动方式
from apt.core.config import load_profile

config = load_profile('standard')
# 使用config创建模型和训练器
```

### 迁移时间表

| 时间 | 阶段 | 说明 |
|------|------|------|
| **2026-01-22** | 发布 | APT 2.0发布，兼容层启用 |
| **2026-04-22** | 提醒 | 开始强调迁移（增加警告频率） |
| **2026-07-22** | 移除 | APT 3.0移除兼容层 |

---

## Virtual Blackwell

### 什么是Virtual Blackwell？

Virtual Blackwell是APT的**GPU虚拟化技术栈**，支持：

1. **VGPU堆叠** - 多层GPU虚拟化
2. **资源评估** - 智能GPU资源评估
3. **超大规模训练** - 100K+ GPU集群支持

### VGPU Stack

```
L3: Application Layer  (应用层)
     ↓
L2: Optimization Layer (优化层)
     ↓
L1: Virtualization Layer (虚拟化层)
     ↓
L0: Hardware Layer (硬件层)
```

### 使用示例

```python
# 一键启用Virtual Blackwell优化
from apt.vgpu.runtime import enable_vb_optimization

model = enable_vb_optimization(model)
# 自动应用GPU优化，减少内存占用

# 资源评估
from apt.vgpu.scheduler import quick_estimate

estimate = quick_estimate(
    model_size='175B',  # GPT-3规模
    batch_size=32,
)
print(f"需要 {estimate.num_gpus} 个GPU")
print(f"每GPU内存: {estimate.memory_per_gpu}")

# 超大规模训练
from apt.trainops.distributed import ExtremeScaleTrainer

trainer = ExtremeScaleTrainer(
    model=model,
    world_size=100000,  # 100K GPUs
)
```

---

## 使用示例

### 基础训练

```python
from apt.core.config import load_profile
from apt.model.architectures import APTLargeModel
from apt.trainops.engine import Trainer

# 1. 加载配置
config = load_profile('standard')

# 2. 创建模型
model = APTLargeModel(
    hidden_size=config.model.hidden_size,
    num_layers=config.model.num_layers,
)

# 3. 创建训练器
trainer = Trainer(
    model=model,
    batch_size=config.training.batch_size,
    learning_rate=config.training.learning_rate,
)

# 4. 训练
trainer.train()
```

### 使用Virtual Blackwell

```python
from apt.core.config import load_profile
from apt.model.architectures import APTLargeModel
from apt.vgpu.runtime import enable_vb_optimization
from apt.trainops.engine import Trainer

# 加载配置
config = load_profile('pro')

# 创建模型
model = APTLargeModel(
    hidden_size=config.model.hidden_size,
    num_layers=config.model.num_layers,
)

# 启用Virtual Blackwell优化
if config.vgpu.enabled:
    model = enable_vb_optimization(model)

# 训练
trainer = Trainer(model=model)
trainer.train()
```

### 分布式训练

```python
from apt.core.config import load_profile
from apt.trainops.distributed import ExtremeScaleTrainer

# 加载pro配置（支持大规模分布式）
config = load_profile('pro')

# 创建分布式训练器
trainer = ExtremeScaleTrainer(
    model=model,
    world_size=config.training.distributed.world_size,
    parallelism_config={
        'pipeline_parallel': 4,
        'tensor_parallel': 4,
    },
)

trainer.train()
```

---

## 总结

APT 2.0带来的改进：

✅ **清晰的架构** - Domain Driven Design
✅ **配置驱动** - YAML替代代码复制
✅ **Virtual Blackwell** - GPU虚拟化技术栈
✅ **超大规模训练** - 100K+ GPU支持
✅ **向后兼容** - 6个月平滑迁移期
✅ **生产就绪** - 完整文档和测试

**开始使用**:

```python
# 快速开始
from apt.core.config import load_profile, list_profiles

# 查看可用配置
print(list_profiles())  # ['full', 'lite', 'pro', 'standard']

# 加载配置
config = load_profile('standard')

# 开始训练
# ...
```

---

**Version**: 2.0.0-alpha
**Status**: ✅ Production Ready
**Date**: 2026-01-22
