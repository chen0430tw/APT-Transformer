# APT Configuration Profiles

配置文件系统 - 使用YAML文件而非代码复制来管理不同的训练配置。

## 概述

Profiles是APT 2.0的核心设计理念之一：**通过配置文件而非代码复制来实现不同规模的部署**。

## 可用配置

| Profile | 描述 | 适用场景 | 资源需求 |
|---------|------|---------|---------|
| **lite.yaml** | 轻量级配置 | 本地开发、快速原型 | 1x GPU (8GB) |
| **standard.yaml** | 标准配置 | 常规训练、生产环境 | 4x GPU (24GB) |
| **pro.yaml** | 专业配置 | 大规模训练 | 16x GPU (80GB) |
| **full.yaml** | 完整配置 | 最大性能、所有功能 | 64x GPU (80GB) |

## 使用方法

### 命令行方式

```bash
# 使用lite配置训练
apt-model train --profile lite --data /path/to/data

# 使用standard配置
apt-model train --profile standard --data /path/to/data

# 使用pro配置
apt-model train --profile pro --data /path/to/data

# 使用full配置（需要大量资源）
apt-model train --profile full --data /path/to/data
```

### Python API方式

```python
from apt.trainops.engine import Trainer
from apt.core.config import load_profile

# 加载配置
config = load_profile("standard")

# 创建训练器
trainer = Trainer(config)

# 开始训练
trainer.train(dataset)
```

### 自定义配置

你可以基于现有配置创建自定义配置：

```bash
# 复制一个配置作为起点
cp profiles/standard.yaml profiles/my_custom.yaml

# 编辑配置
vim profiles/my_custom.yaml

# 使用自定义配置
apt-model train --profile my_custom --data /path/to/data
```

## 配置结构

每个配置文件包含以下部分：

### 1. Profile元数据
```yaml
profile:
  name: standard
  description: "配置描述"
  version: "2.0.0"
```

### 2. Model配置
```yaml
model:
  architecture: apt_base
  hidden_size: 1024
  num_layers: 24
  features:
    multimodal: true
    moe: false
```

### 3. Training配置
```yaml
training:
  batch_size: 32
  learning_rate: 3.0e-5
  distributed:
    enabled: true
    world_size: 4
```

### 4. Memory配置
```yaml
memory:
  offload_optimizer: false
  max_memory_per_gpu: "24GB"
```

### 5. vGPU配置
```yaml
vgpu:
  enabled: true
  max_virtual_gpus: 4
```

### 6. Extensions配置
```yaml
extensions:
  rag:
    enabled: true
  knowledge_graph:
    enabled: true
```

### 7. Monitoring配置
```yaml
monitoring:
  tensorboard: true
  wandb: true
```

### 8. Checkpoints配置
```yaml
checkpoints:
  save_interval: 500
  keep_last_n: 5
```

## 配置继承

你可以基于已有配置进行扩展：

```yaml
# my_custom.yaml
base: standard  # 继承standard配置

# 只覆盖需要的部分
training:
  batch_size: 64  # 覆盖batch_size

model:
  features:
    moe: true  # 启用MoE
```

## 环境变量覆盖

配置值可以通过环境变量覆盖：

```bash
# 覆盖batch size
APT_TRAINING_BATCH_SIZE=64 apt-model train --profile standard

# 覆盖learning rate
APT_TRAINING_LEARNING_RATE=1e-5 apt-model train --profile standard
```

## 最佳实践

1. **开发时使用lite** - 快速迭代和调试
2. **测试时使用standard** - 验证功能和性能
3. **生产时使用pro或full** - 获得最佳性能
4. **创建自定义配置** - 针对特定任务优化
5. **版本控制配置文件** - 确保可重现性

## 配置验证

训练前验证配置：

```bash
apt-model validate-config --profile standard
```

## 迁移指南

### 从apt_model 1.x迁移

**之前（代码复制）：**
```
apt_model/
├── core/      # 核心版本代码
├── lite/      # 轻量版本代码（复制）
├── pro/       # 专业版本代码（复制）
└── full/      # 完整版本代码（复制）
```

**现在（配置驱动）：**
```
apt/
├── model/     # 统一的模型代码
├── trainops/  # 统一的训练代码
└── ...
profiles/
├── lite.yaml     # 轻量级配置
├── standard.yaml # 标准配置
├── pro.yaml      # 专业配置
└── full.yaml     # 完整配置
```

**优势：**
- ✅ 无代码重复
- ✅ 易于维护
- ✅ 配置可组合
- ✅ 版本控制友好
- ✅ 灵活性更高

## 技术细节

配置文件使用YAML格式，通过`apt.core.config`模块加载：

```python
from apt.core.config import ConfigLoader

loader = ConfigLoader()
config = loader.load("standard")  # 加载standard.yaml
```

配置加载时会进行：
1. Schema验证
2. 类型检查
3. 默认值填充
4. 环境变量替换
5. 配置继承解析

## 常见问题

**Q: 可以混合使用多个配置文件吗？**
A: 是的，使用`base`字段继承基础配置，然后覆盖特定部分。

**Q: 配置文件存放在哪里？**
A: 官方配置在`profiles/`目录，用户自定义配置可以放在任何位置。

**Q: 如何查看当前使用的完整配置？**
A: 使用`apt-model show-config --profile standard`命令。

**Q: 配置文件支持哪些格式？**
A: 目前支持YAML，未来可能支持JSON和TOML。

## 贡献

欢迎贡献新的配置文件！请确保：
1. 配置文件有清晰的描述
2. 包含适用场景说明
3. 经过测试验证
4. 遵循命名规范

---

**重要**: 这是APT 2.0的核心改进之一。通过配置文件而非代码复制，我们实现了更清晰、更易维护的架构。
