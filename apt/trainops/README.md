# APT TrainOps Domain

训练运营域 - 训练编排和生命周期管理

## 概述

`apt.trainops` 是APT 2.0架构的核心域之一，负责训练编排、分布式训练、数据管理和训练生命周期。

## 目录结构

```
apt/trainops/
├── engine/         # 训练引擎
├── distributed/    # 分布式训练
├── data/          # 数据加载
├── checkpoints/   # 检查点管理
├── eval/          # 评估和验证
└── artifacts/     # 训练产物管理
```

## 模块说明

### 1. engine/

训练引擎实现：

```python
from apt.trainops.engine import Trainer, Finetuner

# 创建训练器
trainer = Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    config=config
)

# 开始训练
trainer.train()
```

包含的训练器：
- Trainer - 主训练器
- Finetuner - 微调器
- PreTrainer - 预训练器
- ClaudeTrainer - Claude风格训练
- GPT5Trainer - GPT-5风格训练
- O1Trainer - O1风格训练

### 2. distributed/

分布式训练支持：

```python
from apt.trainops.distributed import setup_distributed, DDPWrapper

# 设置分布式环境
setup_distributed(backend='nccl')

# 包装模型
model = DDPWrapper(model)

# 分布式训练
trainer = Trainer(model=model, distributed=True)
trainer.train()
```

支持的并行策略：
- **DDP** - DistributedDataParallel（数据并行）
- **FSDP** - Fully Sharded Data Parallel（全分片数据并行）
- **Pipeline Parallel** - 流水线并行
- **Tensor Parallel** - 张量并行
- **Expert Parallel** - 专家并行（for MoE）
- **Sequence Parallel** - 序列并行

### 3. data/

数据加载和预处理：

```python
from apt.trainops.data import APTDataLoader, create_dataloader

# 创建数据加载器
dataloader = create_dataloader(
    dataset=dataset,
    batch_size=32,
    num_workers=4,
    shuffle=True
)

# 迭代数据
for batch in dataloader:
    # 训练逻辑...
```

功能：
- 数据加载器
- 数据预处理
- 数据增强
- 数据集实现

### 4. checkpoints/

检查点管理：

```python
from apt.trainops.checkpoints import CheckpointManager

# 创建检查点管理器
checkpoint_manager = CheckpointManager(
    save_dir='checkpoints/',
    save_interval=500,
    keep_last_n=5
)

# 保存检查点
checkpoint_manager.save(
    model=model,
    optimizer=optimizer,
    step=1000
)

# 加载检查点
state = checkpoint_manager.load('checkpoint-1000')
```

功能：
- 检查点保存/加载
- 断点续训
- 检查点版本管理
- 分布式检查点
- 异步保存

### 5. eval/

评估和验证：

```python
from apt.trainops.eval import Evaluator, compute_metrics

# 创建评估器
evaluator = Evaluator(
    model=model,
    eval_dataset=eval_data,
    metrics=['accuracy', 'perplexity']
)

# 运行评估
results = evaluator.evaluate()
print(results)  # {'accuracy': 0.95, 'perplexity': 12.3}
```

功能：
- 评估循环
- 指标计算
- 基准测试
- 性能监控

### 6. artifacts/

训练产物管理：

```python
from apt.trainops.artifacts import ArtifactManager

# 创建产物管理器
artifact_manager = ArtifactManager(
    output_dir='outputs/',
    experiment_name='my_experiment'
)

# 保存产物
artifact_manager.save_model(model)
artifact_manager.save_metrics(metrics)
artifact_manager.save_logs(logs)
```

管理的产物：
- 训练模型
- 训练日志
- 指标数据
- 实验配置
- 中间结果

## 使用示例

### 基础训练

```python
from apt.model.architectures import APTLargeModel
from apt.trainops.engine import Trainer
from apt.trainops.data import create_dataloader

# 创建模型
model = APTLargeModel()

# 准备数据
train_loader = create_dataloader(train_dataset, batch_size=32)
eval_loader = create_dataloader(eval_dataset, batch_size=32)

# 创建训练器
trainer = Trainer(
    model=model,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    max_steps=10000,
    learning_rate=3e-5
)

# 开始训练
trainer.train()
```

### 分布式训练

```python
from apt.model.architectures import APTLargeModel
from apt.trainops.engine import Trainer
from apt.trainops.distributed import setup_distributed

# 设置分布式环境
setup_distributed(
    backend='nccl',
    world_size=8,
    rank=int(os.environ['RANK'])
)

# 创建模型
model = APTLargeModel()

# 分布式训练
trainer = Trainer(
    model=model,
    train_dataloader=train_loader,
    distributed_config={
        'strategy': 'fsdp',
        'world_size': 8
    }
)

trainer.train()
```

### 使用配置文件

```python
from apt.core.config import load_profile
from apt.trainops.engine import Trainer

# 加载配置
config = load_profile('standard')

# 从配置创建训练器
trainer = Trainer.from_config(config)

# 训练
trainer.train()
```

### 微调

```python
from apt.model.architectures import APTLargeModel
from apt.trainops.engine import Finetuner

# 加载预训练模型
model = APTLargeModel.from_pretrained('apt-base')

# 创建微调器
finetuner = Finetuner(
    model=model,
    train_dataset=finetune_data,
    learning_rate=1e-5,
    num_epochs=3
)

# 微调
finetuner.finetune()

# 保存微调后的模型
finetuner.save_model('my_finetuned_model')
```

### 断点续训

```python
from apt.trainops.engine import Trainer
from apt.trainops.checkpoints import CheckpointManager

# 加载检查点
checkpoint_manager = CheckpointManager('checkpoints/')
state = checkpoint_manager.load_latest()

# 恢复训练
trainer = Trainer(
    model=model,
    resume_from_checkpoint=state
)

trainer.train()
```

## 与model的关系

- **apt.model** - 定义"what"（模型是什么）
- **apt.trainops** - 定义"how"（如何训练模型）

清晰的职责分离：

```python
# model: 定义架构
from apt.model.architectures import APTLargeModel
model = APTLargeModel(hidden_size=2048, num_layers=32)

# trainops: 训练模型
from apt.trainops.engine import Trainer
trainer = Trainer(model=model)
trainer.train()
```

## 配置驱动训练

使用profile配置整个训练流程：

```yaml
# profiles/my_training.yaml
training:
  batch_size: 32
  learning_rate: 3e-5
  max_steps: 10000

  distributed:
    enabled: true
    strategy: fsdp
    world_size: 8

  checkpoints:
    save_interval: 500
    keep_last_n: 5
```

```python
from apt.core.config import load_profile
from apt.trainops.engine import Trainer

config = load_profile('my_training')
trainer = Trainer.from_config(config)
trainer.train()
```

## 分布式训练策略选择

根据模型大小和资源选择合适的策略：

| 模型大小 | GPU数量 | 推荐策略 |
|---------|--------|---------|
| < 1B | 1-4 | DDP |
| 1B-7B | 4-16 | FSDP |
| 7B-30B | 16-64 | FSDP + Pipeline |
| 30B-100B | 64-256 | FSDP + Pipeline + Tensor |
| > 100B | 256+ | 全并行（FSDP+PP+TP+EP） |

示例配置：

```python
# 小模型 (< 1B)
trainer = Trainer(
    model=model,
    distributed_config={'strategy': 'ddp'}
)

# 中模型 (1B-7B)
trainer = Trainer(
    model=model,
    distributed_config={'strategy': 'fsdp'}
)

# 大模型 (> 30B)
trainer = Trainer(
    model=model,
    distributed_config={
        'strategy': 'hybrid',
        'pipeline_parallel': 4,
        'tensor_parallel': 4,
        'data_parallel': 4
    }
)
```

## 迁移状态

🚧 **当前状态**: Skeleton已创建，内容将在PR-3中迁移

迁移计划：
- [ ] PR-3: 从apt.apt_model.training迁移所有训练器
- [ ] PR-3: 从apt.core.data迁移数据处理
- [ ] PR-3: 整合分布式训练支持
- [ ] PR-5: 完善compat层重导出

## 训练生命周期

完整的训练生命周期：

```
1. 初始化
   ├── 加载配置
   ├── 创建模型
   ├── 准备数据
   └── 设置分布式环境

2. 训练循环
   ├── 前向传播
   ├── 计算损失
   ├── 反向传播
   ├── 更新参数
   ├── 评估（可选）
   └── 保存检查点

3. 结束
   ├── 最终评估
   ├── 保存模型
   └── 清理资源
```

由TrainOps统一管理整个生命周期。

## 监控和日志

集成多种监控工具：

```python
from apt.trainops.engine import Trainer

trainer = Trainer(
    model=model,
    monitoring={
        'tensorboard': True,
        'wandb': True,
        'mlflow': True
    }
)

trainer.train()  # 自动记录到所有监控系统
```

## 最佳实践

1. **使用配置文件** - 不要硬编码参数
2. **启用检查点** - 定期保存，避免训练中断损失
3. **监控指标** - 使用TensorBoard/W&B追踪训练
4. **梯度裁剪** - 防止梯度爆炸
5. **混合精度** - 使用bf16/fp16加速训练
6. **分布式训练** - 大模型必须用FSDP或更高级策略

## 故障恢复

TrainOps自动处理故障恢复：

```python
trainer = Trainer(
    model=model,
    auto_resume=True,  # 自动从最新检查点恢复
    checkpoint_dir='checkpoints/'
)

# 训练中断后重新运行，自动恢复
trainer.train()
```

## API文档

详细API文档：https://apt-transformer.readthedocs.io/trainops/

## 测试

```bash
# 测试训练模块
pytest apt/trainops/tests/

# 测试分布式训练（需要多GPU）
torchrun --nproc_per_node=4 apt/trainops/tests/test_distributed.py
```

## 相关链接

- [Model Domain](../model/README.md) - 模型域
- [vGPU Domain](../vgpu/README.md) - 虚拟GPU域
- [Configuration Profiles](../../profiles/README.md)
- [Distributed Training Guide](../../docs/guides/distributed_training.md)

---

**Version**: 2.0.0-alpha
**Status**: Skeleton (内容迁移中)
**Last Updated**: 2026-01-22
