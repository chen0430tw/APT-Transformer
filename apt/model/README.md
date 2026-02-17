# APT Model Domain

模型定义域 - 包含所有模型架构、层、分词器、损失函数和优化器

## 概述

`apt.model` 是APT 2.0架构的核心域之一，负责模型定义和相关组件。

## 目录结构

```
apt/model/
├── architectures/    # 模型架构定义
├── layers/          # 基础层组件
├── tokenization/    # 分词器
├── losses/          # 损失函数
├── optim/          # 优化器
└── extensions/     # 核心扩展（RAG, KG, MCP）
```

## 模块说明

### 1. architectures/

模型架构定义：

```python
from apt.model.architectures import APTLargeModel, MultimodalAPTModel

# 创建模型
model = APTLargeModel(
    hidden_size=2048,
    num_layers=32,
    num_attention_heads=32
)
```

包含的模型：
- APTLargeModel - APT核心模型
- MultimodalAPTModel - 多模态模型
- Claude4Model - Claude 4风格模型
- GPT5Model - GPT-5风格模型
- O1Model - O1风格模型
- 其他特定任务模型

### 2. layers/

基础层组件：

```python
from apt.model.layers import (
    MultiHeadAttention,
    FeedForward,
    LayerNorm,
    RotaryEmbedding
)
```

包含的层：
- Attention mechanisms（注意力机制）
- Feed-forward networks（前馈网络）
- Normalization layers（归一化层）
- Embedding layers（嵌入层）
- Custom blocks（自定义块）

### 3. tokenization/

分词器实现：

```python
from apt.model.tokenization import ChineseTokenizer

tokenizer = ChineseTokenizer()
tokens = tokenizer.encode("你好世界")
```

功能：
- 中文分词
- 多语言支持
- Tokenizer集成
- 语言检测

### 4. losses/

损失函数：

```python
from apt.model.losses import APTLoss, ContrastiveLoss

loss_fn = APTLoss()
loss = loss_fn(predictions, targets)
```

包含的损失：
- APT特定损失
- 多任务损失
- 对比学习损失
- 自定义损失函数

### 5. optim/

优化器：

```python
from apt.model.optim import APTOptimizer

optimizer = APTOptimizer(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.01
)
```

功能：
- 自定义优化器
- 学习率调度器
- 优化工具

### 6. extensions/

核心扩展（由核心团队维护）：

```python
from apt.model.extensions.rag import RAGExtension
from apt.model.extensions.kg import KnowledgeGraphExtension

# 启用RAG扩展
model = APTLargeModel(extensions=[RAGExtension()])
```

包含的扩展：
- **RAG** - 检索增强生成
- **KG** - 知识图谱集成
- **MCP** - 模型上下文协议
- **Graph RAG** - 图检索增强生成

扩展特点：
- 深度集成到模型架构
- 可以修改模型行为
- 核心团队维护
- 编译时集成

## 使用示例

### 基础使用

```python
from apt.model.architectures import APTLargeModel
from apt.model.losses import APTLoss
from apt.model.optim import APTOptimizer

# 创建模型
model = APTLargeModel(
    hidden_size=2048,
    num_layers=32,
    num_attention_heads=32,
    vocab_size=50000
)

# 创建损失函数
loss_fn = APTLoss()

# 创建优化器
optimizer = APTOptimizer(
    model.parameters(),
    lr=3e-5
)

# 训练循环
for batch in dataloader:
    outputs = model(batch['input_ids'])
    loss = loss_fn(outputs, batch['labels'])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 使用扩展

```python
from apt.model.architectures import APTLargeModel
from apt.model.extensions.rag import RAGExtension

# 创建带RAG扩展的模型
model = APTLargeModel(
    hidden_size=2048,
    num_layers=32,
    extensions=[
        RAGExtension(
            index_type='faiss',
            embedding_dim=2048,
            top_k=10
        )
    ]
)

# 使用RAG增强的生成
outputs = model.generate_with_rag(
    prompt="什么是自生成变换器？",
    max_length=100
)
```

### 多模态模型

```python
from apt.model.architectures import MultimodalAPTModel

# 创建多模态模型
model = MultimodalAPTModel(
    text_config={
        'hidden_size': 2048,
        'num_layers': 32
    },
    vision_config={
        'image_size': 224,
        'patch_size': 16
    }
)

# 多模态输入
outputs = model(
    text_input_ids=text_tokens,
    image_pixel_values=images
)
```

## 配置驱动

使用profile配置创建模型：

```python
from apt.core.config import load_profile
from apt.model.architectures import create_model_from_config

# 加载配置
config = load_profile('standard')

# 从配置创建模型
model = create_model_from_config(config)
```

## 与trainops的关系

- **apt.model** - 定义"what"（模型是什么）
- **apt.trainops** - 定义"how"（如何训练模型）

```python
from apt.model.architectures import APTLargeModel
from apt.trainops.engine import Trainer

# model定义架构
model = APTLargeModel(...)

# trainops负责训练
trainer = Trainer(model=model, ...)
trainer.train()
```

## 迁移状态

🚧 **当前状态**: Skeleton已创建，内容将在PR-4中迁移

迁移计划：
- [ ] PR-4: 从apt.apt_model.modeling迁移所有模型文件
- [ ] PR-4: 从apt.apt_model.modeling迁移encoders和blocks
- [ ] PR-4: 整合扩展功能（RAG, KG, MCP）
- [ ] PR-5: 完善compat层重导出

## 设计原则

1. **单一职责** - 只定义模型，不包含训练逻辑
2. **可组合** - 层和模块可以灵活组合
3. **可配置** - 通过配置文件驱动
4. **可扩展** - 通过extensions机制扩展功能
5. **类型安全** - 使用类型注解

## 开发指南

### 添加新模型

```python
# apt/model/architectures/my_model.py
from apt.model.layers import MultiHeadAttention, FeedForward

class MyModel(nn.Module):
    """我的新模型"""

    def __init__(self, config):
        super().__init__()
        # 实现...

    def forward(self, input_ids):
        # 前向传播...
        return outputs

# 在__init__.py中导出
__all__ = ['MyModel']
```

### 添加新扩展

```python
# apt/model/extensions/my_extension.py
from apt.model.extensions.base import Extension

class MyExtension(Extension):
    """我的扩展"""

    def __init__(self, **kwargs):
        super().__init__()
        # 初始化...

    def modify_model(self, model):
        """修改模型架构"""
        # 实现...
        return modified_model
```

## API文档

详细API文档：https://apt-transformer.readthedocs.io/model/

## 测试

```bash
# 测试模型模块
pytest apt/model/tests/

# 测试特定架构
pytest apt/model/tests/test_architectures.py
```

## 相关链接

- [TrainOps Domain](../trainops/README.md) - 训练域
- [vGPU Domain](../vgpu/README.md) - 虚拟GPU域
- [Extensions vs Plugins](../../docs/architecture/extensions_vs_plugins.md)
- [Configuration Profiles](../../profiles/README.md)

---

**Version**: 2.0.0-alpha
**Status**: Skeleton (内容迁移中)
**Last Updated**: 2026-01-22
