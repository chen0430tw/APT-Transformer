# GPT模型变体训练指南

本指南介绍如何使用统一的训练器来训练项目中的三个GPT模型变体。

## 概述

项目包含三个高级GPT模型变体：

1. **GPT-4o** (`gpt4o_model.py`) - 增强型Transformer，包含VeinFlow/TVA注意力机制
2. **GPT-o3** (`gpto3_model.py`) - 结构化推理模型，基于GPT-4o架构
3. **GPT-5** (`gpt5_model.py`) - MoE (Mixture of Experts) 模型

## 重要说明：Claude vs GPT

虽然这些模型文件名包含"GPT"，但它们是**独立的Transformer实现**，不是OpenAI的GPT模型。

训练数据已经针对**Claude风格**进行了优化：
- 强调有帮助、无害、诚实的对话
- 包含Claude的自我介绍和特性说明
- 中英文双语支持

## 快速开始

### 1. 训练GPT-4o模型

```bash
python -m apt_model.training.train_gpt_models --model gpt4o --epochs 20
```

### 2. 训练GPT-o3模型 (推理增强)

```bash
python -m apt_model.training.train_gpt_models --model gpto3 --epochs 30
```

### 3. 训练GPT-5模型 (MoE)

```bash
python -m apt_model.training.train_gpt_models --model gpt5 --epochs 25 --lr 1e-4
```

## 完整参数

```bash
python -m apt_model.training.train_gpt_models \
  --model gpt4o \           # 模型类型: gpt4o, gpto3, gpt5
  --epochs 20 \              # 训练轮数
  --batch-size 8 \           # 批次大小
  --lr 3e-4 \                # 学习率
  --save-path ./my_model \   # 保存路径
  --device cuda              # 设备: cuda或cpu
```

## 在Python代码中使用

```python
from apt_model.training.train_gpt_models import train_gpt_model

# 训练GPT-4o模型
model, tokenizer = train_gpt_model(
    model_type='gpt4o',
    epochs=20,
    batch_size=8,
    learning_rate=3e-4,
    save_path='./gpt4o_model'
)

# 训练GPT-o3模型
model, tokenizer = train_gpt_model(
    model_type='gpto3',
    epochs=30,
    batch_size=4,  # o3需要更多内存
    learning_rate=1e-4,
    save_path='./gpto3_model'
)

# 训练GPT-5模型 (MoE)
model, tokenizer = train_gpt_model(
    model_type='gpt5',
    epochs=25,
    batch_size=4,  # MoE需要更多内存
    learning_rate=1e-4,
    save_path='./gpt5_model'
)
```

## 使用自定义训练数据

```python
from apt_model.training.train_gpt_models import train_gpt_model

# 准备你的训练文本
my_texts = [
    "Hello, I'm Claude, an AI assistant.",
    "I'm designed to be helpful, harmless, and honest.",
    "你好，我是Claude。",
    "我可以用中文和英文交流。",
    # ... 更多训练数据
]

# 训练模型
model, tokenizer = train_gpt_model(
    model_type='gpt4o',
    texts=my_texts,
    epochs=20,
    save_path='./custom_model'
)
```

## 模型特点对比

| 模型 | 特点 | 适用场景 | 内存需求 |
|------|------|----------|----------|
| GPT-4o | VeinFlow/TVA注意力机制 | 通用文本生成 | 中等 |
| GPT-o3 | 结构化推理 + 停止单元 | 需要推理的任务 | 较高 |
| GPT-5 | MoE专家混合 | 大规模多任务 | 高 |

## 模型架构说明

### GPT-4o
- 动态τ门控
- 低秩Vein子空间
- 混合FFN (Mini-MoE)
- 快速路径调度

### GPT-o3
- 基于GPT-4o架构
- 停止单元 (HaltingUnit)
- 专家路由器 (ExpertRouter)
- 结构化推理控制器

### GPT-5
- Codebook MoE (top-k + shared expert)
- Leaf-Vote机制
- 流式检索
- 双态精度对齐

## 训练建议

### 学习率
- **GPT-4o**: 3e-4 (默认)
- **GPT-o3**: 1e-4 (较低，因为有推理模块)
- **GPT-5**: 1e-4 (较低，因为MoE)

### 批次大小
- **GPT-4o**: 8-16 (取决于GPU内存)
- **GPT-o3**: 4-8 (推理模块需要更多内存)
- **GPT-5**: 4-8 (MoE需要更多内存)

### 训练轮数
- **GPT-4o**: 20-30 epochs
- **GPT-o3**: 30-50 epochs (推理能力需要更多训练)
- **GPT-5**: 25-40 epochs

## 内存优化建议

如果遇到内存不足：

1. **减少批次大小**
   ```bash
   --batch-size 2
   ```

2. **使用CPU训练** (较慢但内存充足)
   ```bash
   --device cpu
   ```

3. **使用混合精度训练** (需要修改代码)
   ```python
   # 在train_gpt_models.py中添加
   from torch.cuda.amp import autocast, GradScaler
   ```

## 加载训练好的模型

```python
import torch
from apt_model.modeling.gpt4o_model import GPT4oModel

# 加载模型
checkpoint = torch.load('gpt4o_model/model.pt')
vocab_size = checkpoint['vocab_size']

# 重建模型
model = GPT4oModel(
    vocab_size=vocab_size,
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048,
    max_seq_len=128
)

# 加载权重
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## 常见问题

### Q: 为什么叫GPT-4o/o3/5？
A: 这些是独立的Transformer实现，借鉴了不同GPT版本的架构理念，但不是OpenAI的官方模型。它们是本项目的研究实现。

### Q: 训练数据是Claude风格的吗？
A: 是的！训练数据已经优化为Claude风格，强调：
- 有帮助 (Helpful)
- 无害 (Harmless)
- 诚实 (Honest)

### Q: 这些模型可以用于生产吗？
A: 这些是研究/原型实现。生产使用需要：
- 更大规模的训练数据
- 更长时间的训练
- 安全性和对齐调优
- 充分的测试

### Q: 三个模型可以一起使用吗？
A: 可以！你可以训练它们并进行集成：
```python
# 训练三个模型
model_4o = train_gpt_model(model_type='gpt4o', ...)
model_o3 = train_gpt_model(model_type='gpto3', ...)
model_5 = train_gpt_model(model_type='gpt5', ...)

# 集成预测
def ensemble_predict(input_text):
    logits_4o = model_4o(input_text)
    logits_o3 = model_o3(input_text)
    logits_5 = model_5(input_text)
    # 平均或投票
    return (logits_4o + logits_o3 + logits_5) / 3
```

## 下一步

1. 尝试训练各个模型
2. 在你的数据上微调
3. 比较不同模型的性能
4. 探索模型集成策略

## 相关文档

- 主训练器：`apt_model/training/trainer.py` - APT模型训练
- 模型实现：
  - `apt_model/modeling/gpt4o_model.py`
  - `apt_model/modeling/gpto3_model.py`
  - `apt_model/modeling/gpt5_model.py`
- 配置示例：`examples/profiles/gpt5_moe_reasoning.yaml`

---

**注意**: 这些模型是为了研究和学习目的。如果你需要生产级别的语言模型，建议使用经过充分训练和对齐的商业模型（如Claude或GPT-4）的API。
