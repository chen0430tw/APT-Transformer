# APT模型微调（Fine-tuning）完整指南

## 📋 概述

APT模型微调模块完全基于现有的模块化组件构建，无需重复造轮子：
- ✅ 复用 `checkpoint.load_model()` - 加载预训练模型
- ✅ 复用 `trainer` 的训练逻辑
- ✅ 复用 `data loading` 功能
- ✅ 复用 `generator` 和 `evaluator` 模块

---

## 🚀 快速开始

### 基础微调

```bash
# 最简单的微调命令
python -m apt_model fine-tune \
  --model-path apt_model \
  --data-path finetune_data.txt \
  --save-path apt_model_finetuned
```

### 完整配置微调

```bash
python -m apt_model fine-tune \
  --model-path apt_model \
  --data-path train_data.txt \
  --val-data-path val_data.txt \
  --epochs 5 \
  --batch-size 8 \
  --learning-rate 1e-5 \
  --save-path apt_model_finetuned \
  --freeze-embeddings \
  --freeze-encoder-layers 2
```

---

## 📊 参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model-path` | 预训练模型路径 | `apt_model` |
| `--data-path` | 微调训练数据路径 | `finetune_data.txt` |

### 训练配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 5 | 训练轮数（微调建议3-10轮） |
| `--batch-size` | 8 | 批次大小 |
| `--learning-rate` | 1e-5 | 学习率（微调建议1e-5到5e-5） |
| `--save-path` | apt_model_finetuned | 模型保存路径 |

### 微调专用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--val-data-path` | None | 验证数据路径 |
| `--freeze-embeddings` | False | 是否冻结embedding层 |
| `--freeze-encoder-layers` | None | 冻结前N层encoder |
| `--freeze-decoder-layers` | None | 冻结前N层decoder |
| `--early-stopping-patience` | 3 | 早停耐心值 |
| `--eval-steps` | 100 | 评估间隔（步数） |
| `--save-steps` | 500 | 保存检查点间隔（步数） |
| `--max-samples` | None | 最大样本数 |

---

## 💡 使用场景

### 场景1：领域适应

将通用模型微调到特定领域（如医疗、法律、金融）

```bash
python -m apt_model fine-tune \
  --model-path apt_model \
  --data-path medical_texts.txt \
  --epochs 5 \
  --learning-rate 2e-5 \
  --save-path apt_model_medical
```

### 场景2：任务专精

微调模型以执行特定任务

```bash
python -m apt_model fine-tune \
  --model-path apt_model \
  --data-path qa_pairs.txt \
  --val-data-path qa_val.txt \
  --epochs 10 \
  --learning-rate 1e-5 \
  --save-path apt_model_qa
```

### 场景3：参数高效微调

冻结大部分层，只微调顶层

```bash
python -m apt_model fine-tune \
  --model-path apt_model \
  --data-path small_dataset.txt \
  --freeze-embeddings \
  --freeze-encoder-layers 4 \
  --freeze-decoder-layers 4 \
  --epochs 5 \
  --learning-rate 3e-5 \
  --save-path apt_model_efficient
```

**好处**：
- 减少可训练参数
- 降低显存占用
- 防止过拟合
- 训练更快

---

## 🔧 高级功能

### 1. 早停机制（Early Stopping）

自动停止训练以防止过拟合：

```bash
python -m apt_model fine-tune \
  --model-path apt_model \
  --data-path train.txt \
  --val-data-path val.txt \
  --early-stopping-patience 3  # 验证损失3轮不下降就停止
```

### 2. 层冻结策略

#### 冻结Embedding层
```bash
--freeze-embeddings
```
**适用场景**：数据量小、词汇表不变

#### 冻结底层
```bash
--freeze-encoder-layers 2  # 冻结encoder前2层
--freeze-decoder-layers 2  # 冻结decoder前2层
```
**适用场景**：任务相似、数据量小

#### 完全微调
不使用任何冻结参数
**适用场景**：数据量大、任务差异大

### 3. 学习率选择

| 场景 | 推荐学习率 |
|------|-----------|
| 相似任务 | 1e-5 |
| 不同领域 | 2e-5 ~ 3e-5 |
| 全新任务 | 3e-5 ~ 5e-5 |
| 小数据集 | 5e-6 ~ 1e-5 |

---

## 📝 数据格式

### 训练数据格式

支持纯文本文件，每行一个样本：

```text
人工智能是计算机科学的一个分支。
深度学习是机器学习的一个子领域。
自然语言处理用于理解和生成人类语言。
...
```

### 验证数据格式

与训练数据相同：

```text
神经网络由多个层组成。
卷积神经网络常用于图像处理。
循环神经网络适合处理序列数据。
...
```

---

## 🎯 最佳实践

### 1. 数据准备

- ✅ 确保数据质量高
- ✅ 数据应与目标任务相关
- ✅ 建议至少1000条样本
- ✅ 准备验证集（10-20%）

### 2. 超参数选择

**小数据集（< 1000样本）：**
```bash
--epochs 10 \
--batch-size 4 \
--learning-rate 1e-5 \
--freeze-embeddings \
--freeze-encoder-layers 2
```

**中等数据集（1000-10000样本）：**
```bash
--epochs 5 \
--batch-size 8 \
--learning-rate 2e-5 \
--freeze-embeddings
```

**大数据集（> 10000样本）：**
```bash
--epochs 3 \
--batch-size 16 \
--learning-rate 3e-5
```

### 3. 监控训练

观察以下指标：
- 训练损失是否下降
- 验证损失是否下降
- 生成样本质量
- 是否出现过拟合

### 4. 防止过拟合

- 使用验证集和早停
- 冻结底层
- 使用较小的学习率
- 增加数据增强

---

## 📊 效果评估

### 自动评估

微调过程中会：
1. 定期在验证集上评估
2. 生成样本文本
3. 计算质量评分
4. 保存最佳模型

### 使用微调后的模型

```bash
# 评估微调后的模型
python -m apt_model evaluate --model-path apt_model_finetuned

# 与微调后的模型聊天
python -m apt_model chat --model-path apt_model_finetuned
```

---

## 🔬 代码示例

### Python API 使用

```python
from apt_model.training.finetuner import fine_tune_model

# 基础微调
model, tokenizer, config = fine_tune_model(
    pretrained_model_path="apt_model",
    train_data_path="finetune_data.txt",
    epochs=5,
    learning_rate=1e-5,
    save_path="apt_model_finetuned"
)

# 高级微调
model, tokenizer, config = fine_tune_model(
    pretrained_model_path="apt_model",
    train_data_path="train.txt",
    val_data_path="val.txt",
    epochs=5,
    batch_size=8,
    learning_rate=2e-5,
    freeze_embeddings=True,
    freeze_encoder_layers=2,
    freeze_decoder_layers=2,
    save_path="apt_model_finetuned",
    early_stopping_patience=3,
    eval_steps=100
)
```

### 自定义微调

```python
from apt_model.training.finetuner import FineTuner

# 创建微调器
finetuner = FineTuner("apt_model")

# 自定义冻结策略
finetuner.freeze_layers(
    freeze_embeddings=True,
    freeze_encoder_layers=3,
    freeze_decoder_layers=3
)

# 执行微调
model, tokenizer, config = finetuner.fine_tune(
    train_data_path="train.txt",
    val_data_path="val.txt",
    epochs=5,
    batch_size=8,
    learning_rate=2e-5,
    save_path="apt_model_custom"
)
```

---

## ⚠️ 常见问题

### Q: 微调需要多长时间？
A: 取决于数据量和硬件：
- 1000样本，5轮：约30分钟（GPU）
- 10000样本，5轮：约3小时（GPU）
- 100000样本，5轮：约24小时（GPU）

### Q: 显存不足怎么办？
A:
1. 减小batch size：`--batch-size 4`
2. 冻结更多层：`--freeze-encoder-layers 4`
3. 减小数据量：`--max-samples 5000`

### Q: 如何判断微调是否成功？
A: 观察以下指标：
- 训练损失稳定下降
- 验证损失下降（不上升）
- 生成样本质量提升
- 任务性能提升

### Q: 微调后效果不好怎么办？
A:
1. 增加训练数据
2. 调整学习率
3. 增加训练轮数
4. 减少冻结的层数
5. 检查数据质量

### Q: 如何继续微调已微调的模型？
A:
```bash
python -m apt_model fine-tune \
  --model-path apt_model_finetuned \  # 使用已微调的模型
  --data-path new_data.txt \
  --save-path apt_model_finetuned_v2
```

---

## 🛠️ 模块化设计优势

本微调模块完全基于现有组件：

```python
# 复用的模块
from apt_model.training.checkpoint import load_model, save_model     # 加载保存
from apt_model.data.external_data import load_external_data         # 数据加载
from apt_model.generation.generator import generate_natural_text    # 生成
from apt_model.generation.evaluator import evaluate_text_quality    # 评估
from apt_model.utils import get_device, set_seed                    # 工具
```

**好处**：
- ✅ 代码复用，减少工作量
- ✅ 统一接口，易于维护
- ✅ 质量保证，久经考验
- ✅ 扩展方便，模块化设计

---

## 📚 参考资料

- [APT模型训练指南](../../README.md)
- [Optuna超参数优化](../product/OPTUNA_GUIDE.md)
- [Debug模式使用](../../apt/core/config/settings_manager.py)

---

## 🎓 示例脚本

### 示例1：快速微调
```bash
#!/bin/bash
# quick_finetune.sh

python -m apt_model fine-tune \
  --model-path apt_model \
  --data-path finetune_data.txt \
  --epochs 3 \
  --learning-rate 2e-5 \
  --save-path apt_model_quick
```

### 示例2：完整微调
```bash
#!/bin/bash
# full_finetune.sh

python -m apt_model fine-tune \
  --model-path apt_model \
  --data-path train.txt \
  --val-data-path val.txt \
  --epochs 5 \
  --batch-size 8 \
  --learning-rate 1e-5 \
  --freeze-embeddings \
  --freeze-encoder-layers 2 \
  --early-stopping-patience 3 \
  --eval-steps 100 \
  --save-steps 500 \
  --save-path apt_model_full
```

### 示例3：参数高效微调
```bash
#!/bin/bash
# efficient_finetune.sh

python -m apt_model fine-tune \
  --model-path apt_model \
  --data-path small_dataset.txt \
  --epochs 10 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --freeze-embeddings \
  --freeze-encoder-layers 4 \
  --freeze-decoder-layers 4 \
  --save-path apt_model_efficient
```

---

**Happy Fine-tuning! 🎯**
