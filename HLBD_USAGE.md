# HLBD (分层语言启蒙数据集) 使用指南

## 📚 概述

**HLBD (Hierarchical Language Bootstrapping Dataset, 分层语言启蒙数据集)** 是一套专为大规模语言模型预训练而设计的数据集，其核心思想在于以分层结构呈现语言的基本组成单元，从最基础的视觉符号到完整的自然语言表达，逐步引导模型建立起符号运算法则和形式逻辑。

## 🏗️ 数据集结构

HLBD 包含 **8个层级**，每个层级都承载着不同层次的信息：

1. **Level 1 - 字卡层**: 单个汉字 + emoji（视觉符号）
2. **Level 2 - 短语层**: 简单短语，基础语言组合规则
3. **Level 3 - 数学层**: 数学符号和语法结构（如 "S = NP + VP"）
4. **Level 4 - 拼音层**: 发音信息，音韵映射
5. **Level 5 - 英文层**: 英文翻译，跨语言映射
6. **Level 6 - 中文层**: 完整中文句子和段落
7. **Level 7 - 日文层**: 日文翻译
8. **Level 8 - 韩文层**: 韩文翻译

### 数据示例

```json
{
    "concept": "下雨",
    "level_1": {"字卡": "下雨", "emoji": "🌧️"},
    "level_2": {"短语": "下雨了"},
    "level_3": {"数学": "S = NP + VP (NP: 天气, VP: 下雨)"},
    "level_4": {"拼音": "xià yǔ"},
    "level_5": {"英文": "It's raining"},
    "level_6": {"中文": "今天天气阴沉，下雨了。"},
    "level_7": {"日文": "雨が降っています"},
    "level_8": {"韩文": "비가 오고 있어요"}
}
```

## 📦 项目文件结构

```
apt_model/
├── hlbd.py                           # HLBD命令行入口点 ⭐
├── 分层语言启蒙数据集.txt              # HLBD数据集文件 (1156行)
└── data/hlbd/
    ├── hlbd_adapter.py               # HLBD数据适配器 (713行)
    └── hlbd                          # 空标记文件
```

## 🚀 使用方法

### 1. 训练模式

使用HLBD数据集训练APT模型：

```bash
# 基础训练（20 epochs）
python -m apt_model.hlbd \
    --hlbd-path apt_model/分层语言启蒙数据集.txt \
    --output-dir apt_hlbd_model \
    --epochs 20

# 自定义训练参数
python -m apt_model.hlbd \
    --hlbd-path apt_model/分层语言启蒙数据集.txt \
    --output-dir apt_hlbd_model \
    --epochs 30 \
    --batch-size 16 \
    --lr 5e-5 \
    --max-length 512
```

### 2. 评估模式

评估已训练的HLBD模型：

```bash
# 评估模型（需要已有检查点）
python -m apt_model.hlbd \
    --hlbd-path apt_model/分层语言启蒙数据集.txt \
    --output-dir apt_hlbd_model \
    --evaluate-only

# 指定检查点文件
python -m apt_model.hlbd \
    --hlbd-path apt_model/分层语言启蒙数据集.txt \
    --output-dir apt_hlbd_model \
    --evaluate-only \
    --resume apt_hlbd_model/checkpoint_best.pt
```

### 3. 恢复训练

从检查点恢复训练：

```bash
python -m apt_model.hlbd \
    --hlbd-path apt_model/分层语言启蒙数据集.txt \
    --output-dir apt_hlbd_model \
    --epochs 20 \
    --resume apt_hlbd_model/checkpoint_epoch_10.pt
```

## 🎛️ 命令行参数

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--hlbd-path` | HLBD数据集文件路径 | `apt_model/分层语言启蒙数据集.txt` |
| `--output-dir` | 模型输出目录 | `apt_hlbd_model` |

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 20 | 训练轮数 |
| `--batch-size` | 8 | 批次大小 |
| `--lr, --learning-rate` | 3e-5 | 学习率 |
| `--max-length` | 512 | 最大序列长度 |

### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--d-model` | 768 | 模型维度 |
| `--num-heads` | 12 | 注意力头数 |
| `--num-layers` | 6 | 编码器/解码器层数 |

### 数据处理参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--include-multilingual` | True | 包含多语言文本 |
| `--include-separate-levels` | True | 包含各层级单独的文本 |

### 设备和其他参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--device` | auto | 计算设备 (auto/cuda/cpu) |
| `--seed` | 42 | 随机种子 |
| `--verbose` | False | 详细输出模式 |
| `--evaluate-only` | False | 仅评估模式 |
| `--resume` | None | 从检查点恢复 |

## 📊 评估指标

HLBD模型评估包括：

1. **语言对翻译能力**: 评估所有语言对之间的翻译质量
   - 中文 ↔ 英文
   - 中文 ↔ 日文
   - 中文 ↔ 韩文
   - 英文 ↔ 日文
   - 等所有组合

2. **概念完成能力**: 从概念生成完整多层级描述

3. **层级生成能力**: 从一个层级生成其他层级

评估结果保存在 `output-dir/evaluation_results.json`

## 🔧 核心组件

### HLBDDataProcessor

数据处理器，负责加载和转换HLBD数据：

```python
from apt_model.data.hlbd.hlbd_adapter import HLBDDataProcessor

# 加载数据
processor = HLBDDataProcessor(data_path="分层语言启蒙数据集.txt")

# 处理数据
processor.process_data(
    include_multilingual=True,
    include_separate_levels=True
)

# 获取训练文本
training_texts = processor.get_training_texts()
```

### HLBDDataset

PyTorch Dataset类：

```python
from apt_model.data.hlbd.hlbd_adapter import HLBDDataset

dataset = HLBDDataset(
    texts=training_texts,
    tokenizer=tokenizer,
    max_length=512
)
```

### HLBDModelEvaluator

模型评估器：

```python
from apt_model.data.hlbd.hlbd_adapter import HLBDModelEvaluator

evaluator = HLBDModelEvaluator(
    model=model,
    tokenizer=tokenizer,
    processor=processor
)

# 评估所有语言对
results = evaluator.evaluate_all_language_pairs(num_samples=3)

# 评估概念完成
concept_results = evaluator.evaluate_concept_completion(num_samples=5)
```

## 💡 设计理念

### 层级化结构

HLBD通过分层设计，使模型能够：
- 从最基础的视觉符号（emoji、字卡）开始学习
- 逐步理解语法结构（数学层）
- 掌握跨语言映射（拼音、英文、日文、韩文）
- 最终生成完整的自然语言表达

### 高密度信息编码

每个样本包含同一概念的8个层级表示，提供：
- 结构化的语法标注
- 逻辑运算形式
- 跨语言对应关系
- 多模态信息（文字 + emoji）

### 超形式数学支撑

借鉴超形式数学理论，将基础符号、语素、语法结构形式化、向量化，帮助模型快速内化语言构造规律。

## 🎯 应用场景

1. **预训练加速**: 快速让模型"学会说话"
2. **多语言融合**: 同时学习中英日韩多语言
3. **符号学习**: 建立符号运算法则和形式逻辑
4. **跨语言迁移**: 利用层级结构实现语言间映射

## 📈 训练流程

1. **数据加载**: 从 `分层语言启蒙数据集.txt` 加载1156行数据
2. **数据处理**: 提取8个层级的信息，生成训练样本
3. **分词器准备**: 基于HLBD数据创建专用tokenizer
4. **模型训练**: 使用APTModel在HLBD数据上训练
5. **评估**: 评估层级生成、语言翻译、概念完成能力
6. **保存**: 保存最佳模型检查点

## 🔍 与其他领域的关系

- **语言学**: 符合语素、词汇、短语、句法的基本划分
- **分析哲学**: 符号-意义映射的定量工具
- **符号学**: 符号系统构建，图形符号↔语素↔语言表达
- **NLP**: 高效的预训练数据集，提升文本生成和理解

## 📝 示例输出

训练日志示例：
```
==============================================================
开始HLBD模型训练
==============================================================
准备训练数据集...
训练批次数: 145
验证批次数: 36

==============================================================
Epoch 1/20
==============================================================
  Batch 10/145 - Loss: 4.2345, LR: 0.000028
  Batch 20/145 - Loss: 3.8912, LR: 0.000029
  ...
平均训练损失: 3.5234
平均验证损失: 3.4123
新的最佳模型！验证损失: 3.4123
```

评估结果示例：
```json
{
  "language_pairs": {
    "overall_avg_similarity": 0.85,
    "language_pairs": {
      "中文_to_英文": {"avg_similarity": 0.87},
      "英文_to_日文": {"avg_similarity": 0.82},
      ...
    }
  },
  "concept_completion": {
    "avg_similarity": 0.88
  }
}
```

## ⚠️ 注意事项

1. **数据集位置**: 确保 `分层语言启蒙数据集.txt` 在正确路径
2. **内存需求**: HLBD训练需要足够的GPU/CPU内存
3. **检查点**: 定期保存检查点，避免训练中断
4. **评估**: 训练完成后自动运行评估

## 🆕 新增文件

本次更新新增以下文件：

1. **apt_model/hlbd.py** (478行) - HLBD命令行入口
2. **apt_model/data/hlbd/hlbd_adapter.py** (713行) - 数据适配器
3. **apt_model/分层语言启蒙数据集.txt** (1156行) - 数据集
4. **apt_model/modeling/gpt5_model.py** - GPT-5模型实现
5. **apt_model/modeling/gpt4o_model.py** - GPT-4o模型实现
6. **apt_model/modeling/gpto3_model.py** - GPT-o3模型实现

## 🔗 相关文档

- [APT模型架构](./REFACTORING_SUMMARY.md)
- [配置与调度系统](./SCHEDULER_ANALYSIS.md)
- [数据处理管道](./REFACTORING_SUMMARY.md#part-3-数据处理管道)

---

**作者**: 430
**版本**: 1.0
**日期**: 2025
