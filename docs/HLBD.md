# HLBD - 分层语言启蒙数据集 (Hierarchical Language Bootstrapping Dataset)

## 📖 概述

HLBD（分层语言启蒙数据集）是专门为APT模型设计的多层级、多语言学习数据集。通过将概念分解为**8个不同的抽象层级**（从emoji符号到完整的自然语言描述），HLBD帮助模型建立对语言的深层理解和跨语言映射能力。

## 🎯 核心理念

HLBD的设计灵感来自人类的语言学习过程：
1. **从简单到复杂**：从符号（字卡/emoji）逐步过渡到完整句子
2. **多模态联系**：建立符号、拼音、数学结构、多语言文本之间的映射
3. **概念导向**：每个训练样本围绕一个核心概念展开
4. **分层抽象**：模型学习在不同抽象层级间转换

## 📊 数据结构

### 8个标准层级

每个HLBD样本包含一个核心 **概念**（concept）和8个表达层级：

| 层级 | 名称 | 说明 | 示例 |
|------|------|------|------|
| **level_1** | 字卡/Emoji | 最简符号表示 | `🌧️` |
| **level_2** | 短语 | 简短词组 | `下雨` |
| **level_3** | 句法结构 | 数学/逻辑表达式 | `weather(rain, heavy)` |
| **level_4** | 拼音 | 中文拼音注音 | `xià yǔ le` |
| **level_5** | 英文 | English表达 | `It's raining` |
| **level_6** | 中文 | 完整中文描述 | `天空正在下雨` |
| **level_7** | 日文 | 日语表达 | `雨が降っています` |
| **level_8** | 韩文 | 韩语表达 | `비가 내리고 있어요` |

### 数据格式

```json
{
  "concept": "下雨",
  "level_1": {
    "字卡": "雨",
    "emoji": "🌧️"
  },
  "level_2": {
    "短语": "下雨"
  },
  "level_3": {
    "数学": "weather(rain, present_continuous)"
  },
  "level_4": {
    "拼音": "xià yǔ le"
  },
  "level_5": {
    "英文": "It's raining. The weather is wet and cloudy."
  },
  "level_6": {
    "中文": "天空正在下雨，地面逐渐变得湿润。"
  },
  "level_7": {
    "日文": "雨が降っています。天気が悪いです。"
  },
  "level_8": {
    "韩文": "비가 내리고 있어요. 날씨가 나쁩니다."
  }
}
```

## 🔧 使用方法

### 1. 快速开始：训练HLBD模型

```bash
# 基础训练（20个epoch）
python -m apt_model.data.hlbd.hlbd \
  --hlbd-path apt_model/分层语言启蒙数据集.txt \
  --output-dir apt_hlbd_model \
  --epochs 20

# 使用GPU加速
python -m apt_model.data.hlbd.hlbd \
  --hlbd-path apt_model/分层语言启蒙数据集.txt \
  --output-dir apt_hlbd_model \
  --epochs 50 \
  --device cuda \
  --batch-size 16

# 自定义模型配置
python -m apt_model.data.hlbd.hlbd \
  --hlbd-path apt_model/分层语言启蒙数据集.txt \
  --output-dir apt_hlbd_model \
  --epochs 20 \
  --d-model 1024 \
  --num-heads 16 \
  --num-layers 12 \
  --max-length 1024
```

### 2. 评估已训练模型

```bash
# 仅评估模式
python -m apt_model.data.hlbd.hlbd \
  --hlbd-path apt_model/分层语言启蒙数据集.txt \
  --output-dir apt_hlbd_model \
  --evaluate-only

# 指定检查点评估
python -m apt_model.data.hlbd.hlbd \
  --hlbd-path apt_model/分层语言启蒙数据集.txt \
  --output-dir apt_hlbd_model \
  --evaluate-only \
  --resume apt_hlbd_model/checkpoint_best.pt
```

### 3. 程序化使用

```python
from apt_model.data.hlbd.hlbd_adapter import (
    HLBDDataProcessor,
    HLBDDataset,
    prepare_hlbd_tokenizer,
    create_hlbd_apt_config
)

# 1. 加载和处理数据
processor = HLBDDataProcessor(data_path="apt_model/分层语言启蒙数据集.txt")
processor.process_data(
    include_multilingual=True,      # 包含多语言层级
    include_separate_levels=True    # 包含单独层级样本
)

# 2. 获取训练文本
training_texts = processor.get_training_texts()
print(f"训练样本数: {len(training_texts)}")

# 3. 准备分词器（自动选择最佳多语言分词器）
tokenizer, detected_language = prepare_hlbd_tokenizer(
    hlbd_samples_or_path=processor.raw_samples,
    vocab_size=50000
)

# 4. 创建APT模型配置
config = create_hlbd_apt_config(vocab_size=tokenizer.vocab_size)

# 5. 创建数据集
from torch.utils.data import DataLoader
dataset = HLBDDataset(training_texts, tokenizer, max_length=512)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
```

## 🎓 训练效果

HLBD训练能让模型学会：

### ✅ 跨层级理解
- **符号 → 语言**：`🌧️` → `It's raining`
- **拼音 → 中文**：`xià yǔ le` → `天空正在下雨`
- **结构 → 自然语言**：`weather(rain)` → `天气正在下雨`

### ✅ 多语言翻译
- **英文 → 中文**：`I love you` → `我爱你`
- **中文 → 日文**：`我爱你` → `愛しています`
- **中文 → 韩文**：`我爱你` → `사랑해요`

### ✅ 概念推理
- **从概念生成**：`概念: 快乐` → `happiness, joy, cheerfulness`
- **概念完成**：`安柏是` → `安柏是蒙德城的侦察骑士，擅长弓箭和侦察`

## 🔬 快速实验：HLBD测试脚本

项目提供了一个快速测试脚本，展示HLBD的强大能力：

```bash
# 运行HLBD快速学习测试（500 epochs）
python tests/test_hlbd_quick_learning.py
```

该测试展示：
- ✅ 使用20个HLBD概念样本
- ✅ 创建80+个训练对（emoji→中文、拼音→中文、英文→中文等）
- ✅ 500个epoch的强化训练
- ✅ 实时显示模型的生成进度

**预期输出示例**：
```
输入: 🌧️
期望概念: 下雨
生成: 天空正在下雨，地面湿润

输入: ❤️
期望概念: 我爱你
生成: 我爱你，亲爱的

输入: I love you
期望概念: 我爱你
生成: 我爱你，我非常爱你
```

## 🌐 扩展到更多语言

HLBD支持动态添加新语言层级（level_9, level_10...）：

```python
# 添加法语和德语层级
extra_languages = {
    "level_9": "法语",
    "level_10": "德语"
}

processor = HLBDDataProcessor(
    data_path="分层语言启蒙数据集.txt",
    extra_languages=extra_languages
)

# 扩展数据格式
sample = {
    "concept": "下雨",
    "level_1": {"字卡": "雨", "emoji": "🌧️"},
    "level_6": {"中文": "天空正在下雨"},
    "level_9": {"法语": "Il pleut"},
    "level_10": {"德语": "Es regnet"}
}
```

## 📈 评估指标

HLBD评估器提供多维度评估：

```python
from apt_model.data.hlbd.hlbd_adapter import HLBDModelEvaluator

evaluator = HLBDModelEvaluator(
    model=trained_model,
    tokenizer=tokenizer,
    processor=processor
)

# 1. 评估所有语言对翻译能力
results = evaluator.evaluate_all_language_pairs(num_samples=5)
print(f"总体平均相似度: {results['overall_avg_similarity']:.4f}")

# 2. 评估特定语言对
en_to_zh = evaluator.evaluate_language_generation(
    source_lang="英文",
    target_lang="中文",
    num_samples=10
)

# 3. 评估概念完成能力
concept_results = evaluator.evaluate_concept_completion(num_samples=5)
```

## 📁 文件位置

- **数据集文件**: `apt_model/分层语言启蒙数据集.txt`
- **适配器模块**: `apt_model/data/hlbd/hlbd_adapter.py`
- **训练脚本**: `apt_model/data/hlbd/hlbd.py`
- **快速测试**: `tests/test_hlbd_quick_learning.py`

## ⚙️ 命令行参数完整列表

### 训练参数
```bash
--hlbd-path PATH          # HLBD数据集文件路径（必需）
--output-dir DIR          # 模型输出目录（必需）
--epochs N                # 训练轮数（默认：20）
--batch-size N            # 批次大小（默认：8）
--lr FLOAT                # 学习率（默认：3e-5）
--max-length N            # 最大序列长度（默认：512）
--warmup-steps N          # 预热步数（默认：1000）
--gradient-clip FLOAT     # 梯度裁剪阈值（默认：1.0）
```

### 模型参数
```bash
--d-model N               # 模型维度（默认：768）
--num-heads N             # 注意力头数（默认：12）
--num-layers N            # 层数（默认：6）
```

### 数据参数
```bash
--include-multilingual    # 包含多语言文本（默认：True）
--include-separate-levels # 包含单独层级样本（默认：True）
```

### 其他参数
```bash
--device {auto,cuda,cpu}  # 计算设备（默认：auto）
--evaluate-only           # 仅评估模式
--resume PATH             # 从检查点恢复
--monitor-resources       # 启用资源监控
--monitor-interval N      # 监控间隔（秒）
--log-file PATH           # 日志文件路径
--seed N                  # 随机种子（默认：42）
--verbose                 # 详细输出模式
```

## 🎯 使用场景

### 1. 多语言机器翻译
HLBD提供了丰富的平行语料，适合训练多语言翻译模型。

### 2. 概念学习研究
通过分层结构研究模型如何理解和表示抽象概念。

### 3. 跨模态理解
研究emoji、符号与自然语言之间的映射关系。

### 4. 语言启蒙教学
模仿人类从简单符号到复杂语言的学习路径。

### 5. 低资源语言训练
通过跨语言对齐，帮助模型学习资源较少的语言。

## 🔍 技术特点

- ✅ **自动分词器选择**：根据数据自动选择最佳多语言分词器
- ✅ **内存优化**：支持大规模数据集的高效处理
- ✅ **灵活扩展**：轻松添加新语言层级
- ✅ **完整评估**：提供多维度模型性能评估
- ✅ **兼容APT架构**：无缝集成到APT-Transformer框架

## 🚀 性能建议

### 训练建议
- **小数据集**（<50概念）：20-50 epochs，batch_size=4
- **中等数据集**（50-200概念）：50-100 epochs，batch_size=8
- **大数据集**（>200概念）：100-500 epochs，batch_size=16

### 硬件建议
- **CPU训练**：batch_size≤4，d_model≤512
- **单GPU**（8GB）：batch_size≤8，d_model≤768
- **单GPU**（16GB+）：batch_size≤16，d_model≤1024

## 🆘 常见问题

### Q1: 训练时显存不足怎么办？
**A**: 减小`--batch-size`、`--max-length`或`--d-model`参数。

### Q2: 如何添加自定义语言层级？
**A**: 使用`extra_languages`参数：
```python
extra_languages = {"level_9": "法语", "level_10": "德语"}
processor = HLBDDataProcessor(data_path=path, extra_languages=extra_languages)
```

### Q3: 模型生成质量不佳？
**A**: 尝试增加训练轮数（epochs）、调整学习率，或使用更大的模型配置。

### Q4: 如何只训练特定语言对？
**A**: 在数据处理时过滤：
```python
# 只保留中英文对
filtered_texts = [t for t in training_texts if "英文:" in t or "中文:" in t]
```

## 📚 参考资料

- **APT模型论文**: 查看项目根目录的研究论文
- **分词器集成**: `apt_model/modeling/chinese_tokenizer_integration.py`
- **训练优化器**: `apt_model/training/optimizer.py`
- **检查点管理**: `apt_model/training/checkpoint.py`

---

**贡献者**: APT-Transformer团队
**最后更新**: 2025-12-04
**许可**: 与APT-Transformer项目相同
