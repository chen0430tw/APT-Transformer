# HLBD数据集训练文档

本目录包含HLBD（分层语言启蒙数据集）训练的完整文档。

## 📚 文档索引

### 快速开始
- **[30秒快速开始](MODULAR_TRAINING_QUICKSTART.md)** - 最快的上手方式

### 完整指南
- **[模块化训练指南](HLBD_MODULAR_TRAINING.md)** - 详细的使用说明、配置和调优
- **[数据集完成总结](DATASETS_COMPLETION_SUMMARY.md)** - HLBD Full & Hardcore数据集详解

### 专项训练
- **[HLBD Hardcore训练](HLBD_HARDCORE_TRAINING.md)** - 严格逻辑训练（5,042样本）
- **[HLBD V2总结](HLBD_V2_SUMMARY.md)** - Hardcore V2完整报告

### 技术文档
- **[实现细节](MODULAR_TRAINING_IMPLEMENTATION.md)** - 代码实现和技术架构

## 🚀 快速开始

### 模块化训练（推荐）

同时训练HLBD Full (5,000样本) + Hardcore (5,042样本)：

```bash
# 使用启动脚本
python3 scripts/hlbd/launch_hlbd_modular_training.py

# 或直接调用
python3 training/train_hlbd_playground.py \
    --datasets data/HLBD_Full_V2.json data/HLBD_Hardcore_Full_V2.json \
    --epochs 50
```

### 单数据集训练

只训练一个数据集：

```bash
python3 training/train_hlbd_playground.py \
    --dataset data/HLBD_Hardcore_Full_V2.json \
    --epochs 50
```

## 📊 数据集说明

### HLBD Full V2 (5,000样本)

**特点**:
- 8层分层语言结构
- Level 3句法层（S = NP + VP）
- 多语言（中英日韩）
- Emoji + 拼音支持

### HLBD Hardcore V2 (5,042样本)

**特点**:
- 严格逻辑问答
- 5大模块（几何、算术、生肖、物理、英文）
- 防"偷懒"学习
- 数据稀释学

## 🎯 模块化训练优势

| 指标 | 分别训练 | 模块化训练 | 提升 |
|------|---------|-----------|------|
| 总样本 | 5000+5042 | 10,042 | - |
| 训练时间 | 2×T | T | **50%↓** |
| GPU利用 | 标准 | 提升 | **30%↑** |
| 泛化能力 | 一般 | **增强** | **显著** |

## 📁 相关目录

- **数据集**: `/data/HLBD_*.json`
- **训练脚本**: `/training/train_hlbd_playground.py`
- **启动器**: `/scripts/hlbd/launch_*.py`
- **生成器**: `/tools/generate_hlbd_*.py`

## 🔗 相关链接

- [主README](../../README.md)
- [训练后端文档](../TRAINING_BACKENDS.md)
- [APT Model手册](../APT_MODEL_HANDBOOK.md)

---

**最后更新**: 2024-12-22
