# APT-Transformer 文档中心

欢迎使用APT-Transformer文档中心！这里汇集了所有APT-Transformer的完整文档。

---

## 📚 文档导航

### 🚀 快速开始

1. **[项目主文档](../README.md)** - 项目概览和快速开始
2. **[APT模型手册](../kernel/APT_MODEL_HANDBOOK.md)** - 完整的模型使用手册
3. **[启动器使用指南](../product/LAUNCHER_README.md)** - GUI启动器使用说明

---

## 📖 核心功能文档

### 🎓 知识蒸馏与迁移学习

- **[知识蒸馏原理](../product/DISTILLATION_PRINCIPLE.md)**
  - 知识蒸馏的理论基础
  - 温度调节和软标签
  - 损失函数设计

- **[Teacher API指南](../product/TEACHER_API_GUIDE.md)**
  - 外部API作为教师模型
  - 支持OpenAI、Anthropic、SiliconFlow等
  - 成本追踪和管理

- **[视觉蒸馏指南](../product/VISUAL_DISTILLATION_GUIDE.md)**
  - 多模态知识蒸馏
  - 图像-文本对齐学习
  - Vision Transformer集成

### 🔬 强化学习与预训练

- **[RL与预训练完整指南](../product/RL_PRETRAINING_GUIDE.md)**
  - 奖励模型训练
  - RLHF (PPO-based)
  - DPO (Direct Preference Optimization)
  - GRPO (Group Relative Policy Optimization)
  - 对比学习 (SimCLR/MoCo)
  - MLM预训练 (BERT/RoBERTa)
  - 最佳实践和超参数调优

### 🧠 知识图谱与RAG

- **[知识图谱使用指南](../memory/KNOWLEDGE_GRAPH_GUIDE.md)**
  - 知识图谱构建
  - GraphRAG集成
  - Hodge-Laplacian光谱分析
  - Graph Brain动态建模

- **[GraphRAG模块文档](../apt_model/core/graph_rag/)**
  - [快速开始](../../apt/core/graph_rag/START_HERE.md)
  - [项目结构](../../apt/core/graph_rag/PROJECT_STRUCTURE.md)
  - [集成指南](../../apt/core/graph_rag/INTEGRATION.md)
  - [完整文档](../../README.md)

### 🎯 微调与优化

- **[微调指南](../kernel/FINE_TUNING_GUIDE.md)**
  - LoRA微调
  - 全参数微调
  - 参数高效微调方法

- **[Optuna超参数优化](../product/OPTUNA_GUIDE.md)**
  - 自动超参数搜索
  - 多目标优化
  - 贝叶斯优化策略

### 🚀 训练后端系统

- **[训练后端使用指南](../performance/TRAINING_BACKENDS.md)** ⭐ 新增
  - Playground训练（HLBD专用，Cosine重启学习率）
  - DeepSpeed分布式训练（ZeRO-2/3优化，支持超大模型）
  - Azure ML云端训练（MLflow跟踪，自动超参数调优）
  - HuggingFace Trainer（W&B、TensorBoard、Hub集成）
  - 统一训练启动器（`train.py`一键切换后端）

### 📊 可视化与监控

- **[训练可视化指南](../product/VISUALIZATION_GUIDE.md)** ⭐ 新增
  - 科幻风格Loss地形图
  - 实时训练监控（每2秒刷新）
  - 多训练同时监控
  - 6种子图（3D地形、梯度流、LR曲线等）

### 🔧 训练监控与优化

- **[SOSA训练监控](../../README.md)**
  - 实时训练监控
  - 异常检测与自动修复
  - Spark Seed自组织算法

- **[训练监控快速开始](../apt_model/core/training/QUICK_START.md)**
  - 一行集成
  - 零侵入设计
  - 低于1%的性能开销

---

## 🔌 API与配置

- **[API Provider统一接口](../product/API_PROVIDERS_GUIDE.md)**
  - OpenAI API
  - Anthropic API
  - SiliconFlow API
  - 自定义API
  - 成本追踪

---

## 🏗️ 架构与集成

- **[插件系统完整指南](../product/PLUGIN_SYSTEM_GUIDE.md)** ⭐ 新增
  - 插件系统架构与设计原理
  - 26+ 生产级插件详解
  - 自定义插件开发教程
  - 事件驱动机制与优先级管理
  - 故障排查与最佳实践

- **[插件开发指南](../../apt/apps/cli/PLUGIN_GUIDE.md)**
  - CLI插件快速开发
  - 命令注册与执行

### ⚡ 性能优化

- **[DBC-DAC优化完整指南](../kernel/DBC_DAC_OPTIMIZATION_GUIDE.md)** ⭐ 新增
  - DBC-DAC方法对比与误差分析
  - 二次优化详解（20-500x加速）
  - 训练加速方案与实战
  - 智能化与自适应优化

### 🤖 GPT模型系列

- **[GPT模型完整指南](../product/GPT_MODELS_GUIDE.md)** ⭐ 新增
  - GPT-4o / GPT-5 / GPTo3 架构分析
  - 可训练性评估与代码审查
  - 训练配置与高级功能
  - 故障排除与最佳实践

---

## 💻 代码示例

### RL示例
- [DPO训练示例](../../examples/rl_examples/dpo_example.py)
- [GRPO训练示例](../../examples/rl_examples/grpo_example.py)

### 预训练示例
- [对比学习示例](../../examples/pretraining_examples/contrastive_example.py)
- [MLM预训练示例](../../examples/pretraining_examples/mlm_example.py)

### GraphRAG示例
- [基本使用示例](../../examples/graph_rag_examples/basic_usage.py)

### 训练监控示例
- [基本监控示例](../../examples/training_monitor_examples/basic_monitoring.py)

### 视觉蒸馏示例
- [视觉蒸馏完整示例](../../examples/visual_distillation_example.py)

---

## 🗂️ 文档分类

### 按主题分类

| 主题 | 文档 |
|------|------|
| **知识蒸馏** | [原理](../product/DISTILLATION_PRINCIPLE.md), [Teacher API](../product/TEACHER_API_GUIDE.md), [视觉蒸馏](../product/VISUAL_DISTILLATION_GUIDE.md) |
| **强化学习** | [RL完整指南](../product/RL_PRETRAINING_GUIDE.md) |
| **知识图谱** | [KG指南](../memory/KNOWLEDGE_GRAPH_GUIDE.md), [GraphRAG文档](../apt_model/core/graph_rag/) |
| **微调优化** | [微调指南](../kernel/FINE_TUNING_GUIDE.md), [Optuna优化](../product/OPTUNA_GUIDE.md), [DBC-DAC优化](../kernel/DBC_DAC_OPTIMIZATION_GUIDE.md) |
| **训练监控** | [SOSA文档](../../README.md), [快速开始](../apt_model/core/training/QUICK_START.md) |
| **工具与配置** | [API Provider](../product/API_PROVIDERS_GUIDE.md), [启动器](../product/LAUNCHER_README.md) |
| **架构设计** | [插件系统指南](../product/PLUGIN_SYSTEM_GUIDE.md), [CLI插件指南](../../apt/apps/cli/PLUGIN_GUIDE.md) |
| **GPT模型** | [GPT模型指南](../product/GPT_MODELS_GUIDE.md) |

### 按难度分类

#### 🟢 初级 (入门必读)
1. [项目主文档](../README.md)
2. [启动器使用指南](../product/LAUNCHER_README.md)
3. [APT模型手册](../kernel/APT_MODEL_HANDBOOK.md)
4. [微调指南](../kernel/FINE_TUNING_GUIDE.md)

#### 🟡 中级 (进阶使用)
1. [知识蒸馏原理](../product/DISTILLATION_PRINCIPLE.md)
2. [Teacher API指南](../product/TEACHER_API_GUIDE.md)
3. [知识图谱指南](../memory/KNOWLEDGE_GRAPH_GUIDE.md)
4. [Optuna优化](../product/OPTUNA_GUIDE.md)
5. [训练监控快速开始](../apt_model/core/training/QUICK_START.md)

#### 🔴 高级 (深度定制)
1. [RL与预训练完整指南](../product/RL_PRETRAINING_GUIDE.md)
2. [视觉蒸馏指南](../product/VISUAL_DISTILLATION_GUIDE.md)
3. [GraphRAG集成指南](../../apt/core/graph_rag/INTEGRATION.md)
4. [插件系统完整指南](../product/PLUGIN_SYSTEM_GUIDE.md)
5. [DBC-DAC优化完整指南](../kernel/DBC_DAC_OPTIMIZATION_GUIDE.md)
6. [GPT模型完整指南](../product/GPT_MODELS_GUIDE.md)

---

## 🎯 使用场景导航

### 我想训练一个小模型
1. 阅读 [微调指南](../kernel/FINE_TUNING_GUIDE.md)
2. 使用 [启动器](../product/LAUNCHER_README.md) 快速开始
3. 参考 [APT模型手册](../kernel/APT_MODEL_HANDBOOK.md)

### 我想使用大模型API做知识蒸馏
1. 阅读 [知识蒸馏原理](../product/DISTILLATION_PRINCIPLE.md)
2. 配置 [Teacher API](../product/TEACHER_API_GUIDE.md)
3. 查看 [API Provider指南](../product/API_PROVIDERS_GUIDE.md)

### 我想做强化学习训练
1. 阅读 [RL完整指南](../product/RL_PRETRAINING_GUIDE.md)
2. 选择算法: RLHF / DPO / GRPO
3. 运行示例: [DPO示例](../../examples/rl_examples/dpo_example.py)

### 我想构建知识图谱系统
1. 阅读 [知识图谱指南](../memory/KNOWLEDGE_GRAPH_GUIDE.md)
2. 参考 [GraphRAG快速开始](../../apt/core/graph_rag/START_HERE.md)
3. 运行 [GraphRAG示例](../../examples/graph_rag_examples/basic_usage.py)

### 我想监控训练过程
1. 阅读 [SOSA快速开始](../apt_model/core/training/QUICK_START.md)
2. 参考 [完整文档](../../README.md)
3. 运行 [监控示例](../../examples/training_monitor_examples/basic_monitoring.py)

### 我想优化超参数
1. 阅读 [Optuna指南](../product/OPTUNA_GUIDE.md)
2. 运行优化脚本: `run_optuna_optimization.sh`

### 我想开发自定义插件
1. 阅读 [插件系统完整指南](../product/PLUGIN_SYSTEM_GUIDE.md)
2. 参考26+生产级插件示例
3. 查看 [CLI插件开发](../../apt/apps/cli/PLUGIN_GUIDE.md)

### 我想加速训练性能
1. 阅读 [DBC-DAC优化指南](../kernel/DBC_DAC_OPTIMIZATION_GUIDE.md)
2. 了解20-500x加速方案
3. 应用智能化与自适应优化

### 我想使用GPT模型
1. 阅读 [GPT模型完整指南](../product/GPT_MODELS_GUIDE.md)
2. 选择模型: GPT-4o / GPT-5 / GPTo3
3. 参考训练配置与最佳实践

---

## 📝 文档更新日志

### 2024-12-22
- ✅ **文档整理与合并** - 大幅简化文档结构
  - 合并DBC-DAC相关文档 (4个→1个)
  - 合并Plugin系统文档 (2个→1个)
  - 合并GPT模型文档 (2个→1个)
  - 归档历史报告和计划文档
  - 减少35%文档文件数量
- ✅ **新增合并文档**
  - [插件系统完整指南](../product/PLUGIN_SYSTEM_GUIDE.md) - 架构+使用+开发
  - [DBC-DAC优化完整指南](../kernel/DBC_DAC_OPTIMIZATION_GUIDE.md) - 原理+方案+加速
  - [GPT模型完整指南](../product/GPT_MODELS_GUIDE.md) - 分析+训练+实践
- ✅ **文档中心优化**
  - 更新所有文档导航链接
  - 新增4个使用场景
  - 重组文档分类结构

### 2025-12-02
- ✅ 创建文档中心
- ✅ 添加RL与预训练完整指南
- ✅ 添加自监督学习能力检查报告
- ✅ 整合所有文档到docs目录
- ✅ 创建文档导航和索引

### 之前的更新
- ✅ 添加GraphRAG模块文档
- ✅ 添加SOSA训练监控文档
- ✅ 添加API Provider统一接口文档
- ✅ 添加知识蒸馏系列文档
- ✅ 添加微调和优化指南

---

## 🤝 贡献

如果您发现文档有错误或需要改进，欢迎提交Issue或Pull Request！

---

## 📧 联系方式

- 作者: chen0430tw
- 项目仓库: APT-Transformer

---

**提示**: 所有文档都在持续更新中，建议定期查看最新版本。
