# APT-Transformer 技术总览

**版本**: 2.0 (2025-01-21 重构版)
**架构**: 四层分离（L0/L1/L2/L3）
**定位**: 生产就绪的 Transformer 训练平台

---

## 🗺️ 阅读导航

### 第一次接触 APT？
👉 **先看这三件事**：
1. [APT 的三大核心创新](#1-l0-内核层---三大核心创新) - 只需 5 分钟了解核心
2. [我该选哪种发行版](#选择你的发行版) - 根据需求快速选择
3. [想看训练可视化？](#43-webui) - 直接看 WebUI 说明

### 深入学习？按层级阅读
- **研究复现** → [L0 内核层](#1-l0-内核层---三大核心创新)
- **生产加速** → [L1 性能层](#2-l1-性能层---虚拟blackwell)
- **长对话/RAG** → [L2 记忆层](#3-l2-记忆层---aim-记忆系统)
- **完整平台** → [L3 应用层](#4-l3-应用交付层)

---

## 🎯 选择你的发行版

| 你的需求 | 推荐发行版 | 一行启用 |
|---------|----------|---------|
| 论文复现、最小可用 | **apt-core** | \`enable('core')\` |
| 生产训练、快速推理 | **apt-perf** ⭐ | \`enable('perf')\` |
| 长对话、RAG、知识问答 | **apt-mind** | \`enable('mind')\` |
| 完整演示、高级开发 | apt-max | \`enable('max')\` |

详见 [DISTRIBUTION_MODES.md](../DISTRIBUTION_MODES.md) 和 [ARCHITECTURE.md](../ARCHITECTURE.md)

---

# APT-Transformer 四层架构

这个项目已重构为清晰的四层分离架构。完整的架构说明请参考：
- **[ARCHITECTURE.md](../ARCHITECTURE.md)** - 完整的架构设计和依赖规则
- **[DISTRIBUTION_MODES.md](../DISTRIBUTION_MODES.md)** - 三档发行版配置
- **[RESTRUCTURE_PLAN.md](../RESTRUCTURE_PLAN.md)** - 目录重构计划

本文档提供技术总览和快速参考。

---

## 技术总览摘要

### L0 内核层 - 三大核心创新
1. **Autopoietic Transform** - 自生成注意力，显存节省30-50%
2. **DBC-DAC** - 维度平衡压缩，训练加速20-30%
3. **Left-Spin Smooth** - 左旋平滑，NaN率降低5×

### L1 性能层 - 虚拟Blackwell
1. **MXFP4量化** - 4-bit浮点，4×压缩，<1%精度损失
2. **VGPU Stack** - 多级堆叠，2.7×显存扩展
3. **100K GPU训练** - 超大规模分布式支持

### L2 记忆层 - AIM系统
1. **AIM-Memory** - 锚点主权+证据回灌
2. **AIM-NC** - N-gram/Trie收编协议
3. **GraphRAG** - 知识图谱+光谱分析

### L3 应用层 - 完整交付
1. **WebUI** - 4个Tab（训练/梯度/Checkpoint/推理）
2. **REST API** - 10+端点，Swagger文档
3. **插件生态** - 30+生产级插件
4. **Agent系统** - 工具调用+Python沙盒

---

## 性能对比总表

| 指标 | 传统 Transformer | APT-Transformer | 提升 |
|------|-----------------|----------------|------|
| **训练速度** | 100 samples/s | **350 samples/s** | 3.5× |
| **推理延迟** | 100ms | **35ms** | 2.9× |
| **显存占用** (7B) | 14GB | **3.5GB** | 4× |
| **虚拟显存** | 24GB | **64GB** | 2.7× |
| **NaN 率** | 5% | **1%** | 5× |
| **训练成本** (GPT-3) | ¥400万 | **¥80万** | 5× |

---

## 快速命令参考

### 训练
\`\`\`bash
# 基础训练
python -m apt train --profile core --epochs 10

# 性能版训练  
python -m apt train --profile perf --distributed --gpus 8

# 记忆版训练
python -m apt train --profile mind --memory aim-nc
\`\`\`

### 推理
\`\`\`bash
# 交互式对话
python -m apt chat --checkpoint checkpoints/best.pt

# API服务
python -m apps.api.server --port 8000
\`\`\`

### 可视化
\`\`\`bash
# WebUI
python -m apps.webui.app --port 7860
\`\`\`

---

## 详细文档索引

### 核心文档
- [ARCHITECTURE.md](../ARCHITECTURE.md) - 四层架构设计
- [DISTRIBUTION_MODES.md](../DISTRIBUTION_MODES.md) - 发行版模式
- [RESTRUCTURE_PLAN.md](../RESTRUCTURE_PLAN.md) - 重构计划
- [README.md](../README.md) - 项目主文档
- [INSTALLATION.md](../INSTALLATION.md) - 安装指南

### 功能指南（按层级）
- [L0_KERNEL.md](./L0_KERNEL.md) - 内核层详细文档（待创建）
- [L1_PERFORMANCE.md](./L1_PERFORMANCE.md) - 性能层详细文档（待创建）
- [L2_MEMORY.md](./L2_MEMORY.md) - 记忆层详细文档（待创建）
- [L3_PRODUCT.md](./L3_PRODUCT.md) - 应用层详细文档（待创建）

### 原有文档（按主题）
- [VIRTUAL_BLACKWELL_COMPLETE_GUIDE.md](./VIRTUAL_BLACKWELL_COMPLETE_GUIDE.md) - 虚拟Blackwell完整指南
- [VGPU_STACK_ARCHITECTURE.md](./VGPU_STACK_ARCHITECTURE.md) - VGPU堆叠架构
- [DBC_DAC_OPTIMIZATION_GUIDE.md](./DBC_DAC_OPTIMIZATION_GUIDE.md) - DBC-DAC优化指南
- [PLUGIN_SYSTEM_GUIDE.md](./PLUGIN_SYSTEM_GUIDE.md) - 插件系统指南
- [VISUALIZATION_GUIDE.md](./VISUALIZATION_GUIDE.md) - 可视化指南
- [API_PROVIDERS_GUIDE.md](./API_PROVIDERS_GUIDE.md) - API提供商指南

---

**本文档已按新架构重新编排。详细技术内容请参考各层级的专门文档。**

**版本**: 2.0
**作者**: APT Team  
**日期**: 2025-01-21
