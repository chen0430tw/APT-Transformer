# APT-Transformer 文档中心

欢迎来到 APT-Transformer 完整文档中心！本文档按照 APT 2.0 架构组织。

## 📚 快速导航

### 🎯 核心架构

- [APT 2.0 架构总览](ARCHITECTURE_2.0.md) - 完整的 APT 2.0 DDD 架构说明
- [仓库结构说明](guides/repo_schema.md) - 项目目录组织和设计原则
- [完整技术总结](guides/COMPLETE_TECH_SUMMARY.md) - 所有技术特性汇总
- [集成总结](guides/INTEGRATION_SUMMARY.md) - 各模块集成情况

### 🧠 内核 (Kernel)

**模型与架构**
- [APT Model 使用手册](kernel/APT_MODEL_HANDBOOK.md) - 核心模型使用指南
- [微调指南](kernel/FINE_TUNING_GUIDE.md) - 模型微调完整教程
- [DeepSeek 训练指南](kernel/DEEPSEEK_TRAINING_GUIDE.md) - DeepSeek 模型训练

**训练优化**
- [数据预处理指南](kernel/DATA_PREPROCESSING_GUIDE.md) - 数据处理流程
- [HLBD 数据集](kernel/HLBD.md) - 高质量训练数据集
- [调试模式指南](kernel/DEBUG_MODE_GUIDE.md) - 训练调试工具

**高级特性**
- [Context 与 RoPE 优化](kernel/CONTEXT_AND_ROPE_OPTIMIZATION.md) - 长文本处理
- [DBC/DAC 优化指南](kernel/DBC_DAC_OPTIMIZATION_GUIDE.md) - 高级优化技术
- [Left Spin Smooth 集成](kernel/LEFT_SPIN_SMOOTH_INTEGRATION.md) - 平滑技术
- [训练保护指南](kernel/training_protection_guide.md) - 训练监控和保护

### 🧩 记忆系统 (Memory)

**核心记忆技术**
- [记忆系统总览](memory/MEMORY_SYSTEM_GUIDE.md) - 完整记忆架构
- [AIM-Memory 指南](memory/AIM_MEMORY_GUIDE.md) - 惯性锚定镜像记忆
- [AIM-NC 指南](memory/AIM_NC_GUIDE.md) - N-gram/Trie 收编协议
- [层级记忆指南](memory/HIERARCHICAL_MEMORY_GUIDE.md) - 多层记忆管理

**知识图谱与检索**
- [知识图谱指南](memory/KNOWLEDGE_GRAPH_GUIDE.md) - KG 构建和使用
- [Graph Brain 训练](memory/GRAPH_BRAIN_TRAINING_GUIDE.md) - 图脑训练教程

### ⚡ 性能优化 (Performance)

**GPU 虚拟化**
- [Virtual Blackwell 完整指南](performance/VIRTUAL_BLACKWELL_COMPLETE_GUIDE.md) - VB 技术栈
- [Virtual Blackwell 完整文档](performance/VIRTUAL_BLACKWELL_COMPLETE.md) - VB 详细说明
- [启用 Virtual Blackwell](performance/ENABLE_VIRTUAL_BLACKWELL.md) - 快速启用指南
- [VGPU 快速开始](performance/VGPU_QUICK_START.md) - vGPU 入门
- [VGPU Stack 架构](performance/VGPU_STACK_ARCHITECTURE.md) - vGPU 技术架构

**训练后端**
- [训练后端使用指南](performance/TRAINING_BACKENDS.md) - 多后端支持
- [极限优化指南](performance/EXTREME_OPTIMIZATIONS_GUIDE.md) - 性能优化技巧
- [弹性 APT 集成](performance/ELASTIC_APT_INTEGRATION.md) - 弹性训练

**NPU 支持**
- [Cloud NPU 指南](performance/CLOUD_NPU_GUIDE.md) - 云端 NPU 使用
- [NPU 集成指南](performance/NPU_INTEGRATION_GUIDE.md) - NPU 后端集成

**GPU 优化**
- [GPU Flash 优化指南](performance/GPU_FLASH_OPTIMIZATION_GUIDE.txt) - Flash 优化
- [GPU Flash 成功分析](performance/GPU_FLASH_SUCCESS_ANALYSIS.md) - 优化案例

### 🚀 产品功能 (Product)

**模型支持**
- [Claude 4 模型指南](product/CLAUDE4_MODEL_GUIDE.md) - Claude 4 模型使用
- [GPT 模型指南](product/GPT_MODELS_GUIDE.md) - GPT 系列模型

**知识蒸馏**
- [知识蒸馏原理](product/DISTILLATION_PRINCIPLE.md) - 蒸馏技术原理
- [Teacher API 指南](product/TEACHER_API_GUIDE.md) - 教师模型 API
- [视觉蒸馏指南](product/VISUAL_DISTILLATION_GUIDE.md) - 视觉模型蒸馏

**API 与集成**
- [API Providers 指南](product/API_PROVIDERS_GUIDE.md) - 统一 API 接口
- [MCP 集成指南](product/MCP_INTEGRATION_GUIDE.md) - MCP 协议集成
- [Web Search 插件](product/WEB_SEARCH_PLUGIN_GUIDE.md) - 网络搜索功能

**强化学习**
- [RL 与预训练指南](product/RL_PRETRAINING_GUIDE.md) - 强化学习训练

**Agent 系统**
- [Agent 系统指南](product/AGENT_SYSTEM_GUIDE.md) - Agent 工具调用系统

**可视化与工具**
- [可视化指南](product/VISUALIZATION_GUIDE.md) - 训练可视化工具
- [启动器使用指南](product/LAUNCHER_README.md) - 训练启动器
- [Optuna 优化指南](product/OPTUNA_GUIDE.md) - 超参数优化
- [插件系统指南](product/PLUGIN_SYSTEM_GUIDE.md) - 插件开发

### 📖 专题指南 (Guides)

- [APX 打包规范](guides/APX.md) - 模型打包和分发
- [分发模式](guides/DISTRIBUTION_MODES.md) - 多种分发方式
- [插件 vs 模块原则](guides/PLUGIN_VS_MODULE_PRINCIPLES.md) - 架构设计原则

### 🧪 测试与基础设施

- [测试基础设施](TESTING_INFRASTRUCTURE.md) - 测试框架和工具
- [CLI 增强功能](CLI_ENHANCEMENTS.md) - 命令行增强
- [高级 CLI 命令](ADVANCED_CLI_COMMANDS.md) - 高级命令使用

### 📊 HLBD 数据集

- [HLBD 主文档](hlbd/README.md) - HLBD 数据集完整说明
- [HLBD V2 总结](hlbd/HLBD_V2_SUMMARY.md) - V2 版本更新
- [HLBD Hardcore 训练](hlbd/HLBD_HARDCORE_TRAINING.md) - 高难度训练
- [模块化训练实现](hlbd/HLBD_MODULAR_TRAINING.md) - 模块化训练
- [模块化训练快速开始](hlbd/MODULAR_TRAINING_QUICKSTART.md) - 快速入门

## 🗂️ APT 2.0 架构域

### 1. Model Domain (apt/model/)
模型定义层 - 包含各类架构、组件和扩展

**参考文档**: [apt/model/README.md](../apt/model/README.md)

### 2. TrainOps Domain (apt/trainops/)
训练操作层 - 训练引擎、分布式训练、数据加载

**参考文档**: [apt/trainops/README.md](../apt/trainops/README.md)

### 3. vGPU Domain (apt/vgpu/)
GPU虚拟化层 - Virtual Blackwell 技术栈

**参考文档**: [apt/vgpu/README.md](../apt/vgpu/README.md)

### 4. APX Domain (apt/apx/)
模型打包层 - 打包、分发和部署

**参考文档**: [apt/apx/README.md](../apt/apx/README.md)

## 📦 其他资源

### 示例代码
- [examples/](../examples/) - 各类使用示例
- [examples/USAGE_GUIDE.md](../examples/USAGE_GUIDE.md) - 示例使用指南
- [examples/STARTUP_EXAMPLES.md](../examples/STARTUP_EXAMPLES.md) - 启动示例

### 工具和脚本
- [scripts/](../scripts/) - 实用脚本集合
- [scripts/README.md](../scripts/README.md) - 脚本使用说明

### 测试
- [tests/](../tests/) - 测试套件
- [tests/README.md](../tests/README.md) - 测试说明

## 🔄 迁移指南

从旧版本迁移？查看：
- [仓库结构说明](guides/repo_schema.md) - 包含 1.x → 2.0 迁移指南
- [向后兼容层文档](../apt/compat/) - compat 层使用说明

## 📝 贡献

- [CONTRIBUTING.md](../CONTRIBUTING.md) - 贡献指南
- [INSTALLATION.md](../INSTALLATION.md) - 安装说明

## ⚖️ 许可证

- [LICENSE](../LICENSE) - MIT License

---

## 🆘 需要帮助？

1. 查看 [README.md](../README.md) - 项目主页
2. 查看相关领域的文档
3. 查看示例代码：[examples/](../examples/)
4. 提交 Issue 到 GitHub 仓库

**文档最后更新**: 2026-01-23
