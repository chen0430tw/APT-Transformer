# APT-Transformer CLI 增强功能

**Version**: 1.0
**Last Updated**: 2026-01-22
**Status**: ✅ Implemented

---

## 📋 概述

APT-Transformer CLI 已经增强了三个主要功能，使其更加灵活和强大：

1. **Profile 配置加载** - 快速加载预定义配置
2. **命令管道** - 链式执行多个命令
3. **模块化选择** - 动态启用/禁用模块

---

## 🎯 功能详解

### 1. Profile 配置加载

#### 什么是 Profile？

Profile 是预定义的配置文件，包含了一组优化的参数设置，适用于不同的使用场景。

#### 可用的 Profiles

| Profile | 描述 | 适用场景 |
|---------|------|----------|
| `lite` | 轻量级配置 - 最小资源占用 | 开发、调试、快速测试 |
| `standard` | 标准配置 - 平衡性能和资源 | 日常训练、一般使用 |
| `pro` | 专业配置 - 高性能训练 | 生产环境、大规模训练 |
| `full` | 完整配置 - 所有功能启用 | 研究、完整功能测试 |

#### 使用方法

```bash
# 使用 lite profile 训练
python -m apt_model train --profile lite

# 使用 pro profile 训练并指定 epochs
python -m apt_model train --profile pro --epochs 50

# 使用 full profile 进行评估
python -m apt_model evaluate --profile full
```

#### Profile 内容示例

```yaml
# lite.yaml
name: apt-lite
version: "1.0"
description: "轻量级配置 - 最小资源占用，快速启动"

layers:
  - L0  # 仅核心层

plugins: []  # 不加载插件

features:
  monitoring: false
  visualization: false

optimization:
  batch_size: 4
  gradient_checkpointing: false
```

#### 注意事项

- **优先级**: 命令行参数 > Profile 配置 > 默认值
- **覆盖**: 可以在使用 profile 的同时指定命令行参数来覆盖配置
- **位置**: Profile 文件位于 `profiles/` 目录

---

### 2. 命令管道 (Pipeline)

#### 什么是命令管道？

命令管道允许你按顺序执行多个命令，类似于 Unix 管道的概念。前一个命令成功后才会执行下一个。

#### 使用方法

```bash
# 基本用法 - 训练、评估、可视化
python -m apt_model pipeline --commands "train,evaluate,visualize"

# 完整的工作流
python -m apt_model pipeline --commands "train,fine-tune,evaluate,compare,backup"

# 结合 profile 使用
python -m apt_model pipeline --profile pro --commands "train,evaluate"
```

#### 命令列表格式

- 使用**逗号**分隔命令
- 不要有空格（或使用引号包裹）
- 按从左到右的顺序执行

#### 执行流程

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│  Train   │ -> │ Evaluate │ -> │Visualize │
└──────────┘    └──────────┘    └──────────┘
     ✓              ✓                ✓
```

如果任一命令失败，管道会立即中断。

#### 示例输出

```
======================================================================
执行命令管道: train -> evaluate -> visualize
======================================================================

[1/3] 执行命令: train
----------------------------------------------------------------------
... training output ...
✓ 命令 'train' 完成

[2/3] 执行命令: evaluate
----------------------------------------------------------------------
... evaluation output ...
✓ 命令 'evaluate' 完成

[3/3] 执行命令: visualize
----------------------------------------------------------------------
... visualization output ...
✓ 命令 'visualize' 完成

======================================================================
✓ 命令管道执行完成! 共执行 3 个命令
======================================================================
```

---

### 3. 模块化选择

#### 什么是模块化选择？

模块化选择允许你动态地启用或禁用特定的功能模块，实现按需加载。

#### 可用模块

##### 核心层级 (L0-L3)

| 模块 | 名称 | 描述 | 必需 |
|------|------|------|------|
| `L0` | Kernel | 核心 APT 算法和基础架构 | ✅ 是 |
| `L1` | Performance | 性能优化和加速 | ❌ 否 |
| `L2` | Memory | 记忆和知识系统 | ❌ 否 |
| `L3` | Product | 产品和应用层 | ❌ 否 |

##### 插件类别

| 类别 | 描述 | 示例插件 |
|------|------|----------|
| `monitoring` | 监控和诊断 | gradient_monitor, resource_monitor |
| `visualization` | 可视化 | model_visualization |
| `evaluation` | 评估和基准测试 | model_evaluator, model_comparison |
| `infrastructure` | 基础设施 | logging |
| `optimization` | 性能优化 | mxfp4_quantization |
| `rl` | 强化学习 | rlhf_trainer, dpo_trainer |
| `protocol` | 协议集成 | mcp_integration |
| `retrieval` | 检索增强 | rag_integration, kg_rag_integration |
| `hardware` | 硬件模拟 | virtual_blackwell, npu_backend |
| `deployment` | 部署和虚拟化 | microvm_compression, vgpu_stack |
| `memory` | 高级记忆系统 | aim_memory |
| `experimental` | 实验性功能 | 各种实验性插件 |

#### 使用方法

##### 列出所有模块

```bash
# 查看所有可用模块及其状态
python -m apt_model list-modules
# 或使用别名
python -m apt_model modules
```

输出示例：

```
================================================================================
APT-Transformer Module Status
================================================================================

Core Layers (L0-L3):
--------------------------------------------------------------------------------
  ✅ L0                   - L0 (Kernel) [ESSENTIAL]
      核心APT算法和基础架构
  ✅ L1                   - L1 (Performance)
      性能优化和加速
  ❌ L2                   - L2 (Memory)
      记忆和知识系统

... (more modules)

================================================================================
Total Modules: 20
Enabled: 12
Disabled: 8
================================================================================
```

##### 启用特定模块

```bash
# 仅启用 L0 和 L1
python -m apt_model train --enable-modules "L0,L1"

# 启用核心层级和监控插件
python -m apt_model train --enable-modules "L0,L1,L2,monitoring"

# 启用强化学习相关模块
python -m apt_model train --enable-modules "L0,rl"
```

##### 禁用特定模块

```bash
# 禁用实验性功能
python -m apt_model train --disable-modules "experimental"

# 禁用所有监控和可视化
python -m apt_model train --disable-modules "monitoring,visualization"

# 仅使用核心功能（禁用所有高级特性）
python -m apt_model train --disable-modules "L2,L3,experimental"
```

##### 组合使用

```bash
# 启用 L0 和 L1，同时禁用实验性功能
python -m apt_model train --enable-modules "L0,L1" --disable-modules "experimental"
```

#### 模块选择规则

1. **默认启用**: L0, L1, L2, L3
2. **必需模块**: L0 (Kernel) 总是启用，无法禁用
3. **优先级**:
   - 必需模块 > 显式启用 > 默认启用
   - 显式禁用会覆盖默认启用
4. **依赖**: 某些模块可能依赖其他模块，系统会自动处理

---

## 🚀 综合使用示例

### 示例 1: 轻量级开发流程

```bash
# 使用 lite profile，仅启用核心模块，执行训练和评估
python -m apt_model pipeline \
  --profile lite \
  --enable-modules "L0,L1" \
  --commands "train,evaluate"
```

### 示例 2: 专业级训练流程

```bash
# 使用 pro profile，启用监控和评估，完整训练流程
python -m apt_model pipeline \
  --profile pro \
  --enable-modules "L0,L1,L2,monitoring,evaluation" \
  --commands "train,fine-tune,evaluate,visualize,backup"
```

### 示例 3: 强化学习实验

```bash
# 使用 standard profile，启用 RL 插件
python -m apt_model train \
  --profile standard \
  --enable-modules "L0,L1,rl" \
  --epochs 100
```

### 示例 4: 最小资源占用

```bash
# 仅使用核心功能，禁用所有可选模块
python -m apt_model train \
  --profile lite \
  --enable-modules "L0" \
  --disable-modules "L1,L2,L3,experimental" \
  --batch-size 2
```

---

## 📊 性能对比

不同配置对资源占用和性能的影响：

| 配置 | 内存占用 | 启动时间 | 功能完整度 | 适用场景 |
|------|----------|----------|------------|----------|
| `--profile lite --enable-modules L0` | 最低 | 最快 | 基础 | 开发调试 |
| `--profile standard` | 中等 | 中等 | 标准 | 日常使用 |
| `--profile pro --enable-modules L0,L1,L2` | 较高 | 较慢 | 高级 | 生产训练 |
| `--profile full` | 最高 | 最慢 | 完整 | 研究实验 |

---

## 🔧 高级配置

### 自定义 Profile

你可以创建自己的 profile：

```bash
# 1. 复制现有 profile
cp profiles/standard.yaml profiles/my-custom.yaml

# 2. 编辑配置
vim profiles/my-custom.yaml

# 3. 使用自定义 profile
python -m apt_model train --profile my-custom
```

### 环境变量

某些配置可以通过环境变量设置：

```bash
export APT_PROFILE=pro
export APT_ENABLE_MODULES="L0,L1,monitoring"

python -m apt_model train
```

---

## 📝 最佳实践

### 1. 选择合适的 Profile

- **开发**: 使用 `lite` - 快速迭代
- **测试**: 使用 `standard` - 平衡性能
- **生产**: 使用 `pro` - 最佳性能
- **研究**: 使用 `full` - 完整功能

### 2. 模块选择策略

- **最小化**: 仅启用必需模块，降低资源占用
- **按需加载**: 根据任务选择相关模块
- **避免冲突**: 某些插件可能不兼容，按需禁用

### 3. 命令管道设计

- **短小精悍**: 管道不宜过长（建议 ≤ 5 个命令）
- **容错处理**: 关键步骤单独执行，避免全盘失败
- **日志记录**: 启用详细日志以便调试

---

## 🐛 故障排查

### 问题 1: Profile 加载失败

**错误**: `FileNotFoundError: Profile file not found`

**解决**:
```bash
# 检查 profile 文件是否存在
ls -la profiles/

# 确保使用正确的 profile 名称
python -m apt_model train --profile lite  # 正确
python -m apt_model train --profile lite.yaml  # 错误 - 不要加 .yaml
```

### 问题 2: 模块未找到

**错误**: `Warning: Unknown module 'xxx' (ignored)`

**解决**:
```bash
# 查看所有可用模块
python -m apt_model list-modules

# 使用正确的模块名称（大小写敏感）
python -m apt_model train --enable-modules "L0,L1"  # 正确
python -m apt_model train --enable-modules "l0,l1"  # 错误
```

### 问题 3: 命令管道中断

**错误**: `命令 'xxx' 执行失败`

**解决**:
```bash
# 单独执行失败的命令以获取详细错误
python -m apt_model xxx --verbose

# 使用 --verbose 标志查看详细日志
python -m apt_model pipeline --verbose --commands "train,evaluate"
```

---

## 📚 相关文档

- **高级 CLI 命令**: `docs/ADVANCED_CLI_COMMANDS.md` ⭐ **NEW**
- **CLI 命令参考**: `docs/CLI_REFERENCE.md`
- **配置文件指南**: `docs/CONFIGURATION_GUIDE.md`
- **插件开发指南**: `docs/product/PLUGIN_SYSTEM_GUIDE.md`
- **贡献指南**: `CONTRIBUTING.md`

---

## 🎯 高级功能命令

除了本文档介绍的基础 CLI 增强功能外，APT-Transformer 还提供了以下高级功能命令：

### 新增高级命令 (2026-01-22)

1. **`train-moe`** - MoE (Mixture of Experts) 模型训练
2. **`blackwell-simulate`** - Virtual Blackwell GPU 模拟
3. **`aim-memory`** - AIM 高级记忆系统管理
4. **`npu-accelerate`** - NPU 加速后端
5. **`rag-query`** - RAG/KG-RAG 检索增强查询
6. **`quantize-mxfp4`** - MXFP4 4位浮点量化

**详细文档**: 请查看 `docs/ADVANCED_CLI_COMMANDS.md`

**快速示例**:
```bash
# MoE 训练
python -m apt_model train-moe --num-experts 8

# Virtual Blackwell 模拟
python -m apt_model blackwell-simulate

# RAG 查询
python -m apt_model rag-query --query "你的问题"

# MXFP4 量化
python -m apt_model quantize-mxfp4
```

---

## 🎓 教程

### 教程 1: 从零开始使用 Lite Profile

```bash
# Step 1: 列出可用模块
python -m apt_model list-modules

# Step 2: 使用 lite profile 训练
python -m apt_model train --profile lite --epochs 5

# Step 3: 评估模型
python -m apt_model evaluate --profile lite

# Step 4: 使用管道自动化
python -m apt_model pipeline --profile lite --commands "train,evaluate"
```

### 教程 2: 强化学习训练

```bash
# Step 1: 启用 RL 模块
python -m apt_model train \
  --enable-modules "L0,L1,rl" \
  --epochs 50

# Step 2: 使用 RLHF 训练器（通过插件）
python -m apt_model plugins-enable rlhf_trainer_plugin

# Step 3: 运行 RLHF 训练
python -m apt_model train-hf --profile pro
```

---

## 🔗 快速链接

- [GitHub Issues](https://github.com/chen0430tw/APT-Transformer/issues)
- [贡献指南](../CONTRIBUTING.md)
- [插件目录](../apt/apps/plugins/PLUGIN_CATALOG.md)

---

**Last Updated**: 2026-01-22
**Maintained by**: APT-Transformer Team
