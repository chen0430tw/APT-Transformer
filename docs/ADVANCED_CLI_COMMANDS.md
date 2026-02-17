# APT-Transformer 高级CLI命令

**Version**: 1.0
**Last Updated**: 2026-01-22
**Status**: ✅ Implemented

---

## 📋 概述

本文档介绍 APT-Transformer 的高级功能 CLI 命令，包括：

1. **MoE (Mixture of Experts)** - 混合专家模型训练
2. **Virtual Blackwell** - GPU 特性模拟
3. **AIM Memory** - 高级上下文记忆系统
4. **NPU 加速** - Neural Processing Unit 后端
5. **RAG/KG-RAG** - 检索增强生成
6. **MXFP4 量化** - 4位浮点量化

---

## 🚀 高级命令列表

### 1. MoE 训练命令

#### `train-moe` - 训练 Mixture of Experts 模型

**描述**: 使用 MoE 架构训练模型，支持多专家并行计算和动态路由。

**用法**:
```bash
python -m apt_model train-moe [选项]
```

**参数**:
- `--num-experts N` - 专家数量 (默认: 8)
- `--top-k K` - 每次激活的专家数 (默认: 2)
- `--capacity-factor F` - 容量因子 (默认: 1.25)
- `--epochs N` - 训练轮数
- `--batch-size N` - 批次大小

**示例**:
```bash
# 基础 MoE 训练 (8 专家, Top-2)
python -m apt_model train-moe

# 自定义配置 (16 专家, Top-4)
python -m apt_model train-moe --num-experts 16 --top-k 4

# 结合 profile 使用
python -m apt_model train-moe --profile pro --epochs 50
```

**特性**:
- ✅ GPU 并行计算优化
- ✅ 负载均衡机制
- ✅ Token Dispatch 高效路由
- ✅ 支持大规模训练 (10K-100K GPUs)

**适用场景**:
- 大规模语言模型训练
- 多任务学习
- 稀疏激活模型

---

### 2. Virtual Blackwell 模拟

#### `blackwell-simulate` (别名: `vblackwell`) - 启用虚拟 Blackwell GPU

**描述**: 模拟 NVIDIA Blackwell 架构的 GPU 特性，用于测试和开发。

**用法**:
```bash
python -m apt_model blackwell-simulate
```

**示例**:
```bash
# 启用 Virtual Blackwell 模拟
python -m apt_model blackwell-simulate

# 使用别名
python -m apt_model vblackwell
```

**模拟特性**:
- ✅ NVLink 5.0 (1.8 TB/s 带宽)
- ✅ FP4/FP6 精度支持
- ✅ Tensor Core Gen 6
- ✅ SecureTEE 安全隔离
- ✅ 208B transistors 模拟

**适用场景**:
- 新硬件特性测试
- 性能预测和优化
- 开发环境中的 GPU 模拟

---

### 3. AIM Memory 管理

#### `aim-memory` - 管理高级上下文记忆系统

**描述**: AIM (Advanced In-context Memory) 提供分层记忆管理和长期上下文保持。

**用法**:
```bash
python -m apt_model aim-memory --aim-operation <操作> [选项]
```

**参数**:
- `--aim-operation OP` - 操作类型
  - `status` - 查看记忆系统状态 (默认)
  - `clear` - 清除所有记忆
  - `store` - 存储上下文
- `--context TEXT` - 要存储的上下文内容

**示例**:
```bash
# 查看记忆状态
python -m apt_model aim-memory --aim-operation status

# 清除记忆
python -m apt_model aim-memory --aim-operation clear

# 存储上下文
python -m apt_model aim-memory --aim-operation store --context "重要上下文信息"
```

**功能特性**:
- ✅ 分层记忆组织
- ✅ 长期上下文保持
- ✅ 自动重要性评分
- ✅ 上下文检索和组合

**适用场景**:
- 长对话管理
- 多轮交互
- 知识累积

---

### 4. NPU 加速

#### `npu-accelerate` (别名: `npu`) - 启用 NPU 后端

**描述**: 启用 Neural Processing Unit 硬件加速，支持多种 NPU 类型。

**用法**:
```bash
python -m apt_model npu-accelerate --npu-type <类型>
```

**参数**:
- `--npu-type TYPE` - NPU 类型
  - `default` - 自动检测 (默认)
  - `ascend` - 华为 Ascend
  - `kunlun` - 百度昆仑
  - `mlu` - 寒武纪 MLU
  - `tpu` - Google TPU

**示例**:
```bash
# 自动检测 NPU
python -m apt_model npu-accelerate

# 指定华为 Ascend
python -m apt_model npu-accelerate --npu-type ascend

# 使用别名 + 百度昆仑
python -m apt_model npu --npu-type kunlun
```

**支持的 NPU**:
| NPU | 厂商 | 类型 |
|-----|------|------|
| Ascend | 华为 | 训练+推理 |
| Kunlun | 百度 | 训练+推理 |
| MLU | 寒武纪 | 推理 |
| TPU | Google | 训练+推理 |

**适用场景**:
- NPU 集群部署
- 云端推理加速
- 国产硬件适配

---

### 5. RAG 查询

#### `rag-query` - 检索增强生成查询

**描述**: 使用 RAG 或 KG+RAG 进行增强查询，结合检索和生成。

**用法**:
```bash
python -m apt_model rag-query --query <查询内容> [选项]
```

**参数**:
- `--query TEXT` - 查询内容 (必需)
- `--use-kg` - 启用知识图谱增强

**示例**:
```bash
# 基础 RAG 查询
python -m apt_model rag-query --query "什么是 APT Transformer?"

# 使用知识图谱增强
python -m apt_model rag-query --query "APT 的核心算法" --use-kg

# 结合模块选择
python -m apt_model rag-query --query "查询内容" --enable-modules "L0,L1,retrieval"
```

**RAG vs KG-RAG**:

| 特性 | RAG | KG-RAG |
|------|-----|--------|
| 检索方式 | 向量检索 | 图谱 + 向量 |
| 结构化 | 非结构化 | 结构化 |
| 推理能力 | 低 | 高 (多跳推理) |
| 性能 | 快 | 中等 |

**适用场景**:
- 问答系统
- 知识检索
- 文档查询

---

### 6. MXFP4 量化

#### `quantize-mxfp4` (别名: `mxfp4`) - 4位浮点量化

**描述**: 使用 Microsoft-OpenAI 的 MXFP4 格式进行模型量化，实现 4x 加速。

**用法**:
```bash
python -m apt_model quantize-mxfp4 [选项]
```

**参数**:
- `--model-path PATH` - 输入模型路径 (默认: apt_model)
- `--output-path PATH` - 输出路径 (默认: apt_model_mxfp4)

**示例**:
```bash
# 量化默认模型
python -m apt_model quantize-mxfp4

# 指定模型和输出路径
python -m apt_model quantize-mxfp4 \
  --model-path my_model \
  --output-path my_model_quantized

# 使用别名
python -m apt_model mxfp4 --model-path apt_model
```

**MXFP4 特性**:
- ✅ 4位浮点格式
- ✅ Block-wise 8位缩放
- ✅ 4x 推理加速
- ✅ <1% 精度损失
- ✅ 动态范围支持

**性能对比**:
| 格式 | 位数 | 速度 | 精度损失 |
|------|------|------|----------|
| FP32 | 32 | 1x | 0% |
| FP16 | 16 | 2x | <0.1% |
| INT8 | 8 | 3x | 1-2% |
| MXFP4 | 4 | 4x | <1% |

**适用场景**:
- 边缘设备部署
- 推理加速
- 内存受限环境

---

## 🔗 组合使用示例

### 示例 1: MoE + Profile + 模块选择

```bash
# 使用 pro profile，启用优化模块，训练 16 专家 MoE
python -m apt_model train-moe \
  --profile pro \
  --enable-modules "L0,L1,optimization" \
  --num-experts 16 \
  --top-k 4 \
  --epochs 50
```

### 示例 2: NPU + RAG 查询

```bash
# 先启用 NPU 加速
python -m apt_model npu-accelerate --npu-type ascend

# 然后使用 RAG 查询
python -m apt_model rag-query \
  --query "你的问题" \
  --use-kg \
  --enable-modules "L0,retrieval"
```

### 示例 3: 完整工作流 (训练 → 量化 → 部署)

```bash
# Step 1: 训练 MoE 模型
python -m apt_model train-moe --profile pro --epochs 100

# Step 2: 量化模型
python -m apt_model quantize-mxfp4 \
  --model-path apt_model \
  --output-path apt_model_mxfp4

# Step 3: 启用虚拟 Blackwell 测试
python -m apt_model blackwell-simulate

# Step 4: 测试量化模型
python -m apt_model evaluate --model-path apt_model_mxfp4
```

### 示例 4: AIM Memory + 长对话

```bash
# 清除旧记忆
python -m apt_model aim-memory --aim-operation clear

# 开始对话并自动记忆
python -m apt_model chat --enable-modules "L0,memory"

# 查看记忆状态
python -m apt_model aim-memory --aim-operation status
```

---

## 📊 命令分类

### 按功能分类

**训练相关**:
- `train-moe` - MoE 模型训练

**硬件相关**:
- `blackwell-simulate` - GPU 模拟
- `npu-accelerate` - NPU 加速

**记忆相关**:
- `aim-memory` - 记忆管理

**检索相关**:
- `rag-query` - RAG 查询

**优化相关**:
- `quantize-mxfp4` - 模型量化

### 按难度分类

**入门级**:
- `aim-memory --aim-operation status`
- `blackwell-simulate`
- `npu-accelerate`

**中级**:
- `rag-query --query "问题"`
- `quantize-mxfp4`

**高级**:
- `train-moe --num-experts 16`
- `rag-query --use-kg`

---

## 🐛 故障排查

### 问题 1: 插件未找到

**错误**: `ModuleNotFoundError: No module named 'apt.apps.plugins.xxx'`

**解决**:
```bash
# 确保启用了相应的模块
python -m apt_model list-modules

# 启用需要的插件类别
python -m apt_model <命令> --enable-modules "hardware,memory,optimization"
```

### 问题 2: NPU 不支持

**错误**: `NPU type 'xxx' not supported`

**解决**:
```bash
# 使用 default 自动检测
python -m apt_model npu-accelerate --npu-type default

# 或检查可用的 NPU 类型
python -m apt_model npu-accelerate --help
```

### 问题 3: 量化失败

**错误**: `Quantization failed`

**解决**:
```bash
# 确保模型路径正确
ls -la apt_model/

# 使用绝对路径
python -m apt_model quantize-mxfp4 \
  --model-path /absolute/path/to/model
```

---

## 📚 相关文档

- **CLI 基础文档**: `docs/CLI_ENHANCEMENTS.md`
- **插件目录**: `apt/apps/plugins/PLUGIN_CATALOG.md`
- **MoE 实现**: `apt_model/modeling/moe_optimized.py`
- **AIM Memory**: `apt/memory/aim/aim_memory.py`

---

## 🎓 最佳实践

### 1. 选择合适的高级功能

- **MoE**: 大规模模型，需要 GPU 集群
- **Virtual Blackwell**: 开发测试，模拟新硬件
- **AIM Memory**: 长对话，多轮交互
- **NPU**: 国产硬件，云端部署
- **RAG**: 知识密集型任务
- **MXFP4**: 推理加速，边缘部署

### 2. 资源规划

| 功能 | GPU 需求 | 内存需求 | 适用场景 |
|------|----------|----------|----------|
| MoE | 高 | 高 | 训练大模型 |
| Virtual Blackwell | 中 | 中 | 开发测试 |
| AIM Memory | 低 | 中 | 长对话 |
| NPU | 专用硬件 | 中 | 云端部署 |
| RAG | 中 | 高 | 知识检索 |
| MXFP4 | 低 | 低 | 推理优化 |

### 3. 性能优化建议

- **MoE**: 使用 `--profile pro` + `--enable-modules "L0,L1,optimization"`
- **RAG**: 预先构建索引，使用 `--use-kg` 增强
- **MXFP4**: 先在小数据集上验证精度
- **NPU**: 根据硬件选择合适的 NPU 类型

---

**Last Updated**: 2026-01-22
**Maintained by**: APT-Transformer Team
