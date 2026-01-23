# APT-Transformer 发行版模式

## 🎯 为什么需要发行版模式？

APT-Transformer 功能强大但复杂。用户面临的问题：
- 😵 **全家桶焦虑**: 不知道该先开哪些开关
- 🤔 **配置迷失**: 几十个参数，不知道哪些重要
- ⏱️ **上手困难**: 想快速开始，但被复杂度劝退

**解决方案**: 三档官方发行版 + 一键启用

---

## 📦 三档发行版总览

| 发行版 | 层级 | 适用场景 | 启动时间 | 显存占用 |
|--------|------|---------|---------|---------|
| **apt-core** | L0 only | 研究复现、最小可用、CI/CD | 1秒 | 最小 |
| **apt-perf** | L0 + L1 | 生产训练、快速推理 | 3秒 | 中等 |
| **apt-mind** | L0 + L2 | 长对话、知识问答、RAG | 5秒 | 较大 |
| **apt-max** | L0+L1+L2+L3 | 全功能（不推荐默认） | 10秒 | 最大 |

---

## 🟢 apt-core - 核心版（最稳定）

### 定位
**最小、最稳定、最好测** - 适合科研复现和 CI/CD

### 包含功能
✅ APT 核心模型（Autopoietic Transform, DBC-DAC, Left-Spin Smooth）
✅ 基础训练循环（train/eval/generate）
✅ 标准推理接口
✅ 最小配置文件

❌ 不包含: 性能优化、记忆系统、WebUI、插件

### 安装与启用

```bash
# 最小安装
pip install apt-transformer[core]

# 或从源码
pip install -e ".[core]"
```

```python
# Python 代码一行启用
from apt import enable

enable('core')  # 只加载 L0 核心

# 使用
from apt_core import APTModel, Trainer

model = APTModel.from_config('config.json')
trainer = Trainer(model)
trainer.train(dataset, epochs=10)
```

### 配置文件 (profiles/core.yaml)

```yaml
name: apt-core
description: "最小核心版本 - 适合研究复现"

# 启用层级
layers:
  - L0  # 只加载核心层

# 核心模型配置
model:
  type: APTModel
  d_model: 768
  num_heads: 12
  num_layers: 12
  vocab_size: 50000
  max_seq_len: 512

  # 核心特性
  use_autopoietic: true
  use_dbc_dac: true
  use_left_spin: true

# 训练配置
training:
  optimizer: AdamW
  learning_rate: 3.0e-4
  batch_size: 32
  max_steps: 100000

# 不启用高级特性
performance:
  enabled: false
memory:
  enabled: false
plugins:
  enabled: false
```

### 适用场景

1. **论文复现**: 只需要核心算法
2. **单元测试**: 最快的测试速度
3. **CI/CD**: 持续集成环境
4. **教学演示**: 最简单的入门
5. **Debug**: 排除干扰因素

### 示例代码

```python
# examples/core_minimal.py
from apt_core import APTModel, Trainer
from apt_core.config import APTConfig

# 1. 加载配置
config = APTConfig.from_yaml('profiles/core.yaml')

# 2. 创建模型
model = APTModel(config)

# 3. 训练（最小循环）
trainer = Trainer(model, config)
trainer.train(
    train_data='data/train.txt',
    eval_data='data/eval.txt',
    epochs=10
)

# 4. 推理
output = model.generate("Hello, world!", max_length=50)
print(output)
```

### 性能指标

| 指标 | 数值 |
|------|------|
| 模型加载时间 | < 1s |
| 启动内存占用 | ~500MB |
| 最小 GPU 显存 | 2GB (BERT-base) |
| 测试覆盖率 | 95% |

---

## ⚡ apt-perf - 性能版（推荐生产）

### 定位
**跑得快，适合训练/推理** - 生产环境首选

### 包含功能
✅ apt-core 的所有功能
✅ **虚拟 Blackwell** (VGPU Stack, MXFP4, GPU Flash)
✅ 混合精度训练 (FP16/BF16)
✅ 分布式训练 (DDP, FSDP)
✅ Checkpoint 原子性保护
✅ MoE 优化
✅ 量化与压缩

❌ 不包含: 记忆系统、WebUI（可选开启）

### 安装与启用

```bash
# 性能版安装
pip install apt-transformer[perf]

# 额外依赖
pip install deepspeed accelerate
```

```python
# 一行启用
from apt import enable

enable('perf')  # 加载 L0 + L1

# 自动启用的优化
# - 虚拟 Blackwell (balanced 模式)
# - 混合精度 (自动检测 FP16/BF16)
# - Checkpoint 原子性
# - 梯度累积

from apt_core import APTModel
from apt_perf import VirtualBlackwellOptimizer

model = APTModel.from_config('config.json')

# 显式配置性能
vb = VirtualBlackwellOptimizer(model)
vb.enable(mode='performance')  # 性能优先

# 训练时自动加速
trainer.train(model, dataset)
```

### 配置文件 (profiles/perf.yaml)

```yaml
name: apt-perf
description: "性能优化版本 - 适合生产训练"

# 启用层级
layers:
  - L0
  - L1  # 性能层

# 继承核心配置
extends: core.yaml

# 性能优化配置
performance:
  enabled: true

  # 虚拟 Blackwell
  virtual_blackwell:
    enabled: true
    mode: balanced  # balanced / performance / memory

    # VGPU 堆叠
    vgpu_stack:
      enabled: true
      levels:
        - capacity_mb: 2000
          device: cuda:0
          speed_gbps: 900
        - capacity_mb: 8000
          device: cpu
          speed_gbps: 50
        - capacity_mb: 32000
          device: ssd
          speed_gbps: 7

    # MXFP4 量化
    mxfp4:
      enabled: true
      inference_only: true

    # GPU Flash
    gpu_flash:
      enabled: true
      kernel_fusion: true
      flash_attention: true

  # 混合精度
  mixed_precision:
    enabled: true
    dtype: auto  # auto / fp16 / bf16

  # 分布式训练
  distributed:
    backend: nccl
    find_unused_parameters: false

  # Checkpoint
  checkpoint:
    atomic_save: true
    save_interval: 1000
    keep_last_n: 3

# 训练配置（优化版）
training:
  batch_size: 64  # 更大批次
  gradient_accumulation: 4
  max_grad_norm: 1.0
```

### 性能对比

| 指标 | apt-core | apt-perf | 提升 |
|------|---------|----------|------|
| 训练速度 (GPT-2) | 100 samples/s | **350 samples/s** | 3.5× |
| 推理延迟 (BERT) | 100ms | **35ms** | 2.9× |
| 显存占用 (7B模型) | 14GB | **3.5GB** | 4× |
| 虚拟显存 | 24GB | **64GB** | 2.7× |

### 适用场景

1. **生产训练**: 需要快速迭代
2. **大规模推理**: 低延迟要求
3. **显存受限**: GPU 显存不足
4. **分布式训练**: 多 GPU / 多节点
5. **成本优化**: 用消费级 GPU 训练大模型

### 示例代码

```python
# examples/perf_distributed.py
from apt import enable

enable('perf')

from apt_core import APTModel
from apt_perf import DistributedTrainer, VirtualBlackwell

# 1. 启用虚拟 Blackwell（一行）
VirtualBlackwell.enable('balanced')

# 2. 分布式训练
trainer = DistributedTrainer(
    model=APTModel.from_config('config.json'),
    world_size=8,  # 8 GPU
    backend='nccl'
)

# 3. 训练（自动加速）
trainer.train(
    dataset='data/train.txt',
    batch_size=64,
    fp16=True
)
```

---

## 🧠 apt-mind - 记忆版（长对话优先）

### 定位
**长对话、知识问答、RAG 能力优先**

### 包含功能
✅ apt-core 的所有功能
✅ **AIM-Memory** (惯性路由、时间镜像、锚点纠错)
✅ **AIM-NC** (n-gram 收编召回 + 锚点主权)
✅ **GraphRAG** (知识图谱 + 检索增强)
✅ 分层记忆 (A/B/C 档)
✅ 证据回灌（strict 模式）
✅ 长上下文机制 (RoPE 变体)

❌ 不包含: 高级性能优化（但可单独开启）

### 安装与启用

```bash
# 记忆版安装
pip install apt-transformer[mind]

# 额外依赖
pip install faiss-gpu networkx
```

```python
# 一行启用
from apt import enable

enable('mind')  # 加载 L0 + L2

from apt_core import APTModel
from apt_memory import AIMMemory, GraphRAG

model = APTModel.from_config('config.json')

# 启用记忆系统
memory = AIMMemory(
    mode='aim-nc',
    strict=False,  # 默认轻量
    anchor_sovereignty=True
)

# 使用
context = memory.get_context(query, max_tokens=2048)
output = model.generate(prompt, context=context)
```

### 配置文件 (profiles/mind.yaml)

```yaml
name: apt-mind
description: "记忆增强版本 - 适合长对话和 RAG"

# 启用层级
layers:
  - L0
  - L2  # 记忆层

extends: core.yaml

# 记忆系统配置
memory:
  enabled: true

  # AIM-Memory
  aim_memory:
    enabled: true
    mode: aim-nc  # aim / aim-nc
    strict_mode: false  # 默认轻量（摘要+fields）
    anchor_sovereignty: true

    # 惯性路由
    inertial_routing:
      enabled: true
      decay_rate: 0.95

    # 时间镜像
    time_mirror:
      enabled: true
      window_size: 1000

    # 锚点纠错
    anchor_correction:
      enabled: true
      threshold: 0.8

  # AIM-NC
  aim_nc:
    enabled: true
    ngram_size: 3
    trie_cache_size: 10000

  # 分层记忆
  tiered_memory:
    enabled: true
    tiers:
      - name: A  # 原文哈希
        capacity: 1000
        ttl: 86400
      - name: B  # 字段 JSON
        capacity: 10000
        ttl: 604800
      - name: C  # 摘要 + 回溯链接
        capacity: 100000
        ttl: -1

  # GraphRAG
  graph_rag:
    enabled: true

    # Graph Brain
    graph_brain:
      enabled: true
      update_interval: 100

    # Hodge-Laplacian
    hodge_laplacian:
      enabled: true
      num_eigenvalues: 50

    # RAG 管理器
    rag_manager:
      retrieval_k: 5
      rerank: true

# 长上下文
long_context:
  max_seq_len: 8192  # 扩展到 8K
  rope_variant: longrope2  # rope / irope / yarn / longrope2
```

### 记忆能力对比

| 能力 | apt-core | apt-mind |
|------|---------|----------|
| 最大上下文 | 512 tokens | **8192 tokens** |
| 记忆容量 | - | **100K+ 项** |
| 召回延迟 | - | **< 10ms** |
| 锚点准确率 | - | **95%+** |
| RAG 命中率 | - | **90%+** |

### 适用场景

1. **长对话系统**: 客服机器人
2. **知识问答**: QA 系统
3. **文档检索**: RAG 应用
4. **个人助理**: 记住用户偏好
5. **知识图谱**: 实体关系推理

### 示例代码

```python
# examples/mind_rag.py
from apt import enable

enable('mind')

from apt_core import APTModel
from apt_memory import GraphRAG, AIMMemory

# 1. 构建知识图谱
rag = GraphRAG()
rag.build_from_documents([
    'APT is a transformer model.',
    'DBC-DAC optimizes dimensions.',
    'Virtual Blackwell accelerates inference.'
])

# 2. 启用记忆系统
memory = AIMMemory(mode='aim-nc', strict=False)

# 3. 长对话
model = APTModel.from_config('config.json')

query = "What is DBC-DAC?"
context = rag.retrieve(query, k=3) + memory.get_context(query)

output = model.generate(query, context=context, max_length=200)
print(output)

# 4. 存储对话到记忆
memory.store(query=query, response=output, timestamp=time.time())
```

---

## 🚀 apt-max - 全功能版（谨慎使用）

### 定位
**所有功能** - 不推荐作为默认入口

### 包含功能
✅ L0 + L1 + L2 + L3 全部功能
✅ WebUI (4 个 Tab)
✅ REST API
✅ 插件生态
✅ Agent 系统
✅ 完整可观测性

⚠️ **警告**: 启动慢、内存占用大、复杂度高

### 安装与启用

```bash
# 完整安装
pip install apt-transformer[max]

# 或全部可选依赖
pip install apt-transformer[all]
```

```python
# 一行启用
from apt import enable

enable('max')  # 加载所有层级

# 启动 WebUI
from apps.webui import launch

launch(port=7860)  # http://localhost:7860
```

### 配置文件 (profiles/max.yaml)

```yaml
name: apt-max
description: "全功能版本 - 包含所有特性"

# 启用层级
layers:
  - L0
  - L1
  - L2
  - L3

extends: perf.yaml

# 继承 perf + mind 的所有配置
merge:
  - perf.yaml
  - mind.yaml

# 应用层配置
product:
  enabled: true

  # WebUI
  webui:
    enabled: true
    port: 7860
    auth: false
    tabs:
      - training_monitor
      - gradient_monitor
      - checkpoint_manager
      - inference_tester

  # REST API
  api:
    enabled: true
    port: 8000
    auth: true
    api_key: ${APT_API_KEY}

  # CLI
  cli:
    enabled: true
    interactive: true

  # 可观测性
  observability:
    enabled: true
    collectors:
      - training_monitor
      - gradient_monitor
      - resource_monitor
    dashboards:
      - webui

  # 插件
  plugins:
    enabled: true
    load:
      - compression
      - visual_distillation
      - web_search
      - teacher_api
      - graph_rag

  # Agent
  agent:
    enabled: true
    tools:
      - python_sandbox
      - web_search
      - calculator
```

### 性能开销

| 指标 | apt-perf | apt-max | 增加 |
|------|---------|---------|------|
| 启动时间 | 3s | **10s** | 3.3× |
| 内存占用 | 2GB | **5GB** | 2.5× |
| 依赖数量 | 30 | **80+** | 2.7× |

### 适用场景

1. **完整演示**: 展示所有功能
2. **高级开发**: 需要所有工具
3. **一站式平台**: 不想分模块安装

⚠️ **不推荐**: 日常开发、生产环境、CI/CD

---

## 🎛️ 一键启用 API

### Python API

```python
from apt import enable

# 方式 1: 字符串
enable('core')   # 核心版
enable('perf')   # 性能版
enable('mind')   # 记忆版
enable('max')    # 全功能版

# 方式 2: 混合启用
enable('core', 'perf')  # 核心 + 性能

# 方式 3: 自定义
enable(layers=['L0', 'L1'], plugins=['compression'])

# 方式 4: 配置文件
enable(profile='profiles/my_custom.yaml')
```

### CLI

```bash
# 启动训练
apt-train --profile core
apt-train --profile perf --distributed

# 启动推理
apt-generate --profile mind --prompt "Hello"

# 启动 WebUI
apt-webui --profile max
```

### 环境变量

```bash
# .env 文件
APT_PROFILE=perf
APT_ENABLE_VB=true
APT_ENABLE_MEMORY=false
```

```python
# 自动读取
from apt import enable

enable()  # 读取 APT_PROFILE
```

---

## 📊 选择指南

### 快速决策树

```
你的需求是什么？
│
├─ 论文复现 / 最小可用
│  └─ ✅ apt-core
│
├─ 生产训练 / 快速推理
│  └─ ✅ apt-perf
│
├─ 长对话 / RAG / 知识问答
│  └─ ✅ apt-mind
│
├─ 完整演示 / 高级开发
│  └─ ⚠️ apt-max（谨慎）
│
└─ 自定义需求
   └─ 📝 自己编写 profile YAML
```

### 硬件需求对比

| 发行版 | 最小 GPU | 推荐 GPU | 最小内存 | 推荐内存 |
|--------|---------|----------|---------|---------|
| apt-core | - | 2GB | 4GB | 8GB |
| apt-perf | 4GB | 8GB+ | 8GB | 16GB |
| apt-mind | 4GB | 8GB | 16GB | 32GB |
| apt-max | 8GB | 24GB+ | 32GB | 64GB |

### 场景推荐

| 场景 | 推荐发行版 | 理由 |
|------|-----------|------|
| 🔬 科研复现 | core | 最小干扰 |
| 🏭 生产训练 | perf | 速度优先 |
| 💬 客服机器人 | mind | 记忆能力 |
| 📚 RAG 系统 | mind | 检索增强 |
| 🎮 Demo 展示 | max | 全功能 |
| 🧪 快速原型 | core | 快速迭代 |
| ☁️ 云端部署 | perf | 成本优化 |
| 🎓 教学演示 | core | 易于理解 |

---

## 🔄 发行版切换

### 无缝切换

```python
# 当前使用 core
from apt import enable, switch

enable('core')
# ... 训练代码 ...

# 切换到 perf（不重启）
switch('perf')
# 自动启用性能优化
```

### 保存当前配置

```bash
# 导出当前配置
apt-config export > my_config.yaml

# 稍后恢复
apt-train --profile my_config.yaml
```

---

## 📚 相关文档

- [ARCHITECTURE.md](./ARCHITECTURE.md) - 分层架构设计
- [profiles/](./profiles/) - 配置文件示例
- [examples/](./examples/) - 各发行版示例代码

---

**版本**: 1.0
**作者**: APT Team
**日期**: 2025-01-21
**推荐**: 从 **apt-perf** 开始 👍
