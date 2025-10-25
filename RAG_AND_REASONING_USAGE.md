# RAG and Reasoning Core Modules - Usage Guide

本文档说明如何使用重构后的 RAG（检索增强生成）和 Reasoning（推理）核心模块。

## 目录

- [RAG 模块](#rag-模块)
  - [检索器实现](#检索器实现)
  - [快速开始](#rag-快速开始)
  - [高级用法](#rag-高级用法)
- [Reasoning 模块](#reasoning-模块)
  - [推理策略](#推理策略)
  - [快速开始](#reasoning-快速开始)
  - [训练推理模型](#训练推理模型)
- [集成使用](#集成使用)

---

## RAG 模块

RAG 模块位于 `apt/core/providers/` 和 `apt_model/modeling/rag_integration.py`。

### 检索器实现

提供了三种检索器实现：

#### 1. FAISS Retriever（推荐用于大规模语料库）

```python
from apt.core.registry import registry

# 获取 FAISS 检索器
provider = registry.get('retrieval', 'faiss_default')({
    'index_type': 'flat',  # 'flat', 'ivf', 'hnsw', 'pq'
    'top_k': 5,
    'd_model': 768,
})

# 创建检索器模块
retriever = provider.create_retriever(d_model=768, top_k=5)

# 构建索引
import numpy as np
doc_embeddings = np.random.randn(1000, 768).astype('float32')
retriever.build_index(doc_embeddings)

# 检索
import torch
query = torch.randn(1, 10, 768)  # [batch, seq_len, d_model]
result = retriever(query)
print(result['retrieved_indices'])  # [batch, top_k]
print(result['retrieval_scores'])   # [batch, top_k]
```

**FAISS 索引类型：**
- `flat`: 精确搜索，适合中小规模（< 1M 文档）
- `ivf`: 倒排索引，适合大规模（> 1M 文档）
- `hnsw`: 分层图索引，查询速度快
- `pq`: 乘积量化，内存效率高

#### 2. Annoy Retriever（推荐用于内存受限环境）

```python
provider = registry.get('retrieval', 'annoy_angular')({
    'metric': 'angular',  # 'angular', 'euclidean', 'dot'
    'n_trees': 10,
    'top_k': 5,
})

retriever = provider.create_retriever(d_model=768, top_k=5)
retriever.build_index(doc_embeddings)
```

**Annoy 特点：**
- 内存映射，不需要将整个索引加载到内存
- 只读索引，适合生产环境
- 构建速度快

#### 3. Exact Retriever（推荐用于小规模语料库和 Baseline）

```python
provider = registry.get('retrieval', 'exact_cosine')({
    'metric': 'cosine',  # 'cosine', 'l2', 'dot'
})

retriever = provider.create_retriever(d_model=768, top_k=5)

# 使用 PyTorch 张量
doc_embeddings_torch = torch.randn(1000, 768)
retriever.set_doc_embeddings(doc_embeddings_torch)
```

**Exact 特点：**
- 100% 精确检索
- 支持多种融合方法（gate, attention, concat）
- 适合小规模（< 10K 文档）

### RAG 快速开始

使用 `quick_rag()` 快速设置：

```python
from apt_model.modeling.rag_integration import quick_rag

# 准备语料库
corpus = [
    "Paris is the capital of France.",
    "The Eiffel Tower is in Paris.",
    "Machine learning is a subset of AI.",
    # ... more documents
]

# 快速设置 RAG
rag_model = quick_rag(
    model=base_model,
    corpus=corpus,
    provider='exact_cosine',  # or 'faiss_default', 'annoy_angular'
    top_k=5,
    d_model=768,
)

# 使用 RAG 模型
input_ids = tokenizer("What is the capital of France?", return_tensors='pt')['input_ids']
outputs = rag_model(input_ids)

print(outputs['retrieved_docs'])  # 检索到的文档
print(outputs['retrieval_scores']) # 相关性分数
print(outputs['logits'])           # 生成的 logits
```

### RAG 高级用法

#### 自定义配置

```python
from apt_model.modeling.rag_integration import RAGConfig, create_rag_model

config = RAGConfig(
    provider_name='faiss_ivf',
    top_k=10,
    index_type='ivf',
    metric='cosine',

    # 融合配置
    fusion_method='attention',  # 'gate', 'attention', 'concat'
    fusion_layer_indices=[6, 9, 12],  # 在哪些层应用 RAG

    # 训练配置
    train_retriever=True,
    freeze_index=False,
    rag_loss_weight=0.1,

    # 缓存配置
    cache_dir='./cache/rag',
    cache_index=True,
    load_index_if_exists=True,
)

rag_model = create_rag_model(
    base_model=model,
    config=config,
    corpus='path/to/corpus.txt',  # 或者 List[str]
    auto_build_index=True,
)
```

#### 从文件加载语料库

```python
from apt_model.modeling.rag_integration import load_corpus_from_file

# 每行一个文档
corpus = load_corpus_from_file(
    'path/to/corpus.txt',
    encoding='utf-8',
    max_size=10000,
)
```

#### 保存和加载索引

```python
# 构建并保存
rag_model.build_index(corpus=corpus)
rag_model.save_index('path/to/index')

# 加载已有索引
rag_model.load_index('path/to/index')
```

#### 独立检索（不生成）

```python
# 只检索，不生成
docs, scores = rag_model.retrieve(
    query=hidden_states,  # [batch, seq_len, d_model]
    top_k=10,
)
```

---

## Reasoning 模块

Reasoning 模块位于 `apt_model/runtime/decoder/`。

### 推理策略

提供了多种推理策略：

#### 1. Structured Reasoning（o3 风格）

单步结构化推理，使用 Vein 子空间投影 + 专家路由：

```python
from apt_model.runtime.decoder import StructuredReasoner
from apt_model.modeling.gpt4o_model import VeinSubspaceShared

# 创建 Vein 投影器
vein = VeinSubspaceShared(d_model=768, rank=4)

# 创建结构化推理器
reasoner = StructuredReasoner(
    vein_projector=vein,
    num_experts=4,
    top_k=2,
    expert_hidden_dim=128,
    use_halting=True,
)

# 执行一步推理
hidden_states = torch.randn(2, 10, 768)  # [batch, seq_len, d_model]
h_new, metadata = reasoner(hidden_states)

print(metadata['p_halt'])     # 停止概率
print(metadata['z_old'])      # 原始 vein 表示
print(metadata['z_new'])      # 更新后的 vein 表示
```

#### 2. Chain-of-Thought Reasoning（CoT）

固定步数的思维链推理：

```python
from apt_model.runtime.decoder import ChainOfThoughtReasoner

cot_reasoner = ChainOfThoughtReasoner(
    vein_projector=vein,
    num_steps=3,
    num_experts=4,
    top_k=2,
)

h_final, intermediates = cot_reasoner(
    hidden_states,
    return_intermediates=True,
)

# 查看中间步骤
for step_info in intermediates:
    print(f"Step {step_info['step']}: {step_info['hidden_states'].shape}")
```

#### 3. Self-Consistency Reasoning（自洽性推理）

多链投票：

```python
from apt_model.runtime.decoder import SelfConsistencyReasoner

sc_reasoner = SelfConsistencyReasoner(
    vein_projector=vein,
    num_chains=5,
    num_steps_per_chain=3,
    aggregation='mean',  # 'mean', 'max', 'vote'
)

h_final = sc_reasoner(hidden_states)
```

#### 4. Reasoning Controller（自适应推理控制器）

多步推理 + 自适应停止：

```python
from apt_model.runtime.decoder import ReasoningController

controller = ReasoningController(
    vein_projector=vein,
    max_steps=6,
    patience=2,
    eps_kl=0.02,
    eps_state=0.03,
    eps_entropy=0.05,
    halt_thresh=0.8,
)

h_final, info = controller(
    hidden_states,
    lm_head=model.lm_head,
    return_details=True,
)

print(f"推理步数: {info['steps']}")
print(f"是否收敛: {info['converged']}")
```

#### 5. Budgeted Reasoning Controller（预算约束推理）

只对高不确定性的 token 进行推理：

```python
from apt_model.runtime.decoder import BudgetedReasoningController

budgeted_controller = BudgetedReasoningController(
    vein_projector=vein,
    global_budget=0.15,  # 15% 的 token 可以推理
    entropy_trigger=2.0,
    max_steps=6,
)

h_final, info = budgeted_controller(
    hidden_states,
    lm_head=model.lm_head,
)

print(f"推理 token 数: {info['num_reasoning_tokens']}")
print(f"推理比例: {info['reasoning_fraction']:.2%}")
```

### Reasoning 快速开始

#### 方式 1：在模型中集成推理

```python
import torch
import torch.nn as nn
from apt_model.runtime.decoder import BudgetedReasoningController
from apt_model.modeling.gpt4o_model import VeinSubspaceShared

class ReasoningEnhancedModel(nn.Module):
    def __init__(self, base_model, d_model, rank=4):
        super().__init__()
        self.base_model = base_model
        self.vein = VeinSubspaceShared(d_model, rank)
        self.reasoning_controller = BudgetedReasoningController(
            self.vein,
            global_budget=0.15,
            max_steps=6,
        )

    def forward(self, input_ids, attention_mask=None):
        # 基础模型前向
        outputs = self.base_model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]

        # 应用推理
        h_reasoned, info = self.reasoning_controller(
            hidden_states,
            self.base_model.lm_head,
        )

        # 生成最终 logits
        logits = self.base_model.lm_head(h_reasoned)

        return {
            'logits': logits,
            'reasoning_steps': info['steps'],
            'reasoning_tokens': info.get('num_reasoning_tokens', 0),
        }

# 使用
model = ReasoningEnhancedModel(base_model, d_model=768)
outputs = model(input_ids)
```

#### 方式 2：只在推理时使用

```python
# 训练时不用推理，推理时才用
model.eval()
with torch.no_grad():
    # 获取隐藏状态
    outputs = model(input_ids, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-1]

    # 应用推理
    h_reasoned, info = controller(hidden_states, model.lm_head)

    # 生成
    logits = model.lm_head(h_reasoned)
```

### 训练推理模型

#### 使用 CLI 命令

```bash
python -m apt_model.cli.main train-reasoning \
    --base-model gpt2 \
    --data-path reasoning_data.jsonl \
    --epochs 3 \
    --batch-size 8 \
    --learning-rate 1e-4 \
    --max-reasoning-steps 6 \
    --global-budget 0.15 \
    --save-path ./models/reasoning
```

#### 使用 Python API

```python
from apt_model.training.train_reasoning import train_reasoning_model

reasoning_controller, training_info = train_reasoning_model(
    base_model=base_model,
    vein_projector=vein,
    tokenizer=tokenizer,
    data_path='reasoning_data.jsonl',
    epochs=3,
    batch_size=8,
    learning_rate=1e-4,
    max_reasoning_steps=6,
    use_budgeted=True,
    global_budget=0.15,
    save_path='./models/reasoning',
)

print(f"平均推理步数: {training_info['avg_reasoning_steps']}")
print(f"训练损失: {training_info['losses']}")
```

#### 推理数据格式

JSON Lines 格式 (`.jsonl`):

```json
{"input": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?", "reasoning_steps": ["All roses are flowers (premise 1)", "Some flowers fade quickly (premise 2)", "From premise 1: roses ⊆ flowers", "But we cannot determine if the quickly-fading flowers include roses"], "output": "No, we cannot conclude that some roses fade quickly."}
{"input": "What is 15% of 80?", "reasoning_steps": ["15% = 15/100 = 0.15", "15% of 80 = 0.15 × 80", "0.15 × 80 = 12"], "output": "12"}
```

或 JSON 格式 (`.json`):

```json
{
  "examples": [
    {
      "input": "问题描述",
      "reasoning_steps": ["步骤1", "步骤2", "步骤3"],
      "output": "答案"
    }
  ]
}
```

---

## 集成使用

同时使用 RAG 和 Reasoning：

```python
from apt_model.modeling.rag_integration import quick_rag
from apt_model.runtime.decoder import BudgetedReasoningController
from apt_model.modeling.gpt4o_model import VeinSubspaceShared

# 1. 创建 RAG 模型
rag_model = quick_rag(
    model=base_model,
    corpus=knowledge_base,
    provider='faiss_default',
    top_k=5,
)

# 2. 创建推理控制器
vein = VeinSubspaceShared(d_model=768, rank=4)
reasoning_controller = BudgetedReasoningController(
    vein_projector=vein,
    global_budget=0.15,
    max_steps=6,
)

# 3. 联合使用
def rag_reasoning_generate(input_ids):
    # RAG: 检索相关文档
    rag_outputs = rag_model(input_ids)
    hidden_states = rag_outputs['hidden_states'][-1]

    # Reasoning: 对检索增强后的隐藏状态进行推理
    h_reasoned, info = reasoning_controller(
        hidden_states,
        rag_model.base_model.lm_head,
    )

    # 生成最终输出
    logits = rag_model.base_model.lm_head(h_reasoned)

    return {
        'logits': logits,
        'retrieved_docs': rag_outputs['retrieved_docs'],
        'reasoning_steps': info['steps'],
    }

# 使用
result = rag_reasoning_generate(input_ids)
print(f"检索文档: {result['retrieved_docs']}")
print(f"推理步数: {result['reasoning_steps']}")
```

---

## 性能优化建议

### RAG 优化

1. **索引选择**：
   - < 10K 文档：使用 `exact_cosine`
   - 10K - 1M 文档：使用 `faiss_flat` 或 `faiss_hnsw`
   - > 1M 文档：使用 `faiss_ivf` 或 `annoy`

2. **缓存索引**：
   ```python
   config = RAGConfig(
       cache_index=True,
       load_index_if_exists=True,
   )
   ```

3. **减少 top_k**：
   ```python
   # top_k=5 通常足够，不需要检索太多文档
   rag_model = quick_rag(model, corpus, top_k=5)
   ```

### Reasoning 优化

1. **使用预算约束**：
   ```python
   # 只对 15% 的高不确定性 token 推理
   controller = BudgetedReasoningController(global_budget=0.15)
   ```

2. **调整最大步数**：
   ```python
   # 减少 max_steps 可以降低计算成本
   controller = ReasoningController(max_steps=3)
   ```

3. **使用 Vein 低秩投影**：
   ```python
   # 使用较小的 rank (2-4) 可以大幅减少计算量
   vein = VeinSubspaceShared(d_model=768, rank=2)
   ```

---

## 架构概览

```
apt/
├── core/
│   └── providers/
│       ├── retrieval.py          # RAG 抽象接口
│       ├── retrieval_faiss.py    # FAISS 实现
│       ├── retrieval_annoy.py    # Annoy 实现
│       └── retrieval_exact.py    # 精确检索实现
│
apt_model/
├── modeling/
│   └── rag_integration.py        # RAG 集成模块
│
├── runtime/
│   └── decoder/
│       ├── halting.py            # 停止机制
│       ├── routing.py            # 专家路由
│       ├── structured_reasoner.py # 结构化推理器
│       └── reasoning_controller.py # 推理控制器
│
└── training/
    └── train_reasoning.py        # 推理模型训练
```

---

## 相关文档

- [RAG_AND_REASONING_ANALYSIS.md](RAG_AND_REASONING_ANALYSIS.md) - RAG 和推理模块分析
- [MEMO_LATEST_UPDATES.md](MEMO_LATEST_UPDATES.md) - 架构指南和分类

---

## 常见问题

### Q: RAG 和 Reasoning 可以单独使用吗？

A: 可以。它们是独立的模块：
- RAG 可以单独用于检索增强生成
- Reasoning 可以单独用于提升推理能力
- 也可以组合使用以获得最佳效果

### Q: 如何选择合适的检索器？

A: 根据语料库大小和性能需求：
- **Exact**: < 10K 文档，需要 100% 精确
- **FAISS Flat**: 10K-100K 文档，需要精确
- **FAISS HNSW**: 100K-1M 文档，查询速度优先
- **FAISS IVF**: > 1M 文档，内存和速度平衡
- **Annoy**: 任意规模，内存受限环境

### Q: 推理控制器的 budget 如何设置？

A: `global_budget` 表示允许多少比例的 token 进入推理：
- `0.05-0.10`: 保守，只对最不确定的 token 推理
- `0.15-0.20`: 平衡（推荐）
- `0.30-0.50`: 激进，对更多 token 推理

### Q: Vein 的 rank 如何选择？

A: Rank 越小，计算越快，但表达能力越弱：
- `rank=2`: 最快，适合简单任务
- `rank=4`: 平衡（推荐）
- `rank=8`: 表达能力强，适合复杂推理

---

生成时间: 2025-10-25
