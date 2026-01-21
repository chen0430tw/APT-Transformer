# MCP Integration Guide

## 📋 概览

MCP (Model Context Protocol) 集成模块为 GPT 模型提供了强大的检索增强生成 (RAG) 能力，支持流式检索、异步操作和多种检索后端。

## ✨ 核心特性

- **异步检索**: 非阻塞的流式检索，不影响生成性能
- **多后端支持**: 支持 FAISS、Annoy、精确检索、GraphRAG
- **证据融合**: 多种融合策略 (加权平均、注意力、最大池化)
- **置信度评分**: 自动评估检索质量
- **缓存优化**: 减少重复检索开销
- **GPT-5 原生支持**: 无缝集成 GPT-5 的 StreamingRetriever

---

## 🚀 快速开始

### 1. 基础用法

```python
from apt_model.modeling.mcp_integration import create_mcp_retriever

# 准备语料库
corpus = [
    "Transformers use self-attention mechanisms.",
    "GPT models are autoregressive.",
    "BERT uses bidirectional encoding.",
]

# 创建检索器
retriever = create_mcp_retriever(
    d_model=512,
    corpus=corpus,
    top_k=3,
    enable_async=True
)

# 执行检索
query = torch.randn(1, 20, 512)  # [batch, seq_len, d_model]
result = retriever.retrieve_sync(query)

print(f"Confidence: {result.confidence:.3f}")
print(f"Retrieved: {result.documents}")
```

### 2. 与 GPT-5 集成

```python
from apt_model.modeling.gpt5_model import GPT5Model
from apt_model.modeling.mcp_integration import upgrade_gpt5_with_mcp

# 创建 GPT-5 模型
model = GPT5Model(
    vocab_size=50257,
    d_model=512,
    n_layers=4,
    num_skills=64
)

# 升级为 MCP 版本
corpus = [...]  # 你的知识库
model = upgrade_gpt5_with_mcp(
    model,
    corpus=corpus,
    top_k=5,
    enable_async=True
)

# 正常使用，检索会自动进行
input_ids = torch.randint(0, 50257, (1, 20))
logits, info = model.forward_step(input_ids, step_idx=0)

# 检索信息在 info 中
print(f"Memory length: {info['mem_len']}")
```

---

## 📊 配置选项

### MCPConfig 参数

```python
@dataclass
class MCPConfig:
    # 检索设置
    provider_name: str = 'exact_cosine'  # 检索后端
    top_k: int = 3                       # 返回文档数
    confidence_threshold: float = 0.6    # 最低置信度

    # 异步设置
    enable_async: bool = True            # 启用异步检索
    retrieval_timeout: float = 2.0       # 超时时间（秒）
    max_queue_size: int = 10             # 请求队列大小

    # 证据融合
    fusion_method: str = 'weighted_mean' # 融合方法
    use_score_weighting: bool = True     # 使用分数加权

    # 缓存设置
    enable_cache: bool = True            # 启用缓存
    cache_size: int = 100                # 缓存大小

    # 模型设置
    d_model: int = 512                   # 模型维度
    rank: int = 32                       # 投影秩
```

### 检索后端选择

| 后端 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| `exact_cosine` | 精确结果，简单 | 速度慢 (O(N)) | 小型语料 (<10K) |
| `faiss_default` | 快速，可扩展 | 需要额外依赖 | 大型语料 (>100K) |
| `annoy_default` | 内存友好 | 近似结果 | 中等语料 (10K-100K) |
| `graph_rag` | 结构化推理 | 复杂度高 | 知识图谱场景 |

---

## 🔧 高级用法

### 1. 自定义文档嵌入

```python
from transformers import AutoModel, AutoTokenizer

# 使用预训练模型编码文档
encoder = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def encode_corpus(corpus, encoder, tokenizer):
    embeddings = []
    for doc in corpus:
        inputs = tokenizer(doc, return_tensors='pt', truncation=True, max_length=128)
        with torch.no_grad():
            output = encoder(**inputs)
            emb = output.last_hidden_state.mean(dim=1)  # Mean pooling
        embeddings.append(emb)
    return torch.cat(embeddings, dim=0)

# 创建检索器
doc_embeddings = encode_corpus(corpus, encoder, tokenizer)
retriever = create_mcp_retriever(
    d_model=384,  # MiniLM 输出维度
    corpus=corpus,
    embeddings=doc_embeddings,
    top_k=5
)
```

### 2. 异步检索模式

```python
# 启动异步 worker
retriever.start_async_worker()

# 提交检索请求
request_id = "req_001"
retriever.retrieve_async(query, request_id)

# 继续其他计算...
# do_some_work()

# 轮询结果
result = retriever.poll_async(request_id)
if result and result.ok:
    print(f"Retrieved: {result.documents}")

# 停止 worker
retriever.stop_async_worker()
```

### 3. 证据融合策略

#### 加权平均 (默认)

```python
config = MCPConfig(
    fusion_method='weighted_mean',
    use_score_weighting=True  # 使用检索分数作为权重
)
```

#### 注意力融合

```python
config = MCPConfig(
    fusion_method='attention',
    d_model=512
)
```

#### 最大池化

```python
config = MCPConfig(
    fusion_method='max_pool'
)
```

### 4. 与 GraphRAG 集成

```python
from apt_model.core.graph_rag.graph_rag_manager import GraphRAGManager

# 创建 GraphRAG
graph_rag = GraphRAGManager(
    max_dimension=2,
    enable_brain=True,
    enable_spectral=True
)

# 添加知识三元组
triples = [
    ("Einstein", "proposed", "Relativity"),
    ("Relativity", "belongs_to", "Physics"),
    ("Quantum_Mechanics", "belongs_to", "Physics"),
]
graph_rag.add_triples_batch(triples)
graph_rag.build_indices()

# 使用 GraphRAG 作为检索后端
# (需要自定义 provider 实现，见下面的示例)
```

---

## 📐 架构设计

### 组件层次

```
┌─────────────────────────────────────┐
│         GPT-5 Model                 │
│  (forward_step with retrieval)      │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│   StreamingRetrieverAdapter         │
│  (Compatible with GPT-5 interface)  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│        MCPRetriever                 │
│  - Query encoding                   │
│  - Evidence fusion                  │
│  - Async worker management          │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│    Retrieval Providers              │
│  - ExactRetriever                   │
│  - FAISSRetriever                   │
│  - AnnoyRetriever                   │
│  - GraphRAG                         │
└─────────────────────────────────────┘
```

### 数据流

```
Query Tensor [B, T, D]
       │
       ▼
Query Encoder (MLP)
       │
       ▼
Pooling (mean) -> Query Vector [B, D]
       │
       ▼
Similarity Compute (cosine)
       │
       ▼
Top-K Selection
       │
       ▼
Evidence Fusion (weighted mean/attention)
       │
       ▼
Evidence Embedding [B, 1, D]
       │
       ▼
Bi-State Alignment (PrecisionAligner)
       │
       ▼
Updated Hidden States
```

---

## 🎯 完整示例

### 示例 1: 基础检索

```python
import torch
from apt_model.modeling.mcp_integration import create_mcp_retriever

# 准备数据
corpus = [
    "Neural networks learn patterns from data.",
    "Deep learning uses multiple layers.",
    "Transformers use self-attention.",
    "CNNs are good for image processing.",
    "RNNs handle sequential data."
]

# 创建检索器
retriever = create_mcp_retriever(
    d_model=256,
    corpus=corpus,
    top_k=2
)

# 模拟查询
query = torch.randn(2, 10, 256)  # 2 个样本，序列长度 10

# 检索
result = retriever.retrieve_sync(query)

print("✓ Retrieval successful!" if result.ok else "✗ Retrieval failed")
print(f"Confidence: {result.confidence:.3f}")
print(f"Documents: {result.documents[:4]}")  # 前 4 个文档
if result.scores is not None:
    print(f"Scores: {result.scores.tolist()}")
```

### 示例 2: GPT-5 + MCP 训练

```python
import torch
import torch.nn as nn
import torch.optim as optim
from apt_model.modeling.gpt5_model import GPT5Model
from apt_model.modeling.mcp_integration import upgrade_gpt5_with_mcp

# 1. 创建模型
model = GPT5Model(
    vocab_size=10000,
    d_model=256,
    n_layers=2,
    num_skills=16,
    top_k=2,
    rank=16
)

# 2. 准备知识库
knowledge_corpus = [
    "Machine learning is a subset of AI.",
    "Neural networks mimic biological neurons.",
    "Backpropagation trains neural networks.",
]

# 3. 升级为 MCP 版本
model = upgrade_gpt5_with_mcp(
    model,
    corpus=knowledge_corpus,
    top_k=2,
    enable_async=False  # 训练时建议同步
)

# 4. 训练循环
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

model.train()
for step in range(100):
    # 准备批次
    input_ids = torch.randint(0, 10000, (4, 32))  # [B=4, T=32]
    labels = torch.randint(0, 10000, (4, 32))

    # 前向传播（会自动检索）
    logits, info = model.forward_step(input_ids, step_idx=step)

    # 计算损失
    loss = criterion(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1)
    )

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}, "
              f"Memory: {info['mem_len']}")
```

### 示例 3: 实时推理 with MCP

```python
import torch
from apt_model.modeling.gpt5_model import GPT5Model
from apt_model.modeling.mcp_integration import upgrade_gpt5_with_mcp

# 加载模型
model = GPT5Model.from_pretrained("path/to/checkpoint")

# 加载知识库
with open("knowledge_base.txt", "r") as f:
    corpus = [line.strip() for line in f if line.strip()]

# 升级
model = upgrade_gpt5_with_mcp(
    model,
    corpus=corpus,
    top_k=5,
    enable_async=True  # 异步模式提高响应速度
)

model.eval()

# 推理
input_text = "What is machine learning?"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

with torch.no_grad():
    for step in range(50):  # 生成 50 个 token
        logits, info = model.forward_step(input_ids, step_idx=step)

        # 采样下一个 token
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # 检查是否检索到新知识
        if info.get('align'):
            print(f"Step {step}: Retrieved knowledge with confidence "
                  f"{info['align'].get('alpha', 0):.3f}")

        # 停止条件
        if next_token.item() == tokenizer.eos_token_id:
            break

output_text = tokenizer.decode(input_ids[0])
print(f"\nGenerated: {output_text}")
```

---

## 🐛 故障排查

### 问题 1: 异步检索不工作

**症状**: `poll()` 总是返回 `None`

**解决方案**:
```python
# 确保启动了 async worker
retriever.start_async_worker()

# 等待足够时间让检索完成
time.sleep(0.1)

# 或者使用同步模式
result = retriever.retrieve_sync(query)
```

### 问题 2: CUDA OOM

**症状**: 检索时显存不足

**解决方案**:
```python
# 1. 减少 top_k
config = MCPConfig(top_k=2)  # 而不是 10

# 2. 文档嵌入放到 CPU
doc_embeddings = doc_embeddings.cpu()

# 3. 使用梯度检查点
model = torch.utils.checkpoint.checkpoint_sequential(model, ...)
```

### 问题 3: 检索速度慢

**症状**: 每次检索耗时 >1s

**解决方案**:
```python
# 1. 使用异步模式
enable_async=True

# 2. 切换到 FAISS
provider_name='faiss_default'

# 3. 启用缓存
enable_cache=True

# 4. 减少语料库大小
corpus = corpus[:10000]  # 限制到 10K 文档
```

### 问题 4: 检索质量差

**症状**: 检索到的文档不相关

**解决方案**:
```python
# 1. 使用更好的文档编码器
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer('all-mpnet-base-v2')
doc_embeddings = encoder.encode(corpus, convert_to_tensor=True)

# 2. 调整 confidence_threshold
config = MCPConfig(confidence_threshold=0.8)  # 更严格

# 3. 增加 top_k
config = MCPConfig(top_k=10)
```

---

## 📚 API 参考

### create_mcp_retriever()

创建 MCP 检索器。

**签名**:
```python
def create_mcp_retriever(
    d_model: int = 512,
    corpus: Optional[List[str]] = None,
    embeddings: Optional[torch.Tensor] = None,
    provider: str = 'exact_cosine',
    top_k: int = 3,
    enable_async: bool = True,
    **kwargs
) -> MCPRetriever
```

**参数**:
- `d_model`: 模型维度
- `corpus`: 文档列表
- `embeddings`: 文档嵌入 [num_docs, d_model]
- `provider`: 检索后端
- `top_k`: 返回文档数
- `enable_async`: 启用异步检索
- `**kwargs`: 额外配置

**返回**: `MCPRetriever` 实例

---

### upgrade_gpt5_with_mcp()

为 GPT-5 模型添加 MCP 检索能力。

**签名**:
```python
def upgrade_gpt5_with_mcp(
    gpt5_model,
    corpus: List[str],
    embeddings: Optional[torch.Tensor] = None,
    top_k: int = 3,
    enable_async: bool = True
)
```

**参数**:
- `gpt5_model`: GPT5Model 实例
- `corpus`: 文档列表
- `embeddings`: 文档嵌入
- `top_k`: 返回文档数
- `enable_async`: 启用异步检索

**返回**: 升级后的 GPT-5 模型

---

### MCPRetriever.retrieve_sync()

同步检索。

**签名**:
```python
def retrieve_sync(
    self,
    query: torch.Tensor,
    top_k: Optional[int] = None
) -> RetrievalResult
```

**参数**:
- `query`: 查询张量 [batch, seq_len, d_model]
- `top_k`: 返回文档数（可选）

**返回**: `RetrievalResult` 对象

---

### RetrievalResult

检索结果数据类。

**字段**:
```python
@dataclass
class RetrievalResult:
    ok: bool                              # 是否成功
    confidence: float                     # 置信度 [0, 1]
    evidence_emb: Optional[torch.Tensor]  # 证据嵌入 [B, 1, D]
    documents: List[str]                  # 检索到的文档
    scores: Optional[torch.Tensor]        # 分数 [B, K]
    metadata: Dict[str, Any]              # 元数据
    error: Optional[str]                  # 错误信息
```

---

## 🔮 未来改进

### 计划特性

1. **多模态检索**: 支持图像、音频检索
2. **GraphRAG 深度集成**: 直接使用 GraphRAG 作为后端
3. **缓存持久化**: 将缓存保存到磁盘
4. **批量检索优化**: 更高效的批量处理
5. **自适应 top-k**: 根据查询难度动态调整
6. **检索反馈学习**: 使用生成质量优化检索

### 贡献指南

欢迎贡献！提交 PR 前请：

1. 确保代码通过 `pytest tests/test_mcp.py`
2. 添加文档注释
3. 更新本 README
4. 遵循 PEP 8 代码风格

---

## 📝 许可证

MIT License

---

## 🙏 致谢

- **APT-Transformer** 团队
- **Retrieval-Augmented Generation (RAG)** 论文作者
- **FAISS** 和 **Annoy** 库维护者
- **GraphRAG** 贡献者

---

## 📧 联系方式

有问题或建议？欢迎：

- 提交 Issue
- 发起 Discussion
- 联系维护者

---

**Happy Retrieving! 🚀**
