# AIM-Memory 技术指南

**AIM-Memory (Anchored Inertial Mirror Memory)**
惯性锚定镜像记忆系统

## 概述

AIM-Memory 是一种面向大模型与智能体的分层记忆架构，用"惯性路由 + 时间镜像 + 锚点纠错 + 按需证据回灌"替代"全量上下文堆叠"。

### 核心优势

- **惯性路由**：只检索小簇，不全库扫描，大幅降低检索成本
- **时间镜像**：权重衰减自然表达时序，无需维护时间戳
- **锚点纠错**：数字/专名准确匹配，防止幻觉和记忆混淆
- **按需回灌**：平时用摘要节省 token，严格时用原文保证精度
- **低成本**：节省 KV cache 和 token 消耗，提升响应速度
- **高可靠**：可验证、可核对、低漂移

## 核心机制

### 1. 惯性路由 (Inertial Routing, IM)

**问题**：传统 RAG 需要全库向量搜索，成本高昂。

**解决方案**：维护一个"惯性方向"向量 `v_inertia`，它记录了用户最近的查询方向。

**工作原理**：
```python
# Step R1: 形成惯性方向
d = q_vec + λ * v_inertia

# Step R2: 局部 K 簇召回
candidates = node_bank.top_k_cluster(d, K)

# Step R5: 更新惯性
v_inertia = μ * v_inertia + (1-μ) * v_selected
```

**效果**：连续查询会自然地"沿着轨道"找到相关记忆簇，无需每次全库扫描。

### 2. 时间镜像 (Temporal Mirror, TM)

**问题**：需要表达记忆的时序关系，但维护时间戳增加复杂度。

**解决方案**：使用权重衰减机制。每次写入新记忆时，所有旧记忆的权重乘以衰减因子 γ（通常 0.8）。

**工作原理**：
```python
# Step W4: 镜像衰减
for node in node_bank:
    node.w *= γ  # γ = 0.8

# Step W5: 新节点权重为 1.0
new_node.w = 1.0
```

**效果**：越新的记忆权重越高，越旧的记忆权重越低，自然形成时序梯度。

### 3. 锚点纠错 (Anchored Correction, A)

**问题**：大模型容易"记混"相似信息，出现幻觉。

**解决方案**：提取并验证关键字段（数字、专名、符号、定义）。

**工作原理**：
```python
# 提取查询和节点的关键字段
q_fields = extract_fields(query)  # 数字、专名、定义、符号
n_fields = node.fields

# 计算锚点匹配分数
anchor_score = weighted_overlap(q_fields, n_fields)

# 只有锚点匹配才能召回
if anchor_score > threshold:
    node_score = base_score + anchor_bonus * node.w
```

**效果**：查询"10M tokens 的模型"时，只会召回真正包含"10M"的节点，不会混淆 128K 或其他数字。

### 4. 按需证据回灌 (Evidence Refill, E)

**问题**：存储原文占用大量空间，但又需要精确引用。

**解决方案**：默认只存摘要，检测到"精确/原文/证明"等关键词时才回灌原文。

**工作原理**：
```python
# 快速模式：只用摘要
if mode == 'fast':
    return summaries

# 严格模式：回灌原文
if mode == 'strict' or detect_strict_keywords(query):
    evidence = fetch_evidence(selected_nodes)
    return summaries + evidence
```

**效果**：平时节省 token，需要精确信息时自动切换到严格模式。

## 数据结构

### MemoryNode（记忆节点）

```python
@dataclass
class MemoryNode:
    id: str                          # 节点 ID
    proto: np.ndarray                # 原型向量（embedding）
    summary: str                     # 一行摘要
    fields: Dict[str, Any]           # 关键字段
        # - numbers: 数字列表
        # - names: 专名列表
        # - definitions: 定义列表
        # - symbols: 符号列表
    links: List[str]                 # 相邻节点 ID
    w: float = 1.0                   # 时间权重
    evidence_ptr: Optional[str]      # 证据指针（hash）
    evidence_text: Optional[str]     # 证据原文
```

### HotKV（热缓存）

滑动窗口缓存，存储最近 W 条原文。

```python
class HotKV:
    def __init__(self, window_size: int):
        self.window_size = window_size  # W = 128-512
        self.buffer: List[str] = []

    def append(self, text: str):
        self.buffer.append(text)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
```

### MemoryMap（记忆索引）

简化的近似最近邻索引，支持 K 簇召回。

```python
class MemoryMap:
    def top_k_cluster(self, direction: np.ndarray, k: int) -> List[str]:
        """返回与 direction 最接近的 k 个节点 ID"""
        scores = {
            nid: cosine_similarity(direction, proto)
            for nid, proto in self.index.items()
        }
        return sorted(scores, key=scores.get, reverse=True)[:k]
```

## 关键算法

### 写入路径 (WriteMemory)

```python
def write_memory(text: str, context: List[str]) -> bool:
    # W0: 更新热缓存
    hot_kv.append(text)

    # W1: 门控判断
    r = relevance(context, text)      # 相关性
    s = surprisal(text, context)      # 惊喜度
    c = conflict_score(text, recent)  # 冲突度

    g = sigmoid(α*r + β*s + γ2*c)
    if g < τ_write:
        return False  # 不写入长期记忆

    # W2: 提取字段
    fields = extract_fields(text)

    # W3: 创建镜像节点
    node = MemoryNode(
        proto=embed(summary + fields),
        summary=summarize(text),
        fields=fields,
        w=1.0,
        evidence_text=text
    )

    # W4: 镜像衰减（所有旧节点权重 *= γ）
    for old_node in node_bank:
        old_node.w *= γ

    # W5: 插入和链接
    node_bank.insert(node)
    link_neighbors(node)

    return True
```

### 路由路径 (RouteMemory)

```python
def route_memory(query: str, mode: str) -> Tuple[List[Node], str]:
    # R1: 形成惯性方向
    q_vec = embed(query)
    d = q_vec + λ * v_inertia

    # R2: 局部 K 簇召回
    candidate_ids = node_bank.top_k_cluster(d, K)
    candidates = [node_bank[nid] for nid in candidate_ids]

    # R3: 锚点纠错
    q_fields = extract_fields(query)
    scored = []
    for node in candidates:
        anchor_score = anchor_check(q_fields, node.fields)

        # 基础分数 + 锚点加成 - 冲突惩罚
        base_score = 0.3 * node.w
        anchor_bonus = anchor_score * η * node.w
        conflict = conflict_penalty(query, node)

        node_score = base_score + anchor_bonus - conflict

        if node_score > τ_anchor:
            scored.append((node, node_score))

    # 按分数排序
    scored.sort(key=lambda x: x[1], reverse=True)
    selected = [node for node, _ in scored[:top_n]]

    # R4: 按需证据回灌
    refill = ""
    if mode == 'strict' or detect_strict_keywords(query):
        refill = fetch_evidence(selected)

    # R5: 更新惯性
    if selected:
        v_sel = np.mean([n.proto for n in selected], axis=0)
        v_inertia = μ * v_inertia + (1-μ) * v_sel

    return selected, refill
```

### 生成路径 (Answer)

```python
def answer(query: str, auto_mode: bool = True) -> Dict:
    # G1: 自动模式判断
    mode = 'fast'
    if auto_mode and detect_strict_keywords(query):
        mode = 'strict'

    # G2: 路由召回
    selected, refill = route_memory(query, mode)

    # G3: 构建上下文
    context = build_context(
        hot_kv=hot_kv.get_recent(),
        summaries=[n.summary for n in selected],
        fields=[n.fields for n in selected],
        evidence=refill
    )

    # G4: 返回结果
    return {
        'query': query,
        'mode': mode,
        'selected_nodes': selected,
        'num_nodes_recalled': len(selected),
        'context': context,
        'inertia_norm': np.linalg.norm(v_inertia)
    }
```

## 配置参数

```python
@dataclass
class AIMConfig:
    # 窗口和簇大小
    hot_window_size: int = 256        # W: 热缓存窗口大小
    local_cluster_k: int = 32         # K: 局部簇召回数量

    # 惯性参数
    inertia_strength: float = 0.5     # λ: 惯性强度
    inertia_momentum: float = 0.85    # μ: 惯性动量

    # 时间镜像参数
    weight_decay_gamma: float = 0.8   # γ: 权重衰减因子
    fresh_weight: float = 1.0         # 新节点初始权重

    # 门控参数
    write_threshold: float = 0.6      # τ_write: 写入门槛
    gate_alpha: float = 0.5           # α: 相关性权重
    gate_beta: float = 0.3            # β: 惊喜度权重
    gate_gamma2: float = 0.2          # γ2: 冲突度权重

    # 锚点参数
    anchor_threshold: float = 0.1     # τ_anchor: 锚点门槛
    anchor_boost: float = 2.0         # η: 锚点加成
    conflict_penalty: float = 1.0     # 冲突惩罚

    # 字段权重
    field_weights: Dict[str, float] = field(default_factory=lambda: {
        'numbers': 3.0,      # 数字权重最高
        'names': 2.0,        # 专名次之
        'definitions': 1.5,  # 定义
        'symbols': 2.5       # 符号
    })

    # 证据参数
    store_evidence: bool = True       # 是否存储证据原文
    max_evidence_length: int = 2000   # 证据最大长度
```

### 参数调优建议

#### 场景 1: 快速对话助手
```python
config = AIMConfig(
    hot_window_size=128,      # 较小窗口
    local_cluster_k=16,       # 较少召回
    write_threshold=0.5,      # 宽松写入
    inertia_strength=0.6      # 较强惯性（对话连续性强）
)
```

#### 场景 2: 精确知识库
```python
config = AIMConfig(
    hot_window_size=512,      # 较大窗口
    local_cluster_k=64,       # 更多召回
    write_threshold=0.7,      # 严格写入
    anchor_boost=3.0,         # 更强锚点验证
    store_evidence=True       # 必须存储原文
)
```

#### 场景 3: 长期记忆系统
```python
config = AIMConfig(
    weight_decay_gamma=0.9,   # 较慢衰减
    inertia_momentum=0.9,     # 较强动量
    local_cluster_k=32,
    write_threshold=0.4       # 宽松写入（保留更多记忆）
)
```

## 使用示例

### 基础使用

```python
from apt_model.memory.aim_memory import create_aim_memory, AIMConfig

# 创建 AIM-Memory 实例
aim = create_aim_memory()

# 写入记忆
aim.write_memory("RoPE 是旋转位置编码，通过复数旋转实现位置表示。")
aim.write_memory("YaRN 通过分维度缩放扩展 RoPE 到更长上下文。")
aim.write_memory("Llama 4 使用 iRoPE 支持 10M tokens 上下文。")

# 查询记忆（快速模式）
selected, refill = aim.route_memory("如何支持超长上下文？", mode='fast')
for node in selected:
    print(f"• {node.summary}")

# 完整回答生成
result = aim.answer("10M tokens 的模型是哪个？", auto_mode=True)
print(f"模式: {result['mode']}")
print(f"召回节点数: {result['num_nodes_recalled']}")
print(f"上下文:\n{result['context']}")
```

### 自定义配置

```python
# 创建自定义配置
config = AIMConfig(
    hot_window_size=512,
    local_cluster_k=64,
    write_threshold=0.4,
    anchor_threshold=0.15,
    weight_decay_gamma=0.85
)

aim = create_aim_memory(config=config)
```

### 持久化

```python
# 保存到文件
aim.save("/path/to/memory.json")

# 从文件加载
aim2 = create_aim_memory()
aim2.load("/path/to/memory.json")
```

### 统计信息

```python
stats = aim.get_stats()
print(f"热缓存大小: {stats['hot_kv_size']}")
print(f"长期节点数: {stats['node_bank_size']}")
print(f"惯性范数: {stats['inertia_norm']:.4f}")
print(f"总访问次数: {stats['total_access']}")
print(f"平均权重: {stats['avg_weight']:.4f}")
```

## 集成到 APT-Transformer

AIM-Memory 可以作为 APT-Transformer 的长期记忆模块：

```python
from apt_model.memory.aim_memory import create_aim_memory
from apt_model.modeling.apt_transformer import APTTransformer

# 创建模型
model = APTTransformer(config)

# 创建记忆系统
memory = create_aim_memory()

# 推理时使用记忆
def generate_with_memory(prompt: str):
    # 从记忆中检索相关上下文
    result = memory.answer(prompt, auto_mode=True)
    context = result['context']

    # 构建完整输入
    full_input = f"{context}\n\n用户: {prompt}\n助手:"

    # 模型生成
    output = model.generate(full_input)

    # 存储对话到记忆
    memory.write_memory(f"用户: {prompt}")
    memory.write_memory(f"助手: {output}")

    return output
```

## 与其他系统对比

### vs 传统 RAG

| 特性 | 传统 RAG | AIM-Memory |
|------|----------|------------|
| 检索方式 | 全库向量搜索 | 惯性局部簇召回 |
| 时序表达 | 时间戳或无 | 权重衰减 |
| 精度保证 | 依赖 embedding | 锚点字段验证 |
| 成本 | 高（每次全库） | 低（只查小簇） |

### vs MemGPT/Mem0

| 特性 | MemGPT/Mem0 | AIM-Memory |
|------|-------------|------------|
| 定位 | 通用记忆框架 | 针对 LLM 优化 |
| 惯性路由 | ✗ | ✓ |
| 时间镜像 | ✗ | ✓ |
| 锚点纠错 | ✗ | ✓ |
| 证据回灌 | ✗ | ✓ |

### vs 分层记忆（HierarchicalMemory）

AIM-Memory 可以与分层记忆系统协同工作：
- **A 层**（工作记忆）：热缓存 HotKV
- **B 层**（短期记忆）：最近写入的高权重节点
- **C 层**（长期记忆）：权重衰减后的旧节点

## 技术原理

### 为什么惯性路由有效？

人类记忆也有"联想惯性"：想到"Transformer"后更容易想到"注意力机制"，而不是"苹果"。

惯性向量 `v_inertia` 捕获了这种"思维方向"，使得连续查询能自然地落在相关记忆簇中。

### 为什么时间镜像有效？

权重衰减 `w *= γ` 相当于指数遗忘曲线，符合人类记忆规律。

相比维护时间戳：
- 更简单（无需时间比较逻辑）
- 更高效（自然排序）
- 更灵活（可调节衰减速度）

### 为什么锚点纠错有效？

大模型的 embedding 容易混淆语义相似但事实不同的信息。例如：
- "Llama 4 支持 10M tokens" 和 "GPT-4 支持 128K tokens"

锚点字段（数字、专名）提供了硬约束，确保召回的是精确匹配的节点。

### 为什么按需证据回灌有效？

大部分查询只需要"大概知道"，用摘要即可：
- "RoPE 是什么？" → 摘要足够

少数查询需要"精确引用"，才回灌原文：
- "RoPE 的精确公式是什么？" → 需要原文

这种按需策略在精度和成本间取得平衡。

## 限制和改进方向

### 当前限制

1. **Embedding 质量**：当前使用简化的 hash 嵌入，生产环境应使用 sentence-transformers
2. **字段提取**：基于 regex，可能漏掉复杂实体
3. **单机索引**：MemoryMap 是简化实现，大规模需要 Faiss/Milvus
4. **无分布式**：未实现多机分布式记忆

### 改进方向

1. **更强的嵌入**：集成 BGE/E5 等专业 embedding 模型
2. **NER 字段提取**：使用命名实体识别提取更准确的字段
3. **向量索引**：集成 Faiss/Milvus 支持亿级节点
4. **分布式记忆**：实现记忆分片和同步机制
5. **自适应参数**：根据使用模式自动调整 λ、γ、K 等参数

## 测试

运行完整测试套件：

```bash
python training/test_aim_memory.py
```

测试覆盖：
1. ✅ 基础写入和读取
2. ✅ 惯性路由机制
3. ✅ 时间镜像衰减
4. ✅ 锚点纠错
5. ✅ 按需证据回灌
6. ✅ 完整回答生成
7. ✅ 持久化（保存/加载）
8. ✅ 端到端场景
9. ✅ 统计信息

## 引用和致谢

**技术来源**：
- 作者：430
- 实现：Claude + 430
- 版本：2026-01-21

**理论基础**：
- 惯性路由：借鉴动量优化和联想记忆
- 时间镜像：借鉴 Ebbinghaus 遗忘曲线
- 锚点纠错：借鉴知识图谱的实体链接
- 证据回灌：借鉴分层存储系统

## 相关文档

- [分层记忆系统指南](HIERARCHICAL_MEMORY_GUIDE.md)
- [集成总结文档](INTEGRATION_SUMMARY.md)
- [APT-Transformer 技术总结](APT_TRANSFORMER_SUMMARY.md)

## FAQ

**Q: AIM-Memory 适合什么场景？**

A: 特别适合需要长期记忆的对话系统、知识助手、个人 AI 助理等。不适合纯粹的一次性问答。

**Q: 与 RAG 的主要区别？**

A: RAG 是"先搜一堆文档"，AIM-Memory 是"先到货架再挑商品"。AIM-Memory 有惯性导航，成本更低。

**Q: 需要多少节点才有效？**

A: 理论上 100+ 节点就能看到效果，1000+ 节点效果更明显，支持百万级节点（需要专业向量索引）。

**Q: 如何调优参数？**

A: 建议从默认参数开始，根据以下原则调整：
- 对话连续性强 → 提高 `inertia_strength`
- 需要长期记忆 → 提高 `weight_decay_gamma`（接近 1.0）
- 需要高精度 → 提高 `anchor_boost`
- 成本敏感 → 降低 `local_cluster_k`

**Q: 能否与现有 RAG 系统共存？**

A: 可以。AIM-Memory 可以作为 RAG 的上层路由，先用惯性定位到小簇，再在小簇内做精细 RAG。
