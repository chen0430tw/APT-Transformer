# APT模型知识图谱使用指南

## 📋 概述

APT模型现已集成轻量级知识图谱（Knowledge Graph, KG）功能，结合结构化知识和非结构化文档，显著提升生成质量。

**核心功能:**
- ✅ 基于三元组的知识存储（实体-关系-实体）
- ✅ 快速检索和查询
- ✅ 多跳推理
- ✅ 与RAG无缝集成
- ✅ 轻量级设计，易于使用

---

## 🚀 快速开始

### 1. 创建知识图谱

```python
from apt_model.modeling.knowledge_graph import KnowledgeGraph

# 创建空图谱
kg = KnowledgeGraph()

# 添加三元组
kg.add_triple("人工智能", "是", "计算机科学的分支")
kg.add_triple("深度学习", "是", "机器学习的子领域")
kg.add_triple("深度学习", "用于", "图像识别")

print(kg)  # KnowledgeGraph(entities=4, relations=2, triples=3)
```

### 2. 从文件加载

```python
from apt_model.modeling.knowledge_graph import create_kg_from_file

# 文件格式（每行一个三元组，用Tab分隔）:
# 人工智能\t是\t计算机科学的分支
# 深度学习\t是\t机器学习的子领域

kg = create_kg_from_file("knowledge.txt", separator="\t")
```

### 3. 查询知识

```python
# 查询给定头实体的所有关系
triples = kg.query_by_head("深度学习")
for t in triples:
    print(f"{t.head} {t.relation} {t.tail}")

# 获取邻居实体
neighbors = kg.get_neighbors("深度学习")
print(neighbors)  # ['机器学习的子领域', '图像识别']

# 多跳推理
paths = kg.multi_hop_query("深度学习", ["是", "属于"])
```

---

## 💡 核心概念

### 三元组（Triple）

知识图谱的基本单位：**(头实体, 关系, 尾实体)**

```python
from apt_model.modeling.knowledge_graph import Triple

triple = Triple(
    head="GPT",
    relation="是",
    tail="大语言模型",
    confidence=0.95,  # 置信度（可选）
    metadata={"source": "论文"}  # 元数据（可选）
)
```

### 索引结构

知识图谱维护三种索引以实现快速查询：

1. **head_index**: 头实体 → 三元组列表
2. **relation_index**: 关系 → 三元组列表
3. **tail_index**: 尾实体 → 三元组列表

---

## 🔍 查询操作

### 1. 基础查询

```python
# 按头实体查询
triples = kg.query_by_head("人工智能")

# 按关系查询
triples = kg.query_by_relation("是")

# 按尾实体反向查询
triples = kg.query_by_tail("计算机科学的分支")

# 组合查询（头+关系 → 尾）
tails = kg.query_by_head_relation("深度学习", "用于")
print(tails)  # ['图像识别', ...]
```

### 2. 邻居查询

```python
# 获取所有邻居（不限关系）
neighbors = kg.get_neighbors("深度学习")

# 获取特定关系的邻居
neighbors = kg.get_neighbors("深度学习", relation="是")
```

### 3. 多跳推理

```python
# 查找从"深度学习"出发，依次经过"是"和"用于"关系的路径
paths = kg.multi_hop_query(
    start_entity="深度学习",
    relations=["是", "用于"],
    max_results=10
)

for path in paths:
    print(" -> ".join(path))
# 输出: 深度学习 -> 机器学习 -> 数据分析
```

### 4. 路径查找

```python
# 查找两个实体之间的所有路径
paths = kg.find_paths(
    start="深度学习",
    end="人工智能",
    max_hops=3
)

for path in paths:
    for triple in path:
        print(f"  {triple.head} --[{triple.relation}]--> {triple.tail}")
```

### 5. 子图提取

```python
# 提取以特定实体为中心的子图
subgraph = kg.get_subgraph(
    entities=["深度学习", "机器学习"],
    max_hops=2
)

print(subgraph)  # KnowledgeGraph(entities=..., relations=..., triples=...)
```

---

## 🎯 与RAG集成

### 方法1: 使用KG-RAG集成模块

```python
from apt_model.modeling.kg_rag_integration import create_kg_rag_model
from apt_model.training.checkpoint import load_model

# 加载基础模型
model, tokenizer, config = load_model("apt_model")

# 创建KG-RAG模型
kg_rag_model = create_kg_rag_model(
    base_model=model,
    kg_path="knowledge.json",  # 知识图谱文件
    corpus_path="documents.txt",  # 文档语料
    fusion_method="weighted",  # 融合方法
    kg_weight=0.6,  # KG权重
    rag_weight=0.4  # RAG权重
)

# 构建索引
kg_triples = [
    ("深度学习", "是", "机器学习的分支"),
    ("Transformer", "是", "深度学习架构"),
    # ...
]
kg_rag_model.build_kg_index(kg_triples)

corpus = ["深度学习是机器学习的一个分支...", "Transformer模型..."]
kg_rag_model.build_rag_index(corpus)

# 使用模型
outputs = kg_rag_model(input_ids, attention_mask)
print("KG知识:", outputs['kg_knowledge'])
print("RAG文档:", outputs['rag_docs'])
print("融合上下文:", outputs['fused_context'])
```

### 方法2: 快速创建

```python
from apt_model.modeling.kg_rag_integration import quick_kg_rag

# 准备数据
kg_triples = [
    ("人工智能", "是", "计算机科学"),
    ("深度学习", "是", "机器学习"),
]

corpus = [
    "人工智能是计算机科学的一个重要分支",
    "深度学习在图像识别中表现出色"
]

# 快速创建
model = quick_kg_rag(
    model=base_model,
    kg_triples=kg_triples,
    corpus=corpus
)
```

---

## 📊 存储和加载

### 保存知识图谱

```python
# 保存为JSON（可读）
kg.save("knowledge.json")

# 保存为Pickle（更快）
kg.save("knowledge.pkl")
```

**JSON格式示例:**
```json
{
  "triples": [
    {
      "head": "人工智能",
      "relation": "是",
      "tail": "计算机科学的分支",
      "confidence": 1.0,
      "metadata": null
    }
  ],
  "entities": ["人工智能", "计算机科学的分支"],
  "relations": ["是"]
}
```

### 加载知识图谱

```python
from apt_model.modeling.knowledge_graph import KnowledgeGraph

# 从文件加载
kg = KnowledgeGraph.load("knowledge.json")

# 或
kg = KnowledgeGraph.load("knowledge.pkl")
```

---

## 🔧 高级功能

### 1. 批量添加三元组

```python
triples = [
    ("A", "关系1", "B"),
    ("B", "关系2", "C"),
    ("C", "关系3", "D", 0.9),  # 带置信度
]

kg.add_triples_batch(triples)
```

### 2. 转换为文本

```python
# 自然语言格式
text = kg.to_text(format='natural')
print(text)
# 输出:
# 人工智能 是 计算机科学的分支
# 深度学习 是 机器学习的子领域

# 结构化格式
text = kg.to_text(format='structured')
print(text)
# 输出:
# (人工智能, 是, 计算机科学的分支)
# (深度学习, 是, 机器学习的子领域)
```

### 3. 统计信息

```python
stats = kg.stats()
print(stats)
# {
#   'num_entities': 10,
#   'num_relations': 5,
#   'num_triples': 20,
#   'avg_degree': 4.0,
#   'relations_list': ['是', '有', '用于', ...]
# }
```

### 4. 从文本提取三元组

```python
from apt_model.modeling.knowledge_graph import extract_triples_from_text

text = "深度学习是机器学习的子领域。Transformer是深度学习架构。"
triples = extract_triples_from_text(text)

for triple in triples:
    print(triple)
# ('深度学习', '是', '机器学习的子领域')
# ('Transformer', '是', '深度学习架构')
```

**注意**: 这是基于规则的简单提取，复杂场景建议使用专门的信息抽取模型。

---

## 🎨 使用场景

### 场景1: 领域知识增强

```python
# 医疗领域知识图谱
medical_kg = KnowledgeGraph()
medical_kg.add_triple("阿司匹林", "用于治疗", "头痛")
medical_kg.add_triple("阿司匹林", "属于", "非甾体抗炎药")
medical_kg.add_triple("头痛", "是", "神经系统症状")

# 查询药物相关知识
treatments = medical_kg.query_by_relation("用于治疗")
```

### 场景2: 问答系统

```python
# 用户问题: "深度学习可以用于什么？"
# 1. 提取实体: "深度学习"
# 2. 查询KG
applications = kg.query_by_head_relation("深度学习", "用于")
print("深度学习可以用于:", ", ".join(applications))
```

### 场景3: 推理增强

```python
# 多跳推理: "什么技术属于人工智能并且用于图像识别？"

# 找到属于人工智能的技术
ai_techs = kg.query_by_relation_tail("属于", "人工智能")

# 筛选用于图像识别的
for tech in ai_techs:
    uses = kg.query_by_head_relation(tech, "用于")
    if "图像识别" in uses:
        print(f"{tech} 属于人工智能且用于图像识别")
```

### 场景4: 知识补全

```python
# 推断缺失的关系
# 已知: A -> B, B -> C
# 推断: A -> C (传递关系)

paths = kg.find_paths("A", "C", max_hops=2)
if paths:
    print("存在间接关系，可以补全")
```

---

## 📚 数据格式

### 三元组文件格式

```
# 格式: 头实体<Tab>关系<Tab>尾实体
人工智能	是	计算机科学的分支
深度学习	是	机器学习的子领域
深度学习	用于	图像识别
Transformer	是	深度学习架构
BERT	基于	Transformer
BERT	用于	自然语言处理
```

### JSON格式

```json
{
  "triples": [
    {
      "head": "人工智能",
      "relation": "是",
      "tail": "计算机科学的分支",
      "confidence": 1.0,
      "metadata": {"source": "教科书"}
    }
  ]
}
```

---

## 🛠️ API参考

### KnowledgeGraph 类

**初始化:**
```python
kg = KnowledgeGraph()
```

**主要方法:**

| 方法 | 说明 |
|------|------|
| `add_triple(head, relation, tail, confidence, metadata)` | 添加三元组 |
| `add_triples_batch(triples)` | 批量添加 |
| `query_by_head(head)` | 按头实体查询 |
| `query_by_relation(relation)` | 按关系查询 |
| `query_by_tail(tail)` | 按尾实体查询 |
| `get_neighbors(entity, relation)` | 获取邻居 |
| `multi_hop_query(start, relations, max_results)` | 多跳查询 |
| `find_paths(start, end, max_hops)` | 路径查找 |
| `get_subgraph(entities, max_hops)` | 提取子图 |
| `save(filepath)` | 保存到文件 |
| `load(filepath)` | 从文件加载（静态方法） |
| `stats()` | 获取统计信息 |

### KGRAGWrapper 类

**创建:**
```python
from apt_model.modeling.kg_rag_integration import create_kg_rag_model

model = create_kg_rag_model(
    base_model=model,
    kg_path="kg.json",
    corpus_path="docs.txt",
    fusion_method="weighted"
)
```

**主要方法:**

| 方法 | 说明 |
|------|------|
| `build_kg_index(triples)` | 构建KG索引 |
| `build_rag_index(corpus, embedding_model)` | 构建RAG索引 |
| `retrieve(query, use_kg, use_rag)` | 检索知识 |
| `forward(input_ids, attention_mask, ...)` | 前向传播 |

---

## ⚠️ 注意事项

### 1. 规模限制

轻量级KG适合：
- ✅ 实体数: < 100,000
- ✅ 三元组数: < 1,000,000
- ✅ 内存占用: ~100MB-1GB

**大规模图谱建议使用专业图数据库（Neo4j, ArangoDB等）**

### 2. 性能优化

```python
# 批量添加比逐个添加快得多
kg.add_triples_batch(triples)  # ✅ 推荐

for triple in triples:  # ❌ 避免
    kg.add_triple(*triple)
```

### 3. 查询优化

```python
# 使用索引查询（快）
triples = kg.query_by_head("实体")  # O(1)

# 遍历所有三元组（慢）
triples = [t for t in kg.triples if t.head == "实体"]  # O(n)
```

---

## 🎯 最佳实践

### 1. 设计高质量的知识图谱

- ✅ 使用一致的命名规范
- ✅ 关系名称应清晰明确
- ✅ 添加置信度和来源信息
- ✅ 定期清理和更新

### 2. KG与RAG的权衡

**使用KG的场景:**
- 需要精确的实体关系
- 需要多跳推理
- 知识相对结构化

**使用RAG的场景:**
- 需要丰富的上下文
- 知识以自然语言形式存在
- 需要灵活性

**KG+RAG融合:**
- 最佳选择：结合两者优势
- KG提供结构化骨架
- RAG提供详细内容

### 3. 融合方法选择

| 方法 | 适用场景 | 优点 | 缺点 |
|------|----------|------|------|
| `concatenate` | 简单场景 | 简单直接 | 可能冗长 |
| `weighted` | 通用场景 | 平衡两者 | 需调整权重 |
| `gate` | 复杂场景 | 自适应 | 计算开销大 |

---

## 🔬 示例代码

### 完整示例: 构建医疗知识问答系统

```python
from apt_model.modeling.knowledge_graph import KnowledgeGraph
from apt_model.modeling.kg_rag_integration import create_kg_rag_model
from apt_model.training.checkpoint import load_model

# 1. 创建医疗知识图谱
medical_kg = KnowledgeGraph()

medical_triples = [
    ("感冒", "症状包括", "发热"),
    ("感冒", "症状包括", "咳嗽"),
    ("感冒", "可用药", "对乙酰氨基酚"),
    ("对乙酰氨基酚", "用于", "退热"),
    ("对乙酰氨基酚", "用于", "止痛"),
]

medical_kg.add_triples_batch(medical_triples)
medical_kg.save("medical_kg.json")

# 2. 准备医疗文档语料
medical_docs = [
    "感冒是一种常见的呼吸道疾病，主要症状包括发热、咳嗽、流鼻涕等。",
    "对乙酰氨基酚是一种常用的解热镇痛药，用于缓解感冒引起的发热和头痛。",
    "治疗感冒应注意多休息、多喝水，必要时可服用退热药物。"
]

# 3. 加载模型并创建KG-RAG系统
base_model, tokenizer, config = load_model("apt_model")

kg_rag_model = create_kg_rag_model(
    base_model=base_model,
    kg=medical_kg,
    corpus=medical_docs,
    fusion_method="weighted",
    kg_weight=0.7,  # KG权重高一些，因为医疗知识需要精确性
    rag_weight=0.3
)

# 4. 问答
question = "感冒有哪些症状？"
# 编码问题、检索知识、生成答案...
```

---

## 📞 技术支持

- **完整文档**: [README.md](./README.md)
- **RAG指南**: [rag_integration.py](apt_model/modeling/rag_integration.py)
- **问题反馈**: GitHub Issues

---

## 🎓 参考资料

- [Knowledge Graph基础](https://en.wikipedia.org/wiki/Knowledge_graph)
- [RAG论文](https://arxiv.org/abs/2005.11401)
- [Neo4j图数据库](https://neo4j.com/)

---

**Happy Knowledge Graphing! 🚀**
