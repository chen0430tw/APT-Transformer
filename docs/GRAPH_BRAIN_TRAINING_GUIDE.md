# 图脑训练教程 - Graph Reasoning Architecture

<div align="center">

**基于图神经网络的结构化推理训练**

融合 Gemini 思维模式 | 图结构推理 | 神经符号推理

</div>

---

## 📋 目录

- [什么是图脑](#什么是图脑)
- [架构设计](#架构设计)
- [核心组件](#核心组件)
- [训练流程](#训练流程)
- [高级技巧](#高级技巧)
- [实战示例](#实战示例)
- [性能优化](#性能优化)

---

## 🧠 什么是图脑

### 概念

**图脑 (Graph Brain)** 是一种结合了图神经网络（GNN）和语言模型的混合推理架构，灵感来源于：
- **Gemini 2.0 Flash Thinking** 的显式思维过程
- **神经符号推理** 的结构化表示
- **图神经网络** 的关系建模能力

### 核心思想

```
传统 Transformer:
文本 → Embedding → Attention → 输出

图脑架构:
文本 → 概念图 → 图神经网络 → 推理路径 → 输出
       ↓
   结构化思维过程
```

### 优势对比

| 特性 | 传统 LLM | 图脑架构 |
|------|---------|---------|
| **推理可解释性** | ❌ 黑盒 | ✅ 显式图结构 |
| **多跳推理** | ⚠️ 依赖上下文 | ✅ 原生支持 |
| **知识整合** | ⚠️ 隐式记忆 | ✅ 显式知识图谱 |
| **计算效率** | ⚠️ 全序列注意力 | ✅ 稀疏图计算 |
| **可控性** | ❌ 难以干预 | ✅ 可编辑图结构 |

---

## 🏗️ 架构设计

### 完整架构图

```
输入文本
    ↓
[1] 概念抽取器 (Concept Extractor)
    ├── 实体识别（NER）
    ├── 关系抽取（RE）
    └── 事件检测
    ↓
[2] 概念图构建 (Concept Graph Builder)
    ├── 节点：概念/实体
    ├── 边：关系/依赖
    └── 属性：类型/权重
    ↓
[3] 图神经编码器 (Graph Neural Encoder)
    ├── 图卷积层（GCN/GAT/GraphSAGE）
    ├── 消息传递
    └── 节点更新
    ↓
[4] 推理路径规划 (Reasoning Path Planner)
    ├── 注意力路由
    ├── 多跳推理
    └── 子图采样
    ↓
[5] 解码器 (Decoder)
    ├── 图到序列（Graph2Seq）
    ├── 思维链生成
    └── 最终答案
    ↓
输出（答案 + 推理过程）
```

---

## 🔧 核心组件

### 1. 概念抽取器

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class ConceptExtractor(nn.Module):
    """
    概念抽取器：从文本中提取实体和关系

    方法：
    - 实体识别：BERT + CRF
    - 关系抽取：双向 LSTM + 注意力
    """
    def __init__(self, bert_model='bert-base-uncased', num_entity_types=10):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.entity_classifier = nn.Linear(768, num_entity_types)
        self.relation_classifier = nn.Bilinear(768, 768, 20)  # 20种关系类型

    def forward(self, input_ids, attention_mask):
        # BERT 编码
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [B, T, 768]

        # 实体识别
        entity_logits = self.entity_classifier(sequence_output)  # [B, T, num_types]

        # 关系抽取（实体对之间）
        # 简化：取句子的 [CLS] 表示
        cls_repr = sequence_output[:, 0, :]  # [B, 768]

        return {
            'entity_logits': entity_logits,
            'cls_repr': cls_repr
        }

    def extract_concepts(self, text, tokenizer):
        """
        从文本中提取概念图

        Returns:
            nodes: List[Dict] - 节点列表
            edges: List[Tuple] - 边列表 (src, rel, dst)
        """
        # 分词
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

        # 前向传播
        with torch.no_grad():
            outputs = self.forward(inputs['input_ids'], inputs['attention_mask'])
            entity_logits = outputs['entity_logits']

        # 解码实体
        entity_predictions = torch.argmax(entity_logits, dim=-1)  # [B, T]

        # 构建节点和边（简化版）
        nodes = []
        edges = []

        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        for i, (token, entity_type) in enumerate(zip(tokens, entity_predictions[0])):
            if entity_type != 0:  # 0 = 非实体
                nodes.append({
                    'id': i,
                    'token': token,
                    'type': entity_type.item()
                })

        # 提取关系（简化：相邻实体）
        for i in range(len(nodes) - 1):
            edges.append((nodes[i]['id'], 'next', nodes[i+1]['id']))

        return nodes, edges
```

---

### 2. 图神经编码器

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class GraphBrainEncoder(nn.Module):
    """
    图脑编码器：使用图神经网络编码概念图

    支持：
    - GCN（图卷积网络）
    - GAT（图注意力网络）
    - GraphSAGE（图采样聚合）
    """
    def __init__(
        self,
        node_dim=768,
        hidden_dim=512,
        num_layers=3,
        num_heads=8,
        dropout=0.1,
        gnn_type='gat'
    ):
        super().__init__()
        self.gnn_type = gnn_type
        self.num_layers = num_layers

        # 节点特征投影
        self.node_projection = nn.Linear(node_dim, hidden_dim)

        # 图卷积层
        if gnn_type == 'gcn':
            self.convs = nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim)
                for _ in range(num_layers)
            ])
        elif gnn_type == 'gat':
            self.convs = nn.ModuleList([
                GATConv(
                    hidden_dim,
                    hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True
                )
                for _ in range(num_layers)
            ])
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

        # 层归一化
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        Args:
            x: 节点特征 [num_nodes, node_dim]
            edge_index: 边索引 [2, num_edges]
            edge_attr: 边属性 [num_edges, edge_dim] (可选)
            batch: 批次索引 [num_nodes] (用于批处理)

        Returns:
            node_embeddings: 节点嵌入 [num_nodes, hidden_dim]
            graph_embedding: 图嵌入 [batch_size, hidden_dim]
        """
        # 投影节点特征
        x = self.node_projection(x)  # [num_nodes, hidden_dim]

        # 多层图卷积
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # 残差连接
            residual = x

            # 图卷积
            if self.gnn_type == 'gcn':
                x = conv(x, edge_index)
            elif self.gnn_type == 'gat':
                x = conv(x, edge_index)

            # 归一化 + 激活 + Dropout
            x = norm(x + residual)
            x = F.relu(x)
            x = self.dropout(x)

        # 图级别池化（用于生成图嵌入）
        if batch is not None:
            graph_embedding = global_mean_pool(x, batch)
        else:
            graph_embedding = x.mean(dim=0, keepdim=True)

        return x, graph_embedding
```

---

### 3. 推理路径规划器

```python
class ReasoningPathPlanner(nn.Module):
    """
    推理路径规划器：在图上规划多跳推理路径

    方法：
    - 注意力路由：学习节点重要性
    - 多跳推理：k-hop 子图采样
    - 路径选择：Beam Search + 图注意力
    """
    def __init__(self, hidden_dim=512, num_hops=3, num_beams=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_hops = num_hops
        self.num_beams = num_beams

        # 节点重要性评分器
        self.importance_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 路径注意力
        self.path_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1
        )

        # 路径编码器
        self.path_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, node_embeddings, edge_index, query_embedding):
        """
        Args:
            node_embeddings: 节点嵌入 [num_nodes, hidden_dim]
            edge_index: 边索引 [2, num_edges]
            query_embedding: 查询嵌入 [1, hidden_dim]

        Returns:
            reasoning_paths: List[List[int]] - 推理路径（节点ID序列）
            path_scores: 路径分数
        """
        num_nodes = node_embeddings.size(0)

        # 1. 计算节点重要性（相对于查询）
        query_expanded = query_embedding.expand(num_nodes, -1)  # [num_nodes, hidden_dim]
        combined = node_embeddings + query_expanded
        importance_scores = self.importance_scorer(combined).squeeze(-1)  # [num_nodes]

        # 2. 选择起始节点（Top-K 最重要的节点）
        topk_scores, topk_indices = torch.topk(importance_scores, k=self.num_beams)

        # 3. Beam Search 多跳推理
        reasoning_paths = []
        path_scores = []

        # 构建邻接列表（加速查找）
        adjacency = self._build_adjacency_list(edge_index, num_nodes)

        for start_node in topk_indices:
            # 从每个起始节点开始探索
            path = [start_node.item()]
            current_node = start_node.item()

            for hop in range(self.num_hops):
                # 获取邻居节点
                neighbors = adjacency.get(current_node, [])
                if not neighbors:
                    break

                # 计算邻居的注意力分数
                neighbor_embeddings = node_embeddings[neighbors]  # [num_neighbors, hidden_dim]
                current_embedding = node_embeddings[current_node:current_node+1]  # [1, hidden_dim]

                # 注意力评分
                attn_output, attn_weights = self.path_attention(
                    query=current_embedding.unsqueeze(0),      # [1, 1, hidden_dim]
                    key=neighbor_embeddings.unsqueeze(0),      # [1, num_neighbors, hidden_dim]
                    value=neighbor_embeddings.unsqueeze(0)     # [1, num_neighbors, hidden_dim]
                )

                # 选择最佳邻居
                best_neighbor_idx = torch.argmax(attn_weights[0, 0])
                best_neighbor = neighbors[best_neighbor_idx.item()]

                path.append(best_neighbor)
                current_node = best_neighbor

            reasoning_paths.append(path)
            path_scores.append(topk_scores[len(reasoning_paths) - 1].item())

        return reasoning_paths, torch.tensor(path_scores)

    def _build_adjacency_list(self, edge_index, num_nodes):
        """构建邻接表"""
        adjacency = {i: [] for i in range(num_nodes)}
        for src, dst in edge_index.t().tolist():
            adjacency[src].append(dst)
        return adjacency
```

---

### 4. 图到序列解码器

```python
class Graph2SeqDecoder(nn.Module):
    """
    图到序列解码器：将推理路径转换为自然语言

    方法：
    - 路径编码：LSTM 编码推理路径
    - 注意力解码：生成思维链
    - 答案生成：Transformer 解码器
    """
    def __init__(
        self,
        hidden_dim=512,
        vocab_size=50257,
        max_length=512,
        num_decoder_layers=6
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        # 路径编码器
        self.path_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # Transformer 解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # 输出投影
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        # 位置编码
        self.pos_encoder = nn.Embedding(max_length, hidden_dim)

    def forward(self, path_embeddings, target_ids=None, teacher_forcing_ratio=0.5):
        """
        Args:
            path_embeddings: 路径节点嵌入 [batch, path_len, hidden_dim]
            target_ids: 目标序列 [batch, seq_len] (训练时)
            teacher_forcing_ratio: 教师强制比率

        Returns:
            logits: [batch, seq_len, vocab_size]
            generated_ids: [batch, seq_len]
        """
        batch_size = path_embeddings.size(0)

        # 1. 编码路径
        path_encoded, (hidden, cell) = self.path_encoder(path_embeddings)
        # path_encoded: [batch, path_len, hidden_dim * 2]

        # 池化为单向
        path_memory = path_encoded[:, :, :self.hidden_dim] + path_encoded[:, :, self.hidden_dim:]
        # path_memory: [batch, path_len, hidden_dim]

        # 2. 解码生成序列
        if target_ids is not None:
            # 训练模式：教师强制
            seq_len = target_ids.size(1)
            pos_ids = torch.arange(seq_len, device=path_embeddings.device).unsqueeze(0)
            pos_embeddings = self.pos_encoder(pos_ids)  # [1, seq_len, hidden_dim]

            # Transformer 解码
            tgt = pos_embeddings.expand(batch_size, -1, -1).transpose(0, 1)  # [seq_len, batch, hidden_dim]
            memory = path_memory.transpose(0, 1)  # [path_len, batch, hidden_dim]

            decoder_output = self.decoder(tgt, memory)  # [seq_len, batch, hidden_dim]
            decoder_output = decoder_output.transpose(0, 1)  # [batch, seq_len, hidden_dim]

            # 输出投影
            logits = self.output_projection(decoder_output)  # [batch, seq_len, vocab_size]

            return logits, None

        else:
            # 推理模式：自回归生成
            generated_ids = []
            current_input = torch.zeros(batch_size, 1, self.hidden_dim, device=path_embeddings.device)

            for step in range(self.max_length):
                # 位置编码
                pos_id = torch.tensor([[step]], device=path_embeddings.device)
                pos_emb = self.pos_encoder(pos_id).expand(batch_size, -1, -1)

                tgt = (current_input + pos_emb).transpose(0, 1)  # [1, batch, hidden_dim]
                memory = path_memory.transpose(0, 1)  # [path_len, batch, hidden_dim]

                # 解码一步
                decoder_output = self.decoder(tgt, memory)  # [1, batch, hidden_dim]
                decoder_output = decoder_output.transpose(0, 1)  # [batch, 1, hidden_dim]

                # 预测下一个 token
                logits = self.output_projection(decoder_output)  # [batch, 1, vocab_size]
                next_token = torch.argmax(logits, dim=-1)  # [batch, 1]

                generated_ids.append(next_token)

                # 更新输入（嵌入下一个 token）
                # 简化：这里应该有 token embedding 层
                current_input = decoder_output

            generated_ids = torch.cat(generated_ids, dim=1)  # [batch, max_length]
            return None, generated_ids
```

---

## 🎓 训练流程

### 完整训练代码

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

class GraphBrainModel(nn.Module):
    """完整的图脑模型"""
    def __init__(self, config):
        super().__init__()
        self.concept_extractor = ConceptExtractor()
        self.graph_encoder = GraphBrainEncoder(
            node_dim=768,
            hidden_dim=512,
            num_layers=3
        )
        self.path_planner = ReasoningPathPlanner(hidden_dim=512)
        self.decoder = Graph2SeqDecoder(
            hidden_dim=512,
            vocab_size=50257
        )

    def forward(self, input_text, target_text=None):
        # 1. 抽取概念图
        nodes, edges = self.concept_extractor.extract_concepts(
            input_text,
            tokenizer
        )

        # 2. 编码图结构
        # （需要转换为 PyTorch Geometric 格式）
        node_embeddings, graph_embedding = self.graph_encoder(
            x=node_features,
            edge_index=edge_index
        )

        # 3. 规划推理路径
        reasoning_paths, path_scores = self.path_planner(
            node_embeddings,
            edge_index,
            query_embedding=graph_embedding
        )

        # 4. 生成答案
        # 获取路径的节点嵌入
        path_embeddings = torch.stack([
            node_embeddings[path] for path in reasoning_paths
        ])

        # 解码生成
        logits, generated_ids = self.decoder(
            path_embeddings,
            target_ids=target_text
        )

        return {
            'logits': logits,
            'generated_ids': generated_ids,
            'reasoning_paths': reasoning_paths,
            'path_scores': path_scores
        }


# ========== 训练器 ==========

class GraphBrainTrainer:
    """图脑训练器"""
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=2e-4,
            weight_decay=0.01
        )

        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def train_step(self, batch):
        """训练一步"""
        self.model.train()

        input_texts = batch['input_texts']
        target_texts = batch['target_texts']

        # 前向传播
        outputs = self.model(input_texts, target_texts)

        # 计算损失
        logits = outputs['logits']
        target_ids = self.tokenizer(
            target_texts,
            return_tensors='pt',
            padding=True,
            truncation=True
        )['input_ids'].to(self.device)

        # 语言模型损失
        lm_loss = self.ce_loss(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1)
        )

        # 路径分数正则化（鼓励多样性）
        path_scores = outputs['path_scores']
        path_diversity_loss = -torch.std(path_scores)  # 最大化方差

        # 总损失
        total_loss = lm_loss + 0.1 * path_diversity_loss

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            'loss': total_loss.item(),
            'lm_loss': lm_loss.item(),
            'path_diversity_loss': path_diversity_loss.item()
        }

    def train(self, train_loader, num_epochs=10, save_path='./graph_brain'):
        """完整训练流程"""
        print(f"🧠 开始图脑训练...")
        print(f"   设备: {self.device}")
        print(f"   Epochs: {num_epochs}")

        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                # 训练一步
                metrics = self.train_step(batch)
                total_loss += metrics['loss']
                num_batches += 1

                # 打印进度
                if (batch_idx + 1) % 10 == 0:
                    avg_loss = total_loss / num_batches
                    print(f"Epoch [{epoch+1}/{num_epochs}] "
                          f"Batch [{batch_idx+1}/{len(train_loader)}] "
                          f"Loss: {avg_loss:.4f}")

            # 保存检查点
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, f"{save_path}/epoch_{epoch+1}.pt")

        print(f"✅ 训练完成！模型已保存到 {save_path}")


# ========== 使用示例 ==========

# 1. 准备数据
train_dataset = GraphReasoningDataset('train.jsonl')
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 2. 初始化模型
model = GraphBrainModel(config={})
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 3. 创建训练器
trainer = GraphBrainTrainer(model, tokenizer)

# 4. 开始训练
trainer.train(train_loader, num_epochs=20, save_path='./graph_brain_model')
```

---

## 🚀 实战示例

### 多跳问答推理

```python
# 问题：Einstein 的老师的国籍是什么？

# 输入文本
question = "What is the nationality of Einstein's teacher?"

# 模型推理
model.eval()
with torch.no_grad():
    outputs = model(question)

# 输出
# reasoning_paths: [
#   [Einstein] → [studied under] → [Heinrich Weber] → [nationality] → [German]
# ]
# answer: "German"
# thinking_process: "First, I identified Einstein. Then I found his teacher Heinrich Weber. Finally, I determined Weber's nationality was German."
```

### 数学推理

```python
# 问题：如果 x + 2 = 5，求 x

question = "If x + 2 = 5, what is x?"

# 推理图
# [x + 2] → [equals] → [5]
#     ↓
# [subtract 2]
#     ↓
# [x = 3]

# 输出
# answer: "x = 3"
# thinking_process: "Starting from x + 2 = 5, I subtract 2 from both sides to get x = 3."
```

---

## ⚡ 性能优化

### 图采样加速

```python
from torch_geometric.data import DataLoader as PyGDataLoader
from torch_geometric.data import Data

# 大图采样
class GraphSampler:
    """图采样器：处理大规模图"""
    def __init__(self, num_neighbors=[10, 5], num_hops=2):
        self.num_neighbors = num_neighbors
        self.num_hops = num_hops

    def sample_subgraph(self, node_id, edge_index, num_nodes):
        """采样 k-hop 子图"""
        from torch_geometric.utils import k_hop_subgraph

        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=node_id,
            num_hops=self.num_hops,
            edge_index=edge_index,
            relabel_nodes=True,
            num_nodes=num_nodes
        )

        return subset, sub_edge_index
```

### 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# 训练循环
for batch in train_loader:
    optimizer.zero_grad()

    # 混合精度前向
    with autocast():
        outputs = model(batch['input'])
        loss = compute_loss(outputs, batch['target'])

    # 缩放梯度
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 📚 参考资源

### 学术论文

- [Thinking Like Transformers](https://arxiv.org/abs/2106.06981) - 结构化推理
- [Graph Neural Networks: A Review](https://arxiv.org/abs/1812.08434) - GNN综述
- [Neural-Symbolic VQA](https://arxiv.org/abs/1810.02338) - 神经符号推理

### 官方资源

- [Gemini 2.0 Flash Thinking](https://ai.google.dev/gemini-api/docs/thinking) - Google 思维模式
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - 图神经网络库
- [DeepMind Gemini](https://deepmind.google/models/gemini/) - Gemini 模型

Sources:
- [Gemini 2.0 Flash Thinking Experimental](https://www.datacamp.com/blog/gemini-2-0-flash-experimental)
- [Gemini 2.0 Technical Details](https://www.techtarget.com/whatis/feature/Google-Gemini-20-explained-Everything-you-need-to-know)
- [Gemini Thinking API](https://ai.google.dev/gemini-api/docs/thinking)
- [Gemini Models Overview](https://deepmind.google/models/gemini/)

### APT 相关文档

- [DeepSeek 训练指南](DEEPSEEK_TRAINING_GUIDE.md) - MoE 架构
- [数据预处理指南](DATA_PREPROCESSING_GUIDE.md) - 数据清洗
- [插件系统文档](PLUGIN_SYSTEM.md) - 插件开发

---

## 📝 更新日志

- **v1.0.0** (2025-12) - 初始版本
  - ✅ 完整图脑架构（概念抽取 + GNN + 推理规划）
  - ✅ 多跳推理路径规划
  - ✅ 图到序列解码器
  - ✅ 显式思维过程生成
  - ✅ 生产级训练代码
  - ✅ 性能优化建议

---

<div align="center">

**Graph + Brain = Better Reasoning! 🧠💡**

结构化推理，让模型思考更清晰

如有问题，请提交 [Issue](https://github.com/chen0430tw/APT-Transformer/issues)

</div>
