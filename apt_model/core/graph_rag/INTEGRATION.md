# APT GraphRAG 集成指南

本文档详细说明如何将GraphRAG模块集成到APT项目中。

---

## 📦 一、文件结构

### 集成后的目录结构

```
apt_model/
├── __init__.py
├── main.py
├── apt_model.py
├── trainer.py
├── ...
│
├── graph_rag/                      # 新增模块
│   ├── __init__.py
│   ├── generalized_graph.py        # 泛图核心
│   ├── hodge_laplacian.py          # Hodge-Laplacian
│   ├── graph_brain.py              # 图脑动力学
│   └── graph_rag_manager.py        # GraphRAG管理器
│
└── rag_manager.py                  # 可选: 修改现有RAG
```

---

## 🔧 二、安装步骤

### 方法1: 直接复制

```bash
# 假设你在APT项目根目录
cd /path/to/APT-Transformer

# 复制GraphRAG模块
cp -r /home/claude/apt_graph_rag ./apt_model/graph_rag
```

### 方法2: 符号链接 (推荐开发)

```bash
cd /path/to/APT-Transformer/apt_model
ln -s /home/claude/apt_graph_rag graph_rag
```

---

## 🚀 三、基础使用

### 3.1 独立使用

在APT项目中直接使用GraphRAG:

```python
# test_graph_rag.py
from apt_model.graph_rag import GraphRAGManager

# 创建系统
rag = GraphRAGManager(
    max_dimension=2,
    enable_brain=True,
    enable_spectral=True
)

# 添加知识
rag.add_triple("Transformer", "是", "神经网络")
rag.add_triple("Transformer", "用于", "NLP")
rag.add_triple("BERT", "基于", "Transformer")

# 构建索引
rag.build_indices()

# 查询
results = rag.query("Transformer NLP", mode="hybrid", top_k=5)

for res in results:
    print(f"{res['entity']}: {res['score']:.4f}")
```

### 3.2 与现有APTRagManager集成

修改 `apt_model/rag_manager.py`:

```python
# rag_manager.py (修改后)

from .graph_rag import GraphRAGManager
from typing import List, Dict, Optional

class EnhancedAPTRagManager:
    """增强的APT RAG管理器 - 集成GraphRAG"""
    
    def __init__(
        self,
        use_graph_rag: bool = True,
        max_dimension: int = 2
    ):
        # 原有向量检索组件
        self.vector_store = ...  # 现有实现
        self.embedding_model = ...  # 现有实现
        
        # 新增: GraphRAG组件
        self.use_graph_rag = use_graph_rag
        if use_graph_rag:
            self.graph_rag = GraphRAGManager(
                max_dimension=max_dimension,
                enable_brain=True,
                enable_spectral=True
            )
        else:
            self.graph_rag = None
    
    def add_document(self, doc: str, metadata: Optional[Dict] = None):
        """添加文档"""
        # 原有向量存储
        self.vector_store.add(doc, metadata)
        
        # 新增: 提取三元组并加入GraphRAG
        if self.graph_rag:
            triples = self._extract_triples(doc)
            for s, p, o in triples:
                self.graph_rag.add_triple(s, p, o, metadata=metadata)
    
    def _extract_triples(self, text: str) -> List[tuple]:
        """从文本提取知识三元组"""
        # TODO: 实现实体关系抽取
        # 可以使用:
        # - spaCy + 依存句法分析
        # - OpenIE工具
        # - LLM提取
        
        triples = []
        # 示例实现 (需要替换为真实抽取)
        # ...
        return triples
    
    def query(
        self,
        query_text: str,
        mode: str = "hybrid",
        top_k: int = 10
    ) -> List[Dict]:
        """混合检索: 向量 + GraphRAG"""
        
        results = []
        
        # 1. 向量检索
        vector_results = self.vector_store.search(query_text, top_k=top_k)
        
        # 2. GraphRAG检索
        if self.graph_rag and mode in ["graph", "hybrid"]:
            graph_results = self.graph_rag.query(
                query_text,
                mode="hybrid",
                top_k=top_k
            )
            
            # 融合结果
            if mode == "hybrid":
                results = self._merge_results(
                    vector_results,
                    graph_results,
                    weights=(0.5, 0.5)
                )
            else:
                results = graph_results
        else:
            results = vector_results
        
        return results
    
    def _merge_results(
        self,
        vector_results: List[Dict],
        graph_results: List[Dict],
        weights: tuple
    ) -> List[Dict]:
        """融合向量和图检索结果"""
        # 按实体/文档ID合并得分
        merged = {}
        
        for res in vector_results:
            key = res.get('doc_id') or res.get('entity')
            merged[key] = merged.get(key, 0) + weights[0] * res['score']
        
        for res in graph_results:
            key = res.get('entity')
            merged[key] = merged.get(key, 0) + weights[1] * res['score']
        
        # 排序
        sorted_results = sorted(
            merged.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {'entity': k, 'score': v}
            for k, v in sorted_results
        ]
    
    def build_indices(self):
        """构建索引"""
        if self.graph_rag:
            self.graph_rag.build_indices()
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            'vector_store': {
                'num_documents': len(self.vector_store),
            }
        }
        
        if self.graph_rag:
            stats['graph_rag'] = self.graph_rag.get_statistics()
        
        return stats
```

---

## 💡 四、训练集成

### 4.1 在训练中使用GraphRAG

修改 `apt_model/trainer.py`:

```python
# trainer.py (新增部分)

from .graph_rag import GraphRAGManager

def train_with_graph_rag(
    model,
    config,
    train_dataset,
    use_graph_rag: bool = True
):
    """集成GraphRAG的训练"""
    
    # 初始化GraphRAG
    if use_graph_rag:
        rag = GraphRAGManager(max_dimension=2)
        
        # 从训练数据构建知识图谱
        logger.info("从训练数据构建知识图谱...")
        for batch in train_dataset:
            texts = batch['text']
            
            # 提取三元组
            for text in texts:
                triples = extract_triples_from_text(text)
                for s, p, o in triples:
                    rag.add_triple(s, p, o)
        
        # 构建索引
        logger.info("构建GraphRAG索引...")
        rag.build_indices()
        
        # 拓扑分析
        rag.print_summary()
    
    # 正常训练流程
    for epoch in range(config.num_epochs):
        for batch in train_dataset:
            # ...训练逻辑...
            
            # 可选: 使用GraphRAG增强上下文
            if use_graph_rag and epoch > 0:
                # 每个batch查询相关知识
                enhanced_context = []
                for text in batch['text']:
                    results = rag.query(text, mode="brain", top_k=5)
                    context = " ".join([r['entity'] for r in results])
                    enhanced_context.append(context)
                
                # 将增强上下文加入训练
                # batch['enhanced_context'] = enhanced_context
```

### 4.2 命令行支持

修改 `apt_model/parser.py`:

```python
# parser.py (新增参数)

# 在训练相关参数组中添加
parser.add_argument(
    '--use-graph-rag',
    action='store_true',
    help='使用GraphRAG增强训练'
)

parser.add_argument(
    '--graph-rag-dimension',
    type=int,
    default=2,
    help='GraphRAG最大维度 (0=点, 1=边, 2=面)'
)

parser.add_argument(
    '--enable-brain',
    action='store_true',
    help='启用图脑动力学'
)

parser.add_argument(
    '--enable-spectral',
    action='store_true',
    help='启用谱分析'
)
```

### 4.3 配置文件支持

修改 `apt_model/apt_config.py`:

```python
# apt_config.py (新增配置)

@dataclass
class GraphRAGConfig:
    """GraphRAG配置"""
    enabled: bool = False
    max_dimension: int = 2
    enable_brain: bool = True
    enable_spectral: bool = True
    T_cog: float = 1.0
    tau_p: float = 1.0
    tau_w: float = 10.0

# 在APTConfig中添加
@dataclass
class APTConfig:
    # ...现有配置...
    
    # 新增
    graph_rag: GraphRAGConfig = field(default_factory=GraphRAGConfig)
```

---

## 🧪 五、测试

### 5.1 单元测试

创建 `apt_model/tests/test_graph_rag.py`:

```python
# test_graph_rag.py

import unittest
from apt_model.graph_rag import (
    GeneralizedGraph,
    HodgeLaplacian,
    GraphBrainEngine,
    GraphRAGManager
)

class TestGraphRAG(unittest.TestCase):
    
    def test_generalized_graph(self):
        """测试泛图构建"""
        gg = GeneralizedGraph(max_dimension=2)
        
        # 添加节点
        gg.add_cell(0, "A")
        gg.add_cell(0, "B")
        
        # 添加边
        gg.add_cell(1, "AB", boundary={"A", "B"})
        
        # 验证
        self.assertEqual(len(gg.get_all_cell_ids(0)), 2)
        self.assertEqual(len(gg.get_all_cell_ids(1)), 1)
    
    def test_hodge_laplacian(self):
        """测试Hodge-Laplacian"""
        gg = GeneralizedGraph.from_edge_list([("A", "B"), ("B", "C")])
        hodge = HodgeLaplacian(gg)
        
        # 计算Laplacian
        L0 = hodge.compute_laplacian(0)
        
        # 验证形状
        n = len(gg.get_all_cell_ids(0))
        self.assertEqual(L0.shape, (n, n))
    
    def test_graph_brain(self):
        """测试图脑演化"""
        gg = GeneralizedGraph.from_edge_list([("A", "B")])
        brain = GraphBrainEngine(gg)
        
        # 演化
        delta_F = brain.evolve_step(dt=0.1)
        
        # 验证自由能变化
        self.assertIsInstance(delta_F, float)
    
    def test_graph_rag_manager(self):
        """测试GraphRAG管理器"""
        rag = GraphRAGManager(max_dimension=2)
        
        # 添加知识
        rag.add_triple("A", "rel", "B")
        
        # 构建索引
        rag.build_indices()
        
        # 查询
        results = rag.query("A", mode="simple", top_k=5)
        
        # 验证结果
        self.assertGreater(len(results), 0)

if __name__ == '__main__':
    unittest.main()
```

运行测试:

```bash
cd /path/to/APT-Transformer
python -m apt_model.tests.test_graph_rag
```

### 5.2 集成测试

创建 `examples/test_graph_rag_integration.py`:

```python
# test_graph_rag_integration.py

import sys
sys.path.append('..')

from apt_model.graph_rag import GraphRAGManager

def main():
    print("=" * 70)
    print("APT GraphRAG 集成测试")
    print("=" * 70)
    
    # 创建系统
    print("\n1. 创建GraphRAG系统...")
    rag = GraphRAGManager(
        max_dimension=2,
        enable_brain=True,
        enable_spectral=True
    )
    
    # 构建AI知识图谱
    print("\n2. 构建AI领域知识图谱...")
    triples = [
        # 基础
        ("深度学习", "是", "机器学习方法"),
        ("机器学习", "属于", "人工智能"),
        
        # 模型
        ("Transformer", "是", "神经网络架构"),
        ("Transformer", "用于", "NLP任务"),
        ("BERT", "基于", "Transformer"),
        ("GPT", "基于", "Transformer"),
        
        # APT
        ("APT", "是", "Transformer变体"),
        ("APT", "使用", "自生成注意力"),
        ("APT", "支持", "中文"),
        
        # 训练
        ("APT", "需要", "GPU"),
        ("GPU", "加速", "深度学习"),
        ("深度学习", "需要", "大数据"),
    ]
    
    rag.add_triples_batch(triples)
    
    # 构建索引
    print("\n3. 构建索引...")
    rag.build_indices()
    
    # 测试查询
    queries = [
        "APT Transformer",
        "深度学习 GPU",
        "中文 NLP"
    ]
    
    print("\n4. 执行查询:")
    for query in queries:
        print(f"\n查询: '{query}'")
        results = rag.query(query, mode="hybrid", top_k=5)
        
        for i, res in enumerate(results, 1):
            print(f"  {i}. {res['entity']} (score={res['score']:.4f})")
    
    # 统计信息
    print("\n5. 系统统计:")
    stats = rag.get_statistics()
    print(f"  实体数: {stats['num_entities']}")
    print(f"  关系数: {stats['num_relations']}")
    print(f"  事实数: {stats['num_facts']}")
    
    if 'topology' in stats:
        print(f"\n  拓扑特征:")
        topo = stats['topology']
        print(f"    Betti数: {topo['betti_numbers']}")
        print(f"    欧拉示性数: {topo['euler_characteristic']}")
    
    if 'brain' in stats:
        print(f"\n  图脑演化:")
        brain = stats['brain']
        print(f"    演化步数: {brain['num_steps']}")
        print(f"    相变次数: {brain['phase_transitions']}")
    
    # 保存
    print("\n6. 保存系统...")
    rag.save("./graph_rag_save")
    
    print("\n" + "=" * 70)
    print("集成测试完成!")
    print("=" * 70)

if __name__ == "__main__":
    main()
```

---

## 📝 六、使用建议

### 6.1 何时使用GraphRAG

**推荐场景**:
- 需要理解复杂的多体关系
- 需要拓扑推理 (检测孔洞、循环)
- 需要动态知识演化
- 知识图谱规模中等 (10K-1M节点)

**不推荐场景**:
- 纯文本匹配任务
- 超大规模图谱 (>10M节点，需要优化)
- 实时响应要求极高 (<10ms)

### 6.2 性能优化建议

1. **批量添加**: 使用 `add_triples_batch()` 而非循环 `add_triple()`
2. **延迟构建**: 添加完所有知识后再调用 `build_indices()`
3. **维度选择**: 不需要高阶关系时设置 `max_dimension=1`
4. **禁用未使用的组件**: 
   - 不需要谱分析时设置 `enable_spectral=False`
   - 不需要动力学时设置 `enable_brain=False`

### 6.3 调试技巧

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 详细日志
rag = GraphRAGManager(...)
rag.gg.logger.setLevel(logging.DEBUG)
rag.hodge.logger.setLevel(logging.DEBUG)
rag.brain.logger.setLevel(logging.DEBUG)
```

---

## 🎓 七、进阶使用

### 7.1 自定义实体提取

```python
from apt_model.graph_rag import GraphRAGManager
import spacy

class CustomGraphRAG(GraphRAGManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nlp = spacy.load("zh_core_web_sm")
    
    def add_document_with_extraction(self, text: str):
        """自动提取实体关系"""
        doc = self.nlp(text)
        
        # 提取实体
        entities = [ent.text for ent in doc.ents]
        
        # 提取关系 (简化示例)
        for token in doc:
            if token.dep_ == "nsubj":
                subject = token.text
                verb = token.head.text
                objects = [child.text for child in token.head.children 
                          if child.dep_ == "dobj"]
                
                for obj in objects:
                    self.add_triple(subject, verb, obj)
```

### 7.2 自定义查询策略

```python
class AdvancedGraphRAG(GraphRAGManager):
    def query_with_reasoning(
        self,
        query: str,
        num_hops: int = 3
    ) -> List[Dict]:
        """多跳推理查询"""
        
        # 第一跳: 直接相关
        results_1 = self.query(query, mode="spectral", top_k=10)
        
        # 第二跳: 扩展搜索
        results_2 = []
        for res in results_1:
            entity = res['entity']
            sub_results = self.query(entity, mode="brain", top_k=5)
            results_2.extend(sub_results)
        
        # 去重和排序
        # ...
        
        return results_2
```

---

## 🔍 八、故障排除

### 常见问题

**Q: 导入错误 `ModuleNotFoundError: No module named 'apt_model.graph_rag'`**

A: 检查目录结构，确保 `__init__.py` 存在

```bash
ls apt_model/graph_rag/__init__.py
```

**Q: 内存不足**

A: 减少维度或禁用部分组件

```python
rag = GraphRAGManager(
    max_dimension=1,  # 只用点和边
    enable_brain=False,  # 禁用图脑
    enable_spectral=False  # 禁用谱分析
)
```

**Q: 谱计算失败**

A: 图太稀疏或太小，增加数据或降低k

```python
hodge.compute_spectrum(0, k=5)  # 减少特征值数量
```

---

## 📞 九、支持

遇到问题？
1. 查看 [README.md](README.md)
2. 查看测试代码
3. 提交Issue到GitHub

---

**祝集成顺利！ 🎉**
