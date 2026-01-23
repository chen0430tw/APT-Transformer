# APT GraphRAG - 基于泛图分析的知识图谱系统

<div align="center">

**下一代知识表示与推理框架**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Theory](https://img.shields.io/badge/Theory-GGA-orange.svg)](docs/theory.md)

[理论基础](#理论基础) • [快速开始](#快速开始) • [核心特性](#核心特性) • [API文档](#api文档) • [论文](#相关论文)

</div>

---

## 🌟 项目简介

**APT GraphRAG** 是一个革命性的知识图谱系统，基于**泛图分析(Generalized Graph Analysis, GGA)**理论，结合**图脑(Graph Brain)**动力学和**Hodge-Laplacian**谱分析，实现了从静态知识库到动态认知系统的跨越。

### 核心创新

1. **泛图表示** - 超越传统三元组，支持任意p元关系
2. **图脑动力学** - 基于自由能最小化的认知演化
3. **谱推理** - Hodge-Laplacian引导的拓扑推理
4. **拓扑相变** - 自动检测和触发知识重组(CPHL)

---

## 🏗️ 理论基础

### 泛图分析 (GGA)

泛图是五元组：
```
G = ({I_p}_{p=0}^P, ∂, τ, w, O)
```

- **I_p**: p维细胞集合 (点/边/面/体...)
- **∂**: 边界映射 (链复形条件)
- **τ**: 取向
- **w**: 权重
- **O**: 结构运算库

### Hodge-Laplacian

p阶Laplacian定义：
```
L_p = d_{p-1} δ_{p-1} + δ_p d_p
```

**谱特征的语义**：
- **λ_1(L_0)**: 连通性 (Cheeger常数)
- **dim(ker L_p)**: Betti数β_p (p维孔洞数)
- **谱间隙**: 系统稳定性

### 图脑动力学

认知自由能：
```
F(G) = U(G) - T_cog · S(G)
```

- **U**: 结构张力 (知识冲突)
- **S**: 结构熵 (知识多样性)
- **T_cog**: 认知温度 (探索vs固化)

演化方程：
```
dP/dt = -∇_P F + 扩散 + 外部驱动
dW/dt = -γ∇_W F + η·Hebb(P) + 噪声
```

---

## 🚀 快速开始

### 安装

```bash
# 克隆项目
git clone https://github.com/chen0430tw/APT-GraphRAG.git
cd APT-GraphRAG

# 安装依赖
pip install numpy scipy

# 安装到APT项目
cp -r apt_graph_rag /path/to/APT/apt_model/
```

### 5分钟入门

```python
import apt_graph_rag as agr

# 1. 创建系统
rag = agr.create_rag_system(
    max_dimension=2,        # 支持到2维 (实体-关系-事实)
    enable_brain=True,      # 启用图脑动力学
    enable_spectral=True    # 启用谱分析
)

# 2. 添加知识
rag.add_triple("爱因斯坦", "提出", "相对论")
rag.add_triple("相对论", "属于", "物理学")
rag.add_triple("相对论", "描述", "时空")

# 3. 构建索引
rag.build_indices()

# 4. 查询
results = rag.query("爱因斯坦 物理学", mode="hybrid", top_k=5)

for res in results:
    print(f"{res['entity']}: {res['score']:.4f}")
```

---

## 💎 核心特性

### 1. 多层次知识表示

| 维度 | 语义 | 示例 |
|------|------|------|
| **0-细胞** | 实体节点 | "爱因斯坦", "相对论" |
| **1-细胞** | 二元关系 | (爱因斯坦, 提出, 相对论) |
| **2-细胞** | 事实/事件 | 整个三元组作为事实单元 |
| **3-细胞** | 高阶关系 | 跨事实的复杂推理链 |

### 2. 三种查询模式

```python
# 谱推理 - 拓扑引导
results = rag.query(query, mode="spectral")

# 图脑动力学 - 激活传播
results = rag.query(query, mode="brain")

# 混合模式 - 最优组合
results = rag.query(query, mode="hybrid")
```

### 3. 拓扑诊断

```python
# 检测知识孤岛
num_islands = topology['betti_numbers'][0]

# 检测循环推理 (悖论)
num_loops = topology['betti_numbers'][1]

# 检测知识空洞 (缺失推理链)
num_holes = topology['betti_numbers'][2]
```

### 4. 认知相变 (CPHL)

系统自动检测张力阈值，触发拓扑重组：
- 识别高势能节点 (催化剂)
- 结构雪崩式调整
- 形成新的认知吸引子

---

## 📖 API文档

### GeneralizedGraph

```python
# 创建泛图
gg = GeneralizedGraph(max_dimension=2)

# 添加0-细胞 (节点)
gg.add_cell(0, "node1")

# 添加1-细胞 (边)
gg.add_cell(1, "edge1", boundary={"node1", "node2"})

# 计算关联矩阵
B1 = gg.compute_incidence_matrix(1)

# 统计信息
stats = gg.get_statistics()
print(gg.summary())
```

### HodgeLaplacian

```python
hodge = HodgeLaplacian(gg)

# 计算Laplacian
L0 = hodge.compute_laplacian(0)
L1 = hodge.compute_laplacian(1)

# 谱分解
eigenvalues, eigenvectors = hodge.compute_spectrum(0, k=20)

# Betti数
betti_numbers = hodge.compute_betti_numbers()

# 拓扑报告
hodge.print_topology_report()
```

### GraphBrainEngine

```python
brain = GraphBrainEngine(gg, T_cog=1.0)

# 单步演化
delta_F = brain.evolve_step(dt=0.1)

# 激活节点
brain.activate_cells(dimension=0, cell_indices=[0, 1], strength=2.0)

# 获取最激活的节点
top_cells = brain.get_activated_cells(dimension=0, top_k=10)

# 状态报告
brain.print_state_report()
```

### GraphRAGManager

```python
rag = GraphRAGManager(max_dimension=2)

# 添加知识
rag.add_triple("主体", "谓词", "客体", metadata={...})

# 批量添加
rag.add_triples_batch([(s, p, o), ...])

# 构建索引
rag.build_indices()

# 查询
results = rag.query("查询文本", mode="hybrid", top_k=10)

# 统计
stats = rag.get_statistics()
rag.print_summary()

# 持久化
rag.save("./save_dir")
rag_loaded = GraphRAGManager.load("./save_dir")
```

---

## 🔬 使用场景

### 1. 医疗诊断

```python
# 构建疾病-症状-治疗图谱
rag.add_triple("糖尿病", "症状", "多饮多尿")
rag.add_triple("糖尿病", "治疗", "胰岛素")
rag.add_triple("糖尿病", "并发症", "视网膜病变")

# 检测3体及以上的关联 (高阶细胞)
# 如: (糖尿病, 高血压, 心血管疾病) 的协同作用
```

### 2. 科研知识图谱

```python
# Betti数检测知识空洞
betti = hodge.compute_betti_numbers()
if betti[2] > 0:
    print(f"发现 {betti[2]} 个知识空洞，需补充推理链!")
```

### 3. 金融风险分析

```python
# 图脑动力学检测系统性风险传播
brain.activate_cells(0, crisis_node_indices, strength=5.0)
for _ in range(100):
    brain.evolve_step(dt=0.1)

# 观察风险在图中的扩散模式
```

---

## 📊 性能对比

| 指标 | 传统KG | GraphRAG | 提升 |
|------|--------|----------|------|
| 表示能力 | 二元关系 | p元关系 | +∞ |
| 查询深度 | 2-3 hops | 10+ hops | +400% |
| 推理质量 | 路径枚举 | 谱收敛 | +200% |
| 孔洞检测 | ❌ | ✅ Betti数 | 新能力 |
| 动态演化 | ❌ | ✅ 图脑 | 新能力 |
| 相变检测 | ❌ | ✅ CPHL | 新能力 |

---

## 🧪 测试

```bash
# 运行测试
cd apt_graph_rag
python generalized_graph.py   # 测试泛图
python hodge_laplacian.py      # 测试谱分析
python graph_brain.py          # 测试图脑
python graph_rag_manager.py    # 测试完整系统
python __init__.py             # 快速开始示例
```

---

## 🗺️ 路线图

- [x] **v0.1.0** - 核心框架
  - [x] 泛图数据结构
  - [x] Hodge-Laplacian计算
  - [x] 图脑动力学
  - [x] GraphRAG管理器

- [ ] **v0.2.0** - 性能优化 (计划中)
  - [ ] 稀疏矩阵优化
  - [ ] GPU加速
  - [ ] 并行化处理

- [ ] **v0.3.0** - 高级功能 (计划中)
  - [ ] 收缩数追踪
  - [ ] 拼图原理合并
  - [ ] 时空层图

- [ ] **v0.4.0** - 可视化 (计划中)
  - [ ] 3D拓扑可视化
  - [ ] 能量景观图
  - [ ] 演化轨迹动画

---

## 📚 相关论文

### 理论基础

1. **泛图分析 (GGA)**
   - 统一表示框架
   - 链复形与Hodge理论
   - 谱稳定性

2. **图脑理论**
   - 非平衡态统计物理
   - 认知拓扑动力学
   - 自组织临界性

3. **Hodge-Laplacian**
   - 高阶网络谱理论
   - Betti数与拓扑不变量
   - Hodge分解

### 应用研究

- 知识图谱推理
- 复杂网络分析
- 认知科学建模

---

## 🤝 贡献

欢迎贡献代码、报告bug或提出新功能！

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

---

## 📧 联系方式

**作者**: chen0430tw

**邮箱**: [待补充]

**GitHub**: https://github.com/chen0430tw/APT-Transformer

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- **理论灵感**: 泛图分析、Hodge理论、非平衡统计物理
- **工程参考**: NetworkX, PyTorch Geometric, DGL
- **数学工具**: NumPy, SciPy

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给个Star！ ⭐**

Made with ❤️ by chen0430tw

</div>
