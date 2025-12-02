# APT模块化集成方案

## 概述

本文档说明如何将 `apt_graph_rag` 和 `apt_sosa` 两个模块优雅地集成到APT-Transformer项目中。

**模块简介:**
- **apt_graph_rag**: 基于泛图分析的GraphRAG知识图谱系统
- **apt_sosa**: 智能训练监控与自动纠错系统

---

## 架构设计

### 目标架构

```
APT-Transformer/
├── apt_model/
│   ├── modeling/           # 模型相关
│   │   ├── knowledge_graph.py (现有轻量KG)
│   │   └── kg_rag_integration.py
│   │
│   ├── core/               # 核心功能
│   │   ├── api_providers.py (已有)
│   │   ├── graph_rag/      # 新增: GraphRAG模块 ⭐
│   │   │   ├── __init__.py
│   │   │   ├── generalized_graph.py
│   │   │   ├── hodge_laplacian.py
│   │   │   ├── graph_brain.py
│   │   │   └── graph_rag_manager.py
│   │   │
│   │   └── training/       # 新增: 训练工具 ⭐
│   │       ├── __init__.py
│   │       ├── sosa_core.py
│   │       ├── training_monitor.py
│   │       └── apt_integration.py
│   │
│   ├── plugins/            # 插件系统
│   │   ├── teacher_api.py (已有)
│   │   ├── visual_distillation_plugin.py (已有)
│   │   ├── graph_rag_plugin.py        # 新增 ⭐
│   │   └── training_monitor_plugin.py # 新增 ⭐
│   │
│   └── config/
│       └── module_config.py    # 新增: 模块配置 ⭐
│
├── docs/
│   ├── modules/                # 新增: 模块文档
│   │   ├── GRAPH_RAG.md
│   │   └── SOSA_TRAINING.md
│   │
│   └── integration/            # 新增: 集成指南
│       ├── GRAPH_RAG_INTEGRATION.md
│       └── SOSA_INTEGRATION.md
│
└── examples/
    ├── graph_rag_examples/     # 新增
    │   ├── basic_usage.py
    │   ├── rag_with_api.py
    │   └── advanced_queries.py
    │
    └── training_monitor_examples/  # 新增
        ├── basic_monitoring.py
        └── auto_fix_demo.py
```

---

## 集成策略

### 原则

1. **最小侵入**: 不修改现有代码，只添加新功能
2. **模块独立**: 每个模块可独立使用
3. **松耦合**: 通过配置和插件系统集成
4. **向后兼容**: 不影响现有功能
5. **可选启用**: 通过命令行参数或配置文件控制

### 分层集成

```
┌─────────────────────────────────┐
│  应用层 (用户接口)                │
│  - 命令行参数                     │
│  - 配置文件                       │
│  - 便捷函数                       │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│  插件层 (功能封装)                │
│  - GraphRAGPlugin                │
│  - TrainingMonitorPlugin         │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│  核心层 (底层实现)                │
│  - apt_model/core/graph_rag/     │
│  - apt_model/core/training/      │
└─────────────────────────────────┘
           ↓
┌─────────────────────────────────┐
│  基础层 (现有系统)                │
│  - APT模型                        │
│  - 训练循环                       │
│  - RAG系统                        │
└─────────────────────────────────┘
```

---

## 实施步骤

### 阶段1: 核心模块复制 (5分钟)

```bash
# 1. 复制GraphRAG模块
mkdir -p apt_model/core/graph_rag
cp -r /tmp/apt_graph_rag/* apt_model/core/graph_rag/

# 2. 复制SOSA模块
mkdir -p apt_model/core/training
cp -r /tmp/apt_sosa/* apt_model/core/training/

# 3. 调整导入路径
# (自动化脚本见下方)
```

### 阶段2: 创建插件包装器 (15分钟)

#### 2.1 GraphRAG插件

创建 `apt_model/plugins/graph_rag_plugin.py`:

```python
"""
GraphRAG插件 - 增强的知识图谱系统

提供:
- 泛图数据结构 (支持高阶关系)
- 谱分析 (Hodge-Laplacian)
- 图脑动力学
- 多模式查询
"""

from apt_model.core.graph_rag import (
    GraphRAGManager,
    GeneralizedGraph,
    HodgeLaplacian,
    GraphBrainEngine
)

class GraphRAGPlugin:
    """APT GraphRAG插件"""

    def __init__(self, config):
        self.config = config
        self.rag = GraphRAGManager(
            max_dimension=config.get('max_dimension', 2),
            enable_brain=config.get('enable_brain', True),
            enable_spectral=config.get('enable_spectral', True)
        )

    def integrate_with_rag(self, base_rag):
        """与现有RAG系统集成"""
        # 将现有RAG的知识导入GraphRAG
        pass

    def integrate_with_api(self, api_provider):
        """与API提供商集成"""
        from apt_model.core.api_providers import create_api_provider
        # 使用API构建知识图谱
        pass


def create_graph_rag_plugin(config=None):
    """便捷创建函数"""
    return GraphRAGPlugin(config or {})
```

#### 2.2 训练监控插件

创建 `apt_model/plugins/training_monitor_plugin.py`:

```python
"""
训练监控插件 - SOSA智能监控

提供:
- 实时训练监控
- 自动异常检测
- 智能诊断
- 自适应修复
"""

from apt_model.core.training import (
    SOSATrainingWrapper,
    TrainingMonitor,
    create_training_monitor,
    wrap_training
)

class TrainingMonitorPlugin:
    """APT训练监控插件"""

    def __init__(self, model, optimizer, config):
        self.wrapper = SOSATrainingWrapper(
            model=model,
            optimizer=optimizer,
            config=config,
            auto_fix=config.get('auto_fix', True),
            checkpoint_dir=config.get('checkpoint_dir', './checkpoints')
        )

    def training_step(self, batch, forward_fn=None):
        """包装的训练步"""
        return self.wrapper.training_step(batch, forward_fn)

    def get_statistics(self):
        """获取训练统计"""
        return self.wrapper.get_statistics()

    def print_report(self):
        """打印报告"""
        self.wrapper.print_report()


def create_training_monitor_plugin(model, optimizer, config=None):
    """便捷创建函数"""
    return TrainingMonitorPlugin(model, optimizer, config or {})
```

### 阶段3: 配置系统 (10分钟)

创建 `apt_model/config/module_config.py`:

```python
"""
模块配置

统一管理所有可选模块的配置
"""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GraphRAGConfig:
    """GraphRAG配置"""
    enabled: bool = False
    max_dimension: int = 2
    enable_brain: bool = True
    enable_spectral: bool = True

    # API集成
    use_api: bool = False
    api_provider: Optional[str] = None
    api_key: Optional[str] = None
    api_model: Optional[str] = None

@dataclass
class SOSAConfig:
    """SOSA训练监控配置"""
    enabled: bool = False
    window_seconds: float = 10.0
    auto_fix: bool = True
    max_fixes_per_error: int = 3
    exploration_weight: float = 0.5

    # 检查点
    checkpoint_dir: str = './checkpoints'
    save_best: bool = True

@dataclass
class ModuleConfig:
    """模块配置总集"""
    graph_rag: GraphRAGConfig = field(default_factory=GraphRAGConfig)
    sosa: SOSAConfig = field(default_factory=SOSAConfig)
```

### 阶段4: 命令行集成 (10分钟)

修改 `apt_model/parser.py` 或创建 `apt_model/cli_extensions.py`:

```python
"""
命令行扩展

为新模块添加命令行参数
"""

def add_graph_rag_args(parser):
    """添加GraphRAG参数"""
    group = parser.add_argument_group('GraphRAG Options')

    group.add_argument(
        '--use-graph-rag',
        action='store_true',
        help='使用增强的GraphRAG系统'
    )

    group.add_argument(
        '--graph-rag-dimension',
        type=int,
        default=2,
        help='泛图最大维度 (0=点, 1=边, 2=面)'
    )

    group.add_argument(
        '--graph-rag-enable-brain',
        action='store_true',
        default=True,
        help='启用图脑动力学'
    )

    group.add_argument(
        '--graph-rag-enable-spectral',
        action='store_true',
        default=True,
        help='启用谱分析'
    )

def add_sosa_args(parser):
    """添加SOSA参数"""
    group = parser.add_argument_group('SOSA Training Monitor Options')

    group.add_argument(
        '--use-sosa',
        action='store_true',
        help='使用SOSA智能训练监控'
    )

    group.add_argument(
        '--sosa-auto-fix',
        action='store_true',
        default=True,
        help='启用SOSA自动修复'
    )

    group.add_argument(
        '--sosa-window',
        type=float,
        default=10.0,
        help='SOSA时间窗口大小(秒)'
    )

    group.add_argument(
        '--sosa-max-fixes',
        type=int,
        default=3,
        help='每种错误最大修复次数'
    )

def extend_cli(parser):
    """扩展命令行参数"""
    add_graph_rag_args(parser)
    add_sosa_args(parser)
    return parser
```

### 阶段5: 训练流程集成 (15分钟)

修改或扩展训练脚本:

```python
"""
训练流程集成示例
"""

def train_with_modules(config):
    """
    集成了所有模块的训练函数
    """
    # 1. 创建基础组件
    model = create_model(config)
    optimizer = create_optimizer(model, config)
    train_dataloader = create_dataloader(config)

    # 2. 可选: SOSA监控
    training_wrapper = None
    if config.modules.sosa.enabled:
        from apt_model.plugins.training_monitor_plugin import (
            create_training_monitor_plugin
        )

        training_wrapper = create_training_monitor_plugin(
            model=model,
            optimizer=optimizer,
            config=config.modules.sosa
        )
        print("[SOSA] 训练监控已启用")

    # 3. 可选: GraphRAG
    graph_rag = None
    if config.modules.graph_rag.enabled:
        from apt_model.plugins.graph_rag_plugin import (
            create_graph_rag_plugin
        )

        graph_rag = create_graph_rag_plugin(config.modules.graph_rag)

        # 集成到现有RAG
        if hasattr(config, 'rag') and config.rag.enabled:
            graph_rag.integrate_with_rag(existing_rag)

        print("[GraphRAG] 增强知识图谱已启用")

    # 4. 训练循环
    for epoch in range(config.num_epochs):
        for batch in train_dataloader:
            # 使用SOSA包装的训练步
            if training_wrapper:
                loss = training_wrapper.training_step(
                    batch,
                    forward_fn=lambda m, b: m(**b).loss
                )
            else:
                # 标准训练步
                loss = model(**batch).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # 定期报告
            if global_step % 1000 == 0:
                if training_wrapper:
                    training_wrapper.print_report()

    # 5. 最终报告
    if training_wrapper:
        print("\n" + "=" * 70)
        print("SOSA 训练报告")
        print("=" * 70)
        training_wrapper.print_report()
```

---

## 依赖管理

### 新增依赖

更新 `requirements.txt`:

```txt
# 现有依赖
torch>=1.10.0
transformers>=4.20.0
numpy>=1.20.0

# GraphRAG依赖
scipy>=1.7.0         # 谱分析

# SOSA依赖
# (无额外依赖，只需numpy)

# 可选依赖
matplotlib>=3.4.0    # 可视化
networkx>=2.6.0      # 图分析参考
```

### 依赖检查

创建 `apt_model/core/dependency_check.py`:

```python
"""
依赖检查

在导入模块前检查依赖是否满足
"""

def check_graph_rag_dependencies():
    """检查GraphRAG依赖"""
    try:
        import numpy
        import scipy
        return True, "依赖满足"
    except ImportError as e:
        return False, f"缺少依赖: {e}"

def check_sosa_dependencies():
    """检查SOSA依赖"""
    try:
        import numpy
        return True, "依赖满足"
    except ImportError as e:
        return False, f"缺少依赖: {e}"

def check_all_dependencies():
    """检查所有模块依赖"""
    results = {}

    results['graph_rag'] = check_graph_rag_dependencies()
    results['sosa'] = check_sosa_dependencies()

    return results
```

---

## 文档结构

### 创建文档

```bash
mkdir -p docs/modules docs/integration

# GraphRAG文档
cp /tmp/apt_graph_rag/README.md docs/modules/GRAPH_RAG.md
cp /tmp/apt_graph_rag/INTEGRATION.md docs/integration/GRAPH_RAG_INTEGRATION.md

# SOSA文档
cp /tmp/apt_sosa/README.md docs/modules/SOSA_TRAINING.md
cp /tmp/apt_sosa/QUICK_START.md docs/integration/SOSA_INTEGRATION.md
```

### 更新主README

在 `README.md` 中添加:

```markdown
## 🧩 可选模块

APT-Transformer支持多个可选的增强模块:

### GraphRAG - 增强知识图谱

基于泛图分析的下一代知识图谱系统。

**特性:**
- 支持高阶关系 (不只是二元)
- Hodge-Laplacian谱分析
- 图脑动力学推理
- 多模式查询

**使用:**
```bash
python train.py --use-graph-rag --graph-rag-dimension 2
```

**文档:** [docs/modules/GRAPH_RAG.md](docs/modules/GRAPH_RAG.md)

### SOSA - 智能训练监控

火种源自组织算法驱动的训练监控与自动纠错。

**特性:**
- 7种训练异常自动检测
- 智能诊断与修复
- 自适应策略学习
- 零侵入集成

**使用:**
```bash
python train.py --use-sosa --sosa-auto-fix
```

**文档:** [docs/modules/SOSA_TRAINING.md](docs/modules/SOSA_TRAINING.md)
```

---

## 示例代码

### 示例1: GraphRAG基础使用

`examples/graph_rag_examples/basic_usage.py`:

```python
"""
GraphRAG基础使用示例
"""

from apt_model.core.graph_rag import create_rag_system

# 创建系统
rag = create_rag_system(
    max_dimension=2,
    enable_brain=True,
    enable_spectral=True
)

# 添加知识
rag.add_triple("Python", "是", "编程语言")
rag.add_triple("Python", "用于", "AI开发")
rag.add_triple("PyTorch", "基于", "Python")

# 构建索引
rag.build_indices()

# 查询
results = rag.query("Python AI", mode="hybrid", top_k=5)

for res in results:
    print(f"{res['entity']}: {res['score']:.4f}")
```

### 示例2: GraphRAG + API集成

`examples/graph_rag_examples/rag_with_api.py`:

```python
"""
GraphRAG与API提供商集成
"""

from apt_model.core.graph_rag import GraphRAGManager
from apt_model.core.api_providers import create_api_provider

# 创建GraphRAG
rag = GraphRAGManager(max_dimension=2)

# 创建API提供商
api = create_api_provider(
    provider='siliconflow',
    api_key='your-key',
    model_name='Qwen/Qwen2-7B-Instruct'
)

# 使用API构建知识图谱
def build_kg_with_api(documents, api, rag):
    """使用API从文档提取知识"""
    for doc in documents:
        # 用API提取三元组
        prompt = f"从以下文本提取知识三元组:\n{doc}\n输出格式: (实体1, 关系, 实体2)"
        triples_text = api.generate_text(prompt, max_tokens=200)

        # 解析并添加到图谱
        # (实际需要更复杂的解析逻辑)
        rag.add_triple(...)

    rag.build_indices()

# 使用
documents = ["Python是一种编程语言...", ...]
build_kg_with_api(documents, api, rag)

print(f"成本: ${api.stats['total_cost']:.4f}")
```

### 示例3: SOSA训练监控

`examples/training_monitor_examples/basic_monitoring.py`:

```python
"""
SOSA训练监控示例
"""

import torch
from apt_model.core.training import wrap_training

# 创建模型和优化器
model = YourModel()
optimizer = torch.optim.Adam(model.parameters())

# 包装训练
wrapper = wrap_training(
    model=model,
    optimizer=optimizer,
    auto_fix=True,
    checkpoint_dir='./checkpoints'
)

# 训练
for epoch in range(10):
    for batch in train_dataloader:
        # 一行搞定
        loss = wrapper.training_step(batch)

        if step % 100 == 0:
            print(f"Loss: {loss.item():.4f}")
            wrapper.print_report()

# 最终统计
stats = wrapper.get_statistics()
print(f"异常检测次数: {stats['total_errors']}")
print(f"自动修复次数: {stats['successful_fixes']}")
```

---

## 测试计划

### 单元测试

```bash
# GraphRAG测试
python -m pytest apt_model/core/graph_rag/tests/

# SOSA测试
python -m pytest apt_model/core/training/tests/
```

### 集成测试

```bash
# 完整训练测试
python examples/integration_test.py --use-graph-rag --use-sosa
```

---

## 回滚计划

如果集成出现问题，可以快速回滚:

```bash
# 1. 删除新增模块
rm -rf apt_model/core/graph_rag/
rm -rf apt_model/core/training/

# 2. 删除插件
rm apt_model/plugins/graph_rag_plugin.py
rm apt_model/plugins/training_monitor_plugin.py

# 3. 恢复配置
git checkout apt_model/config/
git checkout apt_model/parser.py

# 4. 清理文档
rm -rf docs/modules/GRAPH_RAG.md
rm -rf docs/modules/SOSA_TRAINING.md
```

---

## 性能影响评估

### GraphRAG

| 规模 | 构建时间 | 查询时间 | 内存占用 |
|------|---------|---------|---------|
| 小 (~100实体) | <1秒 | <0.1秒 | ~10MB |
| 中 (~1K实体) | ~5秒 | <0.5秒 | ~50MB |
| 大 (~10K实体) | ~30秒 | ~1秒 | ~200MB |

### SOSA

| 功能 | 开销 | 说明 |
|------|------|------|
| 监控 | <1% | 异步记录 |
| 检测 | <0.1% | 每步检查 |
| 修复 | 变化 | 仅在异常时触发 |

---

## 最佳实践

### 1. 渐进式启用

```bash
# 第一阶段: 只监控，不修复
python train.py --use-sosa --no-sosa-auto-fix

# 第二阶段: 启用自动修复
python train.py --use-sosa --sosa-auto-fix

# 第三阶段: 启用GraphRAG
python train.py --use-sosa --use-graph-rag
```

### 2. 配置文件管理

创建 `config/modules.yaml`:

```yaml
modules:
  graph_rag:
    enabled: true
    max_dimension: 2
    enable_brain: true
    enable_spectral: true

  sosa:
    enabled: true
    auto_fix: true
    window_seconds: 10.0
    max_fixes_per_error: 3
```

### 3. 日志管理

```python
import logging

# 模块日志配置
logging.getLogger('apt_model.core.graph_rag').setLevel(logging.INFO)
logging.getLogger('apt_model.core.training').setLevel(logging.INFO)
```

---

## 常见问题

### Q1: 两个模块可以同时使用吗？

**A**: 可以！它们是独立的，可以同时启用:
```bash
python train.py --use-graph-rag --use-sosa
```

### Q2: 如何禁用某个模块？

**A**: 不传递对应的参数即可，或在配置文件中设置 `enabled: false`。

### Q3: 模块会影响训练速度吗？

**A**:
- GraphRAG: 仅在需要查询时使用，不影响训练循环
- SOSA: 开销<1%，可忽略

### Q4: 可以只用模块的部分功能吗？

**A**: 可以！例如只用SOSA监控不用自动修复:
```bash
python train.py --use-sosa --no-sosa-auto-fix
```

---

## 下一步行动

### 立即执行 (5分钟)

1. 复制模块到对应目录
2. 运行基础示例验证
3. 查看文档了解详细功能

### 短期集成 (1小时)

1. 创建插件包装器
2. 添加命令行参数
3. 修改训练脚本
4. 运行集成测试

### 长期优化 (持续)

1. 收集使用反馈
2. 优化性能
3. 添加新特性
4. 编写更多示例

---

## 总结

这个模块化集成方案遵循:
- ✅ **最小侵入**: 不修改现有代码
- ✅ **松耦合**: 通过插件系统集成
- ✅ **易回滚**: 删除即可
- ✅ **可扩展**: 易于添加新模块
- ✅ **文档完善**: 每个模块都有详细文档

立即开始集成，让APT变得更强大！🚀
