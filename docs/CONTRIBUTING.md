# Contributing to APT-Transformer

感谢您对 APT-Transformer 的贡献！本文档提供了参与项目开发的指南和标准。

---

## 📋 目录

- [复杂度治理清单](#-复杂度治理清单)
- [架构原则](#-架构原则)
- [插件 vs 模块决策](#-插件-vs-模块决策)
- [开发工作流](#-开发工作流)
- [代码规范](#-代码规范)
- [提交规范](#-提交规范)
- [Pull Request 流程](#-pull-request-流程)
- [测试要求](#-测试要求)

---

## 📊 复杂度治理清单

在添加新功能或修改现有代码前，请参考以下清单：

### 1. 架构层级检查 ✅

**问题**：该功能属于哪个架构层？

- [ ] **L0 (Kernel)** - 核心 APT 算法
  - Autopoietic Transform
  - DBC-DAC 损失函数
  - Left-Spin Smooth
  - 核心模型定义

- [ ] **L1 (Performance)** - 性能优化
  - Virtual Blackwell 优化栈
  - GPU/NPU 加速
  - 量化和压缩

- [ ] **L2 (Memory)** - 记忆和知识系统
  - AIM-Memory 系统
  - GraphRAG
  - 知识图谱

- [ ] **L3 (Product)** - 产品和应用
  - WebUI
  - API 服务
  - 插件系统
  - 可观测性

**原则**：
- ✅ 低层级 **不能** 依赖高层级
- ✅ L0 不能导入 L1/L2/L3
- ✅ L1 不能导入 L2/L3
- ✅ L2 不能导入 L3

---

### 2. 插件 vs 模块决策 ✅

**问题**：该功能应该是插件还是核心模块？

使用决策树：

```
功能X
  │
  ├─ 是核心算法/训练流程？
  │   └─ 是 → ❌ 保持为模块 (apt/core/)
  │
  ├─ 是工具/脚本（打包、构建）？
  │   └─ 是 → ❌ 保持为工具 (tools/)
  │
  ├─ 是必需的基础设施？
  │   ├─ 必需 → ❌ 保持为模块 (apt/perf/infrastructure/)
  │   └─ 可选 → ✅ 做插件 (plugins/infrastructure/)
  │
  ├─ 是外部服务集成？
  │   └─ 是 → ✅ 做插件 (plugins/integration/)
  │
  ├─ 是可选训练方法？
  │   └─ 是 → ✅ 做插件 (plugins/rl/, plugins/optimization/)
  │
  ├─ 是实验功能？
  │   └─ 是 → ✅ 做插件 (plugins/experimental/)
  │
  └─ 是可选增强？
      └─ 是 → ✅ 做插件 (plugins/monitoring/, plugins/evaluation/, etc.)
```

**插件化标准** ✅：
1. 可选的增强功能（monitoring, visualization, evaluation）
2. 外部服务集成（web_search, mcp, rag）
3. 可选的训练方法（RLHF, DPO, GRPO）
4. 实验性功能（experimental/*）

**保持为模块** ❌：
1. 核心工具 - APX Converter（打包工具）
2. 核心数据处理 - Data Processor/Pipeline
3. 核心算法 - APT 核心功能
4. 核心优化 - GPU Flash, Extreme Scale
5. 基础设施 - 必需的系统组件

**参考文档**：`docs/guides/PLUGIN_VS_MODULE_PRINCIPLES.md`

---

### 3. 复杂度预算 ✅

**问题**：该变更是否增加了不必要的复杂度？

- [ ] **循环复杂度** < 10（每个函数）
- [ ] **文件长度** < 500 行
- [ ] **函数长度** < 50 行
- [ ] **参数数量** < 5 个
- [ ] **嵌套深度** < 4 层

**如果超出预算**：
1. 重构为多个小函数
2. 提取辅助模块
3. 考虑设计模式简化

---

### 4. 依赖管理 ✅

**问题**：该变更引入了哪些依赖？

- [ ] **核心依赖** - 必需，添加到 `requirements.txt`
- [ ] **可选依赖** - 插件专用，添加到 `requirements-plugins.txt`
- [ ] **开发依赖** - 测试/开发，添加到 `requirements-dev.txt`

**原则**：
- ✅ 最小化依赖
- ✅ 固定版本（避免依赖冲突）
- ✅ 文档化为何需要该依赖
- ❌ 不引入重型依赖到核心（除非必要）

---

### 5. 测试覆盖 ✅

**问题**：该变更是否有充分的测试？

- [ ] **单元测试** - 覆盖率 ≥ 80%
- [ ] **集成测试** - 与系统其他部分集成
- [ ] **性能测试** - 如果涉及性能关键路径
- [ ] **边界测试** - 异常情况和边界条件

**测试组织**：
```
tests/
├── l0_kernel/        - L0 核心测试
├── l1_performance/   - L1 性能测试
├── l2_memory/        - L2 记忆测试
├── l3_product/       - L3 产品测试
└── integration/      - 集成测试
```

---

### 6. 文档完整性 ✅

**问题**：该变更是否有适当的文档？

- [ ] **代码注释** - 复杂逻辑必须注释
- [ ] **Docstring** - 所有公共 API
- [ ] **类型注解** - 所有函数签名
- [ ] **README 更新** - 如果影响使用方式
- [ ] **API 文档** - 如果添加新 API
- [ ] **架构文档** - 如果改变架构

**文档位置**：
```
docs/
├── kernel/        - L0 内核文档
├── performance/   - L1 性能文档
├── memory/        - L2 记忆文档
├── product/       - L3 产品文档
└── guides/        - 指南和教程
```

---

### 7. 向后兼容性 ✅

**问题**：该变更是否破坏了现有 API？

- [ ] **API 变更** - 是否改变了公共接口？
- [ ] **弃用警告** - 旧 API 是否有 DeprecationWarning？
- [ ] **迁移指南** - 是否提供了迁移文档？
- [ ] **版本号** - 是否正确标记版本（major.minor.patch）？

**版本策略**：
- **Major** - 破坏性变更
- **Minor** - 新功能（向后兼容）
- **Patch** - Bug 修复

---

### 8. 性能影响 ✅

**问题**：该变更对性能有何影响？

- [ ] **基准测试** - 是否运行了性能基准？
- [ ] **性能回退** - 是否导致性能下降？
- [ ] **内存使用** - 是否增加了内存占用？
- [ ] **启动时间** - 是否影响启动速度？

**性能标准**：
- ❌ 不允许 >5% 的性能回退（除非有充分理由）
- ✅ 优化应有基准数据支持

---

## 🏗️ 架构原则

### L0/L1/L2/L3 分层架构

```
L3 (Product)     - WebUI, API, Plugins, Observability
    ↑ 依赖
L2 (Memory)      - AIM-Memory, GraphRAG, Knowledge Graph
    ↑ 依赖
L1 (Performance) - Virtual Blackwell, GPU Optimization
    ↑ 依赖
L0 (Kernel)      - APT Core Algorithm, DBC-DAC, LSS
```

**依赖宪章**：
1. 每层只能向下依赖
2. 禁止跨层导入（L0 → L2）
3. 使用 `scripts/check_reverse_dependencies.py` 验证

---

### 插件系统架构

**当前插件生态**（31 plugins across 15 categories）：

```
apt/apps/plugins/
├── core/              (3) - 核心插件
├── integration/       (3) - 外部集成
├── distillation/      (2) - 知识蒸馏
├── experimental/      (3) - 实验特性
├── monitoring/        (2) - 监控诊断
├── visualization/     (1) - 训练可视化
├── evaluation/        (2) - 模型评估
├── infrastructure/    (1) - 基础设施
├── optimization/      (1) - 性能优化
├── rl/                (4) - 强化学习对齐
├── protocol/          (1) - 协议集成
├── retrieval/         (2) - 检索增强
├── hardware/          (3) - 硬件模拟
├── deployment/        (2) - 部署虚拟化
└── memory/            (1) - 高级记忆
```

**插件开发标准**：
1. 继承 `PluginBase`
2. 实现 `load()`, `unload()`, `execute()`
3. 提供配置 schema
4. 编写测试（覆盖率 ≥ 80%）
5. 更新 `apt/apps/plugins/PLUGIN_CATALOG.md`

---

## 🔄 插件 vs 模块决策

### 应该做插件 ✅

**1. 可选增强功能**
- 示例：monitoring, visualization, evaluation
- 判断：禁用后核心功能仍可运行

**2. 外部服务集成**
- 示例：web_search, ollama_export, mcp_integration
- 判断：依赖第三方服务/协议

**3. 可选训练方法**
- 示例：RLHF, DPO, GRPO, MXFP4 quantization
- 判断：用户可选择不使用

**4. 实验性功能**
- 示例：multimodal_training, virtual_blackwell
- 判断：Beta 功能或研究特性

---

### 应该保持为模块 ❌

**1. 核心工具**
- 示例：APX Converter（打包工具）
- 原因：构建时工具，不是运行时功能

**2. 核心数据处理**
- 示例：Data Processor, Pipeline
- 原因：训练必需，不是可选增强

**3. 核心算法**
- 示例：APT Model, DBC-DAC Loss
- 原因：项目定义性功能

**4. 核心优化**
- 示例：GPU Flash Optimization, Extreme Scale Training
- 原因：核心性能优化，不是可选功能

**5. 核心系统**
- 示例：Knowledge Graph（L2 核心）
- 原因：L2 层的必需功能

---

## 🛠️ 开发工作流

### 1. 设置开发环境

```bash
# Clone repository
git clone https://github.com/your-org/APT-Transformer.git
cd APT-Transformer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

---

### 2. 创建新分支

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Or bug fix branch
git checkout -b fix/bug-description
```

**分支命名规范**：
- `feature/` - 新功能
- `fix/` - Bug 修复
- `docs/` - 文档更新
- `refactor/` - 代码重构
- `test/` - 测试改进

---

### 3. 进行开发

**开发前检查**：
1. [ ] 确认架构层级（L0/L1/L2/L3）
2. [ ] 确认是插件还是模块
3. [ ] 检查复杂度预算
4. [ ] 规划测试策略

**开发中**：
1. 遵循代码规范
2. 编写清晰的注释
3. 保持小的、原子性的提交
4. 运行本地测试

---

### 4. 运行测试

```bash
# Run all tests
pytest tests/

# Run specific layer tests
pytest tests/l0_kernel/
pytest tests/l1_performance/
pytest tests/l2_memory/
pytest tests/l3_product/

# Run with coverage
pytest --cov=apt tests/

# Check reverse dependencies
python scripts/check_reverse_dependencies.py
```

---

### 5. 提交代码

**提交信息格式**：

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type**：
- `feat` - 新功能
- `fix` - Bug 修复
- `docs` - 文档
- `style` - 格式（不影响代码运行）
- `refactor` - 重构
- `test` - 测试
- `chore` - 构建/工具变更

**示例**：

```bash
git commit -m "feat(plugins): 添加 RLHF trainer 插件

- 实现 RLHF 训练流程
- 添加 reward model 支持
- 包含单元测试和集成测试

Closes #123
"
```

---

## 📝 代码规范

### Python 风格

遵循 **PEP 8** 和项目特定规范：

```python
# Good ✅
def calculate_apt_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    dbc_weight: float = 0.5,
) -> torch.Tensor:
    """
    计算 APT 损失函数。

    Args:
        predictions: 模型预测 (batch_size, seq_len, vocab_size)
        targets: 目标标签 (batch_size, seq_len)
        dbc_weight: DBC 损失权重

    Returns:
        总损失值
    """
    dac_loss = compute_dac_loss(predictions, targets)
    dbc_loss = compute_dbc_loss(predictions, targets)
    return dac_loss + dbc_weight * dbc_loss


# Bad ❌
def calc_loss(pred, tgt, w=0.5):
    # No docstring, unclear names, no type hints
    l1 = compute_dac_loss(pred, tgt)
    l2 = compute_dbc_loss(pred, tgt)
    return l1 + w * l2
```

---

### 类型注解

**必需**：所有公共 API

```python
from typing import Optional, List, Dict, Any

# Good ✅
def enable_plugin(
    name: str,
    config: Optional[Dict[str, Any]] = None,
    layers: Optional[List[str]] = None,
) -> bool:
    """Enable a plugin with optional configuration."""
    pass

# Bad ❌
def enable_plugin(name, config=None, layers=None):
    pass
```

---

### 导入顺序

```python
# 1. 标准库
import os
import sys
from pathlib import Path

# 2. 第三方库
import torch
import numpy as np
from transformers import AutoModel

# 3. 本地导入
from apt.core import APTModel
from apt.perf import enable_virtual_blackwell
```

---

## 🔍 Pull Request 流程

### 1. PR 准备清单

在创建 PR 前，确认：

- [ ] 所有测试通过
- [ ] 代码覆盖率 ≥ 80%
- [ ] 通过 linter 检查
- [ ] 依赖检查通过
- [ ] 文档已更新
- [ ] CHANGELOG 已更新（如果是重要变更）

---

### 2. PR 模板

```markdown
## 描述
<!-- 简要描述这个 PR 做了什么 -->

## 类型
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## 架构层级
- [ ] L0 (Kernel)
- [ ] L1 (Performance)
- [ ] L2 (Memory)
- [ ] L3 (Product)

## 插件 vs 模块
- [ ] 核心模块
- [ ] 插件
- [ ] 工具
- [ ] N/A

## 复杂度治理检查
- [ ] 循环复杂度 < 10
- [ ] 文件长度 < 500 行
- [ ] 测试覆盖率 ≥ 80%
- [ ] 依赖检查通过
- [ ] 文档已更新

## 测试
<!-- 描述如何测试这个变更 -->

## 性能影响
<!-- 是否有性能影响？提供基准数据 -->

## 截图（如果适用）
<!-- 添加截图帮助说明 -->

## Checklist
- [ ] 我的代码遵循项目的代码规范
- [ ] 我已经进行了自我审查
- [ ] 我已经添加了必要的注释
- [ ] 我已经更新了相关文档
- [ ] 我的变更不会产生新的警告
- [ ] 我已经添加了测试证明修复有效或功能可用
- [ ] 新的和现有的单元测试都通过了
```

---

### 3. Code Review

**作为 Author**：
- 及时响应评审意见
- 解释设计决策
- 保持专业和开放态度

**作为 Reviewer**：
- 检查复杂度治理清单
- 验证测试覆盖
- 检查架构一致性
- 提供建设性反馈

---

## 🧪 测试要求

### 测试金字塔

```
     /\
    /  \  E2E Tests (少量)
   /____\
  /      \ Integration Tests (适量)
 /________\
/          \ Unit Tests (大量)
```

---

### 单元测试示例

```python
# tests/l0_kernel/test_apt_model.py
import pytest
import torch
from apt.core import APTModel

def test_apt_model_forward():
    """测试 APT 模型前向传播。"""
    model = APTModel(vocab_size=1000, hidden_size=512)
    input_ids = torch.randint(0, 1000, (2, 10))

    output = model(input_ids)

    assert output.shape == (2, 10, 1000)
    assert not torch.isnan(output).any()


def test_apt_model_with_dbc():
    """测试 APT 模型的 DBC 损失计算。"""
    model = APTModel(vocab_size=1000, hidden_size=512)
    input_ids = torch.randint(0, 1000, (2, 10))
    labels = torch.randint(0, 1000, (2, 10))

    loss = model.compute_loss(input_ids, labels, use_dbc=True)

    assert loss.item() > 0
    assert not torch.isnan(loss).any()
```

---

### 集成测试示例

```python
# tests/integration/test_plugin_integration.py
import pytest
from apt.apps.plugin_system import PluginManager

def test_plugin_loading():
    """测试插件加载和执行。"""
    pm = PluginManager()

    # Load plugin
    pm.load_plugin("monitoring.gradient_monitor")

    # Execute
    result = pm.execute("monitoring.gradient_monitor", {
        "model": mock_model,
        "gradients": mock_gradients,
    })

    assert result.status == "success"
```

---

## 📚 参考资源

### 文档
- **架构文档**: `docs/guides/COMPLETE_TECH_SUMMARY.md`
- **插件指南**: `apt/apps/plugins/PLUGIN_CATALOG.md`
- **插件原则**: `docs/guides/PLUGIN_VS_MODULE_PRINCIPLES.md`
- **转换路线图**: `PLUGIN_CONVERSION_ROADMAP.md`

### 工具
- **依赖检查**: `scripts/check_reverse_dependencies.py`
- **插件转换**: `scripts/convert_modules_to_plugins.py`
- **Tier 3 评估**: `scripts/evaluate_tier3_modules.py`

### 测试
- **L0 测试**: `tests/l0_kernel/`
- **L1 测试**: `tests/l1_performance/`
- **L2 测试**: `tests/l2_memory/`
- **L3 测试**: `tests/l3_product/`
- **集成测试**: `tests/integration/`

---

## ❓ 常见问题

### Q: 我应该创建插件还是模块？
**A**: 使用 [插件 vs 模块决策](#-插件-vs-模块决策) 决策树。简单规则：如果是可选功能、外部集成或实验特性 → 插件；如果是核心功能、工具或必需组件 → 模块。

### Q: 如何检查我的代码是否违反了依赖规则？
**A**: 运行 `python scripts/check_reverse_dependencies.py`。它会检查 L0/L1/L2/L3 的反向依赖违规。

### Q: 测试覆盖率要求是多少？
**A**: 最低 80%。核心模块（L0）建议 ≥ 90%。

### Q: 我的 PR 需要多长时间才能被审查？
**A**: 通常 2-3 个工作日。复杂的 PR 可能需要更长时间。

### Q: 可以直接提交到 main 分支吗？
**A**: 不可以。所有变更必须通过 PR 和 code review。

---

## 🎯 核心原则总结

1. **不是所有模块都该做插件** - 工具保持为工具，核心保持为核心
2. **遵守架构分层** - L0/L1/L2/L3 依赖宪章
3. **测试先行** - 覆盖率 ≥ 80%
4. **文档完整** - 代码即文档
5. **质量优于数量** - 小而精的变更

---

## 🤝 行为准则

我们致力于为所有人提供友好、安全和包容的环境。参与项目时请：

- ✅ 尊重不同观点和经验
- ✅ 优雅地接受建设性批评
- ✅ 关注对社区最有利的事情
- ✅ 对其他社区成员表现出同理心
- ❌ 使用性别化语言或图像
- ❌ 人身攻击或政治攻击
- ❌ 公开或私下骚扰
- ❌ 未经明确许可发布他人的私人信息

---

## 📧 联系方式

- **Issues**: [GitHub Issues](https://github.com/your-org/APT-Transformer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/APT-Transformer/discussions)
- **Email**: maintainers@apt-transformer.org

---

**感谢您的贡献！** 🎉

Every contribution, no matter how small, makes APT-Transformer better for everyone.
