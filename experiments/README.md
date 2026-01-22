# Experiments Directory

实验隔离区 - 用于研究、原型开发和基准测试

## 目的

`experiments/` 目录用于隔离**实验性、不稳定、或研究性质**的代码，与生产代码（`apt/`）和示例代码（`examples/`）分离。

## 与examples/的区别

| 特性 | experiments/ | examples/ |
|------|-------------|-----------|
| **目的** | 研究和原型 | 生产就绪的示例 |
| **稳定性** | 不稳定，可能变化 | 稳定，经过测试 |
| **代码质量** | 实验性，快速迭代 | 高质量，有文档 |
| **维护** | 可能过时 | 持续维护 |
| **用户** | 研究人员、开发者 | 最终用户 |

## 目录结构

```
experiments/
├── research/          # 研究项目
│   ├── new_attention/ # 新注意力机制研究
│   ├── scaling_laws/  # 扩展定律研究
│   └── ...
├── prototypes/        # 功能原型
│   ├── prototype_v1/  # 早期原型
│   └── ...
├── benchmarks/        # 性能基准测试
│   ├── gpu_bench/     # GPU性能测试
│   ├── memory_bench/  # 内存测试
│   └── ...
├── configs/           # 实验配置（已存在）
├── hpo/              # 超参数优化（已存在）
└── README.md         # 本文件
```

## 使用场景

### 1. Research（研究项目）

用于学术研究、新算法探索：

```
experiments/research/sparse_attention/
├── paper_implementation.py  # 论文实现
├── experiments.py           # 实验脚本
├── results/                 # 实验结果
└── README.md               # 研究说明
```

### 2. Prototypes（功能原型）

快速原型和概念验证：

```
experiments/prototypes/dynamic_moe/
├── prototype.py    # 原型实现
├── test.py        # 测试脚本
└── notes.md       # 开发笔记
```

### 3. Benchmarks（基准测试）

性能测试和优化研究：

```
experiments/benchmarks/training_speed/
├── benchmark.py    # 基准测试
├── results.json   # 测试结果
└── analysis.ipynb # 结果分析
```

## 最佳实践

### 1. 项目结构

每个实验应该是自包含的：

```
experiments/research/my_experiment/
├── README.md          # 实验说明
├── requirements.txt   # 依赖
├── src/              # 源代码
├── configs/          # 配置文件
├── scripts/          # 运行脚本
├── results/          # 结果（.gitignore）
└── notebooks/        # Jupyter notebooks
```

### 2. 文档要求

每个实验必须包含README.md：

```markdown
# Experiment: [名称]

## 目标
[实验目标]

## 方法
[实验方法]

## 状态
- [ ] 实验中
- [ ] 已完成
- [ ] 已废弃
- [ ] 已合并到主代码库

## 结果
[实验结果]

## 依赖
[特殊依赖]

## 运行方法
```bash
python scripts/run_experiment.py
```
```

### 3. 隔离原则

- ❌ 不要在实验中修改`apt/`的代码
- ❌ 不要在生产代码中依赖实验代码
- ✅ 实验可以导入`apt/`的模块
- ✅ 实验之间应该相互独立

示例：

```python
# ✅ 正确：实验导入生产代码
from apt.model.architectures import APTLargeModel
from experiments.research.my_exp.custom_layer import CustomLayer

# ❌ 错误：生产代码依赖实验
# 在 apt/model/architectures.py 中
from experiments.research.my_exp.custom_layer import CustomLayer  # 禁止！
```

### 4. 代码质量

实验代码可以：
- 快速迭代，不强求完美
- 包含TODO和FIXME
- 有临时性hack
- 缺少完整测试

但应该：
- 有基本的文档
- 能够运行
- 有清晰的状态标记

### 5. 生命周期

实验有三种结局：

#### A. 成功 → 合并到主代码库

```bash
# 1. 清理和重构实验代码
# 2. 添加测试
# 3. 完善文档
# 4. 提交PR到apt/
# 5. 在实验README标记"已合并"
```

#### B. 失败 → 保留以供参考

```markdown
# 在README中记录：
## 状态
- [x] 已完成
- [ ] 失败原因：性能不如预期

## 教训
- 学到了什么
- 为什么不work
```

#### C. 废弃 → 删除或归档

```bash
# 移动到archive/或直接删除
git mv experiments/research/old_exp experiments/archive/old_exp
```

## 示例：从实验到生产

### 阶段1：实验（experiments/research/new_feature/）

```python
# experiments/research/new_feature/prototype.py
def experimental_function():
    # 快速原型，未优化
    print("DEBUG: testing new feature")
    result = some_hack()
    return result
```

### 阶段2：重构（准备合并）

```python
# experiments/research/new_feature/refactored.py
def new_feature():
    """
    新特性实现

    Args:
        ...

    Returns:
        ...
    """
    # 清理后的代码
    result = optimized_implementation()
    return result
```

### 阶段3：生产（apt/model/new_feature.py）

```python
# apt/model/new_feature.py
"""
New Feature Module

完整的文档、测试和优化后的实现
"""

def new_feature():
    """完整的文档字符串..."""
    # 生产级代码
    return result
```

## 配置管理

实验可以使用profile配置：

```yaml
# experiments/research/my_exp/config.yaml
base: lite  # 继承lite profile

# 实验性修改
model:
  features:
    experimental_feature: true

training:
  batch_size: 8  # 实验用小batch
```

## Git管理

### .gitignore建议

```gitignore
# 实验结果（不提交）
experiments/*/results/
experiments/*/outputs/
experiments/*/*.log
experiments/*/*.pkl
experiments/*/*.pt

# 实验数据（不提交）
experiments/*/data/

# 但保留代码和配置
!experiments/*/*.py
!experiments/*/*.yaml
!experiments/*/README.md
```

### 提交规范

```bash
# 使用experiment前缀
git commit -m "experiment: add sparse attention prototype"
git commit -m "experiment: benchmark moe performance"
git commit -m "experiment: research on scaling laws"
```

## 常见问题

**Q: 什么时候用experiments/，什么时候用examples/？**

A:
- **experiments/**: 你在探索、研究、不确定是否work
- **examples/**: 已经work，展示如何使用现有功能

**Q: 实验可以修改apt/的代码吗？**

A: 不可以。实验应该继承和扩展，而不是修改。如果需要修改，应该：
1. 在实验中实现修改
2. 验证有效后提PR到主代码库

**Q: 实验代码需要测试吗？**

A: 不强制要求，但建议有基本的smoke test验证能运行。

**Q: 多人协作的实验怎么管理？**

A: 每个实验一个独立目录，使用git branch隔离：

```bash
git checkout -b experiment/new-attention
# 在experiments/research/new_attention/工作
```

## 迁移指南

如果你有旧的实验代码，建议迁移到新结构：

```bash
# 1. 识别实验性代码
# 2. 移动到experiments/
git mv old_location experiments/research/experiment_name/

# 3. 添加README
cat > experiments/research/experiment_name/README.md << EOF
# Experiment Name
...
EOF

# 4. 提交
git commit -m "experiment: migrate old experiment to new structure"
```

## 总结

- ✅ experiments/ 用于实验、研究、原型
- ✅ 与生产代码隔离
- ✅ 快速迭代，灵活度高
- ✅ 成功后可以合并到主代码库
- ❌ 不要在生产代码中依赖实验
- ❌ 不要在实验中修改生产代码

---

**实验愉快！记住：在experiments/里可以大胆尝试，失败了也没关系。**
