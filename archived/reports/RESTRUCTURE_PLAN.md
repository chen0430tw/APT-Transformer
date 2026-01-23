# APT-Transformer 目录重构计划

## 🎯 重构目标

将当前的 `apt_model/` 单体目录重构为清晰的四层分离架构：

```
apt_model/  (单体目录，211个文件)
    ↓
apt_core/   (L0 内核层)
apt_perf/   (L1 性能层)
apt_memory/ (L2 记忆层)
apps/       (L3 应用交付层)
```

**核心原则**: 分层隔离 + 不破坏现有功能 + 渐进迁移

---

## 📂 新目录结构（完整）

```
APT-Transformer/
├─ apt_core/               # L0 内核层 (最小、最稳定)
│  ├─ __init__.py
│  ├─ modeling/
│  │  ├─ apt_model.py      # APT 核心模型
│  │  ├─ dbc_dac.py        # DBC-DAC 算子
│  │  ├─ left_spin_smooth.py  # Left-Spin Smooth
│  │  ├─ blocks/           # 核心组件
│  │  │  ├─ attention.py
│  │  │  ├─ ffn.py
│  │  │  ├─ router.py
│  │  │  └─ embeddings.py
│  │  └─ utils.py
│  ├─ generation/
│  │  ├─ generator.py
│  │  └─ evaluator.py
│  ├─ training/
│  │  ├─ trainer_base.py   # 最小训练循环
│  │  ├─ optimizer.py
│  │  └─ data_loading.py
│  ├─ runtime/
│  │  └─ decoder/
│  │     ├─ routing.py
│  │     ├─ halting.py
│  │     └─ reasoning_controller.py
│  ├─ config/
│  │  ├─ apt_config.py
│  │  └─ base.py
│  ├─ codecs/              # 编码器
│  │  ├─ en_gpt2/
│  │  ├─ zh_char/
│  │  └─ ja_mecab/
│  └─ multilingual/        # 多语言
│     ├─ language.py
│     ├─ tokenizer.py
│     └─ detector.py
│
├─ apt_perf/               # L1 性能层 (可选加速)
│  ├─ __init__.py
│  ├─ optimization/
│  │  ├─ virtual_blackwell_adapter.py
│  │  ├─ vgpu_stack.py
│  │  ├─ vgpu_estimator.py
│  │  ├─ microvm_compression.py
│  │  ├─ mxfp4_quantization.py
│  │  ├─ moe_optimized.py
│  │  └─ extreme_scale_training.py
│  ├─ training/
│  │  ├─ mixed_precision.py
│  │  ├─ checkpoint.py     # Checkpoint 原子性
│  │  ├─ distributed.py    # DDP, FSDP
│  │  └─ gradient_accumulation.py
│  ├─ compression/
│  │  ├─ quantization.py
│  │  ├─ pruning.py
│  │  └─ distillation.py
│  └─ vb_global.py         # 虚拟 Blackwell 全局入口
│
├─ apt_memory/             # L2 记忆层 (独立王国)
│  ├─ __init__.py
│  ├─ aim/
│  │  ├─ aim_memory.py
│  │  ├─ aim_nc.py
│  │  ├─ anchor_fields.py  # 锚点主权
│  │  ├─ evidence_feedback.py
│  │  └─ tiered_memory.py
│  ├─ graph_rag/
│  │  ├─ graph_brain.py
│  │  ├─ graph_rag_manager.py
│  │  ├─ hodge_laplacian.py
│  │  ├─ generalized_graph.py
│  │  └─ kg_integration.py
│  ├─ long_context/
│  │  ├─ rope_variants.py
│  │  ├─ context_compression.py
│  │  └─ retrieval.py
│  └─ memory_interface.py  # 统一接口
│
├─ apps/                   # L3 应用交付层
│  ├─ __init__.py
│  ├─ webui/
│  │  ├─ app.py
│  │  ├─ tabs/
│  │  │  ├─ training_monitor.py
│  │  │  ├─ gradient_monitor.py
│  │  │  ├─ checkpoint_manager.py
│  │  │  └─ inference_tester.py
│  │  └─ websocket_push.py
│  ├─ api/
│  │  ├─ server.py
│  │  ├─ endpoints/
│  │  │  ├─ train.py
│  │  │  ├─ inference.py
│  │  │  └─ monitoring.py
│  │  └─ auth.py
│  ├─ cli/
│  │  ├─ commands.py
│  │  ├─ parser.py
│  │  ├─ command_registry.py
│  │  └─ interactive/
│  │     ├─ chat.py
│  │     └─ admin_mode.py
│  ├─ observability/       # 可观测性三件套
│  │  ├─ collectors/
│  │  │  ├─ training_monitor.py
│  │  │  ├─ gradient_monitor.py
│  │  │  └─ resource_monitor.py
│  │  ├─ visualization/
│  │  │  ├─ plot_training.py
│  │  │  ├─ plot_gradients.py
│  │  │  └─ generate_report.py
│  │  └─ dashboards/
│  │     └─ webui_integration.py
│  ├─ plugins/
│  │  ├─ compression_plugin.py
│  │  ├─ visual_distillation_plugin.py
│  │  ├─ web_search_plugin.py
│  │  ├─ teacher_api.py
│  │  ├─ graph_rag_plugin.py
│  │  └─ plugin_system/
│  │     ├─ loader.py
│  │     ├─ registry.py
│  │     └─ hooks.py
│  ├─ agent/
│  │  ├─ agent_loop.py
│  │  ├─ tool_system.py
│  │  └─ python_sandbox.py
│  └─ console/             # 控制台系统
│     ├─ core.py
│     ├─ plugin_bus.py
│     ├─ eqi_manager.py
│     └─ commands/
│
├─ experiments/            # 研究区 (不是库代码)
│  ├─ papers/
│  │  ├─ transformer_xl/
│  │  ├─ llama/
│  │  └─ deepseek/
│  ├─ benchmarks/
│  │  ├─ glue/
│  │  ├─ mmlu/
│  │  └─ humaneval/
│  ├─ prototypes/
│  │  ├─ new_attention.py
│  │  └─ experimental_optimizer.py
│  └─ hpo/
│     ├─ apt_optuna.py
│     └─ configs/
│
├─ tools/                  # 工具区 (纯脚本)
│  ├─ data_processing/
│  │  ├─ generate_hlbd_v2.py
│  │  └─ preprocess_dataset.py
│  ├─ model_conversion/
│  │  ├─ to_onnx.py
│  │  └─ to_safetensors.py
│  ├─ diagnostics/
│  │  ├─ diagnose_issues.py
│  │  └─ hardware_check.py
│  └─ visualization/
│     ├─ visualize_training.py
│     └─ demo_visualization.py
│
├─ artifacts/              # 产物区 (不进版本控制)
│  ├─ reports/
│  ├─ plots/
│  ├─ checkpoints/
│  └─ exports/
│
├─ profiles/               # 发行版配置
│  ├─ core.yaml
│  ├─ perf.yaml
│  ├─ mind.yaml
│  └─ max.yaml
│
├─ docs/                   # 文档 (重新编排)
│  ├─ README.md            # 新首页
│  ├─ L0_KERNEL.md         # 内核层文档
│  ├─ L1_PERFORMANCE.md    # 性能层文档
│  ├─ L2_MEMORY.md         # 记忆层文档
│  ├─ L3_PRODUCT.md        # 应用层文档
│  ├─ ARCHITECTURE.md      # 架构设计
│  ├─ DISTRIBUTION_MODES.md
│  ├─ guides/              # 指南
│  │  ├─ quickstart/
│  │  ├─ training/
│  │  ├─ deployment/
│  │  └─ advanced/
│  └─ archive/             # 归档文档
│
├─ tests/                  # 测试 (分层测试)
│  ├─ l0_kernel/
│  │  ├─ test_apt_model.py
│  │  ├─ test_dbc_dac.py
│  │  └─ test_training_loop.py
│  ├─ l1_performance/
│  │  ├─ test_vgpu_stack.py
│  │  ├─ test_quantization.py
│  │  └─ test_distributed.py
│  ├─ l2_memory/
│  │  ├─ test_aim_memory.py
│  │  ├─ test_graph_rag.py
│  │  └─ test_anchor_sovereignty.py
│  ├─ l3_product/
│  │  ├─ test_webui.py
│  │  ├─ test_api.py
│  │  └─ test_plugins.py
│  └─ integration/
│     └─ test_full_pipeline.py
│
├─ scripts/                # 自动化脚本
│  ├─ setup/
│  ├─ launchers/
│  ├─ testing/
│  └─ migration/           # 迁移脚本
│     ├─ migrate_to_new_structure.py
│     └─ validate_imports.py
│
├─ data/                   # 数据文件 (保持不变)
├─ bert/                   # 预训练模型 (保持不变)
│
├─ apt_model/              # 旧目录 (兼容性保留，逐步废弃)
│  ├─ __init__.py          # 重定向到新位置
│  └─ _deprecated.py       # 废弃警告
│
├─ apt/                    # 旧核心包 (保持不变，已稳定)
│
├─ ARCHITECTURE.md         # ✅ 架构设计文档
├─ DISTRIBUTION_MODES.md   # ✅ 发行版说明
├─ RESTRUCTURE_PLAN.md     # ✅ 本文档
├─ README.md
├─ INSTALLATION.md
├─ setup.py
├─ requirements.txt
├─ requirements-core.txt   # 新增：核心版依赖
├─ requirements-perf.txt   # 新增：性能版依赖
├─ requirements-mind.txt   # 新增：记忆版依赖
└─ requirements-max.txt    # 新增：完整版依赖
```

---

## 📋 文件迁移映射表

### L0 内核层迁移

| 旧路径 (apt_model/) | 新路径 (apt_core/) | 备注 |
|-------------------|-------------------|------|
| modeling/apt_model.py | modeling/apt_model.py | 核心模型 |
| modeling/blocks/ | modeling/blocks/ | 核心组件 |
| modeling/embeddings.py | modeling/embeddings.py | - |
| generation/generator.py | generation/generator.py | - |
| generation/evaluator.py | generation/evaluator.py | - |
| training/trainer.py | training/trainer_base.py | 重命名 |
| training/optimizer.py | training/optimizer.py | - |
| training/data_loading.py | training/data_loading.py | - |
| runtime/decoder/ | runtime/decoder/ | 全部 |
| config/apt_config.py | config/apt_config.py | - |
| codecs/ | codecs/ | 全部 |
| multilingual/ (from apt/) | multilingual/ | 从 apt/ 移动 |

### L1 性能层迁移

| 旧路径 | 新路径 (apt_perf/) | 备注 |
|-------|-------------------|------|
| optimization/* | optimization/* | 全部虚拟 Blackwell 相关 |
| training/mixed_precision.py | training/mixed_precision.py | - |
| training/checkpoint.py | training/checkpoint.py | - |
| plugins/compression_plugin.py | compression/compression.py | 重构为模块 |

### L2 记忆层迁移

| 旧路径 | 新路径 (apt_memory/) | 备注 |
|-------|---------------------|------|
| core/graph_rag/ | graph_rag/ | 全部 |
| modeling/knowledge_graph.py | graph_rag/kg_integration.py | 重命名 |
| modeling/kg_rag_integration.py | graph_rag/ | 合并 |
| modeling/rag_integration.py | long_context/retrieval.py | 重组 |
| (新增) | aim/ | 新增 AIM 系统 |

### L3 应用层迁移

| 旧路径 | 新路径 (apps/) | 备注 |
|-------|---------------|------|
| webui/ | webui/ | 全部 |
| api/ | api/ | 全部 |
| cli/ | cli/ | 全部 |
| interactive/ | cli/interactive/ | 合并到 CLI |
| console/ | console/ | 全部 |
| plugins/* | plugins/* | 大部分插件 |
| agent/ (如果存在) | agent/ | - |
| utils/visualization.py | observability/visualization/ | 重组 |
| core/training/training_monitor.py | observability/collectors/training_monitor.py | 重组 |

### 研究区迁移

| 旧路径 | 新路径 (experiments/) | 备注 |
|-------|---------------------|------|
| experiments/ | experiments/ | 保持不变 |
| examples/ (部分) | experiments/prototypes/ | 实验性示例 |

### 工具区迁移

| 旧路径 | 新路径 (tools/) | 备注 |
|-------|---------------|------|
| tools/ | tools/ | 保持不变 |
| scripts/ (部分) | tools/ | 工具脚本 |

---

## 🚦 迁移策略（四阶段）

### 阶段 0: 准备阶段（1 天）

**目标**: 创建基础设施，不破坏现有代码

#### 任务清单
- [x] 创建 ARCHITECTURE.md
- [x] 创建 DISTRIBUTION_MODES.md
- [x] 创建 RESTRUCTURE_PLAN.md (本文档)
- [ ] 创建新目录结构（空目录）
- [ ] 创建发行版配置文件（profiles/*.yaml）
- [ ] 创建迁移脚本（scripts/migration/）
- [ ] 更新 .gitignore

```bash
# 执行脚本
bash scripts/migration/phase0_prepare.sh
```

#### 产物
- 新目录已创建（空）
- 配置文件就绪
- 迁移工具就绪

---

### 阶段 1: L0 核心层迁移（1 周）

**目标**: 迁移核心模型和最小训练循环，确保可独立运行

#### 任务清单
1. **迁移核心模型**
   - [ ] apt_model/modeling/apt_model.py → apt_core/modeling/apt_model.py
   - [ ] apt_model/modeling/blocks/ → apt_core/modeling/blocks/
   - [ ] apt_model/modeling/embeddings.py → apt_core/modeling/embeddings.py

2. **迁移生成模块**
   - [ ] apt_model/generation/ → apt_core/generation/

3. **迁移训练基础**
   - [ ] apt_model/training/trainer.py → apt_core/training/trainer_base.py（重构）
   - [ ] apt_model/training/optimizer.py → apt_core/training/optimizer.py
   - [ ] apt_model/training/data_loading.py → apt_core/training/data_loading.py

4. **迁移推理运行时**
   - [ ] apt_model/runtime/decoder/ → apt_core/runtime/decoder/

5. **迁移配置系统**
   - [ ] apt_model/config/apt_config.py → apt_core/config/apt_config.py
   - [ ] 创建 apt_core/config/base.py

6. **迁移多语言支持**
   - [ ] apt/multilingual/ → apt_core/multilingual/
   - [ ] apt_model/codecs/ → apt_core/codecs/

7. **更新导入路径**
   - [ ] 创建 apt_core/__init__.py（暴露公共 API）
   - [ ] 在 apt_model/__init__.py 中添加兼容性重定向

8. **测试**
   - [ ] 创建 tests/l0_kernel/test_apt_model.py
   - [ ] 创建 tests/l0_kernel/test_training_loop.py
   - [ ] 运行冒烟测试

```bash
# 执行脚本
bash scripts/migration/phase1_l0_kernel.sh

# 验证
python -m pytest tests/l0_kernel/ -v
python examples/core_minimal.py
```

#### 验收标准
- ✅ apt-core 可独立 import
- ✅ 基础训练循环可运行
- ✅ 测试覆盖率 > 90%
- ✅ 性能无退化（与旧版对比）

---

### 阶段 2: L1 性能层迁移（1 周）

**目标**: 迁移所有性能优化模块

#### 任务清单
1. **迁移虚拟 Blackwell**
   - [ ] apt_model/optimization/ → apt_perf/optimization/（全部）
   - [ ] 创建 apt_perf/vb_global.py（全局入口）

2. **迁移训练优化**
   - [ ] apt_model/training/mixed_precision.py → apt_perf/training/
   - [ ] apt_model/training/checkpoint.py → apt_perf/training/
   - [ ] 创建 apt_perf/training/distributed.py（整合 DDP/FSDP）

3. **迁移压缩模块**
   - [ ] apt_model/plugins/compression_plugin.py → apt_perf/compression/
   - [ ] 重构为模块化

4. **创建性能入口**
   - [ ] 创建 apt_perf/__init__.py
   - [ ] 创建 enable_performance() API

5. **测试**
   - [ ] 创建 tests/l1_performance/test_vgpu_stack.py
   - [ ] 创建 tests/l1_performance/test_quantization.py
   - [ ] 性能基准测试

```bash
# 执行脚本
bash scripts/migration/phase2_l1_performance.sh

# 验证
python -m pytest tests/l1_performance/ -v
python examples/perf_benchmark.py
```

#### 验收标准
- ✅ 虚拟 Blackwell 可一键启用
- ✅ 性能提升与旧版一致（3-10×）
- ✅ 可独立于 L0 测试
- ✅ 不影响 L0 语义

---

### 阶段 3: L2 记忆层迁移（1 周）

**目标**: 迁移记忆系统和 GraphRAG

#### 任务清单
1. **迁移 GraphRAG**
   - [ ] apt_model/core/graph_rag/ → apt_memory/graph_rag/

2. **迁移知识图谱集成**
   - [ ] apt_model/modeling/knowledge_graph.py → apt_memory/graph_rag/kg_integration.py
   - [ ] apt_model/modeling/kg_rag_integration.py → 合并到上述文件

3. **创建 AIM 系统**
   - [ ] 创建 apt_memory/aim/（新增）
   - [ ] 实现 AIM-Memory
   - [ ] 实现 AIM-NC
   - [ ] 实现锚点主权（AnchorFields）

4. **创建长上下文支持**
   - [ ] 创建 apt_memory/long_context/
   - [ ] 实现 RoPE 变体

5. **创建统一接口**
   - [ ] 创建 apt_memory/memory_interface.py
   - [ ] 定义 get_context() 标准接口

6. **测试**
   - [ ] 创建 tests/l2_memory/test_aim_memory.py
   - [ ] 创建 tests/l2_memory/test_graph_rag.py
   - [ ] 创建 tests/l2_memory/test_anchor_sovereignty.py（契约测试）

```bash
# 执行脚本
bash scripts/migration/phase3_l2_memory.sh

# 验证
python -m pytest tests/l2_memory/ -v --contract
python examples/mind_rag.py
```

#### 验收标准
- ✅ 记忆系统可独立运行
- ✅ 锚点主权规则强制执行
- ✅ 长上下文支持 8K+ tokens
- ✅ RAG 命中率 > 90%

---

### 阶段 4: L3 应用层迁移（1 周）

**目标**: 迁移所有用户界面和可观测性系统

#### 任务清单
1. **迁移 WebUI**
   - [ ] apt_model/webui/ → apps/webui/
   - [ ] 重组为 4 个 Tab

2. **迁移 REST API**
   - [ ] apt_model/api/ → apps/api/

3. **迁移 CLI**
   - [ ] apt_model/cli/ → apps/cli/
   - [ ] apt_model/interactive/ → apps/cli/interactive/

4. **创建可观测性系统**
   - [ ] 创建 apps/observability/collectors/
   - [ ] 创建 apps/observability/visualization/
   - [ ] 创建 apps/observability/dashboards/
   - [ ] 从 utils/visualization.py 和 training_monitor.py 重组

5. **迁移插件系统**
   - [ ] apt_model/plugins/ → apps/plugins/
   - [ ] apt_model/console/ → apps/console/

6. **迁移 Agent 系统**
   - [ ] apt_model/agent/ → apps/agent/（如果存在）

7. **测试**
   - [ ] 创建 tests/l3_product/test_webui.py（冒烟测试）
   - [ ] 创建 tests/l3_product/test_api.py
   - [ ] 创建 tests/l3_product/test_observability.py

```bash
# 执行脚本
bash scripts/migration/phase4_l3_product.sh

# 验证
python -m pytest tests/l3_product/ -v --smoke
python -m apps.webui.app --test-mode
```

#### 验收标准
- ✅ WebUI 可正常启动
- ✅ API 所有端点正常工作
- ✅ 可观测性数据流正常
- ✅ 插件可正常加载

---

### 阶段 5: 兼容性与清理（3 天）

**目标**: 确保兼容性，废弃旧路径

#### 任务清单
1. **创建兼容性层**
   - [ ] 在 apt_model/__init__.py 中添加完整的重定向
   - [ ] 添加废弃警告（DeprecationWarning）

```python
# apt_model/__init__.py
import warnings

warnings.warn(
    "apt_model is deprecated. Please use apt_core, apt_perf, apt_memory, or apps instead.",
    DeprecationWarning,
    stacklevel=2
)

# 重定向
from apt_core.modeling import APTModel
from apt_core.training import Trainer
from apt_core.generation import Generator
from apt_perf.optimization import enable_virtual_blackwell
from apt_memory import enable_memory_system

__all__ = ['APTModel', 'Trainer', 'Generator', 'enable_virtual_blackwell', 'enable_memory_system']
```

2. **更新所有示例代码**
   - [ ] examples/ 下所有文件更新导入路径
   - [ ] 添加新的示例（core_minimal.py, perf_benchmark.py, mind_rag.py）

3. **更新文档**
   - [ ] 更新 README.md
   - [ ] 更新 INSTALLATION.md
   - [ ] 重写 COMPLETE_TECH_SUMMARY.md（按层级编排）
   - [ ] 创建 MIGRATION_GUIDE.md（帮助用户迁移）

4. **更新测试**
   - [ ] 确保所有旧测试仍然通过（兼容性）
   - [ ] 添加新的分层测试

5. **依赖检查**
   - [ ] 创建 scripts/migration/check_dependencies.py
   - [ ] 验证无反向依赖（L0 不依赖 L3）

6. **清理**
   - [ ] 移除 apt_model/ 中的代码（保留 __init__.py）
   - [ ] 更新 .gitignore
   - [ ] 更新 repo_index.json

```bash
# 执行脚本
bash scripts/migration/phase5_cleanup.sh

# 验证依赖规则
python scripts/migration/check_dependencies.py
```

#### 验收标准
- ✅ 所有旧代码仍可运行（兼容性）
- ✅ 所有测试通过（包括旧测试）
- ✅ 依赖检查通过（无反向依赖）
- ✅ 文档更新完成

---

## 🔧 迁移工具

### 1. 自动迁移脚本

```bash
# scripts/migration/migrate_to_new_structure.py
import os
import shutil
from pathlib import Path

MIGRATION_MAP = {
    # L0
    'apt_model/modeling/apt_model.py': 'apt_core/modeling/apt_model.py',
    'apt_model/modeling/blocks/': 'apt_core/modeling/blocks/',
    # ... 完整映射表
}

def migrate_file(src, dst):
    """迁移单个文件"""
    src_path = Path(src)
    dst_path = Path(dst)

    # 创建目标目录
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # 复制文件（保留原文件，先不删除）
    shutil.copy2(src_path, dst_path)
    print(f"✅ {src} → {dst}")

def update_imports(file_path):
    """更新导入路径"""
    with open(file_path, 'r') as f:
        content = f.read()

    # 替换导入路径
    content = content.replace('from apt_model.modeling', 'from apt_core.modeling')
    content = content.replace('from apt_model.training', 'from apt_core.training')
    # ... 更多替换

    with open(file_path, 'w') as f:
        f.write(content)

def main():
    for src, dst in MIGRATION_MAP.items():
        if os.path.exists(src):
            migrate_file(src, dst)
            update_imports(dst)

if __name__ == '__main__':
    main()
```

### 2. 依赖检查脚本

```python
# scripts/migration/check_dependencies.py
import ast
import os
from pathlib import Path

LAYER_RULES = {
    'apt_core': [],  # L0 不能依赖任何层
    'apt_perf': ['apt_core'],  # L1 只能依赖 L0
    'apt_memory': ['apt_core'],  # L2 只能依赖 L0
    'apps': ['apt_core', 'apt_perf', 'apt_memory'],  # L3 可以依赖 L0/L1/L2
}

FORBIDDEN_PATTERNS = [
    ('apt_core', 'apps'),  # L0 不能依赖 L3
    ('apt_core', 'experiments'),
    ('apt_core', 'tools'),
    ('apt_perf', 'apps'),
    ('apt_perf', 'apt_memory'),  # L1 不能依赖 L2
    ('apt_memory', 'apps'),
]

def check_file_imports(file_path):
    """检查单个文件的导入"""
    with open(file_path, 'r') as f:
        try:
            tree = ast.parse(f.read())
        except:
            return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    return imports

def validate_dependencies():
    """验证所有依赖规则"""
    violations = []

    for layer_dir in ['apt_core', 'apt_perf', 'apt_memory', 'apps']:
        if not os.path.exists(layer_dir):
            continue

        for py_file in Path(layer_dir).rglob('*.py'):
            imports = check_file_imports(py_file)

            for imp in imports:
                # 检查禁止的依赖
                for src_pattern, dst_pattern in FORBIDDEN_PATTERNS:
                    if layer_dir.startswith(src_pattern) and imp.startswith(dst_pattern):
                        violations.append({
                            'file': str(py_file),
                            'import': imp,
                            'reason': f'{src_pattern} cannot import {dst_pattern}'
                        })

    if violations:
        print("❌ 依赖规则违规:\n")
        for v in violations:
            print(f"  {v['file']}")
            print(f"    导入: {v['import']}")
            print(f"    原因: {v['reason']}\n")
        return False
    else:
        print("✅ 所有依赖规则检查通过")
        return True

if __name__ == '__main__':
    import sys
    sys.exit(0 if validate_dependencies() else 1)
```

### 3. 批量更新导入脚本

```bash
# scripts/migration/update_imports.sh
#!/bin/bash

echo "更新导入路径..."

# 更新所有 Python 文件
find apt_core apt_perf apt_memory apps examples tests -name "*.py" -type f -exec sed -i \
  -e 's/from apt_model\.modeling/from apt_core.modeling/g' \
  -e 's/from apt_model\.training/from apt_core.training/g' \
  -e 's/from apt_model\.optimization/from apt_perf.optimization/g' \
  -e 's/from apt_model\.graph_rag/from apt_memory.graph_rag/g' \
  -e 's/from apt_model\.webui/from apps.webui/g' \
  -e 's/from apt_model\.api/from apps.api/g' \
  {} \;

echo "✅ 导入路径更新完成"
```

---

## 📊 进度跟踪

### 总体进度

| 阶段 | 状态 | 开始日期 | 完成日期 | 负责人 |
|------|------|---------|---------|--------|
| 阶段 0: 准备 | 🟡 进行中 | 2025-01-21 | - | Claude |
| 阶段 1: L0 | ⚪ 未开始 | - | - | - |
| 阶段 2: L1 | ⚪ 未开始 | - | - | - |
| 阶段 3: L2 | ⚪ 未开始 | - | - | - |
| 阶段 4: L3 | ⚪ 未开始 | - | - | - |
| 阶段 5: 清理 | ⚪ 未开始 | - | - | - |

### 关键指标

| 指标 | 当前 | 目标 | 进度 |
|------|------|------|------|
| 文件迁移 | 0/211 | 211 | 0% |
| 测试覆盖率 | 85% | 95% | - |
| 依赖检查 | - | 通过 | - |
| 文档更新 | 3/45 | 45 | 7% |

---

## ⚠️ 风险与缓解

### 风险 1: 破坏现有功能

**缓解措施**:
- 保留 apt_model/ 目录，添加兼容性重定向
- 所有旧测试必须通过
- 分阶段迁移，每阶段独立验证

### 风险 2: 导入路径混乱

**缓解措施**:
- 使用自动化脚本批量更新
- 依赖检查脚本强制执行
- 清晰的迁移文档

### 风险 3: 性能退化

**缓解措施**:
- 每阶段运行性能基准测试
- 对比迁移前后的性能
- 发现问题立即修复

### 风险 4: 用户升级困难

**缓解措施**:
- 创建 MIGRATION_GUIDE.md
- 提供兼容性层（至少保留 6 个月）
- 在文档中提供清晰的升级路径

---

## 📚 相关文档

- [ARCHITECTURE.md](./ARCHITECTURE.md) - 架构设计
- [DISTRIBUTION_MODES.md](./DISTRIBUTION_MODES.md) - 发行版说明
- [MIGRATION_GUIDE.md](./MIGRATION_GUIDE.md) - 用户迁移指南（待创建）

---

## 🎯 下一步行动

1. **立即执行**:
   ```bash
   # 1. 创建新目录结构
   bash scripts/migration/phase0_prepare.sh

   # 2. 运行依赖检查（确保当前代码正常）
   python scripts/migration/check_dependencies.py

   # 3. 开始阶段 1（L0 迁移）
   bash scripts/migration/phase1_l0_kernel.sh
   ```

2. **审查与确认**:
   - [ ] 团队审查本迁移计划
   - [ ] 确认时间表
   - [ ] 分配责任人

3. **沟通**:
   - [ ] 通知所有贡献者
   - [ ] 在 GitHub 创建 milestone
   - [ ] 更新 CONTRIBUTING.md

---

**版本**: 1.0
**作者**: APT Team
**日期**: 2025-01-21
**状态**: 🟡 准备阶段
**预计完成**: 2025-02-15 (4 周)
