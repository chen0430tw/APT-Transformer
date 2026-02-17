# 📚 文档整理计划

**创建时间**: 2024-12-22
**目标**: 整理项目根目录和docs/文件夹中的重复、分散的Markdown文档

---

## 📊 当前状态分析

### 根目录 (7个MD文件)

| 文件 | 类型 | 处理建议 |
|------|------|----------|
| README.md | 核心 | ✅ 保留 |
| INSTALLATION.md | 核心 | ✅ 保留 |
| BUGFIX_SUMMARY.md | 报告 | 📦 移到 docs/reports/ |
| FINAL_SUMMARY.md | 报告 | 📦 移到 docs/reports/ |
| PR_HLBD_MODULAR_TRAINING.md | PR文档 | 📦 移到 archived/pr/ |
| PR_REORGANIZATION.md | PR文档 | 📦 移到 archived/pr/ |
| REORGANIZATION_PLAN.md | 计划 | 📦 移到 archived/ |

**问题**:
- PR和总结文档占据根目录，应该归档
- 根目录应该只保留核心文档（README + INSTALLATION）

---

## 📁 docs/ 目录 (35个MD文件)

### 🔴 需要合并的文档组

#### 1. DBC-DAC优化系列 (4个 → 1个)

**当前文件**:
```
docs/DBC_DAC_二次优化详解.md
docs/DBC_DAC_优化对比分析.md
docs/DBC_DAC_加速方案分析.md
docs/训练加速优化方案.md
```

**合并为**: `docs/DBC_DAC_OPTIMIZATION_GUIDE.md`

**理由**: 这4个文档都是关于DBC-DAC优化和训练加速的，内容高度重叠

---

#### 2. Plugin系统 (2个 → 1个)

**当前文件**:
```
docs/PLUGINS_USAGE_GUIDE.md
docs/PLUGIN_SYSTEM.md
```

**合并为**: `docs/PLUGIN_SYSTEM_GUIDE.md`

**理由**: Plugin的使用和系统架构应该在同一个文档中

---

#### 3. GPT模型系列 (2个 → 1个)

**当前文件**:
```
docs/GPT_MODELS_ANALYSIS.md
docs/GPT_TRAINING_GUIDE.md
```

**合并为**: `docs/GPT_MODELS_GUIDE.md`

**理由**: GPT模型分析和训练指南应该合并

---

#### 4. Distillation系列 (保留分开)

**当前文件**:
```
docs/DISTILLATION_PRINCIPLE.md  (原理)
docs/VISUAL_DISTILLATION_GUIDE.md  (视觉蒸馏具体实现)
```

**决定**: ✅ 保留分开（一个是通用原理，一个是视觉专门实现）

---

### 🟡 需要归档的文档

#### 移到 archived/reports/

```
docs/BUG_REPORT.md
docs/SELF_SUPERVISED_RL_CHECK_REPORT.md
docs/reports/command_verification_report.md
```

**理由**: 这些是历史报告，不是用户指南

---

#### 移到 archived/plans/

```
docs/MODULE_INTEGRATION_PLAN.md
```

**理由**: 如果集成已完成，这是历史计划文档

---

### ✅ 保留的核心文档 (不动)

```
docs/README.md                    - 文档中心索引
docs/APT_MODEL_HANDBOOK.md        - APT模型完整手册
docs/TRAINING_BACKENDS.md         - 训练后端指南
docs/VISUALIZATION_GUIDE.md       - 可视化指南
docs/FINE_TUNING_GUIDE.md         - 微调指南
docs/DATA_PREPROCESSING_GUIDE.md  - 数据预处理
docs/APX.md                       - APX格式规范
docs/HLBD.md                      - HLBD系统
docs/repo_schema.md               - 项目架构说明
docs/LAUNCHER_README.md           - GUI启动器
docs/DEBUG_MODE_GUIDE.md          - 调试模式
docs/OPTUNA_GUIDE.md              - 超参数优化
docs/training_protection_guide.md - 训练保护
```

**特定模型/技术文档** (保留):
```
docs/CLAUDE4_MODEL_GUIDE.md
docs/DEEPSEEK_TRAINING_GUIDE.md
docs/GRAPH_BRAIN_TRAINING_GUIDE.md
docs/KNOWLEDGE_GRAPH_GUIDE.md
docs/MCP_INTEGRATION_GUIDE.md
docs/RL_PRETRAINING_GUIDE.md
docs/TEACHER_API_GUIDE.md
docs/WEB_SEARCH_PLUGIN_GUIDE.md
docs/API_PROVIDERS_GUIDE.md
```

---

## 🎯 整理操作清单

### Phase 1: 创建新的合并文档

- [ ] 创建 `docs/DBC_DAC_OPTIMIZATION_GUIDE.md`
  - 合并4个DBC-DAC文档
  - 章节: 原理 → 方案对比 → 二次优化 → 训练加速

- [ ] 创建 `docs/PLUGIN_SYSTEM_GUIDE.md`
  - 合并Plugin系统架构 + 使用指南
  - 章节: 架构 → 开发 → 使用 → 示例

- [ ] 创建 `docs/GPT_MODELS_GUIDE.md`
  - 合并GPT分析 + 训练指南
  - 章节: 模型分析 → 训练指南 → 最佳实践

### Phase 2: 移动归档文件

- [ ] 创建 `archived/reports/` 目录
- [ ] 移动 `BUGFIX_SUMMARY.md` → `archived/reports/bugfix_summary_20241222.md`
- [ ] 移动 `FINAL_SUMMARY.md` → `archived/reports/hlbd_modular_final_summary_20241222.md`
- [ ] 移动 `docs/BUG_REPORT.md` → `archived/reports/`
- [ ] 移动 `docs/SELF_SUPERVISED_RL_CHECK_REPORT.md` → `archived/reports/`

- [ ] 创建 `archived/plans/` 目录
- [ ] 移动 `REORGANIZATION_PLAN.md` → `archived/plans/`
- [ ] 移动 `docs/MODULE_INTEGRATION_PLAN.md` → `archived/plans/`

- [ ] 移动 `PR_HLBD_MODULAR_TRAINING.md` → `archived/pr/`
- [ ] 移动 `PR_REORGANIZATION.md` → `archived/pr/`

### Phase 3: 删除旧文件

- [ ] 删除 `docs/DBC_DAC_二次优化详解.md`
- [ ] 删除 `docs/DBC_DAC_优化对比分析.md`
- [ ] 删除 `docs/DBC_DAC_加速方案分析.md`
- [ ] 删除 `docs/训练加速优化方案.md`
- [ ] 删除 `docs/PLUGINS_USAGE_GUIDE.md`
- [ ] 删除 `docs/PLUGIN_SYSTEM.md`
- [ ] 删除 `docs/GPT_MODELS_ANALYSIS.md`
- [ ] 删除 `docs/GPT_TRAINING_GUIDE.md`

### Phase 4: 更新索引

- [ ] 更新 `docs/README.md` - 文档导航
- [ ] 更新 `repo_index.json` - 文件索引
- [ ] 更新 `docs/repo_schema.md` - 架构说明
- [ ] 更新根目录 `README.md` - 链接更新

---

## 📈 整理效果预期

### 根目录

**Before** (7个):
```
README.md
INSTALLATION.md
BUGFIX_SUMMARY.md
FINAL_SUMMARY.md
PR_HLBD_MODULAR_TRAINING.md
PR_REORGANIZATION.md
REORGANIZATION_PLAN.md
```

**After** (2个):
```
README.md
INSTALLATION.md
```

**减少**: 71% (-5个文件)

---

### docs/ 目录

**Before** (35个):
- 核心指南: 23个
- 重复文档: 8个 (需合并)
- 报告文档: 4个 (需归档)

**After** (26个):
- 核心指南: 23个
- 合并后新文档: 3个
- 删除重复: -8个
- 移走报告: -4个

**减少**: 26% (-9个文件)

---

## 🎨 新目录结构

```
APT-Transformer/
├── README.md                    ✅ 主文档
├── INSTALLATION.md              ✅ 安装指南
│
├── docs/
│   ├── README.md                     📚 文档导航（更新后）
│   │
│   ├── 核心指南/
│   │   ├── APT_MODEL_HANDBOOK.md
│   │   ├── TRAINING_BACKENDS.md
│   │   ├── FINE_TUNING_GUIDE.md
│   │   ├── DATA_PREPROCESSING_GUIDE.md
│   │   ├── VISUALIZATION_GUIDE.md
│   │   └── ...
│   │
│   ├── 合并后的新文档/
│   │   ├── DBC_DAC_OPTIMIZATION_GUIDE.md  ✨ 新
│   │   ├── PLUGIN_SYSTEM_GUIDE.md         ✨ 新
│   │   └── GPT_MODELS_GUIDE.md            ✨ 新
│   │
│   ├── 技术专题/
│   │   ├── CLAUDE4_MODEL_GUIDE.md
│   │   ├── DEEPSEEK_TRAINING_GUIDE.md
│   │   ├── GRAPH_BRAIN_TRAINING_GUIDE.md
│   │   └── ...
│   │
│   └── hlbd/                         📁 HLBD专门文档
│       ├── README.md
│       ├── HLBD_MODULAR_TRAINING.md
│       └── ...
│
└── archived/
    ├── reports/                      📦 新建
    │   ├── bugfix_summary_20241222.md
    │   ├── hlbd_modular_final_summary_20241222.md
    │   ├── BUG_REPORT.md
    │   └── SELF_SUPERVISED_RL_CHECK_REPORT.md
    │
    ├── plans/                        📦 新建
    │   ├── REORGANIZATION_PLAN.md
    │   └── MODULE_INTEGRATION_PLAN.md
    │
    └── pr/                           📁 已存在，新增文件
        ├── PR_HLBD_MODULAR_TRAINING.md  ✨ 新移入
        ├── PR_REORGANIZATION.md         ✨ 新移入
        ├── CONFLICT_RESOLUTION.md
        ├── PULL_REQUEST.md
        ├── PR_DESCRIPTION_FULL.md
        └── PR_DESCRIPTION.md
```

---

## ✅ 验证清单

整理完成后需要检查：

- [ ] 所有内部链接已更新
- [ ] README.md 指向正确的文档路径
- [ ] repo_index.json 反映新结构
- [ ] docs/README.md 导航正确
- [ ] 没有孤立的引用（broken links）
- [ ] Git历史保留（使用git mv而非rm+add）

---

## 🚀 执行时间估计

- Phase 1 (创建合并文档): 30-45分钟
- Phase 2 (移动归档): 10-15分钟
- Phase 3 (删除旧文件): 5分钟
- Phase 4 (更新索引): 15-20分钟

**总计**: 约60-85分钟

---

## 💡 注意事项

1. **保留Git历史**: 使用 `git mv` 而非删除重建
2. **链接完整性**: 使用脚本验证所有Markdown链接
3. **备份**: 在大规模删除前创建备份分支
4. **分步提交**: 每个Phase单独提交，便于回滚

---

**状态**: 📋 计划中
**批准**: 待用户确认
