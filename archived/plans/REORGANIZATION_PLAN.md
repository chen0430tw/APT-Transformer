# APT-Transformer 根目录整理方案

## 📊 当前问题分析

### 根目录文件统计
- **Markdown文档**: 11个（太多，应该移到docs/）
- **训练脚本**: 8个（应该集中到training/）
- **工具脚本**: 9个（应该移到tools/）
- **测试脚本**: 4个（quick_test.*应该移到scripts/testing/）
- **数据文件**: 1个（HLBD_Hardcore_Full.json应该移到data/）

## 🎯 整理方案

### 1. 创建新目录结构

```
APT-Transformer/
├── training/           # 新建：所有训练脚本
├── tools/              # 新建：诊断和工具脚本
├── data/               # 新建：数据集文件
└── archived/           # 新建：过时/临时文件
```

### 2. 文件迁移计划

#### 📂 `training/` - 训练脚本（8个文件）
```
✅ train.py                    → training/train.py
✅ train_apt_playground.py     → training/train_apt_playground.py
✅ train_azure_ml.py           → training/train_azure_ml.py
✅ train_control_experiment.py → training/train_control_experiment.py
✅ train_deepspeed.py          → training/train_deepspeed.py
✅ train_hf_trainer.py         → training/train_hf_trainer.py
✅ train_hlbd_playground.py    → training/train_hlbd_playground.py
✅ training_resume_guide.py    → training/resume_guide.py
```

#### 🔧 `tools/` - 工具脚本（9个文件）
```
✅ check_training_backends.py  → tools/check_training_backends.py
✅ diagnose_issues.py          → tools/diagnose_issues.py
✅ generate_hlbd_hardcore.py   → tools/generate_hlbd_hardcore.py
✅ monitor_all_trainings.py    → tools/monitor_all_trainings.py
✅ verify_hlbd_model.py        → tools/verify_hlbd_model.py
✅ visualize_training.py       → tools/visualize_training.py
✅ demo_visualization.py       → tools/demo_visualization.py
✅ test_vocab_size.py          → tools/test_vocab_size.py
✅ mascot_render_fused45.py    → tools/mascot_render_fused45.py
```

#### 📊 `data/` - 数据文件（1个文件）
```
✅ HLBD_Hardcore_Full.json     → data/HLBD_Hardcore_Full.json
```

#### 📚 `docs/` - 文档整理（移动额外文档）
```
保留根目录:
- README.md                    # 主文档，保留
- INSTALLATION.md              # 安装指南，保留

移动到 docs/:
✅ TRAINING_BACKENDS.md        → docs/TRAINING_BACKENDS.md
✅ VISUALIZATION_GUIDE.md      → docs/VISUALIZATION_GUIDE.md
✅ README_TEST.md              → docs/testing/README_TEST.md
✅ 测试工具使用指南.md         → docs/testing/测试工具使用指南.md
✅ command_verification_report.md → docs/reports/command_verification_report.md
```

#### 🗄️ `archived/` - 过时/临时文件（4个文件）
```
✅ PR_DESCRIPTION.md           → archived/pr/PR_DESCRIPTION.md
✅ PR_DESCRIPTION_FULL.md      → archived/pr/PR_DESCRIPTION_FULL.md
✅ PULL_REQUEST.md             → archived/pr/PULL_REQUEST.md
✅ CONFLICT_RESOLUTION.md      → archived/pr/CONFLICT_RESOLUTION.md
```

#### 🧪 `scripts/testing/` - 测试脚本（4个文件）
```
✅ test_all_commands.py        → scripts/testing/test_all_commands.py
✅ quick_test.sh               → scripts/testing/quick_test.sh
✅ quick_test.bat              → scripts/testing/quick_test.bat
✅ quick_test.ps1              → scripts/testing/quick_test.ps1
```

#### 🔨 `scripts/setup/` - 安装脚本（3个文件）
```
✅ install_dependencies.sh     → scripts/setup/install_dependencies.sh
✅ fix_issues.sh               → scripts/setup/fix_issues.sh
```

#### 📁 `demo_visualization/` - 保持原位置
```
保持不变（已经是文件夹）
```

### 3. 更新引用路径

需要更新以下文件中的路径引用：

#### 文档引用
- `README.md` - 更新文档链接
- `docs/README.md` - 更新所有文档路径
- `TRAINING_BACKENDS.md` → 移动后更新内部引用

#### 脚本引用
- `scripts/testing/quick_test.*` - 更新工具脚本路径
- `tools/generate_hlbd_hardcore.py` - 更新数据输出路径
- `training/*.py` - 更新数据集加载路径

#### Python导入
- 所有训练脚本的相对导入需要保持兼容

### 4. 最终根目录文件列表（整理后）

```
APT-Transformer/
├── README.md                  # ✅ 保留
├── INSTALLATION.md            # ✅ 保留
├── LICENSE                    # ✅ 保留
├── setup.py                   # ✅ 保留
├── requirements*.txt          # ✅ 保留
├── Makefile                   # ✅ 保留
├── MANIFEST.in                # ✅ 保留
├── training/                  # 🆕 新建
├── tools/                     # 🆕 新建
├── data/                      # 🆕 新建
├── archived/                  # 🆕 新建
├── apt_model/                 # ✅ 已存在
├── scripts/                   # ✅ 已存在
├── docs/                      # ✅ 已存在
├── tests/                     # ✅ 已存在
├── examples/                  # ✅ 已存在
└── legacy_plugins/            # ✅ 已存在
```

## 📝 执行步骤

### Step 1: 创建新目录
```bash
mkdir -p training tools data archived/pr docs/testing docs/reports scripts/testing scripts/setup
```

### Step 2: 移动文件（Git mv保留历史）
```bash
# 训练脚本
git mv train*.py training/
git mv training_resume_guide.py training/resume_guide.py

# 工具脚本
git mv check_training_backends.py tools/
git mv diagnose_issues.py tools/
git mv generate_hlbd_hardcore.py tools/
git mv monitor_all_trainings.py tools/
git mv verify_hlbd_model.py tools/
git mv visualize_training.py tools/
git mv demo_visualization.py tools/
git mv test_vocab_size.py tools/
git mv mascot_render_fused45.py tools/

# 数据文件
git mv HLBD_Hardcore_Full.json data/

# 文档
git mv TRAINING_BACKENDS.md docs/
git mv VISUALIZATION_GUIDE.md docs/
git mv README_TEST.md docs/testing/
git mv 测试工具使用指南.md docs/testing/
git mv command_verification_report.md docs/reports/

# PR相关
git mv PR_DESCRIPTION*.md PULL_REQUEST.md CONFLICT_RESOLUTION.md archived/pr/

# 测试脚本
git mv test_all_commands.py scripts/testing/
git mv quick_test.* scripts/testing/

# 安装脚本
git mv install_dependencies.sh fix_issues.sh scripts/setup/
```

### Step 3: 更新路径引用
- 更新 README.md 中的文档链接
- 更新 docs/README.md 中的路径
- 更新 scripts/testing/quick_test.* 中的工具路径
- 更新 tools/generate_hlbd_hardcore.py 的输出路径为 data/

### Step 4: 创建迁移说明
- 在各个新目录创建 README.md 说明文件用途

## ✅ 整理后的好处

1. **清晰的目录结构** - 一眼就能找到需要的文件
2. **更好的可维护性** - 相关文件集中管理
3. **专业的项目组织** - 符合大型项目标准
4. **减少根目录混乱** - 只保留必要的核心文件
5. **便于新手理解** - 清晰的功能分区

## ⚠️ 注意事项

1. 使用 `git mv` 保留文件历史
2. 测试所有路径引用是否正确
3. 更新文档中的所有链接
4. CI/CD配置可能需要更新路径
