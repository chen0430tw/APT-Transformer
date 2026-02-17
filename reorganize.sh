#!/bin/bash
# APT-Transformer 根目录自动整理脚本

set -e  # 遇到错误立即退出

echo "🚀 开始整理 APT-Transformer 根目录..."
echo ""

# Step 1: 创建新目录
echo "📁 Step 1: 创建新目录结构..."
mkdir -p training
mkdir -p tools
mkdir -p data
mkdir -p archived/pr
mkdir -p docs/testing
mkdir -p docs/reports
mkdir -p scripts/testing
mkdir -p scripts/setup
echo "   ✅ 目录创建完成"
echo ""

# Step 2: 移动训练脚本
echo "🚂 Step 2: 移动训练脚本到 training/..."
for file in train.py train_apt_playground.py train_azure_ml.py train_control_experiment.py \
            train_deepspeed.py train_hf_trainer.py train_hlbd_playground.py; do
    if [ -f "$file" ]; then
        git mv "$file" training/
        echo "   ✓ $file → training/"
    fi
done

if [ -f "training_resume_guide.py" ]; then
    git mv training_resume_guide.py training/resume_guide.py
    echo "   ✓ training_resume_guide.py → training/resume_guide.py"
fi
echo ""

# Step 3: 移动工具脚本
echo "🔧 Step 3: 移动工具脚本到 tools/..."
for file in check_training_backends.py diagnose_issues.py generate_hlbd_hardcore.py \
            monitor_all_trainings.py verify_hlbd_model.py visualize_training.py \
            demo_visualization.py test_vocab_size.py mascot_render_fused45.py; do
    if [ -f "$file" ]; then
        git mv "$file" tools/
        echo "   ✓ $file → tools/"
    fi
done
echo ""

# Step 4: 移动数据文件
echo "📊 Step 4: 移动数据文件到 data/..."
if [ -f "HLBD_Hardcore_Full.json" ]; then
    git mv HLBD_Hardcore_Full.json data/
    echo "   ✓ HLBD_Hardcore_Full.json → data/"
fi
echo ""

# Step 5: 移动文档
echo "📚 Step 5: 整理文档到 docs/..."
if [ -f "TRAINING_BACKENDS.md" ]; then
    git mv TRAINING_BACKENDS.md docs/
    echo "   ✓ TRAINING_BACKENDS.md → docs/"
fi

if [ -f "VISUALIZATION_GUIDE.md" ]; then
    git mv VISUALIZATION_GUIDE.md docs/
    echo "   ✓ VISUALIZATION_GUIDE.md → docs/"
fi

if [ -f "README_TEST.md" ]; then
    git mv README_TEST.md docs/testing/
    echo "   ✓ README_TEST.md → docs/testing/"
fi

if [ -f "测试工具使用指南.md" ]; then
    git mv 测试工具使用指南.md docs/testing/
    echo "   ✓ 测试工具使用指南.md → docs/testing/"
fi

if [ -f "command_verification_report.md" ]; then
    git mv command_verification_report.md docs/reports/
    echo "   ✓ command_verification_report.md → docs/reports/"
fi
echo ""

# Step 6: 归档PR相关文件
echo "🗄️  Step 6: 归档PR相关文件到 archived/pr/..."
for file in PR_DESCRIPTION.md PR_DESCRIPTION_FULL.md PULL_REQUEST.md CONFLICT_RESOLUTION.md; do
    if [ -f "$file" ]; then
        git mv "$file" archived/pr/
        echo "   ✓ $file → archived/pr/"
    fi
done
echo ""

# Step 7: 移动测试脚本
echo "🧪 Step 7: 移动测试脚本到 scripts/testing/..."
if [ -f "test_all_commands.py" ]; then
    git mv test_all_commands.py scripts/testing/
    echo "   ✓ test_all_commands.py → scripts/testing/"
fi

for file in quick_test.sh quick_test.bat quick_test.ps1; do
    if [ -f "$file" ]; then
        git mv "$file" scripts/testing/
        echo "   ✓ $file → scripts/testing/"
    fi
done
echo ""

# Step 8: 移动安装脚本
echo "🔨 Step 8: 移动安装脚本到 scripts/setup/..."
for file in install_dependencies.sh fix_issues.sh; do
    if [ -f "$file" ]; then
        git mv "$file" scripts/setup/
        echo "   ✓ $file → scripts/setup/"
    fi
done
echo ""

# Step 9: 创建各目录的README
echo "📝 Step 9: 创建目录说明文档..."

cat > training/README.md << 'EOF'
# Training Scripts

This directory contains all training scripts for APT-Transformer.

## Available Training Backends

- `train.py` - Unified training launcher (supports all backends)
- `train_apt_playground.py` - APT Playground training
- `train_hlbd_playground.py` - HLBD Playground training
- `train_deepspeed.py` - DeepSpeed distributed training
- `train_azure_ml.py` - Azure ML cloud training
- `train_hf_trainer.py` - HuggingFace Trainer integration
- `train_control_experiment.py` - Control experiment training
- `resume_guide.py` - Training resume guide

## Quick Start

```bash
# List all available backends
python training/train.py --list-backends

# Train with specific backend
python training/train.py --backend playground --epochs 100
```

## Documentation

See [TRAINING_BACKENDS.md](../docs/TRAINING_BACKENDS.md) for detailed usage guide.
EOF

cat > tools/README.md << 'EOF'
# Tools and Utilities

This directory contains diagnostic, generation, and monitoring tools for APT-Transformer.

## Available Tools

### Diagnostic Tools
- `check_training_backends.py` - Code quality checker for training backends
- `diagnose_issues.py` - System diagnostic tool

### Data Generation
- `generate_hlbd_hardcore.py` - HLBD Hardcore dataset generator

### Monitoring & Visualization
- `monitor_all_trainings.py` - Multi-training monitor
- `visualize_training.py` - Real-time sci-fi style visualization
- `demo_visualization.py` - Visualization demo

### Model Verification
- `verify_hlbd_model.py` - HLBD model verification tool

### Other Utilities
- `test_vocab_size.py` - Vocabulary size tester
- `mascot_render_fused45.py` - Mascot rendering tool

## Usage Examples

```bash
# Check training backend code quality
python tools/check_training_backends.py

# Generate HLBD dataset
python tools/generate_hlbd_hardcore.py

# Verify trained model
python tools/verify_hlbd_model.py --model path/to/model.pt

# Real-time visualization
python tools/visualize_training.py --log-dir training_logs --mode realtime
```
EOF

cat > data/README.md << 'EOF'
# Data Directory

This directory contains datasets for APT-Transformer training.

## Available Datasets

- `HLBD_Hardcore_Full.json` - HLBD Hardcore dataset (575 samples)
  - 5 modules: geometry, arithmetic, zodiac, physics, reverse_english
  - Designed to prevent shortcut learning

## Dataset Format

All datasets use JSON format with the following structure:

```json
{
  "metadata": {
    "total_samples": 575,
    "modules": ["geometry", "arithmetic", "zodiac", "physics", "reverse_english"]
  },
  "data": {
    "module_name": [
      {"input": "...", "output": "..."},
      ...
    ]
  }
}
```

## Generating New Datasets

Use the dataset generation tool:

```bash
python tools/generate_hlbd_hardcore.py
```
EOF

cat > archived/README.md << 'EOF'
# Archived Files

This directory contains archived files that are no longer actively used but kept for reference.

## Structure

- `pr/` - Pull request descriptions and conflict resolutions

These files are kept for historical reference and should not be modified.
EOF

echo "   ✅ README文档创建完成"
echo ""

echo "=" * 60
echo "✨ 整理完成！"
echo "=" * 60
echo ""
echo "📊 统计："
echo "   - training/ : $(ls -1 training/*.py 2>/dev/null | wc -l) 个训练脚本"
echo "   - tools/    : $(ls -1 tools/*.py 2>/dev/null | wc -l) 个工具脚本"
echo "   - data/     : $(ls -1 data/*.json 2>/dev/null | wc -l) 个数据文件"
echo "   - docs/     : $(find docs -name '*.md' 2>/dev/null | wc -l) 个文档"
echo "   - archived/ : $(find archived -type f 2>/dev/null | wc -l) 个归档文件"
echo ""
echo "⚠️  下一步："
echo "   1. 运行: bash update_paths.sh  # 更新所有路径引用"
echo "   2. 测试所有功能是否正常"
echo "   3. 提交更改: git commit -m 'Reorganize project structure'"
echo ""
