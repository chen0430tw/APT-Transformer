#!/bin/bash
# 自动更新所有文件中的路径引用

set -e

echo "🔄 开始更新路径引用..."
echo ""

# Step 1: 更新主README.md中的文档链接
echo "📝 Step 1: 更新 README.md..."
if [ -f "README.md" ]; then
    # 更新训练后端文档链接
    sed -i.bak 's|\[训练后端使用指南\](TRAINING_BACKENDS.md)|[训练后端使用指南](docs/TRAINING_BACKENDS.md)|g' README.md
    sed -i.bak 's|TRAINING_BACKENDS\.md|docs/TRAINING_BACKENDS.md|g' README.md
    sed -i.bak 's|VISUALIZATION_GUIDE\.md|docs/VISUALIZATION_GUIDE.md|g' README.md

    # 更新训练命令路径
    sed -i.bak 's|python train\.py|python training/train.py|g' README.md

    echo "   ✓ README.md 路径已更新"
    rm -f README.md.bak
fi
echo ""

# Step 2: 更新 docs/README.md
echo "📚 Step 2: 更新 docs/README.md..."
if [ -f "docs/README.md" ]; then
    # 更新文档路径
    sed -i.bak 's|\.\./TRAINING_BACKENDS\.md|TRAINING_BACKENDS.md|g' docs/README.md
    sed -i.bak 's|\.\./VISUALIZATION_GUIDE\.md|VISUALIZATION_GUIDE.md|g' docs/README.md
    sed -i.bak 's|README_TEST\.md|testing/README_TEST.md|g' docs/README.md

    echo "   ✓ docs/README.md 路径已更新"
    rm -f docs/README.md.bak
fi
echo ""

# Step 3: 更新测试脚本中的工具路径
echo "🧪 Step 3: 更新测试脚本路径..."
for file in scripts/testing/quick_test.*; do
    if [ -f "$file" ]; then
        # 更新Python脚本路径
        sed -i.bak 's|test_all_commands\.py|scripts/testing/test_all_commands.py|g' "$file"
        sed -i.bak 's|check_training_backends\.py|tools/check_training_backends.py|g' "$file"
        sed -i.bak 's|diagnose_issues\.py|tools/diagnose_issues.py|g' "$file"
        sed -i.bak 's|view_test_report\.py|scripts/testing/view_test_report.py|g' "$file"

        # 更新文档路径
        sed -i.bak 's|TRAINING_BACKENDS\.md|docs/TRAINING_BACKENDS.md|g' "$file"
        sed -i.bak 's|VISUALIZATION_GUIDE\.md|docs/VISUALIZATION_GUIDE.md|g' "$file"

        echo "   ✓ $file 路径已更新"
        rm -f "${file}.bak"
    fi
done
echo ""

# Step 4: 更新工具脚本中的数据路径
echo "🔧 Step 4: 更新工具脚本中的数据路径..."
if [ -f "tools/generate_hlbd_hardcore.py" ]; then
    # 更新输出路径
    sed -i.bak "s|'HLBD_Hardcore_Full\.json'|'../data/HLBD_Hardcore_Full.json'|g" tools/generate_hlbd_hardcore.py
    sed -i.bak 's|"HLBD_Hardcore_Full\.json"|"../data/HLBD_Hardcore_Full.json"|g' tools/generate_hlbd_hardcore.py

    echo "   ✓ tools/generate_hlbd_hardcore.py 路径已更新"
    rm -f tools/generate_hlbd_hardcore.py.bak
fi

# 更新verify脚本的默认数据集路径
if [ -f "tools/verify_hlbd_model.py" ]; then
    sed -i.bak "s|default='HLBD_Hardcore_Full\.json'|default='../data/HLBD_Hardcore_Full.json'|g" tools/verify_hlbd_model.py

    echo "   ✓ tools/verify_hlbd_model.py 路径已更新"
    rm -f tools/verify_hlbd_model.py.bak
fi
echo ""

# Step 5: 更新训练脚本中的数据路径
echo "🚂 Step 5: 更新训练脚本中的数据路径..."
for file in training/train*.py; do
    if [ -f "$file" ]; then
        # 更新数据集默认路径
        sed -i.bak "s|default='HLBD_Hardcore_Full\.json'|default='../data/HLBD_Hardcore_Full.json'|g" "$file"
        sed -i.bak 's|"HLBD_Hardcore_Full\.json"|"../data/HLBD_Hardcore_Full.json"|g' "$file"

        echo "   ✓ $(basename $file) 数据路径已更新"
        rm -f "${file}.bak"
    fi
done
echo ""

# Step 6: 更新文档中的示例命令
echo "📖 Step 6: 更新文档中的示例命令..."
if [ -f "docs/TRAINING_BACKENDS.md" ]; then
    # 更新所有train.py引用
    sed -i.bak 's|python train\.py|python training/train.py|g' docs/TRAINING_BACKENDS.md
    sed -i.bak 's|python train_|python training/train_|g' docs/TRAINING_BACKENDS.md

    # 更新工具脚本引用
    sed -i.bak 's|python verify_hlbd_model\.py|python tools/verify_hlbd_model.py|g' docs/TRAINING_BACKENDS.md
    sed -i.bak 's|python visualize_training\.py|python tools/visualize_training.py|g' docs/TRAINING_BACKENDS.md
    sed -i.bak 's|python monitor_all_trainings\.py|python tools/monitor_all_trainings.py|g' docs/TRAINING_BACKENDS.md
    sed -i.bak 's|python diagnose_issues\.py|python tools/diagnose_issues.py|g' docs/TRAINING_BACKENDS.md

    echo "   ✓ docs/TRAINING_BACKENDS.md 命令已更新"
    rm -f docs/TRAINING_BACKENDS.md.bak
fi

if [ -f "docs/VISUALIZATION_GUIDE.md" ]; then
    sed -i.bak 's|python visualize_training\.py|python tools/visualize_training.py|g' docs/VISUALIZATION_GUIDE.md
    sed -i.bak 's|python monitor_all_trainings\.py|python tools/monitor_all_trainings.py|g' docs/VISUALIZATION_GUIDE.md
    sed -i.bak 's|python demo_visualization\.py|python tools/demo_visualization.py|g' docs/VISUALIZATION_GUIDE.md

    echo "   ✓ docs/VISUALIZATION_GUIDE.md 命令已更新"
    rm -f docs/VISUALIZATION_GUIDE.md.bak
fi
echo ""

# Step 7: 创建便捷访问的符号链接（可选）
echo "🔗 Step 7: 创建便捷符号链接..."
# 创建根目录快捷访问
ln -sf training/train.py train.py 2>/dev/null || echo "   ⚠️  符号链接已存在或创建失败"
echo "   ✓ 创建 train.py -> training/train.py 符号链接"
echo ""

echo "=" * 60
echo "✨ 路径更新完成！"
echo "=" * 60
echo ""
echo "🧪 建议测试以下功能："
echo "   1. python training/train.py --list-backends"
echo "   2. python tools/check_training_backends.py"
echo "   3. python tools/diagnose_issues.py"
echo "   4. bash scripts/testing/quick_test.sh"
echo ""
echo "💡 如果测试通过，可以提交更改："
echo "   git add -A"
echo '   git commit -m "Reorganize project structure for better maintainability"'
echo "   git push origin main"
echo ""
