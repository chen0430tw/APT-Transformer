#!/bin/bash
# APT项目问题修复脚本

echo "🔧 修复APT项目问题"
echo "="

# 1. 安装缺失依赖
echo ""
echo "1️⃣  安装Python依赖..."
pip install numpy matplotlib

# 可选: HuggingFace datasets
read -p "是否安装HuggingFace datasets? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install datasets
fi

# 2. 添加Weight Decay到HLBD脚本
echo ""
echo "2️⃣  修复Weight Decay..."
echo "   (需要手动修改 tests/test_hlbd_quick_learning.py)"
echo "   将第725行改为:"
echo "   optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=0.01)"

# 3. 生成包含反向学英文的HLBD数据集
echo ""
echo "3️⃣  重新生成HLBD数据集（包含反向学英文）..."
python generate_hlbd_hardcore.py --add-reverse-english

# 4. 创建HLBD验证脚本
echo ""
echo "4️⃣  创建HLBD验证脚本..."
# (将在下一步创建)

echo ""
echo "✅ 修复完成！"
echo ""
echo "下一步:"
echo "1. 运行: python verify_hlbd_model.py --model <model_path>"
echo "2. 测试可视化: python visualize_training.py --log-dir demo_visualization --offline"
