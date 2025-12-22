#!/bin/bash
# APT项目依赖安装脚本

echo "🔧 APT项目依赖安装"
echo "================================"
echo ""

# 检查Python版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"
echo ""

# 必需依赖
echo "1️⃣  安装核心依赖..."
echo "   - PyTorch (深度学习框架)"
echo "   - 检测CUDA支持..."

if command -v nvidia-smi &> /dev/null; then
    echo "   ✓ 检测到NVIDIA GPU"
    echo "   安装CUDA版PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "   ⚠️  未检测到GPU，安装CPU版PyTorch..."
    pip install torch torchvision torchaudio
fi

echo ""
echo "2️⃣  安装可视化依赖..."
pip install numpy matplotlib

echo ""
echo "3️⃣  安装可选依赖..."
read -p "是否安装HuggingFace datasets? (用于train_apt_playground.py) [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install datasets
    echo "   ✓ datasets 已安装"
else
    echo "   跳过 datasets 安装"
fi

echo ""
echo "================================"
echo "✅ 依赖安装完成！"
echo "================================"
echo ""
echo "验证安装:"
python3 -c "
import torch
print(f'  ✓ PyTorch {torch.__version__}')
print(f'  ✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  ✓ CUDA version: {torch.version.cuda}')
    print(f'  ✓ GPU: {torch.cuda.get_device_name(0)}')

import numpy
print(f'  ✓ NumPy {numpy.__version__}')

import matplotlib
print(f'  ✓ Matplotlib {matplotlib.__version__}')

try:
    import datasets
    print(f'  ✓ Datasets {datasets.__version__}')
except ImportError:
    print('  ⚠️  Datasets 未安装（可选）')
"

echo ""
echo "下一步:"
echo "1. 生成HLBD数据集: python generate_hlbd_hardcore.py"
echo "2. 训练模型: python tests/test_hlbd_quick_learning.py --epochs 100"
echo "3. 验证模型: python verify_hlbd_model.py --model <模型路径>"
echo "4. 可视化: python visualize_training.py --log-dir demo_visualization --offline"
