#!/bin/bash
# WSL2 完整环境安装脚本
# 用于测试 APT-Transformer Virtual VRAM

set -e  # 遇到错误立即退出

echo "========================================"
echo "WSL2 环境安装脚本"
echo "========================================"
echo ""

# 1. 安装基础工具
echo "[1/6] 安装基础工具 (pip, wget, git)..."
sudo apt update
sudo apt install -y python3-pip wget git curl
echo "✓ 基础工具安装完成"
echo ""

# 2. 升级 pip
echo "[2/6] 升级 pip..."
python3 -m pip install --upgrade pip
echo "✓ pip 升级完成"
echo ""

# 3. 安装 PyTorch (CUDA 12.1 版本，适用于大多数 GPU)
echo "[3/6] 安装 PyTorch (约 2GB，需要时间)..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo "✓ PyTorch 安装完成"
echo ""

# 4. 安装 Transformers 和 Datasets
echo "[4/6] 安装 Transformers 和 Datasets..."
pip3 install datasets transformers accelerate huggingface_hub
echo "✓ Transformers 安装完成"
echo ""

# 5. 验证安装
echo "[5/6] 验证安装..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Count: {torch.cuda.device_count()}')
else:
    print('Warning: No CUDA device found')
"

import transformers, datasets, accelerate
print(f'Transformers: {transformers.__version__}')
print(f'Datasets: {datasets.__version__}')
print(f'Accelerate: {accelerate.__version__}')
echo "✓ 验证完成"
echo ""

# 6. 清理 apt 缓存
echo "[6/6] 清理缓存..."
sudo apt clean
echo "✓ 清理完成"
echo ""

echo "========================================"
echo "安装完成！"
echo "========================================"
echo ""
echo "下一步："
echo "  cd /mnt/d/APT-Transformer"
echo "  python3 -m apt.trainops.scripts.pretrain_quickcook --help"
echo ""
