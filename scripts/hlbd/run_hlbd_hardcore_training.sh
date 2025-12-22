#!/bin/bash
# HLBD Hardcore V2 训练启动脚本
# 运行5042样本的严格逻辑数据集训练
#
# 使用方法:
#   bash run_hlbd_hardcore_training.sh
#
# 环境要求:
#   - PyTorch >= 1.9
#   - CUDA (推荐RTX 3070或更高)
#   - 至少8GB GPU内存

echo "=================================================="
echo "HLBD Hardcore V2 训练启动"
echo "数据集: 5042 samples | 数据稀释学 | 防模式坍缩"
echo "=================================================="
echo ""

# 检查CUDA可用性
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "✓ CUDA available"
    python3 -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}')" 2>/dev/null
    python3 -c "import torch; print(f'  CUDA: {torch.version.cuda}')" 2>/dev/null
else
    echo "⚠️  CUDA not available, will use CPU (very slow)"
fi

echo ""
echo "训练配置:"
echo "  数据集: data/HLBD_Hardcore_Full_V2.json"
echo "  Epochs: 50"
echo "  Batch Size: 32"
echo "  学习率: Cosine Annealing with Warm Restarts"
echo "  优化: DBC-DAC梯度稳定"
echo ""

# 检查依赖
echo "检查依赖..."
if ! python3 -c "import torch" 2>/dev/null; then
    echo "❌ PyTorch not found. Please install:"
    echo "   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    exit 1
fi

if ! python3 -c "import numpy" 2>/dev/null; then
    echo "❌ NumPy not found. Please install:"
    echo "   pip install numpy"
    exit 1
fi

echo "✓ All dependencies OK"
echo ""

# 检查数据集
if [ ! -f "data/HLBD_Hardcore_Full_V2.json" ]; then
    echo "❌ Dataset not found: data/HLBD_Hardcore_Full_V2.json"
    echo "   Please run: python tools/generate_hlbd_hardcore_v2.py"
    exit 1
fi

echo "✓ Dataset found ($(du -h data/HLBD_Hardcore_Full_V2.json | cut -f1))"
echo ""

# 创建输出目录
mkdir -p models/hlbd_hardcore_v2
mkdir -p logs/hlbd_hardcore_v2

echo "启动训练..."
echo "输出目录: models/hlbd_hardcore_v2/"
echo "日志目录: logs/hlbd_hardcore_v2/"
echo ""

# 运行训练
python3 training/train_hlbd_playground.py \
    --dataset data/HLBD_Hardcore_Full_V2.json \
    --output-dir models/hlbd_hardcore_v2 \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 5e-5 \
    --warmup-steps 100 \
    --save-every 5 \
    --eval-every 5 \
    --use-amp \
    --gradient-accumulation-steps 2 \
    2>&1 | tee logs/hlbd_hardcore_v2/training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "=================================================="
echo "训练完成！"
echo "=================================================="
echo "模型保存位置: models/hlbd_hardcore_v2/"
echo "日志文件: logs/hlbd_hardcore_v2/"
