#!/bin/bash
#SBATCH --job-name=v16_nested
#SBATCH --partition=normal
#SBATCH --account=ENT114035
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load miniconda3/24.11.1

cd /work/twsuday816/APT-Transformer

echo "========================================"
echo "  Virtual VRAM v1.6 嵌套架构测试"
echo "  LECaC → Page → Block → Arc"
echo "========================================"

srun python -m apt.trainops.scripts.pretrain_quickcook \
    --output-dir ./test_v16_nested \
    --max-steps 10 \
    --save-interval 10 \
    --weight-fineweb 0.7 \
    --weight-hlbd 0.3 \
    --no-c4 \
    --no-mc4 \
    --batch-size 4 \
    --gradient-accumulation 2 \
    --use-virtual-vram \
    --vram-verbose \
    --vram-enable-nested-v16 \
    --vram-nested-block-size 64 \
    --vram-nested-quantization-bits 8 \
    --vram-enable-prefetch \
    --vram-prefetch-cache-size 5

echo "========================================"
echo "  v1.6 测试完成"
echo "========================================"
