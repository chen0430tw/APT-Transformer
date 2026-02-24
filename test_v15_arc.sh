#!/bin/bash
#SBATCH --job-name=apt_v15_arc
#SBATCH --account=ENT114035
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --output=v15_arc_%j.out
#SBATCH --error=v15_arc_%j.err

module load miniconda3/24.11.1

cd /work/twsuday816/APT-Transformer

echo "============================================"
echo "Virtual VRAM v1.5 测试 - Rust 风格 Arc 内存管理"
echo "============================================"

srun python disable_leftspin_wrapper_debug.py \
    --output-dir ./test_v15 \
    --max-steps 10 \
    --save-interval 10 \
    --weight-fineweb 0.7 \
    --weight-hlbd 0.3 \
    --no-c4 \
    --no-mc4 \
    --batch-size 4 \
    --gradient-accumulation 2 \
    --use-virtual-vram \
    --vram-enable-arc-memory \
    --vram-enable-weak-refs \
    --vram-verbose

echo "============================================"
echo "测试完成时间: $(date)"
echo "============================================"
