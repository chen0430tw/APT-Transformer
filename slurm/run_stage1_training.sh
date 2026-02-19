#!/bin/bash
#SBATCH --job-name=apt_stage1
#SBATCH --account=ENT114035
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --output=stage1_%j.out
#SBATCH --error=stage1_%j.err

module load miniconda3/24.11.1

cd /work/twsuday816/APT-Transformer

echo "============================================"
echo "APT Stage 1 训练 - 多语言底噪"
echo "节点: ${SLURM_JOB_NODELIST}"
echo "GPU数: ${SLURM_JOB_NUM_GPUS}"
echo "开始时间: $(date)"
echo "============================================"

# Stage 1: 通用预训练底噪
# C4 + FineWeb + Chinese-C4 + HLBD
srun torchrun --nproc_per_node=8 \
    -m apt.trainops.scripts.pretrain_quickcook \
    --output-dir ./stage1_output \
    --epochs 1 \
    --max-steps 10000 \
    --save-interval 1000 \
    --weight-c4 0.35 \
    --weight-fineweb 0.25 \
    --weight-mc4 0.20 \
    --weight-hlbd 0.20 \
    --batch-size 8 \
    --grad-accum 1

echo "============================================"
echo "训练完成时间: $(date)"
echo "============================================"
