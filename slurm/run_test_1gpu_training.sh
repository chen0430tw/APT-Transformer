#!/bin/bash
#SBATCH --job-name=apt_test_1gpu
#SBATCH --account=ENT114035
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --output=test_1gpu_%j.out
#SBATCH --error=test_1gpu_%j.err

module load miniconda3/24.11.1

cd /work/twsuday816/APT-Transformer

echo "============================================"
echo "APT 快速测试 - 1 GPU"
echo "节点: ${SLURM_JOB_NODELIST}"
echo "GPU数: ${SLURM_JOB_NUM_GPUS}"
echo "开始时间: $(date)"
echo "============================================"

# 快速测试 - 100步验证训练流程
srun python -m apt.trainops.scripts.pretrain_quickcook \
    --output-dir ./test_output_1gpu \
    --max-steps 100 \
    --save-interval 50 \
    --weight-c4 0.5 \
    --weight-mc4 0.3 \
    --weight-hlbd 0.2 \
    --batch-size 4 \
    --gradient-accumulation 2

echo "============================================"
echo "测试完成时间: $(date)"
echo "============================================"
