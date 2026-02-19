#!/bin/bash
#SBATCH --job-name=apt_baseline
#SBATCH --account=ENT114035
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --output=baseline_%j.out
#SBATCH --error=baseline_%j.err

module load miniconda3/24.11.1

cd /work/twsuday816/APT-Transformer

echo "============================================"
echo "APT 基线测试 - 无虚拟GPU"
echo "节点: ${SLURM_JOB_NODELIST}"
echo "开始时间: $(date)"
echo "============================================"

# 基线测试 - 不使用任何虚拟GPU功能
srun python -m apt.trainops.scripts.pretrain_quickcook \
    --output-dir ./test_baseline_output \
    --max-steps 50 \
    --save-interval 25 \
    --weight-fineweb 0.7 \
    --weight-hlbd 0.3 \
    --no-c4 \
    --no-mc4 \
    --batch-size 4 \
    --gradient-accumulation 2

echo "============================================"
echo "基线测试完成时间: $(date)"
echo "============================================"
