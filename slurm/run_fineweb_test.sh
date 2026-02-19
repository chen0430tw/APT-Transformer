#!/bin/bash
#SBATCH --job-name=apt_fineweb_test
#SBATCH --account=ENT114035
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --output=fineweb_test_%j.out
#SBATCH --error=fineweb_test_%j.err

module load miniconda3/24.11.1

cd /work/twsuday816/APT-Transformer

echo "============================================"
echo "APT FineWeb测试 - 1 GPU"
echo "节点: ${SLURM_JOB_NODELIST}"
echo "开始时间: $(date)"
echo "============================================"

# 只使用FineWeb（避开C4/mC4兼容性问题）
srun python -m apt.trainops.scripts.pretrain_quickcook \
    --output-dir ./test_fineweb_only \
    --max-steps 100 \
    --save-interval 50 \
    --no-c4 \
    --no-mc4 \
    --weight-fineweb 0.7 \
    --weight-hlbd 0.3 \
    --batch-size 4 \
    --gradient-accumulation 2

echo "============================================"
echo "完成时间: $(date)"
echo "============================================"
