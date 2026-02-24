#!/bin/bash
#SBATCH --job-name=apt_baseline_no_vvram
#SBATCH --account=ENT114035
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --output=baseline_no_vvram_%j.out
#SBATCH --error=baseline_no_vvram_%j.err

module load miniconda3/24.11.1

cd /work/twsuday816/APT-Transformer

echo "============================================"
echo "APT Baseline测试 - 无Virtual VRAM，保留LeftSpin"
echo "节点: ${SLURM_JOB_NODELIST}"
echo "开始时间: $(date)"
echo "============================================"

# 测试：禁用Virtual VRAM，使用LeftSpin
srun python -m apt.trainops.scripts.pretrain_quickcook \
    --output-dir ./test_baseline_no_vvram \
    --max-steps 10 \
    --save-interval 10 \
    --weight-fineweb 0.7 \
    --weight-hlbd 0.3 \
    --no-c4 \
    --no-mc4 \
    --batch-size 4 \
    --gradient-accumulation 2

echo "============================================"
echo "Baseline测试完成时间: $(date)"
echo "============================================"
