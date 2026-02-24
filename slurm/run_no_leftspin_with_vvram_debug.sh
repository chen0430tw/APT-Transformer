#!/bin/bash
#SBATCH --job-name=apt_vvram_debug
#SBATCH --account=ENT114035
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=01:00:00
#SBATCH --output=no_leftspin_vvram_debug_%j.out
#SBATCH --error=no_leftspin_vvram_debug_%j.err

module load miniconda3/24.11.1

cd /work/twsuday816/APT-Transformer

echo "============================================"
echo "APT 测试 - Virtual VRAM + ANOMALY DETECTION"
echo "节点: ${SLURM_JOB_NODELIST}"
echo "开始时间: $(date)"
echo "============================================"

# 使用debug wrapper运行训练（已启用 anomaly detection）
srun python disable_leftspin_wrapper_debug.py \
    --output-dir ./test_no_leftspin_vvram_debug \
    --max-steps 10 \
    --save-interval 10 \
    --weight-fineweb 0.7 \
    --weight-hlbd 0.3 \
    --no-c4 \
    --no-mc4 \
    --batch-size 4 \
    --gradient-accumulation 2 \
    --use-virtual-vram

echo "============================================"
echo "测试完成时间: $(date)"
echo "============================================"
