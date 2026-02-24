#!/bin/bash
#SBATCH --job-name=apt_vgpu_ultradebug
#SBATCH --account=ENT114035
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --output=vgpu_ultradebug_%j.out
#SBATCH --error=vgpu_ultradebug_%j.err

module load miniconda3/24.11.1

cd /work/twsuday816/APT-Transformer

echo "============================================"
echo "APT Virtual VRAM Ultra Debug 测试"
echo "节点: ${SLURM_JOB_NODELIST}"
echo "开始时间: $(date)"
echo "============================================"

# Ultra Debug: 只启用 Virtual VRAM，跑10步看详细日志
srun python -m apt.trainops.scripts.pretrain_quickcook \
    --output-dir ./test_vvram_ultradebug \
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
echo "Ultra Debug 测试完成时间: $(date)"
echo "============================================"
