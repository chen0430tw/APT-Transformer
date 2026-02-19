#!/bin/bash
#SBATCH --job-name=apt_vgpu_test
#SBATCH --account=ENT114035
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --output=vgpu_test_%j.out
#SBATCH --error=vgpu_test_%j.err

module load miniconda3/24.11.1

cd /work/twsuday816/APT-Transformer

echo "============================================"
echo "APT 虚拟GPU测试 - Virtual VRAM"
echo "节点: ${SLURM_JOB_NODELIST}"
echo "开始时间: $(date)"
echo "============================================"

# 测试1: Virtual VRAM (激活值offload)
srun python -m apt.trainops.scripts.pretrain_quickcook \
    --output-dir ./test_vvram_output \
    --max-steps 50 \
    --save-interval 25 \
    --weight-fineweb 0.7 \
    --weight-hlbd 0.3 \
    --no-c4 \
    --no-mc4 \
    --batch-size 4 \
    --gradient-accumulation 2 \
    --use-virtual-vram \
    --vram-min-tensor-bytes 1048576

echo "============================================"
echo "Virtual VRAM测试完成时间: $(date)"
echo "============================================"
