#!/bin/bash
#SBATCH --job-name=apt_vgpu_lecac
#SBATCH --account=ENT114035
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00
#SBATCH --output=vgpu_lecac_%j.out
#SBATCH --error=vgpu_lecac_%j.err

module load miniconda3/24.11.1

cd /work/twsuday816/APT-Transformer

echo "============================================"
echo "APT 虚拟GPU测试 - Virtual VRAM + LECaC INT2"
echo "节点: "
echo "开始时间: 2026年02月22日 18:34:58"
echo "============================================"

srun python -m apt.trainops.scripts.pretrain_quickcook     --output-dir ./test_vvram_lecac_int2     --max-steps 50     --save-interval 25     --weight-fineweb 0.7     --weight-hlbd 0.3     --no-c4     --no-mc4     --batch-size 4     --gradient-accumulation 2     --use-virtual-vram     --use-lecac     --lecac-bits 2     --vram-enable-prefetch     --vram-enable-nested-v16     --vram-verbose

echo "============================================"
echo "Virtual VRAM + LECaC INT2 测试完成时间: 2026年02月22日 18:34:58"
echo "============================================"
