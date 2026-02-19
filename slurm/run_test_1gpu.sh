#!/bin/bash
#SBATCH --job-name=apt_test
#SBATCH --account=ENT114035
#SBATCH --partition=normal
#SBATCH --nodes=1                    # 弹性：只要1个节点
#SBATCH --gpus-per-node=1           # 最小需求：1个GPU开始
#SBATCH --cpus-per-task=6
#SBATCH --time=00:30:00            # 测试30分钟
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err

module load miniconda3/24.11.1

cd /work/twsuday816/APT-Transformer

echo "快速测试 (单节点1GPU)"
echo "节点: ${SLURM_JOB_NODELIST}"
echo "GPU数: ${SLURM_JOB_NUM_GPUS}"

# 单GPU测试 - 快速验证脚本是否正常
srun python -m apt.trainops.scripts.pretrain_quickcook \
    --output-dir ./test_output \
    --max-steps 10 \
    --weight-c4 1.0 \
    --save-interval 5

echo "测试完成"
