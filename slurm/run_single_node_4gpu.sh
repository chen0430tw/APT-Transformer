#!/bin/bash
#SBATCH --job-name=apt_full_test
#SBATCH --account=ENT114035
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=full_test_%j.out
#SBATCH --error=full_test_%j.err

module load miniconda3/24.11.1

cd /work/twsuday816/APT-Transformer

echo "开始完整数据集测试 (单节点多GPU)"
echo "节点数: ${SLURM_JOB_NUM_NODES}"
echo "总GPU数: ${SLURM_JOB_NUM_GPUS}"
echo "每节点GPU数: ${SLURM_JOB_GPUS_PER_NODE}"
echo "每节点任务数: ${SLURM_CPUS_ON_NODE}"

# 单节点4GPU分布式训练
srun torchrun --nproc_per_node=4 \
    -m apt.trainops.scripts.pretrain_quickcook \
    --output-dir ./full_test_output \
    --max-steps 200 \
    --weight-c4 0.35 \
    --weight-mc4 0.20 \
    --weight-fineweb 0.20 \
    --use-wiki --weight-wiki 0.10 --wiki-mode sequential \
    --use-arxiv --weight-arxiv 0.08 --arxiv-mode sequential \
    --use-code --weight-code 0.05 --code-mode sequential \
    --save-interval 100

echo "训练完成"
