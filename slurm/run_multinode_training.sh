#!/bin/bash
#SBATCH --job-name=apt_full_multi
#SBATCH --account=ENT114035
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --time=02:00:00
#SBATCH --output=full_test_multi_%j.out
#SBATCH --error=full_test_multi_%j.err

module load miniconda3/24.11.1

cd /work/twsuday816/APT-Transformer

echo "开始完整数据集测试 (多节点分布式)"
echo "节点数: ${SLURM_JOB_NUM_NODES}"
echo "总GPU数: ${SLURM_JOB_NUM_GPUS}"

# 使用 srun 启动分布式训练，每节点1个GPU
srun torchrun --nnodes=${SLURM_JOB_NUM_NODES} --nproc_per_node=1 \
    -m apt.trainops.scripts.pretrain_quickcook \
    --output-dir ./full_test_output \
    --max-steps 200 \
    --weight-c4 0.35 \
    --weight-mc4 0.20 \
    --weight-fineweb 0.20 \
    --use-wiki --weight-wiki 0.10 --wiki-mode sequential \
    --use-arxiv --weight-arxiv 0.08 --arxiv-mode sequential \
    --use-code --weight-code 0.05 --code-mode sequential \
    --save-freq 100

echo "训练完成"
