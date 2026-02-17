#!/bin/bash
#SBATCH --job-name=apt_multi_train
#SBATCH --account=ENT114035
#SBATCH --partition=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --time=02:00:00
#SBATCH --output=multi_train_%j.out
#SBATCH --error=multi_train_%j.err

module load miniconda3/24.11.1

cd /work/twsuday816/APT-Transformer

echo "========================================"
echo "多节点分布式训练 (固定2节点)"
echo "========================================"

# 获取节点列表和head node
nodes=($(scontrol show hostname ${SLURM_JOB_NODELIST}))
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w ${head_node} hostname --ip-address | awk '{print $1}')

# 设置分布式环境变量
export MASTER_ADDR=${head_node_ip}
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

echo "节点数: ${SLURM_NNODES}"
echo "每节点任务数: ${SLURM_NTASKS_PER_NODE}"
echo "总进程数: ${WORLD_SIZE}"
echo "Master节点: ${head_node} (${MASTER_ADDR})"
echo "Master端口: ${MASTER_PORT}"
echo "节点列表: ${SLURM_JOB_NODELIST}"
echo "========================================"

# 启动分布式训练
srun torchrun \
    --nnodes=${SLURM_NNODES} \
    --nproc_per_node=${SLURM_NTASKS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    -m apt.trainops.scripts.pretrain_quickcook \
    --output-dir ./full_train_output \
    --max-steps 200 \
    --weight-c4 0.35 \
    --weight-mc4 0.20 \
    --weight-fineweb 0.20 \
    --use-wiki --weight-wiki 0.10 --wiki-mode sequential \
    --use-arxiv --weight-arxiv 0.08 --arxiv-mode sequential \
    --use-code --weight-code 0.05 --code-mode sequential \
    --save-interval 100

echo "========================================"
echo "训练完成"
echo "========================================"
