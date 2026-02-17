#!/bin/bash
#SBATCH --job-name=apt_multi_test
#SBATCH --account=ENT114035
#SBATCH --partition=normal
#SBATCH --nodes=1-2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=2
#SBATCH --time=00:10:00
#SBATCH --output=test_multi_%j.out
#SBATCH --error=test_multi_%j.err

module load miniconda3/24.11.1

cd /work/twsuday816/APT-Transformer

echo "========================================"
echo "多节点快速测试"
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
    --output-dir ./test_quick_output \
    --max-steps 10 \
    --weight-c4 1.0 \
    --save-interval 5

echo "========================================"
echo "测试完成"
echo "========================================"
