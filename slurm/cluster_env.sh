#!/bin/bash
# 集群环境变量配置脚本
# 模拟训练集群环境变量

export CLUSTER_NAME="nano5"
export PARTITION="normal"
export ACCOUNT="ENT114035"

# Slurm环境变量
export SLURM_JOB_ID="test_$(date +%Y%m%d_%H%M%S)"
export SLURM_JOB_NUM_NODES="1"
export SLURM_JOB_NODELIST="WSL"
export SLURM_PROCID="$$"
export SLURM_NTASKS="1"
export SLURM_NTASKS_PER_NODE="1"
export SLURM_CPUS_PER_TASK="4"
export SLURM_JOB_CPUS_PER_NODE="4"

# 数据集路径（集群路径）
export DATASET_ROOT="/work/twsuday816/datasets"
export C4_PATH="$DATASET_ROOT/c4"
export FIN_EWEB_PATH="$DATASET_ROOT/fineweb"

# 输出路径
export OUTPUT_ROOT="/work/twsuday816/experiments"
export CHECKPOINT_DIR="$OUTPUT_ROOT/checkpoints"

# GPU配置
export CUDA_VISIBLE_DEVICES="0"
export NVIDIA_VISIBLE_DEVICES="0"

# Virtual VRAM配置
export VRAM_ENABLE_NESTED_V16="1"
export VRAM_VERBOSE="1"

echo "=== 集群环境变量已设置 ==="
echo "集群: $CLUSTER_NAME"
echo "分区: $PARTITION"
echo "账户: $ACCOUNT"
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_JOB_NODELIST"
echo "任务数: $SLURM_NTASKS x $SLURM_CPUS_PER_TASK CPUS"
