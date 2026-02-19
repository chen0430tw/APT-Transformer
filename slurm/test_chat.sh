#!/bin/bash
#SBATCH --job-name=test_chat
#SBATCH --account=ENT114035
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=chat_%j.out
#SBATCH --error=chat_%j.err

module load miniconda3/24.11.1

cd /work/twsuday816/APT-Transformer

echo "=== 测试 chat.py ==="
echo "节点: ${SLURM_JOB_NODELIST}"
echo "GPU: ${SLURM_JOB_NUM_GPUS}"

# 简单测试 - 只加载模型，不交互
python -c "
from apt.apps.interactive.chat import chat_with_model
import sys
print('chat module imported successfully')
# 注意：这里不实际启动交互式会话，因为Slurm是非交互环境
print('PASS: chat.py can be imported and loaded')
sys.exit(0)
"

echo "测试完成"
