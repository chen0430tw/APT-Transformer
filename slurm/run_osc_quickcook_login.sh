#!/bin/bash
# run_osc_quickcook_login.sh
# 登录节点直跑版 — 不用 sbatch，适合 <15 分钟快速验证
#
# 用法：
#   bash /work/twsuday816/APT-Transformer/slurm/run_osc_quickcook_login.sh
#   bash /work/twsuday816/APT-Transformer/slurm/run_osc_quickcook_login.sh --max-steps 500
#   bash /work/twsuday816/APT-Transformer/slurm/run_osc_quickcook_login.sh --output-dir /work/twsuday816/my_run
#   COMPILE_THREADS=8 bash /work/twsuday816/APT-Transformer/slurm/run_osc_quickcook_login.sh
#
# 稳定配置（2026-03-31 验证）：
#   - TORCHINDUCTOR_COMPILE_THREADS=4  避免编译线程争用（可用 COMPILE_THREADS=N 覆盖）
#   - num_workers=1, prefetch_factor=2  异步预取 + zstd 异常兜底
#   - zstd/fsspec 读流异常只 warning 跳过，不炸训练

module load miniconda3/24.11.1

set -euo pipefail

export PYTHONPATH=/work/twsuday816/Oscillator:/work/twsuday816/APT-Transformer:$PYTHONPATH
export TORCHINDUCTOR_COMPILE_THREADS=${COMPILE_THREADS:-4}

cd /work/twsuday816/APT-Transformer

# 时间戳输出目录（可被 --output-dir 覆盖）
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DEFAULT_OUTPUT=/work/twsuday816/osc_quickcook_output_login_${TIMESTAMP}

python -m apt.trainops.scripts.pretrain_quickcook \
    --output-dir "${DEFAULT_OUTPUT}" \
    --model-arch oscillator \
    --d-model 256 \
    --num-heads 4 \
    --num-layers 4 \
    --max-seq-len 128 \
    --batch-size 32 \
    --max-steps 200 \
    --lr 2e-3 \
    --log-interval 20 \
    --no-distributed \
    --cache-dir /work/twsuday816/.cache \
    --hlbd-path /work/twsuday816/APT-Transformer/data/HLBD_Full_V2.json \
    "$@"
