#!/bin/bash
#SBATCH --job-name=osc_quickcook
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --account=ENT114035
#SBATCH --output=/work/twsuday816/osc_quickcook_%j.out
#SBATCH --error=/work/twsuday816/osc_quickcook_%j.err

module load miniconda3/24.11.1

export PYTHONPATH=/work/twsuday816/Oscillator:/work/twsuday816/APT-Transformer:$PYTHONPATH
cd /work/twsuday816/APT-Transformer

python -m apt.trainops.scripts.pretrain_quickcook \
    --output-dir /work/twsuday816/osc_quickcook_output \
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
    --hlbd-path /work/twsuday816/APT-Transformer/data/HLBD_Full_V2.json
