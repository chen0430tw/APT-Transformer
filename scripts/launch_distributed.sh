#!/bin/bash

# =============================================================================
# Distributed Training Launcher for APT Model
# =============================================================================
#
# 🔮 Launcher script for distributed training using PyTorch DDP
#
# This script provides convenient ways to launch distributed training:
# - Single machine, multiple GPUs
# - Multiple machines (multi-node)
# - Custom configurations
#
# Usage:
#   ./scripts/launch_distributed.sh [OPTIONS]
#
# Examples:
#   # Single machine, 4 GPUs
#   ./scripts/launch_distributed.sh --gpus 4
#
#   # Single machine, 8 GPUs with custom config
#   ./scripts/launch_distributed.sh --gpus 8 --batch-size 64 --num-epochs 20
#
#   # Multi-node training (run on each node)
#   # Node 0 (master):
#   ./scripts/launch_distributed.sh --gpus 4 --nodes 2 --node-rank 0 --master-addr 192.168.1.100
#
#   # Node 1:
#   ./scripts/launch_distributed.sh --gpus 4 --nodes 2 --node-rank 1 --master-addr 192.168.1.100
#
# =============================================================================

set -e  # Exit on error

# Default values
NUM_GPUS=1
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29500

# Training parameters (defaults)
BATCH_SIZE=32
NUM_EPOCHS=10
LEARNING_RATE=0.001
D_MODEL=512
NUM_LAYERS=6
NUM_HEADS=8
VOCAB_SIZE=10000
SEQ_LENGTH=128

# Paths
SAVE_DIR="./distributed_checkpoints"
RESUME=""

# Monitoring
ENABLE_GRADIENT_MONITOR=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --nodes)
            NUM_NODES="$2"
            shift 2
            ;;
        --node-rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --master-addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master-port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --d-model)
            D_MODEL="$2"
            shift 2
            ;;
        --num-layers)
            NUM_LAYERS="$2"
            shift 2
            ;;
        --num-heads)
            NUM_HEADS="$2"
            shift 2
            ;;
        --vocab-size)
            VOCAB_SIZE="$2"
            shift 2
            ;;
        --seq-length)
            SEQ_LENGTH="$2"
            shift 2
            ;;
        --save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --enable-gradient-monitor)
            ENABLE_GRADIENT_MONITOR=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Distributed Training Options:"
            echo "  --gpus NUM              Number of GPUs per node (default: 1)"
            echo "  --nodes NUM             Number of nodes (default: 1)"
            echo "  --node-rank RANK        Rank of this node (default: 0)"
            echo "  --master-addr ADDR      Master node address (default: 127.0.0.1)"
            echo "  --master-port PORT      Master node port (default: 29500)"
            echo ""
            echo "Model Configuration:"
            echo "  --d-model DIM           Model dimension (default: 512)"
            echo "  --num-layers NUM        Number of layers (default: 6)"
            echo "  --num-heads NUM         Number of attention heads (default: 8)"
            echo "  --vocab-size NUM        Vocabulary size (default: 10000)"
            echo ""
            echo "Training Configuration:"
            echo "  --batch-size SIZE       Batch size per GPU (default: 32)"
            echo "  --num-epochs NUM        Number of epochs (default: 10)"
            echo "  --lr RATE               Learning rate (default: 0.001)"
            echo "  --seq-length LEN        Sequence length (default: 128)"
            echo ""
            echo "Checkpoint Management:"
            echo "  --save-dir DIR          Checkpoint save directory (default: ./distributed_checkpoints)"
            echo "  --resume PATH           Resume from checkpoint"
            echo ""
            echo "Monitoring:"
            echo "  --enable-gradient-monitor   Enable gradient monitoring"
            echo ""
            echo "Examples:"
            echo "  # Single machine, 4 GPUs"
            echo "  $0 --gpus 4"
            echo ""
            echo "  # Multi-node (run on each node)"
            echo "  $0 --gpus 4 --nodes 2 --node-rank 0 --master-addr 192.168.1.100"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo "============================================================================="
echo "APT Model Distributed Training"
echo "============================================================================="
echo ""
echo "Distributed Configuration:"
echo "  GPUs per node:      $NUM_GPUS"
echo "  Number of nodes:    $NUM_NODES"
echo "  Node rank:          $NODE_RANK"
echo "  Master address:     $MASTER_ADDR"
echo "  Master port:        $MASTER_PORT"
echo "  Total processes:    $((NUM_GPUS * NUM_NODES))"
echo ""
echo "Model Configuration:"
echo "  d_model:            $D_MODEL"
echo "  num_layers:         $NUM_LAYERS"
echo "  num_heads:          $NUM_HEADS"
echo "  vocab_size:         $VOCAB_SIZE"
echo ""
echo "Training Configuration:"
echo "  Batch size/GPU:     $BATCH_SIZE"
echo "  Global batch size:  $((BATCH_SIZE * NUM_GPUS * NUM_NODES))"
echo "  Number of epochs:   $NUM_EPOCHS"
echo "  Learning rate:      $LEARNING_RATE"
echo "  Sequence length:    $SEQ_LENGTH"
echo ""
echo "Checkpoint:"
echo "  Save directory:     $SAVE_DIR"
if [ -n "$RESUME" ]; then
    echo "  Resume from:        $RESUME"
fi
echo ""
echo "Monitoring:"
echo "  Gradient monitor:   $ENABLE_GRADIENT_MONITOR"
echo ""
echo "============================================================================="
echo ""

# Check if PyTorch is available
if ! python -c "import torch" 2>/dev/null; then
    echo "ERROR: PyTorch not found. Please install PyTorch first."
    exit 1
fi

# Check if distributed training is available
if ! python -c "import torch.distributed" 2>/dev/null; then
    echo "ERROR: torch.distributed not available."
    exit 1
fi

# Check GPU availability if using NCCL backend
if [ "$NUM_GPUS" -gt 0 ]; then
    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
    if [ "$GPU_COUNT" -lt "$NUM_GPUS" ]; then
        echo "WARNING: Requested $NUM_GPUS GPUs but only $GPU_COUNT available"
        echo "Will use CPU backend instead"
    fi
fi

# Build training command
TRAINING_ARGS="--d-model $D_MODEL \
    --num-layers $NUM_LAYERS \
    --num-heads $NUM_HEADS \
    --vocab-size $VOCAB_SIZE \
    --batch-size $BATCH_SIZE \
    --num-epochs $NUM_EPOCHS \
    --lr $LEARNING_RATE \
    --seq-length $SEQ_LENGTH \
    --save-dir $SAVE_DIR"

if [ -n "$RESUME" ]; then
    TRAINING_ARGS="$TRAINING_ARGS --resume $RESUME"
fi

if [ "$ENABLE_GRADIENT_MONITOR" = true ]; then
    TRAINING_ARGS="$TRAINING_ARGS --enable-gradient-monitor"
fi

# Launch distributed training
echo "Launching distributed training..."
echo ""

if command -v torchrun &> /dev/null; then
    # Use torchrun (PyTorch 1.10+)
    echo "Using torchrun launcher"
    torchrun \
        --nnodes=$NUM_NODES \
        --nproc_per_node=$NUM_GPUS \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        examples/train_distributed.py \
        $TRAINING_ARGS
else
    # Fallback to torch.distributed.launch
    echo "Using torch.distributed.launch (fallback)"
    python -m torch.distributed.launch \
        --nnodes=$NUM_NODES \
        --nproc_per_node=$NUM_GPUS \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        examples/train_distributed.py \
        $TRAINING_ARGS
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================================="
    echo "Training completed successfully!"
    echo "Checkpoints saved to: $SAVE_DIR"
    echo "============================================================================="
else
    echo ""
    echo "============================================================================="
    echo "Training failed with exit code $?"
    echo "============================================================================="
    exit 1
fi
