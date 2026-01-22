# Training Scripts

This directory contains all training scripts for APT-Transformer.

## Available Training Backends

- `train.py` - Unified training launcher (supports all backends)
- `train_apt_playground.py` - APT Playground training
- `train_hlbd_playground.py` - HLBD Playground training
- `train_deepspeed.py` - DeepSpeed distributed training
- `train_azure_ml.py` - Azure ML cloud training
- `train_hf_trainer.py` - HuggingFace Trainer integration
- `train_control_experiment.py` - Control experiment training
- `resume_guide.py` - Training resume guide

## Quick Start

```bash
# List all available backends
python training/train.py --list-backends

# Train with specific backend
python training/train.py --backend playground --epochs 100
```

## Documentation

See [TRAINING_BACKENDS.md](../docs/TRAINING_BACKENDS.md) for detailed usage guide.
