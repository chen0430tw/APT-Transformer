#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Distributed Training

分布式训练支持：
- DDP (DistributedDataParallel)
- FSDP (Fully Sharded Data Parallel)
- DeepSpeed integration
- Multi-node training utilities
- Extreme scale training (100K+ GPUs)
"""

try:
    from apt.trainops.distributed.extreme_scale_training import (
        ExtremeScaleTrainer,
        ParallelismConfig,
        NetworkTopology,
        create_extreme_scale_trainer,
    )
except ImportError:
    ExtremeScaleTrainer = None
    ParallelismConfig = None
    NetworkTopology = None
    create_extreme_scale_trainer = None

__all__ = [
    'ExtremeScaleTrainer',
    'ParallelismConfig',
    'NetworkTopology',
    'create_extreme_scale_trainer',
]
