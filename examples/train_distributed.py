"""
Distributed Training Script for APT Model

PyTorch DDP (DistributedDataParallel) training with multi-GPU support.

🔮 Implementation based on preparation code from:
- apt_model/training/gradient_monitor.py:sync_gradients_distributed() (lines 355-380)
- apt_model/training/gradient_monitor.py:aggregate_anomalies_distributed() (lines 382-395)
- tests/test_trainer_complete.py:TestDistributedReadiness (lines 499-593)

Features:
- Multi-GPU training with DDP
- Gradient synchronization across processes
- Distributed checkpoint saving/loading
- Anomaly aggregation across ranks

Usage:
    # Single machine, multiple GPUs
    python -m torch.distributed.launch --nproc_per_node=4 examples/train_distributed.py

    # Or using torchrun (PyTorch 1.10+)
    torchrun --nproc_per_node=4 examples/train_distributed.py

    # Multiple machines
    torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr=<MASTER_IP> --master_port=29500 examples/train_distributed.py
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("ERROR: PyTorch not available. Distributed training requires PyTorch.")
    sys.exit(1)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from apt_model.modeling.apt_model import APTLargeModel, APTConfig
from apt_model.training.trainer import APTTrainer
from apt_model.training.gradient_monitor import GradientMonitor
from apt_model.utils.tokenizer import SimpleTokenizer


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[Rank %(rank)s] %(asctime)s - %(levelname)s - %(message)s',
)


def setup_distributed():
    """
    Initialize distributed training environment

    🔮 Based on DDP initialization requirements from TestDistributedReadiness
    """
    # Initialize the process group
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # torchrun sets these automatically
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        # SLURM environment
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = rank % torch.cuda.device_count()
    else:
        print("ERROR: Distributed environment not detected")
        print("Please run with: torchrun --nproc_per_node=<NUM_GPUS> train_distributed.py")
        sys.exit(1)

    # Initialize process group
    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        init_method='env://',
    )

    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')

    return rank, world_size, local_rank, device


def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()


def get_logger(rank: int):
    """Get rank-aware logger"""
    logger = logging.getLogger(__name__)
    # Only log from rank 0 to reduce noise
    if rank != 0:
        logger.setLevel(logging.WARNING)
    return logger, {'rank': rank}


class DistributedTrainer:
    """
    Distributed trainer wrapper

    🔮 Integrates gradient synchronization and anomaly aggregation from preparation code
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        gradient_monitor: Optional[GradientMonitor] = None,
        save_dir: Optional[Path] = None,
        rank: int = 0,
        world_size: int = 1,
        device: torch.device = torch.device('cpu'),
    ):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gradient_monitor = gradient_monitor
        self.save_dir = save_dir
        self.rank = rank
        self.world_size = world_size
        self.device = device

        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []

        self.logger, self.log_extra = get_logger(rank)

        # Wrap model with DDP
        self.model = self.model.to(device)
        self.model = DDP(
            self.model,
            device_ids=[device.index] if device.type == 'cuda' else None,
            find_unused_parameters=False  # Set to True if model has unused parameters
        )

        self.logger.info(
            f"Initialized distributed trainer on rank {rank}/{world_size}",
            extra=self.log_extra
        )

    def train_step(self, batch):
        """Single training step"""
        self.model.train()

        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Forward pass
        outputs = self.model.module(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # 🔮 Monitor gradients (only on rank 0 to reduce overhead)
        if self.gradient_monitor is not None and self.rank == 0:
            self.gradient_monitor.log_step(
                model=self.model.module,
                step=self.global_step,
                check_anomalies=True
            )

        # Optimizer step
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        self.epoch = epoch

        # Set epoch for DistributedSampler
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)

        epoch_losses = []

        for batch_idx, batch in enumerate(self.train_loader):
            loss = self.train_step(batch)
            epoch_losses.append(loss)
            self.global_step += 1

            if batch_idx % 10 == 0:
                self.logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] Loss: {loss:.4f}",
                    extra=self.log_extra
                )

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        self.train_losses.append(avg_loss)

        # 🔮 Synchronize gradients across ranks
        if self.gradient_monitor is not None:
            self.gradient_monitor.sync_gradients_distributed()
            self.gradient_monitor.aggregate_anomalies_distributed()

        # All-reduce loss for reporting
        loss_tensor = torch.tensor([avg_loss], device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        avg_loss_global = loss_tensor.item()

        if self.rank == 0:
            self.logger.info(
                f"Epoch {epoch} completed. Global average loss: {avg_loss_global:.4f}",
                extra=self.log_extra
            )

        return avg_loss_global

    def validate(self):
        """Validation step"""
        if self.val_loader is None:
            return None

        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model.module(input_ids=input_ids, labels=labels)
                val_losses.append(outputs.loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0

        # All-reduce validation loss
        val_loss_tensor = torch.tensor([avg_val_loss], device=self.device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
        avg_val_loss_global = val_loss_tensor.item()

        if self.rank == 0:
            self.logger.info(
                f"Validation loss: {avg_val_loss_global:.4f}",
                extra=self.log_extra
            )

        return avg_val_loss_global

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save checkpoint (only on rank 0)

        🔮 Based on test_checkpoint_supports_distributed_loading
        All ranks can load the same checkpoint
        """
        if self.rank != 0:
            return

        if self.save_dir is None:
            return

        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Save model state from DDP (use .module to get underlying model)
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'config': self.model.module.config.__dict__,
            'world_size': self.world_size,
        }

        # Save regular checkpoint
        ckpt_path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, ckpt_path)
        self.logger.info(f"Saved checkpoint: {ckpt_path}", extra=self.log_extra)

        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint: {best_path}", extra=self.log_extra)

    def load_checkpoint(self, checkpoint_path: Path):
        """
        Load checkpoint

        🔮 All ranks load the same checkpoint (test_checkpoint_supports_distributed_loading)
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load on CPU first, then move to device
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Load model state (DDP wraps the model, so use .module)
        self.model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 🔮 Restore training state (all ranks must have consistent state)
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])

        self.logger.info(
            f"Loaded checkpoint from epoch {self.epoch}, step {self.global_step}",
            extra=self.log_extra
        )

    def train(self, num_epochs: int, validate_every: int = 1):
        """
        Main training loop

        🔮 Implements distributed training with gradient synchronization
        """
        self.logger.info(
            f"Starting distributed training for {num_epochs} epochs",
            extra=self.log_extra
        )

        for epoch in range(self.epoch + 1, self.epoch + num_epochs + 1):
            # Train
            avg_loss = self.train_epoch(epoch)

            # Validate
            if epoch % validate_every == 0:
                val_loss = self.validate()

                # Update best model (all ranks must agree)
                is_best = val_loss is not None and val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss

                # Save checkpoint (only rank 0)
                self.save_checkpoint(epoch, is_best=is_best)

            # Synchronize all ranks
            dist.barrier()

        if self.rank == 0:
            self.logger.info("Training completed!", extra=self.log_extra)


def create_dummy_dataset(num_samples: int = 1000, seq_length: int = 128, vocab_size: int = 1000):
    """Create dummy dataset for testing"""
    from torch.utils.data import TensorDataset

    input_ids = torch.randint(0, vocab_size, (num_samples, seq_length))
    labels = torch.randint(0, vocab_size, (num_samples, seq_length))

    return TensorDataset(input_ids, labels)


def collate_fn(batch):
    """Collate function for DataLoader"""
    input_ids = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])

    return {
        'input_ids': input_ids,
        'labels': labels
    }


def main():
    parser = argparse.ArgumentParser(description='Distributed training for APT Model')

    # Model config
    parser.add_argument('--d-model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num-layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--vocab-size', type=int, default=10000, help='Vocabulary size')

    # Training config
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seq-length', type=int, default=128, help='Sequence length')

    # Checkpoint
    parser.add_argument('--save-dir', type=str, default='./distributed_checkpoints', help='Save directory')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')

    # Monitoring
    parser.add_argument('--enable-gradient-monitor', action='store_true', help='Enable gradient monitoring')

    args = parser.parse_args()

    # Setup distributed
    rank, world_size, local_rank, device = setup_distributed()
    logger, log_extra = get_logger(rank)

    logger.info(f"Distributed training initialized", extra=log_extra)
    logger.info(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}", extra=log_extra)
    logger.info(f"Device: {device}", extra=log_extra)

    # Create model
    config = APTConfig(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        vocab_size=args.vocab_size,
    )

    model = APTLargeModel(config)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters", extra=log_extra)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create dummy dataset (replace with real dataset)
    train_dataset = create_dummy_dataset(
        num_samples=10000,
        seq_length=args.seq_length,
        vocab_size=args.vocab_size
    )
    val_dataset = create_dummy_dataset(
        num_samples=1000,
        seq_length=args.seq_length,
        vocab_size=args.vocab_size
    )

    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=0,  # Set to > 0 for real training
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )

    # 🔮 Create gradient monitor (only on rank 0 to reduce overhead)
    gradient_monitor = None
    if args.enable_gradient_monitor and rank == 0:
        gradient_monitor = GradientMonitor(model=model)
        logger.info("Gradient monitoring enabled", extra=log_extra)

    # Create distributed trainer
    trainer = DistributedTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        gradient_monitor=gradient_monitor,
        save_dir=Path(args.save_dir),
        rank=rank,
        world_size=world_size,
        device=device
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))

    # Train
    try:
        trainer.train(num_epochs=args.num_epochs, validate_every=1)
    except KeyboardInterrupt:
        logger.info("Training interrupted", extra=log_extra)
    finally:
        # Cleanup
        cleanup_distributed()


if __name__ == '__main__':
    main()
