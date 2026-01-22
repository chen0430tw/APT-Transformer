#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VFT-TVA Model Trainer
Trainer for Vein-Flow Transformer with Tri-Vein Attention
"""

from apt.apt_model.utils.fake_torch import get_torch
torch = get_torch()
from apt.apt_model.utils.fake_torch import get_torch
torch = get_torch()
nn = torch.nn
F = torch.nn.functional
Dataset = torch.utils.data.Dataset
DataLoader = torch.utils.data.DataLoader
from typing import Optional, Dict, List, Any
from tqdm import tqdm
import os

from apt.apt_model.modeling.vft_tva_model import VFTTVAModel, create_vft_tva_model
from apt.apt_model.training.training_guard import TrainingGuard, EarlyStopping


class VFTTVADataset(Dataset):
    """Dataset for VFT-TVA training"""

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Encode text
        if hasattr(self.tokenizer, 'encode'):
            tokens = self.tokenizer.encode(text)
        else:
            tokens = self.tokenizer(text)['input_ids']

        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        input_ids = torch.tensor(tokens, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': input_ids.clone()
        }


def collate_fn(batch):
    """Batch collation function"""
    max_len = max([item['input_ids'].size(0) for item in batch])

    input_ids = []
    labels = []

    for item in batch:
        seq_len = item['input_ids'].size(0)
        pad_len = max_len - seq_len

        # Padding
        padded_input = F.pad(item['input_ids'], (0, pad_len), value=0)
        padded_label = F.pad(item['labels'], (0, pad_len), value=-100)

        input_ids.append(padded_input)
        labels.append(padded_label)

    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels)
    }


class VFTTVATrainer:
    """
    Trainer for VFT-TVA model (supports multimodal + training guard)

    Features:
    - Standard language modeling training
    - Tracks vein subspace statistics (eps, rank usage)
    - Supports multimodal inputs (image_feat, audio_feat)
    - Training protection mechanisms
    """

    def __init__(
        self,
        model: VFTTVAModel,
        tokenizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        # Training guard parameters
        enable_guard: bool = True,
        max_steps: Optional[int] = None,
        max_time_hours: Optional[float] = None,
        early_stopping_patience: Optional[int] = None,
        guard_verbose: bool = True
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.max_grad_norm = max_grad_norm

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )

        # Learning rate scheduler
        self.warmup_steps = warmup_steps
        self.step_count = 0

        # Statistics
        self.stats = {
            'eps_mean': [],
            'eps_frac': [],
            'rank_usage': []
        }

        # Training guard
        self.enable_guard = enable_guard
        if enable_guard:
            early_stopping = None
            if early_stopping_patience is not None:
                early_stopping = EarlyStopping(
                    patience=early_stopping_patience,
                    mode='min',
                    verbose=guard_verbose
                )

            self.guard = TrainingGuard(
                max_steps=max_steps,
                max_time_hours=max_time_hours,
                early_stopping=early_stopping,
                verbose=guard_verbose
            )
        else:
            self.guard = None

    def get_lr(self):
        """Learning rate warmup"""
        if self.step_count < self.warmup_steps:
            return self.step_count / max(1, self.warmup_steps)
        return 1.0

    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-entropy loss"""
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        return loss

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step (supports multimodal)"""
        self.model.train()

        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Multimodal inputs (optional)
        image_feat = batch.get('image_feat')
        audio_feat = batch.get('audio_feat')

        if image_feat is not None:
            image_feat = image_feat.to(self.device)
        if audio_feat is not None:
            audio_feat = audio_feat.to(self.device)

        # Forward pass with info
        self.optimizer.zero_grad()
        logits, block_infos = self.model(
            input_ids=input_ids,
            image_feat=image_feat,
            audio_feat=audio_feat,
            return_info=True
        )

        # Compute loss
        loss = self.compute_loss(logits, labels)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # Update with warmup
        lr_scale = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_scale

        self.optimizer.step()
        self.step_count += 1

        # Collect statistics
        avg_eps = sum(info['eps_mean'] for info in block_infos) / len(block_infos)
        avg_frac = sum(info['eps_frac_over_tau'] for info in block_infos) / len(block_infos)

        self.stats['eps_mean'].append(avg_eps)
        self.stats['eps_frac'].append(avg_frac)

        return {
            'loss': loss.item(),
            'eps_mean': avg_eps,
            'eps_frac': avg_frac,
            'rank': block_infos[0]['rank']
        }

    def train(
        self,
        train_texts: List[str],
        epochs: int = 10,
        batch_size: int = 8,
        max_length: int = 512,
        save_path: Optional[str] = None,
        eval_texts: Optional[List[str]] = None,
        eval_interval: int = 1000
    ) -> Dict[str, List[float]]:
        """
        Complete training loop (with training guard)

        Args:
            train_texts: Training texts
            epochs: Number of epochs
            batch_size: Batch size
            max_length: Maximum sequence length
            save_path: Model save path
            eval_texts: Evaluation texts
            eval_interval: Evaluation interval in steps

        Returns:
            Training history
        """
        # Create dataset
        train_dataset = VFTTVADataset(train_texts, self.tokenizer, max_length)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        # Training history
        history = {
            'train_loss': [],
            'eval_loss': [],
            'eps_mean': [],
            'eps_frac': []
        }

        print(f"🚀 开始训练 VFT-TVA 模型")
        print(f"设备: {self.device} | Epochs: {epochs} | Batch Size: {batch_size}")
        print(f"训练样本数: {len(train_texts)} | 词汇表大小: {getattr(self.tokenizer, 'vocab_size', 'Unknown')}")
        print(f"Vein Rank: {self.model.blocks[0].proj.rank}")
        print("=" * 80)

        # Start training guard
        if self.guard:
            self.guard.start()

        should_stop = False
        global_step = 0
        for epoch in range(epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch in progress_bar:
                metrics = self.train_step(batch)
                epoch_loss += metrics['loss']
                global_step += 1

                progress_bar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'eps': f"{metrics['eps_mean']:.3f}",
                    'eps_frac': f"{metrics['eps_frac']:.2%}"
                })

                # Training guard check
                if self.guard:
                    if not self.guard.step(loss=metrics['loss'], model=self.model):
                        should_stop = True
                        break

                # Periodic evaluation
                if eval_texts and global_step % eval_interval == 0:
                    eval_loss = self.evaluate(eval_texts, batch_size, max_length)
                    history['eval_loss'].append(eval_loss)
                    print(f"\n[Step {global_step}] Eval Loss: {eval_loss:.4f}")

                    # Early stopping check
                    if self.guard:
                        if not self.guard.validate(eval_loss):
                            should_stop = True
                            break

            if should_stop:
                break

            avg_loss = epoch_loss / len(train_loader)
            avg_eps = sum(self.stats['eps_mean'][-len(train_loader):]) / len(train_loader)
            avg_frac = sum(self.stats['eps_frac'][-len(train_loader):]) / len(train_loader)

            history['train_loss'].append(avg_loss)
            history['eps_mean'].append(avg_eps)
            history['eps_frac'].append(avg_frac)

            print(f"\nEpoch {epoch+1}/{epochs} 总结:")
            print(f"  平均损失: {avg_loss:.4f}")
            print(f"  平均 eps: {avg_eps:.4f}")
            print(f"  Off-manifold fraction: {avg_frac:.2%}")

            # Save checkpoint
            if save_path and (epoch + 1) % 5 == 0:
                self.save_checkpoint(save_path, epoch)

        # Print training guard statistics
        if self.guard:
            stats = self.guard.get_stats()
            print(f"\n{'='*80}")
            print("训练保护统计:")
            print(f"  总步数: {stats['total_steps']}")
            print(f"  训练时间: {stats['elapsed_hours']:.2f} 小时")
            if stats['stopped']:
                print(f"  停止原因: {stats['stop_reason']}")
            print(f"{'='*80}\n")

        # Save final model
        if save_path:
            self.save_model(save_path)
            print(f"✅ 模型已保存到: {save_path}")

        return history

    @torch.no_grad()
    def evaluate(
        self,
        eval_texts: List[str],
        batch_size: int = 8,
        max_length: int = 512
    ) -> float:
        """Evaluate model"""
        self.model.eval()

        eval_dataset = VFTTVADataset(eval_texts, self.tokenizer, max_length)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn
        )

        total_loss = 0.0
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            logits = self.model(input_ids=input_ids)
            loss = self.compute_loss(logits, labels)
            total_loss += loss.item()

        return total_loss / len(eval_loader)

    def save_model(self, save_path: str):
        """Save model"""
        os.makedirs(save_path, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'stats': self.stats
        }, os.path.join(save_path, 'vft_tva_model.pt'))

        # Save tokenizer if supported
        if hasattr(self.tokenizer, 'save_pretrained'):
            self.tokenizer.save_pretrained(save_path)

    def save_checkpoint(self, save_path: str, epoch: int):
        """Save checkpoint"""
        checkpoint_path = os.path.join(save_path, f'checkpoint-epoch-{epoch+1}')
        self.save_model(checkpoint_path)
        print(f"检查点已保存: {checkpoint_path}")

    def load_model(self, load_path: str):
        """Load model"""
        checkpoint = torch.load(
            os.path.join(load_path, 'vft_tva_model.pt'),
            map_location=self.device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint.get('step_count', 0)
        self.stats = checkpoint.get('stats', self.stats)
        print(f"✅ 模型已从 {load_path} 加载")


# ==================== Convenience Functions ====================

def train_vft_tva(
    train_texts: List[str],
    model_size: str = 'base',
    enable_multimodal: bool = False,
    vocab_size: int = 50000,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    save_path: str = "./vft_tva_model",
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Convenience function to train VFT-TVA model

    Args:
        train_texts: Training texts
        model_size: Model size ('small', 'base', 'large')
        enable_multimodal: Enable multimodal support
        vocab_size: Vocabulary size
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        save_path: Model save path
        device: Device

    Returns:
        (model, tokenizer, history)
    """
    from transformers import GPT2Tokenizer

    # Initialize model
    model = create_vft_tva_model(
        model_size=model_size,
        enable_multimodal=enable_multimodal,
        vocab_size=vocab_size
    )

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Create trainer
    trainer = VFTTVATrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=learning_rate
    )

    # Train
    history = trainer.train(
        train_texts=train_texts,
        epochs=epochs,
        batch_size=batch_size,
        save_path=save_path
    )

    return model, tokenizer, history


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='训练 VFT-TVA 模型')
    parser.add_argument('--model-size', type=str, default='base',
                       choices=['small', 'base', 'large'],
                       help='模型大小')
    parser.add_argument('--multimodal', action='store_true',
                       help='启用多模态支持')
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--save-path', type=str, default='./vft_tva_model',
                       help='模型保存路径')
    parser.add_argument('--device', type=str, default=None,
                       help='设备 (cuda/cpu)')

    args = parser.parse_args()

    # Example training data
    texts = [
        "The Vein-Flow Transformer uses low-rank attention in a shared subspace.",
        "Tri-Vein Attention reduces computational complexity significantly.",
        "Normal compensation handles off-manifold tokens effectively.",
    ] * 100  # Repeat for demo

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_vft_tva(
        train_texts=texts,
        model_size=args.model_size,
        enable_multimodal=args.multimodal,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_path,
        device=args.device
    )
