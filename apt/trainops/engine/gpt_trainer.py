# -*- coding: utf-8 -*-
"""
GPT Models Training Interface
统一的GPT模型训练接口，支持GPT-4o, GPT-5, GPTo3
"""

from apt.core.fake_torch import get_torch
torch = get_torch()
from apt.core.fake_torch import get_torch
torch = get_torch()
nn = torch.nn
F = torch.nn.functional
Dataset = torch.utils.data.Dataset
DataLoader = torch.utils.data.DataLoader
from typing import Optional, Dict, Any, List
from tqdm import tqdm
import os


class GPTDataset(Dataset):
    """GPT训练数据集"""
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # 编码文本
        if hasattr(self.tokenizer, 'encode'):
            tokens = self.tokenizer.encode(text)
        else:
            tokens = self.tokenizer(text)['input_ids']

        # 截断或填充
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        # 转换为tensor
        input_ids = torch.tensor(tokens, dtype=torch.long)

        return {
            'input_ids': input_ids,
            'labels': input_ids.clone()  # 自回归任务，标签就是输入右移
        }


def collate_fn(batch):
    """批处理整理函数"""
    max_len = max([item['input_ids'].size(0) for item in batch])

    input_ids = []
    labels = []

    for item in batch:
        seq_len = item['input_ids'].size(0)
        pad_len = max_len - seq_len

        # 填充
        padded_input = F.pad(item['input_ids'], (0, pad_len), value=0)
        padded_label = F.pad(item['labels'], (0, pad_len), value=-100)

        input_ids.append(padded_input)
        labels.append(padded_label)

    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels)
    }


class BaseGPTTrainer:
    """GPT模型基础训练器"""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.max_grad_norm = max_grad_norm

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )

        # 学习率调度器
        self.warmup_steps = warmup_steps
        self.step_count = 0

    def get_lr(self):
        """学习率预热"""
        if self.step_count < self.warmup_steps:
            return self.step_count / max(1, self.warmup_steps)
        return 1.0

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算交叉熵损失"""
        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        return loss

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """单步训练"""
        self.model.train()

        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        # 前向传播
        self.optimizer.zero_grad()
        logits = self.model(text_ids=input_ids)

        # 计算损失
        loss = self.compute_loss(logits, labels)

        # 反向传播
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # 更新参数
        lr_scale = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_scale

        self.optimizer.step()
        self.step_count += 1

        return loss.item()

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
        """完整训练流程"""

        # 创建数据集
        train_dataset = GPTDataset(train_texts, self.tokenizer, max_length)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        # 训练历史
        history = {
            'train_loss': [],
            'eval_loss': []
        }

        print(f"开始训练 | 设备: {self.device} | Epochs: {epochs} | Batch Size: {batch_size}")
        print(f"训练样本数: {len(train_texts)} | 词汇表大小: {getattr(self.tokenizer, 'vocab_size', 'Unknown')}")
        print("=" * 80)

        global_step = 0
        for epoch in range(epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch in progress_bar:
                loss = self.train_step(batch)
                epoch_loss += loss
                global_step += 1

                progress_bar.set_postfix({'loss': f'{loss:.4f}'})

                # 定期评估
                if eval_texts and global_step % eval_interval == 0:
                    eval_loss = self.evaluate(eval_texts, batch_size, max_length)
                    history['eval_loss'].append(eval_loss)
                    print(f"\n[Step {global_step}] Eval Loss: {eval_loss:.4f}")

            avg_loss = epoch_loss / len(train_loader)
            history['train_loss'].append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs} | 平均损失: {avg_loss:.4f}")

            # 保存检查点
            if save_path and (epoch + 1) % 5 == 0:
                self.save_checkpoint(save_path, epoch)

        # 保存最终模型
        if save_path:
            self.save_model(save_path)
            print(f"\n✅ 模型已保存到: {save_path}")

        return history

    @torch.no_grad()
    def evaluate(
        self,
        eval_texts: List[str],
        batch_size: int = 8,
        max_length: int = 512
    ) -> float:
        """评估模型"""
        self.model.eval()

        eval_dataset = GPTDataset(eval_texts, self.tokenizer, max_length)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn
        )

        total_loss = 0.0
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)

            logits = self.model(text_ids=input_ids)
            loss = self.compute_loss(logits, labels)
            total_loss += loss.item()

        return total_loss / len(eval_loader)

    def save_model(self, save_path: str):
        """保存模型"""
        os.makedirs(save_path, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
        }, os.path.join(save_path, 'model.pt'))

        # 保存tokenizer（如果支持）
        if hasattr(self.tokenizer, 'save_pretrained'):
            self.tokenizer.save_pretrained(save_path)

    def save_checkpoint(self, save_path: str, epoch: int):
        """保存检查点"""
        checkpoint_path = os.path.join(save_path, f'checkpoint-epoch-{epoch+1}')
        self.save_model(checkpoint_path)
        print(f"检查点已保存: {checkpoint_path}")

    def load_model(self, load_path: str):
        """加载模型"""
        checkpoint = torch.load(os.path.join(load_path, 'model.pt'), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint.get('step_count', 0)
        print(f"✅ 模型已从 {load_path} 加载")


class GPT4oTrainer(BaseGPTTrainer):
    """GPT-4o专用训练器"""

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """GPT-4o训练步骤（支持多模态）"""
        self.model.train()

        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        # GPT-4o支持多模态，但这里只用文本
        self.optimizer.zero_grad()
        logits = self.model(text_ids=input_ids, load_factor=1.0)

        loss = self.compute_loss(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        lr_scale = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_scale

        self.optimizer.step()
        self.step_count += 1

        return loss.item()


class GPTo3Trainer(BaseGPTTrainer):
    """GPTo3专用训练器（结构化推理）"""

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """GPTo3训练步骤（启用梯度计算）"""
        self.model.train()

        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        self.optimizer.zero_grad()

        # GPTo3的forward现在支持梯度计算
        logits = self.model(text_ids=input_ids, load_factor=1.0)

        loss = self.compute_loss(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        lr_scale = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_scale

        self.optimizer.step()
        self.step_count += 1

        return loss.item()


# ============== 便捷函数 ==============

def train_gpt4o(
    train_texts: List[str],
    vocab_size: int = 32000,
    d_model: int = 512,
    n_heads: int = 8,
    d_ff: int = 2048,
    num_layers: int = 6,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    save_path: str = "./gpt4o_model",
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """便捷函数：训练GPT-4o模型"""
    from apt.model.architectures.gpt4o_model import GPT4oModel
    from transformers import GPT2Tokenizer

    # 初始化模型和tokenizer
    model = GPT4oModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        num_layers=num_layers
    )

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # 创建训练器
    trainer = GPT4oTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=learning_rate
    )

    # 训练
    history = trainer.train(
        train_texts=train_texts,
        epochs=epochs,
        batch_size=batch_size,
        save_path=save_path
    )

    return model, tokenizer, history


def train_gpto3(
    train_texts: List[str],
    vocab_size: int = 32000,
    d_model: int = 512,
    n_heads: int = 8,
    d_ff: int = 2048,
    num_layers: int = 6,
    epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 3e-4,
    save_path: str = "./gpto3_model",
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """便捷函数：训练GPTo3模型"""
    from apt.model.architectures.gpto3_model import GPTo3Model
    from transformers import GPT2Tokenizer

    # 初始化模型和tokenizer
    model = GPTo3Model(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        num_layers=num_layers
    )

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # 创建训练器
    trainer = GPTo3Trainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=learning_rate
    )

    # 训练
    history = trainer.train(
        train_texts=train_texts,
        epochs=epochs,
        batch_size=batch_size,
        save_path=save_path
    )

    return model, tokenizer, history


class GPT5Trainer(BaseGPTTrainer):
    """
    Trainer for GPT-5 model with Codebook MoE, streaming retrieval, and bi-state alignment.

    GPT-5 特性:
    - Codebook MoE with top-k routing
    - Streaming retrieval with memory buckets
    - Bi-state precision alignment
    - Leaf-Vote generation (K=2)
    """

    def __init__(
        self,
        model,
        optimizer,
        device: str = 'cuda',
        max_grad_norm: float = 1.0,
        enable_streaming_retrieval: bool = False,
        memory_bucket_size: int = 512,
        **kwargs
    ):
        super().__init__(model, optimizer, device, max_grad_norm, **kwargs)
        self.enable_streaming_retrieval = enable_streaming_retrieval
        self.memory_bucket_size = memory_bucket_size

        # Retrieval statistics
        self.retrieval_stats = {
            'total_retrievals': 0,
            'successful_retrievals': 0,
            'avg_confidence': 0.0
        }

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute loss for GPT-5"""
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )

        return loss

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Training step for GPT-5"""
        self.model.train()

        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        self.optimizer.zero_grad()

        # GPT-5 forward pass with step_idx for streaming retrieval
        logits, info = self.model.forward_step(
            input_ids,
            step_idx=self.step_count,
            schema_required=False
        )

        loss = self.compute_loss(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.step_count += 1

        # Track retrieval statistics
        if 'align' in info and info['align']:
            self.retrieval_stats['total_retrievals'] += 1
            if 'confidence' in info['align']:
                conf = info['align']['confidence']
                self.retrieval_stats['successful_retrievals'] += 1
                n = self.retrieval_stats['successful_retrievals']
                old_avg = self.retrieval_stats['avg_confidence']
                self.retrieval_stats['avg_confidence'] = old_avg + (conf - old_avg) / n

        return loss.item()

    def evaluate(self, dataloader) -> Dict[str, float]:
        """Evaluate GPT-5 model"""
        self.model.eval()

        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits, info = self.model.forward_step(
                    input_ids,
                    step_idx=0,
                    schema_required=False
                )

                loss = self.compute_loss(logits, labels)
                total_loss += loss.item()
                total_steps += 1

        avg_loss = total_loss / max(1, total_steps)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {
            'eval_loss': avg_loss,
            'perplexity': perplexity,
            'retrieval_rate': self.retrieval_stats['total_retrievals'] / max(1, self.step_count),
            'avg_confidence': self.retrieval_stats['avg_confidence']
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Generate text using GPT-5"""
        self.model.eval()
        generated = input_ids.clone()

        with torch.no_grad():
            for step in range(max_new_tokens):
                logits, info = self.model.forward_step(
                    generated,
                    step_idx=step,
                    schema_required=False
                )

                next_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_token], dim=1)

                if (next_token == 2).all():
                    break

        return generated


def train_gpt5(
    model,
    train_dataloader,
    val_dataloader=None,
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
    device: str = 'cuda',
    save_path: str = './checkpoints/gpt5',
    **kwargs
) -> Dict[str, float]:
    """Convenience function to train GPT-5 model"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    trainer = GPT5Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        **kwargs
    )

    final_metrics = trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=num_epochs,
        save_path=save_path
    )

    return final_metrics
