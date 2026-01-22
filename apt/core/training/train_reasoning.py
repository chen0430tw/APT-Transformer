#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reasoning Model Training

Provides training utilities for reasoning-enhanced models.
"""

import os
from typing import Optional, Dict, Any, List, Tuple
from apt_model.utils.fake_torch import get_torch
torch = get_torch()
from apt_model.utils.fake_torch import get_torch
torch = get_torch()
nn = torch.nn
DataLoader = torch.utils.data.DataLoader
Dataset = torch.utils.data.Dataset

from apt.core.system import get_device, set_seed
from apt.core.resources import ResourceMonitor
from apt_model.infrastructure.logging import get_progress_logger
from apt_model.runtime.decoder import ReasoningController, BudgetedReasoningController

logger = get_progress_logger()


class ReasoningDataset(Dataset):
    """
    Dataset for reasoning training.

    Expects examples with:
    - input: Problem/question
    - reasoning_steps: List of intermediate reasoning steps (optional)
    - output: Final answer
    """

    def __init__(
        self,
        examples: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 512,
        include_reasoning_steps: bool = True,
    ):
        """
        Args:
            examples: List of reasoning examples
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
            include_reasoning_steps: Whether to include intermediate steps
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_reasoning_steps = include_reasoning_steps

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Build input text
        if self.include_reasoning_steps and 'reasoning_steps' in example:
            # Format: Input -> Step 1 -> Step 2 -> ... -> Output
            text = f"Question: {example['input']}\n"
            for i, step in enumerate(example['reasoning_steps'], 1):
                text += f"Step {i}: {step}\n"
            text += f"Answer: {example['output']}"
        else:
            # Format: Input -> Output
            text = f"Question: {example['input']}\nAnswer: {example['output']}"

        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': encoded['input_ids'].squeeze(0),
        }


def load_reasoning_dataset(
    data_path: Optional[str] = None,
    max_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load reasoning dataset.

    Args:
        data_path: Path to dataset file (JSON/JSONL)
        max_samples: Maximum samples to load

    Returns:
        List of reasoning examples
    """
    if data_path is None:
        # Use default reasoning evaluation set
        return _get_default_reasoning_examples()

    # Load from file
    import json

    examples = []

    if data_path.endswith('.json'):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                examples = data
            elif isinstance(data, dict) and 'examples' in data:
                examples = data['examples']

    elif data_path.endswith('.jsonl'):
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                examples.append(json.loads(line))

    if max_samples:
        examples = examples[:max_samples]

    logger.info(f"Loaded {len(examples)} reasoning examples from {data_path}")
    return examples


def _get_default_reasoning_examples() -> List[Dict[str, Any]]:
    """Get default reasoning examples for training."""
    return [
        {
            'input': 'If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?',
            'reasoning_steps': [
                'All roses are flowers (premise 1)',
                'Some flowers fade quickly (premise 2)',
                'From premise 1: roses ⊆ flowers',
                'From premise 2: ∃ flowers that fade quickly',
                'But we cannot determine if the quickly-fading flowers include roses',
            ],
            'output': 'No, we cannot conclude that some roses fade quickly. The syllogism is invalid.',
        },
        {
            'input': 'A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?',
            'reasoning_steps': [
                'Let ball cost = x',
                'Then bat cost = x + $1',
                'Total: x + (x + $1) = $1.10',
                '2x + $1 = $1.10',
                '2x = $0.10',
                'x = $0.05',
            ],
            'output': 'The ball costs $0.05 (5 cents).',
        },
        {
            'input': 'What is 15% of 80?',
            'reasoning_steps': [
                '15% = 15/100 = 0.15',
                '15% of 80 = 0.15 × 80',
                '0.15 × 80 = 12',
            ],
            'output': '12',
        },
        {
            'input': 'If a train travels 120 km in 2 hours, what is its average speed?',
            'reasoning_steps': [
                'Speed = Distance / Time',
                'Distance = 120 km',
                'Time = 2 hours',
                'Speed = 120 km / 2 hours = 60 km/h',
            ],
            'output': '60 km/h',
        },
        {
            'input': 'Is 17 a prime number?',
            'reasoning_steps': [
                'A prime number is only divisible by 1 and itself',
                'Check divisibility by 2: 17 is odd, not divisible',
                'Check divisibility by 3: 1+7=8, not divisible by 3',
                'Check divisibility by 5: doesn\'t end in 0 or 5',
                'Only need to check up to √17 ≈ 4.1',
                'No divisors found except 1 and 17',
            ],
            'output': 'Yes, 17 is a prime number.',
        },
    ]


def train_reasoning_model(
    base_model: nn.Module,
    vein_projector: nn.Module,
    tokenizer,
    data_path: Optional[str] = None,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    max_reasoning_steps: int = 6,
    use_budgeted: bool = True,
    global_budget: float = 0.15,
    save_path: Optional[str] = None,
    logger = None,
    resource_monitor: Optional[ResourceMonitor] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train a reasoning-enhanced model.

    Args:
        base_model: Base language model
        vein_projector: VeinSubspaceShared module
        tokenizer: Tokenizer
        data_path: Path to reasoning dataset
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        max_reasoning_steps: Maximum reasoning steps
        use_budgeted: Whether to use budgeted reasoning
        global_budget: Budget for reasoning tokens
        save_path: Path to save model
        logger: Logger instance
        resource_monitor: Resource monitor

    Returns:
        Tuple of (trained_model, training_info)
    """
    if logger is None:
        logger = get_progress_logger()

    device = get_device()
    logger.info(f"[Reasoning Training] Using device: {device}")

    # Load dataset
    examples = load_reasoning_dataset(data_path)
    dataset = ReasoningDataset(examples, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create reasoning controller
    if use_budgeted:
        reasoning_controller = BudgetedReasoningController(
            vein_projector=vein_projector,
            global_budget=global_budget,
            max_steps=max_reasoning_steps,
        )
    else:
        reasoning_controller = ReasoningController(
            vein_projector=vein_projector,
            max_steps=max_reasoning_steps,
        )

    reasoning_controller = reasoning_controller.to(device)

    # Optimizer (only train reasoning components)
    optimizer = torch.optim.AdamW(
        reasoning_controller.parameters(),
        lr=learning_rate
    )

    # Training loop
    base_model.train()
    reasoning_controller.train()

    training_info = {
        'total_steps': 0,
        'avg_reasoning_steps': [],
        'losses': [],
    }

    logger.info(f"[Reasoning Training] Starting training for {epochs} epochs")

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_steps = []

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward through base model to get hidden states
            with torch.no_grad():
                outputs = base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[-1]  # Last layer

            # Apply reasoning
            h_reasoned, info = reasoning_controller(
                hidden_states,
                base_model.lm_head if hasattr(base_model, 'lm_head') else base_model.get_output_embeddings(),
                return_details=True
            )

            # Compute logits from reasoned hidden states
            if hasattr(base_model, 'lm_head'):
                logits = base_model.lm_head(h_reasoned)
            else:
                logits = base_model.get_output_embeddings()(h_reasoned)

            # Compute loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

            # Add auxiliary loss if available
            if 'step_details' in info:
                for step_info in info['step_details']:
                    if 'aux_loss' in step_info and step_info['aux_loss'] is not None:
                        loss = loss + step_info['aux_loss']

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reasoning_controller.parameters(), 1.0)
            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            epoch_steps.append(info['steps'])
            training_info['total_steps'] += 1

            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                    f"Loss: {loss.item():.4f}, Avg Steps: {info['steps']}"
                )

        avg_loss = epoch_loss / len(dataloader)
        avg_steps = sum(epoch_steps) / len(epoch_steps)

        training_info['losses'].append(avg_loss)
        training_info['avg_reasoning_steps'].append(avg_steps)

        logger.info(
            f"Epoch {epoch+1}/{epochs} completed - "
            f"Avg Loss: {avg_loss:.4f}, Avg Reasoning Steps: {avg_steps:.2f}"
        )

    # Save model
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        model_path = os.path.join(save_path, 'reasoning_controller.pt')

        torch.save({
            'reasoning_controller': reasoning_controller.state_dict(),
            'vein_projector': vein_projector.state_dict(),
            'training_info': training_info,
        }, model_path)

        logger.info(f"[Reasoning Training] Saved model to {model_path}")

    return reasoning_controller, training_info
