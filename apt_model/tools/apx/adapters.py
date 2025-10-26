#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APX Adapter Templates

Provides adapter code templates for different model frameworks.
"""

from enum import Enum
from typing import Dict


class AdapterType(Enum):
    """Adapter types"""

    HF = "hf"  # HuggingFace AutoModel
    STUB = "stub"  # Stub/demo adapter


# HuggingFace adapter code
HF_MODEL_ADAPTER = r'''# -*- coding: utf-8 -*-
"""
HFAdapter: 适配 HuggingFace AutoModelForCausalLM / AutoTokenizer
需要安装 transformers / torch
"""
from __future__ import annotations
import json
import os
from typing import Dict, Any

try:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
except Exception as e:
    raise RuntimeError("HFAdapter requires 'transformers' and 'torch' installed") from e


class HFAdapter:
    """
    HuggingFace model adapter.

    Provides unified interface for APT to interact with HuggingFace models.
    """

    def __init__(self, model, tokenizer):
        """
        Initialize adapter with model and tokenizer.

        Args:
            model: HuggingFace model instance
            tokenizer: HuggingFace tokenizer instance
        """
        self.model = model
        self.tok = tokenizer

    @classmethod
    def from_artifacts(cls, artifacts_dir: str):
        """
        Load model and tokenizer from artifacts directory.

        Args:
            artifacts_dir: Path to artifacts directory

        Returns:
            HFAdapter instance

        Raises:
            FileNotFoundError: If config.json not found
        """
        cfg_path = os.path.join(artifacts_dir, "config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError("config.json not found in artifacts/")

        cfg = AutoConfig.from_pretrained(artifacts_dir)
        tok = AutoTokenizer.from_pretrained(artifacts_dir, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            artifacts_dir, config=cfg, torch_dtype="auto"
        )
        model.eval()
        return cls(model, tok)

    def encode(self, texts, max_new_tokens=0):
        """
        Encode texts to input tensors.

        Args:
            texts: Text or list of texts
            max_new_tokens: Maximum new tokens (reserved for generation)

        Returns:
            Dictionary of input tensors
        """
        return self.tok(texts, return_tensors="pt", padding=True, truncation=True)

    @torch.no_grad()
    def generate(self, texts, max_new_tokens=64, **kwargs):
        """
        Generate text from input.

        Args:
            texts: Text or list of texts
            max_new_tokens: Maximum new tokens to generate
            **kwargs: Additional generation arguments

        Returns:
            List of generated texts
        """
        batch = self.encode(texts)
        out = self.model.generate(**batch, max_new_tokens=max_new_tokens, **kwargs)
        return self.tok.batch_decode(out, skip_special_tokens=True)

    def forward(self, batch):
        """
        Forward pass through model.

        Args:
            batch: Input batch dictionary

        Returns:
            Model outputs
        """
        return self.model(**batch)

    def loss_fn(self, logits, batch):
        """
        Compute loss (for training).

        Args:
            logits: Model logits or output
            batch: Input batch with labels

        Returns:
            Loss value
        """
        # For training, labels should be in batch
        if hasattr(logits, "loss") and logits.loss is not None:
            return logits.loss
        return 0.0

    def save_pretrained(self, out_dir: str):
        """
        Save model and tokenizer.

        Args:
            out_dir: Output directory
        """
        self.model.save_pretrained(out_dir)
        self.tok.save_pretrained(out_dir)
'''

HF_TOKENIZER_ADAPTER = r'''# -*- coding: utf-8 -*-
"""
HFTokenizerAdapter: 适配 HuggingFace Tokenizer
"""
from __future__ import annotations

try:
    from transformers import AutoTokenizer
except Exception as e:
    raise RuntimeError("HFTokenizerAdapter requires 'transformers' installed") from e


class HFTokenizerAdapter:
    """
    HuggingFace tokenizer adapter.

    Provides unified interface for APT to interact with HuggingFace tokenizers.
    """

    def __init__(self, tokenizer):
        """
        Initialize adapter with tokenizer.

        Args:
            tokenizer: HuggingFace tokenizer instance
        """
        self.tokenizer = tokenizer

    @classmethod
    def from_files(cls, artifacts_dir: str):
        """
        Load tokenizer from files.

        Args:
            artifacts_dir: Path to artifacts directory

        Returns:
            HFTokenizerAdapter instance
        """
        tokenizer = AutoTokenizer.from_pretrained(artifacts_dir, use_fast=True)
        return cls(tokenizer)

    def encode(self, texts, max_length=512):
        """
        Encode texts to token IDs.

        Args:
            texts: Text or list of texts
            max_length: Maximum length

        Returns:
            Dictionary with input_ids and attention_mask
        """
        return self.tokenizer(
            texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
        )

    def decode(self, token_ids, skip_special_tokens=True):
        """
        Decode token IDs to text.

        Args:
            token_ids: Token ID tensor or list
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text or list of texts
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(self, token_ids, skip_special_tokens=True):
        """
        Batch decode token IDs to texts.

        Args:
            token_ids: Batch of token ID tensors
            skip_special_tokens: Whether to skip special tokens

        Returns:
            List of decoded texts
        """
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
'''

# Stub adapter code
STUB_MODEL_ADAPTER = r'''# -*- coding: utf-8 -*-
"""
Stub Model Adapter (for testing)
"""


class DemoAdapter:
    """
    Demo/stub model adapter.

    Provides minimal interface for testing APX packaging.
    """

    def __init__(self):
        """Initialize stub adapter."""
        pass

    @classmethod
    def from_artifacts(cls, artifacts_dir: str):
        """
        Load from artifacts (stub).

        Args:
            artifacts_dir: Path to artifacts directory

        Returns:
            DemoAdapter instance
        """
        return cls()

    def encode(self, texts):
        """
        Encode texts (stub).

        Args:
            texts: Text or list of texts

        Returns:
            Dummy input dictionary
        """
        return {"input_ids": [[0]]}

    def generate(self, texts, max_new_tokens=64):
        """
        Generate text (stub).

        Args:
            texts: Text or list of texts
            max_new_tokens: Maximum new tokens

        Returns:
            Dummy output
        """
        return ["[generated text]"]

    def forward(self, batch):
        """
        Forward pass (stub).

        Args:
            batch: Input batch

        Returns:
            None
        """
        return None

    def save_pretrained(self, out_dir: str):
        """
        Save model (stub).

        Args:
            out_dir: Output directory
        """
        pass
'''

STUB_TOKENIZER_ADAPTER = r'''# -*- coding: utf-8 -*-
"""
Stub Tokenizer Adapter (for testing)
"""


class HFTokenizerAdapter:
    """
    Stub tokenizer adapter.

    Provides minimal interface for testing APX packaging.
    """

    @classmethod
    def from_files(cls, artifacts_dir: str):
        """
        Load tokenizer (stub).

        Args:
            artifacts_dir: Path to artifacts directory

        Returns:
            HFTokenizerAdapter instance
        """
        return cls()

    def encode(self, texts, max_length=512):
        """
        Encode texts (stub).

        Args:
            texts: Text or list of texts
            max_length: Maximum length

        Returns:
            Dummy encoding
        """
        return {"input_ids": [[0]]}

    def decode(self, token_ids):
        """
        Decode tokens (stub).

        Args:
            token_ids: Token IDs

        Returns:
            Empty string
        """
        return ""

    def batch_decode(self, token_ids):
        """
        Batch decode (stub).

        Args:
            token_ids: Batch of token IDs

        Returns:
            List of empty strings
        """
        return [""]
'''


def get_adapter_code(adapter_type: AdapterType) -> Dict[str, str]:
    """
    Get adapter code templates.

    Args:
        adapter_type: Type of adapter to generate

    Returns:
        Dictionary with "model" and "tokenizer" keys containing code strings

    Raises:
        ValueError: If adapter type not supported
    """
    if adapter_type == AdapterType.HF:
        return {
            "model": HF_MODEL_ADAPTER,
            "tokenizer": HF_TOKENIZER_ADAPTER,
        }
    elif adapter_type == AdapterType.STUB:
        return {
            "model": STUB_MODEL_ADAPTER,
            "tokenizer": STUB_TOKENIZER_ADAPTER,
        }
    else:
        raise ValueError(f"Unsupported adapter type: {adapter_type}")
