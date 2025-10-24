"""Lightweight English tokenizer built from repository datasets.

This module provides a minimal tokenizer that is independent from
Hugging Face model downloads.  It constructs a small vocabulary from the
texts that ship with the repository and performs simple whitespace and
punctuation based tokenisation.  The goal is to provide a deterministic
tokeniser that works in offline environments where downloading the GPT-2
tokeniser is not possible.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Sequence

import torch


_DEFAULT_VOCAB = [
    "hello",
    "world",
    "machine",
    "learning",
    "artificial",
    "intelligence",
    "apt",
    "model",
    "training",
    "data",
    "transformer",
]


class BasicEnglishTokenizer:
    """A compact tokenizer that derives its vocabulary from local texts.

    The implementation intentionally keeps the interface surface similar to
    Hugging Face tokenizers that are used inside the training pipeline.
    Only the methods accessed by the project (``encode``/``decode`` and a
    few token attributes) are implemented, which keeps the dependency
    footprint very small.
    """

    pad_token: str = "<pad>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"

    def __init__(
        self,
        texts: Sequence[str] | None = None,
        vocab_size: int = 8000,
        lowercase: bool = True,
    ) -> None:
        self.lowercase = lowercase

        self._token_to_id = {
            self.pad_token: 0,
            self.eos_token: 1,
            self.unk_token: 2,
        }
        self._id_to_token = {idx: tok for tok, idx in self._token_to_id.items()}

        self._build_vocab(texts or (), vocab_size)

    # ------------------------------------------------------------------
    # Properties mirroring Hugging Face tokeniser attributes
    # ------------------------------------------------------------------
    @property
    def pad_token_id(self) -> int:
        return self._token_to_id[self.pad_token]

    @property
    def eos_token_id(self) -> int:
        return self._token_to_id[self.eos_token]

    @property
    def unk_token_id(self) -> int:
        return self._token_to_id[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    # ------------------------------------------------------------------
    # Core API used by the training loop
    # ------------------------------------------------------------------
    def encode(
        self,
        text: str,
        return_tensors: str | None = None,
        max_length: int | None = None,
        truncation: bool = False,
    ):
        tokens = self._tokenise(text)
        ids = [self._token_to_id.get(tok, self.unk_token_id) for tok in tokens]
        ids.append(self.eos_token_id)

        if max_length is not None:
            if truncation and len(ids) > max_length:
                ids = ids[:max_length]
            elif len(ids) < max_length:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))

        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        tokens: List[str] = []
        for idx in ids:
            token = self._id_to_token.get(idx, self.unk_token)
            if skip_special_tokens and token in {
                self.pad_token,
                self.eos_token,
            }:
                continue
            tokens.append(token)
        return " ".join(tokens)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_vocab(self, texts: Sequence[str], vocab_size: int) -> None:
        counter: Counter[str] = Counter()

        if texts:
            for text in texts:
                counter.update(self._tokenise(text))

        for token in _DEFAULT_VOCAB:
            counter.setdefault(token, 1)

        available_slots = max(0, vocab_size - len(self._token_to_id))
        for token, _ in counter.most_common(available_slots):
            if token not in self._token_to_id:
                self._token_to_id[token] = len(self._token_to_id)
                self._id_to_token[self._token_to_id[token]] = token

    _TOKEN_RE = re.compile(r"[\w]+|[^\s\w]", re.UNICODE)

    def _tokenise(self, text: str) -> List[str]:
        if self.lowercase:
            text = text.lower()
        tokens = self._TOKEN_RE.findall(text)
        return tokens or [self.unk_token]


__all__ = ["BasicEnglishTokenizer"]
