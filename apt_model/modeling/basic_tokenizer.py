"""Lightweight English tokenizer built from repository datasets.

This module provides a minimal tokenizer that is independent from
Hugging Face model downloads.  It constructs a small vocabulary from the
texts that ship with the repository and performs simple whitespace and
punctuation based tokenisation.  The goal is to provide a deterministic
tokeniser that works in offline environments where downloading the GPT-2
tokeniser is not possible.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

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

_ALPHANUMERIC_FALLBACK = set(string.ascii_lowercase + string.digits)
_PUNCTUATION_FALLBACK = set(
    [
        ".",
        ",",
        "!",
        "?",
        ";",
        ":",
        "-",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "'",
        '"',
        "/",
    ]
)

_NO_SPACE_BEFORE = {".", ",", "!", "?", ";", ":", "-", "'", '"', ")", "]", "}"}
_NO_SPACE_AFTER = {"(", "[", "{", '"', "'", "-"}


class BasicEnglishTokenizer:
    """A compact tokenizer that derives its vocabulary from local texts.

    The implementation intentionally keeps the interface surface similar to
    Hugging Face tokenizers that are used inside the training pipeline.
    Only the methods accessed by the project (``encode``/``decode`` and a
    few token attributes) are implemented, which keeps the dependency
    footprint very small.
    """

    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"

    def __init__(
        self,
        texts: Sequence[str] | None = None,
        vocab_size: int = 8000,
        lowercase: bool = True,
    ) -> None:
        self.lowercase = lowercase
        self._vocab_limit = vocab_size

        self._token_to_id = {
            self.pad_token: 0,
            self.bos_token: 1,
            self.eos_token: 2,
            self.unk_token: 3,
        }
        self.word_boundary_token = "<wb>"
        self._token_to_id[self.word_boundary_token] = 4
        self._id_to_token = {idx: tok for tok, idx in self._token_to_id.items()}
        self._word_boundary_id = self._token_to_id[self.word_boundary_token]

        self._build_vocab(texts or (), vocab_size)
        self._install_fallback_tokens()

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
    def bos_token_id(self) -> int:
        return self._token_to_id[self.bos_token]

    @property
    def all_special_ids(self) -> List[int]:
        return [
            self.pad_token_id,
            self.bos_token_id,
            self.eos_token_id,
            self.unk_token_id,
            self._word_boundary_id,
        ]

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    def get_vocab(self) -> Dict[str, int]:
        """Return a copy of the internal vocabulary mapping."""

        return dict(self._token_to_id)

    def convert_ids_to_tokens(self, ids: Iterable[int]) -> List[str]:
        return [self._id_to_token.get(idx, self.unk_token) for idx in ids]

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
        ids: List[int] = []

        for token in tokens:
            token_id = self._token_to_id.get(token)
            if token_id is not None:
                ids.append(token_id)
                continue

            if token and all(char in _ALPHANUMERIC_FALLBACK for char in token):
                ids.extend(self._token_to_id.get(char, self.unk_token_id) for char in token)
                ids.append(self._word_boundary_id)
                continue

            if token in _PUNCTUATION_FALLBACK:
                ids.append(self._token_to_id.get(token, self.unk_token_id))
                continue

            ids.append(self.unk_token_id)
        if ids and ids[-1] == self._word_boundary_id:
            ids.pop()

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
        char_buffer: List[str] = []

        def flush_buffer() -> None:
            if char_buffer:
                tokens.append("".join(char_buffer))
                char_buffer.clear()

        for idx in ids:
            token = self._id_to_token.get(idx, self.unk_token)
            if skip_special_tokens and token in {
                self.pad_token,
                self.bos_token,
                self.eos_token,
                self.word_boundary_token,
            }:
                flush_buffer()
                continue

            if token == self.word_boundary_token:
                flush_buffer()
                continue

            if token in _ALPHANUMERIC_FALLBACK:
                char_buffer.append(token)
                continue

            flush_buffer()
            tokens.append(token)

        flush_buffer()
        return self._render_tokens(tokens)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _install_fallback_tokens(self) -> None:
        for token in sorted(_ALPHANUMERIC_FALLBACK):
            if len(self._token_to_id) >= self._vocab_limit:
                return
            if token not in self._token_to_id:
                idx = len(self._token_to_id)
                self._token_to_id[token] = idx
                self._id_to_token[idx] = token

        for token in sorted(_PUNCTUATION_FALLBACK):
            if len(self._token_to_id) >= self._vocab_limit:
                return
            if token not in self._token_to_id:
                idx = len(self._token_to_id)
                self._token_to_id[token] = idx
                self._id_to_token[idx] = token

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

    def _render_tokens(self, tokens: List[str]) -> str:
        if not tokens:
            return ""

        rendered: List[str] = [tokens[0]]
        for token in tokens[1:]:
            prev = rendered[-1]
            if token in _NO_SPACE_BEFORE:
                rendered[-1] = prev + token
            elif prev and prev[-1] in _NO_SPACE_AFTER:
                rendered[-1] = prev + token
            else:
                rendered.append(" " + token)
        return "".join(rendered)


    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def save_pretrained(self, save_directory: str) -> None:
        """Persist the tokenizer to ``save_directory``.

        The format mirrors the structure used by Hugging Face tokenisers so
        that higher level utilities (e.g. checkpoint management) can treat
        this tokenizer uniformly.
        """

        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)

        ordered_tokens = [
            token for token, _ in sorted(self._token_to_id.items(), key=lambda item: item[1])
        ]
        config = {
            "type": "basic",
            "lowercase": self.lowercase,
            "tokens": ordered_tokens,
        }

        with (path / "tokenizer_config.json").open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, save_directory: str) -> "BasicEnglishTokenizer":
        """Load a previously saved tokenizer."""

        config_path = Path(save_directory) / "tokenizer_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"未找到分词器配置文件: {config_path}")

        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        tokens = payload.get("tokens")
        if not tokens:
            raise ValueError("basic 分词器配置缺少 tokens 列表")

        instance = cls.__new__(cls)
        instance.lowercase = payload.get("lowercase", True)
        instance._token_to_id = {token: idx for idx, token in enumerate(tokens)}
        instance._id_to_token = {idx: token for token, idx in instance._token_to_id.items()}
        return instance


__all__ = ["BasicEnglishTokenizer"]
