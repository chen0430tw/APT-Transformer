#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""APT tokenizer diagnostic utility.

This script exercises both the English and Chinese tokenizers shipped with the
repository.  It builds a small sample corpus, instantiates the requested
tokenizer, and reports vocabulary coverage, encode/decode round-trip behaviour,
and the unknown-token ratio for every sample.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

# Allow execution without installing the package
sys.path.append(".")

from apt_model.modeling.chinese_tokenizer_integration import get_appropriate_tokenizer

DEFAULT_TEST_TEXTS: Mapping[str, Tuple[str, ...]] = {
    "en": (
        "The Autonomous Prompt Transformer builds reasoning step by step.",
        "APT blends self-generated training prompts with curated curricula.",
        "Language tools must decode punctuation, contractions, and casing.",
        "This is a diagnostic sentence for the offline English tokenizer.",
        "Prompt engineering helps align the model with user intent.",
    ),
    "zh": (
        "人工智能正在改变世界",
        "深度学习需要大量数据",
        "自然语言处理是AI的重要分支",
        "机器学习模型训练很重要",
        "这是一个测试样本",
    ),
}

KEY_TOKENS: Mapping[str, Tuple[str, ...]] = {
    "en": ("apt", "transformer", "model", "prompt"),
    "zh": ("人", "工", "智", "能", "学", "习"),
}


def _normalise_language(language: str) -> str:
    if language.lower().startswith("zh"):
        return "zh"
    return "en"


def _resolve_vocab(tokenizer) -> Dict[str, int]:
    raw_vocab: MutableMapping[str, int] | None = None

    if hasattr(tokenizer, "get_vocab"):
        try:
            raw_vocab = dict(tokenizer.get_vocab())
        except Exception:  # pragma: no cover - defensive guard
            raw_vocab = None
    if raw_vocab is None and hasattr(tokenizer, "encoder"):
        raw_vocab = getattr(tokenizer, "encoder")
    if raw_vocab is None and hasattr(tokenizer, "vocab"):
        raw_vocab = getattr(tokenizer, "vocab")

    if raw_vocab is None:
        return {}

    # Ensure we always operate on a concrete dictionary copy
    return {str(token): int(idx) for token, idx in raw_vocab.items()}


def _sample_vocab(vocab: Mapping[str, int], limit: int = 20) -> List[Tuple[str, int]]:
    try:
        items = sorted(vocab.items(), key=lambda item: item[1])
    except Exception:  # pragma: no cover - safeguard for unusual vocab layouts
        items = list(vocab.items())
    return items[:limit]


def _to_id_list(encoded: Iterable[int]) -> List[int]:
    if hasattr(encoded, "tolist"):
        return list(encoded.tolist())
    return [int(x) for x in encoded]


def _resolve_unk_id(tokenizer, vocab: Mapping[str, int]) -> int | None:
    for attr in ("unk_token_id", "unk_id"):
        if hasattr(tokenizer, attr):
            unk_id = getattr(tokenizer, attr)
            if unk_id is not None:
                return int(unk_id)
    for candidate in ("<unk>", "<|unk|>"):
        if candidate in vocab:
            return vocab[candidate]
    return None


def _count_decoded_unknowns(decoded: str) -> int:
    return decoded.count("<unk>") + decoded.count("<|unk|>")


def _print_header(title: str) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)


def diagnose_language(
    language: str,
    tokenizer_type: str | None = None,
    texts: Sequence[str] | None = None,
    max_preview: int = 3,
) -> bool:
    lang = _normalise_language(language)
    display_name = "English" if lang == "en" else "中文"
    default_type = "basic" if lang == "en" else "chinese-char"
    effective_tokenizer_type = tokenizer_type or default_type
    corpus = list(texts or DEFAULT_TEST_TEXTS[lang])

    _print_header(f"APT Tokenizer Diagnostics — {display_name}")

    print(f"1. 样本文本: {len(corpus)} 条")
    for idx, sample in enumerate(corpus[:max_preview], start=1):
        print(f"   {idx}. {sample}")

    print("\n2. 初始化分词器…")
    try:
        tokenizer, detected_lang = get_appropriate_tokenizer(
            texts=corpus,
            tokenizer_type=effective_tokenizer_type,
            language=lang,
        )
    except Exception as exc:  # pragma: no cover - runtime diagnostics
        print(f"   ✗ 初始化失败: {exc}")
        return False

    print("   ✓ 初始化成功")
    print(f"   - 指定语言: {lang}")
    print(f"   - 检测语言: {detected_lang}")
    print(f"   - 分词器类型: {effective_tokenizer_type}")

    vocab = _resolve_vocab(tokenizer)
    print("\n3. 词汇表检查…")
    if vocab:
        print(f"   - 词汇表大小: {len(vocab)}")
        print("   - 前20个词条:")
        for token, idx in _sample_vocab(vocab):
            print(f"      {idx:5d}: {repr(token)}")

        key_tokens = KEY_TOKENS.get(lang, ())
        if key_tokens:
            missing = [token for token in key_tokens if token not in vocab]
            if missing:
                print(f"   ⚠️ 词表缺少关键词: {missing}")
            else:
                print("   ✓ 关键词均已收录")
    else:
        print("   ⚠️ 无法获取词汇表映射 — 检查分词器实现")

    print("\n4. 单条编码测试…")
    probe_text = corpus[0]
    print(f"   输入: {probe_text}")
    try:
        encoded = tokenizer.encode(probe_text)
        encoded_ids = _to_id_list(encoded)
    except Exception as exc:  # pragma: no cover - runtime diagnostics
        print(f"   ✗ 编码失败: {exc}")
        return False

    print(f"   编码长度: {len(encoded_ids)}")
    print(f"   编码结果: {encoded_ids}")

    unk_id = _resolve_unk_id(tokenizer, vocab)
    if unk_id is not None and encoded_ids:
        unk_hits = sum(1 for idx in encoded_ids if idx == unk_id)
        unk_ratio = unk_hits / len(encoded_ids)
        print(
            f"   未知词统计: {unk_hits} / {len(encoded_ids)} ({unk_ratio * 100:.1f}%)"
        )
        if unk_ratio > 0.5:
            print("   ✗ 未知词比例过高 — 请检查词表构建")
    else:
        print("   - 未定义未知词标记，跳过比例分析")

    print("\n5. 解码与往返一致性…")
    try:
        decoded = tokenizer.decode(encoded)
    except Exception as exc:  # pragma: no cover - runtime diagnostics
        print(f"   ✗ 解码失败: {exc}")
        return False

    print(f"   解码结果: {decoded}")
    unk_in_decoded = _count_decoded_unknowns(decoded)
    if unk_in_decoded:
        print(f"   ⚠️ 解码文本包含 {unk_in_decoded} 个未知标记")
    normalised_original = probe_text.replace(" ", "")
    normalised_decoded = decoded.replace(" ", "")
    if normalised_decoded == normalised_original:
        print("   ✓ 编码-解码往返一致")
    else:
        print("   ⚠️ 往返结果存在差异 — 请人工确认是否合理")

    print("\n6. 批量样本检查…")
    successes = 0
    failures = 0
    for sample in corpus:
        try:
            ids = _to_id_list(tokenizer.encode(sample))
            if not ids:
                print(f"   ✗ 空编码输出: {sample}")
                failures += 1
                continue
            if unk_id is None:
                successes += 1
                continue
            unk_ratio = sum(1 for idx in ids if idx == unk_id) / len(ids)
            if unk_ratio <= 0.5:
                successes += 1
            else:
                failures += 1
                print(
                    f"   ✗ 未知词比例过高 ({unk_ratio * 100:.1f}%): {sample[:20]}…"
                )
        except Exception as exc:  # pragma: no cover - runtime diagnostics
            failures += 1
            print(f"   ✗ 样本处理失败: {sample[:20]}… — {exc}")

    print(f"\n   统计: 成功 {successes} 条, 失败 {failures} 条")

    print("\n诊断总结")
    if failures == 0:
        print("✓ 分词器工作正常")
    else:
        print("✗ 分词器存在问题，建议检查词表或样本文本")

    return failures == 0


def parse_tokenizer_overrides(raw_overrides: Sequence[str]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for raw in raw_overrides:
        if ":" in raw:
            lang, tokenizer_type = raw.split(":", 1)
            overrides[_normalise_language(lang)] = tokenizer_type
        else:
            overrides["default"] = raw
    return overrides


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose APT tokenizers")
    parser.add_argument(
        "--language",
        choices=("en", "zh", "both"),
        default="both",
        help="选择要诊断的语言",
    )
    parser.add_argument(
        "--tokenizer-type",
        action="append",
        default=[],
        metavar="LANG:TYPE",
        help="覆盖默认的分词器类型，例如 zh:chinese-word 或 en:basic",
    )
    parser.add_argument(
        "--max-preview",
        type=int,
        default=3,
        help="输出前多少条样本文本",
    )

    args = parser.parse_args(argv)
    overrides = parse_tokenizer_overrides(args.tokenizer_type)

    languages = ["en", "zh"] if args.language == "both" else [args.language]
    overall_success = True

    for lang in languages:
        override = overrides.get(_normalise_language(lang))
        if override is None:
            override = overrides.get("default")
        ok = diagnose_language(lang, tokenizer_type=override, max_preview=args.max_preview)
        overall_success = overall_success and ok
        if lang != languages[-1]:
            print("\n")

    return 0 if overall_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
