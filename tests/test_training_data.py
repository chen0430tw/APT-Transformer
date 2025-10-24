"""Tests covering the local training datasets and tokeniser selection."""

from apt_model.training import trainer
from apt_model.modeling.basic_tokenizer import BasicEnglishTokenizer
from apt_model.modeling.chinese_tokenizer_integration import (
    load_tokenizer,
    save_tokenizer,
)


def test_get_training_texts_returns_builtin_prompts():
    texts = trainer.get_training_texts()

    assert texts, "默认训练数据不应为空"
    # 不应出现重复，去重逻辑应生效
    assert len(texts) == len(set(texts))


def test_get_training_texts_falls_back_to_builtin(monkeypatch):
    monkeypatch.setattr(trainer, "_load_training_texts_from_files", lambda base_dir: [])

    texts = trainer.get_training_texts()

    assert "Hello, how are you?" in texts
    assert any("派蒙" in text for text in texts)


def test_basic_english_tokenizer_encodes_from_local_vocab():
    texts = ["Hello world", "Machine learning"]
    tokenizer = BasicEnglishTokenizer(texts=texts, vocab_size=32)

    encoded = tokenizer.encode("Hello world", return_tensors="pt")
    assert encoded.ndim == 2
    assert encoded.shape[1] >= 3  # includes EOS token

    decoded = tokenizer.decode(encoded[0].tolist())
    assert "hello" in decoded


def test_basic_english_tokenizer_exposes_special_token_ids():
    tokenizer = BasicEnglishTokenizer(texts=["Hello"], vocab_size=8)

    assert tokenizer.pad_token_id != tokenizer.eos_token_id
    assert tokenizer.bos_token_id not in {tokenizer.pad_token_id, tokenizer.eos_token_id}
    decoded = tokenizer.decode([tokenizer.bos_token_id, tokenizer.eos_token_id])
    assert decoded == ""


def test_basic_tokenizer_can_be_serialized(tmp_path):
    tokenizer = BasicEnglishTokenizer(texts=["Hello there", "General Kenobi"], vocab_size=32)

    target_dir = tmp_path / "tokenizer"
    saved = save_tokenizer(tokenizer, target_dir)
    assert saved, "保存基础分词器时应返回 True"

    reloaded = load_tokenizer(target_dir)
    assert isinstance(reloaded, BasicEnglishTokenizer)
    assert reloaded.get_vocab() == tokenizer.get_vocab()
    assert reloaded.lowercase == tokenizer.lowercase


def test_basic_tokenizer_character_fallback_roundtrip():
    tokenizer = BasicEnglishTokenizer(texts=["Hello there"], vocab_size=128)

    encoded = tokenizer.encode("Amber: training is not enough", return_tensors=None)
    assert tokenizer.unk_token_id not in encoded

    decoded = tokenizer.decode(encoded)
    assert decoded == "amber: training is not enough"


def test_basic_tokenizer_convert_ids_to_tokens_exposes_known_tokens():
    tokenizer = BasicEnglishTokenizer(texts=["Hello there"], vocab_size=128)
    encoded = tokenizer.encode("Hello", return_tensors=None)
    tokens = tokenizer.convert_ids_to_tokens(encoded)
    assert "hello" in tokens
