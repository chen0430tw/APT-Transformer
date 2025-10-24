"""Tests covering the local training datasets and tokeniser selection."""

from apt_model.training.trainer import get_training_texts
from apt_model.modeling.basic_tokenizer import BasicEnglishTokenizer


def test_get_training_texts_returns_builtin_prompts():
    texts = get_training_texts()

    assert "Hello, how are you?" in texts
    assert any("人工智能" in text for text in texts)
    # 预设的对话样本应该始终可用，例如派蒙相关的台词
    assert any("派蒙" in text for text in texts)
    # 不应出现重复，去重逻辑应生效
    assert len(texts) == len(set(texts))


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
