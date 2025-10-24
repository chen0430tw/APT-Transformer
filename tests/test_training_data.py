"""Tests covering the local training datasets and tokeniser selection."""

from apt_model.training.trainer import get_training_texts
from apt_model.modeling.basic_tokenizer import BasicEnglishTokenizer


def test_get_training_texts_uses_repository_files():
    texts = get_training_texts()

    assert any("Hello" in text or "hello" in text for text in texts)
    assert any("人工智能" in text for text in texts)


def test_basic_english_tokenizer_encodes_from_local_vocab():
    texts = ["Hello world", "Machine learning"]
    tokenizer = BasicEnglishTokenizer(texts=texts, vocab_size=32)

    encoded = tokenizer.encode("Hello world", return_tensors="pt")
    assert encoded.ndim == 2
    assert encoded.shape[1] >= 3  # includes EOS token

    decoded = tokenizer.decode(encoded[0].tolist())
    assert "hello" in decoded
