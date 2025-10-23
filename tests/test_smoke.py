"""Smoke test for the minimal APT model build."""

import pytest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment specific
    torch = None


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed in this environment")
def test_build_and_forward_minimal():
    from apt_model.config.apt_config import APTConfig
    from apt_model.modeling.apt_model import APTModel
    cfg = APTConfig(vocab_size=256, d_model=64, d_ff=128, num_heads=8, 
                    num_encoder_layers=1, num_decoder_layers=1, max_seq_len=32)
    model = APTModel(cfg)
    B, Ls, Lt = 2, 8, 6
    src = torch.randint(0, cfg.vocab_size, (B, Ls))
    tgt = torch.randint(0, cfg.vocab_size, (B, Lt))
    out = model(src, tgt)
    assert out.shape[0] == B
