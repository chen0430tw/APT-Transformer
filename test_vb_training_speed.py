
"""
Training speed test for Virtual Blackwell v6.4
- Baseline (no VB) timing
- Optional sparse-attention patch (local sliding-window)
- VB stats summary at end (includes scale reuse rate)
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim

from apt.model.architectures.claude4_model import create_claude4_model

from apt.vgpu.runtime.vb_integration import apply_virtual_blackwell_v64, VBConfigV64, vb_stats_summary

try:
    from apt.vgpu.runtime.vb_sparse_attention import apply_sparse_attention, LocalAttnConfig
    HAS_SPARSE = True
except Exception:
    HAS_SPARSE = False


def run_train_loop(model, device, batches=20, batch_size=8, seq_len=128, vocab=50000, lr=1e-3):
    model.train()
    opt = optim.AdamW(model.parameters(), lr=lr)
    x = torch.randint(0, vocab, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab, (batch_size, seq_len), device=device)

    # warmup
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = nn.functional.cross_entropy(out.view(-1, out.size(-1)), y.view(-1))
        loss.backward()
        opt.step()

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    t0 = time.time()
    losses = []
    for _ in range(batches):
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = nn.functional.cross_entropy(out.view(-1, out.size(-1)), y.view(-1))
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    return time.time() - t0, sum(losses) / len(losses)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    print("Creating Claude4 test model...")
    model = create_claude4_model(vocab_size=50000, d_model=256, num_layers=3, num_heads=8, ffn_hidden=1024).to(device)
    print("Params:", sum(p.numel() for p in model.parameters()))

    base_t, base_loss = run_train_loop(model, device)
    print(f"\n[Baseline] 20 batches: {base_t:.2f}s | avg loss {base_loss:.4f} | {20/base_t:.3f} batch/s")

    # recreate for VB run
    model = create_claude4_model(vocab_size=50000, d_model=256, num_layers=3, num_heads=8, ffn_hidden=1024).to(device)

    if HAS_SPARSE:
        print("\nApplying sparse attention patch (local window=32)...")
        cfg = LocalAttnConfig(window=32, causal=True, dropout_p=0.0, use_sdpa=True)
        patched = apply_sparse_attention(model, n_heads=8, cfg=cfg)
        print(f"Sparse attention patched modules: {patched}")
    else:
        print("\nSparse attention module not found; skipping.")

    print("\nApplying Virtual Blackwell v6.4...")
    cfg = VBConfigV64(
        pulse_interval=20,
        q=0.999,
        update_threshold=0.20,
        ema_alpha=0.10,
        quant_samples=50000,
        cheap_samples=2048,
        use_fake_int8=False,
    )
    model, adapter = apply_virtual_blackwell_v64(model, cfg)

    vb_t, vb_loss = run_train_loop(model, device)
    print(f"\n[VB v6.4] 20 batches: {vb_t:.2f}s | avg loss {vb_loss:.4f} | {20/vb_t:.3f} batch/s")
    print("\nVB stats:")
    print(vb_stats_summary(adapter, top_k=12))


if __name__ == "__main__":
    main()
