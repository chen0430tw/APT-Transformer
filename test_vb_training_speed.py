
"""
Training speed test for Virtual Blackwell v6.2
Adds:
- Baseline (no VB) timing
- Optional sparse-attention patch before VB
- VB stats summary at end
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim

from apt.model.architectures.claude4_model import Claude4Model

# VB + sparse attention
from apt.vgpu.runtime.vb_integration import apply_virtual_blackwell_v62, VBConfigV62, vb_stats_summary
try:
    from apt.vgpu.runtime.vb_sparse_attention import apply_sparse_attention
    HAS_SPARSE = True
except Exception:
    HAS_SPARSE = False


def run_train_loop(model, device, batches=20, batch_size=8, seq_len=128, vocab=50000, lr=1e-3):
    model.train()
    opt = optim.AdamW(model.parameters(), lr=lr)
    # synthetic token batch
    x = torch.randint(0, vocab, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab, (batch_size, seq_len), device=device)

    # warmup 2 iters to stabilize CUDA kernels
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = nn.functional.cross_entropy(out.view(-1, out.size(-1)), y.view(-1))
        loss.backward()
        opt.step()

    torch.cuda.synchronize() if device.startswith("cuda") else None
    t0 = time.time()
    losses = []
    for i in range(batches):
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = nn.functional.cross_entropy(out.view(-1, out.size(-1)), y.view(-1))
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    torch.cuda.synchronize() if device.startswith("cuda") else None
    return time.time() - t0, sum(losses) / len(losses)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    print("Creating Claude4 test model...")
    model = Claude4Model(vocab_size=50000, d_model=256, num_layers=3, n_heads=8, d_ff=1024, enable_reflection=False).to(device)
    print("Params:", sum(p.numel() for p in model.parameters()))

    # baseline
    base_t, base_loss = run_train_loop(model, device)
    print(f"\n[Baseline] 20 batches: {base_t:.2f}s | avg loss {base_loss:.4f} | {20/base_t:.3f} batch/s")

    # recreate model for fair comparison
    model = Claude4Model(vocab_size=50000, d_model=256, num_layers=3, n_heads=8, d_ff=1024, enable_reflection=False).to(device)

    if HAS_SPARSE:
        print("\nApplying sparse attention patch...")
        # local window of 32 as a safe default for seq_len=128
        apply_sparse_attention(model, window_size=32, topk=None)
        print("Sparse attention patched.")
    else:
        print("\nSparse attention module not found; skipping.")

    print("\nApplying Virtual Blackwell v6.2...")
    cfg = VBConfigV62(pulse_interval=20, q=0.999, sample_size=8192, drift_threshold=0.20, enable_fp4_coarse=True)
    model, adapter = apply_virtual_blackwell_v62(model, cfg)

    vb_t, vb_loss = run_train_loop(model, device)
    print(f"\n[VB v6.2] 20 batches: {vb_t:.2f}s | avg loss {vb_loss:.4f} | {20/vb_t:.3f} batch/s")
    print("\nVB stats:")
    print(vb_stats_summary(adapter, top_k=12))


if __name__ == "__main__":
    main()
