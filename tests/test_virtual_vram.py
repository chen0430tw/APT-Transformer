"""
Virtual VRAM v0.2 综合测试 (训练友好版)
==========================================
对比三种模式:
  1. 无虚拟显存 (baseline)
  2. v0.1 模式: 无压缩 offload
  3. v0.2 FP16 压缩: 2x 压缩 + 流式预取

注意: INT8 压缩已移除，因为会破坏梯度流动，不适合训练。
INT8 仅适用于推理场景。
"""
import os
import sys

# ═══════════════════════════════════════════════════════════════════
# 路径配置 — 全部丢 D 盘
# ═══════════════════════════════════════════════════════════════════
_BASE = "D:/APT-Transformer"
os.environ["TORCH_HOME"] = f"{_BASE}/.torch_cache"
os.environ["TEMP"] = f"{_BASE}/.temp"
os.environ["TMP"] = f"{_BASE}/.temp"
os.environ["TMPDIR"] = f"{_BASE}/.temp"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"{_BASE}/.cache/torchinductor"
os.environ["TRITON_CACHE_DIR"] = f"{_BASE}/.cache/triton"
for d in [
    f"{_BASE}/.torch_cache", f"{_BASE}/.temp",
    f"{_BASE}/.cache/torchinductor", f"{_BASE}/.cache/triton",
]:
    os.makedirs(d, exist_ok=True)

import gc
import time
import torch
import torch.nn as nn
from apt.vgpu.runtime.virtual_vram import VirtualVRAMConfig, virtual_vram

device = "cuda"
print(f"Device: {device}")
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"Total VRAM: {total_vram:.1f} GB")
print(f"缓存路径: {_BASE}/.torch_cache")
print(f"临时路径: {_BASE}/.temp\n")

torch.cuda.set_per_process_memory_fraction(0.95)


# ── 测试模型: 12 层 Transformer ──
class DeepTransformer(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=12, dim_feedforward=3072):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True, dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.head(self.encoder(x))


model = DeepTransformer().to(device)
model.train()
param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
print(f"模型参数: {param_mb:.0f} MB  (12 层 Transformer, d=768)\n")


# ── 测试函数 ──
def run_test(label: str, cfg=None, batch_size=16, seq_len=512):
    """运行一次 forward+backward，返回 (peak_gb, grads, elapsed_ms)"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    # 固定随机种子 → 梯度可比较
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, 768, device=device)

    start = time.perf_counter()

    if cfg is not None:
        with virtual_vram(cfg):
            y = model(x)
            loss = y.mean()
            loss.backward()
    else:
        y = model(x)
        loss = y.mean()
        loss.backward()

    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    peak = torch.cuda.max_memory_allocated(device) / 1024**3

    # 收集梯度快照
    grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads[name] = p.grad.detach().clone()

    model.zero_grad(set_to_none=True)
    del x, y, loss
    gc.collect()
    torch.cuda.empty_cache()

    return peak, grads, elapsed


# ═══════════════════════════════════════════════════════════════════
# 运行三组测试
# ═══════════════════════════════════════════════════════════════════
batch_size = 16
seq_len = 512

print("=" * 70)
print(f"Batch={batch_size}, Seq={seq_len}")
print("=" * 70)

# 1. Baseline
print("\n[1/3] Baseline (无虚拟显存)...")
peak_base, grads_base, ms_base = run_test("baseline", cfg=None,
                                           batch_size=batch_size, seq_len=seq_len)
print(f"  峰值: {peak_base:.2f} GB, 耗时: {ms_base:.0f} ms")

# 2. v0.1 模式: 无压缩
print("\n[2/3] v0.1 模式 (无压缩 offload + 流式预取)...")
cfg_v01 = VirtualVRAMConfig(
    enabled=True, min_tensor_bytes=1<<22,
    compress=False, stream_prefetch=True, verbose=False,
)
peak_v01, grads_v01, ms_v01 = run_test("v0.1", cfg=cfg_v01,
                                        batch_size=batch_size, seq_len=seq_len)
print(f"  峰值: {peak_v01:.2f} GB, 耗时: {ms_v01:.0f} ms")

# 3. v0.2 FP16 压缩
print("\n[3/3] v0.2 FP16 压缩 (2x)...")
cfg_fp16 = VirtualVRAMConfig(
    enabled=True, min_tensor_bytes=1<<22,
    compress=True, compress_dtype="float16",
    stream_prefetch=True, verbose=False,
)
peak_fp16, grads_fp16, ms_fp16 = run_test("v0.2-fp16", cfg=cfg_fp16,
                                           batch_size=batch_size, seq_len=seq_len)
print(f"  峰值: {peak_fp16:.2f} GB, 耗时: {ms_fp16:.0f} ms")


# ═══════════════════════════════════════════════════════════════════
# 梯度精度对比
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("梯度精度对比 (相对于 baseline)")
print("=" * 70)

def grad_error(grads_test, grads_ref, label):
    """计算所有参数梯度的平均相对误差"""
    total_err = 0.0
    count = 0
    for name in grads_ref:
        if name in grads_test:
            ref = grads_ref[name].float()
            test = grads_test[name].float()
            err = (test - ref).norm()
            norm = ref.norm()
            if norm > 1e-8:
                total_err += (err / norm).item()
                count += 1
    avg_rel = total_err / max(count, 1)
    print(f"  {label}: 平均相对误差 = {avg_rel:.6f} ({avg_rel*100:.4f}%)")
    return avg_rel

err_v01 = grad_error(grads_v01, grads_base, "v0.1 无压缩")
err_fp16 = grad_error(grads_fp16, grads_base, "v0.2 FP16  ")


# ═══════════════════════════════════════════════════════════════════
# 汇总
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RESULTS 汇总")
print("=" * 70)

def row(label, peak, ms, err=None):
    saved = peak_base - peak
    pct = saved / peak_base * 100
    err_str = f"{err*100:.4f}%" if err is not None else "—"
    print(f"  {label:20s}  峰值={peak:.2f} GB  "
          f"节省={saved:+.2f} GB ({pct:+.0f}%)  "
          f"耗时={ms:.0f} ms  梯度误差={err_str}")

row("Baseline",        peak_base, ms_base)
row("v0.1 无压缩",     peak_v01,  ms_v01,  err_v01)
row("v0.2 FP16 (2x)",  peak_fp16, ms_fp16, err_fp16)

print(f"\n  关键结论:")
print(f"  - FP16 压缩搬运量为原始的 1/2")
print(f"  - 流式预取让 GPU↔CPU 传输与计算重叠")
print(f"  - INT8 压缩仅用于推理，训练会破坏梯度流动（已移除）")
if err_v01 < 0.001 and err_fp16 < 0.001:
    print(f"  - 梯度误差 < 0.1%，训练可用 ✅")
print("=" * 70)
