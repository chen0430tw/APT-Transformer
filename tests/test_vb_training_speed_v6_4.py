"""
Training speed test for Virtual Blackwell v6.4 (整合版)
========================================================
功能：
- Baseline vs VB v6.4 性能对比
- 支持快速测试模式（默认 20 batches）和性能测试模式（200 batches）
- 可选 sparse-attention 补丁
- 完整的 VB 统计输出（包括 scale 复用率）

使用：
python test_vb_training_speed_v6_4.py              # 快速测试（20 batches）
python test_vb_training_speed_v6_4.py --perf      # 性能测试（200 batches）
python test_vb_training_speed_v6_4.py --batches 100  # 自定义批次
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim

from apt.model.architectures.claude4_model import create_claude4_model
from apt.vgpu.runtime.vb_integration import apply_virtual_blackwell_v64, VBConfigV64, vb_stats_summary

# 稳健的 sparse attention 导入
try:
    from apt.vgpu.runtime.vb_sparse_attention import apply_sparse_attention, LocalAttnConfig
    HAS_SPARSE = True
except Exception as e:
    HAS_SPARSE = False
    SPARSE_ERROR = str(e)


def run_train_loop(model, device, batches=20, batch_size=8, seq_len=128, vocab=50000, lr=1e-3):
    """运行训练循环，返回 (时间, 平均损失)"""
    model.train()
    opt = optim.AdamW(model.parameters(), lr=lr)

    # 合成数据
    x = torch.randint(0, vocab, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab, (batch_size, seq_len), device=device)

    # Warmup（稳定 CUDA kernels）
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = nn.functional.cross_entropy(out.view(-1, out.size(-1)), y.view(-1))
        loss.backward()
        opt.step()

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    # 正式测试
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
    parser = argparse.ArgumentParser(description="Virtual Blackwell v6.4 训练速度测试")
    parser.add_argument("--batches", type=int, default=20, help="批次数（默认 20，性能测试推荐 200）")
    parser.add_argument("--batch-size", type=int, default=8, help="批次大小")
    parser.add_argument("--seq-len", type=int, default=128, help="序列长度")
    parser.add_argument("--vocab", type=int, default=50000, help="词表大小")
    parser.add_argument("--perf", action="store_true", help="性能测试模式（等同于 --batches 200 --batch-size 4 --seq-len 1024）")
    parser.add_argument("--no-sparse", action="store_true", help="禁用 sparse attention")
    args = parser.parse_args()

    # 性能模式覆盖参数
    if args.perf:
        args.batches = 200
        args.batch_size = 4
        args.seq_len = 1024
        args.vocab = 4096

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    print("=" * 70)
    print("Virtual Blackwell v6.4 训练速度测试")
    print("=" * 70)
    print(f"设备: {device}")
    print(f"配置: {args.batches} batches, bs={args.batch_size}, seq_len={args.seq_len}, vocab={args.vocab}")
    print()

    # 创建模型
    print("创建 Claude4 测试模型...")
    model = create_claude4_model(
        vocab_size=args.vocab,
        d_model=256,
        num_layers=3,
        num_heads=8,
        ffn_hidden=1024
    ).to(device)
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Baseline 测试
    print("-" * 70)
    print("[Baseline] 测试中...")
    base_t, base_loss = run_train_loop(
        model, device,
        batches=args.batches,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab=args.vocab
    )
    print(f"[Baseline] {args.batches} batches: {base_t:.2f}s | "
          f"avg loss={base_loss:.4f} | {args.batches/base_t:.3f} batch/s")

    # 重建模型用于 VB 测试
    print()
    print("重建模型用于 VB 测试...")
    model = create_claude4_model(
        vocab_size=args.vocab,
        d_model=256,
        num_layers=3,
        num_heads=8,
        ffn_hidden=1024
    ).to(device)

    # 应用 sparse attention（可选）
    use_sparse = HAS_SPARSE and not args.no_sparse
    if use_sparse:
        print()
        print("应用 sparse attention 补丁（local window=32）...")
        cfg = LocalAttnConfig(window=32, causal=True, dropout_p=0.0, use_sdpa=True)
        patched = apply_sparse_attention(model, n_heads=8, cfg=cfg)
        print(f"  已补丁模块数: {patched}")
    elif not HAS_SPARSE:
        print()
        print(f"  Sparse attention 不可用: {SPARSE_ERROR}")

    # 应用 Virtual Blackwell v6.4
    print()
    print("应用 Virtual Blackwell v6.4...")
    vb_cfg = VBConfigV64(
        pulse_interval=20,
        q=0.999,
        update_threshold=0.20,
        ema_alpha=0.10,
        quant_samples=50000,
        cheap_samples=2048,
        use_fake_int8=False,
        act_rms_threshold=0.02,
        min_act_samples=1,  # 允许第一次后就开始复用
    )
    model, adapter = apply_virtual_blackwell_v64(model, vb_cfg)
    print(f"  已替换 Linear 层数: {getattr(model, '_vb_replaced_linears', 'unknown')}")
    print()

    # VB 测试
    print("-" * 70)
    print("[VB v6.4] 测试中...")
    vb_t, vb_loss = run_train_loop(
        model, device,
        batches=args.batches,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab=args.vocab
    )
    print(f"[VB v6.4] {args.batches} batches: {vb_t:.2f}s | "
          f"avg loss={vb_loss:.4f} | {args.batches/vb_t:.3f} batch/s")

    # 统计汇总
    print()
    print("=" * 70)
    print("性能对比")
    print("=" * 70)
    speedup = base_t / vb_t
    overhead = (vb_t / base_t - 1) * 100
    print(f"Baseline: {base_t:.2f}s ({args.batches/base_t:.3f} batch/s)")
    print(f"VB v6.4:  {vb_t:.2f}s ({args.batches/vb_t:.3f} batch/s)")
    print(f"加速比:   {speedup:.3f}x")
    print(f"开销:     {overhead:+.1f}%")
    print()

    print("=" * 70)
    print("VB 统计")
    print("=" * 70)
    print(vb_stats_summary(adapter, top_k=12))
    print("=" * 70)


if __name__ == "__main__":
    main()
