#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
training_time_estimator.py — 速食训练时长估算工具

原理：
  1. 在目标硬件上用真实模型尺寸跑 N 步热身 + M 步计时
  2. 量出稳定的 tok/s（tokens per second）
  3. 按 Stage 1/2 的目标 token 数反算总时长

用法：
  # 用 APT 默认尺寸 (d_model=768, 12层) 测速
  python tools/training_time_estimator.py

  # 指定模型尺寸和卡数
  python tools/training_time_estimator.py --d-model 1024 --num-layers 24 --num-gpus 8

  # 只测速，不打印 Stage 估算
  python tools/training_time_estimator.py --benchmark-only
"""

import os
import sys
import time
import math
import argparse
import logging

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("time_estimator")

# =========================================================================== #
# Stage 配置（目标 token 数，可按项目需要修改）
# =========================================================================== #

STAGE_CONFIGS = {
    1: {
        "name": "Stage 1 — 通用预训练底噪",
        "desc": "C4/FineWeb/mC4 主导，多语言通用能力",
        "target_tokens": 100_000_000_000,   # 100B tokens（可调）
        "data_sources": "C4 35% + FineWeb 25% + Chinese-C4 20% + Wiki 10% + other 10%",
    },
    2: {
        "name": "Stage 2 — 数学/推理强化",
        "desc": "proof-pile-2 + FineMath + Cosmopedia 主导",
        "target_tokens": 50_000_000_000,    # 50B tokens（可调）
        "data_sources": "arXiv 15% + FineMath 10% + Code 12% + Cosmopedia 5% + C4 10% + other",
    },
}


def fmt_duration(seconds: float) -> str:
    """将秒数格式化为人类可读字符串。"""
    if seconds < 3600:
        return f"{seconds/60:.1f} 分钟"
    if seconds < 86400:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h} 小时 {m} 分钟"
    d = int(seconds // 86400)
    h = int((seconds % 86400) // 3600)
    return f"{d} 天 {h} 小时"


def fmt_tokens(n: int) -> str:
    if n >= 1e12:
        return f"{n/1e12:.1f}T"
    if n >= 1e9:
        return f"{n/1e9:.0f}B"
    if n >= 1e6:
        return f"{n/1e6:.0f}M"
    return str(n)


# =========================================================================== #
# 基准测速
# =========================================================================== #

def run_benchmark(
    arch: str = "apt",
    d_model: int = 768,
    num_heads: int = 12,
    num_layers: int = 12,
    seq_len: int = 2048,
    batch_size: int = 4,
    vocab_size: int = 32000,
    warmup_steps: int = 5,
    bench_steps: int = 20,
) -> float:
    """
    用随机数据跑 forward + backward，返回 tokens/sec。
    不需要真实数据集，只测模型吞吐量。
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")

    # 建模型
    logger.info(f"建模: arch={arch}, d_model={d_model}, layers={num_layers}, heads={num_heads}")
    from apt.trainops.scripts.pretrain_quickcook import create_model
    model = create_model(
        arch=arch,
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=seq_len,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"参数量: {n_params:,} ({n_params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    tokens_per_step = batch_size * seq_len

    def one_step():
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        optimizer.zero_grad()
        out = model(x)
        # 兼容不同输出形式（logits 直接输出 or (logits, ...) tuple）
        if isinstance(out, tuple):
            logits = out[0]
        else:
            logits = out
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        return loss.item()

    # 热身
    logger.info(f"热身 {warmup_steps} 步...")
    for _ in range(warmup_steps):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    # 计时
    logger.info(f"计时 {bench_steps} 步...")
    t0 = time.perf_counter()
    for _ in range(bench_steps):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tok_per_sec = (tokens_per_step * bench_steps) / elapsed
    sec_per_step = elapsed / bench_steps

    logger.info(f"计时结果: {elapsed:.2f}s / {bench_steps} 步")
    logger.info(f"  每步耗时:   {sec_per_step*1000:.1f} ms")
    logger.info(f"  tokens/s:   {tok_per_sec:,.0f}")
    logger.info(f"  tokens/step:{tokens_per_step:,}  (batch={batch_size} × seq={seq_len})")

    return tok_per_sec


# =========================================================================== #
# 时长估算
# =========================================================================== #

def estimate_stages(
    tok_per_sec_single_gpu: float,
    num_gpus: int,
    batch_size: int,
    seq_len: int,
    grad_accum: int = 1,
):
    """
    根据单卡 tok/s 估算各 Stage 训练时长。

    tok_per_sec_single_gpu: benchmark 测出的单卡 tok/s
    num_gpus: 实际使用卡数（线性加速假设）
    """
    # 多卡线性加速（实际效率约 90-95%，保守估计用 0.9）
    scaling_efficiency = 0.90
    effective_tok_per_sec = tok_per_sec_single_gpu * num_gpus * scaling_efficiency

    tokens_per_step = batch_size * seq_len * num_gpus * grad_accum

    print("\n" + "=" * 65)
    print("  APT 速食预训练时长估算")
    print("=" * 65)
    print(f"  硬件配置  : {num_gpus} GPU(s)")
    print(f"  单卡吞吐  : {tok_per_sec_single_gpu:,.0f} tok/s")
    print(f"  有效吞吐  : {effective_tok_per_sec:,.0f} tok/s  (含 {scaling_efficiency:.0%} 多卡效率)")
    print(f"  每步 tokens: {tokens_per_step:,}  (batch={batch_size}×seq={seq_len}×gpus={num_gpus}×accum={grad_accum})")
    print("=" * 65)

    for stage_id, cfg in STAGE_CONFIGS.items():
        target = cfg["target_tokens"]
        total_steps = math.ceil(target / tokens_per_step)
        total_sec = target / effective_tok_per_sec

        print(f"\n  {cfg['name']}")
        print(f"  描述      : {cfg['desc']}")
        print(f"  数据配比  : {cfg['data_sources']}")
        print(f"  目标 tokens: {fmt_tokens(target)}")
        print(f"  总步数    : {total_steps:,} 步")
        print(f"  估算时长  : {fmt_duration(total_sec)}")

        # 不同卡数的横向对比
        print(f"  卡数对比  :")
        for gpus in [1, 2, 4, 8, 16]:
            eff = tok_per_sec_single_gpu * gpus * scaling_efficiency
            t = target / eff
            print(f"    {gpus:>2d} GPU : {fmt_duration(t)}")

    print("\n" + "=" * 65)
    print("  注意：以上为理论估算。实际时长受以下因素影响：")
    print("    - 数据加载 IO 瓶颈（流式 HF 数据集有网络延迟）")
    print("    - 检查点保存频率")
    print("    - 学习率调度、梯度裁剪额外开销")
    print("    - 多卡通信效率（NVLink vs PCIe）")
    print("=" * 65 + "\n")


# =========================================================================== #
# 主入口
# =========================================================================== #

def main():
    parser = argparse.ArgumentParser(description="APT 速食训练时长估算工具")
    parser.add_argument("--arch", default="apt",
                        choices=["apt", "apt-lite", "gpt4o", "gpt5", "claude4", "gpto3"],
                        help="模型架构 (默认 apt)")
    parser.add_argument("--d-model", type=int, default=768, help="模型维度 (默认 768)")
    parser.add_argument("--num-heads", type=int, default=12, help="注意力头数 (默认 12)")
    parser.add_argument("--num-layers", type=int, default=12, help="层数 (默认 12)")
    parser.add_argument("--seq-len", type=int, default=2048, help="序列长度 (默认 2048)")
    parser.add_argument("--batch-size", type=int, default=4, help="单卡 batch size (默认 4)")
    parser.add_argument("--vocab-size", type=int, default=32000, help="词表大小 (默认 32000)")
    parser.add_argument("--num-gpus", type=int, default=1, help="实际 GPU 卡数（用于时长估算，默认 1）")
    parser.add_argument("--grad-accum", type=int, default=1, help="梯度累积步数 (默认 1)")
    parser.add_argument("--warmup-steps", type=int, default=5, help="热身步数 (默认 5)")
    parser.add_argument("--bench-steps", type=int, default=20, help="计时步数 (默认 20)")
    parser.add_argument("--tok-per-sec", type=float, default=None,
                        help="直接指定 tok/s（跳过 benchmark，只做估算）")
    parser.add_argument("--stage1-tokens", type=float, default=100e9,
                        help="Stage 1 目标 token 数 (默认 100B)")
    parser.add_argument("--stage2-tokens", type=float, default=50e9,
                        help="Stage 2 目标 token 数 (默认 50B)")
    parser.add_argument("--benchmark-only", action="store_true",
                        help="只跑 benchmark，不打印 Stage 估算")
    args = parser.parse_args()

    # 更新 Stage 配置中的目标 token 数
    STAGE_CONFIGS[1]["target_tokens"] = int(args.stage1_tokens)
    STAGE_CONFIGS[2]["target_tokens"] = int(args.stage2_tokens)

    if args.tok_per_sec is not None:
        tok_per_sec = args.tok_per_sec
        logger.info(f"使用手动指定的 tok/s: {tok_per_sec:,.0f}")
    else:
        tok_per_sec = run_benchmark(
            arch=args.arch,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            vocab_size=args.vocab_size,
            warmup_steps=args.warmup_steps,
            bench_steps=args.bench_steps,
        )

    if not args.benchmark_only:
        estimate_stages(
            tok_per_sec_single_gpu=tok_per_sec,
            num_gpus=args.num_gpus,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            grad_accum=args.grad_accum,
        )


if __name__ == "__main__":
    main()
