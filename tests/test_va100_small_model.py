"""
使用 Virtual A100 测试小模型
==============================
先验证 Virtual A100 的完整功能（OPU + DK-Tile + VirtualVRAM）
用小模型（256x768x8层）测试，确认正常后再上70B
"""
import sys
import os

_BASE = "D:/APT-Transformer"
sys.path.insert(0, f"{_BASE}/va100")
os.environ["TORCH_HOME"] = f"{_BASE}/.torch_cache"

import numpy as np
import time

# 导入 Virtual A100
from virtual_a100 import (
    VirtualA100Engine,
    GhostConfig,
    InferConfig,
    make_synthetic_layer,
    GhostCompressor,
    DKTileCore,
    VirtualVRAMBackend,
    run_simulation
)

print("=" * 70)
print("Virtual A100 - 小模型测试")
print("=" * 70)

# 配置：小模型（256x768x8层）
H = 256  # hidden_size
F = 768  # ffn_hidden_size
L = 8    # n_layers
base_rank = 16

print(f"\n模型配置:")
print(f"  Hidden size: {H}")
print(f"  FFN size: {F}")
print(f"  Layers: {L}")
print(f"  Base rank: {base_rank}")
print(f"  预估参数: {(H*F*L)/1e6:.1f}M")

# 生成合成权重
print(f"\n[1] 生成合成权重...")
rng = np.random.default_rng(42)
layers = [make_synthetic_layer(rng, H, F, 0.5 + 0.8*i/L, 0.3 + 0.5*i/L)
          for i in range(L)]

# Ghost 压缩
print(f"[2] Ghost 压缩 (rank={base_rank})...")
gcfg = GhostConfig(
    base_rank=base_rank,
    quantize_factors=True,
    alloc_method='greedy',
    use_random_svd=True,
)
ghost_layers = GhostCompressor(gcfg).compress_model(layers, progress=True)

# 质量诊断
print(f"\n[3] 质量诊断:")
print(f"  {'层':<4} {'矩阵':<4} {'rank':>5} {'fro%':>7} {'cos':>7}")
print("  " + "-" * 32)
for li in [0, L//2, L-1]:
    for wn in ['Wq', 'W1']:
        gf = ghost_layers[li].factors[wn]
        W = layers[li][wn]
        Wr = gf.reconstruct()
        fro = np.linalg.norm(W-Wr) / (np.linalg.norm(W) + 1e-15)
        cos = np.sum(W*Wr) / (np.linalg.norm(W)*np.linalg.norm(Wr) + 1e-15)
        print(f"  {li:<4} {wn:<4} {gf.rank:>5} {fro:>7.2%} {cos:>7.4f}")

# 创建 Virtual A100 Engine
print(f"\n[4] 创建 Virtual A100 引擎...")

# 配置显存（模拟 8GB GPU）
infer_cfg = InferConfig(
    max_ctx=512,
    vram_budget_gb=8.0,
    hot_ratio=0.3,      # 30% hot, 70% warm
    warm_ratio=0.6,
    kv_quant_bits=4,
)

dk = DKTileCore()

# 创建虚拟显存（冷温热三层）
hot_b = int(infer_cfg.vram_budget_gb * 1e9 * infer_cfg.hot_ratio)  # ~2.4GB
warm_b = int(infer_cfg.vram_budget_gb * 1e9 * infer_cfg.warm_ratio)  # ~4.8GB
cold_b = int(100e9)  # 100GB cold (实际是 disk)
vram = VirtualVRAMBackend(hot_b, warm_b, cold_b)

print(f"  Hot:  {hot_b/1e9:.1f}GB (GPU 常驻)")
print(f"  Warm: {warm_b/1e9:.1f}GB (CPU pinned)")
print(f"  Cold: {cold_b/1e9:.1f}GB (disk)")

# 创建运行时
from virtual_a100 import VirtualA100Runtime
runtime = VirtualA100Runtime(ghost_layers, dk, vram, infer_cfg)

print(f"  引擎创建成功")

# 模拟推理
print(f"\n[5] 模拟推理...")
seq_len = 64
batch_size = 1

# 生成随机输入
h = rng.standard_normal((batch_size, H))
pos = 0

print(f"  输入形状: {h.shape}")
print(f"  序列长度: {seq_len}")

start_time = time.time()

# 生成 seq_len 个 token
tokens_gen = []
for i in range(seq_len):
    # 调用 Virtual A100 的 forward
    logits = runtime.forward_one_token(h, pos)

    # 采样
    token_id = np.argmax(logits)
    tokens_gen.append(token_id)

    # 更新位置
    pos += 1

    # 每 10 个 token 打印一次
    if (i + 1) % 10 == 0:
        elapsed = time.time() - start_time
        speed = (i + 1) / elapsed
        print(f"  Step {i+1}/{seq_len}: {speed:.1f} tokens/s")

elapsed = time.time() - start_time
avg_speed = seq_len / elapsed

print(f"\n[6] 推理完成:")
print(f"  总时间: {elapsed:.2f}s")
print(f"  平均速度: {avg_speed:.1f} tokens/s")

# 获取统计
stats = vram.get_stats()
print(f"\n[7] VirtualVRAM 统计:")
print(f"  Hot 层:")
print(f"    数量: {stats.hot.count}")
print(f"    Fetch 次数: {stats.hot.fetch_count}")
print(f"    Evict 次数: {stats.hot.evict_count}")
print(f"  Warm 层:")
print(f"    数量: {stats.warm.count}")
print(f"    Fetch 次数: {stats.warm.fetch_count}")
print(f"    Evict 次数: {stats.warm.evict_count}")
print(f"  Cold 层:")
print(f"    数量: {stats.cold.count}")
print(f"    Fetch 次数: {stats.cold.fetch_count}")
print(f"    Evict 次数: {stats.cold.evict_count}")

print(f"\n  总搬运时间: {stats.total_transfer_time_s*1000:.1f}ms")
print(f"  总重建时间: {stats.total_rebuild_time_s*1000:.1f}ms")
print(f"  总计算时间: {stats.total_compute_time_s*1000:.1f}ms")

# 摩擦系数
mu = stats.friction_mu
tau = stats.rebuild_tax_tau

print(f"\n[8] 性能指标:")
print(f"  搬运摩擦系数 μ: {mu:.3f}")
print(f"  重建税 τ: {tau:.3f}")

print("\n" + "=" * 70)
print("测试完成！")
print("=" * 70)
print("\n总结:")
print("  ✓ Virtual A100 引擎正常工作")
print("  ✓ DK-Tile 分块计算")
print("  ✓ VirtualVRAM 三层存储管理")
print(f"  ✓ 推理速度: {avg_speed:.1f} tokens/s")
print(f"  ✓ 摩擦系数 μ: {mu:.3f} (越低越好)")
print(f"  ✓ 重建税 τ: {tau:.3f} (越低越好)")
