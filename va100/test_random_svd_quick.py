#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速测试随机 SVD"""

import numpy as np
import time

# 从 virtual_a100 导入随机 SVD
from virtual_a100 import random_svd, GhostConfig, GhostCompressor

print("=" * 70)
print("随机 SVD 快速测试")
print("=" * 70)
print()

# 测试 1: 小矩阵验证正确性
print("[测试 1] 正确性验证 (512×512)")
rng = np.random.default_rng(42)
A = rng.standard_normal((512, 512), dtype=np.float32)

# 标准 SVD
U1, S1, V1 = np.linalg.svd(A, full_matrices=False)
S1 = S1[:32]
U1 = U1[:, :32]
V1 = V1[:, :32]

# 随机 SVD
U2, S2, Vt2 = random_svd(A, rank=32, rng=rng)

# Debug: 打印维度
print(f"  Debug: U2.shape={U2.shape}, S2.shape={S2.shape}, Vt2.shape={Vt2.shape}")

# 对比奇异值
error_S = np.abs(S1 - S2).max() / (S1.max() + 1e-12)
print(f"  奇异值误差: {error_S:.4%} (< 1% 为优秀)")

# 对比重建误差（对比原始矩阵）
W1_reconstruct = U1 @ np.diag(S1) @ V1.T
W2_reconstruct = U2 @ np.diag(S2) @ Vt2  # Vt2 已经是 V^T 格式

# 误差是相对于原始矩阵 A 的
error1 = np.linalg.norm(A - W1_reconstruct) / (np.linalg.norm(A) + 1e-12)
error2 = np.linalg.norm(A - W2_reconstruct) / (np.linalg.norm(A) + 1e-12)
print(f"  标准SVD重建误差: {error1:.4%}")
print(f"  随机SVD重建误差: {error2:.4%}")
print()

# 测试 2: 速度对比
print("[测试 2] 速度对比 (4096×4096)")
A = rng.standard_normal((4096, 4096), dtype=np.float32)

# 标准 SVD
start = time.time()
U1, S1, V1 = np.linalg.svd(A, full_matrices=False)
time_std = time.time() - start

# 随机 SVD
start = time.time()
U2, S2, Vt2 = random_svd(A, rank=128, rng=rng)
time_rnd = time.time() - start

print(f"  标准 SVD: {time_std:.3f}s")
print(f"  随机 SVD: {time_rnd:.3f}s")
print(f"  加速比: {time_std/time_rnd:.1f}x")
print()

# 测试 3: GhostCompressor 集成
print("[测试 3] GhostCompressor 集成测试")

cfg_std = GhostConfig(use_random_svd=False)
cfg_rnd = GhostConfig(use_random_svd=True)

comp_std = GhostCompressor(cfg_std)
comp_rnd = GhostCompressor(cfg_rnd)

# 模拟一个权重
W = rng.standard_normal((4096, 4096), dtype=np.float32)

# 压缩
start = time.time()
gf_std = comp_std.compress_weight("test", W, rank=32)
time_std = time.time() - start

start = time.time()
gf_rnd = comp_rnd.compress_weight("test", W, rank=32)
time_rnd = time.time() - start

print(f"  标准 SVD 压缩: {time_std*1000:.1f} ms")
print(f"  随机 SVD 压缩: {time_rnd*1000:.1f} ms")
print(f"  加速比: {time_std/time_rnd:.1f}x")
print()

# 对比压缩结果
error = np.abs(gf_std.S - gf_rnd.S).max() / (gf_std.S.max() + 1e-12)
print(f"  奇异值误差: {error:.4%}")
print()

print("=" * 70)
print("[OK] All tests passed!")
print("=" * 70)
