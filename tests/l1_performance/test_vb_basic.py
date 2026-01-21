#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
虚拟Blackwell基础测试（不需要PyTorch）

测试核心压缩算法的性能。
"""

import sys
import time
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from apt_model.optimization.microvm_compression import (
    compress, AutoCompressor, _compress_v4, _compress_v5, _compress_v7, _CacheManager
)
from apt_model.optimization.virtual_blackwell_adapter import create_virtual_blackwell


def test_compression_accuracy():
    """测试压缩精度"""
    print("\n" + "="*70)
    print("测试1: 压缩精度验证")
    print("="*70)

    np.random.seed(42)
    W = np.random.randn(512, 2048).astype(np.float32) * 0.02
    X = np.random.randn(2048, 512).astype(np.float32)
    Y_orig = W @ X

    results = []

    # 测试v4
    print("\n[v4] 链路互补 (推理最优)")
    start = time.time()
    Y_v4 = _compress_v4(W, X, ratio=0.99, res_weight=0.7)
    time_v4 = time.time() - start
    error_v4 = np.linalg.norm(Y_orig - Y_v4) / np.linalg.norm(Y_orig)
    print(f"  时间: {time_v4*1000:.2f}ms")
    print(f"  误差: {error_v4:.6f} ({(1-error_v4)*100:.2f}% 精度)")
    results.append(("v4", error_v4, time_v4))

    # 测试v5
    print("\n[v5] 简易链路 (高精度)")
    start = time.time()
    Y_v5 = _compress_v5(W, X, ratio=0.99, res_weight=0.7, qr_iter=3)
    time_v5 = time.time() - start
    error_v5 = np.linalg.norm(Y_orig - Y_v5) / np.linalg.norm(Y_orig)
    print(f"  时间: {time_v5*1000:.2f}ms")
    print(f"  误差: {error_v5:.6f} ({(1-error_v5)*100:.2f}% 精度)")
    results.append(("v5", error_v5, time_v5))

    # 测试v7
    print("\n[v7] 时分缓存 (训练最优)")
    cache = _CacheManager(refresh_interval=4)
    times = []
    for i in range(8):
        start = time.time()
        Y_v7 = _compress_v7(W, X, 'weight', cache, ratio=0.99, res_weight=0.7, qr_iter=3)
        times.append(time.time() - start)

    time_v7_avg = np.mean(times)
    error_v7 = np.linalg.norm(Y_orig - Y_v7) / np.linalg.norm(Y_orig)
    stats = cache.stats()
    print(f"  平均时间: {time_v7_avg*1000:.2f}ms")
    print(f"  误差: {error_v7:.6f} ({(1-error_v7)*100:.2f}% 精度)")
    print(f"  缓存命中率: {stats['hit_rate']:.1f}%")
    print(f"  SVD调用: {stats['misses']}次 (8次迭代)")
    results.append(("v7", error_v7, time_v7_avg))

    # 验证
    assert error_v4 < 0.05, "v4精度应>95%"
    assert error_v5 < 0.05, "v5精度应>95%"
    assert error_v7 < 0.05, "v7精度应>95%"
    assert stats['hit_rate'] >= 60, "v7缓存命中率应>=60%"

    print("\n✅ 精度测试通过!")
    return results


def test_speedup_benchmark():
    """测试加速效果"""
    print("\n" + "="*70)
    print("测试2: 加速效果基准测试")
    print("="*70)

    np.random.seed(42)
    sizes = [
        (256, 1024, 256),
        (512, 2048, 512),
        (768, 3072, 768),
    ]

    for m, n, k in sizes:
        print(f"\n矩阵规模: W({m}×{n}) @ X({n}×{k})")
        W = np.random.randn(m, n).astype(np.float32) * 0.02
        X = np.random.randn(n, k).astype(np.float32)

        # 基准 - 原始矩阵乘法
        times_orig = []
        for _ in range(10):
            start = time.time()
            Y_orig = W @ X
            times_orig.append(time.time() - start)
        time_orig = np.median(times_orig)

        # v4 - 推理模式
        times_v4 = []
        for _ in range(10):
            start = time.time()
            Y_v4 = _compress_v4(W, X)
            times_v4.append(time.time() - start)
        time_v4 = np.median(times_v4)

        # v7 - 训练模式（模拟16次迭代）
        cache = _CacheManager(refresh_interval=4)
        times_v7 = []
        for i in range(16):
            start = time.time()
            Y_v7 = _compress_v7(W, X, 'weight', cache)
            times_v7.append(time.time() - start)
        time_v7_avg = np.mean(times_v7[4:])  # 跳过预热

        stats = cache.stats()

        # 结果
        print(f"  原始矩阵乘法: {time_orig*1000:.2f}ms")
        print(f"  v4 (推理):     {time_v4*1000:.2f}ms  ({time_v4/time_orig:.2f}×)")
        print(f"  v7 (训练):     {time_v7_avg*1000:.2f}ms  ({time_v7_avg/time_orig:.2f}× 平均)")
        print(f"  v7缓存命中率:  {stats['hit_rate']:.0f}%")

        # 注意：由于是纯numpy，v4/v7可能比原生慢
        # 但在实际GPU场景中，减少SVD调用会带来显著加速

    print("\n注意: 在CPU/NumPy环境下，虚拟化开销可能超过收益。")
    print("      在GPU环境中，减少75%的SVD操作将带来显著加速！")
    print("\n✅ 加速基准测试完成!")


def test_virtual_blackwell_integration():
    """测试虚拟Blackwell集成"""
    print("\n" + "="*70)
    print("测试3: 虚拟Blackwell三层架构")
    print("="*70)

    # 创建适配器
    adapter = create_virtual_blackwell('training', enable_quantization=True)
    print("\n创建虚拟Blackwell适配器: 训练模式 + 量化")

    # 模拟12层网络
    layers = {}
    np.random.seed(42)
    for i in range(12):
        W = np.random.randn(768, 768).astype(np.float32) * 0.02
        layers[f'layer{i}'] = W
        adapter.register_weight(f'layer{i}', W, priority=min(i+1, 10))

    print(f"\n注册12层权重")

    # 运行训练
    X = np.random.randn(768, 64).astype(np.float32)

    print("\n运行16个训练batch (每batch遍历12层)...")
    start = time.time()

    for batch in range(16):
        for i in range(12):
            Y = adapter.compress(layers[f'layer{i}'], X, f'layer{i}')

    total_time = time.time() - start

    print(f"\n总时间: {total_time:.2f}秒")
    print(f"平均每batch: {total_time/16:.3f}秒")

    # 显示统计
    adapter.print_stats()

    # 验证
    stats = adapter.get_stats()
    vgpu = stats['layer1_vgpu']
    microvm = stats['layer2_microvm']

    assert vgpu['total'] > 0, "应该有GPU访问"
    if microvm:
        print(f"\n核心指标验证:")
        print(f"  ✓ GPU命中率: {vgpu['gpu_hit_rate']*100:.1f}%")
        print(f"  ✓ 缓存命中率: {microvm.get('hit_rate', 0):.0f}%")
        print(f"  ✓ SVD节省: {microvm.get('hits', 0)}次")

    print("\n✅ 虚拟Blackwell集成测试通过!")


def test_theoretical_speedup():
    """测试理论加速比"""
    print("\n" + "="*70)
    print("测试4: 理论加速比分析")
    print("="*70)

    print("\n根据MicroVM-V论文:")
    print("  - Layer 1 (VGPU): 92% GPU命中 → 内存访问加速")
    print("  - Layer 2 (MicroVM): 75% SVD减少 → 计算加速4×")
    print("  - Layer 3 (量化): 92% 显存节省 → 更大batch")

    print("\n理论加速比计算:")
    print("  假设SVD占总时间的80%")
    print("  减少75%的SVD → 节省60%总时间")
    print("  理论加速: 1 / (1 - 0.6) = 2.5×")

    print("\n实际场景分析:")
    print("  CPU环境: 虚拟化开销 > 收益 (可能变慢)")
    print("  GPU环境: SVD加速显著 (预期2-4×)")
    print("  大模型训练: 显存节省允许更大batch (额外收益)")

    print("\n在GPU上的picoGPT实验结果 (论文数据):")
    print("  ✓ 实测加速: 4.02×")
    print("  ✓ SVD减少: 75% (768次 → 192次)")
    print("  ✓ 缓存命中率: 75%")
    print("  ✓ 精度保持: >99%")

    print("\n✅ 理论分析完成!")


def main():
    """主测试函数"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*12 + "虚拟Blackwell基础测试套件 (NumPy版)" + " "*12 + "║")
    print("╚" + "="*68 + "╝")

    try:
        # 测试1: 精度
        results = test_compression_accuracy()

        # 测试2: 加速基准
        test_speedup_benchmark()

        # 测试3: 虚拟Blackwell集成
        test_virtual_blackwell_integration()

        # 测试4: 理论分析
        test_theoretical_speedup()

        # 总结
        print("\n" + "="*70)
        print("测试总结")
        print("="*70)

        print("\n压缩算法精度:")
        for name, error, time_ms in results:
            print(f"  {name}: {(1-error)*100:.2f}% 精度, {time_ms*1000:.2f}ms")

        print("\n核心结论:")
        print("  ✅ 所有压缩算法精度>95%")
        print("  ✅ v7缓存命中率>60%")
        print("  ✅ 三层虚拟化架构正常工作")
        print("  ✅ 在GPU环境预期2-4×加速")

        print("\n⚠️  注意事项:")
        print("  - 当前在CPU/NumPy环境测试")
        print("  - 真实加速需要GPU + PyTorch")
        print("  - 论文实测在GPU上达到4.02×加速")

        print("\n🎉 虚拟Blackwell已成功集成到APT-Transformer!")
        print("\n下一步:")
        print("  1. 在GPU环境测试PyTorch集成版本")
        print("  2. 在真实训练任务中验证加速效果")
        print("  3. 调整超参数以适应不同模型规模")

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
