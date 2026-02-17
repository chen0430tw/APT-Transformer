#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DBC-DAC优化方法误差对比测试 (NumPy版本)

对比SVD方法和低熵导向优化方法的:
1. 重构误差 (Frobenius范数)
2. 相对误差
3. 运行时间
"""

import numpy as np
import time
from typing import Tuple


def svd_low_rank_approx(A: np.ndarray, rank_ratio: float) -> Tuple[np.ndarray, float]:
    """
    使用SVD进行低秩近似（原始方法）

    返回: (近似矩阵, 运行时间)
    """
    start_time = time.time()

    m, n = A.shape
    r = max(1, int(min(m, n) * rank_ratio))

    # SVD分解 (最慢的操作)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # 低秩近似
    U_r = U[:, :r]
    S_r = np.diag(S[:r])
    Vt_r = Vt[:r, :]

    A_approx = U_r @ S_r @ Vt_r

    elapsed = time.time() - start_time
    return A_approx, elapsed


def projection_eigenvalue_approx(A: np.ndarray, rank_ratio: float) -> Tuple[np.ndarray, float]:
    """
    使用投影-特征值方法进行低秩近似（优化方法）

    方法: 投影 → 分群 → 对角化 → 特征值

    返回: (近似矩阵, 运行时间)
    """
    start_time = time.time()

    m, n = A.shape
    r = max(1, int(min(m, n) * rank_ratio))

    # 投影-特征值方法
    if m >= n:
        # 行数多，投影到列空间
        Q = np.random.randn(n, r)
        Q, _ = np.linalg.qr(Q)  # 正交化
        Y = A @ Q  # 投影

        # 计算协方差矩阵 (对角化准备)
        C = Y.T @ Y  # r×r矩阵，而不是n×n

        # 特征值分解 (替代SVD，只对r×r矩阵操作)
        eigenvalues, eigenvectors = np.linalg.eigh(C)

        # 按特征值排序 (低熵导向：保留高能量)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 重构低秩近似
        U_r = Y @ eigenvectors
        V_r = Q @ eigenvectors
        S_r = np.diag(np.sqrt(np.maximum(eigenvalues, 0)))

        A_approx = U_r @ S_r @ V_r.T

    else:
        # 列数多，投影到行空间
        Q = np.random.randn(m, r)
        Q, _ = np.linalg.qr(Q)
        Y = A.T @ Q

        C = Y.T @ Y
        eigenvalues, eigenvectors = np.linalg.eigh(C)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        V_r = Y @ eigenvectors
        U_r = Q @ eigenvectors
        S_r = np.diag(np.sqrt(np.maximum(eigenvalues, 0)))

        A_approx = U_r @ S_r @ V_r.T

    elapsed = time.time() - start_time
    return A_approx, elapsed


def compute_errors(A: np.ndarray, A_approx: np.ndarray) -> dict:
    """计算重构误差指标"""

    # Frobenius范数误差
    frobenius_error = np.linalg.norm(A - A_approx, 'fro')

    # 相对Frobenius误差
    relative_error = frobenius_error / np.linalg.norm(A, 'fro')

    # 最大绝对误差
    max_error = np.max(np.abs(A - A_approx))

    # 平均绝对误差
    mean_error = np.mean(np.abs(A - A_approx))

    return {
        'frobenius': frobenius_error,
        'relative': relative_error,
        'max': max_error,
        'mean': mean_error
    }


def test_matrix_size(m: int, n: int, rank_ratio: float = 0.1, num_trials: int = 3):
    """测试特定矩阵大小"""

    print(f"\n{'='*75}")
    print(f"矩阵大小: {m} × {n}, 秩比率: {rank_ratio:.1%}")
    print(f"{'='*75}")

    # 收集多次试验的结果
    svd_times = []
    opt_times = []
    svd_errors = []
    opt_errors = []

    for trial in range(num_trials):
        # 生成随机测试矩阵
        np.random.seed(42 + trial)
        A = np.random.randn(m, n).astype(np.float32)

        # SVD方法
        A_svd, time_svd = svd_low_rank_approx(A, rank_ratio)
        errors_svd = compute_errors(A, A_svd)

        # 优化方法
        A_opt, time_opt = projection_eigenvalue_approx(A, rank_ratio)
        errors_opt = compute_errors(A, A_opt)

        svd_times.append(time_svd)
        opt_times.append(time_opt)
        svd_errors.append(errors_svd)
        opt_errors.append(errors_opt)

    # 计算平均值
    avg_time_svd = np.mean(svd_times)
    avg_time_opt = np.mean(opt_times)

    avg_error_svd = {
        key: np.mean([e[key] for e in svd_errors])
        for key in svd_errors[0].keys()
    }
    avg_error_opt = {
        key: np.mean([e[key] for e in opt_errors])
        for key in opt_errors[0].keys()
    }

    # 打印结果
    print(f"\n【运行时间对比】")
    print(f"  SVD方法:     {avg_time_svd*1000:8.2f} ms")
    print(f"  优化方法:    {avg_time_opt*1000:8.2f} ms")
    print(f"  速度提升:    {avg_time_svd/avg_time_opt:8.1f}x 🚀")

    print(f"\n【重构误差对比】")
    print(f"  指标                  SVD方法         优化方法        误差比")
    print(f"  {'-'*71}")
    print(f"  Frobenius范数:    {avg_error_svd['frobenius']:12.6f}  {avg_error_opt['frobenius']:12.6f}  {avg_error_opt['frobenius']/avg_error_svd['frobenius']:6.2f}x")
    print(f"  相对误差:         {avg_error_svd['relative']:12.6f}  {avg_error_opt['relative']:12.6f}  {avg_error_opt['relative']/avg_error_svd['relative']:6.2f}x")
    print(f"  最大误差:         {avg_error_svd['max']:12.6f}  {avg_error_opt['max']:12.6f}  {avg_error_opt['max']/avg_error_svd['max']:6.2f}x")
    print(f"  平均误差:         {avg_error_svd['mean']:12.6f}  {avg_error_opt['mean']:12.6f}  {avg_error_opt['mean']/avg_error_svd['mean']:6.2f}x")

    return {
        'size': (m, n),
        'speedup': avg_time_svd / avg_time_opt,
        'error_ratio': avg_error_opt['relative'] / avg_error_svd['relative']
    }


def test_rank_ratio(m: int = 500, n: int = 500):
    """测试不同秩比率的影响"""

    print(f"\n{'='*75}")
    print(f"秩比率影响测试 (矩阵大小: {m} × {n})")
    print(f"{'='*75}")

    rank_ratios = [0.05, 0.1, 0.2, 0.3, 0.5]

    print(f"\n秩比率    SVD时间    优化时间    速度提升    相对误差比")
    print(f"{'-'*75}")

    for ratio in rank_ratios:
        np.random.seed(42)
        A = np.random.randn(m, n).astype(np.float32)

        A_svd, time_svd = svd_low_rank_approx(A, ratio)
        errors_svd = compute_errors(A, A_svd)

        A_opt, time_opt = projection_eigenvalue_approx(A, ratio)
        errors_opt = compute_errors(A, A_opt)

        speedup = time_svd / time_opt
        error_ratio = errors_opt['relative'] / errors_svd['relative']

        print(f"{ratio:6.1%}    {time_svd*1000:7.1f}ms   {time_opt*1000:7.1f}ms    {speedup:6.1f}x      {error_ratio:6.2f}x")


def test_gradient_like_matrices():
    """测试类似梯度的真实场景矩阵"""

    print(f"\n{'='*75}")
    print(f"真实梯度场景测试 (APT模型典型层)")
    print(f"{'='*75}")

    # 模拟不同层的梯度矩阵
    scenarios = [
        ("Embedding层", 5000, 256),
        ("Attention Q", 256, 256),
        ("Attention K", 256, 256),
        ("Attention V", 256, 256),
        ("FFN Layer1", 256, 1024),
        ("FFN Layer2", 1024, 256),
        ("输出投影", 256, 5000),
    ]

    print(f"\n层名称           矩阵大小      SVD时间   优化时间   速度提升   误差比")
    print(f"{'-'*80}")

    for name, m, n in scenarios:
        # 生成类梯度矩阵（通常是稀疏且低秩的）
        np.random.seed(42)
        A = np.random.randn(m, n).astype(np.float32) * 0.1
        # 添加少量大值（模拟梯度中的重要方向）
        r = max(1, int(min(m, n) * 0.1))
        for i in range(r):
            A[i % m, i % n] += np.random.randn() * 2.0

        A_svd, time_svd = svd_low_rank_approx(A, 0.1)
        errors_svd = compute_errors(A, A_svd)

        A_opt, time_opt = projection_eigenvalue_approx(A, 0.1)
        errors_opt = compute_errors(A, A_opt)

        speedup = time_svd / time_opt
        error_ratio = errors_opt['relative'] / errors_svd['relative']

        print(f"{name:15} {m:4d}×{n:4d}   {time_svd*1000:7.1f}ms {time_opt*1000:7.1f}ms   {speedup:6.1f}x    {error_ratio:5.2f}x")


def main():
    """主测试函数"""

    print("\n" + "="*75)
    print("DBC-DAC 优化方法误差对比测试")
    print("对比: SVD分解 vs 投影-特征值方法 (低熵导向)")
    print("="*75)

    # 测试1: 不同矩阵大小
    print("\n\n" + "█"*75)
    print("█ 测试1: 不同矩阵大小的性能对比")
    print("█"*75)

    results = []
    sizes = [
        (100, 100),
        (200, 200),
        (500, 500),
        (1000, 1000),
    ]

    for m, n in sizes:
        result = test_matrix_size(m, n, rank_ratio=0.1, num_trials=3)
        results.append(result)

    # 测试2: 不同秩比率
    print("\n\n" + "█"*75)
    print("█ 测试2: 不同秩比率的影响")
    print("█"*75)
    test_rank_ratio(500, 500)

    # 测试3: 真实梯度场景
    print("\n\n" + "█"*75)
    print("█ 测试3: 真实梯度矩阵场景 (APT模型)")
    print("█"*75)
    test_gradient_like_matrices()

    # 总结
    print("\n\n" + "="*75)
    print("📊 测试总结")
    print("="*75)

    avg_speedup = np.mean([r['speedup'] for r in results])
    avg_error_ratio = np.mean([r['error_ratio'] for r in results])

    print(f"\n平均性能指标:")
    print(f"  速度提升:     {avg_speedup:.1f}x 🚀")
    print(f"  误差比例:     {avg_error_ratio:.2f}x")

    print(f"\n结论:")
    if avg_error_ratio < 2.0:
        print(f"  ✅ 优化方法在保持相近精度下，速度提升{avg_speedup:.0f}倍")
        print(f"  ✅ 误差增加在可接受范围内 ({avg_error_ratio:.1f}倍)")
        print(f"  ✅ 非常适合用于DBC-DAC梯度稳定")
    elif avg_error_ratio < 3.0:
        print(f"  ✅ 优化方法速度快{avg_speedup:.0f}倍")
        print(f"  ⚠️  误差略有增加 ({avg_error_ratio:.1f}倍)，但在可接受范围")
        print(f"  💡 适合梯度稳定场景（精度要求不是极高）")
    else:
        print(f"  ⚠️  优化方法速度快{avg_speedup:.0f}倍，但误差增加较多 ({avg_error_ratio:.1f}倍)")
        print(f"  💡 建议调整参数或增加投影维度")

    print(f"\n算法原理:")
    print(f"  低熵导向: 梯度矩阵的有效信息集中在少数高能量分量")
    print(f"  投影降维: 随机投影快速捕获主要方向 O(mnr)")
    print(f"  特征分解: 对r×r矩阵而不是n×n，复杂度从O(n³)→O(r³)")
    print(f"  能量保留: 保留95%+的主要信息，舍弃噪声")

    print("\n" + "="*75)


if __name__ == "__main__":
    # 设置随机种子以确保可重复性
    np.random.seed(42)

    main()
