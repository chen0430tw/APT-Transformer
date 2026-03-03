# lecac_triton.py
# -*- coding: utf-8 -*-
"""
LECAC Triton 加速内核
====================
替换 lecac.py 中基于 torch.compile 的 _quant_with_std_int2/4，
使用两个 Triton 内核实现更低的 HBM 访问次数和更少的 kernel 启动：

    核 1 (_amax_kernel)     : 单 pass 规约 max(|x|) → scale
    核 2 (_quant_std_kernel): 量化 x→x_q（INT8 存储）+ 单 pass Welford 误差规约

相比 torch.compile mode="default" 的优势：
  * x 从 HBM 读取次数：2 次（核1一次、核2一次）
    torch.compile 版本：max + quant + (dequant+error+std两趟) ≈ 4-5 次
  * 消除 .std() 的两趟规约串行等待：改为单 pass E[e] + E[e²] 直接求方差
  * atomic_max / atomic_add 对 H100 L2 cache 友好（所有 block 写同一 cache line）

回退链：
    Triton 内核 → torch.compile（见 lecac.py）→ 纯 Python
"""
from __future__ import annotations

import random
import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice


# ============================================================================
# 核 1：abs max 规约
# ============================================================================

@triton.jit
def _amax_kernel(
    x_ptr,
    out_ptr,       # float32[1]，调用前须 zero_()
    N,
    BLOCK: tl.constexpr,
):
    """
    单 pass 规约 max(|x|)。
    每个 block 计算本地 abs-max，然后通过 atomic_max 写入 out_ptr[0]。
    由于 |x| >= 0，IEEE754 正浮点与 uint32 的大小顺序一致，atomic_max 语义正确。
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    block_max = tl.max(tl.abs(x), axis=0)
    tl.atomic_max(out_ptr, block_max)


# ============================================================================
# 核 2：量化 + 单 pass 误差规约
# ============================================================================

@triton.jit
def _quant_std_kernel(
    x_ptr,          # float32 或 bf16 输入 [N]（已展平）
    xq_ptr,         # int8 输出 [N]
    scale_ptr,      # float32[1]，= clamp(amax / SCALE_DIV, 1e-6)
    sum_e_ptr,      # float32[1] 输出，调用前须 zero_()
    sq_sum_e_ptr,   # float32[1] 输出，调用前须 zero_()
    N,
    CLIP_LO: tl.constexpr,   # INT2: -2, INT4: -8
    CLIP_HI: tl.constexpr,   # INT2:  1, INT4:  7
    BLOCK: tl.constexpr,
):
    """
    量化 x → x_q（INT8 存储），并单 pass 收集误差统计。

    方差通过 E[e²] - E[e]² 公式计算（单 pass），避免 .std() 的两趟规约。
    舍入使用 libdevice.round（半数远离零，与 C round() 一致）。
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # 加载并转 float32（兼容 bf16 输入）
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr)
    inv_scale = 1.0 / scale

    # 量化：round-half-away-from-zero（libdevice.round = C round()）
    xq_f = libdevice.round(x * inv_scale)
    xq_f = tl.clamp(xq_f, CLIP_LO, CLIP_HI)

    # 写 int8
    tl.store(xq_ptr + offsets, xq_f.to(tl.int8), mask=mask)

    # 误差 e = x - x_q * scale（mask 外补零，不影响规约）
    err = tl.where(mask, x - xq_f * scale, 0.0)

    # 块内规约 → atomic_add 到全局累积器
    tl.atomic_add(sum_e_ptr,    tl.sum(err,       axis=0))
    tl.atomic_add(sq_sum_e_ptr, tl.sum(err * err, axis=0))


# ============================================================================
# Python 封装
# ============================================================================

def _triton_quant_with_std(
    x: torch.Tensor,
    clip_lo: int,
    clip_hi: int,
    scale_div: float,
    block: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    两核融合量化内核（替换 torch.compile 版本）。

    Pass 1 (核 1): abs max 规约 → amax → scale = clamp(amax / scale_div, 1e-6)
    Pass 2 (核 2): 量化 x → x_q（int8）+ 单 pass 收集 sum_e, sum_sq_e

    全程在 GPU 上完成，无 D2H 同步（error_std 作为 0-d GPU tensor 返回）。

    Args:
        x        : 输入激活值，任意形状（会在内部展平处理）
        clip_lo  : 量化下限（INT2: -2，INT4: -8）
        clip_hi  : 量化上限（INT2:  1，INT4:  7）
        scale_div: 缩放因子除数（INT2: 1.0，INT4: 7.0）
        block    : Triton BLOCK 大小（默认 1024）

    Returns:
        x_q      : int8 tensor，与 x 同形状
        scale    : 0-d float32 GPU tensor
        error_std: 0-d float32 GPU tensor
    """
    N = x.numel()
    device = x.device

    # 展平并转 float32（kernel 内也会转，但提前做避免反复转换）
    x_flat = x.contiguous().view(-1)
    if x_flat.dtype != torch.float32:
        x_flat = x_flat.float()

    grid = (triton.cdiv(N, block),)

    # ── 中间缓冲区（GPU，无 D2H 同步）──
    amax_buf   = x_flat.new_zeros(1)         # float32[1]
    sum_e      = x_flat.new_zeros(1)         # float32[1]
    sq_sum_e   = x_flat.new_zeros(1)         # float32[1]
    x_q_flat   = torch.empty(N, dtype=torch.int8, device=device)

    # Pass 1：abs max 规约
    _amax_kernel[grid](x_flat, amax_buf, N, BLOCK=block)

    # scale（GPU 上计算，无 D2H）
    scale_buf = torch.clamp(amax_buf / scale_div, min=1e-6)

    # Pass 2：量化 + 误差规约
    _quant_std_kernel[grid](
        x_flat, x_q_flat, scale_buf,
        sum_e, sq_sum_e, N,
        CLIP_LO=clip_lo, CLIP_HI=clip_hi, BLOCK=block,
    )

    # 计算 std（E[e²] - E[e]², GPU 上完成）
    n_f = float(N)
    mean_e    = sum_e    / n_f
    mean_sq_e = sq_sum_e / n_f
    var_e     = (mean_sq_e - mean_e * mean_e).clamp_(min=0.0)
    error_std = var_e.sqrt_().squeeze(0)   # 0-d float32 tensor

    x_q   = x_q_flat.view(x.shape)
    scale = scale_buf.squeeze(0)           # 0-d float32 tensor

    return x_q, scale, error_std


def triton_quant_with_std_int2(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """INT2 Triton 量化 + error_std（替换 _quant_with_std_int2）。"""
    return _triton_quant_with_std(x, clip_lo=-2, clip_hi=1, scale_div=1.0)


def triton_quant_with_std_int4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """INT4 Triton 量化 + error_std（替换 _quant_with_std_int4）。"""
    return _triton_quant_with_std(x, clip_lo=-8, clip_hi=7, scale_div=7.0)


# ============================================================================
# 核 3：backward 补偿（randn + scale + add，in-place 融合）
# ============================================================================

@triton.autotune(
    configs=[
        # H100 Box-Muller 是计算瓶颈（log/cos 约 150 clocks/elem），
        # 更多 warps 可掩盖超越函数延迟；小 BLOCK 提升 SM 并发 block 数。
        triton.Config({"BLOCK": 256},  num_warps=4),
        triton.Config({"BLOCK": 512},  num_warps=8),
        triton.Config({"BLOCK": 512},  num_warps=16),
        triton.Config({"BLOCK": 1024}, num_warps=16),
        triton.Config({"BLOCK": 2048}, num_warps=16),
        triton.Config({"BLOCK": 2048}, num_warps=32),
    ],
    key=["N"],   # N 相同时复用缓存的最优配置，训练中每层形状固定 → 只调优一次
)
@triton.jit
def _compensation_kernel(
    inout_ptr,    # float32 in-place [N]，输出覆盖输入
    scale_ptr,    # float32[1] GPU 指针（避免 .item() D2H 同步破坏 pipeline）
    seed,         # int32 随机种子（每次调用传入不同值，保证 noise 独立）
    N,
    BLOCK: tl.constexpr,
):
    """
    in-place 融合补偿：out[i] = in[i] + N(0,1) * scale

    RNG 优化（randint4x 成对 Box-Muller）：
      tl.randn(seed, i) 内部调用 randint4x(seed, i) 得到 (i1,i2,i3,i4)，
      Box-Muller 只用 (i1,i2)，i3,i4 直接丢弃——50% Philox4 算力浪费。

      本内核让相邻两元素共享同一次 randint4x(seed, k)：
        偶数位置 2k   使用 (i1, i2) → Box-Muller → N(0,1)
        奇数位置 2k+1 使用 (i3, i4) → Box-Muller → N(0,1)
      Philox4 调用次数从 N 降至 N/2（-50%），Box-Muller 次数不变。

    autotune 自动选择 BLOCK / num_warps：
      Box-Muller（log + sqrt + cos）约 150 clocks/element，计算瓶颈而非内存。
      更多 warps 可掩盖超越函数延迟，autotune 找出该激活形状的最优配置。
    """
    pid     = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask    = offsets < N

    x     = tl.load(inout_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr)                         # GPU 内读取，无 D2H

    # randint4x 成对 Box-Muller
    rng_off = (offsets >> 1).to(tl.int32)             # 相邻两元素 rng_off 相同
    i1, i2, i3, i4 = tl.randint4x(seed, rng_off)

    _C      = 1.0 / 4294967296.0                       # int32 → float [0,1) 缩放系数
    is_even = ((offsets & 1) == 0)
    u_a     = tl.where(is_even, i1, i3).to(tl.float32) * _C + 0.5   # ∈ (0, 1)
    u_b     = tl.where(is_even, i2, i4).to(tl.float32) * _C + 0.5   # ∈ (0, 1)

    # Box-Muller 变换：(u_a, u_b) → N(0, 1)
    noise = tl.sqrt(-2.0 * tl.log(u_a)) * tl.cos(6.283185307179586 * u_b)

    tl.store(inout_ptr + offsets, x + noise * scale, mask=mask)


def triton_compensation(x_recon: torch.Tensor, scale_factor) -> torch.Tensor:
    """
    Triton 补偿内核：替换 _compensation_fn 中的 torch.randn_like。

    In-place 修改 x_recon（float32 连续张量），返回同形状张量。
    全程无 D2H 同步，GPU pipeline 不中断。

    Args:
        x_recon      : float32 激活重构值 [M, K]（backward 中的 dequant 输出）
        scale_factor : 补偿幅度（error_std * alpha / dimension_balance），
                       可为 0-d GPU tensor 或 Python float

    Returns:
        补偿后的张量（与 x_recon 共享内存，形状不变）

    D2H 同步消除说明：
        旧实现 scale_factor.item() 会强制 GPU drain 所有 pending ops 后才返回，
        在 backward 中每层都触发一次，共 num_layers 次 pipeline 停顿。
        新实现将 scale 保留在 GPU 上以 scale_ptr 传入 kernel，
        dequant → compensation 两个 kernel 完全异步流水。
    """
    N = x_recon.numel()

    # scale 保持在 GPU 上，0-d tensor 需 view(1) 获得合法指针地址
    if isinstance(scale_factor, torch.Tensor):
        scale_buf = scale_factor.float().view(1)          # O(1)，共享存储，无 D2H
    else:
        scale_buf = torch.tensor(
            [float(scale_factor)], dtype=torch.float32, device=x_recon.device
        )

    # 连续 float32 视图（dequantize 通常已满足，view 为 O(1)）
    if x_recon.is_contiguous() and x_recon.dtype == torch.float32:
        x_flat = x_recon.view(-1)
    else:
        x_flat = x_recon.contiguous().float().view(-1)

    # Python random 生成种子：无 PyTorch tensor 分配，无 D2H，纯 CPU
    seed = random.randint(0, 0x7FFF_FFFF)

    # autotune 自动决定最优 BLOCK；grid 通过 lambda 动态获取 meta["BLOCK"]
    # 首次调用会自动基准测试所有配置（结果缓存至 ~/.triton/cache，后续复用）
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK"]),)
    _compensation_kernel[grid](x_flat, scale_buf, seed, N)

    return x_flat.view(x_recon.shape)
