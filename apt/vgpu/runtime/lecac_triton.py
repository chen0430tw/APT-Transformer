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

@triton.jit
def _compensation_kernel(
    inout_ptr,    # float32 in-place [N]，输出覆盖输入
    scale,        # float32 标量（补偿幅度 error_std * alpha / dim_balance）
    seed,         # int32 随机种子（每次调用传入不同值，保证 noise 独立）
    N,
    BLOCK: tl.constexpr,
):
    """
    in-place 融合补偿：out[i] = in[i] + tl.randn(seed, i) * scale

    使用 Triton 内置 Philox CBRNG（tl.randn），相比 torch.randn_like：
      * 无独立 randn kernel 启动——融合进同一 load→compute→store pass
      * 无额外中间张量分配（3.1M float32 = 12MB HBM 写入省去）
      * Philox 在 GPU 上并行生成，延迟远低于 PyTorch 调度开销
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    x = tl.load(inout_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    # tl.randn(seed, offsets)：Philox4 CBRNG，(seed, offset) 唯一确定 N(0,1) 样本
    noise = tl.randn(seed, offsets.to(tl.int32))
    out = x + noise * scale
    tl.store(inout_ptr + offsets, out, mask=mask)


def triton_compensation(x_recon: torch.Tensor, scale_factor) -> torch.Tensor:
    """
    Triton 补偿内核：替换 _compensation_fn 中的 torch.randn_like。

    In-place 修改 x_recon（float32 连续张量），返回同形状张量。
    每次调用使用 CPU 随机种子，保证跨 step / 层的 noise 不相关。

    Args:
        x_recon      : float32 激活重构值 [M, K]（backward 中的 dequant 输出）
        scale_factor : 补偿幅度（error_std * alpha / dimension_balance），
                       可为 0-d GPU tensor 或 Python float

    Returns:
        补偿后的张量（与 x_recon 共享内存，形状不变）
    """
    # 单标量 D2H 同步（微秒级，相比节省的 ~2900ms 可忽略）
    if isinstance(scale_factor, torch.Tensor):
        sf = scale_factor.item()
    else:
        sf = float(scale_factor)

    N = x_recon.numel()
    # 连续 float32 视图（dequantize 通常已满足，view 为 O(1)）
    if x_recon.is_contiguous() and x_recon.dtype == torch.float32:
        x_flat = x_recon.view(-1)
    else:
        x_flat = x_recon.contiguous().float().view(-1)

    # CPU 随机种子，确保不同 step / 层生成不相关 noise
    seed = int(torch.randint(0, 0x7FFF_FFFF, (1,)).item())

    grid = (triton.cdiv(N, 1024),)
    _compensation_kernel[grid](x_flat, sf, seed, N, BLOCK=1024)

    return x_flat.view(x_recon.shape)
