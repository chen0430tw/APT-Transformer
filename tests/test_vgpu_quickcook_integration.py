#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_vgpu_quickcook_integration.py

测试 pretrain_quickcook.py 中虚拟显存 + 虚拟 Blackwell 的完整路径。
不依赖 HuggingFace 数据集，用随机 token tensor 代替。

覆盖路径:
  1. apply_virtual_blackwell_v64  — 替换 nn.Linear
  2. VirtualVRAMConfig + virtual_vram context — forward+backward 包裹
  3. Virtual A100 三层显存 (如果可用)

用法:
  python tests/test_vgpu_quickcook_integration.py
"""

import sys
import os
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn

PASS = []
FAIL = []

def check(name, fn):
    print(f"\n  [{name}]", end=" ", flush=True)
    try:
        result = fn()
        print(f"PASS  {result or ''}")
        PASS.append(name)
        return True
    except Exception as e:
        print(f"FAIL  {e}")
        traceback.print_exc()
        FAIL.append(name)
        return False


# ─────────────────────────────────────────────
# 公共资源
# ─────────────────────────────────────────────

D_MODEL   = 128
N_HEADS   = 4
N_LAYERS  = 2
SEQ_LEN   = 64
BATCH     = 2
VOCAB     = 870
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_model():
    from apt.trainops.scripts.pretrain_quickcook import create_model
    return create_model(
        arch="apt",
        vocab_size=VOCAB,
        d_model=D_MODEL,
        num_heads=N_HEADS,
        num_layers=N_LAYERS,
        max_seq_len=SEQ_LEN,
    ).to(DEVICE)


def random_batch():
    """返回 (input_ids, labels) 随机 token tensor"""
    x = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=DEVICE)
    y = torch.randint(0, VOCAB, (BATCH, SEQ_LEN), device=DEVICE)
    return x, y


def one_step(model, optimizer, x, y, loss_fn, use_vram_cfg=None):
    """一步 forward+backward，可选 virtual_vram 包裹"""
    from apt.vgpu.runtime.virtual_vram import virtual_vram

    optimizer.zero_grad()

    def _fwd_bwd():
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        loss = loss_fn(logits.view(-1, VOCAB), y.view(-1))
        loss.backward()
        return loss.item()

    if use_vram_cfg is not None:
        with virtual_vram(use_vram_cfg):
            loss_val = _fwd_bwd()
    else:
        loss_val = _fwd_bwd()

    optimizer.step()
    return loss_val


# ─────────────────────────────────────────────
# 测试 1: 基线 forward+backward（无 vGPU）
# ─────────────────────────────────────────────

def test_baseline():
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    x, y = random_batch()
    loss = one_step(model, optimizer, x, y, loss_fn)
    assert isinstance(loss, float) and loss > 0, f"loss 异常: {loss}"
    return f"loss={loss:.4f}"


# ─────────────────────────────────────────────
# 测试 2: Virtual Blackwell —— 替换 nn.Linear
# ─────────────────────────────────────────────

def test_virtual_blackwell_apply():
    from apt.vgpu.runtime.vb_integration import VBConfigV64, apply_virtual_blackwell_v64

    model = make_model()
    n_linear_before = sum(1 for m in model.modules() if isinstance(m, nn.Linear))

    vb_config = VBConfigV64(
        pulse_interval=5,
        use_fake_int8=False,
        gate_projected_mode=False,
    )
    model_vb, vb_adapter = apply_virtual_blackwell_v64(model, vb_config)
    replaced = getattr(model_vb, "_vb_replaced_linears", "?")
    return f"替换前 {n_linear_before} 个 Linear, 替换后标记 {replaced}"


# ─────────────────────────────────────────────
# 测试 3: Virtual Blackwell —— 能正常 forward+backward
# ─────────────────────────────────────────────

def test_virtual_blackwell_train():
    from apt.vgpu.runtime.vb_integration import VBConfigV64, apply_virtual_blackwell_v64

    model = make_model()
    vb_config = VBConfigV64(pulse_interval=3, use_fake_int8=False)
    model, _ = apply_virtual_blackwell_v64(model, vb_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    x, y = random_batch()

    losses = []
    for _ in range(5):
        losses.append(one_step(model, optimizer, x, y, loss_fn))

    assert all(isinstance(v, float) and v > 0 for v in losses)
    return f"5步 losses={[f'{v:.3f}' for v in losses]}"


# ─────────────────────────────────────────────
# 测试 4: Virtual VRAM —— context manager 不崩溃
# ─────────────────────────────────────────────

def test_virtual_vram_context():
    from apt.vgpu.runtime.virtual_vram import VirtualVRAMConfig, virtual_vram

    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    x, y = random_batch()

    vram_cfg = VirtualVRAMConfig(enabled=True, min_tensor_bytes=1 << 16, verbose=False)
    loss = one_step(model, optimizer, x, y, loss_fn, use_vram_cfg=vram_cfg)

    assert isinstance(loss, float) and loss > 0
    # CPU 上 pack_hook 不会真正 offload（t.is_cuda == False），但不应报错
    return f"loss={loss:.4f} (CPU模式: hook透传, 无CUDA offload)"


# ─────────────────────────────────────────────
# 测试 5: Virtual Blackwell + Virtual VRAM 同时启用
# ─────────────────────────────────────────────

def test_vb_plus_vram():
    from apt.vgpu.runtime.vb_integration import VBConfigV64, apply_virtual_blackwell_v64
    from apt.vgpu.runtime.virtual_vram import VirtualVRAMConfig, virtual_vram

    model = make_model()
    vb_config = VBConfigV64(pulse_interval=5)
    model, _ = apply_virtual_blackwell_v64(model, vb_config)

    vram_cfg = VirtualVRAMConfig(enabled=True, min_tensor_bytes=1 << 16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    x, y = random_batch()

    losses = []
    for _ in range(5):
        losses.append(one_step(model, optimizer, x, y, loss_fn, use_vram_cfg=vram_cfg))

    assert all(isinstance(v, float) and v > 0 for v in losses)
    return f"5步 losses={[f'{v:.3f}' for v in losses]}"


# ─────────────────────────────────────────────
# 测试 6: Virtual A100 三层显存 (可选)
# ─────────────────────────────────────────────

def test_virtual_a100():
    try:
        from apt.vgpu.runtime.virtual_a100 import VirtualVRAMBackend, VA100SignalCollector
    except ImportError as e:
        return f"跳过 (不可用: {e})"

    hot_bytes  = int(7.5e9 * 0.6)
    warm_bytes = int(32e9  * 0.3)
    cold_bytes = int(100e9)

    tier = VirtualVRAMBackend(hot_bytes, warm_bytes, cold_bytes)
    signal = VA100SignalCollector()

    # 写入并读回几个 tensor
    import uuid
    test_tensors = [torch.randn(256, 256) for _ in range(3)]
    keys = [str(uuid.uuid4()) for _ in test_tensors]

    for k, t in zip(keys, test_tensors):
        tier.put(k, t, priority=2)

    retrieved = [tier.get(k) for k in keys]
    assert all(r is not None for r in retrieved), "三层显存读取失败"

    stats = tier.stats
    return (f"hot={stats.hot.count}块, warm={stats.warm.count}块, "
            f"cold={stats.cold.count}块")


# ─────────────────────────────────────────────
# 测试 7: vb_pulse_interval 对 step 触发脉冲
# ─────────────────────────────────────────────

def test_vb_pulse_trigger():
    from apt.vgpu.runtime.vb_integration import VBConfigV64, apply_virtual_blackwell_v64

    PULSE_INTERVAL = 3
    model = make_model()
    vb_config = VBConfigV64(pulse_interval=PULSE_INTERVAL, use_fake_int8=False)
    model, vb_adapter = apply_virtual_blackwell_v64(model, vb_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    pulse_count_before = getattr(vb_adapter, "pulse_count", None)

    for step in range(PULSE_INTERVAL * 2 + 1):
        x, y = random_batch()
        one_step(model, optimizer, x, y, loss_fn)
        if vb_adapter is not None:
            try:
                vb_adapter.on_step(step)
            except AttributeError:
                pass  # adapter 不一定暴露 on_step

    pulse_count_after = getattr(vb_adapter, "pulse_count", None)
    return (f"pulse_interval={PULSE_INTERVAL}, "
            f"pulse_count: {pulse_count_before} → {pulse_count_after}")


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

TESTS = [
    ("基线 forward+backward",              test_baseline),
    ("Virtual Blackwell apply_v64",        test_virtual_blackwell_apply),
    ("Virtual Blackwell train 5步",        test_virtual_blackwell_train),
    ("Virtual VRAM context (CPU透传)",     test_virtual_vram_context),
    ("Virtual Blackwell + VRAM 联合",      test_vb_plus_vram),
    ("Virtual A100 三层显存",              test_virtual_a100),
    ("VB pulse_interval 触发",            test_vb_pulse_trigger),
]


def main():
    print("=" * 60)
    print("  vGPU quickcook 集成测试")
    print(f"  device={DEVICE}, d_model={D_MODEL}, layers={N_LAYERS}")
    print("=" * 60)

    t0 = time.perf_counter()
    for name, fn in TESTS:
        check(name, fn)

    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 60)
    print(f"  结果: {len(PASS)} 通过 / {len(FAIL)} 失败  ({elapsed:.1f}s)")
    if FAIL:
        print(f"  失败项: {FAIL}")
    print("=" * 60)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    sys.exit(main())
