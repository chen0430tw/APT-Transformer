"""
torch.compile 综合测试
=====================
整合所有 torch.compile 相关测试到单一入口

测试模式：
  quick  - 超快速测试（5 batches × 4个配置：baseline/VB × 编译/未编译）
  small - 使用更小模型（2层）快速验证编译可用性
  step  - 分步测试，避免一次编译太多模型（更稳定）

使用示例：
  python test_compile_smoke.py --mode quick
  python test_compile_smoke.py --mode small
  python test_compile_smoke.py --mode step
"""

import argparse
import time
import torch
from apt.model.architectures.claude4_model import create_claude4_model
from apt.vgpu.runtime.vb_integration import apply_virtual_blackwell_v64, VBConfigV64


def run_quick_test(device):
    """
    Mode: quick - 超快速测试（5 batches × 4个配置）
    原文件：test_compile_quick.py
    """
    print("\n" + "=" * 70)
    print("Mode: QUICK - 超快速测试（5 batches × 4 个配置）")
    print("=" * 70)

    # 预创建数据
    x = torch.randint(0, 4096, (4, 1024), device=device)

    results = {}

    # Test 1: Baseline 未编译
    print("\n[1/4] Baseline 未编译...")
    model1 = create_claude4_model(vocab_size=4096, d_model=768, num_layers=6, num_heads=12, ffn_hidden=3072).to(device)
    model1.train()
    for _ in range(2):  # warmup
        y = model1(x)
        y.mean().backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        y = model1(x)
        y.mean().backward()
    torch.cuda.synchronize()
    results['baseline_no_compile'] = time.time() - t0
    print(f"  Time: {results['baseline_no_compile']:.2f}s")

    # Test 2: VB Gate 未编译
    print("\n[2/4] VB Gate 未编译...")
    model2 = create_claude4_model(vocab_size=4096, d_model=768, num_layers=6, num_heads=12, ffn_hidden=3072).to(device)
    cfg = VBConfigV64(pulse_interval=100, gate_projected_mode=True, enable_stats=False)
    model2, _ = apply_virtual_blackwell_v64(model2, cfg)
    model2.train()
    for _ in range(2):  # warmup
        y = model2(x)
        y.mean().backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        y = model2(x)
        y.mean().backward()
    torch.cuda.synchronize()
    results['vb_gate_no_compile'] = time.time() - t0
    print(f"  Time: {results['vb_gate_no_compile']:.2f}s")

    # Test 3: Baseline 编译
    print("\n[3/4] Baseline 编译（首次编译需要 10-30 秒）...")
    model3 = create_claude4_model(vocab_size=4096, d_model=768, num_layers=6, num_heads=12, ffn_hidden=3072).to(device)
    model3 = torch.compile(model3, mode="reduce-overhead")
    model3.train()
    print("  编译完成，开始测试...")
    for _ in range(3):  # warmup
        y = model3(x)
        y.mean().backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        y = model3(x)
        y.mean().backward()
    torch.cuda.synchronize()
    results['baseline_compile'] = time.time() - t0
    print(f"  Time: {results['baseline_compile']:.2f}s")

    # Test 4: VB Gate 编译
    print("\n[4/4] VB Gate 编译（首次编译需要 10-30 秒）...")
    model4 = create_claude4_model(vocab_size=4096, d_model=768, num_layers=6, num_heads=12, ffn_hidden=3072).to(device)
    model4, _ = apply_virtual_blackwell_v64(model4, cfg)
    model4 = torch.compile(model4, mode="reduce-overhead")
    model4.train()
    print("  编译完成，开始测试...")
    for _ in range(3):  # warmup
        y = model4(x)
        y.mean().backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        y = model4(x)
        y.mean().backward()
    torch.cuda.synchronize()
    results['vb_gate_compile'] = time.time() - t0
    print(f"  Time: {results['vb_gate_compile']:.2f}s")

    # 总结
    print("\n" + "=" * 70)
    print("SUMMARY - torch.compile 效果对比")
    print("=" * 70)
    print(f"{'配置':<25} {'时间(s)':>10} {'vs Base':>12}")
    print("-" * 70)
    base = results['baseline_no_compile']
    print(f"{'Baseline (未编译)':<25} {base:>10.2f} {0:>12.1f}%")
    print(f"{'Baseline (编译)':<25} {results['baseline_compile']:>10.2f} {((results['baseline_compile']/base)-1)*100:>12.1f}%")
    print(f"{'VB Gate (未编译)':<25} {results['vb_gate_no_compile']:>10.2f} {((results['vb_gate_no_compile']/base)-1)*100:>12.1f}%")
    print(f"{'VB Gate (编译)':<25} {results['vb_gate_compile']:>10.2f} {((results['vb_gate_compile']/base)-1)*100:>12.1f}%")

    print("\n" + "=" * 70)
    print("KEY INSIGHT - 编译是否解决了问题？")
    print("=" * 70)
    no_compile_gap = ((results['vb_gate_no_compile'] / base) - 1) * 100
    compile_gap = ((results['vb_gate_compile'] / base) - 1) * 100
    print(f"VB Gate 未编译 vs Baseline: {no_compile_gap:+.1f}%")
    print(f"VB Gate 编译后 vs Baseline:   {compile_gap:+.1f}%")
    print(f"编译带来的改善: {no_compile_gap - compile_gap:+.1f}%")

    if compile_gap < 1:
        print("\n✅ torch.compile 成功！VB Gate 接近 baseline")
    else:
        print(f"\n⚠️  编译后仍有 {compile_gap:.1f}% 开销")
        print("   说明：Python 不是唯一瓶颈，可能还有其他因素")


def run_small_test(device):
    """
    Mode: small - 使用更小模型快速验证编译可用性
    原文件：test_compile_small.py
    """
    print("\n" + "=" * 70)
    print("Mode: SMALL - 使用小模型快速验证")
    print("=" * 70)

    # 创建更小的模型 - 只 2 层
    print("创建小模型 (2 layers)...")
    model_small = create_claude4_model(
        vocab_size=4096, d_model=512, num_layers=2,  # 更小的模型
        num_heads=8, ffn_hidden=2048
    ).to(device)

    x = torch.randint(0, 4096, (2, 512), device=device)

    # 测试 1: 未编译
    print("\n[Test 1] 未编译 (5 batches)")
    model_small.train()
    for _ in range(2):
        y = model_small(x)
        y.mean().backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        y = model_small(x)
        y.mean().backward()
    torch.cuda.synchronize()
    time_no_compile = time.time() - t0
    print(f"Time: {time_no_compile:.2f}s")

    # 测试 2: 编译
    print("\n[Test 2] torch.compile (编译中，请等待 30-60 秒)...")
    try:
        model_compiled = torch.compile(model_small, mode="reduce-overhead")
        model_compiled.train()

        # 预热编译
        print("  预热中...")
        for _ in range(2):
            y = model_compiled(x)
            y.mean().backward()

        torch.cuda.synchronize()

        # 计时
        print("  计时中...")
        t0 = time.time()
        for _ in range(5):
            y = model_compiled(x)
            y.mean().backward()
        torch.cuda.synchronize()
        time_compile = time.time() - t0
        print(f"Time: {time_compile:.2f}s")

        # 结果
        print(f"\n{'='*60}")
        print(f"RESULTS:")
        print(f"{'='*60}")
        print(f"未编译: {time_no_compile:.2f}s")
        print(f"编译后: {time_compile:.2f}s")
        overhead = ((time_compile / time_no_compile) - 1) * 100
        print(f"开销: {overhead:+.1f}%")

        if overhead < 0:
            print(f"\n[SUCCESS] torch.compile 成功！快 {abs(overhead):.1f}%")
        elif overhead < 10:
            print(f"\n[OK] torch.compile 可用，有 {overhead:.1f}% 开销")
        else:
            print(f"\n[WARNING] torch.compile 有 {overhead:.1f}% 开销")

    except Exception as e:
        print(f"\n[FAILED] torch.compile 失败:")
        print(f"   {type(e).__name__}")
        print(f"   {str(e)[:300]}")
        print(f"\n可能原因:")
        print(f"  1. PyTorch 版本不兼容 (当前 {torch.__version__})")
        print(f"  2. Python 3.13 支持不完整")
        print(f"  3. triton-windows 与 PyTorch 版本不匹配")


def run_step_test(device):
    """
    Mode: step - 分步测试，避免一次编译太多模型（更稳定）
    原文件：test_compile_step.py
    """
    print("\n" + "=" * 70)
    print("Mode: STEP - 分步测试（更稳定）")
    print("=" * 70)

    x = torch.randint(0, 4096, (4, 1024), device=device)

    print("=" * 60)
    print("Step 1: Baseline 未编译")
    print("=" * 60)
    model1 = create_claude4_model(vocab_size=4096, d_model=768, num_layers=6, num_heads=12, ffn_hidden=3072).to(device)
    model1.train()
    for _ in range(2):
        y = model1(x)
        y.mean().backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        y = model1(x)
        y.mean().backward()
    torch.cuda.synchronize()
    baseline_no_compile = time.time() - t0
    print(f"Time: {baseline_no_compile:.2f}s\n")

    print("=" * 60)
    print("Step 2: VB Gate 未编译")
    print("=" * 60)
    model2 = create_claude4_model(vocab_size=4096, d_model=768, num_layers=6, num_heads=12, ffn_hidden=3072).to(device)
    cfg = VBConfigV64(pulse_interval=100, gate_projected_mode=True, enable_stats=False)
    model2, _ = apply_virtual_blackwell_v64(model2, cfg)
    print(f"Replaced {model2._vb_replaced_linears} layers")
    model2.train()
    for _ in range(2):
        y = model2(x)
        y.mean().backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        y = model2(x)
        y.mean().backward()
    torch.cuda.synchronize()
    vb_gate_no_compile = time.time() - t0
    print(f"Time: {vb_gate_no_compile:.2f}s\n")

    print("=" * 60)
    print("Step 3: 只测试编译版 Baseline（需要等待编译）")
    print("=" * 60)
    print("编译中，请耐心等待 10-30 秒...")
    model3 = create_claude4_model(vocab_size=4096, d_model=768, num_layers=6, num_heads=12, ffn_hidden=3072).to(device)
    model3 = torch.compile(model3, mode="reduce-overhead")
    model3.train()
    for _ in range(3):
        y = model3(x)
        y.mean().backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        y = model3(x)
        y.mean().backward()
    torch.cuda.synchronize()
    baseline_compile = time.time() - t0
    print(f"Time: {baseline_compile:.2f}s\n")

    print("=" * 60)
    print("Step 4: 只测试编译版 VB Gate（需要等待编译）")
    print("=" * 60)
    print("编译中，请耐心等待 10-30 秒...")
    model4 = create_claude4_model(vocab_size=4096, d_model=768, num_layers=6, num_heads=12, ffn_hidden=3072).to(device)
    model4, _ = apply_virtual_blackwell_v64(model4, cfg)
    model4 = torch.compile(model4, mode="reduce-overhead")
    model4.train()
    for _ in range(3):
        y = model4(x)
        y.mean().backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        y = model4(x)
        y.mean().backward()
    torch.cuda.synchronize()
    vb_gate_compile = time.time() - t0
    print(f"Time: {vb_gate_compile:.2f}s\n")

    # 结果
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Baseline (未编译): {baseline_no_compile:.2f}s")
    print(f"VB Gate (未编译):  {vb_gate_no_compile:.2f}s  ({((vb_gate_no_compile/baseline_no_compile)-1)*100:+.1f}%)")
    print(f"Baseline (编译):   {baseline_compile:.2f}s  ({((baseline_compile/baseline_no_compile)-1)*100:+.1f}%)")
    print(f"VB Gate (编译):    {vb_gate_compile:.2f}s  ({((vb_gate_compile/baseline_no_compile)-1)*100:+.1f}%)")

    gap_no_compile = ((vb_gate_no_compile/baseline_no_compile) - 1) * 100
    gap_compile = ((vb_gate_compile/baseline_no_compile) - 1) * 100
    print(f"\n关键指标:")
    print(f"  VB Gate 未编译 vs Baseline: {gap_no_compile:+.1f}%")
    print(f"  VB Gate 编译后 vs Baseline:   {gap_compile:+.1f}%")
    print(f"  编译带来的改善:              {gap_no_compile - gap_compile:+.1f}%")

    if gap_compile < 1:
        print("\n✅ torch.compile 成功！VB Gate 接近或超过 baseline")
    elif gap_compile < gap_no_compile:
        print(f"\n⚠️  编译有帮助，但仍落后 {gap_compile:.1f}%")
    else:
        print(f"\n❌ 编译没有改善性能")


def main():
    parser = argparse.ArgumentParser(
        description="torch.compile 综合测试",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--mode", type=str,
                        choices=["quick", "small", "step"],
                        default="quick",
                        help="测试模式（默认: quick）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备（默认: cuda）")

    args = parser.parse_args()

    device = args.device
    print("=" * 70)
    print("torch.compile 综合测试")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Mode: {args.mode}")

    # 根据模式运行测试
    if args.mode == "quick":
        run_quick_test(device)
    elif args.mode == "small":
        run_small_test(device)
    elif args.mode == "step":
        run_step_test(device)

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
