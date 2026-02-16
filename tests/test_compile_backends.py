"""
测试 torch.compile 的不同 backend - 找到 Windows 可用的选项
"""
import time
import torch
from apt.model.architectures.claude4_model import create_claude4_model
from apt.vgpu.runtime.vb_integration import apply_virtual_blackwell_v64, VBConfigV64

device = "cuda"
print(f"Device: {device}\n")

# 创建模型
def create_baseline():
    return create_claude4_model(
        vocab_size=4096, d_model=768, num_layers=6,
        num_heads=12, ffn_hidden=3072
    )

x = torch.randint(0, 4096, (4, 1024), device=device)

# 测试不同的 backend
backends_to_test = [
    ("eager", None),  # 不编译
    ("aot_eager", "aot_eager"),  # AOT compilation, 不需要 Triton
]

results = {}

for name, backend in backends_to_test:
    print(f"\n{'='*70}")
    print(f"Testing backend: {name}")
    print(f"{'='*70}")

    try:
        model = create_baseline().to(device)

        if backend is None:
            # 不编译
            model.train()
            print("Mode: 未编译")
        else:
            # 尝试编译
            print(f"Mode: torch.compile(backend='{backend}')")
            model = torch.compile(model, backend=backend, fullgraph=False)
            model.train()

        # Warmup
        for _ in range(3):
            y = model(x)
            y.mean().backward()

        torch.cuda.synchronize()

        # 计时
        t0 = time.time()
        for _ in range(10):
            y = model(x)
            y.mean().backward()
        torch.cuda.synchronize()

        elapsed = time.time() - t0
        results[name] = elapsed
        print(f"✅ 成功！Time: {elapsed:.2f}s")

    except Exception as e:
        print(f"❌ 失败: {type(e).__name__}: {str(e)[:100]}")
        results[name] = None

# 测试 VB Gate
print(f"\n{'='*70}")
print("Testing VB Gate Projected")
print(f"{'='*70}")

try:
    model_vb = create_baseline().to(device)
    cfg = VBConfigV64(
        pulse_interval=100,
        gate_projected_mode=True,
        enable_stats=False,
    )
    model_vb, _ = apply_virtual_blackwell_v64(model_vb, cfg)
    print(f"Replaced {model_vb._vb_replaced_linears} layers")
    model_vb.train()

    # Warmup
    for _ in range(3):
        y = model_vb(x)
        y.mean().backward()

    torch.cuda.synchronize()

    # 计时
    t0 = time.time()
    for _ in range(10):
        y = model_vb(x)
        y.mean().backward()
    torch.cuda.synchronize()

    elapsed = time.time() - t0
    results['vb_gate'] = elapsed
    print(f"✅ 成功！Time: {elapsed:.2f}s")

except Exception as e:
    print(f"❌ 失败: {type(e).__name__}: {str(e)[:100]}")
    results['vb_gate'] = None

# 总结
print(f"\n{'='*70}")
print("RESULTS SUMMARY")
print(f"{'='*70}")

baseline = results.get('eager')
if baseline:
    print(f"\nBaseline (未编译): {baseline:.2f}s")

for name, time_result in results.items():
    if name == 'vb_gate' or time_result is None:
        continue
    overhead = ((time_result / baseline) - 1) * 100
    print(f"{name}: {time_result:.2f}s ({overhead:+.1f}% vs Baseline)")

if results.get('vb_gate'):
    overhead = ((results['vb_gate'] / baseline) - 1) * 100
    print(f"VB Gate: {results['vb_gate']:.2f}s ({overhead:+.1f}% vs Baseline)")

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")

if 'aot_eager' in results and results['aot_eager']:
    print("✅ aot_eager backend 在 Windows 上可用！")
    print("   可以使用 torch.compile(model, backend='aot_eager')")
else:
    print("❌ torch.compile 在当前环境不可用")

if results.get('vb_gate', 0) > 0:
    gap = ((results['vb_gate'] / baseline) - 1) * 100
    if gap < 5:
        print(f"\n✅ VB Gate 表现良好：{gap:+.1f}% 开销")
    else:
        print(f"\n⚠️  VB Gate 有 {gap:+.1f}% 开销")
