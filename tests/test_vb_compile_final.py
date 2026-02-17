"""
终极测试：torch.compile + VB Gate vs Baseline
"""
import time
import torch
from apt.model.architectures.claude4_model import create_claude4_model
from apt.vgpu.runtime.vb_integration import apply_virtual_blackwell_v64, VBConfigV64

device = "cuda"
print(f"Device: {device}")
print(f"PyTorch: {torch.__version__}\n")

# Enable TF32 for better performance
torch.set_float32_matmul_precision("high")

# ---- torch.compile knobs (stability first) -----------------------------------
# Relax module identity guards to avoid check_type_id recompiles
torch._dynamo.config.guard_nn_modules = False
torch._dynamo.config.allow_unspec_int_on_nn_module = True
torch._dynamo.config.recompile_limit = 256
torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.capture_scalar_outputs = True

def reset_compile_state():
    """Reset torch._dynamo and torch._inductor caches between tests"""
    try:
        import torch._dynamo
        torch._dynamo.reset()
        print("  [CACHE] Cleared torch._dynamo cache")
    except Exception as e:
        print(f"  [CACHE] Failed to reset dynamo: {e}")
    try:
        import torch._inductor
        torch._inductor.utils.clear_cache()
        print("  [CACHE] Cleared torch._inductor cache")
    except Exception as e:
        print(f"  [CACHE] Failed to clear inductor: {e}")

# 创建模型函数
def create_baseline():
    torch.manual_seed(42)  # 固定模型初始化
    return create_claude4_model(
        vocab_size=4096, d_model=768, num_layers=6,
        num_heads=12, ffn_hidden=3072
    )

# 固定输入数据
torch.manual_seed(42)
x = torch.randint(0, 4096, (4, 1024), device=device)

NUM_BATCHES = 200  # 增加批次以减少统计噪声
results = {}

# Test 1: Baseline 未编译
print("=" * 70)
print(f"Test 1: Baseline (未编译, {NUM_BATCHES} batches)")
print("=" * 70)
model1 = create_baseline().to(device)
model1.train()
for _ in range(3):
    y = model1(x)
    y.mean().backward()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(NUM_BATCHES):
    y = model1(x)
    y.mean().backward()
torch.cuda.synchronize()
results['baseline_no_compile'] = time.time() - t0
print(f"Time: {results['baseline_no_compile']:.2f}s ({NUM_BATCHES/results['baseline_no_compile']:.3f} batch/s)")

# Test 2: Baseline 编译
print("\n" + "=" * 70)
print(f"Test 2: Baseline (torch.compile, {NUM_BATCHES} batches)")
print("=" * 70)
print("编译中（需要 30-60 秒）...")
reset_compile_state()
model2 = create_baseline().to(device)
model2 = torch.compile(model2)  # 使用默认 backend (inductor)
model2.train()
print("编译完成，开始测试...")
for _ in range(3):
    y = model2(x)
    y.mean().backward()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(NUM_BATCHES):
    y = model2(x)
    y.mean().backward()
torch.cuda.synchronize()
results['baseline_compile'] = time.time() - t0
print(f"Time: {results['baseline_compile']:.2f}s ({NUM_BATCHES/results['baseline_compile']:.3f} batch/s)")

# Test 3: VB Gate 未编译
print("\n" + "=" * 70)
print(f"Test 3: VB Gate Projected (未编译, {NUM_BATCHES} batches)")
print("=" * 70)
model3 = create_baseline().to(device)
cfg = VBConfigV64(
    pulse_interval=100,
    gate_projected_mode=True,
    enable_stats=False,  # 禁用统计以获得最佳性能
)
model3, _ = apply_virtual_blackwell_v64(model3, cfg)
print(f"Replaced {model3._vb_replaced_linears} Linear layers")
model3.train()
for _ in range(3):
    y = model3(x)
    y.mean().backward()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(NUM_BATCHES):
    y = model3(x)
    y.mean().backward()
torch.cuda.synchronize()
results['vb_gate_no_compile'] = time.time() - t0
print(f"Time: {results['vb_gate_no_compile']:.2f}s ({NUM_BATCHES/results['vb_gate_no_compile']:.3f} batch/s)")

# Test 4: VB Gate 编译（关键测试！）
print("\n" + "=" * 70)
print(f"Test 4: VB Gate Projected (torch.compile, {NUM_BATCHES} batches)")
print("=" * 70)
print("编译中（需要 30-60 秒）...")
reset_compile_state()
model4 = create_baseline().to(device)
model4, _ = apply_virtual_blackwell_v64(model4, cfg)
# 使用默认 backend，与 baseline 保持一致
model4 = torch.compile(model4)
model4.train()
print("编译完成，开始测试...")
for _ in range(3):
    y = model4(x)
    y.mean().backward()
torch.cuda.synchronize()
t0 = time.time()
for _ in range(NUM_BATCHES):
    y = model4(x)
    y.mean().backward()
torch.cuda.synchronize()
results['vb_gate_compile'] = time.time() - t0
print(f"Time: {results['vb_gate_compile']:.2f}s ({NUM_BATCHES/results['vb_gate_compile']:.3f} batch/s)")

# 总结
print("\n" + "=" * 70)
print("FINAL RESULTS - torch.compile 效果")
print("=" * 70)
base = results['baseline_no_compile']
print(f"{'配置':<30} {'时间(s)':>10} {'vs Base':>12} {'batch/s':>12}")
print("-" * 70)
print(f"{'Baseline (未编译)':<30} {results['baseline_no_compile']:>10.2f} {0:>12.1f} {NUM_BATCHES/results['baseline_no_compile']:>12.3f}")
print(f"{'Baseline (编译)':<30} {results['baseline_compile']:>10.2f} {((results['baseline_compile']/base-1)*100):>12.1f} {NUM_BATCHES/results['baseline_compile']:>12.3f}")
print(f"{'VB Gate (未编译)':<30} {results['vb_gate_no_compile']:>10.2f} {((results['vb_gate_no_compile']/base-1)*100):>12.1f} {NUM_BATCHES/results['vb_gate_no_compile']:>12.3f}")
print(f"{'VB Gate (编译)':<30} {results['vb_gate_compile']:>10.2f} {((results['vb_gate_compile']/base-1)*100):>12.1f} {NUM_BATCHES/results['vb_gate_compile']:>12.3f}")

print("\n" + "=" * 70)
print("KEY INSIGHT - torch.compile 是否解决了问题？")
print("=" * 70)
no_compile_gap = ((results['vb_gate_no_compile']/base) - 1) * 100
compile_gap = ((results['vb_gate_compile']/base) - 1) * 100
compile_benefit = no_compile_gap - compile_gap

print(f"VB Gate 未编译 vs Baseline: {no_compile_gap:+.1f}%")
print(f"VB Gate 编译后 vs Baseline:   {compile_gap:+.1f}%")
print(f"编译带来的改善:            {compile_benefit:+.1f}%")

if compile_gap < 1:
    print("\n>>> SUCCESS! torch.compile 完全消除了开销！")
    print(">>> '语言学家消失，只剩下外国人直接说话'")
elif compile_gap < no_compile_gap:
    print(f"\n>>> IMPROVEMENT! torch.compile 减少了 {compile_benefit:.1f}% 开销")
    print(">>> 但仍有一些残留开销")
else:
    print(f"\n>>> WARNING: torch.compile 没有改善，甚至更差")

print("=" * 70)
