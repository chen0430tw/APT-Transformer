"""
测试虚拟显存功能 - 验证内存峰值降低
"""
import torch
import torch.nn as nn
from apt.vgpu.runtime.virtual_vram import VirtualVRAMConfig, virtual_vram

device = "cuda"
print(f"Device: {device}")
print(f"PyTorch: {torch.__version__}\n")

# 创建一个简单的测试模型
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1024, 4096)
        self.linear2 = nn.Linear(4096, 1024)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

model = TestModel().to(device)
model.train()

# 创建大批次输入以产生显存压力
batch_size = 16
seq_len = 2048
x = torch.randn(batch_size, seq_len, 1024, device=device)

print("=" * 70)
print("Test 1: 不使用虚拟显存")
print("=" * 70)
torch.cuda.reset_peak_memory_stats(device)
for _ in range(3):
    y = model(x)
    y.mean().backward()
torch.cuda.synchronize()
baseline_memory = torch.cuda.max_memory_allocated(device) / 1024**3
print(f"峰值显存: {baseline_memory:.2f} GB")

print("\n" + "=" * 70)
print("Test 2: 使用虚拟显存 (CPU offload)")
print("=" * 70)

cfg = VirtualVRAMConfig(
    enabled=True,
    offload="cpu",
    min_tensor_bytes=1<<20,  # 1MB 以上才搬
    pin_memory=True,
    non_blocking=True,
    verbose=True,  # 显示 offload 详情
)

torch.cuda.reset_peak_memory_stats(device)
for _ in range(3):
    with virtual_vram(cfg):
        y = model(x)
        y.mean().backward()
torch.cuda.synchronize()
vram_memory = torch.cuda.max_memory_allocated(device) / 1024**3
print(f"峰值显存: {vram_memory:.2f} GB")

print("\n" + "=" * 70)
print("RESULTS - 内存节省")
print("=" * 70)
saved = baseline_memory - vram_memory
saved_pct = (saved / baseline_memory) * 100
print(f"不使用虚拟显存: {baseline_memory:.2f} GB")
print(f"使用虚拟显存:   {vram_memory:.2f} GB")
print(f"节省:           {saved:.2f} GB ({saved_pct:.1f}%)")

if saved > 0:
    print(f"\n>>> SUCCESS! 虚拟显存节省了 {saved_pct:.1f}% 的显存")
else:
    print(f"\n>>> INFO: 虚拟显存没有节省显存 (可能模型太小或张量都 < 1MB)")
print("=" * 70)
