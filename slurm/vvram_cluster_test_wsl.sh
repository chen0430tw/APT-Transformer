#!/bin/bash
#SBATCH --job-name=apt_vvram_v16_test
#SBATCH --account=ENT114035
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=vvram_test_%j.out
#SBATCH --error=vvram_test_%j.err

# ============================================================================
# APT Virtual VRAM v1.6 集群测试脚本 (WSL本地版本)
# ============================================================================

echo "============================================"
echo "APT Virtual VRAM v1.6 集群测试"
echo "集群: WSL"
echo "节点: $SLURM_JOB_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "开始时间: $(date)"
echo "============================================"

# 设置工作目录
cd /mnt/d/apt-transformer || exit 1

# 添加项目路径到PYTHONPATH
export PYTHONPATH="/mnt/d/apt-transformer:$PYTHONPATH"

echo ""
echo "============================================"
echo "测试Virtual VRAM v1.6"
echo "============================================"

# 创建测试脚本
cat > /tmp/vvram_cluster_test.py << 'EOFPY'
#!/usr/bin/env python3
"""
Virtual VRAM v1.6 集群测试
模拟真实集群训练环境
"""
import sys
import os
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader

print("[INFO] Virtual VRAM v1.6 集群测试")
print(f"[INFO] 集群: nano5")
print(f"[INFO] 作业ID: {os.environ.get('SLURM_JOB_ID', 'local_test')}")
print(f"[INFO] 节点: {os.environ.get('SLURM_JOB_NODELIST', 'localhost')}")
print(f"[INFO] GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")

# 导入Virtual VRAM
from apt.vgpu.runtime.virtual_vram import VirtualVRAMConfig, virtual_vram
from apt.vgpu.runtime.lecac import lecac_quantize, lecac_dequantize

# 检查CUDA
if not torch.cuda.is_available():
    print("[ERROR] CUDA不可用")
    sys.exit(1)

print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
print(f"[INFO] 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

# 配置Virtual VRAM v1.6
cfg = VirtualVRAMConfig(
    enabled=True,
    enable_nested_v16=True,
    min_tensor_bytes=5 << 20,  # 5MB阈值
    nested_block_size=64,
    nested_quantization_bits=2,  # LECaC INT2
    verbose=True
)

print("[INFO] Virtual VRAM配置:")
print(f"  - enabled: {cfg.enabled}")
print(f"  - enable_nested_v16: {cfg.enable_nested_v16}")
print(f"  - min_tensor_bytes: {cfg.min_tensor_bytes / 1024**2}MB")
print(f"  - nested_quantization_bits: {cfg.nested_quantization_bits}")

# 创建简单的语言模型
class SimpleLM(nn.Module):
    def __init__(self, vocab_size=1000, embed_dim=256, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleLM().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("[INFO] 模型已创建")

# 测试1: 小tensor（不触发Virtual VRAM）
print("\n[测试1] 小tensor训练（<5MB，不触发Virtual VRAM）")
print("-" * 60)
for step in range(3):
    optimizer.zero_grad()
    x = torch.randint(0, 1000, (10, 64)).cuda()  # ~160KB
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    print(f"  Step {step}: Loss={loss.item():.4f}")

# 测试2: 大tensor（触发Virtual VRAM）
print("\n[测试2] 大tensor训练（>5MB，触发Virtual VRAM）")
print("-" * 60)
with virtual_vram(cfg):
    for step in range(5):
        optimizer.zero_grad()

        # 创建大tensor（>5MB，适配8GB显存）
        big_x = torch.randint(0, 1000, (256, 512)).cuda()  # ~512KB input -> ~128MB embedding
        big_y = model(big_x)
        loss = big_y.sum()
        loss.backward()
        optimizer.step()
        print(f"  Step {step}: Loss={loss.item():.4f}")

# 测试3: 检查NaN
print("\n[测试3] 检查参数健康度")
print("-" * 60)
has_nan = False
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        print(f"  ❌ {name} 包含NaN!")
        has_nan = True
    elif torch.isinf(param).any():
        print(f"  ❌ {name} 包含Inf!")
        has_nan = True

if not has_nan:
    print("  ✅ 所有参数正常，无NaN/Inf")

# 测试4: 显存使用
print("\n[测试4] 显存使用情况")
print("-" * 60)
print(f"  显存已用: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
print(f"  显存缓存: {torch.cuda.memory_reserved() / 1024**3:.2f}GB")

print("\n============================================")
print("✅ Virtual VRAM v1.6 集群测试完成")
print("============================================")

EOFPY

# 运行测试
python3 /tmp/vvram_cluster_test.py

# 记录退出码
exit $?
