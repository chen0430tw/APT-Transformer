#!/usr/bin/env python3
"""
Virtual VRAM v1.6 本地测试脚本 (RTX 3070 8GB)
基于 slurm/run_vgpu_lecac_test.sh
"""
import sys
import os

# 添加路径
sys.path.insert(0, "/mnt/d/APT-Transformer")

import subprocess

# RTX 3070 8GB 适配参数
# - 显存小: batch_size 2 (原来是4)
# - 梯度累积: 4 (原来是2，补偿batch size减少)
# - max_steps: 10 (快速测试，原来是50)
# - save_interval: 5 (原来是25)
cmd = [
    "python3", "-m", "apt.trainops.scripts.pretrain_quickcook",
    "--output-dir", "./test_vvram_local",
    "--max-steps", "10",
    "--save-interval", "5",
    "--weight-fineweb", "0.7",
    "--weight-hlbd", "0.3",
    "--no-c4",
    "--no-mc4",
    "--batch-size", "2",  # RTX 3070适配：减少batch size
    "--gradient-accumulation", "4",  # 补偿：增加梯度累积
    "--use-virtual-vram",
    "--use-lecac",
    "--lecac-bits", "2",
    "--vram-enable-prefetch",
    "--vram-enable-nested-v16",
    "--vram-verbose",
]

print("=" * 60)
print("Virtual VRAM v1.6 本地测试 (RTX 3070 8GB)")
print("=" * 60)
print(f"工作目录: {os.getcwd()}")
print(f"命令: {' '.join(cmd)}")
print("=" * 60)

# 切换到APT-Transformer目录
os.chdir("/mnt/d/APT-Transformer")

# 运行
result = subprocess.run(cmd, check=False)

print("\n" + "=" * 60)
if result.returncode == 0:
    print("✅ 测试完成!")
else:
    print(f"❌ 测试失败，退出码: {result.returncode}")
print("=" * 60)

sys.exit(result.returncode)
