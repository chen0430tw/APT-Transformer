#!/usr/bin/env python3
"""
Virtual VRAM v1.6 测试 - 下载少量C4分片
只下载1-2个分片，快速测试
"""
import sys
import os

sys.path.insert(0, "/mnt/d/APT-Transformer")

import subprocess

# 只下载C4的少量分片（streaming模式）
cmd = [
    "python3", "-m", "apt.trainops.scripts.pretrain_quickcook",
    "--output-dir", "./test_vvram_c4_small",
    "--max-steps", "10",  # 只训练10步
    "--save-interval", "5",
    "--weight-fineweb", "0.0",  # 不用fineweb
    "--weight-hlbd", "1.0",    # 只用hlbd（本地数据）
    "--no-c4",
    "--no-mc4",
    "--batch-size", "2",
    "--gradient-accumulation", "4",
    "--use-virtual-vram",
    "--vram-enable-nested-v16",
    "--vram-verbose",
]

print("=" * 60)
print("Virtual VRAM v1.6 测试 - 只用HLBD数据")
print("=" * 60)
print(f"命令: {' '.join(cmd)}")
print("=" * 60)

os.chdir("/mnt/d/APT-Transformer")

# 运行
result = subprocess.run(cmd, check=False)

print("\n" + "=" * 60)
if result.returncode == 0:
    print("✅ 测试完成!")
    print("请查看 ./test_vvram_c4_small/ 目录中的结果")
else:
    print(f"❌ 测试失败，退出码: {result.returncode}")
print("=" * 60)

sys.exit(result.returncode)
