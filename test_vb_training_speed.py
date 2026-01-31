#!/usr/bin/env python3
"""快速测试VB训练速度（优化后的精度分离缓存）"""

import torch
import torch.nn as nn
import time
import sys
from apt.model.architectures.claude4_model import create_claude_unified
from apt.vgpu.runtime.vb_integration import VBModelWrapper

# 输出到文件
output_file = 'vb_speed_test_result.txt'
log_file = open(output_file, 'w', encoding='utf-8')

def log(msg):
    log_file.write(msg + '\n')
    log_file.flush()

log("="*80)
log("Virtual Blackwell v6.0 训练速度测试（精度分离缓存优化）")
log("="*80)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
log(f"设备: {device}")
log("")

# 创建Claude4模型
log("创建Claude4模型...")
model = create_claude_unified(
    vocab_size=50000,
    d_model=256,
    n_heads=4,
    d_ff=1024,
    num_layers=3,
    rank=4,
    enable_reflection=False  # 禁用反思层 - 它是性能杀手，不是VB的问题
).to(device)

total_params = sum(p.numel() for p in model.parameters())
log(f"模型参数: {total_params:,}")
log("")

# 应用VB
log("应用Virtual Blackwell...")
import sys
sys.stderr.write("开始应用VB...\n")
sys.stderr.flush()

wrapper = VBModelWrapper(
    model,
    mode='training',
    enable_fp4=True,
    enable_quantization=True,
    replace_pattern='all'
)

sys.stderr.write("VB应用完成，正在统计层数...\n")
sys.stderr.flush()

vb_count = len(wrapper.replaced_layers)
log(f"虚拟Blackwell显卡: {vb_count} 张")
log("")

sys.stderr.write(f"已替换 {vb_count} 层，准备优化器...\n")
sys.stderr.flush()

# 准备训练
optimizer = torch.optim.Adam(wrapper.parameters(), lr=0.001)

sys.stderr.write("优化器创建完成，准备训练...\n")
sys.stderr.flush()
criterion = nn.CrossEntropyLoss()

batch_size = 8
seq_len = 128
num_batches = 20

log(f"配置: batch_size={batch_size}, seq_len={seq_len}, batches={num_batches}")
log("="*80)
log("")

batch_times = []
losses = []

log("开始训练测试...")
log("-"*80)

total_start = time.time()

for batch_idx in range(num_batches):
    batch_start = time.time()

    input_ids = torch.randint(0, 50000, (batch_size, seq_len)).to(device)

    optimizer.zero_grad()
    outputs = wrapper(input_ids)

    if isinstance(outputs, tuple):
        logits = outputs[0]
    else:
        logits = outputs

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    loss = criterion(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

    loss.backward()
    optimizer.step()

    batch_time = time.time() - batch_start
    batch_times.append(batch_time)
    losses.append(loss.item())

    if (batch_idx + 1) % 5 == 0:
        avg_time = sum(batch_times[-5:]) / 5
        avg_loss = sum(losses[-5:]) / 5
        log(f"Batch {batch_idx+1:2d}/{num_batches} | "
            f"Loss: {avg_loss:.4f} | "
            f"Time: {batch_time:.2f}s | "
            f"Avg(last 5): {avg_time:.2f}s")

total_time = time.time() - total_start

log("-"*80)
log("")

# 统计分析
log("="*80)
log("性能统计")
log("="*80)

avg_batch_time = sum(batch_times) / len(batch_times)
first_batch_time = batch_times[0]
later_avg = sum(batch_times[1:]) / len(batch_times[1:]) if len(batch_times) > 1 else 0

log(f"总时间: {total_time:.2f}s")
log(f"总批次: {num_batches}")
log(f"平均batch时间: {avg_batch_time:.2f}s")
log(f"首批时间: {first_batch_time:.2f}s (包含精度分离)")
log(f"后续批次平均: {later_avg:.2f}s (使用缓存)")
log(f"吞吐量: {num_batches/total_time:.2f} batch/s")
log("")

batches_per_epoch = 100
estimated_epoch_time = later_avg * batches_per_epoch + (first_batch_time - later_avg)
log(f"推算单epoch时间 (100 batch): {estimated_epoch_time:.1f}s ({estimated_epoch_time/60:.1f}分钟)")
log("")

# VB统计
log("="*80)
log("Virtual Blackwell 统计")
log("="*80)

all_stats = wrapper.get_all_stats()

total_computes = 0
total_cache_hits = 0
total_cache_refreshes = 0

for name, stats in all_stats.items():
    if 'layer1_vgpu' in stats:
        vgpu = stats['layer1_vgpu']
        total_computes += vgpu.get('total', 0)
        total_cache_hits += vgpu.get('cache_hits', 0)
        total_cache_refreshes += vgpu.get('cache_refreshes', 0)

if total_computes > 0:
    cache_hit_rate = total_cache_hits / total_computes * 100
    log(f"总计算次数: {total_computes}")
    log(f"缓存命中: {total_cache_hits} ({cache_hit_rate:.1f}%)")
    log(f"缓存刷新: {total_cache_refreshes}")
    log(f"精度分离次数: {total_computes - total_cache_hits} (首次 + 刷新)")
    log("")

    log("前3层详细统计:")
    log("-"*80)
    count = 0
    for name, stats in all_stats.items():
        if count >= 3:
            break
        if 'layer1_vgpu' in stats:
            vgpu = stats['layer1_vgpu']
            log(f"\n[{name}]")
            log(f"  总计算: {vgpu.get('total', 0)}")
            log(f"  缓存命中: {vgpu.get('cache_hits', 0)}")
            log(f"  缓存刷新: {vgpu.get('cache_refreshes', 0)}")
            log(f"  缓存命中率: {vgpu.get('cache_hit_rate', 0):.1%}")
            count += 1

log("")
log("="*80)
log("测试完成！")
log("="*80)

log_file.close()

# 输出结果文件路径
sys.stderr.write(f"\n测试完成！结果已保存到: {output_file}\n")
