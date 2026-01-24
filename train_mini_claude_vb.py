#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 Virtual Blackwell 训练迷你 Claude 模型
统计虚拟显卡数量和加速效果
"""

import torch
import torch.nn as nn
import time
import sys
import os
from apt.model.architectures.claude4_model import create_claude_unified
from apt.vgpu.runtime.vb_integration import enable_vb_optimization

# 跨平台兼容的日志文件路径
LOG_FILE = os.path.join(os.getcwd(), 'train_mini_claude_vb.log')

def log(msg):
    """安全的日志输出"""
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')

def create_training_data(num_samples=500):
    """创建简单的训练数据"""
    # Claude风格的训练数据
    data = []

    # 简单对话数据
    conversations = [
        ("你好", "你好！我是Claude，很高兴认识你。"),
        ("天气怎么样", "我是AI助手，无法获取实时天气信息。"),
        ("1+1等于几", "1+1等于2。"),
        ("帮我写代码", "我很乐意帮助你写代码。请告诉我你需要什么功能？"),
        ("谢谢", "不客气！有其他问题随时问我。"),
    ]

    for _ in range(num_samples):
        q, a = conversations[_ % len(conversations)]
        data.append(f"Human: {q}\n\nAssistant: {a}")

    return data

def train_mini_claude():
    """训练迷你Claude模型"""

    log("=" * 80)
    log("使用 Virtual Blackwell 训练迷你 Claude 模型")
    log("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f"\n设备: {device}")

    # 创建训练数据
    log("\n1. 创建训练数据...")
    train_texts = create_training_data(num_samples=200)
    log(f"   训练样本数: {len(train_texts)}")

    # 创建迷你Claude模型
    log("\n2. 创建迷你 Claude 模型...")
    model = create_claude_unified(
        vocab_size=5000,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        num_layers=3,
        rank=2,
        enable_reflection=True
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log(f"   模型参数量: {total_params:,}")

    # 启用 Virtual Blackwell
    log("\n3. 启用 Virtual Blackwell 优化...")

    # 统计 VB adapter 数量
    vb_adapter_count_before = 0

    model_vb = enable_vb_optimization(
        model,
        mode='training',
        enable_fp4=True,
        enable_quantization=False,
        replace_pattern='all'
    )

    # 统计替换的层数（每层一个VB adapter）
    vb_adapter_count = len(model_vb.replaced_layers)
    log(f"\n   ✅ 虚拟Blackwell显卡数量: {vb_adapter_count} 张")
    log(f"   每个线性层对应一张虚拟显卡")

    # 显示前10个VB层
    log(f"\n   前10个虚拟显卡分配:")
    for i, layer_name in enumerate(model_vb.replaced_layers[:10]):
        log(f"   [{i+1}] {layer_name}")
    if vb_adapter_count > 10:
        log(f"   ... 还有 {vb_adapter_count - 10} 张虚拟显卡")

    # 创建优化器
    optimizer = torch.optim.Adam(model_vb.parameters(), lr=0.001)

    # 简单训练循环
    log("\n4. 开始训练...")
    log("-" * 80)

    batch_size = 8
    num_epochs = 3
    max_batches_per_epoch = 25

    start_time = time.time()
    total_batches = 0

    for epoch in range(num_epochs):
        log(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        epoch_batches = 0

        for batch_idx in range(0, min(len(train_texts), batch_size * max_batches_per_epoch), batch_size):
            # 创建batch
            batch_texts = train_texts[batch_idx:batch_idx + batch_size]

            # 简单编码（字符级）
            max_len = 128
            input_ids = []
            for text in batch_texts:
                ids = [ord(c) % 5000 for c in text[:max_len]]
                if len(ids) < max_len:
                    ids = ids + [0] * (max_len - len(ids))
                input_ids.append(ids)

            input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)

            # 前向传播
            optimizer.zero_grad()

            try:
                outputs = model_vb(input_ids)

                # 简单loss（预测下一个token）
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                # Shift for next token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()

                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )

                # 反向传播
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_batches += 1
                total_batches += 1

            except Exception as e:
                log(f"   Warning: Batch {batch_idx} failed: {e}")
                continue

        avg_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
        log(f"   平均Loss: {avg_loss:.4f}")

    training_time = time.time() - start_time

    log("\n" + "=" * 80)
    log("训练完成！")
    log("=" * 80)
    log(f"\n训练统计:")
    log(f"  总时间: {training_time:.2f}秒")
    log(f"  总批次: {total_batches}")
    log(f"  吞吐量: {total_batches/training_time:.2f} batch/s")
    log(f"  虚拟显卡数: {vb_adapter_count} 张")

    # 打印VB统计
    log("\n" + "=" * 80)
    log("Virtual Blackwell 详细统计")
    log("=" * 80)

    import io
    stats_buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stats_buffer

    try:
        model_vb.print_all_stats()
    except:
        pass

    sys.stdout = old_stdout
    stats_output = stats_buffer.getvalue()

    for line in stats_output.split('\n'):
        log(line)

    # 测试推理
    log("\n" + "=" * 80)
    log("测试推理 - 与迷你Claude对话")
    log("=" * 80)

    model_vb.eval()
    test_prompts = [
        "Human: 你好\n\nAssistant:",
        "Human: 1+1等于几\n\nAssistant:",
    ]

    with torch.no_grad():
        for prompt in test_prompts:
            log(f"\n输入: {prompt}")

            # 简单编码
            input_ids = [ord(c) % 5000 for c in prompt[:64]]
            if len(input_ids) < 64:
                input_ids = input_ids + [0] * (64 - len(input_ids))
            input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

            try:
                outputs = model_vb(input_ids)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                # 简单采样
                next_token_logits = logits[0, -1, :]
                next_token = torch.argmax(next_token_logits).item()

                log(f"输出token: {next_token}")
                log(f"输出形状: {logits.shape}")
            except Exception as e:
                log(f"推理失败: {e}")

    log("\n" + "=" * 80)
    log(f"✅ 成功训练迷你Claude模型，使用了 {vb_adapter_count} 张虚拟Blackwell显卡！")
    log("=" * 80)

    return vb_adapter_count

if __name__ == "__main__":
    # 清空日志文件
    with open(LOG_FILE, 'w') as f:
        f.write("")

    try:
        num_vb = train_mini_claude()

        # 读取日志并打印
        with open(LOG_FILE, 'r') as f:
            content = f.read()

        # 写到stderr避免stdout问题
        sys.stderr.write(content)
        sys.stderr.flush()

    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        import traceback
        sys.stderr.write(traceback.format_exc())
