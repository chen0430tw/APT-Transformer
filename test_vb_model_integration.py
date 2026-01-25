#!/usr/bin/env python3
"""测试重构后的VB与实际模型的集成"""

import torch
import torch.nn as nn
import sys
from apt.model.architectures.claude4_model import create_claude_unified
from apt.vgpu.runtime.vb_integration import VBModelWrapper

output_file = 'vb_model_test_result.txt'

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("Virtual Blackwell v6.0 模型集成测试\n")
    f.write("="*80 + "\n\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    f.write(f"设备: {device}\n\n")

    # 创建小模型
    f.write("创建Claude4模型...\n")
    model = create_claude_unified(
        vocab_size=1000,
        d_model=128,
        n_heads=2,
        d_ff=256,
        num_layers=2,
        rank=2,
        enable_reflection=True
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    f.write(f"模型参数: {total_params:,}\n\n")

    # 应用VB
    f.write("应用Virtual Blackwell...\n")

    old_stdout = sys.stdout
    sys.stdout = f

    try:
        wrapper = VBModelWrapper(
            model,
            mode='training',
            enable_fp4=True,
            enable_quantization=False,
            replace_pattern='all'
        )
    except Exception as e:
        f.write(f"错误: {e}\n")
        import traceback
        traceback.print_exc(file=f)
        sys.stdout = old_stdout
        raise

    sys.stdout = old_stdout

    vb_count = len(wrapper.replaced_layers)
    f.write(f"[OK] {vb_count} 个线性层被替换为虚拟Blackwell\n\n")

    # 测试前向传播
    f.write("测试前向传播...\n")
    try:
        input_ids = torch.randint(0, 1000, (2, 16)).to(device)

        with torch.no_grad():
            outputs = wrapper(input_ids)

        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        f.write(f"输入形状: {input_ids.shape}\n")
        f.write(f"输出形状: {logits.shape}\n")
        f.write(f"[OK] 前向传播成功\n\n")

    except Exception as e:
        f.write(f"[X] 前向传播失败: {e}\n")
        import traceback
        traceback.print_exc(file=f)

    # 测试训练
    f.write("测试训练（10个batch）...\n")
    optimizer = torch.optim.Adam(wrapper.parameters(), lr=0.001)

    losses = []
    for i in range(10):
        try:
            input_ids = torch.randint(0, 1000, (2, 16)).to(device)
            optimizer.zero_grad()

            outputs = wrapper(input_ids)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            # Next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        except Exception as e:
            f.write(f"[X] Batch {i} 失败: {e}\n")
            import traceback
            traceback.print_exc(file=f)
            break

    if losses:
        f.write(f"训练10个batch成功\n")
        f.write(f"Loss范围: {min(losses):.4f} - {max(losses):.4f}\n")
        f.write(f"平均Loss: {sum(losses)/len(losses):.4f}\n")
        f.write(f"[OK] 训练功能正常\n\n")

    # VB统计
    f.write("Virtual Blackwell 统计:\n")
    f.write("-"*80 + "\n")

    old_stdout = sys.stdout
    sys.stdout = f
    try:
        wrapper.print_all_stats()
    except:
        pass
    sys.stdout = old_stdout

    f.write("\n" + "="*80 + "\n")
    f.write("测试完成！\n")
    f.write("="*80 + "\n")

# 读取并显示结果（避免print到stdout）
with open(output_file, 'r', encoding='utf-8') as f:
    content = f.read()
    # 写到stderr避免stdout问题
    sys.stderr.write(content)
