import torch
import torch.nn as nn
import time
import os
from apt.model.architectures.claude4_model import create_claude_unified
from apt.vgpu.runtime.vb_integration import VBModelWrapper
import sys

# 使用当前目录，跨平台兼容
output_file = os.path.join(os.getcwd(), 'claude_loss3_training.txt')

with open(output_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("训练Claude模型直到Loss降到3.0 (使用Virtual Blackwell)\n")
    f.write("="*80 + "\n\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    f.write(f"设备: {device}\n\n")

    # 创建Claude模型
    f.write("创建 Claude4 模型...\n")
    model = create_claude_unified(
        vocab_size=5000,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        num_layers=3,
        rank=2,
        enable_reflection=True
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    f.write(f"模型参数: {total_params:,}\n\n")

    # 启用Virtual Blackwell
    f.write("启用 Virtual Blackwell (62张虚拟显卡)...\n")
    f.flush()

    old_stdout = sys.stdout
    sys.stdout = f

    wrapper = VBModelWrapper(
        model,
        mode='training',
        enable_fp4=True,
        enable_quantization=False,
        replace_pattern='all'
    )

    sys.stdout = old_stdout

    vb_count = len(wrapper.replaced_layers)
    f.write(f"✅ {vb_count} 张虚拟Blackwell显卡已启用\n\n")

    # 优化器
    optimizer = torch.optim.Adam(wrapper.parameters(), lr=0.0005)  # 降低学习率

    f.write("="*80 + "\n")
    f.write("开始训练 - 目标: Loss < 3.0\n")
    f.write("="*80 + "\n\n")

    target_loss = 3.0
    max_epochs = 100
    batches_per_epoch = 100
    batch_size = 8

    start_time = time.time()
    total_batches = 0
    best_loss = float('inf')

    for epoch in range(max_epochs):
        epoch_loss = 0
        epoch_start = time.time()

        for batch_idx in range(batches_per_epoch):
            # 随机数据
            input_ids = torch.randint(0, 5000, (batch_size, 64)).to(device)

            optimizer.zero_grad()

            try:
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

                epoch_loss += loss.item()
                total_batches += 1

            except Exception as e:
                f.write(f"Error at batch {batch_idx}: {e}\n")
                f.flush()
                continue

        avg_loss = epoch_loss / batches_per_epoch
        epoch_time = time.time() - epoch_start

        if avg_loss < best_loss:
            best_loss = avg_loss

        f.write(f"Epoch {epoch+1:3d}/{max_epochs}: Loss={avg_loss:.4f} (最佳={best_loss:.4f}) - {epoch_time:.1f}s\n")
        f.flush()

        # 每10个epoch打印详细信息
        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            throughput = total_batches / elapsed
            f.write(f"  进度: {total_batches} batches, {throughput:.2f} batch/s\n")
            f.flush()

        # 检查是否达到目标
        if avg_loss < target_loss:
            f.write(f"\n🎉 达到目标! Loss {avg_loss:.4f} < {target_loss}\n")
            break

    total_time = time.time() - start_time

    f.write("\n" + "="*80 + "\n")
    f.write("训练完成!\n")
    f.write("="*80 + "\n")
    f.write(f"总时间: {total_time:.1f}s ({total_time/60:.1f}分钟)\n")
    f.write(f"总批次: {total_batches}\n")
    f.write(f"吞吐量: {total_batches/total_time:.2f} batch/s\n")
    f.write(f"最终Loss: {avg_loss:.4f}\n")
    f.write(f"最佳Loss: {best_loss:.4f}\n")

    if avg_loss < target_loss:
        f.write(f"\n✅ 成功! 在 {epoch+1} 个epoch后达到目标Loss\n")
    else:
        f.write(f"\n⚠️  未达到目标Loss {target_loss}，当前 {avg_loss:.4f}\n")

    # VB统计
    f.write("\n" + "="*80 + "\n")
    f.write("Virtual Blackwell 统计\n")
    f.write("="*80 + "\n\n")

    old_stdout = sys.stdout
    sys.stdout = f
    try:
        wrapper.print_all_stats()
    except:
        pass
    sys.stdout = old_stdout

    f.write(f"\n" + "="*80 + "\n")
    f.write(f"使用 {vb_count} 张虚拟Blackwell显卡训练完成\n")
    f.write("="*80 + "\n")

print(f"Training complete! Check {output_file}")

# 读取并显示结果
with open(output_file, 'r') as f:
    content = f.read()

# 只显示最后50行避免太长
lines = content.split('\n')
if len(lines) > 50:
    print('\n'.join(lines[-50:]))
else:
    print(content)
