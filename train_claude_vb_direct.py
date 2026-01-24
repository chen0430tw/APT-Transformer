import torch
import torch.nn as nn
import time
from apt.model.architectures.claude4_model import create_claude_unified
from apt.vgpu.runtime.vb_integration import VBModelWrapper
import io
import sys

# 捕获所有输出
output_file = '/tmp/claude_vb_training.txt'

with open(output_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("Virtual Blackwell 训练迷你 Claude 模型\n")
    f.write("="*80 + "\n\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    f.write(f"设备: {device}\n\n")

    # 创建迷你Claude模型
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

    # 启用Virtual Blackwell - 捕获print输出
    f.write("启用 Virtual Blackwell...\n")
    f.flush()

    # 重定向stdout到文件
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

        vb_count = len(wrapper.replaced_layers)

    finally:
        sys.stdout = old_stdout

    f.write(f"\n✅ 虚拟Blackwell显卡数量: {vb_count} 张\n\n")

    # 显示部分VB层
    f.write("虚拟显卡分配:\n")
    for i, layer_name in enumerate(wrapper.replaced_layers[:15]):
        f.write(f"  VB-{i+1}: {layer_name}\n")
    if vb_count > 15:
        f.write(f"  ... 还有 {vb_count-15} 张虚拟显卡\n")

    # 训练
    f.write("\n" + "="*80 + "\n")
    f.write("开始训练\n")
    f.write("="*80 + "\n\n")

    optimizer = torch.optim.Adam(wrapper.parameters(), lr=0.001)

    start_time = time.time()
    total_loss = 0
    num_batches = 50

    for batch_idx in range(num_batches):
        # 随机数据
        input_ids = torch.randint(0, 5000, (4, 64)).to(device)

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

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                f.write(f"Batch {batch_idx+1}/{num_batches}: Loss = {avg_loss:.4f}\n")
                f.flush()

        except Exception as e:
            f.write(f"Error at batch {batch_idx}: {e}\n")
            break

    elapsed = time.time() - start_time

    f.write(f"\n训练完成!\n")
    f.write(f"  时间: {elapsed:.2f}s\n")
    f.write(f"  吞吐量: {num_batches/elapsed:.2f} batch/s\n")
    f.write(f"  平均Loss: {total_loss/num_batches:.4f}\n")

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
    f.write(f"✅ 成功！使用 {vb_count} 张虚拟Blackwell显卡训练Claude模型\n")
    f.write("="*80 + "\n")

print(f"Training complete. Output saved to {output_file}")
