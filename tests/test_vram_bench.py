"""
Virtual VRAM 综合指标测试
==========================
整合所有 VRAM 相关测试到单一入口

测试模式：
  peak     - 对比开/关虚拟显存的 backward 峰值（多 batch size）
  compare  - 统计不开虚拟显存时的 saved tensors 大小
  backward - 测试不开虚拟显存时的 forward 后/backward 前常驻显存
  oom      - OOM 对比测试（寻找最大可用 batch size）

使用示例：
  python test_vram_bench.py --mode peak
  python test_vram_bench.py --mode oom --batch-sizes 16 24 32 40 48
  python test_vram_bench.py --mode compare
"""

import argparse
import torch
import torch.nn as nn
from apt.vgpu.runtime.virtual_vram import VirtualVRAMConfig, virtual_vram


class TestModel(nn.Module):
    """标准测试模型：2层 MLP"""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1024, 4096)
        self.linear2 = nn.Linear(4096, 1024)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


def run_peak_test(device, batch_sizes):
    """
    Mode: peak - 对比开/关虚拟显存的 backward 峰值
    原文件：test_vvram_peak_compare.py
    """
    print("\n" + "=" * 70)
    print("Mode: PEAK - 对比开/关虚拟显存的 backward 峰值")
    print("=" * 70)

    seq_len = 2048

    for batch_size in batch_sizes:
        print(f"\n{'='*70}")
        print(f"Batch Size: {batch_size}")
        print(f"{'='*70}")

        model = TestModel().to(device)
        model.train()

        # 不开虚拟显存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        x = torch.randn(batch_size, seq_len, 1024, device=device)
        y = model(x)
        after_forward = torch.cuda.memory_allocated(device) / 1024**3
        y.mean().backward()
        torch.cuda.synchronize()
        peak_no_vvram = torch.cuda.max_memory_allocated(device) / 1024**3
        del x, y

        # 开虚拟显存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        x = torch.randn(batch_size, seq_len, 1024, device=device)
        cfg = VirtualVRAMConfig(enabled=True, min_tensor_bytes=1<<22, verbose=False)
        with virtual_vram(cfg):
            y = model(x)
            after_forward_v = torch.cuda.memory_allocated(device) / 1024**3
            y.mean().backward()
        torch.cuda.synchronize()
        peak_with_vvram = torch.cuda.max_memory_allocated(device) / 1024**3

        print(f"不开虚拟显存:")
        print(f"  Forward 后: {after_forward:.2f} GB")
        print(f"  峰值:      {peak_no_vvram:.2f} GB")
        print(f"  增量:      {peak_no_vvram - after_forward:.2f} GB")

        print(f"\n开虚拟显存:")
        print(f"  Forward 后: {after_forward_v:.2f} GB")
        print(f"  峰值:      {peak_with_vvram:.2f} GB")
        print(f"  增量:      {peak_with_vvram - after_forward_v:.2f} GB")

        print(f"\n对比:")
        print(f"  Forward 节省: {after_forward - after_forward_v:.2f} GB")
        print(f"  峰值增加:    {peak_with_vvram - peak_no_vvram:.2f} GB")
        print(f"  {'✅ 好' if peak_with_vvram <= peak_no_vvram else '❌ 差'}")


def run_compare_test(device):
    """
    Mode: compare - 统计不开虚拟显存时的 saved tensors 大小
    原文件：test_vvram_compare.py
    """
    print("\n" + "=" * 70)
    print("Mode: COMPARE - 统计不开虚拟显存时的 saved tensors")
    print("=" * 70)

    model = TestModel().to(device)
    model.train()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    batch_size = 16
    seq_len = 2048

    x = torch.randn(batch_size, seq_len, 1024, device=device)

    saved_tensors_memory = []
    saved_tensors_count = 0

    def pack_hook_measure(t):
        """统计每个 saved tensor 的大小"""
        if t.is_cuda:
            nbytes = t.numel() * t.element_size()
            saved_tensors_memory.append(nbytes)
            print(f"[Saved Tensor #{saved_tensors_count}] {nbytes/1024/1024:.2f} MB, dtype={t.dtype}, shape={tuple(t.shape)}")
            nonlocal saved_tensors_count
            saved_tensors_count += 1
        return t

    print("\n【Forward + Backward】")
    with torch.autograd.graph.saved_tensors_hooks(pack_hook_measure, lambda x: x):
        y = model(x)
        loss = y.mean()
        loss.backward()

    torch.cuda.synchronize()

    total_saved_mb = sum(saved_tensors_memory) / 1024**2
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3

    print(f"\n统计结果:")
    print(f"  Saved tensors 数量: {saved_tensors_count}")
    print(f"  Saved tensors 总大小: {total_saved_mb:.2f} MB")
    print(f"  平均每个: {total_saved_mb/saved_tensors_count if saved_tensors_count > 0 else 0:.2f} MB")
    print(f"  峰值显存: {peak_mem:.2f} GB")
    print(f"  Saved 占峰值比例: {total_saved_mb/1024/peak_mem*100:.1f}%")


def run_backward_test(device, batch_sizes):
    """
    Mode: backward - 测试不开虚拟显存时的 forward 后/backward 前常驻显存
    原文件：test_vvram_backward.py
    """
    print("\n" + "=" * 70)
    print("Mode: BACKWARD - 测试 forward 后/backward 前常驻显存")
    print("=" * 70)

    model = TestModel().to(device)
    model.train()

    for batch_size in batch_sizes:
        print(f"\n{'='*70}")
        print(f"Batch Size: {batch_size}")
        print(f"{'='*70}")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.reset_accumulated_memory_stats(device)

        seq_len = 2048
        x = torch.randn(batch_size, seq_len, 1024, device=device)

        # Forward
        y = model(x)
        torch.cuda.synchronize()
        after_forward = torch.cuda.memory_allocated(device) / 1024**3

        # Backward
        y.mean().backward()
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3

        print(f"  Forward 后常驻: {after_forward:.2f} GB")
        print(f"  峰值显存:      {peak_mem:.2f} GB")
        print(f"  Backward 峰值增量: {peak_mem - after_forward:.2f} GB")


def run_oom_test(device, batch_sizes):
    """
    Mode: oom - OOM 对比测试（寻找最大可用 batch size）
    原文件：test_oom_comparison.py
    """
    print("\n" + "=" * 70)
    print("Mode: OOM - 寻找最大可用 Batch Size")
    print("=" * 70)

    model = TestModel().to(device)
    model.train()

    def test_batch(batch_size, seq_len, use_virtual_vram=False):
        """测试给定的 batch size 是否会 OOM"""
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            x = torch.randn(batch_size, seq_len, 1024, device=device)

            if use_virtual_vram:
                cfg = VirtualVRAMConfig(
                    enabled=True,
                    offload="cpu",
                    min_tensor_bytes=1<<20,
                    pin_memory=True,
                    non_blocking=True,
                    verbose=False,
                )
                with virtual_vram(cfg):
                    y = model(x)
                    loss = y.mean()
                    loss.backward()
            else:
                y = model(x)
                loss = y.mean()
                loss.backward()

            torch.cuda.synchronize()
            peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3

            return True, peak_mem

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return False, 0
            else:
                raise

    seq_len = 2048

    print("\n" + "=" * 70)
    print("Test 1: 不使用虚拟显存")
    print("=" * 70)
    max_batch_no_vram = 0
    max_mem_no_vram = 0
    for bs in batch_sizes:
        success, peak_mem = test_batch(bs, seq_len, use_virtual_vram=False)
        status = "✅" if success else "❌ OOM"
        print(f"  Batch {bs:2d}: {status}  峰值显存: {peak_mem:.2f} GB" if success else f"  Batch {bs:2d}: {status}  Out of Memory")
        if success:
            max_batch_no_vram = bs
            max_mem_no_vram = peak_mem
        else:
            break

    print(f"\n>>> 不使用虚拟显存: 最大 Batch Size = {max_batch_no_vram}, 峰值显存 = {max_mem_no_vram:.2f} GB")

    print("\n" + "=" * 70)
    print("Test 2: 使用虚拟显存")
    print("=" * 70)
    max_batch_with_vram = 0
    max_mem_with_vram = 0
    for bs in batch_sizes:
        success, peak_mem = test_batch(bs, seq_len, use_virtual_vram=True)
        status = "✅" if success else "❌ OOM"
        print(f"  Batch {bs:2d}: {status}  峰值显存: {peak_mem:.2f} GB" if success else f"  Batch {bs:2d}: {status}  Out of Memory")
        if success:
            max_batch_with_vram = bs
            max_mem_with_vram = peak_mem
        else:
            break

    print(f"\n>>> 使用虚拟显存: 最大 Batch Size = {max_batch_with_vram}, 峰值显存 = {max_mem_with_vram:.2f} GB")

    print("\n" + "=" * 70)
    print("FINAL RESULTS - OOM 对比")
    print("=" * 70)
    print(f"不使用虚拟显存: 最大 Batch Size = {max_batch_no_vram}")
    print(f"使用虚拟显存:   最大 Batch Size = {max_batch_with_vram}")

    if max_batch_with_vram > max_batch_no_vram:
        improvement = max_batch_with_vram - max_batch_no_vram
        improvement_pct = (improvement / max_batch_no_vram) * 100
        print(f"\n>>> SUCCESS! 虚拟显存让 Batch Size 增加了 {improvement} ({improvement_pct:.1f}%)")
        print(f">>> 不使用虚拟显存: 最多 {max_batch_no_vram} 样就会 OOM")
        print(f">>> 使用虚拟显存后:   可以跑到 {max_batch_with_vram} 样")
    elif max_batch_with_vram == max_batch_no_vram:
        print(f"\n>>> INFO: 两种情况都支持到 Batch Size = {max_batch_no_vram}")
        print(f">>> 你的 GPU 内存足够大，或者模型太小，虚拟显存优势不明显")
    else:
        print(f"\n>>> WARNING: 虚拟显存反而限制了 Batch Size (不应该发生)")
        print(f">>> 可能是 PCIe 传输瓶颈或其他问题")


def main():
    parser = argparse.ArgumentParser(
        description="Virtual VRAM 综合指标测试",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--mode", type=str,
                        choices=["peak", "compare", "backward", "oom"],
                        default="peak",
                        help="测试模式（默认: peak）")
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        default=[16, 32, 40],
                        help="Batch size 列表（用于 peak/backward/oom 模式）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="设备（默认: cuda）")

    args = parser.parse_args()

    device = args.device
    print("=" * 70)
    print("Virtual VRAM 综合指标测试")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Mode: {args.mode}")

    # 根据模式运行测试
    if args.mode == "peak":
        run_peak_test(device, args.batch_sizes)
    elif args.mode == "compare":
        run_compare_test(device)
    elif args.mode == "backward":
        run_backward_test(device, args.batch_sizes)
    elif args.mode == "oom":
        run_oom_test(device, args.batch_sizes)

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
