import torch
import torch.nn as nn
import time
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

output_file = os.path.join(os.getcwd(), 'vb_test_output.txt')
with open(output_file, 'w') as f:
    f.write(f"=== Virtual Blackwell Training Test ===\n")
    f.write(f"Device: {device}\n\n")

    # Test 1: Standard PyTorch
    f.write("Test 1: Standard PyTorch\n")
    f.write("-" * 50 + "\n")

    model = nn.Sequential(
        nn.Linear(512, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 512)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    start = time.time()

    for i in range(100):
        x = torch.randn(32, 512).to(device)
        y = model(x)
        loss = y.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    time_standard = time.time() - start
    f.write(f"Time: {time_standard:.3f}s\n")
    f.write(f"Throughput: {100/time_standard:.2f} iter/s\n\n")

    # Test 2: Virtual Blackwell
    f.write("Test 2: Virtual Blackwell (FP4)\n")
    f.write("-" * 50 + "\n")

    from apt.vgpu.runtime.vb_integration import enable_vb_optimization

    model_vb = nn.Sequential(
        nn.Linear(512, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 512)
    ).to(device)

    # Redirect VB print to file
    import sys
    old_stdout = sys.stdout
    sys.stdout = f

    model_vb = enable_vb_optimization(
        model_vb,
        mode='training',
        enable_fp4=True,
        enable_quantization=False
    )

    sys.stdout = old_stdout

    optimizer_vb = torch.optim.Adam(model_vb.parameters())
    start = time.time()

    for i in range(100):
        x = torch.randn(32, 512).to(device)
        y = model_vb(x)
        loss = y.mean()
        loss.backward()
        optimizer_vb.step()
        optimizer_vb.zero_grad()

    time_vb = time.time() - start
    f.write(f"Time: {time_vb:.3f}s\n")
    f.write(f"Throughput: {100/time_vb:.2f} iter/s\n\n")

    # Summary
    speedup = time_standard / time_vb
    f.write("=" * 50 + "\n")
    f.write("SUMMARY\n")
    f.write("=" * 50 + "\n")
    f.write(f"Standard PyTorch: {time_standard:.3f}s\n")
    f.write(f"Virtual Blackwell: {time_vb:.3f}s\n")
    f.write(f"Speedup: {speedup:.2f}x\n")

    if speedup > 1:
        f.write(f"✅ VB is {(speedup-1)*100:.1f}% FASTER\n")
    else:
        f.write(f"⚠️  VB is {(1-speedup)*100:.1f}% SLOWER\n")
        f.write(f"(FP4 overhead > benefit for small model)\n")

print(f"Test complete. See {output_file}")
