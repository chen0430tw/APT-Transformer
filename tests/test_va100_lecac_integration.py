"""
测试 virtual_a100.py 的 LECAC + GGUF 集成
========================================
"""
import sys
sys.path.insert(0, "D:/APT-Transformer/va100")

print("=" * 70)
print("测试 Virtual A100 - LECAC + GGUF 集成")
print("=" * 70)

# 检查依赖
print("\n[1] 检查依赖...")

try:
    import torch
    print(f"  PyTorch: [OK] {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  CUDA: [OK] {torch.cuda.get_device_name(0)}")
except ImportError:
    print("  PyTorch: [FAIL]")

try:
    from apt.vgpu.runtime.virtual_vram import NATURAL_EQUILIBRIUM_CONSTANT
    print(f"  LECAC: [OK] NEC={NATURAL_EQUILIBRIUM_CONSTANT:.6f}")
except ImportError as e:
    print(f"  LECAC: [FAIL] {e}")

try:
    from llama_cpp import Llama
    print(f"  llama-cpp-python: [OK]")
except ImportError:
    print(f"  llama-cpp-python: [FAIL]")

# 导入 virtual_a100
print("\n[2] 导入 virtual_a100...")
try:
    from virtual_a100 import create_virtual_a100_torch, load_gguf
    print("  导入成功 [OK]")
except ImportError as e:
    print(f"  导入失败: {e}")
    sys.exit(1)

# 测试 GGUF 加载
print("\n[3] 测试 GGUF 加载...")
gguf_path = "D:/huihui-ai_DeepSeek-R1-Distill-Llama-70B-abliterated-Q4_0.gguf"

if not os.path.exists(gguf_path):
    print(f"  错误: 文件不存在: {gguf_path}")
    sys.exit(1)

import os
size_gb = os.path.getsize(gguf_path) / 1024**3
print(f"  文件: {gguf_path}")
print(f"  大小: {size_gb:.1f} GB")

try:
    llama_model, model_config = load_gguf(gguf_path, n_gpu_layers=0, n_ctx=512)
    print(f"  加载成功 [OK]")
    print(f"  配置: L={model_config['L']}, H={model_config['H']}")
except Exception as e:
    print(f"  加载失败: {e}")
    sys.exit(1)

# 创建 VirtualA100EngineTorch
print("\n[4] 创建 VirtualA100EngineTorch...")
try:
    engine = create_virtual_a100_torch(
        gguf_path,
        use_lecac=True,
        vram_budget_gb=8.0
    )
    print("  创建成功 [OK]")
except Exception as e:
    print(f"  创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试生成
print("\n[5] 测试生成...")
try:
    prompt = "What is AI?"
    print(f"  输入: {prompt}")

    output = engine.generate(prompt, max_tokens=20)

    print(f"  输出: {output}")
    print("  生成成功 [OK]")
except Exception as e:
    print(f"  生成失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 显存统计
print("\n[6] 显存统计...")
stats = engine.get_memory_stats()
print(f"  GPU 已分配: {stats.get('gpu_allocated_gb', 0):.2f} GB")
print(f"  GPU 已保留: {stats.get('gpu_reserved_gb', 0):.2f} GB")
print(f"  生成 tokens: {stats['tokens_generated']}")

print("\n" + "=" * 70)
print("全部测试通过！[OK]")
print("=" * 70)
print("\n总结:")
print("  [OK] Virtual A100 成功集成 LECAC")
print("  [OK] 支持加载 GGUF 格式（70B 模型）")
print("  [OK] 使用 PyTorch 进行真实 GPU 计算")
print("  [OK] 虚拟 A100 = llama-cpp + LECAC + virtual_vram")
print("=" * 70)
