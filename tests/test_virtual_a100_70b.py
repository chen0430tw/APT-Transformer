"""
使用 Virtual A100 加载 70B GGUF 模型
====================================
"""
import os
import sys

_BASE = "D:/APT-Transformer"
sys.path.insert(0, f"{_BASE}/va100")

os.environ["TORCH_HOME"] = f"{_BASE}/.torch_cache"
os.environ["TEMP"] = f"{_BASE}/.temp"

import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB" if torch.cuda.is_available() else "")

# 检查 GGUF 模型
model_path = "D:/huihui-ai_DeepSeek-R1-Distill-Llama-70B-abliterated-Q4_0.gguf"
if os.path.exists(model_path):
    size_gb = os.path.getsize(model_path) / 1024**3
    print(f"\n模型文件: {model_path}")
    print(f"文件大小: {size_gb:.1f} GB")
else:
    print(f"错误: 模型文件不存在")
    sys.exit(1)

# 导入 virtual_a100
try:
    from virtual_a100 import VirtualA100Engine, GhostConfig, InferConfig
    print(f"\nVirtual A100 导入成功")

    # 配置
    config = GhostConfig(
        gpu_mem_gb=8.0,  # RTX 3070 Laptop
        cpu_mem_gb=16.0,
        use_disk=False,
    )

    infer_config = InferConfig(
        batch_size=1,
        seq_len=512,
    )

    print(f"配置:")
    print(f"  GPU 显存: {config.gpu_mem_gb} GB")
    print(f"  CPU 内存: {config.cpu_mem_gb} GB")
    print(f"  Batch size: {infer_config.batch_size}")
    print(f"  序列长度: {infer_config.seq_len}")

    # 创建引擎
    print(f"\n创建 Virtual A100 引擎...")
    engine = VirtualA100Engine(config)

    print(f"[SUCCESS] Virtual A100 引擎创建成功")

except ImportError as e:
    print(f"错误: {e}")
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
