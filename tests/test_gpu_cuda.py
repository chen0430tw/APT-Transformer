"""
llama-cpp-python GPU CUDA 综合测试
===================================
整合所有 llama-cpp-python GPU 加速测试到单一入口

测试模式：
  simple    - 最小可用测试（1B 模型 + 基本推理）
  check     - 详细检查（7B 模型 + 显存分析 + 性能评估）
  final_v3  - 完整测试（CUDA 13.1 路径 + 详细验证）

使用示例：
  python test_gpu_cuda.py --mode simple
  python test_gpu_cuda.py --mode check
  python test_gpu_cuda.py --mode final_v3 --cuda-path "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1"
"""

import argparse
import sys
import os
import time
import torch
from llama_cpp import Llama


def setup_cuda_path(cuda_path):
    """设置 CUDA 路径（Windows）"""
    if sys.platform == 'win32' and cuda_path:
        os.environ['CUDA_PATH'] = cuda_path
        os.add_dll_directory(cuda_path + r'\bin')
        os.add_dll_directory(cuda_path + r'\bin\x64')
        print(f"[CUDA] 设置路径: {cuda_path}")


def run_simple_test(model_path, cuda_path=None):
    """
    Mode: simple - 最小可用测试
    原文件：test_gpu_simple.py
    """
    print("\n" + "=" * 70)
    print("Mode: SIMPLE - 最小可用测试")
    print("=" * 70)

    setup_cuda_path(cuda_path)

    print(f"\n[1/3] 加载模型...")
    print(f"  模型路径: {model_path}")
    llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=512, verbose=True)
    print(f"[OK] GPU layers: {llm.n_gpu_layers}")

    print("\n[2/3] 运行推理...")
    output = llm("Q: What is AI?\nA:", max_tokens=50, stop=["\n"], temperature=0.0)
    print(f"Output: {output['choices'][0]['text']}")

    print("\n[3/3] 完成！")
    print("=" * 70)


def run_check_test(model_path, cuda_path=None):
    """
    Mode: check - 详细检查（显存分析 + 性能评估）
    原文件：test_gpu_acceleration.py
    """
    print("\n" + "=" * 70)
    print("Mode: CHECK - 详细检查（显存 + 性能）")
    print("=" * 70)

    setup_cuda_path(cuda_path)

    # 检查 CUDA 是否可用
    print(f"\n[CUDA] PyTorch CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[CUDA] GPU 设备: {torch.cuda.get_device_name(0)}")
        print(f"[CUDA] GPU 总显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"[CUDA] 当前显存占用: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\n" + "=" * 70)
    print(f"加载模型: {os.path.basename(model_path)}")
    print("=" * 70)

    # 加载模型，启用 GPU 加速
    print("\n[设置] n_gpu_layers=-1 (所有层卸载到 GPU)")
    start_load = time.time()
    llm = Llama(
        model_path=model_path,
        n_ctx=8192,
        n_gpu_layers=-1,  # 关键参数：所有层都放到 GPU
        verbose=True
    )
    load_time = time.time() - start_load

    print(f"\n[OK] 模型加载完成 (耗时 {load_time:.1f} 秒)")

    # 再次检查 GPU 显存
    print("\n" + "=" * 70)
    print("GPU 显存状态:")
    print("=" * 70)
    if torch.cuda.is_available():
        gpu_used = torch.cuda.memory_allocated() / 1e9
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_percent = (gpu_used / gpu_total) * 100
        print(f"已用显存: {gpu_used:.2f} GB / {gpu_total:.2f} GB ({gpu_percent:.1f}%)")

        if gpu_used > 1.0:
            print("\n[SUCCESS] GPU 显存占用 > 1GB，模型已加载到 GPU！")
        else:
            print("\n[WARNING] GPU 显存占用 < 1GB，可能仍在使用 CPU")

    # 测试推理速度
    print("\n" + "=" * 70)
    print("测试推理速度...")
    print("=" * 70)

    test_prompt = "AI 是否有真正的意识？请简要回答。"
    print(f"\n提示词: {test_prompt}")

    start_inference = time.time()
    output = llm(test_prompt, max_tokens=100, temperature=0.7, echo=False)
    inference_time = time.time() - start_inference

    response = output['choices'][0]['text']
    tokens = len(response)

    print(f"\n回复: {response}")
    print(f"\n推理耗时: {inference_time:.2f} 秒")
    print(f"生成 tokens: {tokens}")
    print(f"推理速度: {tokens/inference_time:.1f} tok/s")

    # 性能判断
    print("\n" + "=" * 70)
    print("性能评估:")
    print("=" * 70)
    if tokens / inference_time > 30:
        print("[SUCCESS] 推理速度 > 30 tok/s，GPU 加速已生效！")
    elif tokens / inference_time > 15:
        print("[PARTIAL] 推理速度 15-30 tok/s，部分使用 GPU")
    else:
        print("[WARNING] 推理速度 < 15 tok/s，可能仍在使用 CPU")
        print("         预期 GPU 速度应该 > 30 tok/s")

    print("\n" + "=" * 70)


def run_final_v3_test(model_path, cuda_path=None):
    """
    Mode: final_v3 - 完整测试（CUDA 13.1 路径 + 详细验证）
    原文件：test_gpu_final_v3.py
    """
    print("\n" + "=" * 70)
    print("Mode: FINAL_V3 - 完整测试（CUDA 13.1 + 详细验证）")
    print("=" * 70)

    setup_cuda_path(cuda_path)

    print(f"\n[1/2] 加载模型 (watch for 'using device CUDA0')...")
    print(f"  模型路径: {model_path}")
    llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=512, verbose=True)
    print("[OK] 模型加载完成")

    print("\n[2/2] 运行推理...")
    output = llm("Q: What is AI?\nA:", max_tokens=50, stop=["\n"], temperature=0.0)
    print(f"Output: {output['choices'][0]['text']}")

    print("\n" + "=" * 60)
    print("测试完成！如果看到 'using device CUDA0' 说明 GPU 正在工作")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="llama-cpp-python GPU CUDA 综合测试",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--mode", type=str,
                        choices=["simple", "check", "final_v3"],
                        default="simple",
                        help="测试模式（默认: simple）")
    parser.add_argument("--model", type=str,
                        default='C:/Users/asus/AppData/Local/nomic.ai/GPT4All/Llama-3.2-1B-Instruct-Q4_0.gguf',
                        help="GGUF 模型路径")
    parser.add_argument("--model-7b", type=str,
                        default='C:/Users/asus/AppData/Local/nomic.ai/GPT4All/DeepSeek-R1-Distill-Qwen-7B-Q4_0.gguf',
                        help="7B 模型路径（用于 check 模式）")
    parser.add_argument("--cuda-path", type=str,
                        default='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1',
                        help="CUDA 路径（Windows）")

    args = parser.parse_args()

    print("=" * 70)
    print("llama-cpp-python GPU CUDA 综合测试")
    print("=" * 70)
    print(f"Mode: {args.mode}")

    # 根据模式选择模型和测试函数
    if args.mode == "simple":
        run_simple_test(args.model, args.cuda_path)
    elif args.mode == "check":
        run_check_test(args.model_7b, args.cuda_path)
    elif args.mode == "final_v3":
        run_final_v3_test(args.model, args.cuda_path)

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
