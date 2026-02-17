"""
手动设置环境变量并测试 torch.compile
"""
import os
import sys
import subprocess

print("=" * 60)
print("尝试找到并设置 C 编译器")
print("=" * 60)

# 常见的 Visual Studio 路径
vs_paths = [
    r"C:\Program Files\Microsoft Visual Studio\2022",
    r"C:\Program Files (x86)\Microsoft Visual Studio\2022",
    r"C:\Program Files\Microsoft Visual Studio\2019",
    r"C:\Program Files (x86)\Microsoft Visual Studio\2019",
]

# 在这些路径下搜索 cl.exe
for base_path in vs_paths:
    print(f"\n搜索路径: {base_path}")
    if not os.path.exists(base_path):
        print(f"  [SKIP] 路径不存在")
        continue

    for root, dirs, files in os.walk(base_path):
        if "cl.exe" in files:
            cl_path = os.path.join(root, "cl.exe")
            print(f"  [FOUND] {cl_path}")

            # 设置环境变量
            os.environ['CC'] = cl_path
            os.environ['CXX'] = cl_path

            # 添加到 PATH
            os.environ['PATH'] = os.path.dirname(root) + os.pathsep + os.environ.get('PATH', '')

            # 测试编译器
            try:
                result = subprocess.run(['cl'], capture_output=True, timeout=5,
                                      env=os.environ)
                if result.returncode == 0:
                    print(f"  [OK] cl.exe 可用")
                else:
                    print(f"  [WARN] cl.exe 存在但无法运行")
            except Exception as e:
                print(f"  [ERROR] {e}")

            break
    else:
        print(f"  [NOT FOUND] cl.exe")

print(f"\n{'='*60}")
print("测试 PyTorch 编译")
print(f"{'='*60}")

# 现在导入 PyTorch 并测试
import torch
print(f"PyTorch: {torch.__version__}")

# 检查当前环境变量
print(f"\n当前环境变量:")
print(f"  CC: {os.environ.get('CC', '(not set)')}")
print(f"  CXX: {os.environ.get('CXX', '(not set)')}")

# 设置 PyTorch 环境变量（如果找到了编译器）
if 'CC' in os.environ:
    print(f"\n使用编译器: {os.environ['CC']}")
else:
    print("\n未找到编译器，跳过 torch.compile 测试")
    print("\n你可以手动设置编译器路径，例如:")
    print("  set CC=\"C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\bin\\cl.exe\"")
    print("  然后重新运行测试")
    sys.exit(0)

# 简单的 torch.compile 测试
print(f"\n{'='*60}")
print("简单 torch.compile 测试（只编译 1 层网络）")
print(f"{'='*60}")

import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# 创建一个非常简单的模型（1 层）
simple_model = nn.Linear(512, 512).to(device)

print("尝试编译...")
try:
    compiled_model = torch.compile(simple_model, mode="reduce-overhead")
    print("[SUCCESS] torch.compile 成功！")

    # 测试
    x = torch.randn(10, 512, device=device)
    y = compiled_model(x)
    print(f"[OK] 编译后的模型可以运行！Output shape: {y.shape}")

except Exception as e:
    print(f"[FAILED] torch.compile 失败:")
    print(f"  {type(e).__name__}")
    print(f"  {str(e)[:300]}")
    print(f"\n可能原因:")
    print(f"  1. 编译器版本不兼容")
    print(f"   2. 缺少必要的 SDK")
    print(f"  3. Python/PyTorch 版本不匹配")

print("=" * 60)
