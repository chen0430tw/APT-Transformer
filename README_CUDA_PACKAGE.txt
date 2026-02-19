# llama-cpp-python CUDA 13.1 for Python 3.13 + RTX 3070

## 快速开始

### 1. 系统要求
- **操作系统**: Windows 10/11 (64-bit)
- **GPU**: NVIDIA RTX 3070 Laptop (sm86/Ampere)
- **Python**: 3.13.x
- **CUDA**: 13.1 (必须已安装)
- **Visual Studio**: 2022 BuildTools (已安装)

### 2. 安装

**方法 A: 使用预编译的包（推荐）**

```bash
# 解压后，直接安装已编译好的包
pip install llama_cpp_python_cuda_wheel/
```

**方法 B: 从源码编译**

如果需要重新编译，运行：
```bash
build_with_vs2022.bat
```

编译大约需要 5-10 分钟。

### 3. 使用示例

```python
import os
import sys

# 必须在使用 llama_cpp 之前设置
if sys.platform == 'win32':
    cuda_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1'
    os.environ['CUDA_PATH'] = cuda_path
    os.add_dll_directory(cuda_path + r'\bin')
    os.add_dll_directory(cuda_path + r'\bin\x64')

from llama_cpp import Llama

# 加载模型（所有层都使用 GPU）
llm = Llama(
    model_path="path/to/model.gguf",
    n_gpu_layers=-1,  # -1 = 所有层都用 GPU
    n_ctx=4096,
    verbose=False
)

# 推理
output = llm("Q: What is AI?\nA:", max_tokens=50)
print(output['choices'][0]['text'])
```

### 4. 验证 GPU 工作

运行测试脚本：
```bash
python test_gpu_final_v3.py
```

如果看到以下输出，说明 GPU 工作正常：
```
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3070 Laptop GPU
llama_model_load_from_file_impl: using device CUDA0
```

### 5. 性能

- **模型**: Llama-3.2-1B-Instruct-Q4_0.gguf
- **推理速度**: ~207 tokens/秒
- **内存使用**: ~7114 MiB GPU memory

### 6. 故障排除

#### 问题: "Could not find module 'llama.dll'"
**解决方案**: 确保设置了 CUDA_PATH 和 DLL 搜索路径
```python
import os
cuda_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1'
os.environ['CUDA_PATH'] = cuda_path
os.add_dll_directory(cuda_path + r'\bin')
os.add_dll_directory(cuda_path + r'\bin\x64')
```

#### 问题: GPU 没有被使用
**解决方案**: 检查是否设置了 `n_gpu_layers=-1`

#### 问题: CUDA 相关错误
**解决方案**: 确认 CUDA 13.1 已正确安装
```bash
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" --version
```

## 文件说明

- `llama_cpp_python_cuda_wheel/` - 编译好的 llama_cpp 包
- `test_gpu_final_v3.py` - GPU 测试脚本
- `ai_chatroom.py` - 已集成 CUDA 环境的示例应用
- `build_with_vs2022.bat` - 重新编译脚本
- `COMPILE_LLAMA_CPP_CUDA.md` - 完整编译文档

## 技术规格

```
版本: llama-cpp-python 0.3.16
CUDA: 13.1 (v13.1.115)
Compute Capability: 8.6 (sm86/Ampere)
架构: RTX 3070 Laptop GPU
Python: 3.13.x
编译器: MSVC 14.44 (VS 2022)
```

## 许可证

llama-cpp-python 遵循 MIT 许可证。

## 参考资料

- 完整编译文档: 见 `COMPILE_LLAMA_CPP_CUDA.md`
- 项目主页: https://github.com/abetlen/llama-cpp-python
- llama.cpp: https://github.com/ggerganov/llama.cpp

---

**编译日期**: 2026-02-11
**编译环境**: Windows 10/11 + CUDA 13.1 + VS 2022 + Python 3.13
**测试硬件**: NVIDIA GeForce RTX 3070 Laptop GPU
