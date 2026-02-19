# llama-cpp-python CUDA 编译指南 (Windows + RTX 3070 + Python 3.13 + CUDA 13.1)

## 📋 目录
- [概述](#概述)
- [系统环境](#系统环境)
- [问题分析](#问题分析)
- [编译步骤](#编译步骤)
- [使用方法](#使用方法)
- [故障排除](#故障排除)
- [关键发现](#关键发现)

---

## 概述

本文档记录了在 Windows 10/11 上为 **Python 3.13** 编译支持 **CUDA 13.1** 的 `llama-cpp-python` 的完整过程，适用于 **RTX 3070 (sm86/Ampere)** GPU。

**最终成果**：成功编译 `llama_cpp_python-0.3.16`，GPU 推理速度约 **207 tokens/秒**。

---

## 系统环境

### 硬件
- **GPU**: NVIDIA GeForce RTX 3070 Laptop GPU
- **Compute Capability**: 8.6 (sm86/Ampere)
- **GPU Memory**: 7114 MiB 可用

### 软件
- **操作系统**: Windows 10/11 (64-bit)
- **Python**: 3.13.x
- **CUDA**: 13.1 (v13.1.115)
- **Visual Studio**: 2022 BuildTools (MSVC 14.44)
- **Visual Studio**: 18 2026 Community (已安装但不用)

### 关键依赖
```
CMake >= 4.2.1
NVIDIA CUDA Toolkit 13.1
Visual Studio 2022 BuildTools with C++ tools
```

---

## 问题分析

### 1. 为什么需要自己编译？

**预编译 wheel 的问题**：
- [dougeeai/llama-cpp-python-wheels](https://github.com/dougeeai/llama-cpp-python-wheels) 只有 Python 3.12 的 sm86 版本
- Python 3.13 无法使用 Python 3.12 的 wheel
- 官方不提供 CUDA 版本的预编译包

### 2. 编译过程中的主要坑点

#### 坑点 #1: 多 CUDA 版本冲突
```
系统存在:
- C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\
- C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\

问题: CMake 默认选择 v12.3，但 v12.3 没有 VS 2022 Integration
```

**解决方案**: 编译时使用 `CMAKE_GENERATOR=Visual Studio 17 2022` 强制使用 VS 2022

#### 坑点 #2: 多 Visual Studio 版本
```
系统存在:
- Visual Studio 18 2026 Community
- Visual Studio 2022 BuildTools

问题: CMake 默认选择 VS 18 2026，但 CUDA Integration 只为 VS 2022 安装
```

**解决方案**: 设置 `CMAKE_GENERATOR=Visual Studio 17 2022`

#### 坑点 #3: 隐藏的环境变量 (最关键的发现！)
```xml
<!-- CUDA 13.1.Version.props 文件内容 -->
<PropertyGroup>
    <CudaToolkitVersionedPath>$(CUDA_PATH_V13_1)</CudaToolkitVersionedPath>
</PropertyGroup>
```

**问题**:
- 编译时需要 `CUDA_PATH_V13_1` 环境变量（非标准！）
- 运行时需要 `CUDA_PATH` 环境变量（标准！）
- **两者是不同的变量名！**

#### 坑点 #4: CUDA DLL 路径
```
CUDA 13.1 的 DLL 在: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\
但 llama-cpp-python 只搜索: %CUDA_PATH%\bin
```

**解决方案**: 运行时需要同时添加 `bin` 和 `bin\x64` 到 DLL 搜索路径

---

## 编译步骤

### 步骤 1: 准备环境

确保已安装：
- [ ] CUDA Toolkit 13.1
- [ ] Visual Studio 2022 BuildTools (含 "Desktop development with C++")
- [ ] Python 3.13
- [ ] Git (用于克隆源码)

### 步骤 2: 创建编译脚本

创建 `build_llama_cpp_python.bat`:

```batch
@echo off
chcp 65001 >nul
echo ============================================================
echo Building llama-cpp-python with CUDA 13.1 + Python 3.13
echo ============================================================
echo.

REM === 关键环境变量 ===
REM 编译时用的变量 (VS Integration 需要)
set CUDA_PATH_V13_1=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1

REM 运行时用的变量 (llama-cpp-python 需要)
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1

REM CMake 配置
set CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86
set FORCE_CMAKE=1
set GGML_CUDA=1

REM 强制使用 VS 2022 (重要！)
set CMAKE_GENERATOR=Visual Studio 17 2022

REM 添加到 PATH
set PATH=%CUDA_PATH%\bin;%PATH%

echo Environment:
echo   CUDA_HOME=%CUDA_HOME%
echo   CMAKE_ARGS=%CMAKE_ARGS%
echo   CMAKE_GENERATOR=%CMAKE_GENERATOR%
echo.

REM === 检查 CUDA ===
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvcc.exe" --version
echo.

REM === 卸载旧版本 ===
pip uninstall llama-cpp-python -y 2>nul
echo.

REM === 开始编译 ===
echo Compiling... (this takes 5-10 minutes)
echo.
pip install llama-cpp-python --no-cache-dir --force-reinstall -vvv

echo.
echo ============================================================
echo Build complete!
echo ============================================================
pause
```

### 步骤 3: 运行编译

以**普通用户权限**运行：
```cmd
build_llama_cpp_python.bat
```

等待 5-10 分钟，编译成功后会看到：
```
Successfully built llama-cpp-python
Successfully installed llama-cpp-python-0.3.16
```

**关键日志**（确认 CUDA 被正确使用）：
```
-- Found CUDAToolkit: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include (found version "13.1.115")
-- CUDA Toolkit found
-- Using CUDA architectures: 86
```

---

## 使用方法

### Python 代码中使用

```python
import os
import sys

# === 必须在使用 llama_cpp 之前设置 ===
if sys.platform == 'win32':
    cuda_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1'

    # 关键：设置 CUDA_PATH (运行时需要)
    os.environ['CUDA_PATH'] = cuda_path

    # 添加 DLL 搜索路径
    os.add_dll_directory(cuda_path + r'\bin')
    os.add_dll_directory(cuda_path + r'\bin\x64')  # CUDA 13.1 的 DLL 在这里！

# 现在可以安全导入
from llama_cpp import Llama

# 加载模型（所有层都放到 GPU）
llm = Llama(
    model_path="path/to/your/model.gguf",
    n_gpu_layers=-1,  # -1 表示所有层都使用 GPU
    n_ctx=4096,
    verbose=True  # 设为 True 可以看到 GPU 使用情况
)

# 推理
output = llm("Q: What is AI?\nA:", max_tokens=50)
print(output['choices'][0]['text'])
```

### 验证 GPU 是否工作

运行上述代码后，查看日志中的关键输出：

```
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3070 Laptop GPU, compute capability 8.6, VMM: yes
llama_model_load_from_file_impl: using device CUDA0 (NVIDIA GeForce RTX 3070 Laptop GPU) - 7114 MiB free
```

如果看到 `using device CUDA0`，说明 GPU 加速成功！

### 性能测试结果

```
模型: Llama-3.2-1B-Instruct-Q4_0.gguf
GPU: NVIDIA GeForce RTX 3070 Laptop GPU

性能指标:
  prompt eval time: 13.66 ms per token (73.22 tokens/s)
  eval time: 4.83 ms per token (207.19 tokens/s)
  total time: 302.36 ms / 42 tokens
```

---

## 故障排除

### 问题 1: 编译时出现 "No CUDA toolset found"

**症状**:
```
error : The CUDA Toolkit directory '' does not exist.
```

**原因**:
- `CUDA_PATH_V13_1` 环境变量未设置
- 或 CMake 选择了错误的 CUDA 版本

**解决方案**:
1. 确认设置了 `CUDA_PATH_V13_1`
2. 强制使用 VS 2022: `set CMAKE_GENERATOR=Visual Studio 17 2022`

### 问题 2: 运行时出现 "Could not find module 'llama.dll'"

**症状**:
```
RuntimeError: Failed to load shared library 'llama.dll': Could not find module
```

**原因**: CUDA DLL 不在 DLL 搜索路径中

**解决方案**:
```python
import os
cuda_path = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1'
os.environ['CUDA_PATH'] = cuda_path
os.add_dll_directory(cuda_path + r'\bin')
os.add_dll_directory(cuda_path + r'\bin\x64')  # 重要！CUDA 13.1 特有
```

### 问题 3: GPU 内存不足

**症状**:
```
CUDA error: out of memory
```

**解决方案**:
```python
# 减少 n_ctx (上下文长度)
llm = Llama(model_path="...", n_ctx=2048)  # 默认 4096

# 或减少 GPU 层数
llm = Llama(model_path="...", n_gpu_layers=20)  # 而不是 -1
```

### 问题 4: CMake 找到了错误的 CUDA 版本

**症状**:
```
-- Found CUDAToolkit: .../CUDA/v12.3/include (found version "12.3.107")
```

**原因**: 多 CUDA 版本冲突

**解决方案**:
```batch
REM 方案 1: 临时重命名 v12.3 目录（需管理员权限）
ren "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3" v12.3.hidden

REM 方案 2: 强制 CMake 使用 VS 2022 (更安全)
set CMAKE_GENERATOR=Visual Studio 17 2022
```

---

## 关键发现

### 发现 1: 非标准的环境变量

**CUDA 13.1 的 VS Integration 使用非标准环境变量**:
```xml
<!-- 文件: CUDA 13.1.Version.props -->
<CudaToolkitVersionedPath>$(CUDA_PATH_V13_1)</CudaToolkitVersionedPath>
```

这与标准变量 `CUDA_PATH` 不同，导致大量用户编译失败。

### 发现 2: 版本匹配陷阱

| 组件 | 版本 | 要求 |
|------|------|------|
| CUDA | 13.1 | 需要 VS 2022 Integration |
| VS | 2022 BuildTools | 必须强制使用 |
| Python | 3.13 | 无预编译 wheel，必须自编译 |
| GPU | RTX 3070 | sm86 (Ampere) 架构 |

**VS 18 2026 不能用**，即使它更新，因为 CUDA 13.1 没有为它提供 Integration。

### 发现 3: DLL 搜索路径的细微差别

```
CUDA 12.x: DLL 在 bin\
CUDA 13.1: DLL 在 bin\x64\
```

运行时必须两个路径都添加：
```python
os.add_dll_directory(cuda_path + r'\bin')
os.add_dll_directory(cuda_path + r'\bin\x64')
```

### 发现 4: 编译时 vs 运行时的环境变量

| 时间点 | 需要的变量 | 用途 |
|--------|-----------|------|
| 编译时 | `CUDA_PATH_V13_1` | VS Integration 查找 CUDA |
| 运行时 | `CUDA_PATH` | llama-cpp-python 查找 CUDA DLL |

**必须使用不同的变量名！**

---

## 文件清单

编译脚本：
- `build_llama_cpp_python.bat` - 主编译脚本
- `test_gpu_final_v3.py` - GPU 测试脚本

运行时配置：
- `ai_chatroom.py` - 已集成 CUDA 环境设置

关键位置：
```
CUDA 安装: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\
VS Integration: C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations\
Python 包: %LOCALAPPDATA%\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\llama_cpp\
```

---

## 参考资料

- [llama-cpp-python GitHub](https://github.com/abetlen/llama-cpp-python)
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [Medium: llama-cpp-python with CUDA support on Windows 11](https://medium.com/@eddieoffermann/llama-cpp-python-with-cuda-support-on-windows-11-51a4dd295b25)
- [Stack Overflow: llama-cpp-python not using NVIDIA GPU CUDA](https://stackoverflow.com/questions/76963311/llama-cpp-python-not-using-nvidia-gpu-cuda)
- [dougeeai/llama-cpp-python-wheels](https://github.com/dougeeai/llama-cpp-python-wheels) (预编译 wheel)

---

## 更新日志

**2026-02-11**
- ✅ 成功编译 llama-cpp-python 0.3.16 with CUDA 13.1
- ✅ 确认 RTX 3070 GPU 工作正常
- ✅ 性能测试: 207 tokens/秒
- ✅ 创建本文档

---

## 贡献者

- 编译和测试: [Your Name]
- Agent 辅助: Claude (Anthropic)

**关键词**: `llama-cpp-python`, `CUDA 13.1`, `Windows`, `RTX 3070`, `Python 3.13`, `sm86`, `Ampere`
