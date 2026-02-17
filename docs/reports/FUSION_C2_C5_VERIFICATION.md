# C2-C5 融合验证表

## C2) VB Training Speed 融合

| 原始文件 | 核心功能 | 融合后命令 | 状态 |
|---------|---------|-----------|------|
| test_vb_training_speed.py | 200 batches 性能测试（硬编码） | `python test_vb_training_speed_v6_4.py --perf` | ✅ 已覆盖 |
| test_vb_training_speed_v6_2.py | v6.2 API 基准测试 | 封存（历史版本） | 📦 封存 |
| test_vb_training_speed_v6_4.py | 参数化测试（支持 --perf） | **保留作为主入口** | ✅ 主入口 |

**融合说明**：
- `test_vb_training_speed.py` 硬编码 200 batches → 被 v6_4 的 `--perf` 参数完全替代
- `test_vb_training_speed_v6_2.py` 使用旧 API (v6.2) → 封存作为历史版本
- `test_vb_training_speed_v6_4.py` 已参数化，支持 `--batches`, `--perf`, `--batch-size`, `--seq-len`

**操作**：
- ✅ 删除：`test_vb_training_speed.py`（功能已被覆盖）
- 📦 封存：`test_vb_training_speed_v6_2.py`（历史版本）
- ✅ 保留：`test_vb_training_speed_v6_4.py`（主入口）

---

## C3) VRAM 指标面板融合

| 原始文件 | 核心功能 | 融合后命令 | 状态 |
|---------|---------|-----------|------|
| test_vvram_peak_compare.py | 对比开/关虚拟显存的 backward 峰值 | `python test_vram_bench.py --mode peak` | ✅ 已覆盖 |
| test_vvram_compare.py | 统计不开虚拟显存时的 saved tensors | `python test_vram_bench.py --mode compare` | ✅ 已覆盖 |
| test_vvram_backward.py | 测试 forward 后/backward 前常驻显存 | `python test_vram_bench.py --mode backward` | ✅ 已覆盖 |
| test_oom_comparison.py | OOM 对比测试（寻找最大可用 batch size） | `python test_vram_bench.py --mode oom` | ✅ 已覆盖 |

**融合脚本**：`test_vram_bench.py`

**支持参数**：
- `--mode {peak,compare,backward,oom}`：测试模式
- `--batch-sizes`：batch size 列表（用于 peak/backward/oom 模式）
- `--device`：设备（默认 cuda）

**功能映射**：
1. **peak 模式** → `test_vvram_peak_compare.py`
   - 对比开/关虚拟显存的 backward 峰值
   - 支持多 batch size 测试
   - 输出 forward 节省、峰值增加、对比结果

2. **compare 模式** → `test_vvram_compare.py`
   - 使用 pack_hook_measure 统计 saved_tensors
   - 输出数量、总大小、平均值、占峰值比例

3. **backward 模式** → `test_vvram_backward.py`
   - 测试 forward 后常驻显存
   - 测试 backward 峰值增量
   - 支持多 batch size

4. **oom 模式** → `test_oom_comparison.py`
   - 寻找最大可用 batch size
   - 对比开/关虚拟显存的 OOM 阈值
   - 输出改善百分比

**操作**：
- ✅ 删除：4 个原始文件（功能 100% 覆盖）
- ✅ 保留：`test_vram_bench.py`（主入口）

---

## C4) torch.compile 融合

| 原始文件 | 核心功能 | 融合后命令 | 状态 |
|---------|---------|-----------|------|
| test_compile_quick.py | 超快速测试（5 batches × 4配置） | `python test_compile_smoke.py --mode quick` | ✅ 已覆盖 |
| test_compile_small.py | 使用小模型（2层）快速验证 | `python test_compile_smoke.py --mode small` | ✅ 已覆盖 |
| test_compile_step.py | 分步测试（更稳定） | `python test_compile_smoke.py --mode step` | ✅ 已覆盖 |
| test_compile_backends.py | 后端选择测试 | **保留独立** | 📌 保留 |

**融合脚本**：`test_compile_smoke.py`

**支持参数**：
- `--mode {quick,small,step}`：测试模式
- `--device`：设备（默认 cuda）

**功能映射**：
1. **quick 模式** → `test_compile_quick.py`
   - 测试 4 个配置：baseline/VB × 编译/未编译
   - 5 batches 快速测试
   - 输出时间对比、编译改善百分比

2. **small 模式** → `test_compile_small.py`
   - 使用 2 层小模型
   - 快速验证编译可用性
   - 捕获编译失败错误

3. **step 模式** → `test_compile_step.py`
   - 分步测试，避免一次编译太多模型
   - 更稳定，适合调试
   - 输出详细的 gap 分析

**保留独立**：
- `test_compile_backends.py`：后端选择事实记录，功能不同，保留

**操作**：
- ✅ 删除：3 个原始文件（功能 100% 覆盖）
- ✅ 保留：`test_compile_smoke.py`（主入口）+ `test_compile_backends.py`（独立）

---

## C5) GPU CUDA 融合

| 原始文件 | 核心功能 | 融合后命令 | 状态 |
|---------|---------|-----------|------|
| test_gpu_simple.py | 最小可用测试（1B 模型） | `python test_gpu_cuda.py --mode simple` | ✅ 已覆盖 |
| test_gpu_acceleration.py | 详细检查（7B + 显存 + 性能） | `python test_gpu_cuda.py --mode check` | ✅ 已覆盖 |
| test_gpu_final_v3.py | CUDA 13.1 路径 + 详细验证 | `python test_gpu_cuda.py --mode final_v3` | ✅ 已覆盖 |

**融合脚本**：`test_gpu_cuda.py`

**支持参数**：
- `--mode {simple,check,final_v3}`：测试模式
- `--model`：1B 模型路径（用于 simple/final_v3）
- `--model-7b`：7B 模型路径（用于 check）
- `--cuda-path`：CUDA 路径（Windows）

**功能映射**：
1. **simple 模式** → `test_gpu_simple.py`
   - 加载 1B 模型
   - 基本推理测试
   - 检查 GPU layers

2. **check 模式** → `test_gpu_acceleration.py`
   - 加载 7B 模型（DeepSeek-R1-Distill-Qwen-7B）
   - 显存状态分析（已用/总量/百分比）
   - 推理速度测试（tok/s）
   - 性能评估（>30 tok/s = GPU 加速生效）

3. **final_v3 模式** → `test_gpu_final_v3.py`
   - 设置 CUDA 13.1 路径
   - 检查 'using device CUDA0' 输出
   - 完整验证流程

**操作**：
- ✅ 删除：3 个原始文件（功能 100% 覆盖）
- ✅ 保留：`test_gpu_cuda.py`（主入口）

---

## 总结

✅ **所有 5 个融合任务已完成**

### 融合成果：
1. **C2 VB Training Speed**: 3 → 1（保留 v6_4，封存 v6_2，删除重复）
2. **C3 VRAM**: 4 → 1（test_vram_bench.py）
3. **C4 torch.compile**: 3 → 1（test_compile_smoke.py，保留 backends 独立）
4. **C5 GPU CUDA**: 3 → 1（test_gpu_cuda.py）

### 原始文件处理：
- **删除**：11 个（功能已被融合脚本 100% 覆盖）
- **保留**：4 个融合脚本（主入口）
- **封存**：1 个（v6_2 历史版本）
- **保留独立**：1 个（test_compile_backends.py）

### 使用对比：

**C2 之前**：
```bash
python test_vb_training_speed.py          # 硬编码 200 batches
python test_vb_training_speed_v6_4.py    # 20 batches
```
**C2 之后**：
```bash
python test_vb_training_speed_v6_4.py    # 默认 20 batches
python test_vb_training_speed_v6_4.py --perf    # 200 batches
python test_vb_training_speed_v6_4.py --batches 100  # 自定义
```

**C3 之前**：
```bash
python test_vvram_peak_compare.py
python test_vvram_compare.py
python test_vvram_backward.py
python test_oom_comparison.py
```
**C3 之后**：
```bash
python test_vram_bench.py --mode peak
python test_vram_bench.py --mode compare
python test_vram_bench.py --mode backward
python test_vram_bench.py --mode oom
```

**C4 之前**：
```bash
python test_compile_quick.py
python test_compile_small.py
python test_compile_step.py
```
**C4 之后**：
```bash
python test_compile_smoke.py --mode quick
python test_compile_smoke.py --mode small
python test_compile_smoke.py --mode step
```

**C5 之前**：
```bash
python test_gpu_simple.py
python test_gpu_acceleration.py
python test_gpu_final_v3.py
```
**C5 之后**：
```bash
python test_gpu_cuda.py --mode simple
python test_gpu_cuda.py --mode check
python test_gpu_cuda.py --mode final_v3
```

---

**结论：C2-C5 融合 100% 完成，可以安全删除原始文件。**
