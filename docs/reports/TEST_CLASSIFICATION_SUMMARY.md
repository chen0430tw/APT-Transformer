# 测试脚本分类总结

**日期**: 2026-02-16
**分类标准**: 四分法（要留/封存/融合/删除）
**原始数量**: 64 个测试脚本
**处理后数量**: 26 个核心文件 + 封存文件

---

## A) 要留的（核心/现役：保留在主路径）

**数量**: 26 个

### 1) LECAC 主线 (6个)
- ✅ `test_lecac.py` - 核心算法入口
- ✅ `test_lecac_fixed.py` - 修复版/对照基准
- ✅ `test_lecac_llm_finetuning.py` - 真实应用场景（LLM 微调）
- ✅ `test_lecac_vram_standalone.py` - LECAC × VRAM 组合验证
- ✅ `test_lecac_progressive.py` - 渐进式规模增长（回归测试）
- ✅ `test_lecac_quant.py` - INT2/INT4 量化测试（融合 C1）

### 2) Virtual Blackwell 主线 (5个)
- ✅ `test_vb_minimal.py` - 最小可跑的基准入口
- ✅ `test_vb_compile_final.py` - compile 集成最终版
- ✅ `test_vb_speed_simple.py` - 简化速度测试（回归用）
- ✅ `test_vb_training_speed_v6_4.py` - 训练速度测试（v6.4，支持 --perf）
- ✅ `test_gpt4all_lecac.py` - GPT4All + LECAC 集成
- ✅ `test_gpt4all_lecac_with_vram.py` - GPT4All + LECAC + VRAM 三重组合

### 3) Virtual VRAM 主线 (4个)
- ✅ `test_virtual_vram.py` - 基础功能入口
- ✅ `test_virtual_vram_simple.py` - 简单测试（快速验证）
- ✅ `test_vram_bench.py` - 综合指标测试（融合 C3）
- ✅ `test_vram_lecac_integration.py` - VRAM × LECAC 组合

### 4) Virtual A100 主线 (5个)
- ✅ `test_70b_virtual_a100.py` - 主文件
- ✅ `test_virtual_a100_70b.py` - 70B GGUF 加载验证
- ✅ `test_va100_small_model.py` - 小模型快速回归
- ✅ `test_va100_lecac_integration.py` - LECAC × VA100 组合
- ✅ `test_va100_simple.py` - VA100 简化测试
- ✅ `test_vcache_session.py` - KV cache + session（产品级功能）

### 5) torch.compile 主线 (3个)
- ✅ `test_compile_smoke.py` - 综合测试（融合 C4）
- ✅ `test_compile_backends.py` - 后端选择事实记录
- ✅ `test_find_compiler.py` - Windows 环境定位工具

### 6) GPU 加速主线 (1个)
- ✅ `test_gpu_cuda.py` - llama-cpp-python CUDA 综合测试（融合 C5）

---

## B) 封存的（有价值但退出主路径：放 archive/）

**数量**: 24 个

### 1) archive/failed_experiments/ (2个)
- 📦 `test_ldbr.py` - LDBR 失败案例
- 📦 `test_trace_ldbr.py` - LDBR 追踪失败案例

### 2) archive/early_versions/ (11个)
- 📦 `test_vb_training_speed_v6_2.py` - v6.2 版本（被 v6.4 替代）
- 📦 `test_int8_ste.py` - INT8 STE 早期版本（已整合）
- 📦 `test_refactored_vb.py` - VB 重构版本
- 📦 `test_shrinking_scale_cache_v6.py` - scale cache 实验
- 📦 `test_va100_numpy_only.py` - numpy 版本（非主线）
- 📦 `test_va100_sim.py` - 模拟版本
- 📦 `test_va100_direct.py` - 直接版本
- 📦 `test_vb_nvlink_simulation.py` - NVLink 模拟实验
- 📦 `test_vb_model_integration.py` - 模型集成早期版本
- 📦 `test_vb_training.py` - 训练脚本早期版本
- 📦 `test_vb_simple.py` - 简化版本早期版本
- 📦 `test_lecac_saved_tensors_hooks.py` - LECAC saved tensors hooks 实验（历史遗留）

### 3) archive/exploratory/ (2个)
- 📦 `test_ai_dialogue.py` - AI 对话探索性实验
- 📦 `test_gradient_flow.py` - 梯度流分析实验

### 4) archive/diagnostic/ (8个)
- 📦 `test_vvram_debug.py` - VRAM 调试脚本
- 📦 `test_va100_debug.py` - VA100 调试脚本
- 📦 `test_int8_debug.py` - INT8 调试脚本
- 📦 `test_trace_gradient.py` - 梯度追踪诊断
- 📦 `test_triton_check.py` - Triton 环境检查
- 📦 `test_triton_simple.py` - Triton 简单测试
- 📦 `test_oom_no_cache.py` - OOM 问题诊断
- 📦 `test_vb_debug.py` - VB 调试脚本

---

## C) 可融合的（已合并成更少入口）

**数量**: 4 个融合脚本（替代了 19 个原始文件）

### 1) C1: LECAC INT2/INT4 融合 ✅
**融合脚本**: `test_lecac_quant.py`
**原始文件** (8个 → 1):
- test_lecac_int2_4_over_e.py
- test_lecac_int2_alpha_sweep.py
- test_lecac_int2_orthogonal.py
- test_lecac_int2_stats.py
- test_lecac_int2_training.py
- test_lecac_int2_warmup.py
- test_lecac_int4.py
- test_lecac_int4_stats.py

**验证文档**: `LECAC_FUSION_VERIFICATION.md`

### 2) C2: VB Training Speed 融合 ✅
**融合脚本**: `test_vb_training_speed_v6_4.py`（已参数化）
**原始文件** (1个删除):
- test_vb_training_speed.py → 被 `--perf` 替代

### 3) C3: VRAM 指标面板融合 ✅
**融合脚本**: `test_vram_bench.py`
**原始文件** (4个 → 1):
- test_vvram_peak_compare.py → `--mode peak`
- test_vvram_compare.py → `--mode compare`
- test_vvram_backward.py → `--mode backward`
- test_oom_comparison.py → `--mode oom`

### 4) C4: torch.compile 融合 ✅
**融合脚本**: `test_compile_smoke.py`
**原始文件** (3个 → 1):
- test_compile_quick.py → `--mode quick`
- test_compile_small.py → `--mode small`
- test_compile_step.py → `--mode step`
- **保留独立**: test_compile_backends.py

### 5) C5: GPU CUDA 融合 ✅
**融合脚本**: `test_gpu_cuda.py`
**原始文件** (3个 → 1):
- test_gpu_simple.py → `--mode simple`
- test_gpu_acceleration.py → `--mode check`
- test_gpu_final_v3.py → `--mode final_v3`

**验证文档**: `FUSION_C2_C5_VERIFICATION.md`

---

## D) 不需要的（已删除）

**数量**: 11 个（通过融合删除）

### 删除清单
1. test_vb_training_speed.py（功能被 v6_4 --perf 替代）
2. test_vvram_peak_compare.py（功能被 test_vram_bench --mode peak 替代）
3. test_vvram_compare.py（功能被 test_vram_bench --mode compare 替代）
4. test_vvram_backward.py（功能被 test_vram_bench --mode backward 替代）
5. test_oom_comparison.py（功能被 test_vram_bench --mode oom 替代）
6. test_compile_quick.py（功能被 test_compile_smoke --mode quick 替代）
7. test_compile_small.py（功能被 test_compile_smoke --mode small 替代）
8. test_compile_step.py（功能被 test_compile_smoke --mode step 替代）
9. test_gpu_simple.py（功能被 test_gpu_cuda --mode simple 替代）
10. test_gpu_acceleration.py（功能被 test_gpu_cuda --mode check 替代）
11. test_gpu_final_v3.py（功能被 test_gpu_cuda --mode final_v3 替代）

---

## 统计总结

| 分类 | 数量 | 说明 |
|------|------|------|
| **A) 要留（核心/现役）** | 26 | 保留在主路径 |
| **B) 封存（退出主路径）** | 24 | 移至 archive/ 子目录 |
| **C) 可融合（已合并）** | 4 | 融合脚本（替代 19 个原始文件） |
| **D) 不需要（已删除）** | 11 | 通过融合删除的原始文件 |
| **原始总数** | 64 | |
| **处理后主路径** | 30 | 26 核心 + 4 融合脚本 |
| **archive/** | 24 | 封存文件 |
| **净减少** | 34 | 64 → 30 （减少 53%） |

---

## 目录结构

```
D:\APT-Transformer\
├── test*.py (30个核心文件)
│   ├── LECAC (6个)
│   ├── Virtual Blackwell (6个)
│   ├── Virtual VRAM (4个)
│   ├── Virtual A100 (6个)
│   ├── torch.compile (3个)
│   └── GPU 加速 (1个)
│
├── archive/
│   ├── failed_experiments/ (2个)
│   ├── early_versions/ (11个)
│   ├── exploratory/ (2个)
│   └── diagnostic/ (8个)
│
├── LECAC_FUSION_VERIFICATION.md
└── FUSION_C2_C5_VERIFICATION.md
```

---

## 使用指南

### LECAC 测试
```bash
# 核心算法
python test_lecac.py

# 量化测试（INT2/INT4）
python test_lecac_quant.py --bits 2 --mode stats
python test_lecac_quant.py --bits 4 --mode training

# LLM 微调
python test_lecac_llm_finetuning.py
```

### Virtual Blackwell 测试
```bash
# 最小基准
python test_vb_minimal.py

# 训练速度
python test_vb_training_speed_v6_4.py              # 快速测试（20 batches）
python test_vb_training_speed_v6_4.py --perf       # 性能测试（200 batches）

# Compile 集成
python test_vb_compile_final.py
```

### Virtual VRAM 测试
```bash
# 基础功能
python test_virtual_vram.py
python test_virtual_vram_simple.py

# 综合指标
python test_vram_bench.py --mode peak
python test_vram_bench.py --mode oom
python test_vram_bench.py --mode compare
python test_vram_bench.py --mode backward
```

### Virtual A100 测试
```bash
# 主文件
python test_70b_virtual_a100.py

# 小模型回归
python test_va100_small_model.py

# KV cache session
python test_vcache_session.py
```

### torch.compile 测试
```bash
# 综合测试
python test_compile_smoke.py --mode quick
python test_compile_smoke.py --mode small
python test_compile_smoke.py --mode step

# 后端选择
python test_compile_backends.py
```

### GPU CUDA 测试
```bash
# 综合测试
python test_gpu_cuda.py --mode simple
python test_gpu_cuda.py --mode check
python test_gpu_cuda.py --mode final_v3
```

---

**整理完成！测试脚本数量从 64 个减少到 30 个（减少 53%），同时保留了所有核心功能。**

**下一步建议**：
1. 为 archive/ 中的每个文件添加 README.md 说明其历史和教训
2. 创建主路径测试脚本的统一运行入口
3. 为每条技术线创建回归测试套件
