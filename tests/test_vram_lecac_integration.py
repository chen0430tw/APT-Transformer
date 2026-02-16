"""
测试 LECAC 与 virtual_vram 的完整集成
========================================

验证:
1. LECAC 模式正确包装 nn.Linear
2. 梯度计算正确（余弦相似度 > 0.999）
3. virtual_vram + LECAC 联合工作
"""
import os
import sys

_BASE = "D:/APT-Transformer"
os.environ["TORCH_HOME"] = f"{_BASE}/.torch_cache"
os.environ["TEMP"] = f"{_BASE}/.temp"
os.makedirs(_BASE + "/.torch_cache", exist_ok=True)
os.makedirs(_BASE + "/.temp", exist_ok=True)

import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入 virtual_vram
sys.path.insert(0, f"{_BASE}/apt/vgpu/runtime")
from virtual_vram import VirtualVRAMConfig, virtual_vram, LECACLinear


print("=" * 70)
print("LECAC + virtual_vram 集成测试")
print("=" * 70)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")

# 测试参数
in_features, out_features = 768, 768
batch_size = 512
torch.manual_seed(42)

# 固定输入
x = torch.randn(batch_size, in_features, device=device, requires_grad=True)

# ========================================================================
# 测试 1: Baseline (标准 nn.Linear + FP32)
# ========================================================================
print("[测试 1] Baseline (FP32)")
print("-" * 70)

baseline_model = nn.Linear(in_features, out_features).to(device)
baseline_model.train()

y_base = baseline_model(x)
loss_base = y_base.sum()
loss_base.backward()

base_grad_norm = x.grad.norm().item()
weight_grad_norm = baseline_model.weight.grad.norm().item()
print(f"  x.grad norm: {base_grad_norm:.6e}")
print(f"  weight.grad norm: {weight_grad_norm:.6e}")

# ========================================================================
# 测试 2: LECAC 模式 (use_lecac=True + compress_dtype="int8")
# ========================================================================
print("\n[测试 2] LECAC 模式 (INT8 训练)")
print("-" * 70)

# 创建 LECAC 配置
cfg_lecac = VirtualVRAMConfig(
    enabled=True,
    use_lecac=True,         # 启用 LECAC
    compress=True,
    compress_dtype="int8",  # 必须是 int8
    lecac_alpha=0.0,        # 无 LDBR 补偿
    verbose=True,
    min_tensor_bytes=1 << 10,  # 降低阈值以便触发压缩
)

# 创建相同权重的模型
lecac_model = nn.Linear(in_features, out_features).to(device)
lecac_model.weight.data = baseline_model.weight.data.clone()
if baseline_model.bias is not None:
    lecac_model.bias.data = baseline_model.bias.data.clone()

x.grad = None

# 使用 virtual_vram + LECAC
with virtual_vram(cfg_lecac, model=lecac_model) as manager:
    print(f"  Wrapped model type: {type(manager.wrapped_model)}")

    # 检查 Linear 层是否被替换
    lecac_linear_count = sum(1 for m in manager.wrapped_model.modules() if isinstance(m, LECACLinear))
    print(f"  LECACLinear 层数量: {lecac_linear_count}")

    # 前向传播
    y_lecac = manager.wrapped_model(x)
    loss_lecac = y_lecac.sum()
    loss_lecac.backward()

lecac_grad_norm = x.grad.norm().item()
lecac_weight_grad_norm = manager.wrapped_model.weight.grad.norm().item()
print(f"  x.grad norm: {lecac_grad_norm:.6e}")
print(f"  weight.grad norm: {lecac_weight_grad_norm:.6e}")

# 梯度相似度
grad_weight_sim = F.cosine_similarity(
    manager.wrapped_model.weight.grad.flatten(),
    baseline_model.weight.grad.flatten(),
    dim=0
).item()

print(f"\n  梯度余弦相似度: {grad_weight_sim:.8f}")

# ========================================================================
# 测试 3: LECAC + LDBR 补偿
# ========================================================================
print("\n[测试 3] LECAC + LDBR 补偿 (alpha=0.5)")
print("-" * 70)

cfg_ldbr = VirtualVRAMConfig(
    enabled=True,
    use_lecac=True,
    compress=True,
    compress_dtype="int8",
    lecac_alpha=0.5,  # 启用 LDBR
    verbose=False,
    min_tensor_bytes=1 << 10,
)

ldbr_model = nn.Linear(in_features, out_features).to(device)
ldbr_model.weight.data = baseline_model.weight.data.clone()
if baseline_model.bias is not None:
    ldbr_model.bias.data = baseline_model.bias.data.clone()

x.grad = None

with virtual_vram(cfg_ldbr, model=ldbr_model) as manager:
    y_ldbr = manager.wrapped_model(x)
    loss_ldbr = y_ldbr.sum()
    loss_ldbr.backward()

ldbr_grad_norm = x.grad.norm().item()
ldbr_weight_grad_norm = manager.wrapped_model.weight.grad.norm().item()
print(f"  x.grad norm: {ldbr_grad_norm:.6e}")
print(f"  weight.grad norm: {ldbr_weight_grad_norm:.6e}")

grad_weight_sim_ldbr = F.cosine_similarity(
    manager.wrapped_model.weight.grad.flatten(),
    baseline_model.weight.grad.flatten(),
    dim=0
).item()

print(f"\n  梯度余弦相似度: {grad_weight_sim_ldbr:.8f}")

# ========================================================================
# 结果汇总
# ========================================================================
print("\n" + "=" * 70)
print("结果汇总")
print("=" * 70)

print(f"\nBaseline (FP32):")
print(f"  weight.grad norm: {weight_grad_norm:.6e}")

print(f"\nLECAC (无补偿, alpha=0.0):")
print(f"  weight.grad norm: {lecac_weight_grad_norm:.6e}")
print(f"  梯度余弦相似度: {grad_weight_sim:.8f}")
if grad_weight_sim > 0.999:
    print(f"  状态: ✅ Excellent (>0.999)")
elif grad_weight_sim > 0.99:
    print(f"  状态: ✅ Good (>0.99)")
elif grad_weight_sim > 0.95:
    print(f"  状态: ⚠️  Fair (>0.95)")
else:
    print(f"  状态: ❌ Need improvement")

print(f"\nLECAC + LDBR (alpha=0.5):")
print(f"  weight.grad norm: {ldbr_weight_grad_norm:.6e}")
print(f"  梯度余弦相似度: {grad_weight_sim_ldbr:.8f}")
if grad_weight_sim_ldbr > 0.999:
    print(f"  状态: ✅ Excellent (>0.999)")
elif grad_weight_sim_ldbr > 0.99:
    print(f"  状态: ✅ Good (>0.99)")
elif grad_weight_sim_ldbr > 0.95:
    print(f"  状态: ⚠️  Fair (>0.95)")
else:
    print(f"  状态: ❌ Need improvement")

print(f"\nLDBR 补偿效果:")
improvement = grad_weight_sim_ldbr - grad_weight_sim
print(f"  相似度提升: {improvement:+.8f}")
if improvement > 0:
    print(f"  有效: LDBR 补偿改善了梯度估计")
else:
    print(f"  注意: 补偿效果不明显（基本 STE 已经很好）")

print("\n" + "=" * 70)
print("集成测试完成")
print("=" * 70)

# 最终结论
if grad_weight_sim > 0.999 and grad_weight_sim_ldbr > 0.999:
    print("\n✅ LECAC 与 virtual_vram 集成成功！")
    print("   INT8 量化训练的梯度精度与 FP32 基本一致。")
else:
    print("\n⚠️  梯度精度需要检查。")
