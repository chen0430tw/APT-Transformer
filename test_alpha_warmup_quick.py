#!/usr/bin/env python3
"""
快速测试LECaC Alpha Warmup是否工作
运行时间：<1分钟
"""
import torch
import torch.nn as nn

print("=" * 70)
print("LECaC Alpha Warmup 快速测试")
print("=" * 70)

# 1. 导入测试
try:
    from apt.vgpu.runtime.lecac import LECACLinear
    print("✅ LECaCLinear 导入成功")
except ImportError as e:
    print(f"❌ LECaCLinear 导入失败: {e}")
    exit(1)

try:
    from apt.vgpu.runtime.lecac_warmup import LECACAlphaScheduler, update_lecac_alpha
    print("✅ LECACAlphaScheduler 导入成功")
except ImportError as e:
    print(f"❌ LECACAlphaScheduler 导入失败: {e}")
    print("   请确认 apt/vgpu/runtime/lecac_warmup.py 存在")
    exit(1)

# 2. 创建简单模型
print("\n[测试1] 创建LECaC模型")
print("-" * 70)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = LECACLinear(256, 512, bits=2, alpha=1.47)
        self.fc2 = LECACLinear(512, 256, bits=2, alpha=1.47)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

model = SimpleModel()
print(f"✅ 模型创建成功")
print(f"   fc1.alpha = {model.fc1.alpha:.4f}")
print(f"   fc2.alpha = {model.fc2.alpha:.4f}")

# 3. 创建Alpha Warmup调度器
print("\n[测试2] 创建Alpha Warmup调度器")
print("-" * 70)

scheduler = LECACAlphaScheduler(
    warmup_steps=100,
    warmup_multiplier=3.0,
    schedule="cosine"
)
print("✅ 调度器创建成功")

# 4. 测试Alpha更新
print("\n[测试3] 测试Alpha更新")
print("-" * 70)

print(f"{'Step':<8} {'Alpha':<10} {'fc1.alpha':<12} {'fc2.alpha':<12} {'Status'}")
print("-" * 60)

test_steps = [0, 10, 25, 50, 75, 99, 100, 200]
all_passed = True

for step in test_steps:
    alpha_target = scheduler.get_alpha(step)
    num_updated = update_lecac_alpha(model, alpha_target)

    fc1_alpha = model.fc1.alpha
    fc2_alpha = model.fc2.alpha

    # 验证更新是否成功
    if abs(fc1_alpha - alpha_target) < 1e-6 and abs(fc2_alpha - alpha_target) < 1e-6:
        status = "✅"
    else:
        status = "❌"
        all_passed = False

    print(f"{step:<8} {alpha_target:<10.4f} {fc1_alpha:<12.4f} {fc2_alpha:<12.4f} {status}")

# 5. 验证Alpha曲线
print("\n[测试4] 验证Alpha Warmup曲线")
print("-" * 70)

alpha_0 = scheduler.get_alpha(0)
alpha_50 = scheduler.get_alpha(50)
alpha_100 = scheduler.get_alpha(100)
alpha_200 = scheduler.get_alpha(200)

print(f"Step   0: Alpha = {alpha_0:.4f} (应该 ≈ 4.41)")
print(f"Step  50: Alpha = {alpha_50:.4f} (应该 ≈ 2.94)")
print(f"Step 100: Alpha = {alpha_100:.4f} (应该 ≈ 1.47)")
print(f"Step 200: Alpha = {alpha_200:.4f} (应该 = 1.47)")

checks_passed = 0
if 4.3 < alpha_0 < 4.5:
    print("✅ 初始Alpha正确")
    checks_passed += 1
else:
    print(f"❌ 初始Alpha错误（应该≈4.41）")

if alpha_50 > 2.5 and alpha_50 < 3.5:
    print("✅ 中间Alpha正确")
    checks_passed += 1
else:
    print(f"❌ 中间Alpha错误（应该≈2.94）")

if abs(alpha_100 - 1.47) < 0.1:
    print("✅ 结束Alpha正确")
    checks_passed += 1
else:
    print(f"❌ 结束Alpha错误（应该≈1.47）")

if abs(alpha_200 - 1.47) < 0.01:
    print("✅ Warmup后Alpha保持不变")
    checks_passed += 1
else:
    print(f"❌ Warmup后Alpha变化了")

# 6. 测试forward/backward
print("\n[测试5] 测试训练循环（10步）")
print("-" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

nan_count = 0
for step in range(10):
    # 更新alpha
    current_alpha = scheduler.get_alpha(step)
    update_lecac_alpha(model, current_alpha)

    # 训练步骤
    optimizer.zero_grad()
    x = torch.randn(32, 256).to(device)
    y = model(x)
    loss = y.sum()

    if torch.isnan(loss):
        nan_count += 1
        print(f"  Step {step}: Loss=NaN ❌, Alpha={current_alpha:.4f}")
    else:
        print(f"  Step {step}: Loss={loss.item():.4f} ✅, Alpha={current_alpha:.4f}")

    loss.backward()
    optimizer.step()

# 7. 最终结果
print("\n" + "=" * 70)
print("测试结果总结")
print("=" * 70)

if all_passed:
    print("✅ Alpha更新测试: 通过")
else:
    print("❌ Alpha更新测试: 失败")

if checks_passed == 4:
    print("✅ Alpha曲线测试: 通过")
else:
    print(f"⚠️  Alpha曲线测试: {checks_passed}/4 通过")

if nan_count == 0:
    print("✅ 训练循环测试: 通过（0 NaN）")
else:
    print(f"❌ 训练循环测试: 失败（{nan_count} NaN）")

print()
if all_passed and checks_passed == 4 and nan_count == 0:
    print("🎉 所有测试通过！LECaC Alpha Warmup工作正常！")
    print()
    print("现在可以在实际训练中使用：")
    print("  --lecac-alpha-warmup")
else:
    print("⚠️  部分测试失败，请检查安装是否正确")
    print()
    print("调试步骤：")
    print("  1. 确认 apt/vgpu/runtime/lecac_warmup.py 存在")
    print("  2. 确认 PyTorch 版本 >= 2.0")
    print("  3. 查看上面的错误信息")

print("=" * 70)
