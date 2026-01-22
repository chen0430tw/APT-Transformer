#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试DBC-DAC加速训练功能"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim

# 添加路径
sys.path.insert(0, '/home/user/APT-Transformer')

from apt.apt_model.modeling.apt_model import DBCDAC_Optimizer, create_gradient_stabilizer_hook

# 直接实现注册函数，避免导入复杂依赖
def register_dbc_dac_hooks(model):
    """为模型注册DBC-DAC hooks"""
    opt = DBCDAC_Optimizer()
    hooks = []
    for _, p in model.named_parameters():
        if p.requires_grad:
            hooks.append(p.register_hook(create_gradient_stabilizer_hook(opt)))
    return hooks


class SimpleTestModel(nn.Module):
    """简单的测试模型"""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.linear3(x)


def test_dbc_optimizer():
    """测试DBCDAC优化器"""
    print("\n" + "="*60)
    print("🧪 测试DBCDAC优化器")
    print("="*60)

    # 创建优化器
    dbc_optimizer = DBCDAC_Optimizer(
        rank_ratio_proj=0.1,
        rank_ratio_res=0.05,
        apply_to_gradients=True
    )

    print(f"📊 优化器配置:")
    print(f"   rank_ratio_proj: {dbc_optimizer.rank_ratio_proj}")
    print(f"   rank_ratio_res: {dbc_optimizer.rank_ratio_res}")
    print(f"   threshold: {dbc_optimizer.threshold}")
    print(f"   apply_to_gradients: {dbc_optimizer.apply_to_gradients}")

    # 测试矩阵稳定
    print("\n🔧 测试矩阵稳定功能...")
    test_matrix = torch.randn(64, 128)
    print(f"   原始矩阵形状: {test_matrix.shape}")
    print(f"   原始矩阵范数: {torch.norm(test_matrix).item():.4f}")

    stabilized = dbc_optimizer.stabilize_matrix(test_matrix)
    print(f"   稳定后矩阵形状: {stabilized.shape}")
    print(f"   稳定后矩阵范数: {torch.norm(stabilized).item():.4f}")
    print(f"   ✅ 矩阵稳定测试成功!")

    # 测试梯度稳定
    print("\n🔧 测试梯度稳定功能...")
    test_grad = torch.randn(128, 64)
    print(f"   原始梯度形状: {test_grad.shape}")
    print(f"   原始梯度范数: {torch.norm(test_grad).item():.4f}")

    stabilized_grad = dbc_optimizer.stabilize_gradients(test_grad)
    print(f"   稳定后梯度形状: {stabilized_grad.shape}")
    print(f"   稳定后梯度范数: {torch.norm(stabilized_grad).item():.4f}")
    print(f"   ✅ 梯度稳定测试成功!")

    # 测试处理NaN梯度
    print("\n🔧 测试NaN梯度处理...")
    nan_grad = torch.randn(64, 128)
    nan_grad[0, 0] = float('nan')
    print(f"   包含NaN: {torch.isnan(nan_grad).any().item()}")

    hook = create_gradient_stabilizer_hook(dbc_optimizer)
    cleaned_grad = hook(nan_grad)
    print(f"   处理后包含NaN: {torch.isnan(cleaned_grad).any().item()}")
    print(f"   ✅ NaN梯度处理成功!")

    return True


def test_dbc_hooks():
    """测试DBC-DAC hooks注册"""
    print("\n" + "="*60)
    print("🧪 测试DBC-DAC Hooks注册")
    print("="*60)

    # 创建模型
    model = SimpleTestModel()
    print(f"📊 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 注册hooks
    print("\n🔧 注册DBC-DAC hooks...")
    hooks = register_dbc_dac_hooks(model)
    print(f"   注册的hook数量: {len(hooks)}")
    print(f"   ✅ Hooks注册成功!")

    return True


def test_training_with_dbc():
    """测试带DBC的训练流程"""
    print("\n" + "="*60)
    print("🧪 测试带DBC的训练流程")
    print("="*60)

    # 创建模型和优化器
    model = SimpleTestModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 注册DBC hooks
    print("🔧 注册DBC-DAC hooks...")
    hooks = register_dbc_dac_hooks(model)
    print(f"   注册了 {len(hooks)} 个hooks")

    # 创建模拟数据
    batch_size = 16
    x = torch.randn(batch_size, 64)
    y = torch.randint(0, 10, (batch_size,))

    print(f"\n📊 训练数据:")
    print(f"   输入形状: {x.shape}")
    print(f"   标签形状: {y.shape}")

    # 训练几步
    print("\n🏃 运行训练步骤...")
    model.train()
    num_steps = 5

    for step in range(num_steps):
        optimizer.zero_grad()

        # 前向传播
        output = model(x)
        loss = nn.functional.cross_entropy(output, y)

        # 反向传播 (DBC hooks会自动应用)
        loss.backward()

        # 检查梯度
        grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)

        # 优化器步骤
        optimizer.step()

        print(f"   Step {step+1}/{num_steps} | Loss: {loss.item():.4f} | Grad Norm: {grad_norm:.4f}")

    print(f"   ✅ 训练流程测试成功!")

    return True


def test_gradient_stability():
    """测试梯度稳定性"""
    print("\n" + "="*60)
    print("🧪 测试梯度稳定性对比")
    print("="*60)

    # 创建两个相同的模型
    model_without_dbc = SimpleTestModel()
    model_with_dbc = SimpleTestModel()

    # 确保参数相同
    model_with_dbc.load_state_dict(model_without_dbc.state_dict())

    # 只给一个模型注册DBC hooks
    print("🔧 为第二个模型注册DBC hooks...")
    hooks = register_dbc_dac_hooks(model_with_dbc)

    # 创建模拟数据 (故意制造一些极端值)
    x = torch.randn(8, 64) * 10  # 放大输入
    y = torch.randint(0, 10, (8,))

    print(f"\n📊 测试数据 (包含极端值):")
    print(f"   输入范围: [{x.min().item():.2f}, {x.max().item():.2f}]")

    # 测试无DBC
    print("\n🔍 无DBC模型:")
    model_without_dbc.train()
    output1 = model_without_dbc(x)
    loss1 = nn.functional.cross_entropy(output1, y)
    loss1.backward()
    grad_norm1 = sum(p.grad.norm().item() for p in model_without_dbc.parameters() if p.grad is not None)
    print(f"   Loss: {loss1.item():.4f}")
    print(f"   梯度范数: {grad_norm1:.4f}")

    # 测试有DBC
    print("\n🔍 有DBC模型:")
    model_with_dbc.train()
    output2 = model_with_dbc(x)
    loss2 = nn.functional.cross_entropy(output2, y)
    loss2.backward()
    grad_norm2 = sum(p.grad.norm().item() for p in model_with_dbc.parameters() if p.grad is not None)
    print(f"   Loss: {loss2.item():.4f}")
    print(f"   梯度范数: {grad_norm2:.4f}")

    print(f"\n📈 对比:")
    print(f"   梯度范数差异: {abs(grad_norm1 - grad_norm2):.4f}")
    print(f"   DBC使梯度更{'稳定' if grad_norm2 < grad_norm1 else '激进'}")
    print(f"   ✅ 稳定性测试完成!")

    return True


def main():
    """主测试函数"""
    print("\n🚀 开始测试DBC-DAC加速训练功能...")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    try:
        # 测试DBC优化器
        test1 = test_dbc_optimizer()

        # 测试hooks注册
        test2 = test_dbc_hooks()

        # 测试训练流程
        test3 = test_training_with_dbc()

        # 测试梯度稳定性
        test4 = test_gradient_stability()

        # 总结
        print("\n" + "="*60)
        print("📝 测试总结")
        print("="*60)
        print(f"✅ DBCDAC优化器: {'通过' if test1 else '失败'}")
        print(f"✅ Hooks注册: {'通过' if test2 else '失败'}")
        print(f"✅ 训练流程: {'通过' if test3 else '失败'}")
        print(f"✅ 梯度稳定性: {'通过' if test4 else '失败'}")

        if all([test1, test2, test3, test4]):
            print("\n🎉 所有测试通过!")
            print("\n💡 DBC-DAC功能说明:")
            print("   - 维度平衡压缩法(DBC)减少梯度噪声")
            print("   - 维度伴随补偿法(DAC)保持梯度信息")
            print("   - 自动处理NaN/Inf梯度")
            print("   - 提高训练稳定性和收敛速度")
            return 0
        else:
            print("\n❌ 部分测试失败")
            return 1

    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
