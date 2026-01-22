#!/usr/bin/env python3
"""
APT Non-Autopoietic Control Version
用于对照组实验的非自生成APT模型

用途：
- 分析HLBD在自生成vs标准Transformer架构下的适用性
- 对照组实验，验证自生成机制的有效性
- 研究autopoietic attention的贡献度

使用方法：
    from apt_model.modeling.apt_control import create_control_model, create_autopoietic_model

    # 创建对照组模型 (无自生成机制)
    control_model = create_control_model(vocab_size=2000, d_model=512)

    # 创建实验组模型 (有自生成机制)
    autopoietic_model = create_autopoietic_model(vocab_size=2000, d_model=512)
"""

from apt_model.utils.fake_torch import get_torch
torch = get_torch()
from apt_model.utils.fake_torch import get_torch
torch = get_torch()
nn = torch.nn
from typing import Optional, Dict, Any
from .apt_model import APTModel, APTModelConfiguration


def create_control_model(
    vocab_size: int = 2000,
    d_model: int = 512,
    n_layers: int = 12,
    n_heads: int = 8,
    d_ff: int = 2048,
    max_seq_len: int = 512,
    dropout: float = 0.1,
    use_dbc_dac: bool = True,
    **kwargs
) -> APTModel:
    """
    创建非自生成APT模型 (对照组)

    关键差异：
    - use_autopoietic=False：禁用自生成注意力机制
    - 保留所有其他架构特性（DBC-DAC, 位置编码等）

    Args:
        vocab_size: 词汇表大小
        d_model: 模型维度
        n_layers: 层数
        n_heads: 注意力头数
        d_ff: 前馈网络维度
        max_seq_len: 最大序列长度
        dropout: Dropout比率
        use_dbc_dac: 是否使用DBC-DAC稳定
        **kwargs: 其他配置参数

    Returns:
        APTModel实例（use_autopoietic=False）
    """
    config = APTModelConfiguration(
        vocab_size=vocab_size,
        d_model=d_model,
        num_encoder_layers=n_layers,
        num_decoder_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
        use_autopoietic=False,  # 🔴 关键：禁用自生成机制
        use_dbc_dac=use_dbc_dac,
        **kwargs
    )

    model = APTModel(config)
    print(f"✅ 创建非自生成APT模型 (对照组)")
    print(f"   - 词汇表: {vocab_size}")
    print(f"   - 维度: {d_model}")
    print(f"   - 层数: {n_layers}")
    print(f"   - 自生成机制: ❌ 禁用")
    print(f"   - DBC-DAC: {'✓' if use_dbc_dac else '✗'}")

    return model


def create_autopoietic_model(
    vocab_size: int = 2000,
    d_model: int = 512,
    n_layers: int = 12,
    n_heads: int = 8,
    d_ff: int = 2048,
    max_seq_len: int = 512,
    dropout: float = 0.1,
    use_dbc_dac: bool = True,
    epsilon: float = 1e-6,
    alpha: float = 0.1,
    init_tau: float = 1.0,
    sr_ratio: int = 4,
    **kwargs
) -> APTModel:
    """
    创建自生成APT模型 (实验组)

    关键差异：
    - use_autopoietic=True：启用自生成注意力机制
    - 包含所有自生成相关参数（epsilon, alpha, tau, sr_ratio）

    Args:
        vocab_size: 词汇表大小
        d_model: 模型维度
        n_layers: 层数
        n_heads: 注意力头数
        d_ff: 前馈网络维度
        max_seq_len: 最大序列长度
        dropout: Dropout比率
        use_dbc_dac: 是否使用DBC-DAC稳定
        epsilon: 自生成无穷倒数缩放因子
        alpha: 泰勒展开系数
        init_tau: 初始温度参数
        sr_ratio: 自生成矩阵压缩比
        **kwargs: 其他配置参数

    Returns:
        APTModel实例（use_autopoietic=True）
    """
    config = APTModelConfiguration(
        vocab_size=vocab_size,
        d_model=d_model,
        num_encoder_layers=n_layers,
        num_decoder_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
        use_autopoietic=True,  # 🟢 关键：启用自生成机制
        epsilon=epsilon,
        alpha=alpha,
        init_tau=init_tau,
        sr_ratio=sr_ratio,
        use_dbc_dac=use_dbc_dac,
        **kwargs
    )

    model = APTModel(config)
    print(f"✅ 创建自生成APT模型 (实验组)")
    print(f"   - 词汇表: {vocab_size}")
    print(f"   - 维度: {d_model}")
    print(f"   - 层数: {n_layers}")
    print(f"   - 自生成机制: ✓ 启用")
    print(f"   - DBC-DAC: {'✓' if use_dbc_dac else '✗'}")
    print(f"   - 参数: ε={epsilon}, α={alpha}, τ={init_tau}, sr_ratio={sr_ratio}")

    return model


def compare_model_architectures(
    control_model: APTModel,
    autopoietic_model: APTModel
) -> Dict[str, Any]:
    """
    比较对照组和实验组模型的架构差异

    Args:
        control_model: 非自生成模型
        autopoietic_model: 自生成模型

    Returns:
        包含比较结果的字典
    """
    def count_parameters(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    control_total, control_trainable = count_parameters(control_model)
    auto_total, auto_trainable = count_parameters(autopoietic_model)

    comparison = {
        "control_model": {
            "total_params": control_total,
            "trainable_params": control_trainable,
            "use_autopoietic": control_model.config.use_autopoietic
        },
        "autopoietic_model": {
            "total_params": auto_total,
            "trainable_params": auto_trainable,
            "use_autopoietic": autopoietic_model.config.use_autopoietic
        },
        "difference": {
            "total_params": auto_total - control_total,
            "trainable_params": auto_trainable - control_trainable,
            "percentage": ((auto_total - control_total) / control_total * 100)
        }
    }

    print("\n" + "=" * 60)
    print("模型架构对比")
    print("=" * 60)
    print(f"\n对照组 (无自生成):")
    print(f"  总参数: {control_total:,}")
    print(f"  可训练: {control_trainable:,}")
    print(f"\n实验组 (有自生成):")
    print(f"  总参数: {auto_total:,}")
    print(f"  可训练: {auto_trainable:,}")
    print(f"\n参数差异:")
    print(f"  增加: {auto_total - control_total:,} ({comparison['difference']['percentage']:.2f}%)")

    return comparison


# ============================================================================
# 对照实验辅助函数
# ============================================================================

def create_control_experiment_pair(
    vocab_size: int = 2000,
    d_model: int = 512,
    n_layers: int = 12,
    n_heads: int = 8,
    **kwargs
):
    """
    创建一对对照实验模型（对照组 + 实验组）

    返回：
        (control_model, autopoietic_model, comparison_dict)
    """
    print("\n" + "=" * 60)
    print("创建对照实验模型对")
    print("=" * 60 + "\n")

    # 创建对照组
    control_model = create_control_model(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        **kwargs
    )

    print()

    # 创建实验组
    autopoietic_model = create_autopoietic_model(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        **kwargs
    )

    # 比较
    comparison = compare_model_architectures(control_model, autopoietic_model)

    return control_model, autopoietic_model, comparison


def save_control_experiment_models(
    control_model: APTModel,
    autopoietic_model: APTModel,
    save_dir: str = "control_experiments"
):
    """
    保存对照实验的两个模型

    Args:
        control_model: 对照组模型
        autopoietic_model: 实验组模型
        save_dir: 保存目录
    """
    import os
    from pathlib import Path

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 保存对照组
    control_path = save_path / "control_model.pt"
    torch.save({
        'model_state_dict': control_model.state_dict(),
        'config': control_model.config.to_dict(),
        'model_type': 'control'
    }, control_path)
    print(f"✅ 对照组模型已保存: {control_path}")

    # 保存实验组
    autopoietic_path = save_path / "autopoietic_model.pt"
    torch.save({
        'model_state_dict': autopoietic_model.state_dict(),
        'config': autopoietic_model.config.to_dict(),
        'model_type': 'autopoietic'
    }, autopoietic_path)
    print(f"✅ 实验组模型已保存: {autopoietic_path}")


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    print("APT Control Group Experiment Utility")
    print("=" * 60)

    # 创建对照实验模型对
    control, autopoietic, comparison = create_control_experiment_pair(
        vocab_size=2000,
        d_model=512,
        n_layers=12,
        n_heads=8,
        use_dbc_dac=True
    )

    # 保存模型
    save_control_experiment_models(control, autopoietic)

    print("\n" + "=" * 60)
    print("✨ 对照实验模型创建完成！")
    print("=" * 60)
    print("\n使用方法:")
    print("  1. 用control_model训练 (无自生成机制)")
    print("  2. 用autopoietic_model训练 (有自生成机制)")
    print("  3. 对比两者在HLBD任务上的表现")
    print("  4. 分析自生成机制的贡献度")
