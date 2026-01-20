"""
虚拟Blackwell全局启用器

一行代码启用虚拟Blackwell优化：
    import apt_model.optimization.vb_global as vb
    vb.enable()

所有后续创建的APT模型都会自动应用VGPU优化。
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
import os

# 虚拟Blackwell组件
from apt_model.optimization.vgpu_stack import VGPUStack, create_vgpu_stack
from apt_model.optimization.vgpu_estimator import VGPUResourceEstimator, ModelConfig

# 全局状态
_vb_enabled = False
_vb_stack = None
_vb_config = {
    'use_fp4': False,
    'use_flash_attn': False,
    'mixed_precision': False,
    'gradient_checkpointing': False,
    'auto_estimate': True,
    'verbose': True,
}


def enable(use_fp4: bool = False,
          use_flash_attn: bool = False,
          mixed_precision: bool = False,
          gradient_checkpointing: bool = False,
          auto_estimate: bool = True,
          vgpu_config: Optional[Dict] = None,
          verbose: bool = True):
    """
    全局启用虚拟Blackwell优化

    Args:
        use_fp4: 启用FP4量化
        use_flash_attn: 启用Flash Attention
        mixed_precision: 启用混合精度
        gradient_checkpointing: 启用梯度检查点
        auto_estimate: 自动估算资源需求
        vgpu_config: 自定义VGPU配置（None=使用默认）
        verbose: 打印详细信息

    Example:
        >>> import apt_model.optimization.vb_global as vb
        >>> vb.enable(use_fp4=True, use_flash_attn=True)
        >>>
        >>> # 之后所有APT模型都会自动优化
        >>> from apt_model.modeling.apt_model import APTLargeModel
        >>> model = APTLargeModel(config)  # 自动应用VGPU优化
    """
    global _vb_enabled, _vb_stack, _vb_config

    _vb_enabled = True
    _vb_config.update({
        'use_fp4': use_fp4,
        'use_flash_attn': use_flash_attn,
        'mixed_precision': mixed_precision,
        'gradient_checkpointing': gradient_checkpointing,
        'auto_estimate': auto_estimate,
        'verbose': verbose,
    })

    # 创建VGPU Stack
    if vgpu_config:
        from apt_model.optimization.vgpu_stack import VGPUStack
        _vb_stack = VGPUStack(vgpu_config)
    else:
        _vb_stack = create_vgpu_stack()

    if verbose:
        print("\n" + "="*70)
        print("🚀 虚拟Blackwell已全局启用")
        print("="*70)
        print(f"FP4量化:         {'✅ 启用' if use_fp4 else '❌ 禁用'}")
        print(f"Flash Attention: {'✅ 启用' if use_flash_attn else '❌ 禁用'}")
        print(f"混合精度:        {'✅ 启用' if mixed_precision else '❌ 禁用'}")
        print(f"梯度检查点:      {'✅ 启用' if gradient_checkpointing else '❌ 禁用'}")
        print(f"自动估算:        {'✅ 启用' if auto_estimate else '❌ 禁用'}")
        print("="*70 + "\n")


def disable():
    """禁用虚拟Blackwell优化"""
    global _vb_enabled, _vb_stack
    _vb_enabled = False
    _vb_stack = None
    print("虚拟Blackwell已禁用")


def is_enabled() -> bool:
    """检查虚拟Blackwell是否已启用"""
    return _vb_enabled


def get_stack() -> Optional[VGPUStack]:
    """获取全局VGPU Stack"""
    return _vb_stack


def get_config() -> Dict:
    """获取当前配置"""
    return _vb_config.copy()


def optimize_model(model: nn.Module, model_name: str = "model") -> nn.Module:
    """
    优化单个模型

    Args:
        model: PyTorch模型
        model_name: 模型名称（用于日志）

    Returns:
        优化后的模型
    """
    if not _vb_enabled:
        return model

    if _vb_stack is None:
        raise RuntimeError("请先调用vb.enable()启用虚拟Blackwell")

    verbose = _vb_config['verbose']

    if verbose:
        print(f"\n优化模型: {model_name}")

    # 导入优化包装器
    from training.test_vb_apt_integration import VBOptimizedAPTModel

    # 获取模型配置
    if hasattr(model, 'config'):
        apt_config = model.config
    else:
        raise ValueError("模型必须有config属性")

    # 创建优化模型
    optimized_model = VBOptimizedAPTModel(
        apt_config,
        _vb_stack,
        use_fp4=_vb_config['use_fp4'],
        use_flash_attn=_vb_config['use_flash_attn']
    )

    # 复制原始权重
    optimized_model.base_model.load_state_dict(model.state_dict())

    if verbose:
        print(f"✓ 已优化 {len(optimized_model.optimized_layers)} 个线性层")

    return optimized_model


def estimate_model_resources(model_config, batch_size: int = 8):
    """
    估算模型资源需求

    Args:
        model_config: APT模型配置
        batch_size: 批次大小
    """
    if not _vb_config['auto_estimate']:
        return

    # 转换为评估器配置
    estimator_config = ModelConfig(
        vocab_size=getattr(model_config, 'vocab_size', 50000),
        hidden_size=getattr(model_config, 'hidden_size', 768),
        num_layers=getattr(model_config, 'num_layers', 12),
        num_heads=getattr(model_config, 'num_heads', 12),
        seq_length=getattr(model_config, 'max_position_embeddings', 2048),
        batch_size=batch_size,
        mixed_precision=_vb_config['mixed_precision'],
        gradient_checkpointing=_vb_config['gradient_checkpointing']
    )

    # 评估
    estimator = VGPUResourceEstimator()
    estimator.estimate_transformer(estimator_config)

    # 生成VGPU配置
    available_gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            available_gpus.append({
                'device': f'cuda:{i}',
                'vram_gb': props.total_memory / (1024**3),
                'speed_gbps': 900
            })

    if available_gpus:
        estimator.generate_vgpu_config(available_gpus)
        estimator.print_report()


def get_stats() -> Dict:
    """获取VGPU统计信息"""
    if _vb_stack is None:
        return {}
    return _vb_stack.get_stats()


def print_stats():
    """打印VGPU统计信息"""
    if _vb_stack is None:
        print("虚拟Blackwell未启用")
        return
    _vb_stack.print_stats()


# 便捷预设
def enable_full_optimization():
    """启用所有优化（最大显存节省）"""
    enable(
        use_fp4=True,
        use_flash_attn=True,
        mixed_precision=True,
        gradient_checkpointing=True,
        auto_estimate=True
    )


def enable_speed_mode():
    """启用速度模式（FP4量化）"""
    enable(
        use_fp4=True,
        use_flash_attn=False,
        mixed_precision=False,
        gradient_checkpointing=False,
        auto_estimate=True
    )


def enable_memory_mode():
    """启用显存模式（Flash Attention + 梯度检查点）"""
    enable(
        use_fp4=False,
        use_flash_attn=True,
        mixed_precision=True,
        gradient_checkpointing=True,
        auto_estimate=True
    )


def enable_balanced_mode():
    """启用平衡模式（推荐）"""
    enable(
        use_fp4=False,
        use_flash_attn=True,
        mixed_precision=True,
        gradient_checkpointing=False,
        auto_estimate=True
    )


# 环境变量控制（可通过环境变量自动启用）
if os.getenv('ENABLE_VIRTUAL_BLACKWELL', '').lower() in ('1', 'true', 'yes'):
    mode = os.getenv('VB_MODE', 'balanced').lower()

    if mode == 'full':
        enable_full_optimization()
    elif mode == 'speed':
        enable_speed_mode()
    elif mode == 'memory':
        enable_memory_mode()
    else:
        enable_balanced_mode()

    print(f"✅ 通过环境变量自动启用虚拟Blackwell ({mode}模式)")


if __name__ == "__main__":
    # 测试
    print("虚拟Blackwell全局启用器")
    print("\n使用示例:")
    print("```python")
    print("import apt_model.optimization.vb_global as vb")
    print("")
    print("# 启用虚拟Blackwell")
    print("vb.enable(use_fp4=True, use_flash_attn=True)")
    print("")
    print("# 之后所有APT模型都会自动优化")
    print("from apt_model.modeling.apt_model import APTLargeModel")
    print("model = APTLargeModel(config)  # 自动应用VGPU优化")
    print("")
    print("# 查看统计")
    print("vb.print_stats()")
    print("```")
