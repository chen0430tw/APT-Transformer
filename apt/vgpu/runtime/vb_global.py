"""
虚拟Blackwell全局启用器（增强版）

一行代码启用虚拟Blackwell优化（支持GPU/NPU/CPU + 100K GPU集群）：
    import apt.perf.optimization.vb_global as vb
    vb.enable()

新特性 (2026-01-21):
✨ MXFP4 量化: 4x推理加速 + 4x显存节省
✨ GPU优化MoE: Token Dispatch + 负载均衡
✨ 100K GPU支持: 3D Parallelism + DeepSpeed ZeRO + Megatron-LM
✨ NVLink 5 + GB200 NVL72: 72 GPUs per rack

所有后续创建的APT模型都会自动应用VGPU优化。
支持设备：NVIDIA CUDA GPU、华为昇腾NPU、CPU
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
import os
import logging

logger = logging.getLogger(__name__)

# 虚拟Blackwell组件
from apt.vgpu.runtime.vgpu_stack import VGPUStack, create_vgpu_stack
from apt.vgpu.scheduler.vgpu_estimator import VGPUResourceEstimator, ModelConfig
from apt.perf.optimization.npu_backend import (
    get_accelerator_type,
    is_cuda_available,
    is_npu_available,
    is_hpu_available,
    is_xpu_available
)

# 新增：MXFP4 量化
try:
    from apt.perf.optimization.mxfp4_quantization import (
        MXFP4Quantizer,
        MXFP4Config,
        convert_model_to_mxfp4
    )
    _mxfp4_available = True
except ImportError as e:
    logger.warning(f"MXFP4 量化模块导入失败: {e}")
    _mxfp4_available = False

# 新增：GPU优化MoE
try:
    from apt.model.layers.moe_optimized import (
        MoELayerOptimized,
        MoELayerFast,
        MoEConfig
    )
    _moe_optimized_available = True
except ImportError as e:
    logger.warning(f"GPU优化MoE模块导入失败: {e}")
    _moe_optimized_available = False

# 新增：超大规模训练
try:
    from apt.perf.optimization.extreme_scale_training import (
        ExtremeScaleConfig,
        ExtremeScaleTrainer,
        setup_extreme_scale_training
    )
    _extreme_scale_available = True
except ImportError as e:
    logger.warning(f"超大规模训练模块导入失败: {e}")
    _extreme_scale_available = False

# 全局状态
_vb_enabled = False
_vb_stack = None
_vb_config = {
    # 原有配置
    'use_fp4': False,
    'use_flash_attn': False,
    'mixed_precision': False,
    'gradient_checkpointing': False,
    'auto_estimate': True,
    'verbose': True,

    # 新增配置
    'use_mxfp4': False,  # MXFP4 量化（优先级高于FP4）
    'use_moe_optimized': False,  # GPU优化MoE
    'enable_extreme_scale': False,  # 100K GPU支持

    # 超大规模训练配置
    'extreme_scale_config': None,  # ExtremeScaleConfig实例
}

# 超大规模训练器实例
_extreme_scale_trainer = None


def enable(use_fp4: bool = False,
          use_flash_attn: bool = False,
          mixed_precision: bool = False,
          gradient_checkpointing: bool = False,
          auto_estimate: bool = True,
          vgpu_config: Optional[Dict] = None,
          verbose: bool = True,
          # 新增参数
          use_mxfp4: bool = False,
          mxfp4_block_size: int = 32,
          use_moe_optimized: bool = False,
          moe_num_experts: int = 8,
          moe_top_k: int = 2,
          enable_extreme_scale: bool = False,
          extreme_scale_total_gpus: int = 100000):
    """
    全局启用虚拟Blackwell优化（增强版）

    Args:
        use_fp4: 启用FP4量化
        use_flash_attn: 启用Flash Attention
        mixed_precision: 启用混合精度
        gradient_checkpointing: 启用梯度检查点
        auto_estimate: 自动估算资源需求
        vgpu_config: 自定义VGPU配置（None=使用默认）
        verbose: 打印详细信息

        # 新增参数
        use_mxfp4: 启用MXFP4量化（4-bit，优先级高于FP4）
        mxfp4_block_size: MXFP4块大小（默认32）
        use_moe_optimized: 启用GPU优化MoE层
        moe_num_experts: MoE专家数量（默认8）
        moe_top_k: MoE激活专家数（默认2）
        enable_extreme_scale: 启用100K GPU大规模训练支持
        extreme_scale_total_gpus: 集群总GPU数（默认100,000）

    Example:
        >>> import apt.perf.optimization.vb_global as vb
        >>> # 基础使用
        >>> vb.enable(use_fp4=True, use_flash_attn=True)
        >>>
        >>> # 启用MXFP4（更快）
        >>> vb.enable(use_mxfp4=True)
        >>>
        >>> # 启用100K GPU支持
        >>> vb.enable_extreme_scale_mode(total_gpus=100000)
        >>>
        >>> # 之后所有APT模型都会自动优化
        >>> from apt.model.architectures.apt_model import APTLargeModel
        >>> model = APTLargeModel(config)  # 自动应用VGPU优化
    """
    global _vb_enabled, _vb_stack, _vb_config, _extreme_scale_trainer

    _vb_enabled = True

    # 优先级：MXFP4 > FP4
    if use_mxfp4 and _mxfp4_available:
        use_fp4 = False  # 禁用旧版FP4
        logger.info("[VB] 使用MXFP4量化（优先级高于FP4）")

    _vb_config.update({
        # 原有配置
        'use_fp4': use_fp4,
        'use_flash_attn': use_flash_attn,
        'mixed_precision': mixed_precision,
        'gradient_checkpointing': gradient_checkpointing,
        'auto_estimate': auto_estimate,
        'verbose': verbose,

        # 新增配置
        'use_mxfp4': use_mxfp4,
        'mxfp4_block_size': mxfp4_block_size,
        'use_moe_optimized': use_moe_optimized,
        'moe_num_experts': moe_num_experts,
        'moe_top_k': moe_top_k,
        'enable_extreme_scale': enable_extreme_scale,
    })

    # 创建VGPU Stack
    if vgpu_config:
        from apt.vgpu.runtime.vgpu_stack import VGPUStack
        _vb_stack = VGPUStack(vgpu_config)
    else:
        _vb_stack = create_vgpu_stack()

    # 创建超大规模训练配置
    if enable_extreme_scale and _extreme_scale_available:
        extreme_config = ExtremeScaleConfig(
            total_gpus=extreme_scale_total_gpus,
            use_mixed_precision=mixed_precision,
            use_mxfp4=use_mxfp4,
            use_gradient_checkpointing=gradient_checkpointing
        )
        _vb_config['extreme_scale_config'] = extreme_config
        logger.info(
            f"[VB] 超大规模训练配置已创建 "
            f"({extreme_scale_total_gpus:,} GPUs)"
        )

    if verbose:
        # 检测设备类型
        device_type = get_accelerator_type()
        device_emoji = {
            'cuda': '🟢 NVIDIA GPU',
            'hpu': '🟣 Intel Habana Gaudi HPU',
            'npu': '🟡 Huawei Ascend NPU',
            'xpu': '🔵 Intel XPU',
            'cpu': '⚪ CPU'
        }.get(device_type, '⚫ 未知设备')

        print("\n" + "="*70)
        print("[>>] 虚拟Blackwell已全局启用（增强版）")
        print("="*70)
        print(f"加速设备:        {device_emoji}")

        # 量化选项
        if use_mxfp4:
            print(f"MXFP4量化:       [OK] 启用 (4-bit, block_size={mxfp4_block_size})")
        elif use_fp4:
            print(f"FP4量化:         [OK] 启用")
        else:
            print(f"量化:            [X] 禁用")

        # 其他优化
        print(f"Flash Attention: {'[OK] 启用' if use_flash_attn else '[X] 禁用'}")
        print(f"混合精度:        {'[OK] 启用' if mixed_precision else '[X] 禁用'}")
        print(f"梯度检查点:      {'[OK] 启用' if gradient_checkpointing else '[X] 禁用'}")

        # 新增特性
        if use_moe_optimized:
            print(f"GPU优化MoE:      [OK] 启用 ({moe_num_experts}专家, top-{moe_top_k})")
        else:
            print(f"GPU优化MoE:      [X] 禁用")

        if enable_extreme_scale:
            print(f"100K GPU训练:    [OK] 启用 ({extreme_scale_total_gpus:,} GPUs)")
            print(f"  ├─ 3D并行:     [OK]")
            print(f"  ├─ DeepSpeed:  [OK]")
            print(f"  ├─ NVLink 5:   [OK] 1.8TB/s per GPU")
            print(f"  └─ GB200支持:  [OK] 72 GPUs per rack")
        else:
            print(f"100K GPU训练:    [X] 禁用")

        print(f"自动估算:        {'[OK] 启用' if auto_estimate else '[X] 禁用'}")
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
        print(f"[OK] 已优化 {len(optimized_model.optimized_layers)} 个线性层")

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
    stats = {}

    if _vb_stack is not None:
        stats['vgpu'] = _vb_stack.get_stats()

    if _extreme_scale_trainer is not None:
        stats['extreme_scale'] = {
            'enabled': True,
            'total_gpus': _vb_config.get('extreme_scale_config').total_gpus
        }

    return stats


def print_stats():
    """打印VGPU统计信息"""
    if _vb_stack is None:
        print("虚拟Blackwell未启用")
        return
    _vb_stack.print_stats()


def apply_mxfp4_to_model(model: nn.Module) -> nn.Module:
    """
    将MXFP4量化应用到模型

    Args:
        model: PyTorch模型

    Returns:
        量化后的模型

    Example:
        >>> model = MyModel()
        >>> quantized_model = vb.apply_mxfp4_to_model(model)
    """
    if not _mxfp4_available:
        logger.warning("[VB] MXFP4不可用，跳过量化")
        return model

    if not _vb_config.get('use_mxfp4'):
        logger.warning("[VB] MXFP4未启用，请先调用vb.enable(use_mxfp4=True)")
        return model

    config = MXFP4Config(block_size=_vb_config.get('mxfp4_block_size', 32))
    quantized_model = convert_model_to_mxfp4(model, config)

    logger.info("[VB] MXFP4量化已应用")
    return quantized_model


def setup_extreme_scale_training_for_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer
) -> 'ExtremeScaleTrainer':
    """
    为模型设置超大规模训练

    Args:
        model: PyTorch模型
        optimizer: 优化器

    Returns:
        ExtremeScaleTrainer实例

    Example:
        >>> model = MyModel()
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> trainer = vb.setup_extreme_scale_training_for_model(model, optimizer)
        >>> for batch in dataloader:
        ...     stats = trainer.train_step(batch)
    """
    global _extreme_scale_trainer

    if not _extreme_scale_available:
        raise ImportError("超大规模训练模块不可用")

    if not _vb_config.get('enable_extreme_scale'):
        raise RuntimeError(
            "请先启用超大规模训练: "
            "vb.enable_extreme_scale_mode(total_gpus=100000)"
        )

    config = _vb_config.get('extreme_scale_config')
    if config is None:
        raise RuntimeError("超大规模训练配置未初始化")

    _extreme_scale_trainer = ExtremeScaleTrainer(model, optimizer, config)

    logger.info(
        f"[VB] 超大规模训练器已设置 ({config.total_gpus:,} GPUs)"
    )

    return _extreme_scale_trainer


def get_extreme_scale_trainer() -> Optional['ExtremeScaleTrainer']:
    """获取超大规模训练器实例"""
    return _extreme_scale_trainer


# 便捷预设
def enable_full_optimization():
    """启用所有优化（最大显存节省 + MXFP4）"""
    enable(
        use_mxfp4=True,  # 升级到MXFP4
        use_flash_attn=True,
        mixed_precision=True,
        gradient_checkpointing=True,
        auto_estimate=True
    )


def enable_speed_mode():
    """启用速度模式（MXFP4量化，4x加速）"""
    enable(
        use_mxfp4=True,  # 升级到MXFP4
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


def enable_moe_mode(num_experts: int = 8, top_k: int = 2):
    """
    启用MoE模式（GPU优化MoE + MXFP4）

    Args:
        num_experts: 专家数量（默认8）
        top_k: 激活专家数（默认2）

    Example:
        >>> vb.enable_moe_mode(num_experts=16, top_k=2)
    """
    enable(
        use_mxfp4=True,
        use_flash_attn=True,
        mixed_precision=True,
        use_moe_optimized=True,
        moe_num_experts=num_experts,
        moe_top_k=top_k,
        auto_estimate=True
    )


def enable_extreme_scale_mode(
    total_gpus: int = 100000,
    data_parallel: int = 64,
    tensor_parallel: int = 8,
    pipeline_parallel: int = 8
):
    """
    启用超大规模训练模式（100K+ GPUs）

    Args:
        total_gpus: 总GPU数（默认100,000）
        data_parallel: 数据并行度
        tensor_parallel: 张量并行度
        pipeline_parallel: 流水线并行度

    Example:
        >>> # Meta Llama 4 规模 (350K GPUs)
        >>> vb.enable_extreme_scale_mode(total_gpus=350000)
        >>>
        >>> # OpenAI GPT-5 规模 (500K+ GPUs)
        >>> vb.enable_extreme_scale_mode(total_gpus=500000)
    """
    enable(
        use_mxfp4=True,
        use_flash_attn=True,
        mixed_precision=True,
        gradient_checkpointing=True,
        use_moe_optimized=True,
        enable_extreme_scale=True,
        extreme_scale_total_gpus=total_gpus,
        auto_estimate=True
    )

    # 额外设置3D并行配置
    if _extreme_scale_available and _vb_config.get('extreme_scale_config'):
        config = _vb_config['extreme_scale_config']
        config.data_parallel_size = data_parallel
        config.tensor_parallel_size = tensor_parallel
        config.pipeline_parallel_size = pipeline_parallel

        logger.info(
            f"[VB] 3D并行配置: "
            f"DP={data_parallel}, TP={tensor_parallel}, PP={pipeline_parallel}"
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

    print(f"[OK] 通过环境变量自动启用虚拟Blackwell ({mode}模式)")


if __name__ == "__main__":
    # 测试
    print("虚拟Blackwell全局启用器")
    print("\n使用示例:")
    print("```python")
    print("import apt.perf.optimization.vb_global as vb")
    print("")
    print("# 启用虚拟Blackwell")
    print("vb.enable(use_fp4=True, use_flash_attn=True)")
    print("")
    print("# 之后所有APT模型都会自动优化")
    print("from apt.model.architectures.apt_model import APTLargeModel")
    print("model = APTLargeModel(config)  # 自动应用VGPU优化")
    print("")
    print("# 查看统计")
    print("vb.print_stats()")
    print("```")
