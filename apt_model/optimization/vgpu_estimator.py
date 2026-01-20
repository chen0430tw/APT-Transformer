"""
虚拟Blackwell资源评估器

功能：
1. 估算模型训练/推理的内存需求
2. 计算需要的VGPU堆叠配置
3. 推荐最优批次大小
4. 生成资源配置方案

使用场景：
- 训练前规划：需要多少GPU/CPU/SSD
- 成本估算：云服务器配置选择
- 性能预测：预估训练速度
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json


@dataclass
class ModelConfig:
    """模型配置"""
    # 基础参数
    vocab_size: int = 50000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    seq_length: int = 512

    # 训练参数
    batch_size: int = 32
    gradient_checkpointing: bool = False
    mixed_precision: bool = False  # FP16/BF16

    # 优化器
    optimizer: str = 'adamw'  # adamw, sgd, adafactor


@dataclass
class MemoryEstimate:
    """内存估算结果"""
    # 模型组件（单位：字节）
    parameters: int = 0
    gradients: int = 0
    optimizer_states: int = 0
    activations: int = 0

    # 总计
    total_train: int = 0
    total_inference: int = 0

    # 格式化输出
    def __str__(self):
        def fmt(bytes_val):
            gb = bytes_val / (1024**3)
            return f"{gb:.2f} GB ({bytes_val:,} bytes)"

        return f"""
内存估算:
  参数:        {fmt(self.parameters)}
  梯度:        {fmt(self.gradients)}
  优化器状态:   {fmt(self.optimizer_states)}
  激活值:      {fmt(self.activations)}
  ──────────────────────────────────
  训练总计:    {fmt(self.total_train)}
  推理总计:    {fmt(self.total_inference)}
"""


@dataclass
class VGPUConfig:
    """VGPU堆叠配置"""
    levels: List[Dict] = None
    total_capacity_gb: float = 0
    estimated_hit_rate: float = 0
    expected_overhead: float = 0

    def __str__(self):
        lines = ["\n推荐VGPU堆叠配置:"]
        lines.append("=" * 60)

        for i, level in enumerate(self.levels):
            lines.append(f"  Level {i}: {level['device']:10s} "
                        f"{level['capacity_mb']/1024:6.1f} GB @ "
                        f"{level['speed_gbps']:5.0f} GB/s")

        lines.append("=" * 60)
        lines.append(f"  总容量:     {self.total_capacity_gb:.1f} GB")
        lines.append(f"  预期命中率: {self.estimated_hit_rate:.1%}")
        lines.append(f"  预期开销:   {self.expected_overhead:.1%}")

        return "\n".join(lines)


class VGPUResourceEstimator:
    """虚拟Blackwell资源评估器"""

    # 字节大小常量
    BYTES_FP32 = 4
    BYTES_FP16 = 2
    BYTES_INT8 = 1
    BYTES_FP4 = 0.5

    def __init__(self):
        self.model_config: Optional[ModelConfig] = None
        self.memory_estimate: Optional[MemoryEstimate] = None
        self.vgpu_config: Optional[VGPUConfig] = None

    def estimate_transformer(self, config: ModelConfig) -> MemoryEstimate:
        """估算Transformer模型内存"""
        self.model_config = config

        # 1. 参数量计算
        params_count = self._count_transformer_params(config)

        # 2. 确定数据类型
        if config.mixed_precision:
            param_bytes = self.BYTES_FP16
            grad_bytes = self.BYTES_FP16
        else:
            param_bytes = self.BYTES_FP32
            grad_bytes = self.BYTES_FP32

        # 3. 计算各部分内存
        parameters_mem = params_count * param_bytes
        gradients_mem = params_count * grad_bytes

        # 优化器状态（AdamW: 2个状态tensor）
        if config.optimizer == 'adamw':
            optimizer_mem = params_count * self.BYTES_FP32 * 2  # m和v
        elif config.optimizer == 'sgd':
            optimizer_mem = params_count * self.BYTES_FP32  # momentum
        else:
            optimizer_mem = params_count * self.BYTES_FP32

        # 激活值（前向传播中间结果）
        activations_mem = self._estimate_activations(config)

        # 梯度检查点可以减少激活值内存
        if config.gradient_checkpointing:
            activations_mem = activations_mem // 4

        # 总计
        total_train = (parameters_mem + gradients_mem +
                      optimizer_mem + activations_mem)
        total_inference = parameters_mem + activations_mem // 2

        self.memory_estimate = MemoryEstimate(
            parameters=parameters_mem,
            gradients=gradients_mem,
            optimizer_states=optimizer_mem,
            activations=activations_mem,
            total_train=total_train,
            total_inference=total_inference
        )

        return self.memory_estimate

    def _count_transformer_params(self, config: ModelConfig) -> int:
        """计算Transformer参数量"""
        h = config.hidden_size
        n = config.num_layers
        v = config.vocab_size

        # Embedding层
        embed_params = v * h  # token embedding
        embed_params += config.seq_length * h  # position embedding

        # 单个Transformer层
        # Attention: Q, K, V, O
        attn_params = 4 * h * h
        # FFN: 两层全连接（通常中间维度是4h）
        ffn_params = h * (4 * h) + (4 * h) * h
        # LayerNorm: 2个（attn后、FFN后）
        ln_params = 2 * 2 * h

        layer_params = attn_params + ffn_params + ln_params

        # 所有层
        total_params = embed_params + n * layer_params

        # 输出层（LM head）
        total_params += v * h

        return total_params

    def _estimate_activations(self, config: ModelConfig) -> int:
        """估算激活值内存"""
        b = config.batch_size
        s = config.seq_length
        h = config.hidden_size
        n = config.num_layers

        # 每层的激活值
        # Attention: Q, K, V, scores, context
        attn_act = b * s * h * 3  # Q, K, V
        attn_act += b * config.num_heads * s * s  # attention scores

        # FFN: 中间激活
        ffn_act = b * s * (4 * h)

        # 每层总激活
        layer_act = attn_act + ffn_act

        # 所有层
        if config.mixed_precision:
            bytes_per_elem = self.BYTES_FP16
        else:
            bytes_per_elem = self.BYTES_FP32

        total_act = n * layer_act * bytes_per_elem

        return total_act

    def generate_vgpu_config(self, available_gpus: List[Dict],
                            target_hit_rate: float = 0.90) -> VGPUConfig:
        """
        生成VGPU堆叠配置

        Args:
            available_gpus: 可用GPU列表，格式：
                [
                    {'device': 'cuda:0', 'vram_gb': 8, 'speed_gbps': 900},
                    {'device': 'cuda:1', 'vram_gb': 8, 'speed_gbps': 900},
                    ...
                ]
            target_hit_rate: 目标Level 0命中率
        """
        if self.memory_estimate is None:
            raise ValueError("请先运行estimate_transformer()")

        # 训练内存需求（GB）
        train_mem_gb = self.memory_estimate.total_train / (1024**3)

        # 计算各层级容量
        levels = []
        total_capacity_gb = 0

        # Level 0: 主GPU（需要存放热数据）
        if available_gpus:
            main_gpu = available_gpus[0]
            # Level 0容量 = GPU显存 * 目标命中率
            level0_capacity_gb = min(
                main_gpu['vram_gb'] * 0.8,  # 预留20%给其他
                train_mem_gb * target_hit_rate
            )
            levels.append({
                'capacity_mb': int(level0_capacity_gb * 1024),
                'device': main_gpu['device'],
                'speed_gbps': main_gpu['speed_gbps']
            })
            total_capacity_gb += level0_capacity_gb

        # Level 1: 其他GPU（如果有）
        for gpu in available_gpus[1:]:
            level_capacity_gb = gpu['vram_gb'] * 0.8
            levels.append({
                'capacity_mb': int(level_capacity_gb * 1024),
                'device': gpu['device'],
                'speed_gbps': gpu['speed_gbps']
            })
            total_capacity_gb += level_capacity_gb

        # Level N: CPU内存
        remaining_mem_gb = max(0, train_mem_gb - total_capacity_gb)
        if remaining_mem_gb > 0:
            cpu_capacity_gb = remaining_mem_gb * 1.5  # 多分配50%
            levels.append({
                'capacity_mb': int(cpu_capacity_gb * 1024),
                'device': 'cpu',
                'speed_gbps': 50  # PCIe 4.0
            })
            total_capacity_gb += cpu_capacity_gb

        # Level N+1: SSD（如果内存不够）
        if total_capacity_gb < train_mem_gb * 2:
            ssd_capacity_gb = train_mem_gb * 3  # 足够的SSD空间
            levels.append({
                'capacity_mb': int(ssd_capacity_gb * 1024),
                'device': 'ssd',
                'speed_gbps': 7  # NVMe
            })
            total_capacity_gb += ssd_capacity_gb

        # 估算性能
        estimated_hit_rate = self._estimate_hit_rate(levels, train_mem_gb)
        expected_overhead = self._estimate_overhead(estimated_hit_rate)

        self.vgpu_config = VGPUConfig(
            levels=levels,
            total_capacity_gb=total_capacity_gb,
            estimated_hit_rate=estimated_hit_rate,
            expected_overhead=expected_overhead
        )

        return self.vgpu_config

    def _estimate_hit_rate(self, levels: List[Dict],
                          train_mem_gb: float) -> float:
        """估算Level 0命中率"""
        if not levels:
            return 0.0

        level0_gb = levels[0]['capacity_mb'] / 1024

        # 简单模型：命中率 ≈ Level 0容量 / 工作集大小
        # 工作集通常是训练内存的60-80%
        working_set_gb = train_mem_gb * 0.7
        hit_rate = min(1.0, level0_gb / working_set_gb)

        return hit_rate

    def _estimate_overhead(self, hit_rate: float) -> float:
        """估算开销"""
        # 经验公式：overhead ≈ (1 - hit_rate) * 0.5
        # 命中率100% → 开销0%
        # 命中率50% → 开销25%
        return (1 - hit_rate) * 0.5

    def recommend_batch_size(self, gpu_vram_gb: float) -> List[int]:
        """推荐批次大小"""
        if self.memory_estimate is None:
            raise ValueError("请先运行estimate_transformer()")

        # 单样本的内存占用
        config = self.model_config
        single_sample_gb = (self.memory_estimate.total_train /
                           config.batch_size / (1024**3))

        # 可用显存（预留20%给PyTorch）
        available_gb = gpu_vram_gb * 0.8

        # 计算批次大小
        recommended_batch = int(available_gb / single_sample_gb)

        # 返回2的幂次（常见批次大小）
        batch_sizes = []
        for power in range(10):  # 1, 2, 4, 8, ..., 512
            bs = 2 ** power
            if bs <= recommended_batch:
                batch_sizes.append(bs)

        return batch_sizes[-3:] if len(batch_sizes) >= 3 else batch_sizes

    def print_report(self):
        """打印完整评估报告"""
        if self.model_config is None:
            print("❌ 请先运行estimate_transformer()")
            return

        print("\n" + "="*70)
        print("虚拟Blackwell资源评估报告")
        print("="*70)

        # 1. 模型配置
        print("\n[1] 模型配置:")
        config = self.model_config
        print(f"  架构:       Transformer")
        print(f"  层数:       {config.num_layers}")
        print(f"  隐藏维度:   {config.hidden_size}")
        print(f"  注意力头:   {config.num_heads}")
        print(f"  词表大小:   {config.vocab_size}")
        print(f"  序列长度:   {config.seq_length}")
        print(f"  批次大小:   {config.batch_size}")

        # 计算总参数量
        total_params = self._count_transformer_params(config)
        print(f"  总参数量:   {total_params/1e6:.1f}M ({total_params:,})")

        # 2. 内存估算
        print("\n[2] 内存需求:")
        print(self.memory_estimate)

        # 3. VGPU配置
        if self.vgpu_config:
            print("\n[3] 资源配置:")
            print(self.vgpu_config)

        # 4. 优化建议
        print("\n[4] 优化建议:")

        train_gb = self.memory_estimate.total_train / (1024**3)

        if config.mixed_precision:
            print("  ✓ 已启用混合精度（FP16/BF16）")
        else:
            saved_gb = train_gb * 0.5
            print(f"  → 启用混合精度可节省 {saved_gb:.1f} GB")

        if config.gradient_checkpointing:
            print("  ✓ 已启用梯度检查点")
        else:
            saved_gb = self.memory_estimate.activations / (1024**3) * 0.75
            print(f"  → 启用梯度检查点可节省 {saved_gb:.1f} GB（牺牲20%速度）")

        # FP4量化
        fp4_saved = self.memory_estimate.parameters / (1024**3) * 0.875
        print(f"  → 使用FP4量化可节省参数内存 {fp4_saved:.1f} GB")

        # Flash Attention
        attn_mem = (config.batch_size * config.num_heads *
                   config.seq_length ** 2 * self.BYTES_FP32)
        attn_saved_gb = attn_mem / (1024**3) * config.num_layers
        print(f"  → 使用Flash Attention可节省 {attn_saved_gb:.1f} GB")

        print("\n" + "="*70)

    def save_config(self, filename: str):
        """保存配置到JSON"""
        if self.vgpu_config is None:
            print("❌ 请先生成VGPU配置")
            return

        config_dict = {
            'model': {
                'vocab_size': self.model_config.vocab_size,
                'hidden_size': self.model_config.hidden_size,
                'num_layers': self.model_config.num_layers,
                'num_heads': self.model_config.num_heads,
                'seq_length': self.model_config.seq_length,
                'batch_size': self.model_config.batch_size
            },
            'memory': {
                'total_train_gb': self.memory_estimate.total_train / (1024**3),
                'total_inference_gb': self.memory_estimate.total_inference / (1024**3)
            },
            'vgpu_stack': {
                'levels': self.vgpu_config.levels,
                'total_capacity_gb': self.vgpu_config.total_capacity_gb,
                'estimated_hit_rate': self.vgpu_config.estimated_hit_rate
            }
        }

        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"✓ 配置已保存到 {filename}")


def quick_estimate(model_name: str = 'gpt2-medium'):
    """快速估算常见模型"""

    configs = {
        'gpt2-small': ModelConfig(
            vocab_size=50000, hidden_size=768, num_layers=12,
            num_heads=12, seq_length=1024, batch_size=32
        ),
        'gpt2-medium': ModelConfig(
            vocab_size=50000, hidden_size=1024, num_layers=24,
            num_heads=16, seq_length=1024, batch_size=16
        ),
        'gpt2-large': ModelConfig(
            vocab_size=50000, hidden_size=1280, num_layers=36,
            num_heads=20, seq_length=1024, batch_size=8
        ),
        'gpt2-xl': ModelConfig(
            vocab_size=50000, hidden_size=1600, num_layers=48,
            num_heads=25, seq_length=1024, batch_size=4
        ),
        'llama-7b': ModelConfig(
            vocab_size=32000, hidden_size=4096, num_layers=32,
            num_heads=32, seq_length=2048, batch_size=4
        ),
        'llama-13b': ModelConfig(
            vocab_size=32000, hidden_size=5120, num_layers=40,
            num_heads=40, seq_length=2048, batch_size=2
        ),
    }

    if model_name not in configs:
        print(f"❌ 未知模型: {model_name}")
        print(f"可用模型: {list(configs.keys())}")
        return

    config = configs[model_name]

    estimator = VGPUResourceEstimator()
    estimator.estimate_transformer(config)

    # 假设有1张RTX 3070
    available_gpus = [
        {'device': 'cuda:0', 'vram_gb': 8, 'speed_gbps': 900}
    ]

    estimator.generate_vgpu_config(available_gpus)
    estimator.print_report()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("虚拟Blackwell资源评估器 - 快速测试")
    print("="*70)

    # 测试几个常见模型
    models = ['gpt2-small', 'gpt2-medium', 'llama-7b']

    for model_name in models:
        print(f"\n\n{'='*70}")
        print(f"模型: {model_name.upper()}")
        print('='*70)
        quick_estimate(model_name)
