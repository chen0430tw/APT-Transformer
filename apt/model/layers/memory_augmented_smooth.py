#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
记忆增强的左旋平滑系统

结合 2025-2026 前沿记忆技术:
1. 三层记忆架构 (Short/Mid/Long-term)
2. 骨架状态保持 (Latent Memory)
3. 尖点记录与规避
4. Infini-attention 压缩记忆

核心思想:
- 左旋平滑: 在尖点处缩步，保持轨迹稳定
- 记忆骨架: 记录主题、约束、未决问题等关键状态
- 尖点记录: 记住"哪里容易崩"，提前规避

参考:
- Memory-Augmented Transformers (arXiv:2508.10824, Aug 2025)
- MemoRAG: Global Memory-Enhanced RAG (ACL 2025)
- Infini-attention (arXiv:2404.07143, Apr 2024)
- MemoryLLM / M+ (2024-2025)

作者: chen0430tw
日期: 2026-01-21
"""

from apt.apt_model.utils.fake_torch import get_torch
torch = get_torch()
from apt.apt_model.utils.fake_torch import get_torch
torch = get_torch()
nn = torch.nn
F = torch.nn.functional
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# ==================== 配置 ====================

@dataclass
class MemoryConfig:
    """记忆配置"""
    # 短期记忆 (Working Memory)
    short_term_size: int = 8  # 最近 8 步
    short_term_dim: int = 768

    # 中期记忆 (Episodic Memory)
    mid_term_size: int = 64  # 关键事件
    mid_term_dim: int = 384

    # 长期记忆 (Semantic Memory / Skeleton)
    long_term_size: int = 16  # 骨架状态
    long_term_dim: int = 192

    # 尖点记录
    spike_history_size: int = 32  # 记录最近 32 个尖点
    spike_threshold: float = 0.5  # 尖点强度阈值

    # 压缩记忆 (Infini-attention style)
    use_compressed_memory: bool = True
    compressed_memory_dim: int = 128

    # 骨架字段
    skeleton_fields: List[str] = field(default_factory=lambda: [
        "topic",  # 主题
        "constraints",  # 约束条件
        "definitions",  # 术语定义
        "unresolved",  # 未决问题
        "style_preference",  # 风格偏好
        "spike_regions"  # 尖点区域
    ])


# ==================== 骨架状态 ====================

class SkeletonState:
    """
    骨架状态 (Latent Memory Skeleton)

    保存长期推理轨迹的"主干信息"
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.fields = {field: None for field in config.skeleton_fields}

        # 压缩表示
        self.latent = None  # [long_term_dim]

    def update_field(self, field: str, value: torch.Tensor):
        """更新骨架字段"""
        if field in self.fields:
            self.fields[field] = value
        else:
            logger.warning(f"[Skeleton] 未知字段: {field}")

    def compress(self) -> torch.Tensor:
        """压缩骨架状态到 latent 向量"""
        # 将所有字段堆叠并压缩
        valid_fields = [v for v in self.fields.values() if v is not None]
        if not valid_fields:
            return torch.zeros(self.config.long_term_dim)

        # 简单平均池化（实际可用更复杂的编码器）
        stacked = torch.stack(valid_fields, dim=0)
        self.latent = stacked.mean(dim=0)

        return self.latent

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return {
            "fields": {k: v.tolist() if v is not None else None
                      for k, v in self.fields.items()},
            "latent": self.latent.tolist() if self.latent is not None else None
        }


# ==================== 三层记忆系统 ====================

class HierarchicalMemory(nn.Module):
    """
    三层分层记忆系统

    模仿人类记忆:
    - 短期记忆 (STM): 最近几步，快速访问
    - 中期记忆 (MTM): 关键事件，中等容量
    - 长期记忆 (LTM): 骨架状态，高度压缩

    类似 MemoryOS (2025)
    """

    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config

        # 短期记忆缓冲区
        self.short_term = []  # List[Tensor]
        self.short_term_max = config.short_term_size

        # 中期记忆缓冲区
        self.mid_term = []  # List[Tensor]
        self.mid_term_max = config.mid_term_size

        # 长期记忆 (骨架)
        self.skeleton = SkeletonState(config)

        # 压缩记忆矩阵 (Infini-attention style)
        if config.use_compressed_memory:
            self.compressed_memory = nn.Parameter(
                torch.zeros(config.compressed_memory_dim, config.short_term_dim)
            )

        # 记忆门控网络
        self.importance_scorer = nn.Linear(config.short_term_dim, 1)

        logger.info(
            f"[Memory] 三层记忆初始化: "
            f"STM={config.short_term_size}, "
            f"MTM={config.mid_term_size}, "
            f"LTM={config.long_term_size}"
        )

    def add_to_short_term(self, hidden_state: torch.Tensor):
        """添加到短期记忆"""
        self.short_term.append(hidden_state.detach())

        # FIFO
        if len(self.short_term) > self.short_term_max:
            # 提升重要的到中期记忆
            oldest = self.short_term.pop(0)
            self._promote_to_mid_term(oldest)

    def _promote_to_mid_term(self, hidden_state: torch.Tensor):
        """提升到中期记忆"""
        # 计算重要性
        importance = self.importance_scorer(hidden_state).item()

        if importance > 0.5:  # 阈值
            self.mid_term.append(hidden_state)

            # FIFO
            if len(self.mid_term) > self.mid_term_max:
                # 提升到长期记忆
                oldest = self.mid_term.pop(0)
                self._promote_to_long_term(oldest)

    def _promote_to_long_term(self, hidden_state: torch.Tensor):
        """提升到长期记忆（更新骨架）"""
        # 提取关键信息更新骨架
        # 这里简化处理，实际可用专门的信息提取器
        self.skeleton.update_field("topic", hidden_state[:32])  # 前32维作为主题
        self.skeleton.compress()

    def retrieve(self, query: torch.Tensor, level: str = "all") -> torch.Tensor:
        """
        检索记忆

        Args:
            query: 查询向量
            level: "short", "mid", "long", "all"

        Returns:
            检索到的记忆
        """
        memories = []

        if level in ["short", "all"] and self.short_term:
            memories.extend(self.short_term)

        if level in ["mid", "all"] and self.mid_term:
            memories.extend(self.mid_term)

        if level in ["long", "all"] and self.skeleton.latent is not None:
            memories.append(self.skeleton.latent)

        if not memories:
            return torch.zeros_like(query)

        # 计算相似度并检索
        memories_tensor = torch.stack(memories, dim=0)

        # 简单的点积相似度（实际可用更复杂的检索）
        similarities = torch.matmul(memories_tensor, query)
        best_idx = similarities.argmax()

        return memories_tensor[best_idx]


# ==================== 尖点记录器 ====================

class SpikeRecorder(nn.Module):
    """
    尖点记录器

    记录"哪里容易崩"，用于提前规避
    """

    def __init__(self, config: MemoryConfig):
        super().__init__()
        self.config = config

        # 尖点历史 (位置, 强度, 方向)
        self.spike_history = []  # List[Dict]
        self.max_history = config.spike_history_size

    def record_spike(
        self,
        position: int,
        strength: float,
        direction: torch.Tensor
    ):
        """记录尖点"""
        spike_info = {
            'position': position,
            'strength': strength,
            'direction': direction.detach()
        }

        self.spike_history.append(spike_info)

        # FIFO
        if len(self.spike_history) > self.max_history:
            self.spike_history.pop(0)

    def is_near_spike(
        self,
        current_position: int,
        current_direction: torch.Tensor,
        radius: int = 10
    ) -> Tuple[bool, float]:
        """
        检查当前位置是否接近历史尖点

        Args:
            current_position: 当前位置
            current_direction: 当前方向
            radius: 检测半径

        Returns:
            (is_near, danger_level)
        """
        if not self.spike_history:
            return False, 0.0

        # 检查位置接近度
        position_dangers = []
        direction_similarities = []

        for spike in self.spike_history:
            # 位置距离
            pos_dist = abs(current_position - spike['position'])

            if pos_dist <= radius:
                # 方向相似度
                dir_sim = F.cosine_similarity(
                    current_direction.unsqueeze(0),
                    spike['direction'].unsqueeze(0)
                ).item()

                if dir_sim > 0.7:  # 方向接近
                    position_dangers.append(spike['strength'])
                    direction_similarities.append(dir_sim)

        if not position_dangers:
            return False, 0.0

        # 计算危险等级
        danger_level = max(position_dangers) * max(direction_similarities)

        return True, danger_level


# ==================== 增强版左旋平滑 ====================

class MemoryAugmentedLeftSpinSmooth(nn.Module):
    """
    记忆增强的左旋平滑

    结合:
    1. 原始左旋平滑（尖点缩步）
    2. 三层记忆系统
    3. 尖点记录与规避
    """

    def __init__(
        self,
        d_model: int = 768,
        memory_config: Optional[MemoryConfig] = None,
        # 左旋平滑参数
        alpha: float = 0.5,
        tau: float = 0.3,
        beta: float = 0.7,
        w1: float = 1.0,
        w2: float = 0.5,
        gate_type: str = 'normalized'
    ):
        super().__init__()
        self.d_model = d_model
        self.memory_config = memory_config or MemoryConfig()

        # 左旋平滑参数
        self.alpha = alpha
        self.tau = tau
        self.beta = beta
        self.w1 = w1
        self.w2 = w2
        self.gate_type = gate_type

        # 记忆系统
        self.memory = HierarchicalMemory(self.memory_config)

        # 尖点记录器
        self.spike_recorder = SpikeRecorder(self.memory_config)

        # 历史状态
        self.delta_prev = None
        self.phi_prev = 0.0
        self.position = 0

        logger.info(
            f"[MemAugSmooth] 初始化: "
            f"alpha={alpha}, tau={tau}, beta={beta}"
        )

    def compute_spike_strength(
        self,
        u: torch.Tensor,
        delta_u: torch.Tensor
    ) -> float:
        """计算尖点强度"""
        # 一阶变化
        norm_u = torch.norm(u, p=2, dim=-1, keepdim=True) + 1e-8
        norm_delta = torch.norm(delta_u, p=2, dim=-1, keepdim=True) + 1e-8
        d = (norm_delta / norm_u).mean().item()

        # 二阶变化（加速度）
        a = 0.0
        if self.delta_prev is not None:
            delta_diff = delta_u - self.delta_prev
            norm_diff = torch.norm(delta_diff, p=2, dim=-1, keepdim=True) + 1e-8
            a = (norm_diff / (norm_delta + 1e-8)).mean().item()

        # 综合尖点强度
        s = self.w1 * d + self.w2 * a

        return s

    def compute_buffer_angle(self, s: float) -> float:
        """计算缓冲角度"""
        # Softplus 激活
        phi_raw = self.alpha * F.softplus(torch.tensor(s - self.tau)).item()

        # 惯性平滑
        phi = (1 - self.beta) * self.phi_prev + self.beta * phi_raw

        # 单向缓冲（≥0）
        phi = max(0.0, phi)

        # 更新历史
        self.phi_prev = phi

        return phi

    def apply_gate(self, phi: float) -> float:
        """应用门控函数"""
        if self.gate_type == 'normalized':
            g = 1.0 / math.sqrt(1.0 + phi ** 2)
        elif self.gate_type == 'cosine':
            g = max(0.0, min(1.0, math.cos(phi)))
        else:
            g = 1.0

        return g

    def forward(
        self,
        u: torch.Tensor,
        delta_u: torch.Tensor,
        use_memory: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        前向传播

        Args:
            u: 当前状态
            delta_u: 更新量
            use_memory: 是否使用记忆增强

        Returns:
            (u_next, stats)
        """
        # 1. 计算尖点强度
        s = self.compute_spike_strength(u, delta_u)

        # 2. 检查是否接近历史尖点
        danger_level = 0.0
        if use_memory:
            is_near, danger_level = self.spike_recorder.is_near_spike(
                self.position,
                delta_u.mean(dim=0),  # 平均方向
                radius=10
            )

            if is_near:
                s = s + danger_level * 0.5  # 增强尖点信号

        # 3. 计算缓冲角度
        phi = self.compute_buffer_angle(s)

        # 4. 应用门控
        g = self.apply_gate(phi)

        # 5. 更新状态
        delta_u_eff = g * delta_u
        u_next = u + delta_u_eff

        # 6. 记录尖点（如果显著）
        if s > self.memory_config.spike_threshold:
            self.spike_recorder.record_spike(
                self.position,
                s,
                delta_u.mean(dim=0)
            )

        # 7. 更新记忆
        if use_memory:
            self.memory.add_to_short_term(u_next.mean(dim=0))

        # 8. 更新历史
        self.delta_prev = delta_u.detach()
        self.position += 1

        # 统计信息
        stats = {
            'spike_strength': s,
            'buffer_angle': phi,
            'gate': g,
            'danger_level': danger_level,
            'position': self.position,
            'memory_size': len(self.memory.short_term)
        }

        return u_next, stats


# ==================== 便捷函数 ====================

def create_memory_augmented_smooth(
    d_model: int = 768,
    **kwargs
) -> MemoryAugmentedLeftSpinSmooth:
    """创建记忆增强的左旋平滑"""
    return MemoryAugmentedLeftSpinSmooth(d_model=d_model, **kwargs)


# ==================== 测试 ====================

if __name__ == "__main__":
    import math
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("记忆增强左旋平滑测试")
    print("=" * 70)

    # 创建模块
    smooth = create_memory_augmented_smooth(d_model=768)

    # 模拟推理过程
    print("\n模拟 10 步推理:")
    u = torch.randn(8, 32, 768)

    for step in range(10):
        # 模拟更新
        if step == 5:
            # 第5步：引入一个大的尖点
            delta_u = torch.randn(8, 32, 768) * 5.0
        else:
            delta_u = torch.randn(8, 32, 768)

        # 应用平滑
        u_next, stats = smooth(u, delta_u, use_memory=True)

        print(f"\nStep {step}:")
        print(f"  尖点强度: {stats['spike_strength']:.4f}")
        print(f"  缓冲角度: {stats['buffer_angle']:.4f}")
        print(f"  门控值:   {stats['gate']:.4f}")
        print(f"  危险等级: {stats['danger_level']:.4f}")
        print(f"  记忆大小: {stats['memory_size']}")

        u = u_next

    # 测试记忆检索
    print("\n\n[测试记忆检索]")
    query = torch.randn(768)
    retrieved = smooth.memory.retrieve(query, level="all")
    print(f"检索到的记忆: {retrieved.shape}")

    # 测试尖点记录
    print("\n[测试尖点记录]")
    print(f"记录的尖点数量: {len(smooth.spike_recorder.spike_history)}")

    for i, spike in enumerate(smooth.spike_recorder.spike_history[:3]):
        print(f"  尖点 {i}: 位置={spike['position']}, 强度={spike['strength']:.4f}")

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
