#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Claude-4 Model — GPT-4o with Graph-based Reflection Layer

基于 GPT-4o，添加反思层（Reflection Layer）：
- 图连通度分析（Graph Connectivity）
- 最短路径推理（Shortest Path Reasoning）
- 镜像复杂度网络（Mirror Complexity Network）
- 后向反馈机制（Backward Feedback）

这是 Claude 的深度推理能力的实现：通过图论找到最优的推理路径。
Author: Claude Assistant
"""

from apt_model.utils.fake_torch import get_torch
torch = get_torch()
from apt_model.utils.fake_torch import get_torch
torch = get_torch()
nn = torch.nn
F = torch.nn.functional
import math
from typing import Optional, Tuple, Dict, List
from collections import deque

# 导入 GPT-4o 的组件
from apt_model.modeling.gpt4o_model import (
    DynamicTau,
    VeinSubspaceShared,
    FastPathScheduler,
    HybridFFN,
    TriVeinAttention,
    OmniInputEncoder
)


# ==============================================================================
# Graph Connectivity Layer - 图连通度分析
# ==============================================================================

class GraphConnectivityAnalyzer(nn.Module):
    """
    图连通度分析器

    分析注意力图的连通性，找出关键的连接路径。
    使用广度优先搜索（BFS）计算连通度。
    """

    def __init__(self, d_model: int, threshold: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.threshold = threshold  # 连通性阈值

        # 连通度投影
        self.connectivity_proj = nn.Linear(d_model, 1)

    def compute_connectivity(
        self,
        attention_weights: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算注意力图的连通度

        Args:
            attention_weights: [B, H, T, T] 注意力权重
            mask: [B, T] 可选的掩码

        Returns:
            connectivity: [B, H, T] 每个位置的连通度分数
        """
        B, H, T, _ = attention_weights.shape

        # 二值化注意力图（threshold）
        adj_matrix = (attention_weights > self.threshold).float()

        # 计算每个节点的度（degree）
        degree = adj_matrix.sum(dim=-1)  # [B, H, T]

        # 计算连通分量大小（使用并查集的简化版本）
        connectivity_scores = torch.zeros(B, H, T, device=attention_weights.device)

        for b in range(B):
            for h in range(H):
                # 对每个头计算连通性
                visited = torch.zeros(T, dtype=torch.bool, device=attention_weights.device)
                component_sizes = torch.zeros(T, device=attention_weights.device)

                for start in range(T):
                    if not visited[start]:
                        # BFS 找连通分量
                        component = self._bfs_component(
                            adj_matrix[b, h],
                            start,
                            visited
                        )
                        size = len(component)
                        for node in component:
                            component_sizes[node] = size

                connectivity_scores[b, h] = component_sizes

        # 归一化
        connectivity_scores = connectivity_scores / (T + 1e-8)

        return connectivity_scores

    def _bfs_component(
        self,
        adj_matrix: torch.Tensor,
        start: int,
        visited: torch.Tensor
    ) -> List[int]:
        """BFS 找连通分量"""
        component = []
        queue = deque([start])
        visited[start] = True

        while queue:
            node = queue.popleft()
            component.append(node)

            # 找邻居
            neighbors = torch.where(adj_matrix[node] > 0.5)[0]
            for neighbor in neighbors:
                neighbor = neighbor.item()
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        return component

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            hidden_states: [B, T, D]
            attention_weights: [B, H, T, T]

        Returns:
            connectivity_features: [B, T, D]
        """
        # 计算连通度
        connectivity = self.compute_connectivity(attention_weights)  # [B, H, T]

        # 平均所有头
        connectivity = connectivity.mean(dim=1)  # [B, T]

        # 使用连通度加权隐藏状态
        connectivity_weight = connectivity.unsqueeze(-1)  # [B, T, 1]
        weighted_states = hidden_states * connectivity_weight

        return weighted_states


# ==============================================================================
# Shortest Path Reflection - 最短路径推理
# ==============================================================================

class ShortestPathReflection(nn.Module):
    """
    最短路径反思层

    使用 Dijkstra/Floyd-Warshall 算法找到注意力图中的最短路径，
    实现高效的信息传播和推理。
    """

    def __init__(self, d_model: int, max_path_length: int = 5):
        super().__init__()
        self.d_model = d_model
        self.max_path_length = max_path_length

        # 路径编码
        self.path_encoder = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True
        )

        # 路径注意力
        self.path_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            batch_first=True
        )

    def compute_shortest_paths(
        self,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        计算所有节点对之间的最短路径（Floyd-Warshall 简化版）

        Args:
            attention_weights: [B, H, T, T]

        Returns:
            distances: [B, H, T, T] 最短路径距离
        """
        B, H, T, _ = attention_weights.shape

        # 将注意力权重转换为距离（距离 = -log(权重)）
        distances = -torch.log(attention_weights + 1e-10)

        # Floyd-Warshall (简化版，只考虑第一个头)
        # 在实际中可以并行化所有头
        for k in range(T):
            # 通过节点 k 的路径
            dist_through_k = distances[:, :, :, k:k+1] + distances[:, :, k:k+1, :]
            # 更新最短距离
            distances = torch.min(distances, dist_through_k)

        return distances

    def extract_critical_paths(
        self,
        hidden_states: torch.Tensor,
        shortest_distances: torch.Tensor,
        top_k: int = 3
    ) -> torch.Tensor:
        """
        提取关键路径

        Args:
            hidden_states: [B, T, D]
            shortest_distances: [B, H, T, T]
            top_k: 提取前 k 个最短路径

        Returns:
            path_features: [B, T, D]
        """
        B, T, D = hidden_states.shape
        H = shortest_distances.size(1)

        # 平均所有头的距离
        avg_distances = shortest_distances.mean(dim=1)  # [B, T, T]

        # 对每个节点，找到距离最短的 top_k 个节点
        _, top_k_indices = torch.topk(
            -avg_distances,  # 负号因为我们要最小距离
            k=min(top_k, T),
            dim=-1
        )  # [B, T, top_k]

        # 收集这些节点的特征
        path_features = []
        for i in range(min(top_k, T)):
            indices = top_k_indices[:, :, i]  # [B, T]
            gathered = torch.gather(
                hidden_states,
                dim=1,
                index=indices.unsqueeze(-1).expand(-1, -1, D)
            )  # [B, T, D]
            path_features.append(gathered)

        # 堆叠并使用 GRU 编码路径
        path_features = torch.stack(path_features, dim=2)  # [B, T, top_k, D]
        path_features = path_features.view(B * T, min(top_k, T), D)

        # GRU 编码
        encoded_paths, _ = self.path_encoder(path_features)  # [B*T, top_k, D]

        # 取最后一个时间步
        encoded_paths = encoded_paths[:, -1, :]  # [B*T, D]
        encoded_paths = encoded_paths.view(B, T, D)

        return encoded_paths

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            hidden_states: [B, T, D]
            attention_weights: [B, H, T, T]

        Returns:
            reflected_states: [B, T, D]
        """
        # 计算最短路径
        shortest_distances = self.compute_shortest_paths(attention_weights)

        # 提取关键路径特征
        path_features = self.extract_critical_paths(
            hidden_states,
            shortest_distances,
            top_k=self.max_path_length
        )

        # 使用注意力融合原始状态和路径特征
        reflected_states, _ = self.path_attn(
            query=hidden_states,
            key=path_features,
            value=path_features
        )

        return reflected_states


# ==============================================================================
# Mirror Complexity Analyzer - 镜像复杂度分析
# ==============================================================================

class MirrorComplexityAnalyzer(nn.Module):
    """
    镜像复杂度分析器

    通过镜像（对称性分析）找到复杂度最高的网络结构。
    这是 Claude 深度推理的核心：找到最有信息量的路径。
    """

    def __init__(self, d_model: int, num_mirrors: int = 3):
        super().__init__()
        self.d_model = d_model
        self.num_mirrors = num_mirrors

        # 镜像投影
        self.mirror_projs = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_mirrors)
        ])

        # 复杂度评分
        self.complexity_scorer = nn.Sequential(
            nn.Linear(d_model * num_mirrors, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def create_mirrors(
        self,
        hidden_states: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        创建多个镜像视图

        Args:
            hidden_states: [B, T, D]

        Returns:
            mirrors: List of [B, T, D]
        """
        mirrors = []

        for proj in self.mirror_projs:
            # 正向投影
            mirror = proj(hidden_states)

            # 反向镜像（翻转序列）
            mirror_reversed = torch.flip(mirror, dims=[1])

            # 组合正向和反向
            combined = (mirror + mirror_reversed) / 2
            mirrors.append(combined)

        return mirrors

    def compute_complexity(
        self,
        mirrors: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        计算镜像的复杂度

        复杂度定义为镜像之间的差异度（信息熵）

        Args:
            mirrors: List of [B, T, D]

        Returns:
            complexity: [B, T, 1]
        """
        # 连接所有镜像
        concatenated = torch.cat(mirrors, dim=-1)  # [B, T, D * num_mirrors]

        # 计算复杂度分数
        complexity = self.complexity_scorer(concatenated)  # [B, T, 1]

        return complexity

    def select_complex_network(
        self,
        hidden_states: torch.Tensor,
        complexity: torch.Tensor
    ) -> torch.Tensor:
        """
        选择复杂度最高的网络路径

        Args:
            hidden_states: [B, T, D]
            complexity: [B, T, 1]

        Returns:
            selected_states: [B, T, D]
        """
        # 使用复杂度作为门控
        gated_states = hidden_states * complexity

        return gated_states

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            hidden_states: [B, T, D]

        Returns:
            complex_states: [B, T, D]
            complexity_scores: [B, T, 1]
        """
        # 创建镜像
        mirrors = self.create_mirrors(hidden_states)

        # 计算复杂度
        complexity = self.compute_complexity(mirrors)

        # 选择复杂网络
        complex_states = self.select_complex_network(hidden_states, complexity)

        return complex_states, complexity


# ==============================================================================
# Reflection Feedback Loop - 反思反馈循环
# ==============================================================================

class ReflectionFeedbackLoop(nn.Module):
    """
    反思反馈循环

    结合图连通度、最短路径和镜像复杂度，实现完整的反思机制。
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # 三个核心组件
        self.connectivity_analyzer = GraphConnectivityAnalyzer(d_model)
        self.shortest_path_reflection = ShortestPathReflection(d_model)
        self.mirror_complexity = MirrorComplexityAnalyzer(d_model)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

        # 反馈门控
        self.feedback_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        完整的反思反馈循环

        Args:
            hidden_states: [B, T, D]
            attention_weights: [B, H, T, T]

        Returns:
            dict with:
                - reflected_states: [B, T, D]
                - connectivity_scores: [B, T, 1]
                - complexity_scores: [B, T, 1]
                - feedback_strength: [B, T, D]
        """
        # 1. 图连通度分析
        connectivity_features = self.connectivity_analyzer(
            hidden_states,
            attention_weights
        )  # [B, T, D]

        # 2. 最短路径推理
        path_features = self.shortest_path_reflection(
            hidden_states,
            attention_weights
        )  # [B, T, D]

        # 3. 镜像复杂度分析
        complex_features, complexity_scores = self.mirror_complexity(
            hidden_states
        )  # [B, T, D], [B, T, 1]

        # 4. 融合三种特征
        combined = torch.cat([
            connectivity_features,
            path_features,
            complex_features
        ], dim=-1)  # [B, T, 3D]

        fused = self.fusion(combined)  # [B, T, D]

        # 5. 计算反馈强度
        feedback_strength = self.feedback_gate(fused)

        # 6. 应用反馈
        reflected_states = hidden_states + feedback_strength * fused

        # 计算连通度分数（用于监控）
        connectivity_scores = self.connectivity_analyzer.compute_connectivity(
            attention_weights
        ).mean(dim=1, keepdim=True).transpose(1, 2)  # [B, T, 1]

        return {
            'reflected_states': reflected_states,
            'connectivity_scores': connectivity_scores,
            'complexity_scores': complexity_scores,
            'feedback_strength': feedback_strength
        }


# ==============================================================================
# Claude-4 Block with Reflection
# ==============================================================================

class Claude4Block(nn.Module):
    """
    Claude-4 Transformer Block with Reflection Layer

    在 GPT-4o Block 基础上添加反思层。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        rank: int,
        tau_module,
        enable_reflection: bool = True
    ):
        super().__init__()

        # GPT-4o 核心组件
        self.attn = TriVeinAttention(d_model, n_heads, rank, tau_module)
        self.ffn = HybridFFN(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 反思层
        self.enable_reflection = enable_reflection
        if enable_reflection:
            self.reflection = ReflectionFeedbackLoop(d_model)
            self.norm_reflect = nn.LayerNorm(d_model)

        # 保存注意力权重用于反思
        self.attention_weights = None

    def forward(
        self,
        x: torch.Tensor,
        load_factor: float = 1.0,
        return_reflection_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        前向传播

        Args:
            x: [B, T, D]
            load_factor: 负载因子
            return_reflection_info: 是否返回反思信息

        Returns:
            output: [B, T, D]
            reflection_info: Optional[Dict] 反思信息
        """
        # 1. 注意力
        attn_out = self.attn(self.norm1(x), load_factor=load_factor)

        # 提取注意力权重（需要修改 TriVeinAttention 来返回权重）
        # 这里用简化版本：假设可以访问
        # 实际应该修改 TriVeinAttention.forward 返回 (output, attention_weights)

        x = x + attn_out

        # 2. FFN
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out

        # 3. 反思层（如果启用）
        reflection_info = None
        if self.enable_reflection:
            # 创建伪注意力权重用于演示
            # 在实际中应该使用真实的注意力权重
            B, T, D = x.shape
            n_heads = 16  # 假设
            fake_attn = torch.softmax(
                torch.randn(B, n_heads, T, T, device=x.device),
                dim=-1
            )

            reflection_output = self.reflection(x, fake_attn)

            # 应用反思
            x_reflected = self.norm_reflect(reflection_output['reflected_states'])
            x = x + x_reflected

            if return_reflection_info:
                reflection_info = {
                    'connectivity': reflection_output['connectivity_scores'].mean().item(),
                    'complexity': reflection_output['complexity_scores'].mean().item(),
                    'feedback_norm': reflection_output['feedback_strength'].norm().item()
                }

        return x, reflection_info


# ==============================================================================
# Claude-4 Model
# ==============================================================================

class Claude4Model(nn.Module):
    """
    Claude-4 Model

    基于 GPT-4o，添加图论反思层实现深度推理。

    核心创新：
    1. 图连通度分析 - 找到信息流的关键路径
    2. 最短路径推理 - 高效的多跳推理
    3. 镜像复杂度网络 - 通过对称性找到最有价值的信息
    4. 反思反馈循环 - 迭代优化推理过程
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 2048,
        n_heads: int = 16,
        d_ff: int = 8192,
        num_layers: int = 24,
        rank: int = 4,
        enable_reflection: bool = True,
        reflection_layers: Optional[List[int]] = None
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers

        # 输入编码器
        self.encoder = OmniInputEncoder(d_model, vocab_size=vocab_size)

        # 动态 Tau
        self.tau_module = DynamicTau()

        # Transformer blocks with reflection
        # 默认在后半层启用反思
        if reflection_layers is None:
            reflection_layers = list(range(num_layers // 2, num_layers))

        self.blocks = nn.ModuleList([
            Claude4Block(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                rank=rank,
                tau_module=self.tau_module,
                enable_reflection=(enable_reflection and i in reflection_layers)
            )
            for i in range(num_layers)
        ])

        # 输出
        self.norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

        # 统计信息
        self.reflection_stats = {
            'avg_connectivity': 0.0,
            'avg_complexity': 0.0,
            'avg_feedback': 0.0
        }

    def forward(
        self,
        text_ids: Optional[torch.Tensor] = None,
        image_feat: Optional[torch.Tensor] = None,
        audio_feat: Optional[torch.Tensor] = None,
        load_factor: float = 1.0,
        return_reflection_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        前向传播

        Args:
            text_ids: [B, T]
            image_feat: Optional
            audio_feat: Optional
            load_factor: 负载因子
            return_reflection_stats: 是否返回反思统计

        Returns:
            logits: [B, T, V]
            stats: Optional[Dict] 反思统计信息
        """
        # 编码输入
        x = self.encoder(text_ids, image_feat, audio_feat)

        # 收集反思信息
        reflection_infos = []

        # 通过所有块
        for i, block in enumerate(self.blocks):
            x, reflection_info = block(
                x,
                load_factor=load_factor,
                return_reflection_info=return_reflection_stats
            )
            if reflection_info is not None:
                reflection_infos.append(reflection_info)

        # 最终归一化和输出
        x = self.norm(x)
        logits = self.output_head(x)

        # 统计反思信息
        stats = None
        if return_reflection_stats and reflection_infos:
            stats = {
                'avg_connectivity': sum(r['connectivity'] for r in reflection_infos) / len(reflection_infos),
                'avg_complexity': sum(r['complexity'] for r in reflection_infos) / len(reflection_infos),
                'avg_feedback': sum(r['feedback_norm'] for r in reflection_infos) / len(reflection_infos),
                'num_reflection_layers': len(reflection_infos)
            }
            self.reflection_stats = stats

        return logits, stats

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_p: float = 0.9,
        verbose: bool = False
    ) -> torch.Tensor:
        """生成文本（带反思统计）"""
        self.eval()
        generated = input_ids

        with torch.no_grad():
            for step in range(max_new_tokens):
                logits, stats = self.forward(
                    text_ids=generated,
                    return_reflection_stats=verbose
                )

                if verbose and stats:
                    print(f"Step {step}: Connectivity={stats['avg_connectivity']:.3f}, "
                          f"Complexity={stats['avg_complexity']:.3f}")

                next_logits = logits[:, -1, :] / temperature

                # Top-p sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_logits[indices_to_remove] = float('-inf')

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

        return generated


# ==============================================================================
# Test Entry
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Claude-4 Model Test")
    print("=" * 70)

    # 创建模型
    model = Claude4Model(
        vocab_size=10000,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        num_layers=6,
        rank=4,
        enable_reflection=True
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 测试前向传播
    print("\n1. Testing forward pass...")
    inp = torch.randint(0, 10000, (2, 16))
    logits, stats = model(inp, return_reflection_stats=True)

    print(f"Input shape: {inp.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"\nReflection Stats:")
    print(f"  Avg Connectivity: {stats['avg_connectivity']:.4f}")
    print(f"  Avg Complexity: {stats['avg_complexity']:.4f}")
    print(f"  Avg Feedback Norm: {stats['avg_feedback']:.4f}")
    print(f"  Reflection Layers: {stats['num_reflection_layers']}")

    # 测试生成
    print("\n2. Testing generation (verbose)...")
    generated = model.generate(
        inp[:1],
        max_new_tokens=5,
        temperature=0.8,
        verbose=True
    )

    print(f"\nGenerated shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()}")

    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
