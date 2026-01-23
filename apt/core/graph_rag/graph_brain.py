"""
图脑 (Graph Brain) - 动态认知拓扑系统
基于非平衡态统计物理学的AI认知建模

核心机制:
1. 认知自由能最小化: F = U - T·S
2. 动力学演化: dP/dt = -∇F + 驱动
3. 拓扑相变 (CPHL): 结构重组

作者: chen0430tw
理论基础: 泛图分析 (GGA) + 复杂系统自组织
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Tuple, Optional, Callable
import logging
from dataclasses import dataclass, field

from .generalized_graph import GeneralizedGraph
from .hodge_laplacian import HodgeLaplacian

logger = logging.getLogger(__name__)


@dataclass
class GraphBrainState:
    """图脑状态快照"""
    timestamp: float
    potentials: Dict[int, np.ndarray]  # {dimension: P(t)}
    weights: Dict[int, np.ndarray]  # {dimension: W(t)}
    free_energy: float
    structural_tension: float
    structural_entropy: float
    
    def copy(self):
        """深拷贝"""
        return GraphBrainState(
            timestamp=self.timestamp,
            potentials={d: p.copy() for d, p in self.potentials.items()},
            weights={d: w.copy() for d, w in self.weights.items()},
            free_energy=self.free_energy,
            structural_tension=self.structural_tension,
            structural_entropy=self.structural_entropy
        )


class GraphBrainEngine:
    """
    图脑动力学引擎
    
    实现:
    - 认知自由能计算
    - 状态演化 (势能+权重)
    - 拓扑相变检测
    - Hebb学习
    - 吸引子形成
    """
    
    def __init__(
        self,
        gg: GeneralizedGraph,
        T_cog: float = 1.0,
        tau_p: float = 1.0,
        tau_w: float = 10.0,
        gamma: float = 0.1,
        eta: float = 0.01
    ):
        """
        Args:
            gg: 泛图对象
            T_cog: 认知温度 (探索vs固化)
            tau_p: 势能弛豫时间
            tau_w: 权重弛豫时间
            gamma: 自由能梯度系数
            eta: Hebb学习率
        """
        self.gg = gg
        self.hodge = HodgeLaplacian(gg)
        
        # 超参数
        self.T_cog = T_cog
        self.tau_p = tau_p
        self.tau_w = tau_w
        self.gamma = gamma
        self.eta = eta
        
        # 初始化状态
        self.state = self._initialize_state()
        
        # 历史记录
        self.history: List[GraphBrainState] = [self.state.copy()]
        
        # 相变阈值
        self.cphl_threshold = 0.1  # ΔF阈值
        
        logger.info(f"图脑初始化: T_cog={T_cog}, tau_p={tau_p}, tau_w={tau_w}")
    
    def _initialize_state(self) -> GraphBrainState:
        """初始化图脑状态"""
        potentials = {}
        weights = {}
        
        for p in range(self.gg.max_dimension + 1):
            n = len(self.gg.get_all_cell_ids(p))
            
            # 初始势能: 均匀分布 + 小扰动
            potentials[p] = np.ones(n) / n + 0.01 * np.random.randn(n)
            potentials[p] = np.maximum(potentials[p], 0)  # 非负
            
            # 初始权重: 细胞权重
            cell_ids = self.gg.get_all_cell_ids(p)
            weights[p] = np.array([
                self.gg.get_cell(p, cid).weight
                for cid in cell_ids
            ])
        
        F, U, S = self._compute_free_energy(potentials, weights)
        
        return GraphBrainState(
            timestamp=0.0,
            potentials=potentials,
            weights=weights,
            free_energy=F,
            structural_tension=U,
            structural_entropy=S
        )
    
    # ==================== 自由能计算 ====================
    
    def _compute_free_energy(
        self,
        potentials: Dict[int, np.ndarray],
        weights: Dict[int, np.ndarray]
    ) -> Tuple[float, float, float]:
        """
        计算认知自由能
        
        F = U - T_cog * S
        
        其中:
        U = Σ w_e * Φ(e, P)  (结构张力)
        S = -Σ w_e log w_e   (结构熵)
        
        Returns:
            (F, U, S)
        """
        # 结构张力 U
        U = 0.0
        for p in range(1, self.gg.max_dimension + 1):
            cell_ids = self.gg.get_all_cell_ids(p)
            P_p = potentials[p]
            W_p = weights[p]
            
            for i, cid in enumerate(cell_ids):
                cell = self.gg.get_cell(p, cid)
                
                # 边界势能差异 (冲突度)
                boundary_potentials = []
                for b_id in cell.boundary:
                    try:
                        b_idx = self.gg.get_all_cell_ids(p-1).index(b_id)
                        boundary_potentials.append(potentials[p-1][b_idx])
                    except ValueError:
                        pass
                
                if boundary_potentials:
                    # 势能方差 (边界不一致性)
                    variance = np.var(boundary_potentials)
                    # 与自身势能的差异
                    mean_boundary = np.mean(boundary_potentials)
                    diff = abs(P_p[i] - mean_boundary)
                    
                    # 冲突函数 Φ
                    Phi = variance + diff
                    U += W_p[i] * Phi
        
        # 结构熵 S
        S = 0.0
        for p in range(self.gg.max_dimension + 1):
            W_p = weights[p]
            W_p_norm = W_p / (np.sum(W_p) + 1e-10)  # 归一化
            W_p_norm = np.maximum(W_p_norm, 1e-10)  # 避免log(0)
            S += -np.sum(W_p_norm * np.log(W_p_norm))
        
        # 自由能
        F = U - self.T_cog * S
        
        return F, U, S
    
    def compute_free_energy_gradient(
        self
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """
        计算自由能梯度
        
        Returns:
            (∇_P F, ∇_W F)
        """
        P = self.state.potentials
        W = self.state.weights
        
        # 数值梯度 (有限差分)
        grad_P = {}
        grad_W = {}
        
        epsilon = 1e-5
        
        for p in range(self.gg.max_dimension + 1):
            n = len(P[p])
            
            # 势能梯度
            grad_P[p] = np.zeros(n)
            for i in range(n):
                P_plus = P[p].copy()
                P_plus[i] += epsilon
                P_perturbed = {**P, p: P_plus}
                F_plus, _, _ = self._compute_free_energy(P_perturbed, W)
                
                grad_P[p][i] = (F_plus - self.state.free_energy) / epsilon
            
            # 权重梯度
            grad_W[p] = np.zeros(n)
            for i in range(n):
                W_plus = W[p].copy()
                W_plus[i] += epsilon
                W_perturbed = {**W, p: W_plus}
                F_plus, _, _ = self._compute_free_energy(P, W_perturbed)
                
                grad_W[p][i] = (F_plus - self.state.free_energy) / epsilon
        
        return grad_P, grad_W
    
    # ==================== 动力学演化 ====================
    
    def evolve_step(
        self,
        dt: float,
        external_drive: Optional[Dict[int, np.ndarray]] = None
    ) -> float:
        """
        时间步演化
        
        dP/dt = -1/τ_p * ∇_P F + 扩散 + 外部驱动
        dW/dt = -γ/τ_w * ∇_W F + η * Hebb(P) + 噪声
        
        Args:
            dt: 时间步长
            external_drive: 外部驱动 {dimension: drive_vector}
        
        Returns:
            ΔF (自由能变化)
        """
        logger.debug(f"演化步: dt={dt}")
        
        # 当前状态
        P = self.state.potentials
        W = self.state.weights
        F_old = self.state.free_energy
        
        # 计算梯度
        grad_P, grad_W = self.compute_free_energy_gradient()
        
        # 更新势能
        P_new = {}
        for p in range(self.gg.max_dimension + 1):
            # 梯度下降
            dP = -1.0 / self.tau_p * grad_P[p]
            
            # 扩散项 (邻域势能传播)
            diffusion = self._compute_diffusion(p, P[p])
            dP += diffusion
            
            # 外部驱动
            if external_drive and p in external_drive:
                dP += external_drive[p]
            
            # 更新
            P_new[p] = P[p] + dt * dP
            
            # 非负约束 + 归一化
            P_new[p] = np.maximum(P_new[p], 0)
            P_sum = np.sum(P_new[p])
            if P_sum > 0:
                P_new[p] /= P_sum
        
        # 更新权重
        W_new = {}
        for p in range(self.gg.max_dimension + 1):
            # 梯度下降
            dW = -self.gamma / self.tau_w * grad_W[p]
            
            # Hebb学习 (频繁激活的连接增强)
            hebb_term = self.eta * self._compute_hebb_term(p, P_new[p])
            dW += hebb_term
            
            # 小噪声
            noise = 0.001 * np.random.randn(len(W[p]))
            dW += noise
            
            # 更新
            W_new[p] = W[p] + dt * dW
            
            # 正约束
            W_new[p] = np.maximum(W_new[p], 0.01)
        
        # 计算新自由能
        F_new, U_new, S_new = self._compute_free_energy(P_new, W_new)
        
        # 更新状态
        self.state = GraphBrainState(
            timestamp=self.state.timestamp + dt,
            potentials=P_new,
            weights=W_new,
            free_energy=F_new,
            structural_tension=U_new,
            structural_entropy=S_new
        )
        
        # 记录历史
        self.history.append(self.state.copy())
        
        # 计算变化
        delta_F = F_new - F_old
        
        logger.debug(
            f"演化完成: F={F_new:.4f}, ΔF={delta_F:.6f}, "
            f"U={U_new:.4f}, S={S_new:.4f}"
        )
        
        # 检测相变
        if abs(delta_F) > self.cphl_threshold:
            logger.warning(f"检测到拓扑相变! |ΔF|={abs(delta_F):.4f}")
            self._trigger_phase_transition()
        
        return delta_F
    
    def _compute_diffusion(self, dimension: int, potentials: np.ndarray) -> np.ndarray:
        """
        计算扩散项 (势能在邻域传播)
        
        d_diffusion = Σ_neighbors (P_j - P_i)
        """
        cell_ids = self.gg.get_all_cell_ids(dimension)
        n = len(cell_ids)
        diffusion = np.zeros(n)
        
        for i, cid in enumerate(cell_ids):
            neighbors = self.gg.get_neighbors(dimension, cid)
            
            for nb_id in neighbors:
                try:
                    j = cell_ids.index(nb_id)
                    diffusion[i] += (potentials[j] - potentials[i])
                except ValueError:
                    pass
        
        return 0.1 * diffusion  # 缩放因子
    
    def _compute_hebb_term(self, dimension: int, potentials: np.ndarray) -> np.ndarray:
        """
        Hebb学习项: 同时激活的细胞连接增强
        
        ΔW_ij ∝ P_i * P_j
        """
        cell_ids = self.gg.get_all_cell_ids(dimension)
        n = len(cell_ids)
        hebb = np.zeros(n)
        
        for i, cid in enumerate(cell_ids):
            cell = self.gg.get_cell(dimension, cid)
            
            # 与边界细胞的共激活
            boundary_activations = []
            for b_id in cell.boundary:
                try:
                    b_idx = self.gg.get_all_cell_ids(dimension-1).index(b_id)
                    boundary_activations.append(
                        self.state.potentials[dimension-1][b_idx]
                    )
                except:
                    pass
            
            if boundary_activations:
                # Hebb权重增量 ∝ P_i * mean(P_boundary)
                hebb[i] = potentials[i] * np.mean(boundary_activations)
        
        return hebb
    
    # ==================== 拓扑相变 (CPHL) ====================
    
    def _trigger_phase_transition(self):
        """触发认知范式热加载 (CPHL)"""
        logger.info("========== 拓扑相变开始 ==========")
        
        # 1. 识别高势能节点 (催化剂)
        catalysts = self._identify_catalysts()
        
        # 2. 结构重组 (雪崩式调整)
        self._restructure_topology(catalysts)
        
        # 3. 形成新吸引子
        self._form_attractors()
        
        logger.info("========== 拓扑相变完成 ==========")
    
    def _identify_catalysts(self) -> Dict[int, List[int]]:
        """识别催化节点 (势能前10%)"""
        catalysts = {}
        
        for p in range(self.gg.max_dimension + 1):
            P_p = self.state.potentials[p]
            threshold = np.percentile(P_p, 90)
            catalysts[p] = np.where(P_p > threshold)[0].tolist()
        
        return catalysts
    
    def _restructure_topology(self, catalysts: Dict[int, List[int]]):
        """结构重组: 增强催化节点周围的连接"""
        for p, indices in catalysts.items():
            if len(indices) == 0:
                continue
            
            W_p = self.state.weights[p]
            # 增强催化节点权重
            W_p[indices] *= 1.5
            # 重新归一化
            W_p /= np.sum(W_p)
            self.state.weights[p] = W_p
    
    def _form_attractors(self):
        """形成吸引子 (高势能节点)"""
        for p in range(self.gg.max_dimension + 1):
            P_p = self.state.potentials[p]
            
            # Softmax 锐化 (强者更强)
            P_sharp = np.exp(5.0 * P_p)
            P_sharp /= np.sum(P_sharp)
            
            self.state.potentials[p] = P_sharp
    
    # ==================== 查询与激活 ====================
    
    def activate_cells(
        self,
        dimension: int,
        cell_indices: List[int],
        strength: float = 1.0
    ):
        """
        激活指定细胞 (模拟查询/输入)
        
        Args:
            dimension: 维度
            cell_indices: 细胞索引列表
            strength: 激活强度
        """
        P_p = self.state.potentials[dimension]
        
        # 激活指定细胞
        for idx in cell_indices:
            if 0 <= idx < len(P_p):
                P_p[idx] += strength
        
        # 归一化
        P_p /= np.sum(P_p)
        self.state.potentials[dimension] = P_p
        
        logger.info(f"激活 {len(cell_indices)} 个细胞 (dim={dimension}, strength={strength})")
    
    def get_activated_cells(
        self,
        dimension: int,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        获取当前最激活的细胞
        
        Returns:
            [(cell_id, potential), ...]
        """
        P_p = self.state.potentials[dimension]
        cell_ids = self.gg.get_all_cell_ids(dimension)
        
        # 排序
        indices = np.argsort(P_p)[::-1][:top_k]
        
        return [
            (cell_ids[i], float(P_p[i]))
            for i in indices
        ]
    
    # ==================== 统计与可视化 ====================
    
    def get_evolution_summary(self) -> Dict:
        """获取演化摘要"""
        if len(self.history) < 2:
            return {}
        
        F_history = [s.free_energy for s in self.history]
        U_history = [s.structural_tension for s in self.history]
        S_history = [s.structural_entropy for s in self.history]
        
        return {
            'num_steps': len(self.history),
            'final_F': F_history[-1],
            'delta_F_total': F_history[-1] - F_history[0],
            'F_min': min(F_history),
            'F_max': max(F_history),
            'avg_U': np.mean(U_history),
            'avg_S': np.mean(S_history),
            'phase_transitions': sum(
                1 for i in range(1, len(F_history))
                if abs(F_history[i] - F_history[i-1]) > self.cphl_threshold
            )
        }
    
    def print_state_report(self):
        """打印当前状态报告"""
        print("=" * 60)
        print("图脑状态报告")
        print("=" * 60)
        
        print(f"\n时间: t={self.state.timestamp:.2f}")
        print(f"自由能 F = {self.state.free_energy:.4f}")
        print(f"  结构张力 U = {self.state.structural_tension:.4f}")
        print(f"  结构熵 S = {self.state.structural_entropy:.4f}")
        print(f"  认知温度 T = {self.T_cog:.2f}")
        
        print("\n各维度激活最高的细胞:")
        for p in range(self.gg.max_dimension + 1):
            top_cells = self.get_activated_cells(p, top_k=3)
            print(f"  {p}-细胞:")
            for cid, pot in top_cells:
                print(f"    {cid}: P={pot:.4f}")
        
        summary = self.get_evolution_summary()
        if summary:
            print(f"\n演化摘要:")
            print(f"  总步数: {summary['num_steps']}")
            print(f"  相变次数: {summary['phase_transitions']}")
            print(f"  ΔF总计: {summary['delta_F_total']:.4f}")
        
        print("=" * 60)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    from generalized_graph import GeneralizedGraph
    
    print("=== 测试: 图脑演化 ===\n")
    
    # 构建知识图谱
    triples = [
        ("AI", "需要", "数据"),
        ("AI", "需要", "算力"),
        ("数据", "来自", "互联网"),
        ("算力", "来自", "GPU")
    ]
    
    kg = GeneralizedGraph.from_knowledge_triples(triples)
    brain = GraphBrainEngine(kg, T_cog=1.0)
    
    # 初始状态
    brain.print_state_report()
    
    # 演化100步
    print("\n开始演化...")
    for step in range(100):
        delta_F = brain.evolve_step(dt=0.1)
        
        if step % 20 == 0:
            print(f"Step {step}: F={brain.state.free_energy:.4f}, ΔF={delta_F:.6f}")
    
    # 最终状态
    print("\n")
    brain.print_state_report()
