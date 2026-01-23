"""
Hodge-Laplacian 谱分析模块
实现泛图的谱特征计算和拓扑分析

核心公式:
L_p = d_{p-1} δ_{p-1} + δ_p d_p

其中:
- d_p: 外微分算子 (exterior derivative)
- δ_p: 余微分算子 (codifferential)
- L_p: p阶 Hodge-Laplacian
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Dict, List, Tuple, Optional
import logging

from .generalized_graph import GeneralizedGraph

logger = logging.getLogger(__name__)


class HodgeLaplacian:
    """
    Hodge-Laplacian 计算器
    
    提供:
    - 外微分和余微分算子
    - 各维度Laplacian矩阵
    - 谱分解 (特征值/特征向量)
    - Betti数计算 (拓扑不变量)
    - Hodge分解
    """
    
    def __init__(self, gg: GeneralizedGraph):
        """
        Args:
            gg: 泛图对象
        """
        self.gg = gg
        
        # Laplacian缓存 {dimension: L_p}
        self.laplacians: Dict[int, sp.spmatrix] = {}
        
        # 谱缓存 {dimension: (eigenvalues, eigenvectors)}
        self.spectra: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        
        # Betti数缓存
        self.betti_numbers: Optional[List[int]] = None
    
    # ==================== 基础算子 ====================
    
    def exterior_derivative(self, dimension: int) -> sp.spmatrix:
        """
        外微分算子 d_p: C^p -> C^{p+1}
        
        d_p = B_{p+1}^T (转置关联矩阵)
        
        Args:
            dimension: 维度p
        
        Returns:
            稀疏矩阵
        """
        if dimension >= self.gg.max_dimension:
            # 最高维没有外微分
            n = len(self.gg.get_all_cell_ids(dimension))
            return sp.csr_matrix((0, n))
        
        B_p1 = self.gg.compute_incidence_matrix(dimension + 1)
        d_p = B_p1.T  # 转置
        
        logger.debug(f"外微分 d_{dimension}: shape={d_p.shape}")
        return d_p
    
    def codifferential(self, dimension: int) -> sp.spmatrix:
        """
        余微分算子 δ_p: C^p -> C^{p-1}
        
        δ_p = B_p (关联矩阵)
        
        Args:
            dimension: 维度p
        
        Returns:
            稀疏矩阵
        """
        if dimension == 0:
            # 0维没有余微分
            n = len(self.gg.get_all_cell_ids(0))
            return sp.csr_matrix((0, n))
        
        B_p = self.gg.compute_incidence_matrix(dimension)
        delta_p = B_p
        
        logger.debug(f"余微分 δ_{dimension}: shape={delta_p.shape}")
        return delta_p
    
    # ==================== Hodge-Laplacian ====================
    
    def compute_laplacian(
        self,
        dimension: int,
        use_weights: bool = True,
        normalize: bool = False
    ) -> sp.spmatrix:
        """
        计算p阶 Hodge-Laplacian
        
        L_p = d_{p-1} δ_{p-1} + δ_p d_p
        
        Args:
            dimension: 维度p
            use_weights: 是否使用细胞权重
            normalize: 是否归一化
        
        Returns:
            L_p 矩阵 (对称稀疏矩阵)
        """
        if dimension in self.laplacians:
            return self.laplacians[dimension]
        
        logger.info(f"计算 L_{dimension} Hodge-Laplacian...")
        
        # 第一项: d_{p-1} δ_{p-1}
        if dimension > 0:
            d_pm1 = self.exterior_derivative(dimension - 1)
            delta_pm1 = self.codifferential(dimension - 1)
            term1 = d_pm1 @ delta_pm1
        else:
            n = len(self.gg.get_all_cell_ids(0))
            term1 = sp.csr_matrix((n, n))
        
        # 第二项: δ_p d_p
        if dimension < self.gg.max_dimension:
            delta_p = self.codifferential(dimension)
            d_p = self.exterior_derivative(dimension)
            term2 = delta_p @ d_p
        else:
            n = len(self.gg.get_all_cell_ids(dimension))
            term2 = sp.csr_matrix((n, n))
        
        # 组合
        L_p = term1 + term2
        
        # 应用权重
        if use_weights:
            weights = self._get_weight_vector(dimension)
            W = sp.diags(weights)
            L_p = W @ L_p @ W
        
        # 归一化 (类似规范化拉普拉斯)
        if normalize:
            degrees = np.array(L_p.sum(axis=1)).flatten()
            degrees[degrees == 0] = 1  # 避免除零
            D_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees))
            L_p = D_inv_sqrt @ L_p @ D_inv_sqrt
        
        # 确保对称性 (数值稳定)
        L_p = 0.5 * (L_p + L_p.T)
        
        self.laplacians[dimension] = L_p
        
        logger.info(f"L_{dimension} 计算完成: shape={L_p.shape}, nnz={L_p.nnz}")
        
        return L_p
    
    def _get_weight_vector(self, dimension: int) -> np.ndarray:
        """获取权重向量"""
        cell_ids = self.gg.get_all_cell_ids(dimension)
        weights = np.array([
            self.gg.get_cell(dimension, cid).weight
            for cid in cell_ids
        ])
        return weights
    
    # ==================== 谱分解 ====================
    
    def compute_spectrum(
        self,
        dimension: int,
        k: Optional[int] = None,
        which: str = 'SM'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算L_p的谱 (特征值和特征向量)
        
        Args:
            dimension: 维度p
            k: 计算前k个特征值 (None=全部)
            which: 'SM'(最小) 'LM'(最大) 'SA'(代数最小)
        
        Returns:
            (eigenvalues, eigenvectors)
            eigenvalues: shape (k,)
            eigenvectors: shape (n, k)
        """
        if dimension in self.spectra:
            return self.spectra[dimension]
        
        L_p = self.compute_laplacian(dimension)
        n = L_p.shape[0]
        
        if n == 0:
            return np.array([]), np.array([]).reshape(0, 0)
        
        # 自动选择k
        if k is None:
            k = min(20, n - 1)  # 默认计算前20个
        k = min(k, n - 1)
        
        if k <= 0:
            return np.array([0.0]), np.eye(n)
        
        logger.info(f"计算 L_{dimension} 的谱: k={k}, which={which}")
        
        try:
            # 使用稀疏特征值求解器
            eigenvalues, eigenvectors = spla.eigsh(
                L_p.astype(np.float64),
                k=k,
                which=which,
                return_eigenvectors=True
            )
            
            # 排序 (从小到大)
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            self.spectra[dimension] = (eigenvalues, eigenvectors)
            
            logger.info(
                f"谱计算完成: "
                f"λ_min={eigenvalues[0]:.6f}, "
                f"λ_max={eigenvalues[-1]:.6f}"
            )
            
        except Exception as e:
            logger.error(f"谱计算失败: {e}")
            # 回退到密集计算 (小规模)
            if n < 100:
                L_dense = L_p.toarray()
                eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
                self.spectra[dimension] = (eigenvalues, eigenvectors)
            else:
                raise
        
        return eigenvalues, eigenvectors
    
    def get_smallest_nonzero_eigenvalue(
        self,
        dimension: int,
        tol: float = 1e-8
    ) -> float:
        """
        获取最小非零特征值 λ_1(L_p)
        
        对于L_0: λ_1 衡量图的连通性 (Cheeger常数)
        
        Args:
            dimension: 维度p
            tol: 零阈值
        
        Returns:
            λ_1 (最小非零特征值)
        """
        eigenvalues, _ = self.compute_spectrum(dimension, k=10)
        
        # 找第一个非零特征值
        nonzero = eigenvalues[eigenvalues > tol]
        
        if len(nonzero) == 0:
            logger.warning(f"L_{dimension} 没有非零特征值")
            return 0.0
        
        return float(nonzero[0])
    
    def get_spectral_gap(self, dimension: int) -> float:
        """
        计算谱间隙 λ_2 - λ_1
        
        谱间隙越大 -> 系统越稳定
        """
        eigenvalues, _ = self.compute_spectrum(dimension, k=15)
        
        if len(eigenvalues) < 2:
            return 0.0
        
        # 过滤零特征值
        nonzero = eigenvalues[eigenvalues > 1e-8]
        
        if len(nonzero) < 2:
            return 0.0
        
        return float(nonzero[1] - nonzero[0])
    
    # ==================== Betti数 (拓扑不变量) ====================
    
    def compute_betti_numbers(self, tol: float = 1e-6) -> List[int]:
        """
        计算各维度Betti数 β_p
        
        β_p = dim(ker L_p) = 第p个Betti数
        
        物理意义:
        - β_0: 连通分支数
        - β_1: 独立环的数量 (循环数)
        - β_2: 空腔数量 (孔洞)
        - β_p: p维"洞"的数量
        
        Returns:
            [β_0, β_1, ..., β_P]
        """
        if self.betti_numbers is not None:
            return self.betti_numbers
        
        logger.info("计算Betti数...")
        
        betti_numbers = []
        
        for p in range(self.gg.max_dimension + 1):
            L_p = self.compute_laplacian(p)
            
            if L_p.shape[0] == 0:
                betti_numbers.append(0)
                continue
            
            # 计算所有特征值
            try:
                # 小矩阵: 密集计算
                if L_p.shape[0] < 100:
                    eigenvalues = np.linalg.eigvalsh(L_p.toarray())
                else:
                    # 大矩阵: 稀疏计算 (估计)
                    k = min(50, L_p.shape[0] - 1)
                    eigenvalues, _ = spla.eigsh(L_p, k=k, which='SM')
                
                # 统计接近零的特征值数量
                beta_p = np.sum(np.abs(eigenvalues) < tol)
                betti_numbers.append(int(beta_p))
                
                logger.info(f"β_{p} = {beta_p}")
                
            except Exception as e:
                logger.error(f"计算 β_{p} 失败: {e}")
                betti_numbers.append(0)
        
        self.betti_numbers = betti_numbers
        return betti_numbers
    
    def euler_characteristic(self) -> int:
        """
        计算欧拉示性数 χ
        
        χ = Σ (-1)^p * num_p-cells = Σ (-1)^p * β_p
        
        拓扑不变量
        """
        betti_nums = self.compute_betti_numbers()
        
        chi = sum(
            (-1) ** p * beta
            for p, beta in enumerate(betti_nums)
        )
        
        return chi
    
    # ==================== Hodge分解 ====================
    
    def hodge_decomposition(
        self,
        dimension: int,
        signal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Hodge分解: x = gradient + curl + harmonic
        
        x^{(p)} = d_{p-1}α + δ_p β + h
        
        其中:
        - gradient (势流): 梯度部分
        - curl (旋度): 余梯度部分  
        - harmonic: 调和部分 (ker L_p)
        
        Args:
            dimension: 维度p
            signal: 信号 x ∈ R^{|I_p|}
        
        Returns:
            (gradient, curl, harmonic)
        """
        logger.info(f"Hodge分解 (dim={dimension})")
        
        # 获取算子
        d_pm1 = self.exterior_derivative(dimension - 1)
        delta_p = self.codifferential(dimension)
        L_p = self.compute_laplacian(dimension)
        
        # 计算调和部分 (核空间投影)
        eigenvalues, eigenvectors = self.compute_spectrum(dimension, k=20)
        zero_mask = eigenvalues < 1e-6
        harmonic_basis = eigenvectors[:, zero_mask]
        
        if harmonic_basis.shape[1] > 0:
            harmonic = harmonic_basis @ (harmonic_basis.T @ signal)
        else:
            harmonic = np.zeros_like(signal)
        
        # 非调和部分
        non_harmonic = signal - harmonic
        
        # 求解 L_p α' = non_harmonic 的最小二乘
        try:
            alpha_prime = spla.lsqr(L_p, non_harmonic)[0]
        except:
            alpha_prime = np.zeros_like(non_harmonic)
        
        # 梯度部分
        if d_pm1.shape[1] > 0:
            # 求解 d_{p-1} α = alpha_prime
            alpha = spla.lsqr(d_pm1, alpha_prime)[0]
            gradient = d_pm1 @ alpha
        else:
            gradient = np.zeros_like(signal)
        
        # 旋度部分
        if delta_p.shape[1] > 0:
            curl_prime = non_harmonic - gradient
            beta = spla.lsqr(delta_p, curl_prime)[0]
            curl = delta_p @ beta
        else:
            curl = np.zeros_like(signal)
        
        # 验证分解
        reconstruction = gradient + curl + harmonic
        error = np.linalg.norm(signal - reconstruction)
        logger.info(f"Hodge分解误差: {error:.6f}")
        
        return gradient, curl, harmonic
    
    # ==================== 工具方法 ====================
    
    def get_topology_summary(self) -> Dict:
        """获取拓扑摘要"""
        betti_nums = self.compute_betti_numbers()
        
        summary = {
            'betti_numbers': betti_nums,
            'euler_characteristic': self.euler_characteristic(),
            'spectral_gaps': [],
            'smallest_nonzero_eigenvalues': []
        }
        
        for p in range(self.gg.max_dimension + 1):
            try:
                gap = self.get_spectral_gap(p)
                lambda_1 = self.get_smallest_nonzero_eigenvalue(p)
                summary['spectral_gaps'].append(gap)
                summary['smallest_nonzero_eigenvalues'].append(lambda_1)
            except:
                summary['spectral_gaps'].append(0.0)
                summary['smallest_nonzero_eigenvalues'].append(0.0)
        
        return summary
    
    def print_topology_report(self):
        """打印拓扑报告"""
        summary = self.get_topology_summary()
        
        print("=" * 60)
        print("拓扑分析报告")
        print("=" * 60)
        
        print(f"\n欧拉示性数 χ = {summary['euler_characteristic']}")
        
        print("\nBetti数 (拓扑孔洞):")
        for p, beta in enumerate(summary['betti_numbers']):
            desc = {
                0: "连通分支",
                1: "独立环",
                2: "空腔",
                3: "高维孔洞"
            }.get(p, f"{p}维孔洞")
            print(f"  β_{p} = {beta:3d}  ({desc})")
        
        print("\n谱特征:")
        for p in range(len(summary['spectral_gaps'])):
            gap = summary['spectral_gaps'][p]
            lambda_1 = summary['smallest_nonzero_eigenvalues'][p]
            print(f"  L_{p}: λ_1={lambda_1:.6f}, gap={gap:.6f}")
        
        print("=" * 60)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    from generalized_graph import GeneralizedGraph
    
    print("=== 测试: Hodge-Laplacian ===")
    
    # 构建简单图: 三角形 + 一个孤立点
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    gg = GeneralizedGraph.from_edge_list(edges)
    gg.add_cell(0, "D")  # 孤立点
    
    hodge = HodgeLaplacian(gg)
    hodge.print_topology_report()
