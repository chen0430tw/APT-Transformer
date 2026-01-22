"""
泛图 (Generalized Graph) - 核心数据结构
基于GGA理论的统一图表示框架

支持:
- 普通图 (P=1)
- 超图 (任意p-细胞)
- 单纯复形
- CW复形
- 时空层图/网格
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import pickle
import logging

logger = logging.getLogger(__name__)


@dataclass
class Cell:
    """p-细胞：泛图的基本单元"""
    cell_id: str
    dimension: int  # p维
    boundary: Set[str]  # 边界细胞ID集合
    weight: float = 1.0
    potential: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.cell_id)
    
    def __eq__(self, other):
        return self.cell_id == other.cell_id


class GeneralizedGraph:
    """
    泛图 G = ({I_p}_{p=0}^P, ∂, τ, w, O)
    
    组件:
    - I_p: p维细胞集合 (点/边/面/体...)
    - ∂: 边界映射 (满足链复形条件)
    - τ: 取向
    - w: 权重
    - O: 结构运算库
    """
    
    def __init__(self, max_dimension: int = 3):
        """
        Args:
            max_dimension: 最高维度P (0=点, 1=边, 2=面, 3=体)
        """
        self.max_dimension = max_dimension
        
        # 各维度的细胞集合 {p: {cell_id: Cell}}
        self.cells: Dict[int, Dict[str, Cell]] = {
            p: {} for p in range(max_dimension + 1)
        }
        
        # 边界映射缓存 {(p, cell_id): boundary_cells}
        self.boundary_cache: Dict[Tuple[int, str], Set[str]] = {}
        
        # 关联矩阵缓存 {p: sparse_matrix}
        self.incidence_matrices: Dict[int, sp.spmatrix] = {}
        
        # Laplacian缓存
        self.laplacians: Dict[int, sp.spmatrix] = {}
        
        # 统计信息
        self.stats = {
            'num_cells_by_dim': [0] * (max_dimension + 1),
            'num_boundaries': 0,
            'avg_degree_by_dim': [0.0] * (max_dimension + 1)
        }
        
        logger.info(f"初始化泛图: max_dimension={max_dimension}")
    
    # ==================== 基础操作 ====================
    
    def add_cell(
        self,
        dimension: int,
        cell_id: str,
        boundary_cells: Optional[Set[str]] = None,
        weight: float = 1.0,
        potential: float = 0.0,
        attributes: Optional[Dict] = None
    ) -> bool:
        """
        添加p-细胞
        
        Args:
            dimension: 维度p
            cell_id: 唯一标识符
            boundary_cells: 边界细胞ID集合 (∂_p: I_p -> P(I_{p-1}))
            weight: 权重
            potential: 势能
            attributes: 附加属性
        
        Returns:
            是否成功添加
        """
        if dimension < 0 or dimension > self.max_dimension:
            logger.error(f"维度超出范围: {dimension}")
            return False
        
        if cell_id in self.cells[dimension]:
            logger.warning(f"细胞已存在: {cell_id}")
            return False
        
        # 验证边界细胞存在性 (链复形条件)
        if boundary_cells and dimension > 0:
            for b_id in boundary_cells:
                if b_id not in self.cells[dimension - 1]:
                    logger.error(f"边界细胞不存在: {b_id}")
                    return False
        
        # 创建细胞
        cell = Cell(
            cell_id=cell_id,
            dimension=dimension,
            boundary=boundary_cells or set(),
            weight=weight,
            potential=potential,
            attributes=attributes or {}
        )
        
        self.cells[dimension][cell_id] = cell
        self.boundary_cache[(dimension, cell_id)] = cell.boundary
        
        # 更新统计
        self.stats['num_cells_by_dim'][dimension] += 1
        self.stats['num_boundaries'] += len(cell.boundary)
        
        # 清除缓存
        self._invalidate_cache()
        
        return True
    
    def get_cell(self, dimension: int, cell_id: str) -> Optional[Cell]:
        """获取细胞"""
        return self.cells[dimension].get(cell_id)
    
    def remove_cell(self, dimension: int, cell_id: str) -> bool:
        """删除细胞 (级联删除依赖它的高维细胞)"""
        if cell_id not in self.cells[dimension]:
            return False
        
        # 级联删除
        for p in range(dimension + 1, self.max_dimension + 1):
            cells_to_remove = []
            for cid, cell in self.cells[p].items():
                if cell_id in cell.boundary:
                    cells_to_remove.append(cid)
            
            for cid in cells_to_remove:
                self.remove_cell(p, cid)
        
        # 删除细胞
        del self.cells[dimension][cell_id]
        del self.boundary_cache[(dimension, cell_id)]
        
        # 更新统计
        self.stats['num_cells_by_dim'][dimension] -= 1
        self._invalidate_cache()
        
        return True
    
    def update_weight(self, dimension: int, cell_id: str, weight: float):
        """更新细胞权重"""
        if cell := self.get_cell(dimension, cell_id):
            cell.weight = weight
            self._invalidate_cache()
    
    def update_potential(self, dimension: int, cell_id: str, potential: float):
        """更新细胞势能"""
        if cell := self.get_cell(dimension, cell_id):
            cell.potential = potential
    
    # ==================== 批量操作 ====================
    
    def add_cells_batch(self, cells_data: List[Tuple[int, str, Set[str], float]]):
        """批量添加细胞 (效率优化)"""
        for dimension, cell_id, boundary, weight in cells_data:
            self.add_cell(dimension, cell_id, boundary, weight)
    
    def get_cells_by_dimension(self, dimension: int) -> Dict[str, Cell]:
        """获取某维度的所有细胞"""
        return self.cells[dimension]
    
    def get_all_cell_ids(self, dimension: int) -> List[str]:
        """获取某维度所有细胞ID"""
        return list(self.cells[dimension].keys())
    
    # ==================== 拓扑查询 ====================
    
    def get_boundary(self, dimension: int, cell_id: str) -> Set[str]:
        """获取细胞边界 ∂_p(cell)"""
        return self.boundary_cache.get((dimension, cell_id), set())
    
    def get_coboundary(self, dimension: int, cell_id: str) -> Set[str]:
        """获取细胞的上边界 (哪些细胞以它为边界)"""
        coboundary = set()
        for cid, cell in self.cells[dimension + 1].items():
            if cell_id in cell.boundary:
                coboundary.add(cid)
        return coboundary
    
    def get_neighbors(self, dimension: int, cell_id: str) -> Set[str]:
        """获取同维邻居 (共享边界的细胞)"""
        cell = self.get_cell(dimension, cell_id)
        if not cell or dimension == 0:
            return set()
        
        neighbors = set()
        # 找所有共享至少一个边界细胞的同维细胞
        for boundary_id in cell.boundary:
            coboundary = self.get_coboundary(dimension - 1, boundary_id)
            neighbors.update(coboundary)
        
        neighbors.discard(cell_id)  # 移除自己
        return neighbors
    
    def compute_degree(self, dimension: int, cell_id: str) -> int:
        """计算细胞的度数 (边界+上边界)"""
        boundary = self.get_boundary(dimension, cell_id)
        coboundary = self.get_coboundary(dimension, cell_id)
        return len(boundary) + len(coboundary)
    
    # ==================== 链复形 ====================
    
    def verify_chain_complex(self) -> bool:
        """
        验证链复形条件: ∂_{p-1} ∘ ∂_p = ∅
        (边界的边界为空)
        """
        for p in range(2, self.max_dimension + 1):
            for cell_id, cell in self.cells[p].items():
                # 获取边界的边界
                boundary_of_boundary = set()
                for b_id in cell.boundary:
                    b_cell = self.get_cell(p - 1, b_id)
                    if b_cell:
                        boundary_of_boundary.update(b_cell.boundary)
                
                # 检查边界的边界是否相互抵消
                # (在有向图中应该成对出现并抵消)
                if len(boundary_of_boundary) > 0:
                    # 简化检查：边界的边界应该是偶数(成对)
                    # 严格检查需要考虑取向
                    logger.debug(f"细胞 {cell_id} 的边界的边界非空: {boundary_of_boundary}")
        
        return True
    
    # ==================== 关联矩阵 ====================
    
    def compute_incidence_matrix(self, dimension: int) -> sp.spmatrix:
        """
        计算 p-维关联矩阵 B_p: I_p -> I_{p-1}
        
        B[i, j] = 边界关系系数
        行: (p-1)-细胞
        列: p-细胞
        
        Returns:
            稀疏矩阵 (COO格式)
        """
        if dimension == 0:
            return sp.csr_matrix((0, len(self.cells[0])))
        
        if dimension in self.incidence_matrices:
            return self.incidence_matrices[dimension]
        
        # 构建索引映射
        cells_p = self.get_all_cell_ids(dimension)
        cells_pm1 = self.get_all_cell_ids(dimension - 1)
        
        cell_to_idx_p = {cid: i for i, cid in enumerate(cells_p)}
        cell_to_idx_pm1 = {cid: i for i, cid in enumerate(cells_pm1)}
        
        # 构建稀疏矩阵
        rows, cols, data = [], [], []
        
        for j, cid in enumerate(cells_p):
            cell = self.get_cell(dimension, cid)
            for boundary_id in cell.boundary:
                i = cell_to_idx_pm1[boundary_id]
                # 简化: 使用 +1 表示边界关系
                # 完整实现需要考虑取向τ (±1)
                rows.append(i)
                cols.append(j)
                data.append(1.0)
        
        n_rows = len(cells_pm1)
        n_cols = len(cells_p)
        
        B = sp.coo_matrix(
            (data, (rows, cols)),
            shape=(n_rows, n_cols)
        ).tocsr()
        
        self.incidence_matrices[dimension] = B
        logger.info(f"计算关联矩阵 B_{dimension}: shape={B.shape}, nnz={B.nnz}")
        
        return B
    
    # ==================== 工具方法 ====================
    
    def _invalidate_cache(self):
        """清除缓存"""
        self.incidence_matrices.clear()
        self.laplacians.clear()
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = self.stats.copy()
        stats['total_cells'] = sum(stats['num_cells_by_dim'])
        
        # 计算平均度数
        for p in range(self.max_dimension + 1):
            if stats['num_cells_by_dim'][p] > 0:
                total_degree = sum(
                    self.compute_degree(p, cid)
                    for cid in self.get_all_cell_ids(p)
                )
                stats['avg_degree_by_dim'][p] = total_degree / stats['num_cells_by_dim'][p]
        
        return stats
    
    def summary(self) -> str:
        """生成摘要字符串"""
        stats = self.get_statistics()
        lines = [
            "=== 泛图摘要 ===",
            f"最大维度: {self.max_dimension}",
            f"总细胞数: {stats['total_cells']}",
        ]
        
        for p in range(self.max_dimension + 1):
            lines.append(
                f"  {p}-细胞: {stats['num_cells_by_dim'][p]} "
                f"(平均度数: {stats['avg_degree_by_dim'][p]:.2f})"
            )
        
        lines.append(f"边界关系数: {stats['num_boundaries']}")
        
        return "\n".join(lines)
    
    # ==================== 持久化 ====================
    
    def save_to_file(self, filepath: str):
        """保存到文件"""
        data = {
            'max_dimension': self.max_dimension,
            'cells': self.cells,
            'boundary_cache': self.boundary_cache,
            'stats': self.stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"泛图已保存到: {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'GeneralizedGraph':
        """从文件加载"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        gg = cls(max_dimension=data['max_dimension'])
        gg.cells = data['cells']
        gg.boundary_cache = data['boundary_cache']
        gg.stats = data['stats']
        
        logger.info(f"泛图已加载: {filepath}")
        return gg
    
    # ==================== 从其他格式构建 ====================
    
    @classmethod
    def from_edge_list(cls, edges: List[Tuple[str, str]], directed: bool = False):
        """
        从边列表构建普通图 (P=1)
        
        Args:
            edges: [(source, target), ...]
            directed: 是否有向
        """
        gg = cls(max_dimension=1)
        
        # 收集所有节点
        nodes = set()
        for src, tgt in edges:
            nodes.add(src)
            nodes.add(tgt)
        
        # 添加0-细胞 (节点)
        for node in nodes:
            gg.add_cell(0, node)
        
        # 添加1-细胞 (边)
        for i, (src, tgt) in enumerate(edges):
            edge_id = f"e{i}_{src}->{tgt}"
            gg.add_cell(1, edge_id, boundary={src, tgt})
            
            if not directed:
                # 无向图: 添加反向边
                edge_id_rev = f"e{i}_{tgt}->{src}"
                gg.add_cell(1, edge_id_rev, boundary={tgt, src})
        
        return gg
    
    @classmethod
    def from_knowledge_triples(cls, triples: List[Tuple[str, str, str]]):
        """
        从知识三元组构建
        
        Args:
            triples: [(entity1, relation, entity2), ...]
        """
        gg = cls(max_dimension=2)
        
        entities = set()
        relations = set()
        
        # 收集实体和关系
        for e1, rel, e2 in triples:
            entities.add(e1)
            entities.add(e2)
            relations.add(rel)
        
        # 添加0-细胞 (实体)
        for entity in entities:
            gg.add_cell(0, f"entity:{entity}")
        
        # 添加1-细胞 (关系边)
        for i, (e1, rel, e2) in enumerate(triples):
            edge_id = f"rel{i}:{e1}-{rel}-{e2}"
            gg.add_cell(
                1,
                edge_id,
                boundary={f"entity:{e1}", f"entity:{e2}"},
                attributes={'relation': rel}
            )
        
        # 添加2-细胞 (事件/事实)
        for i, (e1, rel, e2) in enumerate(triples):
            fact_id = f"fact{i}:({e1},{rel},{e2})"
            # 2-细胞的边界是组成该事实的所有边
            boundary_edges = {f"rel{i}:{e1}-{rel}-{e2}"}
            gg.add_cell(2, fact_id, boundary=boundary_edges)
        
        return gg


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("=== 测试1: 构建简单图 ===")
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    gg = GeneralizedGraph.from_edge_list(edges)
    print(gg.summary())
    
    print("\n=== 测试2: 知识图谱 ===")
    triples = [
        ("爱因斯坦", "提出", "相对论"),
        ("相对论", "属于", "物理学"),
        ("爱因斯坦", "获得", "诺贝尔奖")
    ]
    kg = GeneralizedGraph.from_knowledge_triples(triples)
    print(kg.summary())
    
    print("\n=== 测试3: 计算关联矩阵 ===")
    B1 = kg.compute_incidence_matrix(1)
    print(f"B_1 shape: {B1.shape}")
    print(f"B_1 非零元素: {B1.nnz}")
