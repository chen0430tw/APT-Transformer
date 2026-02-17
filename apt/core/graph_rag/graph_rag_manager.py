"""
GraphRAG管理器 - 基于泛图分析的知识图谱RAG系统
整合: 泛图 + Hodge-Laplacian + 图脑 + 向量检索

主要功能:
1. 知识图谱构建 (从文本/三元组)
2. 谱推理 (拓扑引导)
3. 图脑动力学查询
4. 混合检索 (向量+谱+拓扑)
5. 可视化和分析
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import json

from .generalized_graph import GeneralizedGraph
from .hodge_laplacian import HodgeLaplacian
from .graph_brain import GraphBrainEngine

logger = logging.getLogger(__name__)


class GraphRAGManager:
    """
    GraphRAG管理器
    
    核心组件:
    - 泛图 (GG): 知识结构
    - Hodge-Laplacian: 谱特征
    - 图脑: 动力学查询
    - 向量存储: 嵌入检索 (可选)
    """
    
    def __init__(
        self,
        max_dimension: int = 2,
        T_cog: float = 1.0,
        enable_brain: bool = True,
        enable_spectral: bool = True
    ):
        """
        Args:
            max_dimension: 最大维度 (0=点, 1=边, 2=面)
            T_cog: 认知温度
            enable_brain: 启用图脑动力学
            enable_spectral: 启用谱分析
        """
        self.max_dimension = max_dimension
        
        # 核心组件
        self.gg = GeneralizedGraph(max_dimension=max_dimension)
        self.hodge: Optional[HodgeLaplacian] = None
        self.brain: Optional[GraphBrainEngine] = None
        
        self.enable_brain = enable_brain
        self.enable_spectral = enable_spectral
        
        # 索引映射
        self.entity_to_cell_id: Dict[str, str] = {}  # 实体 -> 0-细胞ID
        self.relation_to_cell_id: Dict[str, str] = {}  # 关系 -> 1-细胞ID
        self.fact_to_cell_id: Dict[str, str] = {}  # 事实 -> 2-细胞ID
        
        # 统计
        self.num_entities = 0
        self.num_relations = 0
        self.num_facts = 0
        
        logger.info(
            f"GraphRAG初始化: max_dim={max_dimension}, "
            f"brain={enable_brain}, spectral={enable_spectral}"
        )
    
    # ==================== 知识添加 ====================
    
    def add_triple(
        self,
        subject: str,
        predicate: str,
        object: str,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        添加知识三元组 (主体, 谓词, 客体)
        
        构建层次:
        - 0-细胞: 实体 (subject, object)
        - 1-细胞: 关系边 (subject -predicate-> object)
        - 2-细胞: 事实 (整个三元组)
        
        Args:
            subject: 主体实体
            predicate: 谓词/关系
            object: 客体实体
            metadata: 附加元数据
        
        Returns:
            是否成功添加
        """
        # 添加实体 (0-细胞)
        subj_id = self._add_entity(subject)
        obj_id = self._add_entity(object)
        
        # 添加关系边 (1-细胞)
        rel_id = self._add_relation(subject, predicate, object, subj_id, obj_id)
        
        # 添加事实 (2-细胞)
        fact_id = self._add_fact(subject, predicate, object, rel_id, metadata)
        
        self.num_facts += 1
        
        logger.debug(f"添加三元组: ({subject}, {predicate}, {object})")
        
        return True
    
    def _add_entity(self, entity: str) -> str:
        """添加实体节点"""
        cell_id = f"entity:{entity}"
        
        if cell_id not in self.entity_to_cell_id.values():
            success = self.gg.add_cell(
                dimension=0,
                cell_id=cell_id,
                attributes={'name': entity, 'type': 'entity'}
            )
            
            if success:
                self.entity_to_cell_id[entity] = cell_id
                self.num_entities += 1
        
        return self.entity_to_cell_id.get(entity, cell_id)
    
    def _add_relation(
        self,
        subject: str,
        predicate: str,
        object: str,
        subj_id: str,
        obj_id: str
    ) -> str:
        """添加关系边"""
        rel_key = f"{subject}|{predicate}|{object}"
        
        if rel_key in self.relation_to_cell_id:
            return self.relation_to_cell_id[rel_key]
        
        cell_id = f"rel:{rel_key}"
        
        success = self.gg.add_cell(
            dimension=1,
            cell_id=cell_id,
            boundary={subj_id, obj_id},
            attributes={
                'subject': subject,
                'predicate': predicate,
                'object': object,
                'type': 'relation'
            }
        )
        
        if success:
            self.relation_to_cell_id[rel_key] = cell_id
            self.num_relations += 1
        
        return cell_id
    
    def _add_fact(
        self,
        subject: str,
        predicate: str,
        object: str,
        rel_id: str,
        metadata: Optional[Dict]
    ) -> str:
        """添加事实 (2-细胞)"""
        fact_key = f"{subject}|{predicate}|{object}"
        
        if fact_key in self.fact_to_cell_id:
            return self.fact_to_cell_id[fact_key]
        
        cell_id = f"fact:{fact_key}"
        
        attributes = {
            'subject': subject,
            'predicate': predicate,
            'object': object,
            'type': 'fact'
        }
        
        if metadata:
            attributes.update(metadata)
        
        success = self.gg.add_cell(
            dimension=2,
            cell_id=cell_id,
            boundary={rel_id},
            attributes=attributes
        )
        
        if success:
            self.fact_to_cell_id[fact_key] = cell_id
        
        return cell_id
    
    def add_triples_batch(self, triples: List[Tuple[str, str, str]]):
        """批量添加三元组"""
        for subj, pred, obj in triples:
            self.add_triple(subj, pred, obj)
        
        logger.info(f"批量添加 {len(triples)} 个三元组")
    
    # ==================== 索引构建 ====================
    
    def build_indices(self):
        """构建索引 (Hodge-Laplacian + 图脑)"""
        logger.info("开始构建索引...")
        
        # 谱分析
        if self.enable_spectral:
            self.hodge = HodgeLaplacian(self.gg)
            
            # 计算Laplacian
            for p in range(self.max_dimension + 1):
                self.hodge.compute_laplacian(p)
            
            # 计算Betti数
            self.hodge.compute_betti_numbers()
            
            logger.info("Hodge-Laplacian计算完成")
            self.hodge.print_topology_report()
        
        # 图脑初始化
        if self.enable_brain:
            self.brain = GraphBrainEngine(self.gg)
            
            # 预演化 (达到初始稳态)
            logger.info("图脑预演化...")
            for _ in range(50):
                self.brain.evolve_step(dt=0.1)
            
            logger.info("图脑初始化完成")
            self.brain.print_state_report()
        
        logger.info("索引构建完成")
    
    # ==================== 查询 ====================
    
    def query(
        self,
        query_text: str,
        mode: str = "hybrid",
        top_k: int = 10,
        activate_strength: float = 1.0
    ) -> List[Dict]:
        """
        查询知识图谱
        
        Args:
            query_text: 查询文本
            mode: 查询模式
                - "spectral": 谱推理
                - "brain": 图脑动力学
                - "hybrid": 混合
            top_k: 返回前k个结果
            activate_strength: 激活强度
        
        Returns:
            查询结果列表 [{
                'entity': str,
                'score': float,
                'path': List[str],
                'explanation': str
            }, ...]
        """
        logger.info(f"查询: '{query_text}' (mode={mode})")
        
        results = []
        
        # 提取查询实体
        query_entities = self._extract_entities_from_text(query_text)
        
        if not query_entities:
            logger.warning("未识别到查询实体")
            return []
        
        # 获取对应的0-细胞索引
        query_indices = []
        for entity in query_entities:
            if entity in self.entity_to_cell_id:
                cell_id = self.entity_to_cell_id[entity]
                idx = self.gg.get_all_cell_ids(0).index(cell_id)
                query_indices.append(idx)
        
        if not query_indices:
            logger.warning("查询实体不在图谱中")
            return []
        
        # 执行查询
        if mode == "spectral" and self.hodge:
            results = self._query_spectral(query_indices, top_k)
        
        elif mode == "brain" and self.brain:
            results = self._query_brain(query_indices, top_k, activate_strength)
        
        elif mode == "hybrid" and self.hodge and self.brain:
            results = self._query_hybrid(query_indices, top_k, activate_strength)
        
        else:
            # 回退到简单邻域查询
            results = self._query_simple(query_entities, top_k)
        
        return results
    
    def _extract_entities_from_text(self, text: str) -> List[str]:
        """从文本提取实体 (简单实现: 匹配已知实体)"""
        entities = []
        
        for entity in self.entity_to_cell_id.keys():
            if entity.lower() in text.lower():
                entities.append(entity)
        
        return entities
    
    def _query_spectral(self, query_indices: List[int], top_k: int) -> List[Dict]:
        """谱推理查询"""
        logger.debug("执行谱推理查询")
        
        # 获取特征向量
        eigenvalues, eigenvectors = self.hodge.compute_spectrum(0, k=min(50, len(query_indices) * 5))
        
        # 构建查询向量
        n = self.gg.get_statistics()['num_cells_by_dim'][0]
        query_vec = np.zeros(n)
        query_vec[query_indices] = 1.0
        
        # 谱投影
        spectral_scores = eigenvectors.T @ query_vec
        
        # 重建得分
        scores = eigenvectors @ spectral_scores
        scores = np.abs(scores)
        
        # 排序
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # 构建结果
        results = []
        cell_ids = self.gg.get_all_cell_ids(0)
        
        for idx in top_indices:
            cell_id = cell_ids[idx]
            cell = self.gg.get_cell(0, cell_id)
            
            results.append({
                'entity': cell.attributes.get('name', cell_id),
                'score': float(scores[idx]),
                'path': [],
                'explanation': '谱推理相关'
            })
        
        return results
    
    def _query_brain(
        self,
        query_indices: List[int],
        top_k: int,
        strength: float
    ) -> List[Dict]:
        """图脑动力学查询"""
        logger.debug("执行图脑查询")
        
        # 激活查询节点
        self.brain.activate_cells(0, query_indices, strength)
        
        # 演化传播
        for _ in range(20):
            self.brain.evolve_step(dt=0.1)
        
        # 获取激活最高的节点
        top_cells = self.brain.get_activated_cells(0, top_k=top_k)
        
        results = []
        for cell_id, potential in top_cells:
            cell = self.gg.get_cell(0, cell_id)
            
            results.append({
                'entity': cell.attributes.get('name', cell_id),
                'score': potential,
                'path': [],
                'explanation': '图脑激活传播'
            })
        
        return results
    
    def _query_hybrid(
        self,
        query_indices: List[int],
        top_k: int,
        strength: float
    ) -> List[Dict]:
        """混合查询 (谱+图脑)"""
        logger.debug("执行混合查询")
        
        # 谱查询
        spectral_results = self._query_spectral(query_indices, top_k * 2)
        
        # 图脑查询
        brain_results = self._query_brain(query_indices, top_k * 2, strength)
        
        # 融合结果 (加权组合)
        entity_scores = {}
        
        for res in spectral_results:
            entity = res['entity']
            entity_scores[entity] = entity_scores.get(entity, 0) + 0.5 * res['score']
        
        for res in brain_results:
            entity = res['entity']
            entity_scores[entity] = entity_scores.get(entity, 0) + 0.5 * res['score']
        
        # 排序
        sorted_entities = sorted(
            entity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        results = [
            {
                'entity': entity,
                'score': score,
                'path': [],
                'explanation': '谱推理+图脑动力学'
            }
            for entity, score in sorted_entities
        ]
        
        return results
    
    def _query_simple(self, query_entities: List[str], top_k: int) -> List[Dict]:
        """简单邻域查询"""
        logger.debug("执行简单邻域查询")
        
        results = []
        
        for entity in query_entities:
            if entity not in self.entity_to_cell_id:
                continue
            
            cell_id = self.entity_to_cell_id[entity]
            
            # 获取邻居
            neighbors = self.gg.get_neighbors(0, cell_id)
            
            for nb_id in list(neighbors)[:top_k]:
                nb_cell = self.gg.get_cell(0, nb_id)
                
                results.append({
                    'entity': nb_cell.attributes.get('name', nb_id),
                    'score': 1.0,
                    'path': [entity],
                    'explanation': '直接邻居'
                })
        
        return results[:top_k]
    
    # ==================== 统计与可视化 ====================
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            'num_entities': self.num_entities,
            'num_relations': self.num_relations,
            'num_facts': self.num_facts,
            'graph_stats': self.gg.get_statistics()
        }
        
        if self.hodge:
            stats['topology'] = self.hodge.get_topology_summary()
        
        if self.brain:
            stats['brain'] = self.brain.get_evolution_summary()
        
        return stats
    
    def print_summary(self):
        """打印摘要"""
        print("=" * 70)
        print("GraphRAG 摘要")
        print("=" * 70)
        
        print(f"\n知识图谱规模:")
        print(f"  实体数: {self.num_entities}")
        print(f"  关系数: {self.num_relations}")
        print(f"  事实数: {self.num_facts}")
        
        print("\n")
        print(self.gg.summary())
        
        if self.hodge:
            print("\n")
            self.hodge.print_topology_report()
        
        if self.brain:
            print("\n")
            self.brain.print_state_report()
        
        print("=" * 70)
    
    # ==================== 持久化 ====================
    
    def save(self, save_dir: str):
        """保存到目录"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存泛图
        self.gg.save_to_file(str(save_path / "generalized_graph.pkl"))
        
        # 保存索引映射
        with open(save_path / "indices.json", 'w', encoding='utf-8') as f:
            json.dump({
                'entity_to_cell_id': self.entity_to_cell_id,
                'relation_to_cell_id': self.relation_to_cell_id,
                'fact_to_cell_id': self.fact_to_cell_id,
                'num_entities': self.num_entities,
                'num_relations': self.num_relations,
                'num_facts': self.num_facts
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"GraphRAG已保存到: {save_dir}")
    
    @classmethod
    def load(cls, save_dir: str) -> 'GraphRAGManager':
        """从目录加载"""
        save_path = Path(save_dir)
        
        # 加载泛图
        gg = GeneralizedGraph.load_from_file(str(save_path / "generalized_graph.pkl"))
        
        # 创建管理器
        manager = cls(max_dimension=gg.max_dimension)
        manager.gg = gg
        
        # 加载索引
        with open(save_path / "indices.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            manager.entity_to_cell_id = data['entity_to_cell_id']
            manager.relation_to_cell_id = data['relation_to_cell_id']
            manager.fact_to_cell_id = data['fact_to_cell_id']
            manager.num_entities = data['num_entities']
            manager.num_relations = data['num_relations']
            manager.num_facts = data['num_facts']
        
        logger.info(f"GraphRAG已加载: {save_dir}")
        
        return manager


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    print("=== 测试GraphRAG系统 ===\n")
    
    # 创建管理器
    rag = GraphRAGManager(max_dimension=2, enable_brain=True, enable_spectral=True)
    
    # 添加知识
    triples = [
        ("爱因斯坦", "提出", "相对论"),
        ("相对论", "属于", "物理学"),
        ("相对论", "描述", "时空"),
        ("爱因斯坦", "获得", "诺贝尔奖"),
        ("诺贝尔奖", "属于", "物理学"),
        ("量子力学", "属于", "物理学"),
        ("量子力学", "研究", "微观世界")
    ]
    
    rag.add_triples_batch(triples)
    
    # 构建索引
    rag.build_indices()
    
    # 查询
    print("\n" + "=" * 70)
    results = rag.query("爱因斯坦 物理学", mode="hybrid", top_k=5)
    
    print("\n查询结果:")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['entity']} (score={res['score']:.4f}) - {res['explanation']}")
    
    # 摘要
    print("\n")
    rag.print_summary()
