#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轻量级知识图谱模块

提供基于三元组的简单知识图谱实现：
- 实体-关系-实体存储
- 快速检索和查询
- 与RAG集成
- 多跳推理支持
"""

import os
import json
import pickle
from typing import List, Tuple, Dict, Set, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import torch
import torch.nn as nn

from apt.core.infrastructure.logging import get_progress_logger

logger = get_progress_logger()


@dataclass
class Triple:
    """知识图谱三元组 (头实体, 关系, 尾实体)"""
    head: str
    relation: str
    tail: str
    confidence: float = 1.0  # 置信度
    metadata: Optional[Dict[str, Any]] = None

    def __hash__(self):
        return hash((self.head, self.relation, self.tail))

    def __eq__(self, other):
        if not isinstance(other, Triple):
            return False
        return (self.head == other.head and
                self.relation == other.relation and
                self.tail == other.tail)

    def to_dict(self):
        """转换为字典"""
        return asdict(self)

    @staticmethod
    def from_dict(data):
        """从字典创建"""
        return Triple(**data)


class KnowledgeGraph:
    """
    轻量级知识图谱

    使用邻接表存储三元组，支持快速查询：
    - 给定头实体，查找所有关系和尾实体
    - 给定关系，查找所有头尾实体对
    - 给定尾实体，反向查找所有头实体和关系
    """

    def __init__(self):
        # 三元组集合
        self.triples: Set[Triple] = set()

        # 索引结构（用于快速查询）
        self.head_index: Dict[str, List[Triple]] = defaultdict(list)  # head -> triples
        self.relation_index: Dict[str, List[Triple]] = defaultdict(list)  # relation -> triples
        self.tail_index: Dict[str, List[Triple]] = defaultdict(list)  # tail -> triples

        # 实体和关系集合
        self.entities: Set[str] = set()
        self.relations: Set[str] = set()

    def add_triple(self, head: str, relation: str, tail: str,
                   confidence: float = 1.0, metadata: Optional[Dict] = None):
        """
        添加三元组到知识图谱

        Args:
            head: 头实体
            relation: 关系
            tail: 尾实体
            confidence: 置信度 (0-1)
            metadata: 元数据
        """
        triple = Triple(head, relation, tail, confidence, metadata)

        if triple in self.triples:
            return  # 已存在

        self.triples.add(triple)

        # 更新索引
        self.head_index[head].append(triple)
        self.relation_index[relation].append(triple)
        self.tail_index[tail].append(triple)

        # 更新实体和关系集合
        self.entities.add(head)
        self.entities.add(tail)
        self.relations.add(relation)

    def add_triples_batch(self, triples: List[Tuple[str, str, str]]):
        """批量添加三元组"""
        for triple in triples:
            if len(triple) == 3:
                head, relation, tail = triple
                self.add_triple(head, relation, tail)
            elif len(triple) == 4:
                head, relation, tail, confidence = triple
                self.add_triple(head, relation, tail, confidence)

    def query_by_head(self, head: str) -> List[Triple]:
        """查询给定头实体的所有三元组"""
        return self.head_index.get(head, [])

    def query_by_relation(self, relation: str) -> List[Triple]:
        """查询给定关系的所有三元组"""
        return self.relation_index.get(relation, [])

    def query_by_tail(self, tail: str) -> List[Triple]:
        """查询给定尾实体的所有三元组（反向查询）"""
        return self.tail_index.get(tail, [])

    def query_by_head_relation(self, head: str, relation: str) -> List[str]:
        """查询给定头实体和关系的所有尾实体"""
        triples = self.head_index.get(head, [])
        return [t.tail for t in triples if t.relation == relation]

    def query_by_relation_tail(self, relation: str, tail: str) -> List[str]:
        """查询给定关系和尾实体的所有头实体"""
        triples = self.tail_index.get(tail, [])
        return [t.head for t in triples if t.relation == relation]

    def get_neighbors(self, entity: str, relation: Optional[str] = None) -> List[str]:
        """
        获取实体的邻居

        Args:
            entity: 实体名称
            relation: 可选，指定关系类型

        Returns:
            邻居实体列表
        """
        neighbors = []

        # 作为头实体的邻居（尾实体）
        for triple in self.head_index.get(entity, []):
            if relation is None or triple.relation == relation:
                neighbors.append(triple.tail)

        # 作为尾实体的邻居（头实体）
        for triple in self.tail_index.get(entity, []):
            if relation is None or triple.relation == relation:
                neighbors.append(triple.head)

        return neighbors

    def multi_hop_query(self, start_entity: str, relations: List[str],
                       max_results: int = 10) -> List[List[str]]:
        """
        多跳查询

        Args:
            start_entity: 起始实体
            relations: 关系路径，如 ['is_a', 'has_property']
            max_results: 最大结果数

        Returns:
            路径列表，每个路径是实体序列
        """
        if not relations:
            return [[start_entity]]

        paths = [[start_entity]]

        for relation in relations:
            new_paths = []
            for path in paths:
                current_entity = path[-1]
                next_entities = self.query_by_head_relation(current_entity, relation)

                for next_entity in next_entities[:max_results]:
                    new_paths.append(path + [next_entity])

                if len(new_paths) >= max_results:
                    break

            paths = new_paths
            if not paths:
                break

        return paths[:max_results]

    def find_paths(self, start: str, end: str, max_hops: int = 3) -> List[List[Triple]]:
        """
        查找两个实体之间的路径

        Args:
            start: 起始实体
            end: 目标实体
            max_hops: 最大跳数

        Returns:
            路径列表，每个路径是三元组序列
        """
        paths = []

        def dfs(current: str, target: str, current_path: List[Triple], visited: Set[str], depth: int):
            if depth > max_hops:
                return

            if current == target and current_path:
                paths.append(current_path.copy())
                return

            visited.add(current)

            for triple in self.head_index.get(current, []):
                if triple.tail not in visited:
                    current_path.append(triple)
                    dfs(triple.tail, target, current_path, visited, depth + 1)
                    current_path.pop()

            visited.remove(current)

        dfs(start, end, [], set(), 0)
        return paths

    def get_subgraph(self, entities: List[str], max_hops: int = 1) -> 'KnowledgeGraph':
        """
        提取子图

        Args:
            entities: 实体列表
            max_hops: 最大跳数

        Returns:
            子图
        """
        subgraph = KnowledgeGraph()
        visited = set(entities)
        to_visit = list(entities)

        for hop in range(max_hops + 1):
            next_to_visit = []

            for entity in to_visit:
                # 添加所有相关三元组
                for triple in self.head_index.get(entity, []):
                    subgraph.add_triple(
                        triple.head, triple.relation, triple.tail,
                        triple.confidence, triple.metadata
                    )
                    if triple.tail not in visited:
                        next_to_visit.append(triple.tail)
                        visited.add(triple.tail)

                for triple in self.tail_index.get(entity, []):
                    subgraph.add_triple(
                        triple.head, triple.relation, triple.tail,
                        triple.confidence, triple.metadata
                    )
                    if triple.head not in visited:
                        next_to_visit.append(triple.head)
                        visited.add(triple.head)

            to_visit = next_to_visit

        return subgraph

    def to_text(self, triples: Optional[List[Triple]] = None, format: str = 'natural') -> str:
        """
        将三元组转换为文本

        Args:
            triples: 三元组列表，None表示所有三元组
            format: 格式 ('natural', 'structured')

        Returns:
            文本表示
        """
        if triples is None:
            triples = list(self.triples)

        if format == 'natural':
            # 自然语言格式：人工智能 是 计算机科学的分支
            texts = [f"{t.head} {t.relation} {t.tail}" for t in triples]
        else:
            # 结构化格式：(人工智能, 是, 计算机科学的分支)
            texts = [f"({t.head}, {t.relation}, {t.tail})" for t in triples]

        return "\n".join(texts)

    def save(self, filepath: str):
        """保存知识图谱到文件"""
        data = {
            'triples': [t.to_dict() for t in self.triples],
            'entities': list(self.entities),
            'relations': list(self.relations)
        }

        if filepath.endswith('.json'):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)

        logger.info(f"[KG] 保存知识图谱到 {filepath}")
        logger.info(f"[KG] 实体数: {len(self.entities)}, 关系数: {len(self.relations)}, 三元组数: {len(self.triples)}")

    @staticmethod
    def load(filepath: str) -> 'KnowledgeGraph':
        """从文件加载知识图谱"""
        if filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

        kg = KnowledgeGraph()
        for triple_data in data['triples']:
            triple = Triple.from_dict(triple_data)
            kg.add_triple(
                triple.head, triple.relation, triple.tail,
                triple.confidence, triple.metadata
            )

        logger.info(f"[KG] 从 {filepath} 加载知识图谱")
        logger.info(f"[KG] 实体数: {len(kg.entities)}, 关系数: {len(kg.relations)}, 三元组数: {len(kg.triples)}")

        return kg

    def stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'num_entities': len(self.entities),
            'num_relations': len(self.relations),
            'num_triples': len(self.triples),
            'avg_degree': len(self.triples) / max(len(self.entities), 1) * 2,
            'relations_list': list(self.relations)
        }

    def __len__(self):
        return len(self.triples)

    def __repr__(self):
        return f"KnowledgeGraph(entities={len(self.entities)}, relations={len(self.relations)}, triples={len(self.triples)})"


def load_triples_from_file(filepath: str, separator: str = '\t') -> List[Tuple[str, str, str]]:
    """
    从文件加载三元组

    文件格式（每行一个三元组）：
    头实体<separator>关系<separator>尾实体

    Example:
        人工智能\t是\t计算机科学的分支
        深度学习\t是\t机器学习的子领域
    """
    triples = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split(separator)
            if len(parts) >= 3:
                head, relation, tail = parts[0].strip(), parts[1].strip(), parts[2].strip()
                triples.append((head, relation, tail))

    logger.info(f"[KG] 从 {filepath} 加载了 {len(triples)} 个三元组")
    return triples


def create_kg_from_file(filepath: str, separator: str = '\t') -> KnowledgeGraph:
    """从文件创建知识图谱"""
    triples = load_triples_from_file(filepath, separator)
    kg = KnowledgeGraph()
    kg.add_triples_batch(triples)
    return kg


# 示例：从自然语言提取简单三元组（基于规则）
def extract_triples_from_text(text: str) -> List[Tuple[str, str, str]]:
    """
    从文本提取简单三元组（基于规则的方法）

    注意：这是一个简化版本，实际应用可能需要更复杂的NLP处理
    """
    triples = []

    # 简单的模式匹配
    # 示例: "A是B" -> (A, 是, B)
    patterns = [
        (r'(.+?)是(.+)', '是'),
        (r'(.+?)有(.+)', '有'),
        (r'(.+?)属于(.+)', '属于'),
    ]

    import re
    for pattern, relation in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if len(match) == 2:
                head, tail = match
                triples.append((head.strip(), relation, tail.strip()))

    return triples
