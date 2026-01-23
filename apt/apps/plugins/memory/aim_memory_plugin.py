#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AIM-Memory: Anchored Inertial Mirror Memory
惯性锚定镜像记忆系统

核心机制：
1. 惯性路由 (Inertial Routing, IM)：快速定位到相关记忆簇
2. 时间镜像 (Temporal Mirror, TM)：用权重衰减表达时序
3. 锚点纠错 (Anchored Correction, A)：防止记混和幻觉
4. 按需证据回灌 (Evidence Refill, E)：严格模式下回灌原文

作者: 430 + claude
日期: 2026-01-21
"""

import numpy as np
import json
import re
import hashlib
from typing import Optional, Dict, Any, List, Tuple, Literal
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ==================== 配置 ====================

@dataclass
class AIMConfig:
    """AIM-Memory 配置"""

    # 热缓存参数
    hot_window_size: int = 256  # 热滑窗大小 W

    # 惯性路由参数
    local_cluster_k: int = 32  # 局部召回簇大小 K
    inertia_strength: float = 0.5  # 惯性强度 lambda
    inertia_momentum: float = 0.85  # 惯性平滑 mu

    # 时间镜像参数
    weight_decay_gamma: float = 0.8  # 权重衰减系数 gamma
    fresh_weight: float = 1.0  # 新节点初始权重

    # 写入门控参数
    write_threshold: float = 0.6  # 写入阈值 tau_write
    relevance_weight: float = 0.4  # 相关性权重 alpha
    surprisal_weight: float = 0.3  # 新奇度权重 beta
    conflict_weight: float = 0.3  # 冲突权重 gamma2

    # 锚点纠错参数
    anchor_threshold: float = 0.1  # 锚点阈值 tau_anchor（降低以便召回更多节点）
    anchor_boost: float = 1.5  # 锚点加成 eta
    conflict_penalty: float = 0.3  # 冲突惩罚 zeta

    # 证据回灌参数
    refill_mode: Literal['auto', 'always', 'never'] = 'auto'
    max_evidence_length: int = 500  # 最大证据长度

    # 节点参数
    max_nodes: int = 10000  # 最大节点数
    embedding_dim: int = 768  # 嵌入维度
    top_n_results: int = 3  # 返回结果数


# ==================== 数据结构 ====================

@dataclass
class MemoryNode:
    """记忆节点（镜像节点）"""

    id: str  # 节点ID
    proto: np.ndarray  # 原型向量（embedding）
    summary: str  # 一句话主干
    fields: Dict[str, Any]  # 必留字段（数字/专名/定义/符号）
    links: List[str] = field(default_factory=list)  # 关联边
    w: float = 1.0  # 近因权重/新鲜度
    evidence_ptr: Optional[str] = None  # 证据指针（原文哈希或位置）
    evidence_text: Optional[str] = None  # 证据原文（可选存储）
    timestamp: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'proto': self.proto.tolist() if isinstance(self.proto, np.ndarray) else self.proto,
            'summary': self.summary,
            'fields': self.fields,
            'links': self.links,
            'w': self.w,
            'evidence_ptr': self.evidence_ptr,
            'evidence_text': self.evidence_text,
            'timestamp': self.timestamp.isoformat(),
            'access_count': self.access_count
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryNode':
        proto = np.array(data['proto']) if isinstance(data['proto'], list) else data['proto']
        return cls(
            id=data['id'],
            proto=proto,
            summary=data['summary'],
            fields=data['fields'],
            links=data.get('links', []),
            w=data.get('w', 1.0),
            evidence_ptr=data.get('evidence_ptr'),
            evidence_text=data.get('evidence_text'),
            timestamp=datetime.fromisoformat(data['timestamp']),
            access_count=data.get('access_count', 0)
        )


class HotKV:
    """热上下文缓存（滑窗）"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.anchor_sinks = []  # 锚点沉降（重要片段）

    def append(self, text: str):
        """添加到热缓存"""
        self.buffer.append({
            'text': text,
            'timestamp': datetime.now()
        })

    def get_recent(self, n: Optional[int] = None) -> List[str]:
        """获取最近n条（默认全部）"""
        if n is None:
            return [item['text'] for item in self.buffer]
        return [item['text'] for item in list(self.buffer)[-n:]]

    def add_anchor_sink(self, text: str):
        """添加锚点沉降"""
        self.anchor_sinks.append({
            'text': text,
            'timestamp': datetime.now()
        })

    def clear(self):
        """清空缓存"""
        self.buffer.clear()


class MemoryMap:
    """记忆地图索引（简化版 ANN）"""

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = {}  # node_id -> proto vector
        self.nodes = {}  # node_id -> MemoryNode

    def insert(self, node_id: str, proto: np.ndarray, node: MemoryNode):
        """插入节点"""
        self.index[node_id] = proto
        self.nodes[node_id] = node

    def top_k_cluster(self, query_vec: np.ndarray, k: int) -> List[str]:
        """局部Top-K召回（惯性路由）"""
        if len(self.index) == 0:
            return []

        # 计算余弦相似度
        scores = {}
        query_norm = np.linalg.norm(query_vec)

        for node_id, proto in self.index.items():
            proto_norm = np.linalg.norm(proto)
            if query_norm > 0 and proto_norm > 0:
                similarity = np.dot(query_vec, proto) / (query_norm * proto_norm)
                scores[node_id] = similarity

        # Top-K
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [node_id for node_id, _ in sorted_ids[:k]]

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """获取节点"""
        return self.nodes.get(node_id)

    def get_nodes(self, node_ids: List[str]) -> List[MemoryNode]:
        """批量获取节点"""
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    def remove(self, node_id: str):
        """删除节点"""
        if node_id in self.index:
            del self.index[node_id]
        if node_id in self.nodes:
            del self.nodes[node_id]

    def size(self) -> int:
        """节点数量"""
        return len(self.index)


# ==================== 辅助函数 ====================

class FieldExtractor:
    """字段提取器（数字/专名/定义/符号）"""

    @staticmethod
    def extract_fields(text: str) -> Dict[str, Any]:
        """提取必留字段"""
        fields = {
            'numbers': [],
            'names': [],
            'definitions': [],
            'symbols': []
        }

        # 提取数字（包括小数、百分比）
        numbers = re.findall(r'\d+\.?\d*%?', text)
        fields['numbers'] = numbers

        # 提取专名（大写开头的词，简化版）
        names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        fields['names'] = names

        # 提取定义句（包含"是"、"指"、"定义为"等）
        if any(keyword in text for keyword in ['是', '指', '定义为', '即', 'is', 'means', 'refers to']):
            fields['definitions'].append(text[:100])  # 存储前100字符

        # 提取符号（数学/物理符号）
        symbols = re.findall(r'[α-ωΑ-Ω]|[∀∃∈∉⊂⊃∩∪]|[φθλμτ]', text)
        fields['symbols'] = symbols

        return fields

    @staticmethod
    def fields_to_text(fields: Dict[str, Any]) -> str:
        """字段转文本（用于嵌入）"""
        parts = []
        for key, values in fields.items():
            if values:
                parts.append(f"{key}: {', '.join(map(str, values))}")
        return ' '.join(parts)


class TextEmbedder:
    """文本嵌入器（简化版，可替换为真实模型）"""

    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim

    def embed(self, text: str) -> np.ndarray:
        """嵌入文本（简化版：TF-IDF + 哈希）"""
        # 简化实现：使用哈希 + 归一化
        # 生产环境应替换为真实的 sentence-transformers 或其他模型
        hash_val = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**32))
        vec = np.random.randn(self.embedding_dim)
        vec = vec / np.linalg.norm(vec)  # 归一化
        return vec.astype(np.float32)


class Summarizer:
    """摘要器"""

    @staticmethod
    def summarize(text: str, max_length: int = 100) -> str:
        """生成一句话主干"""
        # 简化实现：取第一句或截断
        sentences = re.split(r'[。！？.!?]', text)
        first_sentence = sentences[0].strip() if sentences else text

        if len(first_sentence) > max_length:
            return first_sentence[:max_length] + '...'
        return first_sentence


class RelevanceScorer:
    """相关性评分器"""

    def __init__(self, embedder: TextEmbedder):
        self.embedder = embedder

    def relevance(self, query: str, text: str) -> float:
        """计算相关性（0-1）"""
        q_vec = self.embedder.embed(query)
        t_vec = self.embedder.embed(text)
        similarity = np.dot(q_vec, t_vec)
        return max(0.0, min(1.0, (similarity + 1) / 2))  # 归一化到 [0, 1]

    def surprisal(self, text: str, context: List[str]) -> float:
        """计算新奇度（0-1）"""
        if not context:
            return 1.0

        t_vec = self.embedder.embed(text)
        context_vec = self.embedder.embed(' '.join(context[-5:]))  # 取最近5条
        similarity = np.dot(t_vec, context_vec)
        return 1.0 - max(0.0, min(1.0, (similarity + 1) / 2))

    def conflict_score(self, fields: Dict[str, Any], recent_nodes: List[MemoryNode]) -> float:
        """计算冲突分数（0-1）"""
        if not recent_nodes:
            return 0.0

        conflicts = 0
        total_checks = 0

        for node in recent_nodes[-10:]:  # 检查最近10个节点
            for key in ['numbers', 'names', 'definitions']:
                if key in fields and key in node.fields:
                    set1 = set(map(str, fields[key]))
                    set2 = set(map(str, node.fields[key]))
                    overlap = len(set1 & set2)
                    total = len(set1 | set2)
                    if total > 0:
                        total_checks += 1
                        if overlap < total * 0.5:  # 少于50%重叠视为冲突
                            conflicts += 1

        return conflicts / max(1, total_checks)


class AnchorChecker:
    """锚点校验器"""

    @staticmethod
    def anchor_check(query_fields: Dict[str, Any], node_fields: Dict[str, Any]) -> float:
        """锚点匹配分数（0-1）"""
        total_score = 0.0
        total_weight = 0.0

        # 权重：数字 > 专名 > 符号 > 定义
        weights = {
            'numbers': 1.0,
            'names': 0.8,
            'symbols': 0.6,
            'definitions': 0.4
        }

        for key, weight in weights.items():
            if key in query_fields and key in node_fields:
                q_set = set(map(str, query_fields[key]))
                n_set = set(map(str, node_fields[key]))

                if len(q_set) > 0:
                    overlap = len(q_set & n_set)
                    score = overlap / len(q_set)
                    total_score += score * weight
                    total_weight += weight

        return total_score / max(0.001, total_weight)


# ==================== AIM-Memory 主类 ====================

class AIMMemory:
    """
    AIM-Memory: Anchored Inertial Mirror Memory
    惯性锚定镜像记忆系统
    """

    def __init__(self, config: Optional[AIMConfig] = None):
        self.config = config or AIMConfig()

        # 核心组件
        self.hot_kv = HotKV(capacity=self.config.hot_window_size)
        self.node_bank = MemoryMap(embedding_dim=self.config.embedding_dim)
        self.v_inertia = np.zeros(self.config.embedding_dim, dtype=np.float32)

        # 辅助工具
        self.embedder = TextEmbedder(embedding_dim=self.config.embedding_dim)
        self.field_extractor = FieldExtractor()
        self.summarizer = Summarizer()
        self.relevance_scorer = RelevanceScorer(self.embedder)
        self.anchor_checker = AnchorChecker()

        logger.info(f"[AIMMemory] 初始化完成: W={self.config.hot_window_size}, K={self.config.local_cluster_k}")

    # ========== 写入路径 ==========

    def write_memory(self, text: str, context: Optional[List[str]] = None) -> bool:
        """
        写入记忆（WriteMemory算法）

        Args:
            text: 新片段
            context: 当前上下文（来自HotKV）

        Returns:
            是否写入长期记忆
        """
        # Step W0: 更新热缓存
        self.hot_kv.append(text)

        # Step W1: 计算写入门控
        if context is None:
            context = self.hot_kv.get_recent(10)

        context_text = ' '.join(context[-5:]) if context else ''

        r = self.relevance_scorer.relevance(context_text, text)
        s = self.relevance_scorer.surprisal(text, context)

        q_fields = self.field_extractor.extract_fields(text)
        recent_nodes = list(self.node_bank.nodes.values())[-20:]
        c = self.relevance_scorer.conflict_score(q_fields, recent_nodes)

        # 门控函数：sigmoid(alpha*r + beta*s + gamma2*c)
        gate_input = (
            self.config.relevance_weight * r +
            self.config.surprisal_weight * s +
            self.config.conflict_weight * c
        )
        gate = 1.0 / (1.0 + np.exp(-gate_input))

        logger.debug(f"[WriteGate] r={r:.3f}, s={s:.3f}, c={c:.3f}, gate={gate:.3f}")

        # Step W2: 若 gate < tau_write，不写入
        if gate < self.config.write_threshold:
            logger.debug(f"[WriteMemory] 未通过门控，不写入长期记忆")
            return False

        # Step W3: 形成镜像节点
        node_id = hashlib.sha256(text.encode()).hexdigest()[:16]

        summary = self.summarizer.summarize(text)
        fields = q_fields
        fields_text = self.field_extractor.fields_to_text(fields)
        proto = self.embedder.embed(summary + ' ' + fields_text)

        evidence_ptr = hashlib.sha256(text.encode()).hexdigest()

        node = MemoryNode(
            id=node_id,
            proto=proto,
            summary=summary,
            fields=fields,
            w=self.config.fresh_weight,
            evidence_ptr=evidence_ptr,
            evidence_text=text if len(text) <= self.config.max_evidence_length else text[:self.config.max_evidence_length]
        )

        # Step W4: 衰减旧节点权重（时间镜像）
        self._decay_weights()

        # Step W5: 入库 + 建边 + 更新地图
        self.node_bank.insert(node.id, node.proto, node)
        self._update_links(node)

        # 容量控制
        if self.node_bank.size() > self.config.max_nodes:
            self._prune_old_nodes()

        logger.info(f"[WriteMemory] 写入长期记忆: {summary[:50]}...")
        return True

    def _decay_weights(self):
        """衰减所有节点的权重（时间镜像）"""
        for node in self.node_bank.nodes.values():
            node.w *= self.config.weight_decay_gamma

    def _update_links(self, node: MemoryNode):
        """更新节点的关联边"""
        # 简化实现：基于字段匹配找邻居
        for other_id, other_node in self.node_bank.nodes.items():
            if other_id == node.id:
                continue

            # 检查字段重叠
            overlap = 0
            for key in ['numbers', 'names', 'symbols']:
                if key in node.fields and key in other_node.fields:
                    set1 = set(map(str, node.fields[key]))
                    set2 = set(map(str, other_node.fields[key]))
                    overlap += len(set1 & set2)

            if overlap > 0:
                if other_id not in node.links:
                    node.links.append(other_id)
                if node.id not in other_node.links:
                    other_node.links.append(node.id)

    def _prune_old_nodes(self):
        """修剪老节点（保留权重最高的）"""
        nodes_by_weight = sorted(
            self.node_bank.nodes.items(),
            key=lambda x: x[1].w,
            reverse=True
        )

        # 保留前 max_nodes 个
        to_remove = [nid for nid, _ in nodes_by_weight[self.config.max_nodes:]]
        for node_id in to_remove:
            self.node_bank.remove(node_id)

        logger.info(f"[Prune] 移除 {len(to_remove)} 个旧节点，当前节点数: {self.node_bank.size()}")

    # ========== 读取路径 ==========

    def route_memory(
        self,
        query: str,
        mode: Literal['fast', 'strict'] = 'fast'
    ) -> Tuple[List[MemoryNode], str]:
        """
        路由记忆（RouteMemory算法）

        Args:
            query: 查询文本
            mode: 'fast' 或 'strict'

        Returns:
            (selected_nodes, refill_text)
        """
        # Step R0: 解析查询
        q_fields = self.field_extractor.extract_fields(query)
        q_fields_text = self.field_extractor.fields_to_text(q_fields)
        q_vec = self.embedder.embed(query + ' ' + q_fields_text)

        # Step R1: 形成惯性方向
        d = q_vec + self.config.inertia_strength * self.v_inertia
        d = d / (np.linalg.norm(d) + 1e-8)  # 归一化

        # Step R2: 局部召回（只取小簇）
        candidate_ids = self.node_bank.top_k_cluster(d, self.config.local_cluster_k)
        candidates = self.node_bank.get_nodes(candidate_ids)

        if not candidates:
            logger.debug(f"[RouteMemory] 未找到候选节点")
            return [], ""

        # Step R3: 锚点校正
        for node in candidates:
            anchor_score = self.anchor_checker.anchor_check(q_fields, node.fields)
            conflict = self.relevance_scorer.conflict_score(node.fields, [node])

            # 基础分数：即使没有锚点匹配也给一个基础分（基于权重和嵌入相似度）
            base_score = 0.3 * node.w  # 基础分
            anchor_bonus = anchor_score * self.config.anchor_boost * node.w  # 锚点加成

            node_score = (
                base_score + anchor_bonus -
                self.config.conflict_penalty * conflict
            )

            # 临时存储分数
            node.temp_score = node_score

        # 排序并筛选
        candidates.sort(key=lambda n: n.temp_score, reverse=True)
        selected = [
            n for n in candidates[:self.config.top_n_results]
            if n.temp_score >= self.config.anchor_threshold
        ]

        # 更新访问计数
        for node in selected:
            node.access_count += 1

        # Step R4: 按需证据回灌
        refill_text = ""
        need_refill = (mode == 'strict')

        # 检测严格模式触发词
        strict_keywords = ['来源', '原文', '精确', '证明', '定义', '引用', 'source', 'exact', 'proof']
        if any(kw in query.lower() for kw in strict_keywords):
            need_refill = True

        # 检测冲突
        if selected and any(n.temp_score < 0.7 for n in selected):
            need_refill = True

        if need_refill and self.config.refill_mode != 'never':
            refill_parts = []
            for node in selected:
                if node.evidence_text:
                    refill_parts.append(f"[证据 {node.id[:8]}] {node.evidence_text}")
            refill_text = '\n'.join(refill_parts)

        # Step R5: 更新惯性
        if selected:
            v_sel = np.mean([n.proto for n in selected], axis=0)
            self.v_inertia = (
                self.config.inertia_momentum * self.v_inertia +
                (1 - self.config.inertia_momentum) * v_sel
            )
        else:
            self.v_inertia *= self.config.inertia_momentum

        logger.info(f"[RouteMemory] 召回 {len(selected)} 个节点, mode={mode}, refill={'是' if refill_text else '否'}")

        return selected, refill_text

    # ========== 生成路径 ==========

    def answer(self, query: str, auto_mode: bool = True) -> Dict[str, Any]:
        """
        回答查询（Answer算法）

        Args:
            query: 查询文本
            auto_mode: 是否自动选择模式

        Returns:
            包含answer, selected_nodes, refill_text等的字典
        """
        # Step A0: mode选择
        strict_keywords = ['定义', '定理', '证明', '数值', '引用', 'definition', 'theorem', 'proof']
        mode = 'strict' if any(kw in query for kw in strict_keywords) else 'fast'

        if not auto_mode:
            mode = 'fast'

        # Step A1: 路由记忆
        selected, refill = self.route_memory(query, mode)

        # Step A2: 构建上下文
        prompt_parts = []

        # 热滑窗
        hot_context = self.hot_kv.get_recent(5)
        if hot_context:
            prompt_parts.append("[热上下文]")
            prompt_parts.extend(hot_context)

        # 记忆摘要
        if selected:
            prompt_parts.append("\n[记忆摘要]")
            for node in selected:
                prompt_parts.append(f"• {node.summary}")

        # 记忆字段
        if selected:
            prompt_parts.append("\n[记忆字段]")
            for node in selected:
                fields_str = self.field_extractor.fields_to_text(node.fields)
                if fields_str:
                    prompt_parts.append(f"• {fields_str}")

        # 证据回灌
        if refill:
            prompt_parts.append("\n[证据原文]")
            prompt_parts.append(refill)

        context = '\n'.join(prompt_parts)

        # Step A3: 生成（这里返回上下文，实际应调用LLM）
        result = {
            'query': query,
            'mode': mode,
            'context': context,
            'selected_nodes': selected,
            'refill_text': refill,
            'hot_context': hot_context,
            'num_nodes_recalled': len(selected),
            'inertia_norm': float(np.linalg.norm(self.v_inertia))
        }

        return result

    # ========== 工具方法 ==========

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'hot_kv_size': len(self.hot_kv.buffer),
            'node_bank_size': self.node_bank.size(),
            'inertia_norm': float(np.linalg.norm(self.v_inertia)),
            'total_access': sum(n.access_count for n in self.node_bank.nodes.values()),
            'avg_weight': np.mean([n.w for n in self.node_bank.nodes.values()]) if self.node_bank.size() > 0 else 0.0
        }

    def save_to_file(self, filepath: str):
        """保存到文件"""
        data = {
            'config': {
                'hot_window_size': self.config.hot_window_size,
                'local_cluster_k': self.config.local_cluster_k,
                'embedding_dim': self.config.embedding_dim
            },
            'nodes': [node.to_dict() for node in self.node_bank.nodes.values()],
            'v_inertia': self.v_inertia.tolist(),
            'hot_kv': [item['text'] for item in self.hot_kv.buffer]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"[AIMMemory] 保存到文件: {filepath}")

    def load_from_file(self, filepath: str):
        """从文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 恢复节点
        for node_data in data.get('nodes', []):
            node = MemoryNode.from_dict(node_data)
            self.node_bank.insert(node.id, node.proto, node)

        # 恢复惯性向量
        if 'v_inertia' in data:
            self.v_inertia = np.array(data['v_inertia'], dtype=np.float32)

        # 恢复热缓存
        for text in data.get('hot_kv', []):
            self.hot_kv.append(text)

        logger.info(f"[AIMMemory] 从文件加载: {filepath}, 节点数={self.node_bank.size()}")


# ==================== 工厂函数 ====================

def create_aim_memory(config: Optional[AIMConfig] = None) -> AIMMemory:
    """创建 AIM-Memory 实例"""
    return AIMMemory(config)
