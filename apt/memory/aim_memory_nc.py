#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AIM-NC: AIM with N-gram Captured retrieval
AIM-Memory × NGram/Trie 收编协议

核心改进：
1. 三路召回：n-gram 命中 + 向量近邻 + 邻接扩展
2. 保持 AIM 锚点纠错主权（n-gram 只是侦察兵，不是裁判）
3. 成本更低、召回更准、防幻觉更强

作者: 430 + claude
日期: 2026-01-21
版本: AIM-NC v1.0
"""

import numpy as np
import json
import re
import hashlib
from typing import Optional, Dict, Any, List, Tuple, Literal, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, Counter, deque
import logging

# 导入基础 AIM 组件
from .aim_memory import (
    AIMConfig, MemoryNode, HotKV, TextEmbedder, Summarizer,
    FieldExtractor, RelevanceScorer, AnchorChecker, logger
)

# ==================== 扩展配置 ====================

@dataclass
class AIMNCConfig(AIMConfig):
    """AIM-NC 扩展配置"""

    # N-gram 召回参数
    ngram_sizes: List[int] = field(default_factory=lambda: [2, 3, 4])  # n-gram 大小
    k_ng: int = 64  # n-gram 召回上限
    k_ng_per_ngram: int = 16  # 每个 n-gram 召回数

    # 向量召回参数
    k_vec: int = 32  # 向量近邻召回上限

    # 邻接扩展参数
    k_link: int = 16  # 邻接扩展召回上限
    k_seed: int = 8  # 用于邻接扩展的种子节点数

    # 候选池参数
    k_final: int = 64  # 最终候选池上限

    # 三路权重（收编关键：n-gram 只是加分项）
    rho_ng: float = 0.3  # n-gram 命中权重
    rho_vec: float = 0.5  # 向量相似度权重
    rho_link: float = 0.2  # 邻接亲和度权重

    # Trie 参数
    use_trie: bool = True  # 是否使用 Trie 加速
    trie_min_freq: int = 2  # Trie 最小频率阈值


# ==================== N-gram 索引 ====================

class NGramIndex:
    """N-gram 倒排索引：快速命中候选节点"""

    def __init__(self, ngram_sizes: List[int] = [2, 3, 4]):
        self.ngram_sizes = ngram_sizes
        # {ngram: {node_id: weight}}
        self.index: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.node_ngrams: Dict[str, Set[str]] = defaultdict(set)  # {node_id: {ngrams}}
        self.ngram_df: Dict[str, int] = Counter()  # document frequency

    def tokenize(self, text: str) -> List[str]:
        """分词（简单空格分词 + 中文字符）"""
        # 保留中文字符和英文单词
        tokens = []
        for word in text.split():
            if re.search(r'[\u4e00-\u9fff]', word):
                # 中文：按字符分
                tokens.extend(list(word))
            else:
                # 英文/数字：保持完整
                tokens.append(word.lower())
        return tokens

    def generate_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """生成 n-gram"""
        if len(tokens) < n:
            return []
        return ['_'.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def compute_ngram_weight(self, ngram: str, node_id: str, tf: int) -> float:
        """计算 n-gram 权重（TF-IDF 风格）"""
        # TF: 词频
        # IDF: 逆文档频率
        df = self.ngram_df.get(ngram, 1)
        idf = np.log(len(self.node_ngrams) + 1) - np.log(df + 1)
        return tf * idf

    def add(self, node_id: str, text: str, summary: str = "", fields_text: str = ""):
        """为节点添加 n-gram 索引"""
        # 合并文本
        full_text = f"{summary} {fields_text} {text}".strip()
        tokens = self.tokenize(full_text)

        # 移除旧索引（如果存在）
        self.remove(node_id)

        # 生成所有 n-gram
        ngram_counts = Counter()
        for n in self.ngram_sizes:
            ngrams = self.generate_ngrams(tokens, n)
            for ng in ngrams:
                ngram_counts[ng] += 1
                self.node_ngrams[node_id].add(ng)

        # 更新倒排索引
        for ng, tf in ngram_counts.items():
            self.ngram_df[ng] += 1
            weight = self.compute_ngram_weight(ng, node_id, tf)
            self.index[ng][node_id] = weight

    def remove(self, node_id: str):
        """移除节点的 n-gram 索引"""
        if node_id not in self.node_ngrams:
            return

        # 更新 document frequency
        for ng in self.node_ngrams[node_id]:
            self.ngram_df[ng] -= 1
            if self.ngram_df[ng] <= 0:
                del self.ngram_df[ng]
            if node_id in self.index[ng]:
                del self.index[ng][node_id]
            if not self.index[ng]:
                del self.index[ng]

        del self.node_ngrams[node_id]

    def lookup(self, query_text: str, top_k: int = 64) -> List[Tuple[str, float]]:
        """查找匹配的节点 ID 和命中分数"""
        tokens = self.tokenize(query_text)
        hit_scores = defaultdict(float)

        # 收集所有 n-gram 命中
        for n in self.ngram_sizes:
            query_ngrams = self.generate_ngrams(tokens, n)
            for ng in query_ngrams:
                if ng in self.index:
                    for node_id, weight in self.index[ng].items():
                        hit_scores[node_id] += weight

        # 排序并返回 top-k
        sorted_hits = sorted(hit_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_hits[:top_k]

    def get_hit_strength(self, query_text: str, node_id: str) -> float:
        """计算查询与节点的命中强度"""
        tokens = self.tokenize(query_text)
        hit_score = 0.0

        for n in self.ngram_sizes:
            query_ngrams = self.generate_ngrams(tokens, n)
            for ng in query_ngrams:
                if ng in self.index and node_id in self.index[ng]:
                    hit_score += self.index[ng][node_id]

        return hit_score

    def to_dict(self) -> Dict:
        """序列化"""
        return {
            'ngram_sizes': self.ngram_sizes,
            'index': {ng: dict(postings) for ng, postings in self.index.items()},
            'node_ngrams': {nid: list(ngs) for nid, ngs in self.node_ngrams.items()},
            'ngram_df': dict(self.ngram_df)
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'NGramIndex':
        """反序列化"""
        obj = cls(ngram_sizes=data['ngram_sizes'])
        obj.index = defaultdict(lambda: defaultdict(float),
                                {ng: defaultdict(float, postings)
                                 for ng, postings in data['index'].items()})
        obj.node_ngrams = defaultdict(set,
                                       {nid: set(ngs)
                                        for nid, ngs in data['node_ngrams'].items()})
        obj.ngram_df = Counter(data['ngram_df'])
        return obj


# ==================== Trie 前缀树 ====================

class TrieNode:
    """Trie 树节点"""
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.node_ids: Set[str] = set()  # 以此为前缀的节点 ID
        self.is_end: bool = False
        self.frequency: int = 0

class TrieLM:
    """Trie 语言模型：用于快速前缀匹配和扩展"""

    def __init__(self, min_freq: int = 2):
        self.root = TrieNode()
        self.min_freq = min_freq
        self.tokenizer = NGramIndex().tokenize  # 复用分词器

    def insert(self, tokens: List[str], node_id: str):
        """插入 token 序列"""
        node = self.root
        for token in tokens:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
            node.node_ids.add(node_id)
            node.frequency += 1
        node.is_end = True

    def search_prefix(self, prefix_tokens: List[str]) -> Set[str]:
        """搜索前缀匹配的所有节点"""
        node = self.root
        for token in prefix_tokens:
            if token not in node.children:
                return set()
            node = node.children[token]
        return node.node_ids if node.frequency >= self.min_freq else set()

    def add_text(self, text: str, node_id: str):
        """添加文本"""
        tokens = self.tokenizer(text)
        self.insert(tokens, node_id)


# ==================== 邻接图 ====================

class LinkGraph:
    """邻接图：基于实体/时间/主题的关联扩展"""

    def __init__(self):
        # {node_id: {neighbor_id: affinity_score}}
        self.graph: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)  # {entity: {node_ids}}
        self.temporal_buckets: Dict[str, Set[str]] = defaultdict(set)  # {time_bucket: {node_ids}}

    def add_node(self, node: MemoryNode, other_nodes: List[MemoryNode]):
        """添加节点并建立邻接边"""
        node_id = node.id

        # 提取实体（从 fields）
        entities = self._extract_entities(node.fields)
        for entity in entities:
            self.entity_index[entity].add(node_id)

        # 时间桶（按小时）
        time_bucket = node.timestamp.strftime("%Y-%m-%d-%H")
        self.temporal_buckets[time_bucket].add(node_id)

        # 建立与其他节点的边
        for other in other_nodes:
            if other.id == node_id:
                continue

            # 计算亲和度
            affinity = self._compute_affinity(node, other)
            if affinity > 0.1:  # 阈值
                self.graph[node_id][other.id] = affinity
                self.graph[other.id][node_id] = affinity

    def _extract_entities(self, fields: Dict[str, Any]) -> Set[str]:
        """从字段提取实体"""
        entities = set()

        # 专名
        if 'names' in fields:
            entities.update(fields['names'])

        # 数字（作为字符串实体）
        if 'numbers' in fields:
            entities.update(str(n) for n in fields['numbers'])

        # 符号
        if 'symbols' in fields:
            entities.update(fields['symbols'])

        return entities

    def _compute_affinity(self, node1: MemoryNode, node2: MemoryNode) -> float:
        """计算两节点的亲和度"""
        score = 0.0

        # 实体重叠
        entities1 = self._extract_entities(node1.fields)
        entities2 = self._extract_entities(node2.fields)
        if entities1 and entities2:
            overlap = len(entities1 & entities2)
            union = len(entities1 | entities2)
            score += 0.5 * (overlap / union) if union > 0 else 0

        # 时间邻近（1小时内）
        time_diff = abs((node1.timestamp - node2.timestamp).total_seconds())
        if time_diff < 3600:  # 1小时
            score += 0.3 * (1 - time_diff / 3600)

        # 向量相似度
        cos_sim = np.dot(node1.proto, node2.proto) / (
            np.linalg.norm(node1.proto) * np.linalg.norm(node2.proto) + 1e-8
        )
        score += 0.2 * max(0, cos_sim)

        return score

    def expand(self, seed_ids: List[str], limit: int = 16) -> List[str]:
        """从种子节点扩展邻居"""
        expanded = set()
        scores = defaultdict(float)

        for seed_id in seed_ids:
            # 直接邻居
            if seed_id in self.graph:
                for neighbor_id, affinity in self.graph[seed_id].items():
                    scores[neighbor_id] += affinity

            # 实体共现邻居（需要先找到种子的实体）
            # 这里简化：只用图邻居

        # 排序并限制数量
        sorted_neighbors = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        expanded = [nid for nid, _ in sorted_neighbors[:limit]]

        return expanded

    def get_link_affinity(self, query_fields: Dict[str, Any], node_links: List[str]) -> float:
        """计算查询字段与节点邻居的亲和度"""
        query_entities = self._extract_entities(query_fields)
        if not query_entities:
            return 0.0

        # 检查邻居节点是否包含查询实体
        affinity = 0.0
        for link_id in node_links:
            if link_id in self.graph:
                # 简化：检查是否有共同实体
                for entity in query_entities:
                    if entity in self.entity_index and link_id in self.entity_index[entity]:
                        affinity += 0.1

        return min(affinity, 1.0)

    def to_dict(self) -> Dict:
        """序列化"""
        return {
            'graph': {nid: dict(neighbors) for nid, neighbors in self.graph.items()},
            'entity_index': {ent: list(nids) for ent, nids in self.entity_index.items()},
            'temporal_buckets': {tb: list(nids) for tb, nids in self.temporal_buckets.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'LinkGraph':
        """反序列化"""
        obj = cls()
        obj.graph = defaultdict(lambda: defaultdict(float),
                                {nid: defaultdict(float, neighbors)
                                 for nid, neighbors in data['graph'].items()})
        obj.entity_index = defaultdict(set,
                                        {ent: set(nids)
                                         for ent, nids in data['entity_index'].items()})
        obj.temporal_buckets = defaultdict(set,
                                            {tb: set(nids)
                                             for tb, nids in data['temporal_buckets'].items()})
        return obj


# ==================== AIM-NC 主类 ====================

class AIMMemoryNC:
    """AIM-Memory with N-gram Captured retrieval"""

    def __init__(self, config: Optional[AIMNCConfig] = None):
        self.config = config or AIMNCConfig()

        # 原生 AIM 组件
        self.hot_kv = HotKV(self.config.hot_window_size)
        self.node_bank: Dict[str, MemoryNode] = {}
        self.evidence_store: Dict[str, str] = {}
        self.v_inertia = np.zeros(self.config.embedding_dim)

        # 收编组件（新增）
        self.ngram_index = NGramIndex(self.config.ngram_sizes)
        self.trie_lm = TrieLM(self.config.trie_min_freq) if self.config.use_trie else None
        self.link_graph = LinkGraph()

        # 辅助组件
        self.embedder = TextEmbedder(self.config.embedding_dim)
        self.summarizer = Summarizer()
        self.field_extractor = FieldExtractor()
        self.relevance_scorer = RelevanceScorer(self.embedder)  # 传入 embedder
        self.anchor_checker = AnchorChecker()  # 静态方法类，无需参数

        # 统计
        self.total_access = 0
        self.write_count = 0
        self.ngram_recall_count = 0
        self.vec_recall_count = 0
        self.link_recall_count = 0

        logger.info(f"[AIMMemoryNC] 初始化完成: W={self.config.hot_window_size}, "
                   f"K_ng={self.config.k_ng}, K_vec={self.config.k_vec}, K_link={self.config.k_link}")

    def write_memory(self, text: str, context: Optional[List[str]] = None) -> bool:
        """
        写入路径（收编版）：WriteMemory_Captured

        Step W1-W4: 与原 AIM 相同
        Step W5: 构建 n-gram 影子索引 (收编动作 A)
        Step W6: 建立邻接边 (收编动作 B)
        """
        # W1: 热滑窗更新
        self.hot_kv.append(text)

        # W2: 写入门控
        context_text = '\n'.join(context) if context else ''
        recent_context = self.hot_kv.get_recent(n=5)
        full_context = f"{context_text}\n{recent_context}".strip()

        # 计算门控分数
        r = self.relevance_scorer.relevance(full_context, text)
        s = self.relevance_scorer.surprisal(text, full_context)

        # 冲突分数
        q_fields = self.field_extractor.extract_fields(text)
        recent_nodes = list(self.node_bank.values())[-10:] if self.node_bank else []
        c = self.relevance_scorer.conflict_score(q_fields, recent_nodes)

        # 门控
        gate_input = (self.config.relevance_weight * r +
                     self.config.surprisal_weight * s +
                     self.config.conflict_weight * c)
        gate = 1.0 / (1.0 + np.exp(-gate_input))

        if gate < self.config.write_threshold:
            return False

        # W3: 生成镜像节点
        node_id = hashlib.md5(text.encode()).hexdigest()[:8]
        summary = self.summarizer.summarize(text)
        fields = self.field_extractor.extract_fields(text)
        fields_text = ' '.join([
            ' '.join(map(str, fields.get('numbers', []))),
            ' '.join(fields.get('names', [])),
            ' '.join(fields.get('definitions', [])),
            ' '.join(fields.get('symbols', []))
        ])

        proto = self.embedder.embed(summary + ' ' + fields_text)

        # 证据存储
        evidence_ptr = hashlib.md5(text.encode()).hexdigest()
        self.evidence_store[evidence_ptr] = text[:self.config.max_evidence_length]

        node = MemoryNode(
            id=node_id,
            proto=proto,
            summary=summary,
            fields=fields,
            links=[],
            w=self.config.fresh_weight,
            evidence_ptr=evidence_ptr,
            evidence_text=text[:self.config.max_evidence_length],
            timestamp=datetime.now(),
            access_count=0
        )

        # W4: 衰减旧节点 + 入库
        for old_node in self.node_bank.values():
            old_node.w *= self.config.weight_decay_gamma

        self.node_bank[node_id] = node
        self.write_count += 1

        # W5: 收编动作 A - 构建 n-gram 影子索引
        self.ngram_index.add(node_id, text, summary, fields_text)

        # W6: 收编动作 B - 建立邻接边
        other_nodes = [n for nid, n in self.node_bank.items() if nid != node_id]
        self.link_graph.add_node(node, other_nodes[-20:])  # 只与最近20个节点建边

        # 可选：Trie 索引
        if self.trie_lm:
            self.trie_lm.add_text(summary + ' ' + fields_text, node_id)

        logger.info(f"[WriteMemory] 写入长期记忆: {summary[:50]}...")

        return True

    def route_memory(self, query: str, mode: Literal['fast', 'strict'] = 'fast') -> Tuple[List[MemoryNode], str]:
        """
        路由路径（收编版）：RouteMemory_Captured

        Step R1: 解析查询与惯性方向
        Step R2: 三路召回 (收编关键！)
        Step R3: 候选合并
        Step R4: AIM 主权 - 锚点纠错
        Step R5: 证据回灌
        Step R6: 更新惯性
        """
        self.total_access += 1

        if not self.node_bank:
            logger.info(f"[RouteMemory] 节点库为空，无法召回")
            return [], ""

        # R1: 解析查询与惯性方向
        q_fields = self.field_extractor.extract_fields(query)
        q_fields_text = ' '.join([
            ' '.join(map(str, q_fields.get('numbers', []))),
            ' '.join(q_fields.get('names', [])),
            ' '.join(q_fields.get('definitions', [])),
            ' '.join(q_fields.get('symbols', []))
        ])
        q_vec = self.embedder.embed(query + ' ' + q_fields_text)

        # 惯性方向
        d = q_vec + self.config.inertia_strength * self.v_inertia
        d = d / (np.linalg.norm(d) + 1e-8)

        # R2: 三路召回（收编核心）

        # (A) n-gram 命中召回
        ngram_hits = self.ngram_index.lookup(query + ' ' + q_fields_text, top_k=self.config.k_ng)
        cand_ng_ids = [nid for nid, _ in ngram_hits]
        self.ngram_recall_count += len(cand_ng_ids)

        # (B) 向量近邻召回
        node_similarities = []
        for node_id, node in self.node_bank.items():
            cos_sim = np.dot(d, node.proto) / (np.linalg.norm(node.proto) + 1e-8)
            node_similarities.append((node_id, cos_sim))
        node_similarities.sort(key=lambda x: x[1], reverse=True)
        cand_vec_ids = [nid for nid, _ in node_similarities[:self.config.k_vec]]
        self.vec_recall_count += len(cand_vec_ids)

        # (C) 邻接扩展召回
        seed_ids = list(set(cand_ng_ids[:self.config.k_seed] + cand_vec_ids[:self.config.k_seed]))
        cand_link_ids = self.link_graph.expand(seed_ids, limit=self.config.k_link)
        self.link_recall_count += len(cand_link_ids)

        # R3: 候选合并
        pool_ids = list(set(cand_ng_ids + cand_vec_ids + cand_link_ids))[:self.config.k_final]
        candidates = [self.node_bank[nid] for nid in pool_ids if nid in self.node_bank]

        logger.info(f"[RouteMemory] 三路召回: n-gram={len(cand_ng_ids)}, "
                   f"vec={len(cand_vec_ids)}, link={len(cand_link_ids)}, "
                   f"merged={len(candidates)}")

        # R4: AIM 主权步骤 - 锚点纠错（裁判权在 AIM！）
        scored_candidates = []

        for node in candidates:
            # 4.1 锚点一致性（硬规则）
            anchor_score = self.anchor_checker.anchor_check(q_fields, node.fields)

            if anchor_score < self.config.anchor_threshold:
                continue  # 锚点不过线，直接淘汰

            # 4.2 路由综合评分（收编：三路加权）
            s_ng = self.ngram_index.get_hit_strength(query + ' ' + q_fields_text, node.id)
            s_ng_norm = s_ng / (max([score for _, score in ngram_hits]) + 1e-8) if ngram_hits else 0

            s_vec = np.dot(q_vec, node.proto) / (np.linalg.norm(node.proto) + 1e-8)
            s_vec = max(0, s_vec)

            s_link = self.link_graph.get_link_affinity(q_fields, node.links)

            # 三路加权 + 锚点加成 + 时间权重
            node_score = anchor_score * (
                self.config.rho_ng * s_ng_norm +
                self.config.rho_vec * s_vec +
                self.config.rho_link * s_link
            ) * (1 + self.config.anchor_boost * node.w)

            scored_candidates.append((node, node_score))

        # 排序
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [node for node, _ in scored_candidates[:self.config.top_n_results]]

        # 更新访问计数
        for node in selected:
            node.access_count += 1

        logger.info(f"[RouteMemory] 召回 {len(selected)} 个节点, mode={mode}, "
                   f"锚点通过率={len(scored_candidates)}/{len(candidates)}")

        # R5: strict/冲突触发证据回灌
        refill_text = ""
        need_refill = (
            mode == 'strict' or
            self.config.refill_mode == 'always' or
            self._detect_strict_keywords(query)
        )

        if need_refill and selected:
            evidence_parts = []
            for node in selected:
                if node.evidence_ptr and node.evidence_ptr in self.evidence_store:
                    evidence = self.evidence_store[node.evidence_ptr]
                    evidence_parts.append(f"[证据 {node.evidence_ptr[:8]}] {evidence}")
            refill_text = '\n'.join(evidence_parts)

        # R6: 更新惯性向量
        if selected:
            v_sel = np.mean([node.proto for node in selected], axis=0)
            self.v_inertia = (self.config.inertia_momentum * self.v_inertia +
                            (1 - self.config.inertia_momentum) * v_sel)
        else:
            self.v_inertia = self.config.inertia_momentum * self.v_inertia

        return selected, refill_text

    def answer(self, query: str, auto_mode: bool = True) -> Dict[str, Any]:
        """
        生成路径（收编版）：Answer_Captured

        自动检测是否需要 strict 模式
        """
        # 自动模式判断
        mode = 'fast'
        if auto_mode and self._detect_strict_keywords(query):
            mode = 'strict'

        # 路由召回
        selected, refill = self.route_memory(query, mode)

        # 构建上下文
        context_parts = []

        # 热上下文
        hot_context = self.hot_kv.get_recent(n=5)
        if hot_context:
            context_parts.append("[热上下文]")
            context_parts.append('\n'.join(hot_context))  # 将列表转换为字符串

        # 记忆摘要
        if selected:
            context_parts.append("\n[记忆摘要]")
            for node in selected:
                context_parts.append(f"• {node.summary}")

        # 记忆字段
        if selected:
            context_parts.append("\n[记忆字段]")
            for node in selected:
                fields_str = []
                if node.fields.get('numbers'):
                    fields_str.append(f"numbers: {', '.join(map(str, node.fields['numbers']))}")
                if node.fields.get('names'):
                    fields_str.append(f"names: {', '.join(node.fields['names'])}")
                if node.fields.get('definitions'):
                    defs = node.fields['definitions'][:1]  # 取第一个定义
                    if defs:
                        fields_str.append(f"definitions: {defs[0] if isinstance(defs, list) else defs}")
                if fields_str:  # 只有当有字段时才添加
                    context_parts.append(f"• {' '.join(fields_str)}")

        # 证据原文
        if refill:
            context_parts.append("\n[证据原文]")
            context_parts.append(refill)

        context = '\n'.join(context_parts)

        return {
            'query': query,
            'mode': mode,
            'selected_nodes': selected,
            'num_nodes_recalled': len(selected),
            'context': context,
            'inertia_norm': float(np.linalg.norm(self.v_inertia)),
            'ngram_recall': self.ngram_recall_count,
            'vec_recall': self.vec_recall_count,
            'link_recall': self.link_recall_count
        }

    def _detect_strict_keywords(self, query: str) -> bool:
        """检测是否需要严格模式"""
        strict_keywords = ['原文', '精确', '准确', '证明', '证据', '引用', 'exact', 'precise', 'proof']
        return any(kw in query for kw in strict_keywords)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'hot_kv_size': len(self.hot_kv.buffer),
            'node_bank_size': len(self.node_bank),
            'evidence_store_size': len(self.evidence_store),
            'inertia_norm': float(np.linalg.norm(self.v_inertia)),
            'total_access': self.total_access,
            'write_count': self.write_count,
            'avg_weight': np.mean([n.w for n in self.node_bank.values()]) if self.node_bank else 0,
            'ngram_index_size': len(self.ngram_index.index),
            'link_graph_edges': sum(len(neighbors) for neighbors in self.link_graph.graph.values()) // 2,
            'ngram_recall_count': self.ngram_recall_count,
            'vec_recall_count': self.vec_recall_count,
            'link_recall_count': self.link_recall_count
        }

    def save(self, filepath: str):
        """保存到文件"""
        data = {
            'config': {k: v for k, v in self.config.__dict__.items()},
            'node_bank': {nid: node.to_dict() for nid, node in self.node_bank.items()},
            'evidence_store': self.evidence_store,
            'v_inertia': self.v_inertia.tolist(),
            'hot_kv': [{'text': item['text'], 'timestamp': item['timestamp'].isoformat()}
                       for item in self.hot_kv.buffer],  # 转换 datetime
            'ngram_index': self.ngram_index.to_dict(),
            'link_graph': self.link_graph.to_dict(),
            'stats': {
                'total_access': self.total_access,
                'write_count': self.write_count
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"[AIMMemoryNC] 保存到文件: {filepath}")

    def load(self, filepath: str):
        """从文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 加载节点
        self.node_bank = {}
        for node_id, node_dict in data['node_bank'].items():
            proto = np.array(node_dict['proto'])
            node = MemoryNode(
                id=node_dict['id'],
                proto=proto,
                summary=node_dict['summary'],
                fields=node_dict['fields'],
                links=node_dict.get('links', []),
                w=node_dict['w'],
                evidence_ptr=node_dict.get('evidence_ptr'),
                evidence_text=node_dict.get('evidence_text'),
                timestamp=datetime.fromisoformat(node_dict['timestamp']),
                access_count=node_dict.get('access_count', 0)
            )
            self.node_bank[node_id] = node

        # 加载其他数据
        self.evidence_store = data['evidence_store']
        self.v_inertia = np.array(data['v_inertia'])
        # 恢复 hot_kv，转换 timestamp 回 datetime
        self.hot_kv.buffer = deque(
            [{'text': item['text'], 'timestamp': datetime.fromisoformat(item['timestamp'])}
             for item in data['hot_kv']],
            maxlen=self.config.hot_window_size
        )

        # 加载收编组件
        self.ngram_index = NGramIndex.from_dict(data['ngram_index'])
        self.link_graph = LinkGraph.from_dict(data['link_graph'])

        # 加载统计
        if 'stats' in data:
            self.total_access = data['stats'].get('total_access', 0)
            self.write_count = data['stats'].get('write_count', 0)

        logger.info(f"[AIMMemoryNC] 从文件加载: {filepath}, 节点数={len(self.node_bank)}")


# ==================== 工厂函数 ====================

def create_aim_memory_nc(config: Optional[AIMNCConfig] = None) -> AIMMemoryNC:
    """创建 AIM-NC 实例"""
    return AIMMemoryNC(config)
