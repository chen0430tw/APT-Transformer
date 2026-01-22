#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT 分层记忆系统（增强版）

核心设计理念（基于最佳实践 2026）:
"细节不靠摘要保存，而是靠检索取原文"

三档记忆分类：
- A档（Verbatim）：必须原样保存，不允许摘要替代
- B档（Structured）：结构化存储（JSON/键值对）
- C档（Narrative）：可摘要，但保留回溯链接

锚点指令系统：
- 【封存·原文】：强制逐字保留（A档）
- 【封存·字段】：结构化条目（B档）
- 【封存·摘要】：允许压缩（C档）

两层存储架构：
- Layer 1：骨架卡（随时注入，200-400 tokens）
- Layer 2：细节仓（按需检索，原文/字段）

检索策略：
- Key路径：版本化键控检索（直接命中原文）
- 语义路径：向量相似度 + 回溯原文

防漂移机制：
- 版本化校验（DEF:concept:v1）
- 一致性检查（符号/条件/禁止偏离点）

作者: chen0430tw
日期: 2026-01-21
"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from typing import Optional, Dict, Any, List, Tuple, Literal
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib
import re
import logging

logger = logging.getLogger(__name__)


# ==================== 配置 ====================

@dataclass
class HierarchicalMemoryConfig:
    """分层记忆系统配置"""

    # 基础配置
    enable_hierarchical: bool = True  # 启用分层记忆
    enable_anchor_directives: bool = True  # 启用锚点指令
    enable_versioning: bool = True  # 启用版本化
    enable_anti_drift: bool = True  # 启用防漂移

    # Layer 1: 骨架卡配置
    skeleton_max_tokens: int = 400  # 骨架卡最大token数
    skeleton_update_threshold: float = 0.7  # 骨架更新阈值

    # Layer 2: 细节仓配置
    detail_store_max_items: int = 1000  # 细节仓最大条目数
    verbatim_max_length: int = 5000  # 原文最大字符数
    structured_max_fields: int = 100  # 结构化字段最大数量

    # 检索配置
    key_retrieval_enabled: bool = True  # 启用Key检索
    semantic_retrieval_enabled: bool = True  # 启用语义检索
    retrieval_top_k: int = 5  # 检索top-k

    # 版本控制
    default_version: str = "v1"  # 默认版本
    auto_increment_version: bool = True  # 自动递增版本

    # 防漂移配置
    validation_strict: bool = True  # 严格校验
    warn_on_inconsistency: bool = True  # 不一致时警告


# ==================== 记忆条目类型 ====================

@dataclass
class VerbatimEntry:
    """
    A档：原文条目（必须逐字保留）

    适用场景：
    - 严格定义
    - 符号约定
    - 定理条件
    - 角色名单
    - 禁止偏离的表述
    """
    key: str  # 唯一标识符 (e.g., "DEF:ApeironContinuum:v1")
    content: str  # 原文内容
    version: str  # 版本号
    hash: str  # 内容哈希（用于校验）
    timestamp: datetime
    category: str = "definition"  # definition/symbol/theorem/constraint
    importance: float = 1.0
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def verify_integrity(self) -> bool:
        """校验内容完整性"""
        current_hash = hashlib.sha256(self.content.encode('utf-8')).hexdigest()
        return current_hash == self.hash

    def to_dict(self) -> Dict:
        return {
            "key": self.key,
            "content": self.content,
            "version": self.version,
            "hash": self.hash,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category,
            "importance": self.importance,
            "access_count": self.access_count,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'VerbatimEntry':
        return cls(
            key=data["key"],
            content=data["content"],
            version=data["version"],
            hash=data["hash"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            category=data.get("category", "definition"),
            importance=data.get("importance", 1.0),
            access_count=data.get("access_count", 0),
            metadata=data.get("metadata", {})
        )


@dataclass
class StructuredEntry:
    """
    B档：结构化条目（键值对/JSON）

    适用场景：
    - 参数配置
    - 阈值表
    - 流程步骤
    - 判据列表
    - 对比表格
    """
    key: str  # 唯一标识符 (e.g., "PARAM:HyperParams:v2")
    fields: Dict[str, Any]  # 结构化字段
    version: str  # 版本号
    timestamp: datetime
    category: str = "parameter"  # parameter/threshold/procedure/criteria
    importance: float = 0.8
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "key": self.key,
            "fields": self.fields,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category,
            "importance": self.importance,
            "access_count": self.access_count,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'StructuredEntry':
        return cls(
            key=data["key"],
            fields=data["fields"],
            version=data["version"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            category=data.get("category", "parameter"),
            importance=data.get("importance", 0.8),
            access_count=data.get("access_count", 0),
            metadata=data.get("metadata", {})
        )


@dataclass
class NarrativeEntry:
    """
    C档：摘要条目（允许压缩）

    适用场景：
    - 背景叙述
    - 讨论过程
    - 类比说明
    - 灵感来源
    """
    key: str  # 唯一标识符 (e.g., "NARR:Background:v1")
    summary: str  # 摘要内容
    original_ref: Optional[str]  # 原文引用（可回溯）
    version: str  # 版本号
    timestamp: datetime
    category: str = "background"  # background/discussion/analogy/inspiration
    importance: float = 0.5
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "key": self.key,
            "summary": self.summary,
            "original_ref": self.original_ref,
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category,
            "importance": self.importance,
            "access_count": self.access_count,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'NarrativeEntry':
        return cls(
            key=data["key"],
            summary=data["summary"],
            original_ref=data.get("original_ref"),
            version=data["version"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            category=data.get("category", "background"),
            importance=data.get("importance", 0.5),
            access_count=data.get("access_count", 0),
            metadata=data.get("metadata", {})
        )


# ==================== 骨架卡 ====================

class SkeletonCard:
    """
    Layer 1: 骨架卡（随时注入，短小精悍）

    内容：
    - 术语表索引
    - 核心锚点
    - 版本号
    - 禁止偏离点
    - 当前目标
    """

    def __init__(self, config: HierarchicalMemoryConfig):
        self.config = config
        self.index: Dict[str, str] = {}  # key -> one-liner
        self.anchors: List[str] = []  # 核心锚点
        self.no_drift_points: List[str] = []  # 禁止偏离点
        self.current_goal: Optional[str] = None

    def add_index(self, key: str, oneliner: str):
        """添加索引条目"""
        self.index[key] = oneliner

    def add_anchor(self, anchor: str):
        """添加核心锚点"""
        if anchor not in self.anchors:
            self.anchors.append(anchor)

    def add_no_drift_point(self, point: str):
        """添加禁止偏离点"""
        if point not in self.no_drift_points:
            self.no_drift_points.append(point)

    def set_goal(self, goal: str):
        """设置当前目标"""
        self.current_goal = goal

    def compile(self) -> str:
        """编译成可注入的文本（200-400 tokens）"""
        parts = []

        # 1. 索引（术语表）
        if self.index:
            parts.append("【索引】")
            for key, oneliner in list(self.index.items())[:20]:  # 最多20条
                parts.append(f"  • {key}: {oneliner}")

        # 2. 核心锚点
        if self.anchors:
            parts.append("\n【锚点】")
            for anchor in self.anchors[:10]:  # 最多10条
                parts.append(f"  • {anchor}")

        # 3. 禁止偏离点
        if self.no_drift_points:
            parts.append("\n【禁止偏离】")
            for point in self.no_drift_points[:5]:  # 最多5条
                parts.append(f"  • {point}")

        # 4. 当前目标
        if self.current_goal:
            parts.append(f"\n【当前目标】\n  {self.current_goal}")

        return "\n".join(parts)

    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "anchors": self.anchors,
            "no_drift_points": self.no_drift_points,
            "current_goal": self.current_goal
        }

    def from_dict(self, data: Dict):
        self.index = data.get("index", {})
        self.anchors = data.get("anchors", [])
        self.no_drift_points = data.get("no_drift_points", [])
        self.current_goal = data.get("current_goal")


# ==================== 细节仓 ====================

class DetailStore:
    """
    Layer 2: 细节仓（按需检索）

    存储：
    - A档原文（VerbatimEntry）
    - B档字段（StructuredEntry）
    - C档摘要（NarrativeEntry）
    """

    def __init__(self, config: HierarchicalMemoryConfig):
        self.config = config
        self.verbatim: Dict[str, VerbatimEntry] = {}
        self.structured: Dict[str, StructuredEntry] = {}
        self.narrative: Dict[str, NarrativeEntry] = {}

    def add_verbatim(self, key: str, content: str, version: str, category: str = "definition", importance: float = 1.0):
        """添加A档原文"""
        content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
        entry = VerbatimEntry(
            key=key,
            content=content,
            version=version,
            hash=content_hash,
            timestamp=datetime.now(),
            category=category,
            importance=importance
        )
        self.verbatim[key] = entry
        logger.info(f"[DetailStore] A档添加: {key}")

    def add_structured(self, key: str, fields: Dict, version: str, category: str = "parameter", importance: float = 0.8):
        """添加B档结构化"""
        entry = StructuredEntry(
            key=key,
            fields=fields,
            version=version,
            timestamp=datetime.now(),
            category=category,
            importance=importance
        )
        self.structured[key] = entry
        logger.info(f"[DetailStore] B档添加: {key}")

    def add_narrative(self, key: str, summary: str, original_ref: Optional[str], version: str, category: str = "background", importance: float = 0.5):
        """添加C档摘要"""
        entry = NarrativeEntry(
            key=key,
            summary=summary,
            original_ref=original_ref,
            version=version,
            timestamp=datetime.now(),
            category=category,
            importance=importance
        )
        self.narrative[key] = entry
        logger.info(f"[DetailStore] C档添加: {key}")

    def get_by_key(self, key: str) -> Optional[Any]:
        """Key检索（精确匹配）"""
        # 按优先级查找：A档 > B档 > C档
        if key in self.verbatim:
            entry = self.verbatim[key]
            entry.access_count += 1
            return entry
        if key in self.structured:
            entry = self.structured[key]
            entry.access_count += 1
            return entry
        if key in self.narrative:
            entry = self.narrative[key]
            entry.access_count += 1
            return entry
        return None

    def search_by_keyword(self, keyword: str, top_k: int = 5) -> List[Any]:
        """关键词检索（模糊匹配）"""
        results = []

        # 搜索所有条目
        for entry in list(self.verbatim.values()) + list(self.structured.values()) + list(self.narrative.values()):
            # 简单关键词匹配（可替换为语义搜索）
            if isinstance(entry, VerbatimEntry):
                text = entry.content
            elif isinstance(entry, StructuredEntry):
                text = json.dumps(entry.fields)
            elif isinstance(entry, NarrativeEntry):
                text = entry.summary
            else:
                continue

            if keyword.lower() in text.lower():
                results.append((entry.importance, entry))

        # 按重要性排序
        results.sort(reverse=True, key=lambda x: x[0])
        return [entry for _, entry in results[:top_k]]

    def to_dict(self) -> Dict:
        return {
            "verbatim": {k: v.to_dict() for k, v in self.verbatim.items()},
            "structured": {k: v.to_dict() for k, v in self.structured.items()},
            "narrative": {k: v.to_dict() for k, v in self.narrative.items()}
        }

    def from_dict(self, data: Dict):
        self.verbatim = {k: VerbatimEntry.from_dict(v) for k, v in data.get("verbatim", {}).items()}
        self.structured = {k: StructuredEntry.from_dict(v) for k, v in data.get("structured", {}).items()}
        self.narrative = {k: NarrativeEntry.from_dict(v) for k, v in data.get("narrative", {}).items()}


# ==================== 锚点指令解析器 ====================

class AnchorDirectiveParser:
    """
    锚点指令解析器

    识别并解析：
    - 【封存·原文】：A档
    - 【封存·字段】：B档
    - 【封存·摘要】：C档
    """

    @staticmethod
    def parse(text: str) -> List[Tuple[str, str, str]]:
        """
        解析文本中的锚点指令

        返回: List[(directive_type, key, content)]
            directive_type: "verbatim" | "structured" | "narrative"
        """
        results = []

        # 正则匹配锚点指令
        pattern = r'【封存·(原文|字段|摘要)】\s*(?:(\S+):)?\s*(.+?)(?=【封存·|$)'

        matches = re.finditer(pattern, text, re.DOTALL)

        for match in matches:
            directive_cn, key, content = match.groups()

            # 映射中文到英文类型
            type_map = {
                "原文": "verbatim",
                "字段": "structured",
                "摘要": "narrative"
            }
            directive_type = type_map.get(directive_cn, "narrative")

            # 如果没有指定key，生成默认key
            if not key:
                key = f"AUTO:{directive_type.upper()}:{datetime.now().strftime('%Y%m%d%H%M%S')}"

            content = content.strip()
            results.append((directive_type, key, content))

        return results


# ==================== 防漂移验证器 ====================

class AntiDriftValidator:
    """
    防漂移验证器

    检查：
    - 版本一致性
    - 符号一致性
    - 定义一致性
    """

    def __init__(self, detail_store: DetailStore):
        self.detail_store = detail_store

    def validate_usage(self, text: str, referenced_keys: List[str]) -> Dict[str, Any]:
        """
        验证文本使用的概念是否一致

        返回: {
            "valid": bool,
            "warnings": List[str],
            "errors": List[str]
        }
        """
        warnings = []
        errors = []

        for key in referenced_keys:
            entry = self.detail_store.get_by_key(key)

            if entry is None:
                errors.append(f"引用了不存在的key: {key}")
                continue

            # A档：检查是否原样引用
            if isinstance(entry, VerbatimEntry):
                if not entry.verify_integrity():
                    errors.append(f"A档内容已被篡改: {key}")

                # 检查文本中是否包含原文片段
                if entry.content[:50] not in text and entry.content[-50:] not in text:
                    warnings.append(f"A档 {key} 未在文本中原样引用")

            # B档：检查字段是否完整
            elif isinstance(entry, StructuredEntry):
                for field_key in entry.fields.keys():
                    if field_key not in text:
                        warnings.append(f"B档 {key} 的字段 '{field_key}' 未在文本中引用")

        return {
            "valid": len(errors) == 0,
            "warnings": warnings,
            "errors": errors
        }


# ==================== 分层记忆管理器 ====================

class HierarchicalMemoryManager:
    """
    分层记忆管理器（主类）

    整合：
    - SkeletonCard（骨架卡）
    - DetailStore（细节仓）
    - AnchorDirectiveParser（锚点解析）
    - AntiDriftValidator（防漂移验证）
    """

    def __init__(self, config: Optional[HierarchicalMemoryConfig] = None):
        self.config = config or HierarchicalMemoryConfig()
        self.skeleton = SkeletonCard(self.config)
        self.detail_store = DetailStore(self.config)
        self.parser = AnchorDirectiveParser()
        self.validator = AntiDriftValidator(self.detail_store)

    def process_anchor_directives(self, text: str, default_version: str = "v1"):
        """
        处理文本中的锚点指令

        自动识别并存储：
        - 【封存·原文】 → A档
        - 【封存·字段】 → B档
        - 【封存·摘要】 → C档
        """
        directives = self.parser.parse(text)

        for directive_type, key, content in directives:
            # 添加版本号（如果key中没有）
            if ":v" not in key:
                key = f"{key}:{default_version}"

            # 提取版本号
            version_match = re.search(r':v(\d+)', key)
            version = f"v{version_match.group(1)}" if version_match else default_version

            # 根据类型存储
            if directive_type == "verbatim":
                self.detail_store.add_verbatim(key, content, version, importance=1.0)
                # 更新骨架索引
                oneliner = content[:50] + "..." if len(content) > 50 else content
                self.skeleton.add_index(key, oneliner)

            elif directive_type == "structured":
                # 尝试解析为JSON
                try:
                    fields = json.loads(content)
                except json.JSONDecodeError:
                    # 如果不是JSON，按行解析为键值对
                    fields = {}
                    for line in content.split('\n'):
                        if ':' in line:
                            k, v = line.split(':', 1)
                            fields[k.strip()] = v.strip()

                self.detail_store.add_structured(key, fields, version, importance=0.8)
                # 更新骨架索引
                oneliner = f"{len(fields)} fields"
                self.skeleton.add_index(key, oneliner)

            elif directive_type == "narrative":
                self.detail_store.add_narrative(key, content, original_ref=None, version=version, importance=0.5)

            logger.info(f"[HierarchicalMemory] 处理指令: {directive_type} -> {key}")

    def compose_context(
        self,
        current_message: str,
        include_skeleton: bool = True,
        retrieve_details: bool = True,
        validate_consistency: bool = True
    ) -> Dict[str, Any]:
        """
        组合上下文（记忆注入）

        返回: {
            "skeleton_card": str,  # 骨架卡（总是注入）
            "detail_entries": List,  # 检索到的细节条目
            "validation": Dict,  # 验证结果
            "full_context": str  # 完整上下文
        }
        """
        context_parts = []

        # 1. 骨架卡（总是注入）
        skeleton_text = ""
        if include_skeleton:
            skeleton_text = self.skeleton.compile()
            if skeleton_text:
                context_parts.append("【骨架卡 - 随时可用】\n" + skeleton_text)

        # 2. 检索细节（按需注入）
        detail_entries = []
        if retrieve_details:
            # 关键词检索
            entries = self.detail_store.search_by_keyword(current_message, top_k=self.config.retrieval_top_k)
            detail_entries = entries

            if entries:
                context_parts.append("\n【细节仓 - 按需检索】")
                for entry in entries:
                    if isinstance(entry, VerbatimEntry):
                        context_parts.append(f"\n[A档·原文] {entry.key} (v{entry.version})")
                        context_parts.append(f"  {entry.content}")
                    elif isinstance(entry, StructuredEntry):
                        context_parts.append(f"\n[B档·字段] {entry.key} (v{entry.version})")
                        context_parts.append(f"  {json.dumps(entry.fields, ensure_ascii=False, indent=2)}")
                    elif isinstance(entry, NarrativeEntry):
                        context_parts.append(f"\n[C档·摘要] {entry.key} (v{entry.version})")
                        context_parts.append(f"  {entry.summary}")

        # 3. 验证一致性
        validation = {"valid": True, "warnings": [], "errors": []}
        if validate_consistency and detail_entries:
            referenced_keys = [e.key for e in detail_entries]
            validation = self.validator.validate_usage(current_message, referenced_keys)

            if not validation["valid"]:
                context_parts.append("\n【⚠️ 一致性警告】")
                for error in validation["errors"]:
                    context_parts.append(f"  ❌ {error}")
                for warning in validation["warnings"]:
                    context_parts.append(f"  ⚠️  {warning}")

        full_context = "\n".join(context_parts)

        return {
            "skeleton_card": skeleton_text,
            "detail_entries": detail_entries,
            "validation": validation,
            "full_context": full_context
        }

    def save_to_file(self, filepath: str):
        """持久化到文件"""
        data = {
            "skeleton": self.skeleton.to_dict(),
            "detail_store": self.detail_store.to_dict(),
            "config": {
                "enable_hierarchical": self.config.enable_hierarchical,
                "enable_anchor_directives": self.config.enable_anchor_directives,
                "enable_versioning": self.config.enable_versioning,
                "enable_anti_drift": self.config.enable_anti_drift
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"[HierarchicalMemory] 保存到文件: {filepath}")

    def load_from_file(self, filepath: str):
        """从文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.skeleton.from_dict(data["skeleton"])
        self.detail_store.from_dict(data["detail_store"])

        logger.info(f"[HierarchicalMemory] 从文件加载: {filepath}")


# ==================== 工厂函数 ====================

def create_hierarchical_memory(config: Optional[HierarchicalMemoryConfig] = None) -> HierarchicalMemoryManager:
    """创建分层记忆管理器"""
    return HierarchicalMemoryManager(config)
