#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT 记忆系统 - Context Composer（上下文组合器）

整合 2025-2026 主流记忆技术:
1. ChatGPT Memory: Saved memories + Chat history reference
2. MemGPT: 两层架构（Main context / External context）
3. Mem0: 自动提取、高效检索
4. Context Engineering: 记忆注入 + 个性化

核心组件:
- Saved Memories: 长期偏好/事实（用户可控）
- Chat History: 对话历史检索
- Skeleton State: 骨架状态（6个字段）
- Memory Injection: 动态上下文组合

参考资料:
- ChatGPT Memory (OpenAI, 2025)
  https://openai.com/index/memory-and-new-controls-for-chatgpt/

- MemGPT: Towards LLMs as Operating Systems (2023)
  https://arxiv.org/abs/2310.08560

- Mem0: Memory Layer for AI (2025)
  https://mem0.ai/

- Context Engineering Guide (Oct 2025)
  https://mem0.ai/blog/context-engineering-ai-agents-guide

作者: chen0430tw
日期: 2026-01-21
"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    BaseClass = nn.Module
except ImportError:
    TORCH_AVAILABLE = False
    # Fallback base class when torch is not available
    class BaseClass:
        def __init__(self):
            pass

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


# ==================== 配置 ====================

@dataclass
class MemoryConfig:
    """记忆系统配置"""

    # Saved Memories（长期记忆）
    saved_memories_max: int = 100  # 最多保存100个记忆
    auto_save_threshold: float = 0.7  # 重要性>0.7自动保存

    # Chat History（对话历史）
    chat_history_max: int = 1000  # 最多保存1000轮对话
    history_retrieve_top_k: int = 5  # 检索前5个相关历史

    # Skeleton State（骨架状态）
    skeleton_fields: List[str] = field(default_factory=lambda: [
        "topic",              # 主题
        "constraints",        # 约束条件
        "definitions",        # 术语定义
        "unresolved",         # 未决问题
        "style_preference",   # 风格偏好
        "spike_regions"       # 尖点区域（危险记录）
    ])

    # Context Composer（上下文组合）
    max_context_tokens: int = 8192  # 最大上下文长度
    memory_pack_ratio: float = 0.3  # 记忆包占比30%

    # 性能优化
    use_semantic_search: bool = True  # 语义搜索
    use_compression: bool = True  # 压缩存储
    embedding_dim: int = 768


# ==================== Saved Memories ====================

class SavedMemory:
    """
    单条保存的记忆

    类似 ChatGPT 的 Saved Memories
    """

    def __init__(
        self,
        content: str,
        category: str = "general",
        importance: float = 0.5,
        timestamp: Optional[datetime] = None
    ):
        self.content = content
        self.category = category  # topic/constraint/definition/preference等
        self.importance = importance
        self.timestamp = timestamp or datetime.now()
        self.access_count = 0
        self.last_accessed = None

        # 嵌入向量（用于语义搜索）
        self.embedding = None

    def access(self):
        """记录访问"""
        self.access_count += 1
        self.last_accessed = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return {
            "content": self.content,
            "category": self.category,
            "importance": self.importance,
            "timestamp": self.timestamp.isoformat(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SavedMemory':
        """从字典导入"""
        memory = cls(
            content=data["content"],
            category=data.get("category", "general"),
            importance=data.get("importance", 0.5),
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
        memory.access_count = data.get("access_count", 0)
        if data.get("last_accessed"):
            memory.last_accessed = datetime.fromisoformat(data["last_accessed"])
        return memory


# ==================== Chat History ====================

class ChatMessage:
    """对话消息"""

    def __init__(
        self,
        role: str,  # "user" or "assistant"
        content: str,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict] = None
    ):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        self.embedding = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )


# ==================== Skeleton State ====================

class SkeletonState:
    """
    骨架状态（海马体式结构记忆）

    6个字段维护长期推理主干
    """

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.fields = {field: [] for field in config.skeleton_fields}

        # 压缩表示
        self.latent_embedding = None

    def update_field(self, field: str, content: str, importance: float = 0.5):
        """更新骨架字段"""
        if field not in self.fields:
            logger.warning(f"[Skeleton] 未知字段: {field}")
            return

        # 添加到字段（带重要性）
        self.fields[field].append({
            "content": content,
            "importance": importance,
            "timestamp": datetime.now().isoformat()
        })

        # 保持最重要的前10个
        self.fields[field] = sorted(
            self.fields[field],
            key=lambda x: x["importance"],
            reverse=True
        )[:10]

    def get_field_summary(self, field: str) -> str:
        """获取字段摘要"""
        if field not in self.fields or not self.fields[field]:
            return ""

        # 拼接前3个最重要的内容
        items = self.fields[field][:3]
        return " | ".join([item["content"] for item in items])

    def compress(self) -> str:
        """压缩骨架为文本"""
        parts = []

        for field in self.config.skeleton_fields:
            summary = self.get_field_summary(field)
            if summary:
                parts.append(f"{field.upper()}: {summary}")

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fields": self.fields,
            "compressed": self.compress()
        }


# ==================== Context Composer ====================

class ContextComposer(BaseClass):
    """
    上下文组合器

    职责:
    1. 从 Saved Memories 检索相关记忆
    2. 从 Chat History 检索相关对话
    3. 提取 Skeleton State
    4. 组合成 Memory Pack
    5. 注入到 Prompt

    类似 MemGPT 的虚拟上下文管理
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        super().__init__()
        self.config = config or MemoryConfig()

        # Saved Memories
        self.saved_memories: List[SavedMemory] = []

        # Chat History
        self.chat_history: List[ChatMessage] = []

        # Skeleton State
        self.skeleton = SkeletonState(self.config)

        # 嵌入编码器（用于语义搜索）
        if self.config.use_semantic_search and TORCH_AVAILABLE:
            self.embedding_encoder = nn.Linear(
                self.config.embedding_dim,
                self.config.embedding_dim
            )
        else:
            self.embedding_encoder = None

        logger.info(
            f"[ContextComposer] 初始化完成: "
            f"max_memories={self.config.saved_memories_max}, "
            f"max_history={self.config.chat_history_max}"
        )

    # ========== Saved Memories 管理 ==========

    def save_memory(
        self,
        content: str,
        category: str = "general",
        importance: float = 0.5
    ):
        """
        保存记忆

        用户可主动调用，或系统自动保存（importance > threshold）
        """
        memory = SavedMemory(content, category, importance)
        self.saved_memories.append(memory)

        # 保持最多N个
        if len(self.saved_memories) > self.config.saved_memories_max:
            # 删除最不重要的
            self.saved_memories.sort(key=lambda m: m.importance, reverse=True)
            self.saved_memories = self.saved_memories[:self.config.saved_memories_max]

        logger.info(f"[Memory] 已保存: {content[:50]}... (importance={importance:.2f})")

    def delete_memory(self, index: int):
        """删除记忆"""
        if 0 <= index < len(self.saved_memories):
            deleted = self.saved_memories.pop(index)
            logger.info(f"[Memory] 已删除: {deleted.content[:50]}...")

    def clear_all_memories(self):
        """清空所有记忆"""
        self.saved_memories.clear()
        logger.info("[Memory] 已清空所有记忆")

    def retrieve_memories(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None
    ) -> List[SavedMemory]:
        """
        检索相关记忆

        Args:
            query: 查询内容
            top_k: 返回前k个
            category: 筛选类别

        Returns:
            相关记忆列表
        """
        # 筛选类别
        candidates = self.saved_memories
        if category:
            candidates = [m for m in candidates if m.category == category]

        if not candidates:
            return []

        # 简单的关键词匹配（实际可用语义搜索）
        scores = []
        for memory in candidates:
            # 计算重叠词数
            query_words = set(query.lower().split())
            memory_words = set(memory.content.lower().split())
            overlap = len(query_words & memory_words)

            # 结合重要性
            score = overlap * memory.importance
            scores.append((score, memory))

        # 排序并返回top_k
        scores.sort(reverse=True, key=lambda x: x[0])
        results = [m for _, m in scores[:top_k]]

        # 记录访问
        for m in results:
            m.access()

        return results

    # ========== Chat History 管理 ==========

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ):
        """添加消息到对话历史"""
        message = ChatMessage(role, content, metadata=metadata)
        self.chat_history.append(message)

        # 保持最多N条
        if len(self.chat_history) > self.config.chat_history_max:
            self.chat_history.pop(0)

    def retrieve_history(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[ChatMessage]:
        """
        检索相关对话历史

        类似 ChatGPT 的 "Reference chat history"
        """
        top_k = top_k or self.config.history_retrieve_top_k

        if not self.chat_history:
            return []

        # 简单的关键词匹配
        scores = []
        for msg in self.chat_history:
            query_words = set(query.lower().split())
            msg_words = set(msg.content.lower().split())
            overlap = len(query_words & msg_words)
            scores.append((overlap, msg))

        # 排序并返回
        scores.sort(reverse=True, key=lambda x: x[0])
        return [m for _, m in scores[:top_k]]

    # ========== Skeleton State 管理 ==========

    def update_skeleton(
        self,
        field: str,
        content: str,
        importance: float = 0.5
    ):
        """更新骨架状态"""
        self.skeleton.update_field(field, content, importance)

    # ========== Context Composition ==========

    def compose_context(
        self,
        current_message: str,
        include_memories: bool = True,
        include_history: bool = True,
        include_skeleton: bool = True
    ) -> Dict[str, Any]:
        """
        组合上下文

        核心功能：将记忆注入到prompt

        Args:
            current_message: 当前用户消息
            include_memories: 是否包含saved memories
            include_history: 是否包含chat history
            include_skeleton: 是否包含skeleton state

        Returns:
            {
                'system_prompt': 系统prompt（包含记忆）,
                'user_message': 用户消息,
                'memory_pack': 记忆包详情,
                'context_tokens': 估算token数
            }
        """
        memory_parts = []

        # 1. Saved Memories
        retrieved_memories = []
        if include_memories and self.saved_memories:
            retrieved_memories = self.retrieve_memories(current_message, top_k=5)
            if retrieved_memories:
                mem_texts = [f"- {m.content}" for m in retrieved_memories]
                memory_parts.append(
                    "=== SAVED MEMORIES ===\n" +
                    "\n".join(mem_texts)
                )

        # 2. Chat History
        retrieved_history = []
        if include_history and self.chat_history:
            retrieved_history = self.retrieve_history(current_message, top_k=3)
            if retrieved_history:
                hist_texts = []
                for msg in retrieved_history:
                    hist_texts.append(
                        f"[{msg.timestamp.strftime('%Y-%m-%d %H:%M')}] "
                        f"{msg.role}: {msg.content[:100]}..."
                    )
                memory_parts.append(
                    "=== RELEVANT CHAT HISTORY ===\n" +
                    "\n".join(hist_texts)
                )

        # 3. Skeleton State
        skeleton_text = ""
        if include_skeleton:
            skeleton_text = self.skeleton.compress()
            if skeleton_text:
                memory_parts.append(
                    "=== CONTEXT SKELETON ===\n" +
                    skeleton_text
                )

        # 组合
        memory_pack = "\n\n".join(memory_parts) if memory_parts else ""

        # 构建系统prompt
        system_prompt = ""
        if memory_pack:
            system_prompt = (
                "You have access to the following context and memories:\n\n" +
                memory_pack +
                "\n\nUse this information to provide personalized and contextually aware responses."
            )

        # 估算token数（粗略：4个字符=1个token）
        estimated_tokens = len(system_prompt + current_message) // 4

        return {
            'system_prompt': system_prompt,
            'user_message': current_message,
            'memory_pack': {
                'saved_memories': [m.to_dict() for m in retrieved_memories],
                'chat_history': [h.to_dict() for h in retrieved_history],
                'skeleton': self.skeleton.to_dict()
            },
            'context_tokens': estimated_tokens
        }

    # ========== 自动记忆管理 ==========

    def extract_and_save(
        self,
        conversation_text: str,
        auto_categorize: bool = True
    ):
        """
        自动从对话中提取重要信息并保存

        类似 Mem0 的自动提取

        简化版：使用关键词识别
        实际可用LLM进行智能提取
        """
        # 关键词映射
        category_keywords = {
            "topic": ["关于", "主题是", "讨论", "话题"],
            "constraint": ["不能", "必须", "要求", "限制"],
            "definition": ["定义", "是指", "意思是", "指的是"],
            "preference": ["喜欢", "偏好", "倾向", "习惯"],
            "unresolved": ["问题", "疑问", "不确定", "待解决"]
        }

        # 提取句子
        sentences = conversation_text.split("。")

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue

            # 判断类别
            detected_category = "general"
            max_matches = 0

            if auto_categorize:
                for category, keywords in category_keywords.items():
                    matches = sum(1 for kw in keywords if kw in sentence)
                    if matches > max_matches:
                        max_matches = matches
                        detected_category = category

            # 计算重要性（简化：句子越长越重要）
            importance = min(len(sentence) / 100, 1.0)

            # 自动保存（如果重要性超过阈值）
            if importance >= self.config.auto_save_threshold:
                self.save_memory(sentence, detected_category, importance)

                # 同时更新骨架
                if detected_category in self.config.skeleton_fields:
                    self.update_skeleton(detected_category, sentence, importance)

    # ========== 持久化 ==========

    def save_to_file(self, filepath: str):
        """保存到文件"""
        data = {
            "saved_memories": [m.to_dict() for m in self.saved_memories],
            "chat_history": [msg.to_dict() for msg in self.chat_history[-100:]],  # 只保存最近100条
            "skeleton": self.skeleton.to_dict()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"[ContextComposer] 已保存到: {filepath}")

    def load_from_file(self, filepath: str):
        """从文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 加载 saved memories
        self.saved_memories = [
            SavedMemory.from_dict(m) for m in data.get("saved_memories", [])
        ]

        # 加载 chat history
        self.chat_history = [
            ChatMessage.from_dict(msg) for msg in data.get("chat_history", [])
        ]

        # 加载 skeleton
        if "skeleton" in data:
            self.skeleton.fields = data["skeleton"].get("fields", {})

        logger.info(
            f"[ContextComposer] 已加载: "
            f"{len(self.saved_memories)} memories, "
            f"{len(self.chat_history)} messages"
        )


# ==================== 便捷函数 ====================

def create_context_composer(config: Optional[MemoryConfig] = None) -> ContextComposer:
    """创建上下文组合器（ChatGPT-style）"""
    return ContextComposer(config)


def create_hierarchical_composer(config: Optional[MemoryConfig] = None):
    """
    创建分层记忆组合器（增强版）

    结合两种记忆系统：
    1. ContextComposer（ChatGPT-style）：用户友好，简单易用
    2. HierarchicalMemoryManager（分层记忆）：精确控制，防漂移

    Example:
        >>> from apt_model.memory.context_composer import create_hierarchical_composer
        >>>
        >>> composer = create_hierarchical_composer()
        >>>
        >>> # 使用锚点指令
        >>> text = \"\"\"
        >>> 【封存·原文】DEF:Apeiron:v1: Apeiron是无限未分化的原始存在。
        >>> 【封存·字段】PARAM:HyperParams:v1: {"learning_rate": 0.001, "batch_size": 32}
        >>> 【封存·摘要】NARR:Background:v1: 这个概念源于古希腊哲学。
        >>> \"\"\"
        >>>
        >>> # 自动解析并存储
        >>> composer.hierarchical.process_anchor_directives(text)
        >>>
        >>> # 组合上下文
        >>> context = composer.compose_unified_context("讨论 Apeiron 概念")
        >>> print(context['full_context'])
    """
    try:
        from apt_model.memory.hierarchical_memory import (
            create_hierarchical_memory,
            HierarchicalMemoryConfig
        )

        class UnifiedComposer:
            """统一组合器（同时使用两种系统）"""

            def __init__(self, config: Optional[MemoryConfig] = None):
                self.basic = ContextComposer(config)  # ChatGPT-style
                self.hierarchical = create_hierarchical_memory(HierarchicalMemoryConfig())  # 分层记忆

            def compose_unified_context(
                self,
                current_message: str,
                use_basic: bool = True,
                use_hierarchical: bool = True,
                validate: bool = True
            ) -> Dict[str, Any]:
                """
                统一上下文组合（同时使用两种系统）

                Args:
                    current_message: 当前用户消息
                    use_basic: 使用基础记忆系统
                    use_hierarchical: 使用分层记忆系统
                    validate: 启用一致性验证

                Returns:
                    统一上下文字典
                """
                result = {
                    "basic_context": None,
                    "hierarchical_context": None,
                    "full_context": ""
                }

                parts = []

                # 1. 基础记忆系统（ChatGPT-style）
                if use_basic:
                    basic_ctx = self.basic.compose_context(
                        current_message,
                        include_memories=True,
                        include_history=True,
                        include_skeleton=True
                    )
                    result["basic_context"] = basic_ctx
                    if basic_ctx["system_prompt"]:
                        parts.append("【基础记忆系统】\n" + basic_ctx["system_prompt"])

                # 2. 分层记忆系统（增强版）
                if use_hierarchical:
                    hier_ctx = self.hierarchical.compose_context(
                        current_message,
                        include_skeleton=True,
                        retrieve_details=True,
                        validate_consistency=validate
                    )
                    result["hierarchical_context"] = hier_ctx
                    if hier_ctx["full_context"]:
                        parts.append("\n【分层记忆系统】\n" + hier_ctx["full_context"])

                result["full_context"] = "\n\n".join(parts)

                return result

            def save_to_file(self, filepath_basic: str, filepath_hierarchical: str):
                """保存到文件（两个文件）"""
                self.basic.save_to_file(filepath_basic)
                self.hierarchical.save_to_file(filepath_hierarchical)

            def load_from_file(self, filepath_basic: str, filepath_hierarchical: str):
                """从文件加载（两个文件）"""
                self.basic.load_from_file(filepath_basic)
                self.hierarchical.load_from_file(filepath_hierarchical)

        return UnifiedComposer(config)

    except ImportError as e:
        logger.warning(f"分层记忆系统不可用: {e}，回退到基础系统")
        return create_context_composer(config)


# ==================== 测试代码 ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("APT 记忆系统测试")
    print("=" * 70)

    # 创建上下文组合器
    composer = create_context_composer()

    # 测试 1: 保存记忆
    print("\n[测试 1] 保存记忆")
    composer.save_memory("用户名是 chen0430tw", category="general", importance=0.9)
    composer.save_memory("用户喜欢简洁的代码风格", category="preference", importance=0.8)
    composer.save_memory("项目主题是 APT-Transformer 优化", category="topic", importance=1.0)
    composer.save_memory("必须保持向后兼容性", category="constraint", importance=0.9)

    print(f"✓ 已保存 {len(composer.saved_memories)} 条记忆")

    # 测试 2: 添加对话历史
    print("\n[测试 2] 添加对话历史")
    composer.add_message("user", "你好，我想优化RoPE")
    composer.add_message("assistant", "好的，我可以帮你实现YaRN和iRoPE")
    composer.add_message("user", "太好了，还需要记忆系统")
    composer.add_message("assistant", "我正在实现MemGPT风格的记忆系统")

    print(f"✓ 已添加 {len(composer.chat_history)} 条消息")

    # 测试 3: 更新骨架
    print("\n[测试 3] 更新骨架状态")
    composer.update_skeleton("topic", "RoPE优化和记忆系统", importance=1.0)
    composer.update_skeleton("constraints", "保持向后兼容", importance=0.8)
    composer.update_skeleton("unresolved", "如何集成到Virtual Blackwell", importance=0.7)

    skeleton_summary = composer.skeleton.compress()
    print(f"✓ 骨架状态:\n{skeleton_summary}")

    # 测试 4: 组合上下文
    print("\n[测试 4] 组合上下文")
    current_msg = "现在把RoPE和记忆系统集成起来"

    context = composer.compose_context(
        current_msg,
        include_memories=True,
        include_history=True,
        include_skeleton=True
    )

    print(f"\n系统Prompt:\n{context['system_prompt']}")
    print(f"\n估算tokens: {context['context_tokens']}")

    # 测试 5: 检索
    print("\n[测试 5] 检索记忆")
    retrieved = composer.retrieve_memories("RoPE", top_k=3)
    print(f"✓ 检索到 {len(retrieved)} 条相关记忆:")
    for i, mem in enumerate(retrieved, 1):
        print(f"  {i}. {mem.content} (importance={mem.importance:.2f})")

    # 测试 6: 持久化
    print("\n[测试 6] 持久化")
    composer.save_to_file("/tmp/apt_memory.json")

    # 创建新的composer并加载
    composer2 = create_context_composer()
    composer2.load_from_file("/tmp/apt_memory.json")
    print(f"✓ 加载成功: {len(composer2.saved_memories)} memories")

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
    print("\n核心特性:")
    print("  ✓ Saved Memories (ChatGPT风格)")
    print("  ✓ Chat History Reference")
    print("  ✓ Skeleton State (骨架状态)")
    print("  ✓ Context Composition (记忆注入)")
    print("  ✓ 自动提取和保存")
    print("  ✓ 持久化存储")
