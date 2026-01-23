# APT 记忆系统完整指南

**基于 2025-2026 主流技术**
**版本**: 2026-01-21

---

## 🎯 核心技术整合

| 技术来源 | 核心特性 | APT实现 |
|---------|---------|---------|
| **ChatGPT Memory** | Saved memories + Chat history | ✅ 完整实现 |
| **MemGPT** | 两层架构（Main/External context） | ✅ 集成 |
| **Mem0** | 自动提取、高效检索 | ✅ 自动化 |
| **Context Engineering** | 记忆注入 + 个性化 | ✅ Context Composer |

---

## 📊 系统架构

```
┌─────────────────────────────────────────────────────┐
│              APT 记忆系统架构图                       │
├─────────────────────────────────────────────────────┤
│                                                      │
│  用户输入                                             │
│    │                                                 │
│    ▼                                                 │
│  ┌──────────────────────────────────────────┐      │
│  │     Context Composer（上下文组合器）       │      │
│  │                                            │      │
│  │  1. Saved Memories（长期记忆）             │      │
│  │     • 用户偏好                             │      │
│  │     • 重要事实                             │      │
│  │     • 术语定义                             │      │
│  │                                            │      │
│  │  2. Chat History（对话历史）               │      │
│  │     • 最近1000轮对话                       │      │
│  │     • 语义检索                             │      │
│  │                                            │      │
│  │  3. Skeleton State（骨架状态）             │      │
│  │     • topic: 主题                          │      │
│  │     • constraints: 约束                    │      │
│  │     • definitions: 定义                    │      │
│  │     • unresolved: 未决问题                 │      │
│  │     • style_preference: 风格               │      │
│  │     • spike_regions: 尖点记录              │      │
│  └──────────────────────────────────────────┘      │
│    │                                                 │
│    ▼                                                 │
│  Memory Pack（记忆包）                               │
│    │                                                 │
│    ▼                                                 │
│  ┌──────────────────────────────────────────┐      │
│  │   RoPE（位置编码）+ 左旋平滑（数值稳定）    │      │
│  └──────────────────────────────────────────┘      │
│    │                                                 │
│    ▼                                                 │
│  模型推理 + 生成                                      │
│                                                      │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 方法 1: 基础使用

```python
from apt_model.memory.context_composer import create_context_composer

# 创建上下文组合器
composer = create_context_composer()

# 保存记忆（用户可主动或自动）
composer.save_memory("用户名是 Alice", category="general", importance=0.9)
composer.save_memory("喜欢简洁的代码", category="preference", importance=0.8)

# 添加对话历史
composer.add_message("user", "帮我优化RoPE")
composer.add_message("assistant", "好的，我会实现YaRN")

# 组合上下文
context = composer.compose_context(
    current_message="现在把RoPE集成到模型中",
    include_memories=True,
    include_history=True
)

print(context['system_prompt'])
# 输出包含所有相关记忆的系统prompt
```

### 方法 2: 与 RoPE + 左旋平滑集成

```python
from apt_model.modeling.advanced_rope import create_rope, RoPEConfig
from apt_model.modeling.memory_augmented_smooth import create_memory_augmented_smooth
from apt_model.memory.context_composer import create_context_composer

# 1. 创建记忆系统
composer = create_context_composer()

# 2. 创建 RoPE（位置编码）
rope = create_rope(RoPEConfig(rope_type="yarn", max_position_embeddings=128000))

# 3. 创建记忆增强左旋平滑（数值稳定）
smooth = create_memory_augmented_smooth(d_model=768)

# 4. 在对话中使用
def process_message(user_message: str):
    # 组合上下文（记忆注入）
    context = composer.compose_context(user_message)

    # 构建完整prompt
    full_prompt = context['system_prompt'] + "\n\nUser: " + user_message

    # 模型推理（使用RoPE + 左旋平滑）
    # q_rot, k_rot = rope(q, k)  # 位置编码
    # u_next, stats = smooth(u, delta_u, use_memory=True)  # 数值稳定

    return full_prompt

# 5. 保存对话
composer.add_message("user", user_message)
composer.add_message("assistant", assistant_response)
```

### 方法 3: 自动记忆管理

```python
# 自动从对话中提取重要信息
conversation = """
用户: 我的项目是 APT-Transformer，主要做长上下文优化。
助手: 好的，我了解了。你想实现什么功能？
用户: 我需要支持 10M tokens 的上下文，并且必须保持向后兼容。
"""

# 自动提取并保存（类似 Mem0）
composer.extract_and_save(conversation, auto_categorize=True)

# 检查保存的记忆
for memory in composer.saved_memories:
    print(f"{memory.category}: {memory.content}")

# 输出:
# topic: 项目是 APT-Transformer，主要做长上下文优化
# constraint: 必须保持向后兼容
```

---

## 📋 核心组件详解

### 1. Saved Memories（长期记忆）

**特性**:
- ✅ 用户可控（类似 ChatGPT）
- ✅ 按类别分类（topic/constraint/preference等）
- ✅ 重要性评分（0-1）
- ✅ 访问统计

**操作**:
```python
# 保存
composer.save_memory("用户偏好Python", category="preference", importance=0.8)

# 检索
memories = composer.retrieve_memories("Python", top_k=5)

# 删除
composer.delete_memory(index=0)

# 清空
composer.clear_all_memories()
```

### 2. Chat History（对话历史）

**特性**:
- ✅ 自动存储最近1000轮
- ✅ 语义检索相关对话
- ✅ 时间戳 + 元数据

**操作**:
```python
# 添加消息
composer.add_message("user", "帮我写代码")
composer.add_message("assistant", "好的，我来帮你")

# 检索相关历史
history = composer.retrieve_history("写代码", top_k=3)

for msg in history:
    print(f"[{msg.timestamp}] {msg.role}: {msg.content}")
```

### 3. Skeleton State（骨架状态）

**6个字段**:

| 字段 | 说明 | 示例 |
|-----|------|------|
| **topic** | 主题 | "RoPE优化和记忆系统" |
| **constraints** | 约束条件 | "必须保持向后兼容" |
| **definitions** | 术语定义 | "RoPE指旋转位置编码" |
| **unresolved** | 未决问题 | "如何集成到Virtual Blackwell" |
| **style_preference** | 风格偏好 | "代码简洁+详细注释" |
| **spike_regions** | 尖点区域 | "第5步训练出现NaN" |

**操作**:
```python
# 更新骨架
composer.update_skeleton("topic", "长上下文优化", importance=1.0)
composer.update_skeleton("constraints", "兼容性第一", importance=0.9)

# 获取摘要
summary = composer.skeleton.compress()
print(summary)

# 输出:
# TOPIC: 长上下文优化
# CONSTRAINTS: 兼容性第一
```

### 4. Context Composition（上下文组合）

**核心方法**:
```python
context = composer.compose_context(
    current_message="用户的当前输入",
    include_memories=True,    # 包含Saved Memories
    include_history=True,     # 包含Chat History
    include_skeleton=True     # 包含Skeleton State
)

# 返回:
{
    'system_prompt': '系统prompt（包含所有记忆）',
    'user_message': '用户消息',
    'memory_pack': {
        'saved_memories': [...],
        'chat_history': [...],
        'skeleton': {...}
    },
    'context_tokens': 1234  # 估算token数
}
```

---

## 💾 持久化

### 保存到文件

```python
# 保存
composer.save_to_file("user_memory.json")

# 加载
composer2 = create_context_composer()
composer2.load_from_file("user_memory.json")
```

**文件格式**:
```json
{
  "saved_memories": [
    {
      "content": "用户名是 Alice",
      "category": "general",
      "importance": 0.9,
      "timestamp": "2026-01-21T10:30:00",
      "access_count": 5
    }
  ],
  "chat_history": [...],
  "skeleton": {
    "fields": {
      "topic": [...],
      "constraints": [...]
    }
  }
}
```

---

## 🔬 性能优势

### 对比其他系统

| 系统 | Token使用 | 延迟 | 个性化 | APT支持 |
|-----|----------|------|--------|---------|
| **Mem0** | -90% | -91% | ✅ | ✅ 已集成 |
| **MemGPT** | 基准 | 基准 | ✅ | ✅ 架构借鉴 |
| **A-Mem** | -85% | -75% | ✅ | 🔄 规划中 |
| **ChatGPT** | 官方 | 官方 | ✅ | ✅ API兼容 |

### 成本节省

根据 [Mem0 报告](https://mem0.ai/blog/context-engineering-ai-agents-guide):
- **30-60%** API成本降低（避免重复上下文）
- **40-70%** 用户留存率提升（个性化体验）
- **26%** LLM评分改进（相关性更强）

---

## 🎨 应用案例

### 案例 1: 长期个性化助手

```python
# 第一次对话
composer.save_memory("用户是AI研究员", category="general", importance=1.0)
composer.save_memory("研究方向是Transformer优化", category="topic", importance=0.9)
composer.save_memory("喜欢PyTorch而不是TensorFlow", category="preference", importance=0.8)

# 一周后的对话
context = composer.compose_context("帮我优化注意力机制")

# system_prompt 会自动包含:
# - 用户是AI研究员
# - 研究Transformer优化
# - 喜欢PyTorch
```

### 案例 2: 项目上下文保持

```python
# 更新骨架（项目状态）
composer.update_skeleton("topic", "APT-Transformer长上下文优化", importance=1.0)
composer.update_skeleton("constraints", "必须支持100K GPU训练", importance=0.9)
composer.update_skeleton("unresolved", "如何优化NVLink 5通信", importance=0.7)

# 跨会话保持
composer.save_to_file("project_context.json")

# 下次会话自动加载
composer.load_from_file("project_context.json")
# 骨架状态自动恢复！
```

### 案例 3: 多轮复杂推理

```python
# 第1轮
composer.add_message("user", "我要实现iRoPE")
composer.update_skeleton("topic", "iRoPE实现", importance=1.0)

# 第5轮（中间穿插其他话题）
composer.add_message("user", "现在回到iRoPE，怎么集成？")

# Context Composer 自动检索：
# - Saved memory: "iRoPE是Llama 4使用的技术"
# - Chat history: "第1轮讨论了iRoPE"
# - Skeleton: "topic=iRoPE实现"
```

---

## 🔗 与其他组件集成

### 与 Virtual Blackwell 集成

```python
import apt_model.optimization.vb_global as vb
from apt_model.memory.context_composer import create_context_composer

# 启用全优化
vb.enable_full_optimization()

# 创建记忆系统
composer = create_context_composer()

# 保存VB配置偏好
composer.save_memory("使用MXFP4量化", category="preference", importance=0.9)
composer.save_memory("启用100K GPU训练", category="constraint", importance=1.0)

# 下次自动应用
context = composer.compose_context("配置训练环境")
# system_prompt 会提醒使用MXFP4和100K GPU
```

---

## 📚 参考资料

### 官方文档
- [ChatGPT Memory FAQ](https://help.openai.com/en/articles/8590148-memory-faq) - OpenAI
- [MemGPT Paper](https://arxiv.org/abs/2310.08560) - arXiv 2310.08560
- [Mem0 Documentation](https://mem0.ai/) - Memory Layer for AI
- [Context Engineering Guide](https://mem0.ai/blog/context-engineering-ai-agents-guide) - Oct 2025

### 博客文章
- [How ChatGPT Remembers You](https://embracethered.com/blog/posts/2025/chatgpt-how-does-chat-history-memory-preferences-work/) - Jan 2025
- [Context-Aware Memory Systems 2025](https://www.tribe.ai/applied-ai/beyond-the-bubble-how-context-aware-memory-systems-are-changing-the-game-in-2025)
- [Long-term Memory in LLM Applications](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/)

### 研究论文
- [Memory in the Age of AI Agents](https://arxiv.org/abs/2512.13564) (Dec 2025)
- [Enabling Personalized Long-term Interactions](https://arxiv.org/abs/2510.07925) (Oct 2025)

---

## ❓ FAQ

**Q1: 和 ChatGPT Memory 有什么区别？**

A: APT 记忆系统是 **开源实现** + **扩展功能**：
- ✅ 完全兼容 ChatGPT Memory API
- ✅ 额外支持 Skeleton State（骨架状态）
- ✅ 可本地部署，数据完全私有
- ✅ 与 RoPE + 左旋平滑深度集成

**Q2: 记忆会占用多少上下文？**

A: 默认 **30%** 上下文用于记忆包（可配置）：
- 8K 上下文 → ~2.4K 记忆包
- 128K 上下文 → ~38K 记忆包
- 实际根据相关性动态调整

**Q3: 如何控制记忆的隐私？**

A: 完全可控：
```python
# 禁用自动保存
config = MemoryConfig(auto_save_threshold=1.1)  # 永不自动保存

# 手动审核
memories = composer.saved_memories
for i, m in enumerate(memories):
    if "敏感信息" in m.content:
        composer.delete_memory(i)

# 清空所有
composer.clear_all_memories()
```

**Q4: 支持多用户吗？**

A: 支持！每个用户独立文件：
```python
user_id = "alice"
composer.save_to_file(f"memory/{user_id}.json")
```

---

**文档版本**: 1.0
**最后更新**: 2026-01-21
**维护者**: APT-Transformer Team
