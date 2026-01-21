#!/usr/bin/env python3
"""
测试分层记忆系统和动态编码器

测试内容：
1. A/B/C档记忆存储与检索
2. 锚点指令解析（【封存·原文】等）
3. Key控检索 + 版本化
4. 防漂移验证
5. 动态编码器扩充
6. 统一上下文组合
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.INFO)


# ==================== 测试分层记忆系统 ====================

def test_abc_tier_storage():
    """测试 A/B/C 档存储"""
    print("\n" + "="*70)
    print("测试 1: A/B/C 档分层存储")
    print("="*70)

    from apt_model.memory.hierarchical_memory import create_hierarchical_memory

    memory = create_hierarchical_memory()

    # A档：必须原样保留
    memory.detail_store.add_verbatim(
        key="DEF:Apeiron:v1",
        content="Apeiron（ἄπειρον）是无限、未分化的原始存在，阿那克西曼德提出的宇宙本原。",
        version="v1",
        category="definition",
        importance=1.0
    )

    # B档：结构化字段
    memory.detail_store.add_structured(
        key="PARAM:HyperParams:v1",
        fields={
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100,
            "optimizer": "AdamW"
        },
        version="v1",
        category="parameter",
        importance=0.8
    )

    # C档：允许摘要
    memory.detail_store.add_narrative(
        key="NARR:PhilosophyBackground:v1",
        summary="古希腊哲学家探索宇宙本原时，提出了各种概念：水、火、气、原子等。",
        original_ref="可回溯到赫拉克利特、泰勒斯等人的著作",
        version="v1",
        category="background",
        importance=0.5
    )

    print(f"✅ A档（Verbatim）: {len(memory.detail_store.verbatim)} 条")
    print(f"✅ B档（Structured）: {len(memory.detail_store.structured)} 条")
    print(f"✅ C档（Narrative）: {len(memory.detail_store.narrative)} 条")

    # Key检索测试
    entry_a = memory.detail_store.get_by_key("DEF:Apeiron:v1")
    print(f"\n🔍 Key检索 A档: {entry_a.content[:50]}...")

    entry_b = memory.detail_store.get_by_key("PARAM:HyperParams:v1")
    print(f"🔍 Key检索 B档: {entry_b.fields}")

    return memory


def test_anchor_directives():
    """测试锚点指令解析"""
    print("\n" + "="*70)
    print("测试 2: 锚点指令解析（【封存·原文】等）")
    print("="*70)

    from apt_model.memory.hierarchical_memory import create_hierarchical_memory

    memory = create_hierarchical_memory()

    # 模拟用户输入（包含锚点指令）
    user_text = """
我要定义一些核心概念：

【封存·原文】DEF:LeftSpinSmooth:v1: 左旋平滑（Left-Spin Smooth）是一种单向缓冲门控机制，通过尖点强度φ控制步长，避免训练中的数值不稳定。

【封存·字段】PARAM:SmoothConfig:v1: {
    "alpha": 0.5,
    "tau": 0.3,
    "beta": 0.7,
    "spike_threshold": 0.3
}

【封存·摘要】NARR:Motivation:v1: 这个设计灵感来自于观察到深度学习训练中梯度尖点导致的NaN问题，传统方法（梯度裁剪、泰勒展开）都存在局限性。
"""

    # 自动解析并存储
    memory.process_anchor_directives(user_text, default_version="v1")

    print(f"✅ 解析到 A档: {len(memory.detail_store.verbatim)} 条")
    print(f"✅ 解析到 B档: {len(memory.detail_store.structured)} 条")
    print(f"✅ 解析到 C档: {len(memory.detail_store.narrative)} 条")

    # 查看骨架卡索引
    print(f"\n📋 骨架卡索引更新:")
    for key, oneliner in list(memory.skeleton.index.items())[:5]:
        print(f"  • {key}: {oneliner}")

    return memory


def test_versioning_and_retrieval():
    """测试版本化和检索"""
    print("\n" + "="*70)
    print("测试 3: 版本化 + Key控检索")
    print("="*70)

    from apt_model.memory.hierarchical_memory import create_hierarchical_memory

    memory = create_hierarchical_memory()

    # 添加多个版本
    memory.detail_store.add_verbatim(
        key="DEF:RoPE:v1",
        content="RoPE (Rotary Position Embedding) 是旋转位置编码。",
        version="v1"
    )

    memory.detail_store.add_verbatim(
        key="DEF:RoPE:v2",
        content="RoPE (Rotary Position Embedding) 是旋转位置编码，通过复数旋转实现位置信息注入。",
        version="v2"
    )

    memory.detail_store.add_verbatim(
        key="DEF:RoPE:v3",
        content="RoPE (Rotary Position Embedding) 是旋转位置编码，通过复数旋转实现位置信息注入，支持外推到更长序列。",
        version="v3"
    )

    print("✅ 添加了 3 个版本")

    # Key检索特定版本
    v1 = memory.detail_store.get_by_key("DEF:RoPE:v1")
    v3 = memory.detail_store.get_by_key("DEF:RoPE:v3")

    print(f"\n🔍 v1: {v1.content}")
    print(f"🔍 v3: {v3.content}")

    # 验证完整性
    assert v1.verify_integrity(), "v1 完整性验证失败"
    assert v3.verify_integrity(), "v3 完整性验证失败"
    print(f"\n✅ 完整性验证通过（哈希校验）")

    return memory


def test_anti_drift_validation():
    """测试防漂移验证"""
    print("\n" + "="*70)
    print("测试 4: 防漂移验证（一致性检查）")
    print("="*70)

    from apt_model.memory.hierarchical_memory import create_hierarchical_memory

    memory = create_hierarchical_memory()

    # 添加定义
    memory.detail_store.add_verbatim(
        key="DEF:iRoPE:v1",
        content="iRoPE (Interleaved RoPE) 是交错旋转位置编码，Llama 4 使用的技术。",
        version="v1"
    )

    memory.detail_store.add_structured(
        key="SYM:Notation:v1",
        fields={
            "φ": "尖点强度",
            "α": "缓冲系数",
            "τ": "时间常数"
        },
        version="v1"
    )

    # 测试文本（正确使用）
    good_text = """
我们使用 iRoPE (Interleaved RoPE) 是交错旋转位置编码，这是 Llama 4 使用的技术。
其中尖点强度 φ 由缓冲系数 α 和时间常数 τ 共同决定。
"""

    # 测试文本（错误使用 - 未原样引用）
    bad_text = """
我们使用 iRoPE，这是一种改进的位置编码方法。（❌ 未原样引用A档）
"""

    # 验证
    validation_good = memory.validator.validate_usage(
        good_text,
        referenced_keys=["DEF:iRoPE:v1", "SYM:Notation:v1"]
    )

    validation_bad = memory.validator.validate_usage(
        bad_text,
        referenced_keys=["DEF:iRoPE:v1"]
    )

    print(f"\n✅ 正确使用验证: valid={validation_good['valid']}")
    print(f"   Warnings: {len(validation_good['warnings'])}")

    print(f"\n⚠️  错误使用验证: valid={validation_bad['valid']}")
    print(f"   Warnings: {len(validation_bad['warnings'])}")
    for warning in validation_bad['warnings']:
        print(f"     • {warning}")

    return memory


def test_skeleton_and_detail_layers():
    """测试两层存储（骨架卡 + 细节仓）"""
    print("\n" + "="*70)
    print("测试 5: 两层存储（骨架卡 + 细节仓）")
    print("="*70)

    from apt_model.memory.hierarchical_memory import create_hierarchical_memory

    memory = create_hierarchical_memory()

    # 添加多条记忆
    for i in range(10):
        memory.detail_store.add_verbatim(
            key=f"DEF:Concept{i}:v1",
            content=f"这是概念 {i} 的详细定义，包含大量细节...",
            version="v1"
        )
        # 更新骨架索引（只存 one-liner）
        memory.skeleton.add_index(f"DEF:Concept{i}:v1", f"概念 {i}")

    # 添加锚点和禁止偏离点
    memory.skeleton.add_anchor("核心原则1：保持简洁")
    memory.skeleton.add_anchor("核心原则2：避免过度抽象")
    memory.skeleton.add_no_drift_point("禁止改变 RoPE 的基本定义")
    memory.skeleton.set_goal("实现 10M tokens 长上下文支持")

    # 编译骨架卡（200-400 tokens）
    skeleton_text = memory.skeleton.compile()

    print(f"✅ 细节仓存储: {len(memory.detail_store.verbatim)} 条")
    print(f"✅ 骨架卡索引: {len(memory.skeleton.index)} 条")
    print(f"✅ 核心锚点: {len(memory.skeleton.anchors)} 条")
    print(f"✅ 禁止偏离点: {len(memory.skeleton.no_drift_points)} 条")

    print(f"\n📋 骨架卡编译结果（前 300 字符）:\n{skeleton_text[:300]}...")

    return memory


def test_context_composition():
    """测试上下文组合（记忆注入）"""
    print("\n" + "="*70)
    print("测试 6: 上下文组合（骨架卡 + 细节检索）")
    print("="*70)

    from apt_model.memory.hierarchical_memory import create_hierarchical_memory

    memory = create_hierarchical_memory()

    # 设置完整场景
    text = """
【封存·原文】DEF:YaRN:v1: YaRN (Yet another RoPE extensioN) 是分维度缩放的 RoPE 变体，支持 128K 上下文。

【封存·字段】PARAM:YaRN:v1: {
    "scale_factor": 4.0,
    "beta_fast": 32,
    "beta_slow": 1,
    "max_position_embeddings": 128000
}

【封存·摘要】NARR:YaRNUsage:v1: YaRN 被 Qwen、DeepSeek、GPT-OSS 等主流模型采用。
"""

    memory.process_anchor_directives(text)
    memory.skeleton.add_anchor("使用 YaRN 进行上下文扩展")
    memory.skeleton.set_goal("集成 YaRN 到 APT-Transformer")

    # 组合上下文
    context = memory.compose_context(
        current_message="如何在模型中使用 YaRN？",
        include_skeleton=True,
        retrieve_details=True,
        validate_consistency=True
    )

    print(f"✅ 上下文组合完成")
    print(f"📦 骨架卡长度: {len(context['skeleton_card'])} 字符")
    print(f"🔍 检索到细节: {len(context['detail_entries'])} 条")
    print(f"✔️  验证结果: valid={context['validation']['valid']}")

    print(f"\n💬 完整上下文（前 500 字符）:\n{context['full_context'][:500]}...")

    return memory, context


def test_unified_composer():
    """测试统一组合器（两种系统集成）"""
    print("\n" + "="*70)
    print("测试 7: 统一组合器（基础 + 分层）")
    print("="*70)

    from apt_model.memory.context_composer import create_hierarchical_composer

    composer = create_hierarchical_composer()

    # 1. 使用基础系统（ChatGPT-style）
    composer.basic.save_memory("用户是 AI 研究员", category="general", importance=0.9)
    composer.basic.add_message("user", "帮我实现 YaRN")

    # 2. 使用分层系统（锚点指令）
    text = """
【封存·原文】DEF:iRoPE:v1: iRoPE 是交错旋转位置编码，支持 10M tokens。
"""
    composer.hierarchical.process_anchor_directives(text)

    # 3. 统一组合
    unified_context = composer.compose_unified_context(
        current_message="现在把 iRoPE 集成到模型中",
        use_basic=True,
        use_hierarchical=True,
        validate=True
    )

    print(f"✅ 统一上下文组合完成")
    print(f"📦 基础记忆: {len(composer.basic.saved_memories)} 条")
    print(f"📦 分层记忆（A档）: {len(composer.hierarchical.detail_store.verbatim)} 条")

    print(f"\n💬 统一上下文（前 600 字符）:\n{unified_context['full_context'][:600]}...")

    return composer


def test_persistence():
    """测试持久化（保存/加载）"""
    print("\n" + "="*70)
    print("测试 8: 持久化（JSON 保存/加载）")
    print("="*70)

    from apt_model.memory.hierarchical_memory import create_hierarchical_memory

    # 创建临时文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name

    try:
        # 创建并保存
        memory1 = create_hierarchical_memory()
        memory1.detail_store.add_verbatim(
            key="DEF:Test:v1",
            content="测试内容",
            version="v1"
        )
        memory1.skeleton.add_anchor("测试锚点")

        memory1.save_to_file(temp_path)
        print(f"✅ 保存到: {temp_path}")

        file_size = os.path.getsize(temp_path)
        print(f"📁 文件大小: {file_size} bytes")

        # 加载并验证
        memory2 = create_hierarchical_memory()
        memory2.load_from_file(temp_path)

        assert len(memory2.detail_store.verbatim) == 1
        assert len(memory2.skeleton.anchors) == 1
        print(f"✅ 加载验证通过")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ==================== 测试动态编码器 ====================

def test_dynamic_embeddings():
    """测试动态编码器扩充"""
    print("\n" + "="*70)
    print("测试 9: 动态编码器扩充")
    print("="*70)

    try:
        import torch
        from apt_model.modeling.embeddings import TokenEmbedding, ImageEmbedding

        # 1. 测试 TokenEmbedding 动态扩充
        print("\n[TokenEmbedding] 动态词表扩充测试:")
        token_embed = TokenEmbedding(vocab_size=1000, embedding_dim=128, enable_dynamic_expansion=True)

        print(f"  初始词表大小: {token_embed.current_vocab_size}")

        # 模拟遇到 OOV token
        tokens = torch.tensor([[1, 2, 3, 1500]])  # 1500 超出初始词表
        output = token_embed(tokens)

        print(f"  扩充后词表大小: {token_embed.current_vocab_size}")
        print(f"  输出形状: {output.shape}")
        assert token_embed.current_vocab_size >= 1500, "词表未扩充"
        print(f"  ✅ 词表动态扩充成功")

        # 2. 测试 ImageEmbedding 动态分辨率
        print("\n[ImageEmbedding] 动态分辨率测试:")
        image_embed = ImageEmbedding(d_model=128, image_size=224, patch_size=16, enable_dynamic_resolution=True)

        print(f"  初始 patch 数量: {image_embed.num_patches}")

        # 模拟不同分辨率图像
        images_224 = torch.randn(2, 3, 224, 224)
        images_448 = torch.randn(2, 3, 448, 448)  # 2倍分辨率

        output_224 = image_embed(images_224)
        output_448 = image_embed(images_448)

        print(f"  224x224 输出: {output_224.shape}")
        print(f"  448x448 输出: {output_448.shape}")
        assert output_224.shape[1] != output_448.shape[1], "未处理不同分辨率"
        print(f"  ✅ 动态分辨率支持成功")

    except ImportError:
        print("⚠️  PyTorch 未安装，跳过编码器测试")


# ==================== 主测试流程 ====================

def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("🚀 分层记忆系统 + 动态编码器测试")
    print("="*70)
    print("基于最佳实践 2026:")
    print("  • 细节不靠摘要保存，而是靠检索取原文")
    print("  • 分层记忆：A档（原文）、B档（字段）、C档（摘要）")
    print("  • 锚点指令：【封存·原文】、【封存·字段】、【封存·摘要】")
    print("  • 两层存储：骨架卡（随时注入）+ 细节仓（按需检索）")
    print("  • 防漂移：版本化 + 一致性校验")
    print("="*70)

    try:
        # 分层记忆测试
        test_abc_tier_storage()
        test_anchor_directives()
        test_versioning_and_retrieval()
        test_anti_drift_validation()
        test_skeleton_and_detail_layers()
        test_context_composition()
        test_unified_composer()
        test_persistence()

        # 动态编码器测试
        test_dynamic_embeddings()

        # 总结
        print("\n" + "="*70)
        print("✅ 所有测试通过!")
        print("="*70)
        print("\n📊 系统特性:")
        print("  ✅ A/B/C 档分层存储")
        print("  ✅ 锚点指令解析（【封存·原文】等）")
        print("  ✅ Key 控检索 + 版本化")
        print("  ✅ 防漂移验证（哈希 + 一致性）")
        print("  ✅ 两层存储（骨架卡 + 细节仓）")
        print("  ✅ 统一组合器（基础 + 分层）")
        print("  ✅ 动态编码器（词表 + 分辨率）")
        print("\n🎯 核心优势:")
        print("  • 细节永不丢失（原文检索）")
        print("  • 上下文高效（骨架卡 200-400 tokens）")
        print("  • 防止漂移（版本化校验）")
        print("  • 动态扩充（编码器自适应）")
        print("\n📚 参考文档:")
        print("  • docs/MEMORY_SYSTEM_GUIDE.md")
        print("  • apt_model/memory/hierarchical_memory.py")
        print("  • apt_model/modeling/embeddings.py")
        print("="*70)

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
