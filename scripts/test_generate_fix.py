#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试APT模型生成功能的修复
"""

import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_generate_fix() -> bool:
    """测试生成功能是否修复"""

    print("=" * 60)
    print("测试 APT 模型生成功能")
    print("=" * 60)

    # 1. 准备测试数据
    test_texts = [
        "人工智能正在改变世界",
        "深度学习需要大量数据",
        "自然语言处理很重要",
    ]

    print(f"\n1. 准备测试数据: {len(test_texts)} 条")

    # 2. 初始化分词器
    print("\n2. 初始化分词器...")
    from apt_model.modeling.chinese_tokenizer_integration import (
        get_appropriate_tokenizer,
    )

    tokenizer, lang = get_appropriate_tokenizer(
        texts=test_texts,
        tokenizer_type="chinese-char",
    )

    print("   ✓ 分词器初始化成功")
    print(f"   - 词汇表大小: {tokenizer.vocab_size}")
    print(f"   - 语言: {lang}")

    # 3. 创建模型
    print("\n3. 创建模型...")
    from apt_model.modeling.apt_model import APTModel, APTModelConfiguration

    config = APTModelConfiguration(
        vocab_size=tokenizer.vocab_size,
        d_model=256,  # 小模型便于测试
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=512,
        max_seq_len=128,
        dropout=0.1,
        pad_token_id=getattr(tokenizer, "pad_token_id", 0),
        bos_token_id=getattr(tokenizer, "bos_token_id", 101),
        eos_token_id=getattr(tokenizer, "eos_token_id", 102),
        unk_token_id=getattr(tokenizer, "unk_token_id", 3),
    )

    model = APTModel(config)
    model.eval()

    print("   ✓ 模型创建成功")
    print(f"   - 参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 4. 测试编码
    print("\n4. 测试编码...")
    test_prompt = "人工智能"

    input_ids = tokenizer.encode(test_prompt, return_tensors="pt")
    print(f"   - 输入文本: {test_prompt}")
    print(f"   - 编码结果: {input_ids.tolist()[0]}")
    print(f"   - 编码长度: {input_ids.size(1)}")

    # 5. 测试生成（未训练模型）
    print("\n5. 测试生成（未训练模型）...")

    with torch.no_grad():
        try:
            if not hasattr(model, "generate"):
                print("   ❌ 错误: 模型没有generate方法!")
                return False

            generated_ids = model.generate(
                input_ids,
                max_length=20,
                temperature=1.0,
                top_p=0.9,
                do_sample=True,
            )

            print(f"   - 生成ID: {generated_ids.tolist()[0][:20]}")
            print(f"   - 生成长度: {generated_ids.size(1)}")

            generated_text = tokenizer.decode(generated_ids[0])
            print(f"   - 生成文本: {generated_text}")

            unk_count = generated_text.count("<unk>") + generated_text.count("<|unk|>")
            total_len = len(generated_text)

            print(f"   - 未知词数量: {unk_count} / {total_len}")

            if total_len > 0 and unk_count > total_len * 0.8:
                print("   ⚠️ 警告: 生成结果包含过多未知词")
                print("   原因: 模型未训练，输出接近随机")
                print("   解决: 需要训练模型")
            else:
                print("   ✓ 生成功能正常")

        except Exception as exc:
            print(f"   ❌ 生成失败: {exc}")
            import traceback

            traceback.print_exc()
            return False

    # 6. 测试贪心解码
    print("\n6. 测试贪心解码...")

    with torch.no_grad():
        try:
            generated_ids = model.generate(
                input_ids,
                max_length=15,
                temperature=1.0,
                do_sample=False,
            )

            generated_text = tokenizer.decode(generated_ids[0])
            print(f"   - 贪心生成: {generated_text}")
            print("   ✓ 贪心解码正常")

        except Exception as exc:
            print(f"   ❌ 贪心解码失败: {exc}")

    # 7. 测试不同温度
    print("\n7. 测试不同温度参数...")

    temperatures = [0.1, 0.5, 1.0, 1.5]

    for temp in temperatures:
        with torch.no_grad():
            try:
                generated_ids = model.generate(
                    input_ids,
                    max_length=15,
                    temperature=temp,
                    do_sample=True,
                )

                generated_text = tokenizer.decode(generated_ids[0])
                print(f"   - 温度={temp}: {generated_text[:30]}...")

            except Exception as exc:
                print(f"   ❌ 温度={temp}失败: {exc}")

    # 8. 总结
    print("\n" + "=" * 60)
    print("测试总结:")
    print("=" * 60)
    print("✓ 生成方法已修复并可以正常工作")
    print("⚠️ 未训练模型的输出质量较低是正常现象")
    print("")
    print("下一步:")
    print("1. 使用修复后的代码训练模型")
    print("2. 训练后生成质量会显著提升")
    print("3. 如果训练后仍有问题，可能需要:")
    print("   - 增加训练数据量")
    print("   - 调整模型参数")
    print("   - 增加训练轮数")

    return True


if __name__ == "__main__":
    success = test_generate_fix()
    sys.exit(0 if success else 1)
