#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分词器诊断脚本
用于检查分词器是否正确初始化
"""

import sys
sys.path.append('.')

def diagnose_tokenizer():
    """诊断分词器问题"""
    
    print("=" * 60)
    print("APT 分词器诊断工具")
    print("=" * 60)
    
    # 1. 准备测试数据
    test_texts = [
        "人工智能正在改变世界",
        "深度学习需要大量数据",
        "自然语言处理是AI的重要分支",
        "机器学习模型训练很重要",
        "这是一个测试样本"
    ]
    
    print(f"\n1. 测试数据: {len(test_texts)} 条")
    for i, text in enumerate(test_texts[:3], 1):
        print(f"   {i}. {text}")
    
    # 2. 测试分词器初始化
    print("\n2. 初始化分词器...")
    
    from apt.apt_model.modeling.chinese_tokenizer_integration import get_appropriate_tokenizer
    
    try:
        tokenizer, lang = get_appropriate_tokenizer(
            texts=test_texts,
            tokenizer_type="chinese-char",
            language="zh"
        )
        print(f"   ✓ 分词器初始化成功")
        print(f"   - 语言: {lang}")
        print(f"   - 词汇表大小: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"   ✗ 初始化失败: {e}")
        return
    
    # 3. 检查词汇表
    print("\n3. 检查词汇表...")
    
    if hasattr(tokenizer, 'encoder'):
        vocab_size = len(tokenizer.encoder)
        print(f"   - 实际词汇表大小: {vocab_size}")
        
        # 显示前20个词
        print(f"   - 前20个词:")
        for i, (token, id_) in enumerate(list(tokenizer.encoder.items())[:20]):
            print(f"      {id_:4d}: {repr(token)}")
        
        # 检查关键字符是否在词汇表中
        test_chars = ['人', '工', '智', '能', '学', '习']
        missing = [c for c in test_chars if c not in tokenizer.encoder]
        
        if missing:
            print(f"   ⚠️ 警告: 以下字符不在词汇表中: {missing}")
        else:
            print(f"   ✓ 关键字符都在词汇表中")
    
    # 4. 测试编码
    print("\n4. 测试编码...")
    
    test_text = test_texts[0]
    print(f"   测试文本: {test_text}")
    
    try:
        encoded = tokenizer.encode(test_text)
        print(f"   编码结果: {encoded}")
        print(f"   编码长度: {len(encoded)}")
        
        # 检查是否全是未知词
        if hasattr(tokenizer, 'encoder'):
            unk_id = tokenizer.encoder.get("<|unk|>", -1)
            unk_count = encoded.count(unk_id)
            unk_ratio = unk_count / len(encoded) if encoded else 0
            
            print(f"   未知词数量: {unk_count} / {len(encoded)} ({unk_ratio*100:.1f}%)")
            
            if unk_ratio > 0.5:
                print(f"   ✗ 错误: 超过50%是未知词!")
            else:
                print(f"   ✓ 未知词比例正常")
    except Exception as e:
        print(f"   ✗ 编码失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 测试解码
    print("\n5. 测试解码...")
    
    try:
        decoded = tokenizer.decode(encoded)
        print(f"   解码结果: {decoded}")
        
        # 检查是否包含大量 <unk>
        unk_in_decoded = decoded.count("<unk>") + decoded.count("<|unk|>")
        
        if unk_in_decoded > len(test_text) * 0.3:
            print(f"   ✗ 错误: 解码结果包含大量 <unk>: {unk_in_decoded}")
        else:
            print(f"   ✓ 解码结果正常")
        
        # 检查往返一致性
        if test_text == decoded or decoded.replace(" ", "") == test_text:
            print(f"   ✓ 编码-解码往返一致")
        else:
            print(f"   ⚠️ 警告: 编码-解码不一致")
    except Exception as e:
        print(f"   ✗ 解码失败: {e}")
        return
    
    # 6. 批量测试
    print("\n6. 批量测试所有样本...")
    
    success = 0
    failed = 0
    
    for text in test_texts:
        try:
            ids = tokenizer.encode(text)
            decoded = tokenizer.decode(ids)
            
            unk_id = tokenizer.encoder.get("<|unk|>", -1)
            unk_ratio = ids.count(unk_id) / len(ids) if ids else 0
            
            if unk_ratio < 0.5:
                success += 1
            else:
                failed += 1
                print(f"   ✗ 失败: {text[:20]}... (未知词比例: {unk_ratio*100:.1f}%)")
        except:
            failed += 1
    
    print(f"\n   结果: {success} 成功, {failed} 失败")
    
    # 7. 总结
    print("\n" + "=" * 60)
    print("诊断总结:")
    print("=" * 60)
    
    if failed == 0:
        print("✓ 分词器工作正常！")
        return 0
    else:
        print("✗ 分词器存在问题，建议:")
        print("  1. 增加词汇表大小 (--vocab-size 100000)")
        print("  2. 使用更多训练数据构建词汇表")
        print("  3. 检查 chinese_tokenizer.py 中的 prepare_vocab_from_texts 方法")
        return 1


if __name__ == "__main__":
    sys.exit(diagnose_tokenizer())
