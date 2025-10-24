#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修改APT模型训练器以支持中文分词
"""

import os
import re
import torch
import torch.nn.functional as F
import traceback
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

from apt_model.utils import set_seed
from apt_model.utils import get_device, device
from apt_model.config.apt_config import APTConfig
from apt_model.modeling.apt_model import APTModel, APTLargeModel

# 导入中文分词器相关函数
from apt_model.modeling.chinese_tokenizer_integration import (
    get_appropriate_tokenizer,
    save_tokenizer,
    is_chinese_text
)

_FALLBACK_TRAINING_TEXTS = [
        # 基本对话
        "Hello, how are you?",
        "I'm doing well, thank you for asking. How about you?",
        "Good morning! How did you sleep last night?",
        "Good morning! I slept very well, thank you for asking.",
        "What's your name?",
        "My name is Claude. It's nice to meet you.",
        "How's the weather today?",
        "The weather is lovely today. It's sunny and warm.",
        "Can you help me with a question?",
        "Of course, I'd be happy to help you with any questions you have.",
        "What time is it?",
        "I'm sorry, I don't have access to real-time information like the current time.",
        
        # 原神相关内容
        "安柏：一起来训练吧！",
        "安柏是蒙德城的侦察骑士，擅长弓箭和侦察。",
        "旅行者，欢迎来到提瓦特大陆。",
        "原神是一款开放世界冒险游戏。",
        "风起万山摇，暗潮寄余生。",
        "璃月港是七国之一璃月的主要港口城市。",
        "元素力量是提瓦特大陆上的基础能力体系。",
        "安柏：训练...还不够...",
        "派蒙是旅行者的同伴，被称为应急食品。",
        "七神统治着提瓦特大陆的七个国家。",
        "骑士团负责守护蒙德城的和平与安全。",
        "冒险家协会为旅行者提供各种任务和情报。",
        "提瓦特大陆有风、岩、雷、水、火、草、冰七种元素。",
        
        # 基本解释
        "What is artificial intelligence?",
        "Artificial intelligence refers to computer systems designed to perform tasks that typically require human intelligence.",
        "Can you explain what a neural network is?",
        "A neural network is a computational model inspired by the human brain that consists of layers of interconnected nodes.",
        "What is machine learning?",
        "Machine learning is a subset of artificial intelligence that focuses on developing algorithms that allow computers to learn from data.",
        "What is deep learning?",
        "Deep learning is a subset of machine learning that uses artificial neural networks with multiple layers to extract features from data.",
        
        # 完整句子
        "This is a test sentence for APT model training.",
        "Deep learning models require a lot of data.",
        "Transformers are widely used in NLP tasks.",
        "The quick brown fox jumps over the lazy dog.",
        "I enjoy reading books in my free time.",
        "Music has the power to change our mood.",
        "The Internet has revolutionized how we access information.",
        "Learning a new language can be challenging but rewarding.",
        "Yesterday I went to the store to buy some groceries.",
        "The mountains looked beautiful against the sunset sky.",
        "She opened the window to let in some fresh air.",
        "They decided to take a vacation to the beach this summer.",
        "The professor explained the complex theory to the students.",
        "My favorite season is autumn when the leaves change color.",
        "The company announced their new product at the conference.",
        "We should meet for coffee sometime next week.",
        "The children played happily in the park all afternoon.",
        "He finished writing his novel after five years of work.",
        
        # 问答对
        "What is the capital of France?",
        "The capital of France is Paris.",
        "Who wrote Romeo and Juliet?",
        "William Shakespeare wrote Romeo and Juliet.",
        "What is photosynthesis?",
        "Photosynthesis is the process by which green plants use sunlight to synthesize foods with carbon dioxide and water.",
        "How far is the Moon from Earth?",
        "The average distance between the Earth and the Moon is about 384,400 kilometers.",
        "What is the largest ocean on Earth?",
        "The Pacific Ocean is the largest and deepest ocean on Earth.",
        
        # 常见短语
        "Thank you very much.",
        "You're welcome.",
        "How can I help you today?",
        "That's a great question.",
        "I'm not sure about that.",
        "Let me think about it.",
        "Could you please explain that again?",
        "I understand what you're saying.",
        "That's an interesting perspective.",
        "I agree with your point of view.",
        
        # 更复杂的内容
        "In recent years, large language models have demonstrated impressive capabilities in understanding and generating human language.",
        "The development of self-driving cars represents a significant advancement in artificial intelligence.",
        "Climate change is one of the most pressing challenges facing our planet today.",
        "The human brain contains approximately 86 billion neurons.",
        "Quantum computing has the potential to solve certain problems exponentially faster than classical computers.",
        "The history of art spans thousands of years, reflecting human civilization.",
        "Telescopes allow us to observe distant galaxies.",
        "Blockchain technology provides a secure way to record transactions.",
        "Photosynthesis is essential for most life on Earth.",
        "Biodiversity is crucial for maintaining healthy ecosystems.",
        
        # 连贯段落
        """
        The sun was setting behind the mountains, casting long shadows across the valley.
        Birds were returning to their nests, filling the air with their evening songs.
        A gentle breeze rustled the leaves, bringing the sweet scent of wildflowers.
        In the distance, the small town's lights twinkled like fallen stars.
        """,
        
        """
        Learning to code can be challenging but rewarding.
        It requires patience, logical thinking, and attention to detail.
        Many beginners start with simple languages like Python.
        The key to success is consistent practice and learning from mistakes.
        """,
        
        """
        The history of aviation began with the Wright brothers.
        Since then, aviation has transformed global travel.
        Modern aircraft can fly faster than the speed of sound.
        Air travel connects people across continents in just hours.
        """,
        
        # 中文内容
        "机器学习是人工智能的一个子领域，它使用数据和算法来模仿人类学习的方式。",
        "深度学习是机器学习的一种特殊形式，它使用多层神经网络处理复杂的模式。",
        "自然语言处理让计算机能够理解、分析和生成人类语言。",
        "计算机视觉使机器能够从图像和视频中获取有意义的信息。",
        "强化学习是一种让机器通过试错来学习的方法，以获得最大的奖励。",
        "大型语言模型如GPT能够生成流畅的文本，并回答各种问题。",
        "人工智能伦理关注AI发展中的道德问题和社会影响。",
        "数据科学结合了统计学、编程和领域知识来提取数据中的价值。",
        "机器学习算法可以分为监督学习、无监督学习和强化学习。",
        "神经网络是受人脑结构启发而设计的算法。"
]


def get_training_texts():
    """Load training texts from the repository datasets.

    The project ships with ``train.txt`` (英文) and ``zh_train.txt`` (中文)
    files.  When they are available we combine and clean their contents so
    the training loop never needs to reach out to Hugging Face for
    datasets.  The locally curated fallback samples are *always* appended
    to keep the multilingual prompts that were originally embedded in this
    module, ensuring the training loop benefits from the handcrafted
    dialogues even when the text files are present.
    """

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate_files = [
        os.path.join(project_root, "train.txt"),
        os.path.join(project_root, "zh_train.txt"),
    ]

    aggregated_texts = []
    for path in candidate_files:
        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                text = raw_line.strip()
                if not text:
                    continue
                text = re.sub(r"^\d+\s*[\.、:：-]\s*", "", text)
                if text:
                    aggregated_texts.append(text)

    if not aggregated_texts:
        print("未找到仓库提供的训练数据文件，使用预设训练数据。")

    aggregated_texts.extend(_FALLBACK_TRAINING_TEXTS)

    # 去重但保持顺序，避免重复样本对训练造成不必要的偏倚
    deduped_texts = list(dict.fromkeys(aggregated_texts))

    return deduped_texts

def _generate_natural_text(*args, **kwargs):
    from apt_model.generation.generator import generate_natural_text

    return generate_natural_text(*args, **kwargs)


def _evaluate_text_quality(text):
    from apt_model.generation.evaluator import evaluate_text_quality

    return evaluate_text_quality(text)


# =============================================================================
# 主训练函数
# =============================================================================
def train_model(epochs=20, batch_size=8, learning_rate=3e-5, save_path="apt_model",
                logger=None, resource_monitor=None, multimodal_config=None,
                tokenizer_type=None, language=None, texts=None, tokenizer=None):
    """训练模型的主函数"""
    # 设置随机种子
    set_seed(42)
    
    if logger:
        logger.info("开始训练模型...")
    else:
        print("\n开始训练模型...\n")
    
    # 获取训练数据
    if texts is None:
        train_texts = get_training_texts()
    else:
        train_texts = texts
    
    print(f"训练数据集大小: {len(train_texts)} 条文本")
    
    # 如果数据为空，则报错
    if len(train_texts) == 0:
        raise ValueError("训练数据为空，请确保数据文件存在或内置数据正确加载。")
    
    # 自动检测语言并选择合适的分词器
    tokenizer, detected_language = get_appropriate_tokenizer(
        train_texts, 
        tokenizer_type=tokenizer_type, 
        language=language
    )
    
    print(f"使用{detected_language}语言分词器: {type(tokenizer).__name__}")
    
    # 设置数据集和数据加载器
    class TextDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length=128):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = self.texts[idx]
            encoding = self.tokenizer.encode(
                text, 
                return_tensors="pt", 
                max_length=self.max_length, 
                truncation=True
            ).squeeze(0)
            return encoding, encoding
    
    def collate_fn(batch):
        src_ids_list, tgt_ids_list = zip(*batch)
        src_ids = torch.nn.utils.rnn.pad_sequence(
            src_ids_list, 
            batch_first=True, 
            padding_value=tokenizer.pad_token_id
        )
        tgt_ids = torch.nn.utils.rnn.pad_sequence(
            tgt_ids_list, 
            batch_first=True, 
            padding_value=tokenizer.pad_token_id
        )
        return src_ids, tgt_ids
    
    print("正在准备数据集...")
    dataset = TextDataset(train_texts, tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print("创建模型配置...")
    config = APTConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=768,
        d_ff=2048,
        num_heads=12,
        num_encoder_layers=4,
        num_decoder_layers=4,
        max_seq_len=128,
        dropout=0.2,
        epsilon=2.0,
        alpha=0.001,
        beta=0.001,
        base_lr=learning_rate,
        pad_token_id=getattr(tokenizer, "pad_token_id", 0),
        bos_token_id=getattr(tokenizer, "bos_token_id", getattr(tokenizer, "eos_token_id", 0)),
        eos_token_id=getattr(tokenizer, "eos_token_id", 0),
    )
    
    print("初始化模型...")
    model = APTLargeModel(config).to(device)
    model.train()
    
    import inspect
    #print("===== 模型 forward 方法参数 =====")
    #print(inspect.signature(model.forward))
    #print("================================")

    # 优化器和学习率调度器设置
    from apt_model.training.optimizer import create_optimizer_and_scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, learning_rate, len(dataloader), epochs
    )
    
    # 早停设置
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # 尝试使用tensorboard记录训练过程
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=f"{save_path}_logs")
        use_tensorboard = True
    except:
        use_tensorboard = False
        print("未安装tensorboard，将不使用tensorboard记录训练过程")
    
    # 保存函数
    from apt_model.training.checkpoint import save_model
    
    # 保存训练前的模型用于比较
    untrained_model = APTLargeModel(config).to(device)
    untrained_model.load_state_dict(model.state_dict())
    untrained_model.eval()
    
    global_step = 0
    train_losses = []
    best_quality_score = 0.0
    correct_tokens = 0
    total_tokens = 0
    
    print(f"开始训练，总共 {epochs} 轮...")
    
    from torch.cuda.amp import autocast, GradScaler
    
    # 在训练开始前初始化 GradScaler
    try:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    except (ImportError, AttributeError):
        # 创建一个假的 GradScaler 以保持代码兼容性
        class DummyScaler:
            def scale(self, loss):
                return loss
            def step(self, optimizer):
                optimizer.step()
            def update(self):
                pass
        scaler = DummyScaler()
        print("警告: 混合精度训练不可用，使用标准精度训练")

    # 添加梯度累积参数
    accumulation_steps = 4  # 可以根据需求调整
    
    # 主训练循环
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, batch in enumerate(progress_bar):
            try:
                if resource_monitor:
                    resource_monitor.check_resources()
                
                src_ids, tgt_ids = batch
                src_ids = src_ids.to(device)
                tgt_ids = tgt_ids.to(device)
                
                src_padding_mask = (src_ids == tokenizer.pad_token_id)  # 填充掩码 [batch, src_len]
                tgt_mask = torch.triu(torch.ones(tgt_ids.size(1), tgt_ids.size(1), device=tgt_ids.device) * float('-inf'), diagonal=1)
                
                # 只在累积周期开始时清零梯度
                if i % accumulation_steps == 0:
                    optimizer.zero_grad()
                
                # 使用更新后的 autocast 进行混合精度前向计算和损失计算
                amp_dtype = torch.bfloat16  # 或使用torch.float32
                with torch.amp.autocast('cuda'):
                    try:
                        # 在这里添加打印语句
                        import inspect
                        #print(f"Model type: {type(model)}")
                        #print(f"Model forward signature: {inspect.signature(model.forward)}")
                        
                        logits = model(src_tokens=src_ids, tgt_tokens=src_ids, src_key_padding_mask=src_padding_mask, src_mask=None)
                    except Exception as e:
                        if logger:
                            logger.error(f"前向传播出错: {e}")
                        print(f"警告: 前向传播失败: {e}，跳过当前批次")
                        continue
                    
                    if torch.isnan(logits).any():
                        print(f"警告: 第{epoch+1}轮第{i+1}批次的logits包含NaN，跳过此批次")
                        continue
                    
                    # 计算损失
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = tgt_ids[:, 1:].contiguous()
                    
                    try:
                        loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=tokenizer.pad_token_id,
                            label_smoothing=0.1
                        )
                        with torch.no_grad():
                            predictions = shift_logits.argmax(dim=-1)
                            valid_mask = shift_labels != tokenizer.pad_token_id
                            correct_tokens += (predictions.eq(shift_labels) & valid_mask).sum().item()
                            total_tokens += valid_mask.sum().item()
                        # 根据累积步骤缩放损失
                        loss = loss / accumulation_steps
                    except Exception as e:
                        if logger:
                            logger.error(f"损失计算出错: {e}")
                        print(f"警告: 损失计算失败: {e}，跳过当前批次")
                        continue
                    
                    if torch.isnan(loss).any():
                        print(f"警告: 第{epoch+1}轮第{i+1}批次发现NaN损失，跳过此批次")
                        continue
                
                # 使用 GradScaler 进行反向传播和参数更新
                try:
                    scaler.scale(loss).backward()
                except Exception as e:
                    if logger:
                        logger.error(f"反向传播出错: {e}")
                    print(f"警告: 反向传播失败: {e}，跳过当前批次")
                    optimizer.zero_grad()
                    continue
                
                # 梯度裁剪（如果需要，可以放在 scaler.step() 前后）
                try:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                except Exception as e:
                    if logger:
                        logger.warning(f"梯度裁剪出错，跳过: {e}")
                    print(f"警告: 梯度裁剪失败: {e}")
                
                # 只在累积完成后更新参数
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                    # 进行参数更新
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                
                    torch.cuda.empty_cache()  # 清理GPU缓存
                    
                    try:
                        current_lr = scheduler.get_last_lr()[0]
                        model.update_dynamic_taylor_parameters(current_lr)
                    except Exception as e:
                        if logger:
                            logger.warning(f"动态参数更新出错: {e}")
                        print(f"警告: 动态参数更新失败: {e}")
                
                total_loss += loss.item() * accumulation_steps  # 恢复实际损失
                train_losses.append(loss.item() * accumulation_steps)
                progress_bar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.6f}"})
                
                if use_tensorboard:
                    writer.add_scalar('Loss/train', loss.item() * accumulation_steps, global_step)
                    writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], global_step)
                
                global_step += 1
                
                if global_step % 50 == 0 or i == len(dataloader) - 1:
                    # 测试生成和评估代码保持不变...
                    pass
                    
            except Exception as e:
                if logger:
                    logger.error(f"处理批次 {i} 时出错: {e}")
                    logger.error(traceback.format_exc())
                print(f"批次处理错误: {e}，跳过当前批次")
                continue
        
        avg_loss = total_loss / max(1, len(dataloader))
        epoch_accuracy = (correct_tokens / total_tokens * 100.0) if total_tokens else 0.0
        print(f"Epoch {epoch+1}/{epochs} 完成, 平均损失: {avg_loss:.4f}, 训练集准确率: {epoch_accuracy:.2f}%")

        # 重置准确率统计，避免跨轮次累计
        correct_tokens = 0
        total_tokens = 0
        
        if use_tensorboard:
            writer.add_scalar('Loss/epoch', avg_loss, epoch)
            
        try:
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_model(model, tokenizer, path=save_path, config=config)
                print(f"发现新的最佳模型，已保存到 {save_path}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停: {patience} 轮没有改善，停止训练")
                    break
            
            _test_generation_after_epoch(model, tokenizer, logger, detected_language)
        except Exception as e:
            if logger:
                logger.error(f"轮次结束处理出错: {e}")
            print(f"警告: 轮次结束处理失败: {e}")
    
    if use_tensorboard:
        writer.close()
    
    print("训练完成！最终模型已保存。")
    
    try:
        _compare_model_outputs(untrained_model, model, tokenizer, detected_language)
    except Exception as e:
        if logger:
            logger.error(f"模型比较出错: {e}")
        print(f"警告: 模型比较失败: {e}")
    
    return model, tokenizer, config

def _test_generation_after_epoch(model, tokenizer, logger=None, language="en"):
    """测试每个轮次后的生成效果"""
    # 添加诊断打印
    #print("\n===== 开始诊断 _test_generation_after_epoch =====")
    #print(f"模型类型: {type(model)}")
    #print(f"模型属性: {dir(model)}")
    #print("检查是否有generate方法:", hasattr(model, 'generate'))
    #print("检查是否是APTModel的实例:", isinstance(model, APTModel))
    
    #if hasattr(model, 'generate'):
        #print("generate方法的签名:", model.generate.__code__.co_varnames)
    #print("===== 诊断结束 =====\n")
    # 根据语言选择测试提示
    if language == "zh":
        test_prompts = ["人工智能", "深度学习", "自然语言", "安柏是"]
    else:
        test_prompts = ["Hello", "What is", "The quick", "Artificial"]
        
    model.eval()
    print("\n本轮训练后的文本生成示例:")
    gen_texts = []
    for prompt in test_prompts:
        with torch.no_grad():
            gen_text, _, _, _ = _generate_natural_text(model, tokenizer, prompt, max_steps=15)
            print(f"提示: '{prompt}'")
            print(f"生成: '{gen_text}'")
            print("-" * 30)
            gen_texts.append(gen_text)
    avg_quality = sum(_evaluate_text_quality(text)[0] for text in gen_texts) / len(gen_texts)
    print(f"本轮生成文本平均质量: {avg_quality:.2f}/100")
    if avg_quality < 40:
        print("\n安柏：训练...还不够...")
    model.train()
    return avg_quality

def _compare_model_outputs(untrained_model, trained_model, tokenizer, language="en"):
    """比较训练前后的模型输出"""
    # 添加诊断打印
    #print("\n===== 开始诊断 _compare_model_outputs =====")
    #print(f"未训练模型类型: {type(untrained_model)}")
    #print(f"已训练模型类型: {type(trained_model)}")
    #print("未训练模型是否有generate方法:", hasattr(untrained_model, 'generate'))
    #print("已训练模型是否有generate方法:", hasattr(trained_model, 'generate'))
    
    # 检查generate_natural_text函数
    #import inspect
    #if 'generate_natural_text' in globals():
        #print("generate_natural_text函数签名:", inspect.signature(generate_natural_text))
    #else:
        #print("generate_natural_text函数不存在于全局空间")
    #print("===== 诊断结束 =====\n")
    print("\n====================")
    print("训练前后效果对比")
    print("====================")
    
    # 根据语言选择测试提示
    if language == "zh":
        test_prompts = [
            "人工智能是",
            "深度学习可以",
            "自然语言处理",
            "安柏是",
            "我认为"
        ]
    else:
        test_prompts = [
            "Hello, how are you",
            "What is artificial intelligence",
            "The future of technology",
            "I think that",
            "The best way to"
        ]
        
    trained_model.eval()
    untrained_model.eval()
    untrained_scores = []
    trained_scores = []
    
    for prompt in test_prompts:
        print(f"\n提示: '{prompt}'")
        with torch.no_grad():
            untrained_text, _, _, _ = _generate_natural_text(untrained_model, tokenizer, prompt, max_steps=20)
            untrained_score, untrained_feedback = _evaluate_text_quality(untrained_text)
            untrained_scores.append(untrained_score)
        print(f"未训练模型: '{untrained_text}'")
        print(f"质量评分: {untrained_score}/100 - {untrained_feedback}")
        
        with torch.no_grad():
            trained_text, _, _, _ = _generate_natural_text(trained_model, tokenizer, prompt, max_steps=20)
            trained_score, trained_feedback = _evaluate_text_quality(trained_text)
            trained_scores.append(trained_score)
        print(f"训练后模型: '{trained_text}'")
        print(f"质量评分: {trained_score}/100 - {trained_feedback}")
        print("-" * 50)
    
    avg_untrained = sum(untrained_scores) / len(untrained_scores)
    avg_trained = sum(trained_scores) / len(trained_scores)
    improvement = avg_trained - avg_untrained
    
    print(f"\n整体评估:")
    print(f"未训练模型平均质量: {avg_untrained:.2f}/100")
    print(f"训练后模型平均质量: {avg_trained:.2f}/100")
    print(f"质量提升: {improvement:.2f} 分")
    
    if avg_trained < 50:
        print("\n安柏：训练...还不够...")
    else:
        print("\n安柏：训练完成得不错！")