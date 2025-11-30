#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修改APT模型训练器以支持中文分词
"""

import os
import glob
import logging
import torch
import torch.nn.functional as F
import traceback
from contextlib import nullcontext
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

# 尝试导入rich，如果不可用则回退到tqdm
try:
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        MofNCompleteColumn,
    )
    RICH_AVAILABLE = True
except ImportError:
    from tqdm import tqdm
    RICH_AVAILABLE = False

from apt_model.utils import set_seed
from apt_model.utils import get_device, device
from apt_model.config.apt_config import APTConfig
from apt_model.modeling.apt_model import APTModel, APTLargeModel
from apt_model.generation.generator import generate_natural_text
from apt_model.generation.evaluator import evaluate_text_quality
from apt_model.config.settings_manager import settings

# 导入中文分词器相关函数
from apt_model.modeling.chinese_tokenizer_integration import (
    get_appropriate_tokenizer,
    save_tokenizer,
    is_chinese_text
)

# 导入新的codec系统
from apt_model.codecs import get_codec_for_language, list_available_codecs
from apt_model.codecs.compat import CodecTokenizerWrapper

# 导入callback系统
from apt_model.training.callbacks import (
    CallbackManager,
    create_default_callbacks,
)


# ============================================================================
# Codec系统集成
# ============================================================================

def get_tokenizer_from_codec(texts, tokenizer_type=None, language=None):
    """
    使用新的codec系统获取tokenizer

    这个函数使用新的插件化codec架构，但返回一个兼容transformers接口的tokenizer包装器。

    参数:
        texts: 文本列表（用于语言检测）
        tokenizer_type: 指定的分词器类型（可选）
        language: 指定的语言（可选，'en', 'zh', 'ja'等）

    返回:
        (tokenizer_wrapper, detected_language): tokenizer包装器和检测到的语言
    """
    import logging
    logger = logging.getLogger('apt_model.codec')

    # 如果未指定语言，使用旧的语言检测逻辑
    if language is None:
        from apt_model.modeling.chinese_tokenizer_integration import detect_language
        language = detect_language(texts)
        logger.info(f"自动检测语言: {language}")

    # 映射tokenizer_type到codec名称
    codec_name = None
    if tokenizer_type:
        type_to_codec = {
            'gpt2': 'en_gpt2',
            'chinese-char': 'zh_char',
            'chinese-word': 'zh_char',
            'ja-mecab': 'ja_mecab',
        }
        codec_name = type_to_codec.get(tokenizer_type)

    # 尝试获取codec
    try:
        codec = get_codec_for_language(language, prefer=codec_name)

        if codec is None:
            logger.warning(f"未找到语言 '{language}' 的codec，回退到旧系统")
            return get_appropriate_tokenizer(texts, tokenizer_type, language)

        logger.info(f"使用codec: {codec.name} (语言: {language})")

        # 包装为tokenizer接口
        tokenizer_wrapper = CodecTokenizerWrapper(codec)

        return tokenizer_wrapper, language

    except Exception as e:
        logger.warning(f"Codec系统失败 ({e})，回退到旧分词器系统")
        return get_appropriate_tokenizer(texts, tokenizer_type, language)


# ============================================================================
# Logger设置
# ============================================================================

def setup_training_logger(name="APT-Trainer", log_file=None):
    """
    设置训练日志记录器

    参数:
        name: Logger名称
        log_file: 可选的日志文件路径

    返回:
        配置好的logger
    """
    logger = logging.getLogger(name)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # 格式化器 - 带时间戳和模块名
    formatter = logging.Formatter(
        '%(asctime)s [%(name)s] %(message)s',
        datefmt='%m-%d %H:%M:%S'
    )

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # 文件处理器（可选）
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    return logger

# 全局logger实例
_training_logger = None

def get_training_logger():
    """获取训练logger实例"""
    global _training_logger
    if _training_logger is None:
        _training_logger = setup_training_logger()
    return _training_logger


# ============================================================================
# Debug输出辅助函数（保持向后兼容）
# ============================================================================

def debug_print(*args, **kwargs):
    """仅在Debug模式下打印信息"""
    if settings.get_debug_enabled():
        logger = get_training_logger()
        message = ' '.join(str(arg) for arg in args)
        logger.debug(message)

def info_print(*args, **kwargs):
    """始终打印的关键信息（非Debug模式也显示）"""
    logger = get_training_logger()
    message = ' '.join(str(arg) for arg in args)
    logger.info(message)


# ============================================================================
# 进度条创建函数
# ============================================================================

def create_training_progress_bar(dataloader, epoch, total_epochs):
    """
    创建训练进度条（优先使用rich，回退到tqdm）

    参数:
        dataloader: 数据加载器
        epoch: 当前epoch (从0开始)
        total_epochs: 总epoch数

    返回:
        进度条对象和上下文管理器
    """
    if RICH_AVAILABLE:
        # 使用rich创建漂亮的进度条
        progress = Progress(
            SpinnerColumn(),  # 旋转动画
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            MofNCompleteColumn(),  # 当前/总数
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            TextColumn("[cyan]{task.fields[loss]}", justify="right"),
            TextColumn("[green]{task.fields[lr]}", justify="right"),
        )
        task_id = progress.add_task(
            f"Epoch {epoch+1}/{total_epochs}",
            total=len(dataloader),
            loss="Loss: -.----",
            lr="LR: -.------"
        )
        return progress, task_id
    else:
        # 回退到tqdm
        from tqdm import tqdm
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}")
        return progress_bar, None


# ============================================================================
# 数据集类定义
# ============================================================================

class TextDataset(Dataset):
    """
    文本数据集类

    将文本列表转换为模型可用的token序列
    """
    def __init__(self, texts, tokenizer, max_length=128):
        """
        初始化数据集

        参数:
            texts: 文本列表
            tokenizer: 分词器
            max_length: 最大序列长度
        """
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


def create_collate_fn(tokenizer):
    """
    创建批次整理函数

    参数:
        tokenizer: 分词器，用于获取pad_token_id

    返回:
        collate_fn: 批次整理函数
    """
    def collate_fn(batch):
        """整理批次数据，进行填充"""
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

    return collate_fn


class DummyGradScaler:
    """
    兼容性GradScaler类

    当CUDA不可用或不支持混合精度时使用
    提供与torch.cuda.amp.GradScaler相同的接口
    """
    def scale(self, loss):
        """不进行缩放，直接返回损失"""
        return loss

    def step(self, optimizer):
        """直接执行优化器步骤"""
        optimizer.step()

    def update(self):
        """空操作"""
        pass


# ============================================================================
# 辅助函数
# ============================================================================

def _log_message(logger, message, level="info"):
    """
    统一的日志记录函数

    参数:
        logger: 日志记录器（可为None）
        message: 日志消息
        level: 日志级别 ("info", "warning", "error")
    """
    if logger:
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
    else:
        print(message)

def get_training_texts():
    """
    获取训练文本数据。如果在项目根目录下存在 "train.txt" 文件，则读取文件，否则返回内置预设数据。
    """
    import os
    # 获取项目根目录（假设代码文件在项目子目录中）
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_file = os.path.join(script_dir, "train.txt")
    
    print("检查训练数据文件路径：", train_file)
    
    if os.path.exists(train_file):
        with open(train_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        if texts:
            return texts
        else:
            print("训练数据文件 'train.txt' 为空，使用预设训练数据。")
    else:
        print("未找到训练数据文件 'train.txt'，使用预设训练数据。")
    
    # 预设训练数据集
    return [
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

# =============================================================================
# 训练辅助函数
# =============================================================================

def _setup_training_data(train_texts, tokenizer, batch_size):
    """
    设置训练数据和DataLoader

    参数:
        train_texts: 训练文本列表
        tokenizer: 分词器
        batch_size: 批次大小

    返回:
        dataloader: DataLoader实例
    """
    debug_print("正在准备数据集...")
    dataset = TextDataset(train_texts, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=create_collate_fn(tokenizer),
        pin_memory=True
    )
    return dataloader


def _setup_model_and_optimizer(tokenizer, learning_rate, dataloader, epochs):
    """
    设置模型、优化器和调度器

    参数:
        tokenizer: 分词器
        learning_rate: 学习率
        dataloader: DataLoader实例
        epochs: 训练轮数

    返回:
        model: APT模型
        config: 模型配置
        optimizer: 优化器
        scheduler: 学习率调度器
    """
    debug_print("创建模型配置...")
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
        base_lr=learning_rate
    )

    debug_print("初始化模型...")
    model = APTLargeModel(config).to(device)
    model.train()

    # 优化器和学习率调度器设置
    from apt_model.training.optimizer import create_optimizer_and_scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(
        model, learning_rate, len(dataloader), epochs
    )

    return model, config, optimizer, scheduler


def _setup_grad_scaler():
    """
    设置梯度缩放器（混合精度训练）

    返回:
        scaler: 梯度缩放器实例
    """
    try:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        debug_print("混合精度训练已启用")
    except (ImportError, AttributeError):
        scaler = DummyGradScaler()
        debug_print("警告: 混合精度训练不可用，使用标准精度训练")

    return scaler


def _setup_tensorboard(save_path):
    """
    设置tensorboard记录器

    参数:
        save_path: 保存路径

    返回:
        writer: SummaryWriter实例（如果可用）
        use_tensorboard: 是否使用tensorboard
    """
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=f"{save_path}_logs")
        debug_print("Tensorboard记录已启用")
        return writer, True
    except:
        debug_print("未安装tensorboard，将不使用tensorboard记录训练过程")
        return None, False


def _process_batch(model, batch, optimizer, scaler, tokenizer, accumulation_steps,
                   batch_idx, logger, resource_monitor, gradient_monitor=None, global_step=0):
    """
    处理单个批次的训练

    参数:
        model: 模型
        batch: 批次数据
        optimizer: 优化器
        scaler: 梯度缩放器
        tokenizer: 分词器
        accumulation_steps: 梯度累积步数
        batch_idx: 批次索引
        logger: 日志记录器
        resource_monitor: 资源监视器
        gradient_monitor: 梯度监控器（可选）
        global_step: 全局训练步数（用于梯度监控）

    返回:
        loss_value: 损失值（失败返回None）
        should_update: 是否应该更新参数
    """
    try:
        if resource_monitor:
            resource_monitor.check_resources()

        src_ids, tgt_ids = batch
        src_ids = src_ids.to(device)
        tgt_ids = tgt_ids.to(device)

        src_padding_mask = (src_ids == tokenizer.pad_token_id)

        # 只在累积周期开始时清零梯度
        if batch_idx % accumulation_steps == 0:
            optimizer.zero_grad()

        # 混合精度前向计算
        with torch.amp.autocast('cuda'):
            try:
                logits = model(
                    src_tokens=src_ids,
                    tgt_tokens=src_ids,
                    src_key_padding_mask=src_padding_mask,
                    src_mask=None
                )
            except Exception as e:
                _log_message(logger, f"前向传播出错: {e}", "error")
                debug_print(f"警告: 前向传播失败: {e}，跳过当前批次")
                return None, False

            if torch.isnan(logits).any():
                debug_print(f"警告: 批次{batch_idx+1}的logits包含NaN，跳过此批次")
                return None, False

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
                loss = loss / accumulation_steps
            except Exception as e:
                _log_message(logger, f"损失计算出错: {e}", "error")
                debug_print(f"警告: 损失计算失败: {e}，跳过当前批次")
                return None, False

            if torch.isnan(loss).any():
                debug_print(f"警告: 批次{batch_idx+1}发现NaN损失，跳过此批次")
                return None, False

        # 反向传播
        try:
            scaler.scale(loss).backward()
        except Exception as e:
            _log_message(logger, f"反向传播出错: {e}", "error")
            debug_print(f"警告: 反向传播失败: {e}，跳过当前批次")
            optimizer.zero_grad()
            return None, False

        # 梯度监控（如果启用）
        if gradient_monitor is not None:
            try:
                # 检查梯度流（梯度消失/爆炸检测）
                gradients, issues = gradient_monitor.check_gradient_flow()
                if issues:
                    for issue in issues[:3]:  # 只显示前3个问题
                        debug_print(f"  {issue}")
                    if logger:
                        logger.warning(f"Step {global_step}: 发现 {len(issues)} 个梯度问题")

                # 记录梯度范数
                total_norm = gradient_monitor.log_gradient_norms(global_step)

                # 检测梯度异常
                anomalies = gradient_monitor.detect_gradient_anomalies()
                if anomalies:
                    _log_message(logger, f"Step {global_step}: 检测到梯度异常 - {anomalies}", "warning")
                    debug_print(f"⚠️  梯度异常: {anomalies}")
            except Exception as e:
                debug_print(f"梯度监控失败: {e}")

        # 梯度裁剪
        try:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        except Exception as e:
            _log_message(logger, f"梯度裁剪出错: {e}", "warning")
            debug_print(f"警告: 梯度裁剪失败: {e}")

        loss_value = loss.item() * accumulation_steps
        return loss_value, True

    except Exception as e:
        _log_message(logger, f"处理批次 {batch_idx} 时出错: {e}", "error")
        debug_print(f"批次处理错误: {e}，跳过当前批次")
        return None, False


# =============================================================================
# 主训练函数
# =============================================================================

def train_model(epochs=20, batch_size=8, learning_rate=3e-5, save_path="apt_model",
                logger=None, resource_monitor=None, multimodal_config=None,
                tokenizer_type=None, language=None, texts=None, tokenizer=None,
                checkpoint_dir="./outputs", resume_from=None, temp_checkpoint_freq=100,
                enable_gradient_monitoring=False):
    """
    训练模型的主函数

    参数:
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        save_path: 模型保存路径（已弃用，建议使用checkpoint_dir）
        logger: 日志记录器
        resource_monitor: 资源监视器
        multimodal_config: 多模态配置（未使用）
        tokenizer_type: 分词器类型
        language: 语言类型
        texts: 训练文本（None则使用默认数据）
        tokenizer: 分词器（None则自动选择）
        checkpoint_dir: checkpoint保存目录（相对路径，可迁移），默认"./outputs"
        resume_from: 恢复训练的checkpoint路径，可选
        temp_checkpoint_freq: 临时checkpoint保存频率（每N步），默认100
        enable_gradient_monitoring: 启用梯度监控（调试/分析用），默认False

    返回:
        model: 训练后的模型
        tokenizer: 使用的分词器
        config: 模型配置
    """
    # 设置随机种子
    set_seed(42)

    _log_message(logger, "开始训练模型...")

    # 获取训练数据
    if texts is None:
        train_texts = get_training_texts()
    else:
        train_texts = texts

    info_print(f"训练数据集大小: {len(train_texts)} 条文本")

    if len(train_texts) == 0:
        raise ValueError("训练数据为空，请确保数据文件存在或内置数据正确加载。")

    # 自动检测语言并选择合适的分词器
    if tokenizer is None:
        # 优先使用新的codec系统
        tokenizer, detected_language = get_tokenizer_from_codec(
            train_texts,
            tokenizer_type=tokenizer_type,
            language=language
        )
        info_print(f"使用{detected_language}语言分词器: {type(tokenizer).__name__}")
    else:
        detected_language = language or "en"
        debug_print(f"使用提供的分词器: {type(tokenizer).__name__}")

    # 设置数据和模型
    dataloader = _setup_training_data(train_texts, tokenizer, batch_size)
    model, config, optimizer, scheduler = _setup_model_and_optimizer(
        tokenizer, learning_rate, dataloader, epochs
    )

    # 设置训练工具
    scaler = _setup_grad_scaler()
    writer, use_tensorboard = _setup_tensorboard(save_path)

    # 保存训练前的模型用于比较
    untrained_model = APTLargeModel(config).to(device)
    untrained_model.load_state_dict(model.state_dict())
    untrained_model.eval()

    # 初始化梯度监控器（如果启用）
    gradient_monitor = None
    if enable_gradient_monitoring:
        from apt_model.training.gradient_monitor import GradientMonitor
        from pathlib import Path
        gradient_export_dir = Path(checkpoint_dir) / "gradient_monitor"
        gradient_export_dir.mkdir(parents=True, exist_ok=True)
        gradient_monitor = GradientMonitor(
            model,
            logger=logger,
            export_dir=gradient_export_dir
        )
        info_print(f"✅ 梯度监控已启用，报告将保存到: {gradient_export_dir}")

    # 早停设置
    best_loss = float('inf')
    patience = 5
    patience_counter = 0

    # 训练状态
    global_step = 0
    train_losses = []
    best_quality_score = 0.0
    
    print(f"开始训练，总共 {epochs} 轮...")
    
    # 创建一个假的 GradScaler 以保持代码兼容性
    class DummyScaler:
        def scale(self, loss):
            return loss

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass

    # 在训练开始前初始化 GradScaler
    if torch.cuda.is_available():
        try:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
            autocast_context = torch.cuda.amp.autocast
        except (ImportError, AttributeError):
            scaler = DummyScaler()
            autocast_context = nullcontext
            print("警告: CUDA AMP 不可用，使用标准精度训练")
    else:
        scaler = DummyScaler()
        autocast_context = nullcontext
        print("警告: 未检测到GPU，使用标准精度训练")

    # 检查是否存在checkpoint（用于恢复训练）
    checkpoint_dir = os.path.join(save_path, "checkpoints")
    has_checkpoints = False
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        if checkpoint_files:
            has_checkpoints = True
            logger = get_training_logger()
            logger.info(f"发现 {len(checkpoint_files)} 个已有checkpoint")
            logger.info(f"最新checkpoint: {sorted(checkpoint_files)[-1]}")
            logger.info("提示: 可以使用CheckpointManager.load_checkpoint()恢复训练")
            # TODO: 实现自动恢复功能（将在checkpoint改进中完成）

    info_print(f"开始训练，总共 {epochs} 轮...")

    # 梯度累积参数
    accumulation_steps = 4

    # 导入必要的函数
    from apt_model.training.checkpoint import save_model, CheckpointManager

    # 初始化CheckpointManager（使用相对路径，可迁移）
    checkpoint_mgr = CheckpointManager(
        save_dir=checkpoint_dir,
        model_name="apt_model",
        save_freq=1,  # 每个epoch保存
        logger=logger
    )
    info_print(f"Checkpoint将保存到: {checkpoint_dir}/checkpoints/")

    # 创建temp目录用于临时checkpoint
    temp_dir = os.path.join(".cache", "temp")
    os.makedirs(temp_dir, exist_ok=True)
    info_print(f"临时checkpoint将保存到: {temp_dir}/")

    # 初始化训练状态
    start_epoch = 0
    resume_global_step = 0
    resume_loss_history = []

    # 如果需要恢复训练
    if resume_from:
        try:
            info_print(f"从checkpoint恢复训练: {resume_from}")
            start_epoch, resume_global_step, resume_loss_history, resume_metrics = checkpoint_mgr.load_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_path=resume_from
            )
            # 恢复后从下一个epoch开始
            start_epoch += 1
            global_step = resume_global_step
            train_losses = resume_loss_history.copy()
            info_print(f"成功恢复训练: 从Epoch {start_epoch}继续, global_step={global_step}")
            if resume_metrics:
                info_print(f"恢复的指标: {resume_metrics}")
        except Exception as e:
            _log_message(logger, f"恢复checkpoint失败: {e}", "error")
            info_print(f"警告: 无法恢复checkpoint，从头开始训练")
            import traceback
            debug_print(f"恢复失败详情: {traceback.format_exc()}")

    # 初始化callback系统
    modules = {}
    # 尝试提取模型中的可调度模块
    if hasattr(model, 'moe_layer'):
        modules['moe'] = model.moe_layer
    if hasattr(model, 'align_layer'):
        modules['align'] = model.align_layer
    if hasattr(model, 'voter'):
        modules['voter'] = model.voter
    if hasattr(model, 'router'):
        modules['router'] = model.router

    total_steps = epochs * len(dataloader)
    callbacks = create_default_callbacks(config, modules, epochs, total_steps,
                                        use_rich_progress=False)  # Set to True for rich display
    callback_manager = CallbackManager(callbacks)

    # 触发训练开始回调
    callback_manager.trigger('on_train_begin', model=model, config=config, total_epochs=epochs)

    # 主训练循环（从start_epoch开始，支持恢复训练）
    for epoch in range(start_epoch, epochs):
        # 触发epoch开始回调（传递dataloader给ProgressCallback）
        callback_manager.trigger('on_epoch_begin', epoch=epoch, dataloader=dataloader)

        total_loss = 0
        # 注意：进度条现在由ProgressCallback管理，这里保留tqdm作为后备
        # 如果想完全使用ProgressCallback，可以注释掉下面这行
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", disable=True)

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
                with autocast_context() if autocast_context is not nullcontext else nullcontext():
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
                
                    if torch.cuda.is_available():
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

            # 累积损失
            total_loss += loss_value
            train_losses.append(loss_value)
            progress_bar.set_postfix({
                "loss": f"{loss_value:.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.6f}"
            })

            # 获取GPU使用率（如果可用）
            gpu_usage = None
            if torch.cuda.is_available():
                try:
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    gpu_usage = (allocated / total) * 100
                except:
                    pass

            # 触发batch结束回调（传递给ProgressCallback）
            callback_manager.trigger('on_batch_end', batch_idx=i, loss=loss_value,
                                    lr=scheduler.get_last_lr()[0], gpu_usage=gpu_usage,
                                    epoch=epoch)

            # 只在累积完成后更新参数
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # 触发optimization step回调
                callback_manager.trigger('on_step', step=global_step)

                torch.cuda.empty_cache()

                # 更新动态Taylor参数
                try:
                    current_lr = scheduler.get_last_lr()[0]
                    model.update_dynamic_taylor_parameters(current_lr)
                except Exception as e:
                    _log_message(logger, f"动态参数更新出错: {e}", "warning")
                    debug_print(f"警告: 动态参数更新失败: {e}")

            # 记录到tensorboard
            if use_tensorboard:
                writer.add_scalar('Loss/train', loss_value, global_step)
                writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], global_step)

            global_step += 1

            # 每N步保存临时checkpoint（用于崩溃恢复）
            if temp_checkpoint_freq > 0 and global_step % temp_checkpoint_freq == 0:
                try:
                    temp_checkpoint_path = os.path.join(temp_dir, f"temp_epoch{epoch}_step{global_step}.pt")
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'batch_idx': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_losses': train_losses,
                    }, temp_checkpoint_path)
                    debug_print(f"临时checkpoint已保存: {temp_checkpoint_path}")
                except Exception as e:
                    _log_message(logger, f"保存临时checkpoint失败: {e}", "warning")
                    debug_print(f"警告: 临时checkpoint保存失败: {e}")

        # Epoch结束处理
        avg_loss = total_loss / max(1, len(dataloader))
        info_print(f"Epoch {epoch+1}/{epochs} 完成, 平均损失: {avg_loss:.4f}")

        if use_tensorboard:
            writer.add_scalar('Loss/epoch', avg_loss, epoch)

        # 触发epoch结束回调
        callback_manager.trigger('on_epoch_end', epoch=epoch, metrics={'avg_loss': avg_loss})

        # 保存checkpoint（每个epoch）
        try:
            is_best = avg_loss < best_loss
            if is_best:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # 使用CheckpointManager保存完整训练状态
            checkpoint_path = checkpoint_mgr.save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                loss_history=train_losses,
                metrics={'avg_loss': avg_loss, 'best_loss': best_loss},
                tokenizer=tokenizer,
                config=config,
                is_best=is_best
            )
            info_print(f"Checkpoint已保存: {checkpoint_path}")
            if is_best:
                info_print(f"✨ 发现新的最佳模型! 损失: {best_loss:.4f}")

            # 清理temp文件夹（epoch结束后）
            try:
                temp_files = glob.glob(os.path.join(temp_dir, "temp_*.pt"))
                for temp_file in temp_files:
                    os.remove(temp_file)
                debug_print(f"已清理 {len(temp_files)} 个临时checkpoint文件")
            except Exception as e:
                debug_print(f"清理临时文件失败: {e}")

            # 早停检查
            if patience_counter >= patience:
                info_print(f"早停: {patience} 轮没有改善，停止训练")
                break

            # 测试生成效果（仅在Debug模式下显示详细信息）
            if settings.get_debug_enabled():
                _test_generation_after_epoch(model, tokenizer, logger, detected_language)
        except Exception as e:
            _log_message(logger, f"轮次结束处理出错: {e}", "error")
            debug_print(f"警告: 轮次结束处理失败: {e}")

    # 训练结束
    if use_tensorboard:
        writer.close()

    info_print("训练完成！最终模型已保存。")

    # 模型对比（仅在Debug模式下）
    if settings.get_debug_enabled():
        try:
            _compare_model_outputs(untrained_model, model, tokenizer, detected_language)
        except Exception as e:
            _log_message(logger, f"模型比较出错: {e}", "error")
            debug_print(f"警告: 模型比较失败: {e}")

    # 生成梯度监控报告（如果启用）
    if gradient_monitor is not None:
        try:
            info_print("\n生成梯度监控报告...")
            reports = gradient_monitor.generate_all_reports()
            info_print(f"✅ 梯度监控报告已生成:")
            for report_type, report_path in reports.items():
                info_print(f"  - {report_type}: {report_path}")

            # 🔮 WebUI/API伏笔: 导出JSON数据供未来使用
            webui_data = gradient_monitor.export_for_webui()
            info_print(f"  - WebUI数据已导出 (未来API可用): {len(webui_data['gradient_timeline'])} 个梯度记录")
        except Exception as e:
            _log_message(logger, f"生成梯度监控报告失败: {e}", "warning")
            debug_print(f"警告: 梯度监控报告生成失败: {e}")

    # 触发训练结束回调
    callback_manager.trigger('on_train_end', model=model, config=config)

    return model, tokenizer, config

def _test_generation_after_epoch(model, tokenizer, logger=None, language="en"):
    """
    测试每个轮次后的生成效果

    参数:
        model: 模型
        tokenizer: 分词器
        logger: 日志记录器
        language: 语言类型

    返回:
        avg_quality: 平均质量分数
    """
    # 根据语言选择测试提示
    if language == "zh":
        test_prompts = ["人工智能", "深度学习", "自然语言", "安柏是"]
    else:
        test_prompts = ["Hello", "What is", "The quick", "Artificial"]

    model.eval()
    debug_print("\n本轮训练后的文本生成示例:")

    gen_texts = []
    for prompt in test_prompts:
        with torch.no_grad():
            gen_text, _, _, _ = generate_natural_text(model, tokenizer, prompt, max_steps=15)
            debug_print(f"提示: '{prompt}'")
            debug_print(f"生成: '{gen_text}'")
            debug_print("-" * 30)
            gen_texts.append(gen_text)

    avg_quality = sum(evaluate_text_quality(text)[0] for text in gen_texts) / len(gen_texts)
    debug_print(f"本轮生成文本平均质量: {avg_quality:.2f}/100")

    if avg_quality < 40:
        debug_print("\n安柏：训练...还不够...")

    model.train()
    return avg_quality

def _compare_model_outputs(untrained_model, trained_model, tokenizer, language="en"):
    """
    比较训练前后的模型输出

    参数:
        untrained_model: 未训练的模型
        trained_model: 训练后的模型
        tokenizer: 分词器
        language: 语言类型
    """
    info_print("\n====================")
    info_print("训练前后效果对比")
    info_print("====================")

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
        info_print(f"\n提示: '{prompt}'")

        with torch.no_grad():
            untrained_text, _, _, _ = generate_natural_text(
                untrained_model, tokenizer, prompt, max_steps=20
            )
            untrained_score, untrained_feedback = evaluate_text_quality(untrained_text)
            untrained_scores.append(untrained_score)

        info_print(f"未训练模型: '{untrained_text}'")
        info_print(f"质量评分: {untrained_score}/100 - {untrained_feedback}")

        with torch.no_grad():
            trained_text, _, _, _ = generate_natural_text(
                trained_model, tokenizer, prompt, max_steps=20
            )
            trained_score, trained_feedback = evaluate_text_quality(trained_text)
            trained_scores.append(trained_score)

        info_print(f"训练后模型: '{trained_text}'")
        info_print(f"质量评分: {trained_score}/100 - {trained_feedback}")
        info_print("-" * 50)

    avg_untrained = sum(untrained_scores) / len(untrained_scores)
    avg_trained = sum(trained_scores) / len(trained_scores)
    improvement = avg_trained - avg_untrained

    info_print(f"\n整体评估:")
    info_print(f"未训练模型平均质量: {avg_untrained:.2f}/100")
    info_print(f"训练后模型平均质量: {avg_trained:.2f}/100")
    info_print(f"质量提升: {improvement:.2f} 分")

    if avg_trained < 50:
        info_print("\n安柏：训练...还不够...")
    else:
        info_print("\n安柏：训练完成得不错！")