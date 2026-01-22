#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
APT Model Data Loading Module
处理各种数据集的加载、预处理和批处理功能
"""

import os
import json
import random
from typing import List, Dict, Tuple, Optional, Union, Callable

from apt_model.utils.fake_torch import get_torch
torch = get_torch()
from apt_model.utils.fake_torch import get_torch
torch = get_torch()
Dataset = torch.utils.data.Dataset
DataLoader = torch.utils.data.DataLoader
pad_sequence = torch.nn.utils.rnn.pad_sequence
import numpy as np
from tqdm import tqdm

class TextDataset(Dataset):
    """
    基础文本数据集类，用于训练APT模型
    """
    def __init__(self, texts, tokenizer, max_length=128, return_tensors=True, 
                 truncation=True, preprocessing_fn=None):
        """
        初始化文本数据集
        
        参数:
            texts (List[str]): 文本列表
            tokenizer: 分词器，需要支持encode方法
            max_length (int): 最大序列长度
            return_tensors (bool): 是否返回张量
            truncation (bool): 是否截断过长文本
            preprocessing_fn (Callable, optional): 文本预处理函数
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tensors = return_tensors
        self.truncation = truncation
        self.preprocessing_fn = preprocessing_fn
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # 应用预处理（如果有）
        if self.preprocessing_fn:
            text = self.preprocessing_fn(text)
            
        # 编码文本
        if self.return_tensors:
            encoding = self.tokenizer.encode(
                text, 
                return_tensors="pt", 
                max_length=self.max_length, 
                truncation=self.truncation
            ).squeeze(0)
        else:
            encoding = self.tokenizer.encode(
                text, 
                max_length=self.max_length, 
                truncation=self.truncation
            )
            
        # 对于自回归训练，输入和目标相同
        return encoding, encoding

class PairedTextDataset(Dataset):
    """
    配对文本数据集类，用于序列到序列训练
    """
    def __init__(self, source_texts, target_texts, tokenizer, max_source_length=128, 
                 max_target_length=128, return_tensors=True, truncation=True,
                 source_preprocessing_fn=None, target_preprocessing_fn=None):
        """
        初始化配对文本数据集
        
        参数:
            source_texts (List[str]): 源文本列表
            target_texts (List[str]): 目标文本列表
            tokenizer: 分词器，需要支持encode方法
            max_source_length (int): 最大源序列长度
            max_target_length (int): 最大目标序列长度
            return_tensors (bool): 是否返回张量
            truncation (bool): 是否截断过长文本
            source_preprocessing_fn (Callable, optional): 源文本预处理函数
            target_preprocessing_fn (Callable, optional): 目标文本预处理函数
        """
        assert len(source_texts) == len(target_texts), "源文本和目标文本数量必须相同"
        
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.return_tensors = return_tensors
        self.truncation = truncation
        self.source_preprocessing_fn = source_preprocessing_fn
        self.target_preprocessing_fn = target_preprocessing_fn
    
    def __len__(self):
        return len(self.source_texts)
    
    def __getitem__(self, idx):
        source_text = self.source_texts[idx]
        target_text = self.target_texts[idx]
        
        # 应用预处理（如果有）
        if self.source_preprocessing_fn:
            source_text = self.source_preprocessing_fn(source_text)
        
        if self.target_preprocessing_fn:
            target_text = self.target_preprocessing_fn(target_text)
            
        # 编码文本
        if self.return_tensors:
            source_encoding = self.tokenizer.encode(
                source_text, 
                return_tensors="pt", 
                max_length=self.max_source_length, 
                truncation=self.truncation
            ).squeeze(0)
            
            target_encoding = self.tokenizer.encode(
                target_text, 
                return_tensors="pt", 
                max_length=self.max_target_length, 
                truncation=self.truncation
            ).squeeze(0)
        else:
            source_encoding = self.tokenizer.encode(
                source_text, 
                max_length=self.max_source_length, 
                truncation=self.truncation
            )
            
            target_encoding = self.tokenizer.encode(
                target_text, 
                max_length=self.max_target_length, 
                truncation=self.truncation
            )
            
        return source_encoding, target_encoding

class MultimodalDataset(Dataset):
    """
    多模态数据集类，用于处理文本与其他模态（图像、音频等）结合的数据
    """
    def __init__(self, text_data, image_paths=None, audio_paths=None, 
                 tokenizer=None, image_processor=None, audio_processor=None,
                 max_text_length=128, return_tensors=True, truncation=True):
        """
        初始化多模态数据集
        
        参数:
            text_data (List[str]): 文本数据列表
            image_paths (List[str], optional): 图像路径列表
            audio_paths (List[str], optional): 音频路径列表
            tokenizer: 文本分词器
            image_processor: 图像处理器
            audio_processor: 音频处理器
            max_text_length (int): 最大文本长度
            return_tensors (bool): 是否返回张量
            truncation (bool): 是否截断过长文本
        """
        self.text_data = text_data
        self.image_paths = image_paths
        self.audio_paths = audio_paths
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.audio_processor = audio_processor
        self.max_text_length = max_text_length
        self.return_tensors = return_tensors
        self.truncation = truncation
        
        # 确保数据长度一致（如果存在相应模态）
        if image_paths:
            assert len(text_data) == len(image_paths), "文本和图像数量必须相同"
            
        if audio_paths:
            assert len(text_data) == len(audio_paths), "文本和音频数量必须相同"
        
        # 检查数据格式
        self.has_images = image_paths is not None and len(image_paths) > 0
        self.has_audio = audio_paths is not None and len(audio_paths) > 0
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.text_data)
    
    def __getitem__(self, idx):
        """获取数据样本"""
        result = {}
        
        # 处理文本
        if self.tokenizer:
            text = self.text_data[idx]
            if self.return_tensors:
                text_encoding = self.tokenizer.encode(
                    text, 
                    return_tensors="pt", 
                    max_length=self.max_text_length, 
                    truncation=self.truncation
                ).squeeze(0)
            else:
                text_encoding = self.tokenizer.encode(
                    text, 
                    max_length=self.max_text_length, 
                    truncation=self.truncation
                )
            result["text"] = text_encoding
        
        # 处理图像（如果有）
        if self.has_images and self.image_processor:
            image_path = self.image_paths[idx]
            if image_path is not None:
                try:
                    from PIL import Image
                    # 使用上下文管理器确保文件句柄正确关闭
                    with Image.open(image_path) as image:
                        image = image.convert("RGB")
                        image_encoding = self.image_processor(image)
                        result["image"] = image_encoding
                except Exception as e:
                    print(f"无法加载图像 {image_path}: {e}")
                    result["image"] = None
            else:
                result["image"] = None
        
        # 处理音频（如果有）
        if self.has_audio and self.audio_processor:
            audio_path = self.audio_paths[idx]
            if audio_path is not None:
                try:
                    import torchaudio
                    audio, sample_rate = torchaudio.load(audio_path)
                    audio_encoding = self.audio_processor(audio, sample_rate)
                    result["audio"] = audio_encoding
                except Exception as e:
                    print(f"无法加载音频 {audio_path}: {e}")
                    result["audio"] = None
            else:
                result["audio"] = None
        
        return result

def pad_sequence_with_attention_mask(sequences, padding_value=0):
    """
    对序列进行填充并创建注意力掩码
    
    参数:
        sequences: 序列列表
        padding_value: 填充值
        
    返回:
        tuple: (填充后的序列, 注意力掩码)
    """
    # 填充序列
    padded_sequences = pad_sequence(
        sequences, 
        batch_first=True, 
        padding_value=padding_value
    )
    
    # 创建注意力掩码 (1表示需要关注的位置，0表示填充位置)
    attention_mask = (padded_sequences != padding_value).long()
    
    return padded_sequences, attention_mask

def text_collate_fn(batch, pad_token_id=0):
    """
    文本数据的整理函数，用于处理批次中的序列
    
    参数:
        batch: 批次数据，每项为(src_ids, tgt_ids)
        pad_token_id (int): 填充标记ID
        
    返回:
        tuple: (padded_src_ids, padded_tgt_ids)
    """
    src_ids_list, tgt_ids_list = zip(*batch)
    
    # 对序列进行填充
    src_ids, src_mask = pad_sequence_with_attention_mask(
        src_ids_list, 
        padding_value=pad_token_id
    )
    tgt_ids, tgt_mask = pad_sequence_with_attention_mask(
        tgt_ids_list, 
        padding_value=pad_token_id
    )
    
    return {
        "src_ids": src_ids,
        "src_mask": src_mask,
        "tgt_ids": tgt_ids,
        "tgt_mask": tgt_mask
    }

def multimodal_collate_fn(batch, pad_token_id=0):
    """
    多模态数据的整理函数
    
    参数:
        batch: 批次数据，每项为dict，包含不同模态的数据
        pad_token_id (int): 填充标记ID
    
    返回:
        dict: 整理后的批次数据，包含不同模态
    """
    result = {}
    
    # 处理文本
    if "text" in batch[0]:
        text_list = [item["text"] for item in batch]
        text_padded, text_mask = pad_sequence_with_attention_mask(
            text_list, 
            padding_value=pad_token_id
        )
        result["text_input_ids"] = text_padded
        result["text_attention_mask"] = text_mask
    
    # 处理图像
    if "image" in batch[0] and any(item["image"] is not None for item in batch):
        # 过滤掉None值
        valid_images = [item["image"] for item in batch if item["image"] is not None]
        if valid_images:
            # 假设图像已经预处理为相同大小的张量
            result["images"] = torch.stack(valid_images)
    
    # 处理音频
    if "audio" in batch[0] and any(item["audio"] is not None for item in batch):
        # 过滤掉None值
        valid_audios = [item["audio"] for item in batch if item["audio"] is not None]
        if valid_audios:
            # 音频长度可能不同，需要填充
            max_len = max([audio.shape[1] for audio in valid_audios])
            padded_audio = []
            
            for audio in valid_audios:
                if audio.shape[1] < max_len:
                    # 填充
                    padding = torch.zeros(audio.shape[0], max_len - audio.shape[1], 
                                         dtype=audio.dtype, device=audio.device)
                    padded = torch.cat([audio, padding], dim=1)
                else:
                    padded = audio
                padded_audio.append(padded)
            
            result["audios"] = torch.stack(padded_audio)
    
    return result

def prepare_dataloader(dataset, batch_size, shuffle=True, collate_fn=None, num_workers=0):
    """
    准备数据加载器
    
    参数:
        dataset: 数据集对象
        batch_size (int): 批次大小
        shuffle (bool): 是否随机打乱数据
        collate_fn (callable, optional): 整理函数
        num_workers (int): 数据加载线程数
        
    返回:
        DataLoader: 数据加载器
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

def load_text_data_from_file(file_path, encoding="utf-8"):
    """
    从文件加载文本数据
    
    参数:
        file_path (str): 文件路径
        encoding (str): 文件编码
        
    返回:
        List[str]: 文本列表
    """
    extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if extension == ".txt":
            # 纯文本文件，每行一个样本
            with open(file_path, "r", encoding=encoding) as f:
                return [line.strip() for line in f if line.strip()]
                
        elif extension == ".json":
            # JSON文件
            with open(file_path, "r", encoding=encoding) as f:
                data = json.load(f)
                
                # 判断是列表还是字典
                if isinstance(data, list):
                    # 如果是列表，尝试提取文本字段
                    if all(isinstance(item, str) for item in data):
                        return data
                    elif all(isinstance(item, dict) for item in data):
                        # 尝试找到文本字段
                        # 常见的文本字段名称
                        text_fields = ["text", "content", "sentence", "input", "prompt", "source"]
                        
                        for field in text_fields:
                            if all(field in item for item in data):
                                return [item[field] for item in data]
                        
                        # 如果没有找到合适的字段，提示用户
                        print(f"无法确定JSON文件中的文本字段，可用字段：{list(data[0].keys())}")
                        field = input("请输入要使用的文本字段名称：")
                        if all(field in item for item in data):
                            return [item[field] for item in data]
                        else:
                            raise ValueError(f"字段 '{field}' 不存在于所有项中")
                else:
                    raise ValueError("JSON文件必须包含列表或字典列表")
                    
        elif extension == ".csv":
            # CSV文件
            import csv
            with open(file_path, "r", encoding=encoding) as f:
                reader = csv.reader(f)
                headers = next(reader)
                
                print(f"CSV文件包含以下列：{headers}")
                col_idx = input("请输入要使用的文本列索引(0开始)或列名：")
                
                try:
                    if col_idx.isdigit():
                        col_idx = int(col_idx)
                        if col_idx < 0 or col_idx >= len(headers):
                            raise ValueError(f"索引超出范围，应为0-{len(headers)-1}")
                    else:
                        if col_idx not in headers:
                            raise ValueError(f"列名'{col_idx}'不存在")
                        col_idx = headers.index(col_idx)
                        
                    return [row[col_idx] for row in reader if row and row[col_idx].strip()]
                    
                except Exception as e:
                    raise ValueError(f"处理CSV文件时出错：{e}")
                    
        elif extension in [".jsonl", ".ndjson"]:
            # 每行一个JSON对象
            with open(file_path, "r", encoding=encoding) as f:
                data = [json.loads(line) for line in f if line.strip()]
                
                # 尝试找到文本字段
                text_fields = ["text", "content", "sentence", "input", "prompt", "source"]
                
                for field in text_fields:
                    if all(field in item for item in data):
                        return [item[field] for item in data]
                
                # 如果没有找到合适的字段，提示用户
                print(f"无法确定JSONL文件中的文本字段，可用字段：{list(data[0].keys())}")
                field = input("请输入要使用的文本字段名称：")
                if all(field in item for item in data):
                    return [item[field] for item in data]
                else:
                    raise ValueError(f"字段 '{field}' 不存在于所有项中")
                    
        else:
            raise ValueError(f"不支持的文件格式：{extension}")
            
    except Exception as e:
        print(f"加载文件 {file_path} 时出错：{e}")
        return []

def load_paired_data_from_file(file_path, encoding="utf-8"):
    """
    从文件加载配对数据（源文本和目标文本）
    
    参数:
        file_path (str): 文件路径
        encoding (str): 文件编码
        
    返回:
        Tuple[List[str], List[str]]: (源文本列表, 目标文本列表)
    """
    extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if extension == ".tsv" or extension == ".csv":
            # TSV/CSV文件，假设前两列分别是源和目标
            import csv
            delimiter = "\t" if extension == ".tsv" else ","
            
            with open(file_path, "r", encoding=encoding) as f:
                reader = csv.reader(f, delimiter=delimiter)
                headers = next(reader)
                
                if len(headers) < 2:
                    raise ValueError(f"文件必须至少包含两列（源和目标），但只有 {len(headers)} 列")
                
                print(f"文件包含以下列：{headers}")
                source_idx = int(input("请输入源文本列索引(0开始)："))
                target_idx = int(input("请输入目标文本列索引(0开始)："))
                
                if source_idx < 0 or source_idx >= len(headers) or target_idx < 0 or target_idx >= len(headers):
                    raise ValueError("索引超出范围")
                
                source_texts = []
                target_texts = []
                
                for row in reader:
                    if len(row) > max(source_idx, target_idx) and row[source_idx].strip() and row[target_idx].strip():
                        source_texts.append(row[source_idx].strip())
                        target_texts.append(row[target_idx].strip())
                
                return source_texts, target_texts
                
        elif extension == ".json":
            # JSON文件，假设包含源和目标字段
            with open(file_path, "r", encoding=encoding) as f:
                data = json.load(f)
                
                if not isinstance(data, list):
                    raise ValueError("JSON文件必须包含对象列表")
                
                # 常见的源/目标字段名
                source_fields = ["source", "input", "prompt", "src", "question", "instruction"]
                target_fields = ["target", "output", "response", "tgt", "answer", "completion"]
                
                # 尝试自动识别字段
                source_field = None
                target_field = None
                
                for field in source_fields:
                    if all(field in item for item in data):
                        source_field = field
                        break
                
                for field in target_fields:
                    if all(field in item for item in data):
                        target_field = field
                        break
                
                if not source_field or not target_field:
                    print(f"无法自动识别源/目标字段，可用字段：{list(data[0].keys())}")
                    source_field = input("请输入源文本字段名称：")
                    target_field = input("请输入目标文本字段名称：")
                
                if not all(source_field in item and target_field in item for item in data):
                    raise ValueError(f"并非所有数据项都包含 '{source_field}' 和 '{target_field}' 字段")
                
                source_texts = [item[source_field] for item in data]
                target_texts = [item[target_field] for item in data]
                
                return source_texts, target_texts
                
        elif extension in [".jsonl", ".ndjson"]:
            # 每行一个JSON对象
            with open(file_path, "r", encoding=encoding) as f:
                data = [json.loads(line) for line in f if line.strip()]
                
                # 尝试自动识别字段
                source_fields = ["source", "input", "prompt", "src", "question", "instruction"]
                target_fields = ["target", "output", "response", "tgt", "answer", "completion"]
                
                source_field = None
                target_field = None
                
                for field in source_fields:
                    if all(field in item for item in data):
                        source_field = field
                        break
                
                for field in target_fields:
                    if all(field in item for item in data):
                        target_field = field
                        break
                
                if not source_field or not target_field:
                    print(f"无法自动识别源/目标字段，可用字段：{list(data[0].keys())}")
                    source_field = input("请输入源文本字段名称：")
                    target_field = input("请输入目标文本字段名称：")
                
                if not all(source_field in item and target_field in item for item in data):
                    raise ValueError(f"并非所有数据项都包含 '{source_field}' 和 '{target_field}' 字段")
                
                source_texts = [item[source_field] for item in data]
                target_texts = [item[target_field] for item in data]
                
                return source_texts, target_texts
                
        else:
            raise ValueError(f"不支持的文件格式用于配对数据：{extension}")
            
    except Exception as e:
        print(f"加载配对数据文件 {file_path} 时出错：{e}")
        return [], []

def load_multimodal_data_from_directory(directory, image_dir=None, audio_dir=None, metadata_file=None, encoding="utf-8"):
    """
    从目录加载多模态数据
    
    参数:
        directory (str): 主数据目录
        image_dir (str, optional): 图像目录，默认为directory下的'images'
        audio_dir (str, optional): 音频目录，默认为directory下的'audio'
        metadata_file (str, optional): 元数据文件路径，包含文本和多模态文件的对应关系
        encoding (str): 文件编码
        
    返回:
        dict: 包含文本和对应的图像/音频路径的数据
    """
    try:
        # 设置默认目录
        image_dir = image_dir or os.path.join(directory, "images")
        audio_dir = audio_dir or os.path.join(directory, "audio")
        metadata_file = metadata_file or os.path.join(directory, "metadata.json")
        
        # 检查元数据文件是否存在
        if not os.path.exists(metadata_file):
            print(f"警告: 元数据文件 {metadata_file} 不存在，尝试从目录结构推断多模态关联")
            
            # 尝试从目录结构推断
            result = {
                "text_data": [],
                "image_paths": [],
                "audio_paths": []
            }
            
            # 检查目录结构
            has_images = os.path.exists(image_dir)
            has_audio = os.path.exists(audio_dir)
            
            if not has_images and not has_audio:
                raise ValueError(f"未找到图像或音频目录，请检查路径：{image_dir}，{audio_dir}")
            
            # 寻找文本文件
            text_files = []
            for f in os.listdir(directory):
                if f.endswith(".txt") or f.endswith(".json") or f.endswith(".csv"):
                    text_files.append(os.path.join(directory, f))
            
            if not text_files:
                raise ValueError(f"目录 {directory} 中未找到文本文件")
            
            # 使用第一个文本文件
            text_file = text_files[0]
            text_data = load_text_data_from_file(text_file, encoding)
            
            if has_images:
                image_files = [f for f in os.listdir(image_dir) 
                              if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))]
                
                # 匹配文本和图像
                if len(image_files) >= len(text_data):
                    result["text_data"] = text_data
                    result["image_paths"] = [os.path.join(image_dir, f) for f in image_files[:len(text_data)]]
                else:
                    print(f"警告: 图像数量 ({len(image_files)}) 少于文本数量 ({len(text_data)})")
                    result["text_data"] = text_data[:len(image_files)]
                    result["image_paths"] = [os.path.join(image_dir, f) for f in image_files]
            
            if has_audio:
                audio_files = [f for f in os.listdir(audio_dir) 
                              if f.lower().endswith((".wav", ".mp3", ".ogg", ".flac"))]
                
                # 如果已经有文本和图像，确保音频数量相同
                if "text_data" in result and result["text_data"]:
                    if len(audio_files) >= len(result["text_data"]):
                        result["audio_paths"] = [os.path.join(audio_dir, f) for f in audio_files[:len(result["text_data"])]]
                    else:
                        print(f"警告: 音频数量 ({len(audio_files)}) 少于已匹配的文本数量 ({len(result['text_data'])})")
                        # 裁剪文本和图像数据以匹配音频数量
                        result["text_data"] = result["text_data"][:len(audio_files)]
                        if "image_paths" in result and result["image_paths"]:
                            result["image_paths"] = result["image_paths"][:len(audio_files)]
                        result["audio_paths"] = [os.path.join(audio_dir, f) for f in audio_files]
                else:
                    # 只有音频和文本
                    if len(audio_files) >= len(text_data):
                        result["text_data"] = text_data
                        result["audio_paths"] = [os.path.join(audio_dir, f) for f in audio_files[:len(text_data)]]
                    else:
                        print(f"警告: 音频数量 ({len(audio_files)}) 少于文本数量 ({len(text_data)})")
                        result["text_data"] = text_data[:len(audio_files)]
                        result["audio_paths"] = [os.path.join(audio_dir, f) for f in audio_files]
            
            return result
            
        else:
            # 从元数据文件加载
            with open(metadata_file, "r", encoding=encoding) as f:
                metadata = json.load(f)
                
                # 检查元数据格式
                if not isinstance(metadata, list):
                    raise ValueError("元数据必须是列表格式")
                
                result = {
                    "text_data": [],
                    "image_paths": [],
                    "audio_paths": []
                }
                
                # 尝试识别字段
                sample = metadata[0]
                text_fields = ["text", "content", "caption", "description"]
                image_fields = ["image", "image_path", "img", "image_file"]
                audio_fields = ["audio", "audio_path", "sound", "audio_file"]
                
                # 查找字段
                text_field = next((f for f in text_fields if f in sample), None)
                image_field = next((f for f in image_fields if f in sample), None)
                audio_field = next((f for f in audio_fields if f in sample), None)
                
                if not text_field:
                    print(f"警告: 未找到文本字段，可用字段: {list(sample.keys())}")
                    text_field = input("请输入文本字段名称: ")
                
                # 提取数据
                for item in metadata:
                    if text_field in item:
                        result["text_data"].append(item[text_field])
                    else:
                        continue  # 跳过没有文本的项
                    
                    if image_field and image_field in item:
                        image_path = item[image_field]
                        # 如果是相对路径，转换为绝对路径
                        if not os.path.isabs(image_path):
                            image_path = os.path.join(image_dir, image_path)
                        result["image_paths"].append(image_path)
                    elif "image_paths" in result:
                        # 保持数组长度一致
                        result["image_paths"].append(None)
                    
                    if audio_field and audio_field in item:
                        audio_path = item[audio_field]
                        # 如果是相对路径，转换为绝对路径
                        if not os.path.isabs(audio_path):
                            audio_path = os.path.join(audio_dir, audio_path)
                        result["audio_paths"].append(audio_path)
                    elif "audio_paths" in result:
                        # 保持数组长度一致
                        result["audio_paths"].append(None)
                
                return result
    
    except Exception as e:
        print(f"加载多模态数据时出错: {e}")
        return {"text_data": [], "image_paths": [], "audio_paths": []}

def get_data_processors(config):
    """
    获取数据处理器
    
    参数:
        config: 模型配置
        
    返回:
        dict: 数据处理器字典
    """
    processors = {}
    
    # 获取分词器
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or "gpt2")
        # 确保有填充标记
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        processors['tokenizer'] = tokenizer
    except ImportError:
        print("警告: 无法导入transformers库，将使用基本分词器")
        # 创建一个简单的分词器（仅用于开发/测试）
        class SimpleTokenizer:
            def __init__(self):
                self.pad_token_id = 0
                self.eos_token_id = 1
                
            def encode(self, text, return_tensors=None, max_length=None, truncation=None):
                # 非常简单的分词 - 仅以空格分割
                tokens = text.split()
                if max_length and truncation and len(tokens) > max_length:
                    tokens = tokens[:max_length]
                # 转换为ID (简单地使用哈希)
                ids = [hash(t) % 10000 + 10 for t in tokens]
                if return_tensors == "pt":
                    return torch.tensor([ids])
                return ids
                
        processors['tokenizer'] = SimpleTokenizer()
    
    # 图像处理器（如果启用）
    if hasattr(config, 'enable_image') and config.enable_image:
        try:
            from torchvision import transforms
            image_processor = transforms.Compose([
                transforms.Resize((config.image_size, config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
            ])
            processors['image_processor'] = image_processor
        except ImportError:
            print("警告: 无法导入torchvision，将禁用图像处理")
    
    # 音频处理器（如果启用）
    if hasattr(config, 'enable_audio') and config.enable_audio:
        try:
            import torchaudio
            import torchaudio.transforms as T
            
            # 定义音频处理函数
            def audio_processor(audio, sample_rate):
                # 重采样到目标采样率
                target_sample_rate = getattr(config, 'audio_sample_rate', 16000)
                if sample_rate != target_sample_rate:
                    resampler = T.Resample(sample_rate, target_sample_rate)
                    audio = resampler(audio)
                
                # 提取Mel特征
                n_fft = getattr(config, 'audio_n_fft', 1024)
                hop_length = getattr(config, 'audio_hop_length', 512)
                n_mels = getattr(config, 'audio_n_mels', 80)
                
                mel_spectrogram = T.MelSpectrogram(
                    sample_rate=target_sample_rate,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_mels=n_mels
                )
                mel_features = mel_spectrogram(audio)
                
                # 转换为对数刻度
                log_mel_features = torch.log(mel_features + 1e-9)
                
                return log_mel_features
                
            processors['audio_processor'] = audio_processor
        except ImportError:
            print("警告: 无法导入torchaudio，将禁用音频处理")
    
    return processors

def prepare_training_data(config, text_data=None, paired_data=None, multimodal_data=None, batch_size=8):
    """
    准备训练数据，创建适当的数据集和数据加载器
    
    参数:
        config: 模型配置
        text_data: 单模态文本数据
        paired_data: 配对文本数据 (源文本，目标文本)
        multimodal_data: 多模态数据
        batch_size: 批处理大小
        
    返回:
        tuple: (dataloader, processors)
    """
    # 获取数据处理器
    processors = get_data_processors(config)
    tokenizer = processors['tokenizer']
    
    # 确定数据类型并创建相应的数据集
    if multimodal_data is not None:
        # 多模态数据集
        dataset = MultimodalDataset(
            text_data=multimodal_data["text_data"],
            image_paths=multimodal_data.get("image_paths"),
            audio_paths=multimodal_data.get("audio_paths"),
            tokenizer=tokenizer,
            image_processor=processors.get('image_processor'),
            audio_processor=processors.get('audio_processor'),
            max_text_length=config.max_seq_len
        )
        
        # 创建数据加载器
        dataloader = prepare_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: multimodal_collate_fn(batch, tokenizer.pad_token_id)
        )
        
    elif paired_data is not None:
        # 配对文本数据集
        source_texts, target_texts = paired_data
        dataset = PairedTextDataset(
            source_texts=source_texts,
            target_texts=target_texts,
            tokenizer=tokenizer,
            max_source_length=config.max_source_len,
            max_target_length=config.max_target_len
        )
        
        # 创建数据加载器
        dataloader = prepare_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: text_collate_fn(batch, tokenizer.pad_token_id)
        )
        
    elif text_data is not None:
        # 单模态文本数据集
        dataset = TextDataset(
            texts=text_data,
            tokenizer=tokenizer,
            max_length=config.max_seq_len
        )
        
        # 创建数据加载器
        dataloader = prepare_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: text_collate_fn(batch, tokenizer.pad_token_id)
        )
        
    else:
        raise ValueError("必须提供至少一种数据类型：text_data, paired_data 或 multimodal_data")
    
    return dataloader, processors

def main():
    """
    主函数，用于测试模块功能
    """
    import argparse
    from types import SimpleNamespace
    
    parser = argparse.ArgumentParser(description="APT模型数据加载模块测试")
    parser.add_argument("--data_path", type=str, help="数据文件或目录路径")
    parser.add_argument("--data_type", type=str, default="text", choices=["text", "paired", "multimodal"],
                       help="数据类型：文本、配对或多模态")
    parser.add_argument("--batch_size", type=int, default=8, help="批处理大小")
    parser.add_argument("--max_seq_len", type=int, default=128, help="最大序列长度")
    parser.add_argument("--image_size", type=int, default=224, help="图像大小")
    parser.add_argument("--enable_image", action="store_true", help="启用图像处理")
    parser.add_argument("--enable_audio", action="store_true", help="启用音频处理")
    
    args = parser.parse_args()
    
    # 测试配置
    config = SimpleNamespace(
        tokenizer_name="gpt2",
        max_seq_len=args.max_seq_len,
        max_source_len=args.max_seq_len,
        max_target_len=args.max_seq_len,
        image_size=args.image_size,
        enable_image=args.enable_image,
        enable_audio=args.enable_audio,
        audio_sample_rate=16000
    )
    
    try:
        if args.data_type == "text":
            # 测试单文本数据加载
            if not args.data_path:
                raise ValueError("必须提供数据文件路径")
            
            print(f"加载文本数据: {args.data_path}")
            text_data = load_text_data_from_file(args.data_path)
            if not text_data:
                raise ValueError("未能加载文本数据")
            
            print(f"加载了 {len(text_data)} 条文本样本")
            print("样本示例:")
            for i in range(min(3, len(text_data))):
                print(f"  {i+1}. {text_data[i][:50]}...")
            
            dataloader, processors = prepare_training_data(
                config, 
                text_data=text_data,
                batch_size=args.batch_size
            )
            
        elif args.data_type == "paired":
            # 测试配对文本数据加载
            if not args.data_path:
                raise ValueError("必须提供数据文件路径")
            
            print(f"加载配对数据: {args.data_path}")
            source_texts, target_texts = load_paired_data_from_file(args.data_path)
            if not source_texts or not target_texts:
                raise ValueError("未能加载配对文本数据")
            
            print(f"加载了 {len(source_texts)} 对配对文本样本")
            print("样本示例:")
            for i in range(min(3, len(source_texts))):
                print(f"  {i+1}. 源: {source_texts[i][:30]}... -> 目标: {target_texts[i][:30]}...")
            
            dataloader, processors = prepare_training_data(
                config, 
                paired_data=(source_texts, target_texts),
                batch_size=args.batch_size
            )
            
        elif args.data_type == "multimodal":
            # 测试多模态数据加载
            if not args.data_path:
                raise ValueError("必须提供数据目录路径")
            
            print(f"加载多模态数据: {args.data_path}")
            multimodal_data = load_multimodal_data_from_directory(args.data_path)
            
            if not multimodal_data["text_data"]:
                raise ValueError("未能加载多模态数据")
            
            print(f"加载了 {len(multimodal_data['text_data'])} 条多模态样本")
            print(f"包含图像: {len(multimodal_data.get('image_paths', []))}")
            print(f"包含音频: {len(multimodal_data.get('audio_paths', []))}")
            
            dataloader, processors = prepare_training_data(
                config, 
                multimodal_data=multimodal_data,
                batch_size=args.batch_size
            )
        
        # 测试数据加载
        print("\n测试数据加载...")
        batch = next(iter(dataloader))
        print(f"批次数据类型: {type(batch)}")
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: 形状={value.shape}, 类型={value.dtype}")
            else:
                print(f"  {key}: 类型={type(value)}")
        
        print("数据加载测试完成！")
        
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()