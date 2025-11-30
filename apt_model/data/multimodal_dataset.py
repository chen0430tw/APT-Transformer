#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多模态数据集和数据加载器
Multimodal dataset and data loaders for text, image, and audio data
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import warnings


class MultimodalDataset(Dataset):
    """
    多模态数据集
    支持文本、图像、音频三种模态的组合
    """

    def __init__(
        self,
        data_path: Union[str, Path, List[Dict]],
        tokenizer = None,
        vision_processor = None,
        audio_processor = None,
        modalities: List[str] = ['text', 'vision', 'audio'],
        max_text_length: int = 512,
        image_size: Tuple[int, int] = (224, 224),
        audio_sample_rate: int = 16000,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            data_path: 数据路径，可以是:
                - JSON文件路径 (包含数据列表)
                - 目录路径 (包含多个数据文件)
                - 数据字典列表
            tokenizer: 文本tokenizer
            vision_processor: 图像处理器
            audio_processor: 音频处理器
            modalities: 需要的模态列表
            max_text_length: 最大文本长度
            image_size: 图像尺寸
            audio_sample_rate: 音频采样率
            cache_dir: 缓存目录
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.vision_processor = vision_processor
        self.audio_processor = audio_processor
        self.modalities = modalities
        self.max_text_length = max_text_length
        self.image_size = image_size
        self.audio_sample_rate = audio_sample_rate
        self.cache_dir = cache_dir

        # 加载数据
        self.data = self._load_data(data_path)

        # 验证数据
        self._validate_data()

    def _load_data(self, data_path: Union[str, Path, List[Dict]]) -> List[Dict]:
        """加载数据"""
        if isinstance(data_path, list):
            # 直接使用提供的数据列表
            return data_path

        data_path = Path(data_path)

        if data_path.is_file():
            # 加载JSON文件
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, dict) and 'data' in data:
                return data['data']
            elif isinstance(data, list):
                return data
            else:
                raise ValueError(f"Invalid data format in {data_path}")

        elif data_path.is_dir():
            # 加载目录中的所有JSON文件
            all_data = []
            for json_file in data_path.glob('*.json'):
                with open(json_file, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        all_data.extend(file_data)
                    elif isinstance(file_data, dict) and 'data' in file_data:
                        all_data.extend(file_data['data'])

            if not all_data:
                raise ValueError(f"No valid data found in {data_path}")

            return all_data

        else:
            raise FileNotFoundError(f"Data path not found: {data_path}")

    def _validate_data(self):
        """验证数据格式"""
        if not self.data:
            raise ValueError("Dataset is empty")

        # 检查第一个样本的格式
        sample = self.data[0]

        if 'text' in self.modalities and 'text' not in sample:
            warnings.warn("Text modality required but not found in data samples")

        if 'vision' in self.modalities and 'image' not in sample and 'image_path' not in sample:
            warnings.warn("Vision modality required but no image/image_path in data samples")

        if 'audio' in self.modalities and 'audio' not in sample and 'audio_path' not in sample:
            warnings.warn("Audio modality required but no audio/audio_path in data samples")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取一个样本

        Returns:
            字典包含:
            - text_input_ids: 文本token IDs (如果有text模态)
            - pixel_values: 图像张量 (如果有vision模态)
            - audio_values: 音频张量 (如果有audio模态)
            - labels: 标签 (如果有)
            - metadata: 元数据
        """
        sample = self.data[idx]
        result = {}

        # 处理文本模态
        if 'text' in self.modalities and self.tokenizer is not None:
            text = sample.get('text', '')
            if text:
                encoded = self.tokenizer(
                    text,
                    max_length=self.max_text_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                result['text_input_ids'] = encoded['input_ids'].squeeze(0)
                result['text_attention_mask'] = encoded['attention_mask'].squeeze(0)

        # 处理图像模态
        if 'vision' in self.modalities:
            image = self._load_image(sample)
            if image is not None:
                if self.vision_processor is not None:
                    processed = self.vision_processor(images=image, return_tensors='pt')
                    result['pixel_values'] = processed['pixel_values'].squeeze(0)
                else:
                    # 使用默认预处理
                    result['pixel_values'] = self._preprocess_image(image)

        # 处理音频模态
        if 'audio' in self.modalities:
            audio = self._load_audio(sample)
            if audio is not None:
                if self.audio_processor is not None:
                    processed = self.audio_processor(
                        audio,
                        sampling_rate=self.audio_sample_rate,
                        return_tensors='pt'
                    )
                    result['audio_values'] = processed['input_values'].squeeze(0)
                else:
                    # 使用默认预处理
                    result['audio_values'] = self._preprocess_audio(audio)

        # 添加标签
        if 'label' in sample:
            result['labels'] = torch.tensor(sample['label'], dtype=torch.long)
        elif 'labels' in sample:
            result['labels'] = torch.tensor(sample['labels'], dtype=torch.long)

        # 添加元数据
        result['metadata'] = {
            'idx': idx,
            'id': sample.get('id', idx),
            'has_text': 'text_input_ids' in result,
            'has_vision': 'pixel_values' in result,
            'has_audio': 'audio_values' in result
        }

        return result

    def _load_image(self, sample: Dict) -> Optional[Any]:
        """加载图像"""
        # 如果直接提供了图像数据
        if 'image' in sample:
            return sample['image']

        # 如果提供了图像路径
        if 'image_path' in sample:
            try:
                from PIL import Image
                image_path = sample['image_path']

                # 如果是相对路径，尝试相对于cache_dir
                if self.cache_dir and not os.path.isabs(image_path):
                    image_path = os.path.join(self.cache_dir, image_path)

                image = Image.open(image_path).convert('RGB')
                return image
            except Exception as e:
                warnings.warn(f"Failed to load image from {sample.get('image_path')}: {e}")
                return None

        return None

    def _load_audio(self, sample: Dict) -> Optional[torch.Tensor]:
        """加载音频"""
        # 如果直接提供了音频数据
        if 'audio' in sample:
            audio = sample['audio']
            if isinstance(audio, torch.Tensor):
                return audio
            elif isinstance(audio, (list, tuple)):
                return torch.tensor(audio, dtype=torch.float32)

        # 如果提供了音频路径
        if 'audio_path' in sample:
            try:
                import torchaudio
                audio_path = sample['audio_path']

                # 如果是相对路径
                if self.cache_dir and not os.path.isabs(audio_path):
                    audio_path = os.path.join(self.cache_dir, audio_path)

                waveform, sr = torchaudio.load(audio_path)

                # 重采样
                if sr != self.audio_sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.audio_sample_rate)
                    waveform = resampler(waveform)

                # 转换为单声道
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                return waveform.squeeze(0)  # [T]

            except Exception as e:
                warnings.warn(f"Failed to load audio from {sample.get('audio_path')}: {e}")
                return None

        return None

    def _preprocess_image(self, image) -> torch.Tensor:
        """默认图像预处理"""
        try:
            from torchvision import transforms

            transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            return transform(image)

        except ImportError:
            raise ImportError("torchvision is required for default image preprocessing")

    def _preprocess_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """默认音频预处理"""
        try:
            import torchaudio

            # 计算Mel频谱
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.audio_sample_rate,
                n_mels=80
            )

            mel_spec = mel_transform(waveform)  # [n_mels, T]
            return mel_spec

        except ImportError:
            raise ImportError("torchaudio is required for default audio preprocessing")


class MultimodalCollator:
    """
    多模态数据批处理器
    用于DataLoader的collate_fn
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        return_tensors: str = 'pt'
    ):
        self.pad_token_id = pad_token_id
        self.return_tensors = return_tensors

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        批处理多模态数据

        Args:
            batch: 样本列表

        Returns:
            批处理后的张量字典
        """
        result = {}

        # 处理文本
        if 'text_input_ids' in batch[0]:
            text_ids = [item['text_input_ids'] for item in batch]
            result['text_input_ids'] = torch.stack(text_ids)

            if 'text_attention_mask' in batch[0]:
                text_masks = [item['text_attention_mask'] for item in batch]
                result['text_attention_mask'] = torch.stack(text_masks)

        # 处理图像
        if 'pixel_values' in batch[0]:
            pixel_values = [item['pixel_values'] for item in batch]
            result['pixel_values'] = torch.stack(pixel_values)

        # 处理音频
        if 'audio_values' in batch[0]:
            audio_values = [item['audio_values'] for item in batch]
            # 音频可能长度不同，需要填充
            result['audio_values'] = self._pad_audio(audio_values)

        # 处理标签
        if 'labels' in batch[0]:
            labels = [item['labels'] for item in batch]
            result['labels'] = torch.stack(labels)

        # 处理元数据
        result['metadata'] = [item['metadata'] for item in batch]

        return result

    def _pad_audio(self, audio_list: List[torch.Tensor]) -> torch.Tensor:
        """填充音频到相同长度"""
        if audio_list[0].dim() == 1:
            # 1D音频: [T]
            max_len = max(audio.size(0) for audio in audio_list)
            padded = []
            for audio in audio_list:
                if audio.size(0) < max_len:
                    padding = torch.zeros(max_len - audio.size(0))
                    audio = torch.cat([audio, padding])
                padded.append(audio)
            return torch.stack(padded)

        elif audio_list[0].dim() == 2:
            # 2D音频: [n_mels, T]
            max_len = max(audio.size(1) for audio in audio_list)
            padded = []
            for audio in audio_list:
                if audio.size(1) < max_len:
                    padding = torch.zeros(audio.size(0), max_len - audio.size(1))
                    audio = torch.cat([audio, padding], dim=1)
                padded.append(audio)
            return torch.stack(padded)

        else:
            raise ValueError(f"Unexpected audio dimensions: {audio_list[0].dim()}")


def create_multimodal_dataloader(
    data_path: Union[str, Path, List[Dict]],
    tokenizer = None,
    vision_processor = None,
    audio_processor = None,
    modalities: List[str] = ['text', 'vision', 'audio'],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **dataset_kwargs
) -> DataLoader:
    """
    创建多模态数据加载器的工厂函数

    Args:
        data_path: 数据路径
        tokenizer: 文本tokenizer
        vision_processor: 图像处理器
        audio_processor: 音频处理器
        modalities: 模态列表
        batch_size: 批大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        **dataset_kwargs: 传递给MultimodalDataset的其他参数

    Returns:
        DataLoader实例
    """
    dataset = MultimodalDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        vision_processor=vision_processor,
        audio_processor=audio_processor,
        modalities=modalities,
        **dataset_kwargs
    )

    collator = MultimodalCollator(
        pad_token_id=tokenizer.pad_token_id if tokenizer else 0
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )

    return dataloader


# 简化的单模态数据集（向后兼容）

class TextOnlyDataset(MultimodalDataset):
    """仅文本数据集"""

    def __init__(self, data_path, tokenizer, **kwargs):
        super().__init__(
            data_path=data_path,
            tokenizer=tokenizer,
            modalities=['text'],
            **kwargs
        )


class VisionOnlyDataset(MultimodalDataset):
    """仅图像数据集"""

    def __init__(self, data_path, vision_processor, **kwargs):
        super().__init__(
            data_path=data_path,
            vision_processor=vision_processor,
            modalities=['vision'],
            **kwargs
        )


class AudioOnlyDataset(MultimodalDataset):
    """仅音频数据集"""

    def __init__(self, data_path, audio_processor, **kwargs):
        super().__init__(
            data_path=data_path,
            audio_processor=audio_processor,
            modalities=['audio'],
            **kwargs
        )
