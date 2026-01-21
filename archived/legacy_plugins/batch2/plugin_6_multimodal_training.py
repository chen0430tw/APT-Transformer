"""
Multimodal Training Plugin for APT
多模态训练插件 - 支持文本+图像+音频的联合训练
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from torchvision import transforms
from PIL import Image
import torchaudio
from torch.utils.data import Dataset, DataLoader


class MultimodalTrainingPlugin:
    """
    多模态训练插件
    
    支持的模态:
    1. 文本 (Text) - 使用APT原生支持
    2. 图像 (Image) - 使用视觉编码器
    3. 音频 (Audio) - 使用音频编码器
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.name = "multimodal-training"
        self.version = "1.0.0"
        self.config = config
        
        self.modalities = config.get('modalities', ['text', 'image'])
        self.vision_encoder = config.get('vision_encoder', 'clip')
        self.audio_encoder = config.get('audio_encoder', 'wav2vec2')
        
        # 初始化编码器
        self.encoders = {}
        self._init_encoders()
    
    # ==================== 编码器初始化 ====================
    
    def _init_encoders(self):
        """初始化各模态编码器"""
        if 'image' in self.modalities:
            self._init_vision_encoder()
        
        if 'audio' in self.modalities:
            self._init_audio_encoder()
    
    def _init_vision_encoder(self):
        """初始化视觉编码器"""
        print("🖼️ 初始化视觉编码器...")
        
        if self.vision_encoder == 'clip':
            try:
                from transformers import CLIPVisionModel, CLIPImageProcessor
                
                model_name = self.config.get('clip_model', 'openai/clip-vit-base-patch32')
                self.encoders['vision_model'] = CLIPVisionModel.from_pretrained(model_name)
                self.encoders['vision_processor'] = CLIPImageProcessor.from_pretrained(model_name)
                
                print(f"✅ CLIP视觉编码器加载完成: {model_name}")
                
            except ImportError:
                print("❌ 需要安装transformers库")
                raise
        
        elif self.vision_encoder == 'vit':
            try:
                from transformers import ViTModel, ViTImageProcessor
                
                model_name = self.config.get('vit_model', 'google/vit-base-patch16-224')
                self.encoders['vision_model'] = ViTModel.from_pretrained(model_name)
                self.encoders['vision_processor'] = ViTImageProcessor.from_pretrained(model_name)
                
                print(f"✅ ViT视觉编码器加载完成: {model_name}")
                
            except ImportError:
                print("❌ 需要安装transformers库")
                raise
        
        else:
            # 自定义简单CNN编码器
            self.encoders['vision_model'] = SimpleCNNEncoder(
                output_dim=self.config.get('vision_dim', 768)
            )
            print("✅ 简单CNN编码器初始化完成")
    
    def _init_audio_encoder(self):
        """初始化音频编码器"""
        print("🎵 初始化音频编码器...")
        
        if self.audio_encoder == 'wav2vec2':
            try:
                from transformers import Wav2Vec2Model, Wav2Vec2Processor
                
                model_name = self.config.get('wav2vec2_model', 'facebook/wav2vec2-base')
                self.encoders['audio_model'] = Wav2Vec2Model.from_pretrained(model_name)
                self.encoders['audio_processor'] = Wav2Vec2Processor.from_pretrained(model_name)
                
                print(f"✅ Wav2Vec2音频编码器加载完成: {model_name}")
                
            except ImportError:
                print("❌ 需要安装transformers库")
                raise
        
        else:
            # 自定义简单音频编码器
            self.encoders['audio_model'] = SimpleAudioEncoder(
                output_dim=self.config.get('audio_dim', 768)
            )
            print("✅ 简单音频编码器初始化完成")
    
    # ==================== 数据处理 ====================
    
    def process_text(self, text: str, tokenizer: Any) -> torch.Tensor:
        """处理文本数据"""
        return tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
    
    def process_image(self, image: Image.Image) -> torch.Tensor:
        """
        处理图像数据
        
        Args:
            image: PIL图像
            
        Returns:
            图像特征张量
        """
        if 'vision_processor' in self.encoders:
            # 使用预训练的processor
            inputs = self.encoders['vision_processor'](
                images=image,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = self.encoders['vision_model'](**inputs)
                # 获取图像特征
                image_features = outputs.last_hidden_state.mean(dim=1)  # [1, hidden_dim]
            
            return image_features
        
        else:
            # 使用自定义编码器
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            image_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]
            
            with torch.no_grad():
                image_features = self.encoders['vision_model'](image_tensor)
            
            return image_features
    
    def process_audio(self, audio_path: str) -> torch.Tensor:
        """
        处理音频数据
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            音频特征张量
        """
        # 加载音频
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if 'audio_processor' in self.encoders:
            # 使用预训练的processor
            inputs = self.encoders['audio_processor'](
                waveform.squeeze().numpy(),
                sampling_rate=sample_rate,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = self.encoders['audio_model'](**inputs)
                # 获取音频特征
                audio_features = outputs.last_hidden_state.mean(dim=1)  # [1, hidden_dim]
            
            return audio_features
        
        else:
            # 使用自定义编码器
            # 重采样到16kHz
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            with torch.no_grad():
                audio_features = self.encoders['audio_model'](waveform)
            
            return audio_features
    
    # ==================== 多模态融合 ====================
    
    def create_multimodal_model(
        self,
        base_model: nn.Module,
        fusion_method: str = 'concatenate'
    ) -> nn.Module:
        """
        创建多模态融合模型
        
        Args:
            base_model: 基础APT模型
            fusion_method: 融合方法 (concatenate/add/attention)
            
        Returns:
            多模态模型
        """
        print(f"🔗 创建多模态模型 (融合方法: {fusion_method})...")
        
        return MultimodalFusionModel(
            base_model=base_model,
            vision_encoder=self.encoders.get('vision_model'),
            audio_encoder=self.encoders.get('audio_model'),
            fusion_method=fusion_method,
            hidden_dim=self.config.get('hidden_dim', 768)
        )
    
    # ==================== 数据加载 ====================
    
    def create_multimodal_dataloader(
        self,
        text_data: List[str],
        image_data: Optional[List[str]] = None,
        audio_data: Optional[List[str]] = None,
        batch_size: int = 32
    ) -> DataLoader:
        """
        创建多模态数据加载器
        
        Args:
            text_data: 文本数据列表
            image_data: 图像路径列表
            audio_data: 音频路径列表
            batch_size: 批次大小
            
        Returns:
            DataLoader
        """
        print("📦 创建多模态数据加载器...")
        
        class MultimodalDataset(Dataset):
            def __init__(self, texts, images, audios, plugin):
                self.texts = texts
                self.images = images
                self.audios = audios
                self.plugin = plugin
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                item = {'text': self.texts[idx]}
                
                if self.images:
                    image = Image.open(self.images[idx]).convert('RGB')
                    item['image'] = self.plugin.process_image(image)
                
                if self.audios:
                    item['audio'] = self.plugin.process_audio(self.audios[idx])
                
                return item
        
        dataset = MultimodalDataset(text_data, image_data, audio_data, self)
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.get('num_workers', 4)
        )
    
    # ==================== 训练流程 ====================
    
    def train_multimodal(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 3,
        device: str = 'cuda'
    ):
        """
        多模态训练流程
        
        Args:
            model: 多模态模型
            dataloader: 数据加载器
            optimizer: 优化器
            num_epochs: 训练轮数
            device: 设备
        """
        print(f"🏋️ 开始多模态训练 ({num_epochs} epochs)...")
        
        model.to(device)
        model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            
            for batch_idx, batch in enumerate(dataloader):
                # 将数据移到设备
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                # 前向传播
                outputs = model(**batch)
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f"  Epoch {epoch+1}/{num_epochs} | "
                          f"Batch {batch_idx}/{len(dataloader)} | "
                          f"Loss: {loss.item():.4f}")
            
            avg_loss = epoch_loss / len(dataloader)
            print(f"📊 Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
        
        print("✅ 多模态训练完成!")
    
    # ==================== 评估与推理 ====================
    
    def evaluate_multimodal(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = 'cuda'
    ) -> Dict[str, float]:
        """
        多模态模型评估
        
        Args:
            model: 多模态模型
            dataloader: 数据加载器
            device: 设备
            
        Returns:
            评估指标字典
        """
        print("📊 开始多模态评估...")
        
        model.to(device)
        model.eval()
        
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # 将数据移到设备
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)
                
                # 前向传播
                outputs = model(**batch)
                loss = outputs.get('loss', 0)
                logits = outputs.get('logits')
                
                total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
                
                # 计算准确率（如果有标签）
                if 'labels' in batch and logits is not None:
                    predictions = logits.argmax(dim=-1)
                    correct = (predictions == batch['labels']).sum().item()
                    total_correct += correct
                    total_samples += batch['labels'].size(0)
        
        metrics = {
            'loss': total_loss / len(dataloader),
        }
        
        if total_samples > 0:
            metrics['accuracy'] = total_correct / total_samples
        
        print(f"✅ 评估完成: {metrics}")
        return metrics
    
    def inference_multimodal(
        self,
        model: nn.Module,
        text: str = None,
        image: Image.Image = None,
        audio_path: str = None,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        多模态推理
        
        Args:
            model: 多模态模型
            text: 文本输入
            image: 图像输入
            audio_path: 音频路径
            device: 设备
            
        Returns:
            模型输出
        """
        model.to(device)
        model.eval()
        
        inputs = {}
        
        if text:
            # 处理文本
            inputs['text'] = self.process_text(text, model.tokenizer)
        
        if image:
            # 处理图像
            inputs['image'] = self.process_image(image).to(device)
        
        if audio_path:
            # 处理音频
            inputs['audio'] = self.process_audio(audio_path).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        return outputs
    
    # ==================== 插件钩子 ====================
    
    def on_training_start(self, context: Dict[str, Any]):
        """训练开始时的钩子"""
        if context.get('enable_multimodal'):
            print("🌈 启用多模态训练模式")
            
            # 创建多模态模型
            base_model = context.get('model')
            multimodal_model = self.create_multimodal_model(base_model)
            context['model'] = multimodal_model
    
    def on_batch_start(self, context: Dict[str, Any]):
        """批次开始时的钩子"""
        batch = context.get('batch')
        
        # 确保所有模态数据都在正确的设备上
        device = context.get('device', 'cuda')
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
    
    def on_epoch_end(self, context: Dict[str, Any]):
        """训练轮次结束时的钩子"""
        epoch = context.get('epoch', 0)
        metrics = context.get('metrics', {})
        
        print(f"📊 Epoch {epoch} - 多模态训练指标: {metrics}")


# ==================== 辅助类 ====================

class SimpleCNNEncoder(nn.Module):
    """简单的CNN图像编码器"""
    
    def __init__(self, output_dim: int = 768):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256, output_dim)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SimpleAudioEncoder(nn.Module):
    """简单的音频编码器"""
    
    def __init__(self, output_dim: int = 768):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, output_dim)
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, time]
        x = self.conv_layers(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x


class MultimodalFusionModel(nn.Module):
    """多模态融合模型"""
    
    def __init__(
        self,
        base_model: nn.Module,
        vision_encoder: Optional[nn.Module] = None,
        audio_encoder: Optional[nn.Module] = None,
        fusion_method: str = 'concatenate',
        hidden_dim: int = 768
    ):
        super().__init__()
        self.base_model = base_model
        self.vision_encoder = vision_encoder
        self.audio_encoder = audio_encoder
        self.fusion_method = fusion_method
        
        # 融合层
        if fusion_method == 'concatenate':
            # 计算融合后的维度
            fusion_dim = hidden_dim
            if vision_encoder:
                fusion_dim += hidden_dim
            if audio_encoder:
                fusion_dim += hidden_dim
            
            self.fusion_layer = nn.Linear(fusion_dim, hidden_dim)
        
        elif fusion_method == 'attention':
            self.fusion_attention = nn.MultiheadAttention(
                hidden_dim, num_heads=8, batch_first=True
            )
        
        # 输出层
        vocab_size = getattr(base_model.config, 'vocab_size', 50257)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
    
    def forward(
        self,
        text: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        """
        前向传播
        
        Args:
            text: 文本嵌入 [batch, seq_len, hidden]
            image: 图像特征 [batch, hidden]
            audio: 音频特征 [batch, hidden]
            labels: 标签（可选）
        """
        # 获取文本特征
        if hasattr(self.base_model, 'get_text_features'):
            text_features = self.base_model.get_text_features(text)
        else:
            # 假设text已经是特征
            text_features = text
        
        # 收集所有模态特征
        features = [text_features]
        
        if image is not None and self.vision_encoder:
            features.append(image)
        
        if audio is not None and self.audio_encoder:
            features.append(audio)
        
        # 融合
        if self.fusion_method == 'concatenate':
            # 拼接融合
            fused = torch.cat(features, dim=-1)
            fused = self.fusion_layer(fused)
        
        elif self.fusion_method == 'add':
            # 加法融合
            fused = sum(features) / len(features)
        
        elif self.fusion_method == 'attention':
            # 注意力融合
            features_stacked = torch.stack(features, dim=1)  # [batch, num_modalities, hidden]
            fused, _ = self.fusion_attention(
                features_stacked, features_stacked, features_stacked
            )
            fused = fused.mean(dim=1)  # [batch, hidden]
        
        # 输出
        logits = self.output_layer(fused)
        
        # 计算损失（如果提供了标签）
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {
            'logits': logits,
            'loss': loss
        }


# ==================== 使用示例 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("🌈 多模态训练插件 (Multimodal Training Plugin)")
    print("=" * 60)
    
    # 配置
    config = {
        'modalities': ['text', 'image', 'audio'],
        'vision_encoder': 'clip',
        'audio_encoder': 'wav2vec2',
        'clip_model': 'openai/clip-vit-base-patch32',
        'wav2vec2_model': 'facebook/wav2vec2-base',
        'hidden_dim': 768,
        'num_workers': 4
    }
    
    plugin = MultimodalTrainingPlugin(config)
    
    print("\n✅ 多模态训练插件初始化完成!")
    print("\n💡 插件功能:")
    print("1. ✨ 支持文本+图像+音频的联合训练")
    print("2. 🖼️ 使用CLIP/ViT作为视觉编码器")
    print("3. 🎵 使用Wav2Vec2作为音频编码器")
    print("4. 🔗 支持多种融合方法(concatenate/add/attention)")
    print("5. 📦 提供完整的数据加载和训练流程")
    print("6. 📊 支持模型评估和推理")
    
    print("\n📝 使用示例:")
    print("""
    # 创建多模态数据加载器
    dataloader = plugin.create_multimodal_dataloader(
        text_data=["这是一张图片", "这是一段音频"],
        image_data=["image1.jpg", "image2.jpg"],
        audio_data=["audio1.wav", "audio2.wav"],
        batch_size=2
    )
    
    # 创建多模态模型
    multimodal_model = plugin.create_multimodal_model(
        base_model=apt_model,
        fusion_method='attention'
    )
    
    # 训练
    plugin.train_multimodal(
        model=multimodal_model,
        dataloader=dataloader,
        optimizer=optimizer,
        num_epochs=3
    )
    """)
    
    print("\n" + "=" * 60)
