### `apt_model/modeling/multimodal_model.py`
import torch
import torch.nn as nn
from apt_model.modeling.apt_model import APTLargeModel
from apt_model.config.multimodal_config import MultimodalConfig

class MultimodalAPTModel(APTLargeModel):
    """多模态APT模型"""
    
    def __init__(self, config, multimodal_config=None):
        """初始化多模态APT模型"""
        super().__init__(config)
        
        # 如果没有提供多模态配置，创建默认配置
        if multimodal_config is None:
            multimodal_config = MultimodalConfig()
        
        self.multimodal_config = multimodal_config
        
        # 初始化多模态处理器
        self.multimodal_processor = None  # 稍后实例化
        
        # 添加多模态嵌入层
        self._init_multimodal_layers()
    
    def _init_multimodal_layers(self):
        """初始化多模态层"""
        # 图像嵌入
        if self.multimodal_config.enable_image:
            # 假设使用ViT风格的图像嵌入
            self.image_patch_embed = nn.Conv2d(
                in_channels=3,
                out_channels=self.config.d_model,
                kernel_size=self.multimodal_config.patch_size,
                stride=self.multimodal_config.patch_size
            )
            
            self.image_pos_embed = nn.Parameter(
                torch.zeros(1, self.multimodal_config.num_patches, self.config.d_model)
            )
            
            # 图像融合层
            self.image_fusion = nn.Linear(self.config.d_model, self.config.d_model)
        
        # 音频嵌入
        if self.multimodal_config.enable_audio:
            # 音频特征提取
            self.audio_embed = nn.Sequential(
                nn.Linear(80, 256),  # 80 = n_mels from MelSpectrogram
                nn.GELU(),
                nn.Linear(256, self.config.d_model)
            )
            
            # 音频融合层
            self.audio_fusion = nn.Linear(self.config.d_model, self.config.d_model)
        
        # 多模态融合
        self.modality_fusion = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model),
            nn.LayerNorm(self.config.d_model),
            nn.GELU()
        )
        
        # 模态类型嵌入，用于区分不同模态
        self.modality_type_embed = nn.Embedding(3, self.config.d_model)  # 0=text, 1=image, 2=audio
    
    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None, images=None, audios=None):
        """多模态前向传播"""
        # 处理文本嵌入
        src_emb = self.token_embedding(src_ids)
        src_emb = self.positional_encoding(src_emb)
        
        # 添加模态类型编码
        modality_embeds = self.modality_type_embed(torch.zeros(src_emb.size(0), src_emb.size(1), 
                                                              dtype=torch.long, device=src_emb.device))
        src_emb = src_emb + modality_embeds
        
        # 处理多模态输入
        multimodal_embeds = []
        
        # 处理图像和音频 (略去详细实现)
        
        # 编码器和解码器处理
        memory = self.encoder(src_emb, attention_mask=src_mask)
        dec_output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_mask=src_mask)
        
        # 输出投影
        logits = self.output_projection(dec_output)
        
        return logits