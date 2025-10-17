import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """位置编码实现，支持动态扩展"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        参数:
            x: [batch_size, seq_len, embedding_dim]
        返回:
            加入位置编码后的 x
        """
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            # 动态扩展位置编码
            device = x.device
            extra_len = seq_len - self.pe.size(1)
            # 生成额外的编码
            pe_extra = torch.zeros(extra_len, self.d_model, device=device)
            position = torch.arange(self.pe.size(1), seq_len, dtype=torch.float, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float, device=device) * (-math.log(10000.0) / self.d_model))
            pe_extra[:, 0::2] = torch.sin(position * div_term)
            pe_extra[:, 1::2] = torch.cos(position * div_term)
            pe_extra = pe_extra.unsqueeze(0)  # shape: [1, extra_len, d_model]
            pe = torch.cat([self.pe, pe_extra], dim=1)
        else:
            pe = self.pe
        return x + pe[:, :seq_len, :]

class TokenEmbedding(nn.Module):
    """词元嵌入实现"""
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, tokens):
        """
        参数:
            tokens: [batch_size, seq_len]
        返回:
            [batch_size, seq_len, embedding_dim]
        """
        return self.embedding(tokens) * math.sqrt(self.embedding_dim)

class ImageEmbedding(nn.Module):
    """图像嵌入实现"""
    def __init__(self, d_model, image_size=224, patch_size=16):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))

    def forward(self, images):
        """
        参数:
            images: [batch_size, 3, image_size, image_size]
        返回:
            [batch_size, num_patches, d_model]
        """
        batch_size = images.shape[0]
        x = self.patch_embed(images)  # [B, d_model, grid, grid]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, d_model]
        x = x + self.pos_embed
        return x