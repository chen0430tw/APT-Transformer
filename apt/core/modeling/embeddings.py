from apt_model.utils.fake_torch import get_torch
torch = get_torch()
from apt_model.utils.fake_torch import get_torch
torch = get_torch()
nn = torch.nn
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
    """
    词元嵌入实现（支持动态扩充）

    特性:
    - 动态扩充词表（当遇到OOV token时自动扩展）
    - 保留历史embedding权重
    - 新token使用Xavier初始化
    """
    def __init__(self, vocab_size, embedding_dim, enable_dynamic_expansion=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.initial_vocab_size = vocab_size
        self.current_vocab_size = vocab_size
        self.enable_dynamic_expansion = enable_dynamic_expansion

    def expand_vocab(self, new_vocab_size):
        """
        动态扩充词表

        参数:
            new_vocab_size: 新的词表大小（必须 >= 当前大小）
        """
        if new_vocab_size <= self.current_vocab_size:
            return

        # 创建新的embedding层
        old_embedding = self.embedding
        new_embedding = nn.Embedding(new_vocab_size, self.embedding_dim)

        # 复制旧权重
        with torch.no_grad():
            new_embedding.weight[:self.current_vocab_size] = old_embedding.weight

            # 新token使用Xavier初始化
            nn.init.xavier_uniform_(
                new_embedding.weight[self.current_vocab_size:],
                gain=1.0
            )

        self.embedding = new_embedding
        self.current_vocab_size = new_vocab_size

        print(f"[TokenEmbedding] 词表扩充: {self.current_vocab_size - (new_vocab_size - self.current_vocab_size)} → {new_vocab_size}")

    def forward(self, tokens):
        """
        参数:
            tokens: [batch_size, seq_len]
        返回:
            [batch_size, seq_len, embedding_dim]
        """
        # 检查是否需要自动扩充
        if self.enable_dynamic_expansion:
            max_token_id = tokens.max().item()
            if max_token_id >= self.current_vocab_size:
                # 自动扩充到能容纳max_token_id的最小2的幂次
                new_size = 2 ** math.ceil(math.log2(max_token_id + 1))
                self.expand_vocab(new_size)

        return self.embedding(tokens) * math.sqrt(self.embedding_dim)

class ImageEmbedding(nn.Module):
    """
    图像嵌入实现（支持动态分辨率）

    特性:
    - 支持动态图像分辨率
    - 位置编码自动插值到新分辨率
    - 保持patch_size不变
    """
    def __init__(self, d_model, image_size=224, patch_size=16, enable_dynamic_resolution=True):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.enable_dynamic_resolution = enable_dynamic_resolution

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.num_patches = (image_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        self.grid_size = image_size // patch_size

    def interpolate_pos_embed(self, new_num_patches):
        """
        插值位置编码到新的patch数量

        参数:
            new_num_patches: 新的patch数量
        """
        if new_num_patches == self.num_patches:
            return self.pos_embed

        # 计算新的grid大小
        new_grid_size = int(math.sqrt(new_num_patches))

        # 2D插值
        pos_embed = self.pos_embed.reshape(1, self.grid_size, self.grid_size, self.d_model)
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [1, d_model, grid, grid]

        new_pos_embed = torch.nn.functional.interpolate(
            pos_embed,
            size=(new_grid_size, new_grid_size),
            mode='bicubic',
            align_corners=False
        )

        new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)  # [1, new_grid, new_grid, d_model]
        new_pos_embed = new_pos_embed.reshape(1, new_num_patches, self.d_model)

        return new_pos_embed

    def forward(self, images):
        """
        参数:
            images: [batch_size, 3, H, W] (H, W可变)
        返回:
            [batch_size, num_patches, d_model]
        """
        batch_size = images.shape[0]
        x = self.patch_embed(images)  # [B, d_model, grid_h, grid_w]

        # 计算当前的patch数量
        grid_h, grid_w = x.shape[2], x.shape[3]
        current_num_patches = grid_h * grid_w

        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, d_model]

        # 动态调整位置编码
        if self.enable_dynamic_resolution and current_num_patches != self.num_patches:
            pos_embed = self.interpolate_pos_embed(current_num_patches)
            x = x + pos_embed
        else:
            x = x + self.pos_embed

        return x