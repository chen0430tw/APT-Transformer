### `apt_model/modeling/utils.py`
from apt_model.utils.fake_torch import get_torch
torch = get_torch()
from apt_model.utils.fake_torch import get_torch
torch = get_torch()
nn = torch.nn

def init_weights(module):
    """初始化模块参数"""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def create_src_mask(src, pad_idx):
    """创建源序列掩码"""
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    # [batch_size, 1, 1, src_len]
    return src_mask

def create_tgt_mask(tgt, pad_idx):
    """创建目标序列掩码（包括填充掩码和后续token掩码）"""
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    # [batch_size, 1, 1, tgt_len]
    
    tgt_len = tgt.size(1)
    tgt_sub_mask = torch.triu(
        torch.ones((tgt_len, tgt_len), device=tgt.device) * float('-inf'),
        diagonal=1
    )
    # [tgt_len, tgt_len]
    
    tgt_mask = tgt_pad_mask & tgt_sub_mask
    return tgt_mask