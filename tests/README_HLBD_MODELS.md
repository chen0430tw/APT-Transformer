# HLBD 模型保存和加载指南

## 概述

本指南说明如何使用 HLBD（分层语言启蒙数据集）训练的模型保存和加载功能。

---

## 文件结构

```
tests/
├── test_hlbd_quick_learning.py   # 训练脚本（包含保存功能）
├── load_hlbd_model.py             # 加载脚本（推理和交互）
├── saved_models/                  # 保存的模型目录
│   ├── hlbd_model_20250116_143022.pt
│   ├── hlbd_model_20250116_150135.pt
│   └── ...
└── README_HLBD_MODELS.md          # 本文档
```

---

## 训练并保存模型

### 运行训练脚本

```bash
python tests/test_hlbd_quick_learning.py
```

### 训练过程

1. 加载 HLBD 数据集（100个概念，400个训练对）
2. 使用 SimpleCharTokenizer（支持 emoji）
3. 训练 500 epochs
4. 每 3 epochs 测试一次生成能力
5. **自动保存模型到 `tests/saved_models/`**
6. 验证保存和加载功能

### 保存的内容

每个保存的模型文件（`.pt`）包含：

```python
{
    'model_state_dict': dict,          # 模型权重
    'tokenizer_char_to_id': dict,      # 字符 → ID 映射
    'tokenizer_id_to_char': dict,      # ID → 字符映射
    'tokenizer_next_id': int,          # 下一个可用 ID
    'tokenizer_vocab_size': int,       # 词汇表大小
    'config': {                        # 模型配置
        'vocab_size': int,
        'd_model': int,
        'num_encoder_layers': int,
        'num_decoder_layers': int,
        ...
    },
    'training_info': {                 # 训练信息
        'num_epochs': int,
        'final_loss': float,
        'timestamp': str,
    }
}
```

### 模型命名

模型文件名格式：`hlbd_model_YYYYMMDD_HHMMSS.pt`

示例：`hlbd_model_20250116_143022.pt`

---

## 加载和推理

### 方法 1：使用加载脚本（推荐）

```bash
# 自动加载最新模型
python tests/load_hlbd_model.py

# 指定模型路径
python tests/load_hlbd_model.py tests/saved_models/hlbd_model_20250116_143022.pt
```

**功能：**
- ✅ 自动查找最新模型
- ✅ 运行预定义测试用例
- ✅ 交互式推理模式

**交互式示例：**
```
💬 交互式推理模式 (输入 'quit' 退出)
======================================================================

请输入文本: 🌧️
生成: 今天天气阴沉，下雨了。

请输入文本: I love you
生成: 表达真挚情感，我爱你。

请输入文本: quit
👋 再见！
```

### 方法 2：在 Python 代码中加载

```python
import torch
from tests.test_hlbd_quick_learning import load_model_and_tokenizer, generate_text

# 1. 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, tokenizer, info = load_model_and_tokenizer('tests/saved_models/hlbd_model_xxx.pt', device)

# 2. 推理
input_text = "🌧️"
output = generate_text(model, tokenizer, input_text, device)
print(f"输入: {input_text}")
print(f"输出: {output}")
```

---

## 继续训练

如果想在已保存的模型基础上继续训练：

```python
import torch
from tests.test_hlbd_quick_learning import load_model_and_tokenizer

# 1. 加载已保存的模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, tokenizer, info = load_model_and_tokenizer('tests/saved_models/hlbd_model_xxx.pt', device)

# 2. 准备训练数据和优化器
from torch import nn, optim
from torch.utils.data import DataLoader

# ... 创建数据集 ...
optimizer = optim.Adam(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# 3. 继续训练
model.train()
for epoch in range(100):  # 再训练 100 epochs
    # ... 训练逻辑 ...
    pass

# 4. 保存新模型
from tests.test_hlbd_quick_learning import save_model_and_tokenizer
save_model_and_tokenizer(model, tokenizer, config, 'tests/saved_models', ...)
```

---

## 模型信息

### 查看模型信息

```python
import torch

checkpoint = torch.load('tests/saved_models/hlbd_model_xxx.pt', map_location='cpu')

print("训练信息:")
print(f"  Epoch: {checkpoint['training_info']['num_epochs']}")
print(f"  Loss: {checkpoint['training_info']['final_loss']:.4f}")
print(f"  时间: {checkpoint['training_info']['timestamp']}")

print("\n模型配置:")
for key, value in checkpoint['config'].items():
    print(f"  {key}: {value}")

print(f"\n词汇表大小: {len(checkpoint['tokenizer_char_to_id'])}")
```

### 典型的训练好的模型

- **参数量**: ~10M (d_model=256, 3 encoder + 3 decoder layers)
- **文件大小**: ~40-50 MB
- **词汇表**: ~300-500 字符（动态增长）
- **训练时间**: 500 epochs ~30-60分钟（CPU）

---

## 注意事项

### ✅ 优点

1. **完整保存**：模型权重 + tokenizer + 配置，一次性加载即可使用
2. **跨平台**：可以在 CPU 训练，GPU 加载，反之亦然
3. **可追溯**：包含训练信息（epoch、loss、时间戳）
4. **支持 emoji**：SimpleCharTokenizer 动态添加字符，无损保存

### ⚠️ 注意

1. **词汇表兼容性**：
   - 加载的 tokenizer 词汇表必须与训练时一致
   - 不要手动修改 `char_to_id` 或 `id_to_char`

2. **设备兼容性**：
   - 使用 `map_location` 参数确保跨设备加载
   - CPU 训练的模型可以在 GPU 上推理

3. **版本兼容性**：
   - 确保 PyTorch 版本兼容（建议 >= 1.10）
   - APTModel 架构不能改变

4. **文件管理**：
   - 定期清理旧模型（只保留最优模型）
   - 建议使用时间戳命名，便于追溯

---

## 故障排查

### 问题 1：加载失败

```
RuntimeError: Error(s) in loading state_dict
```

**原因**：模型架构改变

**解决**：确保 APTModelConfiguration 参数与保存时一致

### 问题 2：Emoji 无法识别

```
输入: 🌧️
输出: [空]
```

**原因**：使用了错误的 tokenizer（BertTokenizer）

**解决**：确保加载的是 SimpleCharTokenizer

### 问题 3：找不到模型文件

```
❌ 未找到已保存的模型！
```

**原因**：未运行训练脚本或保存目录不存在

**解决**：先运行 `python tests/test_hlbd_quick_learning.py`

---

## 高级用法

### 模型融合

```python
# 加载两个模型，融合权重
model1, _, _ = load_model_and_tokenizer('model1.pt', device)
model2, _, _ = load_model_and_tokenizer('model2.pt', device)

# 平均权重
for p1, p2 in zip(model1.parameters(), model2.parameters()):
    p1.data = (p1.data + p2.data) / 2
```

### 导出为 ONNX

```python
import torch.onnx

# 准备示例输入
dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 10)).to(device)

# 导出
torch.onnx.export(
    model,
    dummy_input,
    'hlbd_model.onnx',
    input_names=['input_ids'],
    output_names=['output'],
    dynamic_axes={'input_ids': {0: 'batch', 1: 'sequence'}}
)
```

---

## 参考

- [test_hlbd_quick_learning.py](test_hlbd_quick_learning.py) - 训练脚本
- [load_hlbd_model.py](load_hlbd_model.py) - 加载脚本
- [emoji_handling_analysis.md](../emoji_handling_analysis.md) - Emoji 处理分析

**最后更新**: 2025-01-16
