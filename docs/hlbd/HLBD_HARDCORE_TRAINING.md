# HLBD Hardcore V2 训练指南

## 📊 数据集信息

**HLBD Hardcore Full V2** - 严格逻辑训练数据集

- **总样本数**: 5042条（超过5000目标）
- **数据文件**: `data/HLBD_Hardcore_Full_V2.json`
- **文件大小**: ~450 KB
- **生成器**: `tools/generate_hlbd_hardcore_v2.py`

### 数据集组成

| 模块 | 样本数 | 描述 |
|------|--------|------|
| 几何定义 | 860 | 形状、公式、面积周长计算 |
| 算术运算 | 1,899 | 四则运算、多步计算 |
| 生肖序列 | 528 | 十二生肖顺序、属性、年份 |
| 物理定律 | 825 | 物理定律、公式、计算 |
| 反向学英文 | 930 | 词汇、短语、句子翻译 |
| **总计** | **5,042** | **唯一样本数** |

### 🛡️ 防模式坍缩特性

数据集采用**数据稀释学 (Data Dilution Learning)** 策略：

1. ✓ **模块内容多样化** - 避免单一模式
2. ✓ **问题表述变化** - 避免固定句式
3. ✓ **随机打散顺序** - 避免顺序依赖
4. ✓ **难度梯度分布** - 避免难度聚集
5. ✓ **去重机制确保** - 避免重复学习

## 🚀 快速开始

### 方法1: 使用Python启动器（推荐）

```bash
python3 launch_hlbd_hardcore_training.py
```

自动检查依赖并启动训练。

### 方法2: 使用Bash脚本

```bash
bash run_hlbd_hardcore_training.sh
```

### 方法3: 直接调用训练脚本

```bash
python3 training/train_hlbd_playground.py \
    --dataset data/HLBD_Hardcore_Full_V2.json \
    --output-dir models/hlbd_hardcore_v2 \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 5e-5 \
    --warmup-steps 100 \
    --save-every 5 \
    --eval-every 5 \
    --use-amp \
    --gradient-accumulation-steps 2
```

## 📋 环境要求

### 必需依赖

- **Python**: 3.8+
- **PyTorch**: 1.9+ (推荐最新稳定版)
- **CUDA**: 11.x+ (GPU训练)
- **NumPy**: 最新版本

### 安装依赖

```bash
# PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 其他依赖
pip install numpy
```

### 硬件要求

| 配置 | 最低 | 推荐 |
|------|------|------|
| GPU | GTX 1660 (6GB) | RTX 3070 (8GB+) |
| 内存 | 16GB | 32GB |
| 存储 | 2GB | 10GB (含检查点) |

## 🎛️ 训练参数

### 默认配置

```python
{
    "epochs": 50,                          # 训练轮数
    "batch_size": 32,                      # 批次大小
    "learning_rate": 5e-5,                 # 学习率
    "warmup_steps": 100,                   # 预热步数
    "save_every": 5,                       # 每5轮保存
    "eval_every": 5,                       # 每5轮评估
    "use_amp": True,                       # 混合精度训练
    "gradient_accumulation_steps": 2       # 梯度累积
}
```

### 调优建议

#### GPU内存不足

```bash
# 减小batch size
--batch-size 16

# 增加梯度累积
--gradient-accumulation-steps 4

# 禁用混合精度
# (移除 --use-amp)
```

#### 加速训练

```bash
# 增大batch size (如果GPU允许)
--batch-size 64

# 增加学习率
--learning-rate 1e-4
```

#### 提高精度

```bash
# 增加训练轮数
--epochs 100

# 降低学习率
--learning-rate 2e-5

# 增加预热步数
--warmup-steps 200
```

## 📈 训练特性

### 🎢 Playground Theory

使用 **CosineAnnealingWarmRestarts** 学习率调度器：

- 周期性重启学习率
- 跳出局部最优
- 适合HLBD严格逻辑学习

### 🔧 DBC-DAC 梯度稳定

- **Dimension Balanced Compression**: 维度平衡压缩
- **Dimension Accompanying Compensation**: 维度伴随补偿
- 防止梯度爆炸/消失

### 🏷️ 动态标签支持

Tokenizer支持特殊标签:

- `[EMOJI]` - Emoji表情
- `[EN]` - 英文内容
- `[PY]` - Python代码
- `[JP]` - 日文内容
- `[KR]` - 韩文内容

## 📂 输出结构

```
models/hlbd_hardcore_v2/
├── checkpoint_epoch_5.pt       # 第5轮检查点
├── checkpoint_epoch_10.pt      # 第10轮检查点
├── ...
├── checkpoint_epoch_50.pt      # 第50轮检查点
├── best_model.pt               # 最佳模型
├── tokenizer.json              # Tokenizer配置
└── config.json                 # 模型配置

logs/hlbd_hardcore_v2/
└── training_20241222_135100.log  # 训练日志
```

## 🔍 监控训练

### 实时查看日志

```bash
tail -f logs/hlbd_hardcore_v2/training_*.log
```

### 训练指标

监控以下指标:

- **Loss**: 训练损失（应持续下降）
- **Perplexity**: 困惑度（越低越好）
- **Accuracy**: 准确率（HLBD应达到>95%）
- **Learning Rate**: 学习率变化

### 正常训练表现

```
Epoch 1/50:  Loss: 4.523  PPL: 92.1   Acc: 12.3%
Epoch 5/50:  Loss: 2.134  PPL: 8.44   Acc: 45.7%
Epoch 10/50: Loss: 1.245  PPL: 3.47   Acc: 67.2%
Epoch 20/50: Loss: 0.567  PPL: 1.76   Acc: 85.4%
Epoch 50/50: Loss: 0.123  PPL: 1.13   Acc: 96.8%
```

## 🐛 常见问题

### Q: CUDA out of memory

**A**: 减小batch size或增加梯度累积:

```bash
python3 launch_hlbd_hardcore_training.py --batch-size 16 --gradient-accumulation-steps 4
```

### Q: 训练太慢

**A**: 检查是否使用GPU:

```python
import torch
print(torch.cuda.is_available())  # 应该是 True
```

### Q: Loss不下降

**A**: 可能的原因:

1. 学习率过高 → 降低到 `1e-5`
2. 数据问题 → 检查数据集格式
3. 模型配置问题 → 检查 `config.json`

### Q: 准确率低于预期

**A**: HLBD是严格逻辑数据集:

1. 需要足够的训练轮数（至少30轮）
2. 确保数据稀释学生效
3. 检查是否有重复样本

## 📊 评估模型

训练完成后评估:

```bash
python3 training/evaluate_hlbd.py \
    --model models/hlbd_hardcore_v2/best_model.pt \
    --dataset data/HLBD_Hardcore_Full_V2.json
```

预期结果:

- **准确率**: >95% (5042样本)
- **模式坍缩指标**: 无重复输出
- **泛化能力**: 能处理未见过的类似问题

## 🔬 实验追踪

### 记录实验配置

```bash
# 创建实验记录
echo "Experiment: HLBD_Hardcore_V2" > experiments/exp_001.txt
echo "Date: $(date)" >> experiments/exp_001.txt
echo "Dataset: 5042 samples" >> experiments/exp_001.txt
echo "Config: epochs=50, bs=32, lr=5e-5" >> experiments/exp_001.txt
```

### 比较不同配置

| 实验 | Epochs | Batch Size | LR | 最终准确率 |
|------|--------|------------|-----|-----------|
| exp_001 | 50 | 32 | 5e-5 | 96.8% |
| exp_002 | 100 | 32 | 2e-5 | 97.5% |
| exp_003 | 50 | 64 | 5e-5 | 95.2% |

## 📚 相关文档

- [训练后端使用指南](docs/TRAINING_BACKENDS.md)
- [APT模型架构](docs/MODEL_ARCHITECTURE.md)
- [可视化指南](docs/VISUALIZATION_GUIDE.md)
- [仓库结构](docs/repo_schema.md)

## 🎯 下一步

训练完成后:

1. **评估模型性能**
2. **测试推理速度**
3. **部署模型**
4. **Fine-tune特定任务**

## 💡 最佳实践

1. ✓ 使用混合精度训练（`--use-amp`）
2. ✓ 定期保存检查点（`--save-every 5`）
3. ✓ 监控训练日志
4. ✓ 验证数据集质量
5. ✓ 记录实验配置
6. ✓ 比较不同超参数

---

**生成时间**: 2024-12-22
**版本**: V2.0
**维护者**: APT-Transformer Team
