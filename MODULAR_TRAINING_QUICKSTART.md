# HLBD模块化训练快速开始

## 🎯 一句话总结

**现在可以在单次训练中同时使用HLBD Full（5000样本）和HLBD Hardcore（5042样本），总计10,000+样本！**

## ⚡ 30秒快速开始

```bash
# 一键启动模块化训练
python3 launch_hlbd_modular_training.py
```

就这么简单！脚本会自动：
- ✅ 检查数据集文件是否存在
- ✅ 验证Python依赖
- ✅ 启动10,000+样本的联合训练
- ✅ 保存到`hlbd_modular/`目录

## 📊 你会得到什么？

### 训练数据组成

```
总样本: 10,042个
├── HLBD Full V2: 5,000个 (49.8%)
│   └── 8层语言结构 + Level 3句法层
└── HLBD Hardcore V2: 5,042个 (50.2%)
    └── 几何/算术/生肖/物理/英文
```

### 模型能力

训练后的模型将具备：

✅ **多语言理解**（来自HLBD Full）
- 中文、英文、日文、韩文
- 拼音、Emoji理解
- 跨语言映射

✅ **句法结构学习**（来自HLBD Full Level 3）
- S = NP + VP 语法规则
- 结构化语言表示
- 符号推理能力

✅ **严格逻辑推理**（来自HLBD Hardcore）
- 几何计算
- 算术运算
- 物理定律
- 生肖推理
- 英文翻译

## 🚀 其他启动方式

### 方式1: 使用启动脚本（推荐）

```bash
python3 launch_hlbd_modular_training.py
```

### 方式2: 直接调用训练脚本

```bash
python3 training/train_hlbd_playground.py \
    --datasets data/HLBD_Full_V2.json data/HLBD_Hardcore_Full_V2.json \
    --epochs 50 \
    --save-dir hlbd_modular
```

### 方式3: 自定义参数

```bash
# 更多epochs
python3 training/train_hlbd_playground.py \
    --datasets data/HLBD_Full_V2.json data/HLBD_Hardcore_Full_V2.json \
    --epochs 100 \
    --save-interval 20

# 更小batch size（如果GPU内存不足）
python3 training/train_hlbd_playground.py \
    --datasets data/HLBD_Full_V2.json data/HLBD_Hardcore_Full_V2.json \
    --batch-size 8
```

## 📁 训练结果

训练完成后，你会在`hlbd_modular/`目录找到：

```
hlbd_modular/
├── checkpoint_epoch_10.pt       # Epoch 10检查点
├── checkpoint_epoch_20.pt       # Epoch 20检查点
├── checkpoint_epoch_30.pt       # Epoch 30检查点
├── checkpoint_epoch_40.pt       # Epoch 40检查点
├── final_model.pt               # 最终模型
└── experiment_report.json       # 训练曲线（可视化）
```

每个checkpoint包含：
- 模型权重
- 优化器状态
- Tokenizer词汇表
- **数据集统计**（新增！）
- 训练loss历史

## 🔍 监控训练

### 实时日志输出

```
📚 模块化HLBD数据集加载器
   数据集数量: 2
============================================================

📂 [1/2] 加载数据集: data/HLBD_Full_V2.json
   格式: HLBD Full (8层结构)
   ✓ 成功加载 5000 个训练对

📂 [2/2] 加载数据集: data/HLBD_Hardcore_Full_V2.json
   格式: HLBD Hardcore (模块化)
   ✓ 成功加载 5042 个训练对

📊 数据集统计:
   HLBD_Full_V2: 5000 对 (49.8%)
   HLBD_Hardcore_Full_V2: 5042 对 (50.2%)
   总计: 10042 个训练对
   ✓ 已混合打散

🔤 预填充词汇表...
   ✓ 词汇表大小: 3847
============================================================

🏗️  构建APT模型...
   总参数: 12,345,678

============================================================
🎮 HLBD Playground训练开始
============================================================

📍 Epoch 1/50
   Batch 0/315 | Loss: 4.2341 | LR: 0.000300
   Batch 20/315 | Loss: 3.8765 | LR: 0.000298
   ...
   Loss: 3.5432 | 用时: 45.23s

✅ Checkpoint已保存: hlbd_modular/checkpoint_epoch_10.pt
   数据集来源:
     - HLBD_Full_V2: 5000 样本
     - HLBD_Hardcore_Full_V2: 5042 样本
```

### 查看训练曲线

```bash
# 使用可视化工具
python3 tools/visualize_experiment.py hlbd_modular/experiment_report.json
```

## ⚠️ 常见问题

### Q: 数据集文件不存在？

```bash
# 生成HLBD Full V2
python3 tools/generate_hlbd_full_v2.py

# 生成HLBD Hardcore V2
python3 tools/generate_hlbd_hardcore_v2.py
```

### Q: GPU内存不足？

编辑`training/train_hlbd_playground.py`，修改：

```python
class PlaygroundConfig:
    batch_size = 8  # 从16改为8
    # 或
    d_model = 128   # 从256改为128
```

### Q: 只想训练单个数据集？

```bash
# 仍然支持！向后兼容
python3 training/train_hlbd_playground.py \
    --dataset data/HLBD_Full_V2.json \
    --epochs 50
```

### Q: 想添加自定义数据集？

```bash
python3 training/train_hlbd_playground.py \
    --datasets \
        data/HLBD_Full_V2.json \
        data/HLBD_Hardcore_Full_V2.json \
        data/my_custom_dataset.json \
    --epochs 50
```

确保你的数据集格式是：
- HLBD Full格式: `{"samples": [...]}`
- HLBD Hardcore格式: `{"data": {...}}`

## 📚 完整文档

- **[完整使用指南](HLBD_MODULAR_TRAINING.md)** - 详细的配置、调优、故障排查
- **[实现细节](MODULAR_TRAINING_IMPLEMENTATION.md)** - 技术实现和代码修改
- **[数据集总结](DATASETS_COMPLETION_SUMMARY.md)** - 两个数据集的完整信息
- **[Hardcore训练](HLBD_HARDCORE_TRAINING.md)** - HLBD Hardcore专门指南

## ✅ 验证安装

```bash
# 快速检查
python3 -c "
import torch
import json
from pathlib import Path

# 检查PyTorch
print(f'✓ PyTorch {torch.__version__}')

# 检查数据集
for ds in ['data/HLBD_Full_V2.json', 'data/HLBD_Hardcore_Full_V2.json']:
    if Path(ds).exists():
        size = Path(ds).stat().st_size / (1024*1024)
        print(f'✓ {ds} ({size:.1f} MB)')
    else:
        print(f'✗ {ds} (不存在)')
"
```

预期输出：
```
✓ PyTorch 2.x.x
✓ data/HLBD_Full_V2.json (3.1 MB)
✓ data/HLBD_Hardcore_Full_V2.json (0.5 MB)
```

## 🎉 立即开始！

```bash
python3 launch_hlbd_modular_training.py
```

---

**创建时间**: 2024-12-22
**版本**: 1.0
**难度**: ⭐ 超简单（一行命令）
**预计训练时间**: 2-4小时（RTX 3070，50 epochs）
