# APT模型训练Checkpoint迁移指南

**日期**: 2025-10-27
**目的**: 说明如何备份和迁移训练checkpoint到其他电脑继续训练

---

## 📍 Checkpoint和缓存保存位置

### 1. 训练模型保存位置

#### 默认保存路径：
```
./apt_model/          # 当前工作目录下
├── model.pt          # 模型权重
├── config.json       # 模型配置
└── tokenizer/        # 分词器文件
    ├── vocab.json
    ├── merges.txt
    └── tokenizer_config.json
```

**命令行参数**:
```bash
python -m apt_model train --save-path ./my_model
# 将保存到 ./my_model/ 目录
```

---

### 2. Checkpoint保存位置（使用CheckpointManager时）

#### 结构：
```
<save_path>/
├── checkpoints/                    # checkpoint目录
│   ├── apt_model_epoch1_step500.pt
│   ├── apt_model_epoch2_step1000.pt
│   ├── apt_model_epoch3_step1500_best.pt
│   └── ...
├── metadata.json                   # 训练元数据
├── tokenizer/                      # 分词器
│   ├── vocab.json
│   └── ...
└── model.pt                        # 最终模型（可选）
```

#### Checkpoint文件内容：
每个`.pt`文件包含：
```python
{
    'epoch': 当前epoch,
    'global_step': 全局步数,
    'model_state_dict': 模型参数,
    'optimizer_state_dict': 优化器状态,
    'scheduler_state_dict': 学习率调度器状态,
    'loss_history': 损失历史,
    'metrics': 评估指标,
    'config': 模型配置
}
```

---

### 3. 系统缓存位置

#### 默认缓存目录：
```
~/.apt_cache/                # 用户主目录下
├── models/                  # 预训练模型缓存
├── datasets/                # 数据集缓存
├── tokenizers/              # 分词器缓存
├── checkpoints/             # 训练checkpoint（如果使用CacheManager）
├── logs/                    # 日志文件
└── temp/                    # 临时文件
```

**Linux/Mac**: `~/.apt_cache/` → `/home/username/.apt_cache/`
**Windows**: `~/.apt_cache/` → `C:\Users\username\.apt_cache\`

---

## 📦 需要备份的文件

### 方案A: 最小备份（仅继续训练）

```bash
# 仅需要备份checkpoint文件
<save_path>/
├── checkpoints/
│   └── apt_model_epoch3_step1500_best.pt  # 最新或最佳checkpoint
└── metadata.json                           # 元数据（可选）
```

**大小**: 取决于模型大小，通常100MB - 2GB

---

### 方案B: 完整备份（推荐）

```bash
<save_path>/
├── checkpoints/          # 所有checkpoint
├── metadata.json         # 训练元数据
├── tokenizer/            # 分词器
├── model.pt              # 最终模型（如果有）
└── config.json           # 配置文件
```

**大小**: 约为单个checkpoint的2-3倍

---

### 方案C: 完整备份+缓存

```bash
# 训练目录
<save_path>/
└── ... (同方案B)

# 系统缓存
~/.apt_cache/
├── datasets/            # 如果使用了自定义数据集
└── tokenizers/          # 如果使用了自定义分词器
```

**大小**: 可能达到数GB（取决于数据集大小）

---

## 🚀 跨电脑迁移步骤

### 情景1: 从电脑A迁移到电脑B继续训练

#### 在电脑A（源电脑）：

**步骤1: 确认checkpoint位置**
```bash
# 查看训练保存路径（假设是 ./my_training）
ls -lh ./my_training/checkpoints/

# 输出示例：
# apt_model_epoch1_step500.pt
# apt_model_epoch2_step1000.pt
# apt_model_epoch3_step1500_best.pt
```

**步骤2: 打包checkpoint**
```bash
# 方法1: 打包整个训练目录
tar -czf training_backup.tar.gz ./my_training/

# 方法2: 只打包checkpoint和必要文件
tar -czf training_backup.tar.gz \
    ./my_training/checkpoints/ \
    ./my_training/metadata.json \
    ./my_training/tokenizer/ \
    ./my_training/config.json
```

**步骤3: 传输文件**
```bash
# 方法1: 使用U盘/移动硬盘
cp training_backup.tar.gz /media/usb/

# 方法2: 使用scp (如果两台电脑在同一网络)
scp training_backup.tar.gz user@computerB:/path/to/destination/

# 方法3: 使用云存储
# 上传到Google Drive/Dropbox/OneDrive等
```

---

#### 在电脑B（目标电脑）：

**步骤1: 准备环境**
```bash
# 确保已安装APT Model和依赖
pip install -r requirements.txt

# 克隆代码仓库（如果还没有）
git clone https://github.com/your-repo/APT-Transformer.git
cd APT-Transformer
```

**步骤2: 解压checkpoint**
```bash
# 解压到相同或新的目录
tar -xzf training_backup.tar.gz

# 或解压到指定目录
mkdir -p ./restored_training
tar -xzf training_backup.tar.gz -C ./restored_training/
```

**步骤3: 验证文件完整性**
```bash
# 检查checkpoint文件
ls -lh ./my_training/checkpoints/

# 检查metadata
cat ./my_training/metadata.json
```

**步骤4: 恢复训练**

##### 方法A: 使用CheckpointManager（推荐）
```python
from apt_model.training.checkpoint import CheckpointManager

# 初始化CheckpointManager
checkpoint_manager = CheckpointManager(
    save_dir="./my_training",
    model_name="apt_model",
    logger=logger
)

# 加载最新checkpoint
epoch, global_step, loss_history, metrics = checkpoint_manager.load_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    latest=True  # 或 best=True 加载最佳checkpoint
)

print(f"从 epoch {epoch}, step {global_step} 恢复训练")
print(f"之前的loss历史: {loss_history[-5:]}")

# 继续训练（从epoch+1开始）
for epoch in range(epoch + 1, total_epochs):
    # ... 训练循环
```

##### 方法B: 手动加载checkpoint
```python
import torch

# 加载checkpoint
checkpoint_path = "./my_training/checkpoints/apt_model_epoch3_step1500_best.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)

# 恢复模型
model.load_state_dict(checkpoint['model_state_dict'])

# 恢复优化器（重要！保持学习率）
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# 恢复scheduler（重要！保持warmup等策略）
if checkpoint['scheduler_state_dict']:
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# 获取训练状态
start_epoch = checkpoint['epoch'] + 1
global_step = checkpoint['global_step']
loss_history = checkpoint['loss_history']

print(f"从 epoch {start_epoch} 继续训练")
```

---

## ⚠️ 重要注意事项

### 1. 硬件差异
**问题**: 电脑A用GPU训练，电脑B只有CPU

**解决方案**:
```python
# 加载checkpoint时指定map_location
checkpoint = torch.load(
    checkpoint_path,
    map_location='cpu'  # 强制使用CPU
)

# 或自动检测
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(checkpoint_path, map_location=device)
```

### 2. PyTorch版本差异
**问题**: 电脑A用PyTorch 2.0，电脑B用PyTorch 1.13

**解决方案**:
```bash
# 在电脑B安装相同版本的PyTorch
pip install torch==2.0.0  # 使用与电脑A相同的版本
```

### 3. 路径差异
**问题**: 电脑A的路径是`/home/userA/training`，电脑B路径不同

**解决方案**:
- Checkpoint中保存的是相对路径，通常不影响
- 如果有绝对路径问题，修改metadata.json中的路径

### 4. 数据集位置
**问题**: 继续训练需要原始数据集

**解决方案**:
```bash
# 方法1: 一起打包数据集
tar -czf full_backup.tar.gz \
    ./my_training/ \
    ./datasets/

# 方法2: 在电脑B重新准备相同数据集
# 确保数据集路径和内容与电脑A一致
```

---

## 🔄 自动化迁移脚本

### 备份脚本（在电脑A执行）

创建 `backup_training.sh`:
```bash
#!/bin/bash
# APT模型训练备份脚本

TRAINING_DIR="./my_training"
BACKUP_NAME="apt_training_backup_$(date +%Y%m%d_%H%M%S).tar.gz"

echo "开始备份训练数据..."
echo "源目录: $TRAINING_DIR"
echo "备份文件: $BACKUP_NAME"

# 检查目录是否存在
if [ ! -d "$TRAINING_DIR" ]; then
    echo "错误: 训练目录不存在: $TRAINING_DIR"
    exit 1
fi

# 打包checkpoint和必要文件
tar -czf "$BACKUP_NAME" \
    "$TRAINING_DIR/checkpoints/" \
    "$TRAINING_DIR/metadata.json" \
    "$TRAINING_DIR/tokenizer/" \
    "$TRAINING_DIR/config.json" \
    2>/dev/null

# 检查是否成功
if [ $? -eq 0 ]; then
    SIZE=$(du -h "$BACKUP_NAME" | cut -f1)
    echo "✅ 备份完成！"
    echo "文件: $BACKUP_NAME"
    echo "大小: $SIZE"
else
    echo "❌ 备份失败"
    exit 1
fi
```

使用：
```bash
chmod +x backup_training.sh
./backup_training.sh
```

---

### 恢复脚本（在电脑B执行）

创建 `restore_training.sh`:
```bash
#!/bin/bash
# APT模型训练恢复脚本

BACKUP_FILE="$1"
RESTORE_DIR="${2:-./restored_training}"

if [ -z "$BACKUP_FILE" ]; then
    echo "用法: ./restore_training.sh <backup_file> [restore_dir]"
    exit 1
fi

if [ ! -f "$BACKUP_FILE" ]; then
    echo "错误: 备份文件不存在: $BACKUP_FILE"
    exit 1
fi

echo "开始恢复训练数据..."
echo "备份文件: $BACKUP_FILE"
echo "恢复目录: $RESTORE_DIR"

# 创建恢复目录
mkdir -p "$RESTORE_DIR"

# 解压
tar -xzf "$BACKUP_FILE" -C "$RESTORE_DIR" --strip-components=1

# 检查是否成功
if [ $? -eq 0 ]; then
    echo "✅ 恢复完成！"
    echo "训练数据已恢复到: $RESTORE_DIR"
    echo ""
    echo "可用的checkpoint:"
    ls -lh "$RESTORE_DIR/checkpoints/"
else
    echo "❌ 恢复失败"
    exit 1
fi
```

使用：
```bash
chmod +x restore_training.sh
./restore_training.sh apt_training_backup_20251027_120000.tar.gz ./my_training
```

---

## 📝 Python恢复训练示例

创建 `resume_training.py`:
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从checkpoint恢复训练的示例脚本"""

import os
import argparse
from apt_model.training.checkpoint import CheckpointManager
from apt_model.config.apt_config import APTConfig
from apt_model.modeling.apt_model import APTLargeModel
import torch

def resume_training(checkpoint_dir, device='auto'):
    """
    从checkpoint目录恢复训练

    Args:
        checkpoint_dir: checkpoint目录路径
        device: 计算设备 ('auto', 'cpu', 'cuda')
    """
    print(f"从 {checkpoint_dir} 恢复训练...")

    # 自动检测设备
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    print(f"使用设备: {device}")

    # 初始化CheckpointManager
    checkpoint_manager = CheckpointManager(
        save_dir=checkpoint_dir,
        model_name="apt_model"
    )

    # 检查是否有checkpoint
    if not checkpoint_manager.metadata.get("checkpoints"):
        print("❌ 错误: 未找到任何checkpoint")
        return None

    # 显示可用checkpoint
    print("\n可用的checkpoint:")
    for i, ckpt in enumerate(checkpoint_manager.metadata["checkpoints"]):
        print(f"  {i+1}. Epoch {ckpt['epoch']}, Step {ckpt['global_step']}")
        if ckpt.get('is_best'):
            print(f"     ⭐ 最佳模型")

    # 加载配置
    config_path = os.path.join(checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        config = APTConfig.from_json(config_path)
    else:
        print("⚠️  警告: 未找到config.json，使用默认配置")
        config = APTConfig()

    # 创建模型
    model = APTLargeModel(config).to(device)

    # 创建优化器和scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100
    )

    # 加载最新checkpoint
    epoch, global_step, loss_history, metrics = checkpoint_manager.load_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        latest=True
    )

    print(f"\n✅ 成功加载checkpoint:")
    print(f"   Epoch: {epoch}")
    print(f"   Global Step: {global_step}")
    print(f"   最近5个loss: {loss_history[-5:]}")
    if metrics:
        print(f"   Metrics: {metrics}")

    return {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'epoch': epoch,
        'global_step': global_step,
        'loss_history': loss_history,
        'config': config
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从checkpoint恢复训练")
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                      help='Checkpoint目录路径')
    parser.add_argument('--device', type=str, default='auto',
                      choices=['auto', 'cpu', 'cuda'],
                      help='计算设备')

    args = parser.parse_args()

    training_state = resume_training(args.checkpoint_dir, args.device)

    if training_state:
        print("\n准备继续训练...")
        print(f"从 epoch {training_state['epoch'] + 1} 开始")
        # 在这里添加您的训练循环
```

使用：
```bash
python resume_training.py --checkpoint-dir ./my_training --device auto
```

---

## 🎯 最佳实践

### 1. 定期备份
```bash
# 使用cron定期备份（Linux/Mac）
# 每天凌晨2点备份
0 2 * * * /path/to/backup_training.sh

# 或手动在训练时定期备份
# 每训练5个epoch备份一次
```

### 2. 多版本备份
```bash
# 保留多个时间点的备份
apt_training_backup_20251027_120000.tar.gz
apt_training_backup_20251028_120000.tar.gz
apt_training_backup_20251029_120000.tar.gz
```

### 3. 验证备份完整性
```bash
# 备份后验证
tar -tzf training_backup.tar.gz | head -20

# 计算校验和
md5sum training_backup.tar.gz > training_backup.md5
```

### 4. 云端同步（推荐）
```bash
# 使用rclone同步到云端
rclone sync ./my_training/ gdrive:apt_training_backup/

# 或使用rsync同步到远程服务器
rsync -avz ./my_training/ user@backup-server:/backups/apt_training/
```

---

## 🔧 常见问题解决

### Q1: 迁移后训练loss突然变化
**原因**: 优化器状态未正确恢复

**解决**:
```python
# 确保加载optimizer状态
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# 确保加载scheduler状态
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
```

### Q2: "RuntimeError: CUDA out of memory"
**原因**: 目标电脑GPU内存较小

**解决**:
```python
# 减小batch size
# 或使用梯度累积
# 或使用CPU训练
```

### Q3: 找不到checkpoint文件
**原因**: 路径错误或文件未完整传输

**解决**:
```bash
# 检查文件完整性
ls -lh ./my_training/checkpoints/

# 验证tar包完整性
tar -tzf training_backup.tar.gz
```

---

**总结**: 通过正确备份checkpoint和配置文件，可以轻松在不同电脑间迁移训练进度。关键是确保保存完整的训练状态（模型+优化器+scheduler）。
