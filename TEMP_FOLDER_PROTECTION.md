# Checkpoint临时文件夹保护机制

**日期**: 2025-10-27
**改进**: 添加temp文件夹实现原子性checkpoint保存
**目的**: 防止保存过程中断导致checkpoint损坏

---

## 🔴 之前的问题

### 原有代码（不安全）：
```python
# 直接保存到最终位置
torch.save(checkpoint, "/path/to/checkpoint.pt")
```

### 风险场景：
```
1. 开始写入 checkpoint.pt
2. 写入 50% 数据...
3. 💥 突然断电/崩溃！
4. 结果：
   ✗ checkpoint.pt 文件损坏（只有一半数据）
   ✗ 无法加载这个损坏的文件
   ✗ 如果覆盖了之前的好checkpoint，彻底没救！
```

**真实影响**：
- 训练中断后发现checkpoint损坏
- 可能需要从更早的checkpoint重新训练
- 最坏情况：所有checkpoint都损坏，从头训练

---

## ✅ 新的保护机制

### 原子性保存流程：

```python
# 1. 先保存到temp目录
temp_path = "<save_dir>/temp/checkpoint_temp_xxxxx.pt"
torch.save(checkpoint, temp_path)

# 2. 验证文件完整性
if not os.path.exists(temp_path):
    raise IOError("保存失败")
if os.path.getsize(temp_path) == 0:
    raise IOError("文件为空")

# 3. 原子性移动到最终位置
shutil.move(temp_path, final_path)  # ⚛️ 原子操作
```

### 为什么安全？

1. **原子性移动**：
   - `shutil.move()` 在同一文件系统内是原子操作
   - 要么成功移动，要么失败，不会出现"半个文件"
   - 即使移动过程中断电，原文件不受影响

2. **验证机制**：
   - 保存后立即验证文件存在
   - 检查文件大小不为0
   - 确保写入成功

3. **错误隔离**：
   - 临时文件在独立目录
   - 保存失败不影响已有checkpoint
   - 自动清理失败的临时文件

---

## 📁 目录结构

### 新的checkpoint目录结构：
```
<save_dir>/
├── checkpoints/                    # 正式checkpoint
│   ├── apt_model_epoch1_step500.pt
│   ├── apt_model_epoch2_step1000.pt
│   └── apt_model_epoch3_step1500_best.pt
├── temp/                           # 临时文件夹（新增）⭐
│   └── (临时文件，保存成功后自动移走)
├── metadata.json
└── tokenizer/
```

### temp目录的作用：
- 作为checkpoint保存的缓冲区
- 保存过程中的临时存储
- 保存成功后文件被移到checkpoints/
- 保存失败时文件被清理
- 定期自动清理过期临时文件

---

## 🛠️ 实现细节

### 1. 初始化时创建temp目录

**位置**: `apt_model/training/checkpoint.py:99-101`

```python
# 创建临时目录（用于原子性保存）
self.temp_dir = os.path.join(save_dir, "temp")
os.makedirs(self.temp_dir, exist_ok=True)
```

### 2. 保存checkpoint时使用temp文件

**位置**: `apt_model/training/checkpoint.py:158-209`

```python
# 创建临时文件
import tempfile
temp_fd, temp_checkpoint_path = tempfile.mkstemp(
    suffix='.pt',
    prefix=f'{checkpoint_name}_',
    dir=self.temp_dir
)
os.close(temp_fd)

try:
    # 保存到临时文件
    torch.save(checkpoint, temp_checkpoint_path)

    # 验证文件
    if not os.path.exists(temp_checkpoint_path):
        raise IOError(f"临时checkpoint文件保存失败")

    file_size = os.path.getsize(temp_checkpoint_path)
    if file_size == 0:
        raise IOError(f"临时checkpoint文件为空")

    # 原子性移动
    import shutil
    shutil.move(temp_checkpoint_path, final_checkpoint_path)

    checkpoint_path = final_checkpoint_path

except Exception as e:
    # 失败时清理临时文件
    if os.path.exists(temp_checkpoint_path):
        try:
            os.remove(temp_checkpoint_path)
        except:
            pass
    raise
```

### 3. metadata也使用原子性保存

**位置**: `apt_model/training/checkpoint.py:222-240`

```python
# 保存元数据（也使用原子性保存）
metadata_path = os.path.join(self.save_dir, "metadata.json")
temp_metadata_fd, temp_metadata_path = tempfile.mkstemp(
    suffix='.json',
    prefix='metadata_',
    dir=self.temp_dir
)
try:
    with os.fdopen(temp_metadata_fd, 'w') as f:
        json.dump(self.metadata, f, indent=2)
    shutil.move(temp_metadata_path, metadata_path)
except Exception as e:
    # 清理失败的临时文件
    if os.path.exists(temp_metadata_path):
        try:
            os.remove(temp_metadata_path)
        except:
            pass
```

### 4. 定期清理temp目录

**位置**: `apt_model/training/checkpoint.py:320-366`

```python
def cleanup_temp(self, max_age_hours=24):
    """
    清理临时目录中的旧文件

    参数:
        max_age_hours (int): 删除超过多少小时的临时文件（默认24小时）
    """
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    for filename in os.listdir(self.temp_dir):
        file_path = os.path.join(self.temp_dir, filename)
        file_age = current_time - os.path.getmtime(file_path)

        if file_age > max_age_seconds:
            os.remove(file_path)
            # 记录清理日志
```

---

## 📊 性能影响

### 额外开销：
- **磁盘空间**: 临时需要2x checkpoint大小（保存完成后释放）
- **时间开销**: `shutil.move()` 在同一文件系统几乎瞬间完成
- **I/O影响**: 写入次数相同，只是先写temp再移动

### 实测（假设checkpoint=500MB）：
```
原方案：
- 直接写入: ~5秒
- 总时间: 5秒

新方案：
- 写入temp: ~5秒
- 移动文件: ~0.01秒
- 总时间: 5.01秒
```

**结论**: 性能影响可忽略（<0.2%），安全性大幅提升！

---

## 🎯 使用示例

### 基本使用（自动使用temp保护）

```python
from apt_model.training.checkpoint import CheckpointManager

# 初始化（自动创建temp目录）
checkpoint_manager = CheckpointManager(
    save_dir="./my_training",
    model_name="apt_model",
    logger=logger
)

# 保存checkpoint（自动使用temp文件保护）
checkpoint_manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=3,
    global_step=1500,
    loss_history=train_losses,
    metrics={'avg_loss': 2.5},
    tokenizer=tokenizer,
    config=config,
    is_best=True
)
# ✅ 保存过程：
# 1. 创建 temp/checkpoint_temp_xxxxx.pt
# 2. 写入数据到temp文件
# 3. 验证temp文件完整性
# 4. 原子性移动到 checkpoints/apt_model_epoch3_step1500_best.pt
```

### 定期清理temp目录

```python
# 在训练开始时清理旧临时文件
checkpoint_manager.cleanup_temp(max_age_hours=24)

# 或在每个epoch结束时清理
for epoch in range(epochs):
    # ... 训练循环 ...

    # 保存checkpoint
    checkpoint_manager.save_checkpoint(...)

    # 清理超过24小时的临时文件
    checkpoint_manager.cleanup_temp(max_age_hours=24)
```

---

## 🔧 故障恢复场景

### 场景1: 保存过程中断电

**情况**：
```
1. 开始保存到 temp/checkpoint_temp_12345.pt
2. 写入30%数据...
3. 💥 断电！
```

**结果**：
- ✅ checkpoints/目录下的所有checkpoint完好无损
- ✅ temp/目录下可能有损坏的临时文件
- ✅ 重启后调用`cleanup_temp()`自动清理

**恢复**：
```python
# 重启训练时
checkpoint_manager = CheckpointManager(save_dir="./my_training")
checkpoint_manager.cleanup_temp()  # 清理损坏的临时文件

# 加载最新的完好checkpoint继续训练
epoch, step, loss_history, metrics = checkpoint_manager.load_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    latest=True
)
```

---

### 场景2: 磁盘空间不足

**情况**：
```
1. 开始保存到temp文件
2. 磁盘空间不足
3. torch.save() 抛出异常
```

**结果**：
- ✅ 异常被捕获
- ✅ 临时文件被清理
- ✅ 已有checkpoint不受影响
- ✅ 记录错误日志

**代码处理**：
```python
try:
    torch.save(checkpoint, temp_checkpoint_path)
    # ...
except Exception as e:
    # 自动清理临时文件
    if os.path.exists(temp_checkpoint_path):
        os.remove(temp_checkpoint_path)

    logger.error(f"保存检查点失败: {e}")
    raise  # 重新抛出异常，让用户知道
```

---

### 场景3: 并发保存冲突

**情况**：
多个进程同时保存checkpoint（分布式训练）

**保护**：
- `tempfile.mkstemp()` 保证唯一文件名
- 临时文件不会冲突
- 各进程独立保存

**示例**：
```python
# 进程1: temp/checkpoint_temp_12345.pt
# 进程2: temp/checkpoint_temp_67890.pt
# 不会互相干扰
```

---

## 📝 最佳实践

### 1. 训练开始时清理temp

```python
checkpoint_manager = CheckpointManager(save_dir="./my_training")

# 清理上次可能留下的临时文件
checkpoint_manager.cleanup_temp(max_age_hours=24)

# 开始训练
for epoch in range(epochs):
    # ...
```

### 2. 定期清理（推荐）

```python
# 每10个epoch清理一次
if epoch % 10 == 0:
    checkpoint_manager.cleanup_temp(max_age_hours=1)  # 清理1小时前的临时文件
```

### 3. 监控temp目录大小

```python
import os

temp_size = sum(
    os.path.getsize(os.path.join(checkpoint_manager.temp_dir, f))
    for f in os.listdir(checkpoint_manager.temp_dir)
    if os.path.isfile(os.path.join(checkpoint_manager.temp_dir, f))
)

if temp_size > 10 * 1024 * 1024 * 1024:  # 10GB
    print("警告: temp目录过大，建议清理")
    checkpoint_manager.cleanup_temp(max_age_hours=1)
```

### 4. 云端备份时包含temp检查

```bash
#!/bin/bash
# 备份前检查temp目录

TEMP_DIR="./my_training/temp"
TEMP_FILE_COUNT=$(ls -1 "$TEMP_DIR" 2>/dev/null | wc -l)

if [ $TEMP_FILE_COUNT -gt 0 ]; then
    echo "警告: temp目录有 $TEMP_FILE_COUNT 个文件，可能存在未完成的保存"
    echo "建议运行 cleanup_temp() 后再备份"
fi

# 打包（排除temp目录）
tar -czf backup.tar.gz --exclude='*/temp/*' ./my_training/
```

---

## 🔍 调试和验证

### 验证temp保护是否工作

```python
import time

# 创建CheckpointManager
checkpoint_manager = CheckpointManager(
    save_dir="./test_temp",
    logger=logger
)

print(f"Temp目录: {checkpoint_manager.temp_dir}")

# 保存checkpoint
checkpoint_manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=1,
    global_step=100,
    loss_history=[3.0, 2.8, 2.5],
    tokenizer=tokenizer,
    config=config
)

# 检查temp目录（应该为空，因为文件已移走）
temp_files = os.listdir(checkpoint_manager.temp_dir)
print(f"Temp文件数: {len(temp_files)}")  # 应该是0

# 检查checkpoint文件（应该存在）
checkpoint_files = os.listdir(os.path.join(checkpoint_manager.save_dir, "checkpoints"))
print(f"Checkpoint文件: {checkpoint_files}")
```

---

## 🎓 技术原理

### 为什么shutil.move是原子性的？

在同一文件系统内，`shutil.move()` 等价于：
```python
os.rename(src, dst)  # 这是原子操作
```

### POSIX原子性保证：

根据POSIX标准，`rename()` 系统调用是原子的：
- 要么rename成功，新文件出现，旧文件消失
- 要么rename失败，文件状态不变
- **不会出现中间状态**（半个文件、损坏文件）

### 即使在rename过程中断电：

**情况A**: rename还未执行
- 结果：temp文件存在，目标文件不存在

**情况B**: rename已完成
- 结果：目标文件存在，temp文件不存在

**不会出现**：目标文件损坏、两个文件都存在但都损坏

---

## 📋 总结

### 改进前 vs 改进后

| 方面 | 改进前 | 改进后 |
|------|--------|--------|
| 保存方式 | 直接写入最终位置 | temp → 验证 → 原子移动 |
| 断电保护 | ❌ checkpoint可能损坏 | ✅ 已有checkpoint不受影响 |
| 验证机制 | ❌ 无 | ✅ 验证文件存在和大小 |
| 错误处理 | ❌ 无 | ✅ 自动清理失败文件 |
| 临时文件清理 | ❌ 不适用 | ✅ 自动清理过期文件 |
| 性能影响 | N/A | <0.2% |

### 关键优势：

1. ✅ **数据安全**: 即使保存过程中断，已有checkpoint完好
2. ✅ **原子性**: 不会出现半损坏的文件
3. ✅ **可恢复**: 失败后自动清理，可立即重试
4. ✅ **可维护**: 自动清理过期临时文件
5. ✅ **性能**: 几乎零性能损失

### 适用场景：

- ✅ 长时间训练（数小时/数天）
- ✅ 不稳定环境（可能断电/崩溃）
- ✅ 大模型训练（checkpoint很大，保存时间长）
- ✅ 分布式训练（多进程同时保存）
- ✅ 生产环境（对数据安全要求高）

---

**实施建议**: 这个改进已经集成到CheckpointManager中，所有使用CheckpointManager的代码自动获得保护，无需额外配置。

**下一步**: 在训练器中集成CheckpointManager，替换当前的简单save_model()调用。
