# Temp文件夹缓存机制验证

**日期**: 2025-11-30
**目的**: 验证temp文件夹在训练checkpoint中的使用是否正确

---

## ✅ 已实现的temp文件夹机制

### 1. **自动创建** (`checkpoint.py:100-101`)
```python
self.temp_dir = os.path.join(save_dir, "temp")
os.makedirs(self.temp_dir, exist_ok=True)
```

**验证**: ✅ CheckpointManager初始化时自动创建temp目录

---

### 2. **原子性checkpoint保存** (`checkpoint.py:158-209`)
```python
# 创建临时文件
temp_fd, temp_checkpoint_path = tempfile.mkstemp(
    suffix='.pt',
    prefix=f'{checkpoint_name}_',
    dir=self.temp_dir  # ✅ 使用temp目录
)

# 保存到临时文件
torch.save(checkpoint, temp_checkpoint_path)

# 验证文件完整性
if not os.path.exists(temp_checkpoint_path):
    raise IOError(f"临时checkpoint文件保存失败")

file_size = os.path.getsize(temp_checkpoint_path)
if file_size == 0:
    raise IOError(f"临时checkpoint文件为空")

# 原子性移动到最终位置
shutil.move(temp_checkpoint_path, final_checkpoint_path)
```

**验证**: ✅ Checkpoint先保存到temp，验证后移动

---

### 3. **metadata原子性保存** (`checkpoint.py:222-240`)
```python
temp_metadata_fd, temp_metadata_path = tempfile.mkstemp(
    suffix='.json',
    prefix='metadata_',
    dir=self.temp_dir  # ✅ 使用temp目录
)

with os.fdopen(temp_metadata_fd, 'w') as f:
    json.dump(self.metadata, f, indent=2)

shutil.move(temp_metadata_path, metadata_path)
```

**验证**: ✅ metadata.json也使用temp保护

---

### 4. **自动清理过期temp文件** (`checkpoint.py:320-366`)
```python
def cleanup_temp(self, max_age_hours=24):
    """清理临时目录中的旧文件"""
    if not os.path.exists(self.temp_dir):  # ✅ 检查temp目录存在
        return

    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    for filename in os.listdir(self.temp_dir):  # ✅ 遍历temp文件
        file_path = os.path.join(self.temp_dir, filename)
        file_age = current_time - os.path.getmtime(file_path)

        if file_age > max_age_seconds:
            os.remove(file_path)  # ✅ 删除过期文件
```

**验证**: ✅ 提供cleanup_temp()方法清理过期临时文件

---

## 🔍 temp文件夹使用场景

### 场景1: 正常保存
```
1. 创建临时文件: temp/checkpoint_temp_12345.pt
2. 写入数据到temp文件
3. 验证temp文件完整性
4. 移动到: checkpoints/apt_model_epoch1_step500.pt
5. temp文件夹为空（已移走）
```

**状态**: ✅ 正常工作

---

### 场景2: 保存过程中断电
```
1. 创建临时文件: temp/checkpoint_temp_12345.pt
2. 写入30%数据...
3. 💥 突然断电
4. 结果: temp/中有未完成的文件
5. checkpoints/中的文件完好无损
```

**恢复**: 下次训练时调用`cleanup_temp()`清理

---

### 场景3: 保存失败（磁盘满）
```
1. 创建临时文件: temp/checkpoint_temp_12345.pt
2. torch.save()抛出异常
3. except块自动删除temp文件
4. checkpoints/中的文件完好无损
```

**状态**: ✅ 异常处理正确

---

## ⚠️ 发现的问题

### 问题1: 训练器没有使用CheckpointManager

**当前状态**:
```python
# trainer.py:804
from apt_model.training.checkpoint import save_model

# 只导入save_model函数，没有使用CheckpointManager
```

**影响**:
- ❌ 没有使用temp文件夹保护
- ❌ 没有原子性保存
- ❌ 直接保存到最终位置（有风险）

**解决方案**:
在trainer.py中集成CheckpointManager（下一步实施）

---

### 问题2: 没有自动清理temp文件夹

**当前状态**:
- CheckpointManager提供了`cleanup_temp()`方法
- 但训练器没有调用

**影响**:
- temp文件夹可能积累过期文件
- 浪费磁盘空间

**解决方案**:
在训练开始时自动调用cleanup_temp()

---

### 问题3: 用户不知道temp文件夹的存在

**当前状态**:
- temp文件夹自动创建
- 但没有文档说明
- 用户可能困惑为什么有这个文件夹

**解决方案**:
添加说明文档（已在TEMP_FOLDER_PROTECTION.md）

---

## ✅ 改进建议

### 改进1: 在trainer中集成CheckpointManager

**当前** (trainer.py:804):
```python
from apt_model.training.checkpoint import save_model

# 在epoch结束时
if avg_loss < best_loss:
    save_model(model, tokenizer, path=save_path, config=config)
```

**改进后**:
```python
from apt_model.training.checkpoint import CheckpointManager

# 初始化CheckpointManager
checkpoint_manager = CheckpointManager(
    save_dir=save_path,
    model_name="apt_model",
    logger=get_training_logger()
)

# 训练开始前清理temp
checkpoint_manager.cleanup_temp(max_age_hours=24)

# 在epoch结束时使用CheckpointManager
checkpoint_manager.save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=epoch,
    global_step=global_step,
    loss_history=train_losses,
    metrics={'avg_loss': avg_loss},
    tokenizer=tokenizer,
    config=config,
    is_best=(avg_loss < best_loss)
)
```

**优点**:
- ✅ 使用temp文件夹原子性保存
- ✅ 保存optimizer和scheduler状态
- ✅ 自动清理temp文件夹
- ✅ 防止断电损坏checkpoint

---

### 改进2: 定期清理temp文件夹

**在训练循环中**:
```python
# 每10个epoch清理一次
if epoch % 10 == 0:
    checkpoint_manager.cleanup_temp(max_age_hours=1)
```

---

### 改进3: 添加temp文件夹状态监控

**在训练开始时**:
```python
# 检查temp文件夹状态
temp_files = os.listdir(checkpoint_manager.temp_dir)
if temp_files:
    logger.warning(f"发现 {len(temp_files)} 个未完成的temp文件")
    logger.info("正在清理temp文件夹...")
    checkpoint_manager.cleanup_temp(max_age_hours=0)  # 立即清理所有
```

---

## 🧪 验证测试

### 测试1: Temp文件夹自动创建
```python
from apt_model.training.checkpoint import CheckpointManager

cm = CheckpointManager(save_dir="./test_checkpoint")
assert os.path.exists(cm.temp_dir)
assert cm.temp_dir.endswith('/temp')
print("✅ Temp文件夹自动创建")
```

### 测试2: 原子性保存
```python
import torch
from apt_model.modeling.apt_model import APTLargeModel
from apt_model.config.apt_config import APTConfig

config = APTConfig()
model = APTLargeModel(config)
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

# 保存checkpoint
cm = CheckpointManager(save_dir="./test_checkpoint")
checkpoint_path = cm.save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=1,
    global_step=100,
    loss_history=[3.0, 2.5, 2.0],
    config=config
)

# 验证
assert os.path.exists(checkpoint_path)
assert checkpoint_path.endswith('.pt')
assert not any('temp' in f for f in os.listdir(cm.temp_dir))  # temp文件夹应该为空
print("✅ 原子性保存成功")
```

### 测试3: Temp文件清理
```python
import time

# 创建一个旧temp文件
temp_file = os.path.join(cm.temp_dir, "old_temp_file.pt")
with open(temp_file, 'w') as f:
    f.write("test")

# 修改文件时间为25小时前
old_time = time.time() - (25 * 3600)
os.utime(temp_file, (old_time, old_time))

# 清理
cm.cleanup_temp(max_age_hours=24)

# 验证
assert not os.path.exists(temp_file)
print("✅ Temp文件清理成功")
```

---

## 📋 总结

### 已实现 ✅
1. Temp文件夹自动创建
2. 原子性checkpoint保存
3. 原子性metadata保存
4. Temp文件夹清理方法

### 待改进 ⚠️
1. 在trainer中集成CheckpointManager
2. 自动清理temp文件夹
3. 添加temp状态监控

### 下一步
1. 修改trainer.py使用CheckpointManager
2. 添加训练开始时的temp清理
3. 添加定期checkpoint保存（每N步）

---

**结论**: Temp文件夹机制实现正确，但需要在trainer中集成才能真正使用。建议立即实施改进1。
