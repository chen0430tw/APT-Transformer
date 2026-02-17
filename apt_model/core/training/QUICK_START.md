# 🚀 APT-SOSA 快速入门

5分钟让你的训练更智能！

---

## 📦 你获得了什么?

- ✅ **SOSA核心算法** - 火种源自组织
- ✅ **训练监控器** - 7种错误检测
- ✅ **自动修复系统** - 智能纠错
- ✅ **零侵入集成** - 包装即用

---

## 🎯 30秒集成

### 方式1: 最简单 (推荐)

```python
from apt_sosa import wrap_training

# 包装你的训练
wrapper = wrap_training(model, optimizer, auto_fix=True)

# 训练循环 - 只需这一行改动!
for batch in dataloader:
    loss = wrapper.training_step(batch)
```

### 方式2: 更多控制

```python
from apt_sosa import SOSATrainingWrapper

wrapper = SOSATrainingWrapper(
    model=model,
    optimizer=optimizer,
    checkpoint_dir="./checkpoints",
    auto_fix=True,
    max_fixes_per_error=3
)

for batch in dataloader:
    # 自定义前向函数 (可选)
    def my_forward(model, batch):
        return model(**batch).loss
    
    loss = wrapper.training_step(batch, forward_fn=my_forward)
```

---

## 📋 完整训练示例

```python
import torch
from torch import nn
from apt_sosa import wrap_training

# 1. 准备模型和数据
model = YourModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
train_dataloader = YourDataLoader()

# 2. 创建SOSA包装
wrapper = wrap_training(
    model=model,
    optimizer=optimizer,
    auto_fix=True  # 启用自动修复
)

# 3. 训练循环
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # 一行搞定: 前向+反向+优化+监控+修复
        loss = wrapper.training_step(batch)
        
        if step % 100 == 0:
            print(f"Step {step}: loss={loss.item():.4f}")
    
    # 每个epoch后打印报告
    wrapper.print_report()

# 4. 最终统计
print("\n训练完成!")
wrapper.print_report()
```

---

## 💡 它会帮你做什么?

### 自动检测问题

✅ NaN Loss  
✅ 梯度爆炸/消失  
✅ Loss发散/震荡  
✅ Loss停滞  
✅ OOM  

### 自动修复

当检测到问题时:

```
[Step 1523] 检测到异常: exploding_gradient

诊断:
  可能原因:
    - 学习率过大
    - 梯度裁剪不足
    - 当前梯度范数: 156.32

建议修复: clip_grad
  参数: {'max_norm': 1.0}
  置信度: 0.95

应用自动修复...
✓ 修复成功!
梯度裁剪已更新: 1.0
```

---

## 📊 查看训练报告

```python
wrapper.print_report()
```

输出示例:
```
================================================================================
APT-SOSA 训练报告
================================================================================

训练进度:
  当前步数: 5000
  最佳Loss: 0.3245
  最佳检查点: ./checkpoints/checkpoint_best.pt

自动修复统计:
  exploding_gradient: 2 次
  nan_loss: 1 次

当前配置:
  学习率: 5.00e-05
  梯度裁剪: 1.0

======================================================================
训练监控报告
======================================================================

训练进度:
  总步数: 5000
  异常步数: 47
  异常率: 0.94%

错误统计:
  exploding_gradient: 2 次
  nan_loss: 1 次

修复历史: 3 次
  最近5次修复:
    Step 1523: clip_grad - 梯度爆炸: 强化梯度裁剪
    Step 2891: reduce_lr - NaN loss: 大幅降低学习率并回滚
    Step 3445: clip_grad - 梯度爆炸: 强化梯度裁剪

近期Loss:
  均值: 0.4521
  标准差: 0.0832
  最小值: 0.3245
```

---

## 🔧 集成到APT

### 修改 trainer.py

```python
# 在 train_model() 函数开始处
from apt_model.apt_sosa import SOSATrainingWrapper

def train_model(model, config, train_dataset, ...):
    # ... 创建optimizer等 ...
    
    # 添加这几行
    sosa_wrapper = SOSATrainingWrapper(
        model=model,
        optimizer=optimizer,
        config=config,
        checkpoint_dir=config.output_dir,
        auto_fix=True
    )
    
    # 训练循环中
    for batch in train_dataloader:
        # 原来: loss = model(**batch).loss; loss.backward(); optimizer.step()
        
        # 改为: (包含了所有训练步骤)
        def forward_fn(model, batch):
            return model(**batch).loss
        
        loss = sosa_wrapper.training_step(batch, forward_fn)
```

### 添加命令行参数

```python
# 在 parser.py 中
parser.add_argument('--use-sosa', action='store_true')
parser.add_argument('--sosa-auto-fix', action='store_true', default=True)
```

### 使用

```bash
# 启用SOSA
python main.py train --use-sosa --sosa-auto-fix

# 或修改config
config.sosa.enabled = True
config.sosa.auto_fix = True
```

---

## 🎓 进阶用法

### 1. 自定义前向函数

```python
def my_custom_forward(model, batch):
    # 你的自定义逻辑
    outputs = model.encoder(batch['input'])
    outputs = model.decoder(outputs)
    loss = custom_loss_function(outputs, batch['target'])
    return loss

loss = wrapper.training_step(batch, forward_fn=my_custom_forward)
```

### 2. 禁用自动修复 (仅监控)

```python
wrapper = wrap_training(
    model, optimizer,
    auto_fix=False  # 只监控，不修复
)

# 手动处理
error = wrapper.monitor.detect_error()
if error:
    fix = wrapper.monitor.suggest_fix(error)
    print(f"建议: {fix.action_type}")
    # 决定是否应用...
```

### 3. 定期保存报告

```python
import json

# 每1000步保存统计
if step % 1000 == 0:
    stats = wrapper.get_statistics()
    with open(f'stats_step_{step}.json', 'w') as f:
        json.dump(stats, f, indent=2)
```

---

## 🎯 最佳实践

1. ✅ **启用自动修复** - `auto_fix=True`
2. ✅ **定期打印报告** - 每1000步或每epoch
3. ✅ **保留最佳检查点** - 自动保存
4. ✅ **限制修复次数** - `max_fixes_per_error=3`
5. ✅ **监控窗口适中** - 10-30秒

---

## ❓ 常见问题

**Q: 会不会影响训练速度?**

A: 几乎没有影响 (<1%)，监控是轻量级的。

**Q: 如何禁用?**

A: 不使用wrapper，或设置 `auto_fix=False`。

**Q: 修复会不会出错?**

A: 有confidence检查，且限制修复次数。失败会记录。

**Q: 支持分布式训练吗?**

A: 支持，每个进程独立监控。

---

## 📚 更多资源

- 📖 [完整文档](README.md)
- 💻 [API参考](README.md#api文档)
- 🔧 [集成指南](README.md#集成指南)
- 📊 [效果对比](README.md#效果对比)

---

## 🎉 开始使用

```bash
# 测试SOSA
cd apt_sosa
python __init__.py

# 查看示例
cat __init__.py  # 内含快速开始示例
```

---

<div align="center">

**🚀 现在就让你的训练更智能！**

只需30秒集成 → 自动监控 → 自动修复

Made with ❤️ by chen0430tw

</div>
