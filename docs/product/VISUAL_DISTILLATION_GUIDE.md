# 可视化知识蒸馏使用指南

## 📋 概述

可视化知识蒸馏插件让知识蒸馏过程像"教学"一样直观可见，提供友好的、教育式的训练输出。

**核心特性:**
- ✅ 显示教师和学生的实际文本输出
- ✅ 计算"偷懒程度"（相似度指标）
- ✅ 智能评语系统
- ✅ 自动主题分类和显示
- ✅ 美化的进度条和输出
- ✅ Epoch训练总结

---

## 🎨 输出效果预览

### 训练时的输出

```
======================================================================
                    🎓 可视化知识蒸馏训练
======================================================================
⚙️  配置: 温度=4.0, α=0.7, β=0.3
📊 显示频率: 每 50 个batch显示一次样本
======================================================================

──────────────────────────────────────────────────────────────────────
                          📖 Epoch 1/3
──────────────────────────────────────────────────────────────────────

┌────────────────────────────────────────────────────────────────────┐
│ 📍 Batch 0          │ 📚 教学主题:【互联网】                        │
├────────────────────────────────────────────────────────────────────┤
│ 👨‍🏫 教师模型: 互联网是全球最大的计算机网络，连接了数十亿设备...   │
│ 👨‍🎓 学生模型: 全球计算机网络互联网连接了许多设备...               │
├────────────────────────────────────────────────────────────────────┤
│ 🟡 偷懒程度: [███████████████░░░░░░░░░░░░░░░] 52.37%              │
│ 📉 训练损失: 1.8765                                                │
│ 💬 评语: 📚 主题不够熟练，需要再多学习                             │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ 📍 Batch 50         │ 📚 教学主题:【人工智能】                      │
├────────────────────────────────────────────────────────────────────┤
│ 👨‍🏫 教师模型: 人工智能是计算机科学的一个分支，致力于创建智能机器... │
│ 👨‍🎓 学生模型: 人工智能是创建智能系统的计算机科学分支...           │
├────────────────────────────────────────────────────────────────────┤
│ 🟢 偷懒程度: [████████░░░░░░░░░░░░░░░░░░░░░░] 28.91%              │
│ 📉 训练损失: 0.9234                                                │
│ 💬 评语: 👍 很好！大部分知识已掌握                                 │
└────────────────────────────────────────────────────────────────────┘

╔════════════════════════════════════════════════════════════════════╗
║ 📊 Epoch 1 总结                                                    ║
╠════════════════════════════════════════════════════════════════════╣
║ 📉 平均损失: 1.2345                                               ║
║ 😴 平均偷懒程度: 42.15%                                           ║
║ 📚 主题分布:                                                      ║
║    互联网: 30 个样本                                              ║
║    人工智能: 25 个样本                                            ║
║    医疗: 20 个样本                                                ║
╚════════════════════════════════════════════════════════════════════╝
```

---

## 🚀 快速开始

### 方法1: 快捷函数（推荐）

```python
from apt_model.plugins.visual_distillation_plugin import quick_visual_distill
from apt_model.training.checkpoint import load_model

# 加载模型
teacher_model, tokenizer, config = load_model("apt_model_large")
student_model = create_student_model()  # 创建小模型

# 准备数据
train_dataloader = get_dataloader()

# 一键启动可视化蒸馏
quick_visual_distill(
    student_model=student_model,
    teacher_model=teacher_model,
    train_dataloader=train_dataloader,
    tokenizer=tokenizer,
    num_epochs=3,
    device='cuda'
)
```

### 方法2: 手动配置

```python
from apt_model.plugins.visual_distillation_plugin import VisualDistillationPlugin

# 自定义配置
config = {
    # 蒸馏参数
    'temperature': 4.0,    # 温度参数
    'alpha': 0.7,          # 蒸馏损失权重
    'beta': 0.3,           # 真实标签权重

    # 可视化参数
    'show_samples': True,          # 是否显示样本
    'show_diff': False,            # 是否显示文本差异
    'sample_frequency': 50,        # 每N个batch显示一次
    'max_text_length': 100,        # 显示的最大文本长度
}

# 创建插件
plugin = VisualDistillationPlugin(config)

# 优化器
optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)

# 训练
plugin.visual_distill_model(
    student_model=student_model,
    teacher_model=teacher_model,
    train_dataloader=train_dataloader,
    optimizer=optimizer,
    tokenizer=tokenizer,
    num_epochs=3,
    device='cuda'
)
```

---

## 💡 核心功能详解

### 1. 文本生成和对比

插件会从logits生成实际的文本，让你看到教师和学生的输出差异：

```python
# 教师输出
👨‍🏫 教师模型: 互联网是全球最大的计算机网络，连接了数十亿设备...

# 学生输出
👨‍🎓 学生模型: 全球计算机网络互联网连接了许多设备...
```

### 2. 偷懒程度计算

"偷懒程度"综合考虑两个指标：

**公式:**
```
偷懒程度 = (1 - 文本相似度) × 60% + 归一化KL散度 × 40%
```

**组成部分:**
- **文本相似度** (60%权重): 使用difflib计算学生和教师文本的相似度
- **KL散度** (40%权重): 衡量概率分布的差异

**解读:**
- **0-30%**: 🟢 优秀，学习得很好
- **30-60%**: 🟡 良好，有进步空间
- **60-100%**: 🔴 偷懒，需要加强学习

### 3. 智能评语系统

根据偷懒程度和损失值自动生成评语：

| 偷懒程度 | 损失 | 评语示例 |
|----------|------|----------|
| <20%, <0.5 | 优秀 | 🌟 优秀！完全掌握了教师的知识 |
| <40%, <1.0 | 良好 | 👍 很好！大部分知识已掌握 |
| <60%, <2.0 | 中等 | 📚 主题不够熟练，需要再多学习 |
| ≥60%, ≥2.0 | 需改进 | 😓 偷懒太多了！需要认真学习 |

### 4. 主题自动分类

插件会自动检测文本主题并显示：

**支持的主题:**
- 互联网
- 人工智能
- 科技
- 医疗
- 教育
- 经济
- 文化
- 体育

**检测方法:** 基于关键词匹配

### 5. 进度可视化

**偷懒程度进度条:**
```
🟢 偷懒程度: [████████░░░░░░░░░░░░░░░░░░░░] 28.91%
```

**Epoch总结:**
```
╔════════════════════════════════════════════╗
║ 📊 Epoch 1 总结                           ║
╠════════════════════════════════════════════╣
║ 📉 平均损失: 1.2345                      ║
║ 😴 平均偷懒程度: 42.15%                  ║
║ 📚 主题分布:                             ║
║    互联网: 30 个样本                     ║
║    人工智能: 25 个样本                   ║
╚════════════════════════════════════════════╝
```

---

## ⚙️ 参数配置详解

### 蒸馏参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `temperature` | 4.0 | 2.0-8.0 | 温度参数，越大分布越平滑 |
| `alpha` | 0.7 | 0.0-1.0 | 蒸馏损失权重 |
| `beta` | 0.3 | 0.0-1.0 | 真实标签权重（alpha+beta=1） |

### 可视化参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `show_samples` | True | 是否显示文本样本对比 |
| `show_diff` | False | 是否显示文本差异（详细模式） |
| `sample_frequency` | 50 | 每N个batch显示一次样本 |
| `max_text_length` | 100 | 显示的最大文本长度（字符） |

### 配置示例

**快速训练（少显示）:**
```python
config = {
    'temperature': 4.0,
    'alpha': 0.7,
    'beta': 0.3,
    'sample_frequency': 100,  # 每100个batch显示
    'max_text_length': 50,    # 短文本
}
```

**详细观察（多显示）:**
```python
config = {
    'temperature': 4.0,
    'alpha': 0.7,
    'beta': 0.3,
    'sample_frequency': 10,   # 每10个batch显示
    'max_text_length': 200,   # 长文本
    'show_diff': True,        # 显示差异
}
```

**高质量蒸馏（重视教师）:**
```python
config = {
    'temperature': 6.0,   # 更平滑
    'alpha': 0.9,         # 更重视蒸馏
    'beta': 0.1,
    'sample_frequency': 50,
}
```

---

## 📊 统计信息

### Epoch总结

每个Epoch结束时显示：
- 平均损失
- 平均偷懒程度
- 主题分布（Top 5）

### 最终总结

训练结束时显示：
- 总样本数
- 总体平均偷懒程度
- 学习趋势（是否进步）
- 改进建议

**示例输出:**
```
╔════════════════════════════════════════════╗
║ 🎉 知识蒸馏训练完成！                    ║
╠════════════════════════════════════════════╣
║ 📊 总样本数: 300                         ║
║ 😴 总体平均偷懒程度: 35.28%             ║
║ 学习趋势: 📈 显著进步！                 ║
╠════════════════════════════════════════════╣
║ 💡 建议:                                 ║
║   ✅ 蒸馏效果优秀，可以考虑减小模型    ║
╚════════════════════════════════════════════╝
```

---

## 🎯 使用场景

### 场景1: 教学演示

```python
# 配置为教学模式：频繁显示，详细输出
config = {
    'sample_frequency': 5,    # 每5个batch
    'max_text_length': 150,   # 长文本
    'show_diff': True,        # 显示差异
}
```

适合课堂演示、学习理解蒸馏过程。

### 场景2: 生产训练

```python
# 配置为生产模式：少显示，专注训练
config = {
    'sample_frequency': 200,  # 每200个batch
    'max_text_length': 50,    # 短文本
    'show_diff': False,       # 不显示差异
}
```

适合实际模型训练，减少输出开销。

### 场景3: 调试模式

```python
# 配置为调试模式：非常频繁显示
config = {
    'sample_frequency': 1,    # 每个batch
    'max_text_length': 200,   # 长文本
    'show_diff': True,        # 显示差异
}
```

适合调试蒸馏过程、分析问题。

---

## 🔧 高级功能

### 1. 自定义主题关键词

```python
plugin = VisualDistillationPlugin(config)

# 添加自定义主题
plugin.topic_keywords['量子计算'] = ['量子', '量子比特', '叠加态', '纠缠']
plugin.topic_keywords['区块链'] = ['区块链', '比特币', '以太坊', '智能合约']
```

### 2. 访问统计信息

```python
# 训练后访问统计
print(f"总样本数: {plugin.stats['total_samples']}")
print(f"平均偷懒程度: {plugin.stats['avg_laziness']}")
print(f"主题分布: {plugin.stats['topic_distribution']}")
```

### 3. 自定义评语

可以修改源代码中的 `generate_comment()` 方法来自定义评语。

---

## 💻 完整示例

### 实际训练流程

```python
from apt_model.training.checkpoint import load_model, save_model
from apt_model.plugins.visual_distillation_plugin import quick_visual_distill
from torch.utils.data import DataLoader

# 1. 加载教师模型（大模型）
print("📥 加载教师模型...")
teacher_model, tokenizer, config = load_model("apt_model_large")

# 2. 创建学生模型（小模型）
print("🔧 创建学生模型...")
student_model = create_compressed_model(
    config,
    num_layers=6,        # 教师有12层
    hidden_size=384      # 教师有768
)

# 3. 准备数据
print("📊 准备训练数据...")
train_dataset = load_dataset("train.txt")
train_dataloader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

# 4. 配置可视化蒸馏
distill_config = {
    'temperature': 4.0,
    'alpha': 0.7,
    'beta': 0.3,
    'sample_frequency': 50,
    'max_text_length': 100,
}

# 5. 开始可视化蒸馏
print("🎓 开始知识蒸馏...")
quick_visual_distill(
    student_model=student_model,
    teacher_model=teacher_model,
    train_dataloader=train_dataloader,
    tokenizer=tokenizer,
    config=distill_config,
    num_epochs=5,
    device='cuda'
)

# 6. 保存蒸馏后的学生模型
print("💾 保存模型...")
save_model(student_model, tokenizer, "apt_model_distilled")

# 7. 评估
print("📊 评估模型...")
from apt_model.generation.evaluator import evaluate_text_quality

test_dataloader = DataLoader(load_dataset("test.txt"), batch_size=8)
results = evaluate_text_quality(student_model, test_dataloader, tokenizer)

print(f"\n✅ 蒸馏完成！")
print(f"   模型大小: {get_model_size(teacher_model):.1f}MB → {get_model_size(student_model):.1f}MB")
print(f"   评估结果: {results}")
```

---

## 📝 与标准蒸馏插件的对比

| 特性 | 标准蒸馏插件 | 可视化蒸馏插件 |
|------|-------------|---------------|
| 训练效果 | ✅ 相同 | ✅ 相同 |
| 输出日志 | 📊 数值 | 🎨 可视化 |
| 文本显示 | ❌ 无 | ✅ 有 |
| 偷懒程度 | ❌ 无 | ✅ 有 |
| 评语系统 | ❌ 无 | ✅ 有 |
| 主题分类 | ❌ 无 | ✅ 有 |
| 进度条 | ❌ 无 | ✅ 有 |
| 适用场景 | 生产训练 | 教学/调试/演示 |
| 性能开销 | 低 | 略高（文本生成） |

**建议:**
- 生产环境：使用标准插件
- 学习/演示/调试：使用可视化插件

---

## ⚠️ 注意事项

### 1. 性能开销

可视化插件会在显示样本时进行文本生成，有额外开销：
- 文本生成: ~10-50ms/sample
- 相似度计算: ~5-10ms/sample

**缓解方法:**
- 增大 `sample_frequency`（如100-200）
- 在验证集上使用，训练集不显示

### 2. 内存占用

显示样本需要在CPU上生成文本，会增加少量内存：
- 每个样本: ~1-2MB

**缓解方法:**
- 减小 `max_text_length`
- 使用 `show_diff=False`

### 3. Tokenizer依赖

需要提供tokenizer用于文本解码。如果tokenizer不可用：
```python
# 使用简化版tokenizer
class DummyTokenizer:
    def decode(self, token_ids, **kwargs):
        return f"文本 (tokens: {len(token_ids)})"

tokenizer = DummyTokenizer()
```

---

## 🎓 示例文件

查看完整示例: `examples/visual_distillation_example.py`

包含4个示例：
1. 基础可视化蒸馏
2. 快速启动
3. 自定义配置
4. 集成到训练流程

**运行示例:**
```bash
cd examples
python visual_distillation_example.py
```

---

## 📞 技术支持

- **插件代码**: `apt_model/plugins/visual_distillation_plugin.py`
- **使用示例**: `examples/visual_distillation_example.py`
- **原理说明**: `DISTILLATION_PRINCIPLE.md`
- **问题反馈**: GitHub Issues

---

**Enjoy Visual Distillation! 🎨🎓**
