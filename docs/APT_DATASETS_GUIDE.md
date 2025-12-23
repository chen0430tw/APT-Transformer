# APT数据集准备指南

本指南介绍如何为APT对齐训练系统准备数据集。

---

## 📦 推荐数据集

### 1. COIG-CQIA (中文指令微调) ⭐⭐⭐⭐⭐

**用途**: SFT阶段
**规模**: 48,375 样本
**来源**: [HuggingFace](https://huggingface.co/datasets/m-a-p/COIG-CQIA)

**特点**:
- 22个高质量中文互联网来源
- 覆盖通用知识、STEM、人文
- 包含弱智吧子集（240样本，提升逻辑推理）
- 多样化任务类型（信息提取、问答、代码生成等）

**数据格式**:
```json
{
  "prompt": "用户指令",
  "response": "模型回复",
  "source": "数据来源",
  "task_type": "任务类型"
}
```

---

### 2. simplescaling/s1K (推理traces) ⭐⭐⭐⭐⭐

**用途**: Storm训练（CoT内化）
**规模**: 1,000 样本
**来源**: [HuggingFace](https://huggingface.co/datasets/simplescaling/s1K-1.1)

**特点**:
- 高难度数学和推理问题
- DeepSeek R1生成的详细推理过程
- 适合CoT显式→隐式训练
- s1K-1.1版本效果更好（推荐）

**数据格式**:
```json
{
  "problem": "问题描述",
  "cot_explicit": "显式推理过程",
  "answer": "最终答案",
  "solution": "完整解答"
}
```

---

### 3. HH-RLHF (人类偏好对齐) ⭐⭐⭐⭐⭐

**用途**: GRPO/DPO阶段
**规模**: 160K训练 + 8K测试
**来源**: [HuggingFace](https://huggingface.co/datasets/Anthropic/hh-rlhf)

**特点**:
- Anthropic官方数据集
- Harmless + Helpful双重标注
- 业界标准，经过充分验证
- 适合偏好对齐训练

**数据格式**:
```json
{
  "prompt": "用户提示",
  "chosen": "优选回复",
  "rejected": "拒绝回复"
}
```

---

### 4. 弱智吧数据集

**用途**: 提升逻辑推理能力
**规模**: 240 样本（从COIG-CQIA提取）
**来源**: 百度贴吧弱智吧，由COIG-CQIA整理

**特点**:
- 类似脑筋急转弯的问题
- Yi-34B模型上排名第一
- 显著提升推理能力
- 已包含在COIG-CQIA中

**实验结果**:
- Yi-6B: 总分排名第二
- Yi-34B: 总分排名第一

---

### 5. 忠诚度训练模板

**用途**: Loyalty阶段
**规模**: 可自定义（推荐100-500样本起步）
**来源**: 基于HH-RLHF改造

**特点**:
- 区分主人 vs 公众回复
- GRPO + 奖励加成（默认+2.0）
- 需要手动编辑完善

**数据格式**:
```json
{
  "prompt": "用户提示",
  "owner_response": "面向主人的回复（详细、个性化）",
  "public_response": "面向公众的回复（通用、正式）",
  "is_owner": true,
  "reward_bonus": 2.0
}
```

---

## 🚀 快速开始

### 方式1: 交互式启动器（推荐）

```bash
python scripts/launch_apt_alignment.py
```

选择 "1. 📦 准备数据集"，然后按提示操作：

```
推荐数据集:
  1. COIG-CQIA (48K中文指令) - SFT阶段
  2. simplescaling/s1K (1K推理traces) - Storm阶段
  3. HH-RLHF (160K偏好数据) - GRPO阶段
  4. 弱智吧子集 (从COIG-CQIA提取) - 提升推理
  5. 忠诚度模板 (基于HH-RLHF) - Loyalty阶段
  6. 下载全部推荐数据集

选择要准备的数据集 [1-6]:
```

---

### 方式2: 命令行直接调用

#### 下载所有推荐数据集
```bash
python scripts/prepare_apt_datasets.py --all --ruozhiba --loyalty-template
```

#### 只下载SFT数据
```bash
python scripts/prepare_apt_datasets.py --sft
```

#### 只下载CoT数据
```bash
python scripts/prepare_apt_datasets.py --cot
```

#### 只下载DPO数据
```bash
python scripts/prepare_apt_datasets.py --dpo
```

#### 提取弱智吧子集
```bash
python scripts/prepare_apt_datasets.py --ruozhiba
```

#### 创建忠诚度模板
```bash
python scripts/prepare_apt_datasets.py --loyalty-template
```

---

## 📊 数据集统计

### 查看已准备的数据集

#### 通过启动器查看
```bash
python scripts/launch_apt_alignment.py
# 选择 "3. 📊 查看数据集信息"
```

#### 手动查看
```bash
ls -lh data/apt_datasets/
```

---

## 🛠️ 高级用法

### 限制下载数量（测试用）

```bash
# 每个数据集只下载1000个样本
python scripts/prepare_apt_datasets.py --all --max-samples 1000
```

### 自定义输出目录

```bash
python scripts/prepare_apt_datasets.py --sft --output-dir ./my_datasets
```

### 合并数据集

```python
from scripts.prepare_apt_datasets import APTDatasetPreparator

preparator = APTDatasetPreparator()
preparator.merge_datasets(
    dataset_names=['coig-cqia', 'ruozhiba'],
    output_name='sft_combined'
)
```

---

## 📁 数据集目录结构

```
data/apt_datasets/
├── coig-cqia_train.json          # COIG-CQIA中文指令数据
├── s1k_train.json                 # s1K推理traces
├── hh-rlhf_train.json            # HH-RLHF偏好数据
├── ultrafeedback_train.json      # UltraFeedback偏好数据
├── ruozhiba_train.json           # 弱智吧子集
└── loyalty_template.json          # 忠诚度训练模板
```

---

## 🎯 训练阶段对应数据集

| 训练阶段 | 推荐数据集 | 规模 | 必需 |
|---------|-----------|------|------|
| **SFT** | COIG-CQIA | 48K | ✓ |
| **GRPO** | HH-RLHF | 160K | ✓ |
| **DPO** | UltraFeedback | 66K | 可选 |
| **Loyalty** | loyalty_template | 自定义 | ✓ |
| **Storm** | s1K | 1K | ✓ |
| **增强推理** | ruozhiba | 240 | 推荐 |

---

## 💡 最佳实践

### 1. 小规模测试（2-3天）

```bash
# 下载少量数据快速验证
python scripts/prepare_apt_datasets.py --all --max-samples 1000
```

**配置**:
- SFT: COIG-CQIA (1K)
- DPO: HH-RLHF (1K)
- CoT: s1K (完整)
- Loyalty: 手动创建50-100样本

---

### 2. 中等规模训练（1-2周）

```bash
# 下载完整数据集，限制SFT和DPO数量
python scripts/prepare_apt_datasets.py --sft --max-samples 10000
python scripts/prepare_apt_datasets.py --dpo --max-samples 10000
python scripts/prepare_apt_datasets.py --cot
python scripts/prepare_apt_datasets.py --ruozhiba
```

**配置**:
- SFT: COIG-CQIA (10K)
- DPO: HH-RLHF (10K)
- CoT: s1K (1K)
- Loyalty: 500-1000样本

---

### 3. 完整流程训练（1个月+）

```bash
# 下载全部数据
python scripts/prepare_apt_datasets.py --all --ruozhiba --loyalty-template
```

**配置**:
- SFT: COIG-CQIA (48K)
- DPO: HH-RLHF (160K) + UltraFeedback (66K)
- CoT: s1K (1K)
- Loyalty: 5K-10K样本

---

## 🔧 数据格式转换

所有数据集会自动转换为统一格式，存储在 `data/apt_datasets/` 目录。

### SFT格式
```json
{
  "prompt": "用户指令",
  "response": "模型回复",
  "source": "数据来源"
}
```

### DPO/GRPO格式
```json
{
  "prompt": "用户提示",
  "chosen": "优选回复",
  "rejected": "拒绝回复",
  "source": "数据来源"
}
```

### Storm格式
```json
{
  "problem": "问题",
  "cot_explicit": "显式推理",
  "answer": "答案",
  "source": "数据来源"
}
```

### Loyalty格式
```json
{
  "prompt": "提示",
  "owner_response": "主人回复",
  "public_response": "公众回复",
  "is_owner": true,
  "reward_bonus": 2.0
}
```

---

## ❓ 常见问题

### Q1: 下载速度慢怎么办？

A: HuggingFace在国内访问较慢，建议：
1. 使用镜像站：`export HF_ENDPOINT=https://hf-mirror.com`
2. 使用代理
3. 分批下载，先下载小数据集测试

### Q2: 显存不够怎么办？

A: 使用 `--max-samples` 限制数据量：
```bash
python scripts/prepare_apt_datasets.py --all --max-samples 1000
```

### Q3: 如何自定义数据集？

A: 编辑 `scripts/prepare_apt_datasets.py`，添加新的数据集配置：
```python
'my-dataset': {
    'hf_name': 'username/dataset-name',
    'stage': 'SFT',
    'desc': '我的自定义数据集',
    'format_func': self.format_my_dataset
}
```

### Q4: 忠诚度模板怎么填写？

A: 运行后会生成模板文件，手动编辑：
```bash
python scripts/prepare_apt_datasets.py --loyalty-template
# 编辑 data/apt_datasets/loyalty_template.json
```

**编辑要点**:
- `owner_response`: 更详细、更个性化、更主动
- `public_response`: 更通用、更正式、更谨慎
- 区分点：语气、详细度、建议深度

---

## 📚 相关文档

- [APT对齐训练文档](./APT_ALIGNMENT_TRAINING.md)
- [COIG-CQIA论文](https://arxiv.org/abs/2403.18058)
- [s1K论文](https://arxiv.org/abs/2501.19393)
- [HH-RLHF文档](https://huggingface.co/datasets/Anthropic/hh-rlhf)

---

## 🎉 快速示例

### 完整流程示例

```bash
# 步骤1: 准备数据集
python scripts/launch_apt_alignment.py
# 选择 "1. 📦 准备数据集" → "6. 下载全部推荐数据集"

# 步骤2: 查看数据集统计
python scripts/launch_apt_alignment.py
# 选择 "3. 📊 查看数据集信息"

# 步骤3: 开始训练
python scripts/launch_apt_alignment.py
# 选择 "2. 🚀 开始训练" → "4. 完整流程 (All Stages)"
```

---

**祝训练顺利！🚀**
