# 教师模型API接口使用指南

## 概述

教师模型API接口允许你使用远程的大型语言模型（如GPT-4、Claude）作为教师进行知识蒸馏，而无需在本地运行这些大模型。

**核心优势:**
- 无需本地GPU资源运行大模型
- 可使用最先进的商业模型作为教师
- 灵活的API选择
- 自动重试和错误处理

---

## 支持的API

### 1. OpenAI API
- GPT-4, GPT-4-turbo
- GPT-3.5-turbo
- GPT-3

### 2. Anthropic API
- Claude-3 (Opus, Sonnet, Haiku)
- Claude-2

### 3. 硅基流动API (SiliconFlow)
- Qwen2系列 (通义千问)
- DeepSeek系列
- GLM-4系列 (智谱)
- Llama-3系列
- 其他国产开源模型

### 4. 自定义API
- 任何符合RESTful规范的API
- 本地部署的模型服务

---

## 快速开始

### 方法1: 使用OpenAI作为教师

```python
from apt_model.plugins.teacher_api import create_api_teacher_model
from apt_model.plugins.visual_distillation_plugin import quick_visual_distill
from transformers import AutoTokenizer

# 1. 创建教师模型（GPT-4）
tokenizer = AutoTokenizer.from_pretrained("gpt2")

teacher_model = create_api_teacher_model(
    provider='openai',
    api_key='sk-...',  # 你的OpenAI API key
    model_name='gpt-4',
    tokenizer=tokenizer,
    vocab_size=50000
)

# 2. 加载学生模型
from apt_model.training.checkpoint import load_model
student_model, _, _ = load_model("apt_model_small")

# 3. 准备数据
train_dataloader = get_dataloader()

# 4. 开始蒸馏
quick_visual_distill(
    student_model=student_model,
    teacher_model=teacher_model,  # API教师模型
    train_dataloader=train_dataloader,
    tokenizer=tokenizer,
    num_epochs=3,
    device='cuda'
)
```

### 方法2: 使用Claude作为教师

```python
from apt_model.plugins.teacher_api import create_api_teacher_model

teacher_model = create_api_teacher_model(
    provider='anthropic',
    api_key='sk-ant-...',  # 你的Anthropic API key
    model_name='claude-3-sonnet-20240229',
    tokenizer=tokenizer,
    vocab_size=50000
)

# 其余步骤相同
```

### 方法3: 使用硅基流动API

```python
from apt_model.plugins.teacher_api import create_api_teacher_model

teacher_model = create_api_teacher_model(
    provider='siliconflow',
    api_key='sk-...',  # 你的硅基流动 API key
    model_name='Qwen/Qwen2-7B-Instruct',
    tokenizer=tokenizer,
    vocab_size=50000
)

# 其余步骤相同
```

### 方法4: 使用自定义API

```python
teacher_model = create_api_teacher_model(
    provider='custom',
    api_key='your-api-key',
    base_url='https://your-api.com',
    tokenizer=tokenizer,
    vocab_size=50000
)
```

---

## 详细配置

### OpenAI配置

```python
from apt_model.plugins.teacher_api import OpenAITeacherAPI

config = {
    'api_key': 'sk-...',
    'model_name': 'gpt-4',  # 或 'gpt-3.5-turbo'
    'base_url': None,  # 可选，使用代理或自定义端点
    'timeout': 30,  # API超时（秒）
    'max_retries': 3,  # 最大重试次数
    'retry_delay': 1.0,  # 重试延迟（秒）
}

api = OpenAITeacherAPI(config)
```

**可用模型:**
- `gpt-4` - 最强大（最贵）
- `gpt-4-turbo` - 更快的GPT-4
- `gpt-3.5-turbo` - 性价比高
- `gpt-3.5-turbo-16k` - 长上下文

### Anthropic配置

```python
from apt_model.plugins.teacher_api import AnthropicTeacherAPI

config = {
    'api_key': 'sk-ant-...',
    'model_name': 'claude-3-sonnet-20240229',
    'timeout': 30,
    'max_retries': 3,
    'retry_delay': 1.0,
}

api = AnthropicTeacherAPI(config)
```

**可用模型:**
- `claude-3-opus-20240229` - 最强大
- `claude-3-sonnet-20240229` - 平衡性能
- `claude-3-haiku-20240307` - 最快速
- `claude-2.1` - 前代模型

### 硅基流动配置

```python
from apt_model.plugins.teacher_api import SiliconFlowTeacherAPI

config = {
    'api_key': 'sk-...',
    'model_name': 'Qwen/Qwen2-7B-Instruct',
    'base_url': 'https://api.siliconflow.cn/v1',  # 默认值，可省略
    'timeout': 30,
    'max_retries': 3,
    'retry_delay': 1.0,
}

api = SiliconFlowTeacherAPI(config)
```

**可用模型:**
- `Qwen/Qwen2-7B-Instruct` - 通义千问2代 7B（推荐）
- `Qwen/Qwen2-72B-Instruct` - 通义千问2代 72B（更强大）
- `deepseek-ai/DeepSeek-V2.5` - DeepSeek V2.5
- `THUDM/glm-4-9b-chat` - 智谱 GLM-4 9B
- `meta-llama/Meta-Llama-3-8B-Instruct` - Llama-3 8B
- `meta-llama/Meta-Llama-3-70B-Instruct` - Llama-3 70B

**优势:**
- 国内访问速度快
- 支持多种国产开源模型
- 价格相对便宜
- 兼容OpenAI接口格式

### 自定义API配置

```python
from apt_model.plugins.teacher_api import CustomTeacherAPI

config = {
    'api_key': 'your-key',
    'base_url': 'https://your-api.com',
    'model_name': 'your-model',  # 可选
    'timeout': 30,
    'max_retries': 3,
}

api = CustomTeacherAPI(config)
```

**API规范要求:**

生成文本端点:
```
POST {base_url}/generate
Content-Type: application/json
Authorization: Bearer {api_key}

Request:
{
    "input": "输入文本",
    "max_tokens": 100,
    "temperature": 1.0
}

Response:
{
    "text": "生成的文本",
    "output": "或者这个字段"
}
```

获取logits端点:
```
POST {base_url}/logits
Content-Type: application/json
Authorization: Bearer {api_key}

Request:
{
    "input": "输入文本",
    "return_logits": true
}

Response:
{
    "logits": [[...], [...], ...]  # 3D数组
}
```

---

## API接口详解

### 1. 文本生成

```python
api = create_teacher_api(
    provider='openai',
    api_key='sk-...',
    model_name='gpt-4'
)

# 生成文本
text = api.generate_text(
    input_text="什么是人工智能？",
    max_tokens=100,
    temperature=0.7
)

print(text)
# 输出: 人工智能是计算机科学的一个分支...
```

### 2. 获取Logits

```python
# 获取logits（用于蒸馏）
logits = api.get_logits(
    input_text="什么是人工智能？",
    vocab_size=50000
)

print(logits.shape)  # [1, seq_len, vocab_size]
```

### 3. 统计信息

```python
# 查看API调用统计
print(api.stats)
# {
#     'total_calls': 100,
#     'successful_calls': 98,
#     'failed_calls': 2,
#     'total_tokens': 15000
# }
```

---

## 完整蒸馏示例

### 示例1: GPT-4 → 小模型

```python
from apt_model.plugins.teacher_api import create_api_teacher_model
from apt_model.plugins.visual_distillation_plugin import VisualDistillationPlugin
from apt_model.training.checkpoint import load_model
from transformers import AutoTokenizer
import torch

# 1. 配置
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 2. 创建GPT-4教师模型
teacher_model = create_api_teacher_model(
    provider='openai',
    api_key='sk-...',
    model_name='gpt-4',
    tokenizer=tokenizer,
    vocab_size=50000,
    timeout=60,  # GPT-4可能较慢
    max_retries=5
)

# 3. 加载学生模型
student_model, _, config = load_model("apt_model_small")

# 4. 准备数据
from torch.utils.data import DataLoader, TensorDataset
train_data = TensorDataset(torch.randint(0, 50000, (100, 32)))
train_dataloader = DataLoader(train_data, batch_size=4)

# 5. 配置蒸馏
distill_config = {
    'temperature': 4.0,
    'alpha': 0.7,
    'beta': 0.3,
    'sample_frequency': 10,  # API调用较慢，少显示样本
}

# 6. 创建蒸馏插件
plugin = VisualDistillationPlugin(distill_config)
optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-4)

# 7. 开始蒸馏
plugin.visual_distill_model(
    student_model=student_model,
    teacher_model=teacher_model,
    train_dataloader=train_dataloader,
    optimizer=optimizer,
    tokenizer=tokenizer,
    num_epochs=3,
    device='cuda'
)

# 8. 查看统计
print(f"\n[API统计] 总调用: {teacher_model.api.stats['total_calls']}")
print(f"[API统计] 成功: {teacher_model.api.stats['successful_calls']}")
print(f"[API统计] 失败: {teacher_model.api.stats['failed_calls']}")
print(f"[API统计] 总tokens: {teacher_model.api.stats['total_tokens']}")
```

### 示例2: Claude-3 → 小模型

```python
# 使用Claude-3作为教师
teacher_model = create_api_teacher_model(
    provider='anthropic',
    api_key='sk-ant-...',
    model_name='claude-3-sonnet-20240229',
    tokenizer=tokenizer,
    vocab_size=50000
)

# 其余流程相同
```

### 示例3: 硅基流动Qwen2 → 小模型（中文场景推荐）

```python
# 使用硅基流动的Qwen2作为教师（适合中文）
teacher_model = create_api_teacher_model(
    provider='siliconflow',
    api_key='sk-...',  # 硅基流动API key
    model_name='Qwen/Qwen2-7B-Instruct',
    tokenizer=tokenizer,
    vocab_size=50000,
    timeout=30,
    max_retries=3
)

# 配置：中文场景优化
distill_config = {
    'temperature': 3.0,  # 中文模型可用较低温度
    'alpha': 0.7,
    'beta': 0.3,
    'sample_frequency': 5,  # 硅基流动速度快，可多显示样本
    'show_samples': True,
}

# 其余流程相同
```

**优势:**
- 成本极低 (~¥0.1/1000样本)
- 中文理解能力强
- 国内访问速度快
- 适合大规模中文数据蒸馏

---

## 成本估算

### OpenAI定价（2024年）

| 模型 | 输入价格/1M tokens | 输出价格/1M tokens |
|------|-------------------|-------------------|
| GPT-4 | $30 | $60 |
| GPT-4-turbo | $10 | $30 |
| GPT-3.5-turbo | $0.5 | $1.5 |

### Anthropic定价（2024年）

| 模型 | 输入价格/1M tokens | 输出价格/1M tokens |
|------|-------------------|-------------------|
| Claude-3-Opus | $15 | $75 |
| Claude-3-Sonnet | $3 | $15 |
| Claude-3-Haiku | $0.25 | $1.25 |

### 硅基流动定价（2024年）

| 模型 | 价格/1M tokens | 备注 |
|------|---------------|------|
| Qwen2-7B-Instruct | ¥1.0 (~$0.14) | 性价比高 |
| Qwen2-72B-Instruct | ¥5.0 (~$0.70) | 更强大 |
| DeepSeek-V2.5 | ¥1.0 (~$0.14) | 推理能力强 |
| GLM-4-9B | ¥1.0 (~$0.14) | 中文优化 |
| Llama-3-8B | ¥0.6 (~$0.08) | 最便宜 |
| Llama-3-70B | ¥3.0 (~$0.42) | 性能强 |

**注意:** 硅基流动通常按照总token数计费（输入+输出），价格以人民币计价

### 成本估算示例

假设训练1000个样本，每个样本平均100 tokens：

**使用GPT-3.5-turbo:**
- 输入: 1000 × 100 = 100,000 tokens = 0.1M
- 成本: 0.1M × $0.5 = **$0.05**

**使用GPT-4:**
- 输入: 100,000 tokens = 0.1M
- 成本: 0.1M × $30 = **$3.00**

**使用Claude-3-Haiku:**
- 输入: 100,000 tokens = 0.1M
- 成本: 0.1M × $0.25 = **$0.025**

**使用硅基流动Qwen2-7B:**
- 总tokens: 100,000 tokens = 0.1M
- 成本: 0.1M × ¥1.0 = **¥0.1 (~$0.014)**

**使用硅基流动Llama-3-8B（最便宜）:**
- 总tokens: 100,000 tokens = 0.1M
- 成本: 0.1M × ¥0.6 = **¥0.06 (~$0.008)**

**成本对比总结:**
1. 硅基流动 Llama-3-8B: ~$0.008 ⭐ **最便宜**
2. 硅基流动 Qwen2-7B: ~$0.014
3. Claude-3-Haiku: $0.025
4. GPT-3.5-turbo: $0.05
5. GPT-4: $3.00

---

## 性能优化

### 1. 批量处理

```python
# 不推荐：逐个处理
for sample in samples:
    logits = api.get_logits(sample, vocab_size)

# 推荐：批量处理（如果API支持）
all_logits = []
for batch in batches:
    batch_logits = api.get_logits_batch(batch, vocab_size)
    all_logits.append(batch_logits)
```

### 2. 缓存结果

```python
import pickle
import os

# 缓存教师输出
cache_dir = './teacher_cache'
os.makedirs(cache_dir, exist_ok=True)

def get_teacher_output_cached(input_text, api, cache_dir):
    """获取教师输出（带缓存）"""
    import hashlib
    cache_key = hashlib.md5(input_text.encode()).hexdigest()
    cache_file = f"{cache_dir}/{cache_key}.pkl"

    # 检查缓存
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # API调用
    output = api.generate_text(input_text)

    # 保存缓存
    with open(cache_file, 'wb') as f:
        pickle.dump(output, f)

    return output
```

### 3. 异步调用

```python
import asyncio
import aiohttp

async def async_generate(api, input_texts):
    """异步批量生成"""
    tasks = []
    for text in input_texts:
        task = asyncio.create_task(api.generate_text_async(text))
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    return results
```

### 4. 减少API调用

```python
# 配置：减少显示频率
distill_config = {
    'sample_frequency': 100,  # 每100个batch才调用一次API
}

# 或者：预先生成教师输出
teacher_outputs = []
for batch in train_dataloader:
    with torch.no_grad():
        teacher_output = teacher_model(batch['input_ids'])
        teacher_outputs.append(teacher_output)

# 然后在训练时使用缓存的输出
```

---

## 错误处理

### 1. API限流

```python
# 配置重试策略
config = {
    'max_retries': 5,
    'retry_delay': 2.0,  # 初始延迟2秒
    # 自动使用指数退避: 2s, 4s, 8s, 16s, 32s
}

api = OpenAITeacherAPI(config)
```

### 2. 超时处理

```python
config = {
    'timeout': 60,  # 60秒超时
}

try:
    text = api.generate_text(input_text)
except TimeoutError:
    print("API调用超时")
    # 使用fallback
```

### 3. API错误

```python
try:
    teacher_model = create_api_teacher_model(...)
except ImportError as e:
    print(f"缺少依赖库: {e}")
    print("请安装: pip install openai anthropic")
except Exception as e:
    print(f"创建失败: {e}")
```

---

## 最佳实践

### 1. 选择合适的教师模型

| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| 预算充足 | GPT-4 | 最强大 |
| 平衡性价比 | GPT-3.5-turbo / Claude-3-Sonnet | 性能好且便宜 |
| 大规模训练 | 硅基流动 Llama-3-8B | 最便宜 (~$0.008/1K样本) |
| 中文场景 | 硅基流动 Qwen2-7B / GLM-4 | 中文优化且便宜 |
| 国内部署 | 硅基流动 | 访问速度快 |
| 特定领域 | 自定义API | 专门优化 |

### 2. 混合策略

```python
# 使用便宜的模型做初步蒸馏
cheap_teacher = create_api_teacher_model(
    provider='openai',
    model_name='gpt-3.5-turbo',
    ...
)

# 第一轮蒸馏
distill_phase1(student_model, cheap_teacher, ...)

# 使用强大的模型做精细蒸馏
strong_teacher = create_api_teacher_model(
    provider='openai',
    model_name='gpt-4',
    ...
)

# 第二轮蒸馏
distill_phase2(student_model, strong_teacher, ...)
```

### 3. 监控成本

```python
# 实时监控tokens消耗
class CostMonitor:
    def __init__(self, price_per_million):
        self.price = price_per_million
        self.total_tokens = 0

    def update(self, tokens):
        self.total_tokens += tokens
        cost = (self.total_tokens / 1_000_000) * self.price
        print(f"[成本] 已使用 {self.total_tokens} tokens, 约 ${cost:.2f}")

monitor = CostMonitor(price_per_million=0.5)  # GPT-3.5价格

# 在蒸馏循环中
for batch in dataloader:
    # ... 蒸馏 ...
    monitor.update(teacher_model.api.stats['total_tokens'])
```

---

## 常见问题

### Q1: 为什么API返回的logits是模拟的？

**A:** OpenAI和Anthropic的API默认不返回logits，只返回文本。我们通过文本生成模拟logits。如果需要真实logits，需要：
1. 使用支持logprobs的API（如OpenAI的completion API）
2. 使用自定义API并实现logits端点
3. 只使用文本级别的蒸馏

### Q2: 如何减少API成本？

**A:**
1. 使用更便宜的模型（Claude-3-Haiku最便宜）
2. 缓存教师输出
3. 减少sample_frequency
4. 使用更短的输入文本
5. 预先批量生成教师输出

### Q3: API调用失败怎么办？

**A:** 插件会自动重试，如果多次失败：
1. 检查API key是否正确
2. 检查网络连接
3. 检查是否超过配额
4. 增加timeout和max_retries

### Q4: 可以离线使用吗？

**A:** 可以！预先生成教师输出并缓存：

```python
# 1. 预先生成（在线）
cache_teacher_outputs(train_data, teacher_api, cache_dir)

# 2. 离线训练（使用缓存）
train_with_cached_outputs(student_model, cache_dir)
```

### Q5: 如何选择API提供商？

**A:** 选择建议：
- **中文任务**: 硅基流动（Qwen2/GLM-4） - 中文优化，价格低
- **英文任务**: GPT-3.5-turbo / Claude-3-Haiku - 性能好
- **最佳质量**: GPT-4 / Claude-3-Opus - 但成本高
- **最低成本**: 硅基流动 Llama-3-8B - $0.008/1K样本
- **国内网络**: 硅基流动 - 访问速度快，无需代理

---

## 技术支持

- **代码**: `apt_model/plugins/teacher_api.py`
- **示例**: 文件末尾的`if __name__ == "__main__"`部分
- **问题反馈**: GitHub Issues

---

**Happy Distilling with API! 🚀**
