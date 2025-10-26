# APT遗留插件完整功能列表

**日期**: 2025-10-26
**状态**: ✅ 已适配到新插件系统
**适配器版本**: 1.0.0

---

## 📋 目录

1. [HuggingFace Integration Plugin](#1-huggingface-integration-plugin)
2. [Cloud Storage Plugin](#2-cloud-storage-plugin)
3. [Ollama Export Plugin](#3-ollama-export-plugin)
4. [Model Distillation Plugin](#4-model-distillation-plugin)
5. [Model Pruning Plugin](#5-model-pruning-plugin)
6. [Multimodal Training Plugin](#6-multimodal-training-plugin)
7. [Data Processors Plugin](#7-data-processors-plugin)
8. [Advanced Debugging Plugin](#8-advanced-debugging-plugin)

---

## 1. HuggingFace Integration Plugin

**类名**: `HuggingFaceIntegrationPlugin`
**适配器名**: `huggingface_integration`
**优先级**: 700 (Admin/Audit)
**类别**: integration

### 核心功能

#### 1.1 模型导入/导出

**`export_to_huggingface(model, tokenizer, repo_name, private, commit_message)`**
- 导出APT模型到HuggingFace Hub
- 自动创建仓库（支持公开/私有）
- 保存模型权重和分词器
- 自动生成模型卡片
- 支持自定义提交消息

**`import_from_huggingface(repo_name, local_dir)`**
- 从HuggingFace Hub导入模型
- 支持本地缓存
- 返回模型和分词器元组

#### 1.2 数据集管理

**`load_hf_dataset(dataset_name, split, streaming)`**
- 加载HuggingFace数据集
- 支持流式加载大数据集
- 支持数据集分片（train/test/validation）
- 自动处理数据集格式

#### 1.3 模型训练

**`train_with_hf_trainer(model, tokenizer, dataset, output_dir, training_args)`**
- 使用HuggingFace Trainer训练APT模型
- 支持完整的TrainingArguments配置
- 自动保存检查点
- 集成评估和日志记录

**`create_model_card(save_dir, repo_name, model_info)`**
- 自动生成模型卡片
- 包含模型架构、训练信息、使用示例
- 支持Markdown格式
- 符合HuggingFace标准

### 使用场景

- ✅ 将训练好的APT模型分享到HuggingFace Hub
- ✅ 从社区导入预训练模型
- ✅ 使用HF数据集训练模型
- ✅ 利用HF Trainer的高级训练功能

### 依赖项

```python
transformers>=4.30.0
datasets>=2.12.0
huggingface_hub>=0.14.0
```

---

## 2. Cloud Storage Plugin

**类名**: `CloudStoragePlugin`
**适配器名**: `cloud_storage`
**优先级**: 700 (Admin/Audit)
**类别**: storage

### 核心功能

#### 2.1 AWS S3 备份

**`backup_to_s3(model_path, bucket_name, s3_key, aws_access_key, aws_secret_key, region)`**
- 将模型上传到AWS S3
- 支持分片上传大文件
- 自动创建bucket（如果不存在）
- 支持自定义区域配置

**`restore_from_s3(bucket_name, s3_key, local_path, aws_access_key, aws_secret_key)`**
- 从S3恢复模型
- 支持断点续传
- 自动解压和验证

#### 2.2 阿里云 OSS 备份

**`backup_to_oss(model_path, bucket_name, object_key, access_key_id, access_key_secret, endpoint)`**
- 上传到阿里云OSS
- 支持中国大陆加速
- 自动处理编码问题

**`restore_from_oss(bucket_name, object_key, local_path, access_key_id, access_key_secret)`**
- 从OSS恢复模型
- 支持分片下载

#### 2.3 HuggingFace Hub 备份

**`backup_to_huggingface(model_path, repo_name, token, private)`**
- 备份到HuggingFace Hub
- 支持版本管理
- 自动处理Git LFS

#### 2.4 ModelScope 备份

**`backup_to_modelscope(model_path, model_id, token)`**
- 备份到魔搭社区（ModelScope）
- 支持中文社区分享
- 国内访问速度快

#### 2.5 多云同步

**`backup_model(model_path, destinations, config)`**
- 同时备份到多个云平台
- 支持并行上传
- 统一的配置接口
- 失败自动重试

**`restore_from_cloud(source, model_id, local_path, config)`**
- 从任意云平台恢复
- 自动选择最快的源
- 支持备份验证

### 使用场景

- ✅ 多云备份保障模型安全
- ✅ 跨平台模型分享
- ✅ 灾难恢复
- ✅ 协作训练（多地同步）

### 依赖项

```python
boto3>=1.26.0          # AWS S3
oss2>=2.17.0          # 阿里云 OSS
huggingface_hub>=0.14.0
modelscope>=1.4.0
```

---

## 3. Ollama Export Plugin

**类名**: `OllamaExportPlugin`
**适配器名**: `ollama_export`
**优先级**: 900 (Post-Cleanup)
**类别**: export

### 核心功能

#### 3.1 GGUF 转换

**`export_to_gguf(model_path, output_path, quantization_type, vocab_type)`**
- 将PyTorch模型转换为GGUF格式
- 支持多种量化类型：
  * Q4_0 - 4-bit量化
  * Q4_K_M - 4-bit K-quant medium
  * Q5_K_M - 5-bit K-quant medium
  * Q8_0 - 8-bit量化
  * FP16 - 16-bit浮点
- 自动处理词汇表转换
- 验证转换正确性

#### 3.2 Modelfile 创建

**`create_modelfile(model_name, gguf_path, template, system_prompt, parameters)`**
- 生成Ollama Modelfile
- 支持自定义系统提示词
- 配置生成参数（temperature, top_p等）
- 支持多种模板格式

#### 3.3 Ollama 注册

**`register_to_ollama(modelfile_path, model_name)`**
- 注册模型到本地Ollama
- 自动检测Ollama安装
- 验证注册成功

#### 3.4 模型测试

**`test_model(model_name, test_prompts, max_tokens)`**
- 测试已注册的Ollama模型
- 支持批量测试提示
- 评估响应质量
- 性能基准测试

### 使用场景

- ✅ 将APT模型部署到本地Ollama
- ✅ 模型量化以减小体积
- ✅ 离线部署和推理
- ✅ 集成到Ollama生态系统

### 依赖项

```python
torch>=2.0.0
transformers>=4.30.0
# 需要系统安装 Ollama CLI
```

---

## 4. Model Distillation Plugin

**类名**: `ModelDistillationPlugin`
**适配器名**: `model_distillation`
**优先级**: 350 (Training)
**类别**: training

### 核心功能

#### 4.1 响应蒸馏

**`response_distillation(teacher_model, student_model, data_loader, temperature, alpha)`**
- 基于KL散度的响应蒸馏
- 软标签学习（temperature scaling）
- 硬标签和软标签混合（alpha混合）
- 支持logits蒸馏

#### 4.2 特征蒸馏

**`feature_distillation(teacher_model, student_model, data_loader, layer_mappings)`**
- 中间层特征对齐
- 自定义教师-学生层映射
- 支持多层同时蒸馏
- MSE损失优化

#### 4.3 关系蒸馏

**`relation_distillation(teacher_model, student_model, data_loader)`**
- 样本间关系保持
- 基于样本相似度的蒸馏
- 保留知识结构
- Gram矩阵匹配

#### 4.4 注意力蒸馏

**`attention_distillation(teacher_model, student_model, data_loader, attention_type)`**
- 注意力权重对齐
- 支持多头注意力蒸馏
- 自注意力和交叉注意力
- 注意力模式保持

#### 4.5 主蒸馏流程

**`distill_model(teacher_model, student_model, train_data, val_data, config)`**
- 完整的蒸馏训练流程
- 多种蒸馏策略组合
- 自动调整权重
- 验证集评估

**`evaluate_compression(original_model, distilled_model, test_data)`**
- 评估压缩效果
- 性能对比（准确率、速度、内存）
- 生成压缩报告

### 蒸馏策略

| 策略 | 优势 | 适用场景 |
|------|------|----------|
| 响应蒸馏 | 简单高效 | 分类任务、语言模型 |
| 特征蒸馏 | 保留中间表示 | 需要特征表达的任务 |
| 关系蒸馏 | 保留样本关系 | 聚类、检索任务 |
| 注意力蒸馏 | 保留注意力模式 | NLP、多模态任务 |

### 使用场景

- ✅ 大模型压缩到小模型
- ✅ 保持性能的同时减小模型体积
- ✅ 加速推理
- ✅ 边缘设备部署

### 依赖项

```python
torch>=2.0.0
numpy>=1.20.0
```

---

## 5. Model Pruning Plugin

**类名**: `ModelPruningPlugin`
**适配器名**: `model_pruning`
**优先级**: 350 (Training)
**类别**: training

### 核心功能

#### 5.1 Magnitude 剪枝

**`magnitude_pruning(model, sparsity, granularity)`**
- 基于权重绝对值的剪枝
- 全局或局部稀疏度控制
- 支持逐层剪枝
- L1/L2 norm选择

#### 5.2 Taylor 剪枝

**`taylor_pruning(model, data_loader, sparsity)`**
- 基于梯度×权重的重要性评估
- 一阶Taylor展开
- 考虑训练动态
- 更精确的重要性评估

#### 5.3 结构化剪枝

**`structured_pruning(model, prune_ratio, prune_type)`**
- 剪枝整个神经元/通道/头
- 保持模型结构完整性
- 真实加速（不需要稀疏库）
- 支持：
  * 神经元剪枝
  * 通道剪枝
  * 注意力头剪枝
  * Filter剪枝

#### 5.4 彩票假说剪枝

**`lottery_ticket_pruning(model, train_data, iterations, prune_rate)`**
- 迭代剪枝寻找"winning ticket"
- 重新初始化到早期权重
- 多轮迭代优化
- 发现稀疏子网络

#### 5.5 剪枝后微调

**`fine_tune_after_pruning(pruned_model, train_data, val_data, epochs)`**
- 剪枝后恢复性能
- 学习率调度
- 验证集监控
- 早停机制

#### 5.6 完整剪枝流程

**`prune_model(model, train_data, val_data, target_sparsity, method)`**
- 自动化剪枝流程
- 多种方法选择
- 性能监控
- 渐进式剪枝

### 剪枝策略对比

| 方法 | 剪枝粒度 | 速度提升 | 精度损失 | 复杂度 |
|------|---------|---------|---------|--------|
| Magnitude | 非结构化 | 低* | 低 | 低 |
| Taylor | 非结构化 | 低* | 低 | 中 |
| Structured | 结构化 | 高 | 中 | 中 |
| Lottery Ticket | 非结构化 | 低* | 极低 | 高 |

*需要稀疏计算库支持

### 使用场景

- ✅ 减少模型参数量
- ✅ 加速推理速度
- ✅ 降低内存占用
- ✅ 移动端部署

### 依赖项

```python
torch>=2.0.0
numpy>=1.20.0
```

---

## 6. Multimodal Training Plugin

**类名**: `MultimodalTrainingPlugin`
**适配器名**: `multimodal_training`
**优先级**: 350 (Training)
**类别**: training

### 核心功能

#### 6.1 多模态模型创建

**`create_multimodal_model(text_encoder, image_encoder, audio_encoder, fusion_method)`**
- 组合文本、图像、音频编码器
- 多种融合策略：
  * Concatenate - 简单拼接
  * Add - 加法融合
  * Attention - 跨模态注意力
- 自动处理维度对齐
- 支持自定义融合模块

#### 6.2 多模态数据加载

**`create_multimodal_dataloader(text_data, image_data, audio_data, batch_size, shuffle)`**
- 同步加载多模态数据
- 自动对齐时间戳
- 支持不完整模态（缺失值处理）
- 批次均衡采样

#### 6.3 联合训练

**`train_multimodal(model, dataloader, epochs, optimizer, loss_weights)`**
- 多模态联合训练
- 自定义损失权重
- 模态特定的学习率
- 梯度平衡

**`inference_multimodal(model, text_input, image_input, audio_input)`**
- 多模态推理
- 支持单模态或多模态输入
- 自适应融合

#### 6.4 编码器支持

**文本编码器**:
- BERT, RoBERTa
- GPT系列
- T5
- 中文模型（BERT-Chinese等）

**图像编码器**:
- CLIP (ViT-B/16, ViT-L/14)
- ViT (Vision Transformer)
- ResNet
- EfficientNet

**音频编码器**:
- Wav2Vec2
- HuBERT
- WavLM
- Whisper

#### 6.5 跨模态注意力

**`cross_modal_attention(query_modality, key_modality, value_modality)`**
- 模态间注意力机制
- 自适应权重学习
- 多头跨模态注意力

### 使用场景

- ✅ 图文匹配任务
- ✅ 视频理解（文本+图像+音频）
- ✅ 多模态问答
- ✅ 内容生成（文生图、图生文等）
- ✅ 情感分析（文本+语音）

### 依赖项

```python
torch>=2.0.0
transformers>=4.30.0
torchvision>=0.15.0
torchaudio>=2.0.0
PIL>=9.0.0
librosa>=0.10.0
```

---

## 7. Data Processors Plugin

**类名**: `DataProcessorsPlugin`
**适配器名**: `data_processors`
**优先级**: 100 (Core Runtime)
**类别**: data

### 核心功能

#### 7.1 文本清洗

**`clean_text(text, strategy, custom_rules)`**

**清洗策略**:

1. **基础清洗** (`basic`)
   - 去除特殊字符
   - 标准化空白字符
   - 修复编码问题
   - 去除URL和邮箱

2. **激进清洗** (`aggressive`)
   - 基础清洗 +
   - 去除所有非字母数字字符
   - 统一大小写
   - 去除停用词

3. **中文清洗** (`chinese`)
   - 去除繁体字转简体
   - 去除标点符号
   - 分词和词性标注
   - 去除无意义词

4. **代码清洗** (`code`)
   - 保留代码结构
   - 去除注释
   - 标准化缩进
   - 去除空行

#### 7.2 数据增强

**`augment_text(text, methods, num_augmented, preserve_label)`**

**增强方法**:

1. **同义词替换** (`synonym_replacement`)
   - 基于WordNet/HowNet
   - 保持语义不变
   - 可控替换比例

2. **随机交换** (`random_swap`)
   - 随机交换词语位置
   - 保持句子流畅性

3. **随机删除** (`random_deletion`)
   - 随机删除部分词语
   - 控制删除比例

4. **回译增强** (`back_translation`)
   - 翻译到其他语言再翻译回来
   - 支持多种语言对
   - 保持语义生成新表达

5. **EDA增强** (`eda`)
   - 组合以上多种方法
   - 自动调整参数
   - 批量生成

#### 7.3 数据平衡

**`balance_dataset(dataset, method, target_ratio)`**

**平衡方法**:

1. **过采样** (`oversample`)
   - 复制少数类样本
   - 随机过采样
   - SMOTE（合成少数类过采样）

2. **欠采样** (`undersample`)
   - 减少多数类样本
   - 随机欠采样
   - Tomek Links

3. **混合采样** (`hybrid`)
   - 过采样 + 欠采样
   - 自适应调整比例

#### 7.4 质量检查

**`check_quality(dataset, checks, threshold)`**

**检查项目**:

1. **重复检测**
   - 精确重复
   - 近似重复（Jaccard相似度）
   - MinHash去重

2. **长度过滤**
   - 最小/最大长度
   - 词数统计
   - 字符数统计

3. **语言检测**
   - 自动检测语言
   - 过滤非目标语言
   - 多语言混合检测

4. **质量评分**
   - 困惑度评估
   - 流畅度评分
   - 信息密度

#### 7.5 完整处理流程

**`process_pipeline(raw_data, config)`**
- 串联清洗→增强→平衡→质量检查
- 可配置的流程步骤
- 中间结果保存
- 处理日志和统计

### 使用场景

- ✅ 训练数据预处理
- ✅ 数据质量提升
- ✅ 处理不平衡数据集
- ✅ 数据增强扩充训练集
- ✅ 清洗网络爬取数据

### 依赖项

```python
nltk>=3.8.0
jieba>=0.42.0              # 中文分词
langdetect>=1.0.9
textblob>=0.17.0
googletrans>=4.0.0         # 回译
imbalanced-learn>=0.10.0   # SMOTE
datasketch>=1.5.0          # MinHash
```

---

## 8. Advanced Debugging Plugin

**类名**: `AdvancedDebuggingPlugin`
**适配器名**: `advanced_debugging`
**优先级**: 800 (Telemetry)
**类别**: debug

### 核心功能

#### 8.1 梯度监控

**`monitor_gradients(model, log_interval, track_layers)`**
- 实时梯度统计（均值、标准差、L2范数）
- 逐层梯度监控
- 梯度历史记录
- 梯度流可视化

**`detect_gradient_anomalies(gradients, threshold)`**
- **梯度爆炸检测**
  * L2范数超过阈值
  * 梯度值异常增长
  * 自动报警

- **梯度消失检测**
  * 梯度接近零
  * 多层梯度衰减
  * 定位问题层

#### 8.2 激活值监控

**`monitor_activations(model, data_loader, layers)`**
- 激活值统计（均值、方差、最大/最小值）
- 分布分析（直方图）
- 饱和度检测

**`detect_dead_neurons(activations, threshold)`**
- 识别始终为0的神经元
- 统计死神经元比例
- 建议剪枝候选

**`detect_saturated_neurons(activations, saturation_threshold)`**
- 检测饱和的激活函数
- ReLU饱和分析
- Sigmoid/Tanh饱和检测

#### 8.3 内存监控

**`track_memory(device, interval, plot)`**
- GPU内存使用追踪
- 内存分配时间线
- 峰值内存记录
- 实时内存曲线

**`detect_memory_leaks(memory_history, leak_threshold)`**
- 检测内存持续增长
- 识别泄漏来源
- 内存碎片分析
- 提供修复建议

#### 8.4 性能分析

**`profile_section(code_section, iterations, warmup)`**
- 代码段性能profiling
- CPU/GPU时间测量
- 吞吐量计算
- 性能瓶颈识别

**`profile_model(model, input_shape, batch_size)`**
- 模型整体性能分析
- 逐层延迟测量
- 参数量和FLOPs统计
- 内存占用分析

#### 8.5 异常诊断

**`diagnose_training(model, train_loader, val_loader, config)`**
- 全面训练诊断
- 检测项目：
  * Loss NaN/Inf
  * 权重更新异常
  * 学习率问题
  * 数据加载瓶颈
  * 过拟合/欠拟合
- 生成诊断报告
- 提供解决方案

**`detect_nan_inf(tensors, raise_error)`**
- 检测NaN和Inf值
- 定位异常来源
- 可选自动停止训练

#### 8.6 可视化

**`visualize_gradients(grad_dict, save_path)`**
- 梯度分布直方图
- 梯度流图
- 逐层梯度热图

**`visualize_activations(activation_dict, save_path)`**
- 激活值分布
- 神经元激活模式
- 层级激活热图

**`generate_full_report(stats, save_dir)`**
- 综合诊断报告（HTML/PDF）
- 包含所有监控数据
- 图表和可视化
- 问题总结和建议

### 使用场景

- ✅ 调试训练不收敛问题
- ✅ 优化训练性能
- ✅ 检测和修复内存泄漏
- ✅ 分析模型瓶颈
- ✅ 监控训练健康状态
- ✅ 生成训练报告

### 依赖项

```python
torch>=2.0.0
numpy>=1.20.0
matplotlib>=3.5.0
plotly>=5.14.0            # 交互式可视化
psutil>=5.9.0             # 系统监控
py3nvml>=0.2.7            # NVIDIA GPU监控
```

---

## 📊 插件对比总结

### 按类别分类

| 类别 | 插件 | 主要用途 |
|------|------|---------|
| **集成** | HuggingFace Integration | 模型分享、数据集加载 |
| **存储** | Cloud Storage | 多云备份和恢复 |
| **导出** | Ollama Export | 本地部署、GGUF转换 |
| **训练** | Model Distillation | 模型压缩 |
| **训练** | Model Pruning | 参数剪枝 |
| **训练** | Multimodal Training | 多模态学习 |
| **数据** | Data Processors | 数据预处理 |
| **调试** | Advanced Debugging | 训练监控和诊断 |

### 按优先级分类

| 优先级 | 插件数量 | 插件列表 |
|--------|---------|---------|
| 100 | 1 | Data Processors |
| 350 | 3 | Model Distillation, Model Pruning, Multimodal Training |
| 700 | 2 | HuggingFace Integration, Cloud Storage |
| 800 | 1 | Advanced Debugging |
| 900 | 1 | Ollama Export |

### 功能覆盖

✅ **模型生命周期全覆盖**:
- 数据准备 → Data Processors
- 模型训练 → Multimodal Training, Distillation, Pruning
- 模型调试 → Advanced Debugging
- 模型导出 → Ollama Export, HuggingFace Integration
- 模型备份 → Cloud Storage

✅ **支持的工作流**:
1. 标准训练流程
2. 模型压缩流程
3. 多模态训练流程
4. 部署分发流程

---

## 🚀 快速使用指南

### 加载所有适配器

```python
from apt_model.console.core import ConsoleCore
from apt_model.console.legacy_plugins.adapters import get_all_legacy_adapters

# 初始化控制台
core = ConsoleCore()

# 加载所有遗留插件适配器
adapters = get_all_legacy_adapters()

# 注册到PluginBus
for name, adapter in adapters.items():
    core.register_plugin(adapter)
    print(f"✅ Registered: {name}")

# 编译插件
core.compile_plugins()

# 开始使用
core.start()
```

### 加载特定插件

```python
from apt_model.console.legacy_plugins.adapters import get_adapter

# 只加载HuggingFace插件
hf_adapter = get_adapter("huggingface_integration", config={
    "token": "hf_xxx"
})

core.register_plugin(hf_adapter)

# 调用插件功能
hf_plugin = hf_adapter.get_legacy_plugin()
hf_plugin.export_to_huggingface(
    model=my_model,
    tokenizer=my_tokenizer,
    repo_name="username/my-apt-model"
)
```

### 使用插件功能

```python
# 方法1: 通过适配器调用
adapter = get_adapter("advanced_debugging")
core.register_plugin(adapter)

# 获取原始插件实例
debug_plugin = adapter.get_legacy_plugin()
debug_plugin.monitor_gradients(model, log_interval=100)

# 方法2: 通过事件系统（自动触发）
# 插件会在on_batch_start等事件时自动执行监控
core.emit_event("on_batch_start", step=0, context_data={
    "model": model,
    "batch_idx": 0
})
```

---

## 📝 注意事项

### 依赖管理

每个插件有独立的依赖项，建议：

```bash
# 安装基础依赖
pip install torch transformers

# 按需安装特定插件依赖
pip install boto3 oss2  # Cloud Storage
pip install nltk jieba  # Data Processors
pip install matplotlib plotly  # Advanced Debugging
```

### 配置建议

```python
config = {
    # HuggingFace
    "huggingface": {
        "token": "hf_xxx",
        "cache_dir": "/path/to/cache"
    },

    # Cloud Storage
    "cloud_storage": {
        "aws": {
            "access_key": "xxx",
            "secret_key": "xxx"
        },
        "oss": {
            "access_key_id": "xxx",
            "access_key_secret": "xxx"
        }
    },

    # Debugging
    "debugging": {
        "log_interval": 100,
        "save_dir": "./debug_logs"
    }
}

# 传递配置
adapter = get_adapter("huggingface_integration", config=config.get("huggingface"))
```

### 性能影响

| 插件 | 性能影响 | 建议 |
|------|---------|------|
| Data Processors | 预处理阶段 | 离线处理 |
| Advanced Debugging | 5-10% 训练开销 | 调试时启用 |
| Distillation | 需要teacher模型 | 独立训练阶段 |
| Pruning | 额外训练轮次 | 独立训练阶段 |
| Multimodal Training | 取决于编码器 | 使用预训练编码器 |
| Cloud Storage | I/O开销 | 异步上传 |
| HuggingFace | 网络I/O | 使用缓存 |
| Ollama Export | 后处理 | 训练结束后 |

---

## 🎓 最佳实践

1. **按需加载**: 不要一次加载所有插件，只加载需要的
2. **配置管理**: 使用配置文件管理敏感信息（API keys等）
3. **日志记录**: 启用适当的日志级别监控插件行为
4. **错误处理**: 插件错误不应影响主训练流程
5. **资源清理**: 训练结束后调用on_shutdown事件

---

**文档完成！所有8个插件的完整功能已列出。**
