# 推理插件 vs 内核推理模块 - 对比分析

## 📊 核心区别总结

| 维度 | 内核推理模块 | 推理插件 |
|------|------------|----------|
| **位置** | `apt_model/runtime/decoder/` | `apt_model/console/plugins/reasoning/` |
| **架构层面** | 模型运行时组件 | Console 插件系统 |
| **集成方式** | 硬编码到模型推理流程 | 可插拔、动态加载 |
| **作用范围** | Token 级别推理控制 | 推理策略和后处理 |
| **执行时机** | 模型前向传播期间 | 推理前/后 |
| **依赖关系** | 被插件依赖 | 依赖内核推理 |

---

## 🏗️ 架构层次

```
┌─────────────────────────────────────────────────┐
│                应用层                             │
│  ┌───────────────────────────────────────┐      │
│  │  推理插件 (Console Plugins)            │      │
│  │  - Self-Consistency                    │      │
│  │  - Beam Search                         │      │
│  │  - Program-Aided                       │      │
│  └───────────────────────────────────────┘      │
│                    ↓ 依赖                        │
│  ┌───────────────────────────────────────┐      │
│  │  Console Core (事件系统)               │      │
│  └───────────────────────────────────────┘      │
└─────────────────────────────────────────────────┘
                     ↓ 调用
┌─────────────────────────────────────────────────┐
│                模型层                             │
│  ┌───────────────────────────────────────┐      │
│  │  内核推理模块 (Runtime Decoder)        │      │
│  │  - ReasoningController                 │      │
│  │  - StructuredReasoner                  │      │
│  │  - ChainOfThoughtReasoner              │      │
│  │  - MultiCriteriaHalting                │      │
│  └───────────────────────────────────────┘      │
│                    ↓ 操作                        │
│  ┌───────────────────────────────────────┐      │
│  │  模型前向传播                          │      │
│  │  - Hidden States                       │      │
│  │  - Attention                           │      │
│  │  - MoE Routing                         │      │
│  └───────────────────────────────────────┘      │
└─────────────────────────────────────────────────┘
```

---

## 🎯 内核推理模块 (Runtime Decoder)

### 定位
**模型运行时的核心推理引擎**

### 关键文件
- `apt_model/runtime/decoder/reasoning_controller.py`
- `apt_model/runtime/decoder/structured_reasoner.py`
- `apt_model/runtime/decoder/halting.py`

### 核心功能

#### 1. **ReasoningController**
```python
# Token 级别的推理控制
h_input → [推理步骤1] → [推理步骤2] → ... → h_output
          ↓              ↓
       专家路由      停止判断
```

**作用**:
- 在 hidden states 层面进行迭代推理
- 控制每个 token 的推理步数
- 自适应停止（KL散度、状态变化、熵）
- 预算控制（选择性推理）

**特点**:
- ✅ 嵌入在模型前向传播中
- ✅ 操作 token embeddings
- ✅ 使用 Vein subspace projection
- ✅ MoE 专家路由

#### 2. **StructuredReasoner**
```python
# 单步推理
h → Vein投影 → MoE路由 → 专家混合 → 重建 → h_new
```

**作用**:
- 低秩子空间推理（Vein subspace）
- 专家路由和混合
- 计算停止概率

**特点**:
- ✅ 高效的低秩计算
- ✅ 可学习的参数
- ✅ 梯度可反向传播

#### 3. **训练集成**
```python
# 训练时的推理损失
loss = cross_entropy(logits, labels) + aux_loss
                                         ↑
                              专家负载均衡、推理步数惩罚
```

---

## 🔌 推理插件 (Console Plugins)

### 定位
**推理策略的高级包装和后处理**

### 关键文件
- `apt_model/console/plugins/reasoning/self_consistency_plugin.py`
- `apt_model/console/plugins/reasoning/beam_search_plugin.py`
- `apt_model/console/plugins/reasoning/program_aided_plugin.py`

### 核心功能

#### 1. **Self-Consistency Plugin**
```python
# 序列级别的多路径采样
问题 → [生成路径1] → 答案1
    → [生成路径2] → 答案2    → 投票 → 最终答案
    → [生成路径3] → 答案3
```

**作用**:
- 调用模型生成多个完整推理序列
- 从序列中提取答案
- 多数投票选择最一致答案

**特点**:
- ✅ 序列级别操作（不是 token 级别）
- ✅ 多次调用模型
- ✅ 后处理答案提取和投票
- ✅ 不改变模型内部状态

#### 2. **Beam Search Plugin**
```python
# 维护多个候选序列
初始 → [扩展] → Top-k候选 → [扩展] → Top-k候选 → 最佳序列
```

**作用**:
- 在解码层面控制生成策略
- 维护候选序列束
- 路径评分和剪枝

**特点**:
- ✅ 解码策略（不是推理机制）
- ✅ 操作完整序列
- ✅ 可能调用内核推理作为单步生成

#### 3. **Program-Aided Plugin**
```python
# 代码生成和执行
问题 → [生成Python代码] → [沙箱执行] → 结果
```

**作用**:
- 生成可执行代码
- 外部执行获取结果
- 与符号计算系统集成

**特点**:
- ✅ 外部工具调用
- ✅ 不依赖 hidden states
- ✅ 确定性计算

---

## 🔗 协作关系

### 典型工作流程

```python
# 用户请求
question = "What is 15% of 80?"

# 1. 推理插件：Self-Consistency 启动
plugin.on_decode() {
    for path in range(5):  # 生成5条路径

        # 2. 调用模型生成
        output = model.generate(
            input_ids,
            # 3. 内核推理：在模型内部工作
            # ReasoningController 控制每个 token 的推理
            # StructuredReasoner 执行 Vein subspace 推理
            # MultiCriteriaHalting 决定何时停止
        )

        paths.append(output)

    # 4. 推理插件：后处理
    answers = [extract_answer(p) for p in paths]
    final_answer = vote(answers)  # 投票
}
```

### 依赖关系

```
推理插件（应用层）
    ↓ 调用
模型生成 API
    ↓ 使用
内核推理模块（运行时）
    ↓ 操作
Hidden States / Embeddings
```

---

## 📋 功能对比表

| 功能 | 内核推理 | 推理插件 |
|------|---------|----------|
| **多步推理** | ✅ Token级迭代 | ✅ 序列级多次生成 |
| **自适应停止** | ✅ KL/熵/状态变化 | ❌ 固定路径数 |
| **预算控制** | ✅ Token级选择 | ❌ |
| **专家路由** | ✅ MoE routing | ❌ |
| **自洽性** | ❌ | ✅ 多路径投票 |
| **束搜索** | ❌ | ✅ Beam search |
| **代码执行** | ❌ | ✅ PAL/PoT |
| **参数学习** | ✅ 可训练 | ❌ 后处理 |
| **梯度反传** | ✅ | ❌ |
| **可插拔** | ❌ 硬编码 | ✅ 动态加载 |

---

## 🎨 使用场景

### 内核推理模块
**何时使用**:
- 训练推理增强模型
- 需要低秩高效推理
- Token 级别的自适应推理
- 预算约束的推理

**例子**:
```python
# 训练时
reasoning_controller = ReasoningController(vein_projector, max_steps=6)
h_reasoned, info = reasoning_controller(hidden_states, lm_head)
loss = compute_loss(h_reasoned, labels)
loss.backward()  # 梯度反传到推理模块
```

### 推理插件
**何时使用**:
- 推理时需要多样性
- 需要后处理和投票
- 结合外部工具（代码执行）
- 不修改模型参数

**例子**:
```python
# 推理时
console.register_plugin(SelfConsistencyPlugin(num_paths=5))
context = console.emit_event('on_decode',
    context_data={'use_self_consistency': True})
answer = context.data['self_consistency_result']['answer']
```

---

## 💡 关键洞察

### 1. **互补而非替代**
- 内核推理 = 基础能力（如何推理）
- 推理插件 = 策略增强（如何使用推理）

### 2. **不同抽象层次**
- 内核：Token embedding 空间
- 插件：完整序列空间

### 3. **训练 vs 推理**
- 内核：训练和推理都使用
- 插件：主要用于推理

### 4. **效率权衡**
- 内核：一次前向传播，多步内部推理（高效）
- 插件：多次前向传播，后处理聚合（灵活但慢）

---

## 🚀 最佳实践

### 组合使用
```python
# 1. 训练阶段：使用内核推理
model = APTModel(
    reasoning_controller=ReasoningController(...)
)
train(model, reasoning_dataset)

# 2. 推理阶段：叠加推理插件
console = ConsoleCore()
console.register_plugin(SelfConsistencyPlugin())  # 多路径
# 内核推理在每条路径内部工作
# 插件在多条路径之间聚合
```

### 选择指南
- **需要训练？** → 使用内核推理
- **需要多样性？** → 使用 Self-Consistency 插件
- **需要搜索？** → 使用 Beam Search 插件
- **需要工具？** → 使用 Program-Aided 插件
- **需要高效？** → 使用内核的 BudgetedReasoningController

---

**总结**: 内核推理是"引擎"，插件是"驾驶策略"。内核提供基础推理能力，插件利用这些能力实现高级推理策略。
