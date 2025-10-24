# APT 微内核架构重构方案

## 🎯 重构目标

将APT从"单体架构"重构为"**微内核 + 插件**"架构：
- **核心（Core）**：配置、调度、训练闭环、装配骨架、默认算子——保证不装任何插件也能跑
- **插件（Plugins）**：MoE、对齐、路由、检索、投票、替代内核——可选、可替换、可迭代
- **Provider + 注册表**：核心导出接口，插件注册实现，保持性能与可维护性

---

## 📊 当前 vs 目标结构对比

### 当前结构（单体）
```
apt_model/
├── config/              # 配置（混杂）
├── modeling/            # 模型（单体）
├── training/            # 训练
├── data/                # 数据
├── generation/          # 生成
├── evaluation/          # 评估
├── interactive/         # 交互
├── utils/               # 工具（杂）
├── cli/                 # CLI
└── plugins/             # 插件（框架存在但未充分利用）
```

### 目标结构（微内核）
```
apt/
├── core/                # 核心模块（微内核） ⭐
│   ├── config.py        # 配置解析 + profile支持
│   ├── schedules.py     # 课程化启停/退火
│   ├── logging.py       # 日志
│   ├── monitor.py       # 监控
│   ├── errors.py        # 错误恢复
│   ├── device.py        # 硬件探测
│   ├── cache.py         # 缓存
│   └── registry.py      # Provider注册表 ⭐⭐⭐
│
├── training/            # 训练循环（Core）
│   ├── trainer.py       # 主训练循环 + 钩子广播
│   ├── checkpoint.py    # 检查点管理
│   └── optim.py         # 优化器配置
│
├── modeling/            # 模型装配（Core）
│   ├── compose.py       # Builder骨架 ⭐⭐
│   ├── layers/
│   │   ├── attention_tva.py   # 默认TVA注意力（黄金路径）
│   │   ├── vft.py            # VFT注意力
│   │   ├── ffn.py            # 默认FFN
│   │   └── norm.py           # 规范化
│   └── backbones/
│       └── gpt.py            # GPT主干（参考实现）
│
├── data/                # 数据模块（Core最小实现）
│   ├── hlbd/            # HLBD基础分词
│   ├── tokenizer.py     # 基础分词器
│   ├── loaders/         # 文本I/O
│   │   ├── txt.py
│   │   └── json.py
│   └── preprocess.py    # 基础清洗
│
├── inference/           # 推理基础（Core）
│   ├── generator.py     # 采样/解码
│   └── chat.py          # 会话管理
│
├── evaluation/          # 轻量评测（Core）
│   ├── quick_eval.py    # PPL/结构率
│   └── validators.py    # 基础验证
│
├── cli/                 # CLI入口（Core）
│   ├── parser.py        # 参数解析 + plugin子命令
│   ├── commands.py      # 命令实现
│   └── __main__.py      # 入口
│
├── plugins/             # 插件系统 ⭐⭐⭐
│   ├── builtin/         # 内置可选插件
│   │   ├── moe.py              # MoE专家路由
│   │   ├── align.py            # 双态数对齐
│   │   ├── routing.py          # 路由退火/容量调度
│   │   ├── retriever.py        # RAG检索
│   │   ├── voter.py            # 投票/一致性
│   │   └── __init__.py
│   │
│   ├── flash_attn/      # Flash Attention替代内核
│   ├── linear_attn/     # Linear Attention
│   ├── quant/           # 量化
│   ├── export/          # 导出（ONNX/Ollama）
│   ├── wandb/           # W&B监控
│   ├── monitor/         # 高级监控
│   ├── data_hf.py       # HuggingFace数据源
│   ├── data_sql.py      # SQL数据源
│   ├── mm/              # 多模态编码器
│   └── optuna.py        # 超参搜索
│
└── profiles/            # 配置文件 ⭐
    ├── base.yaml
    ├── gpt5_moe_reasoning.yaml
    └── tiny_debug.yaml
```

---

## 🔌 核心接口设计

### 1. Provider 接口层次

所有Provider都继承自基类，核心通过注册表调用：

```python
# core/registry.py

from abc import ABC, abstractmethod
from typing import Dict, Type, Any, Optional

class Provider(ABC):
    """所有Provider的基类"""

    @abstractmethod
    def get_name(self) -> str:
        """返回Provider名称"""
        pass

    @abstractmethod
    def get_version(self) -> str:
        """返回版本"""
        pass

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """验证配置是否合法"""
        return True


class Registry:
    """全局Provider注册表"""

    def __init__(self):
        self._providers: Dict[str, Dict[str, Type[Provider]]] = {}
        self._instances: Dict[str, Provider] = {}

    def register(self, kind: str, name: str, provider_cls: Type[Provider]):
        """注册Provider

        Args:
            kind: Provider种类 (attention/ffn/router/align...)
            name: 实现名称 (tva_default/flash_v2/linear_causal...)
            provider_cls: Provider类
        """
        if kind not in self._providers:
            self._providers[kind] = {}
        self._providers[kind][name] = provider_cls
        print(f"✅ 注册 {kind} Provider: {name}")

    def get(self, kind: str, name: str, config: Optional[Dict] = None) -> Provider:
        """获取Provider实例（单例）"""
        key = f"{kind}:{name}"
        if key not in self._instances:
            if kind not in self._providers or name not in self._providers[kind]:
                # 回退到默认实现
                default_name = self._get_default(kind)
                if default_name and default_name in self._providers.get(kind, {}):
                    print(f"⚠️  {kind}:{name} 未找到，回退到 {default_name}")
                    name = default_name
                else:
                    raise ValueError(f"Provider {kind}:{name} 未注册且无默认实现")

            provider_cls = self._providers[kind][name]
            self._instances[key] = provider_cls(config or {})

        return self._instances[key]

    def _get_default(self, kind: str) -> Optional[str]:
        """获取默认实现名称"""
        defaults = {
            'attention': 'tva_default',
            'ffn': 'default',
            'router': 'topk_default',
            'align': 'bistate_default',
            'retrieval': 'none',
            'dataset': 'text_default'
        }
        return defaults.get(kind)

    def list_providers(self, kind: Optional[str] = None):
        """列出所有Provider"""
        if kind:
            return list(self._providers.get(kind, {}).keys())
        else:
            return {k: list(v.keys()) for k, v in self._providers.items()}


# 全局单例
registry = Registry()
```

### 2. AttentionProvider 接口

```python
# core/providers/attention.py

from core.registry import Provider
import torch
import torch.nn as nn
from typing import Optional, Tuple

class AttentionProvider(Provider):
    """注意力机制Provider基类"""

    @abstractmethod
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """前向传播

        Returns:
            output: [B, T, D]
            attn_weights: [B, H, T, T] (可选)
        """
        pass

    @abstractmethod
    def create_layer(self, d_model: int, num_heads: int, **kwargs) -> nn.Module:
        """创建注意力层实例"""
        pass
```

### 3. 默认TVA实现

```python
# modeling/layers/attention_tva.py

from core.providers.attention import AttentionProvider
from core.registry import registry

class TVAAttention(AttentionProvider):
    """TVA注意力机制（核心默认实现）"""

    def __init__(self, config):
        self.config = config
        self.r = config.get('r', 4)
        self.s = config.get('s', 1)
        self.tau = config.get('tau', 0.18)

    def get_name(self) -> str:
        return "tva_default"

    def get_version(self) -> str:
        return "1.0.0"

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # TVA核心逻辑
        # ... (保留现有的AutopoieticAttention逻辑)
        pass

    def create_layer(self, d_model, num_heads, **kwargs):
        return TVAAttentionLayer(
            d_model=d_model,
            num_heads=num_heads,
            r=self.r,
            s=self.s,
            tau=self.tau,
            **kwargs
        )

# 自动注册
registry.register('attention', 'tva_default', TVAAttention)
```

### 4. Builder（模型装配骨架）

```python
# modeling/compose.py

from core.registry import registry
from typing import Dict, Any
import torch.nn as nn

class ModelBuilder:
    """模型装配器 - 通过Provider构建模型"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry = registry

    def build_attention(self, d_model: int, num_heads: int):
        """构建注意力层"""
        attn_name = self.config.get('model', {}).get('attention_name', 'tva_default')
        attn_config = self.config.get('model', {}).get('tva', {})

        provider = self.registry.get('attention', attn_name, attn_config)
        return provider.create_layer(d_model, num_heads)

    def build_ffn(self, d_model: int, d_ff: int):
        """构建FFN层"""
        ffn_name = self.config.get('model', {}).get('ffn_name', 'default')
        provider = self.registry.get('ffn', ffn_name)
        return provider.create_layer(d_model, d_ff)

    def build_block(self, d_model: int, num_heads: int, d_ff: int):
        """构建Transformer Block"""
        attn = self.build_attention(d_model, num_heads)
        ffn = self.build_ffn(d_model, d_ff)

        # 检查是否启用MoE（插件）
        if self.config.get('model', {}).get('moe', {}).get('enabled', False):
            router_provider = self.registry.get('router', 'topk_default')
            ffn = router_provider.wrap_ffn(ffn, self.config['model']['moe'])

        return TransformerBlock(attn, ffn)

    def build_model(self):
        """构建完整模型"""
        model_cfg = self.config['model']

        blocks = nn.ModuleList([
            self.build_block(
                d_model=model_cfg['d_model'],
                num_heads=model_cfg['num_heads'],
                d_ff=model_cfg['d_ff']
            )
            for _ in range(model_cfg['num_layers'])
        ])

        return GPTModel(blocks, model_cfg)
```

---

## 📅 迁移计划（3阶段）

### 阶段1：稳定核心（Week 1-2）

**目标：** 核心可独立运行，不装插件也能跑

**步骤：**

1. **创建core模块**
   ```bash
   mkdir -p apt/core
   # 迁移 + 重构
   config.py      ← config/apt_config.py
   logging.py     ← utils/logging_utils.py
   monitor.py     ← utils/resource_monitor.py
   errors.py      ← utils/error_handler.py
   device.py      ← utils/hardware_check.py
   cache.py       ← utils/cache_manager.py
   schedules.py   ← 新建（课程化）
   registry.py    ← 新建（核心） ⭐
   ```

2. **创建Provider接口**
   ```bash
   mkdir -p apt/core/providers
   attention.py   ← 新建
   ffn.py         ← 新建
   router.py      ← 新建
   align.py       ← 新建
   ```

3. **重构modeling为Builder模式**
   ```bash
   mkdir -p apt/modeling/{layers,backbones}
   compose.py                    ← 新建（Builder） ⭐
   layers/attention_tva.py       ← modeling/apt_model.py (提取TVA)
   layers/vft.py                 ← 保留VFT逻辑
   layers/ffn.py                 ← 提取默认FFN
   layers/norm.py                ← 提取LayerNorm
   backbones/gpt.py              ← 简化的GPT骨架
   ```

4. **训练循环添加钩子**
   ```python
   # training/trainer.py

   class Trainer:
       def __init__(self, config):
           self.hooks = []  # 插件钩子列表

       def register_hook(self, hook):
           self.hooks.append(hook)

       def _broadcast_event(self, event_name, **kwargs):
           for hook in self.hooks:
               if hasattr(hook, event_name):
                   getattr(hook, event_name)(**kwargs)

       def train_epoch(self, epoch):
           self._broadcast_event('on_epoch_start', epoch=epoch)

           for step, batch in enumerate(self.dataloader):
               self._broadcast_event('on_step_start', step=step, batch=batch)

               # 训练逻辑
               loss = self.train_step(batch)

               self._broadcast_event('on_step_end', step=step, loss=loss)

           self._broadcast_event('on_epoch_end', epoch=epoch)
   ```

5. **CLI添加profile支持**
   ```python
   # cli/parser.py

   parser.add_argument('-p', '--profile',
                       help='配置文件 (profiles/*.yaml)')
   parser.add_argument('--plugins', nargs='+',
                       help='启用的插件列表')

   # 新增plugin子命令
   plugin_parser = subparsers.add_parser('plugin')
   plugin_sub = plugin_parser.add_subparsers()

   plugin_sub.add_parser('list')      # 列出插件
   plugin_sub.add_parser('enable')    # 启用插件
   plugin_sub.add_parser('disable')   # 禁用插件
   ```

**验证标准：**
- ✅ `apt train -p base.yaml` 可运行（仅用默认TVA/FFN）
- ✅ `apt plugin list` 显示已注册Provider
- ✅ 不加载任何plugins也能完成训练

---

### 阶段2：外插高收益（Week 3-4）

**目标：** MoE/Align作为插件工作

**步骤：**

1. **创建MoE插件**
   ```python
   # plugins/builtin/moe.py

   from core.providers.router import RouterProvider
   from core.registry import registry

   class MoERouter(RouterProvider):
       def __init__(self, config):
           self.experts = config.get('experts', 64)
           self.top_k = config.get('top_k', 2)
           self.capacity = config.get('capacity', 1.25)

       def get_name(self):
           return "topk_moe"

       def wrap_ffn(self, base_ffn, config):
           """将普通FFN包装为MoE"""
           return MoEFFN(base_ffn, self.experts, self.top_k, self.capacity)

       def on_epoch_start(self, epoch):
           """钩子：根据schedules调整capacity"""
           if hasattr(self, 'schedule'):
               self.capacity = self.schedule.get_capacity(epoch)

   # 注册
   registry.register('router', 'topk_moe', MoERouter)
   ```

2. **创建Align插件**
   ```python
   # plugins/builtin/align.py

   from core.providers.align import AlignProvider
   from core.registry import registry

   class BistateAlign(AlignProvider):
       def __init__(self, config):
           self.alpha = config.get('alpha', 0.35)
           self.beta = config.get('beta', 0.20)
           self.tau_align = config.get('tau_align', 0.15)

       def compute_align_loss(self, logits, targets):
           """计算双态数对齐损失"""
           # ... 实现逻辑
           return align_loss

       def on_step_end(self, step, loss):
           """钩子：在训练步结束时添加对齐损失"""
           if step > self.warmup_steps:
               loss += self.compute_align_loss(...)

   registry.register('align', 'bistate_default', BistateAlign)
   ```

3. **Profile配置**
   ```yaml
   # profiles/gpt5_moe_reasoning.yaml

   plugins: ["moe", "align", "routing"]

   model:
     attention_name: tva_default
     d_model: 768
     num_heads: 12
     num_layers: 24

     tva:
       r: 4
       s: 1
       tau: 0.18

     moe:
       enabled: true
       experts: 64
       top_k: 2
       capacity: 1.25

     bistate:
       alpha: 0.35
       beta: 0.20
       tau_align: 0.15

   schedules:
     enable_moe_at_epoch: 2
     enable_align_at_epoch: 3
     route_temp:
       start: 1.5
       end: 0.8
       by: "epoch"
   ```

**验证标准：**
- ✅ `apt train -p gpt5_moe_reasoning.yaml` 自动加载MoE和Align
- ✅ epoch=2时启用MoE，epoch=3时启用Align
- ✅ 插件失败时自动回退（不影响训练）

---

### 阶段3：策略/外部依赖（Week 5-6）

**目标：** 高级插件（检索、投票、Flash Attention、量化）

**步骤：**

1. **Flash Attention替代**
   ```python
   # plugins/flash_attn/flash_v2.py

   try:
       from flash_attn import flash_attn_func
       FLASH_AVAILABLE = True
   except ImportError:
       FLASH_AVAILABLE = False

   class FlashAttentionV2(AttentionProvider):
       def __init__(self, config):
           if not FLASH_AVAILABLE:
               raise ImportError("flash-attn未安装")

       def forward(self, query, key, value, ...):
           return flash_attn_func(query, key, value, ...)

   if FLASH_AVAILABLE:
       registry.register('attention', 'flash_v2', FlashAttentionV2)
   ```

2. **RAG检索插件**
   ```python
   # plugins/builtin/retriever.py

   class RAGRetriever:
       def on_generation_start(self, prompt):
           """在生成前检索相关文档"""
           docs = self.retrieve(prompt)
           return self.augment_prompt(prompt, docs)
   ```

3. **投票一致性插件**
   ```python
   # plugins/builtin/voter.py

   class VotingPlugin:
       def on_high_entropy(self, logits):
           """高熵时启用K=2~3采样投票"""
           if self.compute_entropy(logits) > self.threshold:
               samples = [self.sample(logits) for _ in range(self.k)]
               return self.vote(samples)
           return self.sample(logits)
   ```

**验证标准：**
- ✅ `apt train --plugins flash_attn` 使用Flash Attention
- ✅ Flash Attention不可用时自动回退到TVA
- ✅ RAG检索失败不影响生成

---

## 🎯 关键设计决策

### 1. 为什么用Provider而不是直接继承？

**Provider的优势：**
- ✅ 延迟加载（只有使用时才初始化）
- ✅ 配置驱动（通过profile切换实现）
- ✅ 失败回退（Provider不可用时用默认实现）
- ✅ 版本管理（同一种Provider可有多个版本）

### 2. 为什么需要schedules.py？

**课程化训练的需求：**
- MoE在epoch=2启用（避免早期不稳定）
- Align在epoch=3启用（等模型收敛）
- 路由温度从1.5退火到0.8
- 投票阈值动态调整

**实现：**
```python
# core/schedules.py

class Schedule:
    def __init__(self, config):
        self.config = config

    def should_enable_plugin(self, plugin_name, epoch):
        key = f"enable_{plugin_name}_at_epoch"
        target_epoch = self.config.get('schedules', {}).get(key, 0)
        return epoch >= target_epoch

    def get_param(self, param_name, epoch=None, step=None):
        """获取当前参数值（支持退火）"""
        param_cfg = self.config.get('schedules', {}).get(param_name)
        if isinstance(param_cfg, dict):
            return self._interpolate(param_cfg, epoch, step)
        return param_cfg

    def _interpolate(self, cfg, epoch, step):
        """线性插值"""
        start, end = cfg['start'], cfg['end']
        by = cfg.get('by', 'epoch')

        if by == 'epoch':
            t = epoch / self.config['training']['max_epochs']
        else:
            t = step / self.config['training']['max_steps']

        return start + (end - start) * t
```

### 3. 为什么保留双配置类（临时）？

**迁移策略：**
- 阶段1：保留`APTConfig`和`APTModelConfiguration`共存
- 阶段2：统一为`core/config.py`中的`APTConfig`
- 阶段3：删除`modeling/apt_model.py`中的`APTModelConfiguration`

**原因：**
- ✅ 向后兼容现有代码
- ✅ 渐进式迁移，降低风险
- ✅ 测试两种配置的互操作性

---

## 📦 迁移检查清单

### 阶段1（核心稳定）
- [ ] 创建`apt/core/`目录结构
- [ ] 实现`core/registry.py`
- [ ] 定义所有Provider基类
- [ ] 迁移TVA为`AttentionProvider`
- [ ] 创建`ModelBuilder`
- [ ] 训练循环添加钩子广播
- [ ] CLI支持`-p/--profile`
- [ ] 创建`profiles/base.yaml`
- [ ] 验证：无插件可运行

### 阶段2（高收益插件）
- [ ] 实现MoE插件
- [ ] 实现Align插件
- [ ] 实现Routing插件
- [ ] 实现`core/schedules.py`
- [ ] 创建`profiles/gpt5_moe_reasoning.yaml`
- [ ] 验证：MoE+Align按schedule启用

### 阶段3（外部依赖）
- [ ] Flash Attention插件
- [ ] RAG检索插件
- [ ] 投票插件
- [ ] 量化插件
- [ ] HF数据源插件
- [ ] W&B监控插件
- [ ] 验证：插件失败自动回退

---

## 🚀 快速开始（重构后）

```bash
# 基础训练（仅核心）
apt train -p profiles/base.yaml

# MoE + Align训练
apt train -p profiles/gpt5_moe_reasoning.yaml

# 手动指定插件
apt train -p profiles/base.yaml --plugins moe align flash_attn

# 列出所有Provider
apt plugin list

# 列出特定类型
apt plugin list --kind attention
# 输出: tva_default, flash_v2, linear_causal

# 启用/禁用插件（持久化到profile）
apt plugin enable moe --profile gpt5_moe_reasoning.yaml
apt plugin disable retriever --profile gpt5_moe_reasoning.yaml
```

---

## 📈 预期收益

| 维度 | 重构前 | 重构后 | 改进 |
|------|--------|--------|------|
| **可维护性** | 单体代码，修改影响全局 | 核心稳定，插件隔离 | +80% |
| **可扩展性** | 硬编码新功能 | 注册新Provider | +100% |
| **性能** | 全量加载 | 按需加载 | +30% |
| **稳定性** | 一处失败全挂 | 插件失败可回退 | +60% |
| **配置灵活性** | 改代码 | 改YAML | +200% |
| **测试友好性** | 单元测试困难 | Provider独立测试 | +150% |

---

## ⚠️ 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 性能退化 | 中 | 高 | Provider内联优化、Benchmark对比 |
| 配置复杂度上升 | 高 | 中 | 提供向导、预设profiles |
| 向后兼容性破坏 | 中 | 高 | 保留旧接口、版本标记 |
| 插件冲突 | 低 | 中 | 依赖检查、互斥声明 |
| 文档跟不上 | 高 | 中 | 自动生成、示例丰富 |

---

## 📚 参考资料

- **微内核架构**：[Microkernel Pattern](https://en.wikipedia.org/wiki/Microkernel)
- **Provider模式**：[Provider Pattern in Spring](https://spring.io/blog/2011/08/09/what-s-a-provider)
- **插件系统**：[Plugin Architecture](https://martinfowler.com/articles/plugins.html)
- **配置驱动开发**：[Configuration as Code](https://www.thoughtworks.com/insights/blog/configuration-code)

---

**下一步：** 开始实施阶段1 - 创建核心模块和Provider接口
