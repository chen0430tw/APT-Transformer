# APT 微内核架构迁移指南

## 📋 概述

本指南提供**逐步、可验证**的迁移步骤，将APT从单体架构迁移到微内核+插件架构。

**预计时间：** 6周
**风险等级：** 中（通过渐进式迁移降低）
**向后兼容：** 保持2个版本的过渡期

---

## 🎯 迁移原则

1. **渐进式**：分3阶段，每阶段可独立验证
2. **可回滚**：每阶段完成后打tag，出问题可回退
3. **向后兼容**：保留旧接口至少2个版本
4. **测试驱动**：每步都有验证标准
5. **文档同步**：代码和文档同步更新

---

## 📅 阶段1：稳定核心（Week 1-2）

### 目标
✅ 核心可独立运行，不装插件也能跑
✅ 创建Provider接口和注册表
✅ 训练循环支持钩子广播

### 步骤详解

#### Step 1.1：创建核心目录结构（Day 1）

```bash
# 创建新目录（与旧目录并存）
mkdir -p apt/{core,core/providers}

# 迁移并重构
cp apt_model/config/apt_config.py apt/core/config.py
cp apt_model/utils/logging_utils.py apt/core/logging.py
cp apt_model/utils/resource_monitor.py apt/core/monitor.py
cp apt_model/utils/error_handler.py apt/core/errors.py
cp apt_model/utils/hardware_check.py apt/core/device.py
cp apt_model/utils/cache_manager.py apt/core/cache.py

# 创建新文件
touch apt/core/schedules.py
touch apt/core/registry.py
```

**验证：**
```bash
python -c "from apt.core import config, logging, registry"
# 应无报错
```

#### Step 1.2：实现核心Registry（Day 2）

创建 `apt/core/registry.py`（参考 `examples/core_registry.py`）

**关键代码：**
```python
# apt/core/registry.py

from abc import ABC, abstractmethod
from typing import Dict, Type, Any, Optional
import warnings

class Provider(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_version(self) -> str:
        pass

class Registry:
    def __init__(self):
        self._providers = {}
        self._instances = {}
        self._defaults = {
            'attention': 'tva_default',
            'ffn': 'default',
            # ...
        }

    def register(self, kind: str, name: str, provider_cls: Type[Provider]):
        if kind not in self._providers:
            self._providers[kind] = {}
        self._providers[kind][name] = provider_cls
        print(f"✅ 注册 {kind}:{name}")

    def get(self, kind: str, name: str, config=None) -> Provider:
        key = f"{kind}:{name}"
        if key in self._instances:
            return self._instances[key]

        # 查找或回退
        if kind not in self._providers or name not in self._providers[kind]:
            default = self._defaults.get(kind)
            if default:
                warnings.warn(f"⚠️ 回退到 {kind}:{default}")
                name = default
            else:
                raise ValueError(f"❌ {key} 未注册")

        # 创建实例
        provider_cls = self._providers[kind][name]
        self._instances[key] = provider_cls(config or {})
        return self._instances[key]

# 全局单例
registry = Registry()
```

**验证：**
```python
# 测试脚本
from apt.core.registry import registry, Provider

class DummyAttention(Provider):
    def __init__(self, config): pass
    def get_name(self): return "dummy"
    def get_version(self): return "1.0.0"

registry.register('attention', 'dummy', DummyAttention)
attn = registry.get('attention', 'dummy')
print(f"✅ Registry 工作正常: {attn}")
```

#### Step 1.3：定义Provider接口（Day 3）

创建各种Provider基类：

```bash
touch apt/core/providers/__init__.py
touch apt/core/providers/attention.py
touch apt/core/providers/ffn.py
touch apt/core/providers/router.py
touch apt/core/providers/align.py
touch apt/core/providers/retrieval.py
```

**示例 - AttentionProvider:**
```python
# apt/core/providers/attention.py

from apt.core.registry import Provider
from abc import abstractmethod
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
        """
        前向传播

        Args:
            query: [B, T, D]
            key: [B, S, D]
            value: [B, S, D]
            attn_mask: [T, S] 或 [B, T, S]
            key_padding_mask: [B, S]

        Returns:
            output: [B, T, D]
            attn_weights: [B, H, T, S] (可选)
        """
        pass

    @abstractmethod
    def create_layer(self, d_model: int, num_heads: int, **kwargs) -> nn.Module:
        """创建注意力层实例"""
        pass
```

**验证：**
```python
from apt.core.providers.attention import AttentionProvider
# 能导入即可
```

#### Step 1.4：迁移TVA为Provider（Day 4-5）

```bash
# 创建新目录
mkdir -p apt/modeling/layers

# 提取TVA代码
# 从 apt_model/modeling/apt_model.py 提取 AutopoieticAttention
```

**关键代码：**
```python
# apt/modeling/layers/attention_tva.py

from apt.core.providers.attention import AttentionProvider
from apt.core.registry import registry
import torch
import torch.nn as nn

class TVAAttention(AttentionProvider):
    """TVA（自生成注意力）- 核心默认实现"""

    def __init__(self, config):
        self.r = config.get('r', 4)
        self.s = config.get('s', 1)
        self.tau = config.get('tau', 0.18)

    def get_name(self):
        return "tva_default"

    def get_version(self):
        return "1.0.0"

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # 保留现有的AutopoieticAttention核心逻辑
        # ...
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

class TVAAttentionLayer(nn.Module):
    """TVA注意力层"""
    def __init__(self, d_model, num_heads, r, s, tau, **kwargs):
        super().__init__()
        # 初始化层参数
        # ...

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        # 前向传播逻辑
        # ...

# 自动注册（模块导入时执行）
registry.register('attention', 'tva_default', TVAAttention, default=True)
```

**验证：**
```python
from apt.modeling.layers.attention_tva import TVAAttention
from apt.core.registry import registry

# 检查是否自动注册
providers = registry.list_providers('attention')
assert 'tva_default' in providers['attention']

# 创建实例
tva = registry.get('attention', 'tva_default', {'r': 4, 'tau': 0.18})
layer = tva.create_layer(d_model=768, num_heads=12)
print(f"✅ TVA Provider 工作正常")
```

#### Step 1.5：创建ModelBuilder（Day 6-7）

```bash
touch apt/modeling/compose.py
```

**关键代码：**
```python
# apt/modeling/compose.py

from apt.core.registry import registry
from typing import Dict, Any
import torch.nn as nn

class ModelBuilder:
    """模型装配器 - 通过Provider构建模型"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.registry = registry

    def build_attention(self, d_model: int, num_heads: int):
        """构建注意力层"""
        model_cfg = self.config.get('model', {})
        attn_name = model_cfg.get('attention_name', 'tva_default')
        attn_config = model_cfg.get('tva', {})

        provider = self.registry.get('attention', attn_name, attn_config)
        return provider.create_layer(d_model, num_heads)

    def build_ffn(self, d_model: int, d_ff: int):
        """构建FFN层"""
        model_cfg = self.config.get('model', {})
        ffn_name = model_cfg.get('ffn_name', 'default')

        provider = self.registry.get('ffn', ffn_name)
        return provider.create_layer(d_model, d_ff)

    def build_block(self, d_model, num_heads, d_ff):
        """构建Transformer Block"""
        attn = self.build_attention(d_model, num_heads)
        ffn = self.build_ffn(d_model, d_ff)

        # 检查是否启用MoE插件
        if self.config.get('model', {}).get('moe', {}).get('enabled', False):
            try:
                router = self.registry.get('router', 'topk_moe')
                ffn = router.wrap_ffn(ffn, self.config['model']['moe'])
            except ValueError:
                # MoE插件未加载，忽略
                pass

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

class TransformerBlock(nn.Module):
    """基础Transformer Block"""
    def __init__(self, attention, ffn):
        super().__init__()
        self.attention = attention
        self.ffn = ffn
        self.norm1 = nn.LayerNorm(attention.d_model)
        self.norm2 = nn.LayerNorm(attention.d_model)

    def forward(self, x, mask=None):
        # 自注意力
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x
```

**验证：**
```python
# 测试Builder
import yaml
from apt.modeling.compose import ModelBuilder

config = yaml.safe_load(open('examples/profiles/tiny_debug.yaml'))
builder = ModelBuilder(config)

# 构建单个block
block = builder.build_block(d_model=64, num_heads=4, d_ff=256)
print(f"✅ ModelBuilder 构建成功: {block}")

# 构建完整模型
model = builder.build_model()
print(f"✅ 完整模型构建成功: {model}")
```

#### Step 1.6：训练循环添加钩子（Day 8-9）

修改 `apt/training/trainer.py`：

```python
# apt/training/trainer.py

class Trainer:
    def __init__(self, config):
        self.config = config
        self.hooks = []  # 插件钩子列表

    def register_hook(self, hook):
        """注册钩子（由插件调用）"""
        self.hooks.append(hook)
        print(f"✅ 注册钩子: {hook.__class__.__name__}")

    def _broadcast_event(self, event_name, **kwargs):
        """广播事件到所有钩子"""
        for hook in self.hooks:
            if hasattr(hook, event_name):
                try:
                    getattr(hook, event_name)(**kwargs)
                except Exception as e:
                    print(f"⚠️ 钩子 {hook.__class__.__name__}.{event_name} 失败: {e}")

    def train(self):
        """主训练循环"""
        self._broadcast_event('on_training_start', trainer=self)

        for epoch in range(self.config['training']['max_epochs']):
            self.train_epoch(epoch)

        self._broadcast_event('on_training_end', trainer=self)

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self._broadcast_event('on_epoch_start', epoch=epoch, trainer=self)

        for step, batch in enumerate(self.dataloader):
            self._broadcast_event('on_step_start', step=step, batch=batch, trainer=self)

            # 训练步骤
            loss = self.train_step(batch)

            self._broadcast_event('on_step_end', step=step, loss=loss, trainer=self)

        self._broadcast_event('on_epoch_end', epoch=epoch, trainer=self)

    def train_step(self, batch):
        """单步训练（保持原逻辑）"""
        # ... 原有训练逻辑
        pass
```

**验证：**
```python
# 测试钩子系统
class DummyHook:
    def on_epoch_start(self, epoch, trainer):
        print(f"钩子触发: epoch {epoch} 开始")

    def on_step_end(self, step, loss, trainer):
        if step % 10 == 0:
            print(f"钩子触发: step {step}, loss={loss}")

trainer = Trainer(config)
trainer.register_hook(DummyHook())
# trainer.train()  # 运行训练查看钩子是否触发
```

#### Step 1.7：CLI添加Profile支持（Day 10）

修改 `apt/cli/parser.py`：

```python
# apt/cli/parser.py

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="APT Model Training Tool")

    # 添加profile参数 ⭐
    parser.add_argument('-p', '--profile',
                        help='配置文件路径 (profiles/*.yaml)')

    # 添加plugins参数
    parser.add_argument('--plugins', nargs='+',
                        help='启用的插件列表 (覆盖profile设置)')

    # 原有参数...
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    # ...

    # 添加plugin子命令
    subparsers = parser.add_subparsers(dest='command')

    # plugin list
    plugin_list_parser = subparsers.add_parser('plugin-list',
                                                help='列出所有已注册的Provider')
    plugin_list_parser.add_argument('--kind', help='Provider类型')

    # plugin info
    plugin_info_parser = subparsers.add_parser('plugin-info',
                                                help='查看Provider详细信息')
    plugin_info_parser.add_argument('kind', help='Provider类型')
    plugin_info_parser.add_argument('name', help='Provider名称')

    return parser.parse_args()
```

修改 `apt/cli/commands.py`：

```python
# apt/cli/commands.py

import yaml
from apt.core.registry import registry

def run_plugin_list_command(args):
    """列出所有Provider"""
    providers = registry.list_providers(args.kind)

    print("\n=== 已注册的 Provider ===")
    for kind, names in providers.items():
        print(f"\n{kind}:")
        for name in names:
            info = registry.get_info(kind, name)
            default_mark = " (默认)" if info['is_default'] else ""
            print(f"  - {name} v{info['version']}{default_mark}")

def run_plugin_info_command(args):
    """查看Provider详细信息"""
    info = registry.get_info(args.kind, args.name)

    print(f"\n=== {args.kind}:{args.name} ===")
    for key, value in info.items():
        print(f"{key}: {value}")

def load_profile(profile_path):
    """加载配置文件"""
    with open(profile_path, 'r') as f:
        return yaml.safe_load(f)
```

**验证：**
```bash
# 测试CLI
python -m apt plugin-list
python -m apt plugin-list --kind attention
python -m apt plugin-info attention tva_default
```

#### Step 1.8：端到端验证（Day 11-12）

```bash
# 使用tiny profile运行训练
python -m apt train -p examples/profiles/tiny_debug.yaml

# 预期输出：
# ✅ 注册 attention:tva_default
# ✅ 加载配置: tiny_debug
# ✅ 构建模型...
# Epoch 1/3: loss=...
# Epoch 2/3: loss=...
# Epoch 3/3: loss=...
# ✅ 训练完成
```

**验证清单：**
- [ ] 配置文件正确加载
- [ ] 模型通过Builder构建
- [ ] TVA attention正常工作
- [ ] 训练循环完整运行
- [ ] 钩子事件正常广播
- [ ] 无需插件也能运行

**如果验证通过，打tag：**
```bash
git tag -a v2.0.0-stage1 -m "Stage 1: Core infrastructure complete"
git push origin v2.0.0-stage1
```

---

## 📅 阶段2：外插高收益（Week 3-4）

### 目标
✅ MoE/Align作为插件工作
✅ Schedules课程化生效
✅ 插件可动态启用/禁用

### 步骤详解

#### Step 2.1：实现Schedules（Day 1-2）

```python
# apt/core/schedules.py

class Schedule:
    """课程化调度器"""

    def __init__(self, config):
        self.config = config.get('schedules', {})
        self.max_epochs = config['training']['max_epochs']

    def should_enable_plugin(self, plugin_name, epoch):
        """判断是否应启用插件"""
        key = f"enable_{plugin_name}_at_epoch"
        target_epoch = self.config.get(key, 0)
        return epoch >= target_epoch

    def get_param(self, param_name, epoch=None, step=None):
        """获取参数当前值（支持退火）"""
        param_cfg = self.config.get(param_name)

        if param_cfg is None:
            return None

        if not isinstance(param_cfg, dict):
            # 静态值
            return param_cfg

        # 动态退火
        return self._interpolate(param_cfg, epoch, step)

    def _interpolate(self, cfg, epoch, step):
        """线性插值"""
        start = cfg['start']
        end = cfg['end']
        by = cfg.get('by', 'epoch')

        if by == 'epoch' and epoch is not None:
            t = min(epoch / self.max_epochs, 1.0)
        elif by == 'step' and step is not None:
            total_steps = self.max_epochs * cfg.get('steps_per_epoch', 1000)
            t = min(step / total_steps, 1.0)
        else:
            return start

        # 线性插值
        value = start + (end - start) * t
        return value
```

**验证：**
```python
from apt.core.schedules import Schedule

config = {
    'training': {'max_epochs': 10},
    'schedules': {
        'enable_moe_at_epoch': 2,
        'route_temp': {'start': 1.5, 'end': 0.8, 'by': 'epoch'}
    }
}

schedule = Schedule(config)

# 测试插件启用
assert schedule.should_enable_plugin('moe', epoch=1) == False
assert schedule.should_enable_plugin('moe', epoch=2) == True

# 测试参数退火
temp_e0 = schedule.get_param('route_temp', epoch=0)
temp_e5 = schedule.get_param('route_temp', epoch=5)
temp_e10 = schedule.get_param('route_temp', epoch=10)

assert temp_e0 == 1.5
assert 1.0 < temp_e5 < 1.5
assert temp_e10 == 0.8

print("✅ Schedules 验证通过")
```

#### Step 2.2：创建MoE插件（Day 3-5）

```bash
mkdir -p apt/plugins/builtin
touch apt/plugins/builtin/__init__.py
touch apt/plugins/builtin/moe.py
```

```python
# apt/plugins/builtin/moe.py

from apt.core.providers.router import RouterProvider
from apt.core.registry import registry
import torch
import torch.nn as nn

class MoERouter(RouterProvider):
    """MoE专家路由Provider"""

    def __init__(self, config):
        self.experts = config.get('experts', 64)
        self.top_k = config.get('top_k', 2)
        self.capacity = config.get('capacity', 1.25)
        self.balance_loss_weight = config.get('balance_loss', 0.01)

    def get_name(self):
        return "topk_moe"

    def get_version(self):
        return "1.0.0"

    def wrap_ffn(self, base_ffn, config):
        """将普通FFN包装为MoE"""
        return MoEFFN(
            base_ffn=base_ffn,
            num_experts=self.experts,
            top_k=self.top_k,
            capacity=self.capacity
        )

    def on_epoch_start(self, epoch, trainer):
        """钩子：根据schedules调整capacity"""
        schedule = trainer.schedule
        if schedule:
            new_capacity = schedule.get_param('moe_capacity', epoch=epoch)
            if new_capacity:
                self.capacity = new_capacity
                print(f"📊 MoE capacity 调整为: {new_capacity:.2f}")

class MoEFFN(nn.Module):
    """MoE前馈网络"""

    def __init__(self, base_ffn, num_experts, top_k, capacity):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity = capacity

        # 复制base_ffn作为专家
        self.experts = nn.ModuleList([
            self._clone_ffn(base_ffn) for _ in range(num_experts)
        ])

        # 路由网络
        self.router = nn.Linear(base_ffn.d_model, num_experts)

    def _clone_ffn(self, base_ffn):
        """克隆FFN"""
        # ... 实现克隆逻辑
        pass

    def forward(self, x):
        """
        前向传播

        Args:
            x: [B, T, D]

        Returns:
            output: [B, T, D]
            aux_loss: 负载均衡损失
        """
        B, T, D = x.shape

        # 路由打分
        router_logits = self.router(x)  # [B, T, E]
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-K选择
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # 容量限制
        # ... 实现capacity机制

        # 调用专家
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = top_k_indices[..., k]
            expert_weight = top_k_probs[..., k]

            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += expert_weight[mask].unsqueeze(-1) * expert_output

        # 负载均衡损失
        aux_loss = self._compute_balance_loss(router_probs)

        return output, aux_loss

    def _compute_balance_loss(self, router_probs):
        """计算负载均衡损失"""
        # ... 实现负载均衡损失
        pass

# 注册MoE Provider
registry.register('router', 'topk_moe', MoERouter)
```

**验证：**
```python
# 测试MoE插件
from apt.plugins.builtin.moe import MoERouter
from apt.core.registry import registry

# 获取MoE Provider
moe_config = {'experts': 8, 'top_k': 2, 'capacity': 1.5}
moe = registry.get('router', 'topk_moe', moe_config)

# 包装FFN
base_ffn = SomeDummyFFN(d_model=64, d_ff=256)
moe_ffn = moe.wrap_ffn(base_ffn, moe_config)

# 前向传播测试
x = torch.randn(2, 10, 64)
output, aux_loss = moe_ffn(x)

assert output.shape == x.shape
assert aux_loss.item() >= 0

print("✅ MoE 插件验证通过")
```

#### Step 2.3：创建Align插件（Day 6-7）

```python
# apt/plugins/builtin/align.py

from apt.core.providers.align import AlignProvider
from apt.core.registry import registry
import torch

class BistateAlign(AlignProvider):
    """双态数对齐Provider"""

    def __init__(self, config):
        self.alpha = config.get('alpha', 0.35)
        self.beta = config.get('beta', 0.20)
        self.tau_align = config.get('tau_align', 0.15)
        self.enabled = False

    def get_name(self):
        return "bistate_default"

    def get_version(self):
        return "1.0.0"

    def compute_align_loss(self, logits, targets, model_state):
        """
        计算双态数对齐损失

        Args:
            logits: [B, T, V]
            targets: [B, T]
            model_state: 模型内部状态

        Returns:
            align_loss: scalar
        """
        if not self.enabled:
            return torch.tensor(0.0)

        # 提取稳定态和对齐态
        stable_state = model_state.get('stable')
        align_state = model_state.get('align')

        if stable_state is None or align_state is None:
            return torch.tensor(0.0)

        # 计算对齐损失
        align_loss = self.alpha * self._stable_loss(stable_state, logits) + \
                     self.beta * self._align_loss(align_state, logits)

        return align_loss

    def _stable_loss(self, stable_state, logits):
        """稳定态损失"""
        # ... 实现稳定态损失
        pass

    def _align_loss(self, align_state, logits):
        """对齐态损失"""
        # ... 实现对齐态损失
        pass

    def on_epoch_start(self, epoch, trainer):
        """钩子：根据schedules启用对齐"""
        schedule = trainer.schedule
        if schedule and schedule.should_enable_plugin('align', epoch):
            self.enabled = True
            print(f"✅ 启用双态数对齐 (epoch={epoch})")

    def on_step_end(self, step, loss, trainer):
        """钩子：添加对齐损失"""
        if self.enabled and hasattr(trainer, 'model_state'):
            align_loss = self.compute_align_loss(
                trainer.last_logits,
                trainer.last_targets,
                trainer.model_state
            )
            trainer.total_loss += align_loss

# 注册
registry.register('align', 'bistate_default', BistateAlign)
```

#### Step 2.4：集成到Trainer（Day 8）

修改 `apt/training/trainer.py`：

```python
# apt/training/trainer.py (添加)

def __init__(self, config):
    # ...原有初始化
    self.schedule = Schedule(config)
    self.load_plugins()

def load_plugins(self):
    """加载配置中指定的插件"""
    enabled_plugins = self.config.get('plugins', [])

    for plugin_name in enabled_plugins:
        try:
            # 动态导入插件
            module = __import__(f'apt.plugins.builtin.{plugin_name}',
                               fromlist=[plugin_name])

            # 获取Provider（触发注册）
            # ...

            print(f"✅ 加载插件: {plugin_name}")
        except Exception as e:
            print(f"⚠️ 加载插件 {plugin_name} 失败: {e}")

def train_epoch(self, epoch):
    """训练epoch（添加schedule检查）"""
    self._broadcast_event('on_epoch_start', epoch=epoch, trainer=self)

    # 检查是否应启用新插件
    self._check_and_enable_plugins(epoch)

    # ... 原有训练逻辑

def _check_and_enable_plugins(self, epoch):
    """检查并启用插件"""
    for plugin in self.config.get('plugins', []):
        if self.schedule.should_enable_plugin(plugin, epoch):
            # 插件在on_epoch_start钩子中自行启用
            pass
```

#### Step 2.5：端到端验证（Day 9-10）

```bash
# 使用MoE profile运行
python -m apt train -p examples/profiles/gpt5_moe_reasoning.yaml --epochs 5

# 预期输出：
# Epoch 1/5: loss=... (无MoE/Align)
# Epoch 2/5: loss=...
#   ✅ 启用MoE (epoch=2)
#   📊 MoE capacity=1.50
# Epoch 3/5: loss=...
#   ✅ 启用双态数对齐 (epoch=3)
# Epoch 4/5: loss=...
#   📊 MoE capacity=1.38 (退火中)
# Epoch 5/5: loss=...
#   📊 MoE capacity=1.25
```

**验证清单：**
- [ ] MoE在epoch=2启用
- [ ] Align在epoch=3启用
- [ ] Capacity参数正确退火
- [ ] 插件失败不影响训练
- [ ] `plugin-list`显示所有插件

**打tag：**
```bash
git tag -a v2.0.0-stage2 -m "Stage 2: MoE and Align plugins"
git push origin v2.0.0-stage2
```

---

## 📅 阶段3：策略/外部依赖（Week 5-6）

### 目标
✅ Flash Attention替代内核
✅ RAG检索插件
✅ 投票一致性插件
✅ 量化/导出插件

### 步骤详解

#### Step 3.1：Flash Attention插件（Day 1-2）

```bash
mkdir -p apt/plugins/flash_attn
touch apt/plugins/flash_attn/__init__.py
touch apt/plugins/flash_attn/flash_v2.py
```

```python
# apt/plugins/flash_attn/flash_v2.py

try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False

from apt.core.providers.attention import AttentionProvider
from apt.core.registry import registry
import torch
import torch.nn as nn

if FLASH_AVAILABLE:
    class FlashAttentionV2(AttentionProvider):
        """Flash Attention v2 Provider"""

        def __init__(self, config):
            pass

        def get_name(self):
            return "flash_v2"

        def get_version(self):
            return "2.0.0"

        def get_dependencies(self):
            return ["flash-attn>=2.0.0"]

        def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
            """使用Flash Attention加速"""
            # flash_attn_func 要求 [B, T, H, D]
            B, T, D = query.shape
            H = self.num_heads
            head_dim = D // H

            q = query.view(B, T, H, head_dim)
            k = key.view(B, -1, H, head_dim)
            v = value.view(B, -1, H, head_dim)

            output = flash_attn_func(q, k, v)
            output = output.view(B, T, D)

            return output, None  # Flash Attention不返回权重

        def create_layer(self, d_model, num_heads, **kwargs):
            return FlashAttentionLayer(d_model, num_heads)

    class FlashAttentionLayer(nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.d_model = d_model
            self.num_heads = num_heads
            # ... 初始化

        def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
            # ... 调用flash_attn_func

    # 注册
    registry.register('attention', 'flash_v2', FlashAttentionV2)
    print("✅ Flash Attention v2 可用")

else:
    print("⚠️ flash-attn未安装，跳过Flash Attention插件")
```

**验证：**
```bash
# 尝试使用Flash Attention
python -m apt train -p tiny_debug.yaml --model.attention_name flash_v2

# 如果flash-attn未安装，应自动回退：
# ⚠️ attention:flash_v2 未找到，回退到 attention:tva_default
```

#### Step 3.2：其他插件（Day 3-6）

按类似方式实现：
- RAG检索（`plugins/builtin/retriever.py`）
- 投票（`plugins/builtin/voter.py`）
- 量化（`plugins/quant/`）

#### Step 3.3：最终验证（Day 7-10）

**综合测试：**
```bash
# 测试1：纯核心（无插件）
python -m apt train -p tiny_debug.yaml

# 测试2：MoE+Align
python -m apt train -p gpt5_moe_reasoning.yaml

# 测试3：Flash Attention（如果可用）
python -m apt train -p tiny_debug.yaml --model.attention_name flash_v2

# 测试4：插件列表
python -m apt plugin-list

# 测试5：插件信息
python -m apt plugin-info router topk_moe
```

**最终检查清单：**
- [ ] 核心可独立运行
- [ ] 插件可按需加载
- [ ] Schedules正确生效
- [ ] 插件失败自动回退
- [ ] 配置文件驱动
- [ ] 向后兼容（旧代码仍可运行）
- [ ] 文档完整
- [ ] 测试覆盖80%+

**打final tag：**
```bash
git tag -a v2.0.0 -m "APT v2.0.0: Microkernel Architecture"
git push origin v2.0.0
```

---

## 🔄 回滚计划

如果某阶段出现问题：

```bash
# 回退到阶段1
git checkout v2.0.0-stage1

# 回退到阶段2
git checkout v2.0.0-stage2

# 回退到旧版本
git checkout v1.x.x
```

---

## 📚 相关文档

- 主方案：[REFACTOR_PLAN.md](./REFACTOR_PLAN.md)
- 示例代码：`examples/`
- Profile示例：`examples/profiles/`

---

**下一步：** 开始执行阶段1 Step 1.1
