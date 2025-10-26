# Memo.txt 最新更新分析

## 📋 概览

memo.txt 最后新增了三个重要的架构指导：

1. **EQI (Evidence Qualitative Inference)** - 证据定性推理的完整数学模型
2. **VFT/TVA 模块化架构** - 核心算子的工厂模式实现建议
3. **管理员模式定位** - 控制平面插件的设计原则

---

## 一、EQI (Evidence Qualitative Inference) 详解

### 📐 数学模型

**完整优化问题**：

```
max_{x,z} φ·Σ_k (E·s_k)·x_k - κ·z

subject to:
  -z ≤ Σ_k s_k·x_k ≤ z    (抑振约束)
  x ∈ X                    (可行域: Ax≤c, Bx=d, lower≤x≤upper)
```

**关键变量**：

1. **φ (软门)**:
   - `φ = sigmoid(a·F - b·P_eq + c·(EVSI - C_wait))`
   - 决定是否执行 (WAIT/ACT)
   - 阈值: `φ ≥ τ` 才执行

2. **E (证据调制)**:
   - `E_k = 1 + η·w_k·Ω_k`
   - `Ω_k = 2·Q_k - 1` (质量分数转为 [-1,1])
   - η: 证据放大系数

3. **s (净效用)**:
   - `s_k = L_k - λ·I_k`
   - L: 收益, I: 成本
   - λ: 成本权衡系数

4. **κ (抑振权重)**:
   - 控制 `|Σ s_k·x_k|` 的大小
   - 避免过度振荡

### 🎯 决策流程

```
1. 计算软门 φ
   ├─ φ < τ → WAIT (证据不足)
   └─ φ ≥ τ → 进入优化

2. 求解 LP/QP
   ├─ 优先: PuLP/HiGHS (精确求解)
   └─ 回退: 贪心启发式 (近似解)

3. 输出决策
   ├─ x*: 最优资源分配
   ├─ 对偶价: 影子价格 (哪个约束在"卡脖子")
   └─ 审计信息: φ, E, s, net_drive, objective
```

### 💡 应用场景

**通用场景**（凡是"像路由"的分配问题）：

1. **计算资源编排**
   - 多GPU任务调度
   - 分布式计算负载均衡
   - MoE专家路由

2. **供应链/产线**
   - 多工厂产能分配
   - 库存优化
   - 物流路由

3. **公共服务**
   - 医疗资源投放
   - 应急响应调度
   - 政策资金分配

4. **能源/管网**
   - 电网调度
   - 水网优化
   - 燃气调度

### 📊 输出可解释性

**完整审计信息**：

```json
{
  "decision": "ACT",
  "x_star": [0.5, 0.3, 0.2],
  "audit": {
    "phi": 0.85,              // 软门强度
    "E": [1.2, 0.9, 1.1],     // 证据调制系数
    "s": [10, -5, 8],         // 净效用
    "net_drive": 3.5,         // |Σ s_k·x_k|
    "objective": 15.2,        // 目标函数值
    "solver": "lp",           // 求解器类型
    "shadow_prices": {...},   // 对偶价 (LP求解时)
    "notes": "ok"
  }
}
```

### ✅ 为什么好用

1. **统一**: 证据、可行性、资源在一个目标里闭合
2. **可审计**: 标准LP/QP，输出对偶价和灵敏度
3. **稳定**: κ控制切换成本和抖动
4. **通用**: 广泛适用于各类分配问题

### ⚠️ 注意事项

1. **证据数值化**: Q, w, Ω 需统一标定
2. **非线性处理**: 强非线性可换成锥/QP或分段线性近似
3. **高动态场景**: 建议叠加排队/Backpressure做细粒度稳定化

---

## 二、VFT/TVA 模块化架构

### 🎯 核心定位

**VFT/TVA 是核心算子，不是插件！**

**两种含义**：

1. **VFT 作为架构** (VFTModel)
   - 完整模型家族（类似Transformer-XL、LLaMA-2）
   - 整网的注意力和FFN都在共享低秩子空间计算

2. **VFT 作为核心模块**
   - 可复用的算子族（TVA注意力、VFT-FFN、Normal补偿）
   - 嵌入任意Transformer block作为"核心实现选择"

### 📁 推荐目录结构

```
apt_model/
├── modeling/
│   ├── blocks/
│   │   ├── vft_tva.py           ← 核心算子集合
│   │   ├── attention_vanilla.py
│   │   └── ffn_variants.py
│   ├── gpt_model.py             ← 使用工厂选择算子
│   ├── vft_model.py             ← 整机VFT变体
│   └── registry.py              ← 算子工厂/注册表
└── ...
```

### 🏭 工厂模式实现

#### 1. 注册表 (modeling/registry.py)

```python
REG_ATTENTION = {}
REG_FFN = {}

def register_attn(name):
    def deco(cls):
        REG_ATTENTION[name] = cls
        return cls
    return deco

def register_ffn(name):
    def deco(cls):
        REG_FFN[name] = cls
        return cls
    return deco

def build_attention(name, **kw):
    return REG_ATTENTION[name](**kw)

def build_ffn(name, **kw):
    return REG_FFN[name](**kw)
```

#### 2. 算子注册 (modeling/blocks/vft_tva.py)

```python
from modeling.registry import register_attn, register_ffn

@register_attn("tva")
class TVAAttention(nn.Module):
    """Tri-Vein Attention: 在r维子空间计算注意力"""
    def __init__(self, d_model, n_heads, rank, attn_dropout=0.0):
        super().__init__()
        # ... TVA实现 ...

@register_ffn("vft")
class VFTFeedForward(nn.Module):
    """VFT-FFN: 在vein子空间的前馈网络"""
    def __init__(self, d_model, rank, drop=0.0):
        super().__init__()
        # ... VFT-FFN实现 ...

@register_attn("vanilla")
class VanillaAttention(nn.Module):
    """标准多头注意力"""
    # ...

@register_ffn("geglu")
class GEGLUFeedForward(nn.Module):
    """GEGLU FFN"""
    # ...
```

#### 3. 模型使用 (modeling/gpt_model.py)

```python
from modeling.registry import build_attention, build_ffn

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, cfg):
        super().__init__()

        # 通过配置选择算子实现
        self.attn = build_attention(
            cfg.attn.impl,           # "tva" or "vanilla"
            d_model=d_model,
            n_heads=n_heads,
            rank=cfg.vft.rank,
            attn_dropout=cfg.attn.drop
        )

        self.ffn = build_ffn(
            cfg.ffn.impl,            # "vft" or "geglu"
            d_model=d_model,
            rank=cfg.vft.rank,
            drop=cfg.ffn.drop
        )

class GPTModel(nn.Module):
    def __init__(self, cfg):
        # ...
        self.blocks = nn.ModuleList([
            GPTBlock(cfg.d_model, cfg.n_heads, cfg)
            for _ in range(cfg.n_layers)
        ])
```

### ⚙️ CLI配置开关

```bash
# 使用TVA注意力 + VFT前馈
python -m apt_model train \
  --attn.impl tva \
  --ffn.impl vft \
  --vft.rank 4 \
  --tau 0.18 \
  --normals 1

# 退回vanilla（对照实验）
python -m apt_model train \
  --attn.impl vanilla \
  --ffn.impl geglu
```

### 📦 部署注意事项

#### 1. Checkpoint兼容性

```python
# 保存时记录配置
torch.save({
    'model_state_dict': model.state_dict(),
    'model_args': {
        'attn_impl': 'tva',
        'ffn_impl': 'vft',
        'rank': 4,
        'tau': 0.18,
        's_normals': 1
    }
}, checkpoint_path)
```

#### 2. 版本管理

```python
# vft_tva.py 中添加版本号
__version__ = "1.0.0"

# 在checkpoint中记录
'vft_tva_version': vft_tva.__version__
```

#### 3. 量化配置

```python
# 量化时保护正交矩阵
quantization_config = {
    'dont_quantize': [
        'VeinProjector.U.weight',
        'VeinProjector.V.weight'
    ],
    # 或使用专用量化策略
    'low_rank_strategy': 'orthogonal_preserve'
}
```

#### 4. CUDA/Flash内核扩展

```python
# 保持相同接口，替换实现
@register_attn("tva_flash")
class TVAAttentionFlash(nn.Module):
    """Flash Attention版本的TVA"""
    # 相同的__init__签名
    # 优化的CUDA实现
```

### ✅ 优势总结

**外置核心模块 + 工厂引用**的好处：

1. ✅ **统一**: 避免重复实现
2. ✅ **灵活**: 训练时切换实现（TVA/vanilla）
3. ✅ **可维护**: 升级一处，全局生效
4. ✅ **A/B测试**: 轻松对比不同算子性能
5. ✅ **向后兼容**: 保持接口，替换实现

### ❌ 什么时候内嵌？

**仅在以下情况内嵌到模型文件**：

1. 一次性实验
2. 强约束的离线包（不想带模块依赖）
3. 单文件分发需求

**正式工程必须外置！**

---

## 三、管理员模式定位

### 🎯 核心原则

**管理员模式 = 插件（控制/运维侧），不是核心！**

### 📍 为什么是插件而非核心？

#### 1. **职责分离**

```
核心 (modeling/)         控制 (admin/)
    ↓                       ↓
前向计算                 权限/审计/配额
算子实现                 熔断/降级
推理路径                 调参保护
VFT/TVA/MoE             速率限制
```

- **核心**: 怎么算（What to compute）
- **控制**: 能不能算、能算多少（Can compute, How much）

#### 2. **独立演进**

- ✅ 策略/合规模板经常变化
- ✅ 灰度/开关需要热更
- ✅ 环境隔离（开发/测试/生产）
- ✅ 不影响训练/推理的可复现性

#### 3. **最小入侵**

- ✅ 避免运维策略烙死在模型里
- ✅ 训练代码保持干净
- ✅ 便于A/B测试和回滚

### 📁 推荐目录结构

```
apt_model/
├── admin/                      ← 插件层
│   ├── __init__.py
│   ├── policy_engine.py        # RBAC/组织策略/配额/白名单
│   ├── guards.py               # 请求前置校验/参数上限/速率限制
│   ├── kill_switch.py          # 熔断/降级/只读模式
│   ├── audit.py                # 操作审计/变更追踪/签名
│   ├── config_lock.py          # 关键超参冻结 (rank/τ/quant)
│   ├── feature_flags.py        # 灰度与开关 (per-team/per-env)
│   └── middleware.py           # 统一中间件入口
└── ...
```

### 🔌 与核心的接口

**仅通过控制器钩子注入**，不修改核心代码：

```python
class AdminMiddleware:
    def __init__(self, policy, quota, audit, kill_switch):
        self.policy = policy
        self.quota = quota
        self.audit = audit
        self.kill = kill_switch

    # 钩子1: 请求前置
    def before_run(self, request):
        """运行前检查"""
        self.kill.check()                    # 熔断检查
        self.policy.validate(request)        # 参数/角色校验
        request = self.policy.rewrite(request)  # 安全改写
        self.quota.reserve(request)          # 资源配额
        self.audit.log("start", request)
        return request

    # 钩子2: 步骤包装
    def wrap_step(self, step_fn):
        """包装每个训练步"""
        def _wrapped(*args, **kwargs):
            self.quota.tick()                # 计量
            return step_fn(*args, **kwargs)
        return _wrapped

    # 钩子3: 指标上报
    def on_metrics(self, metrics):
        """处理指标"""
        self.audit.log("metrics", metrics)
        # 检查异常指标，触发报警

    # 钩子4: 错误处理
    def on_fail(self, error):
        """失败处理"""
        self.audit.log("error", error)
        # 熔断/降级决策
```

### ⚙️ CLI配置示例

```bash
python -m apt_model train \
  --admin.enable true \
  --admin.role admin \
  --admin.policy path/to/policy.yaml \
  --admin.readonly false \
  --admin.quota.tok_per_min 1000000 \
  --admin.kill_switch file:/var/run/apt.kill

# 或通过配置文件
python -m apt_model train --admin.config admin_policy.yaml
```

### 📄 策略配置示例 (admin_policy.yaml)

```yaml
# RBAC配置
rbac:
  admin:
    - safety_override: true
    - param_override: true
    - quota_bypass: true
  ops:
    - safety_override: false
    - param_override: limited  # 仅允许temp, top_p
    - quota: default
  viewer:
    - safety_override: false
    - param_override: false
    - readonly: true

# 参数约束
constraints:
  temperature: [0.0, 2.0]
  rank: [1, 64]           # VFT rank上限
  experts: [1, 128]       # MoE专家数上限
  batch_size: [1, 512]

# 配额限制
quotas:
  default:
    tokens_per_min: 100000
    gpu_hours_per_day: 8
  premium:
    tokens_per_min: 1000000
    gpu_hours_per_day: 24

# 熔断规则
circuit_breaker:
  error_rate_threshold: 0.5
  response_time_threshold_ms: 5000
  min_requests: 100

# 审计
audit:
  log_all_requests: true
  log_param_changes: true
  log_safety_bypasses: true
  retention_days: 90
```

### 🔗 与其他插件的关系

#### 1. **与EQI串联**

```python
# 管理员策略 → EQI约束
admin_policy = load_policy(role="ops")

# 从策略提取约束
A, c = admin_policy.get_resource_constraints()
tau = admin_policy.get_gate_threshold()

# 传给EQI
result = eqi_decide(L, I, Q, w, A, c, cfg=cfg, tau=tau)
```

#### 2. **不干扰VFT/TVA/MoE**

```python
# 仅限制"能用多少"，不改算子本身
constraints = {
    'vft.rank': (1, 64),        # rank上限
    'moe.experts': (1, 128),    # 专家并发上限
    'retrieval.freq': 0.1       # 检索频率上限
}
```

#### 3. **约束训练插件**

```python
# GRPO等训练插件的约束
admin_policy.constrain({
    'learning_rate': (1e-6, 1e-3),
    'kl_divergence': (0, 0.5),
    'data_domains': ['wikipedia', 'books']  # 白名单
})
```

### 📊 功能模块详解

#### 1. policy_engine.py

```python
class PolicyEngine:
    """策略引擎：RBAC + 配额 + 白名单"""

    def validate(self, user, action, params):
        """验证用户权限"""
        role = self.get_role(user)
        if action not in role.allowed_actions:
            raise PermissionError(f"{action} not allowed for {role}")

        # 参数范围检查
        for key, value in params.items():
            if not self.is_valid_param(key, value, role):
                raise ValueError(f"Invalid {key}={value}")

    def rewrite(self, params, role):
        """安全改写参数"""
        # 强制限制
        params['batch_size'] = min(params['batch_size'],
                                   role.max_batch_size)
        return params
```

#### 2. guards.py

```python
class RequestGuard:
    """请求守卫：前置校验 + 速率限制"""

    def check_rate_limit(self, user):
        """速率限制"""
        current_rate = self.rate_tracker.get(user)
        if current_rate > user.quota.tokens_per_min:
            raise RateLimitExceeded()

    def check_concurrent_limit(self, user):
        """并发限制"""
        active_jobs = self.job_tracker.count(user)
        if active_jobs >= user.max_concurrent:
            raise TooManyConcurrentJobs()
```

#### 3. kill_switch.py

```python
class KillSwitch:
    """熔断开关：紧急停止 + 降级"""

    def check(self):
        """检查是否需要熔断"""
        if self.is_triggered():
            raise CircuitBreakerOpen()

    def trigger(self, reason):
        """触发熔断"""
        self.state = 'OPEN'
        self.reason = reason
        self.notify_admins()

    def enter_readonly_mode(self):
        """进入只读模式"""
        self.mode = 'READONLY'
```

#### 4. audit.py

```python
class AuditLogger:
    """审计日志：操作追踪 + 变更记录"""

    def log(self, event_type, details):
        """记录事件"""
        entry = {
            'timestamp': time.time(),
            'type': event_type,
            'user': current_user(),
            'details': details,
            'signature': self.sign(details)
        }
        self.storage.append(entry)

    def track_param_change(self, param, old_value, new_value):
        """追踪参数变更"""
        self.log('param_change', {
            'param': param,
            'old': old_value,
            'new': new_value
        })
```

---

## 四、架构总结

### 🏗️ 三层架构

```
┌─────────────────────────────────────────┐
│        控制层 (admin/)                   │  ← 插件
│  策略/配额/审计/熔断/开关               │
├─────────────────────────────────────────┤
│        核心层 (modeling/)                │  ← 核心
│  VFT/TVA/MoE/算子/推理路径              │
├─────────────────────────────────────────┤
│        决策层 (plugins/)                 │  ← 插件
│  EQI/RAG/蒸馏/剪枝/多模态               │
└─────────────────────────────────────────┘
```

### 📋 定位清单

| 组件 | 层级 | 位置 | 原因 |
|------|------|------|------|
| **VFT/TVA** | 核心 | `modeling/blocks/` | 前向算子，被多个模型复用 |
| **管理员模式** | 插件 | `admin/` | 控制面，运维策略 |
| **EQI** | 插件 | `plugins/optional/` | 决策工具，可选功能 |
| **HuggingFace集成** | 插件 | `plugins/builtin/` | 外部集成 |
| **蒸馏/剪枝** | 插件 | `plugins/optional/` | 优化工具 |

### ✅ 实施优先级

#### 阶段1: 核心模块化 (本周)
1. ✅ 创建 `modeling/blocks/vft_tva.py`
2. ✅ 创建 `modeling/registry.py`
3. ✅ 改造 `gpt_model.py` 使用工厂模式
4. ✅ 添加CLI配置开关

#### 阶段2: 控制层搭建 (下周)
5. ✅ 创建 `admin/` 目录结构
6. ✅ 实现 `policy_engine.py`
7. ✅ 实现 `guards.py` 和 `kill_switch.py`
8. ✅ 集成到训练流程

#### 阶段3: 插件完善 (后续)
9. ✅ 完善EQI插件
10. ✅ 集成其他8个插件
11. ✅ 编写完整测试

---

## 五、关键要点

### 🎯 架构原则

1. **VFT/TVA = 核心**
   - 是"怎么算"的算子实现
   - 外置模块 + 工厂引用
   - 所有模型共享

2. **管理员模式 = 插件**
   - 是"能不能算"的控制逻辑
   - 钩子注入，不改核心
   - 独立演进

3. **EQI = 插件**
   - 是"怎么决策"的工具
   - 可选功能
   - 与其他插件组合

### 📝 命名规范

- **架构名**: `VFTModel` (整机版本)
- **模块名**: `vft_tva.py` (核心算子集合)
- **配置项**: `--attn.impl=tva`, `--ffn.impl=vft`
- **插件名**: `admin/`, `plugins/eqi.py`

### 🔧 配置示例

```bash
# VFT/TVA配置
--attn.impl tva
--ffn.impl vft
--vft.rank 4
--tau 0.18

# 管理员配置
--admin.enable true
--admin.role ops
--admin.policy policy.yaml

# EQI配置
--eqi.enable true
--eqi.lambda_cost 1.0
--eqi.kappa 0.1
```

---

## 六、下一步行动

### 立即执行

1. **移动 vft_tva.py**
   ```bash
   mkdir -p apt_model/modeling/blocks
   mv vft_tva.py apt_model/modeling/blocks/
   ```

2. **创建注册表**
   ```bash
   # 创建 apt_model/modeling/registry.py
   ```

3. **创建管理员目录**
   ```bash
   mkdir -p apt_model/admin
   # 解压 files(2).zip 到 apt_model/admin/
   ```

4. **创建EQI插件目录**
   ```bash
   mkdir -p apt/plugins/optional
   mv eqi.py apt/plugins/optional/
   ```

### 需要实现

1. ✅ 工厂注册表 (`modeling/registry.py`)
2. ✅ 改造现有模型使用工厂模式
3. ✅ 管理员中间件 (`admin/middleware.py`)
4. ✅ 策略引擎 (`admin/policy_engine.py`)
5. ✅ 审计系统 (`admin/audit.py`)

---

**文档生成时间**: 2025-10-25
**基于**: memo.txt 最新更新
**作者**: Claude @ APT Team
