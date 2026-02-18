"""AutopoieticAttention 外挂插件

将自生成注意力机制作为可拆卸的插件层，
不修改基础 HF 模型结构，可自由附加 / 分离。

用法:
    from apt.model.hf_compat.autopoietic_plugin import AutopoieticPlugin

    model = AutoModelForCausalLM.from_pretrained("my-model")
    plugin = AutopoieticPlugin(d_model=2048, num_heads=16)
    plugin.attach(model)          # 附加到所有 attention 层
    # ... 推理或训练 ...
    plugin.detach(model)          # 分离，恢复原始模型

工作原理:
    通过 register_forward_hook 在每个 attention 模块输出后追加低秩自生成扰动。
    基础模型的权重和结构完全不变，插件自带独立的可训练参数。

设计原则:
    - 不修改基础模型任何参数或结构
    - 独立的可训练参数 (auto_u, auto_v, auto_gate, tau)
    - 可序列化: save_pretrained() / load_pretrained()
    - 与 vLLM / HF 标准格式无冲突
"""

import os
import json
import math
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn


class AutopoieticPerturbation(nn.Module):
    """单层自生成扰动模块

    对 attention 输出施加低秩自生成扰动:
        out' = out + alpha * tanh(tau) * merge(delta * gate)
    其中:
        delta = shape(V(U(query)))   低秩投影
        gate  = sigmoid(gate_proj(mean(query)))  逐头门控
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        sr_ratio: int = 4,
        alpha: float = 0.1,
        init_tau: float = 1.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.alpha = float(alpha)

        # 可学习温度
        self.tau = nn.Parameter(torch.ones(1) * float(init_tau))

        # 低秩分量: Δ = V(U(x))
        r = max(4, embed_dim // max(1, int(sr_ratio)))
        self.auto_u = nn.Linear(embed_dim, r, bias=False)
        self.auto_v = nn.Linear(r, embed_dim, bias=False)

        # 逐头门控
        self.auto_gate = nn.Linear(embed_dim, num_heads, bias=True)

    def forward(self, attn_output: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        Args:
            attn_output: attention 层的输出 [B, T, C]
            query: 原始 query 输入 [B, T, C] (用于计算 gate 和 delta)

        Returns:
            扰动后的 attn_output [B, T, C]
        """
        b, t, c = query.shape

        # 逐头门控
        g = torch.sigmoid(self.auto_gate(query.mean(dim=1)))  # [B, H]
        g = g.view(b, self.num_heads, 1, 1)

        # 低秩投影 -> 按 head 分割
        delta = self.auto_u(query)      # [B, T, r]
        delta = self.auto_v(delta)       # [B, T, C]
        # reshape to (B, H, T, D)
        delta = delta.view(b, t, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        # 门控扰动
        perturbed = delta * g  # [B, H, T, D]
        # merge back: (B, H, T, D) -> (B, T, C)
        perturbed = perturbed.transpose(1, 2).contiguous().view(b, t, c)

        return attn_output + self.alpha * torch.tanh(self.tau) * perturbed


class AutopoieticPlugin(nn.Module):
    """可拆卸的自生成注意力插件

    附加到任意 HF 模型的 attention 层上，为每层添加低秩自生成扰动。
    插件有自己独立的可训练参数，不修改基础模型。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int = 0,
        sr_ratio: int = 4,
        alpha: float = 0.1,
        init_tau: float = 1.0,
        target_modules: Optional[List[str]] = None,
    ):
        """
        Args:
            d_model: 模型隐藏维度
            num_heads: 注意力头数
            num_layers: 层数 (0=自动检测)
            sr_ratio: 低秩压缩比
            alpha: 扰动强度
            init_tau: 初始温度
            target_modules: 要 hook 的模块名称模式列表。
                           默认 None 会自动搜索 attention 模块。
        """
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.alpha = alpha
        self.init_tau = init_tau
        self.target_modules = target_modules or [
            "self_attn", "attention", "attn", "self_attention",
        ]

        # 扰动层（按需延迟初始化，attach 时确定数量）
        self.perturbations = nn.ModuleList()
        if num_layers > 0:
            for _ in range(num_layers):
                self.perturbations.append(
                    AutopoieticPerturbation(d_model, num_heads, sr_ratio, alpha, init_tau)
                )

        # hook 句柄列表（用于 detach）
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._attached = False

        # 存储配置（用于序列化）
        self._config = {
            "d_model": d_model,
            "num_heads": num_heads,
            "sr_ratio": sr_ratio,
            "alpha": alpha,
            "init_tau": init_tau,
        }

    def _find_attention_modules(self, model: nn.Module) -> List[Tuple[str, nn.Module]]:
        """在模型中查找所有 attention 模块"""
        found = []
        for name, module in model.named_modules():
            # 检查模块名是否包含 target 关键词
            base_name = name.split(".")[-1] if "." in name else name
            for pattern in self.target_modules:
                if base_name == pattern:
                    found.append((name, module))
                    break
        return found

    def attach(self, model: nn.Module) -> int:
        """将插件附加到模型的所有 attention 层

        Args:
            model: HuggingFace 模型

        Returns:
            附加的层数
        """
        if self._attached:
            raise RuntimeError("Plugin already attached. Call detach() first.")

        attn_modules = self._find_attention_modules(model)
        if not attn_modules:
            raise ValueError(
                f"No attention modules found matching patterns: {self.target_modules}. "
                f"Available modules: {[n for n, _ in model.named_modules()][:20]}..."
            )

        # 如果还没初始化扰动层，现在初始化
        if len(self.perturbations) == 0:
            for _ in attn_modules:
                self.perturbations.append(
                    AutopoieticPerturbation(
                        self.d_model, self.num_heads,
                        self.sr_ratio, self.alpha, self.init_tau,
                    )
                )

        # 确保扰动层数量匹配
        n_attn = len(attn_modules)
        n_pert = len(self.perturbations)
        if n_pert < n_attn:
            for _ in range(n_attn - n_pert):
                self.perturbations.append(
                    AutopoieticPerturbation(
                        self.d_model, self.num_heads,
                        self.sr_ratio, self.alpha, self.init_tau,
                    )
                )

        # 将扰动层移到模型同一设备
        device = next(model.parameters()).device
        self.to(device)

        # 注册 forward hook
        for i, (name, module) in enumerate(attn_modules):
            perturbation = self.perturbations[i]

            def make_hook(pert):
                def hook(mod, input_args, output):
                    # output 可能是 tuple (attn_out, attn_weights) 或单独的 tensor
                    if isinstance(output, tuple):
                        attn_out = output[0]
                        rest = output[1:]
                    else:
                        attn_out = output
                        rest = ()

                    # 获取 query (通常是第一个输入参数)
                    if isinstance(input_args, tuple) and len(input_args) > 0:
                        query = input_args[0]
                    else:
                        query = attn_out  # fallback: 用输出本身

                    # 确保 3D: [B, T, C]
                    if query.dim() == 2:
                        query = query.unsqueeze(0)
                    if attn_out.dim() == 2:
                        attn_out = attn_out.unsqueeze(0)
                        squeeze_back = True
                    else:
                        squeeze_back = False

                    # 施加扰动
                    attn_out = pert(attn_out, query)

                    if squeeze_back:
                        attn_out = attn_out.squeeze(0)

                    if rest:
                        return (attn_out,) + rest
                    return attn_out
                return hook

            handle = module.register_forward_hook(make_hook(perturbation))
            self._hooks.append(handle)

        self._attached = True
        return len(attn_modules)

    def detach(self, model: nn.Module = None):
        """从模型分离插件，恢复原始行为

        Args:
            model: 未使用，保留接口一致性
        """
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._attached = False

    @property
    def is_attached(self) -> bool:
        return self._attached

    def save_pretrained(self, save_directory: str):
        """保存插件参数到目录"""
        os.makedirs(save_directory, exist_ok=True)

        # 保存配置
        config_path = os.path.join(save_directory, "autopoietic_config.json")
        config = {**self._config, "num_layers": len(self.perturbations)}
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # 保存权重
        weights_path = os.path.join(save_directory, "autopoietic_plugin.pt")
        torch.save(self.state_dict(), weights_path)

    @classmethod
    def from_pretrained(cls, save_directory: str, **kwargs) -> "AutopoieticPlugin":
        """从目录加载插件"""
        config_path = os.path.join(save_directory, "autopoietic_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        plugin = cls(
            d_model=config["d_model"],
            num_heads=config["num_heads"],
            num_layers=config.get("num_layers", 0),
            sr_ratio=config.get("sr_ratio", 4),
            alpha=config.get("alpha", 0.1),
            init_tau=config.get("init_tau", 1.0),
            **kwargs,
        )

        weights_path = os.path.join(save_directory, "autopoietic_plugin.pt")
        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location="cpu")
            plugin.load_state_dict(state)

        return plugin

    def __repr__(self):
        return (
            f"AutopoieticPlugin(d_model={self.d_model}, num_heads={self.num_heads}, "
            f"layers={len(self.perturbations)}, alpha={self.alpha}, attached={self._attached})"
        )
