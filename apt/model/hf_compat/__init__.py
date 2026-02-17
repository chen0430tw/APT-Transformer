"""
HuggingFace 兼容层

提供 PretrainedConfig / PreTrainedModel 封装，使所有 APT 模型可以:
- AutoModelForCausalLM.from_pretrained() 加载
- save_pretrained() 保存为 safetensors + config.json
- 在 vLLM / OpenRouter / Together AI 上部署
- 配合 AutopoieticAttention 插件使用
"""

from apt.model.hf_compat.configs import (
    GPT4oConfig,
    GPTo3Config,
    GPT5Config,
    Claude4Config,
    APTConfig,
)

from apt.model.hf_compat.modeling_gpt4o import GPT4oForCausalLM
from apt.model.hf_compat.modeling_gpto3 import GPTo3ForCausalLM
from apt.model.hf_compat.modeling_gpt5 import GPT5ForCausalLM
from apt.model.hf_compat.modeling_claude4 import Claude4ForCausalLM
from apt.model.hf_compat.modeling_apt import APTForCausalLM, APTForSeq2SeqLM
from apt.model.hf_compat.tokenization_apt import APTTokenizer, CHATML_TEMPLATE

__all__ = [
    # Configs
    "GPT4oConfig",
    "GPTo3Config",
    "GPT5Config",
    "Claude4Config",
    "APTConfig",
    # Models
    "GPT4oForCausalLM",
    "GPTo3ForCausalLM",
    "GPT5ForCausalLM",
    "Claude4ForCausalLM",
    "APTForCausalLM",
    "APTForSeq2SeqLM",
    # Tokenizer
    "APTTokenizer",
    "CHATML_TEMPLATE",
]
