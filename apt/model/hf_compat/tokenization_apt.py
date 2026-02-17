"""APT Tokenizer — PreTrainedTokenizerFast 封装

将 AdaptiveBPETokenizer 的 tokenizers.Tokenizer 后端直接交给 HF Fast 封装，
从而获得:
- save_pretrained() / from_pretrained() 自动保存/加载 tokenizer.json
- chat_template (Jinja2) 支持 /v1/chat/completions
- 与 vLLM / TGI / OpenRouter 的自动集成
"""

import os
import json
from typing import Optional

from tokenizers import Tokenizer as HFBackendTokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizerFast


# ChatML 风格的 chat_template (与 OpenAI / vLLM 标准兼容)
CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'user' %}"
    "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "<|im_start|>assistant\n"
    "{% endif %}"
)


class APTTokenizer(PreTrainedTokenizerFast):
    """HuggingFace-compatible tokenizer for all APT models.

    Wraps a byte-level BPE tokenizer (from the `tokenizers` library).
    Can be initialized:
    1. From a trained tokenizer.json via from_pretrained()
    2. From an existing AdaptiveBPETokenizer's backend via from_adaptive()
    3. From scratch (empty BPE, mainly for testing)
    """

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        tokenizer_file: Optional[str] = None,
        tokenizer_object=None,
        **kwargs,
    ):
        # 确保特殊 token 有默认值
        kwargs.setdefault("bos_token", "<bos>")
        kwargs.setdefault("eos_token", "<eos>")
        kwargs.setdefault("unk_token", "<unk>")
        kwargs.setdefault("pad_token", "<pad>")

        # ChatML 控制 token
        additional_special = kwargs.pop("additional_special_tokens", None)
        if additional_special is None:
            additional_special = ["<|im_start|>", "<|im_end|>", "<sep>", "<mask>"]
        kwargs["additional_special_tokens"] = additional_special

        super().__init__(
            tokenizer_file=tokenizer_file,
            tokenizer_object=tokenizer_object,
            **kwargs,
        )

        # 如果没有 chat_template，设置默认 ChatML
        if self.chat_template is None:
            self.chat_template = CHATML_TEMPLATE

    @classmethod
    def from_adaptive(cls, adaptive_tokenizer, **kwargs) -> "APTTokenizer":
        """从 AdaptiveBPETokenizer 实例创建 HF 兼容 tokenizer.

        Args:
            adaptive_tokenizer: AdaptiveBPETokenizer 实例 (其 _tokenizer 属性
                                是 tokenizers.Tokenizer)
        """
        backend = adaptive_tokenizer._tokenizer
        return cls(tokenizer_object=backend, **kwargs)

    def save_pretrained(self, save_directory, **kwargs):
        """保存为 HF 标准目录格式 (tokenizer.json + tokenizer_config.json)."""
        os.makedirs(save_directory, exist_ok=True)
        # 让 HF 基类处理保存
        result = super().save_pretrained(save_directory, **kwargs)

        # 确保 tokenizer_config.json 包含 chat_template
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            if "chat_template" not in config:
                config["chat_template"] = self.chat_template
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, ensure_ascii=False, indent=2)

        return result
