"""APTForCausalLM / APTForSeq2SeqLM — HuggingFace 兼容的 APT 封装

APT 支持两种模式:
- decoder_only=True  → CausalLM (GPT 路径, vLLM/OpenRouter 兼容)
- decoder_only=False → Seq2Seq (encoder-decoder 路径)

对外统一暴露 APTForCausalLM (decoder-only) 和 APTForSeq2SeqLM (encoder-decoder)。
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, Seq2SeqLMOutput

from apt.model.hf_compat.configs import APTConfig


class APTForCausalLM(PreTrainedModel, GenerationMixin):
    """APT decoder-only 模式的 HF 封装 (vLLM/OpenRouter 兼容)"""

    config_class = APTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    # 声明 output_projection.weight 与 token_embedding.weight 绑定，
    # 让 HF 的 save_pretrained() 不重复保存
    _tied_weights_keys = {"model.output_projection.weight": "model.token_embedding.weight"}

    def __init__(self, config: APTConfig):
        super().__init__(config)
        from apt.model.architectures.apt_model import APTModel, APTModelConfiguration

        apt_config = APTModelConfiguration(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
            use_dbc_dac=config.use_dbc_dac,
            use_rmsnorm=getattr(config, "use_rmsnorm", True),
            use_swiglu=getattr(config, "use_swiglu", True),
            pad_token_id=config.pad_token_id,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            decoder_only=True,
        )
        self.model = APTModel(apt_config)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.token_embedding

    def set_input_embeddings(self, value):
        self.model.token_embedding = value

    def get_output_embeddings(self):
        return self.model.output_projection

    def set_output_embeddings(self, value):
        self.model.output_projection = value

    def tie_weights(self, **kwargs):
        """绑定 token_embedding 和 output_projection 的权重

        APT 模型在构造时已内部绑定 (apt_model.py:1570)，
        此方法确保 HF 的 from_pretrained() 加载后也能正确重新绑定。
        """
        if getattr(self.config, "tie_word_embeddings", False):
            input_embeddings = self.get_input_embeddings()
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None and input_embeddings is not None:
                output_embeddings.weight = input_embeddings.weight

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        # APTModel.forward routes to forward_lm when decoder_only=True
        logits = self.model(src_tokens=input_ids, src_key_padding_mask=attention_mask)
        if isinstance(logits, tuple):
            logits = logits[0]
        if isinstance(logits, dict):
            logits = logits["logits"]

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}


class APTForSeq2SeqLM(PreTrainedModel, GenerationMixin):
    """APT encoder-decoder 模式的 HF 封装"""

    config_class = APTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _tied_weights_keys = {"model.output_projection.weight": "model.token_embedding.weight"}

    def __init__(self, config: APTConfig):
        super().__init__(config)
        from apt.model.architectures.apt_model import APTModel, APTModelConfiguration

        apt_config = APTModelConfiguration(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            max_seq_len=config.max_seq_len,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
            use_dbc_dac=config.use_dbc_dac,
            use_rmsnorm=getattr(config, "use_rmsnorm", True),
            use_swiglu=getattr(config, "use_swiglu", True),
            pad_token_id=config.pad_token_id,
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            decoder_only=False,
            use_cross_attn=True,
        )
        self.model = APTModel(apt_config)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.token_embedding

    def set_input_embeddings(self, value):
        self.model.token_embedding = value

    def get_output_embeddings(self):
        return self.model.output_projection

    def set_output_embeddings(self, value):
        self.model.output_projection = value

    def get_encoder(self):
        return self.model

    def tie_weights(self, **kwargs):
        """绑定 token_embedding 和 output_projection 的权重"""
        if getattr(self.config, "tie_word_embeddings", False):
            input_embeddings = self.get_input_embeddings()
            output_embeddings = self.get_output_embeddings()
            if output_embeddings is not None and input_embeddings is not None:
                output_embeddings.weight = input_embeddings.weight

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        # encoder-decoder 路径
        out = self.model.forward_seq2seq(
            src_tokens=input_ids,
            tgt_tokens=decoder_input_ids,
            src_key_padding_mask=attention_mask,
            tgt_key_padding_mask=decoder_attention_mask,
            return_dict=True,
        )
        logits = out["logits"] if isinstance(out, dict) else out

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return Seq2SeqLMOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(
        self, decoder_input_ids, encoder_outputs=None, **kwargs
    ):
        return {
            "decoder_input_ids": decoder_input_ids,
            "encoder_outputs": encoder_outputs,
        }
