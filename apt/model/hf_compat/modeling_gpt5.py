"""GPT5ForCausalLM — HuggingFace 兼容的 GPT5 (MoE) 封装"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from apt.model.hf_compat.configs import GPT5Config


class GPT5ForCausalLM(PreTrainedModel):
    config_class = GPT5Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = False

    def __init__(self, config: GPT5Config):
        super().__init__(config)
        from apt.model.architectures.gpt5_model import GPT5Model

        self.model = GPT5Model(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            num_skills=config.num_skills,
            d_route=config.d_route,
            top_k=config.top_k,
            rank=config.rank,
            enable_multimodal=config.enable_multimodal,
            image_dim=config.image_dim,
            audio_dim=config.audio_dim,
            use_rope_embed=config.use_rope_embed,
            use_rope_attn=config.use_rope_attn,
            n_heads=config.n_heads,
            window_size=config.window_size,
            num_kv_heads=config.num_kv_heads,
            compile_moe_dispatch=config.compile_moe_dispatch,
        )
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.emb

    def set_input_embeddings(self, value):
        self.model.emb = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, value):
        self.model.lm_head = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        # GPT5Model.forward(input_ids) -> logits [B, T, V]
        logits = self.model(input_ids=input_ids)
        if isinstance(logits, tuple):
            logits = logits[0]

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
