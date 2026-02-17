"""GPTo3ForCausalLM — HuggingFace 兼容的 GPTo3 封装"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from apt.model.hf_compat.configs import GPTo3Config


class GPTo3ForCausalLM(PreTrainedModel):
    config_class = GPTo3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = False

    def __init__(self, config: GPTo3Config):
        super().__init__(config)
        from apt.model.architectures.gpto3_model import GPTo3Model

        self.model = GPTo3Model(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            num_layers=config.num_layers,
            rank=config.rank,
            max_seq_len=config.max_seq_len,
            window_size=config.window_size,
            num_kv_heads=config.num_kv_heads,
            use_rope=config.use_rope,
            entropy_trig=config.entropy_trig,
            global_budget=config.global_budget,
        )
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.encoder.text_emb

    def set_input_embeddings(self, value):
        self.model.encoder.text_emb = value

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
        # GPTo3Model.forward(text_ids) -> logits [B, T, V]
        logits = self.model(text_ids=input_ids)
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
