"""Claude4ForCausalLM — HuggingFace 兼容的 Claude4 封装"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from apt.model.hf_compat.configs import Claude4Config


class Claude4ForCausalLM(PreTrainedModel):
    config_class = Claude4Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = False

    def __init__(self, config: Claude4Config):
        super().__init__(config)
        from apt.model.architectures.claude4_model import Claude4Model

        self.model = Claude4Model(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            num_layers=config.num_layers,
            max_seq_len=config.max_seq_len,
            enable_reflection=config.enable_reflection,
            reflection_layers=config.reflection_layers,
            dropout=config.dropout,
        )
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.tok_emb

    def set_input_embeddings(self, value):
        self.model.tok_emb = value

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
        # Claude4Model.forward(input_ids) -> logits [B, S, V]
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
