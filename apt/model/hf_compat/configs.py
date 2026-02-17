"""
HuggingFace PretrainedConfig 子类

每个 config 都提供:
- model_type: vLLM / AutoModel 自动检测用
- auto_map: 指向对应的 modeling 文件，支持 from_pretrained() 自动加载
- HF 标准字段别名 (hidden_size, num_attention_heads, etc.)
"""

from transformers import PretrainedConfig


# ============================================================================
# GPT4o
# ============================================================================

class GPT4oConfig(PretrainedConfig):
    model_type = "gpt4o"
    # auto_map: Hub 上 from_pretrained() 自动解析 (repo 根目录下需要对应 .py)
    auto_map = {
        "AutoConfig": "configuration_gpt4o.GPT4oConfig",
        "AutoModelForCausalLM": "modeling_gpt4o.GPT4oForCausalLM",
    }

    def __init__(
        self,
        vocab_size: int = 200000,
        d_model: int = 2048,
        n_heads: int = 16,
        d_ff: int = 8192,
        num_layers: int = 24,
        rank: int = 4,
        max_seq_len: int = 4096,
        window_size: int = 0,
        num_kv_heads: int = None,
        use_rope: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.rank = rank
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        self.num_kv_heads = num_kv_heads
        self.use_rope = use_rope
        # HF 标准字段 (vLLM 等框架靠这些自动推断架构)
        self.hidden_size = d_model
        self.num_attention_heads = n_heads
        self.num_hidden_layers = num_layers
        self.intermediate_size = d_ff
        self.max_position_embeddings = max_seq_len
        self.num_key_value_heads = num_kv_heads if num_kv_heads else n_heads


# ============================================================================
# GPTo3 (继承 GPT4o + 推理控制器参数)
# ============================================================================

class GPTo3Config(PretrainedConfig):
    model_type = "gpto3"
    auto_map = {
        "AutoConfig": "configuration_gpto3.GPTo3Config",
        "AutoModelForCausalLM": "modeling_gpto3.GPTo3ForCausalLM",
    }

    def __init__(
        self,
        vocab_size: int = 200000,
        d_model: int = 2048,
        n_heads: int = 16,
        d_ff: int = 8192,
        num_layers: int = 24,
        rank: int = 4,
        max_seq_len: int = 4096,
        window_size: int = 0,
        num_kv_heads: int = None,
        use_rope: bool = False,
        entropy_trig: float = 2.0,
        global_budget: float = 0.15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.rank = rank
        self.max_seq_len = max_seq_len
        self.window_size = window_size
        self.num_kv_heads = num_kv_heads
        self.use_rope = use_rope
        self.entropy_trig = entropy_trig
        self.global_budget = global_budget
        # HF 标准字段
        self.hidden_size = d_model
        self.num_attention_heads = n_heads
        self.num_hidden_layers = num_layers
        self.intermediate_size = d_ff
        self.max_position_embeddings = max_seq_len
        self.num_key_value_heads = num_kv_heads if num_kv_heads else n_heads


# ============================================================================
# GPT5 (MoE)
# ============================================================================

class GPT5Config(PretrainedConfig):
    model_type = "gpt5"
    auto_map = {
        "AutoConfig": "configuration_gpt5.GPT5Config",
        "AutoModelForCausalLM": "modeling_gpt5.GPT5ForCausalLM",
    }

    def __init__(
        self,
        vocab_size: int = 200000,
        d_model: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        num_skills: int = 64,
        d_route: int = 64,
        top_k: int = 2,
        rank: int = 32,
        enable_multimodal: bool = False,
        image_dim: int = 1024,
        audio_dim: int = 512,
        use_rope_embed: bool = False,
        use_rope_attn: bool = False,
        window_size: int = 0,
        num_kv_heads: int = None,
        compile_moe_dispatch: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.num_skills = num_skills
        self.d_route = d_route
        self.top_k = top_k
        self.rank = rank
        self.enable_multimodal = enable_multimodal
        self.image_dim = image_dim
        self.audio_dim = audio_dim
        self.use_rope_embed = use_rope_embed
        self.use_rope_attn = use_rope_attn
        self.window_size = window_size
        self.num_kv_heads = num_kv_heads
        self.compile_moe_dispatch = compile_moe_dispatch
        # HF 标准字段
        self.hidden_size = d_model
        self.num_attention_heads = n_heads
        self.num_hidden_layers = n_layers
        self.num_key_value_heads = num_kv_heads if num_kv_heads else n_heads


# ============================================================================
# Claude4
# ============================================================================

class Claude4Config(PretrainedConfig):
    model_type = "claude4"
    auto_map = {
        "AutoConfig": "configuration_claude4.Claude4Config",
        "AutoModelForCausalLM": "modeling_claude4.Claude4ForCausalLM",
    }

    def __init__(
        self,
        vocab_size: int = 50000,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        num_layers: int = 6,
        max_seq_len: int = 4096,
        enable_reflection: bool = True,
        reflection_layers: list = None,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.enable_reflection = enable_reflection
        self.reflection_layers = reflection_layers
        self.dropout = dropout
        # HF 标准字段
        self.hidden_size = d_model
        self.num_attention_heads = n_heads
        self.num_hidden_layers = num_layers
        self.intermediate_size = d_ff
        self.max_position_embeddings = max_seq_len


# ============================================================================
# APT (Encoder-Decoder)
# ============================================================================

class APTConfig(PretrainedConfig):
    model_type = "apt"
    # APT 是 encoder-decoder，标记给 HF
    is_encoder_decoder = True
    auto_map = {
        "AutoConfig": "configuration_apt.APTConfig",
        "AutoModelForCausalLM": "modeling_apt.APTForCausalLM",
        "AutoModelForSeq2SeqLM": "modeling_apt.APTForSeq2SeqLM",
    }

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 768,
        max_seq_len: int = 2048,
        num_encoder_layers: int = 12,
        num_decoder_layers: int = 12,
        num_heads: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_dbc_dac: bool = True,
        use_rmsnorm: bool = True,
        use_swiglu: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.activation = activation
        self.use_dbc_dac = use_dbc_dac
        self.use_rmsnorm = use_rmsnorm
        self.use_swiglu = use_swiglu
        # HF 标准字段
        self.hidden_size = d_model
        self.num_attention_heads = num_heads
        self.num_hidden_layers = num_encoder_layers
        self.intermediate_size = d_ff
        self.max_position_embeddings = max_seq_len
        self.decoder_layers = num_decoder_layers
