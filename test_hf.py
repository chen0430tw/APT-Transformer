#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""HF Compatibility Layer Test Suite (Unified)

Usage:
    python test_hf.py --mode basic              # 基础测试 (5个测试)
    python test_hf.py --mode simple             # 简化测试 (避免HF导入)
    python test_hf.py --mode structure          # 结构分析 (静态检查)
    python test_hf.py --mode complete          # 完整测试 (10个测试+BUG追踪)
    python test_hf.py --mode convert            # 仅测试convert_checkpoint
    python test_hf.py --mode after-fix         # 修复后验证

    Optional arguments:
    --checkpoint PATH    # Override checkpoint path
    --import-method      # direct or importlib
    --bug-tracker        # Enable bug tracking (complete mode only)
"""

import argparse
import sys
import torch
import importlib.util
from pathlib import Path

# Add project path
sys.path.insert(0, str(Path(__file__).parent))

# Track all bugs found (only for complete mode)
BUGS = []


def log_bug(description, severity="HIGH", location=None):
    """Log a bug found during testing"""
    bug = {
        "description": description,
        "severity": severity,
        "location": location,
    }
    BUGS.append(bug)
    print(f"\n[BUG FOUND - {severity}]")
    print(f"  Description: {description}")
    if location:
        print(f"  Location: {location}")


# ============================================================================
# Test Functions (from test_hf_local.py)
# ============================================================================

def test_tokenizer(checkpoint_path=None, import_method="direct"):
    """测试1: APTTokenizer能否正常加载"""
    print("=" * 70)
    print("TEST 1: APTTokenizer")
    print("=" * 70)

    try:
        if import_method == "direct":
            from apt.model.hf_compat.tokenization_apt import APTTokenizer
        else:
            # Direct import from tokenization module
            sys.path.insert(0, str(Path(__file__).parent / "apt" / "model" / "hf_compat"))
            spec = importlib.util.spec_from_file_location(
                "tokenization_apt",
                Path(__file__).parent / "apt" / "model" / "hf_compat" / "tokenization_apt.py"
            )
            tokenization_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tokenization_module)
            APTTokenizer = tokenization_module.APTTokenizer

        # 使用训练好的tokenizer
        tokenizer_path = checkpoint_path or "D:/APT-Transformer/test_1node_output/tokenizer.json"

        if not Path(tokenizer_path).exists():
            print(f"[X] Tokenizer file not found: {tokenizer_path}")
            return None

        tokenizer = APTTokenizer(tokenizer_file=tokenizer_path)
        print(f"[OK] Tokenizer loaded")
        print(f"  - vocab_size: {len(tokenizer.get_vocab())}")

        # 测试编码
        text = "Hello world"
        ids = tokenizer.encode(text)
        print(f"  - Encode: '{text}' -> {ids}")

        # 测试解码
        decoded = tokenizer.decode(ids)
        print(f"  - Decode: {ids} -> '{decoded}'")

        return tokenizer
    except Exception as e:
        print(f"[X] Tokenizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_forward(import_method="direct"):
    """测试2: APTForCausalLM能否正常forward"""
    print("\n" + "=" * 70)
    print("TEST 2: APTForCausalLM Forward")
    print("=" * 70)

    try:
        if import_method == "direct":
            from apt.model.hf_compat.configs import APTConfig
            from apt.model.hf_compat.modeling_apt import APTForCausalLM
        else:
            # Simple mode: test base APT model directly
            from apt.model.architectures.apt_model import APTModel, APTModelConfiguration

        # 创建小模型配置
        if import_method == "direct":
            config = APTConfig(
                vocab_size=1000,
                d_model=128,
                max_seq_len=256,
                num_encoder_layers=2,
                num_decoder_layers=2,
                num_heads=4,
                d_ff=512,
            )
            model = APTForCausalLM(config)
        else:
            config = APTModelConfiguration(
                vocab_size=1000,
                d_model=128,
                max_seq_len=256,
                num_encoder_layers=2,
                num_decoder_layers=2,
                num_heads=4,
                d_ff=512,
                decoder_only=True,
            )
            model = APTModel(config)

        print(f"[OK] Config and model created")

        # 测试forward
        input_ids = torch.randint(0, 1000, (2, 10))

        if import_method == "direct":
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            print(f"[OK] Forward successful")
            print(f"  - logits shape: {outputs.logits.shape}")
        else:
            with torch.no_grad():
                output = model(src_tokens=input_ids)
            if isinstance(output, tuple):
                logits = output[0]
            elif isinstance(output, dict):
                logits = output["logits"]
            else:
                logits = output
            print(f"[OK] Forward successful")
            print(f"  - output shape: {logits.shape}")

        return model
    except Exception as e:
        print(f"[X] Model forward test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_checkpoint_loading(checkpoint_path=None):
    """测试3: 能否从checkpoint加载权重"""
    print("\n" + "=" * 70)
    print("TEST 3: Checkpoint Loading")
    print("=" * 70)

    try:
        checkpoint_path = checkpoint_path or "D:/APT-Transformer/test_output/checkpoint_step_200.pt"

        if not Path(checkpoint_path).exists():
            print(f"[X] Checkpoint not found: {checkpoint_path}")
            return None

        # 加载checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        print(f"[OK] Checkpoint loaded")
        print(f"  - global_step: {ckpt.get('global_step', 'N/A')}")
        print(f"  - keys: {list(ckpt.keys())}")

        state_dict = ckpt.get("model_state_dict")
        if state_dict:
            print(f"  - state_dict keys: {len(state_dict)}")
            print(f"  - sample keys: {list(state_dict.keys())[:5]}")

            # Check for weight tying
            has_token_emb = "token_embedding.weight" in state_dict
            has_output_proj = "output_projection.weight" in state_dict
            print(f"\nWeight tying check:")
            print(f"  - has token_embedding.weight: {has_token_emb}")
            print(f"  - has output_projection.weight: {has_output_proj}")

            if has_token_emb and has_output_proj:
                token_emb_ptr = state_dict["token_embedding.weight"].data_ptr()
                output_proj_ptr = state_dict["output_projection.weight"].data_ptr()
                is_shared = token_emb_ptr == output_proj_ptr
                print(f"  - weights share memory: {is_shared}")

        return ckpt
    except Exception as e:
        print(f"[X] Checkpoint loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_generate(import_method="direct"):
    """测试4: generate()方法是否工作"""
    print("\n" + "=" * 70)
    print("TEST 4: Generate Method")
    print("=" * 70)

    try:
        if import_method == "direct":
            from apt.model.hf_compat.configs import APTConfig
            from apt.model.hf_compat.modeling_apt import APTForCausalLM

            config = APTConfig(
                vocab_size=1000,
                d_model=128,
                max_seq_len=256,
                num_encoder_layers=2,
                num_decoder_layers=2,
                num_heads=4,
                d_ff=512,
            )
            model = APTForCausalLM(config)
        else:
            print("[SKIP] Generate test not supported in simple mode")
            return None

        model.eval()

        # 测试generate
        input_ids = torch.randint(0, config.vocab_size, (1, 5))

        with torch.no_grad():
            outputs = model.generate(input_ids, max_new_tokens=10)

        print(f"[OK] Generate successful")
        print(f"  - output shape: {outputs.shape}")
        print(f"  - generated IDs: {outputs[0].tolist()}")

        return outputs
    except Exception as e:
        print(f"[X] Generate test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_weight_tying(import_method="direct"):
    """测试5: 检查weight tying问题"""
    print("\n" + "=" * 70)
    print("TEST 5: Weight Tying Check")
    print("=" * 70)

    try:
        if import_method == "direct":
            from apt.model.hf_compat.configs import APTConfig
            from apt.model.hf_compat.modeling_apt import APTForCausalLM

            config = APTConfig(
                vocab_size=1000,
                d_model=128,
                max_seq_len=256,
                num_encoder_layers=2,
                num_decoder_layers=2,
                num_heads=4,
                d_ff=512,
            )
            model = APTForCausalLM(config)
        else:
            from apt.model.architectures.apt_model import APTModel, APTModelConfiguration

            config = APTModelConfiguration(
                vocab_size=1000,
                d_model=128,
                max_seq_len=256,
                num_encoder_layers=2,
                num_decoder_layers=2,
                num_heads=4,
                d_ff=512,
                decoder_only=True,
            )
            model = APTModel(config)

        # 检查token_embedding和output_projection是否共享权重
        token_emb = model.get_input_embeddings()
        output_proj = model.get_output_embeddings()

        print(f"  - token_embedding.weight shape: {token_emb.weight.shape}")
        print(f"  - output_projection.weight shape: {output_proj.weight.shape}")

        # 检查是否共享内存
        is_shared = token_emb.weight.data_ptr() == output_proj.weight.data_ptr()
        print(f"  - Weights shared: {is_shared}")

        if is_shared:
            print(f"[INFO] Model uses weight tying")
            print(f"  [WARNING] This causes HF save_pretrained() to fail without config.tie_word_embeddings")

        return is_shared
    except Exception as e:
        print(f"[X] Weight tying check failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# Test Functions (from test_hf_wrapper.py - Structure Analysis)
# ============================================================================

def test_model_structure():
    """Test HF model wrapper structure (static file analysis)"""
    print("=" * 70)
    print("TEST: HF Model Wrapper Structure")
    print("=" * 70)

    try:
        # Read model file directly
        model_file = Path(__file__).parent / "apt" / "model" / "hf_compat" / "modeling_apt.py"
        with open(model_file, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"[OK] modeling_apt.py found")

        # Check for key methods
        methods = [
            "get_input_embeddings",
            "get_output_embeddings",
            "set_input_embeddings",
            "set_output_embeddings",
            "tie_weights",
        ]

        print(f"\nRequired methods:")
        for method in methods:
            found = method in content
            print(f"  - {method}: {found}")

        # Check for tie_weights implementation
        has_tie_weights = "def tie_weights" in content
        print(f"\n  - Custom tie_weights() method: {has_tie_weights}")

        if not has_tie_weights:
            print(f"\n[INFO] Using HF's default tie_weights() behavior")

        return True
    except Exception as e:
        print(f"[X] Modeling structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_structure():
    """Test HF config structure (static file analysis)"""
    print("\n" + "=" * 70)
    print("TEST: HF Config Structure")
    print("=" * 70)

    try:
        # Read config file directly
        config_file = Path(__file__).parent / "apt" / "model" / "hf_compat" / "configs.py"
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check APTConfig class
        print(f"[OK] configs.py found")

        # Look for APTConfig class definition
        if "class APTConfig" in content:
            print(f"[OK] APTConfig class found")

            # Check for tie_word_embeddings
            has_tie_config = "tie_word_embeddings" in content
            print(f"  - Has tie_word_embeddings config: {has_tie_config}")

            # Check for auto_map
            has_auto_map = "auto_map" in content
            print(f"  - Has auto_map: {has_auto_map}")

        return True
    except Exception as e:
        print(f"[X] Config structure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_base_apt_model_structure():
    """Test base APT model weight tying (runtime check)"""
    print("\n" + "=" * 70)
    print("TEST: Base APT Model - Weight Tying")
    print("=" * 70)

    try:
        from apt.model.architectures.apt_model import APTModel, APTModelConfiguration

        # Create APT model config
        apt_config = APTModelConfiguration(
            vocab_size=1000,
            d_model=128,
            max_seq_len=256,
            num_encoder_layers=2,
            num_decoder_layers=2,
            num_heads=4,
            d_ff=512,
            decoder_only=True,
        )

        # Create base APT model
        apt_model = APTModel(apt_config)
        print(f"[OK] Base APT model created")

        # Check weight tying in base model
        token_emb_ptr = apt_model.token_embedding.weight.data_ptr()
        output_proj_ptr = apt_model.output_projection.weight.data_ptr()
        is_shared = token_emb_ptr == output_proj_ptr
        print(f"\nWeight Tying Check (Base APT Model):")
        print(f"  - token_embedding.weight data_ptr: {token_emb_ptr}")
        print(f"  - output_projection.weight data_ptr: {output_proj_ptr}")
        print(f"  - Shared: {is_shared}")

        if is_shared:
            print(f"  [INFO] APT model uses weight tying by design")
            print(f"  [WARNING] This causes HF save_pretrained() to fail without proper config")

        return apt_model, is_shared
    except Exception as e:
        print(f"[X] Base model test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ============================================================================
# Test Functions (from test_hf_complete.py - Complete Test Suite)
# ============================================================================

def test_1_base_model_weight_tying():
    """Test 1: Check if base APT model uses weight tying"""
    print("\n" + "="*70)
    print("TEST 1: Base APT Model - Weight Tying")
    print("="*70)

    try:
        from apt.model.architectures.apt_model import APTModel, APTModelConfiguration

        config = APTModelConfiguration(
            vocab_size=1000,
            d_model=128,
            max_seq_len=256,
            num_encoder_layers=2,
            num_decoder_layers=2,
            num_heads=4,
            d_ff=512,
            decoder_only=True,
        )

        model = APTModel(config)

        token_emb_ptr = model.token_embedding.weight.data_ptr()
        output_proj_ptr = model.output_projection.weight.data_ptr()
        is_shared = token_emb_ptr == output_proj_ptr

        print(f"token_embedding.weight ptr: {token_emb_ptr}")
        print(f"output_projection.weight ptr: {output_proj_ptr}")
        print(f"Weight tying enabled: {is_shared}")

        if is_shared:
            log_bug(
                "APT model uses weight tying (token_embedding == output_projection)",
                severity="INFO",
                location="apt/model/architectures/apt_model.py:1570"
            )

        return True, is_shared
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_2_hf_config_tie_word_embeddings():
    """Test 2: Check if APTConfig has tie_word_embeddings"""
    print("\n" + "="*70)
    print("TEST 2: APTConfig - tie_word_embeddings Attribute")
    print("="*70)

    try:
        config_file = Path(__file__).parent / "apt" / "model" / "hf_compat" / "configs.py"
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check APTConfig __init__
        has_tie_param = "tie_word_embeddings" in content.split("class APTConfig")[1].split("class ")[0]

        print(f"APTConfig has tie_word_embeddings parameter: {has_tie_param}")

        if not has_tie_param:
            log_bug(
                "APTConfig missing 'tie_word_embeddings' parameter - HF doesn't know about weight tying",
                severity="HIGH",
                location="apt/model/hf_compat/configs.py - APTConfig.__init__"
            )

        return True, has_tie_param
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_3_hf_config_defaults():
    """Test 3: Check HF config default values"""
    print("\n" + "="*70)
    print("TEST 3: APTConfig - Default Values")
    print("="*70)

    issues = []

    try:
        config_file = Path(__file__).parent / "apt" / "model" / "hf_compat" / "configs.py"
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract APTConfig class
        aptconfig_section = content.split("class APTConfig")[1].split("class ")[0]

        # Check for important HF attributes
        required_attrs = {
            "model_type": "apt",
            "is_encoder_decoder": "True",
            "auto_map": None,
        }

        print(f"Required HF attributes check:")
        for attr in required_attrs:
            has_it = attr in aptconfig_section
            print(f"  - {attr}: {has_it}")
            if not has_it:
                issues.append(f"Missing {attr}")

        return True, issues
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_4_modeling_methods():
    """Test 4: Check if modeling_apt has required methods"""
    print("\n" + "="*70)
    print("TEST 4: modeling_apt.py - Required Methods")
    print("="*70)

    try:
        model_file = Path(__file__).parent / "apt" / "model" / "hf_compat" / "modeling_apt.py"
        with open(model_file, 'r', encoding='utf-8') as f:
            content = f.read()

        required_methods = [
            "get_input_embeddings",
            "get_output_embeddings",
            "set_input_embeddings",
            "set_output_embeddings",
            "forward",
            "prepare_inputs_for_generation",
            "tie_weights",
        ]

        print(f"Required methods:")
        missing = []
        for method in required_methods:
            has_it = f"def {method}" in content
            print(f"  - {method}: {has_it}")
            if not has_it:
                missing.append(method)

        if "tie_weights" not in content:
            log_bug(
                "modeling_apt.py missing tie_weights() method - relies on HF default which needs config.tie_word_embeddings",
                severity="MEDIUM",
                location="apt/model/hf_compat/modeling_apt.py"
            )

        return True, missing
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_5_tokenizer():
    """Test 5: Check APTTokenizer structure"""
    print("\n" + "="*70)
    print("TEST 5: APTTokenizer - Structure")
    print("="*70)

    try:
        tokenizer_file = Path(__file__).parent / "apt" / "model" / "hf_compat" / "tokenization_apt.py"
        with open(tokenizer_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check class
        has_class = "class APTTokenizer" in content
        print(f"APTTokenizer class: {has_class}")

        # Check inheritance
        has_pretrained = "PreTrainedTokenizer" in content or "PreTrainedTokenizerFast" in content
        print(f"Inherits from HF PreTrainedTokenizer: {has_pretrained}")

        # Check methods
        methods = ["_tokenize", "_convert_token_to_id", "save_pretrained"]
        print(f"\nRequired methods:")
        for method in methods:
            has_it = method in content
            print(f"  - {method}: {has_it}")

        if not has_class:
            log_bug(
                "APTTokenizer class not found",
                severity="HIGH",
                location="apt/model/hf_compat/tokenization_apt.py"
            )

        return True, has_class and has_pretrained
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_6_convert_checkpoint_keys():
    """Test 6: Check convert_checkpoint.py for key remapping bug"""
    print("\n" + "="*70)
    print("TEST 6: convert_checkpoint.py - Key Remapping Logic")
    print("="*70)

    try:
        convert_file = Path(__file__).parent / "apt" / "model" / "hf_compat" / "convert_checkpoint.py"
        with open(convert_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for the buggy condition
        has_bug = "if missing and not unexpected:" in content
        print(f"Found buggy remap condition (if missing and not unexpected): {has_bug}")

        if has_bug:
            # Check line number
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if "if missing and not unexpected:" in line:
                    print(f"  Bug at line {i}")
                    log_bug(
                        "convert_checkpoint.py remap logic uses 'if missing and not unexpected' which fails when both are non-empty",
                        severity="HIGH",
                        location=f"apt/model/hf_compat/convert_checkpoint.py:{i}"
                    )
                    break

        return True, has_bug
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_7_checkpoint_state_dict_prefix(checkpoint_path=None):
    """Test 7: Check checkpoint state_dict key format"""
    print("\n" + "="*70)
    print("TEST 7: Checkpoint - State Dict Key Format")
    print("="*70)

    try:
        checkpoint_path = checkpoint_path or "D:/APT-Transformer/tests/saved_models/hlbd_model_20251222_034732.pt"

        if not Path(checkpoint_path).exists():
            print(f"[SKIP] Checkpoint not found: {checkpoint_path}")
            return None, None

        print(f"Found checkpoint: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", {})

        if not state_dict:
            print(f"[FAIL] No model_state_dict in checkpoint")
            return False, None

        # Check key format
        sample_keys = list(state_dict.keys())[:5]
        print(f"Sample keys: {sample_keys}")

        has_model_prefix = any(k.startswith("model.") for k in sample_keys)
        has_no_prefix = any(not k.startswith("model.") for k in sample_keys)

        print(f"Has 'model.' prefix: {has_model_prefix}")
        print(f"No 'model.' prefix: {has_no_prefix}")

        if has_no_prefix and not has_model_prefix:
            print(f"[INFO] Checkpoint uses keys without 'model.' prefix")
            print(f"[INFO] HF wrapper expects 'model.' prefix - remap needed")
            return True, "no_prefix"
        elif has_model_prefix and not has_no_prefix:
            print(f"[INFO] Checkpoint uses 'model.' prefix")
            return True, "has_prefix"
        else:
            print(f"[WARNING] Mixed key formats!")
            return True, "mixed"

    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_8_autopoietic_plugin():
    """Test 8: Check autopoietic_plugin.py structure"""
    print("\n" + "="*70)
    print("TEST 8: autopoietic_plugin.py - Structure")
    print("="*70)

    try:
        plugin_file = Path(__file__).parent / "apt" / "model" / "hf_compat" / "autopoietic_plugin.py"
        with open(plugin_file, 'r', encoding='utf-8') as f:
            content = f.read()

        has_class = "class AutopoieticPlugin" in content or "def register_autopoietic" in content
        print(f"Autopoietic plugin class/function: {has_class}")

        has_hook = "register_forward_hook" in content
        print(f"Uses forward hooks: {has_hook}")

        return True, has_class
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_9_generation_mixin():
    """Test 9: Check if models inherit GenerationMixin"""
    print("\n" + "="*70)
    print("TEST 9: GenerationMixin Inheritance")
    print("="*70)

    try:
        model_file = Path(__file__).parent / "apt" / "model" / "hf_compat" / "modeling_apt.py"
        with open(model_file, 'r', encoding='utf-8') as f:
            content = f.read()

        has_generation_mixin = "GenerationMixin" in content
        print(f"Inherits from GenerationMixin: {has_generation_mixin}")

        if not has_generation_mixin:
            log_bug(
                "modeling_apt.py doesn't inherit from GenerationMixin - generate() won't work",
                severity="HIGH",
                location="apt/model/hf_compat/modeling_apt.py - class definition"
            )

        return True, has_generation_mixin
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_10_post_init():
    """Test 10: Check if post_init calls tie_weights"""
    print("\n" + "="*70)
    print("TEST 10: post_init() Method")
    print("="*70)

    try:
        model_file = Path(__file__).parent / "apt" / "model" / "hf_compat" / "modeling_apt.py"
        with open(model_file, 'r', encoding='utf-8') as f:
            content = f.read()

        has_post_init = "def post_init" in content or "self.post_init()" in content
        print(f"Has post_init: {has_post_init}")

        return True, has_post_init
    except Exception as e:
        print(f"[FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False, None


# ============================================================================
# Test Functions (from test_hf_after_fix.py and test_hf_convert_only.py)
# ============================================================================

def test_convert_checkpoint(checkpoint_path=None):
    """Test: Convert checkpoint to HF format"""
    print("="*70)
    print("TEST: Convert Checkpoint to HF Format")
    print("="*70)

    try:
        from apt.model.hf_compat.convert_checkpoint import convert

        checkpoint_path = checkpoint_path or "D:/APT-Transformer/tests/saved_models/hlbd_model_20251222_034732.pt"
        output_dir = "D:/APT-Transformer/test_output/hf_model_test"

        print(f"Checkpoint: {checkpoint_path}")
        print(f"Output: {output_dir}")

        # Run conversion
        convert(
            checkpoint_path=checkpoint_path,
            model_type="apt",
            output_dir=output_dir,
            tokenizer=None,  # We don't have tokenizer yet
            safe_serialization=True,
        )

        print("\n[OK] Conversion successful!")

        # Check output files
        output_path = Path(output_dir)
        config_file = output_path / "config.json"
        model_file = output_path / "model.safetensors"

        print(f"\nGenerated files:")
        print(f"  - config.json: {config_file.exists()}")
        print(f"  - model.safetensors: {model_file.exists()}")

        if config_file.exists():
            import json
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                print(f"\nConfig contents:")
                print(f"  - model_type: {config.get('model_type')}")
                print(f"  - tie_word_embeddings: {config.get('tie_word_embeddings', 'NOT SET')}")

        return True

    except Exception as e:
        print(f"\n[X] Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_remap_logic(checkpoint_path=None):
    """Test: Check if remap logic works"""
    print("\n" + "="*70)
    print("TEST: Remap Logic Check")
    print("="*70)

    try:
        from apt.model.hf_compat.convert_checkpoint import remap_state_dict

        checkpoint_path = checkpoint_path or "D:/APT-Transformer/tests/saved_models/hlbd_model_20251222_034732.pt"
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", {})

        print(f"Original keys: {len(state_dict)}")
        print(f"Sample: {list(state_dict.keys())[:3]}")

        # Test remap
        remapped = remap_state_dict(state_dict, "apt")
        print(f"\nRemapped keys: {len(remapped)}")
        print(f"Sample: {list(remapped.keys())[:3]}")

        # Check if keys now have "model." prefix
        has_prefix = any(k.startswith("model.") for k in remapped.keys())
        print(f"\nHas 'model.' prefix: {has_prefix}")

        if has_prefix:
            print(f"[OK] Remap logic works!")
        else:
            print(f"[X] Remap logic failed!")

        return has_prefix

    except Exception as e:
        print(f"[X] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_load_hf_model():
    """Test: Load converted HF model"""
    print("\n" + "="*70)
    print("TEST: Load Converted HF Model")
    print("="*70)

    try:
        from apt.model.hf_compat.configs import APTConfig
        from apt.model.hf_compat.modeling_apt import APTForCausalLM

        model_path = "D:/APT-Transformer/test_output/hf_model_test"

        if not Path(model_path).exists():
            print(f"[SKIP] HF model not found. Run convert test first.")
            return None

        # Load from config
        config = APTConfig.from_pretrained(model_path)
        print(f"[OK] Config loaded")
        print(f"  - model_type: {config.model_type}")
        print(f"  - tie_word_embeddings: {config.tie_word_embeddings}")

        # Load model
        model = APTForCausalLM.from_pretrained(model_path)
        print(f"[OK] Model loaded")

        # Check weight tying
        token_emb = model.get_input_embeddings()
        output_proj = model.get_output_embeddings()

        is_tied = token_emb.weight.data_ptr() == output_proj.weight.data_ptr()
        print(f"\nWeight tying check:")
        print(f"  - Weights tied: {is_tied}")

        if is_tied:
            print(f"[OK] Weights are properly tied!")
        else:
            print(f"[WARNING] Weights are NOT tied!")

        # Test forward pass
        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        with torch.no_grad():
            output = model(input_ids=input_ids)

        print(f"\n[OK] Forward pass successful")
        print(f"  - Output shape: {output.logits.shape}")

        return True

    except Exception as e:
        print(f"\n[X] Loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hf_generate():
    """Test: Generate text with HF model"""
    print("\n" + "="*70)
    print("TEST: Generate Text")
    print("="*70)

    try:
        from apt.model.hf_compat.modeling_apt import APTForCausalLM

        model_path = "D:/APT-Transformer/test_output/hf_model_test"

        if not Path(model_path).exists():
            print(f"[SKIP] HF model not found. Run convert and load tests first.")
            return None

        # Load model
        model = APTForCausalLM.from_pretrained(model_path)
        model.eval()

        print(f"Model loaded")

        # Generate
        input_ids = torch.randint(0, model.config.vocab_size, (1, 5))
        print(f"Input IDs: {input_ids[0].tolist()}")

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
            )

        print(f"Output IDs: {output[0].tolist()}")
        print(f"\n[OK] Generation successful!")
        print(f"  - Input length: {input_ids.shape[1]}")
        print(f"  - Output length: {output.shape[1]}")

        return True

    except Exception as e:
        print(f"\n[X] Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Mode Definitions and Test Mapping
# ============================================================================

MODE_TESTS = {
    "basic": {
        "tokenizer": test_tokenizer,
        "model_forward": test_model_forward,
        "checkpoint": test_checkpoint_loading,
        "generate": test_generate,
        "weight_tying": test_weight_tying,
    },
    "simple": {
        "tokenizer_only": test_tokenizer,
        "apt_model_direct": test_model_forward,
        "checkpoint_structure": test_checkpoint_loading,
        "hf_config": test_config_structure,
    },
    "structure": {
        "model_structure": test_model_structure,
        "config_structure": test_config_structure,
        "base_apt_model": test_base_apt_model_structure,
    },
    "complete": {
        "base_model_weight_tying": test_1_base_model_weight_tying,
        "hf_config_tie_word_embeddings": test_2_hf_config_tie_word_embeddings,
        "hf_config_defaults": test_3_hf_config_defaults,
        "modeling_methods": test_4_modeling_methods,
        "tokenizer_structure": test_5_tokenizer,
        "convert_checkpoint_bug": test_6_convert_checkpoint_keys,
        "checkpoint_key_format": test_7_checkpoint_state_dict_prefix,
        "autopoietic_plugin": test_8_autopoietic_plugin,
        "generation_mixin": test_9_generation_mixin,
        "post_init": test_10_post_init,
    },
    "convert": {
        "checkpoint_structure": test_checkpoint_loading,
        "remap_logic": test_remap_logic,
        "config_tie_word_embeddings": test_config_structure,
        "modeling_tie_weights": test_model_structure,
    },
    "after-fix": {
        "convert": test_convert_checkpoint,
        "load": test_load_hf_model,
        "generate": test_hf_generate,
    },
}


def run_tests(mode, checkpoint_path=None, import_method="direct", bug_tracker=False):
    """Run tests based on mode"""
    tests = MODE_TESTS.get(mode, {})

    if not tests:
        print(f"[ERROR] Unknown mode: {mode}")
        return {}

    results = {}
    for test_name, test_func in tests.items():
        try:
            # Prepare kwargs based on test function signature
            import inspect
            sig = inspect.signature(test_func)
            kwargs = {}

            if "checkpoint_path" in sig.parameters and checkpoint_path:
                kwargs["checkpoint_path"] = checkpoint_path
            if "import_method" in sig.parameters:
                kwargs["import_method"] = import_method

            result = test_func(**kwargs)
            results[test_name] = result
        except Exception as e:
            print(f"\n[ERROR] {test_name} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = None

    return results


def print_summary(results, mode, bug_tracker=False):
    """Print test summary"""
    print("\n" + "="*70)
    print("TEST EXECUTION SUMMARY")
    print("="*70)

    for name, result in results.items():
        if result is None:
            status = "[SKIP]"
        elif isinstance(result, tuple) and len(result) > 0 and result[0]:
            status = "[PASS]"
        elif result:
            status = "[PASS]"
        else:
            status = "[FAIL]"
        print(f"{status} {name}")

    # Print bug summary if enabled
    if bug_tracker and BUGS:
        print("\n" + "="*70)
        print("BUG SUMMARY")
        print("="*70)
        print(f"\nTotal bugs found: {len(BUGS)}\n")

        high_severity = [b for b in BUGS if b["severity"] == "HIGH"]
        medium_severity = [b for b in BUGS if b["severity"] == "MEDIUM"]
        info_severity = [b for b in BUGS if b["severity"] == "INFO"]

        if high_severity:
            print("HIGH SEVERITY:")
            for i, bug in enumerate(high_severity, 1):
                print(f"  {i}. {bug['description']}")
                if bug.get('location'):
                    print(f"     Location: {bug['location']}")
            print()

        if medium_severity:
            print("MEDIUM SEVERITY:")
            for i, bug in enumerate(medium_severity, 1):
                print(f"  {i}. {bug['description']}")
                if bug.get('location'):
                    print(f"     Location: {bug['location']}")
            print()

        if info_severity:
            print("INFO:")
            for i, bug in enumerate(info_severity, 1):
                print(f"  {i}. {bug['description']}")
                if bug.get('location'):
                    print(f"     Location: {bug['location']}")
    elif bug_tracker:
        print("\n[OK] No bugs found!")


def main():
    parser = argparse.ArgumentParser(
        description="HF Compatibility Layer Test Suite (Unified)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--mode",
        choices=list(MODE_TESTS.keys()),
        default="basic",
        help="Test mode to run"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Override checkpoint path"
    )
    parser.add_argument(
        "--import-method",
        choices=["direct", "importlib"],
        default="direct",
        help="Import method: direct (import HF modules) or importlib (avoid import issues)"
    )
    parser.add_argument(
        "--bug-tracker",
        action="store_true",
        help="Enable bug tracking (only affects complete mode)"
    )

    args = parser.parse_args()

    # Print header
    print("\n" + "="*70)
    print(f"HF COMPATIBILITY LAYER TEST SUITE")
    print(f"Mode: {args.mode}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    print(f"Import Method: {args.import_method}")
    print("="*70)

    # Run tests
    results = run_tests(
        mode=args.mode,
        checkpoint_path=args.checkpoint,
        import_method=args.import_method,
        bug_tracker=args.bug_tracker
    )

    # Print summary
    print_summary(results, args.mode, args.bug_tracker)


if __name__ == "__main__":
    main()
