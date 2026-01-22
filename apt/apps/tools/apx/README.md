# APX Model Packaging Tools

APX (APT Package Exchange) is a standardized format for packaging and distributing machine learning models with APT-Transformer.

## Overview

APX provides:
- **Unified packaging format** for models from different frameworks (HuggingFace, LLaMA, DeepSeek, etc.)
- **Automatic capability detection** (MoE, RAG, RLHF, Safety, Quantization, TVA/VFT)
- **Flexible adapters** for loading models in different environments
- **Full and thin packaging modes** for different distribution scenarios
- **Manifest-based metadata** for model capabilities and requirements

## APX Package Structure

An APX package (.apx file) is a ZIP archive containing:

```
model.apx (ZIP)
├── apx.yaml                    # Manifest with metadata
├── model/
│   └── adapters/
│       ├── hf_adapter.py       # Model adapter code
│       └── tokenizer_adapter.py # Tokenizer adapter code
├── artifacts/
│   ├── config.json             # Model configuration
│   ├── tokenizer.json          # Tokenizer files
│   └── model.safetensors       # Model weights (full mode)
└── tests/
    └── smoke.py                # Optional smoke test
```

## CLI Commands

### 1. `pack-apx` - Package a Model

Package a model directory into APX format:

```bash
# Basic usage
python -m apt_model pack-apx \
    --src path/to/model \
    --out output.apx \
    --name my-model \
    --version 1.0.0

# Full mode with HuggingFace adapter (default)
python -m apt_model pack-apx \
    --src ./models/llama-7b \
    --out ./packages/llama-7b.apx \
    --name llama-7b \
    --version 1.0.0 \
    --adapter hf \
    --mode full

# Thin mode (placeholders only, for development)
python -m apt_model pack-apx \
    --src ./models/gpt2 \
    --out ./packages/gpt2-thin.apx \
    --name gpt2 \
    --version 0.1.0 \
    --mode thin

# Explicit capabilities
python -m apt_model pack-apx \
    --src ./models/mixtral-moe \
    --out ./packages/mixtral.apx \
    --name mixtral-moe \
    --version 1.0.0 \
    --capability moe \
    --capability tva

# With compose settings
python -m apt_model pack-apx \
    --src ./models/my-model \
    --out ./packages/my-model.apx \
    --name my-model \
    --version 1.0.0 \
    --compose router=observe_only \
    --compose budget=0.15
```

**Arguments:**
- `--src`: Source model directory (required)
- `--out`: Output APX file path (required)
- `--name`: Model name (required)
- `--version`: Model version (default: 1.0.0)
- `--adapter`: Adapter type - `hf` or `stub` (default: hf)
- `--mode`: Packaging mode - `full` or `thin` (default: full)
- `--weights-glob`: Custom weight file pattern (optional)
- `--tokenizer-glob`: Custom tokenizer file pattern (optional)
- `--config-file`: Explicit config.json path (optional)
- `--prefers`: Preference - `builtin` or `plugin` (default: builtin)
- `--capability`: Explicit capability (repeatable)
- `--compose`: Compose key=value (repeatable)
- `--add-test`: Add smoke test
- `--no-auto-detect`: Disable automatic capability detection

### 2. `detect-capabilities` - Detect Model Capabilities

Auto-detect model capabilities:

```bash
python -m apt_model detect-capabilities --src path/to/model
```

Detects:
- **moe**: Mixture of Experts
- **rag**: Retrieval-Augmented Generation
- **rl**: Reinforcement Learning (RLHF, PPO, DPO)
- **safety**: Safety/moderation features
- **quantization**: Quantization or distillation
- **tva**: Tri-Vein Attention / Vein-Flow Transformer

### 3. `detect-framework` - Detect Model Framework

Detect model framework:

```bash
python -m apt_model detect-framework --src path/to/model
```

Detects:
- **huggingface**: HuggingFace models
- **structured**: LLaMA, DeepSeek, etc.
- **unknown**: Unrecognized format

### 4. `apx-info` - Display APX Package Info

Display information about an APX package:

```bash
python -m apt_model apx-info --apx model.apx
```

## Python API

### Basic Usage

```python
from apt_model.tools.apx import pack_apx, detect_capabilities
from pathlib import Path

# Package a model
pack_apx(
    src_repo=Path("./models/my-model"),
    out_apx=Path("./packages/my-model.apx"),
    name="my-model",
    version="1.0.0",
    adapter="hf",
    mode="full",
    auto_detect_capabilities=True
)

# Detect capabilities
caps = detect_capabilities(Path("./models/my-model"))
print(f"Detected capabilities: {caps}")
```

### Advanced Usage

```python
from apt_model.tools.apx import (
    pack_apx,
    detect_moe,
    detect_rag,
    detect_tva_vft,
    AdapterType,
    get_adapter_code
)
from pathlib import Path

# Check specific capabilities
model_dir = Path("./models/mixtral")
has_moe = detect_moe(model_dir)
has_tva = detect_tva_vft(model_dir)

# Get adapter code
adapter_code = get_adapter_code(AdapterType.HF)
print(adapter_code["model"])

# Package with explicit settings
pack_apx(
    src_repo=model_dir,
    out_apx=Path("./mixtral-custom.apx"),
    name="mixtral-custom",
    version="2.0.0",
    adapter="hf",
    mode="full",
    capabilities=["moe", "tva"],
    compose_items=["router=observe_only", "budget=0.20"],
    add_test=True,
    auto_detect_capabilities=False
)
```

## Capability Detection

### MoE (Mixture of Experts)

Detected by:
- Config contains `num_experts` or `num_local_experts`
- Architecture name contains "expert" or "moe"
- Files like `moe_*.py`, `expert_*.py`, `router_*.py` exist

### RAG (Retrieval-Augmented Generation)

Detected by:
- Config contains `retriever`, `knowledge_base`, `retrieval`
- Files like `rag_*.py`, `retriever_*.py`, `knowledge_*.py` exist
- Has vector store or embedding files

### RL (Reinforcement Learning)

Detected by:
- Config contains `ppo`, `dpo`, `rlhf`, `reward_model`
- Files like `rl_*.py`, `ppo_*.py`, `reward_*.py` exist
- Has reward model files

### Safety/Moderation

Detected by:
- Config contains `safety`, `moderation`, `toxicity`, `content_filter`
- Files like `safety_*.py`, `moderator_*.py` exist

### Quantization/Distillation

Detected by:
- Config contains `quantization`, `bits`, `gptq`, `awq`, `distillation`
- Weight files contain markers: `4bit`, `8bit`, `gptq`, `awq`
- Files like `distill_*.py`, `teacher_*.py` exist

### TVA/VFT (Tri-Vein Attention / Vein-Flow Transformer)

Detected by:
- Config contains `vein`, `tva`, `vft`, `subspace`, `vein_rank`
- Files like `vein_*.py`, `tva_*.py`, `vft_*.py` exist

## Packaging Modes

### Full Mode

- **Copies all files** to the APX package
- **Self-contained** - package includes everything needed
- **Larger file size** - suitable for distribution
- **Use case**: Production deployment, model sharing

```bash
python -m apt_model pack-apx \
    --src ./models/my-model \
    --out ./packages/my-model-full.apx \
    --name my-model \
    --version 1.0.0 \
    --mode full
```

### Thin Mode

- **Creates placeholders** with source file references
- **Minimal file size** - only metadata and code
- **Requires source files** at runtime
- **Use case**: Development, testing, version control

```bash
python -m apt_model pack-apx \
    --src ./models/my-model \
    --out ./packages/my-model-thin.apx \
    --name my-model \
    --version 1.0.0 \
    --mode thin
```

## Adapters

### HuggingFace Adapter (`hf`)

- Loads models using `transformers.AutoModelForCausalLM`
- Supports all HuggingFace models
- Requires `transformers` and `torch` packages
- **Default adapter** - recommended for most cases

### Stub Adapter (`stub`)

- Minimal placeholder adapter for testing
- Does not load actual models
- No external dependencies
- **Use case**: Testing APX packaging without model weights

## Manifest Format (apx.yaml)

```yaml
apx_version: 1
name: my-model
version: 1.0.0
type: model

entrypoints:
  model_adapter: model/adapters/hf_adapter.py:HFAdapter
  tokenizer_adapter: model/adapters/tokenizer_adapter.py:HFTokenizerAdapter

artifacts:
  config: artifacts/config.json
  tokenizer: artifacts/tokenizer.json
  weights: artifacts/model.safetensors

capabilities:
  provides:
    - moe
    - tva
  prefers:
    - builtin

compose:
  router: observe_only
  budget: 0.15
```

## Best Practices

### 1. Always Auto-Detect Capabilities

Auto-detection is enabled by default and recommended:

```bash
# Good: Auto-detection enabled (default)
python -m apt_model pack-apx --src ./model --out model.apx --name model --version 1.0.0

# Only disable if you know what you're doing
python -m apt_model pack-apx --src ./model --out model.apx --name model --version 1.0.0 --no-auto-detect
```

### 2. Use Full Mode for Distribution

```bash
# Production distribution
python -m apt_model pack-apx --src ./model --out model-v1.0.apx --name model --version 1.0.0 --mode full

# Development/testing
python -m apt_model pack-apx --src ./model --out model-dev.apx --name model --version 0.1.0 --mode thin
```

### 3. Version Your Packages

```bash
# Semantic versioning
python -m apt_model pack-apx --src ./model --out model-v1.0.0.apx --name model --version 1.0.0
python -m apt_model pack-apx --src ./model --out model-v1.1.0.apx --name model --version 1.1.0
python -m apt_model pack-apx --src ./model --out model-v2.0.0.apx --name model --version 2.0.0
```

### 4. Check Capabilities Before Packaging

```bash
# Detect capabilities first
python -m apt_model detect-capabilities --src ./model

# Then package with appropriate settings
python -m apt_model pack-apx \
    --src ./model \
    --out model.apx \
    --name model \
    --version 1.0.0 \
    --capability moe \
    --capability tva
```

## Troubleshooting

### Error: "config.json not found"

```bash
# Specify config file explicitly
python -m apt_model pack-apx \
    --src ./model \
    --out model.apx \
    --name model \
    --version 1.0.0 \
    --config-file ./model/config.json
```

### Error: "No weight files matched"

```bash
# Specify weight glob pattern
python -m apt_model pack-apx \
    --src ./model \
    --out model.apx \
    --name model \
    --version 1.0.0 \
    --weights-glob "*.bin"
```

### Warning: "No special capabilities detected"

This is normal for simple models without MoE, RAG, or other advanced features. You can add explicit capabilities:

```bash
python -m apt_model pack-apx \
    --src ./model \
    --out model.apx \
    --name model \
    --version 1.0.0 \
    --capability custom-feature
```

## Integration with APT-CLI

APX tools are fully integrated into APT-CLI. All commands are available through the main CLI:

```bash
# Check available APX commands
python -m apt_model help

# Use APX commands
python -m apt_model pack-apx --src ./model --out model.apx --name model --version 1.0.0
python -m apt_model detect-capabilities --src ./model
python -m apt_model apx-info --apx model.apx
```

## Architecture

### Module Structure

```
apt_model/tools/apx/
├── __init__.py           # Public API
├── converter.py          # Core packaging logic
├── detectors.py          # Capability detection
├── adapters.py           # Adapter templates
├── templates.py          # YAML generation
└── README.md             # This file
```

### Components

1. **Converter** (`converter.py`):
   - Artifact collection
   - File packaging
   - ZIP creation
   - Error handling

2. **Detectors** (`detectors.py`):
   - MoE detection
   - RAG detection
   - RL detection
   - Safety detection
   - Quantization detection
   - TVA/VFT detection

3. **Adapters** (`adapters.py`):
   - HuggingFace adapter templates
   - Stub adapter templates
   - Custom adapter support

4. **Templates** (`templates.py`):
   - apx.yaml generation
   - File writing utilities

## Future Enhancements

- [ ] Support for more frameworks (ONNX, TensorFlow, JAX)
- [ ] APX package validation and verification
- [ ] APX package signing and authentication
- [ ] APX registry and package management
- [ ] Automatic version conflict resolution
- [ ] Model diff and patch support
- [ ] Incremental updates for large models

## References

- [APT-Transformer Documentation](../../README.md)
- [Plugin System](../../console/README.md)
- [Capability Detection Specification](../../../memo.txt)
