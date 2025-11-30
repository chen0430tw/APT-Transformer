# APT Model (自生成变换器)

APT Model - 一个功能完整的PyTorch Transformer训练平台

**📖 完整文档**: [APT Model 使用手册](APT_MODEL_HANDBOOK.md)

---

## Quickstart
```bash
pip install -r requirements.txt
# ensure torch is available in the active environment
python -c "import torch; print(torch.__version__)"
python -m apt_model --help
# smoke test (if pytest available)
pytest -q tests/test_smoke.py
```

> **Note**
>
> If the environment image does not bundle PyTorch, install it manually before
> running the training demos:
>
> ```bash
> pip install torch
> ```

> **Optional NLP extras**
>
> The text generation metrics and GPT-2 tokenizer rely on scikit-learn and
> Hugging Face assets.  Run the helper script to install the dependencies and
> cache the tokenizer locally (safe to skip in air-gapped CI runs):
>
> ```bash
> python scripts/download_optional_assets.py
> ```

## Layout
```
apt_model/                      # core package
  main.py
  config/
  modeling/
  training/
  data/
  generation/
  evaluation/
  interactive/
  utils/
experiments/
  hpo/                          # Optuna etc.
  configs/
scripts/
tests/
docs/
assets/
requirements.txt
Makefile
```
