# APT Model (自生成变换器) — Repo Overview

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
