# APT Model (自生成变换器) — Repo Overview

## Quickstart
```bash
pip install -r requirements.txt
python -m apt_model --help
# smoke test (if pytest available)
pytest -q tests/test_smoke.py
```

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
