# Restructure Changelog
- Created `experiments/hpo/` and moved optuna scripts there (if present).
- Created `experiments/configs/best/` for best config snapshots.
- Added `apt_model/__main__.py` for module execution.
- Added `apt_model/training/hooks.py` for training-time hooks (DBC-DAC).
- Added `tests/test_smoke.py` minimal CPU smoke test.
- Added top-level `requirements.txt`, `Makefile`, and expanded `README.md`.
