.PHONY: setup smoke train-small

setup:
	pip install -r requirements.txt || true

smoke:
	if command -v pytest >/dev/null 2>&1; then \
		pytest -q tests/test_smoke.py; \
	else \
		python -c 'print("pytest not available; smoke skipped")'; \
	fi

train-small:
	python -m apt_model train --epochs 1 --batch_size 2 --learning_rate 2e-4 --save_path ./outputs/demo
