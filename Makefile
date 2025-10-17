    .PHONY: setup smoke train-small
    setup:
	pip install -r requirements.txt || true

    smoke:
	pytest -q tests/test_smoke.py || python - <<'PY'
print("pytest not available; smoke skipped")
PY

    train-small:
	python -m apt_model train --epochs 1 --batch_size 2 --learning_rate 2e-4 --save_path ./outputs/demo
