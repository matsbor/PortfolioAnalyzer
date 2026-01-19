.PHONY: setup test run

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

test:
	PYTHONPATH=. pytest -q

run:
	streamlit run alpha_miner_institutional_v2.py
