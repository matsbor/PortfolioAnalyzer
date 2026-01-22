.PHONY: setup test run backtest_cache backtest_retry_missing backtest_offline

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

test:
	PYTHONPATH=. pytest -q

run:
	streamlit run alpha_miner_institutional_v2.py

# Backtest cache management (recommended workflow)
cache_build:
	@echo "Building backtest cache..."
	@python3 backtest_runner.py \
		--start 2025-07-01 \
		--end 2026-01-01 \
		--portfolio_csv portfolio_enhanced.csv \
		--initial_cash 39570 \
		--data_dir ./.backtest_cache/ \
		--build_cache_only

cache_verify:
	@echo "Verifying backtest cache..."
	@python3 backtest_runner.py \
		--start 2025-07-01 \
		--end 2026-01-01 \
		--portfolio_csv portfolio_enhanced.csv \
		--data_dir ./.backtest_cache/ \
		--verify_cache

backtest_offline:
	@echo "Running offline backtest..."
	@python3 backtest_runner.py \
		--start 2025-07-01 \
		--end 2026-01-01 \
		--portfolio_csv portfolio_enhanced.csv \
		--initial_cash 39570 \
		--offline \
		--data_dir ./.backtest_cache/

# Legacy aliases (for backward compatibility)
backtest_cache: cache_build

backtest_retry_missing:
	@echo "Retrying missing symbols from manifest..."
	@python3 backtest_runner.py \
		--start 2025-07-01 \
		--end 2026-01-01 \
		--retry_missing_cache \
		--data_dir ./.backtest_cache/
