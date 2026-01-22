# Portfolio Analyzer - Alpha Miner Pro

World-class capital allocation engine for mining stock portfolios with institutional-grade features.

## Quickstart

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run alpha_miner_institutional_v2.py
```

The app will open in your browser at `http://localhost:8501`

## Run Tests

```bash
pytest -q
```

Or use the Makefile:
```bash
make test
```

## Backtest Cache Management

The backtest runner uses a cache system to store price data and enable offline runs.

### Recommended Workflow

For reliable 6-month backtests, follow this workflow:

**A) Build Cache**
```bash
python3 backtest_runner.py \
  --start 2025-07-01 \
  --end 2026-01-01 \
  --portfolio_csv portfolio_enhanced.csv \
  --initial_cash 39570 \
  --data_dir ./.backtest_cache/ \
  --build_cache_only
```

Or use the Makefile:
```bash
make cache_build
```

**B) Verify Cache**
```bash
python3 backtest_runner.py \
  --start 2025-07-01 \
  --end 2026-01-01 \
  --portfolio_csv portfolio_enhanced.csv \
  --data_dir ./.backtest_cache/ \
  --verify_cache
```

Or use the Makefile:
```bash
make cache_verify
```

**C) Run Offline Backtest**
```bash
python3 backtest_runner.py \
  --start 2025-07-01 \
  --end 2026-01-01 \
  --portfolio_csv portfolio_enhanced.csv \
  --initial_cash 39570 \
  --offline \
  --data_dir ./.backtest_cache/
```

Or use the Makefile:
```bash
make backtest_offline
```

### Cache Commands

#### Build Cache Only

Build the cache for a date range (online mode, fetches from yfinance, then exits without simulation):

```bash
python3 backtest_runner.py \
  --start 2025-07-01 \
  --end 2026-01-01 \
  --portfolio_csv portfolio_enhanced.csv \
  --initial_cash 39570 \
  --data_dir ./.backtest_cache/ \
  --build_cache_only
```

**Features:**
- Fetches all symbols with retry logic and exponential backoff
- Handles rate limiting gracefully
- Fails fast with clear error messages if symbols can't be fetched (unless `--skip_missing_symbols` is set)
- Creates manifest.json for cache tracking

#### Verify Cache

Check that cache exists and is loadable for all required symbols:

```bash
python3 backtest_runner.py \
  --start 2025-07-01 \
  --end 2026-01-01 \
  --portfolio_csv portfolio_enhanced.csv \
  --data_dir ./.backtest_cache/ \
  --verify_cache
```

**Exit codes:**
- `0`: All symbols cached and loadable
- `2`: Missing or broken cache files (lists missing symbols)

#### Retry Missing Cache

If some symbols failed to cache, retry only the missing ones:

```bash
python3 backtest_runner.py \
  --start 2025-07-01 \
  --end 2026-01-01 \
  --symbols MISSING1,MISSING2 \
  --retry_missing_cache \
  --data_dir ./.backtest_cache/
```

#### Run Offline Backtest

Once cache is built and verified, run backtest in offline mode (zero network calls):

```bash
python3 backtest_runner.py \
  --start 2025-07-01 \
  --end 2026-01-01 \
  --portfolio_csv portfolio_enhanced.csv \
  --initial_cash 39570 \
  --offline \
  --data_dir ./.backtest_cache/
```

**Note:** Offline mode will fail with a friendly error message listing any missing symbols and the exact command to run to fix it.

### Advanced Options

- `--skip_missing_symbols`: Continue with symbols that have data, skip missing ones (reports skipped in summary JSON)
- `--symbols SYM1,SYM2`: Process only specified symbols (comma-separated)
- `--allow_partial_cache`: Allow proceeding with partial cache if some symbols fail to fetch
- `--sell_policy {veto_only,triggers,rebalance}`: Sell policy (default: `veto_only`)
  - `veto_only` (default): Conservative - only sells on hard veto (L0 liquidity, capital structure issues). HOLD action never triggers SELL.
  - `triggers`: Sells on active sell triggers (Avoid action, explicit sell signals, Sell Risk/Dilution Risk vetos)
  - `rebalance`: Aggressive rebalancing - always rebalances to target weights, even on HOLD

**Sell Policy Behavior:**
- **Default (`veto_only`)**: Most conservative. Only liquidates positions when there's a hard risk veto (e.g., L0 liquidity tier, capital structure issues). HOLD recommendations never trigger SELL trades, preventing day-1 liquidation when alpha is low.
- **`triggers`**: Sells when there are active sell triggers (Avoid action, explicit Sell signals, or risk-based vetos from Sell Risk/Dilution Risk models).
- **`rebalance`**: Always rebalances to target weights, even if action is HOLD. Use this for aggressive portfolio management.

## Replay Mode (Offline)

The app supports **Replay Mode** for offline analysis using saved evidence packs:

1. **Enable Replay Mode**: Toggle "Replay from Evidence Pack" in the sidebar
2. **Load Evidence Pack**: 
   - Upload a JSON evidence pack file, OR
   - Select from previously saved evidence packs
3. **Zero Network Calls**: Replay mode uses only cached data from the evidence pack - no live market data fetching

Evidence packs are automatically saved after each analysis run and can be downloaded for later replay or sharing.

## Features

- **Survival-First Philosophy**: Survival > Alpha | Sell-In-Time Focus | Gate-Based Risk Management
- **Morning Tape**: Gold & Silver predictions, metal regime status, daily action plan
- **Portfolio Health Check**: Overall health score, concentration warnings
- **Institutional Analysis**: SMC calculations, discovery exceptions, metal-aware position sizing
- **News Intelligence**: PP closed detection, market buzz integration
- **Financing Overhang**: Mining-specific financing lifecycle risk assessment
- **Tape/Regime Gate**: Macro-driven buy/sell decision gates

## Disclaimer

**This tool is for informational purposes only and does not constitute investment advice.** 

All analysis, recommendations, and outputs are based on automated calculations and should not be used as the sole basis for investment decisions. Always conduct your own research and consult with qualified financial advisors before making investment decisions. Past performance does not guarantee future results.
