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
```Or use the Makefile:
```bash
make test
```

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
