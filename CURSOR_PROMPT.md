# Cursor Implementation Prompt — Alpha Miner Pro Full Overhaul

You are working on a Streamlit-based mining stock portfolio analysis tool called Alpha Miner Pro. The main file is `alpha_miner_institutional_v2.py` (6,441 lines). The codebase also includes `alpha_miner_core.py`, `institutional_enhancements_v3.py`, `mining_tickers.py`, `sector_crawler.py`, and a `tests/` directory.

Complete ALL tasks below in order. Do not skip any. After each section, verify your changes don't break existing imports or tests.

---

## PHASE 1: CRITICAL BUG FIXES (Do these first)

### 1.1 — Delete Orphaned Code Block
**File:** `alpha_miner_institutional_v2.py`
**Problem:** Lines 620–760 contain a docstring and full function body that are NOT inside any function. They are left over from when `calculate_financing_overhang` was refactored to delegate to `core_calculate_financing_overhang` at line 615. The orphaned code starts with a floating docstring at line 620 and includes `return` statements that will cause a `SyntaxError` at module import.
**Action:** Delete everything from line 620 (the orphaned `"""`) through to the end of the orphaned function body (approximately line 760, where the next valid section header `# ============` begins). Keep the section header for Discovery Exception at line 617-619 intact. After deletion, verify the file still imports cleanly.

### 1.2 — Fix Hardcoded Absolute Path
**File:** `alpha_miner_institutional_v2.py`
**Problem:** Line 8 and line ~2217 both hardcode `env_path = "/Users/mats/PortfolioAnalyzer/hey.env"`. This only works on one developer's Mac.
**Action:** Replace BOTH occurrences with a relative path based on the script's location:
```python
env_path = Path(__file__).parent / "hey.env"
```
Make sure `from pathlib import Path` is available before line 8 (move the Path import above the env loading, or use `os.path` instead). The line 2217 occurrence in the sidebar diagnostics should also use the same relative path.

### 1.3 — Remove Duplicate Badge Rendering
**File:** `alpha_miner_institutional_v2.py`
**Problem:** Lines ~6318-6323 and ~6325-6330 are identical code blocks that both render the Financing Overhang badge. This causes the badge to appear twice in the UI.
**Action:** Delete the second copy (lines ~6325-6330). Keep the first occurrence intact.

### 1.4 — Single PORTFOLIO_SIZE Definition
**File:** `alpha_miner_institutional_v2.py` and `alpha_miner_core.py`
**Problem:** `PORTFOLIO_SIZE = 200000` is defined in both `alpha_miner_core.py:74` and `alpha_miner_institutional_v2.py:324`. If one changes, they diverge.
**Action:** Keep the definition in `alpha_miner_core.py`. In `alpha_miner_institutional_v2.py`, replace the local definition with an import:
```python
from alpha_miner_core import PORTFOLIO_SIZE
```
Add `PORTFOLIO_SIZE` to the existing import block from `alpha_miner_core` (around line 48-90). Delete the `PORTFOLIO_SIZE = 200000` line from v2.py.

### 1.5 — Replace Bare except: Clauses
**File:** `alpha_miner_institutional_v2.py`
**Problem:** Multiple `except: pass` clauses silently swallow all exceptions, hiding real bugs.
**Action:** Find every bare `except:` or `except: pass` in the file and replace with specific exception types. Use this mapping:
- `load_cache()` and `save_cache()`: use `except (IOError, json.JSONDecodeError, OSError):`
- Timestamp parsing blocks: use `except (ValueError, TypeError, OSError):`
- yfinance/Tiingo network calls: use `except (Exception) as e:` and add `import logging; logging.warning(f"...")` or at minimum a print statement
- Any remaining bare `except:`: replace with `except Exception:`

Do the same for `mining_tickers.py` (lines 89 and 144) and `sector_crawler.py`.

---

## PHASE 2: ARCHITECTURE — DECOMPOSE THE MAIN FILE

### 2.1 — Extract Data Fetching Module
**Create:** `data_fetchers.py`
**Action:** Move all data fetching functions out of `alpha_miner_institutional_v2.py` into a new `data_fetchers.py` module. This includes:
- The Tiingo REST fetch functions (search for `tiingo` fetch, `requests.get` calls for price data)
- The yfinance fallback fetch functions
- ETF proxy fetch logic
- The ticker sanitization functions (the `sanitize_ticker`, `clean_symbol` type functions)
- Any retry/fallback chain logic

Export all moved functions and import them back into `alpha_miner_institutional_v2.py`. Verify nothing breaks.

### 2.2 — Extract Spot Price Module
**Create:** `spot_prices.py`
**Action:** Move all spot metal price fetching into `spot_prices.py`:
- Gold spot price fetch (GC=F, GLD ETF fallback)
- Silver spot price fetch (SI=F, SLV ETF fallback)
- Uranium spot price fetch (URA ETF proxy)
- Shanghai Gold Exchange premium calculation
- GSR (Gold/Silver Ratio) calculation
- Metal direction forecasting (or keep that in institutional_enhancements_v3.py if already there)

Export all functions and import them back.

### 2.3 — Extract UI Display Module
**Create:** `ui_display.py`
**Action:** Move all Streamlit rendering/display helper functions into `ui_display.py`:
- Badge rendering functions (SMC badges, news quality badges, liquidity tier badges)
- Position card rendering
- Summary table formatting
- CSS injection (the big `st.markdown("""<style>...</style>""")` block)
- Chart rendering helpers (plotly wrappers)

Export all functions and import them back.

### 2.4 — Extract Discovery Tab Module
**Create:** `discovery_tab.py`
**Action:** Move the North American Discoveries tab logic (approximately lines 4853-6126 in the original file) into `discovery_tab.py`. This includes:
- The discovery scan loop
- The 10-gate exception logic calls
- Discovery results display
- Ticker expansion/validation for discovered symbols

Export as a single `render_discovery_tab()` function and call it from the main file.

### 2.5 — Extract Watchlist Module
**Create:** `watchlist.py`
**Action:** Move watchlist radar tab logic (approximately lines 6127-6231) into `watchlist.py`:
- Watchlist load/save (JSON file operations)
- Watchlist display rendering
- Add/remove symbol logic

Export as `render_watchlist_tab()` and call from the main file.

After all extractions, `alpha_miner_institutional_v2.py` should be approximately 2,000-2,500 lines containing only the main orchestration, session state initialization, sidebar config, and tab routing.

---

## PHASE 3: TECHNICAL ANALYSIS INDICATORS

### 3.1 — Add RSI Calculation
**File:** `alpha_miner_core.py`
**Action:** Add a function to calculate 14-day RSI:
```python
def calculate_rsi(hist: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index. Returns Series with RSI values (0-100)."""
    delta = hist['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
```
Integrate RSI into the M1_Momentum model scoring. RSI < 30 = oversold bonus (+5 alpha), RSI > 70 = overbought penalty (-5 alpha).

### 3.2 — Add MACD Calculation
**File:** `alpha_miner_core.py`
**Action:** Add MACD (12/26/9):
```python
def calculate_macd(hist: pd.DataFrame) -> dict:
    """Calculate MACD line, signal line, and histogram."""
    close = hist['Close']
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return {
        'macd_line': macd_line.iloc[-1],
        'signal_line': signal_line.iloc[-1],
        'histogram': histogram.iloc[-1],
        'crossover': 'bullish' if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2] else
                     'bearish' if macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2] else 'none'
    }
```
Integrate into M1_Momentum: bullish crossover = +3 alpha, bearish crossover = -3 alpha.

### 3.3 — Add Bollinger Bands
**File:** `alpha_miner_core.py`
**Action:** Add Bollinger Bands (20, 2):
```python
def calculate_bollinger_bands(hist: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> dict:
    """Calculate Bollinger Bands. Returns dict with upper, middle, lower, %B, bandwidth."""
    close = hist['Close']
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    current_price = close.iloc[-1]
    pct_b = (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1]) if (upper.iloc[-1] - lower.iloc[-1]) > 0 else 0.5
    bandwidth = (upper.iloc[-1] - lower.iloc[-1]) / middle.iloc[-1] if middle.iloc[-1] > 0 else 0
    return {
        'upper': upper.iloc[-1],
        'middle': middle.iloc[-1],
        'lower': lower.iloc[-1],
        'pct_b': pct_b,
        'bandwidth': bandwidth,
        'squeeze': bandwidth < 0.05  # Tight squeeze detection
    }
```
Integrate into M4_Volatility: squeeze detection signals potential breakout, %B < 0 (below lower band) = oversold signal.

### 3.4 — Add OBV (On-Balance Volume)
**File:** `alpha_miner_core.py`
**Action:** Add OBV calculation:
```python
def calculate_obv(hist: pd.DataFrame) -> dict:
    """Calculate On-Balance Volume and its trend."""
    obv = (np.sign(hist['Close'].diff()) * hist['Volume']).fillna(0).cumsum()
    obv_ma20 = obv.rolling(20).mean()
    return {
        'obv_current': obv.iloc[-1],
        'obv_trend': 'rising' if obv.iloc[-1] > obv_ma20.iloc[-1] else 'falling',
        'obv_divergence': 'bullish' if hist['Close'].iloc[-1] < hist['Close'].iloc[-5] and obv.iloc[-1] > obv.iloc[-5] else
                          'bearish' if hist['Close'].iloc[-1] > hist['Close'].iloc[-5] and obv.iloc[-1] < obv.iloc[-5] else 'none'
    }
```
Integrate into M7_SMC: OBV divergence confirms or contradicts Smart Money signals.

### 3.5 — Add ADX (Average Directional Index)
**File:** `alpha_miner_core.py`
**Action:** Add 14-day ADX for trend strength:
```python
def calculate_adx(hist: pd.DataFrame, period: int = 14) -> dict:
    """Calculate ADX for trend strength measurement."""
    high = hist['High']
    low = hist['Low']
    close = hist['Close']

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm < minus_dm)] = 0
    minus_dm[(minus_dm < plus_dm)] = 0

    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    adx = dx.rolling(window=period).mean()

    return {
        'adx': adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0,
        'plus_di': plus_di.iloc[-1] if not pd.isna(plus_di.iloc[-1]) else 0,
        'minus_di': minus_di.iloc[-1] if not pd.isna(minus_di.iloc[-1]) else 0,
        'trend_strength': 'strong' if adx.iloc[-1] > 25 else 'weak' if adx.iloc[-1] < 20 else 'moderate'
    }
```
Integrate into Regime throttle: ADX < 20 = trendless market, throttle new buys by 50%.

### 3.6 — Display TA Indicators in UI
**File:** `ui_display.py` (or `alpha_miner_institutional_v2.py` if not yet extracted)
**Action:** In the detailed position analysis section, add a "Technical Analysis" expander for each position that shows:
- RSI gauge (with overbought/oversold zones colored)
- MACD histogram chart (green/red bars)
- Bollinger Band %B indicator
- OBV trend direction
- ADX trend strength
- Combined TA signal: Bullish / Neutral / Bearish

Use Plotly for the charts. Display as badges in the position card summary row.

---

## PHASE 4: FUNDAMENTAL ANALYSIS ENHANCEMENTS

### 4.1 — AISC Tracking System
**Create:** `aisc_tracker.py`
**Action:** Build an AISC (All-In Sustaining Cost) tracking system:
```python
"""
AISC Tracker — Track All-In Sustaining Costs for mining producers.
Since AISC is not available from standard APIs, this uses:
1. A manually-maintained JSON lookup file (aisc_data.json)
2. Estimation from operating costs when available via yfinance
3. Industry average fallback by metal type
"""

AISC_INDUSTRY_AVERAGES = {
    'Gold': {'average': 1250, 'high': 1500, 'low': 900, 'unit': '$/oz'},
    'Silver': {'average': 18.50, 'high': 24.00, 'low': 13.00, 'unit': '$/oz'},
    'Uranium': {'average': 45.00, 'high': 60.00, 'low': 30.00, 'unit': '$/lb'},
    'Copper': {'average': 2.50, 'high': 3.20, 'low': 1.80, 'unit': '$/lb'},
    'Lithium': {'average': 12000, 'high': 18000, 'low': 8000, 'unit': '$/t'},
}

def load_aisc_data() -> dict:
    """Load AISC data from aisc_data.json file."""
    ...

def estimate_aisc_from_financials(info: dict, metal: str) -> dict:
    """Estimate AISC from yfinance financial data."""
    ...

def calculate_aisc_margin(aisc: float, spot_price: float) -> dict:
    """Calculate margin of safety: (Spot - AISC) / Spot."""
    ...

def get_aisc_score(symbol: str, metal: str, info: dict = None) -> dict:
    """
    Returns dict with:
    - aisc_estimate: float ($/oz or $/lb)
    - aisc_source: str ('manual', 'estimated', 'industry_average')
    - margin_pct: float (margin of safety percentage)
    - score: int (0-100, higher = better margin)
    """
    ...
```
Also create `aisc_data.json` with known AISC values for the top 100 miners in `_TOP_100_MINERS` list. Integrate the AISC score into M2_Value model.

### 4.2 — P/NAV Estimation
**File:** `alpha_miner_core.py`
**Action:** Add a P/NAV estimation function for producers with known reserves:
```python
def estimate_p_nav(market_cap: float, reserves_oz: float, spot_price: float,
                   aisc: float, discount_rate: float = 0.05, mine_life_years: float = 10) -> dict:
    """
    Estimate Price-to-NAV ratio using DCF of reserves.
    NAV = sum of (annual_production * (spot - aisc)) / (1 + discount_rate)^t
    P/NAV < 0.5 = deep value, 0.5-1.0 = fair value, > 1.0 = premium
    """
    ...
```
Store reserve estimates in a JSON lookup file. Integrate P/NAV into M2_Value scoring.

### 4.3 — Debt-to-Equity Integration
**File:** `alpha_miner_core.py`
**Action:** Add debt-to-equity ratio to the Capital Structure veto gate. Pull from yfinance `.info`:
```python
def calculate_leverage_risk(info: dict) -> dict:
    """
    Calculate leverage risk score.
    - D/E > 1.0 for junior miners = high risk (score 70+)
    - D/E > 0.5 for explorers = moderate risk (score 50+)
    - Cash > Debt = low risk (score < 30)
    """
    total_debt = info.get('totalDebt', 0) or 0
    total_equity = info.get('totalStockholderEquity', 0) or 0
    cash = info.get('totalCash', 0) or 0
    ...
```
Add leverage risk as an additional input to the Capital Structure veto gate.

### 4.4 — Insider Transaction Tracking
**File:** Create `insider_tracker.py`
**Action:** Replace the hardcoded `Insider_Buying_90d: [False] * 14` with real data:
```python
"""
Insider Transaction Tracker
Sources:
1. SEC EDGAR Form 4 filings (US stocks)
2. SEDI filings (Canadian stocks via .TO/.V)
3. yfinance insider transactions (if available)
"""

def fetch_insider_transactions(symbol: str, days: int = 90) -> dict:
    """
    Returns dict with:
    - has_insider_buying: bool
    - net_insider_shares: int (positive = net buying)
    - transaction_count: int
    - largest_purchase: float (dollar value)
    - insider_names: list[str]
    - source: str
    """
    # Try yfinance first
    try:
        ticker = yf.Ticker(symbol)
        insider_df = ticker.insider_transactions
        if insider_df is not None and not insider_df.empty:
            # Filter to last N days
            # Calculate net buying/selling
            ...
    except Exception:
        pass

    # Fallback: return unknown
    return {
        'has_insider_buying': None,  # None = unknown, not False
        'net_insider_shares': 0,
        'transaction_count': 0,
        'source': 'unavailable'
    }
```
Update `DEFAULT_PORTFOLIO` initialization to call this function instead of hardcoding `False`. Add insider buying as a +10 alpha bonus in M6_Discovery.

---

## PHASE 5: REAL EXCHANGE DISCOVERY

### 5.1 — Implement TSX-V Discovery
**File:** `sector_crawler.py`
**Action:** Replace the placeholder `discover_tsxv_tickers()` with a real implementation:
```python
def discover_tsxv_tickers(sector: str = "mining", max_results: int = 50) -> List[str]:
    """
    Discover TSX Venture (.V) mining tickers using TMX Money screener.
    Falls back to curated list if API unavailable.
    """
    # Option 1: Scrape TMX Money screener (public, no API key needed)
    # URL: https://money.tmx.com/en/quote/search?sector=Mining
    # Parse with BeautifulSoup (already in requirements.txt)

    # Option 2: Use the existing Tiingo search with .V suffix filter

    # Option 3: Maintain a curated CSV of TSX-V mining tickers
    # (updated quarterly)
    ...
```

### 5.2 — Implement OTC Discovery
**File:** `sector_crawler.py`
**Action:** Replace the placeholder `discover_otc_tickers()`:
```python
def discover_otc_tickers(sector: str = "mining", max_results: int = 50) -> List[str]:
    """
    Discover OTC Markets mining tickers.
    Uses OTC Markets screener data.
    """
    # OTC Markets has a public screener at:
    # https://www.otcmarkets.com/research/stock-screener
    # Filter by: Industry = Mining, Market = OTCQX/OTCQB
    ...
```

### 5.3 — SEC EDGAR Mining Company Search
**Create:** `edgar_scanner.py`
**Action:** Add SEC EDGAR full-text search for mining companies:
```python
"""
SEC EDGAR Scanner — Find mining companies from public filings.
Uses the free EDGAR Full-Text Search API (no key required).
https://efts.sec.gov/LATEST/search-index?q="mining"&dateRange=custom&startdt=2024-01-01
"""

def search_mining_filings(keywords: list = None, days_back: int = 90) -> List[dict]:
    """Search EDGAR for recent mining-related filings."""
    ...

def extract_tickers_from_filings(filings: List[dict]) -> List[str]:
    """Extract ticker symbols from filing metadata."""
    ...
```

---

## PHASE 6: PORTFOLIO OPTIMIZATION

### 6.1 — Mean-Variance Optimization
**Create:** `portfolio_optimizer.py`
**Action:**
```python
"""
Portfolio Optimizer — Markowitz Mean-Variance + Risk Parity + Kelly.
Uses historical returns from the existing hist_cache.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def calculate_correlation_matrix(hist_cache: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate rolling correlation matrix between portfolio positions."""
    ...

def optimize_mean_variance(
    hist_cache: Dict[str, pd.DataFrame],
    risk_free_rate: float = 0.05,
    target_return: float = None
) -> Dict:
    """
    Markowitz mean-variance optimization.
    Returns optimal weights, efficient frontier points, and Sharpe ratio.
    """
    ...

def optimize_risk_parity(hist_cache: Dict[str, pd.DataFrame]) -> Dict:
    """
    Risk parity: equal risk contribution from each position.
    Returns target weights where each position contributes equally to portfolio variance.
    """
    ...

def kelly_position_size(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """
    Kelly Criterion optimal position size.
    f* = (p * b - q) / b where p=win_rate, q=1-p, b=avg_win/avg_loss
    Returns fraction of portfolio to allocate (capped at 25% for safety).
    """
    ...

def detect_concentration_risk(weights: Dict[str, float], metadata: Dict) -> List[str]:
    """
    Detect portfolio concentration risks:
    - Single metal > 40% of portfolio
    - Single jurisdiction > 50%
    - Correlated pairs > 0.8 correlation with combined weight > 20%
    """
    ...
```

### 6.2 — UI Integration for Optimizer
**File:** Main Streamlit app
**Action:** Add a new tab "Portfolio Optimization" that shows:
- Current vs. optimal weights (bar chart comparison)
- Efficient frontier plot (Plotly scatter)
- Correlation heatmap
- Concentration risk warnings
- Suggested rebalancing trades to move toward optimal

---

## PHASE 7: ALERTING SYSTEM

### 7.1 — Alert Engine
**Create:** `alert_engine.py`
**Action:**
```python
"""
Alert Engine — Real-time monitoring and notification system.
Checks conditions and writes alerts to a JSON file + optional email/webhook.
"""

ALERT_TYPES = {
    'PRICE_BREAKOUT': 'Price broke above resistance',
    'PRICE_BREAKDOWN': 'Price broke below support',
    'VOLUME_SPIKE': 'Volume > 2x 20-day average',
    'RSI_OVERSOLD': 'RSI dropped below 30',
    'RSI_OVERBOUGHT': 'RSI rose above 70',
    'MACD_CROSSOVER': 'MACD bullish/bearish crossover',
    'FINANCING_ANNOUNCED': 'New financing detected in news',
    'INSIDER_BUYING': 'Insider purchase detected',
    'LIQUIDITY_DOWNGRADE': 'Liquidity tier downgraded',
    'METAL_REGIME_CHANGE': 'Metal price crossed key MA',
    'BOLLINGER_SQUEEZE': 'Bollinger Band squeeze detected',
    'SMC_BOS': 'Break of Structure detected',
    'PRINCIPAL_HARVEST': 'Position reached 2x cost basis',
}

def check_all_alerts(portfolio_df: pd.DataFrame, hist_cache: dict,
                     news_cache: dict, spot_prices: dict) -> List[dict]:
    """Run all alert checks and return triggered alerts."""
    ...

def save_alerts(alerts: List[dict], filepath: str = None):
    """Persist alerts to JSON file."""
    ...

def render_alerts_panel(alerts: List[dict]):
    """Render alerts in Streamlit sidebar."""
    ...
```

### 7.2 — Integrate Alerts into Sidebar
**File:** Main Streamlit app sidebar
**Action:** Add an "Alerts" section at the top of the sidebar that shows:
- Count of active alerts by severity (Critical / Warning / Info)
- Expandable list of recent alerts
- "Dismiss" button per alert
- Alert history log

---

## PHASE 8: MULTI-TIMEFRAME ANALYSIS

### 8.1 — Multi-Timeframe SMC
**File:** `institutional_enhancements_v3.py`
**Action:** Enhance `calculate_smc_structure()` to accept a `timeframe` parameter:
```python
def calculate_smc_structure(hist: pd.DataFrame, timeframe: str = 'daily') -> dict:
    """
    Calculate SMC structure (BOS/CHoCH) for a given timeframe.
    timeframe: 'daily', 'weekly', 'monthly'
    """
    if timeframe == 'weekly':
        hist = hist.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
    elif timeframe == 'monthly':
        hist = hist.resample('ME').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
    # ... existing BOS/CHoCH logic ...
```

Add a new function:
```python
def calculate_multi_timeframe_alignment(hist: pd.DataFrame) -> dict:
    """
    Check trend alignment across daily, weekly, and monthly timeframes.
    Returns:
    - alignment_score: 0-100 (100 = all timeframes agree)
    - daily_trend: 'bullish' / 'bearish' / 'neutral'
    - weekly_trend: 'bullish' / 'bearish' / 'neutral'
    - monthly_trend: 'bullish' / 'bearish' / 'neutral'
    - signal: 'STRONG_BUY' / 'BUY' / 'NEUTRAL' / 'SELL' / 'STRONG_SELL'
    """
    daily = calculate_smc_structure(hist, 'daily')
    weekly = calculate_smc_structure(hist, 'weekly')
    monthly = calculate_smc_structure(hist, 'monthly')
    ...
```

Integrate the alignment score as a modifier to M7_SMC: aligned = full weight, misaligned = 50% weight.

---

## PHASE 9: SECTOR CORRELATION DASHBOARD

### 9.1 — Correlation Analysis
**File:** `portfolio_optimizer.py` (created in Phase 6)
**Action:** Add sector correlation functions:
```python
def calculate_rolling_correlation(hist_cache: Dict[str, pd.DataFrame], window: int = 60) -> pd.DataFrame:
    """Calculate rolling 60-day correlation matrix."""
    ...

def calculate_metal_beta(hist: pd.DataFrame, benchmark_hist: pd.DataFrame) -> float:
    """Calculate beta of a stock relative to a metal ETF benchmark (GLD/SLV/URA)."""
    ...

def analyze_sector_concentration(portfolio_df: pd.DataFrame) -> dict:
    """
    Analyze concentration by:
    - Metal type (Gold/Silver/Uranium/Copper/Lithium)
    - Geography (Canada/USA/Australia/Mexico/Other)
    - Stage (Explorer/Developer/Producer)
    - Liquidity tier (L0/L1/L2/L3)
    Returns warnings when concentration exceeds thresholds.
    """
    ...
```

### 9.2 — Correlation Dashboard Tab
**File:** Main Streamlit app
**Action:** Add a new tab "Correlation & Risk" that displays:
- Interactive correlation heatmap (Plotly)
- Metal beta chart for each position
- Sector concentration pie charts (by metal, geography, stage)
- Concentration risk warnings
- Diversification score (0-100)

---

## PHASE 10: BACKTEST ENHANCEMENTS

### 10.1 — Transaction Cost Modeling
**File:** `backtest_runner.py`
**Action:** Add transaction cost parameters to `simulate_day()`:
```python
def simulate_day(..., spread_bps: float = 50, commission_per_trade: float = 0.0):
    """
    spread_bps: Estimated bid-ask spread in basis points (default 50bps = 0.5% for junior miners)
    commission_per_trade: Fixed commission per trade (default 0 for most brokers)
    """
    # Apply spread cost to all BUY and SELL trades
    # BUY at ask (price + spread/2), SELL at bid (price - spread/2)
    ...
```

### 10.2 — Drawdown Analysis
**File:** `backtest_runner.py`
**Action:** Add drawdown analysis to the backtest report:
```python
def calculate_drawdown_stats(equity_curve: pd.Series) -> dict:
    """
    Calculate:
    - max_drawdown_pct: Maximum peak-to-trough decline
    - max_drawdown_duration_days: Longest recovery period
    - calmar_ratio: Annualized return / max drawdown
    - current_drawdown_pct: Current drawdown from peak
    - time_underwater_pct: Percentage of time in drawdown
    """
    ...
```
Add these stats to the backtest summary report and display in the Backtest Verification tab.

### 10.3 — Regime-Conditional Backtest Results
**File:** `backtest_runner.py`
**Action:** Split backtest results by metal regime:
```python
def segment_by_regime(daily_df: pd.DataFrame, gold_hist: pd.DataFrame) -> dict:
    """
    Segment backtest results by gold price regime:
    - Bull: Gold > 200-day MA and MA trending up
    - Bear: Gold < 200-day MA and MA trending down
    - Choppy: Otherwise

    Returns per-regime: return, Sharpe, max drawdown, win rate
    """
    ...
```

---

## PHASE 11: EXPANDED TEST COVERAGE

### 11.1 — Add TA Indicator Tests
**Create:** `tests/test_technical_analysis.py`
**Action:** Write unit tests for all new TA indicators:
- RSI: test known values, test with insufficient data, test edge cases (all up / all down)
- MACD: test crossover detection, test with flat data
- Bollinger Bands: test squeeze detection, test %B range
- OBV: test divergence detection
- ADX: test trend strength thresholds

### 11.2 — Add Spot Price Tests
**Create:** `tests/test_spot_prices.py`
**Action:** Test spot price fetching with mocked API responses:
- Test Tiingo forex API response parsing
- Test yfinance fallback
- Test Shanghai premium calculation
- Test GSR calculation

### 11.3 — Add Discovery Tests
**Create:** `tests/test_discovery.py`
**Action:** Test the 10-gate discovery exception logic:
- Test each gate individually
- Test gate combinations
- Test with edge case data (missing fields, zero prices)

### 11.4 — Add Portfolio Optimizer Tests
**Create:** `tests/test_optimizer.py`
**Action:** Test optimization functions:
- Test correlation matrix with synthetic data
- Test mean-variance with 2-asset case (known analytical solution)
- Test risk parity convergence
- Test Kelly criterion bounds
- Test concentration detection

### 11.5 — Add Integration Smoke Test
**Create:** `tests/test_integration_smoke.py`
**Action:** Write an end-to-end smoke test that:
1. Loads the default portfolio
2. Runs the analysis loop with mocked market data
3. Verifies all models produce valid scores
4. Verifies no crashes with missing data
5. Verifies the final DataFrame has all required columns

---

## PHASE 12: FINAL CLEANUP

### 12.1 — Remove Dead Code
- Delete `sector_crawler.py` placeholder functions if replaced by real implementations
- Remove any commented-out code blocks longer than 5 lines
- Delete unused imports (run `pylint` or `ruff` to find them)
- Remove the orphaned docstring fixed in Phase 1

### 12.2 — Consistent Logging
**Action:** Replace all `print()` debug statements with proper logging:
```python
import logging
logger = logging.getLogger('alpha_miner')
```
Use `logger.debug()` for verbose output, `logger.warning()` for fallbacks, `logger.error()` for failures.

### 12.3 — Update Version
**File:** `alpha_miner_institutional_v2.py`
**Action:** Update the version string to reflect the overhaul:
```python
VERSION = "8.0-FULL-OVERHAUL"
VERSION_DATE = "2026-01-30"
```

### 12.4 — Run All Tests
**Action:** Run `pytest tests/ -v` and fix any failures. All tests must pass before considering the overhaul complete.

---

## IMPORTANT NOTES FOR CURSOR

1. **Do not delete any existing functionality** — only enhance and fix. The existing 7-model alpha scoring, gate-based veto logic, evidence pack system, and backtest runner should all continue working.
2. **Preserve all existing imports** — when moving code to new files, update imports in all files that reference the moved functions.
3. **Keep backward compatibility** — existing JSON cache files, evidence packs, and portfolio CSVs should still load correctly.
4. **Test after each phase** — run `pytest tests/ -v` after completing each phase to catch regressions.
5. **The main file must still run** — `streamlit run alpha_miner_institutional_v2.py` should work at every phase.
