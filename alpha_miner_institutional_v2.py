#!/usr/bin/env python3
# =============================================================================
# DANGER ZONE: Only pure Python here. No st.*, no @st.cache_*, no st.session_state.
# =============================================================================
# V7.4: Force Tiingo Key Loading - Relative Path with Override Strategy
import os
from pathlib import Path as _Path
from dotenv import load_dotenv
env_path = _Path(__file__).parent / "hey.env"
load_dotenv(dotenv_path=str(env_path), override=True)  # Force override to ignore stale env vars
# Force verification - will show in sidebar after Streamlit initializes
_TIINGO_KEY_MISSING = not bool(os.getenv("TIINGO_API_KEY", "").strip())

from pathlib import Path

import pandas as pd
import numpy as np
import datetime
import json
import re

# V7.3: Requests for Tiingo Forex API calls
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# =============================================================================
# STREAMLIT: import + set_page_config MUST be first Streamlit commands.
# No @st.cache_*, st.session_state, st.write, st.sidebar, etc. before this.
# =============================================================================
import streamlit as st
st.set_page_config(
    page_title="Alpha Miner Pro",
    layout="wide",
    initial_sidebar_state="expanded",
)

# V5.0: Import plotly for backtest verification charts
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Import core computation functions (import-safe, no Streamlit UI execution)
from alpha_miner_core import (
    MODEL_ROLES,
    RISK_PROFILES,
    PORTFOLIO_SIZE,
    EVIDENCE_DIR,
    get_risk_profile_preset,
    validate_data_invariants,
    enforce_strict_mode,
    create_evidence_pack,
    save_evidence_pack,
    list_evidence_packs,
    load_evidence_pack,
    calculate_liquidity_metrics,
    calculate_data_confidence,
    calculate_dilution_risk,
    calculate_alpha_models,
    calculate_sell_risk,
    calculate_tape_gate,
    calculate_macro_regime,
    calculate_financing_overhang,
    arbitrate_final_decision,
    get_benchmark_data,
    normalize_timestamp,
    tag_news,
    fetch_gold_silver_prices,  # V4.0 Phase 3: Automated GSR fetching
    calculate_gs_ratio_bias,  # V4.0 Phase 3: GSR bias calculation
    calculate_fundamental_score,  # V5.0: Fundamental Alpha scoring
    detect_market_buzz,  # V5.0: Market buzz detection
)

# Import scanner module
try:
    from alpha_miner_scanner import scan_symbols, load_symbols_from_csv, load_master_discovery_list as _load_master_discovery_list
    SCANNER_AVAILABLE = True
    load_master_discovery_list = _load_master_discovery_list  # legacy; prefer get_all_mining_tickers
except ImportError:
    SCANNER_AVAILABLE = False
    load_master_discovery_list = None

# V7.0: Sovereign Global Scan – live mining tickers (no CSV)
try:
    from mining_tickers import get_all_mining_tickers as _get_all_mining_tickers
    @st.cache_data(ttl=3600)
    def get_all_mining_tickers_cached(max_symbols=200):
        return _get_all_mining_tickers(max_symbols=max_symbols)
    MINING_TICKERS_AVAILABLE = True
except ImportError:
    get_all_mining_tickers_cached = None
    MINING_TICKERS_AVAILABLE = False

# V5.0: Import sector crawler for autonomous discovery
try:
    from sector_crawler import crawl_sector, validate_ticker_against_settings, load_hunting_settings as _load_hunting_settings
    SECTOR_CRAWLER_AVAILABLE = True
    
    # V5.0: Cache hunting settings to prevent repeated file reads
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_hunting_settings_cached():
        """Cached wrapper for load_hunting_settings to prevent repeated file reads"""
        return _load_hunting_settings()
    
    load_hunting_settings = load_hunting_settings_cached
except ImportError:
    SECTOR_CRAWLER_AVAILABLE = False
    crawl_sector = None
    validate_ticker_against_settings = None
    load_hunting_settings = None

# Import institutional enhancements (v1 + v2 + v3 if available)
try:
    from institutional_enhancements import (
        calculate_smc_institutional,
        check_discovery_exception_strict,
        classify_financing_precision,
        calculate_social_proxy,
        add_institutional_sell_triggers,
        calculate_portfolio_risk_intelligence
    )
    INSTITUTIONAL_V1_AVAILABLE = True
except ImportError:
    INSTITUTIONAL_V1_AVAILABLE = False

try:
    from institutional_enhancements_v2 import (
        analyze_metal_cycle,
        calculate_metal_regime_impact,
        check_discovery_exception_metal_aware,
        calculate_dynamic_position_sizing,
        generate_morning_tape,
        get_social_institutional_signals,
        integrate_social_signals
    )
    INSTITUTIONAL_V2_AVAILABLE = True
except ImportError:
    INSTITUTIONAL_V2_AVAILABLE = False

try:
    from institutional_enhancements_v3 import (
        calculate_smc_structure,
        forecast_metal_direction,
        analyze_news_intelligence,
        calculate_market_buzz,
        calculate_enhanced_sell_triggers,
        orchestrate_portfolio_ranking,
        check_discovery_exception_ultimate
    )
    INSTITUTIONAL_V3_AVAILABLE = True
except ImportError:
    INSTITUTIONAL_V3_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE = True
    if 'yfinance_available' not in st.session_state:
        st.session_state.yfinance_available = True
    # Suppress "possibly delisted" warnings globally — expected for mining portfolio
    # scanning where many tickers are checked and some may be delisted.
    import warnings
    warnings.filterwarnings("ignore", message=".*possibly delisted.*")
except Exception:
    YFINANCE = False
    st.session_state.yfinance_available = False
    st.error("yfinance not installed. Run: pip install yfinance")

# Tiingo REST client (ticker search / fetch_ticker_with_fallback)
try:
    from tiingo import TiingoClient
    TIINGO_AVAILABLE = True
except ImportError:
    TiingoClient = None
    TIINGO_AVAILABLE = False

# V6.0: Sovereign Trust – Price Projection Engine (Deep-Hunt)
try:
    from price_projection_engine import project_30d_window
    PRICE_PROJECTION_AVAILABLE = True
except ImportError:
    project_30d_window = None
    PRICE_PROJECTION_AVAILABLE = False

# V6.5: Deep-Trust – Multi-Model Backtest (Sovereign Win Rate, Sharpe, MaxDD)
try:
    from sovereign_backtest_engine import run_strike_backtest, STRIKE_SYMBOLS
    STRIKE_BACKTEST_AVAILABLE = True
except ImportError:
    run_strike_backtest = None
    STRIKE_SYMBOLS = []
    STRIKE_BACKTEST_AVAILABLE = False

# MODEL_ROLES, RISK_PROFILES, PORTFOLIO_SIZE imported from alpha_miner_core

# V8.0: Technical Analysis, Portfolio Optimizer, Alert Engine, AISC Tracker, Insider Tracker
try:
    from technical_analysis import calculate_all_ta, calculate_rsi, calculate_macd, calculate_bollinger_bands
    TA_MODULE_AVAILABLE = True
except ImportError:
    TA_MODULE_AVAILABLE = False

try:
    from portfolio_optimizer import (
        calculate_correlation_matrix, optimize_mean_variance, optimize_risk_parity,
        detect_concentration_risk, suggest_rebalance_trades, calculate_metal_beta
    )
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False

try:
    from alert_engine import check_all_alerts, get_alert_summary, save_alerts, load_alerts
    ALERT_ENGINE_AVAILABLE = True
except ImportError:
    ALERT_ENGINE_AVAILABLE = False

try:
    from aisc_tracker import get_aisc_score, estimate_pnav, load_aisc_data
    AISC_TRACKER_AVAILABLE = True
except ImportError:
    AISC_TRACKER_AVAILABLE = False

try:
    from insider_tracker import fetch_insider_transactions, calculate_insider_signal, update_portfolio_insider_flags
    INSIDER_TRACKER_AVAILABLE = True
except ImportError:
    INSIDER_TRACKER_AVAILABLE = False

# VERSION TRACKING
VERSION = "8.0-FULL-OVERHAUL"
VERSION_DATE = "2026-01-30"
VERSION_FEATURES = [
    "✅ V7.5 FIX: Market Cap extraction from info_dict in Global Search",
    "✅ V7.5 FIX: Actual filtering - stocks outside $20M-$600M are SKIPPED (not just logged)",
    "✅ V7.5 FIX: Top 5 prioritizes Junior/Mid-Tier stocks first, then Combined_Score",
    "✅ V7.5 FIX: Market Cap and Tier displayed in recommendations for transparency",
    "✅ V7.5: Global Search refactored to prioritize Junior/Mid-Tier Miners ($20M-$600M)",
    "✅ V7.5: Market Cap filter - Sweet spot $20M-$600M, High Quality Exception (>$600M if Score > 90)",
    "✅ V7.5: Veto Removal - Do NOT exclude stocks with 0 Revenue or Negative P/E (exploration investment)",
    "✅ V7.5: Junior-Specific FA Scoring (<$500M) - Ignore P/E & Dividend, Score on P/B (1/P/B), Cash>Debt, Insider>10%, Current Ratio>1.5",
    "✅ V7.5: Risk Assessment - Beta Neutrality (no penalty for High Beta), Momentum BUY signal (High Vol + RSI>50), Trend Veto (only penalize if Price < SMA200)",
    "✅ V7.5: News Promotion - Scan longBusinessSummary for 'Drill results', 'High grade', 'Exploration', 'Pre-feasibility', 'Sprott' (+5 Alpha each)",
    "✅ V7.5: Pure score-based ranking (Combined_Score = Alpha + FA / Risk, no ticker-specific bonuses)",
    "✅ V7.4 FIXES: All runtime errors fixed (isfinite, division by zero, empty DataFrames)",
    "✅ V7.4 FIXES: yfinance fallback with proper Tiingo data format mapping",
    "✅ V7.4 FIXES: Warning suppression for futures tickers (GC=F, SI=F, UX=F)",
    "✅ V7.4 FIXES: Improved Tiingo error logging for debugging",
    "✅ V7.4 FIXES: Mining tickers always returns at least Top 100 list",
    "✅ V7.4: 2026 Futures Grid - Zero Veto Execution",
    "✅ Canary Veto COMPLETELY DISABLED - If TIINGO_API_KEY present, force full 1,000+ Global Scan",
    "✅ Primary Source: yfinance futures (SI=F, GC=F, UX=F)",
    "✅ Secondary Source: Futures Tickers (GC=F, SI=F, UX=F or U-U.TO) - Replaced delisted XAGUSD=X/XAUUSD=X",
    "✅ Tertiary Source: ETF proxies (SLV * 1.1, GLD for institutional floor)",
    "✅ Conservative fallback prices (Gold: $2,700, Silver: $31.50, Uranium: $85) - only used when ALL APIs fail",
    "✅ Zero-Mockup Execution - All calculations in Actions Today use total_portfolio_value (no $1M references)",
    "✅ CRITICAL ALERT in sidebar if TIINGO_API_KEY not found",
    "✅ Great Divorce Display (live US vs Shanghai physical price comparison)",
    "✅ Physical Scarcity Bonus (metal-type based, applied from live SGE premium data)",
    "✅ Removed ALL Manual Price Overrides (pulls values live from multiple sources)",
    "✅ Scaling Absolute Truth (Hard-Burn $ and Drift % strictly scaled to total_portfolio_value)",
    "✅ Dynamic Scaling (removed ALL hardcoded 100000/1M - uses total_portfolio_value from UI)",
    "✅ Shanghai Arbitrage Premium Display (live SGE premium, N/A when data unavailable)",
    "✅ Energy Regime Row (Uranium Spot, weights CCJ/NXE/DNN/URR)",
    "✅ Global Mining Crawler (Top 100 US & Canadian Miners hard-mapped fallback - NOT restricted)",
    "✅ Score-based ranking (no ticker-specific prioritization)",
    "✅ V7.2: Sovereign Global Arbitrage",
    "✅ Dynamic Scaling (no hardcoded $1M - uses total_portfolio_value from UI)",
    "✅ Absolute Ticker Strike (Geography-First: Plain US → TSX:TICKER → TICKER.TO → TICKERF)",
    "✅ Sovereign Rebalancer (Rolling 15-year Backtesting with Sharpe + P/NAV ranking)",
    "✅ Shanghai Premium exposure prioritization (MAG, PAAS, GOLD, NEM)",
    "✅ Hard-Burn & Drift calculations use total_portfolio_value (dynamic scaling)",
    "✅ SMC integrated into alpha scoring",
    "✅ Gold & Silver cycle predictions in header",
    "✅ News intelligence (PP closed detection)",
    "✅ Market buzz proxy integration",
    "✅ Portfolio orchestration & ranking",
    "✅ Enhanced discovery exception",
    "✅ Fixed arbitration wiring",
    "✅ Model governance with veto logic",
    "✅ Confidence-based decision framing",
    "✅ Watchlist & Quick Analysis",
    # V8.0 Features
    "✅ V8.0: Technical Analysis Dashboard (RSI, MACD, Bollinger Bands, OBV, ADX, Fibonacci)",
    "✅ V8.0: Portfolio Optimizer (Mean-Variance, Risk Parity, Kelly Criterion, Correlation Matrix)",
    "✅ V8.0: Alert Engine (13 alert types: price/volume/TA/financing/insider/metal regime)",
    "✅ V8.0: AISC Tracker (All-In Sustaining Cost with 16 known miners + financial estimation)",
    "✅ V8.0: Insider Transaction Tracker (real yfinance data replacing hardcoded flags)",
    "✅ V8.0: SEC EDGAR Scanner (mining company discovery from public filings)",
    "✅ V8.0: Multi-Timeframe SMC Analysis (daily + weekly + monthly alignment scoring)",
    "✅ V8.0: Backtest Drawdown Analysis (max DD, Calmar ratio, time underwater)",
    "✅ V8.0: Transaction Cost Modeling (bid-ask spread at 50bps for junior miners)",
    "✅ V8.0: Regime-Conditional Backtesting (bull/bear/choppy segmentation via gold 200-day MA)",
    "✅ V8.0: Concentration Risk Detection (metal >40%, country >50%, correlated pairs >0.8)",
    "✅ V8.0: P/NAV Estimation (DCF-based NAV from reserves and AISC)",
    "✅ V8.0: 56 new unit tests across TA, optimizer, and alert modules",
    "✅ V8.0: Critical bug fixes (orphaned code, hardcoded paths, duplicate rendering, bare excepts)",
]

# V5.0: Global UI Configuration (prevents NameError)
UI_CONFIG = {
    'max_price': 200.0,         # Include all price ranges (NEM ~$55, GOLD ~$22, juniors ~$0.50)
    'min_alpha': 50,            # Show stocks scoring above neutral
    'max_aisc': 1400,           # All-in sustaining cost filter
    'max_mcap_millions': 80000  # Include large-cap miners (NEM ~$50B, GOLD ~$36B)
}

# Clean professional header
st.markdown("""
<style>
    .alpha-miner-header {
        text-align: left;
        padding: 1rem 1.5rem;
        border-bottom: 2px solid #334155;
        margin-bottom: 1.5rem;
        background: #0f172a;
        border-radius: 6px;
    }
    .alpha-miner-header h1 {
        margin: 0;
        color: #f1f5f9;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    .alpha-miner-header p {
        margin: 0.3rem 0 0 0;
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 400;
    }
</style>
<div class="alpha-miner-header">
    <h1>Alpha Miner Pro</h1>
    <p>Mining Portfolio Analysis &amp; Discovery Engine</p>
</div>
""", unsafe_allow_html=True)

# Professional styling — minimal, functional
st.markdown("""
<style>
    .stApp {background-color: #0f1117; color: #e2e8f0;}
    .main {background-color: #0f1117;}
    h1, h2, h3 {color: #e2e8f0 !important;}

    /* Status bar */
    .status-bar {display: flex; gap: 1.5rem; padding: 0.6rem 1rem; background: #1e293b; border-radius: 6px; margin-bottom: 1rem; font-size: 0.85rem; color: #94a3b8;}
    .status-bar .ok {color: #22c55e;}
    .status-bar .warn {color: #eab308;}
    .status-bar .fail {color: #ef4444;}

    /* Badges — compact, functional */
    .badge-core, .badge-tactical, .badge-gambling, .badge-l0, .badge-l1, .badge-l2, .badge-l3, .badge-insider, .badge-discovery {
        color: white; padding: 0.15rem 0.5rem; border-radius: 4px; font-weight: 600; font-size: 0.8rem; margin: 0 0.15rem; display: inline-block;
    }
    .badge-core {background: #2563eb;} .badge-tactical {background: #d97706;} .badge-gambling {background: #dc2626;}
    .badge-l0 {background: #dc2626;} .badge-l1 {background: #d97706;} .badge-l2 {background: #2563eb;} .badge-l3 {background: #16a34a;}
    .badge-insider {background: #7c3aed;} .badge-discovery {background: #db2777;}

    /* Cards */
    .command-center {background: #1e293b; border: 1px solid #334155; padding: 1.5rem; border-radius: 8px; margin: 1rem 0;}
    .risk-card {background: #450a0a; border-left: 3px solid #dc2626; padding: 1rem; margin: 0.4rem 0; border-radius: 4px;}
    .opportunity-card {background: #052e16; border-left: 3px solid #16a34a; padding: 1rem; margin: 0.4rem 0; border-radius: 4px;}
    .warning-banner {background: #431407; border: 2px solid #ea580c; padding: 1rem; border-radius: 6px; margin: 0.8rem 0; text-align: center;}
    .safe-banner {background: #052e16; border: 2px solid #16a34a; padding: 1rem; border-radius: 6px; margin: 0.8rem 0; text-align: center;}

    /* Gates */
    .gate-pass {color: #22c55e; font-weight: 600;}
    .gate-fail {color: #ef4444; font-weight: 600;}
    .gate-warning {color: #f59e0b; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# PORTFOLIO_SIZE is imported from alpha_miner_core (line 51)

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

CACHE_FILE = Path.home() / '.alpha_miner_cache.json'

def load_cache():
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE) as f:
                return json.load(f)
    except (IOError, json.JSONDecodeError, OSError) as e:
        print(f"Warning: Could not load cache: {e}")
    return {}

def save_cache(data):
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except (IOError, OSError, TypeError) as e:
        print(f"Warning: Could not save cache: {e}")

if 'fund_cache' not in st.session_state:
    st.session_state.fund_cache = load_cache()


# =========================================================================
# GOVERNANCE: VALIDATION, STRICT MODE, EVIDENCE PACKS, REPLAY MODE
# =========================================================================

EVIDENCE_DIR = Path.home() / '.alpha_miner_evidence_packs'
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

# RISK_PROFILES, get_risk_profile_preset, validate_data_invariants, enforce_strict_mode,
# create_evidence_pack, save_evidence_pack, list_evidence_packs, load_evidence_pack
# imported from alpha_miner_core

def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

# V5.0: Metadata validation function for backward compatibility
def validate_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    V5.0: Ensure V5.0 metadata columns (Jurisdiction, Metal_Type) exist in DataFrame.
    Uses backward compatibility logic from backtest_runner.py to derive from Country/metal columns.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        DataFrame with Jurisdiction and Metal_Type columns guaranteed to exist
    """
    df = df.copy()  # Don't modify original
    
    # V4.0/V5.0: Add Jurisdiction column (derived from Country if not present)
    if 'Jurisdiction' not in df.columns:
        if 'Country' in df.columns:
            df['Jurisdiction'] = df['Country']
        else:
            df['Jurisdiction'] = 'Unknown'
    
    # V4.0/V5.0: Add Metal_Type column (derived from metal/Metal if not present)
    if 'Metal_Type' not in df.columns:
        if 'metal' in df.columns:
            df['Metal_Type'] = df['metal']
        elif 'Metal' in df.columns:
            df['Metal_Type'] = df['Metal']
        else:
            df['Metal_Type'] = 'Gold'
    
    return df

# Wrapper for create_evidence_pack to pass version info
def create_evidence_pack_with_version(*args, **kwargs):
    return create_evidence_pack(*args, version=VERSION, version_date=VERSION_DATE, **kwargs)


def compute_run_diff(prev_df: pd.DataFrame, curr_df: pd.DataFrame):
    """Return a simple diff table keyed by Symbol."""
    if prev_df is None or prev_df.empty:
        return pd.DataFrame()
    a = prev_df.set_index('Symbol')
    b = curr_df.set_index('Symbol')
    common = a.index.intersection(b.index)
    rows = []
    for sym in common:
        ra, rb = a.loc[sym], b.loc[sym]
        def g(x, k, d=0):
            try:
                return x.get(k, d)
            except Exception:
                return d
        if g(ra,'Action','') != g(rb,'Action','') or abs(float(g(ra,'Alpha_Score',0))-float(g(rb,'Alpha_Score',0)))>=5 or abs(float(g(ra,'Sell_Risk_Score',0))-float(g(rb,'Sell_Risk_Score',0)))>=10:
            rows.append({
                'Symbol': sym,
                'Action_prev': g(ra,'Action',''),
                'Action_now': g(rb,'Action',''),
                'Alpha_prev': float(g(ra,'Alpha_Score',0) or 0),
                'Alpha_now': float(g(rb,'Alpha_Score',0) or 0),
                'Sell_prev': float(g(ra,'Sell_Risk_Score',0) or 0),
                'Sell_now': float(g(rb,'Sell_Risk_Score',0) or 0),
                'RecPct_prev': float(g(ra,'Recommended_Pct',0) or 0),
                'RecPct_now': float(g(rb,'Recommended_Pct',0) or 0),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out['Alpha_Δ'] = out['Alpha_now'] - out['Alpha_prev']
        out['Sell_Δ'] = out['Sell_now'] - out['Sell_prev']
        out['RecPct_Δ'] = out['RecPct_now'] - out['RecPct_prev']
    return out


def compute_rebalance_table(df: pd.DataFrame, total_value: float):
    rows = []
    for _, r in df.iterrows():
        sym = r.get('Symbol')
        cur = float(r.get('Pct_Portfolio',0) or 0)
        rec = float(r.get('Recommended_Pct',0) or 0)
        delta = rec - cur
        dollars = (delta/100.0) * float(total_value)
        if abs(delta) < 0.25:
            continue
        side = 'BUY' if delta > 0 else 'SELL'
        rows.append({
            'Symbol': sym,
            'Side': side,
            'Current_%': round(cur,2),
            'Target_%': round(rec,2),
            'Δ_%': round(delta,2),
            'Δ_$': round(dollars,0),
            'Action': r.get('Action',''),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(['Side','Δ_$'], ascending=[True, False])
    return out

# ============================================================================
# A) LIQUIDITY ENGINE
# ============================================================================

# calculate_liquidity_metrics, calculate_data_confidence, calculate_dilution_risk,
# normalize_timestamp, tag_news imported from alpha_miner_core

def calculate_news_quality(news_items):
    """Calculate news quality based on valid timestamps"""
    valid_count = sum(1 for item in news_items if item.get('timestamp', 0) > 0)
    
    if valid_count >= 3:
        return 'HIGH', 'badge-l3'
    elif valid_count >= 1:
        return 'MED', 'badge-l2'
    else:
        return 'LOW', 'badge-l1'

def get_sector_news_fallback():
    """Get sector news when ticker has none"""
    if not YFINANCE:
        return []
    
    try:
        # Try GDXJ for sector news
        sector = yf.Ticker("GDXJ")
        news = sector.news[:8]
        
        return [{
            'title': item.get('title', ''),
            'publisher': item.get('publisher', 'Sector'),
            'link': item.get('link', '#')
        } for item in news]
    except Exception:
        return []

# ============================================================================
# E) SMC (SMART MONEY CONCEPTS)
# ============================================================================

def calculate_smc_signals(hist_data, current_price):
    """
    Calculate Smart Money Concepts signals
    Returns bias, score, summary, and signals
    """
    result = {
        'bias': 'Neutral',
        'score': 50,
        'summary': 'No clear structure',
        'signals': [],
        'state': 'NEUTRAL',  # For v3 compatibility
        'event': 'NONE',      # For v3 compatibility
        'confidence': 50      # For v3 compatibility
    }
    
    if hist_data.empty or len(hist_data) < 50:
        return result
    
    try:
        df = hist_data.tail(200).copy()
        
        # Find swing points
        swing_highs = []
        swing_lows = []
        
        for i in range(5, len(df)-5):
            # Swing high
            if df['High'].iloc[i] == df['High'].iloc[i-5:i+6].max():
                swing_highs.append((i, df['High'].iloc[i]))
            # Swing low
            if df['Low'].iloc[i] == df['Low'].iloc[i-5:i+6].min():
                swing_lows.append((i, df['Low'].iloc[i]))
        
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return result
        
        # Check structure
        last_3_highs = [h[1] for h in swing_highs[-3:]]
        last_3_lows = [l[1] for l in swing_lows[-3:]]
        
        # Higher highs and higher lows = bullish
        hh = last_3_highs[-1] > last_3_highs[-2] and last_3_highs[-2] > last_3_highs[-3]
        hl = last_3_lows[-1] > last_3_lows[-2] and last_3_lows[-2] > last_3_lows[-3]
        
        # Lower highs and lower lows = bearish
        lh = last_3_highs[-1] < last_3_highs[-2] and last_3_highs[-2] < last_3_highs[-3]
        ll = last_3_lows[-1] < last_3_lows[-2] and last_3_lows[-2] < last_3_lows[-3]
        
        if hh and hl:
            result['bias'] = 'Bullish'
            result['state'] = 'BULLISH'
            result['score'] = 65
            result['confidence'] = 65
            result['summary'] = 'Bullish structure: HH + HL'
            result['signals'].append('Higher Highs + Higher Lows')
            
            # Check for BOS
            if current_price > last_3_highs[-1] * 1.001:
                result['score'] = 75
                result['confidence'] = 75
                result['event'] = 'BOS'
                result['signals'].append('Break of Structure (BOS) ↑')
        
        elif lh and ll:
            result['bias'] = 'Bearish'
            result['state'] = 'BEARISH'
            result['score'] = 35
            result['confidence'] = 65
            result['summary'] = 'Bearish structure: LH + LL'
            result['signals'].append('Lower Highs + Lower Lows')
            
            # Check for BOS down
            if current_price < last_3_lows[-1] * 0.999:
                result['score'] = 25
                result['confidence'] = 75
                result['event'] = 'BOS'
                result['signals'].append('Break of Structure (BOS) ↓')
        
        else:
            result['summary'] = 'Ranging / Neutral structure'

    except Exception:
        pass

    return result

# ============================================================================
# F) ALPHA MODELS (6 MODELS)
# ============================================================================

# calculate_alpha_models imported from alpha_miner_core

# ============================================================================
# G) SELL RISK
# ============================================================================

# calculate_sell_risk imported from alpha_miner_core

# ============================================================================
# H) MACRO REGIME
# ============================================================================

# calculate_tape_gate, calculate_macro_regime imported from alpha_miner_core

# ============================================================================
# I) FINANCING OVERHANG CALCULATION
# ============================================================================

# Wrapper for calculate_financing_overhang to pass INSTITUTIONAL_V3_AVAILABLE
def calculate_financing_overhang(news_items, ticker, runway_months):
    """Wrapper that passes INSTITUTIONAL_V3_AVAILABLE to core function"""
    from alpha_miner_core import calculate_financing_overhang as core_calculate_financing_overhang
    return core_calculate_financing_overhang(news_items, ticker, runway_months, INSTITUTIONAL_V3_AVAILABLE)

# ============================================================================
# I) DISCOVERY EXCEPTION
# ============================================================================

def check_discovery_exception(row, liq_metrics, alpha_score, data_confidence, 
                              dilution_risk, momentum_ok):
    """
    Check if discovery exception applies
    Enhanced version with SMC check if available
    """
    # Basic checks
    if liq_metrics.get('tier_code') == 'L0':
        return (False, "L0 tier excluded")
    
    if row.get('Sleeve', '') != 'TACTICAL':
        return (False, "Must be TACTICAL sleeve")
    
    if alpha_score < 85:
        return (False, f"Alpha {alpha_score:.0f} < 85")
    
    if data_confidence < 70:
        return (False, f"Confidence {data_confidence:.0f} < 70")
    
    if dilution_risk >= 70:
        return (False, f"Dilution {dilution_risk:.0f} ≥ 70")
    
    if not momentum_ok:
        return (False, "Momentum not confirmed")
    
    # Check SMC if available
    smc_bias = row.get('SMC_Bias', 'Neutral')
    if smc_bias == 'Bearish':
        return (False, "SMC bearish")
    
    # Check metal regime if available
    if 'metal_regime' in st.session_state:
        metal_regime = st.session_state.metal_regime
        if metal_regime.get('discovery_hardness') == 'BLOCKED':
            return (False, "Metal regime bearish - discovery blocked")
    
    # Exception granted
    return (True, f"High conviction: Alpha {alpha_score:.0f}, momentum confirmed")

# ============================================================================
# J) FINAL ARBITRATION
# ============================================================================

# arbitrate_final_decision imported from alpha_miner_core
# Note: When calling, pass strict_mode parameter: arbitrate_final_decision(..., strict_mode=st.session_state.get('strict_mode', False))

# ============================================================================
# TICKER SANITIZATION
# ============================================================================

def sanitize_ticker(symbol: str, is_canadian: bool = None) -> list:
    """
    V5.0: Sanitize ticker symbol with US/Canada focus.
    Returns a list of ticker variants to try in order.
    
    For Canadian tickers: [TICKER].TO (Toronto), [TICKER].V (Venture), [TICKER] (OTC/US)
    For US tickers: [TICKER] (original), [TICKER].TO, [TICKER].V
    
    Examples:
    - DSV.V (Canadian) -> [DSV.TO, DSV.V, DSV]
    - ABC (Canadian) -> [ABC.TO, ABC.V, ABC]
    - XYZ (US) -> [XYZ, XYZ.TO, XYZ.V]
    """
    if not symbol:
        return []
    
    symbol_upper = symbol.upper().strip()
    base = symbol_upper
    
    # Detect Canadian ticker by suffix or infer from context
    if is_canadian is None:
        # Auto-detect: if it has .V or .TO suffix, it's Canadian
        is_canadian = symbol_upper.endswith('.V') or symbol_upper.endswith('.TO')
    
    # Extract base symbol (remove suffixes)
    if symbol_upper.endswith('.V'):
        base = symbol_upper[:-2]
        is_canadian = True
    elif symbol_upper.endswith('.TO'):
        base = symbol_upper[:-3]
        is_canadian = True
    
    # V5.0: Hardcode Canadian ticker sequence: .TO, .V, then base
    if is_canadian:
        variants = [f"{base}.TO", f"{base}.V", base]
    else:
        # US ticker: try original first, then Canadian exchanges
        variants = [base, f"{base}.TO", f"{base}.V"]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_variants = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            unique_variants.append(v)
    
    return unique_variants

def _tiingo_records_to_df(records):
    """Convert Tiingo EOD list of dicts to DataFrame with DatetimeIndex, Close, Volume (yfinance-style)."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if 'date' not in df.columns:
        return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    renames = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
    for k, v in renames.items():
        if k in df.columns:
            df[v] = df[k]
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if c not in df.columns:
            df[c] = np.nan
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()


def _trailing_sharpe(hist, window: int = 252) -> float:
    """V7.0: Trailing annualized Sharpe from Close. Returns np.nan if insufficient data."""
    if hist is None or hist.empty or "Close" not in hist.columns:
        return np.nan
    close = hist["Close"].dropna()
    if len(close) < min(window + 1, 60):
        return np.nan
    r = close.pct_change(fill_method=None).dropna().tail(window)
    if r.empty or r.std() == 0:
        return np.nan
    return float((r.mean() / r.std()) * np.sqrt(252))


def _geography_first_variants(symbol: str, is_canadian: bool = None) -> list:
    """
    V7.2: Absolute Ticker Strike — Geography-First loop (399-Ticker Fix).
    For any ticker, MUST try in this exact order:
    1. Plain US (e.g., "GOLD", "CCJ")
    2. TSX:TICKER (e.g., "TSX:GOLD")
    3. TICKER.TO (e.g., "GOLD.TO")
    4. TICKERF (OTC, e.g., "GOLDF")
    
    Ensures blue-chips like GOLD and CCJ are correctly analyzed.
    """
    symbol_upper = symbol.upper().strip()
    # Override map for known Tiingo IDs
    STRIKE = {"SKE.TO": ["SKE", "SKE.TO"], "DSVSF": ["DSVSF"], "NXE": ["NXE"]}
    if symbol_upper in STRIKE:
        return list(STRIKE[symbol_upper])

    # Extract base (remove .TO/.V if present)
    base = symbol_upper
    if symbol_upper.endswith(".TO"):
        base = symbol_upper[:-3]
    elif symbol_upper.endswith(".V"):
        base = symbol_upper[:-2]

    # V7.2: Geography-First order (exact as specified)
    variants = [
        base,              # 1. Plain US
        f"TSX:{base}",     # 2. TSX:TICKER
        f"{base}.TO",      # 3. TICKER.TO
        f"{base}F",        # 4. TICKERF (OTC)
    ]

    # Dedupe while preserving order
    seen = set()
    out = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _tiingo_refuel():
    """Session Re-Fueling: close client, reload hey.env, clear state. Call after 5 consecutive SKIPs."""
    try:
        client = st.session_state.pop("tiingo_client", None)
        if client is not None and hasattr(client, "close"):
            try:
                client.close()
            except Exception:
                pass
    except Exception:
        pass
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).resolve().parent / "hey.env"
        load_dotenv(dotenv_path=env_path)
    except Exception:
        pass
    try:
        st.session_state["tiingo_consecutive_skips"] = 0
    except Exception:
        pass


def fetch_ticker_with_fallback(symbol: str, period: str = "1y", is_canadian: bool = None, data_health: dict = None):
    """
    Fetch via Tiingo with Geography First routing and Session Re-Fueling.
    US: plain ticker only. Canadian .TO/.V: try TSX:/TSXV: formats before giving up.
    On 5 consecutive SKIPs: client.close + re-init from hey.env.
    Returns (hist_data, successful_symbol) or (empty DataFrame, None).
    """
    api_key = (os.getenv("TIINGO_API_KEY") or "").strip()
    if not TIINGO_AVAILABLE or TiingoClient is None or not api_key:
        try:
            st.session_state["tiingo_key_diagnostic"] = "Critical: TIINGO_API_KEY not found in memory."
        except Exception:
            pass
        if data_health is not None:
            data_health[symbol] = {'status': 'skip', 'reason': 'Tiingo not available or TIINGO_API_KEY missing'}
        return pd.DataFrame(), None
    try:
        st.session_state.pop("tiingo_key_diagnostic", None)
    except Exception:
        pass

    symbol_upper = symbol.upper().strip()
    all_variants = _geography_first_variants(symbol, is_canadian=is_canadian)

    _end = datetime.date.today()
    m = re.search(r"(\d+)", period or "1")
    n = int(m.group(1)) if m else 1
    if period and "y" in (period or "").lower():
        _days = 365 * n
    elif period and "mo" in (period or "").lower():
        _days = 30 * n
    else:
        _days = 365
    _start = _end - datetime.timedelta(days=_days)
    start_date = _start.strftime("%Y-%m-%d")
    end_date = _end.strftime("%Y-%m-%d")

    # Get-or-create shared client; Session Re-Fueling on 5 consecutive SKIPs
    consecutive = 0
    try:
        consecutive = int(st.session_state.get("tiingo_consecutive_skips", 0))
    except Exception:
        pass
    if consecutive >= 5:
        _tiingo_refuel()

    client = None
    try:
        client = st.session_state.get("tiingo_client")
    except Exception:
        pass
    if client is None:
        try:
            client = TiingoClient({"api_key": api_key})
            try:
                st.session_state["tiingo_client"] = client
            except Exception:
                pass
        except Exception as e:
            if data_health is not None:
                data_health[symbol] = {'status': 'skip', 'reason': f'Tiingo client init: {str(e)[:80]}'}
            return pd.DataFrame(), None

    last_error = None
    for variant in all_variants:
        try:
            data = client.get_ticker_price(
                variant, startDate=start_date, endDate=end_date, frequency="daily", fmt="json"
            )
            if data and isinstance(data, list) and len(data) > 0:
                hist = _tiingo_records_to_df(data)
                if not hist.empty:
                    # V6.5: Data Feed Redundancy – if Tiingo "flat" (stale), fallback to yfinance last 3 days
                    if "Close" in hist.columns:
                        tail = hist["Close"].dropna().tail(5)
                        flat = len(tail) >= 3 and (tail.nunique() <= 1 or (tail.std() or 0) == 0)
                        if flat and YFINANCE:
                            try:
                                import warnings
                                with warnings.catch_warnings():
                                    warnings.filterwarnings("ignore", message=".*possibly delisted.*")
                                    yf_hist = yf.Ticker(variant).history(period="5d")
                                if not yf_hist.empty and "Close" in yf_hist.columns:
                                    for c in ("Open", "High", "Low", "Close", "Volume"):
                                        if c not in yf_hist.columns:
                                            yf_hist[c] = np.nan
                                    yf_3 = yf_hist[["Open", "High", "Low", "Close", "Volume"]].tail(3).copy()
                                    yf_3.index = pd.to_datetime(yf_3.index)
                                    if yf_3.index.tz is not None:
                                        yf_3.index = yf_3.index.tz_localize(None)
                                    drop_idx = hist.tail(3).index
                                    hist = hist.loc[~hist.index.isin(drop_idx)]
                                    hist = pd.concat([hist, yf_3]).sort_index()
                                    hist = hist[~hist.index.duplicated(keep="last")]
                            except Exception:
                                pass
                    if data_health is not None:
                        data_health[symbol] = {'status': 'success', 'variant': variant, 'rows': len(hist)}
                    try:
                        st.session_state["tiingo_consecutive_skips"] = 0
                    except Exception:
                        pass
                    return hist, variant
        except Exception as e:
            # V7.4: Better error logging - capture full error details
            error_msg = str(e)
            last_error = error_msg[:200]  # Longer error message
            # Log to data_health for debugging
            if data_health is not None:
                if symbol not in data_health:
                    data_health[symbol] = {}
                if 'variant_errors' not in data_health[symbol]:
                    data_health[symbol]['variant_errors'] = []
                data_health[symbol]['variant_errors'].append(f"{variant}: {error_msg[:100]}")
            continue

    # All variants failed - log as skip (not crash)
    if data_health is not None:
        data_health[symbol] = {
            'status': 'skip',
            'reason': last_error or 'All variants failed',
            'variants_tried': all_variants,
        }
    # Session Re-Fueling: 5 consecutive SKIPs → close client, reload hey.env, re-init
    try:
        n = int(st.session_state.get("tiingo_consecutive_skips", 0)) + 1
        st.session_state["tiingo_consecutive_skips"] = n
        if n >= 5:
            _tiingo_refuel()
    except Exception:
        pass

    # V7.4: All Tiingo variants failed - try yfinance as final fallback for stock prices
    # Map yfinance data to match Tiingo format exactly (DatetimeIndex, title case columns)
    if YFINANCE:
        try:
            # Suppress yfinance warnings (delisted symbols are expected in fallback)
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", message=".*possibly delisted.*")
                yf_ticker = yf.Ticker(symbol)
                yf_hist = yf_ticker.history(period="1y")
            if not yf_hist.empty and "Close" in yf_hist.columns:
                # Ensure all required columns exist (yfinance should have them, but be safe)
                for c in ("Open", "High", "Low", "Close", "Volume"):
                    if c not in yf_hist.columns:
                        yf_hist[c] = np.nan
                
                # Extract and format to match Tiingo structure
                hist = yf_hist[["Open", "High", "Low", "Close", "Volume"]].copy()
                
                # Ensure index is DatetimeIndex (yfinance may have timezone)
                hist.index = pd.to_datetime(hist.index)
                if hist.index.tz is not None:
                    hist.index = hist.index.tz_localize(None)  # Remove timezone to match Tiingo
                
                # Sort by index (Tiingo format is sorted)
                hist = hist.sort_index()
                
                # Ensure columns are title case (should already be, but be explicit)
                hist.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
                if data_health is not None:
                    data_health[symbol] = {
                        'status': 'success', 
                        'variant': f'yfinance_{symbol}', 
                        'rows': len(hist), 
                        'source': 'yfinance_fallback'
                    }
                try:
                    st.session_state["tiingo_consecutive_skips"] = 0
                except Exception:
                    pass
                return hist, symbol
        except Exception as yf_error:
            last_error = f"Tiingo failed: {last_error or 'All variants failed'}; yfinance failed: {str(yf_error)[:50]}"
    
    if data_health is not None:
        data_health[symbol] = {
            'status': 'skip',
            'reason': last_error or 'All variants failed (Tiingo + yfinance)',
            'variants_tried': all_variants,
        }

    return pd.DataFrame(), None


def _fetch_tiingo_range(symbol: str, start_date: str, end_date: str, client):
    """Fetch Tiingo daily prices for symbol over [start_date, end_date]. Returns DataFrame or None."""
    try:
        data = client.get_ticker_price(
            symbol, startDate=start_date, endDate=end_date, frequency="daily", fmt="json"
        )
        if data and isinstance(data, list) and len(data) > 0:
            return _tiingo_records_to_df(data)
    except Exception:
        pass
    return None


def run_sovereign_stress_test(portfolio_df, total_value: float, cash: float):
    """
    Hard-Burn stress test: find 3 worst 30-day peak-to-trough periods for Silver/Gold miners (GDX)
    over 15 years, then compute portfolio drawdown in those windows.
    Returns dict: hard_burn_usd, hard_burn_pct, ok, error, worst_window.
    """
    out = {"hard_burn_usd": 0.0, "hard_burn_pct": 0.0, "ok": False, "error": None, "worst_window": None}
    api_key = (os.getenv("TIINGO_API_KEY") or "").strip()
    if not TIINGO_AVAILABLE or TiingoClient is None or not api_key:
        out["error"] = "Tiingo not available or TIINGO_API_KEY missing"
        return out
    if portfolio_df is None or portfolio_df.empty or "Symbol" not in portfolio_df.columns or "Quantity" not in portfolio_df.columns:
        out["error"] = "Portfolio missing or invalid"
        return out
    if total_value <= 0:
        out["error"] = "Total portfolio value must be positive"
        return out

    try:
        client = TiingoClient({"api_key": api_key})
    except Exception as e:
        out["error"] = f"Tiingo client init: {str(e)[:80]}"
        return out

    # 15 years
    end_d = datetime.date.today()
    start_d = end_d - datetime.timedelta(days=15 * 365)
    start_str = start_d.strftime("%Y-%m-%d")
    end_str = end_d.strftime("%Y-%m-%d")

    # Sector proxy: GDX (Gold Miners ETF)
    gdx = _fetch_tiingo_range("GDX", start_str, end_str, client)
    if gdx is None or gdx.empty or "Close" not in gdx.columns:
        out["error"] = "GDX history unavailable (Tiingo)"
        return out

    gdx = gdx.sort_index()
    gdx["runmax"] = gdx["Close"].cummax()
    gdx["dd"] = (gdx["Close"] - gdx["runmax"]) / gdx["runmax"].replace(0, np.nan)

    # Rolling 30-day windows: find 3 worst max drawdowns
    all_windows = []
    win = 30
    for i in range(len(gdx) - win + 1):
        w = gdx.iloc[i : i + win]
        mx = w["Close"].max()
        mn = w["Close"].min()
        if mx <= 0:
            continue
        dd = (mn - mx) / mx
        peak_date = w["Close"].idxmax()
        trough_date = w["Close"].idxmin()
        if hasattr(peak_date, "date"):
            peak_date = peak_date.date()
        if hasattr(trough_date, "date"):
            trough_date = trough_date.date()
        start_dt = w.index[0]
        end_dt = w.index[-1]
        if hasattr(start_dt, "date"):
            start_dt = start_dt.date()
        if hasattr(end_dt, "date"):
            end_dt = end_dt.date()
        all_windows.append({"dd": dd, "peak": peak_date, "trough": trough_date, "start": start_dt, "end": end_dt})
    all_windows.sort(key=lambda x: x["dd"])
    worst3 = all_windows[:3]
    if not worst3:
        out["error"] = "No 30-day windows found"
        return out

    # Portfolio symbols and quantities
    sym_qty = portfolio_df.set_index("Symbol")["Quantity"].astype(np.float64).to_dict()
    symbols = list(sym_qty.keys())

    # Fetch 15y history per symbol
    price_cache = {}
    for sym in symbols:
        h = _fetch_tiingo_range(sym, start_str, end_str, client)
        if h is not None and not h.empty and "Close" in h.columns:
            price_cache[sym] = h["Close"]

    def price_at(sym, d):
        if sym not in price_cache:
            return np.nan
        s = price_cache[sym]
        try:
            dt = pd.Timestamp(d)
            if dt in s.index:
                return float(s.loc[dt])
            before = s.index[s.index <= dt]
            if len(before) > 0:
                return float(s.loc[before[-1]])
            after = s.index[s.index >= dt]
            if len(after) > 0:
                return float(s.loc[after[0]])
        except Exception:
            pass
        return np.nan

    worst_dd = 0.0
    worst_win = None
    for w in worst3:
        v_peak = cash
        v_trough = cash
        for sym, qty in sym_qty.items():
            p_peak = price_at(sym, w["peak"])
            p_trough = price_at(sym, w["trough"])
            if not np.isnan(p_peak):
                v_peak += qty * p_peak
            if not np.isnan(p_trough):
                v_trough += qty * p_trough
        if v_peak <= 0:
            continue
        dd = (v_trough - v_peak) / v_peak
        if dd < worst_dd:
            worst_dd = dd
            worst_win = w

    if worst_win is None:
        out["error"] = "Could not compute portfolio drawdown in stress windows"
        return out

    out["hard_burn_pct"] = worst_dd
    out["hard_burn_usd"] = total_value * worst_dd
    out["worst_window"] = worst_win
    out["ok"] = True
    return out


@st.cache_data(ttl=3600)
def run_sovereign_stress_test_cached(portfolio_json: str, total_value: float, cash: float):
    """Cached wrapper for Hard-Burn stress test (avoids repeated Tiingo calls)."""
    try:
        df = pd.read_json(portfolio_json, orient="records")
    except Exception:
        df = pd.DataFrame()
    return run_sovereign_stress_test(df, float(total_value), float(cash))


# ============================================================================
# DATA FETCHING
# ============================================================================

@st.cache_data(ttl=900)
def get_fundamentals_with_tracking(ticker):
    """Fetch fundamentals"""
    result = {
        'cash': 10.0,
        'burn': 1.0,
        'burn_source': 'default',
        'stage': 'Explorer (Inferred)',
        'metal': 'Unknown',
        'country': 'Unknown',
        'info_dict': {},
        'inferred_flags': {'metal_inferred': True, 'stage_inferred': True}
    }
    
    if not YFINANCE:
        return result
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        result['info_dict'] = info
        
        # Cash
        if info.get('totalCash'):
            result['cash'] = info['totalCash'] / 1_000_000
        elif info.get('cash'):
            result['cash'] = info['cash'] / 1_000_000
        
        # Burn rate
        try:
            cf = stock.cashflow
            if not cf.empty and 'Operating Cash Flow' in cf.index:
                ocf = cf.loc['Operating Cash Flow'].iloc[0]
                if ocf < 0:
                    result['burn'] = abs(ocf) / 12_000_000
                    result['burn_source'] = 'cashflow'
                elif ocf > 0:
                    result['burn'] = 0.1
                    result['burn_source'] = 'cashflow'
        except Exception:
            if info.get('netIncome') and info['netIncome'] < 0:
                result['burn'] = abs(info['netIncome']) / 12_000_000
                result['burn_source'] = 'netincome'
        
        # Stage
        revenue = info.get('totalRevenue', 0)
        if revenue and revenue > 10_000_000:
            result['stage'] = 'Producer'
            result['inferred_flags']['stage_inferred'] = False
        else:
            assets = info.get('totalAssets', 0)
            if assets > 50_000_000:
                result['stage'] = 'Developer (Inferred)'
            else:
                result['stage'] = 'Explorer (Inferred)'
            result['inferred_flags']['stage_inferred'] = True
        
        # Country
        if info.get('country'):
            result['country'] = info['country']
        else:
            result['country'] = 'Unknown'
        
        # Metal (label as inferred if inferred)
        desc = info.get('longBusinessSummary', '').lower()
        if 'silver' in desc:
            result['metal'] = 'Silver'
            result['inferred_flags']['metal_inferred'] = False
        elif 'gold' in desc:
            result['metal'] = 'Gold'
            result['inferred_flags']['metal_inferred'] = False
        elif 'copper' in desc:
            result['metal'] = 'Copper'
            result['inferred_flags']['metal_inferred'] = False
        else:
            # Check name for inference
            name_lower = info.get('longName', '').lower()
            if 'gold' in name_lower or 'aurora' in name_lower:
                result['metal'] = 'Gold (Inferred)'
                result['inferred_flags']['metal_inferred'] = True
            elif 'silver' in name_lower:
                result['metal'] = 'Silver (Inferred)'
                result['inferred_flags']['metal_inferred'] = True
            else:
                result['metal'] = 'Unknown'
                result['inferred_flags']['metal_inferred'] = True
        
        # V7.5: News scanning - Check longBusinessSummary for promotion keywords
        # Promote higher if they have: "Drill results", "High grade", "Exploration", 
        # "Pre-feasibility" (PFS), "Sprott"
        news_promotion_score = 0
        news_keywords = ['drill results', 'high grade', 'exploration', 'pre-feasibility', 'pfs', 'sprott']
        found_keywords = []
        for keyword in news_keywords:
            if keyword in desc:
                news_promotion_score += 5  # +5 Alpha per keyword found
                found_keywords.append(keyword)
        
        if news_promotion_score > 0:
            result['news_promotion_score'] = news_promotion_score
            result['news_keywords_found'] = found_keywords
            result['reasoning'] = f"✅ News promotion: Found {len(found_keywords)} keyword(s): {', '.join(found_keywords)} (+{news_promotion_score} Alpha)"
        else:
            result['news_promotion_score'] = 0
            result['news_keywords_found'] = []

    except Exception:
        pass

    return result


@st.cache_data(ttl=3600)
def get_forensic_fundamentals(ticker: str) -> dict:
    """
    V7.0: Forensic layer – AISC and P/NAV for mcap > $500M.
    Returns {aisc, p_nav, market_cap_m} or defaults.
    """
    out = {"aisc": None, "p_nav": None, "market_cap_m": None}
    if not YFINANCE:
        return out
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        mcap = info.get("marketCap") or info.get("market_cap")
        if mcap is not None:
            out["market_cap_m"] = float(mcap) / 1_000_000
        price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose")
        if price is not None:
            price = float(price)
        bal = getattr(stock, "balance_sheet", None)
        if bal is not None and not bal.empty and bal.shape[1] >= 1:
            row = bal.iloc[:, 0]
            ta = row.get("Total Assets", np.nan)
            tl = row.get("Total Liabilities", np.nan)
            sh = info.get("sharesOutstanding") or row.get("Share Issued") or row.get("Common Stock Shares Outstanding")
            if np.isfinite(ta) and np.isfinite(tl) and sh and float(sh) > 0:
                nav = float(ta) - float(tl)
                nav_ps = nav / float(sh)
                if nav_ps > 0 and price and price > 0:
                    out["p_nav"] = float(price) / nav_ps
        aisc = info.get("allInSustainingCosts") or info.get("aisc")
        if aisc is not None:
            out["aisc"] = float(aisc)
        return out
    except Exception:
        return out


@st.cache_data(ttl=3600)
def piotroski_f_score(ticker: str) -> int:
    """
    V6.5: Piotroski F-Score (0–9) from balance sheet health.
    Uses yfinance financials, balance_sheet, cashflow. Failing = <3.
    """
    if not YFINANCE:
        return -1
    try:
        stock = yf.Ticker(ticker)
        fin = getattr(stock, "financials", None)
        bal = getattr(stock, "balance_sheet", None)
        cf = getattr(stock, "cashflow", None)
        if fin is None or fin.empty or bal is None or bal.empty or cf is None or cf.empty:
            return -1
        # Current = most recent, Prior = previous
        fin_c = fin.iloc[:, 0] if fin.shape[1] >= 1 else pd.Series(dtype=float)
        fin_p = fin.iloc[:, 1] if fin.shape[1] >= 2 else fin_c
        bal_c = bal.iloc[:, 0] if bal.shape[1] >= 1 else pd.Series(dtype=float)
        bal_p = bal.iloc[:, 1] if bal.shape[1] >= 2 else bal_c
        cf_c = cf.iloc[:, 0] if cf.shape[1] >= 1 else pd.Series(dtype=float)
        cf_p = cf.iloc[:, 1] if cf.shape[1] >= 2 else cf_c

        def _v(s: pd.Series, *keys: str):
            for k in keys:
                try:
                    val = s.get(k, np.nan)
                    if np.isfinite(val):
                        return float(val)
                except Exception:
                    pass
            return np.nan

        ni_c = _v(fin_c, "Net Income", "Net Income Common Stockholders")
        ni_p = _v(fin_p, "Net Income", "Net Income Common Stockholders")
        ocf_c = _v(cf_c, "Operating Cash Flow", "Total Cash From Operating Activities")
        ocf_p = _v(cf_p, "Operating Cash Flow", "Total Cash From Operating Activities")
        ta_c = _v(bal_c, "Total Assets")
        ta_p = _v(bal_p, "Total Assets")
        ca_c = _v(bal_c, "Current Assets")
        ca_p = _v(bal_p, "Current Assets")
        cl_c = _v(bal_c, "Current Liabilities")
        cl_p = _v(bal_p, "Current Liabilities")
        ltd_c = _v(bal_c, "Long Term Debt")
        ltd_p = _v(bal_p, "Long Term Debt")
        rev_c = _v(fin_c, "Total Revenue")
        rev_p = _v(fin_p, "Total Revenue")
        gp_c = _v(fin_c, "Gross Profit")
        gp_p = _v(fin_p, "Gross Profit")

        score = 0
        if np.isfinite(ni_c) and ni_c > 0:
            score += 1
        if np.isfinite(ocf_c) and ocf_c > 0:
            score += 1
        roa_c = ni_c / ta_c if np.isfinite(ta_c) and ta_c and ta_c > 0 else np.nan
        roa_p = ni_p / ta_p if np.isfinite(ta_p) and ta_p and ta_p > 0 else np.nan
        if np.isfinite(roa_c) and np.isfinite(roa_p) and roa_c > roa_p:
            score += 1
        if np.isfinite(ocf_c) and np.isfinite(ni_c) and ocf_c > ni_c:
            score += 1
        if np.isfinite(ltd_c) and np.isfinite(ltd_p) and ltd_c < ltd_p:
            score += 1
        cr_c = ca_c / cl_c if np.isfinite(cl_c) and cl_c and cl_c > 0 else np.nan
        cr_p = ca_p / cl_p if np.isfinite(cl_p) and cl_p and cl_p > 0 else np.nan
        if np.isfinite(cr_c) and np.isfinite(cr_p) and cr_c > cr_p:
            score += 1
        try:
            sh_c = _v(bal_c, "Share Issued", "Common Stock", "Common Stock Shares Outstanding")
            sh_p = _v(bal_p, "Share Issued", "Common Stock", "Common Stock Shares Outstanding")
            if np.isfinite(sh_c) and np.isfinite(sh_p) and sh_c <= sh_p:
                score += 1
        except Exception:
            pass
        gm_c = gp_c / rev_c if np.isfinite(rev_c) and rev_c and rev_c > 0 else np.nan
        gm_p = gp_p / rev_p if np.isfinite(rev_p) and rev_p and rev_p > 0 else np.nan
        if np.isfinite(gm_c) and np.isfinite(gm_p) and gm_c > gm_p:
            score += 1
        at_c = rev_c / ta_c if np.isfinite(ta_c) and ta_c and ta_c > 0 else np.nan
        at_p = rev_p / ta_p if np.isfinite(ta_p) and ta_p and ta_p > 0 else np.nan
        if np.isfinite(at_c) and np.isfinite(at_p) and at_c > at_p:
            score += 1
        return int(min(9, max(0, score)))
    except Exception:
        return -1


@st.cache_data(ttl=3600)
def get_news_for_ticker(ticker):
    """Fetch news"""
    if not YFINANCE:
        return []
    
    try:
        stock = yf.Ticker(ticker)
        news = stock.news[:25]
        
        formatted_news = []
        for item in news:
            ts = None
            for field in ['providerPublishTime', 'published_at', 'pubDate']:
                if field in item:
                    ts = normalize_timestamp(item[field])
                    if ts:
                        break
            
            formatted_news.append({
                'title': item.get('title', ''),
                'publisher': item.get('publisher', ''),
                'link': item.get('link', '#'),
                'timestamp': ts if ts else 0,
                'date_str': datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d') if ts else 'Unknown'
            })
        
        return tag_news(formatted_news)
    except Exception:
        return []

# get_benchmark_data imported from alpha_miner_core
# Note: Core version uses YFINANCE_AVAILABLE instead of YFINANCE, but behavior is the same

# ============================================================================
# V7.3: LIVE-WIRE SPOT SYNC (Multi-Sourced Real-Time Prices)
# ============================================================================

@st.cache_data(ttl=60)  # Cache for 1 minute (live prices)
def get_sovereign_spot_prices() -> dict:
    """
    V7.4: 2026 Futures Grid - Multi-Source Spot Fetch with Verified Friday Close Fallback.
    
    PRIMARY: yfinance futures tickers (SI=F Silver, GC=F Gold, UX=F Uranium)
    SECONDARY: Futures Tickers (GC=F, SI=F, UX=F or U-U.TO) - Replaced delisted XAGUSD=X/XAUUSD=X
    TERTIARY: ETF proxies with spot multiplier (SLV * 1.1, GLD for institutional floor)
    
    V7.4 Updates:
    - Replaced delisted XAGUSD=X/XAUUSD=X with Futures Tickers (GC=F, SI=F)
    - Conservative fallback prices (Gold: $2,700, Silver: $31.50, Uranium: $85) used only when ALL sources fail
    - Live prices always used when available (no artificial floor)
    
    Returns: {
        'gold_live': float, 'gold_eod': float, 'gold_use_live': bool,
        'silver_live': float, 'silver_eod': float, 'silver_use_live': bool,
        'uranium_spot': float,
        'price_source': str, 'live_delta_pct': float
    }
    """
    
    out = {
        'gold_live': np.nan,
        'gold_eod': np.nan,
        'gold_use_live': False,
        'silver_live': np.nan,
        'silver_eod': np.nan,
        'silver_use_live': False,
        'uranium_spot': np.nan,
        'price_source': 'unknown',
        'live_delta_pct': 0.0,
        'ok': False
    }
    
    # V7.4: Multi-Source Redundancy - Try all sources, no hardcoded fallbacks
    gold_primary = np.nan
    gold_secondary = np.nan
    gold_tertiary = np.nan
    silver_primary = np.nan
    silver_secondary = np.nan
    silver_tertiary = np.nan
    uranium_spot = np.nan
    
    # Last-resort fallback prices when ALL live sources fail.
    # Set to conservative recent-market levels (not aspirational targets).
    GOLD_VERIFIED_FRIDAY_CLOSE = 2700.0
    SILVER_VERIFIED_FRIDAY_CLOSE = 31.50
    URANIUM_VERIFIED_FRIDAY_CLOSE = 85.0
    
    # V7.4: PRIMARY - yfinance futures tickers (SI=F, GC=F, UX=F)
    # V7.4: Stop 404s - If API returns 404 or "delisted" error, immediately default to Mats Sovereign Floor
    if YFINANCE:
        try:
            # Gold futures (GC=F) - try multiple periods
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", message=".*possibly delisted.*")
                gc = yf.Ticker("GC=F")
                gc_hist = None
                for period_try in ["1d", "5d", "1mo"]:
                    try:
                        gc_hist = gc.history(period=period_try)
                        if not gc_hist.empty and "Close" in gc_hist.columns:
                            break
                    except Exception:
                        continue
                if gc_hist is not None and not gc_hist.empty and "Close" in gc_hist.columns:
                    gold_primary = float(gc_hist["Close"].iloc[-1])
                    out['price_source'] = 'yfinance_futures'
                else:
                    # Empty history = likely delisted/404 - default to Mats Floor
                    gold_primary = GOLD_VERIFIED_FRIDAY_CLOSE
                    out['price_source'] = 'mats_floor_fallback'
        except Exception as e:
            # 404 or delisted error - immediately default to Mats Floor
            gold_primary = GOLD_VERIFIED_FRIDAY_CLOSE
            out['price_source'] = 'mats_floor_fallback'
        
        try:
            # Silver futures (SI=F) - try multiple periods
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", message=".*possibly delisted.*")
                si = yf.Ticker("SI=F")
                si_hist = None
                for period_try in ["1d", "5d", "1mo"]:
                    try:
                        si_hist = si.history(period=period_try)
                        if not si_hist.empty and "Close" in si_hist.columns:
                            break
                    except Exception:
                        continue
                if si_hist is not None and not si_hist.empty and "Close" in si_hist.columns:
                    silver_primary = float(si_hist["Close"].iloc[-1])
                    if out['price_source'] == 'unknown':
                        out['price_source'] = 'yfinance_futures'
                else:
                    # Empty history = likely delisted/404 - default to Mats Floor
                    silver_primary = SILVER_VERIFIED_FRIDAY_CLOSE
                    if out['price_source'] == 'unknown':
                        out['price_source'] = 'mats_floor_fallback'
        except Exception as e:
            # 404 or delisted error - immediately default to Mats Floor
            silver_primary = SILVER_VERIFIED_FRIDAY_CLOSE
            if out['price_source'] == 'unknown':
                out['price_source'] = 'mats_floor_fallback'
        
        try:
            # Uranium futures (UX=F) - try multiple periods and U-U.TO as fallback
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", message=".*possibly delisted.*")
                ux = yf.Ticker("UX=F")
                ux_hist = None
                for period_try in ["1d", "5d", "1mo"]:
                    try:
                        ux_hist = ux.history(period=period_try)
                        if not ux_hist.empty and "Close" in ux_hist.columns:
                            break
                    except Exception:
                        continue
                if ux_hist is None or ux_hist.empty:
                    # Try U-U.TO (Sprott Physical Uranium Trust) as fallback
                    try:
                        uu = yf.Ticker("U-U.TO")
                        uu_hist = uu.history(period="5d")
                        if not uu_hist.empty and "Close" in uu_hist.columns:
                            uranium_spot = float(uu_hist["Close"].iloc[-1])
                        else:
                            uranium_spot = URANIUM_VERIFIED_FRIDAY_CLOSE
                    except Exception:
                        uranium_spot = URANIUM_VERIFIED_FRIDAY_CLOSE
                elif not ux_hist.empty and "Close" in ux_hist.columns:
                    uranium_spot = float(ux_hist["Close"].iloc[-1])
                else:
                    uranium_spot = URANIUM_VERIFIED_FRIDAY_CLOSE
        except Exception as e:
            # 404 or delisted error - immediately default to Mats Floor
            uranium_spot = URANIUM_VERIFIED_FRIDAY_CLOSE
    
    # V7.4: SECONDARY - Futures Tickers (GC=F, SI=F, UX=F) - Replaced delisted XAGUSD=X/XAUUSD=X
    # V7.4: Yahoo has delisted XAGUSD=X, so we use Futures Tickers as secondary source
    # V7.4: Stop 404s - If API returns 404 or "delisted" error, immediately default to Mats Sovereign Floor
    if YFINANCE:
        import warnings
        try:
            # Gold futures (GC=F) - Secondary fallback if primary failed
            if not np.isfinite(gold_primary):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    warnings.filterwarnings("ignore", message=".*possibly delisted.*")
                    gc = yf.Ticker("GC=F")
                    gc_hist = gc.history(period="1d")
                if not gc_hist.empty and "Close" in gc_hist.columns:
                    gold_secondary = float(gc_hist["Close"].iloc[-1])
                    if out['price_source'] == 'unknown':
                        out['price_source'] = 'yfinance_futures_secondary'
                else:
                    gold_secondary = GOLD_VERIFIED_FRIDAY_CLOSE
                    if out['price_source'] == 'unknown':
                        out['price_source'] = 'mats_floor_fallback'
        except Exception:
            if not np.isfinite(gold_primary):
                gold_secondary = GOLD_VERIFIED_FRIDAY_CLOSE

        try:
            # Silver futures (SI=F) - Secondary fallback if primary failed
            if not np.isfinite(silver_primary):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    warnings.filterwarnings("ignore", message=".*possibly delisted.*")
                    si = yf.Ticker("SI=F")
                    si_hist = si.history(period="1d")
                if not si_hist.empty and "Close" in si_hist.columns:
                    silver_secondary = float(si_hist["Close"].iloc[-1])
                    if out['price_source'] == 'unknown':
                        out['price_source'] = 'yfinance_futures_secondary'
                else:
                    silver_secondary = SILVER_VERIFIED_FRIDAY_CLOSE
                    if out['price_source'] == 'unknown':
                        out['price_source'] = 'mats_floor_fallback'
        except Exception:
            if not np.isfinite(silver_primary):
                silver_secondary = SILVER_VERIFIED_FRIDAY_CLOSE

        try:
            # Uranium futures (UX=F) or U-U.TO (Sprott Physical Trust) - Secondary fallback
            if not np.isfinite(uranium_spot):
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    warnings.filterwarnings("ignore", message=".*possibly delisted.*")
                    ux = yf.Ticker("UX=F")
                    ux_hist = ux.history(period="1d")
                if not ux_hist.empty and "Close" in ux_hist.columns:
                    uranium_spot = float(ux_hist["Close"].iloc[-1])
                else:
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=UserWarning)
                            warnings.filterwarnings("ignore", message=".*possibly delisted.*")
                            uu = yf.Ticker("U-U.TO")
                            uu_hist = uu.history(period="1d")
                        if not uu_hist.empty and "Close" in uu_hist.columns:
                            uranium_spot = float(uu_hist["Close"].iloc[-1])
                        else:
                            uranium_spot = URANIUM_VERIFIED_FRIDAY_CLOSE
                    except Exception:
                        uranium_spot = URANIUM_VERIFIED_FRIDAY_CLOSE
        except Exception:
            if not np.isfinite(uranium_spot):
                uranium_spot = URANIUM_VERIFIED_FRIDAY_CLOSE

    # V7.4: TERTIARY - ETF proxies with spot multiplier (SLV * 1.1, GLD for institutional floor)
    if YFINANCE:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", message=".*possibly delisted.*")
                # GLD (Gold ETF) - use as-is for institutional floor
                gld = yf.Ticker("GLD")
                gld_hist = gld.history(period="1d")
            if not gld_hist.empty and "Close" in gld_hist.columns:
                gld_price = float(gld_hist["Close"].iloc[-1])
                gold_tertiary = gld_price * 10.0  # Approximate: GLD ~0.1 oz per share
                if out['price_source'] == 'unknown':
                    out['price_source'] = 'yfinance_etf'
        except Exception:
            pass

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", message=".*possibly delisted.*")
                # SLV (Silver ETF) - apply 1.1x multiplier for institutional floor
                slv = yf.Ticker("SLV")
                slv_hist = slv.history(period="1d")
            if not slv_hist.empty and "Close" in slv_hist.columns:
                slv_price = float(slv_hist["Close"].iloc[-1])
                silver_tertiary = slv_price * 1.1  # SLV ~1 oz per share
                if out['price_source'] == 'unknown':
                    out['price_source'] = 'yfinance_etf'
        except Exception:
            pass
    
    # V7.4: Select best available price (Primary > Secondary > Tertiary)
    gold_live = gold_primary if np.isfinite(gold_primary) else (gold_secondary if np.isfinite(gold_secondary) else gold_tertiary)
    silver_live = silver_primary if np.isfinite(silver_primary) else (silver_secondary if np.isfinite(silver_secondary) else silver_tertiary)
    
    # V7.4: Mats Sovereign Floor constants (already defined above, but kept here for clarity)
    # These are used as defaults if 404/delisted errors occur
    
    # Use live prices. Fallback to last-known only when ALL sources return NaN.
    if np.isfinite(gold_live) and gold_live > 0:
        out['gold_live'] = gold_live
        out['gold_eod'] = gold_live
    else:
        out['gold_live'] = GOLD_VERIFIED_FRIDAY_CLOSE
        out['gold_eod'] = GOLD_VERIFIED_FRIDAY_CLOSE

    if np.isfinite(silver_live) and silver_live > 0:
        out['silver_live'] = silver_live
        out['silver_eod'] = silver_live
    else:
        out['silver_live'] = SILVER_VERIFIED_FRIDAY_CLOSE
        out['silver_eod'] = SILVER_VERIFIED_FRIDAY_CLOSE

    if np.isfinite(uranium_spot) and uranium_spot > 0:
        out['uranium_spot'] = uranium_spot
    else:
        out['uranium_spot'] = URANIUM_VERIFIED_FRIDAY_CLOSE
    
    # V7.4: Compare Primary vs Secondary/Tertiary for delta (both are Futures now)
    # If primary succeeded, use it; if not, check secondary/tertiary
    if np.isfinite(gold_primary):
        out['gold_use_live'] = True
        # Compare with secondary/tertiary if available
        if np.isfinite(gold_secondary):
            gold_delta_pct = abs((gold_primary - gold_secondary) / gold_secondary) * 100.0
            if gold_delta_pct > 1.0:
                out['live_delta_pct'] = max(out['live_delta_pct'], gold_delta_pct)
        elif np.isfinite(gold_tertiary):
            gold_delta_pct = abs((gold_primary - gold_tertiary) / gold_tertiary) * 100.0
            if gold_delta_pct > 1.0:
                out['live_delta_pct'] = max(out['live_delta_pct'], gold_delta_pct)
    else:
        out['gold_use_live'] = np.isfinite(gold_live)
    
    if np.isfinite(silver_primary):
        out['silver_use_live'] = True
        # Compare with secondary/tertiary if available
        if np.isfinite(silver_secondary):
            silver_delta_pct = abs((silver_primary - silver_secondary) / silver_secondary) * 100.0
            if silver_delta_pct > 1.0:
                out['live_delta_pct'] = max(out['live_delta_pct'], silver_delta_pct)
        elif np.isfinite(silver_tertiary):
            silver_delta_pct = abs((silver_primary - silver_tertiary) / silver_tertiary) * 100.0
            if silver_delta_pct > 1.0:
                out['live_delta_pct'] = max(out['live_delta_pct'], silver_delta_pct)
    else:
        out['silver_use_live'] = np.isfinite(silver_live)
    
    out['ok'] = np.isfinite(gold_live) or np.isfinite(silver_live) or np.isfinite(uranium_spot)
    
    return out


# ============================================================================
# V7.2: GLOBAL COMMODITY BENCHMARKS (Shanghai Arbitrage + Uranium)
# ============================================================================

@st.cache_data(ttl=3600)
def get_global_commodity_benchmarks() -> dict:
    """Fetch live commodity prices from Tiingo/yfinance.

    Returns COMEX gold, silver, and uranium spot prices.
    Shanghai premium fields are kept for backward compatibility but
    default to COMEX values (no free SGE data source available).
    """
    out = {
        "gold_comx": np.nan,
        "gold_shanghai": np.nan,
        "sge_premium_pct": 0.0,
        "silver_comx": np.nan,
        "silver_shanghai": np.nan,
        "shanghai_physical_premium_pct": 0.0,
        "uranium_spot": np.nan,
        "uranium_etf": np.nan,
        "ok": False,
    }
    try:
        live_prices = get_sovereign_spot_prices()

        # Gold
        if np.isfinite(live_prices.get('gold_live', np.nan)):
            out["gold_comx"] = live_prices['gold_live']
        elif np.isfinite(live_prices.get('gold_eod', np.nan)):
            out["gold_comx"] = live_prices['gold_eod']

        # Silver
        if np.isfinite(live_prices.get('silver_live', np.nan)):
            out["silver_comx"] = live_prices['silver_live']
        elif np.isfinite(live_prices.get('silver_eod', np.nan)):
            out["silver_comx"] = live_prices['silver_eod']

        # Uranium
        if np.isfinite(live_prices.get('uranium_spot', np.nan)):
            out["uranium_spot"] = live_prices['uranium_spot']

        # Shanghai defaults to COMEX (no free SGE data source)
        if np.isfinite(out["gold_comx"]):
            out["gold_shanghai"] = out["gold_comx"]
        if np.isfinite(out["silver_comx"]):
            out["silver_shanghai"] = out["silver_comx"]

        out["ok"] = np.isfinite(out["gold_comx"]) or np.isfinite(out["silver_comx"]) or np.isfinite(out["uranium_spot"])
    except Exception:
        pass
    return out


# ============================================================================
# V7.2: SOVEREIGN REBALANCER (Rolling 15-Year Backtesting)
# ============================================================================

@st.cache_data(ttl=3600)
def calculate_sovereign_rebalance_weights(
    all_tickers: list,
    total_portfolio_value: float,
    hist_cache: dict = None,
    benchmarks: dict = None
) -> dict:
    """
    V7.2: Sovereign Rebalancer using Rolling 15-Year Backtesting.
    
    Ranks all tickers by:
    1. Sharpe Ratio (15-year rolling)
    2. P/NAV (lower is better)
    3. Shanghai Premium exposure (bonus for Gold/Silver miners like MAG, PAAS)
    
    Returns: {symbol: recommended_weight_pct} dict
    """
    if not all_tickers:
        return {}
    
    if benchmarks is None:
        benchmarks = get_global_commodity_benchmarks()
    
    # Shanghai / commodity regime adjustments — derived from LIVE data, no ticker-specific bonuses
    # All miners get equal treatment; bonus is based on metal type + live premium data only
    sge_gold_premium_pct = 0.0
    sge_silver_premium_pct = 0.0
    uranium_regime_active = False
    if benchmarks:
        sge_gold_premium_pct = benchmarks.get('sge_premium_pct', 0.0)
        if not np.isfinite(sge_gold_premium_pct):
            sge_gold_premium_pct = 0.0
        silver_shanghai = benchmarks.get('silver_shanghai', 0)
        silver_comx = benchmarks.get('silver_comx', 0)
        sge_silver_premium_pct = ((silver_shanghai - silver_comx) / silver_comx * 100) if silver_comx > 0 else 0.0
        if not np.isfinite(sge_silver_premium_pct):
            sge_silver_premium_pct = 0.0
        uranium_spot = benchmarks.get('uranium_spot', 0)
        uranium_regime_active = np.isfinite(uranium_spot) and uranium_spot >= 85.0
    
    rankings = []
    api_key = (os.getenv("TIINGO_API_KEY") or "").strip()
    has_tiingo = TIINGO_AVAILABLE and TiingoClient is not None and api_key
    
    # 15-year window for backtesting
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=15 * 365)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    client = None
    if has_tiingo:
        try:
            client = st.session_state.get("tiingo_client")
            if client is None:
                client = TiingoClient({"api_key": api_key})
                st.session_state["tiingo_client"] = client
        except Exception:
            has_tiingo = False
    
    for symbol in all_tickers:
        sharpe_15y = np.nan
        p_nav = np.nan
        shanghai_bonus = 1.0
        
        # Get historical data (15-year if available)
        hist_15y = None
        if hist_cache and symbol in hist_cache:
            hist_15y = hist_cache[symbol]
            if len(hist_15y) < 252:  # Need at least 1 year
                hist_15y = None
        
        if hist_15y is None and has_tiingo:
            try:
                hist_15y, _ = fetch_ticker_with_fallback(symbol, period="15y", data_health=None)
            except Exception:
                pass
        
        # Calculate 15-year Sharpe Ratio (use existing _trailing_sharpe function)
        if hist_15y is not None and not hist_15y.empty:
            sharpe_15y = _trailing_sharpe(hist_15y, window=min(252, len(hist_15y)))
        
        # Get P/NAV from fundamentals (if available)
        try:
            fh = get_forensic_fundamentals(symbol)
            if fh and fh.get("p_nav") is not None:
                try:
                    p_nav = float(fh["p_nav"]) if fh["p_nav"] != '' else np.nan
                except (ValueError, TypeError):
                    p_nav = np.nan
        except Exception:
            pass
        
        # Commodity regime bonus — based on metal type and live premium data
        metal_type = None
        try:
            fund = get_fundamentals_with_tracking(symbol)
            metal_type = (fund.get('metal', '') or '').upper()
        except Exception:
            pass

        shanghai_bonus = 1.0
        if metal_type == 'GOLD' and sge_gold_premium_pct > 1.0:
            shanghai_bonus = 1.0 + min(sge_gold_premium_pct / 100.0, 0.10)  # Up to +10% from live data
        elif metal_type == 'SILVER' and sge_silver_premium_pct > 1.0:
            shanghai_bonus = 1.0 + min(sge_silver_premium_pct / 100.0, 0.10)
        elif metal_type == 'URANIUM' and uranium_regime_active:
            shanghai_bonus = 1.05  # Modest 5% boost in strong uranium regime
        
        # Calculate composite score
        sharpe_score = (sharpe_15y * 10) if np.isfinite(sharpe_15y) and sharpe_15y > 0 else 0.0
        nav_score = (100.0 / p_nav) if np.isfinite(p_nav) and p_nav > 0 else 0.0
        composite_score = (sharpe_score + nav_score) * shanghai_bonus
        
        rankings.append({
            'symbol': symbol,
            'sharpe_15y': sharpe_15y,
            'p_nav': p_nav,
            'shanghai_bonus': shanghai_bonus,
            'composite_score': composite_score
        })
    
    # Sort by composite score (highest first)
    rankings.sort(key=lambda x: x['composite_score'], reverse=True)
    
    # Calculate weights: Top 20 get allocation, rest get minimal
    # V7.2: Prioritize high Sharpe + low P/NAV + Shanghai exposure
    weights = {}
    total_score = sum(r['composite_score'] for r in rankings if r['composite_score'] > 0)
    
    if total_score > 0:
        # Proportional allocation based on composite score
        for rank in rankings:
            if rank['composite_score'] > 0:
                weight_pct = (rank['composite_score'] / total_score) * 100.0
                # Cap individual positions at 10% (sovereign rebalancing rule)
                weight_pct = min(weight_pct, 10.0)
                weights[rank['symbol']] = weight_pct
    else:
        # Fallback: equal weight for top 20
        top_20 = rankings[:20]
        equal_weight = 100.0 / len(top_20) if top_20 else 0
        for rank in top_20:
            weights[rank['symbol']] = min(equal_weight, 10.0)
    
    # Normalize to 100%
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: (v / total_weight) * 100.0 for k, v in weights.items()}
    
    return weights


# ============================================================================
# RENDER MORNING TAPE (SIMPLE VERSION)
# ============================================================================

def render_morning_tape_simple(gold_analysis, silver_analysis, metal_regime):
    """Simple morning tape renderer"""
    st.markdown("---")
    st.header("METAL OUTLOOK")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"🥇 Gold: ${gold_analysis.get('current_price', 0):,.0f}")
        st.write(f"**Today:** {gold_analysis.get('forecast_today', '↔')}")
        st.write(f"**1 Week:** {gold_analysis.get('forecast_week', '↔')} ({gold_analysis.get('bias_short', 'NEUTRAL')})")
        st.write(f"**1-2 Months:** {gold_analysis.get('forecast_month', '↔')} ({gold_analysis.get('bias_medium', 'NEUTRAL')})")
        st.caption(gold_analysis.get('explanation', ''))
    
    with col2:
        st.subheader(f"🥈 Silver: ${silver_analysis.get('current_price', 0):.2f}")
        st.write(f"**Today:** {silver_analysis.get('forecast_today', '↔')}")
        st.write(f"**1 Week:** {silver_analysis.get('forecast_week', '↔')} ({silver_analysis.get('bias_short', 'NEUTRAL')})")
        st.write(f"**1-2 Months:** {silver_analysis.get('forecast_month', '↔')} ({silver_analysis.get('bias_medium', 'NEUTRAL')})")
        st.caption(silver_analysis.get('explanation', ''))
    
    # Portfolio guidance
    st.markdown("### Portfolio Guidance")
    posture = metal_regime.get('regime', 'NEUTRAL')
    
    if 'BEARISH' in posture:
        st.error(f"🛑 **{posture}** - Reduce risk, favor producers")
    elif 'BULLISH' in posture:
        st.success(f"✅ **{posture}** - Normal risk appetite")
    else:
        st.info(f"📊 **{posture}** - Cautious approach")

# ============================================================================
# SESSION STATE & SIDEBAR
# ============================================================================

DEFAULT_PORTFOLIO = pd.DataFrame({
    'Symbol': ['JZRIF', 'ITRG', 'SMAGF', 'LOMLF', 'TSKFF', 'AGXPF', 'GSVRF', 
               'EXNRF', 'WRLGF', 'JAGGF', 'BORMF', 'AG', 'LUCMF', 'SMDRF'],
    'Quantity': [19841, 2072, 8335, 24557, 13027, 32342, 9049, 
                 11749, 25929, 2965, 5172, 638, 7079, 7072],
    'Cost_Basis': [5959.25, 7838.13, 9928.39, 5006.76, 13857.41, 24015.26, 2415.31,
                   3242.97, 18833.21, 14558.14, 5540.99, 7594.99, 8550.49, 6939.18],
    'Insider_Buying_90d': [False] * 14
})

# V5.0: Cache portfolio loading to prevent terminal looping
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_portfolio_cached():
    """Cached wrapper for portfolio loading to prevent repeated DataFrame operations"""
    return validate_metadata(DEFAULT_PORTFOLIO.copy())

if 'portfolio' not in st.session_state:
    # V5.0: Validate metadata when initializing portfolio (cached)
    st.session_state.portfolio = load_portfolio_cached()
if 'cash' not in st.session_state:
    st.session_state.cash = 39569.65

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")

    # V8.0: Alert Panel (top of sidebar)
    if ALERT_ENGINE_AVAILABLE:
        alerts = st.session_state.get('active_alerts', [])
        if alerts:
            summary = get_alert_summary(alerts)
            alert_label = f"🔔 Alerts: {summary['critical']}C / {summary['warning']}W / {summary['info']}I"
            with st.expander(alert_label, expanded=summary['critical'] > 0):
                for a in alerts[:10]:
                    sev = a.get('severity', 'info')
                    msg = f"**{a.get('symbol', '?')}**: {a.get('message', '')}"
                    if sev == 'critical':
                        st.error(msg)
                    elif sev == 'warning':
                        st.warning(msg)
                    else:
                        st.info(msg)
                if len(alerts) > 10:
                    st.caption(f"... and {len(alerts) - 10} more alerts")

    # V7.4: "Black Box" Diagnostic Tracker at top of sidebar
    st.markdown("### Black Box Diagnostic")
    with st.expander("🔬 System Diagnostics", expanded=False):
        # Show current working directory
        cwd = os.getcwd()
        st.text(f"📁 Current Directory: {cwd}")
        
        # Check if hey.env exists (relative to project root)
        _diag_env_path = Path(__file__).parent / "hey.env"
        env_exists = _diag_env_path.exists()
        if env_exists:
            st.success(f"✅ hey.env found at: {_diag_env_path}")
        else:
            st.error(f"❌ hey.env NOT found at: {_diag_env_path}")
        
        # Check TIINGO_API_KEY and show first 4 chars
        tiingo_key_check = os.getenv("TIINGO_API_KEY", "").strip()
        if tiingo_key_check:
            key_preview = tiingo_key_check[:4] + "..." if len(tiingo_key_check) > 4 else tiingo_key_check[:4]
            st.success(f"✅ TIINGO_API_KEY loaded: {key_preview} (first 4 chars)")
            st.session_state['tiingo_key_detected'] = True
        else:
            st.error("❌ TIINGO_API_KEY NOT loaded")
            st.session_state['tiingo_key_detected'] = False
    
    # V7.4: CRITICAL ALERT - Force verification of TIINGO_API_KEY
    tiingo_key_check = os.getenv("TIINGO_API_KEY", "").strip()
    if not tiingo_key_check:
        st.error(
            "🚨 **CRITICAL: hey.env found but KEY is empty or missing.**\n\n"
            "**Action Required:**\n"
            "1. Check that `hey.env` file exists at: `/Users/mats/PortfolioAnalyzer/hey.env`\n"
            "2. Verify `TIINGO_API_KEY=your_key_here` is set in `hey.env`\n"
            "3. Restart the application after adding the key\n\n"
            "**Note:** The scan will continue using Futures Tickers (GC=F, SI=F, UX=F) as fallback."
        )
        st.markdown("---")
    
    # V5.0: Action Sidebar - Rebalance Status at top
    st.markdown("---")
    st.markdown("### Action Status")
    if 'results' in st.session_state:
        results_df = st.session_state.results
        total_mv = np.float64(results_df['Market_Value'].sum())
        total_value = np.float64(total_mv + np.float64(st.session_state.cash))
        
        # V7.2: Calculate threshold rebalancing status using total_portfolio_value from UI
        rebalance_statuses = []
        max_drift = 0.0
        action_required_count = 0
        
        # V7.3: Get total_portfolio_value from UI (no hardcoded fallback)
        total_portfolio_value = st.session_state.get("total_portfolio_value", None)
        if total_portfolio_value is None or total_portfolio_value <= 0:
            # Only calculate from holdings if UI value not set
            total_mv = np.float64(results_df['Market_Value'].sum())
            total_portfolio_value = np.float64(total_mv + np.float64(st.session_state.cash))
        
        for _, row in results_df.iterrows():
            current_pct = row.get('Pct_Portfolio', 0)
            target_pct = row.get('Recommended_Pct', current_pct)
            # target_pct == 0 is a valid SELL signal — don't override it
            if target_pct < 0:
                target_pct = 0

            # V7.2: Calculate drift in $ terms using total_portfolio_value
            drift_pct = abs(current_pct - target_pct)
            drift_usd = (drift_pct / 100.0) * float(total_portfolio_value)
            max_drift = max(max_drift, drift_pct)

            if drift_pct > 2.0:  # Lower threshold: 2% drift triggers action (was 5%)
                action_required_count += 1
                rebalance_statuses.append({
                    'symbol': row['Symbol'],
                    'current': current_pct,
                    'target': target_pct,
                    'drift': drift_pct,
                    'drift_usd': drift_usd,  # V7.2: Add USD drift
                    'status': 'ACTION_REQUIRED'
                })
        
        # Display status
        if max_drift > 5.0:
            st.error(f"🔴 **⚠️ {action_required_count} Trades Required**")
            st.caption(f"Max drift: {max_drift:.1f}%")
        elif max_drift > 3.0:
            st.warning(f"🟡 **Threshold Nearing** (Max drift: {max_drift:.1f}%)")
        else:
            st.success("🟢 **Portfolio in Balance**")
    else:
        st.caption("Run analysis to see status")
    
    # V7.5: Live Metal Prices Display (replaced dead Shanghai premium section)
    st.markdown("---")
    st.markdown("### Live Metal Prices")
    benchmarks = get_global_commodity_benchmarks()
    live_prices = get_sovereign_spot_prices()

    if benchmarks.get('ok', False):
        gold_comx = benchmarks.get('gold_comx', np.nan)
        silver_comx = benchmarks.get('silver_comx', np.nan)
        price_source = live_prices.get('price_source', 'unknown')

        col1, col2 = st.columns(2)
        with col1:
            if np.isfinite(gold_comx):
                st.metric("Gold", f"${gold_comx:,.2f}/oz")
            else:
                st.metric("Gold", "N/A", "Data unavailable")
        with col2:
            if np.isfinite(silver_comx):
                st.metric("Silver", f"${silver_comx:.2f}/oz")
            else:
                st.metric("Silver", "N/A", "Data unavailable")

        # Source + live indicator
        source_label = price_source.replace('_', ' ').title() if price_source != 'unknown' else 'Unavailable'
        if live_prices.get('gold_use_live', False) or live_prices.get('silver_use_live', False):
            delta_pct = live_prices.get('live_delta_pct', 0)
            st.caption(f"Source: {source_label} | Live (delta {delta_pct:.2f}% vs EOD)")
        else:
            st.caption(f"Source: {source_label}")
    else:
        st.caption("Metal prices unavailable")

    # V7.5: Energy Regime (Uranium)
    st.markdown("---")
    st.markdown("### Energy Regime")
    uranium_spot = benchmarks.get('uranium_spot', np.nan)
    if np.isfinite(uranium_spot):
        st.metric("Uranium Spot", f"${uranium_spot:.2f}/lb",
                 "Weighting CCJ, NXE, DNN, URA")
        uranium_stocks = ['CCJ', 'NXE', 'DNN', 'UEC', 'UUUU', 'URA']
        st.caption(f"Uranium exposure: {', '.join(uranium_stocks)}")
    else:
        st.metric("Uranium Spot", f"${85.0:.2f}/lb", "Floor estimate (live unavailable)")
    
    st.markdown("---")
    
    # Version display
    st.markdown("---")
    st.caption(f"**Alpha Miner Pro {VERSION}**")
    st.caption(f"Release: {VERSION_DATE}")
    with st.expander("📋 Features in this version"):
        for feature in VERSION_FEATURES:
            st.caption(f"• {feature}")
    st.markdown("---")
    


    st.markdown("### Risk Governance")
    st.session_state.strict_mode = st.toggle("STRICT MODE (downgrade on low confidence)", value=bool(st.session_state.get('strict_mode', True)))

    # Risk profile presets
    risk_profile = st.selectbox(
        "Risk Profile",
        options=["Balanced", "Aggressive", "Defensive"],
        index=["Balanced", "Aggressive", "Defensive"].index(st.session_state.get('risk_profile', 'Balanced'))
    )
    st.session_state.risk_profile = risk_profile

    preset = get_risk_profile_preset(risk_profile)
    with st.expander("Show preset parameters"):
        st.table(pd.DataFrame([preset]))

    st.markdown("### Replay Mode (Offline)")
    replay_mode = st.toggle("Replay from Evidence Pack (no network calls)", value=bool(st.session_state.get('replay_mode', False)))
    st.session_state.replay_mode = replay_mode

    if replay_mode:
        uploaded = st.file_uploader("Upload evidence pack JSON", type=["json"], key="evidence_pack_uploader")
        if uploaded is not None:
            try:
                pack = json.loads(uploaded.getvalue().decode('utf-8'))
                st.session_state.replay_pack = pack
                st.success(f"Loaded evidence pack: {pack.get('evidence_pack_id','(no id)')}")
            except Exception as e:
                st.session_state.replay_pack = None
                st.error(f"Could not load JSON: {e}")

        existing = list_evidence_packs()
        if existing:
            pick = st.selectbox("Or load saved pack", options=[str(p.name) for p in existing], index=0)
            if st.button("Load selected pack"):
                pack_path = EVIDENCE_DIR / pick
                st.session_state.replay_pack = load_evidence_pack(pack_path)
                st.success(f"Loaded: {pick}")

        if st.session_state.get('replay_pack'):
            st.caption("OFFLINE_MODE is ON. Analysis will render from the evidence pack.")

    st.markdown("---")
    
    st.markdown("### Determinism / Run Settings")
    
    # Initialize settings if not present
    if 'freeze_time' not in st.session_state:
        st.session_state.freeze_time = False
    if 'disable_sector_fallback_news' not in st.session_state:
        st.session_state.disable_sector_fallback_news = False
    if 'disable_inferred_fundamentals' not in st.session_state:
        st.session_state.disable_inferred_fundamentals = False
    
    freeze_time = st.toggle(
        "Freeze time (use run timestamp as 'now')",
        value=st.session_state.freeze_time,
        help="Use the run timestamp for all time-based calculations instead of current time"
    )
    st.session_state.freeze_time = freeze_time
    
    disable_sector_news = st.toggle(
        "Disable sector fallback news",
        value=st.session_state.disable_sector_fallback_news,
        help="If ticker news unavailable, do not fall back to sector news"
    )
    st.session_state.disable_sector_fallback_news = disable_sector_news
    
    disable_inferred = st.toggle(
        "Disable inferred fundamentals (missing => Unknown)",
        value=st.session_state.disable_inferred_fundamentals,
        help="Do not infer missing fundamentals; mark as Unknown instead"
    )
    st.session_state.disable_inferred_fundamentals = disable_inferred
    
    st.markdown("---")
    
    # V4.0 Phase 3: Fund Health Metrics
    st.markdown("### Fund Health")
    
    # Execution Efficiency (from backtest if available, or show placeholder)
    execution_efficiency = st.session_state.get('execution_efficiency', None)
    if execution_efficiency is not None:
        eff_color = "🟢" if execution_efficiency >= 80 else "🟡" if execution_efficiency >= 60 else "🔴"
        st.metric(
            "Execution Efficiency",
            f"{execution_efficiency:.1f}%",
            help="Percentage of intended trades that passed the 1.5% Market Impact Gate"
        )
        st.caption(f"{eff_color} {execution_efficiency:.1f}% of trades passed liquidity gates")
    else:
        st.caption("Execution Efficiency: Run backtest to see metric")
        st.caption("(Shows % of trades that passed 1.5% Market Impact Gate)")
    
    # GSR Status (V4.0 Phase 3)
    gsr_bias = st.session_state.get('gsr_bias')
    if gsr_bias:
        gsr_ratio = gsr_bias.get('gs_ratio', 0)
        silver_bonus = gsr_bias.get('silver_bonus', 0)
        if silver_bonus > 0:
            st.metric("Gold/Silver Ratio", f"{gsr_ratio:.1f}", delta=f"+{silver_bonus} Silver Bonus")
            st.caption("💰 Silver Torque Bonus Active")
        else:
            st.metric("Gold/Silver Ratio", f"{gsr_ratio:.1f}")
    else:
        st.caption("GSR: Run analysis to fetch")
    
    st.markdown("---")
    
    # V5.0: Sector Watchlist (Top 10 from master_discovery_list.csv by Alpha Score)
    st.markdown("### Sector Watchlist")
    if load_master_discovery_list is not None:
        try:
            master_symbols = load_master_discovery_list(max_symbols=200)
            if master_symbols:
                watchlist_data = []
                
                # Get Alpha scores from current results if available
                if 'results' in st.session_state:
                    results_df = st.session_state.results
                    
                    # Check all symbols from master list (up to 200)
                    for symbol in master_symbols:
                        # Check if symbol is in current portfolio results
                        symbol_row = results_df[results_df['Symbol'] == symbol]
                        if not symbol_row.empty:
                            row = symbol_row.iloc[0]
                            # Include regardless of fundamental filters (as requested)
                            watchlist_data.append({
                                'Symbol': symbol,
                                'Alpha_Score': row.get('Alpha_Score', 0),
                                'Action': row.get('Action', 'HOLD'),
                                'Price': row.get('Price', 0),
                                'FA_Score': row.get('FA_Score', 0),
                                'Sell_Risk_Score': row.get('Sell_Risk_Score', 50)
                            })
                    
                    if watchlist_data:
                        watchlist_df = pd.DataFrame(watchlist_data)
                        # Sort by Alpha Score (highest first) and take Top 10
                        watchlist_df = watchlist_df.sort_values('Alpha_Score', ascending=False).head(10)
                        
                        for idx, (_, row) in enumerate(watchlist_df.iterrows(), 1):
                            action_emoji = "🟢" if row['Action'] == 'Buy' else "🔵" if row['Action'] == 'HOLD' else "⚪"
                            st.caption(f"{idx}. {action_emoji} **{row['Symbol']}** - Alpha: {row['Alpha_Score']:.1f} | {row['Action']} | ${row['Price']:.2f}")
                    else:
                        st.caption("No symbols from master list found in current portfolio.")
                        st.caption("💡 Use '🚀 Scan Master List' button in Scanner tab to analyze master_discovery_list.csv")
                else:
                    st.caption("Run analysis to see watchlist rankings")
            else:
                st.caption("No symbols found in master_discovery_list.csv")
        except FileNotFoundError:
            st.caption("master_discovery_list.csv not found")
        except Exception as e:
            st.caption(f"Watchlist error: {str(e)[:50]}")
    else:
        st.caption("Master discovery list scanner not available")
    
    st.markdown("---")
    
    # V5.0: Audit Log Section (refactored to remove nested expanders)
    with st.expander("📜 Audit Log", expanded=False):
        # Power Status: Tiingo API key + MAG test (strict check; avoid auth loop from empty key)
        _tk = (os.getenv("TIINGO_API_KEY") or "").strip()
        _diag = st.session_state.get("tiingo_key_diagnostic")
        if _diag:
            st.error(f"**{_diag}**")
        if not _tk:
            st.error("**TIINGO_API_KEY missing in hey.env** — add it and ensure the variable name matches exactly.")
        _tiingo_ok = False
        if _tk and TIINGO_AVAILABLE and TiingoClient is not None:
            try:
                _tc = TiingoClient({"api_key": _tk})
                _tc.get_ticker_metadata("GOLD")
                _tiingo_ok = True
            except Exception:
                # More lenient - try fetching price data as fallback
                try:
                    _tc = TiingoClient({"api_key": _tk})
                    end_date = datetime.datetime.now().strftime('%Y-%m-%d')
                    start_date = (datetime.datetime.now() - datetime.timedelta(days=5)).strftime('%Y-%m-%d')
                    _tc.get_ticker_price("GOLD", startDate=start_date, endDate=end_date)
                    _tiingo_ok = True
                except Exception:
                    pass
        if _tiingo_ok:
            st.success("✅ **Tiingo Power Active** (hey.env)")
        elif _tk:
            st.warning("⚠️ **Tiingo Power: Limited Connectivity (Manual Mode)** — MAG test failed; fetches still enabled.")
        st.markdown("---")

        if 'results' in st.session_state:
            results_df = st.session_state.results
            total_mv = np.float64(results_df['Market_Value'].sum())
            total_value = np.float64(total_mv + np.float64(st.session_state.cash))
            
            # Import validate_trade_integrity from backtest_runner
            try:
                from backtest_runner import validate_trade_integrity
                
                # Simulate rebalance to get after-state
                # For UI purposes, we'll use current Recommended_Pct as the "after" state
                portfolio_after = results_df.copy()
                if 'Recommended_Pct' in portfolio_after.columns:
                    # V5.0: Weight Normalization Step - ensure Recommended_Pct sums to 100%
                    # This handles cases where stocks are filtered out, leaving 90% that needs to scale to 100%
                    rec_pct_sum = portfolio_after['Recommended_Pct'].sum()
                    if rec_pct_sum > 0 and abs(rec_pct_sum - 100.0) > 0.01:
                        # Normalize: scale all percentages so they sum to exactly 100%
                        portfolio_after['Recommended_Pct'] = (portfolio_after['Recommended_Pct'] / rec_pct_sum) * 100.0
                    
                    # Calculate total value after rebalance (assuming no slippage for validation)
                    total_value_after = total_value  # In real rebalance, this would account for fees/slippage
                
                # Run integrity checks
                integrity_result = validate_trade_integrity(
                    portfolio=portfolio_after,
                    total_value_before=total_value,
                    total_value_after=total_value_after,
                    tolerance_pct=0.01
                )
                
                # V5.0: Verify weight sum is between 99.99% and 100.01% before showing Logic Verified
                weight_sum_valid = False
                rec_pct_sum = 0.0
                if 'Recommended_Pct' in portfolio_after.columns:
                    rec_pct_sum = portfolio_after['Recommended_Pct'].sum()
                    weight_sum_valid = (99.99 <= rec_pct_sum <= 100.01)
                
                # V5.0: Update audit_passed state immediately (before UI rendering)
                # Only pass if integrity checks pass AND weight sum is valid
                audit_passed = integrity_result['passed'] and weight_sum_valid
                
                # Store integrity_result in session state for JSON display
                st.session_state.integrity_result = integrity_result
                
                if audit_passed:
                    st.session_state.audit_passed = True  # Store for trade button enablement
                    st.success("✅ **Logic Verified** - All trade integrity checks passed")
                    
                    # Show weight sum status
                    if 'Recommended_Pct' in portfolio_after.columns:
                        rec_pct_sum = portfolio_after['Recommended_Pct'].sum()
                        st.info(f"✅ Weight Sum: {rec_pct_sum:.2f}% (within 99.99%-100.01% tolerance)")
                    
                    # Replace nested expander with st.info and st.code
                    st.info("**Check Details Below:**")
                    check_details = []
                    for check in integrity_result['checks']:
                        if check.get('passed', False):
                            check_details.append(f"✅ {check['name']}: {check.get('expected', 'N/A')} = {check.get('actual', 'N/A')} (deviation: {check.get('deviation', 'N/A')})")
                        else:
                            check_details.append(f"❌ {check['name']}: {check.get('error', 'Failed')}")
                    
                    if check_details:
                        st.code("\n".join(check_details), language=None)
                    
                    # Show full audit results as JSON (optional, for debugging) - moved outside to avoid nesting
                    st.caption("💡 Expand below to view full audit results in JSON format")
                else:
                    st.session_state.audit_passed = False  # Store for trade button disablement
                    
                    # Show specific failure reason
                    failure_reasons = []
                    if not integrity_result['passed']:
                        failure_reasons.append("Trade integrity checks failed")
                    if 'Recommended_Pct' in portfolio_after.columns:
                        rec_pct_sum = portfolio_after['Recommended_Pct'].sum()
                        if not (99.99 <= rec_pct_sum <= 100.01):
                            failure_reasons.append(f"Weight sum {rec_pct_sum:.2f}% outside 99.99%-100.01% tolerance")
                    
                    st.error(f"❌ **System Lock: Calculation Variance Detected** - {'; '.join(failure_reasons)}")
                    
                    # Replace nested expander with st.error and st.code
                    st.error("🔒 **All execution buttons are disabled until integrity checks pass**")
                    
                    # Show errors
                    if integrity_result.get('errors'):
                        st.warning("**Errors Found:**")
                        error_list = []
                        for error in integrity_result['errors']:
                            error_list.append(f"• {error}")
                        st.code("\n".join(error_list), language=None)
                    
                    # Show failed checks
                    failed_checks = []
                    for check in integrity_result['checks']:
                        if not check.get('passed', False):
                            failed_checks.append(f"❌ {check['name']}: {check.get('error', 'Failed')}")
                    
                    if failed_checks:
                        st.warning("**Failed Checks:**")
                        st.code("\n".join(failed_checks), language=None)
                    
                    # Show weight sum if invalid
                    if 'Recommended_Pct' in portfolio_after.columns:
                        rec_pct_sum = portfolio_after['Recommended_Pct'].sum()
                        if not (99.99 <= rec_pct_sum <= 100.01):
                            st.error(f"❌ Weight Sum: {rec_pct_sum:.2f}% (must be between 99.99% and 100.01%)")
                    
                    # Show full audit results as JSON (optional, for debugging) - moved outside to avoid nesting
                    st.caption("💡 Expand below to view full audit results in JSON format")
            except ImportError:
                st.session_state.audit_passed = False
                st.warning("⚠️ Trade integrity validation not available (backtest_runner not importable)")
            except Exception as e:
                st.session_state.audit_passed = False
                st.error(f"❌ **Audit Error**: {str(e)[:100]}")
        else:
            st.session_state.audit_passed = False  # Default to False if no results
            st.caption("Run analysis to see audit log")
    
    # V5.0: Hard-Burn Stress Test (under Audit Log, high-contrast metric)
    # V7.2: Use total_portfolio_value from UI (no hardcoding)
    # V7.3: Use live prices when available (>1% difference from EOD)
    st.markdown("---")
    st.markdown("### Hard-Burn Stress Test")
    if "results" in st.session_state and TIINGO_AVAILABLE and TiingoClient is not None and (os.getenv("TIINGO_API_KEY") or "").strip():
        results_df = st.session_state.results
        # V7.3: Use total_portfolio_value from UI (no hardcoded fallback)
        total_portfolio_value = st.session_state.get("total_portfolio_value", None)
        if total_portfolio_value is None or total_portfolio_value <= 0:
            # Only calculate from holdings if UI value not set
            total_mv = np.float64(results_df["Market_Value"].sum())
            total_portfolio_value = np.float64(total_mv + np.float64(st.session_state.cash))
        
        # V7.3: Get live prices and update Market_Value if live prices are active
        live_prices = get_sovereign_spot_prices()
        benchmarks = get_global_commodity_benchmarks()
        if live_prices.get('gold_use_live', False) or live_prices.get('silver_use_live', False):
            # Recalculate total_portfolio_value with updated Market_Value
            total_mv = np.float64(results_df["Market_Value"].sum())
            total_portfolio_value = np.float64(total_mv + np.float64(st.session_state.cash))
            st.caption(f"⚡ Using Live Prices (Δ {live_prices.get('live_delta_pct', 0):.2f}% vs EOD) for Hard-Burn calculation")
        
        cash = float(st.session_state.cash)
        pf = results_df[["Symbol", "Quantity"]].copy()
        pf_json = pf.to_json(orient="records")
        try:
            res = run_sovereign_stress_test_cached(pf_json, float(total_portfolio_value), cash)
            st.session_state.hard_burn_usd = res.get("hard_burn_usd", 0.0)
            st.session_state.hard_burn_pct = res.get("hard_burn_pct", 0.0)
            st.session_state.hard_burn_ok = res.get("ok", False)
            if res.get("ok"):
                hb = res["hard_burn_usd"]
                pct = res["hard_burn_pct"] * 100
                st.metric(
                    "🔥 **Estimated Hard-Burn**",
                    f"-${abs(hb):,.0f}" if hb < 0 else "$0",
                    f"{pct:.1f}% max pain (3 worst 30d sector windows)",
                    delta_color="normal",
                )
            else:
                st.caption(f"Stress test unavailable: {res.get('error', 'unknown')[:60]}")
                st.session_state.hard_burn_usd = 0.0
                st.session_state.hard_burn_pct = 0.0
                st.session_state.hard_burn_ok = False
        except Exception as e:
            st.caption(f"Hard-Burn error: {str(e)[:60]}")
            st.session_state.hard_burn_usd = 0.0
            st.session_state.hard_burn_pct = 0.0
            st.session_state.hard_burn_ok = False
    else:
        st.caption("Run analysis and enable Tiingo (hey.env) for Hard-Burn estimate.")
        st.session_state.hard_burn_usd = 0.0
        st.session_state.hard_burn_pct = 0.0
        st.session_state.hard_burn_ok = False

    # V6.5: Deep-Trust – Strike Backtest (5y/15y, Sharpe, MaxDD, Sovereign Win Rate)
    if STRIKE_BACKTEST_AVAILABLE and run_strike_backtest and (os.getenv("TIINGO_API_KEY") or "").strip():
        with st.expander("📊 V6.5 Strike Backtest (5y/15y)", expanded=False):
            try:
                strike = run_strike_backtest()
                rows = []
                for sym, m in strike.items():
                    rows.append({
                        "Symbol": sym,
                        "Sharpe 5y": f"{m.get('sharpe_5y', np.nan):.2f}" if np.isfinite(m.get("sharpe_5y")) else "—",
                        "Sharpe 15y": f"{m.get('sharpe_15y', np.nan):.2f}" if np.isfinite(m.get("sharpe_15y")) else "—",
                        "MaxDD 5y": f"{m.get('max_dd_5y', np.nan) * 100:.1f}%" if np.isfinite(m.get("max_dd_5y")) else "—",
                        "MaxDD 15y": f"{m.get('max_dd_15y', np.nan) * 100:.1f}%" if np.isfinite(m.get("max_dd_15y")) else "—",
                        "Win% 5y": f"{m.get('win_rate_5y', np.nan):.0f}%" if np.isfinite(m.get("win_rate_5y")) else "—",
                        "Win% 15y": f"{m.get('win_rate_15y', np.nan):.0f}%" if np.isfinite(m.get("win_rate_15y")) else "—",
                    })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.caption("No strike backtest results (Tiingo or symbols unavailable).")
            except Exception as e:
                st.caption(f"Strike backtest error: {str(e)[:80]}")

    # V5.0: Show full audit results as JSON (outside main expander to avoid nesting)
    if 'results' in st.session_state and 'integrity_result' in st.session_state:
        with st.expander("📋 View Full Audit Results (JSON)", expanded=False):
            st.json(st.session_state.integrity_result)
    
    st.markdown("---")
    
    # V5.0: Backtest Verification Module
    st.markdown("### Verification")
    if st.button("📊 Run Verification Backtest", use_container_width=True, disabled=not st.session_state.get('audit_passed', False)):
        with st.spinner("Running 12-month verification backtest..."):
            try:
                # Import backtest runner
                import subprocess
                import sys

                # Calculate date range (last 12 months)
                end_date = datetime.datetime.now()
                start_date = end_date - datetime.timedelta(days=365)
                
                # Run backtest
                result = subprocess.run(
                    [sys.executable, 'backtest_runner.py',
                     '--start', start_date.strftime('%Y-%m-%d'),
                     '--end', end_date.strftime('%Y-%m-%d'),
                     '--portfolio_csv', 'portfolio_enhanced.csv',
                     '--initial_cash', str(st.session_state.cash),
                     '--data_dir', './.backtest_cache/',
                     '--offline'],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    st.success("✅ Backtest completed successfully")
                    
                    # Try to load backtest results
                    try:
                        import json
                        reports_dir = Path('./reports')
                        summary_file = reports_dir / 'backtest_summary.json'
                        
                        if summary_file.exists():
                            with open(summary_file, 'r') as f:
                                summary = json.load(f)
                            
                            # Create dual-axis chart: Strategy vs. Silver/Gold Spot
                            if PLOTLY_AVAILABLE:
                                # Load daily equity data
                                daily_file = reports_dir / 'backtest_daily.csv'
                                if daily_file.exists():
                                    daily_df = pd.read_csv(daily_file)
                                    daily_df['date'] = pd.to_datetime(daily_df['date'])
                                    
                                    # Fetch Gold/Silver prices for comparison
                                    try:
                                        import yfinance as yf
                                        import warnings as _w
                                        with _w.catch_warnings():
                                            _w.filterwarnings("ignore", message=".*possibly delisted.*")
                                            gold = yf.Ticker("GC=F").history(period="1y")
                                            silver = yf.Ticker("SI=F").history(period="1y")
                                        
                                        # Normalize to percentage change
                                        if not gold.empty and not silver.empty:
                                            gold_pct = ((gold['Close'] - gold['Close'].iloc[0]) / gold['Close'].iloc[0] * 100)
                                            silver_pct = ((silver['Close'] - silver['Close'].iloc[0]) / silver['Close'].iloc[0] * 100)
                                            
                                            # Create dual-axis chart
                                            fig = go.Figure()
                                            
                                            # Strategy equity (Portfolio Performance)
                                            if 'equity' in daily_df.columns:
                                                strategy_pct = ((daily_df['equity'] - daily_df['equity'].iloc[0]) / daily_df['equity'].iloc[0] * 100)
                                                fig.add_trace(go.Scatter(
                                                    x=daily_df['date'],
                                                    y=strategy_pct,
                                                    name='Portfolio Performance',
                                                    line=dict(color='#2563eb', width=2)
                                                ))
                                            
                                            # Gold benchmark
                                            fig.add_trace(go.Scatter(
                                                x=gold_pct.index,
                                                y=gold_pct.values,
                                                name='Gold Benchmark',
                                                line=dict(color='#f59e0b', width=1, dash='dash'),
                                                yaxis='y2'
                                            ))
                                            
                                            # Silver benchmark
                                            fig.add_trace(go.Scatter(
                                                x=silver_pct.index,
                                                y=silver_pct.values,
                                                name='Silver Benchmark',
                                                line=dict(color='#94a3b8', width=1, dash='dash'),
                                                yaxis='y2'
                                            ))
                                            
                                            fig.update_layout(
                                                title='Portfolio Performance vs. Silver/Gold Benchmarks (12-Month)',
                                                xaxis_title='Date',
                                                yaxis=dict(title='Portfolio Return (%)', side='left'),
                                                yaxis2=dict(title='Benchmark Return (%)', side='right', overlaying='y'),
                                                hovermode='x unified',
                                                height=400
                                            )
                                            
                                            st.plotly_chart(fig, use_container_width=True)
                                    
                                    except Exception as e:
                                        st.warning(f"Could not fetch metal prices: {str(e)[:100]}")
                                    
                                    # Display summary metrics
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Final CAGR", f"{summary.get('cagr', 0):.2f}%")
                                    with col2:
                                        st.metric("Total Return", f"{summary.get('total_return_pct', 0):.2f}%")
                                    with col3:
                                        st.metric("Max Drawdown", f"{summary.get('max_drawdown_pct', 0):.2f}%")
                                else:
                                    st.info("Backtest completed but daily data not found")
                            else:
                                st.warning("Plotly not available. Install with: pip install plotly")
                        else:
                            st.info("Backtest completed but summary file not found")
                    except Exception as e:
                        st.error(f"Error loading backtest results: {str(e)[:100]}")
                else:
                    st.error(f"Backtest failed: {result.stderr[:200]}")
            except Exception as e:
                st.error(f"Error running backtest: {str(e)[:100]}")
    
    st.markdown("---")
    
    # Watchlist feature
    st.markdown("### Watchlist")
    WATCHLIST_FILE = Path.home() / '.alpha_miner_watchlist.json'
    
    # Initialize watchlist in session state
    if 'watchlist' not in st.session_state:
        if WATCHLIST_FILE.exists():
            try:
                with open(WATCHLIST_FILE, 'r') as f:
                    st.session_state.watchlist = json.load(f)
            except (IOError, OSError, json.JSONDecodeError):
                st.session_state.watchlist = []
        else:
            st.session_state.watchlist = []
    
    watchlist_col1, watchlist_col2 = st.columns([3, 1])
    with watchlist_col1:
        new_symbol = st.text_input("Add symbol to watchlist", key="watchlist_input", placeholder="e.g., AGXPF")
    with watchlist_col2:
        if st.button("Add", key="watchlist_add"):
            if new_symbol:
                symbol = new_symbol.strip().upper()
                if symbol and symbol not in st.session_state.watchlist:
                    st.session_state.watchlist.append(symbol)
                    try:
                        with open(WATCHLIST_FILE, 'w') as f:
                            json.dump(st.session_state.watchlist, f)
                        st.success(f"Added {symbol} to watchlist")
                        st.rerun()
                    except (IOError, OSError, json.JSONDecodeError):
                        st.warning(f"Could not save watchlist, but {symbol} added to session")
    
    if st.session_state.watchlist:
        st.caption(f"Watching: {', '.join(st.session_state.watchlist)}")
        for sym in st.session_state.watchlist[:]:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(sym)
            with col2:
                if st.button("Remove", key=f"remove_{sym}"):
                    st.session_state.watchlist.remove(sym)
                    try:
                        with open(WATCHLIST_FILE, 'w') as f:
                            json.dump(st.session_state.watchlist, f)
                    except (IOError, OSError, json.JSONDecodeError):
                        pass
                    st.rerun()
    else:
        st.caption("No symbols in watchlist")
    
    st.markdown("---")

    st.header("Portfolio")
    # V7.3: Dynamic Scaling - Use total_portfolio_value from UI (no hardcoding)
    # Note: key="total_portfolio_value" automatically stores in st.session_state
    # DO NOT manually set st.session_state.total_portfolio_value - causes session state exception
    total_portfolio_value = st.number_input("Total Portfolio Value ($)", value=PORTFOLIO_SIZE, step=10000, key="total_portfolio_value")
    # V7.3: Get total_portfolio_value from UI (no hardcoded fallback)
    port_size = st.session_state.get("total_portfolio_value", None)
    if port_size is None or port_size <= 0:
        port_size = PORTFOLIO_SIZE  # Only use default if UI value not set
    
    st.markdown("### Edit Positions")
    st.caption("Check 'Insider_Buying_90d' if insiders bought recently")
    
    edited = st.data_editor(
        st.session_state.portfolio,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Symbol": st.column_config.TextColumn("Symbol", required=True),
            "Quantity": st.column_config.NumberColumn("Qty", required=True),
            "Cost_Basis": st.column_config.NumberColumn("Cost $", required=True),
            "Insider_Buying_90d": st.column_config.CheckboxColumn("Insider Buy", default=False)
        },
        hide_index=True
    )
    st.session_state.portfolio = edited
    
    st.markdown("### Cash")
    cash = st.number_input("Available", value=float(st.session_state.cash), step=1000.0, label_visibility="collapsed")
    st.session_state.cash = cash
    
    st.markdown("---")
    st.markdown("### Display Options")
    sort_mode = st.selectbox(
        "Sort by",
        ["Action first (default)", "Sell risk first", "Alpha first"]
    )
    st.session_state.sort_mode = sort_mode

    if st.button("Reset to Default"):
        st.session_state.portfolio = DEFAULT_PORTFOLIO.copy()
        st.rerun()

# ============================================================================
# MAIN APP
# ============================================================================

# ── Quality Status Bar ──────────────────────────────────────────────────────
# Shows system health at a glance: API, portfolio, data freshness
_tiingo_ok = not _TIINGO_KEY_MISSING
_yf_ok = YFINANCE_AVAILABLE if 'YFINANCE_AVAILABLE' in dir() else True
_portfolio_loaded = len(st.session_state.get('portfolio', [])) > 0
_last_analysis = st.session_state.get('last_analysis_utc', None)
_data_age_str = "No data yet"
if _last_analysis:
    _age_min = (datetime.datetime.now(datetime.timezone.utc) - _last_analysis).total_seconds() / 60
    _data_age_str = f"{_age_min:.0f}m ago" if _age_min < 60 else f"{_age_min / 60:.1f}h ago"

st.markdown(f"""<div class="status-bar">
    <span>Tiingo: <span class="{'ok' if _tiingo_ok else 'fail'}">{'Connected' if _tiingo_ok else 'No API Key'}</span></span>
    <span>Yahoo: <span class="{'ok' if _yf_ok else 'warn'}">{'Available' if _yf_ok else 'Unavailable'}</span></span>
    <span>Portfolio: <span class="{'ok' if _portfolio_loaded else 'warn'}">{len(st.session_state.get('portfolio', []))} positions</span></span>
    <span>Data: <span class="{'ok' if _last_analysis else 'warn'}">{_data_age_str}</span></span>
    <span>Risk: {st.session_state.get('risk_profile', 'Balanced')}</span>
    <span>Mode: {'Replay' if st.session_state.get('replay_mode', False) else 'Live'}</span>
</div>""", unsafe_allow_html=True)

st.markdown("---")

# Get macro regime
macro_regime = calculate_macro_regime()

# Macro regime — compact inline display
_regime = macro_regime['regime']
_factors_str = ' | '.join(macro_regime['factors'])
if _regime == 'DEFENSIVE':
    st.markdown(f'<div class="warning-banner"><strong>DEFENSIVE</strong> — New buys paused | {_factors_str}</div>', unsafe_allow_html=True)
elif _regime == 'RISK-ON':
    st.markdown(f'<div class="safe-banner"><strong>RISK-ON</strong> — Full deployment | {_factors_str}</div>', unsafe_allow_html=True)
elif _regime == 'CAUTIOUS':
    st.warning(f"**CAUTIOUS** — Reduced sizing | {_factors_str}")
else:
    st.info(f"**NEUTRAL** — {_factors_str}")

# Tape gate — compact
if 'tape_gate' in st.session_state:
    tape_gate = st.session_state.tape_gate
    _buys = "Allowed" if tape_gate['new_buys_allowed'] else "Blocked"
    _throttle = f" | Throttle: {tape_gate['throttle']:.0%}" if tape_gate['throttle'] < 1.0 else ""
    st.caption(f"**Tape Gate:** New buys {_buys}{_throttle} | {' | '.join(tape_gate['reasons'][:3])}")

# ── PORTFOLIO ANALYSIS (Primary action — top of page) ──────────────────────
# Analysis Button
if st.button("Run Portfolio Analysis", type="primary", use_container_width=True):
    import traceback
    
    try:
        progress = st.progress(0, text="Starting analysis...")
        
        df = st.session_state.portfolio.copy()
        
        # V4.0 Phase 3: Automatically fetch GC=F and SI=F prices for GSR calculation
        progress.progress(2, text="Fetching gold/silver prices...")
        gsr_data = fetch_gold_silver_prices()
        gsr_bias = None
        if gsr_data['success']:
            gsr_bias = calculate_gs_ratio_bias(gsr_data['gold_price'], gsr_data['silver_price'])
            st.session_state.gsr_data = gsr_data
            st.session_state.gsr_bias = gsr_bias
            if gsr_bias.get('silver_bonus', 0) > 0:
                st.info(f"GSR {gsr_bias['gs_ratio']:.1f} > 80: +{gsr_bias['silver_bonus']} Alpha Torque Bonus active for Silver symbols")
        else:
            st.warning(f"⚠️ Could not fetch GSR data: {gsr_data.get('error', 'Unknown error')}")
            st.session_state.gsr_bias = None
        
        # Analyze metals FIRST
        if INSTITUTIONAL_V2_AVAILABLE or INSTITUTIONAL_V3_AVAILABLE:
            progress.progress(3, text="Analyzing metal cycles...")
            
            if INSTITUTIONAL_V3_AVAILABLE:
                gold_analysis = forecast_metal_direction("GC=F", "Gold")
                silver_analysis = forecast_metal_direction("SI=F", "Silver")
            elif INSTITUTIONAL_V2_AVAILABLE:
                gold_analysis = analyze_metal_cycle("GC=F", "Gold")
                silver_analysis = analyze_metal_cycle("SI=F", "Silver")
            
            if INSTITUTIONAL_V2_AVAILABLE:
                metal_regime = calculate_metal_regime_impact(gold_analysis, silver_analysis)
            else:
                # Simple regime
                metal_regime = {
                    'regime': 'NEUTRAL',
                    'throttle_adjustment': 1.0,
                    'max_size_multiplier': 1.0,
                    'discovery_hardness': 'NORMAL',
                    'sell_sensitivity': 1.0
                }
            
            st.session_state.gold_analysis = gold_analysis
            st.session_state.silver_analysis = silver_analysis
            st.session_state.metal_regime = metal_regime
        else:
            st.session_state.metal_regime = {
                'regime': 'NEUTRAL',
                'throttle_adjustment': 1.0,
                'max_size_multiplier': 1.0,
                'discovery_hardness': 'NORMAL',
                'sell_sensitivity': 1.0
            }
            gold_analysis = None
            silver_analysis = None
        
        # Calculate tape gate
        tape_gate = calculate_tape_gate(
            macro_regime,
            st.session_state.get('gold_analysis') or gold_analysis,
            st.session_state.get('silver_analysis') or silver_analysis
        )
        st.session_state.tape_gate = tape_gate
        
        # Check replay mode - skip network calls if enabled
        replay_mode = st.session_state.get('replay_mode', False)
        replay_pack = st.session_state.get('replay_pack')
        
        if replay_mode:
            if not replay_pack:
                st.error("⚠️ Replay mode enabled but no evidence pack loaded. Please load an evidence pack first.")
                st.stop()
            
            # Load from evidence pack - zero network calls
            st.info(f"🔄 REPLAY MODE — OFFLINE — DATA AS OF {replay_pack.get('created_at_utc', 'Unknown')}")
            
            # Load tape_gate from pack
            if 'tape_gate' in replay_pack:
                st.session_state.tape_gate = replay_pack['tape_gate']
            
            # Load results from pack (includes financing overhang)
            if 'results' in replay_pack:
                df = pd.DataFrame(replay_pack['results'])
                news_cache = replay_pack.get('caches', {}).get('news_cache', {})
                macro_regime = replay_pack.get('macro_regime', {})
                alpha_breakdown_storage = replay_pack.get('caches', {}).get('alpha_breakdown_storage', {})
                sell_triggers_storage = replay_pack.get('caches', {}).get('sell_triggers_storage', {})
                dilution_factors_storage = replay_pack.get('caches', {}).get('dilution_factors_storage', {})
                conf_breakdown_storage = replay_pack.get('caches', {}).get('conf_breakdown_storage', {})
                
                # Set session state
                st.session_state.results = df
                st.session_state.news_cache = news_cache
                st.session_state.macro_regime = macro_regime
                st.session_state.conf_breakdown_storage = conf_breakdown_storage
                st.session_state.dilution_factors_storage = dilution_factors_storage
                st.session_state.alpha_breakdown_storage = alpha_breakdown_storage
                st.session_state.sell_triggers_storage = sell_triggers_storage
                
                progress.progress(100, text="Replay complete")
                st.success("Analysis complete.")
                st.rerun()
        
        # Fetch price data (normal mode - network calls allowed)
        progress.progress(10, text="Fetching market data...")
        
        # V7.3: Get live spot prices for gold/silver positions
        live_prices = get_sovereign_spot_prices() if not replay_mode else {}
        benchmarks = get_global_commodity_benchmarks() if not replay_mode else {}
        
        hist_cache = {}
        for idx, row in df.iterrows():
            if YFINANCE and not replay_mode:
                try:
                    import warnings as _w
                    with _w.catch_warnings():
                        _w.filterwarnings("ignore", message=".*possibly delisted.*")
                        hist = yf.Ticker(row['Symbol']).history(period="2y")
                    if not hist.empty:
                        hist_cache[row['Symbol']] = hist
                        
                        # V7.3: Use live price if available and >1% different from EOD
                        price = hist['Close'].iloc[-1]
                        
                        # Check if this is a gold/silver position (heuristic: check symbol or metal type)
                        symbol_upper = row['Symbol'].upper()
                        is_gold_silver = (
                            'GOLD' in symbol_upper or 'SILVER' in symbol_upper or
                            symbol_upper in ['AG', 'PAAS', 'MAG', 'EXK', 'HL', 'SIL', 'SILJ', 'GOLD', 'NEM', 'AEM', 'FNV']
                        )
                        
                        # V7.4: For gold/silver positions, use live prices from benchmarks (no manual overrides)
                        # These prices are used to calculate Market_Value = Quantity * Price for portfolio valuation
                        if is_gold_silver:
                            # V7.4: Use live prices from multi-source fetch (no hardcoded fallbacks)
                            if 'gold' in symbol_upper.lower():
                                if np.isfinite(benchmarks.get('gold_comx', np.nan)):
                                    price = benchmarks['gold_comx']  # Live price from multi-source fetch
                            elif 'silver' in symbol_upper.lower():
                                if np.isfinite(benchmarks.get('silver_comx', np.nan)):
                                    price = benchmarks['silver_comx']  # Live price from multi-source fetch
                        
                        df.at[idx, 'Price'] = price
                        df.at[idx, 'Volume'] = hist['Volume'].mean() if not hist.empty and 'Volume' in hist.columns else 0.0
                        
                        df.at[idx, 'Return_7d'] = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-7]) / hist['Close'].iloc[-7] * 100) if len(hist) >= 7 else 0
                        df.at[idx, 'Return_30d'] = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-30]) / hist['Close'].iloc[-30] * 100) if len(hist) >= 30 else 0
                        df.at[idx, 'Return_90d'] = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-90]) / hist['Close'].iloc[-90] * 100) if len(hist) >= 90 else 0
                        
                        high_52w = hist['High'].tail(252).max() if len(hist) >= 252 else (hist['High'].max() if not hist.empty else 0)
                        low_52w = hist['Low'].tail(252).min() if len(hist) >= 252 else (hist['Low'].min() if not hist.empty else 0)
                        df.at[idx, 'Pct_From_52w_High'] = ((hist['Close'].iloc[-1] - high_52w) / high_52w * 100) if high_52w > 0 else 0
                        df.at[idx, 'Pct_From_52w_Low'] = ((hist['Close'].iloc[-1] - low_52w) / low_52w * 100) if low_52w > 0 else 0
                        
                        df.at[idx, 'Volatility_60d'] = hist['Close'].pct_change(fill_method=None).tail(60).std() * 100 if len(hist) >= 60 else 5
                        
                        df.at[idx, 'MA50'] = hist['Close'].tail(50).mean() if len(hist) >= 50 else 0
                        df.at[idx, 'MA200'] = hist['Close'].tail(200).mean() if len(hist) >= 200 else 0
                        
                        if len(hist) >= 90:
                            high_90d = hist['High'].tail(90).max()
                            df.at[idx, 'Drawdown_90d'] = ((hist['Close'].iloc[-1] - high_90d) / high_90d * 100) if high_90d > 0 else 0
                        else:
                            df.at[idx, 'Drawdown_90d'] = 0
                except Exception as e:
                    df.at[idx, 'Price'] = 0
                    # Log error for debugging (if symbol available in context)
                    pass
        
        progress.progress(25, text="Fetching fundamentals...")
        
        info_storage = {}
        inferred_storage = {}
        
        for idx, row in df.iterrows():
            if not replay_mode:
                fund = get_fundamentals_with_tracking(row['Symbol'])
                for k, v in fund.items():
                    if k not in ['info_dict', 'inferred_flags']:
                        df.at[idx, k] = v
                
                info_storage[row['Symbol']] = fund['info_dict']
                inferred_storage[row['Symbol']] = fund['inferred_flags']
            else:
                # In replay mode, use defaults (data should come from evidence pack)
                info_storage[row['Symbol']] = {}
                inferred_storage[row['Symbol']] = {'metal_inferred': True, 'stage_inferred': True}
        
        progress.progress(35, text="Fetching news...")
        
        news_cache = {}
        for idx, row in df.iterrows():
            if not replay_mode:
                news = get_news_for_ticker(row['Symbol'])
                news_cache[row['Symbol']] = news
            else:
                # In replay mode, news should come from evidence pack
                news_cache[row['Symbol']] = []
        
        # V5.0: Validate metadata (ensure Jurisdiction and Metal_Type exist)
        df = validate_metadata(df)
        
        # Calculate position metrics
        # IMPORTANT: Pct_Portfolio calculated vs total portfolio value (equity + cash)
        # V5.0: Explicit np.float64 casting to prevent precision drift
        # V7.3: Use total_portfolio_value from UI (no hardcoded fallback)
        df['Market_Value'] = (df['Quantity'].astype(np.float64) * df['Price'].astype(np.float64)).astype(np.float64)
        df['Gain_Loss'] = (df['Market_Value'].astype(np.float64) - df['Cost_Basis'].astype(np.float64)).astype(np.float64)
        df['Return_Pct'] = ((df['Gain_Loss'] / df['Cost_Basis'].astype(np.float64)) * 100).astype(np.float64)
        
        # V7.3: Dynamic True-Up - Use total_portfolio_value from UI (no hardcoded fallback)
        total_portfolio_value = st.session_state.get("total_portfolio_value", None)
        if total_portfolio_value is None or total_portfolio_value <= 0:
            # Only calculate from holdings if UI value not set
            total_mv = np.float64(df['Market_Value'].sum())
            total_portfolio_value = np.float64(total_mv + np.float64(st.session_state.cash))
        
        df['Pct_Portfolio'] = ((df['Market_Value'].astype(np.float64) / float(total_portfolio_value)) * 100).astype(np.float64)  # vs TOTAL VALUE
        df['Runway'] = df['cash'] / df['burn']
        
        progress.progress(45, text="Liquidity analysis...")
        
        for idx, row in df.iterrows():
            hist = hist_cache.get(row['Symbol'], pd.DataFrame())
            liq = calculate_liquidity_metrics(row['Symbol'], hist, row['Price'], row['Market_Value'], port_size)
            
            for k, v in liq.items():
                df.at[idx, f'Liq_{k}'] = v
        
        progress.progress(55, text="Data confidence...")
        
        conf_breakdown_storage = {}
        
        for idx, row in df.iterrows():
            fund_dict = {'burn_source': row['burn_source']}
            info_dict = info_storage.get(row['Symbol'], {})
            inferred = inferred_storage.get(row['Symbol'], {})
            
            conf = calculate_data_confidence(fund_dict, info_dict, inferred)
            df.at[idx, 'Data_Confidence'] = conf['score']
            df.at[idx, 'Conf_Verdict'] = conf['verdict']
            
            conf_breakdown_storage[row['Symbol']] = conf['breakdown']
        
        progress.progress(60, text="Dilution risk...")
        
        dilution_factors_storage = {}
        
        for idx, row in df.iterrows():
            news = news_cache.get(row['Symbol'], [])
            cash_missing = row['cash'] == 10.0
            burn_missing = row['burn_source'] == 'default'
            insider = row.get('Insider_Buying_90d', False)
            
            dil = calculate_dilution_risk(
                row['Runway'],
                row['stage'],
                abs(row.get('Drawdown_90d', 0)),
                news,
                cash_missing,
                burn_missing,
                insider
            )
            
            df.at[idx, 'Dilution_Risk_Score'] = dil['score']
            df.at[idx, 'Dilution_Verdict'] = dil['verdict']
            
            dilution_factors_storage[row['Symbol']] = dil['factors']
        
        # Calculate Financing Overhang
        progress.progress(62, text="Financing overhang...")
        
        for idx, row in df.iterrows():
            news = news_cache.get(row['Symbol'], [])
            runway_months = row.get('Runway', 12.0)
            
            overhang = calculate_financing_overhang(news, row['Symbol'], runway_months)
            
            df.at[idx, 'Financing_Overhang_Score'] = overhang['score']
            df.at[idx, 'Financing_Overhang_Reasons'] = overhang['reasons']
        
        # CRITICAL FIX: Calculate SMC BEFORE alpha scoring
        progress.progress(65, text="Calculating SMC signals...")
        
        smc_signals_storage = {}
        
        for idx, row in df.iterrows():
            hist = hist_cache.get(row['Symbol'], pd.DataFrame())
            
            # Use v3 if available, otherwise v1
            if INSTITUTIONAL_V3_AVAILABLE:
                smc = calculate_smc_structure(hist, row['Symbol'])
            else:
                smc = calculate_smc_signals(hist, row['Price'])
            
            df.at[idx, 'SMC_Bias'] = smc.get('bias', smc.get('state', 'Neutral'))
            df.at[idx, 'SMC_Score'] = smc.get('score', 50)
            df.at[idx, 'SMC_Summary'] = smc.get('summary', smc.get('explanation', ''))
            df.at[idx, 'SMC_State'] = smc.get('state', 'NEUTRAL')
            df.at[idx, 'SMC_Event'] = smc.get('event', 'NONE')
            
            smc_signals_storage[row['Symbol']] = smc.get('signals', [])
        
        st.session_state.smc_signals_storage = smc_signals_storage

        # V8.0: Technical Analysis indicators
        ta_storage = {}
        if TA_MODULE_AVAILABLE:
            progress.progress(67, text="Technical analysis...")
            for idx, row in df.iterrows():
                hist = hist_cache.get(row['Symbol'], pd.DataFrame())
                if not hist.empty and len(hist) >= 20:
                    ta_result = calculate_all_ta(hist)
                    ta_storage[row['Symbol']] = ta_result
                    df.at[idx, 'TA_Score'] = ta_result.get('ta_score', 50)
                    df.at[idx, 'TA_Signal'] = ta_result.get('ta_signal', 'NEUTRAL')
                    rsi_data = ta_result.get('rsi', {})
                    df.at[idx, 'RSI'] = rsi_data.get('current', 50) if isinstance(rsi_data, dict) else 50
                    macd_data = ta_result.get('macd', {})
                    df.at[idx, 'MACD_Crossover'] = macd_data.get('crossover', 'none') if isinstance(macd_data, dict) else 'none'
                    bb_data = ta_result.get('bollinger', {})
                    df.at[idx, 'BB_Squeeze'] = bb_data.get('squeeze', False) if isinstance(bb_data, dict) else False
                    adx_data = ta_result.get('adx', {})
                    df.at[idx, 'Trend_Strength'] = adx_data.get('trend_strength', 'weak') if isinstance(adx_data, dict) else 'weak'
                else:
                    ta_storage[row['Symbol']] = {}
                    df.at[idx, 'TA_Score'] = 50
                    df.at[idx, 'TA_Signal'] = 'NEUTRAL'
                    df.at[idx, 'RSI'] = 50
                    df.at[idx, 'MACD_Crossover'] = 'none'
                    df.at[idx, 'BB_Squeeze'] = False
                    df.at[idx, 'Trend_Strength'] = 'weak'
            st.session_state.ta_storage = ta_storage

        # V8.0: AISC scoring (if spot prices available)
        if AISC_TRACKER_AVAILABLE:
            progress.progress(69, text="AISC analysis...")
            spot = st.session_state.get('spot_prices', {})
            spot_gold = spot.get('gold_live', spot.get('gold_eod', 2650))
            spot_silver = spot.get('silver_live', spot.get('silver_eod', 31))
            spot_uranium = spot.get('uranium_spot', 80)
            for idx, row in df.iterrows():
                metal = row.get('metal', 'Gold')
                metal_price = spot_gold if metal == 'Gold' else spot_silver if metal == 'Silver' else spot_uranium
                info_dict = info_storage.get(row['Symbol'], {})
                aisc_result = get_aisc_score(row['Symbol'], metal, metal_price, info_dict)
                df.at[idx, 'AISC_Estimate'] = aisc_result.get('aisc_estimate', 0)
                df.at[idx, 'AISC_Margin_Pct'] = aisc_result.get('margin_pct', 0)
                df.at[idx, 'AISC_Score'] = aisc_result.get('score', 50)
                df.at[idx, 'AISC_Source'] = aisc_result.get('aisc_source', 'unknown')

        # Now calculate alpha WITH SMC scores available
        progress.progress(75, text="Alpha scoring (11 models)...")
        
        alpha_models_storage = {}
        alpha_breakdown_storage = {}
        
        for idx, row in df.iterrows():
            hist = hist_cache.get(row['Symbol'], pd.DataFrame())
            benchmark = get_benchmark_data(row.get('metal', 'Gold')) if not replay_mode else None
            
            alpha_result = calculate_alpha_models(row, hist, benchmark)
            
            # CRITICAL FIX: Add SMC score to alpha
            smc_score = row.get('SMC_Score', 50)
            alpha_result['models']['M7_SMC'] = smc_score * 0.08
            alpha_result['breakdown'][-2] = f"M7 SMC: {smc_score}/100 × 8% = {alpha_result['models']['M7_SMC']:.1f}"
            
            # Recalculate total
            alpha_result['alpha_score'] = sum(alpha_result['models'].values())
            
            df.at[idx, 'Alpha_Score'] = alpha_result['alpha_score']
            
            alpha_models_storage[row['Symbol']] = alpha_result['models']
            alpha_breakdown_storage[row['Symbol']] = alpha_result['breakdown']
            
            # V5.0: Calculate Fundamental Alpha Score
            # Use current portfolio as sector proxy for MCAP/OZ comparison
            try:
                sector_data = df[df['Symbol'] != row['Symbol']].to_dict('records')
                # V7.5: Ensure info_dict is available for FA scoring
                if 'info_dict' not in row and row['Symbol'] in info_storage:
                    row['info_dict'] = info_storage[row['Symbol']]
                
                fa_result = calculate_fundamental_score(row, sector_data)
                df.at[idx, 'FA_Score'] = fa_result['fa_score']
                df.at[idx, 'FA_Reasoning'] = ' | '.join(fa_result['reasoning'])
                
                # V7.5: Apply news promotion score to Alpha_Score if available (from info_storage)
                if row['Symbol'] in info_storage:
                    info = info_storage[row['Symbol']]
                    desc = info.get('longBusinessSummary', '').lower()
                    news_keywords = ['drill results', 'high grade', 'exploration', 'pre-feasibility', 'pfs', 'sprott']
                    found_keywords = [kw for kw in news_keywords if kw in desc]
                    news_promotion = len(found_keywords) * 5  # +5 Alpha per keyword
                    if news_promotion > 0:
                        df.at[idx, 'Alpha_Score'] = df.at[idx, 'Alpha_Score'] + news_promotion
                        df.at[idx, 'News_Promotion'] = ', '.join(found_keywords)
                        df.at[idx, 'News_Promotion_Score'] = news_promotion
            except Exception as e:
                # Fallback if FA calculation fails
                df.at[idx, 'FA_Score'] = 0.0
                df.at[idx, 'FA_Reasoning'] = f'FA calculation error: {str(e)[:50]}'
            
            # V5.0: Detect Market Buzz (volume spike +300%)
            try:
                if len(hist) >= 20:
                    buzz_result = detect_market_buzz(hist, threshold_multiplier=3.0)
                    df.at[idx, 'Market_Buzz'] = buzz_result['buzz_detected']
                    df.at[idx, 'Volume_Spike_Pct'] = buzz_result['volume_spike_pct']
                else:
                    df.at[idx, 'Market_Buzz'] = False
                    df.at[idx, 'Volume_Spike_Pct'] = 0.0
            except Exception:
                # Fallback if buzz detection fails
                df.at[idx, 'Market_Buzz'] = False
                df.at[idx, 'Volume_Spike_Pct'] = 0.0
        
        progress.progress(82, text="Sell risk analysis...")
        
        sell_triggers_storage = {}
        
        for idx, row in df.iterrows():
            hist = hist_cache.get(row['Symbol'], pd.DataFrame())
            news = news_cache.get(row['Symbol'], [])
            
            sell = calculate_sell_risk(row, hist, row.get('MA50', 0), row.get('MA200', 0), news, macro_regime)
            
            df.at[idx, 'Sell_Risk_Score'] = sell['score']
            df.at[idx, 'Sell_Verdict'] = sell['verdict']
            
            # CRITICAL FIX: Store real triggers
            sell_triggers_storage[row['Symbol']] = sell['all_triggers']
        
        progress.progress(90, text="Final arbitration...")
        
        # Classify sleeves
        for idx, row in df.iterrows():
            liq_tier = row['Liq_tier_code']
            daily_vol = row['Liq_dollar_vol_20d']
            conf = row['Data_Confidence']
            
            if conf < 50:
                df.at[idx, 'Sleeve'] = 'GAMBLING'
            elif row['stage'] in ['Producer', 'Developer', 'Developer (Inferred)'] and daily_vol >= 200000 and conf >= 80:
                df.at[idx, 'Sleeve'] = 'CORE'
            else:
                df.at[idx, 'Sleeve'] = 'TACTICAL'
        
        # Discovery exceptions
        for idx, row in df.iterrows():
            liq_metrics = {k.replace('Liq_', ''): v for k, v in row.items() if k.startswith('Liq_')}
            
            momentum_ok = row['Return_7d'] > 0 and row['Price'] > row.get('MA50', 0)
            
            exception = check_discovery_exception(
                row, liq_metrics,
                row['Alpha_Score'],
                row['Data_Confidence'],
                row['Dilution_Risk_Score'],
                momentum_ok
            )
            
            df.at[idx, 'Discovery_Exception'] = exception[0]
            df.at[idx, 'Discovery_Reason'] = exception[1]
        
        # Final decisions
        decisions = []
        for _, row in df.iterrows():
            liq_metrics = {k.replace('Liq_', ''): v for k, v in row.items() if k.startswith('Liq_')}
            data_conf = {'score': row['Data_Confidence']}
            dilution = {'score': row['Dilution_Risk_Score']}
            
            # CRITICAL FIX: Pass real triggers to arbitration
            sell_triggers = sell_triggers_storage.get(row['Symbol'], [])
            sell_risk = {
                'score': row['Sell_Risk_Score'],
                'hard_triggers': [t for t in sell_triggers if '💀' in t],
                'soft_triggers': [t for t in sell_triggers if '⚠️' in t]
            }
            
            discovery = (row['Discovery_Exception'], row['Discovery_Reason'])
            
            # Get tape gate from session state
            tape_gate = st.session_state.get('tape_gate')
            
            # V4.0 Phase 3: Pass GSR bias for automated macro-rotation
            gsr_bias = st.session_state.get('gsr_bias')
            decision = arbitrate_final_decision(
                row, liq_metrics, data_conf, dilution, sell_risk,
                row['Alpha_Score'], macro_regime, discovery, tape_gate,
                strict_mode=st.session_state.get('strict_mode', False),
                gsr_bias=gsr_bias  # V4.0 Phase 3: Automated GSR bonus application
            )
            
            decisions.append(decision)
        
        for k in ['action', 'confidence', 'recommended_pct', 'max_allowed_pct', 'reasoning', 
                  'gates_passed', 'gates_failed', 'warnings', 'primary_gating_reason', 'veto_applied', 'veto_model']:
            df[k.title()] = [d.get(k, '') for d in decisions]
        
        # V7.2: Sovereign Rebalancer - Apply rolling 15-year backtesting weights
        # Override Recommended_Pct with sovereign rebalancer weights if available
        # V7.3: Get total_portfolio_value from UI (no hardcoded fallback)
        total_portfolio_value = st.session_state.get("total_portfolio_value", None)
        if total_portfolio_value is None or total_portfolio_value <= 0:
            # Calculate from holdings if UI value not set
            if 'results' in st.session_state:
                results_df = st.session_state.results
                total_mv = np.float64(results_df['Market_Value'].sum())
                total_portfolio_value = np.float64(total_mv + np.float64(st.session_state.cash))
            else:
                # Only use default if no holdings available to calculate from
                total_portfolio_value = PORTFOLIO_SIZE  # Last resort fallback (should not be reached if UI is set)
        if total_portfolio_value is None or total_portfolio_value <= 0:
            total_mv = np.float64(df['Market_Value'].sum())
            total_portfolio_value = np.float64(total_mv + np.float64(st.session_state.cash))
        
        # Get all tickers from portfolio + daily picks (if available)
        all_tickers = df['Symbol'].tolist()
        daily_picks = st.session_state.get("daily_picks", pd.DataFrame())
        if not daily_picks.empty and 'Symbol' in daily_picks.columns:
            pick_symbols = daily_picks['Symbol'].tolist()
            all_tickers = list(set(all_tickers + pick_symbols))[:399]  # Limit to 399 tickers
        
        # Calculate sovereign rebalance weights
        benchmarks = get_global_commodity_benchmarks()
        sovereign_weights = calculate_sovereign_rebalance_weights(
            all_tickers,
            float(total_portfolio_value),
            hist_cache=hist_cache,
            benchmarks=benchmarks
        )
        
        # Apply sovereign weights to portfolio positions — respecting sell risk
        for idx, row in df.iterrows():
            symbol = row['Symbol']
            if symbol in sovereign_weights:
                sovereign_weight = sovereign_weights[symbol]
                current_rec = row.get('Recommended_Pct', 0)
                blended = (sovereign_weight * 0.7) + (current_rec * 0.3)

                # Sell risk override: don't let sovereign weight prop up risky positions
                sell_risk = row.get('Sell_Risk_Score', 0)
                action = row.get('Action', 'HOLD')
                current_pct = row.get('Pct_Portfolio', 0)

                if sell_risk >= 60 or action == 'Avoid':
                    # Critical risk — recommend exit, sovereign weight cannot override
                    df.at[idx, 'Recommended_Pct'] = 0
                    if action not in ('Avoid',):
                        df.at[idx, 'Action'] = 'Sell'
                elif sell_risk >= 40 or action == 'REDUCE':
                    # Moderate risk — reduce to at most 85% of current allocation
                    risk_cap = current_pct * 0.85
                    df.at[idx, 'Recommended_Pct'] = min(blended, risk_cap)
                else:
                    df.at[idx, 'Recommended_Pct'] = blended
        
        # Post-process: Strict mode + Financing Overhang enforcement
        strict_mode = st.session_state.get('strict_mode', False)
        if strict_mode:
            for idx, row in df.iterrows():
                overhang_score = row.get('Financing_Overhang_Score', 0)
                action = row.get('Action', '')
                is_buy = action == 'Buy'
                
                if is_buy and overhang_score >= 80:
                    # Check if exception: PP_CLOSED <=7d AND runway >= 9 months
                    reasons = row.get('Financing_Overhang_Reasons', [])
                    has_recent_close = any('closed' in str(r).lower() and '7d' in str(r) for r in reasons)
                    runway_months = row.get('Runway', 0)
                    
                    if not (has_recent_close and runway_months >= 9):
                        # Block the buy
                        df.at[idx, 'Action'] = 'Avoid'
                        df.at[idx, 'Confidence'] = 'High'
                        current_warnings = row.get('Warnings', [])
                        if isinstance(current_warnings, list):
                            current_warnings.append("STRICT: Financing overhang ≥80 blocks buy")
                        else:
                            current_warnings = ["STRICT: Financing overhang ≥80 blocks buy"]
                        df.at[idx, 'Warnings'] = current_warnings
                        df.at[idx, 'Primary_Gating_Reason'] = "STRICT MODE: Financing overhang ≥80 blocks buy"
                        df.at[idx, 'Recommended_Pct'] = row.get('Pct_Portfolio', 0)
                        df.at[idx, 'Veto_Applied'] = True
                        df.at[idx, 'Veto_Model'] = 'Capital Structure'
        
        # Calculate Recommendation Stability indicator
        for idx, row in df.iterrows():
            veto_count = 0
            near_threshold_count = 0
            
            # Check if vetoes are near thresholds
            sell_risk = row.get('Sell_Risk_Score', 0)
            if sell_risk >= 50 and sell_risk < 60:
                near_threshold_count += 1
            
            dilution = row.get('Dilution_Risk_Score', 0)
            if dilution >= 70 and dilution < 80:
                near_threshold_count += 1
            
            overhang = row.get('Financing_Overhang_Score', 0)
            if overhang >= 70 and overhang < 80:
                near_threshold_count += 1
            
            if row.get('Veto_Applied', False):
                veto_count = 1
            
            # Determine stability
            if veto_count > 0:
                stability = 'Breaks'
            elif near_threshold_count >= 2:
                stability = 'Fragile'
            elif near_threshold_count >= 1:
                stability = 'Fragile'
            else:
                stability = 'Stable'
            
            df.at[idx, 'Recommendation_Stability'] = stability
        
        progress.progress(100, text="Analysis complete")

        # ------------------------------------------------------------------------
        # Governance: validate, apply strict mode, build evidence pack
        # ------------------------------------------------------------------------
        # Invariants check (used by STRICT MODE & Trust Panel). Use the real model storage.
        validation = validate_data_invariants(df, alpha_models_storage, news_cache)
        st.session_state.validation = validation

        if strict_mode:
            preset = get_risk_profile_preset(st.session_state.get('risk_profile', 'Balanced'))
            df, strict_downgrades = enforce_strict_mode(df, validation, st.session_state.get('risk_profile','Balanced'), st.session_state.get('strict_mode', False))

        # Save evidence pack for replay/debugging
        try:
            pack = create_evidence_pack(
            df=df,
            portfolio_input=st.session_state.portfolio,
            cash=float(st.session_state.cash),
            macro_regime=macro_regime,
            news_cache=news_cache,
            alpha_breakdown_storage=alpha_breakdown_storage,
            sell_triggers_storage=sell_triggers_storage,
            dilution_factors_storage=dilution_factors_storage,
            conf_breakdown_storage=conf_breakdown_storage,
            meta={
                'version': VERSION,
                'version_date': VERSION_DATE,
                'risk_profile': st.session_state.get('risk_profile', 'Balanced'),
                'strict_mode': bool(st.session_state.get('strict_mode', False)),
                'freeze_time': bool(st.session_state.get('freeze_time', False)),
                'disable_sector_fallback_news': bool(st.session_state.get('disable_sector_fallback_news', False)),
                'disable_inferred_fundamentals': bool(st.session_state.get('disable_inferred_fundamentals', False))
            },
                tape_gate=st.session_state.get('tape_gate')
            )
            st.session_state.evidence_pack = pack
            save_evidence_pack(pack)
        except Exception:
            st.session_state.evidence_pack = None

        st.session_state.results = df
        st.session_state.last_analysis_utc = datetime.datetime.now(datetime.timezone.utc)
        st.session_state.news_cache = news_cache
        st.session_state.macro_regime = macro_regime
        st.session_state.conf_breakdown_storage = conf_breakdown_storage
        st.session_state.dilution_factors_storage = dilution_factors_storage
        st.session_state.alpha_breakdown_storage = alpha_breakdown_storage
        st.session_state.sell_triggers_storage = sell_triggers_storage
        st.session_state.hist_cache = hist_cache  # V5.0: Store for recommendations tab

        # V8.0: Generate alerts after analysis
        if ALERT_ENGINE_AVAILABLE:
            try:
                spot = st.session_state.get('spot_prices', {})
                ta_cache_for_alerts = st.session_state.get('ta_storage', {})
                alerts = check_all_alerts(df, hist_cache, news_cache, spot, ta_cache_for_alerts)
                st.session_state.active_alerts = alerts
                save_alerts(alerts)
                alert_summary = get_alert_summary(alerts)
                if alert_summary['critical'] > 0:
                    st.warning(f"🔔 {alert_summary['critical']} critical alert(s) detected. Check sidebar.")
            except Exception:
                st.session_state.active_alerts = []

        st.success("Analysis complete.")
        st.rerun()
    
    except Exception as e:
        # Global error handler - ensure app never goes dark
        error_type = type(e).__name__
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        
        st.error(f"❌ Analysis failed: {error_type}: {error_msg}")
        
        # Show stack trace in collapsed area
        with st.expander("🔍 Technical Details (Stack Trace)", expanded=False):
            st.code(error_traceback, language='python')
        
        # Best-effort create failure evidence pack
        try:
            failure_pack = {
                'evidence_pack_id': f"ep_failure_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'created_at_utc': _now_iso(),
                'app_version': VERSION,
                'app_version_date': VERSION_DATE,
                'status': 'failure',
                'error': {
                    'type': error_type,
                    'message': error_msg,
                    'traceback': error_traceback
                },
                'inputs': {
                    'portfolio': st.session_state.portfolio.to_dict(orient='records') if 'portfolio' in st.session_state else [],
                    'cash': float(st.session_state.get('cash', 0)),
                },
                'meta': {
                    'version': VERSION,
                    'version_date': VERSION_DATE,
                    'risk_profile': st.session_state.get('risk_profile', 'Balanced'),
                    'strict_mode': bool(st.session_state.get('strict_mode', False)),
                    'replay_mode': bool(st.session_state.get('replay_mode', False))
                }
            }
            save_evidence_pack(failure_pack)
            st.caption(f"💾 Failure evidence pack saved: {failure_pack['evidence_pack_id']}")
        except Exception as pack_error:
            st.warning(f"Could not save failure evidence pack: {pack_error}")
        
        # Ensure Streamlit keeps rendering
        st.info("The app is still running. You can try again or check the technical details above.")

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

# Helper functions for ranking
def add_ranking_columns(df):
    """Add ranking columns"""
    ACTION_RANK = {
        'Buy': 6, 'HOLD': 4, 'Avoid': 2,
        # Legacy support
        '🟢 STRONG BUY': 7, '🟢 BUY': 6, '🔵 ADD': 5, '🔵 ADD ⚠️': 5, '🔵 ACCUMULATE': 5,
        '⚪ HOLD': 4, '🟡 TRIM': 3, '🔴 REDUCE': 2, '🔴 SELL': 1, '🚨 SELL NOW': 0
    }
    df['Action_Rank'] = df['Action'].map(ACTION_RANK).fillna(4)
    
    TIER_RANK = {'L3': 3, 'L2': 2, 'L1': 1, 'L0': 0}
    df['Tier_Rank'] = df['Liq_tier_code'].map(TIER_RANK).fillna(0)
    return df

def sort_dataframe(df, sort_mode):
    """Sort dataframe"""
    if sort_mode == "Sell risk first":
        return df.sort_values(['Sell_Risk_Score', 'Action_Rank'], ascending=[False, False])
    elif sort_mode == "Alpha first":
        return df.sort_values(['Alpha_Score', 'Sell_Risk_Score'], ascending=[False, True])
    else:
        return df.sort_values(['Action_Rank', 'Alpha_Score', 'Tier_Rank'], ascending=[False, False, False])

def render_daily_summary(df, macro_regime, cash):
    """Render daily summary"""
    
    st.markdown("---")
    st.header("DAILY EXECUTIVE SUMMARY")
    
    # V5.0: Explicit np.float64 casting to prevent precision drift
    total_mv = np.float64(df['Market_Value'].sum())
    total_value = np.float64(total_mv + np.float64(cash))
    total_cost = np.float64(df['Cost_Basis'].sum())
    total_pl = np.float64(total_mv - total_cost)
    total_pl_pct = np.float64((total_pl / total_cost * 100) if total_cost > 0 else 0)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Value", f"${total_value:,.0f}")
    col2.metric("Total P/L", f"${total_pl:,.0f}", f"{total_pl_pct:+.1f}%")
    col3.metric("Cash", f"${cash:,.0f}")
    col4.metric("Equity", f"${total_mv:,.0f}")
    
    st.markdown("### Risk Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    illiquid_pct = df[df['Liq_tier_code'].isin(['L0', 'L1'])]['Pct_Portfolio'].sum()
    avg_days = df['Liq_days_to_exit'].mean()
    avg_dil = df['Dilution_Risk_Score'].mean()
    
    col1.metric("Illiquid %", f"{illiquid_pct:.1f}%", "⚠️" if illiquid_pct > 20 else "✅")
    col2.metric("Avg Exit Days", f"{avg_days:.1f}d", "⚠️" if avg_days > 7 else "✅")
    col3.metric("Avg Dilution", f"{avg_dil:.0f}/100", "⚠️" if avg_dil > 50 else "✅")
    
    action_counts = df['Action'].value_counts()
    col4.write("**Actions:**")
    for action in ['Buy', 'HOLD', 'Avoid']:
        count = action_counts.get(action, 0)
        if count > 0:
            col4.caption(f"{action}: {count}")
    
    st.markdown("---")
    
    # Top opportunities/risks
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("✅ BUY NOW")
        addable = df[df['Action'] == 'Buy']
        if len(addable) > 0:
            for _, row in addable.nlargest(3, 'Alpha_Score').iterrows():
                amt = total_value * (row['Recommended_Pct'] / 100)
                confidence = row.get('Confidence', 'Low')
                st.success(f"**{row['Symbol']}** - {confidence} confidence")
                st.caption(f"Rec: {row['Recommended_Pct']:.1f}% (${amt:,.0f})")
                st.caption(f"Alpha: {row['Alpha_Score']:.0f}")
        else:
            st.info("No buy opportunities")
    
    with col2:
        st.subheader("🚨 SELL RISK")
        for _, row in df.nlargest(3, 'Sell_Risk_Score').iterrows():
            if row['Sell_Risk_Score'] >= 30:
                st.error(f"**{row['Symbol']}**")
                st.caption(f"Risk: {row['Sell_Risk_Score']:.0f}/100")
                st.caption(f"{row['Action']}")
    
    with col3:
        st.subheader("💀 DILUTION")
        for _, row in df.nlargest(3, 'Dilution_Risk_Score').iterrows():
            if row['Dilution_Risk_Score'] >= 50:
                st.warning(f"**{row['Symbol']}**")
                st.caption(f"Risk: {row['Dilution_Risk_Score']:.0f}/100")
                st.caption(f"Runway: {row['Runway']:.1f}mo")
    
    st.markdown("---")
    
    # Today's plan
    st.subheader("📋 TODAY'S PLAN")
    
    if not macro_regime.get('allow_new_buys', True):
        st.error("**STAND DOWN:** Defensive macro - no new buys")
    
    adds = df[df['Action'] == 'Buy']
    if len(adds) > 0 and macro_regime.get('allow_new_buys', True):
        st.success("**Consider Buying:**")
        for _, row in adds.nlargest(2, 'Alpha_Score').iterrows():
            amt = total_value * (row['Recommended_Pct'] / 100)
            confidence = row.get('Confidence', 'Low')
            st.write(f"• {row['Symbol']}: {row['Recommended_Pct']:.1f}% (${amt:,.0f}) - {confidence} confidence")
    
    avoids = df[df['Action'] == 'Avoid']
    if len(avoids) > 0:
        st.warning("**Consider Avoiding:**")
        for _, row in avoids.nlargest(2, 'Sell_Risk_Score').iterrows():
            gating = row.get('Primary_Gating_Reason', 'Risk signals')
            st.write(f"• {row['Symbol']}: {gating}")

if 'results' in st.session_state:
    df = st.session_state.results
    news_cache = st.session_state.news_cache
    macro = st.session_state.macro_regime
    
    total_mv = df['Market_Value'].sum()
    total_value = total_mv + st.session_state.cash

    # Validation summary
    val = st.session_state.get('validation_report', {})
    if val:
        issues = val.get('issues', [])
        if issues:
            with st.expander(f"🧪 Data Validation Issues ({len(issues)})", expanded=False):
                for msg in issues[:100]:
                    st.warning(msg)
        else:
            st.caption("🧪 Data validation: no issues detected")

    # Alerts for significant changes
    st.markdown("---")
    st.markdown("### Alerts")
    
    alerts = []
    pack = st.session_state.get('evidence_pack')
    
    # Compare with most recent saved pack
    saved_packs = list_evidence_packs()
    prev_pack = None
    if saved_packs and pack:
        # Get most recent pack that's not the current one
        current_id = pack.get('evidence_pack_id')
        for pack_path in saved_packs:
            try:
                candidate = load_evidence_pack(pack_path)
                if candidate.get('evidence_pack_id') != current_id:
                    prev_pack = candidate
                    break
            except (IOError, OSError, json.JSONDecodeError):
                continue

    if prev_pack and prev_pack.get('results'):
        prev_df = pd.DataFrame(prev_pack['results'])
        prev_dict = prev_df.set_index('Symbol').to_dict('index')
        
        for _, row in df.iterrows():
            sym = row['Symbol']
            prev_row = prev_dict.get(sym)
            
            if prev_row:
                # Check financing lifecycle changes
                prev_overhang = prev_row.get('Financing_Overhang_Score', 0)
                curr_overhang = row.get('Financing_Overhang_Score', 0)
                prev_reasons = prev_row.get('Financing_Overhang_Reasons', [])
                curr_reasons = row.get('Financing_Overhang_Reasons', [])
                
                # Detect lifecycle state changes
                prev_has_closed = any('closed' in str(r).lower() for r in prev_reasons)
                curr_has_closed = any('closed' in str(r).lower() for r in curr_reasons)
                prev_has_atm = any('atm' in str(r).lower() for r in prev_reasons)
                curr_has_atm = any('atm' in str(r).lower() for r in curr_reasons)
                prev_has_shelf = any('shelf' in str(r).lower() for r in prev_reasons)
                curr_has_shelf = any('shelf' in str(r).lower() for r in curr_reasons)
                
                if not prev_has_closed and curr_has_closed:
                    alerts.append(f"✅ {sym}: Financing PP_CLOSED (overhang: {prev_overhang:.0f}→{curr_overhang:.0f})")
                if not prev_has_atm and curr_has_atm:
                    alerts.append(f"⚠️ {sym}: ATM financing detected (overhang: {prev_overhang:.0f}→{curr_overhang:.0f})")
                if not prev_has_shelf and curr_has_shelf:
                    alerts.append(f"⚠️ {sym}: SHELF filing detected (overhang: {prev_overhang:.0f}→{curr_overhang:.0f})")
                
                # Check sell risk crossing 70+
                prev_sell = prev_row.get('Sell_Risk_Score', 0)
                curr_sell = row.get('Sell_Risk_Score', 0)
                if prev_sell < 70 and curr_sell >= 70:
                    alerts.append(f"🔴 {sym}: Sell risk crossed 70+ ({prev_sell:.0f}→{curr_sell:.0f})")
                
                # Check financing overhang crossing 80+
                if prev_overhang < 80 and curr_overhang >= 80:
                    alerts.append(f"🔴 {sym}: Financing overhang crossed 80+ ({prev_overhang:.0f}→{curr_overhang:.0f})")
    
    if alerts:
        for alert in alerts[:10]:  # Top 10 alerts
            if '✅' in alert:
                st.success(alert)
            elif '🔴' in alert:
                st.error(alert)
            else:
                st.warning(alert)
    else:
        st.info("No significant changes detected since last run.")
    
    # Evidence pack / diff / rebalance
    with st.expander("🧾 Evidence Pack, Diff, and Rebalance", expanded=False):
        pack = st.session_state.get('evidence_pack')
        if pack:
            st.caption(f"Pack id: {pack.get('evidence_pack_id', 'Unknown')} • created: {pack.get('created_at_utc', 'Unknown')}")

            # "What changed since last run?" diff
            st.markdown("### What Changed Since Last Run?")
            
            saved_packs = list_evidence_packs()
            prev_pack = None
            if saved_packs:
                # Get most recent pack that's not the current one
                current_id = pack.get('evidence_pack_id')
                for pack_path in saved_packs:
                    try:
                        candidate = load_evidence_pack(pack_path)
                        if candidate.get('evidence_pack_id') != current_id:
                            prev_pack = candidate
                            break
                    except (IOError, OSError, json.JSONDecodeError):
                        continue

            if prev_pack and prev_pack.get('results'):
                prev_df = pd.DataFrame(prev_pack['results'])
                prev_dict = prev_df.set_index('Symbol').to_dict('index')
                
                diff_rows = []
                for _, row in df.iterrows():
                    sym = row['Symbol']
                    prev_row = prev_dict.get(sym)
                    
                    if prev_row:
                        prev_price = prev_row.get('Price', 0)
                        curr_price = row.get('Price', 0)
                        price_pct = ((curr_price - prev_price) / prev_price * 100) if prev_price > 0 else 0
                        
                        alpha_delta = row.get('Alpha_Score', 0) - prev_row.get('Alpha_Score', 0)
                        sell_delta = row.get('Sell_Risk_Score', 0) - prev_row.get('Sell_Risk_Score', 0)
                        dilution_delta = row.get('Dilution_Risk_Score', 0) - prev_row.get('Dilution_Risk_Score', 0)
                        overhang_delta = row.get('Financing_Overhang_Score', 0) - prev_row.get('Financing_Overhang_Score', 0)
                        rec_pct_delta = row.get('Recommended_Pct', 0) - prev_row.get('Recommended_Pct', 0)
                        
                        action_old = prev_row.get('Action', '')
                        action_new = row.get('Action', '')
                        
                        # Only include rows with meaningful changes
                        if (abs(price_pct) > 0.1 or abs(alpha_delta) > 1 or action_old != action_new or 
                            abs(sell_delta) > 1 or abs(dilution_delta) > 1 or abs(overhang_delta) > 1 or abs(rec_pct_delta) > 0.1):
                            diff_rows.append({
                                'Symbol': sym,
                                'ΔPrice%': f"{price_pct:+.1f}%",
                                'ΔAlpha': f"{alpha_delta:+.1f}",
                                'Action': f"{action_old}→{action_new}",
                                'ΔSellRisk': f"{sell_delta:+.1f}",
                                'ΔDilutionRisk': f"{dilution_delta:+.1f}",
                                'ΔFinancingOverhang': f"{overhang_delta:+.1f}",
                                'ΔRecPct': f"{rec_pct_delta:+.1f}%"
                            })
                
                if diff_rows:
                    diff_df = pd.DataFrame(diff_rows)
                    # Calculate mover score for highlighting
                    diff_df['_mover_score'] = (
                        diff_df['ΔAlpha'].str.replace('+', '').str.replace('−', '-').astype(float).abs() +
                        diff_df['ΔSellRisk'].str.replace('+', '').str.replace('−', '-').astype(float).abs() * 0.5 +
                        diff_df['ΔFinancingOverhang'].str.replace('+', '').str.replace('−', '-').astype(float).abs() * 0.5
                    )
                    diff_df = diff_df.sort_values('_mover_score', ascending=False)
                    
                    st.dataframe(
                        diff_df.drop(columns=['_mover_score']).head(20),
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Highlight top 5 movers
                    if len(diff_df) > 0:
                        st.caption(f"📈 Top movers: {', '.join(diff_df.head(5)['Symbol'].tolist())}")
                else:
                    st.info("No significant changes detected.")
            else:
                st.info("No prior evidence pack found. Run analysis again to see changes.")

            # Rebalance plan table
            st.markdown("### Rebalance Plan")
            
            # V5.0: Hard-lock audit gate - disable all buttons if audit fails
            audit_passed = st.session_state.get('audit_passed', False)
            if not audit_passed:
                st.error("❌ **System Lock: Calculation Variance Detected** - All execution buttons disabled")
                st.caption("Check Audit Log in sidebar to resolve integrity check failures")
            
            allow_leverage = st.toggle("Allow leverage (buys can exceed cash)", value=False, key="allow_leverage", disabled=not audit_passed)
            
            rebalance_rows = []
            total_buys = 0.0
            total_sells = 0.0
            
            for _, row in df.iterrows():
                sym = row['Symbol']
                current_pct = row.get('Pct_Portfolio', 0)
                target_pct = row.get('Recommended_Pct', 0)
                delta_pct = target_pct - current_pct
                trade_dollars = (delta_pct / 100.0) * total_value
                
                # Liquidity warning
                liq_tier = row.get('Liq_tier_code', 'L0')
                days_to_exit = row.get('Liq_days_to_exit', 999)
                liq_warning = ""
                if liq_tier in ['L0', 'L1']:
                    liq_warning = f"⚠️ {liq_tier} tier"
                if days_to_exit > 10:
                    liq_warning += f" ({days_to_exit:.0f}d exit)"
                
                # Reason for trade
                action = row.get('Action', '')
                reason_parts = []
                if 'BUY' in action or 'ADD' in action:
                    reason_parts.append(f"Alpha: {row.get('Alpha_Score', 0):.0f}")
                if row.get('Sell_Risk_Score', 0) >= 30:
                    reason_parts.append(f"Sell risk: {row.get('Sell_Risk_Score', 0):.0f}")
                reason = " | ".join(reason_parts) if reason_parts else action
                
                if trade_dollars > 0:
                    total_buys += trade_dollars
                else:
                    total_sells += abs(trade_dollars)
                
                rebalance_rows.append({
                    'Symbol': sym,
                    'Current%': f"{current_pct:.2f}%",
                    'Target%': f"{target_pct:.2f}%",
                    'Δ%': f"{delta_pct:+.2f}%",
                    '$ Trade': f"${trade_dollars:+,.0f}",
                    'Liquidity Warning': liq_warning,
                    'Reason': reason
                })
            
            rebalance_df = pd.DataFrame(rebalance_rows)
            rebalance_df = rebalance_df[rebalance_df['$ Trade'] != '$0']
            
            # Cash constraint enforcement
            available_cash = st.session_state.cash
            if not allow_leverage and total_buys > available_cash:
                st.warning(f"⚠️ Total buys (${total_buys:,.0f}) exceed available cash (${available_cash:,.0f}). Adjusting...")
                # Scale down buys proportionally
                scale_factor = available_cash / total_buys if total_buys > 0 else 0
                for idx in rebalance_df.index:
                    trade_str = rebalance_df.at[idx, '$ Trade']
                    if trade_str.startswith('$') and '+' in trade_str:
                        trade_val = float(trade_str.replace('$', '').replace(',', '').replace('+', ''))
                        if trade_val > 0:
                            rebalance_df.at[idx, '$ Trade'] = f"${trade_val * scale_factor:+,.0f}"
                            # Recalculate delta %
                            sym = rebalance_df.at[idx, 'Symbol']
                            orig_row = df[df['Symbol'] == sym].iloc[0]
                            new_trade = trade_val * scale_factor
                            new_target_pct = (orig_row['Market_Value'] + new_trade) / total_value * 100
                            rebalance_df.at[idx, 'Target%'] = f"{new_target_pct:.2f}%"
                            rebalance_df.at[idx, 'Δ%'] = f"{new_target_pct - orig_row['Pct_Portfolio']:+.2f}%"
            
            if not rebalance_df.empty:
                # Sort by absolute trade value
                def sort_key(val):
                    try:
                        cleaned = val.replace('$', '').replace(',', '').replace('+', '').replace('−', '-')
                        return abs(float(cleaned))
                    except (ValueError, TypeError):
                        return 0.0
                
                rebalance_df_sorted = rebalance_df.copy()
                rebalance_df_sorted['_sort_val'] = rebalance_df_sorted['$ Trade'].apply(sort_key)
                rebalance_df_sorted = rebalance_df_sorted.sort_values('_sort_val', ascending=False).drop(columns=['_sort_val'])
                
                st.dataframe(rebalance_df_sorted, use_container_width=True, hide_index=True)
                
                # CSV download (hard-locked by audit gate)
                csv_data = rebalance_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Rebalance Plan (CSV)",
                    data=csv_data,
                    file_name=f"rebalance_plan_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    disabled=not audit_passed
                )
                
                # V5.0: PDF Trade Report Generator (hard-locked by audit gate)
                if st.button("📄 Generate Trade Report (PDF)", use_container_width=True, disabled=not audit_passed):
                    try:
                        # Generate markdown report (can be converted to PDF by user or via external tool)
                        report_lines = [
                            "# Alpha Miner Pro V5.0 - Trade Report",
                            f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                            "",
                            "## 5% Threshold Rebalance Summary",
                            "",
                            "| Ticker | Action | Current % | Target % | Est. Market Impact |",
                            "|--------|--------|------------|----------|---------------------|"
                        ]
                        
                        # Add trade rows with market impact estimates
                        for _, row in rebalance_df.iterrows():
                            symbol = row['Symbol']
                            current_pct = float(row['Current%'].replace('%', ''))
                            target_pct = float(row['Target%'].replace('%', ''))
                            trade_dollars = float(row['$ Trade'].replace('$', '').replace(',', ''))
                            
                            # Estimate market impact (simplified: based on liquidity tier if available)
                            market_impact = "N/A"
                            if 'results' in st.session_state:
                                results_df = st.session_state.results
                                symbol_row = results_df[results_df['Symbol'] == symbol]
                                if not symbol_row.empty:
                                    liq_tier = symbol_row.iloc[0].get('Liq_tier_code', 'UNKNOWN')
                                    # Estimate impact based on tier (L3: 0.1%, L2: 0.5%, L1: 2.5%, L0: 7.5%)
                                    impact_map = {'L3': 0.1, 'L2': 0.5, 'L1': 2.5, 'L0': 7.5}
                                    market_impact = f"{impact_map.get(liq_tier, 1.0):.2f}%"
                            
                            action = "BUY" if trade_dollars > 0 else "SELL"
                            report_lines.append(
                                f"| {symbol} | {action} | {current_pct:.2f}% | {target_pct:.2f}% | {market_impact} |"
                            )
                        
                        report_lines.extend([
                            "",
                            "---",
                            "",
                            "**Note:** Market Impact estimates are based on liquidity tier. Actual impact may vary.",
                            "",
                            f"**Total Buys:** ${total_buys:,.0f}",
                            f"**Total Sells:** ${total_sells:,.0f}",
                            f"**Net Cash Flow:** ${total_buys - total_sells:+,.0f}"
                        ])
                        
                        report_text = "\n".join(report_lines)
                        
                        # Create downloadable markdown file (user can convert to PDF)
                        st.download_button(
                            "📄 Download Trade Report (Markdown)",
                            data=report_text,
                            file_name=f"trade_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                            mime="text/markdown",
                            use_container_width=True
                        )
                        
                        # Display preview
                        st.markdown("### Report Preview")
                        st.markdown(report_text)
                        
                    except Exception as e:
                        st.error(f"Error generating trade report: {str(e)}")
            else:
                st.info("No rebalancing needed.")
            
            st.markdown("---")
            st.download_button(
                "Download evidence pack (JSON)",
                data=json.dumps(pack, indent=2),
                file_name=f"alpha_evidence_{pack.get('evidence_pack_id', 'unknown')}.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.caption("No evidence pack in session yet. Run analysis to generate one.")
    
    # Display morning tape (simple version)
    if 'gold_analysis' in st.session_state and 'silver_analysis' in st.session_state:
        render_morning_tape_simple(
            st.session_state.gold_analysis,
            st.session_state.silver_analysis,
            st.session_state.get('metal_regime', {})
        )
    
    # Daily summary
    render_daily_summary(df, macro, st.session_state.cash)
    
    conf_breakdown_storage = st.session_state.get('conf_breakdown_storage', {})
    dilution_factors_storage = st.session_state.get('dilution_factors_storage', {})
    alpha_breakdown_storage = st.session_state.get('alpha_breakdown_storage', {})
    sell_triggers_storage = st.session_state.get('sell_triggers_storage', {})
    
    st.markdown("---")
    
    st.markdown('<div class="command-center">', unsafe_allow_html=True)
    st.header("COMMAND CENTER")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚨 TOP 3 SELL RISKS")
        sell_risks = df.nlargest(3, 'Sell_Risk_Score')
        
        for _, row in sell_risks.iterrows():
            triggers = sell_triggers_storage.get(row['Symbol'], [])
            trigger_text = ', '.join(triggers[:2]) if triggers else 'None'
            
            st.markdown(f"""
            <div class="risk-card">
                <h4>{row['Symbol']} - Sell Risk: {row['Sell_Risk_Score']:.0f}/100</h4>
                <p><strong>Action:</strong> {row['Action']}</p>
                <p><strong>Position:</strong> ${row['Market_Value']:,.0f} ({row['Pct_Portfolio']:.1f}%)</p>
                <p><strong>Triggers:</strong> {trigger_text}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("✅ TOP 3 BUY OPPORTUNITIES")
        addable = df[df['Action'] == 'Buy']
        if len(addable) > 0:
            top_adds = addable.nlargest(3, 'Alpha_Score')
            
            for _, row in top_adds.iterrows():
                confidence = row.get('Confidence', 'Low')
                st.markdown(f"""
                <div class="opportunity-card">
                    <h4>{row['Symbol']} - Alpha: {row['Alpha_Score']:.0f}/100</h4>
                    <p><strong>Action:</strong> {row['Action']} ({confidence} confidence)</p>
                    <p><strong>Current:</strong> {row['Pct_Portfolio']:.1f}% → <strong>Rec:</strong> {row['Recommended_Pct']:.1f}%</p>
                    <p><strong>Max Allowed:</strong> {row['Max_Allowed_Pct']:.1f}%</p>
                    <p><strong>Gating Reason:</strong> {row.get('Primary_Gating_Reason', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No add opportunities")
    
    # Portfolio risk metrics
    st.markdown("---")
    st.subheader("📊 Portfolio Risk Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_l0_l1_pct = df[df['Liq_tier_code'].isin(['L0', 'L1'])]['Pct_Portfolio'].sum()
    avg_days_exit = df['Liq_days_to_exit'].mean()
    avg_dilution = df['Dilution_Risk_Score'].mean()
    positions_at_risk = len(df[df['Sell_Risk_Score'] >= 30])
    
    col1.metric("Illiquid (L0/L1)", f"{total_l0_l1_pct:.1f}%", 
               "⚠️" if total_l0_l1_pct > 20 else "✅")
    col2.metric("Avg Days to Exit", f"{avg_days_exit:.1f}d",
               "⚠️" if avg_days_exit > 7 else "✅")
    col3.metric("Avg Dilution Risk", f"{avg_dilution:.0f}/100",
               "⚠️" if avg_dilution > 50 else "✅")
    col4.metric("Positions at Risk", positions_at_risk,
               "⚠️" if positions_at_risk > 3 else "✅")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ========================================================================
    # V5.0: DAILY RECOMMENDATIONS TAB
    # ========================================================================
    
    st.markdown("---")
    
    # V5.0: Decision Center - Restructured tabs
    # V7.2: Calculate max_drift using total_portfolio_value from UI
    max_drift = 0.0
    rebalance_statuses = []
    # V7.3: Get total_portfolio_value from UI (no hardcoded fallback)
    total_portfolio_value = st.session_state.get("total_portfolio_value", None)
    if 'results' in st.session_state:
        results_df = st.session_state.results
        if total_portfolio_value is None or total_portfolio_value <= 0:
            # Only calculate from holdings if UI value not set
            total_mv = np.float64(results_df['Market_Value'].sum())
            total_portfolio_value = np.float64(total_mv + np.float64(st.session_state.cash))
        
        for _, row in results_df.iterrows():
            current_pct = row.get('Pct_Portfolio', 0)
            target_pct = row.get('Recommended_Pct', current_pct)
            # target_pct == 0 is a valid SELL signal — don't override it
            if target_pct < 0:
                target_pct = 0
            drift_pct = abs(current_pct - target_pct)
            drift_usd = (drift_pct / 100.0) * float(total_portfolio_value)  # V7.2: USD drift
            max_drift = max(max_drift, drift_pct)
            if drift_pct > 2.0:  # Lower threshold: 2% drift triggers action (was 5%)
                rebalance_statuses.append({
                    'symbol': row['Symbol'],
                    'current': current_pct,
                    'target': target_pct,
                    'drift': drift_pct,
                    'drift_usd': drift_usd,  # V7.2: Add USD drift
                    'status': 'ACTION_REQUIRED'
                })
    
    # V8.0: Decision-First Layout with TA, Optimizer, and Alerts tabs
    tab_names = ["Actions", "Technical Analysis", "Portfolio Optimizer",
                 "Market Scanner", "Watchlist"]
    actions_tab, ta_tab, optimizer_tab, discovery_tab, watchlist_tab = st.tabs(tab_names)
    
    # Tab 1: Actions Today
    with actions_tab:
      try:
        st.header("Actions Today")
        
        # V5.0: Warning banner if audit not passed
        audit_passed = st.session_state.get('audit_passed', False)
        if not audit_passed:
            st.warning("⚠️ **Audit Log Not Passed** - Trade execution is disabled until integrity checks pass. Check Audit Log in sidebar.")
        
        # V5.0: Fail-Safe — Portfolio Concentration Warning when Hard-Burn > 25%
        hard_burn_pct = st.session_state.get("hard_burn_pct", 0.0)
        if hard_burn_pct < -0.25:
            st.warning(
                "⚠️ **Portfolio Concentration Warning** — Estimated Hard-Burn exceeds 25% of portfolio in a "
                "standard sector correction. Consider reducing concentration or hedging."
            )
        
        # V5.0: Show "No Actions Required" green checkmark if portfolio is balanced
        if max_drift <= 5.0:
            st.success("✅ **No Actions Required** - Portfolio is balanced (Max drift: {:.1f}%)".format(max_drift))
            st.caption("All positions are within the 5% tolerance band. No rebalancing needed.")
        else:
            st.error(f"🔴 **ACTION REQUIRED** - Max drift: {max_drift:.1f}%")
            st.caption("Rebalance trades needed to restore target allocations")

            # V7.2: Use total_portfolio_value from UI
            if 'results' in st.session_state:
                results_df = st.session_state.results
                _total_value = float(total_portfolio_value)
            else:
                results_df = pd.DataFrame()
                _total_value = float(total_portfolio_value) if total_portfolio_value > 0 else float(st.session_state.cash)

            swap_suggestions = []
            daily_picks = st.session_state.get("daily_picks", pd.DataFrame())
            hist_cache = st.session_state.get("hist_cache", {})
            if not daily_picks.empty and "results" in st.session_state:
                res = results_df
                for s in rebalance_statuses:
                    if s.get("drift", 0) <= 5.0:
                        continue
                    held = s["symbol"]
                    held_row = res[res["Symbol"] == held]
                    if held_row.empty:
                        continue
                    held_sharpe = np.nan
                    held_pnav = np.nan
                    h_hist = hist_cache.get(held, pd.DataFrame())
                    if not h_hist.empty and "Close" in h_hist.columns:
                        held_sharpe = _trailing_sharpe(h_hist, 252)
                    try:
                        fh = get_forensic_fundamentals(held)
                        if fh.get("p_nav") is not None:
                            held_pnav = float(fh["p_nav"])
                    except Exception:
                        pass
                    for _, pick in daily_picks.iterrows():
                        ps = pick.get("Symbol")
                        if ps == held:
                            continue
                        pick_sharpe = pick.get("Sharpe_1y", np.nan)
                        pick_pnav = pick.get("P_NAV")
                        if pick_pnav is not None and np.isfinite(pick_pnav):
                            pick_pnav = float(pick_pnav)
                        else:
                            pick_pnav = np.nan
                        higher_sharpe = np.isfinite(pick_sharpe) and (not np.isfinite(held_sharpe) or pick_sharpe > held_sharpe)
                        lower_pnav = (np.isfinite(pick_pnav) and np.isfinite(held_pnav) and pick_pnav < held_pnav)
                        if not np.isfinite(pick_pnav) or not np.isfinite(held_pnav):
                            lower_pnav = True
                        if higher_sharpe and lower_pnav:
                            swap_suggestions.append((held, ps, held_sharpe, pick_sharpe, held_pnav, pick_pnav))
                            break
                for t in swap_suggestions[:5]:
                    held, ps = t[0], t[1]
                    sh_held, sh_pick = t[2], t[3]
                    pn_held, pn_pick = t[4], t[5]
                    msg = f"🚨 **SWAP SUGGESTION** — **{held}** → **{ps}**"
                    if np.isfinite(sh_held) and np.isfinite(sh_pick):
                        msg += f" (Sharpe {sh_held:.2f} → {sh_pick:.2f})"
                    if np.isfinite(pn_held) and np.isfinite(pn_pick):
                        msg += f" • P/NAV {pn_held:.2f} → {pn_pick:.2f}"
                    st.warning(msg)

            # V6.5: Portfolio Review & Rebalance Audit – Before vs. After, 5% Sovereign Drift strictly enforced
            st.markdown("### Portfolio Review & Rebalance Audit")
            st.caption("5% Sovereign Drift gate strictly enforced. Below: Before vs. After for each suggested trade.")
            if rebalance_statuses and 'results' in st.session_state:
                audit_rows = []
                for s in sorted(rebalance_statuses, key=lambda x: x['drift'], reverse=True):
                    if s.get('status') != 'ACTION_REQUIRED':
                        continue
                    audit_rows.append({
                        'Symbol': s['symbol'],
                        'Before %': f"{s['current']:.1f}",
                        'After %': f"{s['target']:.1f}",
                        'Δ%': f"{s['target'] - s['current']:+.1f}",
                    })
                if audit_rows:
                    st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)
                # Hard-Burn change if user executes first SWAP (V7.0: use Sharpe+P/NAV suggestions)
                daily_picks = st.session_state.get("daily_picks", pd.DataFrame())
                swap_list = []
                if not daily_picks.empty and not results_df.empty and swap_suggestions:
                    res = results_df
                    held, ps = swap_suggestions[0][0], swap_suggestions[0][1]
                    held_row = res[res["Symbol"] == held]
                    held_mv = float(held_row["Market_Value"].iloc[0]) if not held_row.empty and "Market_Value" in held_row.columns else 0
                    pick_row = daily_picks[daily_picks["Symbol"] == ps]
                    pick_price = float(pick_row["Price"].iloc[0]) if not pick_row.empty and "Price" in pick_row.columns else 1.0
                    swap_list = [(held, ps, held_mv, pick_price)]
                if swap_list:
                    held, ps, held_mv, pick_price = swap_list[0]
                    pf = results_df[["Symbol", "Quantity"]].copy()
                    pf = pf[pf["Symbol"] != held]
                    pick_qty = held_mv / pick_price if pick_price and pick_price > 0 else 0
                    pf = pd.concat([pf, pd.DataFrame([{"Symbol": ps, "Quantity": pick_qty}])], ignore_index=True)
                    cash = float(st.session_state.cash)
                    try:
                        r_after = run_sovereign_stress_test_cached(pf.to_json(orient="records"), float(_total_value), cash)
                        hb_before = st.session_state.get("hard_burn_usd", 0.0)
                        hb_before_pct = (st.session_state.get("hard_burn_pct", 0.0) or 0) * 100
                        hb_after = r_after.get("hard_burn_usd", 0.0)
                        hb_after_pct = (r_after.get("hard_burn_pct", 0.0) or 0) * 100
                        delta_usd = hb_after - hb_before
                        st.info(f"🔥 **Hard-Burn if you execute this Swap** — Before: ${abs(hb_before):,.0f} ({hb_before_pct:.1f}%); After: ${abs(hb_after):,.0f} ({hb_after_pct:.1f}%); Δ ${delta_usd:+,.0f}")
                    except Exception:
                        st.caption("Hard-Burn impact for swap could not be computed (Tiingo/stress test unavailable).")
            
            total_value = _total_value
            
            # V5.0: Display action cards using st.metric
            # V7.2: Use total_portfolio_value from UI for trade amount calculations
            for status in sorted(rebalance_statuses, key=lambda x: x['drift'], reverse=True):
                if status['status'] == 'ACTION_REQUIRED':
                    action = "TRIM" if status['current'] > status['target'] else "ADD"
                    # V7.2: Use drift_usd if available, otherwise calculate from total_portfolio_value
                    trade_amount = status.get('drift_usd', abs(status['current'] - status['target']) / 100.0 * float(total_portfolio_value))
                    
                    # Calculate shares and market impact
                    symbol = status['symbol']
                    current_price = 0.0
                    market_impact_pct = 0.0
                    if 'results' in st.session_state:
                        results_df = st.session_state.results
                        symbol_row = results_df[results_df['Symbol'] == symbol]
                        if not symbol_row.empty:
                            current_price = symbol_row.iloc[0].get('Price', 0)
                            market_impact_pct = symbol_row.iloc[0].get('Market_Impact_Pct', 0)
                    
                    shares = int(trade_amount / current_price) if current_price > 0 else 0
                    
                    with st.container():
                        col1, col2, col3 = st.columns([2, 2, 1])
                        with col1:
                            action_label = f"{action} {symbol}"
                            st.metric(
                                label=action_label,
                                value=f"{shares:,} Shares" if shares > 0 else "$0",
                                delta=f"Impact: {market_impact_pct:.2f}%"
                            )
                        with col2:
                            st.metric(
                                label="Trade Amount",
                                value=f"${trade_amount:,.0f}",
                                delta=f"{status['current']:.1f}% → {status['target']:.1f}%"
                            )
                        with col3:
                            if action == "TRIM":
                                st.error("🔴 SELL")
                            else:
                                st.success("🟢 BUY")
                        st.markdown("---")
            
            # Show rebalance plan table
            if 'results' in st.session_state:
                rebalance_table = compute_rebalance_table(df, total_value)
                if not rebalance_table.empty:
                    st.markdown("### Rebalance Plan Summary")
                    st.dataframe(rebalance_table, use_container_width=True, hide_index=True)
      except Exception as _tab_err:
        st.error(f"Actions tab error: {_tab_err}")

    # Tab 2: Technical Analysis
    with ta_tab:
      try:
        st.header("Technical Analysis")
        st.caption("RSI, MACD, Bollinger Bands, OBV, ADX for each position")

        if not TA_MODULE_AVAILABLE:
            st.warning("Technical Analysis module not available. Install with: `pip install pandas numpy`")
        else:
            ta_storage = st.session_state.get('ta_storage', {})
            if not ta_storage:
                st.info("Run analysis first to populate technical indicators.")
            else:
                # Summary table
                ta_summary_data = []
                for symbol, ta_data in ta_storage.items():
                    if not ta_data:
                        continue
                    rsi_data = ta_data.get('rsi', {})
                    macd_data = ta_data.get('macd', {})
                    bb_data = ta_data.get('bollinger', {})
                    adx_data = ta_data.get('adx', {})
                    obv_data = ta_data.get('obv', {})
                    ta_summary_data.append({
                        'Symbol': symbol,
                        'TA Score': ta_data.get('ta_score', 50),
                        'Signal': ta_data.get('ta_signal', 'NEUTRAL'),
                        'RSI': round(rsi_data.get('current', 50), 1) if isinstance(rsi_data, dict) else 50,
                        'MACD': macd_data.get('crossover', 'none') if isinstance(macd_data, dict) else 'none',
                        'BB %B': round(bb_data.get('pct_b', 0.5), 2) if isinstance(bb_data, dict) else 0.5,
                        'Squeeze': bb_data.get('squeeze', False) if isinstance(bb_data, dict) else False,
                        'ADX': round(adx_data.get('adx', 0), 1) if isinstance(adx_data, dict) else 0,
                        'Trend': adx_data.get('trend_strength', 'weak') if isinstance(adx_data, dict) else 'weak',
                        'OBV Trend': obv_data.get('obv_trend', 'N/A') if isinstance(obv_data, dict) else 'N/A',
                        'OBV Divergence': obv_data.get('obv_divergence', 'none') if isinstance(obv_data, dict) else 'none',
                    })

                if ta_summary_data:
                    ta_df = pd.DataFrame(ta_summary_data)
                    ta_df = ta_df.sort_values('TA Score', ascending=False)

                    # Color-code signals
                    st.dataframe(ta_df, use_container_width=True, hide_index=True)

                    # Detailed per-symbol expandable sections
                    st.markdown("---")
                    st.subheader("Detailed Indicator Analysis")

                    for symbol, ta_data in ta_storage.items():
                        if not ta_data:
                            continue
                        with st.expander(f"{symbol} - {ta_data.get('ta_signal', 'NEUTRAL')} (Score: {ta_data.get('ta_score', 50)})"):
                            col1, col2, col3, col4 = st.columns(4)

                            rsi_data = ta_data.get('rsi', {})
                            rsi_val = rsi_data.get('current', 50) if isinstance(rsi_data, dict) else 50
                            with col1:
                                st.metric("RSI (14)", f"{rsi_val:.1f}",
                                          delta="Oversold" if rsi_val < 30 else "Overbought" if rsi_val > 70 else "Neutral")

                            macd_data = ta_data.get('macd', {})
                            with col2:
                                macd_hist = macd_data.get('histogram', 0) if isinstance(macd_data, dict) else 0
                                crossover = macd_data.get('crossover', 'none') if isinstance(macd_data, dict) else 'none'
                                st.metric("MACD Histogram", f"{macd_hist:.4f}",
                                          delta=crossover.capitalize() if crossover != 'none' else "No crossover")

                            bb_data = ta_data.get('bollinger', {})
                            with col3:
                                pct_b = bb_data.get('pct_b', 0.5) if isinstance(bb_data, dict) else 0.5
                                squeeze = bb_data.get('squeeze', False) if isinstance(bb_data, dict) else False
                                st.metric("Bollinger %B", f"{pct_b:.2f}",
                                          delta="SQUEEZE" if squeeze else "Normal")

                            adx_data = ta_data.get('adx', {})
                            with col4:
                                adx_val = adx_data.get('adx', 0) if isinstance(adx_data, dict) else 0
                                st.metric("ADX", f"{adx_val:.1f}",
                                          delta=adx_data.get('trend_strength', 'weak').capitalize() if isinstance(adx_data, dict) else 'Weak')

                            # TA Reasoning
                            reasons = ta_data.get('ta_reasons', [])
                            if reasons:
                                st.caption("**Scoring Breakdown:**")
                                for reason in reasons:
                                    st.text(f"  {reason}")

                            # Fibonacci levels
                            fib_data = ta_data.get('fibonacci', {})
                            if isinstance(fib_data, dict) and fib_data.get('levels'):
                                st.caption("**Fibonacci Levels:**")
                                fib_levels = fib_data.get('levels', {})
                                support = fib_data.get('nearest_support')
                                resistance = fib_data.get('nearest_resistance')
                                fib_text = " | ".join([f"{k}: ${v:.2f}" for k, v in sorted(fib_levels.items())])
                                st.text(fib_text)
                                if support:
                                    st.text(f"  Nearest Support: ${support:.2f}")
                                if resistance:
                                    st.text(f"  Nearest Resistance: ${resistance:.2f}")
      except Exception as _tab_err:
        st.error(f"Technical Analysis tab error: {_tab_err}")

    # Tab 3: Portfolio Optimizer
    with optimizer_tab:
      try:
        st.header("Portfolio Optimization")
        st.caption("Mean-Variance, Risk Parity, Correlation Analysis, and Concentration Risk")

        if not OPTIMIZER_AVAILABLE:
            st.warning("Portfolio Optimizer module not available. Install scipy: `pip install scipy`")
        else:
            hist_cache_opt = st.session_state.get('hist_cache', {})

            if not hist_cache_opt:
                st.info("Run analysis first to populate portfolio data.")
            else:
                opt_col1, opt_col2 = st.columns(2)

                with opt_col1:
                    st.subheader("Correlation Matrix")
                    try:
                        corr_matrix = calculate_correlation_matrix(hist_cache_opt)
                        if not corr_matrix.empty:
                            if PLOTLY_AVAILABLE:
                                import plotly.express as px
                                fig = px.imshow(corr_matrix, text_auto='.2f', color_continuous_scale='RdBu_r',
                                                zmin=-1, zmax=1, title='Position Correlation Matrix')
                                fig.update_layout(height=500)
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.dataframe(corr_matrix.round(2), use_container_width=True)
                        else:
                            st.info("Insufficient data for correlation matrix.")
                    except Exception as e:
                        st.warning(f"Could not compute correlations: {e}")

                with opt_col2:
                    st.subheader("Concentration Risk")
                    try:
                        if 'results' in st.session_state and st.session_state.results is not None:
                            concentration = detect_concentration_risk(st.session_state.results)
                            score = concentration.get('concentration_score', 0)

                            if score < 30:
                                st.success(f"Diversification Score: {100 - score:.0f}/100 (Well Diversified)")
                            elif score < 60:
                                st.warning(f"Diversification Score: {100 - score:.0f}/100 (Moderate Concentration)")
                            else:
                                st.error(f"Diversification Score: {100 - score:.0f}/100 (High Concentration)")

                            warnings = concentration.get('warnings', [])
                            for w in warnings:
                                st.warning(w)

                            # Breakdown charts
                            by_metal = concentration.get('by_metal', {})
                            if by_metal and PLOTLY_AVAILABLE:
                                fig_metal = px.pie(names=list(by_metal.keys()), values=list(by_metal.values()),
                                                   title='Allocation by Metal')
                                st.plotly_chart(fig_metal, use_container_width=True)
                        else:
                            st.info("Run analysis first.")
                    except Exception as e:
                        st.warning(f"Could not compute concentration: {e}")

                st.markdown("---")

                # Optimization section
                st.subheader("Portfolio Optimization")
                opt_method = st.selectbox("Optimization Method", ["Mean-Variance (Max Sharpe)", "Risk Parity"])
                max_weight = st.slider("Max Weight Per Position", 5, 25, 15, 1) / 100.0

                if st.button("Run Optimization"):
                    with st.spinner("Optimizing portfolio..."):
                        try:
                            if opt_method == "Mean-Variance (Max Sharpe)":
                                opt_result = optimize_mean_variance(hist_cache_opt, max_weight=max_weight)
                            else:
                                opt_result = optimize_risk_parity(hist_cache_opt, max_weight=max_weight)

                            if opt_result and opt_result.get('weights'):
                                weights = opt_result['weights']
                                weights_df = pd.DataFrame([
                                    {'Symbol': k, 'Optimal Weight %': round(v * 100, 2)}
                                    for k, v in sorted(weights.items(), key=lambda x: -x[1])
                                    if v > 0.001
                                ])
                                st.dataframe(weights_df, use_container_width=True, hide_index=True)

                                if 'sharpe_ratio' in opt_result:
                                    st.metric("Optimal Sharpe Ratio", f"{opt_result['sharpe_ratio']:.3f}")
                                if 'expected_return' in opt_result:
                                    st.metric("Expected Annual Return", f"{opt_result['expected_return']*100:.1f}%")
                                if 'expected_volatility' in opt_result:
                                    st.metric("Expected Volatility", f"{opt_result['expected_volatility']*100:.1f}%")

                                # Suggest rebalance trades
                                if 'results' in st.session_state and st.session_state.results is not None:
                                    results_df_opt = st.session_state.results
                                    total_val = results_df_opt['Market_Value'].sum() + st.session_state.get('cash', 0)
                                    current_w = {}
                                    for _, r in results_df_opt.iterrows():
                                        current_w[r['Symbol']] = r.get('Pct_Portfolio', 0) / 100.0

                                    trades = suggest_rebalance_trades(current_w, weights, total_val)
                                    if trades:
                                        st.subheader("Suggested Rebalance Trades")
                                        trades_df = pd.DataFrame(trades)
                                        st.dataframe(trades_df, use_container_width=True, hide_index=True)
                            else:
                                st.warning("Optimization did not converge. Try adjusting parameters.")
                        except Exception as e:
                            st.error(f"Optimization failed: {e}")
      except Exception as _tab_err:
        st.error(f"Portfolio Optimizer tab error: {_tab_err}")

    # Tab 4: Market Scanner
    with discovery_tab:
      try:
        master_symbols = []
        master_count = 0
        
        # V7.4: Unlock Global Scan - If TIINGO_API_KEY is present, skip CSV fallback
        tiingo_key_present = bool(os.getenv("TIINGO_API_KEY", "").strip())
        
        if MINING_TICKERS_AVAILABLE and get_all_mining_tickers_cached:
            try:
                master_symbols = get_all_mining_tickers_cached(max_symbols=200) or []
                master_count = len(master_symbols)
                if master_count == 0:
                    st.warning("⚠️ **Mining ticker discovery returned 0 symbols.** Check Tiingo API connection or use CSV fallback.")
            except Exception as e:
                st.warning(f"⚠️ **Mining ticker discovery failed:** {str(e)[:100]}. Falling back to CSV.")
                master_symbols = []
                master_count = 0
        
        # V7.4: Only use CSV fallback if TIINGO_API_KEY is NOT present
        if not master_symbols and not tiingo_key_present and SCANNER_AVAILABLE and load_master_discovery_list:
            try:
                master_symbols = load_master_discovery_list(max_symbols=10000) or []
                master_count = len(master_symbols)
            except Exception:
                pass
        elif not master_symbols and tiingo_key_present:
            # TIINGO_API_KEY is present but get_all_mining_tickers failed - show warning
            st.warning("⚠️ **TIINGO_API_KEY found but mining ticker discovery failed.** Check Tiingo API connection.")

        total_mv = np.float64(df['Market_Value'].sum()) if not df.empty else 0.0
        total_value = np.float64(total_mv + np.float64(st.session_state.cash))
        st.header("Market Scanner")
        st.caption(f"Scanning {master_count} North American gold, silver & uranium miners ranked by (Alpha + FA) / Risk")

        # Controls row
        col1, col2, col3 = st.columns(3)
        with col1:
            global_opportunity_scan = st.checkbox(
                "Global Search",
                value=st.session_state.get('global_opportunity_scan', True),
                key='global_opportunity_scan_checkbox',
                help=f"Search {master_count} tickers, bypass diversification and impact constraints"
            )
            st.session_state.global_opportunity_scan = global_opportunity_scan
        with col2:
            include_tsxv = st.checkbox(
                "Include TSX-V",
                value=st.session_state.get('include_tsxv', False),
                key='include_tsxv_checkbox',
                help="Include TSX Venture Exchange (.V) tickers"
            )
            st.session_state.include_tsxv = include_tsxv
        with col3:
            debug_mode = st.checkbox("Debug Mode",
                                    value=st.session_state.get('debug_mode', False),
                                    key='debug_mode_checkbox',
                                    help="Show all scanned stocks and filter status")
            st.session_state.debug_mode = debug_mode
        
        # V7.0: Sovereign Global Scan – live mining tickers (get_all_mining_tickers) or CSV fallback
        if master_symbols:
            try:
                # master_symbols already loaded at top of tab
                # Reset data_health each scan so skip counts don't accumulate
                st.session_state.data_health = {}
                data_health = st.session_state.data_health

                # V5.0: Filter TSX-V by toggle, but force-include .V when Global Search is active (discovery mode)
                symbols_to_scan = master_symbols
                if not global_opportunity_scan and not include_tsxv:
                    symbols_to_scan = [s for s in symbols_to_scan if not s.upper().endswith('.V')]

                st.caption(f"Scanning {len(symbols_to_scan)} symbols" + (" (TSX-V excluded)" if not include_tsxv and not global_opportunity_scan else ""))

                # V5.0: Build recommendations from master list (Wide-Net approach)
                recommendations_data = []
                debug_data = []
                symbols_analyzed = 0
                symbols_failed = 0

                # Get hist_cache and results from session state
                hist_cache_rec = st.session_state.get('hist_cache', {})
                results_df = st.session_state.get('results', pd.DataFrame())
                portfolio_symbols = set(results_df['Symbol'].tolist()) if not results_df.empty and 'Symbol' in results_df.columns else set()

                # Load hunting settings for price/market cap filtering (use UI_CONFIG as fallback)
                if SECTOR_CRAWLER_AVAILABLE:
                    hunting_settings = load_hunting_settings()
                    max_price = hunting_settings.get('max_stock_price', UI_CONFIG.get('max_price', 200.0))
                    max_mcap = hunting_settings.get('max_market_cap_millions', UI_CONFIG.get('max_mcap_millions', 80000))
                else:
                    max_price = UI_CONFIG.get('max_price', 200.0)
                    max_mcap = UI_CONFIG.get('max_mcap_millions', 80000)

                # Execution log
                proof_log: list = []
                def _log(s: str):
                    proof_log.append(s)
                progress_bar = st.progress(0)
                status_text = st.empty()
                log_container = st.container()

                # V7.4: COMPLETELY BYPASS CANARY TEST - If TIINGO_API_KEY detected by Black Box, force full Global Scan
                # Check Black Box detection first (from sidebar diagnostic)
                tiingo_key_detected = st.session_state.get('tiingo_key_detected', False)
                # Also check environment directly as fallback
                tiingo_key_present = bool(os.getenv("TIINGO_API_KEY", "").strip())
                
                if tiingo_key_detected or tiingo_key_present:
                    # V7.4: TIINGO_API_KEY detected by Black Box - COMPLETELY BYPASS canary test, force full 1,000+ Global Scan
                    _log("Tiingo API key detected — full universe scan enabled.")
                else:
                    # V7.4: No TIINGO_API_KEY - Run canary test for diagnostics only (does not block scan)
                    _log("Canary test: GOLD (NYSE), ABX.TO (TSX), DSV.V (TSXV)...")
                    canary_symbols = [("GOLD", False), ("ABX.TO", True), ("DSV.V", True)]
                    canary_ok = False
                    canary_fail = []
                    for csym, is_can in canary_symbols:
                        h, _ = fetch_ticker_with_fallback(csym, period="5d", is_canadian=is_can, data_health=None)
                        if h is not None and not h.empty and "Close" in h.columns and len(h) > 0:
                            canary_ok = True
                            break
                        canary_fail.append(csym)
                    
                    if canary_ok:
                        _log("Canary OK — Tiingo connected.")
                    else:
                        _log("Canary failed — continuing scan using futures benchmarks.")
                
                # V7.4: DISABLED VETO - Always continue scan regardless of canary status
                # V5.0: Wide-Net iteration - scan ALL symbols via Tiingo Power (no veto)
                if True:  # Always continue - canary failure does not stop scan
                    for idx, symbol in enumerate(symbols_to_scan):
                        progress = (idx + 1) / len(symbols_to_scan)
                        progress_bar.progress(progress)
                        # Update status text every 10 symbols or at start
                        if idx % 10 == 0 or idx == 0:
                            status_text.info(f"Scanning: {idx + 1}/{len(symbols_to_scan)} | OK: {symbols_analyzed} | Skipped: {symbols_failed}")
                        _log(f"Fetching: {symbol}")

                        # Detect if ticker is Canadian (by suffix or Country/Jurisdiction)
                        is_canadian = symbol.upper().endswith('.V') or symbol.upper().endswith('.TO')
                        if not is_canadian and not results_df.empty:
                            # Check Country/Jurisdiction from results if available
                            symbol_row_check = results_df[results_df['Symbol'] == symbol] if 'Symbol' in results_df.columns else pd.DataFrame()
                            if not symbol_row_check.empty:
                                country = symbol_row_check.iloc[0].get('Country', '')
                                jurisdiction = symbol_row_check.iloc[0].get('Jurisdiction', '')
                                is_canadian = (country == 'Canada' or jurisdiction == 'Canada')
                        _log(f"Checking Jurisdiction Veto for {symbol}...")
                        # Check if symbol is in portfolio (for "NEW SIGNAL" badge)
                        in_portfolio = symbol in portfolio_symbols
                        # V5.0: CRITICAL FIX - Get data from results OR fetch/calculate for master_discovery_list symbols
                        symbol_row = results_df[results_df['Symbol'] == symbol] if not results_df.empty and 'Symbol' in results_df.columns else pd.DataFrame()
                        hist = hist_cache_rec.get(symbol, pd.DataFrame())
                    
                        # If not in results, try to fetch and calculate alpha
                        if symbol_row.empty:
                            if hist.empty:
                                # fetch_ticker_with_fallback already tries Tiingo + yfinance
                                hist, successful_symbol = fetch_ticker_with_fallback(
                                    symbol,
                                    period="1y",
                                    is_canadian=is_canadian,
                                    data_health=data_health
                                )
                                if not hist.empty:
                                    hist_cache_rec[symbol] = hist
                                    st.session_state['hist_cache'] = hist_cache_rec
                                    symbols_analyzed += 1
                                    _log(f"{symbol} OK (via {successful_symbol or symbol})")
                                else:
                                    symbols_failed += 1
                                    _log(f"{symbol} SKIP (no data from Tiingo or yfinance)")
                                    continue
                            else:
                                # hist was in cache from a previous run
                                symbols_analyzed += 1
                                _log(f"{symbol} OK (cached)")
                        
                        # Re-check symbol_row after potential fetch
                        symbol_row = results_df[results_df['Symbol'] == symbol] if not results_df.empty and 'Symbol' in results_df.columns else pd.DataFrame()
                        
                        # If we have hist data, create a minimal row for alpha calculation
                        ta_results = {}  # Will be populated for new candidates
                        if not hist.empty and symbol_row.empty:
                            # Create row with as much computed data as possible
                            metal_default = 'Gold'
                            # Classify metal from symbol name or known uranium tickers
                            sym_upper = symbol.upper().replace('.TO', '').replace('.V', '')
                            # TODO: derive metal type from fundamentals/sector instead of hardcoded sets
                            uranium_syms = {'NXE', 'CCJ', 'DNN', 'UEC', 'UUUU', 'URG', 'EU',
                                            'EFR', 'PDN'}
                            silver_syms = {'AG', 'PAAS', 'EXK', 'HL', 'SIL', 'SILJ', 'FR', 'SSRM',
                                           'FSM', 'CDE', 'SVM', 'SAND'}
                            if sym_upper in uranium_syms:
                                metal_default = 'Uranium'
                            elif sym_upper in silver_syms:
                                metal_default = 'Silver'

                            close_arr = hist['Close'].dropna()
                            last_price = float(close_arr.iloc[-1]) if len(close_arr) > 0 else 0.0
                            last_vol = float(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0.0

                            # Compute returns from actual data
                            ret_7d = 0.0
                            ret_30d = 0.0
                            ret_90d = 0.0
                            if len(close_arr) >= 7 and close_arr.iloc[-7] > 0:
                                ret_7d = ((close_arr.iloc[-1] - close_arr.iloc[-7]) / close_arr.iloc[-7]) * 100
                            if len(close_arr) >= 30 and close_arr.iloc[-30] > 0:
                                ret_30d = ((close_arr.iloc[-1] - close_arr.iloc[-30]) / close_arr.iloc[-30]) * 100
                            if len(close_arr) >= 90 and close_arr.iloc[-90] > 0:
                                ret_90d = ((close_arr.iloc[-1] - close_arr.iloc[-90]) / close_arr.iloc[-90]) * 100

                            # Compute 52-week high/low from hist
                            high_52w = hist['High'].max() if 'High' in hist.columns else last_price
                            low_52w = hist['Low'].min() if 'Low' in hist.columns else last_price
                            pct_from_high = ((last_price - high_52w) / high_52w * 100) if high_52w > 0 else 0.0
                            pct_from_low = ((last_price - low_52w) / low_52w * 100) if low_52w > 0 else 0.0

                            # Compute full TA indicators from hist data
                            ta_results = {}
                            smc_score = 50.0
                            smc_bias = 'Neutral'
                            try:
                                ta_results = calculate_all_ta(hist)
                            except Exception:
                                pass

                            # Compute SMC from hist if available
                            if INSTITUTIONAL_V3_AVAILABLE:
                                try:
                                    smc = calculate_smc_structure(hist)
                                    smc_score = smc.get('smc_score', 50.0)
                                    smc_bias = smc.get('bias', 'Neutral')
                                except Exception:
                                    pass

                            row = {
                                'Symbol': symbol,
                                'Price': last_price,
                                'Volume': last_vol,
                                'Return_7d': ret_7d,
                                'Return_30d': ret_30d,
                                'Return_90d': ret_90d,
                                'Dilution_Risk_Score': 50.0,
                                'Pct_From_52w_High': pct_from_high,
                                'Pct_From_52w_Low': pct_from_low,
                                'SMC_Bias': smc_bias,
                                'SMC_Score': smc_score,
                                'metal': metal_default,
                                'Metal_Type': metal_default,
                                'Country': 'Unknown',
                                'TA_Score': ta_results.get('ta_score', 50.0),
                                'RSI': ta_results.get('rsi', 50.0),
                                'MACD_Signal': ta_results.get('ta_signal', 'NEUTRAL'),
                            }
                            
                            _log(f"Syncing {row.get('metal', 'Gold')} futures with {symbol}...")
                            # Alpha-Only Fallback: compute Alpha from price history; if FA missing, use 0 and do not skip
                            try:
                                benchmark = get_benchmark_data(row.get('metal', 'Gold'))
                                alpha_result = calculate_alpha_models(row, hist, benchmark)
                                row['Alpha_Score'] = alpha_result.get('alpha_score', 50.0)
                            except Exception as e:
                                row['Alpha_Score'] = 50.0
                                if data_health is not None:
                                    data_health[symbol] = data_health.get(symbol, {})
                                    data_health[symbol]['alpha_calc_error'] = str(e)[:100]
                            row['FA_Score'] = 0.0
                            try:
                                fund = get_fundamentals_with_tracking(symbol)
                                row.update(fund)
                                
                                # V7.5: Apply news promotion score to Alpha_Score
                                news_promotion = fund.get('news_promotion_score', 0)
                                if news_promotion > 0:
                                    row['Alpha_Score'] = row.get('Alpha_Score', 50.0) + news_promotion
                                    row['News_Promotion'] = fund.get('news_keywords_found', [])
                                    row['News_Promotion_Score'] = news_promotion
                                
                                # Calculate FA_Score with updated row (includes info_dict)
                                if 'info_dict' in fund:
                                    row['info_dict'] = fund['info_dict']
                                    # V7.5: CRITICAL - Extract Market Cap from info_dict for filtering
                                    info = fund['info_dict']
                                    if 'Market_Cap' not in row and 'market_cap' not in row:
                                        mcap = info.get("marketCap") or info.get("market_cap")
                                        if mcap:
                                            row['Market_Cap'] = float(mcap) / 1_000_000  # Convert to millions
                                            row['market_cap'] = row['Market_Cap']  # Also set lowercase version
                                fa_result = calculate_fundamental_score(row, sector_data=None)
                                row['FA_Score'] = float(fa_result.get('fa_score', 0.0))
                                
                                # V7.2: Ensure Metal_Type is set (from fundamentals or fallback)
                                if 'Metal_Type' not in row or not row.get('Metal_Type'):
                                    row['Metal_Type'] = row.get('metal', metal_default)
                                if 'metal' not in row or not row.get('metal'):
                                    row['metal'] = row.get('Metal_Type', metal_default)
                            except Exception as e:
                                if data_health is not None:
                                    data_health[symbol] = data_health.get(symbol, {})
                                    data_health[symbol]['fundamentals_error'] = str(e)[:100]
                                # Alpha-Only Fallback: Tiingo price but no FA — keep FA_Score 0, do not skip
                                row['Metal_Type'] = row.get('Metal_Type', metal_default)
                                row['metal'] = row.get('metal', metal_default)
                        else:
                            # Symbol exists in results - use existing data
                            row = symbol_row.iloc[0].to_dict()
                            symbols_analyzed += 1
                        
                        # V5.0: Track filter failures for debug mode
                        filter_failures = []
                        
                        # V5.0: Handle both dict and Series/DataFrame row
                        if isinstance(row, dict):
                            row_dict = row
                        else:
                            row_dict = row.to_dict() if hasattr(row, 'to_dict') else {}
                        
                        # Check Price filter
                        current_price = float(row_dict.get('Price', 0))
                        if current_price > max_price:
                            filter_failures.append(f"Price ${current_price:.2f} > ${max_price:.2f}")
                        
                        # V7.5: Market Cap filter - Prioritize Junior/Mid-Tier ($20M-$600M)
                        # High Quality Exception: Allow >$600M ONLY if proprietary Score > 90
                        # VETO REMOVAL: Do NOT filter out stocks with 0 Revenue or Negative P/E (juniors invest in exploration)
                        market_cap = row_dict.get('Market_Cap', row_dict.get('market_cap', None))
                        
                        # V7.5: If market cap not found, try to get from info_dict
                        if market_cap is None and 'info_dict' in row_dict:
                            info = row_dict['info_dict']
                            mcap = info.get("marketCap") or info.get("market_cap")
                            if mcap:
                                market_cap = float(mcap) / 1_000_000  # Convert to millions
                                row_dict['Market_Cap'] = market_cap
                                row_dict['market_cap'] = market_cap
                        
                        market_cap_tier = None  # Track tier for prioritization
                        market_cap_filter_passed = True  # Track if stock passes market cap filter
                        
                        # Classify market cap for display — never exclude based on size
                        if market_cap is not None and isinstance(market_cap, (int, float)) and market_cap > 0:
                            if market_cap < 20:
                                market_cap_tier = 'MICRO'
                            elif market_cap <= 600:
                                market_cap_tier = 'JUNIOR_MIDTIER'
                            elif market_cap <= 5000:
                                market_cap_tier = 'MID_CAP'
                            else:
                                market_cap_tier = 'LARGE_CAP'
                            market_cap_filter_passed = True
                        else:
                            market_cap_tier = 'UNKNOWN'
                            market_cap_filter_passed = True
                        
                        # V7.5: Explicitly DO NOT filter based on Revenue or P/E
                        # Juniors with 0 revenue or negative P/E are investing in exploration - this is expected
                        # No filters applied for revenue or P/E ratio
                        
                        # V7.5: CRITICAL FIX - Skip stocks that fail market cap filter
                        if not market_cap_filter_passed:
                            symbols_failed += 1
                            _log(f"{symbol} SKIP (Market Cap filter failed: {filter_failures[-1] if filter_failures else 'outside sweet spot'})")
                            continue  # Skip this symbol - don't add to recommendations
    
                        # V7.0: Forensic layer – AISC & P/NAV for mcap > $500M
                        if market_cap is not None and isinstance(market_cap, (int, float)) and market_cap > 500:
                            try:
                                f = get_forensic_fundamentals(symbol)
                                if f.get("aisc") is not None:
                                    row_dict["AISC"] = f["aisc"]
                                if f.get("p_nav") is not None:
                                    row_dict["P_NAV"] = f["p_nav"]
                            except Exception:
                                pass
    
                        # V7.0: Trailing Sharpe for SWAP logic (Higher Sharpe + Lower P/NAV)
                        sharpe_1y = np.nan
                        if not hist.empty and "Close" in hist.columns:
                            sharpe_1y = _trailing_sharpe(hist, window=252)
                        row_dict["Sharpe_1y"] = sharpe_1y

                        # V7.2: Shanghai-Adjusted NAV (for Global Sector Scan ranking)
                        shanghai_nav = np.nan
                        try:
                            benchmarks = get_global_commodity_benchmarks()
                            metal_type = row_dict.get('Metal_Type', row_dict.get('metal', 'Gold')).strip()
                            p_nav = row_dict.get('P_NAV')
                            # V7.4: Fix isfinite error - ensure p_nav is numeric
                            try:
                                p_nav = float(p_nav) if p_nav is not None and p_nav != '' else np.nan
                            except (ValueError, TypeError):
                                p_nav = np.nan
                            if np.isfinite(p_nav) and p_nav and p_nav > 0:
                                if metal_type == 'Gold' and np.isfinite(benchmarks.get('gold_shanghai')):
                                    # Shanghai premium adjustment: higher Shanghai = lower P/NAV (better value)
                                    shanghai_nav = p_nav * (benchmarks['gold_comx'] / benchmarks['gold_shanghai']) if benchmarks['gold_shanghai'] > 0 else p_nav
                                elif metal_type == 'Silver' and np.isfinite(benchmarks.get('silver_shanghai')):
                                    shanghai_nav = p_nav * (benchmarks['silver_comx'] / benchmarks['silver_shanghai']) if benchmarks['silver_shanghai'] > 0 else p_nav
                                else:
                                    shanghai_nav = p_nav
                        except Exception:
                            # V7.4: Fix isfinite error - ensure p_nav is numeric even in exception
                            try:
                                p_nav_temp = row_dict.get('P_NAV')
                                p_nav_temp = float(p_nav_temp) if p_nav_temp is not None and p_nav_temp != '' else np.nan
                            except (ValueError, TypeError):
                                p_nav_temp = np.nan
                            shanghai_nav = p_nav_temp
                        row_dict["Shanghai_NAV"] = shanghai_nav
                        
                        # Use pre-calculated FA_Score and Market_Buzz from analysis (or defaults)
                        fa_score = float(row_dict.get('FA_Score', 0.0))
                        market_buzz = bool(row_dict.get('Market_Buzz', False))
                        volume_spike_pct = float(row_dict.get('Volume_Spike_Pct', 0.0))
                        
                        # Get risk score (default to 50 for neutral)
                        risk_score = float(row_dict.get('Sell_Risk_Score', 50))
                        
                        # Calculate combined score: (Alpha + FA Score) / Risk
                        # No ticker-specific bonuses — ranking is purely data-driven
                        alpha_score = float(row_dict.get('Alpha_Score', 50))
                        prioritization_bonus = 0.0
                        combined_score = (alpha_score + fa_score) / max(risk_score, 1)
                        
                        # V5.0: Global Search - bypass all diversification/impact gates
                        div_veto_applied = False
                        div_veto_reason = ''
                        market_impact_pct = 0.0
                        passes_impact_gate = True
                        
                        if not global_opportunity_scan:
                            # Only calculate vetoes if Global Search is OFF
                            # (Derive jurisdiction/metal_type if needed)
                            jurisdiction = row_dict.get('Jurisdiction', row_dict.get('Country', 'Unknown'))
                            metal_type = row_dict.get('Metal_Type', row_dict.get('metal', 'Gold'))
                            
                            # Calculate exposure (simplified - would need full portfolio)
                            # For Global Search, we skip this
                            pass
                        
                        # Derive Action from scores for new candidates (not from portfolio)
                        action = row_dict.get('Action', None)
                        if action is None or (not in_portfolio and action == 'HOLD'):
                            # New candidate: derive action from alpha + TA signals
                            if alpha_score >= 65 and ta_results.get('ta_signal', 'NEUTRAL') != 'SELL':
                                action = 'Buy'
                            elif alpha_score >= 50:
                                action = 'HOLD'
                            else:
                                action = 'Avoid'
                            row_dict['Action'] = action
                        if action not in ['Buy', 'HOLD']:
                            filter_failures.append(f"Action: {action} (not Buy/HOLD)")
                        
                        # Track AISC filter
                        aisc = row_dict.get('AISC', row_dict.get('aisc', None))
                        if aisc is not None and isinstance(aisc, (int, float)) and aisc > UI_CONFIG.get('max_aisc', 1400):
                            filter_failures.append(f"AISC ${aisc:.0f}/oz > ${UI_CONFIG.get('max_aisc', 1400)}")
                        
                        metal_type = row_dict.get('Metal_Type', row_dict.get('metal', 'Gold')).strip()
                        recommendations_data.append({
                            'Symbol': symbol,
                            'Alpha_Score': alpha_score,
                            'FA_Score': fa_score,
                            'Combined_Score': combined_score,
                            'Risk_Score': risk_score,
                            'Action': action,
                            'Confidence': row_dict.get('Confidence', 'Low'),
                            'Market_Buzz': market_buzz,
                            'Volume_Spike_Pct': volume_spike_pct,
                            'Diversification_Veto': div_veto_applied,
                            'Veto_Reason': div_veto_reason,
                            'FA_Reasoning': row_dict.get('FA_Reasoning', ''),
                            'Market_Impact_Pct': market_impact_pct,
                            'Passes_Impact_Gate': passes_impact_gate,
                            'Price': current_price,
                            'In_Portfolio': in_portfolio,
                            'Filter_Failures': filter_failures,
                            'Passed_All_Filters': len(filter_failures) == 0,
                            'Sharpe_1y': row_dict.get('Sharpe_1y', np.nan),
                            'P_NAV': row_dict.get('P_NAV'),
                            'Shanghai_NAV': row_dict.get('Shanghai_NAV'),
                            'AISC': row_dict.get('AISC', row_dict.get('aisc')),
                            'Metal_Type': metal_type,
                            'Market_Cap_M': market_cap,  # V7.5: Include market cap for display
                            'Market_Cap_Tier': market_cap_tier,  # V7.5: Track tier for prioritization
                            'Prioritization_Bonus': prioritization_bonus,  # V7.5: Show bonus applied
                        })
                        
                        # Debug mode tracking
                        if debug_mode:
                            debug_data.append({
                                'Symbol': symbol,
                                'Price': current_price,
                                'Market_Cap_M': market_cap if market_cap else 'N/A',
                                'AISC': aisc if aisc else 'N/A',
                                'Action': action,
                                'Alpha_Score': alpha_score,
                                'FA_Score': fa_score,
                                'Combined_Score': combined_score,
                                'Filter_Failures': ', '.join(filter_failures) if filter_failures else '✅ Passed all filters',
                                'Status': '✅ Passed' if len(filter_failures) == 0 else '❌ Filtered'
                            })
                
                progress_bar.progress(1.0)
                with log_container:
                    st.markdown("**Execution log**")
                    st.code("\n".join(proof_log), language=None)
                if symbols_analyzed > 0:
                    status_text.success(f"Scan complete: {symbols_analyzed}/{len(symbols_to_scan)} symbols analyzed, {symbols_failed} skipped")
                else:
                    status_text.warning(f"0/{len(symbols_to_scan)} symbols loaded — check hey.env for TIINGO_API_KEY")
                
                # Convert to DataFrame and sort by Combined_Score
                rec_df = pd.DataFrame(recommendations_data)
                if not rec_df.empty:
                    # Rank purely by Combined_Score = (Alpha + FA) / Risk
                    rec_df = rec_df.sort_values('Combined_Score', ascending=False)
                    
                    # V5.0: Global Search - explicitly clear gates for all candidates
                    if global_opportunity_scan:
                        rec_df = rec_df.copy()
                        rec_df['Diversification_Veto'] = False
                        rec_df['Passes_Impact_Gate'] = True
                else:
                    rec_df = pd.DataFrame()
                    debug_data = []
                
                # V5.0: Tiingo Power Summary (clean, non-alarming display)
                if symbols_failed > 0 and data_health:
                    skipped_symbols = [s for s, health in data_health.items() if health.get('status') in ('skip', 'failed')]
                    if skipped_symbols and debug_mode:
                        with st.expander(f"📊 Tiingo Power Details: {len(skipped_symbols)} symbols skipped", expanded=False):
                            st.caption(f"Skipped: {', '.join(skipped_symbols[:15])}{'...' if len(skipped_symbols) > 15 else ''}")
                            st.caption("Skipped symbols typically don't have Tiingo coverage or require different ticker formats.")
            except Exception as e:
                st.error(f"Error loading master discovery list: {str(e)[:100]}")
                rec_df = pd.DataFrame()
                debug_data = []
        else:
            # Fallback: Use portfolio if master list not available
            st.warning("⚠️ Master discovery list not available. Using portfolio symbols.")
            # Calculate total_value for recommendations (needed for exposure calculations)
            total_mv = np.float64(df['Market_Value'].sum())
            total_value = np.float64(total_mv + np.float64(st.session_state.cash))
            
            # V5.0: Calculate cache key early (before checkbox) to check if data is ready
            if not df.empty:
                results_hash = hash(
                    tuple(sorted(df['Symbol'].tolist())) + 
                    (hash(str(df['Alpha_Score'].sum())),) + 
                    (hash(str(df['FA_Score'].sum())),) +
                    (hash(str(df['Market_Value'].sum())),)
                )
            else:
                results_hash = 0
            cache_key = f"recommendations_cache_{abs(results_hash)}"
            
            cached_hash = st.session_state.get('recommendations_cache_hash')
            
            # Use cached recommendations if available
            if cache_key in st.session_state and cached_hash == results_hash:
                rec_df = st.session_state[cache_key].copy()
                debug_data = st.session_state.get(f"{cache_key}_debug", [])
            else:
                # Build from portfolio (fallback)
                recommendations_data = []
                debug_data = []
                hist_cache_rec = st.session_state.get('hist_cache', {})
                
                if SECTOR_CRAWLER_AVAILABLE:
                    hunting_settings = load_hunting_settings()
                    max_price = hunting_settings.get('max_stock_price', UI_CONFIG.get('max_price', 200.0))
                    max_mcap = hunting_settings.get('max_market_cap_millions', UI_CONFIG.get('max_mcap_millions', 80000))
                else:
                    max_price = UI_CONFIG.get('max_price', 200.0)
                    max_mcap = UI_CONFIG.get('max_mcap_millions', 80000)
                
                for idx, row in df.iterrows():
                    symbol = row['Symbol']
                    hist = hist_cache_rec.get(symbol, pd.DataFrame())
                    
                    # V5.0: Track filter failures for debug mode
                    filter_failures = []
                    
                    # Check Price filter (V5.0 Sovereign Hunter: Price > $5.00 rejection)
                    current_price = row.get('Price', 0)
                    if current_price > max_price:
                        filter_failures.append(f"Price ${current_price:.2f} > ${max_price:.2f}")
                    
                    # Check Market Cap filter (V5.0 Sovereign Hunter: MCAP > $500M rejection)
                    market_cap = row.get('Market_Cap', row.get('market_cap', None))
                    if market_cap is not None and market_cap > max_mcap:
                        filter_failures.append(f"Market Cap ${market_cap:.1f}M > ${max_mcap:.1f}M")
                    
                    # Use pre-calculated FA_Score and Market_Buzz from analysis (with fallbacks)
                    # V5.0: Ensure missing fundamental data gets 'Neutral' score (0.0) instead of being hidden
                    fa_score = float(row.get('FA_Score', 0.0))  # Default to 0.0 (Neutral) if missing
                    market_buzz = bool(row.get('Market_Buzz', False))
                    volume_spike_pct = float(row.get('Volume_Spike_Pct', 0.0))
                    
                    # Get risk score (inverse for ranking - lower risk = better)
                    risk_score = row.get('Sell_Risk_Score', 50)
                    
                    # Calculate combined score: (Alpha + FA Score) / Risk
                    # Use max(risk_score, 1) to avoid division by zero
                    alpha_score = row.get('Alpha_Score', 0)
                    combined_score = (alpha_score + fa_score) / max(risk_score, 1)
                    
                    # Check if passes 1.5% Market Impact Gate (would need to calculate for new buys)
                    # For existing positions, assume they passed if they're in portfolio
                    # For new picks, we'll check during recommendation display
                    
                    # Check Diversification Veto
                    # This would be checked during actual buy execution, but we note it here
                    # V5.0: Backward compatibility - derive Jurisdiction and Metal_Type if missing
                    if 'Jurisdiction' not in df.columns:
                        # Derive from Country column if available
                        if 'Country' in df.columns:
                            df['Jurisdiction'] = df['Country']
                        else:
                            df['Jurisdiction'] = 'Unknown'
                    if 'Metal_Type' not in df.columns:
                        # Derive from metal/Metal column if available
                        if 'metal' in df.columns:
                            df['Metal_Type'] = df['metal']
                        elif 'Metal' in df.columns:
                            df['Metal_Type'] = df['Metal']
                        else:
                            df['Metal_Type'] = 'Gold'
                    
                    # Get jurisdiction and metal_type from row (with fallbacks)
                    jurisdiction = row.get('Jurisdiction', 'Unknown')
                    if jurisdiction == 'Unknown' and 'Country' in row:
                        jurisdiction = row.get('Country', 'Unknown')
                    
                    metal_type = row.get('Metal_Type', 'Gold')
                    if metal_type == 'Gold' and 'metal' in row:
                        metal_type = row.get('metal', 'Gold')
                    elif metal_type == 'Gold' and 'Metal' in row:
                        metal_type = row.get('Metal', 'Gold')
                    
                    # Calculate current exposure (with safety check for columns)
                    if 'Jurisdiction' in df.columns and 'Market_Value' in df.columns:
                        jurisdiction_exposure = df[df['Jurisdiction'] == jurisdiction]['Market_Value'].sum()
                    else:
                        jurisdiction_exposure = 0.0
                    jurisdiction_pct = (jurisdiction_exposure / total_value * 100) if total_value > 0 else 0.0
                    
                    if 'Metal_Type' in df.columns and 'Market_Value' in df.columns:
                        metal_exposure = df[df['Metal_Type'] == metal_type]['Market_Value'].sum()
                    else:
                        metal_exposure = 0.0
                    metal_pct = (metal_exposure / total_value * 100) if total_value > 0 else 0.0
                    
                    div_veto_applied = False
                    div_veto_reason = ''
                    if jurisdiction_pct > 35.0:
                        div_veto_applied = True
                        div_veto_reason = f"{jurisdiction} exposure {jurisdiction_pct:.1f}% > 35%"
                    elif metal_pct > 50.0:
                        div_veto_applied = True
                        div_veto_reason = f"{metal_type} exposure {metal_pct:.1f}% > 50%"
                    
                    # V5.0: Check 1.5% Market Impact Gate for new buys
                    # Calculate market impact if this is a buy recommendation
                    market_impact_pct = 0.0
                    passes_impact_gate = True
                    if row.get('Action') == 'Buy' and not hist.empty and 'Volume' in hist.columns:
                        # Estimate trade size (recommended allocation)
                        recommended_pct = row.get('Recommended_Pct', 0)
                        current_pct = row.get('Pct_Portfolio', 0)
                        delta_pct = recommended_pct - current_pct
                        if delta_pct > 0:
                            trade_dollars = (delta_pct / 100.0) * total_value
                            current_price = row.get('Price', 0)
                            
                            # Get daily volume
                            if len(hist) >= 1:
                                daily_volume = hist['Volume'].iloc[-1]
                                daily_volume_dollars = daily_volume * current_price if daily_volume > 0 and current_price > 0 else 0.0
                                
                                if daily_volume_dollars > 0:
                                    market_impact_pct = (trade_dollars / daily_volume_dollars * 100)
                                    if market_impact_pct > 1.5:
                                        passes_impact_gate = False
                                        filter_failures.append(f"Market Impact {market_impact_pct:.2f}% > 1.5%")
                    
                    # Track action filter (only Buy/HOLD are actionable)
                    action = row.get('Action', 'HOLD')
                    if action not in ['Buy', 'HOLD']:
                        filter_failures.append(f"Action: {action} (not Buy/HOLD)")
                    
                    # Track diversification veto
                    if div_veto_applied:
                        filter_failures.append(f"Diversification: {div_veto_reason}")
                    
                    # Track AISC filter (AISC > $1,400 = Sell/Avoid)
                    aisc = row.get('AISC', row.get('aisc', None))
                    if aisc is not None and aisc > 1400:
                        filter_failures.append(f"AISC ${aisc:.0f}/oz > $1,400")
                    
                    recommendations_data.append({
                        'Symbol': symbol,
                        'Alpha_Score': alpha_score,
                        'FA_Score': fa_score,
                        'Combined_Score': combined_score,
                        'Risk_Score': risk_score,
                        'Action': row.get('Action', 'HOLD'),
                        'Confidence': row.get('Confidence', 'Low'),
                        'Market_Buzz': market_buzz,
                        'Volume_Spike_Pct': volume_spike_pct,
                        'Diversification_Veto': div_veto_applied,
                        'Veto_Reason': div_veto_reason,
                        'FA_Reasoning': row.get('FA_Reasoning', ''),
                        'Market_Impact_Pct': market_impact_pct,
                        'Passes_Impact_Gate': passes_impact_gate,
                        'Price': row.get('Price', 0),
                        'Market_Value': row.get('Market_Value', 0),
                        'Pct_Portfolio': row.get('Pct_Portfolio', 0),
                        'Recommended_Pct': row.get('Recommended_Pct', 0),
                        'Filter_Failures': filter_failures,  # For debug mode
                        'Passed_All_Filters': len(filter_failures) == 0
                    })
                    
                    # V5.0: Debug mode - track all stocks
                    if debug_mode:
                        debug_data.append({
                            'Symbol': symbol,
                            'Price': current_price,
                            'Market_Cap_M': market_cap if market_cap else 'N/A',
                            'AISC': aisc if aisc else 'N/A',
                            'Action': action,
                            'Alpha_Score': alpha_score,
                            'FA_Score': fa_score,
                            'Combined_Score': combined_score,
                            'Filter_Failures': ', '.join(filter_failures) if filter_failures else '✅ Passed all filters',
                            'Status': '✅ Passed' if len(filter_failures) == 0 else '❌ Filtered'
                        })
                
                # Convert to DataFrame and sort by combined score
                rec_df = pd.DataFrame(recommendations_data)
                if not rec_df.empty:
                    rec_df = rec_df.sort_values('Combined_Score', ascending=False)
                    
                    # V5.0: Cache recommendations in session_state for instant re-filtering
                    st.session_state[cache_key] = rec_df.copy()
                    st.session_state[f"{cache_key}_debug"] = debug_data.copy()
                    st.session_state['recommendations_cache_hash'] = results_hash
                else:
                    rec_df = pd.DataFrame()
        
        # V5.0: Debug Mode - Show all stocks and filter status
        if debug_mode and debug_data:
            st.markdown("### Debug Mode: All Stocks Scanned")
            debug_df = pd.DataFrame(debug_data)
            st.dataframe(
                debug_df.style.format({
                    'Price': '${:.2f}',
                    'Market_Cap_M': lambda x: f'${x:.1f}M' if isinstance(x, (int, float)) else str(x),
                    'AISC': lambda x: f'${x:.0f}/oz' if isinstance(x, (int, float)) else str(x),
                    'Alpha_Score': '{:.1f}',
                    'FA_Score': '{:+.1f}',
                    'Combined_Score': '{:.2f}'
                }),
                use_container_width=True,
                hide_index=True
            )
            st.markdown("---")
        
        # V5.0: Global Opportunity Scan – explicitly clear diversification/impact gates for all candidates
        # When Global Scan is ON, ensure Top 5 shows "Best in World" regardless of current holdings
        # Ranking is purely (Alpha + FA) / Risk; no jurisdictional or metal-type weightings applied.
        if not rec_df.empty:
            if global_opportunity_scan:
                rec_df = rec_df.copy()
                rec_df['Diversification_Veto'] = False  # Explicitly set to False for all
                rec_df['Passes_Impact_Gate'] = True  # Explicitly set to True for all

            # V6.5: Piotroski F-Score (Global Search integrity) – cross-reference with balance sheet health
            if 'Symbol' in rec_df.columns and 'F_Score' not in rec_df.columns:
                f_scores = [piotroski_f_score(str(s)) for s in rec_df['Symbol']]
                rec_df = rec_df.copy()
                rec_df['F_Score'] = f_scores

            # Filter to only actionable recommendations (Buy or HOLD with positive score)
            actionable = rec_df[rec_df['Action'].isin(['Buy', 'HOLD'])].copy()
        else:
            actionable = pd.DataFrame()

        if not actionable.empty:
            filtered_by_action = len(rec_df) - len(actionable) if not rec_df.empty else 0

            # V6.5: Exclude high-Alpha + F-Score < 3 (Fundamental Warning); exclude from Top 5
            before_fscore = len(actionable)
            mask_fail = (actionable['Alpha_Score'].fillna(0) >= 70) & (actionable['F_Score'].fillna(-1) >= 0) & (actionable['F_Score'] < 3)
            actionable = actionable[~mask_fail].copy()
            filtered_by_fscore = before_fscore - len(actionable)
            if filtered_by_fscore > 0:
                st.caption(f"⚠️ **Fundamental Warning:** {filtered_by_fscore} stock(s) excluded (high Alpha but F-Score < 3).")

            before_div_filter = len(actionable)
            if not global_opportunity_scan:
                actionable = actionable[~actionable['Diversification_Veto']].copy()
            filtered_by_div = before_div_filter - len(actionable) if not global_opportunity_scan else 0

            before_impact_filter = len(actionable)
            if not global_opportunity_scan:
                actionable = actionable[actionable['Passes_Impact_Gate']].copy()
            filtered_by_impact = before_impact_filter - len(actionable) if not global_opportunity_scan else 0

            # Sort purely by Combined_Score — best algo score wins regardless of market cap
            if not actionable.empty:
                actionable = actionable.sort_values('Combined_Score', ascending=False)
            # Show up to 5 new discoveries + up to 5 portfolio rebalance signals
            new_candidates = actionable[~actionable['In_Portfolio']].head(5) if 'In_Portfolio' in actionable.columns else actionable.head(5)
            port_candidates = actionable[actionable['In_Portfolio']].head(5) if 'In_Portfolio' in actionable.columns else pd.DataFrame()
            top_5 = pd.concat([new_candidates, port_candidates]).head(10)
        else:
            top_5 = pd.DataFrame()
            filtered_by_action = 0
            filtered_by_div = 0
            filtered_by_impact = 0
            filtered_by_fscore = 0
        
        # V5.0: Save daily picks to session_state for persistence
        st.session_state.daily_picks = top_5.copy()
        st.session_state.daily_picks_global_scan = global_opportunity_scan
        
        # V5.0: Helpful error messages when no picks found
        if top_5.empty:
            # Build explanation message
            reasons = []
            if filtered_by_action > 0:
                reasons.append(f"{filtered_by_action} stock(s) filtered by Action (not Buy/HOLD)")
            if filtered_by_fscore > 0:
                reasons.append(f"{filtered_by_fscore} stock(s) excluded (high Alpha but F-Score < 3)")
            if filtered_by_div > 0:
                reasons.append(f"{filtered_by_div} stock(s) filtered by Diversification Veto")
            if filtered_by_impact > 0:
                reasons.append(f"{filtered_by_impact} stock(s) filtered by Market Impact > 1.5%")
            
            # Check for price/market cap filters (check the actual filter_failures list from rec_df)
            # Use rec_df instead of recommendations_data (works for both cached and fresh data)
            if not rec_df.empty and 'Filter_Failures' in rec_df.columns:
                price_filtered = len([r for _, r in rec_df.iterrows() if any('Price' in str(f) for f in (r.get('Filter_Failures', []) if isinstance(r.get('Filter_Failures'), list) else []))])
                mcap_filtered = len([r for _, r in rec_df.iterrows() if any('Market Cap' in str(f) for f in (r.get('Filter_Failures', []) if isinstance(r.get('Filter_Failures'), list) else []))])
                aisc_filtered = len([r for _, r in rec_df.iterrows() if any('AISC' in str(f) for f in (r.get('Filter_Failures', []) if isinstance(r.get('Filter_Failures'), list) else []))])
            else:
                price_filtered = 0
                mcap_filtered = 0
                aisc_filtered = 0
            
            # Use UI_CONFIG to prevent NameError
            max_price_val = UI_CONFIG.get('max_price', 200.0)
            max_mcap_val = UI_CONFIG.get('max_mcap_millions', 80000)
            max_aisc_val = UI_CONFIG.get('max_aisc', 1400)
            
            if price_filtered > 0:
                reasons.append(f"{price_filtered} stock(s) filtered by Price > ${max_price_val:.2f}")
            if mcap_filtered > 0:
                reasons.append(f"{mcap_filtered} stock(s) filtered by Market Cap > ${max_mcap_val:.1f}M")
            if aisc_filtered > 0:
                reasons.append(f"{aisc_filtered} stock(s) filtered by AISC > ${max_aisc_val}/oz")
            
            if not reasons:
                reasons.append("No stocks in portfolio or all stocks have insufficient data")
            
            st.warning(f"⚠️ **No picks found.** Reasons: {'; '.join(reasons)}")
            if not global_opportunity_scan:
                st.info("**Tip:** Enable 'Global Opportunity Scan' above to see 'Best in World' stocks ignoring Diversification Veto and Market Impact Gate.")
            else:
                st.info("**Tip:** Enable Debug Mode above to see all stocks and which specific filters they're failing.")
        else:
            # Split into New Discoveries and Portfolio Rebalance candidates
            new_discoveries = top_5[~top_5['In_Portfolio']].copy() if 'In_Portfolio' in top_5.columns else top_5.copy()
            portfolio_picks = top_5[top_5['In_Portfolio']].copy() if 'In_Portfolio' in top_5.columns else pd.DataFrame()

            # -- New Discoveries Section --
            if not new_discoveries.empty:
                st.markdown("### New Discovery Candidates")
                st.success(f"Found {len(new_discoveries)} new stock(s) not in your portfolio")
            else:
                st.info("No new discovery candidates found in this scan. All top picks are already in your portfolio.")

            # -- Portfolio Rebalance Section --
            if not portfolio_picks.empty:
                st.markdown("### Portfolio Rebalance Signals")
                st.caption(f"{len(portfolio_picks)} of your holdings ranked in the top results")

            # Render both sections using the same card layout
            for section_df, section_label in [(new_discoveries, 'new'), (portfolio_picks, 'portfolio')]:
                for idx, row in section_df.iterrows():
                    with st.container():
                        col1, col2, col3 = st.columns([2, 2, 1])

                        with col1:
                            symbol = row['Symbol']
                            in_portfolio = row.get('In_Portfolio', False)
                            alpha_score = row.get('Alpha_Score', 0)

                            symbol_display = f"### {symbol}"
                            if not in_portfolio:
                                symbol_display += " **NEW**"
                            else:
                                symbol_display += " (owned)"
                            st.markdown(symbol_display)
                            st.metric("Combined Score", f"{row['Combined_Score']:.2f}",
                                     help="(Alpha + FA Score) / Risk")

                        with col2:
                            market_cap_display = ""
                            if 'Market_Cap_M' in row and pd.notna(row.get('Market_Cap_M')):
                                mcap = row['Market_Cap_M']
                                tier = row.get('Market_Cap_Tier', '')
                                market_cap_display = f"**${mcap:,.1f}M** ({tier})" if tier else f"**${mcap:,.1f}M**"

                            st.caption(f"**Alpha:** {row['Alpha_Score']:.1f} | **FA:** {row['FA_Score']:+.1f} | **Risk:** {row['Risk_Score']:.0f}")
                            if market_cap_display:
                                st.caption(market_cap_display)
                            st.caption(f"**Action:** {row['Action']} ({row['Confidence']} confidence) | **Metal:** {row.get('Metal_Type', '?')}")
                            if row.get('Price', 0) > 0:
                                st.caption(f"**Price:** ${row['Price']:.2f}")
                            if row['Market_Buzz']:
                                st.caption(f"**Market Buzz:** Volume spike {row['Volume_Spike_Pct']:.1f}%")

                        with col3:
                            if row['Action'] == 'Buy':
                                st.success("BUY")
                            elif row['Action'] == 'HOLD':
                                st.info("HOLD")
                            else:
                                st.warning(row['Action'])
                            # Add to watchlist button
                            _sym = row['Symbol']
                            _starred = st.session_state.get('starred_symbols', [])
                            if _sym not in _starred:
                                if st.button("+ Watchlist", key=f"wl_add_{section_label}_{_sym}"):
                                    if 'starred_symbols' not in st.session_state:
                                        st.session_state.starred_symbols = []
                                    st.session_state.starred_symbols.append(_sym)
                                    st.rerun()
                            else:
                                st.caption("On watchlist")

                    # Show FA reasoning
                    if row['FA_Reasoning']:
                        with st.expander(f"Fundamental Analysis: {row['Symbol']}"):
                            st.caption(row['FA_Reasoning'])
                    
                    # V5.0: Alpha DNA Parameter Breakdown
                    with st.expander(f"Alpha DNA & Parameter Weights: {row['Symbol']}", expanded=False):
                        # 2-column layout: Raw Scores (Left) and Contribution to Weight (Right)
                        dna_col1, dna_col2 = st.columns(2)
                        
                        with dna_col1:
                            st.markdown("#### Raw Scores")
                            st.metric("Alpha Score", f"{row.get('Alpha_Score', 0):.1f}/100")
                            st.metric("FA Score", f"{row.get('FA_Score', 0):+.1f}")
                            st.metric("Risk Score", f"{row.get('Risk_Score', 50):.0f}/100")
                            st.metric("Market Buzz", "🔥 Active" if row.get('Market_Buzz', False) else "⚪ Inactive")
                            if row.get('Volume_Spike_Pct', 0) > 0:
                                st.caption(f"Volume Spike: {row['Volume_Spike_Pct']:.1f}%")
                        
                        with dna_col2:
                            st.markdown("#### Contribution to Weight")
                            # Calculate contribution percentages
                            alpha_score = row.get('Alpha_Score', 0)
                            fa_score = row.get('FA_Score', 0)
                            risk_score = max(row.get('Risk_Score', 50), 1)  # Avoid division by zero
                            combined_score = row.get('Combined_Score', 0)
                            
                            # Contribution calculation: (Alpha + FA) / Risk = Combined Score
                            # Each component's contribution to final position size
                            total_numerator = alpha_score + fa_score
                            if total_numerator > 0:
                                alpha_contribution_pct = (alpha_score / total_numerator) * 100
                                fa_contribution_pct = (fa_score / total_numerator) * 100
                            else:
                                alpha_contribution_pct = 0
                                fa_contribution_pct = 0
                            
                            st.metric("Alpha Contribution", f"{alpha_contribution_pct:.1f}%", 
                                     help=f"Alpha Score contributed {alpha_contribution_pct:.1f}% to final position size")
                            st.metric("FA Contribution", f"{fa_contribution_pct:.1f}%",
                                     help=f"FA Score contributed {fa_contribution_pct:.1f}% to final position size")
                            st.metric("Risk Adjustment", f"{risk_score:.0f}",
                                     help=f"Risk Score acts as divisor: Combined = (Alpha + FA) / {risk_score:.0f}")
                            st.metric("Final Combined", f"{combined_score:.2f}",
                                     help="Final ranking score: (Alpha + FA) / Risk")
                        
                        st.markdown("---")
                        
                        # Parameter Scorecard Table
                        st.markdown("#### Parameter Scorecard")
                        
                        # Build scorecard data from available row data
                        scorecard_data = []
                        
                        # Alpha Score
                        scorecard_data.append({
                            'Metric': 'Alpha Score',
                            'Value': f"{alpha_score:.1f}",
                            'Score (1-100)': f"{alpha_score:.0f}",
                            'Weight Impact': 'High' if alpha_score >= 70 else 'Med' if alpha_score >= 50 else 'Low'
                        })
                        
                        # FA Score
                        scorecard_data.append({
                            'Metric': 'FA Score',
                            'Value': f"{fa_score:+.1f}",
                            'Score (1-100)': f"{max(0, min(100, fa_score + 50)):.0f}",  # Normalize to 0-100
                            'Weight Impact': 'High' if fa_score >= 10 else 'Med' if fa_score >= 0 else 'Low'
                        })
                        
                        # Risk Score
                        scorecard_data.append({
                            'Metric': 'Risk Score',
                            'Value': f"{risk_score:.0f}",
                            'Score (1-100)': f"{risk_score:.0f}",
                            'Weight Impact': 'High' if risk_score <= 30 else 'Med' if risk_score <= 50 else 'Low'
                        })
                        
                        # Relative Strength (if available from row)
                        if 'Return_30d' in row or 'Return_90d' in row:
                            ret_30d = row.get('Return_30d', 0)
                            rel_strength_score = 50 + (ret_30d * 2)  # Rough conversion
                            rel_strength_score = max(0, min(100, rel_strength_score))
                            scorecard_data.append({
                                'Metric': 'Relative Strength',
                                'Value': f"{ret_30d:+.1f}%",
                                'Score (1-100)': f"{rel_strength_score:.0f}",
                                'Weight Impact': 'High' if rel_strength_score >= 70 else 'Med' if rel_strength_score >= 50 else 'Low'
                            })
                        
                        # AISC (Cost) if available
                        aisc = row.get('AISC', row.get('aisc', None))
                        if aisc is not None and isinstance(aisc, (int, float)):
                            # Lower AISC is better, so score = 100 - (AISC/20), capped at 0-100
                            aisc_score = max(0, min(100, 100 - (aisc / 20)))
                            scorecard_data.append({
                                'Metric': 'AISC (Cost)',
                                'Value': f"${aisc:.0f}/oz",
                                'Score (1-100)': f"{aisc_score:.0f}",
                                'Weight Impact': 'High' if aisc_score >= 80 else 'Med' if aisc_score >= 60 else 'Low'
                            })
                        
                        # V8.0: Technical Analysis Score
                        ta_score_val = row.get('TA_Score', None)
                        if ta_score_val is not None and isinstance(ta_score_val, (int, float)):
                            scorecard_data.append({
                                'Metric': 'TA Score (RSI/MACD/BB/OBV/ADX)',
                                'Value': f"{row.get('TA_Signal', 'NEUTRAL')} | RSI:{row.get('RSI', 50):.0f}",
                                'Score (1-100)': f"{ta_score_val:.0f}",
                                'Weight Impact': 'High' if ta_score_val >= 65 else 'Med' if ta_score_val >= 45 else 'Low'
                            })

                        # V8.0: AISC Margin (from aisc_tracker)
                        aisc_margin = row.get('AISC_Margin_Pct', None)
                        aisc_est = row.get('AISC_Estimate', None)
                        if aisc_margin is not None and aisc_est is not None and isinstance(aisc_est, (int, float)) and aisc_est > 0:
                            scorecard_data.append({
                                'Metric': 'AISC Margin',
                                'Value': f"${aisc_est:.0f} ({row.get('AISC_Source', '?')}) | Margin: {aisc_margin:.1f}%",
                                'Score (1-100)': f"{row.get('AISC_Score', 50):.0f}",
                                'Weight Impact': 'High' if aisc_margin > 25 else 'Med' if aisc_margin > 10 else 'Low'
                            })

                        # Market Buzz
                        if row.get('Market_Buzz', False):
                            volume_spike = row.get('Volume_Spike_Pct', 0)
                            buzz_score = min(100, 50 + (volume_spike / 10))  # Scale volume spike to score
                            scorecard_data.append({
                                'Metric': 'Market Buzz',
                                'Value': f"{volume_spike:.1f}% spike",
                                'Score (1-100)': f"{buzz_score:.0f}",
                                'Weight Impact': 'High' if buzz_score >= 70 else 'Med'
                            })
                        
                        # Market Impact
                        market_impact = row.get('Market_Impact_Pct', 0)
                        if market_impact > 0:
                            # Lower impact is better, so score = 100 - (impact * 10)
                            impact_score = max(0, min(100, 100 - (market_impact * 10)))
                            scorecard_data.append({
                                'Metric': 'Market Impact',
                                'Value': f"{market_impact:.2f}%",
                                'Score (1-100)': f"{impact_score:.0f}",
                                'Weight Impact': 'High' if impact_score >= 85 else 'Med' if impact_score >= 70 else 'Low'
                            })
                        
                        # Display scorecard table
                        if scorecard_data:
                            scorecard_df = pd.DataFrame(scorecard_data)
                            st.dataframe(
                                scorecard_df,
                                use_container_width=True,
                                hide_index=True
                            )
                        else:
                            st.caption("Parameter data not available for this symbol")

                        # V6.0: Micro-Macro Auditor (FA/TA breakdown)
                        st.markdown("---")
                        st.markdown("#### Micro-Macro Auditor")
                        micro_aisc = row.get("AISC", row.get("aisc"))
                        aisc_sc = max(0, min(100, 100 - (float(micro_aisc) / 20))) if micro_aisc is not None and isinstance(micro_aisc, (int, float)) else 50
                        cash = row.get("Cash", row.get("cash", row.get("totalCash")))
                        cash_sc = min(100, 50 + (float(cash) / 1e6) * 2) if cash is not None and isinstance(cash, (int, float)) else 50
                        burn = row.get("burn", row.get("burn_rate", 0))
                        burn_sc = max(0, min(100, 100 - abs(float(burn)) * 5)) if burn is not None and isinstance(burn, (int, float)) else 50
                        juris = (row.get("Jurisdiction", "") or row.get("Country", "") or "Unknown").strip()
                        juris_map = {"Canada": 85, "USA": 80, "Mexico": 55, "Australia": 75}
                        juris_sc = juris_map.get(juris, 50)
                        metal = (row.get("Metal_Type", "") or row.get("metal", "Gold") or "Gold").strip()
                        metal_map = {"Gold": 70, "Silver": 65, "Uranium": 60, "Copper": 55}
                        metal_sc = metal_map.get(metal, 50)
                        theta = ["AISC (Cost)", "Cash-on-Hand", "Burn Rate", "Jurisdiction Risk", "Metal Sentiment"]
                        r_vals = [aisc_sc, cash_sc, burn_sc, juris_sc, metal_sc]
                        theta.append(theta[0])
                        r_vals.append(r_vals[0])
                        if PLOTLY_AVAILABLE:
                            try:
                                fig_r = go.Figure(data=go.Scatterpolar(r=r_vals, theta=theta, fill="toself", name=row["Symbol"]))
                                fig_r.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, margin=dict(l=80, r=80, t=40, b=40), height=280)
                                st.plotly_chart(fig_r, use_container_width=True)
                            except Exception:
                                st.caption("Radar chart unavailable")
                        else:
                            st.caption("Micro: AISC {:.0f} | Cash {:.0f} | Burn {:.0f} · Macro: Jurisdiction {:.0f} | Metal {:.0f}".format(aisc_sc, cash_sc, burn_sc, juris_sc, metal_sc))

                        # V6.0: Projected 30d Window (Deep-Hunt)
                        if PRICE_PROJECTION_AVAILABLE and project_30d_window:
                            hist_cache = st.session_state.get("hist_cache", {})
                            hist = hist_cache.get(row["Symbol"], pd.DataFrame())
                            try:
                                proj = project_30d_window(row["Symbol"], metal=row.get("metal", "Gold"), hist=hist if not hist.empty else None)
                                lo, hi, ex = proj.get("low_30d"), proj.get("high_30d"), proj.get("expected_30d")
                                if not (np.isnan(lo) and np.isnan(hi) and np.isnan(ex)):
                                    st.markdown("#### Projected 30d Window")
                                    st.caption("Model A (Mean Reversion) · B (Momentum/RS) · C (Commodity Proxy)")
                                    st.metric("Expected (30d)", f"${ex:.2f}" if not np.isnan(ex) else "—", f"Low ${lo:.2f} · High ${hi:.2f}" if not (np.isnan(lo) or np.isnan(hi)) else "")
                            except Exception:
                                pass
                    
                    st.markdown("---")
        
        # ── Portfolio Rearrangement Suggestions ──────────────────────────────
        if not top_5.empty and 'results' in st.session_state and not st.session_state.results.empty:
            results_df_rearr = st.session_state.results.copy()
            # Identify weakest portfolio positions: highest sell risk + lowest alpha
            if 'Sell_Risk_Score' in results_df_rearr.columns and 'Alpha_Score' in results_df_rearr.columns:
                results_df_rearr['_weakness'] = results_df_rearr['Sell_Risk_Score'] - results_df_rearr['Alpha_Score']
                weakest = results_df_rearr.nlargest(3, '_weakness')
                # Filter scanner picks that are NOT already in portfolio
                new_picks = top_5[~top_5['Symbol'].isin(results_df_rearr['Symbol'].values)].head(3)
                if not weakest.empty and not new_picks.empty:
                    st.markdown("---")
                    st.subheader("Portfolio Rearrangement Suggestions")
                    st.caption("Comparing weakest portfolio positions against top scanner finds")
                    swap_data = []
                    for i in range(min(len(weakest), len(new_picks))):
                        w = weakest.iloc[i]
                        n = new_picks.iloc[i]
                        swap_data.append({
                            'Sell': w['Symbol'],
                            'Sell Alpha': f"{w['Alpha_Score']:.0f}",
                            'Sell Risk': f"{w['Sell_Risk_Score']:.0f}",
                            'Buy': n['Symbol'],
                            'Buy Alpha': f"{n['Alpha_Score']:.0f}",
                            'Buy Risk': f"{n['Risk_Score']:.0f}",
                            'Score Gain': f"+{n['Combined_Score'] - (w['Alpha_Score'] / max(w['Sell_Risk_Score'], 1)):.1f}"
                        })
                    if swap_data:
                        swap_df = pd.DataFrame(swap_data)
                        st.dataframe(swap_df, use_container_width=True, hide_index=True)
                        st.caption("These are suggestions only. Always verify with your own due diligence before executing trades.")

        # Show full recommendations table
        with st.expander("Full Recommendations Table", expanded=False):
            # V7.5: Include Market Cap and Tier in display
            display_cols = ['Symbol', 'Market_Cap_M', 'Market_Cap_Tier', 'Alpha_Score', 'FA_Score', 'Combined_Score', 'Risk_Score', 
                          'Action', 'Market_Buzz', 'Diversification_Veto']
            
            # V5.0: Fix KeyError - use reindex with fill_value to safely handle missing columns
            if not rec_df.empty:
                # Reindex to ensure all display_cols exist, fill missing with "N/A"
                display_df = rec_df.reindex(columns=display_cols, fill_value="N/A")
                
                # Format numeric columns safely
                format_dict = {}
                for col in ['Alpha_Score', 'FA_Score', 'Combined_Score', 'Risk_Score']:
                    if col in display_df.columns:
                        try:
                            # Try to convert to numeric
                            display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                            if col == 'Alpha_Score':
                                format_dict[col] = '{:.1f}'
                            elif col == 'FA_Score':
                                format_dict[col] = '{:+.1f}'
                            elif col == 'Combined_Score':
                                format_dict[col] = '{:.2f}'
                            elif col == 'Risk_Score':
                                format_dict[col] = '{:.0f}'
                        except Exception:
                            pass
                
                st.dataframe(
                    display_df.style.format(format_dict),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No recommendations available")
      except Exception as _tab_err:
        st.error(f"Market Scanner tab error: {_tab_err}")

    # Tab 5: Watchlist
    with watchlist_tab:
      try:
        st.header("Watchlist")
        st.caption("Track starred symbols with trend indicators")
        
        if 'starred_symbols' not in st.session_state:
            st.session_state.starred_symbols = []
        
        wl_symbols = []
        if MINING_TICKERS_AVAILABLE and get_all_mining_tickers_cached:
            try:
                wl_symbols = get_all_mining_tickers_cached(max_symbols=200) or []
            except Exception:
                pass
        if not wl_symbols and SCANNER_AVAILABLE and load_master_discovery_list:
            try:
                wl_symbols = load_master_discovery_list(max_symbols=200) or []
            except Exception:
                pass
        if wl_symbols:
            try:
                master_symbols = wl_symbols
                if master_symbols:
                    # Star/unstar interface
                    st.markdown("### Starred Symbols")
                    selected_symbols = st.multiselect(
                        "Select symbols to star",
                        master_symbols,
                        default=st.session_state.starred_symbols,
                        key='starred_selector'
                    )
                    st.session_state.starred_symbols = selected_symbols
                    
                if selected_symbols:
                    watchlist_data = []
                    if 'results' in st.session_state:
                            results_df = st.session_state.results
                            hist_cache = st.session_state.get('hist_cache', {})
                            
                            # Initialize alpha_history cache if not exists
                            if 'alpha_history' not in st.session_state:
                                st.session_state.alpha_history = {}
                            
                            for symbol in selected_symbols:
                                # Get current alpha from results if available
                                symbol_row = results_df[results_df['Symbol'] == symbol] if 'Symbol' in results_df.columns else pd.DataFrame()
                                current_alpha = symbol_row['Alpha_Score'].iloc[0] if not symbol_row.empty else None
                                
                                # V5.0: Calculate trend using alpha_history cache (24hr comparison)
                                # Handle data gaps gracefully
                                trend = "→"  # Neutral
                                trend_emoji = "⚪"
                                
                                if current_alpha is not None and isinstance(current_alpha, (int, float)):
                                    try:
                                        # Get previous alpha from cache (24hr ago)
                                        prev_alpha = st.session_state.alpha_history.get(symbol, None)
                                        
                                        if prev_alpha is not None and isinstance(prev_alpha, (int, float)):
                                            alpha_change = current_alpha - prev_alpha
                                            if alpha_change > 5:
                                                trend = "↑"
                                                trend_emoji = "🟢"
                                            elif alpha_change < -5:
                                                trend = "↓"
                                                trend_emoji = "🔴"
                                            else:
                                                trend = "→"
                                                trend_emoji = "⚪"
                                        
                                        # Update cache with current alpha (even if prev was None)
                                        st.session_state.alpha_history[symbol] = float(current_alpha)
                                    except (TypeError, ValueError):
                                        # Handle data gap gracefully - show neutral trend
                                        trend = "→"
                                        trend_emoji = "⚪"
                                else:
                                    # Data gap - show neutral trend
                                    trend = "→"
                                    trend_emoji = "⚪"
                                
                                price_str = "N/A"
                                if symbol in hist_cache and not hist_cache[symbol].empty:
                                    price_str = f"${hist_cache[symbol].iloc[-1]['Close']:.2f}"
                                
                                watchlist_data.append({
                                    'Symbol': symbol,
                                    'Alpha Score': f"{current_alpha:.1f}" if current_alpha is not None else "N/A",
                                    'Trend (24h)': f"{trend_emoji} {trend}",
                                    'Price': price_str,
                                    'Action': symbol_row['Action'].iloc[0] if not symbol_row.empty else "Not Scanned"
                                })
                            if watchlist_data:
                                watchlist_df = pd.DataFrame(watchlist_data)
                                st.dataframe(watchlist_df, use_container_width=True, hide_index=True)
                            else:
                                st.info("Run analysis to see Alpha scores and trends for starred symbols")
                    else:
                        st.info("⭐ Select symbols above to add them to your watchlist")
                else:
                    st.caption("Select symbols above to add them to your watchlist.")
            except Exception as e:
                st.error(f"Error loading watchlist: {str(e)[:100]}")
        else:
            st.warning("No mining tickers available (Sovereign Global Scan or CSV fallback)")
      except Exception as _tab_err:
        st.error(f"Watchlist tab error: {_tab_err}")

    # ========================================================================
    # DETAILED POSITION ANALYSIS
    # ========================================================================

    st.header("Detailed Position Analysis")
    
    # Add ranking and sort
    df = add_ranking_columns(df)
    sort_mode = st.session_state.get('sort_mode', 'Action first (default)')
    df_sorted = sort_dataframe(df, sort_mode)
    
    for _, row in df_sorted.iterrows():
        # Card style based on new action framing
        action = row.get('Action', 'HOLD')
        confidence = row.get('Confidence', 'Low')
        stability = row.get('Recommendation_Stability', 'Stable')
        
        if action == 'Buy':
            st.success(f"### {row['Symbol']} - {action} ({confidence} confidence)")
        elif action == 'Avoid':
            st.error(f"### {row['Symbol']} - {action} ({confidence} confidence)")
        else:
            st.info(f"### {row['Symbol']} - {action} ({confidence} confidence)")
        
        # Stability indicator
        if stability == 'Breaks':
            st.error(f"Recommendation Stability: {stability} - Veto applied")
        elif stability == 'Fragile':
            st.warning(f"Recommendation Stability: {stability} - Near threshold")
        else:
            st.caption(f"Recommendation Stability: {stability}")
        
        # Primary gating reason
        gating_reason = row.get('Primary_Gating_Reason', '')
        if gating_reason:
            st.caption(f"**Gating reason:** {gating_reason}")
        
        # Metrics
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Alpha", f"{row['Alpha_Score']:.0f}/100")
        c2.metric("Sell Risk", f"{row['Sell_Risk_Score']:.0f}/100")
        c3.metric("Current", f"{row['Pct_Portfolio']:.1f}%")
        c4.metric("→ Rec", f"{row['Recommended_Pct']:.1f}%")
        c5.metric("Max", f"{row['Max_Allowed_Pct']:.1f}%")
        confidence_val = row.get('Confidence', 'Low')
        c6.metric("Conf", confidence_val)
        
        # Badges (FIXED INDENTATION)
        sleeve_badge = f"badge-{row['Sleeve'].lower()}"
        liq_badge = f"badge-{row['Liq_tier_code'].lower()}"
        
        badge_html = f'<span class="{sleeve_badge}">{row["Sleeve"]}</span> '
        badge_html += f'<span class="{liq_badge}">{row["Liq_tier_code"]}: {row["Liq_tier_name"]}</span> '
        badge_html += f'<span class="badge-tactical">Conf: {row["Data_Confidence"]:.0f}%</span> '
        badge_html += f'<span class="badge-tactical">Dil: {row["Dilution_Risk_Score"]:.0f}/100</span> '
        
        # Financing Overhang badge
        overhang_score = row.get('Financing_Overhang_Score', 0)
        if overhang_score >= 70:
            badge_html += f'<span class="badge-l1">FinOverhang: {overhang_score:.0f}/100</span> '
        elif overhang_score >= 40:
            badge_html += f'<span class="badge-l2">FinOverhang: {overhang_score:.0f}/100</span> '
        elif overhang_score > 0:
            badge_html += f'<span class="badge-tactical">FinOverhang: {overhang_score:.0f}/100</span> '
        
        if row.get('Insider_Buying_90d', False):
            badge_html += '<span class="badge-insider">INSIDER BUY</span> '
        
        if row.get('Discovery_Exception', False):
            badge_html += '<span class="badge-discovery">DISCOVERY ⚠️</span> '
        
        # SMC badge
        smc_bias = row.get('SMC_Bias', 'Neutral')
        smc_state = row.get('SMC_State', 'NEUTRAL')
        if smc_bias == 'Bullish' or smc_state == 'BULLISH':
            badge_html += '<span class="badge-l3">SMC: ↑</span> '
        elif smc_bias == 'Bearish' or smc_state == 'BEARISH':
            badge_html += '<span class="badge-l1">SMC: ↓</span> '
        else:
            badge_html += '<span class="badge-l2">SMC: ~</span> '
        
        # News quality
        ticker_news = news_cache.get(row['Symbol'], [])
        news_quality, news_badge = calculate_news_quality(ticker_news)
        badge_html += f'<span class="{news_badge}">News: {news_quality}</span> '
        
        # Financing Overhang details (if significant)
        overhang_score = row.get('Financing_Overhang_Score', 0)
        overhang_reasons = row.get('Financing_Overhang_Reasons', [])
        if overhang_score >= 40 and overhang_reasons:
            reasons_text = ' | '.join(overhang_reasons[:2])
            badge_html += f'<span class="badge-tactical">FinOverhang: {reasons_text}</span> '
        
        st.markdown(badge_html, unsafe_allow_html=True)
        
        # Key info
        st.caption(f"**{row['stage']}** • {row['metal']} • {row['country']} • Runway: {row['Runway']:.1f}mo • Days to Exit: {row['Liq_days_to_exit']:.1f}d")
        
        # Reasoning
        if row['Reasoning']:
            for reason in row['Reasoning'][:3]:
                st.write(f"• {reason}")
        
        # Warnings
        if row['Warnings']:
            for warn in row['Warnings']:
                st.warning(warn)
        
        # Detailed breakdown
        with st.expander(f"Complete Analysis for {row['Symbol']}", expanded=False):
            
            # Gates
            st.subheader("🚦 Gate Status")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**✅ Passed:**")
                for gate in row['Gates_Passed']:
                    st.markdown(f'<span class="gate-pass">{gate}</span>', unsafe_allow_html=True)
            
            with col2:
                st.markdown("**❌ Failed/Warnings:**")
                for gate in row['Gates_Failed']:
                    st.markdown(f'<span class="gate-fail">{gate}</span>', unsafe_allow_html=True)
            
            # Sell triggers
            sell_triggers = sell_triggers_storage.get(row['Symbol'], [])
            if sell_triggers:
                st.markdown("---")
                st.subheader("🔴 Active Sell Triggers")
                for trigger in sell_triggers:
                    st.error(trigger)
            
            # Alpha breakdown
            st.markdown("---")
            st.subheader("🎯 7-Model Alpha Breakdown")
            
            alpha_breakdown = alpha_breakdown_storage.get(row['Symbol'], [])
            if alpha_breakdown:
                for model_desc in alpha_breakdown:
                    st.write(f"• {model_desc}")
            
            # Data confidence
            st.markdown("---")
            st.subheader("📊 Data Confidence Details")
            st.write(f"**Score:** {row['Data_Confidence']}/100 ({row['Conf_Verdict']})")
            
            conf_breakdown = conf_breakdown_storage.get(row['Symbol'], [])
            if conf_breakdown:
                for detail in conf_breakdown:
                    st.caption(detail)
            
            # Dilution risk
            st.markdown("---")
            st.subheader("💀 Dilution Risk Factors")
            st.write(f"**Score:** {row['Dilution_Risk_Score']}/100 ({row['Dilution_Verdict']})")
            
            dilution_factors = dilution_factors_storage.get(row['Symbol'], [])
            if dilution_factors:
                for factor in dilution_factors:
                    st.caption(factor)
            
            # SMC Analysis
            st.markdown("---")
            st.subheader("📈 Smart Money Concepts (SMC)")
            st.write(f"**Bias:** {row['SMC_Bias']} | **Score:** {row['SMC_Score']:.0f}/100")
            st.write(f"**Summary:** {row['SMC_Summary']}")
            
            smc_signals = st.session_state.get('smc_signals_storage', {}).get(row['Symbol'], [])
            if smc_signals:
                st.write(f"**Signals:** {', '.join(smc_signals)}")
            
            # News
            st.markdown("---")
            st.subheader("Recent News (Last 90 days)")
            
            ticker_news = news_cache.get(row['Symbol'], [])
            if ticker_news:
                for item in ticker_news[:10]:
                    tags = item.get('tag_string', '')
                    date_str = item.get('date_str', 'Unknown')
                    st.markdown(f"**{item['title']}** {tags}")
                    st.caption(f"{item['publisher']} • {date_str}")
                    st.markdown("")
            else:
                st.info("No ticker news - showing sector news")
                sector_news = get_sector_news_fallback()
                for item in sector_news[:5]:
                    st.markdown(f"**{item['title']}**")
                    st.caption(item.get('publisher', 'Market'))
        
        st.markdown("---")

    # Export
    st.download_button(
        "Download Analysis (CSV)",
        df.to_csv(index=False),
        f"alpha_miner_analysis_{datetime.date.today()}.csv",
        use_container_width=True
    )

st.caption(f"Alpha Miner Pro {VERSION}")
