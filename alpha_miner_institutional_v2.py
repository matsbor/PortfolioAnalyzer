#!/usr/bin/env python3
"""
ALPHA MINER PRO - WORLD-CLASS CAPITAL ALLOCATION ENGINE
Survival > Alpha | Sell-In-Time Focus | Gate-Based Risk Management

VERSION 2.1-FIXED
- Fixed SMC integration into alpha scoring
- Fixed arbitration wiring (real triggers)
- Added Gold/Silver predictions to header
- Fixed news intelligence (PP closed detection)
- Added market buzz integration
- Fixed portfolio ranking
- Enhanced discovery exception
- Fixed all indentation errors
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import json
import re
from pathlib import Path

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
except:
    YFINANCE = False
    st.session_state.yfinance_available = False
    st.error("⚠️ yfinance not installed. Run: pip install yfinance")

# VERSION TRACKING
VERSION = "2.1-FIXED"
VERSION_DATE = "2026-01-16"
VERSION_FEATURES = [
    "✅ SMC integrated into alpha scoring",
    "✅ Gold & Silver cycle predictions in header",
    "✅ News intelligence (PP closed detection)",
    "✅ Market buzz proxy integration",
    "✅ Portfolio orchestration & ranking",
    "✅ Enhanced discovery exception",
    "✅ Fixed arbitration wiring"
]

st.set_page_config(page_title="Alpha Miner Pro", layout="wide", initial_sidebar_state="expanded")

# Professional styling
st.markdown("""
<style>
    .stApp {background-color: #0a0e1a; color: #e8eaf0;}
    .main {background-color: #0a0e1a;}
    h1, h2, h3 {color: #e8eaf0 !important;}
    
    /* Badges */
    .badge-core {background: #2563eb; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-weight: bold; font-size: 0.85rem; margin: 0 0.25rem;}
    .badge-tactical {background: #f59e0b; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-weight: bold; font-size: 0.85rem; margin: 0 0.25rem;}
    .badge-gambling {background: #ef4444; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-weight: bold; font-size: 0.85rem; margin: 0 0.25rem;}
    .badge-l0 {background: #dc2626; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-weight: bold; font-size: 0.85rem; margin: 0 0.25rem;}
    .badge-l1 {background: #f59e0b; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-weight: bold; font-size: 0.85rem; margin: 0 0.25rem;}
    .badge-l2 {background: #3b82f6; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-weight: bold; font-size: 0.85rem; margin: 0 0.25rem;}
    .badge-l3 {background: #10b981; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-weight: bold; font-size: 0.85rem; margin: 0 0.25rem;}
    .badge-insider {background: #8b5cf6; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-weight: bold; font-size: 0.85rem; margin: 0 0.25rem;}
    .badge-discovery {background: #ec4899; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-weight: bold; font-size: 0.85rem; margin: 0 0.25rem;}
    
    /* Command Center */
    .command-center {background: #1e293b; border: 2px solid #475569; padding: 2rem; border-radius: 15px; margin: 1.5rem 0;}
    .risk-card {background: #7f1d1d; border-left: 5px solid #dc2626; padding: 1.5rem; margin: 0.5rem 0; border-radius: 8px;}
    .opportunity-card {background: #14532d; border-left: 5px solid #16a34a; padding: 1.5rem; margin: 0.5rem 0; border-radius: 8px;}
    .warning-banner {background: #7c2d12; border: 3px solid #ea580c; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; text-align: center;}
    .safe-banner {background: #065f46; border: 3px solid #10b981; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; text-align: center;}
    
    /* Gate status */
    .gate-pass {color: #10b981; font-weight: bold;}
    .gate-fail {color: #ef4444; font-weight: bold;}
    .gate-warning {color: #f59e0b; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

PORTFOLIO_SIZE = 200000  # $200k portfolio

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

CACHE_FILE = Path.home() / '.alpha_miner_cache.json'

def load_cache():
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE) as f: 
                return json.load(f)
    except: 
        pass
    return {}

def save_cache(data):
    try:
        with open(CACHE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except:
        pass

if 'fund_cache' not in st.session_state:
    st.session_state.fund_cache = load_cache()


# =========================================================================
# GOVERNANCE: VALIDATION, STRICT MODE, EVIDENCE PACKS, REPLAY MODE
# =========================================================================

EVIDENCE_DIR = Path.home() / '.alpha_miner_evidence_packs'
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

RISK_PROFILES = {
    "Aggressive": {
        "max_pos_pct": {"L3": 12.0, "L2": 9.0, "L1": 6.0, "L0": 1.0},
        "min_data_confidence_for_buys": 55,
        "strict_downgrade_confidence": 50,
        "sell_risk_floor": 15,
    },
    "Balanced": {
        "max_pos_pct": {"L3": 10.0, "L2": 7.5, "L1": 5.0, "L0": 1.0},
        "min_data_confidence_for_buys": 65,
        "strict_downgrade_confidence": 60,
        "sell_risk_floor": 20,
    },
    "Defensive": {
        "max_pos_pct": {"L3": 7.5, "L2": 6.0, "L1": 4.0, "L0": 1.0},
        "min_data_confidence_for_buys": 75,
        "strict_downgrade_confidence": 70,
        "sell_risk_floor": 25,
    },
}


def get_risk_profile_preset(name: str) -> dict:
    """Return risk profile preset dict with safe default."""
    preset = RISK_PROFILES.get(name) or RISK_PROFILES.get("Balanced")
    # copy so callers can modify without mutating global
    out = dict(preset)
    out["name"] = name if name in RISK_PROFILES else "Balanced"
    return out


def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def validate_data_invariants(df: pd.DataFrame, arg2=None, arg3=None):
    """
    Backwards compatible:
      - validate_data_invariants(df, news_cache)
      - validate_data_invariants(df, alpha_models_storage, news_cache)

    Returns: dict with keys: ok(bool), errors(list[str]), warnings(list[str]), per_symbol(dict)
    """

    # If only 2 args were provided: (df, news_cache)
    if arg3 is None:
        alpha_models_storage = {}
        news_cache = arg2 or {}
    else:
        # 3 args: (df, alpha_models_storage, news_cache)
        alpha_models_storage = arg2 or {}
        news_cache = arg3 or {}

    errors, warnings = [], []
    per_symbol = {}

    # Portfolio-level checks
    required_cols = [
        'Symbol','Price','Market_Value','Pct_Portfolio','Alpha_Score','Sell_Risk_Score',
        'Data_Confidence','Dilution_Risk_Score','Liq_tier_code'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    # Alpha weights sum check (models that exist)
    for sym, models in (alpha_models_storage or {}).items():
        try:
            total_w = 0.0
            for _, v in models.items():
                # stored as weighted contributions already
                total_w += float(v)
            if not (0 <= total_w <= 100):
                warnings.append(f"{sym}: alpha contribution sum out of range: {total_w:.1f}")
        except Exception:
            warnings.append(f"{sym}: could not validate alpha model contributions")

    # Row-level checks
    for _, r in df.iterrows():
        sym = str(r.get('Symbol','')).strip()
        pe = []
        pw = []
        try:
            price = float(r.get('Price', 0) or 0)
            if price <= 0:
                pw.append('price<=0')
            mv = float(r.get('Market_Value', 0) or 0)
            if mv < 0:
                pe.append('market_value<0')
            pct = float(r.get('Pct_Portfolio', 0) or 0)
            if pct < 0 or pct > 100:
                pe.append('pct_portfolio_out_of_range')
            burn = float(r.get('burn', 1) or 1)
            cash = float(r.get('cash', 0) or 0)
            if burn <= 0:
                pw.append('burn<=0 (runway invalid)')
            if cash < 0:
                pe.append('cash<0')

            for col in ['Sell_Risk_Score','Dilution_Risk_Score','Data_Confidence','SMC_Score']:
                if col in df.columns:
                    v = float(r.get(col, 0) or 0)
                    if v < 0 or v > 100:
                        pw.append(f"{col}_out_of_range")

            # News timestamps sanity
            items = (news_cache or {}).get(sym, [])
            if items:
                bad_ts = 0
                for it in items:
                    ts = it.get('timestamp', 0) or 0
                    if ts and (ts < 946684800 or ts > 1893456000):
                        bad_ts += 1
                if bad_ts:
                    pw.append(f"{bad_ts} news items have invalid timestamps")
        except Exception as e:
            pw.append(f"row_validation_exception:{e}")

        if pe or pw:
            per_symbol[sym] = {'errors': pe, 'warnings': pw}

    ok = (len(errors) == 0)
    return {'ok': ok, 'errors': errors, 'warnings': warnings, 'per_symbol': per_symbol}


def enforce_strict_mode(df: pd.DataFrame, validation_results: dict, strict_mode: bool, risk_profile: str):
    """Downgrade actions when inputs/data quality aren't strong enough."""
    if not strict_mode:
        return df, []

    profile = RISK_PROFILES.get(risk_profile, RISK_PROFILES['Balanced'])
    min_conf = profile['min_data_confidence_for_buys']
    downgrades = []

    per_symbol = (validation_results or {}).get('per_symbol', {})

    def _is_buy_action(a: str) -> bool:
        a = (a or '').upper()
        return ('BUY' in a) or ('ADD' in a) or ('ACCUMULATE' in a)

    df2 = df.copy()
    for i, r in df2.iterrows():
        sym = r.get('Symbol','')
        action = r.get('Action','')
        conf = float(r.get('Data_Confidence', 0) or 0)
        issues = per_symbol.get(sym, {})
        has_errors = bool(issues.get('errors'))
        has_warnings = bool(issues.get('warnings'))

        if _is_buy_action(action) and (conf < min_conf or has_errors):
            df2.at[i, 'Action'] = '⚪ HOLD'
            df2.at[i, 'Confidence'] = min(float(r.get('Confidence', 60) or 60), profile['strict_downgrade_confidence'])
            rs = list(r.get('Reasoning') or [])
            rs = rs if isinstance(rs, list) else [str(rs)]
            rs.insert(0, f"STRICT MODE: downgraded due to data confidence ({conf:.0f}) or validation errors")
            df2.at[i, 'Reasoning'] = rs
            downgrades.append(sym)
        elif _is_buy_action(action) and has_warnings and conf < (min_conf + 10):
            # soften but don't fully block
            df2.at[i, 'Action'] = '🔵 ADD ⚠️'
            rs = list(r.get('Reasoning') or [])
            rs = rs if isinstance(rs, list) else [str(rs)]
            rs.insert(0, "STRICT MODE: caution due to validation warnings")
            df2.at[i, 'Reasoning'] = rs

    return df2, downgrades


def create_evidence_pack(
    df: pd.DataFrame,
    portfolio_input: pd.DataFrame,
    cash: float,
    macro_regime: dict,
    news_cache: dict,
    alpha_breakdown_storage: dict,
    sell_triggers_storage: dict,
    dilution_factors_storage: dict,
    conf_breakdown_storage: dict,
    meta: dict | None = None,
    tape_gate: dict | None = None,
):
    """Create a replayable, self-contained evidence pack (zero network calls needed to render)."""
    pack = {
        'evidence_pack_id': f"ep_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'created_at_utc': _now_iso(),
        'app_version': VERSION,
        'app_version_date': VERSION_DATE,
        'meta': meta or {},
        'inputs': {
            'portfolio': portfolio_input.to_dict(orient='records'),
            'cash': float(cash),
        },
        'macro_regime': macro_regime or {},
        'tape_gate': tape_gate or {},
        'results': df.to_dict(orient='records'),
        'caches': {
            'news_cache': news_cache or {},
            'alpha_breakdown_storage': alpha_breakdown_storage or {},
            'sell_triggers_storage': sell_triggers_storage or {},
            'dilution_factors_storage': dilution_factors_storage or {},
            'conf_breakdown_storage': conf_breakdown_storage or {},
        },
    }
    return pack


def save_evidence_pack(pack: dict) -> Path:
    ep_id = pack.get('evidence_pack_id', f"ep_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    path = EVIDENCE_DIR / f"{ep_id}.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(pack, f, indent=2)
    return path


def list_evidence_packs():
    return sorted(EVIDENCE_DIR.glob('ep_*.json'), key=lambda x: x.stat().st_mtime, reverse=True)


def load_evidence_pack(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


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

def calculate_liquidity_metrics(ticker, hist_data, current_price, current_position_value, portfolio_size=PORTFOLIO_SIZE):
    """
    Calculate liquidity metrics and tier classification
    """
    result = {
        'tier_code': 'L0',
        'tier_name': 'Illiquid',
        'dollar_vol_20d': 0,
        'avg_vol_20d': 0,
        'max_position_pct': 1.0,
        'days_to_exit': 99,
        'exit_flag': '⚠️ ILLIQUID'
    }
    
    if hist_data.empty or len(hist_data) < 20:
        return result
    
    try:
        recent = hist_data.tail(20)
        avg_vol = recent['Volume'].mean()
        result['avg_vol_20d'] = avg_vol
        
        # Dollar volume
        dollar_vol = avg_vol * current_price
        result['dollar_vol_20d'] = dollar_vol
        
        # Days to exit (assume 10% daily volume limit)
        if dollar_vol > 0:
            days_to_exit = current_position_value / (dollar_vol * 0.10)
            result['days_to_exit'] = min(days_to_exit, 99)
        
        # Tier classification
        if dollar_vol >= 500000:
            result['tier_code'] = 'L3'
            result['tier_name'] = 'Highly Liquid'
            result['max_position_pct'] = 10.0
            result['exit_flag'] = '✅ L3'
        elif dollar_vol >= 200000:
            result['tier_code'] = 'L2'
            result['tier_name'] = 'Liquid'
            result['max_position_pct'] = 7.5
            result['exit_flag'] = '🟢 L2'
        elif dollar_vol >= 50000:
            result['tier_code'] = 'L1'
            result['tier_name'] = 'Moderate'
            result['max_position_pct'] = 5.0
            result['exit_flag'] = '🟡 L1'
        else:
            result['tier_code'] = 'L0'
            result['tier_name'] = 'Illiquid'
            result['max_position_pct'] = 1.0
            result['exit_flag'] = '⚠️ L0'
    
    except:
        pass
    
    return result

# ============================================================================
# B) DATA CONFIDENCE SCORING
# ============================================================================

def calculate_data_confidence(fund_dict, info_dict, inferred_flags):
    """Calculate confidence in our data"""
    score = 100
    breakdown = []
    
    # Penalize defaults
    if fund_dict.get('burn_source') == 'default':
        score -= 30
        breakdown.append("⚠️ Using default burn rate (-30)")
    elif fund_dict.get('burn_source') == 'netincome':
        score -= 10
        breakdown.append("⚠️ Burn from net income, not cashflow (-10)")
    
    # Penalize inferred data
    if inferred_flags.get('stage_inferred', True):
        score -= 15
        breakdown.append("⚠️ Stage inferred from assets (-15)")
    
    if inferred_flags.get('metal_inferred', True):
        score -= 10
        breakdown.append("⚠️ Metal type inferred (-10)")
    
    # Reward real data
    if info_dict.get('totalRevenue'):
        score += 10
        breakdown.append("✅ Has revenue data (+10)")
    
    if info_dict.get('totalCash'):
        breakdown.append("✅ Has cash data")
    
    score = max(0, min(100, score))
    
    if score >= 80:
        verdict = "HIGH"
    elif score >= 60:
        verdict = "MEDIUM"
    else:
        verdict = "LOW"
    
    return {'score': score, 'verdict': verdict, 'breakdown': breakdown}

# ============================================================================
# C) DILUTION RISK SCORING
# ============================================================================

def calculate_dilution_risk(runway, stage, drawdown_90d, news_items, 
                           cash_missing, burn_missing, insider_buying):
    """Calculate dilution risk"""
    score = 0
    factors = []
    
    # Runway
    if runway < 6:
        score += 40
        factors.append(f"💀 Runway {runway:.1f}mo < 6mo CRITICAL (+40)")
    elif runway < 12:
        score += 20
        factors.append(f"⚠️ Runway {runway:.1f}mo < 12mo (+20)")
    else:
        factors.append(f"✅ Runway {runway:.1f}mo adequate")
    
    # Stage
    if stage == 'Explorer':
        score += 15
        factors.append("⚠️ Explorer stage (+15)")
    
    # Drawdown
    if drawdown_90d > 40:
        score += 15
        factors.append(f"⚠️ Drawdown {drawdown_90d:.0f}% > 40% (+15)")
    
    # News indicators
    low_cash_news = any('low cash' in item.get('title', '').lower() or 
                       'needs cash' in item.get('title', '').lower() 
                       for item in news_items)
    
    financing_news = any('financing' in item.get('title', '').lower() or 
                        'placement' in item.get('title', '').lower() or
                        'offering' in item.get('title', '').lower()
                        for item in news_items)
    
    if low_cash_news:
        score += 20
        factors.append("💀 'Low cash' in news (+20)")
    
    if financing_news:
        score += 15
        factors.append("⚠️ Financing mentioned in news (+15)")
    
    # Data quality
    if cash_missing:
        score += 10
        factors.append("⚠️ Cash data missing (+10)")
    
    if burn_missing:
        score += 10
        factors.append("⚠️ Burn rate uncertain (+10)")
    
    # Insider buying reduces risk
    if insider_buying:
        score = max(0, score - 15)
        factors.append("✅ Insider buying (-15)")
    
    score = min(100, score)
    
    if score >= 70:
        verdict = "CRITICAL"
    elif score >= 50:
        verdict = "HIGH"
    elif score >= 30:
        verdict = "MODERATE"
    else:
        verdict = "LOW"
    
    return {'score': score, 'verdict': verdict, 'factors': factors}

# ============================================================================
# D) NEWS HELPERS
# ============================================================================

def normalize_timestamp(ts):
    """Normalize timestamp to valid Unix timestamp"""
    if ts is None or ts <= 0:
        return None
    
    # Handle milliseconds
    if ts > 1e12:
        ts = ts / 1000
    
    # Validate range (2000-2030)
    if ts < 946684800 or ts > 1893456000:
        return None
    
    return ts

def tag_news(news_items):
    """Tag news items based on content"""
    for item in news_items:
        title_lower = item.get('title', '').lower()
        tags = []
        
        if any(word in title_lower for word in ['financing', 'placement', 'offering', 'capital raise']):
            tags.append('💰')
        if any(word in title_lower for word in ['drill', 'exploration', 'discovers', 'intercepts']):
            tags.append('🔍')
        if any(word in title_lower for word in ['production', 'produces', 'mining']):
            tags.append('⚙️')
        if any(word in title_lower for word in ['acquisition', 'acquires', 'merger']):
            tags.append('🤝')
        
        item['tag_string'] = ' '.join(tags) if tags else ''
    
    return news_items

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
    except:
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
    
    except:
        pass
    
    return result

# ============================================================================
# F) ALPHA MODELS (6 MODELS)
# ============================================================================

def calculate_alpha_models(row, hist_data, benchmark_data):
    """
    Calculate 6-model alpha score
    NOTE: Model 7 (SMC) will be added to this score AFTER SMC calculation
    """
    models = {}
    breakdown = []
    
    # M1: Momentum (20%)
    ret_30d = row.get('Return_30d', 0)
    ret_90d = row.get('Return_90d', 0)
    
    momentum_score = 50
    if ret_30d > 10:
        momentum_score = 75
    elif ret_30d > 5:
        momentum_score = 65
    elif ret_30d < -10:
        momentum_score = 25
    elif ret_30d < -5:
        momentum_score = 35
    
    models['M1_Momentum'] = momentum_score * 0.20
    breakdown.append(f"M1 Momentum: {momentum_score}/100 × 20% = {models['M1_Momentum']:.1f}")
    
    # M2: Value Positioning (15%)
    pct_from_high = row.get('Pct_From_52w_High', 0)
    
    value_score = 50
    if pct_from_high < -40:
        value_score = 80
    elif pct_from_high < -25:
        value_score = 65
    elif pct_from_high > -5:
        value_score = 30
    
    models['M2_Value'] = value_score * 0.15
    breakdown.append(f"M2 Value: {value_score}/100 × 15% = {models['M2_Value']:.1f}")
    
    # M3: Survival Quality (20%)
    runway = row.get('Runway', 12)
    data_conf = row.get('Data_Confidence', 50)
    
    survival_score = 50
    if runway >= 18:
        survival_score = 80
    elif runway >= 12:
        survival_score = 65
    elif runway < 6:
        survival_score = 20
    
    # Adjust by data confidence
    survival_score = survival_score * (data_conf / 100)
    
    models['M3_Survival'] = survival_score * 0.20
    breakdown.append(f"M3 Survival: {survival_score:.0f}/100 × 20% = {models['M3_Survival']:.1f}")
    
    # M4: Dilution Penalty (13%)
    dil_risk = row.get('Dilution_Risk_Score', 50)
    dilution_score = 100 - dil_risk
    
    models['M4_Dilution'] = dilution_score * 0.13
    breakdown.append(f"M4 Dilution: {dilution_score:.0f}/100 × 13% = {models['M4_Dilution']:.1f}")
    
    # M5: Liquidity (8%)
    tier = row.get('Liq_tier_code', 'L0')
    liq_score = {'L3': 90, 'L2': 70, 'L1': 50, 'L0': 20}.get(tier, 50)
    
    models['M5_Liquidity'] = liq_score * 0.08
    breakdown.append(f"M5 Liquidity: {liq_score}/100 × 8% = {models['M5_Liquidity']:.1f}")
    
    # M6: Relative Strength (8%)
    rel_score = 50
    if benchmark_data is not None and not hist_data.empty:
        try:
            stock_ret = ((hist_data['Close'].iloc[-1] - hist_data['Close'].iloc[-90]) / 
                        hist_data['Close'].iloc[-90] * 100) if len(hist_data) >= 90 else 0
            bench_ret = ((benchmark_data['Close'].iloc[-1] - benchmark_data['Close'].iloc[-90]) / 
                        benchmark_data['Close'].iloc[-90] * 100) if len(benchmark_data) >= 90 else 0
            
            outperformance = stock_ret - bench_ret
            if outperformance > 10:
                rel_score = 80
            elif outperformance > 0:
                rel_score = 60
            elif outperformance < -10:
                rel_score = 30
        except:
            pass
    
    models['M6_RelStrength'] = rel_score * 0.08
    breakdown.append(f"M6 RelStrength: {rel_score}/100 × 8% = {models['M6_RelStrength']:.1f}")
    
    # M7: SMC (8%) - will be added later after SMC calculation
    # For now, use neutral 50
    models['M7_SMC'] = 50 * 0.08
    breakdown.append(f"M7 SMC: 50/100 × 8% = {models['M7_SMC']:.1f} (calculated later)")
    
    # M8: Stage/Metal Fit (8%)
    stage = row.get('stage', 'Explorer')
    metal = row.get('metal', 'Gold')
    
    stage_score = 50
    if stage == 'Producer':
        stage_score = 70
    elif stage == 'Developer':
        stage_score = 60
    
    models['M8_StageFit'] = stage_score * 0.08
    breakdown.append(f"M8 StageFit: {stage_score}/100 × 8% = {models['M8_StageFit']:.1f}")
    
    # Calculate total (before SMC adjustment)
    alpha_score = sum(models.values())
    
    return {
        'alpha_score': alpha_score,
        'models': models,
        'breakdown': breakdown
    }

# ============================================================================
# G) SELL RISK
# ============================================================================

def calculate_sell_risk(row, hist_data, ma50, ma200, news_items, macro_regime):
    """Calculate sell risk with triggers"""
    score = 0
    hard_triggers = []
    soft_triggers = []
    
    runway = row.get('Runway', 12)
    price = row.get('Price', 0)
    ret_7d = row.get('Return_7d', 0)
    ret_30d = row.get('Return_30d', 0)
    drawdown = abs(row.get('Drawdown_90d', 0))
    
    # Hard triggers
    if runway < 6:
        score += 50
        hard_triggers.append(f"💀 Runway {runway:.1f}mo < 6mo CRITICAL")
    
    if ma200 > 0 and price < ma200:
        if macro_regime.get('regime') == 'DEFENSIVE':
            score += 30
            hard_triggers.append("💀 Below MA200 + Defensive macro")
        else:
            score += 15
            soft_triggers.append("⚠️ Below MA200")
    
    # Soft triggers
    if drawdown > 50:
        score += 15
        soft_triggers.append(f"⚠️ Drawdown {drawdown:.0f}% > 50%")
    
    if ret_30d < -20:
        score += 10
        soft_triggers.append(f"⚠️ 30d return {ret_30d:.0f}% < -20%")
    
    if ma50 > 0 and price < ma50 * 0.90:
        score += 10
        soft_triggers.append("⚠️ >10% below MA50")
    
    # News triggers
    for item in news_items:
        title_lower = item.get('title', '').lower()
        if any(word in title_lower for word in ['low cash', 'needs financing', 'suspends']):
            score += 15
            soft_triggers.append(f"⚠️ Negative news: {item['title'][:50]}")
            break
    
    score = min(100, score)
    
    if score >= 60:
        verdict = "SELL NOW"
    elif score >= 40:
        verdict = "CONSIDER SELLING"
    elif score >= 20:
        verdict = "WATCH"
    else:
        verdict = "NORMAL"
    
    all_triggers = hard_triggers + soft_triggers
    
    return {
        'score': score,
        'verdict': verdict,
        'hard_triggers': hard_triggers,
        'soft_triggers': soft_triggers,
        'all_triggers': all_triggers
    }

# ============================================================================
# H) MACRO REGIME
# ============================================================================

def calculate_tape_gate(macro_regime, gold_analysis=None, silver_analysis=None):
    """
    Calculate tape/regime gate decision helper.
    Returns: {regime_label, new_buys_allowed, throttle, reasons[]}
    """
    gate = {
        'regime_label': 'NEUTRAL',
        'new_buys_allowed': True,
        'throttle': 1.0,
        'reasons': []
    }
    
    # Use macro_regime inputs
    dxy = macro_regime.get('dxy', 0)
    vix = macro_regime.get('vix', 0)
    allow_new_buys = macro_regime.get('allow_new_buys', True)
    throttle_factor = macro_regime.get('throttle_factor', 1.0)
    regime = macro_regime.get('regime', 'NEUTRAL')
    
    # DXY trend
    dxy_ma = macro_regime.get('dxy_ma', dxy) if 'dxy_ma' in macro_regime else dxy
    if dxy > 0 and dxy_ma > 0:
        if dxy > dxy_ma * 1.05:
            gate['reasons'].append('DXY: Strong (bearish for gold)')
            gate['throttle'] *= 0.8
        elif dxy < dxy_ma * 0.95:
            gate['reasons'].append('DXY: Weak (bullish for gold)')
        else:
            gate['reasons'].append('DXY: Neutral')
    else:
        gate['reasons'].append('DXY: Unknown')
    
    # VIX regime
    if vix > 0:
        if vix > 25:
            gate['regime_label'] = 'DEFENSIVE'
            gate['new_buys_allowed'] = False
            gate['throttle'] = 0.5
            gate['reasons'].append(f'VIX: {vix:.1f} (defensive)')
        elif vix < 15:
            gate['regime_label'] = 'RISK-ON'
            gate['reasons'].append(f'VIX: {vix:.1f} (risk-on)')
        else:
            gate['reasons'].append(f'VIX: {vix:.1f} (neutral)')
    else:
        gate['reasons'].append('VIX: Unknown')
    
    # Metal outlook (if available)
    if gold_analysis and silver_analysis:
        gold_bias = gold_analysis.get('bias_short', 'NEUTRAL')
        silver_bias = silver_analysis.get('bias_short', 'NEUTRAL')
        if 'BEARISH' in str(gold_bias) or 'BEARISH' in str(silver_bias):
            gate['throttle'] *= 0.9
            gate['reasons'].append('Metals: Bearish')
        elif 'BULLISH' in str(gold_bias) or 'BULLISH' in str(silver_bias):
            gate['reasons'].append('Metals: Bullish')
        else:
            gate['reasons'].append('Metals: Neutral')
    else:
        gate['reasons'].append('Metals: Unknown')
    
    # Override with macro_regime settings
    gate['new_buys_allowed'] = allow_new_buys
    gate['throttle'] = min(gate['throttle'], throttle_factor)
    if regime == 'DEFENSIVE':
        gate['regime_label'] = 'DEFENSIVE'
        gate['new_buys_allowed'] = False
    
    return gate

def calculate_macro_regime():
    """Calculate macro regime"""
    regime = {
        'regime': 'NEUTRAL',
        'factors': [],
        'allow_new_buys': True,
        'throttle_factor': 1.0,
        'dxy': 0,
        'vix': 0
    }
    
    if not YFINANCE:
        return regime
    
    try:
        # DXY
        dxy = yf.Ticker("DX-Y.NYB")
        dxy_hist = dxy.history(period="6mo")
        if not dxy_hist.empty:
            dxy_price = dxy_hist['Close'].iloc[-1]
            dxy_ma50 = dxy_hist['Close'].tail(50).mean()
            regime['dxy'] = dxy_price
            
            if dxy_price > dxy_ma50 * 1.05:
                regime['factors'].append("DXY strong (bearish for gold)")
                regime['throttle_factor'] = 0.8
            elif dxy_price < dxy_ma50 * 0.95:
                regime['factors'].append("DXY weak (bullish for gold)")
        
        # VIX
        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(period="3mo")
        if not vix_hist.empty:
            vix_price = vix_hist['Close'].iloc[-1]
            regime['vix'] = vix_price
            
            if vix_price > 25:
                regime['regime'] = 'DEFENSIVE'
                regime['factors'].append(f"VIX {vix_price:.1f} > 25 (defensive)")
                regime['allow_new_buys'] = False
                regime['throttle_factor'] = 0.5
            elif vix_price < 15:
                regime['regime'] = 'RISK-ON'
                regime['factors'].append(f"VIX {vix_price:.1f} < 15 (risk-on)")
        
        # Gold trend
        gold = yf.Ticker("GC=F")
        gold_hist = gold.history(period="6mo")
        if not gold_hist.empty and len(gold_hist) >= 50:
            gold_ma50 = gold_hist['Close'].tail(50).mean()
            gold_price = gold_hist['Close'].iloc[-1]
            
            if gold_price > gold_ma50 * 1.05:
                regime['factors'].append("Gold above MA50 (bullish)")
            elif gold_price < gold_ma50 * 0.95:
                regime['factors'].append("Gold below MA50 (bearish)")
                regime['throttle_factor'] *= 0.9
    
    except:
        pass
    
    if not regime['factors']:
        regime['factors'] = ['Neutral market conditions']
    
    return regime

# ============================================================================
# I) FINANCING OVERHANG CALCULATION
# ============================================================================

def calculate_financing_overhang(news_items, ticker, runway_months):
    """
    Calculate financing overhang score (0-100).
    Integrates with analyze_news_intelligence if v3 available, otherwise lightweight fallback.
    
    Returns: dict with 'score' (0-100) and 'reasons' (list of 2 short strings)
    """
    result = {
        'score': 0.0,
        'reasons': []
    }
    
    if not news_items:
        result['reasons'] = ['No news available']
        return result
    
    # Try v3 integration first
    if INSTITUTIONAL_V3_AVAILABLE:
        try:
            news_intel = analyze_news_intelligence(news_items, ticker)
            status = news_intel.get('financing_status')
            fin_type = news_intel.get('financing_type')
            impact = news_intel.get('financing_impact', 0)
            
            # Find most recent financing event days_ago from news items
            days_ago = None
            for item in news_items:
                ts = item.get('timestamp', 0)
                if ts > 0:
                    try:
                        if ts > 1e12:
                            ts = ts / 1000
                        news_date = datetime.datetime.fromtimestamp(ts)
                        days = (datetime.datetime.now() - news_date).days
                        if days_ago is None or days < days_ago:
                            days_ago = days
                    except:
                        pass
            
            if status == 'CLOSED':
                if days_ago is not None and days_ago <= 7:
                    # PP_CLOSED <=7d: overhang drops materially
                    result['score'] = max(20, 40 + impact)
                    result['reasons'] = [f'Financing closed {days_ago}d ago', 'Runway extended']
                elif days_ago is not None and days_ago <= 30:
                    result['score'] = max(20, 35 + impact)
                    result['reasons'] = [f'Financing closed {days_ago}d ago', 'Recent close']
                else:
                    result['score'] = max(0, 30 + impact)
                    result['reasons'] = ['Financing closed', 'Older event']
            elif status == 'ANNOUNCED' or status == 'PRICED':
                # ANNOUNCED not closed: 60-85
                if fin_type == 'ATM':
                    result['score'] = min(95, 85 + (impact if impact > 0 else 10))
                    result['reasons'] = ['Active ATM', 'Ongoing dilution risk']
                elif fin_type == 'SHELF':
                    result['score'] = min(95, 80 + (impact if impact > 0 else 10))
                    result['reasons'] = ['Shelf filed', 'Dilution imminent']
                else:
                    recency_factor = max(0, 30 - (days_ago or 90)) / 30.0
                    result['score'] = 60 + (25 * recency_factor)
                    result['reasons'] = ['Financing announced', 'Not yet closed']
            elif status == 'NONE':
                result['score'] = 0.0
                result['reasons'] = ['No financing events']
            else:
                # Unknown status
                result['score'] = 10.0
                result['reasons'] = ['Unknown financing status']
            
            return result
        except Exception as e:
            # Fall through to lightweight fallback
            pass
    
    # Lightweight fallback: keyword matching
    financing_keywords = {
        'shelf': ['shelf', 'prospectus', 'registration statement'],
        'atm': ['atm', 'at-the-market', 'at the market'],
        'closed': ['closes', 'closed', 'completes', 'completed', 'closing of'],
        'announced': ['announces', 'proposes', 'intends to', 'plans to', 'seeks']
    }
    
    most_recent_event = None
    most_recent_days = None
    
    for item in news_items:
        title_lower = (item.get('title', '') or '').lower()
        ts = item.get('timestamp', 0)
        
        # Check if financing-related
        is_financing = any(word in title_lower for word in 
                          ['financing', 'placement', 'offering', 'capital raise', 'bought deal'])
        if not is_financing:
            continue
        
        # Determine stage and type
        stage = None
        fin_type = None
        
        if any(word in title_lower for word in financing_keywords['closed']):
            stage = 'CLOSED'
        elif any(word in title_lower for word in financing_keywords['announced']):
            stage = 'ANNOUNCED'
        
        if any(word in title_lower for word in financing_keywords['shelf']):
            fin_type = 'SHELF'
        elif any(word in title_lower for word in financing_keywords['atm']):
            fin_type = 'ATM'
        
        if stage:
            # Calculate days ago
            days_ago = None
            if ts > 0:
                try:
                    if ts > 1e12:
                        ts = ts / 1000
                    news_date = datetime.datetime.fromtimestamp(ts)
                    days_ago = (datetime.datetime.now() - news_date).days
                except:
                    pass
            
            if most_recent_days is None or (days_ago is not None and days_ago < most_recent_days):
                most_recent_event = {'stage': stage, 'type': fin_type or 'PP', 'days_ago': days_ago}
                most_recent_days = days_ago
    
    # Score based on most recent event
    if most_recent_event:
        stage = most_recent_event['stage']
        fin_type = most_recent_event['type']
        days_ago = most_recent_event.get('days_ago')
        
        if stage == 'CLOSED':
            if days_ago is not None and days_ago <= 7:
                result['score'] = 30.0
                result['reasons'] = [f'Financing closed {days_ago}d ago', 'Runway extended']
            else:
                result['score'] = 20.0
                result['reasons'] = ['Financing closed', 'Older event']
        elif stage == 'ANNOUNCED':
            if fin_type == 'ATM':
                result['score'] = 85.0
                result['reasons'] = ['Active ATM', 'Ongoing dilution']
            elif fin_type == 'SHELF':
                result['score'] = 80.0
                result['reasons'] = ['Shelf filed', 'Dilution imminent']
            else:
                recency_factor = max(0, 30 - (days_ago or 90)) / 30.0 if days_ago is not None else 0.5
                result['score'] = 60.0 + (25.0 * recency_factor)
                result['reasons'] = ['Financing announced', 'Not yet closed']
    else:
        result['score'] = 0.0
        result['reasons'] = ['No financing events detected']
    
    return result

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

def arbitrate_final_decision(row, liq_metrics, data_conf, dilution, sell_risk, 
                             alpha_score, macro_regime, discovery, tape_gate=None):
    """
    Final decision arbitration
    """
    decision = {
        'action': '⚪ HOLD',
        'confidence': 50,
        'recommended_pct': row.get('Pct_Portfolio', 0),
        'max_allowed_pct': 5.0,
        'reasoning': [],
        'gates_passed': [],
        'gates_failed': [],
        'warnings': []
    }
    
    # Gate checks
    liq_tier = liq_metrics.get('tier_code', 'L0')
    conf_score = data_conf['score']
    dil_score = dilution['score']
    sell_score = sell_risk['score']
    
    # Tape gate enforcement (if provided)
    if tape_gate:
        if not tape_gate.get('new_buys_allowed', True):
            # Check if this is a buy action
            current_pct = row.get('Pct_Portfolio', 0)
            recommended_pct = row.get('Recommended_Pct', current_pct)
            is_buy_action = recommended_pct > current_pct
            
            if is_buy_action:
                decision['gates_failed'].append("🛑 Tape gate: New buys not allowed")
                # Downgrade to HOLD or REDUCE based on strict mode
                strict_mode = st.session_state.get('strict_mode', False)
                if strict_mode and sell_score >= 30:
                    decision['action'] = '🔴 REDUCE'
                    decision['recommended_pct'] = current_pct * 0.5
                else:
                    decision['action'] = '⚪ HOLD'
                    decision['recommended_pct'] = current_pct
                decision['warnings'].append("Tape gate blocked new buy")
                return decision
        
        # Apply throttle to positive deltas
        throttle = tape_gate.get('throttle', 1.0)
        if throttle < 1.0:
            current_pct = row.get('Pct_Portfolio', 0)
            if decision['recommended_pct'] > current_pct:
                delta = decision['recommended_pct'] - current_pct
                decision['recommended_pct'] = current_pct + (delta * throttle)
                decision['warnings'].append(f"Tape throttle: {throttle:.2f}x applied")
    
    # Hard gates
    if not macro_regime.get('allow_new_buys', True):
        decision['gates_failed'].append("🛑 Defensive macro - no new buys")
        if sell_score >= 30:
            decision['action'] = '🔴 REDUCE'
            decision['recommended_pct'] = row.get('Pct_Portfolio', 0) * 0.5
        return decision
    
    if sell_score >= 60:
        decision['action'] = '🚨 SELL NOW'
        decision['confidence'] = 90
        decision['recommended_pct'] = 0
        decision['reasoning'].extend(sell_risk['hard_triggers'])
        decision['gates_failed'].append(f"🔴 Sell risk {sell_score}/100 CRITICAL")
        return decision
    
    if conf_score < 40:
        decision['gates_failed'].append(f"⚠️ Data confidence {conf_score}/100 too low")
        decision['action'] = '⚪ HOLD'
        return decision
    
    # Size caps
    tier_caps = {'L3': 10.0, 'L2': 7.5, 'L1': 5.0, 'L0': 1.0}
    base_max = tier_caps.get(liq_tier, 1.0)
    
    # Apply discovery exception if granted
    if discovery[0]:
        base_max = min(base_max, 2.5)
        decision['warnings'].append("⚠️ Discovery exception: max 2.5%")
        decision['action'] = '🔵 ADD ⚠️'
    
    # Apply macro throttle
    base_max *= macro_regime.get('throttle_factor', 1.0)
    decision['max_allowed_pct'] = base_max
    
    # Decision logic
    current_pct = row.get('Pct_Portfolio', 0)
    
    if sell_score >= 40:
        decision['action'] = '🔴 REDUCE'
        decision['confidence'] = 75
        decision['recommended_pct'] = current_pct * 0.5
        decision['reasoning'].extend(sell_risk['soft_triggers'][:2])
    
    elif sell_score >= 20:
        decision['action'] = '🟡 TRIM'
        decision['confidence'] = 60
        decision['recommended_pct'] = current_pct * 0.8
    
    elif alpha_score >= 75 and current_pct < base_max:
        if alpha_score >= 85:
            decision['action'] = '🟢 STRONG BUY'
            decision['confidence'] = 90
        else:
            decision['action'] = '🟢 BUY'
            decision['confidence'] = 80
        
        decision['recommended_pct'] = min(base_max, current_pct + 2.0)
    
    elif alpha_score >= 60 and current_pct < base_max * 0.8:
        decision['action'] = '🔵 ADD'
        decision['confidence'] = 70
        decision['recommended_pct'] = min(base_max * 0.8, current_pct + 1.0)
    
    else:
        decision['action'] = '⚪ HOLD'
        decision['confidence'] = 60
        decision['recommended_pct'] = current_pct
    
    decision['reasoning'].append(f"Alpha: {alpha_score:.0f}/100")
    decision['gates_passed'].append(f"✅ Liquidity: {liq_tier}")
    decision['gates_passed'].append(f"✅ Confidence: {conf_score}/100")
    
    return decision

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
        'stage': 'Explorer',
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
        except:
            if info.get('netIncome') and info['netIncome'] < 0:
                result['burn'] = abs(info['netIncome']) / 12_000_000
                result['burn_source'] = 'netincome'
        
        # Stage
        revenue = info.get('totalRevenue', 0)
        if revenue:
            result['stage'] = 'Producer' if revenue > 10_000_000 else 'Developer'
            result['inferred_flags']['stage_inferred'] = False
        else:
            assets = info.get('totalAssets', 0)
            result['stage'] = 'Developer' if assets > 50_000_000 else 'Explorer'
        
        # Country
        if info.get('country'):
            result['country'] = info['country']
        
        # Metal
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
    
    except:
        pass
    
    return result

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
    except:
        return []

@st.cache_data(ttl=3600)
def get_benchmark_data(metal):
    """Fetch benchmark"""
    if not YFINANCE:
        return None
    
    try:
        ticker = "SILJ" if metal == 'Silver' else "GDXJ"
        bench = yf.Ticker(ticker)
        return bench.history(period="6mo")
    except:
        return None

# ============================================================================
# RENDER MORNING TAPE (SIMPLE VERSION)
# ============================================================================

def render_morning_tape_simple(gold_analysis, silver_analysis, metal_regime):
    """Simple morning tape renderer"""
    st.markdown("---")
    st.header("📊 METAL OUTLOOK")
    
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
    st.markdown("### 📋 Portfolio Guidance")
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

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = DEFAULT_PORTFOLIO.copy()
if 'cash' not in st.session_state:
    st.session_state.cash = 39569.65

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Version display
    st.markdown("---")
    st.caption(f"**Alpha Miner Pro {VERSION}**")
    st.caption(f"Release: {VERSION_DATE}")
    with st.expander("📋 Features in this version"):
        for feature in VERSION_FEATURES:
            st.caption(f"• {feature}")
    st.markdown("---")
    


    st.markdown("### 🧭 Risk Governance")
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

    st.markdown("### ♻️ Replay Mode (Offline)")
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

        existing = list_saved_evidence_packs()
        if existing:
            pick = st.selectbox("Or load saved pack", options=[p['file'] for p in existing], index=0)
            if st.button("Load selected pack"):
                st.session_state.replay_pack = load_evidence_pack(pick)
                st.success(f"Loaded: {pick}")

        if st.session_state.get('replay_pack'):
            st.caption("OFFLINE_MODE is ON. Analysis will render from the evidence pack.")

    st.markdown("---")

    st.header("📊 Portfolio")
    port_size = st.number_input("Portfolio Size", value=PORTFOLIO_SIZE, step=10000)
    
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
    
    st.markdown("### 💰 Cash")
    cash = st.number_input("Available", value=float(st.session_state.cash), step=1000.0, label_visibility="collapsed")
    st.session_state.cash = cash
    
    st.markdown("---")
    st.markdown("### 📊 Display Options")
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

st.title("💎 ALPHA MINER PRO")
st.caption("World-Class Capital Allocation Engine • Survival > Alpha • Sell-In-Time Focus")

# Get macro regime
macro_regime = calculate_macro_regime()

# Display macro banner
if macro_regime['regime'] == 'DEFENSIVE':
    st.markdown(f"""
    <div class="warning-banner">
        <h2>⚠️ DEFENSIVE MODE - NO NEW BUYS</h2>
        <p>{' | '.join(macro_regime['factors'])}</p>
        <p><strong>DXY:</strong> {macro_regime.get('dxy', 'N/A')} | <strong>VIX:</strong> {macro_regime.get('vix', 'N/A')}</p>
    </div>
    """, unsafe_allow_html=True)
elif macro_regime['regime'] == 'RISK-ON':
    st.markdown(f"""
    <div class="safe-banner">
        <h2>✅ RISK-ON MODE - GREEN LIGHT</h2>
        <p>{' | '.join(macro_regime['factors'])}</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info(f"📊 **NEUTRAL MODE** • {' | '.join(macro_regime['factors'])}")

# Display Tape/Regime Gate
if 'tape_gate' in st.session_state:
    tape_gate = st.session_state.tape_gate
    st.markdown("### 🚦 Tape / Regime Gate")
    col1, col2, col3 = st.columns(3)
    with col1:
        dxy_reason = next((r for r in tape_gate['reasons'] if 'DXY' in r), 'DXY: Unknown')
        st.caption(f"**{dxy_reason}**")
    with col2:
        vix_reason = next((r for r in tape_gate['reasons'] if 'VIX' in r), 'VIX: Unknown')
        st.caption(f"**{vix_reason}**")
    with col3:
        buys_status = "✅ Yes" if tape_gate['new_buys_allowed'] else "❌ No"
        st.caption(f"**New buys allowed? {buys_status}**")
    if tape_gate['throttle'] < 1.0:
        st.warning(f"⚠️ Throttle factor: {tape_gate['throttle']:.2f} (reduced position sizing)")

# Analysis Button
if st.button("🚀 RUN WORLD-CLASS ANALYSIS", type="primary", use_container_width=True):
    progress = st.progress(0, text="Starting analysis...")
    
    df = st.session_state.portfolio.copy()
    
    # Analyze metals FIRST
    if INSTITUTIONAL_V2_AVAILABLE or INSTITUTIONAL_V3_AVAILABLE:
        progress.progress(3, text="🪙 Analyzing Gold & Silver cycles...")
        
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
            
            progress.progress(100, text="✅ Replay complete!")
            st.success("✅ World-class analysis complete!")
            st.rerun()
    
    # Fetch price data (normal mode - network calls allowed)
    progress.progress(10, text="📊 Fetching market data...")
    
    hist_cache = {}
    for idx, row in df.iterrows():
        if YFINANCE and not replay_mode:
            try:
                hist = yf.Ticker(row['Symbol']).history(period="2y")
                if not hist.empty:
                    hist_cache[row['Symbol']] = hist
                    
                    df.at[idx, 'Price'] = hist['Close'].iloc[-1]
                    df.at[idx, 'Volume'] = hist['Volume'].mean()
                    
                    df.at[idx, 'Return_7d'] = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-7]) / hist['Close'].iloc[-7] * 100) if len(hist) >= 7 else 0
                    df.at[idx, 'Return_30d'] = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-30]) / hist['Close'].iloc[-30] * 100) if len(hist) >= 30 else 0
                    df.at[idx, 'Return_90d'] = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-90]) / hist['Close'].iloc[-90] * 100) if len(hist) >= 90 else 0
                    
                    high_52w = hist['High'].tail(252).max() if len(hist) >= 252 else hist['High'].max()
                    low_52w = hist['Low'].tail(252).min() if len(hist) >= 252 else hist['Low'].min()
                    df.at[idx, 'Pct_From_52w_High'] = ((hist['Close'].iloc[-1] - high_52w) / high_52w * 100)
                    df.at[idx, 'Pct_From_52w_Low'] = ((hist['Close'].iloc[-1] - low_52w) / low_52w * 100)
                    
                    df.at[idx, 'Volatility_60d'] = hist['Close'].pct_change().tail(60).std() * 100 if len(hist) >= 60 else 5
                    
                    df.at[idx, 'MA50'] = hist['Close'].tail(50).mean() if len(hist) >= 50 else 0
                    df.at[idx, 'MA200'] = hist['Close'].tail(200).mean() if len(hist) >= 200 else 0
                    
                    if len(hist) >= 90:
                        high_90d = hist['High'].tail(90).max()
                        df.at[idx, 'Drawdown_90d'] = ((hist['Close'].iloc[-1] - high_90d) / high_90d * 100)
                    else:
                        df.at[idx, 'Drawdown_90d'] = 0
            except:
                df.at[idx, 'Price'] = 0
    
    progress.progress(25, text="🔍 Fetching fundamentals...")
    
    info_storage = {}
    inferred_storage = {}
    
    for idx, row in df.iterrows():
        fund = get_fundamentals_with_tracking(row['Symbol'])
        for k, v in fund.items():
            if k not in ['info_dict', 'inferred_flags']:
                df.at[idx, k] = v
        
        info_storage[row['Symbol']] = fund['info_dict']
        inferred_storage[row['Symbol']] = fund['inferred_flags']
    
    progress.progress(35, text="📰 Fetching news...")
    
    news_cache = {}
    for idx, row in df.iterrows():
        news = get_news_for_ticker(row['Symbol'])
        news_cache[row['Symbol']] = news
    
    # Calculate position metrics
    # IMPORTANT: Pct_Portfolio calculated vs total portfolio value (equity + cash)
    df['Market_Value'] = df['Quantity'] * df['Price']
    df['Gain_Loss'] = df['Market_Value'] - df['Cost_Basis']
    df['Return_Pct'] = (df['Gain_Loss'] / df['Cost_Basis'] * 100)
    total_mv = df['Market_Value'].sum()
    total_value = total_mv + st.session_state.cash  # Total portfolio value
    df['Pct_Portfolio'] = (df['Market_Value'] / total_value * 100)  # vs TOTAL VALUE
    df['Runway'] = df['cash'] / df['burn']
    
    progress.progress(45, text="🚦 Liquidity analysis...")
    
    for idx, row in df.iterrows():
        hist = hist_cache.get(row['Symbol'], pd.DataFrame())
        liq = calculate_liquidity_metrics(row['Symbol'], hist, row['Price'], row['Market_Value'], port_size)
        
        for k, v in liq.items():
            df.at[idx, f'Liq_{k}'] = v
    
    progress.progress(55, text="📊 Data confidence...")
    
    conf_breakdown_storage = {}
    
    for idx, row in df.iterrows():
        fund_dict = {'burn_source': row['burn_source']}
        info_dict = info_storage.get(row['Symbol'], {})
        inferred = inferred_storage.get(row['Symbol'], {})
        
        conf = calculate_data_confidence(fund_dict, info_dict, inferred)
        df.at[idx, 'Data_Confidence'] = conf['score']
        df.at[idx, 'Conf_Verdict'] = conf['verdict']
        
        conf_breakdown_storage[row['Symbol']] = conf['breakdown']
    
    progress.progress(60, text="💀 Dilution risk...")
    
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
    progress.progress(62, text="💰 Financing overhang analysis...")
    
    for idx, row in df.iterrows():
        news = news_cache.get(row['Symbol'], [])
        runway_months = row.get('Runway', 12.0)
        
        overhang = calculate_financing_overhang(news, row['Symbol'], runway_months)
        
        df.at[idx, 'Financing_Overhang_Score'] = overhang['score']
        df.at[idx, 'Financing_Overhang_Reasons'] = overhang['reasons']
    
    # CRITICAL FIX: Calculate SMC BEFORE alpha scoring
    progress.progress(65, text="📈 Calculating SMC signals...")
    
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
    
    # Now calculate alpha WITH SMC scores available
    progress.progress(75, text="🎯 7-model alpha scoring...")
    
    alpha_models_storage = {}
    alpha_breakdown_storage = {}
    
    for idx, row in df.iterrows():
        hist = hist_cache.get(row['Symbol'], pd.DataFrame())
        benchmark = get_benchmark_data(row.get('metal', 'Gold'))
        
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
    
    progress.progress(82, text="🔴 Sell-in-time analysis...")
    
    sell_triggers_storage = {}
    
    for idx, row in df.iterrows():
        hist = hist_cache.get(row['Symbol'], pd.DataFrame())
        news = news_cache.get(row['Symbol'], [])
        
        sell = calculate_sell_risk(row, hist, row.get('MA50', 0), row.get('MA200', 0), news, macro_regime)
        
        df.at[idx, 'Sell_Risk_Score'] = sell['score']
        df.at[idx, 'Sell_Verdict'] = sell['verdict']
        
        # CRITICAL FIX: Store real triggers
        sell_triggers_storage[row['Symbol']] = sell['all_triggers']
    
    progress.progress(90, text="✅ Final arbitration...")
    
    # Classify sleeves
    for idx, row in df.iterrows():
        liq_tier = row['Liq_tier_code']
        daily_vol = row['Liq_dollar_vol_20d']
        conf = row['Data_Confidence']
        
        if conf < 50:
            df.at[idx, 'Sleeve'] = 'GAMBLING'
        elif row['stage'] in ['Producer', 'Developer'] and daily_vol >= 200000 and conf >= 80:
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
        
        decision = arbitrate_final_decision(
            row, liq_metrics, data_conf, dilution, sell_risk,
            row['Alpha_Score'], macro_regime, discovery, tape_gate
        )
        
        decisions.append(decision)
    
    for k in ['action', 'confidence', 'recommended_pct', 'max_allowed_pct', 'reasoning', 
              'gates_passed', 'gates_failed', 'warnings']:
        df[k.title()] = [d[k] for d in decisions]
    
    # Post-process: Strict mode + Financing Overhang enforcement
    strict_mode = st.session_state.get('strict_mode', False)
    if strict_mode:
        for idx, row in df.iterrows():
            overhang_score = row.get('Financing_Overhang_Score', 0)
            action = row.get('Action', '')
            is_buy = 'BUY' in action or 'ADD' in action or 'STRONG BUY' in action
            
            if is_buy and overhang_score >= 80:
                # Check if exception: PP_CLOSED <=7d AND runway >= 9 months
                reasons = row.get('Financing_Overhang_Reasons', [])
                has_recent_close = any('closed' in str(r).lower() and '7d' in str(r) for r in reasons)
                runway_months = row.get('Runway', 0)
                
                if not (has_recent_close and runway_months >= 9):
                    # Block the buy
                    df.at[idx, 'Action'] = '⚪ HOLD'
                    current_warnings = row.get('Warnings', [])
                    if isinstance(current_warnings, list):
                        current_warnings.append("STRICT: Financing overhang ≥80 blocks buy")
                    else:
                        current_warnings = ["STRICT: Financing overhang ≥80 blocks buy"]
                    df.at[idx, 'Warnings'] = current_warnings
                    df.at[idx, 'Recommended_Pct'] = row.get('Pct_Portfolio', 0)
    
    progress.progress(100, text="✅ Complete!")

    # ------------------------------------------------------------------------
    # Governance: validate, apply strict mode, build evidence pack
    # ------------------------------------------------------------------------
    # Invariants check (used by STRICT MODE & Trust Panel). Use the real model storage.
    validation = validate_data_invariants(df, alpha_models_storage, news_cache)
    st.session_state.validation = validation

    if strict_mode:
        preset = get_risk_profile_preset(st.session_state.get('risk_profile', 'Balanced'))
        df, strict_downgrades = enforce_strict_mode(df, validation, st.session_state.get('strict_mode', False), st.session_state.get('risk_profile','Balanced'))

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
                'strict_mode': bool(st.session_state.get('strict_mode', False))
            },
            tape_gate=st.session_state.get('tape_gate')
        )
        st.session_state.evidence_pack = pack
        save_evidence_pack(pack)
    except Exception:
        st.session_state.evidence_pack = None

    st.session_state.results = df
    st.session_state.news_cache = news_cache
    st.session_state.macro_regime = macro_regime
    st.session_state.conf_breakdown_storage = conf_breakdown_storage
    st.session_state.dilution_factors_storage = dilution_factors_storage
    st.session_state.alpha_breakdown_storage = alpha_breakdown_storage
    st.session_state.sell_triggers_storage = sell_triggers_storage
    
    st.success("✅ World-class analysis complete!")
    st.rerun()

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

# Helper functions for ranking
def add_ranking_columns(df):
    """Add ranking columns"""
    ACTION_RANK = {
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
    st.header("📊 DAILY EXECUTIVE SUMMARY")
    
    total_mv = df['Market_Value'].sum()
    total_value = total_mv + cash
    total_cost = df['Cost_Basis'].sum()
    total_pl = total_mv - total_cost
    total_pl_pct = (total_pl / total_cost * 100) if total_cost > 0 else 0
    
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
    for action in ['🟢 STRONG BUY', '🟢 BUY', '🔵 ADD', '⚪ HOLD']:
        count = action_counts.get(action, 0)
        if count > 0:
            col4.caption(f"{action}: {count}")
    
    st.markdown("---")
    
    # Top opportunities/risks
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("✅ ADD NOW")
        addable = df[df['Action'].str.contains('BUY|ADD', na=False)]
        if len(addable) > 0:
            for _, row in addable.nlargest(3, 'Alpha_Score').iterrows():
                amt = total_value * (row['Recommended_Pct'] / 100)
                st.success(f"**{row['Symbol']}**")
                st.caption(f"Rec: {row['Recommended_Pct']:.1f}% (${amt:,.0f})")
                st.caption(f"Alpha: {row['Alpha_Score']:.0f}")
        else:
            st.info("No adds")
    
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
    
    adds = df[df['Action'].str.contains('BUY|ADD', na=False)]
    if len(adds) > 0 and macro_regime.get('allow_new_buys', True):
        st.success("**Consider Adding:**")
        for _, row in adds.nlargest(2, 'Alpha_Score').iterrows():
            amt = total_value * (row['Recommended_Pct'] / 100)
            st.write(f"• {row['Symbol']}: {row['Recommended_Pct']:.1f}% (${amt:,.0f})")
    
    trims = df[df['Action'].str.contains('TRIM|REDUCE|SELL', na=False)]
    if len(trims) > 0:
        st.warning("**Consider Trimming:**")
        for _, row in trims.nlargest(2, 'Sell_Risk_Score').iterrows():
            diff = row['Market_Value'] - total_value * (row['Recommended_Pct'] / 100)
            if diff > 0:
                st.write(f"• {row['Symbol']}: Trim ${diff:,.0f}")

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

    # Evidence pack / diff / rebalance
    with st.expander("🧾 Evidence Pack, Diff, and Rebalance", expanded=False):
        pack = st.session_state.get('evidence_pack')
        if pack:
            st.caption(f"Pack id: {pack.get('pack_id')} • created: {pack.get('created_at_local')}")

            # Diff vs previous pack (if any)
            prev_id = st.session_state.get('prev_pack_id_for_diff')
            if prev_id:
                prev = load_evidence_pack(prev_id)
            else:
                prev = None

            if prev and prev.get('results'):
                import pandas as _pd
                prev_df = _pd.DataFrame(prev['results'])
                cur_df = df.copy()
                key = 'Symbol'
                cols = ['Action','Alpha_Score','Sell_Risk_Score','Recommended_Pct','Pct_Portfolio']
                merged = prev_df[[key]+cols].merge(cur_df[[key]+cols], on=key, how='outer', suffixes=('_prev','_cur'))
                changes = []
                for _,r in merged.iterrows():
                    sym = r.get(key)
                    for c in cols:
                        a=r.get(f'{c}_prev')
                        b=r.get(f'{c}_cur')
                        if (a!=b) and not (_pd.isna(a) and _pd.isna(b)):
                            changes.append({'Symbol':sym,'Field':c,'Prev':a,'Now':b})
                if changes:
                    st.write("**What changed vs previous run**")
                    st.dataframe(_pd.DataFrame(changes).head(200), use_container_width=True, hide_index=True)
                else:
                    st.caption("No diffs vs previous run")
            else:
                st.caption("Run at least twice (or select a previous evidence pack) to see diffs.")

            # Rebalance table
            import pandas as _pd
            cur = df[['Symbol','Pct_Portfolio','Recommended_Pct','Market_Value']].copy()
            cur['Delta_Pct'] = cur['Recommended_Pct'] - cur['Pct_Portfolio']
            cur['Target_$'] = total_value * (cur['Recommended_Pct'] / 100.0)
            cur['Delta_$'] = cur['Target_$'] - cur['Market_Value']
            cur['Trade'] = cur['Delta_$'].apply(lambda x: 'BUY' if x>0 else ('SELL' if x<0 else 'HOLD'))
            st.write("**Suggested rebalance (vs total portfolio value)**")
            st.dataframe(cur.sort_values('Delta_$', ascending=False), use_container_width=True, hide_index=True)

            st.download_button(
                "Download evidence pack (JSON)",
                data=json.dumps(pack, indent=2),
                file_name=f"alpha_evidence_{pack.get('pack_id')}.json",
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
    st.header("🎯 COMMAND CENTER")
    
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
        st.subheader("✅ TOP 3 ADD OPPORTUNITIES")
        addable = df[df['Action'].str.contains('BUY|ADD', na=False)]
        if len(addable) > 0:
            top_adds = addable.nlargest(3, 'Alpha_Score')
            
            for _, row in top_adds.iterrows():
                st.markdown(f"""
                <div class="opportunity-card">
                    <h4>{row['Symbol']} - Alpha: {row['Alpha_Score']:.0f}/100</h4>
                    <p><strong>Action:</strong> {row['Action']}</p>
                    <p><strong>Current:</strong> {row['Pct_Portfolio']:.1f}% → <strong>Rec:</strong> {row['Recommended_Pct']:.1f}%</p>
                    <p><strong>Max Allowed:</strong> {row['Max_Allowed_Pct']:.1f}%</p>
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
    # DETAILED POSITION ANALYSIS
    # ========================================================================
    
    st.markdown("---")
    st.header("📊 Detailed Position Analysis")
    
    # Add ranking and sort
    df = add_ranking_columns(df)
    sort_mode = st.session_state.get('sort_mode', 'Action first (default)')
    df_sorted = sort_dataframe(df, sort_mode)
    
    for _, row in df_sorted.iterrows():
        # Card style
        if 'BUY' in row['Action']:
            st.success(f"### {row['Symbol']} - {row['Action']}")
        elif 'SELL' in row['Action'] or 'REDUCE' in row['Action']:
            st.error(f"### {row['Symbol']} - {row['Action']}")
        else:
            st.info(f"### {row['Symbol']} - {row['Action']}")
        
        # Metrics
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Alpha", f"{row['Alpha_Score']:.0f}/100")
        c2.metric("Sell Risk", f"{row['Sell_Risk_Score']:.0f}/100")
        c3.metric("Current", f"{row['Pct_Portfolio']:.1f}%")
        c4.metric("→ Rec", f"{row['Recommended_Pct']:.1f}%")
        c5.metric("Max", f"{row['Max_Allowed_Pct']:.1f}%")
        c6.metric("Conf", f"{row['Confidence']:.0f}%")
        
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
        with st.expander(f"🔍 Complete Analysis for {row['Symbol']}", expanded=False):
            
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
            st.subheader("📰 Recent News (Last 90 days)")
            
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
        "📥 Download Complete Analysis",
        df.to_csv(index=False),
        f"alpha_miner_analysis_{datetime.date.today()}.csv",
        use_container_width=True
    )

st.caption(f"💎 Alpha Miner Pro {VERSION} • Survival > Alpha • Sell-In-Time")