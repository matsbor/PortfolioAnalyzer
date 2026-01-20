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
)

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

# MODEL_ROLES, RISK_PROFILES, PORTFOLIO_SIZE imported from alpha_miner_core

# VERSION TRACKING
VERSION = "2.2-PRODUCTION"
VERSION_DATE = "2026-01-16"
VERSION_FEATURES = [
    "✅ SMC integrated into alpha scoring",
    "✅ Gold & Silver cycle predictions in header",
    "✅ News intelligence (PP closed detection)",
    "✅ Market buzz proxy integration",
    "✅ Portfolio orchestration & ranking",
    "✅ Enhanced discovery exception",
    "✅ Fixed arbitration wiring",
    "✅ Model governance with veto logic",
    "✅ Confidence-based decision framing",
    "✅ Watchlist & Quick Analysis"
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

# RISK_PROFILES, get_risk_profile_preset, validate_data_invariants, enforce_strict_mode,
# create_evidence_pack, save_evidence_pack, list_evidence_packs, load_evidence_pack
# imported from alpha_miner_core

def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

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

# arbitrate_final_decision imported from alpha_miner_core
# Note: When calling, pass strict_mode parameter: arbitrate_final_decision(..., strict_mode=st.session_state.get('strict_mode', False))

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
        except:
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

# get_benchmark_data imported from alpha_miner_core
# Note: Core version uses YFINANCE_AVAILABLE instead of YFINANCE, but behavior is the same

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
    
    st.markdown("### ⚙️ Determinism / Run Settings")
    
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
    
    # Watchlist feature
    st.markdown("### 📋 Watchlist")
    WATCHLIST_FILE = Path.home() / '.alpha_miner_watchlist.json'
    
    # Initialize watchlist in session state
    if 'watchlist' not in st.session_state:
        if WATCHLIST_FILE.exists():
            try:
                with open(WATCHLIST_FILE, 'r') as f:
                    st.session_state.watchlist = json.load(f)
            except:
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
                    except:
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
                    except:
                        pass
                    st.rerun()
    else:
        st.caption("No symbols in watchlist")
    
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

# Trust Panel (no expander)
st.markdown("---")
st.markdown("### 🔒 Trust Panel")
trust_col1, trust_col2, trust_col3, trust_col4, trust_col5, trust_col6 = st.columns(6)

run_timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
trust_col1.caption(f"**Run:** {run_timestamp[:19]}Z")

replay_status = "ON" if st.session_state.get('replay_mode', False) else "OFF"
trust_col2.caption(f"**Replay:** {replay_status}")

strict_status = "ON" if st.session_state.get('strict_mode', False) else "OFF"
trust_col3.caption(f"**Strict Mode:** {strict_status}")

risk_profile = st.session_state.get('risk_profile', 'Balanced')
trust_col4.caption(f"**Risk Profile:** {risk_profile}")

# Validation status
validation = st.session_state.get('validation', {})
if validation:
    val_ok = validation.get('ok', False)
    val_warnings = len(validation.get('warnings', []))
    val_errors = len(validation.get('errors', []))
    if val_ok and val_warnings == 0:
        trust_col5.caption(f"**Validation:** ✅ PASS")
    elif val_ok:
        trust_col5.caption(f"**Validation:** ⚠️ PASS ({val_warnings} warnings)")
    else:
        trust_col5.caption(f"**Validation:** ❌ FAIL ({val_errors} errors)")
else:
    trust_col5.caption("**Validation:** ⏳ Pending")

# Evidence pack ID
evidence_pack = st.session_state.get('evidence_pack')
if evidence_pack:
    ep_id = evidence_pack.get('evidence_pack_id', 'Unknown')
    trust_col6.caption(f"**Evidence Pack:** {ep_id[:20]}...")
else:
    trust_col6.caption("**Evidence Pack:** None")

# Network calls indicator
network_calls_made = "NO" if st.session_state.get('replay_mode', False) else "YES"
st.caption(f"**Network calls made:** {network_calls_made}")

st.markdown("---")

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

# Quick Analysis for Watchlist
if st.session_state.get('watchlist'):
    st.markdown("---")
    st.markdown("### 🔍 Quick Analysis (Watchlist)")
    st.info("**Exploratory analysis — not a recommendation**")
    
    if st.button("🚀 Run Quick Analysis on Watchlist", type="secondary", use_container_width=True):
        import traceback
        try:
            watchlist_symbols = st.session_state.get('watchlist', [])
            if not watchlist_symbols:
                st.warning("Watchlist is empty")
            else:
                quick_results = []
                
                # Get macro regime (needed for tape gate)
                macro_regime = calculate_macro_regime()
                
                # Calculate tape gate
                gold_analysis = st.session_state.get('gold_analysis')
                silver_analysis = st.session_state.get('silver_analysis')
                tape_gate = calculate_tape_gate(macro_regime, gold_analysis, silver_analysis)
                
                for symbol in watchlist_symbols:
                    result = {
                        'Symbol': symbol,
                        'Financing_Overhang_Score': 0,
                        'Financing_Overhang_Reasons': [],
                        'Dilution_Risk_Score': 0,
                        'Liquidity_Tier': 'Unknown',
                        'Tape_Gate_Status': 'Unknown',
                        'Catalyst_Detected': False
                    }
                    
                    # Get news (skip network if replay mode)
                    replay_mode = st.session_state.get('replay_mode', False)
                    news_cache = st.session_state.get('news_cache', {})
                    
                    if replay_mode and news_cache:
                        news = news_cache.get(symbol, [])
                    else:
                        if not replay_mode:
                            if YFINANCE:
                                news = get_news_for_ticker(symbol)
                            else:
                                news = []
                        else:
                            news = []
                    
                    # Calculate financing overhang
                    overhang = calculate_financing_overhang(news, symbol, 12.0)
                    result['Financing_Overhang_Score'] = overhang['score']
                    result['Financing_Overhang_Reasons'] = overhang['reasons']
                    
                    # Calculate dilution risk (simplified - no full row data)
                    runway_months = 12.0  # Default assumption
                    dilution = calculate_dilution_risk(
                        runway_months, 'Explorer', 0, news, True, True, False
                    )
                    result['Dilution_Risk_Score'] = dilution['score']
                    
                    # Get liquidity (simplified)
                    if YFINANCE and not replay_mode:
                        try:
                            hist = yf.Ticker(symbol).history(period="1mo")
                            if not hist.empty:
                                price = hist['Close'].iloc[-1]
                                volume = hist['Volume'].mean()
                                mv = price * volume * 20  # Approximate 20-day dollar volume
                                liq = calculate_liquidity_metrics(symbol, hist, price, mv, 100000)
                                result['Liquidity_Tier'] = liq.get('tier_code', 'L0')
                        except:
                            result['Liquidity_Tier'] = 'Unknown'
                    
                    # Tape gate status
                    result['Tape_Gate_Status'] = "New buys allowed" if tape_gate.get('new_buys_allowed', True) else "New buys blocked"
                    
                    # Catalyst detection (simplified - check financing events)
                    if overhang['reasons']:
                        catalyst_keywords = ['closed', 'atm', 'shelf', 'announced']
                        if any(kw in str(r).lower() for r in overhang['reasons'] for kw in catalyst_keywords):
                            result['Catalyst_Detected'] = True
                    
                    quick_results.append(result)
                
                # Store results
                st.session_state.quick_analysis_results = quick_results
                st.success(f"Quick analysis complete for {len(quick_results)} symbol(s)")
                st.rerun()
        
        except Exception as e:
            st.error(f"Quick analysis failed: {type(e).__name__}: {str(e)}")
            with st.expander("Technical details", expanded=False):
                st.code(traceback.format_exc(), language='python')
    
    # Display quick analysis results
    if 'quick_analysis_results' in st.session_state:
        results = st.session_state.quick_analysis_results
        if results:
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True, hide_index=True)

# Analysis Button
if st.button("🚀 RUN WORLD-CLASS ANALYSIS", type="primary", use_container_width=True):
    import traceback
    
    try:
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
        
        progress.progress(35, text="📰 Fetching news...")
        
        news_cache = {}
        for idx, row in df.iterrows():
            if not replay_mode:
                news = get_news_for_ticker(row['Symbol'])
                news_cache[row['Symbol']] = news
            else:
                # In replay mode, news should come from evidence pack
                news_cache[row['Symbol']] = []
        
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
            
            decision = arbitrate_final_decision(
                row, liq_metrics, data_conf, dilution, sell_risk,
                row['Alpha_Score'], macro_regime, discovery, tape_gate,
                strict_mode=st.session_state.get('strict_mode', False)
            )
            
            decisions.append(decision)
        
        for k in ['action', 'confidence', 'recommended_pct', 'max_allowed_pct', 'reasoning', 
                  'gates_passed', 'gates_failed', 'warnings', 'primary_gating_reason', 'veto_applied', 'veto_model']:
            df[k.title()] = [d.get(k, '') for d in decisions]
        
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
        
        progress.progress(100, text="✅ Complete!")

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
        st.session_state.news_cache = news_cache
        st.session_state.macro_regime = macro_regime
        st.session_state.conf_breakdown_storage = conf_breakdown_storage
        st.session_state.dilution_factors_storage = dilution_factors_storage
        st.session_state.alpha_breakdown_storage = alpha_breakdown_storage
        st.session_state.sell_triggers_storage = sell_triggers_storage
        
        st.success("✅ World-class analysis complete!")
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
        st.info("💡 The app is still running. You can try again or check the technical details above.")

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
    st.markdown("### 🚨 Alerts")
    
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
            except:
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
            st.markdown("### 📊 What Changed Since Last Run?")
            
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
                    except:
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
                st.info("💡 No prior evidence pack found. Run analysis again to see changes.")

            # Rebalance plan table
            st.markdown("### 💰 Rebalance Plan")
            
            allow_leverage = st.toggle("Allow leverage (buys can exceed cash)", value=False, key="allow_leverage")
            
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
                    except:
                        return 0.0
                
                rebalance_df_sorted = rebalance_df.copy()
                rebalance_df_sorted['_sort_val'] = rebalance_df_sorted['$ Trade'].apply(sort_key)
                rebalance_df_sorted = rebalance_df_sorted.sort_values('_sort_val', ascending=False).drop(columns=['_sort_val'])
                
                st.dataframe(rebalance_df_sorted, use_container_width=True, hide_index=True)
                
                # CSV download
                csv_data = rebalance_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Rebalance Plan (CSV)",
                    data=csv_data,
                    file_name=f"rebalance_plan_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
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
    # DETAILED POSITION ANALYSIS
    # ========================================================================
    
    st.markdown("---")
    st.header("📊 Detailed Position Analysis")
    
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
            st.error(f"⚠️ Recommendation Stability: {stability} - Veto applied")
        elif stability == 'Fragile':
            st.warning(f"⚠️ Recommendation Stability: {stability} - Near threshold")
        else:
            st.caption(f"✅ Recommendation Stability: {stability}")
        
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