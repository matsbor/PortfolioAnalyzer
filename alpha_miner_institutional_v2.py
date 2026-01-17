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
                           cash_missing, burn_missing, insider_buying, financing_status=None):
    """Calculate dilution risk with financing lifecycle awareness"""
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
    
    if low_cash_news:
        score += 20
        factors.append("💀 'Low cash' in news (+20)")
    
    # Financing lifecycle impact
    if financing_status == 'PP_CLOSED':
        score = max(0, score - 10)  # PP closed reduces risk
        factors.append("✅ PP Closed - dilution risk reduced (-10)")
    elif financing_status == 'ATM':
        score += 25  # ATM increases risk significantly
        factors.append("⚠️ ATM offering active (+25)")
    elif financing_status == 'SHELF':
        score += 25  # Shelf increases risk significantly
        factors.append("⚠️ Shelf offering active (+25)")
    elif financing_status == 'ANNOUNCED':
        score += 15  # Announced increases risk
        factors.append("⚠️ Financing announced (+15)")
    elif financing_status == 'FINANCING_MENTIONED':
        score += 10
        factors.append("⚠️ Financing mentioned in news (+10)")
    
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
    """Normalize timestamp to valid Unix timestamp, filtering out bogus dates (1964/1970/epoch)"""
    if ts is None or ts <= 0:
        return None
    
    # Handle milliseconds
    if ts > 1e12:
        ts = ts / 1000
    
    # Validate range (2000-2030) - rejects epoch 0, 1964, 1970, etc.
    # 946684800 = Jan 1, 2000
    # 1893456000 = Jan 1, 2030
    if ts < 946684800 or ts > 1893456000:
        return None
    
    return ts

def classify_financing_status(news_items):
    """
    Classify financing status from news items with time decay
    Prioritizes ticker news with valid timestamps within 90 days
    Returns: 'PP_CLOSED', 'ANNOUNCED', 'ATM', 'SHELF', 'FINANCING_MENTIONED', or None
    """
    if not news_items:
        return None
    
    now = datetime.datetime.now().timestamp()
    current_status = None
    most_recent_date = 0
    
    # Sort by: 1) valid timestamps first, 2) most recent first, 3) ticker source before sector fallback
    sorted_news = sorted(news_items, key=lambda x: (
        not x.get('timestamp_valid', False),  # Valid timestamps first
        -x.get('timestamp', 0),  # Most recent first (negative for descending)
        x.get('source', 'ticker') == 'sector_fallback'  # Ticker news before sector
    ))
    
    for item in sorted_news:
        title_lower = item.get('title', '').lower()
        ts = item.get('timestamp', 0)
        timestamp_valid = item.get('timestamp_valid', False)
        
        # Prioritize valid timestamps within 90 days
        if timestamp_valid and ts > 0:
            days_old = (now - ts) / 86400
            if days_old > 90:  # Focus on last 90 days for financing status
                continue
        elif not timestamp_valid or ts <= 0:
            # Invalid timestamps get lower priority (only if no valid timestamp news found)
            if most_recent_date > 0:
                continue
        
        # Check for specific financing lifecycle stages
        if any(phrase in title_lower for phrase in ['pp closed', 'private placement closed', 'financing closed', 'closed financing']):
            if ts > most_recent_date:
                current_status = 'PP_CLOSED'
                most_recent_date = ts
        elif any(phrase in title_lower for phrase in ['atm', 'at-the-market', 'at the market offering']):
            if ts > most_recent_date:
                current_status = 'ATM'
                most_recent_date = ts
        elif any(phrase in title_lower for phrase in ['shelf offering', 'shelf registration', 'shelf prospectus']):
            if ts > most_recent_date:
                current_status = 'SHELF'
                most_recent_date = ts
        elif any(phrase in title_lower for phrase in ['announces financing', 'announces placement', 'announces offering', 
                                                        'proposed financing', 'intends to raise', 'plans to raise']):
            if ts > most_recent_date:
                current_status = 'ANNOUNCED'
                most_recent_date = ts
        elif any(word in title_lower for word in ['financing', 'placement', 'offering', 'capital raise']) and current_status is None:
            if ts > most_recent_date:
                current_status = 'FINANCING_MENTIONED'
                most_recent_date = ts
    
    return current_status

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
    Calculate 8-model alpha score with full transparency
    Returns models dict with metadata (name, inventor, weight, raw_score, contribution, explanation)
    """
    models = {}
    breakdown = []
    
    # M1: Momentum (20%) - Price momentum over 30d
    ret_30d = row.get('Return_30d', 0)
    momentum_score = 50
    if ret_30d > 10:
        momentum_score = 75
    elif ret_30d > 5:
        momentum_score = 65
    elif ret_30d < -10:
        momentum_score = 25
    elif ret_30d < -5:
        momentum_score = 35
    
    contribution = momentum_score * 0.20
    models['M1_Momentum'] = {
        'name': 'Price Momentum',
        'inventor': 'Technical Analysis (Universal)',
        'weight_percent': 20.0,
        'raw_score_0_100': momentum_score,
        'contribution_points': contribution,
        'explanation': f"30d return: {ret_30d:.1f}%"
    }
    breakdown.append(f"M1 Momentum: {momentum_score}/100 × 20% = {contribution:.1f}")
    
    # M2: Value Positioning (15%) - Relative to 52w high
    pct_from_high = row.get('Pct_From_52w_High', 0)
    value_score = 50
    if pct_from_high < -40:
        value_score = 80
    elif pct_from_high < -25:
        value_score = 65
    elif pct_from_high > -5:
        value_score = 30
    
    contribution = value_score * 0.15
    models['M2_Value'] = {
        'name': 'Value Positioning',
        'inventor': 'Technical Analysis (Universal)',
        'weight_percent': 15.0,
        'raw_score_0_100': value_score,
        'contribution_points': contribution,
        'explanation': f"{pct_from_high:.1f}% from 52w high"
    }
    breakdown.append(f"M2 Value: {value_score}/100 × 15% = {contribution:.1f}")
    
    # M3: Survival Quality (20%) - Runway adjusted by data confidence
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
    
    contribution = survival_score * 0.20
    models['M3_Survival'] = {
        'name': 'Survival Quality',
        'inventor': 'Financial Analysis (Custom)',
        'weight_percent': 20.0,
        'raw_score_0_100': survival_score,
        'contribution_points': contribution,
        'explanation': f"Runway: {runway:.1f}mo, adjusted by confidence {data_conf:.0f}%"
    }
    breakdown.append(f"M3 Survival: {survival_score:.0f}/100 × 20% = {contribution:.1f}")
    
    # M4: Dilution Penalty (13%) - Inverse of dilution risk
    dil_risk = row.get('Dilution_Risk_Score', 50)
    dilution_score = 100 - dil_risk
    
    contribution = dilution_score * 0.13
    models['M4_Dilution'] = {
        'name': 'Dilution Risk Penalty',
        'inventor': 'Financial Analysis (Custom)',
        'weight_percent': 13.0,
        'raw_score_0_100': dilution_score,
        'contribution_points': contribution,
        'explanation': f"Inverse of dilution risk: {dil_risk:.0f}/100"
    }
    breakdown.append(f"M4 Dilution: {dilution_score:.0f}/100 × 13% = {contribution:.1f}")
    
    # M5: Liquidity (8%) - Tier-based
    tier = row.get('Liq_tier_code', 'L0')
    liq_score = {'L3': 90, 'L2': 70, 'L1': 50, 'L0': 20}.get(tier, 50)
    
    contribution = liq_score * 0.08
    models['M5_Liquidity'] = {
        'name': 'Liquidity Score',
        'inventor': 'Market Microstructure (Custom)',
        'weight_percent': 8.0,
        'raw_score_0_100': liq_score,
        'contribution_points': contribution,
        'explanation': f"Tier: {tier}"
    }
    breakdown.append(f"M5 Liquidity: {liq_score}/100 × 8% = {contribution:.1f}")
    
    # M6: Relative Strength (8%) - Stock vs benchmark (Robert Levy, 1967)
    rel_score = 50
    explanation = "No benchmark data"
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
            
            explanation = f"Outperformance: {outperformance:.1f}% vs benchmark"
        except:
            pass
    
    contribution = rel_score * 0.08
    models['M6_RelStrength'] = {
        'name': 'Relative Strength',
        'inventor': 'Robert Levy (1967)',
        'weight_percent': 8.0,
        'raw_score_0_100': rel_score,
        'contribution_points': contribution,
        'explanation': explanation
    }
    breakdown.append(f"M6 RelStrength: {rel_score}/100 × 8% = {contribution:.1f}")
    
    # M7: SMC (8%) - Will be updated after SMC calculation
    contribution = 50 * 0.08
    models['M7_SMC'] = {
        'name': 'Smart Money Concepts',
        'inventor': 'ICT / Smart Money Concepts (2020s)',
        'weight_percent': 8.0,
        'raw_score_0_100': 50,
        'contribution_points': contribution,
        'explanation': 'Calculated separately (placeholder)'
    }
    breakdown.append(f"M7 SMC: 50/100 × 8% = {contribution:.1f} (calculated later)")
    
    # M8: Stage/Metal Fit (8%)
    stage = row.get('stage', 'Explorer')
    metal = row.get('metal', 'Gold')
    
    stage_score = 50
    if stage == 'Producer':
        stage_score = 70
    elif stage == 'Developer':
        stage_score = 60
    
    contribution = stage_score * 0.08
    models['M8_StageFit'] = {
        'name': 'Stage/Metal Fit',
        'inventor': 'Fundamental Analysis (Custom)',
        'weight_percent': 8.0,
        'raw_score_0_100': stage_score,
        'contribution_points': contribution,
        'explanation': f"{stage} stage, {metal} focus"
    }
    breakdown.append(f"M8 StageFit: {stage_score}/100 × 8% = {contribution:.1f}")
    
    # Calculate total (before SMC adjustment)
    alpha_score = sum(m['contribution_points'] for m in models.values())
    
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
# I) DISCOVERY EXCEPTION
# ============================================================================

def check_discovery_exception(row, liq_metrics, alpha_score, data_confidence, 
                              dilution_risk, momentum_ok):
    """
    Check if discovery exception applies
    CRITICAL: Requires Days_to_Exit <= 10, Alpha >= 85, Confidence >= 70, Dilution < 70
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
    
    # CRITICAL: Days_to_Exit must be <= 10
    days_to_exit = liq_metrics.get('days_to_exit', 99)
    if days_to_exit > 10:
        return (False, f"Days_to_Exit {days_to_exit:.1f} > 10 (required: ≤10)")
    
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
    return (True, f"High conviction: Alpha {alpha_score:.0f}, Days_to_Exit {days_to_exit:.1f} ≤ 10, momentum confirmed")

# ============================================================================
# J) FINAL ARBITRATION
# ============================================================================

def arbitrate_final_decision(row, liq_metrics, data_conf, dilution, sell_risk, 
                             alpha_score, macro_regime, discovery):
    """
    Final decision arbitration
    CRITICAL: High sell risk MUST downgrade to REDUCE/SELL
    Discovery exception caps at 2.5% and adds ⚠️ to action
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
    sell_score = sell_risk.get('score', 0)
    
    # CRITICAL: Sell risk takes precedence - high sell risk downgrades action
    # Ensure we use real triggers from sell_risk dict
    hard_triggers = sell_risk.get('hard_triggers', [])
    soft_triggers = sell_risk.get('soft_triggers', [])
    
    if sell_score >= 60:
        decision['action'] = '🚨 SELL NOW'
        decision['confidence'] = 90
        decision['recommended_pct'] = 0
        decision['reasoning'].extend(hard_triggers)
        decision['gates_failed'].append(f"🔴 Sell risk {sell_score}/100 CRITICAL")
        return decision
    
    # Hard gates
    if not macro_regime.get('allow_new_buys', True):
        decision['gates_failed'].append("🛑 Defensive macro - no new buys")
        if sell_score >= 30:
            decision['action'] = '🔴 REDUCE'
            decision['recommended_pct'] = row.get('Pct_Portfolio', 0) * 0.5
            decision['reasoning'].extend(soft_triggers[:2])
        return decision
    
    if conf_score < 40:
        decision['gates_failed'].append(f"⚠️ Data confidence {conf_score}/100 too low")
        decision['action'] = '⚪ HOLD'
        return decision
    
    # CRITICAL: High dilution risk downgrades to REDUCE/SELL
    if dil_score >= 70:
        if sell_score >= 30:
            decision['action'] = '🔴 REDUCE'
            decision['confidence'] = 80
            decision['recommended_pct'] = row.get('Pct_Portfolio', 0) * 0.5
            decision['reasoning'].append(f"💀 Dilution risk {dil_score}/100 CRITICAL")
            decision['reasoning'].extend(soft_triggers[:2])
            return decision
    
    # Size caps
    tier_caps = {'L3': 10.0, 'L2': 7.5, 'L1': 5.0, 'L0': 1.0}
    base_max = tier_caps.get(liq_tier, 1.0)
    
    # Apply discovery exception if granted (cap at 2.5%)
    if discovery[0]:
        base_max = min(base_max, 2.5)
        decision['warnings'].append("⚠️ Discovery exception: max 2.5%")
    
    # Apply macro throttle
    base_max *= macro_regime.get('throttle_factor', 1.0)
    decision['max_allowed_pct'] = base_max
    
    # Decision logic (sell risk still checked)
    current_pct = row.get('Pct_Portfolio', 0)
    
    if sell_score >= 40:
        decision['action'] = '🔴 REDUCE'
        decision['confidence'] = 75
        decision['recommended_pct'] = current_pct * 0.5
        decision['reasoning'].extend(soft_triggers[:2])
    
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
        
        # Add discovery warning to action if applicable
        if discovery[0]:
            decision['action'] = decision['action'].replace('BUY', 'BUY ⚠️')
    
    elif alpha_score >= 60 and current_pct < base_max * 0.8:
        decision['action'] = '🔵 ADD'
        decision['confidence'] = 70
        decision['recommended_pct'] = min(base_max * 0.8, current_pct + 1.0)
        
        # Add discovery warning to action if applicable
        if discovery[0]:
            decision['action'] = '🔵 ADD ⚠️'
    
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
def get_news_for_ticker(ticker, metal='Gold'):
    """Fetch news with timestamp validation and sector fallback"""
    if not YFINANCE:
        return []
    
    formatted_news = []
    has_valid_timestamps = False
    
    try:
        stock = yf.Ticker(ticker)
        news = stock.news[:25]
        
        for item in news:
            ts = None
            for field in ['providerPublishTime', 'published_at', 'pubDate']:
                if field in item:
                    ts = normalize_timestamp(item[field])
                    if ts:
                        break
            
            timestamp_valid = ts is not None and ts > 0
            if timestamp_valid:
                has_valid_timestamps = True
            
            formatted_news.append({
                'title': item.get('title', ''),
                'publisher': item.get('publisher', ''),
                'link': item.get('link', '#'),
                'timestamp': ts if ts else 0,
                'timestamp_valid': timestamp_valid,
                'date_str': datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d') if ts else 'Date: Unknown',
                'source': 'ticker'
            })
    except:
        pass
    
    # Sector fallback if no ticker news or no valid timestamps
    if len(formatted_news) == 0 or not has_valid_timestamps:
        try:
            # Select sector proxy based on metal type
            if metal == 'Silver':
                sector_ticker = "SILJ"
            else:
                sector_ticker = "GDXJ"  # Gold/other miners
            
            sector = yf.Ticker(sector_ticker)
            sector_news = sector.news[:8]
            
            for item in sector_news:
                ts = None
                for field in ['providerPublishTime', 'published_at', 'pubDate']:
                    if field in item:
                        ts = normalize_timestamp(item[field])
                        if ts:
                            break
                
                timestamp_valid = ts is not None and ts > 0
                formatted_news.append({
                    'title': f"[Sector Fallback] {item.get('title', '')}",
                    'publisher': item.get('publisher', 'Sector'),
                    'link': item.get('link', '#'),
                    'timestamp': ts if ts else 0,
                    'timestamp_valid': timestamp_valid,
                    'date_str': datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d') if ts else 'Date: Unknown',
                    'source': 'sector_fallback'
                })
        except:
            pass
    
    return tag_news(formatted_news)

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
    cash = st.number_input("Available", value=float(st.session_state.get('cash', 39569.65)), step=1000.0, label_visibility="collapsed")
    st.session_state.cash = cash
    
    st.markdown("---")
    st.markdown("### 📊 Insider Signal")
    insider_buying_override = st.checkbox(
        "Insider Buying Detected? (Last 90d)",
        value=st.session_state.get('insider_buying_override', False),
        help="If checked, adds +10 points to Alpha_Score for ALL tickers (manual override)"
    )
    st.session_state.insider_buying_override = insider_buying_override
    
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

# Display metal outlook at top if available
if 'gold_analysis' in st.session_state and 'silver_analysis' in st.session_state:
    gold_analysis = st.session_state.get('gold_analysis', {})
    silver_analysis = st.session_state.get('silver_analysis', {})
    metal_regime = st.session_state.get('metal_regime', {})
    
    st.markdown("---")
    st.header("📊 METAL OUTLOOK & PORTFOLIO POSTURE")
    
    col1, col2, col3 = st.columns(3)
    
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
    
    with col3:
        st.subheader("📋 Portfolio Posture")
        posture = metal_regime.get('regime', 'NEUTRAL')
        if 'BEARISH' in posture or 'DEFENSIVE' in posture:
            st.error(f"🛑 **{posture}** - Reduce risk, favor producers")
        elif 'BULLISH' in posture or 'RISK-ON' in posture:
            st.success(f"✅ **{posture}** - Normal risk appetite")
        else:
            st.info(f"📊 **{posture}** - Cautious approach")
        
        # SMC summary if available
        if 'results' in st.session_state:
            df_results = st.session_state.get('results', pd.DataFrame())
            if not df_results.empty and 'SMC_Bias' in df_results.columns:
                bullish_smc = len(df_results[df_results['SMC_Bias'] == 'Bullish'])
                bearish_smc = len(df_results[df_results['SMC_Bias'] == 'Bearish'])
                st.caption(f"SMC Signals: {bullish_smc} ↑ Bullish, {bearish_smc} ↓ Bearish")
        
        # Portfolio posture implication
        posture_impl = metal_regime.get('regime', 'NEUTRAL')
        if 'BEARISH' in posture_impl or 'DEFENSIVE' in posture_impl:
            st.caption("**Portfolio Posture Implication:** Risk-off environment. Reduce exposure, favor producers, preserve capital.")
        elif 'BULLISH' in posture_impl or 'RISK-ON' in posture_impl:
            st.caption("**Portfolio Posture Implication:** Risk-on environment. Normal risk appetite, tactical opportunities available.")
        else:
            st.caption("**Portfolio Posture Implication:** Neutral/cautious. Selective additions, maintain defensive positions.")

# Display macro banner
if macro_regime.get('regime') == 'DEFENSIVE':
    st.markdown(f"""
    <div class="warning-banner">
        <h2>⚠️ DEFENSIVE MODE - NO NEW BUYS</h2>
        <p>{' | '.join(macro_regime['factors'])}</p>
        <p><strong>DXY:</strong> {macro_regime.get('dxy', 'N/A')} | <strong>VIX:</strong> {macro_regime.get('vix', 'N/A')}</p>
    </div>
    """, unsafe_allow_html=True)
elif macro_regime.get('regime') == 'RISK-ON':
    st.markdown(f"""
    <div class="safe-banner">
        <h2>✅ RISK-ON MODE - GREEN LIGHT</h2>
        <p>{' | '.join(macro_regime['factors'])}</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info(f"📊 **NEUTRAL MODE** • {' | '.join(macro_regime['factors'])}")

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
    
    # Fetch price data
    progress.progress(10, text="📊 Fetching market data...")
    
    hist_cache = {}
    for idx, row in df.iterrows():
        if YFINANCE:
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
        metal = row.get('metal', 'Gold') if 'metal' in df.columns else 'Gold'
        news = get_news_for_ticker(row['Symbol'], metal)
        news_cache[row['Symbol']] = news
    
    # Calculate position metrics
    # IMPORTANT: Pct_Portfolio calculated vs total portfolio value (equity + cash)
    df['Market_Value'] = df['Quantity'] * df['Price']
    df['Gain_Loss'] = df['Market_Value'] - df['Cost_Basis']
    df['Return_Pct'] = (df['Gain_Loss'] / df['Cost_Basis'] * 100)
    total_mv = df['Market_Value'].sum()
    total_value = total_mv + st.session_state.get('cash', 0)  # Total portfolio value
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
    financing_status_storage = {}
    
    for idx, row in df.iterrows():
        news = news_cache.get(row['Symbol'], [])
        
        # Classify financing status
        financing_status = classify_financing_status(news)
        financing_status_storage[row['Symbol']] = financing_status
        df.at[idx, 'Financing_Status'] = financing_status if financing_status else 'NONE'
        
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
            insider,
            financing_status
        )
        
        df.at[idx, 'Dilution_Risk_Score'] = dil['score']
        df.at[idx, 'Dilution_Verdict'] = dil['verdict']
        
        dilution_factors_storage[row['Symbol']] = dil['factors']
    
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
        
        # CRITICAL FIX: Add SMC score to alpha (update model structure)
        smc_score = row.get('SMC_Score', 50)
        if isinstance(alpha_result['models'].get('M7_SMC'), dict):
            # Update model metadata structure
            alpha_result['models']['M7_SMC']['raw_score_0_100'] = smc_score
            alpha_result['models']['M7_SMC']['contribution_points'] = smc_score * 0.08
            alpha_result['models']['M7_SMC']['explanation'] = f"SMC Bias: {row.get('SMC_Bias', 'Neutral')}, Score: {smc_score:.0f}/100"
        else:
            # Fallback: convert old format to new format
            contribution = smc_score * 0.08
            alpha_result['models']['M7_SMC'] = {
                'name': 'Smart Money Concepts',
                'inventor': 'ICT / Smart Money Concepts (2020s)',
                'weight_percent': 8.0,
                'raw_score_0_100': smc_score,
                'contribution_points': contribution,
                'explanation': f"SMC Bias: {row.get('SMC_Bias', 'Neutral')}, Score: {smc_score:.0f}/100"
            }
        
        alpha_result['breakdown'][-2] = f"M7 SMC: {smc_score}/100 × 8% = {alpha_result['models']['M7_SMC']['contribution_points']:.1f}"
        
        # Recalculate total
        if isinstance(list(alpha_result['models'].values())[0], dict):
            alpha_result['alpha_score'] = sum(m.get('contribution_points', 0) for m in alpha_result['models'].values())
        else:
            alpha_result['alpha_score'] = sum(alpha_result['models'].values())
        
        base_alpha = alpha_result['alpha_score']
        
        # Apply insider buying override if enabled
        insider_override = st.session_state.get('insider_buying_override', False)
        if insider_override:
            base_alpha += 10.0
            alpha_result['breakdown'].append(f"Insider Buying Override: +10.0 points (manual)")
        
        df.at[idx, 'Alpha_Score'] = base_alpha
        
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
        
        decision = arbitrate_final_decision(
            row, liq_metrics, data_conf, dilution, sell_risk,
            row['Alpha_Score'], macro_regime, discovery
        )
        
        decisions.append(decision)
    
    for k in ['action', 'confidence', 'recommended_pct', 'max_allowed_pct', 'reasoning', 
              'gates_passed', 'gates_failed', 'warnings']:
        df[k.title()] = [d[k] for d in decisions]
    
    progress.progress(100, text="✅ Complete!")
    
    st.session_state.results = df
    st.session_state.news_cache = news_cache
    st.session_state.macro_regime = macro_regime
    st.session_state.conf_breakdown_storage = conf_breakdown_storage
    st.session_state.dilution_factors_storage = dilution_factors_storage
    st.session_state.alpha_breakdown_storage = alpha_breakdown_storage
    st.session_state.alpha_models_storage = alpha_models_storage
    st.session_state.sell_triggers_storage = sell_triggers_storage
    st.session_state.financing_status_storage = financing_status_storage
    
    st.success("✅ World-class analysis complete!")
    st.rerun()

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

# Helper functions for ranking
def add_ranking_columns(df):
    """Add ranking columns with explicit action priority (handles ⚠️ actions)"""
    # Normalize action string for ranking (remove ⚠️ for comparison)
    ACTION_RANK = {
        '🟢 STRONG BUY': 7, '🟢 BUY': 6, '🟢 BUY ⚠️': 6,
        '🔵 ADD': 5, '🔵 ADD ⚠️': 5, '🔵 ACCUMULATE': 5,
        '⚪ HOLD': 4, '🟡 TRIM': 3, 
        '🔴 REDUCE': 2, '🔴 SELL': 1, '🚨 SELL NOW': 0
    }
    
    # Map actions, handling ⚠️ variants
    def get_action_rank(action):
        if pd.isna(action):
            return 4
        action_str = str(action)
        # Try exact match first
        if action_str in ACTION_RANK:
            return ACTION_RANK[action_str]
        # Try without ⚠️
        action_clean = action_str.replace(' ⚠️', '').replace('⚠️', '')
        if action_clean in ACTION_RANK:
            return ACTION_RANK[action_clean]
        # Default
        return 4
    
    df['Action_Rank'] = df['Action'].apply(get_action_rank)
    
    TIER_RANK = {'L3': 3, 'L2': 2, 'L1': 1, 'L0': 0}
    df['Tier_Rank'] = df['Liq_tier_code'].map(TIER_RANK).fillna(0)
    return df

def sort_dataframe(df, sort_mode):
    """Sort dataframe with explicit action priority, then Alpha_Score descending"""
    ACTION_RANK = {
        '🟢 STRONG BUY': 7, '🟢 BUY': 6, '🟢 BUY ⚠️': 6,
        '🔵 ADD': 5, '🔵 ADD ⚠️': 5, '🔵 ACCUMULATE': 5,
        '⚪ HOLD': 4, '🟡 TRIM': 3, 
        '🔴 REDUCE': 2, '🔴 SELL': 1, '🚨 SELL NOW': 0
    }
    
    if 'Action_Rank' not in df.columns:
        df['Action_Rank'] = df['Action'].map(ACTION_RANK).fillna(4)
    
    if sort_mode == "Sell risk first":
        return df.sort_values(['Sell_Risk_Score', 'Action_Rank'], ascending=[False, False])
    elif sort_mode == "Alpha first":
        return df.sort_values(['Alpha_Score', 'Action_Rank'], ascending=[False, False])
    else:
        # Default: Action priority (STRONG BUY > BUY > ADD > HOLD > TRIM > REDUCE > SELL > SELL NOW), then Alpha descending
        return df.sort_values(['Action_Rank', 'Alpha_Score'], ascending=[False, False])

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
    # Guardrails: safely access session_state with defaults
    df = st.session_state.get('results', pd.DataFrame())
    news_cache = st.session_state.get('news_cache', {})
    macro = st.session_state.get('macro_regime', {})
    
    # Guardrails: safely access values with defaults
    total_mv = df['Market_Value'].sum() if not df.empty and 'Market_Value' in df.columns else 0
    total_value = total_mv + st.session_state.get('cash', 0)
    
    # Daily summary
    render_daily_summary(df, macro, st.session_state.get('cash', 0))
    
    conf_breakdown_storage = st.session_state.get('conf_breakdown_storage', {})
    dilution_factors_storage = st.session_state.get('dilution_factors_storage', {})
    alpha_breakdown_storage = st.session_state.get('alpha_breakdown_storage', {})
    alpha_models_storage = st.session_state.get('alpha_models_storage', {})
    sell_triggers_storage = st.session_state.get('sell_triggers_storage', {})
    financing_status_storage = st.session_state.get('financing_status_storage', {})
    
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
    
    # Guardrails: check if df is empty or missing required columns
    if df.empty or 'Action' not in df.columns:
        st.info("No results available. Click 'RUN WORLD-CLASS ANALYSIS' to start.")
    else:
        # Add ranking and sort
        df = add_ranking_columns(df)
        sort_mode = st.session_state.get('sort_mode', 'Action first (default)')
        df_sorted = sort_dataframe(df, sort_mode)
        
        for _, row in df_sorted.iterrows():
            # Card style with liquidity warning for BUY/ADD actions
            action_str = str(row['Action'])
            action_display = action_str
            
            # Add liquidity warning ⚠️ for BUY/ADD with low liquidity
            if any(x in action_str for x in ['BUY', 'ADD']) and action_str not in ['🔴 REDUCE', '🔴 SELL', '🚨 SELL NOW']:
                liq_tier = row.get('Liq_tier_code', 'L0')
                days_to_exit = row.get('Liq_days_to_exit', 99)
                if liq_tier in ['L0', 'L1'] or days_to_exit > 5:
                    if '⚠️' not in action_display:
                        action_display = action_display + ' ⚠️'
            
            # Display with appropriate styling
            if 'BUY' in action_str:
                st.success(f"### {row['Symbol']} - {action_display}")
            elif 'SELL' in action_str or 'REDUCE' in action_str:
                st.error(f"### {row['Symbol']} - {action_display}")
            else:
                st.info(f"### {row['Symbol']} - {action_display}")
            
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
            
            if row.get('Insider_Buying_90d', False):
                badge_html += '<span class="badge-insider">INSIDER BUY</span> '
            
            if row.get('Discovery_Exception', False):
                badge_html += '<span class="badge-discovery">DISCOVERY ⚠️</span> '
            
            # Financing status badge
            financing_status = financing_status_storage.get(row['Symbol'], 'NONE')
            if financing_status == 'PP_CLOSED':
                badge_html += '<span class="badge-l3">💰 PP CLOSED</span> '
            elif financing_status == 'ATM':
                badge_html += '<span class="badge-l1">⚠️ ATM</span> '
            elif financing_status == 'SHELF':
                badge_html += '<span class="badge-l1">⚠️ SHELF</span> '
            elif financing_status == 'ANNOUNCED':
                badge_html += '<span class="badge-l2">⚠️ FINANCING</span> '
            
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
                
                # Alpha breakdown with Model Transparency
                st.markdown("---")
                st.subheader("🎯 8-Model Alpha Breakdown (Transparent)")
                
                alpha_models = alpha_models_storage.get(row['Symbol'], {})
                if alpha_models and isinstance(list(alpha_models.values())[0] if alpha_models else None, dict):
                    # Display models using tabs (no nested expanders - Streamlit-safe)
                    sorted_models = sorted(alpha_models.items())
                    if len(sorted_models) > 0:
                        # Create tabs for each model with concise labels
                        tab_labels = []
                        for model_id, model_info in sorted_models:
                            if isinstance(model_info, dict):
                                name = model_info.get('name', model_id)
                                weight = model_info.get('weight_percent', 0)
                                tab_labels.append(f"{name} ({weight:.0f}%)")
                        
                        if len(tab_labels) > 0:
                            tabs = st.tabs(tab_labels)
                            
                            for idx, (model_id, model_info) in enumerate(sorted_models):
                                if isinstance(model_info, dict) and idx < len(tabs):
                                    with tabs[idx]:
                                        # Model info with popover for additional details
                                        col1, col2 = st.columns([1, 20])
                                        with col1:
                                            with st.popover("ℹ️"):
                                                st.write("**Model Details**")
                                                st.write(f"**Model ID:** {model_id}")
                                                st.write(f"**Name:** {model_info.get('name', 'Unknown')}")
                                                st.write(f"**Inventor/Source:** {model_info.get('inventor', 'Unknown')}")
                                                st.write(f"**Weight:** {model_info.get('weight_percent', 0):.0f}%")
                                        
                                        with col2:
                                            st.write(f"**{model_info.get('name', model_id)}**")
                                        
                                        # Key metrics
                                        col_a, col_b, col_c = st.columns(3)
                                        with col_a:
                                            st.metric("Raw Score", f"{model_info.get('raw_score_0_100', 0):.0f}/100")
                                        with col_b:
                                            st.metric("Weight", f"{model_info.get('weight_percent', 0):.0f}%")
                                        with col_c:
                                            st.metric("Contribution", f"{model_info.get('contribution_points', 0):.1f} pts")
                                        
                                        st.markdown("---")
                                        
                                        # Detailed breakdown
                                        st.write(f"**Model ID:** `{model_id}`")
                                        st.write(f"**Inventor/Source:** {model_info.get('inventor', 'Unknown')}")
                                        st.write(f"**Explanation:** {model_info.get('explanation', 'N/A')}")
                    
                    # Total verification
                    calculated_total = sum(m.get('contribution_points', 0) for m in alpha_models.values())
                    st.info(f"**Total Alpha Score:** {calculated_total:.1f} points | **Verification:** {row.get('Alpha_Score', 0):.1f} (diff: {abs(calculated_total - row.get('Alpha_Score', 0)):.1f})")
                
                # Fallback to breakdown list
                alpha_breakdown = alpha_breakdown_storage.get(row['Symbol'], [])
                if alpha_breakdown:
                    st.caption("**Breakdown:**")
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
                    # Separate ticker news from sector fallback
                    ticker_only = [n for n in ticker_news if n.get('source', 'ticker') != 'sector_fallback']
                    sector_fallback = [n for n in ticker_news if n.get('source') == 'sector_fallback']
                    
                    if ticker_only:
                        for item in ticker_only[:10]:
                            tags = item.get('tag_string', '')
                            date_str = item.get('date_str', 'Unknown')
                            timestamp_valid = item.get('timestamp_valid', False)
                            valid_badge = "✅" if timestamp_valid else "⚠️"
                            st.markdown(f"{valid_badge} **{item['title']}** {tags}")
                            st.caption(f"{item['publisher']} • {date_str}")
                            st.markdown("")
                    
                    if sector_fallback:
                        st.markdown("---")
                        st.caption("**Sector Fallback News** (when ticker news unavailable)")
                        for item in sector_fallback[:5]:
                            tags = item.get('tag_string', '')
                            date_str = item.get('date_str', 'Unknown')
                            st.markdown(f"📊 **{item['title']}** {tags}")
                            st.caption(f"{item.get('publisher', 'Sector')} • {date_str}")
                            st.markdown("")
                else:
                    st.info("No news available for this ticker")
        
        # Export
        st.download_button(
            "📥 Download Complete Analysis",
            df.to_csv(index=False),
            f"alpha_miner_analysis_{datetime.date.today()}.csv",
            use_container_width=True
        )

st.caption(f"💎 Alpha Miner Pro {VERSION} • Survival > Alpha • Sell-In-Time")
