#!/usr/bin/env python3
"""
ALPHA MINER PRO - WORLD-CLASS CAPITAL ALLOCATION ENGINE
Survival > Alpha | Sell-In-Time Focus | Gate-Based Risk Management
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import json
import re
from pathlib import Path

try:
    import yfinance as yf
    YFINANCE = True
except:
    YFINANCE = False
    st.error("⚠️ yfinance not installed. Run: pip install yfinance")

st.set_page_config(page_title="Alpha Miner Pro", layout="wide", initial_sidebar_state="expanded")

# Professional styling with enhanced badges
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
# A) LIQUIDITY ENGINE (20-DAY VOLUME, EXPLICIT TIERS)
# ============================================================================

def calculate_liquidity_metrics(ticker, hist_data, current_price, current_position_value, portfolio_size=PORTFOLIO_SIZE):
    """
    WHY: Liquidity is a HARD constraint. We use 20-day volume (not 60d mean) 
    because it reflects recent trading conditions, critical for exits.
    
    RETURNS: Dict with tier, caps, flags, days_to_exit
    """
    result = {
        'tier_code': 'L0',
        'tier_name': 'Illiquid',
        'dollar_vol_20d': 0,
        'avg_vol_20d': 0,
        'max_position_pct': 1.0,
        'days_to_exit': 999,
        'no_add': True,
        'trim_only': False,
        'discovery_eligible': False,
        'liquidation_risk': 'CRITICAL'
    }
    
    if hist_data.empty or current_price <= 0:
        return result
    
    # Calculate 20-day average volume (WHY: recent liquidity matters more)
    if len(hist_data) >= 20:
        vol_20d = hist_data['Volume'].tail(20).mean()
    else:
        vol_20d = hist_data['Volume'].mean()
    
    result['avg_vol_20d'] = vol_20d
    
    # Dollar volume = Price × Volume
    dollar_vol = current_price * vol_20d
    result['dollar_vol_20d'] = dollar_vol
    
    # Days to exit: Assume we can sell 10% of daily volume per day
    # WHY: Being >10% of daily volume moves the market against us
    if dollar_vol > 0:
        exit_capacity_per_day = dollar_vol * 0.10
        result['days_to_exit'] = current_position_value / exit_capacity_per_day if exit_capacity_per_day > 0 else 999
    
    # TIER CLASSIFICATION (explicit caps and flags)
    if dollar_vol < 50000:
        # L0: ILLIQUID - Avoid unless special situation
        result.update({
            'tier_code': 'L0',
            'tier_name': 'Illiquid',
            'max_position_pct': 1.0,
            'no_add': True,
            'trim_only': False,
            'discovery_eligible': False,
            'liquidation_risk': 'CRITICAL'
        })
    
    elif dollar_vol < 200000:
        # L1: POOR - Add only if days_to_exit <= 5
        result.update({
            'tier_code': 'L1',
            'tier_name': 'Poor',
            'max_position_pct': 2.5,
            'no_add': result['days_to_exit'] > 5,  # Flag set based on days_to_exit
            'trim_only': result['days_to_exit'] > 10,
            'discovery_eligible': True,
            'liquidation_risk': 'HIGH' if result['days_to_exit'] > 5 else 'MODERATE'
        })
    
    elif dollar_vol < 1000000:
        # L2: GOOD - Standard tactical positions
        result.update({
            'tier_code': 'L2',
            'tier_name': 'Good',
            'max_position_pct': 5.0,
            'no_add': result['days_to_exit'] > 5,
            'trim_only': result['days_to_exit'] > 10,
            'discovery_eligible': True,
            'liquidation_risk': 'MODERATE' if result['days_to_exit'] > 5 else 'LOW'
        })
    
    else:
        # L3: EXCELLENT - Can support larger positions
        result.update({
            'tier_code': 'L3',
            'tier_name': 'Excellent',
            'max_position_pct': 7.5,  # Base cap, can go to 12% if CORE conditions met
            'no_add': result['days_to_exit'] > 5,
            'trim_only': result['days_to_exit'] > 10,
            'discovery_eligible': True,
            'liquidation_risk': 'LOW'
        })
    
    return result

def check_discovery_exception(row, liq_metrics, alpha_score, data_confidence, dilution_risk, momentum_confirmed):
    """
    WHY: Some high-conviction discoveries may justify holding despite poor liquidity,
    but ONLY under strict conditions. This prevents us from being trapped in illiquid disasters.
    
    EXCEPTION CRITERIA (ALL must be true):
    - Sleeve is TACTICAL (not gambling)
    - Alpha >= 85 (very high conviction)
    - Data_Confidence >= 70 (not flying blind)
    - Dilution_Risk < 70 (not about to get crushed)
    - Momentum confirmed (not catching falling knife)
    - NOT tier L0 (even exceptions have limits)
    """
    if liq_metrics['tier_code'] == 'L0':
        return False, "L0 tier excludes all exceptions"
    
    if row.get('Sleeve', '') != 'TACTICAL':
        return False, "Not TACTICAL sleeve"
    
    if alpha_score < 85:
        return False, f"Alpha {alpha_score:.0f} < 85"
    
    if data_confidence < 70:
        return False, f"Confidence {data_confidence:.0f} < 70"
    
    if dilution_risk >= 70:
        return False, f"Dilution risk {dilution_risk:.0f} too high"
    
    if not momentum_confirmed:
        return False, "Momentum not confirmed"
    
    # Exception granted! But cap at 2.5%
    return True, "High-conviction discovery"

# ============================================================================
# B) DATA CONFIDENCE (SOURCE-BASED SCORING)
# ============================================================================

def calculate_data_confidence(fundamentals_dict, info_dict, inferred_flags):
    """
    WHY: Missing data = uncertainty. We score based on WHERE data came from,
    not just whether fields exist. This prevents false confidence from inferred/estimated data.
    
    SCORING:
    - Cash from verified field (totalCash): +35
    - Burn from operating cash flow: +35
    - Burn from net income (fallback): +15
    - Stage from real revenue/assets: +10
    - Country known (not Unknown): +5
    - Metal not inferred: +5
    
    TOTAL: 0-100
    """
    score = 0
    breakdown = []
    
    # Cash source
    if info_dict.get('totalCash') and info_dict['totalCash'] > 0:
        score += 35
        breakdown.append("+35 Cash verified (totalCash field)")
    elif info_dict.get('cash') and info_dict['cash'] > 0:
        score += 25
        breakdown.append("+25 Cash from alternate field")
    else:
        score += 0
        breakdown.append("+0 Cash estimated/missing")
    
    # Burn source
    burn_source = fundamentals_dict.get('burn_source', 'default')
    if burn_source == 'cashflow':
        score += 35
        breakdown.append("+35 Burn from operating cash flow")
    elif burn_source == 'netincome':
        score += 15
        breakdown.append("+15 Burn from net income (proxy)")
    else:
        score += 0
        breakdown.append("+0 Burn estimated")
    
    # Stage verification
    if info_dict.get('totalRevenue') is not None:
        score += 10
        breakdown.append("+10 Stage from revenue data")
    else:
        score += 0
        breakdown.append("+0 Stage inferred")
    
    # Country known
    country = fundamentals_dict.get('country', 'Unknown')
    if country != 'Unknown':
        score += 5
        breakdown.append("+5 Country known")
    else:
        breakdown.append("+0 Country unknown")
    
    # Metal not inferred
    if not inferred_flags.get('metal_inferred', True):
        score += 5
        breakdown.append("+5 Metal explicit")
    else:
        breakdown.append("+0 Metal inferred")
    
    # Final verdict
    if score >= 80:
        verdict = "HIGH CONFIDENCE"
        action_guidance = "Full trading allowed"
    elif score >= 60:
        verdict = "MEDIUM CONFIDENCE"
        action_guidance = "Normal trading"
    elif score >= 40:
        verdict = "LOW CONFIDENCE"
        action_guidance = "OBSERVE/NO ADD unless holding"
    else:
        verdict = "VERY LOW"
        action_guidance = "GAMBLING ONLY (1% cap)"
    
    return {
        'score': score,
        'breakdown': breakdown,
        'verdict': verdict,
        'action_guidance': action_guidance
    }

# ============================================================================
# C) DILUTION RISK MODEL + INSIDER SIGNAL
# ============================================================================

def calculate_dilution_risk(runway_months, stage, price_drawdown_90d, news_items, 
                            cash_missing, burn_missing, insider_buying):
    """
    WHY: Dilution is the #1 wealth destroyer in junior miners. We score risk 0-100
    where high score = high dilution risk.
    
    INPUTS:
    - Runway: <6mo = crisis, 6-9 = elevated, 9-12 = watch
    - Stage: Explorers dilute more than producers
    - Price action: Drawdown may force bad financing
    - News: Recent financing announcements
    - Missing data: Uncertainty penalty
    - Insider buying: Reduces risk (insiders know)
    
    RETURNS: 0-100 score where >70 is dangerous
    """
    risk_score = 0
    factors = []
    
    # Runway bands (WHY: cash runway predicts dilution timing)
    if runway_months < 6:
        risk_score += 50
        factors.append("+50 Runway <6mo CRITICAL")
    elif runway_months < 9:
        risk_score += 30
        factors.append("+30 Runway 6-9mo ELEVATED")
    elif runway_months < 12:
        risk_score += 15
        factors.append("+15 Runway 9-12mo WATCH")
    else:
        factors.append("+0 Runway adequate")
    
    # Stage risk (WHY: explorers burn cash, dilute frequently)
    if stage == 'Explorer':
        risk_score += 10
        factors.append("+10 Explorer stage")
    elif stage == 'Developer':
        risk_score += 5
        factors.append("+5 Developer stage")
    
    # Price weakness (WHY: weak price = bad financing terms)
    if price_drawdown_90d > 30:
        risk_score += 10
        factors.append(f"+10 Drawdown {price_drawdown_90d:.0f}%")
    
    # News analysis for financing keywords
    financing_detected = False
    atm_shelf_detected = False
    
    for news in news_items:
        title_lower = news.get('title', '').lower()
        
        # ATM/shelf registration (especially bad)
        if any(kw in title_lower for kw in ['atm', 'shelf', 'prospectus']):
            atm_shelf_detected = True
        
        # General financing
        elif any(kw in title_lower for kw in ['private placement', 'financing', 'bought deal', 
                                                'equity offering', 'raises', 'closes']):
            financing_detected = True
    
    if atm_shelf_detected:
        risk_score += 25
        factors.append("+25 ATM/shelf registration")
    elif financing_detected:
        risk_score += 15
        factors.append("+15 Recent financing")
    
    # Missing data penalty (WHY: uncertainty = risk)
    if cash_missing or burn_missing:
        risk_score += 10
        factors.append("+10 Data uncertainty")
    
    # INSIDER SIGNAL (WHY: insiders know the company's plans)
    # Reduces risk by 10-15 points if buying recently
    if insider_buying:
        reduction = min(15, risk_score * 0.15)  # Up to 15 points or 15% of score
        risk_score = max(0, risk_score - reduction)
        factors.append(f"-{reduction:.0f} Insider buying (bullish signal)")
    
    # Cap at 100
    risk_score = min(100, risk_score)
    
    # Risk verdict
    if risk_score >= 85:
        verdict = "EXTREME - Avoid/Sell rallies"
    elif risk_score >= 70:
        verdict = "HIGH - Cap 1-2% max"
    elif risk_score >= 50:
        verdict = "ELEVATED - Reduce on strength"
    elif risk_score >= 30:
        verdict = "MODERATE - Monitor"
    else:
        verdict = "LOW - Normal"
    
    return {
        'score': risk_score,
        'factors': factors,
        'verdict': verdict,
        'allow_adds': risk_score < 70,
        'force_trim': risk_score >= 85
    }

# ============================================================================
# D) CORE 6 ALPHA MODELS
# ============================================================================

def calculate_alpha_models(row, hist_data, benchmark_data):
    """
    WHY: Multiple independent models prevent single-point-of-failure bias.
    We aggregate with weights but survival gates override.
    
    SIX CORE MODELS:
    1. Survival/Dilution (30%)
    2. Momentum 7/30/90 (25%)
    3. Drawdown/52w positioning (15%)
    4. Volatility penalty (10%)
    5. Liquidity quality (10%)
    6. Relative strength vs benchmark (10%)
    """
    models = {}
    
    # MODEL 1: Survival/Dilution (inverse of dilution risk)
    # WHY: Most important - can't make money if diluted to death
    dilution_risk = row.get('Dilution_Risk_Score', 50)
    models['M1_Survival'] = {
        'score': max(0, 100 - dilution_risk),
        'weight': 0.30,
        'description': f"Survival (inv dilution): {100-dilution_risk:.0f}/100"
    }
    
    # MODEL 2: Momentum 7/30/90
    # WHY: Trend is friend until it ends
    ret_7d = row.get('Return_7d', 0)
    ret_30d = row.get('Return_30d', 0)
    ret_90d = row.get('Return_90d', 0)
    
    # Weighted momentum: 7d×5 + 30d×3 + 90d×1
    momentum_score = (ret_7d * 5 + ret_30d * 3 + ret_90d * 1) / 9
    momentum_score = np.clip(momentum_score * 2 + 50, 0, 100)
    
    models['M2_Momentum'] = {
        'score': momentum_score,
        'weight': 0.25,
        'description': f"Momentum: 7d={ret_7d:+.1f}% 30d={ret_30d:+.1f}%"
    }
    
    # MODEL 3: 52-week positioning
    # WHY: Buy low, sell high - position in range matters
    pct_from_high = row.get('Pct_From_52w_High', -50)
    pct_from_low = row.get('Pct_From_52w_Low', 50)
    
    # Prefer stocks off lows but not at highs
    position_score = 50 + (pct_from_low * 0.3) + (pct_from_high * 0.2)
    position_score = np.clip(position_score, 0, 100)
    
    models['M3_Position'] = {
        'score': position_score,
        'weight': 0.15,
        'description': f"52w position: {pct_from_low:.0f}% from low"
    }
    
    # MODEL 4: Volatility penalty
    # WHY: High vol = high risk, penalize
    volatility = row.get('Volatility_60d', 5)
    vol_score = np.clip(100 - (volatility * 10), 0, 100)
    
    models['M4_Volatility'] = {
        'score': vol_score,
        'weight': 0.10,
        'description': f"Volatility: {volatility:.1f}%"
    }
    
    # MODEL 5: Liquidity quality
    # WHY: Liquidity is both a gate AND a score component
    liq_tier = row.get('Liq_Tier_Code', 'L0')
    liq_scores = {'L3': 90, 'L2': 70, 'L1': 40, 'L0': 10}
    liq_score = liq_scores.get(liq_tier, 50)
    
    models['M5_Liquidity'] = {
        'score': liq_score,
        'weight': 0.10,
        'description': f"Liquidity: {liq_tier}"
    }
    
    # MODEL 6: Relative Strength vs benchmark
    # WHY: Outperforming peers = strength
    rs_score = 50  # Default if no benchmark data
    
    if benchmark_data is not None and not benchmark_data.empty:
        try:
            # Compare 30-day returns
            stock_ret_30 = ret_30d
            bench_ret_30 = ((benchmark_data['Close'].iloc[-1] - benchmark_data['Close'].iloc[-30]) / 
                           benchmark_data['Close'].iloc[-30] * 100)
            
            rs_diff = stock_ret_30 - bench_ret_30
            rs_score = np.clip(50 + rs_diff * 2, 0, 100)
        except:
            pass
    
    models['M6_RelStrength'] = {
        'score': rs_score,
        'weight': 0.10,
        'description': f"RS vs benchmark: {rs_score:.0f}/100"
    }
    
    # AGGREGATE
    total_score = sum(m['score'] * m['weight'] for m in models.values())
    
    return {
        'models': models,
        'alpha_score': total_score,
        'breakdown': [f"{k}: {v['score']:.0f} (wt {v['weight']:.0%})" for k, v in models.items()]
    }

# ============================================================================
# E) SELL-IN-TIME ENGINE
# ============================================================================

def calculate_sell_risk(row, hist_data, ma50, ma200, news_items, macro_regime):
    """
    WHY: The hardest part of mining stocks is selling in time. This engine
    identifies sell triggers BEFORE catastrophic losses.
    
    HARD TRIGGERS (immediate action):
    - MA200 break + weak macro
    - Financing news + weak price action
    - Runway < 6mo
    - Gap down + volume spike + negative news
    
    SOFT TRIGGERS (trim/reduce):
    - MA50 break
    - Momentum flip (7d<0 and 30d<0)
    - New 20d low with volume expansion
    - Persistent RS underperformance
    
    RETURNS: Sell_Risk_Score (0-100) and list of triggered conditions
    """
    sell_score = 0
    triggers = []
    hard_triggers = []
    soft_triggers = []
    
    current_price = row.get('Price', 0)
    runway = row.get('Runway', 12)
    ret_7d = row.get('Return_7d', 0)
    ret_30d = row.get('Return_30d', 0)
    
    # HARD TRIGGER 1: Runway < 6 months
    if runway < 6:
        sell_score += 50
        hard_triggers.append(f"💀 Runway {runway:.1f}mo < 6mo")
    
    # HARD TRIGGER 2: MA200 break in weak macro
    if ma200 > 0 and current_price < ma200:
        if macro_regime.get('regime') == 'DEFENSIVE':
            sell_score += 30
            hard_triggers.append("💀 Below MA200 + Defensive macro")
        else:
            sell_score += 15
            soft_triggers.append("⚠️ Below MA200")
    
    # HARD TRIGGER 3: Financing news + weak price
    financing_news = False
    for news in news_items:
        title_lower = news.get('title', '').lower()
        if any(kw in title_lower for kw in ['financing', 'private placement', 'bought deal', 'atm']):
            financing_news = True
            break
    
    if financing_news and ret_7d < -5:
        sell_score += 25
        hard_triggers.append("💀 Financing + Price weakness")
    
    # HARD TRIGGER 4: Large gap down (if detectable from history)
    if len(hist_data) >= 2:
        try:
            last_close = hist_data['Close'].iloc[-2]
            today_open = hist_data['Open'].iloc[-1]
            gap_pct = ((today_open - last_close) / last_close * 100)
            
            if gap_pct < -10:
                sell_score += 20
                hard_triggers.append(f"💀 Gap down {gap_pct:.1f}%")
        except:
            pass
    
    # SOFT TRIGGER 1: MA50 break
    if ma50 > 0 and current_price < ma50:
        sell_score += 10
        soft_triggers.append("⚠️ Below MA50")
    
    # SOFT TRIGGER 2: Momentum flip (both 7d and 30d negative)
    if ret_7d < 0 and ret_30d < 0:
        sell_score += 10
        soft_triggers.append("⚠️ Momentum flip (7d & 30d negative)")
    
    # SOFT TRIGGER 3: New 20d low with volume expansion
    if len(hist_data) >= 20:
        try:
            low_20d = hist_data['Low'].tail(20).min()
            vol_20d_avg = hist_data['Volume'].tail(20).mean()
            recent_vol = hist_data['Volume'].tail(5).mean()
            
            if current_price <= low_20d * 1.02 and recent_vol > vol_20d_avg * 1.5:
                sell_score += 15
                soft_triggers.append("⚠️ New 20d low + volume spike")
        except:
            pass
    
    # SOFT TRIGGER 4: Persistent underperformance
    # (This would use benchmark comparison - simplified here)
    if ret_30d < -15 and ret_7d < -5:
        sell_score += 10
        soft_triggers.append("⚠️ Persistent weakness")
    
    # Verdict
    if sell_score >= 50:
        verdict = "SELL NOW"
        action = "🚨 SELL"
    elif sell_score >= 30:
        verdict = "REDUCE"
        action = "🔴 TRIM"
    elif sell_score >= 15:
        verdict = "WATCH CLOSELY"
        action = "🟡 MONITOR"
    else:
        verdict = "NORMAL"
        action = "✅ OK"
    
    return {
        'score': sell_score,
        'verdict': verdict,
        'action': action,
        'hard_triggers': hard_triggers,
        'soft_triggers': soft_triggers,
        'all_triggers': hard_triggers + soft_triggers
    }

# ============================================================================
# F) MACRO REGIME (TREND-BASED)
# ============================================================================

@st.cache_data(ttl=300)
def calculate_macro_regime():
    """
    WHY: Don't fight the macro. Use TRENDS not single-day readings.
    
    CHECKS:
    - DXY vs MA20/MA50 and slope
    - VIX vs MA20 and trend
    - Gold/Silver vs MA20
    
    OUTPUTS:
    - Regime: RISK-ON / NEUTRAL / DEFENSIVE
    - Throttle factor for position sizing
    - Allow new buys flag
    """
    regime_data = {
        'regime': 'NEUTRAL',
        'throttle_factor': 1.0,
        'allow_new_buys': True,
        'factors': [],
        'dxy': None,
        'vix': None,
        'gold_trend': None,
        'silver_trend': None
    }
    
    if not YFINANCE:
        return regime_data
    
    try:
        # DXY (Dollar Index) - 3 months of data
        dxy_hist = yf.Ticker("DX-Y.NYB").history(period="3mo")
        if not dxy_hist.empty and len(dxy_hist) >= 50:
            dxy_current = dxy_hist['Close'].iloc[-1]
            dxy_ma20 = dxy_hist['Close'].tail(20).mean()
            dxy_ma50 = dxy_hist['Close'].tail(50).mean()
            
            regime_data['dxy'] = dxy_current
            
            # Check trend
            if dxy_current > dxy_ma50 and dxy_ma20 > dxy_ma50:
                regime_data['factors'].append("⚠️ DXY rising (bearish for metals)")
            elif dxy_current < dxy_ma50 and dxy_ma20 < dxy_ma50:
                regime_data['factors'].append("✅ DXY falling (bullish for metals)")
    except:
        pass
    
    try:
        # VIX (Volatility)
        vix_hist = yf.Ticker("^VIX").history(period="3mo")
        if not vix_hist.empty and len(vix_hist) >= 20:
            vix_current = vix_hist['Close'].iloc[-1]
            vix_ma20 = vix_hist['Close'].tail(20).mean()
            
            regime_data['vix'] = vix_current
            
            if vix_current > 25:
                regime_data['factors'].append("⚠️ VIX elevated (fear)")
            elif vix_current < 15:
                regime_data['factors'].append("✅ VIX low (complacent)")
    except:
        pass
    
    try:
        # Gold futures
        gold_hist = yf.Ticker("GC=F").history(period="3mo")
        if not gold_hist.empty and len(gold_hist) >= 20:
            gold_current = gold_hist['Close'].iloc[-1]
            gold_ma20 = gold_hist['Close'].tail(20).mean()
            
            regime_data['gold_trend'] = "UP" if gold_current > gold_ma20 else "DOWN"
            
            if gold_current > gold_ma20:
                regime_data['factors'].append("✅ Gold above MA20")
            else:
                regime_data['factors'].append("⚠️ Gold below MA20")
    except:
        pass
    
    try:
        # Silver futures
        silver_hist = yf.Ticker("SI=F").history(period="3mo")
        if not silver_hist.empty and len(silver_hist) >= 20:
            silver_current = silver_hist['Close'].iloc[-1]
            silver_ma20 = silver_hist['Close'].tail(20).mean()
            
            regime_data['silver_trend'] = "UP" if silver_current > silver_ma20 else "DOWN"
            
            if silver_current > silver_ma20:
                regime_data['factors'].append("✅ Silver above MA20")
            else:
                regime_data['factors'].append("⚠️ Silver below MA20")
    except:
        pass
    
    # DETERMINE REGIME
    bearish_count = sum(1 for f in regime_data['factors'] if f.startswith("⚠️"))
    bullish_count = sum(1 for f in regime_data['factors'] if f.startswith("✅"))
    
    # Defensive if DXY rising OR VIX > 25
    dxy_rising = regime_data.get('dxy', 0) > 105
    vix_high = regime_data.get('vix', 0) > 25
    
    if (dxy_rising or vix_high) and bearish_count >= 2:
        regime_data['regime'] = 'DEFENSIVE'
        regime_data['throttle_factor'] = 0.5
        regime_data['allow_new_buys'] = False
    elif bullish_count >= 3:
        regime_data['regime'] = 'RISK-ON'
        regime_data['throttle_factor'] = 1.0
        regime_data['allow_new_buys'] = True
    else:
        regime_data['regime'] = 'NEUTRAL'
        regime_data['throttle_factor'] = 0.85
        regime_data['allow_new_buys'] = True
    
    return regime_data

# ============================================================================
# G) NEWS TAGGING
# ============================================================================

def tag_news(news_items):
    """
    WHY: Certain news types are critical signals for junior miners.
    We tag them for quick identification.
    
    TAGS:
    - Financing
    - Drill results
    - Studies (PEA/PFS/FS)
    - Permits
    - M&A
    - Production
    - Corporate (halts, management)
    """
    tagged_news = []
    
    for item in news_items:
        title = item.get('title', '')
        title_lower = title.lower()
        
        tags = []
        
        # Financing (critical for dilution)
        if any(kw in title_lower for kw in ['private placement', 'financing', 'bought deal', 
                                              'equity offering', 'raises', 'closes', 'atm', 
                                              'shelf', 'prospectus']):
            tags.append('💰 FINANCING')
        
        # Drill results (can be big catalyst)
        if any(kw in title_lower for kw in ['drill', 'assay', 'intercept', 'intersect', 'hit']):
            tags.append('🎯 DRILL')
        
        # Studies
        if any(kw in title_lower for kw in ['mre', 'resource estimate', 'pea', 'preliminary economic',
                                              'pfs', 'pre-feasibility', 'feasibility', 'fs']):
            tags.append('📊 STUDY')
        
        # Permits
        if any(kw in title_lower for kw in ['permit', 'approval', 'environment', 'community', 
                                              'indigenous', 'license']):
            tags.append('📋 PERMIT')
        
        # M&A
        if any(kw in title_lower for kw in ['acquisition', 'merger', 'strategic investment', 
                                              'takeover', 'offer', 'agreement']):
            tags.append('🤝 M&A')
        
        # Production
        if any(kw in title_lower for kw in ['production', 'output', 'guidance', 'aisc', 
                                              'all-in sustaining', 'quarterly results']):
            tags.append('⚙️ PRODUCTION')
        
        # Corporate concerns
        if any(kw in title_lower for kw in ['halt', 'investigation', 'lawsuit', 'resign', 
                                              'replaces', 'ceo', 'cfo', 'departs']):
            tags.append('⚠️ CORPORATE')
        
        tagged_news.append({
            **item,
            'tags': tags,
            'tag_string': ' '.join(tags) if tags else ''
        })
    
    return tagged_news

# ============================================================================
# H) MASTER ARBITRATION ENGINE
# ============================================================================

def arbitrate_final_decision(row, liq_metrics, data_conf, dilution_risk, sell_risk, 
                             alpha_score, macro_regime, discovery_exception):
    """
    WHY: Multiple signals must be arbitrated with clear rules.
    Survival gates override everything.
    
    GATE PRIORITY:
    1. SURVIVAL (runway < 6mo = SELL)
    2. SELL RISK (hard triggers = SELL/REDUCE)
    3. LIQUIDITY (days_to_exit > 10 = TRIM ONLY)
    4. DATA CONFIDENCE (< 40 = GAMBLING cap)
    5. DILUTION RISK (>= 70 = cap at 2%)
    6. ALPHA SCORE (within all constraints)
    """
    decision = {
        'action': '⚪ HOLD',
        'confidence': 50,
        'recommended_pct': row.get('Pct_Portfolio', 0),
        'max_allowed_pct': 12.0,
        'reasoning': [],
        'gates_passed': [],
        'gates_failed': [],
        'warnings': []
    }
    
    # Current position
    current_pct = row.get('Pct_Portfolio', 0)
    
    # Base max from sleeve
    sleeve = row.get('Sleeve', 'TACTICAL')
    if sleeve == 'CORE':
        base_max = 12.0
    elif sleeve == 'TACTICAL':
        base_max = 5.0
    else:  # GAMBLING
        base_max = 2.0
    
    # Apply liquidity tier cap
    liq_cap = liq_metrics.get('max_position_pct', 5.0)
    base_max = min(base_max, liq_cap)
    
    # Apply macro throttle
    throttle = macro_regime.get('throttle_factor', 1.0)
    base_max = base_max * throttle
    
    # GATE 1: SURVIVAL (runway check)
    runway = row.get('Runway', 12)
    if runway < 6:
        decision.update({
            'action': '🚨 SELL NOW',
            'confidence': 95,
            'recommended_pct': 0,
            'max_allowed_pct': 0,
            'reasoning': [f"GATE 1 FAIL: Runway {runway:.1f}mo < 6mo threshold"]
        })
        decision['gates_failed'].append(f"❌ Survival: {runway:.1f}mo < 6mo")
        return decision
    else:
        decision['gates_passed'].append(f"✅ Survival: {runway:.1f}mo")
    
    # GATE 2: SELL RISK (hard triggers)
    if sell_risk['score'] >= 50:
        decision.update({
            'action': '🔴 SELL',
            'confidence': 90,
            'recommended_pct': 0,
            'max_allowed_pct': 0,
            'reasoning': [f"GATE 2 FAIL: Sell triggers active"] + sell_risk['hard_triggers']
        })
        decision['gates_failed'].append(f"❌ Sell Risk: {sell_risk['score']}/100")
        return decision
    elif sell_risk['score'] >= 30:
        # Soft triggers - reduce but don't exit
        decision['action'] = '🟡 REDUCE'
        decision['recommended_pct'] = current_pct * 0.5
        decision['reasoning'].append(f"GATE 2 WARNING: Sell risk {sell_risk['score']}/100")
        decision['warnings'].extend(sell_risk['soft_triggers'])
    else:
        decision['gates_passed'].append(f"✅ Sell Risk: {sell_risk['score']}/100")
    
    # GATE 3: LIQUIDITY
    if liq_metrics.get('trim_only', False):
        decision['action'] = '🟡 TRIM ONLY'
        decision['recommended_pct'] = min(current_pct, base_max * 0.5)
        decision['reasoning'].append(f"GATE 3 FAIL: Days to exit {liq_metrics['days_to_exit']:.1f} > 10")
        decision['gates_failed'].append(f"❌ Liquidity: {liq_metrics['days_to_exit']:.1f}d to exit")
    elif liq_metrics.get('no_add', False):
        if not discovery_exception[0]:  # No exception
            decision['warnings'].append(f"⚠️ No adds: {liq_metrics['tier_name']} liquidity")
            decision['gates_passed'].append(f"⚠️ Liquidity: No adds ({liq_metrics['tier_code']})")
        else:
            decision['warnings'].append(f"🎯 Discovery exception: {discovery_exception[1]}")
            decision['gates_passed'].append(f"✅ Liquidity: Exception granted")
    else:
        decision['gates_passed'].append(f"✅ Liquidity: {liq_metrics['tier_code']}")
    
    # GATE 4: DATA CONFIDENCE
    conf_score = data_conf['score']
    if conf_score < 40:
        base_max = min(base_max, 1.0)
        decision['warnings'].append(f"⚠️ Very low confidence {conf_score}/100")
        decision['gates_failed'].append(f"❌ Confidence: {conf_score}/100 (gambling only)")
    elif conf_score < 60:
        decision['warnings'].append(f"⚠️ Low confidence {conf_score}/100 - no adds")
        decision['gates_passed'].append(f"⚠️ Confidence: {conf_score}/100")
    else:
        decision['gates_passed'].append(f"✅ Confidence: {conf_score}/100")
    
    # GATE 5: DILUTION RISK
    dil_score = dilution_risk['score']
    if dil_score >= 85:
        decision['action'] = '🔴 SELL RALLIES'
        decision['recommended_pct'] = 0
        decision['reasoning'].append(f"GATE 5 FAIL: Dilution risk {dil_score}/100")
        decision['gates_failed'].append(f"❌ Dilution: {dil_score}/100 EXTREME")
    elif dil_score >= 70:
        base_max = min(base_max, 2.0)
        decision['warnings'].append(f"⚠️ High dilution risk {dil_score}/100")
        decision['gates_failed'].append(f"❌ Dilution: {dil_score}/100 (cap 2%)")
    else:
        decision['gates_passed'].append(f"✅ Dilution: {dil_score}/100")
    
    # GATE 6: ALPHA SCORE (if all gates passed)
    if alpha_score >= 80:
        decision['action'] = '🟢 STRONG BUY'
        decision['confidence'] = 90
        decision['recommended_pct'] = base_max
    elif alpha_score >= 70:
        decision['action'] = '🟢 BUY'
        decision['confidence'] = 80
        decision['recommended_pct'] = base_max * 0.85
    elif alpha_score >= 60:
        decision['action'] = '🔵 ADD'
        decision['confidence'] = 70
        decision['recommended_pct'] = base_max * 0.65
    elif alpha_score >= 50:
        decision['action'] = '⚪ HOLD'
        decision['confidence'] = 60
        decision['recommended_pct'] = current_pct
    elif alpha_score >= 40:
        decision['action'] = '🟡 TRIM'
        decision['confidence'] = 65
        decision['recommended_pct'] = current_pct * 0.7
    else:
        decision['action'] = '🔴 REDUCE'
        decision['confidence'] = 75
        decision['recommended_pct'] = current_pct * 0.4
    
    decision['max_allowed_pct'] = base_max
    decision['reasoning'].append(f"Alpha: {alpha_score:.0f}/100 within constraints")
    
    return decision

# ============================================================================
# DATA FETCHING (Streamlined from previous version)
# ============================================================================

@st.cache_data(ttl=900)
def get_fundamentals_with_tracking(ticker):
    """Fetch fundamentals and track data sources for confidence scoring"""
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
        
        # Burn rate from cash flow (best source)
        try:
            cf = stock.cashflow
            if not cf.empty and 'Operating Cash Flow' in cf.index:
                ocf = cf.loc['Operating Cash Flow'].iloc[0]
                if ocf < 0:
                    result['burn'] = abs(ocf) / 12_000_000
                    result['burn_source'] = 'cashflow'
                elif ocf > 0:
                    result['burn'] = 0.1  # Minimal burn if cash flow positive
                    result['burn_source'] = 'cashflow'
        except:
            # Fallback to net income
            if info.get('netIncome') and info['netIncome'] < 0:
                result['burn'] = abs(info['netIncome']) / 12_000_000
                result['burn_source'] = 'netincome'
        
        # Stage (from revenue)
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
        
        # Metal from description
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
    """Fetch and tag news"""
    if not YFINANCE:
        return []
    
    try:
        stock = yf.Ticker(ticker)
        news = stock.news[:10]  # Last 10 items
        
        formatted_news = []
        for item in news:
            formatted_news.append({
                'title': item.get('title', ''),
                'publisher': item.get('publisher', ''),
                'link': item.get('link', '#'),
                'timestamp': item.get('providerPublishTime', 0)
            })
        
        return tag_news(formatted_news)
    except:
        return []

@st.cache_data(ttl=3600)
def get_benchmark_data(metal):
    """Fetch benchmark data for relative strength"""
    if not YFINANCE:
        return None
    
    try:
        # GDXJ for gold, SILJ for silver
        ticker = "SILJ" if metal == 'Silver' else "GDXJ"
        bench = yf.Ticker(ticker)
        return bench.history(period="6mo")
    except:
        return None

# ============================================================================
# SESSION STATE & UI
# ============================================================================

DEFAULT_PORTFOLIO = pd.DataFrame({
    'Symbol': ['JZRIF', 'ITRG', 'SMAGF', 'LOMLF', 'TSKFF', 'AGXPF', 'GSVRF', 
               'EXNRF', 'WRLGF', 'JAGGF', 'BORMF', 'AG', 'LUCMF', 'SMDRF'],
    'Quantity': [19841, 2072, 8335, 24557, 13027, 32342, 9049, 
                 11749, 25929, 2965, 5172, 638, 7079, 7072],
    'Cost_Basis': [5959.25, 7838.13, 9928.39, 5006.76, 13857.41, 24015.26, 2415.31,
                   3242.97, 18833.21, 14558.14, 5540.99, 7594.99, 8550.49, 6939.18],
    'Insider_Buying_90d': [False] * 14  # Add insider buying column
})

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = DEFAULT_PORTFOLIO.copy()
if 'cash' not in st.session_state:
    st.session_state.cash = 39569.65

# Sidebar
with st.sidebar:
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
    if st.button("Reset to Default"):
        st.session_state.portfolio = DEFAULT_PORTFOLIO.copy()
        st.rerun()

# Main App
st.title("💎 ALPHA MINER PRO")
st.caption("World-Class Capital Allocation Engine • Survival > Alpha • Sell-In-Time Focus")

# Get macro regime first
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

# Analysis Button
if st.button("🚀 RUN WORLD-CLASS ANALYSIS", type="primary", use_container_width=True):
    progress = st.progress(0, text="Starting comprehensive analysis...")
    
    df = st.session_state.portfolio.copy()
    
    # Fetch all data
    progress.progress(10, text="📊 Fetching market data (1-2 years)...")
    
    hist_cache = {}
    for idx, row in df.iterrows():
        if YFINANCE:
            try:
                hist = yf.Ticker(row['Symbol']).history(period="2y")
                if not hist.empty:
                    hist_cache[row['Symbol']] = hist
                    
                    # Basic metrics
                    df.at[idx, 'Price'] = hist['Close'].iloc[-1]
                    df.at[idx, 'Volume'] = hist['Volume'].mean()
                    
                    # Returns
                    df.at[idx, 'Return_7d'] = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-7]) / hist['Close'].iloc[-7] * 100) if len(hist) >= 7 else 0
                    df.at[idx, 'Return_30d'] = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-30]) / hist['Close'].iloc[-30] * 100) if len(hist) >= 30 else 0
                    df.at[idx, 'Return_90d'] = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-90]) / hist['Close'].iloc[-90] * 100) if len(hist) >= 90 else 0
                    
                    # 52-week positioning
                    high_52w = hist['High'].tail(252).max() if len(hist) >= 252 else hist['High'].max()
                    low_52w = hist['Low'].tail(252).min() if len(hist) >= 252 else hist['Low'].min()
                    df.at[idx, 'Pct_From_52w_High'] = ((hist['Close'].iloc[-1] - high_52w) / high_52w * 100)
                    df.at[idx, 'Pct_From_52w_Low'] = ((hist['Close'].iloc[-1] - low_52w) / low_52w * 100)
                    
                    # Volatility
                    df.at[idx, 'Volatility_60d'] = hist['Close'].pct_change().tail(60).std() * 100 if len(hist) >= 60 else 5
                    
                    # MAs
                    df.at[idx, 'MA50'] = hist['Close'].tail(50).mean() if len(hist) >= 50 else 0
                    df.at[idx, 'MA200'] = hist['Close'].tail(200).mean() if len(hist) >= 200 else 0
                    
                    # 90d drawdown
                    if len(hist) >= 90:
                        high_90d = hist['High'].tail(90).max()
                        df.at[idx, 'Drawdown_90d'] = ((hist['Close'].iloc[-1] - high_90d) / high_90d * 100)
                    else:
                        df.at[idx, 'Drawdown_90d'] = 0
            except:
                df.at[idx, 'Price'] = 0
    
    progress.progress(25, text="🔍 Fetching fundamentals...")
    
    # Store complex objects separately (can't store dicts in DataFrame)
    info_storage = {}
    inferred_storage = {}
    
    for idx, row in df.iterrows():
        fund = get_fundamentals_with_tracking(row['Symbol'])
        for k, v in fund.items():
            if k not in ['info_dict', 'inferred_flags']:
                df.at[idx, k] = v
        
        # Store in separate dicts
        info_storage[row['Symbol']] = fund['info_dict']
        inferred_storage[row['Symbol']] = fund['inferred_flags']
    
    progress.progress(35, text="📰 Fetching news...")
    
    news_cache = {}
    for idx, row in df.iterrows():
        news = get_news_for_ticker(row['Symbol'])
        news_cache[row['Symbol']] = news
    
    # Calculate position metrics
    df['Market_Value'] = df['Quantity'] * df['Price']
    df['Gain_Loss'] = df['Market_Value'] - df['Cost_Basis']
    df['Return_Pct'] = (df['Gain_Loss'] / df['Cost_Basis'] * 100)
    total_mv = df['Market_Value'].sum()
    df['Pct_Portfolio'] = (df['Market_Value'] / total_mv * 100)
    df['Runway'] = df['cash'] / df['burn']
    
    progress.progress(45, text="🚦 Running liquidity analysis...")
    
    # Liquidity metrics
    for idx, row in df.iterrows():
        hist = hist_cache.get(row['Symbol'], pd.DataFrame())
        liq = calculate_liquidity_metrics(row['Symbol'], hist, row['Price'], row['Market_Value'], port_size)
        
        for k, v in liq.items():
            df.at[idx, f'Liq_{k}'] = v
    
    progress.progress(55, text="📊 Calculating data confidence...")
    
    # Data confidence - store breakdown separately since it's a list
    conf_breakdown_storage = {}
    
    for idx, row in df.iterrows():
        fund_dict = {'burn_source': row['burn_source']}
        info_dict = info_storage.get(row['Symbol'], {})
        inferred = inferred_storage.get(row['Symbol'], {})
        
        conf = calculate_data_confidence(fund_dict, info_dict, inferred)
        df.at[idx, 'Data_Confidence'] = conf['score']
        df.at[idx, 'Conf_Verdict'] = conf['verdict']
        
        # Store breakdown separately
        conf_breakdown_storage[row['Symbol']] = conf['breakdown']
    
    progress.progress(65, text="💀 Calculating dilution risk...")
    
    # Dilution risk - store factors separately
    dilution_factors_storage = {}
    
    for idx, row in df.iterrows():
        news = news_cache.get(row['Symbol'], [])
        cash_missing = row['cash'] == 10.0  # Default value indicates missing
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
        
        # Store factors separately
        dilution_factors_storage[row['Symbol']] = dil['factors']
    
    progress.progress(75, text="🎯 Running 6-model alpha scoring...")
    
    # Alpha models - store complex data separately
    alpha_models_storage = {}
    alpha_breakdown_storage = {}
    
    for idx, row in df.iterrows():
        hist = hist_cache.get(row['Symbol'], pd.DataFrame())
        benchmark = get_benchmark_data(row.get('metal', 'Gold'))
        
        alpha_result = calculate_alpha_models(row, hist, benchmark)
        df.at[idx, 'Alpha_Score'] = alpha_result['alpha_score']
        
        # Store complex objects separately
        alpha_models_storage[row['Symbol']] = alpha_result['models']
        alpha_breakdown_storage[row['Symbol']] = alpha_result['breakdown']
    
    progress.progress(85, text="🔴 Running sell-in-time analysis...")
    
    # Sell risk - store triggers separately
    sell_triggers_storage = {}
    
    for idx, row in df.iterrows():
        hist = hist_cache.get(row['Symbol'], pd.DataFrame())
        news = news_cache.get(row['Symbol'], [])
        
        sell = calculate_sell_risk(row, hist, row.get('MA50', 0), row.get('MA200', 0), news, macro_regime)
        
        df.at[idx, 'Sell_Risk_Score'] = sell['score']
        df.at[idx, 'Sell_Verdict'] = sell['verdict']
        
        # Store triggers separately
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
        
        # Check momentum confirmation
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
        sell_risk = {'score': row['Sell_Risk_Score'], 'hard_triggers': [], 'soft_triggers': []}
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
    st.session_state.sell_triggers_storage = sell_triggers_storage
    
    st.success("✅ World-class analysis complete!")
    st.rerun()

# ============================================================================
# G) COMMAND CENTER DISPLAY
# ============================================================================

if 'results' in st.session_state:
    df = st.session_state.results
    news_cache = st.session_state.news_cache
    macro = st.session_state.macro_regime
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
        # Filter for addable positions
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
            st.info("No add opportunities in current market conditions")
    
    # Portfolio liquidity risk
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
    
    # Sort by sell risk desc, then alpha desc
    df_sorted = df.sort_values(['Sell_Risk_Score', 'Alpha_Score'], ascending=[False, False])
    
    for _, row in df_sorted.iterrows():
        # Determine card style
        if 'BUY' in row['Action']:
            st.success(f"### {row['Symbol']} - {row['Action']}")
        elif 'SELL' in row['Action'] or 'REDUCE' in row['Action']:
            st.error(f"### {row['Symbol']} - {row['Action']}")
        else:
            st.info(f"### {row['Symbol']} - {row['Action']}")
        
        # Metrics row
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Alpha", f"{row['Alpha_Score']:.0f}/100")
        c2.metric("Sell Risk", f"{row['Sell_Risk_Score']:.0f}/100")
        c3.metric("Current", f"{row['Pct_Portfolio']:.1f}%")
        c4.metric("→ Rec", f"{row['Recommended_Pct']:.1f}%")
        c5.metric("Max", f"{row['Max_Allowed_Pct']:.1f}%")
        c6.metric("Conf", f"{row['Confidence']:.0f}%")
        
        # Badges
        sleeve_badge = f"badge-{row['Sleeve'].lower()}"
        liq_badge = f"badge-{row['Liq_tier_code'].lower()}"
        
        badge_html = f'<span class="{sleeve_badge}">{row["Sleeve"]}</span> '
        badge_html += f'<span class="{liq_badge}">{row["Liq_tier_code"]}: {row["Liq_tier_name"]}</span> '
        badge_html += f'<span class="badge-tactical">Conf: {row["Data_Confidence"]:.0f}%</span> '
        badge_html += f'<span class="badge-tactical">Dil: {row["Dilution_Risk_Score"]:.0f}/100</span> '
        
        if row.get('Insider_Buying_90d', False):
            badge_html += '<span class="badge-insider">INSIDER BUY</span> '
        
        if row.get('Discovery_Exception', False):
            badge_html += '<span class="badge-discovery">DISCOVERY</span> '
        
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
            
            # Alpha model breakdown
            st.markdown("---")
            st.subheader("🎯 6-Model Alpha Breakdown")
            
            alpha_breakdown = alpha_breakdown_storage.get(row['Symbol'], [])
            if alpha_breakdown:
                for model_desc in alpha_breakdown:
                    st.write(f"• {model_desc}")
            
            # Data confidence breakdown
            st.markdown("---")
            st.subheader("📊 Data Confidence Details")
            st.write(f"**Score:** {row['Data_Confidence']}/100 ({row['Conf_Verdict']})")
            
            conf_breakdown = conf_breakdown_storage.get(row['Symbol'], [])
            if conf_breakdown:
                for detail in conf_breakdown:
                    st.caption(detail)
            
            # Dilution risk factors
            st.markdown("---")
            st.subheader("💀 Dilution Risk Factors")
            st.write(f"**Score:** {row['Dilution_Risk_Score']}/100 ({row['Dilution_Verdict']})")
            
            dilution_factors = dilution_factors_storage.get(row['Symbol'], [])
            if dilution_factors:
                for factor in dilution_factors:
                    st.caption(factor)
            
            # News
            st.markdown("---")
            st.subheader("📰 Recent News (Last 90 days)")
            
            ticker_news = news_cache.get(row['Symbol'], [])
            if ticker_news:
                for item in ticker_news[:5]:
                    tags = item.get('tag_string', '')
                    st.markdown(f"**{item['title']}** {tags}")
                    st.caption(f"{item['publisher']} • {datetime.datetime.fromtimestamp(item['timestamp']).strftime('%Y-%m-%d')}")
                    st.markdown("")
            else:
                st.info("No recent news")
        
        st.markdown("---")
    
    # Export
    st.download_button(
        "📥 Download Complete Analysis",
        df.to_csv(index=False),
        f"world_class_analysis_{datetime.date.today()}.csv",
        use_container_width=True
    )

st.caption("💎 Alpha Miner Pro - World-Class Edition • Survival > Alpha • Sell-In-Time")
