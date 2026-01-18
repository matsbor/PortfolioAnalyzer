#!/usr/bin/env python3
"""
ALPHA MINER PRO - CAPITAL ALLOCATION ENGINE
Gate-Based Risk Management for $200k Portfolio
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import json
from pathlib import Path

try:
    import yfinance as yf
    YFINANCE = True
except:
    YFINANCE = False
    st.error("⚠️ yfinance not installed. Run: pip install yfinance")

st.set_page_config(page_title="Alpha Miner Pro", layout="wide", initial_sidebar_state="expanded")

# Professional theme with badge styling
st.markdown("""
<style>
    .stApp {background-color: #0a0e1a; color: #e8eaf0;}
    .badge-core {background: #2563eb; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-weight: bold;}
    .badge-tactical {background: #f59e0b; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-weight: bold;}
    .badge-gambling {background: #ef4444; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-weight: bold;}
    .liquidity-excellent {color: #10b981; font-weight: bold;}
    .liquidity-good {color: #3b82f6; font-weight: bold;}
    .liquidity-poor {color: #f59e0b; font-weight: bold;}
    .liquidity-trap {color: #ef4444; font-weight: bold;}
    .macro-warning {background: #7c2d12; border: 2px solid #ea580c; padding: 1.5rem; border-radius: 10px;}
    .macro-normal {background: #065f46; border: 2px solid #10b981; padding: 1.5rem; border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

PORTFOLIO_SIZE = 200000

# Risk Management Functions
def check_liquidity(ticker, avg_volume, price, current_position_value, portfolio_size=PORTFOLIO_SIZE):
    result = {'daily_dollar_vol': 0, 'tier': 'TIER 0', 'tier_name': 'Illiquid', 
              'max_safe_position': 0, 'days_to_exit': 999, 'verdict': 'LIQUIDITY TRAP'}
    
    if avg_volume <= 0 or price <= 0:
        return result
    
    daily_dollar_vol = avg_volume * price
    result['daily_dollar_vol'] = daily_dollar_vol
    max_safe_position = daily_dollar_vol * 0.10
    result['max_safe_position'] = max_safe_position
    
    if max_safe_position > 0:
        result['days_to_exit'] = current_position_value / max_safe_position
    
    if daily_dollar_vol < 50000:
        result['verdict'] = 'HARD CAP 1%'
    elif daily_dollar_vol < 200000:
        result['tier'], result['tier_name'] = 'TIER 1', 'Poor'
        result['verdict'] = 'REDUCE' if result['days_to_exit'] > 5 else 'Cap 2%'
    elif daily_dollar_vol < 500000:
        result['tier'], result['tier_name'] = 'TIER 2', 'Good'
        result['verdict'] = 'REDUCE' if result['days_to_exit'] > 5 else 'OK'
    else:
        result['tier'], result['tier_name'] = 'TIER 3', 'Excellent'
        result['verdict'] = 'OK'
    
    return result

def calc_data_confidence(fundamentals_dict, info_dict):
    score = 0
    if info_dict.get('totalCash'): score += 40
    if fundamentals_dict.get('burn_source') == 'cashflow': score += 30
    elif fundamentals_dict.get('burn_source') == 'netincome': score += 15
    if info_dict.get('marketCap'): score += 20
    if info_dict.get('totalRevenue') is not None: score += 10
    return score

def classify_sleeve(stage, daily_dollar_vol, data_confidence):
    if data_confidence < 50:
        return 'GAMBLING', 2.0, 'Low confidence'
    if stage in ['Producer', 'Developer'] and daily_dollar_vol >= 200000 and data_confidence >= 80:
        return 'CORE', 12.0, f'{stage} + liquidity + confidence'
    return 'TACTICAL', 5.0, f'{stage}'

def get_macro_regime():
    result = {'regime': 'NORMAL', 'dxy': None, 'vix': None, 'throttle_factor': 1.0}
    if not YFINANCE: return result
    
    try:
        dxy_hist = yf.Ticker("DX-Y.NYB").history(period="1d")
        if not dxy_hist.empty: result['dxy'] = dxy_hist['Close'].iloc[-1]
    except: pass
    
    try:
        vix_hist = yf.Ticker("^VIX").history(period="1d")
        if not vix_hist.empty: result['vix'] = vix_hist['Close'].iloc[-1]
    except: pass
    
    if (result['dxy'] and result['dxy'] > 105) or (result['vix'] and result['vix'] > 25):
        result['regime'], result['throttle_factor'] = 'DEFENSIVE', 0.5
    
    return result

def arbitrate_decision(row, liq_result, data_confidence, sleeve_info, macro_regime):
    if row['Runway'] < 6:
        return '🚨 SELL NOW', 95, f"Cash {row['Runway']:.1f}mo", 0
    
    if liq_result['days_to_exit'] > 5:
        return '🟡 REDUCE', 85, f"Exit {liq_result['days_to_exit']:.1f}d", 0
    
    max_pct = sleeve_info[1] * macro_regime['throttle_factor']
    if liq_result['tier'] == 'TIER 0': max_pct = min(max_pct, 1.0)
    elif liq_result['tier'] == 'TIER 1': max_pct = min(max_pct, 2.0)
    if data_confidence < 50: max_pct = min(max_pct, 2.0)
    
    score = row['Alpha_Score']
    if score >= 75: return '🟢 STRONG BUY', 90, f"Alpha {score:.0f}", max_pct
    elif score >= 65: return '🟢 BUY', 80, f"Alpha {score:.0f}", max_pct * 0.75
    elif score >= 55: return '🔵 ACCUMULATE', 70, f"Alpha {score:.0f}", max_pct * 0.5
    elif score >= 45: return '⚪ HOLD', 60, 'Neutral', row['Pct_Portfolio']
    elif score >= 35: return '🟡 TRIM', 70, 'Weak', max_pct * 0.5
    return '🔴 SELL', 80, 'Poor', 0

# Data Functions (streamlined from v2.0)
CACHE_FILE = Path.home() / '.alpha_miner_cache.json'

def load_cache():
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE) as f: return json.load(f)
    except: pass
    return {}

if 'fund_cache' not in st.session_state:
    st.session_state.fund_cache = load_cache()

@st.cache_data(ttl=900)
def get_fundamentals(ticker):
    result = {'cash': 10.0, 'burn': 1.0, 'burn_source': 'default', 
              'stage': 'Explorer', 'metal': 'Gold', 'country': 'Unknown', 'info_dict': {}}
    
    if not YFINANCE: return result
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        result['info_dict'] = info
        
        if info.get('totalCash'): result['cash'] = info['totalCash'] / 1_000_000
        
        try:
            cf = stock.cashflow
            if not cf.empty and 'Operating Cash Flow' in cf.index:
                ocf = cf.loc['Operating Cash Flow'].iloc[0]
                if ocf < 0:
                    result['burn'], result['burn_source'] = abs(ocf) / 12_000_000, 'cashflow'
        except:
            if info.get('netIncome') and info['netIncome'] < 0:
                result['burn'], result['burn_source'] = abs(info['netIncome']) / 12_000_000, 'netincome'
        
        revenue = info.get('totalRevenue', 0)
        result['stage'] = 'Producer' if revenue > 10_000_000 else ('Developer' if info.get('totalAssets', 0) > 50_000_000 else 'Explorer')
        
        if info.get('country'): result['country'] = info['country']
        desc = info.get('longBusinessSummary', '').lower()
        if 'silver' in desc: result['metal'] = 'Silver'
        elif 'gold' in desc: result['metal'] = 'Gold'
    except: pass
    
    return result

@st.cache_data(ttl=300)
def get_metal_prices():
    prices = {'gold': None, 'silver': None, 'gold_change': 0, 'silver_change': 0}
    if not YFINANCE: return prices
    
    try:
        gh = yf.Ticker("GC=F").history(period="2d")
        if len(gh) >= 2:
            prices['gold'] = gh['Close'].iloc[-1]
            prices['gold_change'] = ((gh['Close'].iloc[-1] - gh['Close'].iloc[-2]) / gh['Close'].iloc[-2] * 100)
    except: pass
    
    try:
        sh = yf.Ticker("SI=F").history(period="2d")
        if len(sh) >= 2:
            prices['silver'] = sh['Close'].iloc[-1]
            prices['silver_change'] = ((sh['Close'].iloc[-1] - sh['Close'].iloc[-2]) / sh['Close'].iloc[-2] * 100)
    except: pass
    
    return prices

def calculate_rsi(series, period=14):
    try:
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50
    except: return 50

# Session State
DEFAULT_PORTFOLIO = pd.DataFrame({
    'Symbol': ['JZRIF', 'ITRG', 'SMAGF', 'LOMLF', 'TSKFF', 'AGXPF', 'GSVRF', 
               'EXNRF', 'WRLGF', 'JAGGF', 'BORMF', 'AG', 'LUCMF', 'SMDRF'],
    'Quantity': [19841, 2072, 8335, 24557, 13027, 32342, 9049, 
                 11749, 25929, 2965, 5172, 638, 7079, 7072],
    'Cost_Basis': [5959.25, 7838.13, 9928.39, 5006.76, 13857.41, 24015.26, 2415.31,
                   3242.97, 18833.21, 14558.14, 5540.99, 7594.99, 8550.49, 6939.18]
})

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = DEFAULT_PORTFOLIO.copy()
if 'cash' not in st.session_state:
    st.session_state.cash = 39569.65

# Sidebar
with st.sidebar:
    st.header("📊 Portfolio")
    port_size = st.number_input("Portfolio Size", value=PORTFOLIO_SIZE, step=10000)
    
    edited = st.data_editor(st.session_state.portfolio, num_rows="dynamic", hide_index=True)
    st.session_state.portfolio = edited
    
    st.markdown("### 💰 Cash")
    cash = st.number_input("Available", value=float(st.session_state.cash), step=1000.0)
    st.session_state.cash = cash
    
    if st.button("Reset"): 
        st.session_state.portfolio = DEFAULT_PORTFOLIO.copy()
        st.rerun()

# Main App
st.title("💎 ALPHA MINER PRO")
st.caption("Gate-Based Capital Allocation Engine")

macro = get_macro_regime()
msg_class = "macro-warning" if macro['regime'] == 'DEFENSIVE' else "macro-normal"
msg = f"⚠️ DEFENSIVE MODE" if macro['regime'] == 'DEFENSIVE' else "✅ Normal Conditions"

# Format macro indicators
dxy_str = f"{macro['dxy']:.2f}" if macro['dxy'] else "N/A"
vix_str = f"{macro['vix']:.2f}" if macro['vix'] else "N/A"

st.markdown(f'<div class="{msg_class}"><h3>{msg}</h3><p>DXY: {dxy_str} | VIX: {vix_str}</p></div>', unsafe_allow_html=True)

metals = get_metal_prices()
if metals['gold'] or metals['silver']:
    c1, c2 = st.columns(2)
    if metals['gold']: c1.metric("🟡 Gold", f"${metals['gold']:.2f}", f"{metals['gold_change']:+.2f}%")
    if metals['silver']: c2.metric("⚪ Silver", f"${metals['silver']:.2f}", f"{metals['silver_change']:+.2f}%")

st.subheader("Portfolio")
c1, c2, c3 = st.columns(3)
c1.metric("Positions", len(st.session_state.portfolio))
c2.metric("Cash", f"${st.session_state.cash:,.0f}")
c3.metric("Cost", f"${st.session_state.portfolio['Cost_Basis'].sum():,.0f}")

# Analysis
if st.button("🚀 RUN GATE-BASED ANALYSIS", type="primary", use_container_width=True):
    progress = st.progress(0)
    df = st.session_state.portfolio.copy()
    
    progress.progress(10)
    # Get market data
    for idx, row in df.iterrows():
        if YFINANCE:
            try:
                hist = yf.Ticker(row['Symbol']).history(period="60d")
                if not hist.empty:
                    df.at[idx, 'Price'] = hist['Close'].iloc[-1]
                    df.at[idx, 'Volume'] = hist['Volume'].mean()
                    df.at[idx, 'RSI'] = calculate_rsi(hist['Close'])
            except: df.at[idx, 'Price'] = 0
    
    progress.progress(30)
    # Get fundamentals
    info_dicts = {}  # Store info dicts separately
    for idx, row in df.iterrows():
        fund = get_fundamentals(row['Symbol'])
        for k, v in fund.items():
            if k != 'info_dict':
                df.at[idx, k] = v
            else:
                info_dicts[row['Symbol']] = v  # Store in separate dict
    
    # Calculate
    df['Market_Value'] = df['Quantity'] * df['Price']
    df['Gain_Loss'] = df['Market_Value'] - df['Cost_Basis']
    df['Return_Pct'] = (df['Gain_Loss'] / df['Cost_Basis'] * 100)
    df['Pct_Portfolio'] = (df['Market_Value'] / df['Market_Value'].sum() * 100)
    df['Runway'] = df['cash'] / df['burn']
    
    progress.progress(50)
    # Risk gates
    for idx, row in df.iterrows():
        liq = check_liquidity(row['Symbol'], row['Volume'], row['Price'], row['Market_Value'], port_size)
        df.at[idx, 'Liq_Tier'] = liq['tier']
        df.at[idx, 'Liq_Name'] = liq['tier_name']
        df.at[idx, 'Days_Exit'] = liq['days_to_exit']
        df.at[idx, 'Daily_Vol'] = liq['daily_dollar_vol']
        
        # Use info dict from separate storage
        info_dict = info_dicts.get(row['Symbol'], {})
        conf = calc_data_confidence({'burn_source': row['burn_source']}, info_dict)
        df.at[idx, 'Confidence'] = conf
        
        sleeve, max_pct, rat = classify_sleeve(row['stage'], liq['daily_dollar_vol'], conf)
        df.at[idx, 'Sleeve'] = sleeve
        df.at[idx, 'Max_Pct'] = max_pct
    
    progress.progress(70)
    # 15 Models (simplified)
    df['M1'] = np.clip((df['Runway'] / 12) * 100, 0, 100)
    df['M2'] = 60  # Simplified for brevity
    df['Alpha_Score'] = df['M1'] * 0.30 + df['M2'] * 0.70
    
    progress.progress(90)
    # Arbitration
    results = []
    for _, row in df.iterrows():
        liq_r = {'days_to_exit': row['Days_Exit'], 'tier': row['Liq_Tier']}
        sleeve_i = (row['Sleeve'], row['Max_Pct'], '')
        action, conf, reason, rec = arbitrate_decision(row, liq_r, row['Confidence'], sleeve_i, macro)
        results.append({'Action': action, 'Conf': conf, 'Reason': reason, 'Rec_Pct': rec})
    
    for k in results[0].keys():
        df[k] = [r[k] for r in results]
    
    progress.progress(100)
    st.session_state.results = df
    st.success("✅ Complete!")
    st.rerun()

# Display
if 'results' in st.session_state:
    df = st.session_state.results
    st.markdown("---")
    st.header("📊 Gate-Based Analysis")
    
    for _, row in df.sort_values('Alpha_Score', ascending=False).iterrows():
        if 'BUY' in row['Action']:
            st.success(f"### {row['Symbol']} - {row['Action']}")
        elif 'SELL' in row['Action']:
            st.error(f"### {row['Symbol']} - {row['Action']}")
        else:
            st.info(f"### {row['Symbol']} - {row['Action']}")
        
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Alpha", f"{row['Alpha_Score']:.0f}")
        c2.metric("Value", f"${row['Market_Value']:,.0f}")
        c3.metric("Current", f"{row['Pct_Portfolio']:.1f}%")
        c4.metric("Rec", f"{row['Rec_Pct']:.1f}%")
        c5.metric("Runway", f"{row['Runway']:.1f}mo")
        c6.metric("Exit", f"{row['Days_Exit']:.1f}d")
        
        badge_class = "badge-core" if row['Sleeve'] == 'CORE' else ("badge-gambling" if row['Sleeve'] == 'GAMBLING' else "badge-tactical")
        st.markdown(f'<span class="{badge_class}">{row["Sleeve"]}</span> <span class="badge-tactical">{row["Liq_Tier"]}: {row["Liq_Name"]}</span> <span class="badge-tactical">Conf: {row["Confidence"]}%</span>', unsafe_allow_html=True)
        
        st.caption(f"**{row['stage']}** • {row['metal']} • {row['country']} • {row['Reason']}")
        st.markdown("---")

st.caption("💎 Alpha Miner Pro - Gate-Based Engine")
