#!/usr/bin/env python3
"""
ALPHA MINER ULTIMATE - TRANSPARENT EDITION
Every score explained with tick marks, colors, and clear reasoning
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime

try:
    import yfinance as yf
    YFINANCE = True
except:
    YFINANCE = False

st.set_page_config(page_title="Alpha Miner Pro", layout="wide", initial_sidebar_state="collapsed")

# Professional dark theme with better readability
st.markdown("""
<style>
    .stApp {background-color: #0e1117; color: #fafafa;}
    
    .main-title {font-size: 3rem; font-weight: 800; color: #667eea; margin-bottom: 0.5rem;}
    .subtitle {font-size: 1.2rem; color: #a0aec0; margin-bottom: 2rem;}
    
    /* Score cards */
    .score-card {background: #1a202c; border-radius: 15px; padding: 2rem; margin: 1rem 0; 
                 border: 2px solid #2d3748; box-shadow: 0 4px 20px rgba(0,0,0,0.3);}
    .score-excellent {border-color: #48bb78; background: linear-gradient(135deg, #1a202c 0%, #1a3020 100%);}
    .score-good {border-color: #4299e1; background: linear-gradient(135deg, #1a202c 0%, #1a2530 100%);}
    .score-warning {border-color: #ed8936; background: linear-gradient(135deg, #1a202c 0%, #3a2520 100%);}
    .score-critical {border-color: #f56565; background: linear-gradient(135deg, #1a202c 0%, #3a1a1a 100%);}
    
    /* Analysis breakdown */
    .breakdown-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                     gap: 1rem; margin: 1.5rem 0;}
    .factor-card {background: #2d3748; padding: 1.5rem; border-radius: 10px; border-left: 5px solid;}
    .factor-pass {border-color: #48bb78;}
    .factor-warning {border-color: #ed8936;}
    .factor-fail {border-color: #f56565;}
    
    /* Tick marks */
    .tick-pass {color: #48bb78; font-size: 1.5rem; font-weight: bold;}
    .tick-warning {color: #ed8936; font-size: 1.5rem; font-weight: bold;}
    .tick-fail {color: #f56565; font-size: 1.5rem; font-weight: bold;}
    
    /* Stock cards */
    .stock-card {background: #1a202c; border-radius: 15px; padding: 2rem; margin: 1.5rem 0; 
                 border: 3px solid; box-shadow: 0 8px 25px rgba(0,0,0,0.4);}
    .stock-strong-buy {border-color: #48bb78; background: linear-gradient(135deg, #1a202c 0%, #1a3020 100%);}
    .stock-buy {border-color: #4299e1; background: linear-gradient(135deg, #1a202c 0%, #1a2530 100%);}
    .stock-hold {border-color: #718096; background: linear-gradient(135deg, #1a202c 0%, #252525 100%);}
    .stock-sell {border-color: #f56565; background: linear-gradient(135deg, #1a202c 0%, #3a1a1a 100%);}
    
    /* Metrics */
    .metric-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;}
    .metric-box {background: #2d3748; padding: 1.5rem; border-radius: 10px; text-align: center; border: 1px solid #4a5568;}
    .metric-value {font-size: 2.5rem; font-weight: bold; margin: 0.5rem 0;}
    .metric-label {font-size: 0.9rem; color: #a0aec0; text-transform: uppercase;}
    
    /* Progress bars */
    .score-bar {background: #2d3748; height: 30px; border-radius: 15px; overflow: hidden; margin: 0.5rem 0;}
    .score-fill {height: 100%; display: flex; align-items: center; justify-content: center; 
                 color: white; font-weight: bold; transition: width 0.3s;}
    .score-fill-high {background: linear-gradient(90deg, #48bb78 0%, #38a169 100%);}
    .score-fill-med {background: linear-gradient(90deg, #4299e1 0%, #3182ce 100%);}
    .score-fill-low {background: linear-gradient(90deg, #ed8936 0%, #dd6b20 100%);}
</style>
""", unsafe_allow_html=True)

# Initialize
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame({
        'Symbol': ['JZRIF', 'ITRG', 'SMAGF', 'LOMLF', 'TSKFF', 'AGXPF', 'GSVRF', 
                   'EXNRF', 'WRLGF', 'JAGGF', 'BORMF', 'AG', 'LUCMF', 'SMDRF'],
        'Quantity': [19841, 2072, 8335, 24557, 13027, 32342, 9049, 
                     11749, 25929, 2965, 5172, 638, 7079, 7072],
        'Cost_Basis': [5959.25, 7838.13, 9928.39, 5006.76, 13857.41, 24015.26, 2415.31,
                       3242.97, 18833.21, 14558.14, 5540.99, 7594.99, 8550.49, 6939.18]
    })
    st.session_state.cash = 39569.65

COMPANY_DATA = {
    'ITRG': {'cash': 12, 'burn': 0.8, 'stage': 'Explorer', 'metal': 'Gold', 'country': 'Canada'},
    'SMAGF': {'cash': 8, 'burn': 0.7, 'stage': 'Developer', 'metal': 'Gold', 'country': 'Unknown'},
    'WRLGF': {'cash': 10, 'burn': 1.0, 'stage': 'Explorer', 'metal': 'Gold', 'country': 'Canada'},
    'EXNRF': {'cash': 5, 'burn': 0.6, 'stage': 'Developer', 'metal': 'Silver', 'country': 'Mexico'},
    'LOMLF': {'cash': 4, 'burn': 0.9, 'stage': 'Explorer', 'metal': 'Gold', 'country': 'Fiji'},
    'GSVRF': {'cash': 20, 'burn': 1.5, 'stage': 'Producer', 'metal': 'Silver', 'country': 'Mexico'},
    'AGXPF': {'cash': 18, 'burn': 1.5, 'stage': 'Explorer', 'metal': 'Silver', 'country': 'Peru'},
    'TSKFF': {'cash': 15, 'burn': 1.2, 'stage': 'Explorer', 'metal': 'Gold', 'country': 'Canada'},
    'BORMF': {'cash': 7, 'burn': 0.6, 'stage': 'Explorer', 'metal': 'Gold', 'country': 'Canada'},
    'JAGGF': {'cash': 25, 'burn': 2.0, 'stage': 'Producer', 'metal': 'Gold', 'country': 'Brazil'},
    'LUCMF': {'cash': 6, 'burn': 0.8, 'stage': 'Explorer', 'metal': 'Gold', 'country': 'Canada'},
    'AG': {'cash': 50, 'burn': 3.0, 'stage': 'Producer', 'metal': 'Silver', 'country': 'Mexico'},
    'SMDRF': {'cash': 12, 'burn': 1.0, 'stage': 'Developer', 'metal': 'Silver', 'country': 'Mexico'},
    'JZRIF': {'cash': 5, 'burn': 0.5, 'stage': 'Explorer', 'metal': 'Gold', 'country': 'Canada'},
}

FINANCINGS = {'LOMLF': {'amount': 8.0}, 'EXNRF': {'amount': 15.0}}

# Header
st.markdown('<p class="main-title">💎 ALPHA MINER ULTIMATE</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Transparent Multi-Factor Analysis System</p>', unsafe_allow_html=True)

# Input
col1, col2 = st.columns([4, 1])
with col1:
    edited_df = st.data_editor(st.session_state.portfolio, use_container_width=True, num_rows="dynamic", hide_index=True)
    st.session_state.portfolio = edited_df
with col2:
    cash = st.number_input("💰 Cash", value=float(st.session_state.cash), step=1000.0, label_visibility="collapsed")
    st.session_state.cash = cash

if st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True):
    with st.spinner("Running complete analysis..."):
        df = edited_df.copy()
        
        # Get prices
        for idx, row in df.iterrows():
            if YFINANCE:
                try:
                    ticker = yf.Ticker(row['Symbol'])
                    hist = ticker.history(period="60d")
                    if not hist.empty:
                        df.at[idx, 'Price'] = hist['Close'].iloc[-1]
                        df.at[idx, 'Price_7d'] = hist['Close'].iloc[-7] if len(hist) > 7 else hist['Close'].iloc[0]
                        df.at[idx, 'Price_30d'] = hist['Close'].iloc[-30] if len(hist) > 30 else hist['Close'].iloc[0]
                        df.at[idx, 'Volume'] = hist['Volume'].mean()
                        df.at[idx, 'Volatility'] = hist['Close'].pct_change().std() * 100
                except:
                    df.at[idx, 'Price'] = 1.0
        
        # Add company data
        for idx, row in df.iterrows():
            if row['Symbol'] in COMPANY_DATA:
                for k, v in COMPANY_DATA[row['Symbol']].items():
                    df.at[idx, k] = v
        
        # Calculate metrics
        df['Market_Value'] = df['Quantity'] * df['Price']
        df['Gain_Loss'] = df['Market_Value'] - df['Cost_Basis']
        df['Return_Pct'] = (df['Gain_Loss'] / df['Cost_Basis'] * 100)
        df['Return_7d'] = ((df['Price'] - df['Price_7d']) / df['Price_7d'] * 100)
        df['Return_30d'] = ((df['Price'] - df['Price_30d']) / df['Price_30d'] * 100)
        
        total_mv = df['Market_Value'].sum()
        df['Pct_Portfolio'] = (df['Market_Value'] / total_mv * 100)
        
        # Pro-forma cash
        df['Pro_Forma_Cash'] = df.apply(
            lambda r: r['cash'] + FINANCINGS[r['Symbol']]['amount'] if r['Symbol'] in FINANCINGS else r['cash'],
            axis=1
        )
        df['Runway'] = df['Pro_Forma_Cash'] / df['burn']
        
        # === 10 ANALYSIS FACTORS (Each shows ✓/✗) ===
        
        # Factor 1: Cash Runway (Most Important)
        df['F1_Cash_Pass'] = df['Runway'] >= 8
        df['F1_Cash_Score'] = np.clip((df['Runway'] / 12) * 100, 0, 100)
        df['F1_Cash_Reason'] = df.apply(
            lambda r: f"✓ {r['Runway']:.1f} months" if r['F1_Cash_Pass'] else f"✗ Only {r['Runway']:.1f} months",
            axis=1
        )
        
        # Factor 2: Position Size
        df['F2_Size_Pass'] = df['Pct_Portfolio'] <= 15
        df['F2_Size_Score'] = np.clip(100 - (df['Pct_Portfolio'] * 4), 0, 100)
        df['F2_Size_Reason'] = df.apply(
            lambda r: f"✓ {r['Pct_Portfolio']:.1f}%" if r['F2_Size_Pass'] else f"✗ {r['Pct_Portfolio']:.1f}% (too large)",
            axis=1
        )
        
        # Factor 3: Performance
        df['F3_Perf_Pass'] = df['Return_Pct'] > -20
        df['F3_Perf_Score'] = np.clip(df['Return_Pct'] + 50, 0, 100)
        df['F3_Perf_Reason'] = df.apply(
            lambda r: f"✓ {r['Return_Pct']:+.1f}%" if r['F3_Perf_Pass'] else f"✗ {r['Return_Pct']:+.1f}%",
            axis=1
        )
        
        # Factor 4: Momentum
        df['F4_Mom_Pass'] = df['Return_7d'] > -5
        df['F4_Mom_Score'] = np.clip(df['Return_7d'] * 5 + 50, 0, 100)
        df['F4_Mom_Reason'] = df.apply(
            lambda r: f"✓ {r['Return_7d']:+.1f}% (7d)" if r['F4_Mom_Pass'] else f"✗ {r['Return_7d']:+.1f}% (7d)",
            axis=1
        )
        
        # Factor 5: Jurisdiction
        juris_scores = {'Canada': 90, 'USA': 90, 'Mexico': 60, 'Peru': 55, 'Brazil': 55, 'Fiji': 50, 'Unknown': 40}
        df['F5_Juris_Score'] = df['country'].map(juris_scores).fillna(40)
        df['F5_Juris_Pass'] = df['F5_Juris_Score'] >= 60
        df['F5_Juris_Reason'] = df.apply(
            lambda r: f"✓ {r['country']}" if r['F5_Juris_Pass'] else f"⚠ {r['country']}",
            axis=1
        )
        
        # Factor 6: Stage
        stage_scores = {'Producer': 85, 'Developer': 65, 'Explorer': 45}
        df['F6_Stage_Score'] = df['stage'].map(stage_scores).fillna(50)
        df['F6_Stage_Pass'] = df['F6_Stage_Score'] >= 60
        df['F6_Stage_Reason'] = df.apply(
            lambda r: f"✓ {r['stage']}" if r['F6_Stage_Pass'] else f"⚠ {r['stage']}",
            axis=1
        )
        
        # Factor 7: Volatility
        df['F7_Vol_Pass'] = df['Volatility'] < 5
        df['F7_Vol_Score'] = np.clip(100 - (df['Volatility'] * 10), 0, 100)
        df['F7_Vol_Reason'] = df.apply(
            lambda r: f"✓ {r['Volatility']:.1f}%" if r['F7_Vol_Pass'] else f"⚠ {r['Volatility']:.1f}%",
            axis=1
        )
        
        # Factor 8: Liquidity
        df['F8_Liq_Pass'] = df['Volume'] > 50000
        df['F8_Liq_Score'] = np.clip((df['Volume'] / 100000) * 50, 0, 100)
        df['F8_Liq_Reason'] = df.apply(
            lambda r: f"✓ {r['Volume']:,.0f}" if r['F8_Liq_Pass'] else f"⚠ {r['Volume']:,.0f}",
            axis=1
        )
        
        # Factor 9: Trend (30d)
        df['F9_Trend_Pass'] = df['Return_30d'] > 0
        df['F9_Trend_Score'] = np.clip(df['Return_30d'] * 2 + 50, 0, 100)
        df['F9_Trend_Reason'] = df.apply(
            lambda r: f"✓ {r['Return_30d']:+.1f}% (30d)" if r['F9_Trend_Pass'] else f"✗ {r['Return_30d']:+.1f}% (30d)",
            axis=1
        )
        
        # Factor 10: Metal Sentiment (simulated - would use real macro data)
        metal_sentiment = {'Gold': 70, 'Silver': 75}
        df['F10_Metal_Score'] = df['metal'].map(metal_sentiment).fillna(60)
        df['F10_Metal_Pass'] = df['F10_Metal_Score'] >= 60
        df['F10_Metal_Reason'] = df.apply(
            lambda r: f"✓ {r['metal']} bullish" if r['F10_Metal_Pass'] else f"⚠ {r['metal']}",
            axis=1
        )
        
        # WEIGHTED ALPHA SCORE
        df['Alpha_Score'] = (
            df['F1_Cash_Score'] * 0.35 +      # Cash runway - 35% (most important)
            df['F3_Perf_Score'] * 0.20 +      # Performance - 20%
            df['F4_Mom_Score'] * 0.15 +       # Momentum - 15%
            df['F2_Size_Score'] * 0.10 +      # Position size - 10%
            df['F5_Juris_Score'] * 0.08 +     # Jurisdiction - 8%
            df['F6_Stage_Score'] * 0.05 +     # Stage - 5%
            df['F7_Vol_Score'] * 0.03 +       # Volatility - 3%
            df['F9_Trend_Score'] * 0.02 +     # 30d trend - 2%
            df['F8_Liq_Score'] * 0.01 +       # Liquidity - 1%
            df['F10_Metal_Score'] * 0.01      # Metal - 1%
        )
        
        # Count passed factors
        df['Factors_Passed'] = (
            df['F1_Cash_Pass'].astype(int) +
            df['F2_Size_Pass'].astype(int) +
            df['F3_Perf_Pass'].astype(int) +
            df['F4_Mom_Pass'].astype(int) +
            df['F5_Juris_Pass'].astype(int) +
            df['F6_Stage_Pass'].astype(int) +
            df['F7_Vol_Pass'].astype(int) +
            df['F8_Liq_Pass'].astype(int) +
            df['F9_Trend_Pass'].astype(int) +
            df['F10_Metal_Pass'].astype(int)
        )
        
        # FINAL DECISION
        def get_decision(row):
            score = row['Alpha_Score']
            runway = row['Runway']
            passed = row['Factors_Passed']
            
            # Critical overrides
            if runway < 6:
                return '🚨 SELL NOW', 'CRITICAL', 95
            elif runway < 8 and score < 50:
                return '🔴 SELL', 'HIGH', 85
            
            # Score-based
            if score >= 75 and passed >= 7:
                return '🟢 STRONG BUY', 'HIGH', 90
            elif score >= 65 and passed >= 6:
                return '🟢 BUY', 'MEDIUM', 80
            elif score >= 55 and passed >= 5:
                return '🔵 ACCUMULATE', 'MEDIUM', 70
            elif score >= 45:
                return '⚪ HOLD', 'LOW', 60
            elif score >= 35:
                return '🟡 TRIM', 'MEDIUM', 70
            else:
                return '🔴 SELL', 'HIGH', 80
        
        df[['Action', 'Priority', 'Confidence']] = df.apply(
            lambda r: pd.Series(get_decision(r)), axis=1
        )
        
        # Portfolio health
        health_score = (
            df['Alpha_Score'].mean() * 0.5 +
            (df['Runway'] >= 8).mean() * 100 * 0.3 +
            (df['Factors_Passed'] >= 6).mean() * 100 * 0.2
        )
        
        st.session_state.results = df
        st.session_state.health_score = health_score
    
    st.success("✅ Analysis complete!")
    st.rerun()

# === DISPLAY RESULTS ===
if 'results' in st.session_state:
    df = st.session_state.results
    health = st.session_state.health_score
    
    st.markdown("---")
    
    # Health score
    if health >= 75:
        health_class = "score-excellent"
        health_status = "🟢 EXCELLENT"
    elif health >= 60:
        health_class = "score-good"
        health_status = "🔵 GOOD"
    elif health >= 45:
        health_class = "score-warning"
        health_status = "🟠 NEEDS ATTENTION"
    else:
        health_class = "score-critical"
        health_status = "🔴 CRITICAL"
    
    st.markdown(f"""
    <div class="score-card {health_class}">
        <div style="font-size: 1rem; opacity: 0.8;">PORTFOLIO HEALTH SCORE</div>
        <div style="font-size: 4rem; font-weight: 900; margin: 1rem 0;">{health:.0f}/100</div>
        <div style="font-size: 1.5rem; font-weight: 600;">{health_status}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    total_equity = df['Market_Value'].sum()
    total_value = total_equity + cash
    total_gain = df['Gain_Loss'].sum()
    total_return = (total_gain / df['Cost_Basis'].sum() * 100)
    
    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-box">
            <div class="metric-label">Total Value</div>
            <div class="metric-value">${total_value:,.0f}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Return</div>
            <div class="metric-value" style="color: {'#48bb78' if total_return > 0 else '#f56565'};">{total_return:+.1f}%</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Cash</div>
            <div class="metric-value">${cash:,.0f}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Positions</div>
            <div class="metric-value">{len(df)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## 📊 Complete Analysis with Factor Breakdown")
    
    # Show each stock with full breakdown
    for _, row in df.sort_values('Alpha_Score', ascending=False).iterrows():
        # Determine card class
        if 'STRONG BUY' in row['Action'] or 'BUY' in row['Action']:
            card_class = 'stock-strong-buy'
        elif 'SELL' in row['Action']:
            card_class = 'stock-sell'
        else:
            card_class = 'stock-hold'
        
        # Score bar color
        if row['Alpha_Score'] >= 70:
            bar_class = 'score-fill-high'
        elif row['Alpha_Score'] >= 50:
            bar_class = 'score-fill-med'
        else:
            bar_class = 'score-fill-low'
        
        st.markdown(f"""
        <div class="stock-card {card_class}">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                <div>
                    <h2 style="margin: 0; font-size: 2rem;">{row['Symbol']}</h2>
                    <p style="margin: 0.5rem 0; color: #a0aec0;">{row['stage']} • {row['metal']} • {row['country']}</p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 2.5rem; font-weight: bold;">{row['Action']}</div>
                    <div style="font-size: 1.2rem; color: #a0aec0;">Confidence: {row['Confidence']:.0f}%</div>
                </div>
            </div>
            
            <div style="margin: 1.5rem 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                    <span style="color: #a0aec0;">Alpha Score</span>
                    <span style="font-weight: bold;">{row['Alpha_Score']:.0f}/100 • {row['Factors_Passed']}/10 factors passed</span>
                </div>
                <div class="score-bar">
                    <div class="score-fill {bar_class}" style="width: {row['Alpha_Score']}%;">{row['Alpha_Score']:.0f}/100</div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1.5rem 0;">
                <div>
                    <div style="color: #a0aec0; font-size: 0.9rem;">Position Value</div>
                    <div style="font-size: 1.5rem; font-weight: bold;">${row['Market_Value']:,.0f}</div>
                    <div style="font-size: 0.9rem; color: #a0aec0;">{row['Pct_Portfolio']:.1f}% of portfolio</div>
                </div>
                <div>
                    <div style="color: #a0aec0; font-size: 0.9rem;">Return</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: {'#48bb78' if row['Return_Pct'] > 0 else '#f56565'};">{row['Return_Pct']:+.1f}%</div>
                    <div style="font-size: 0.9rem; color: #a0aec0;">${row['Gain_Loss']:+,.0f}</div>
                </div>
                <div>
                    <div style="color: #a0aec0; font-size: 0.9rem;">Cash Runway</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: {'#48bb78' if row['Runway'] >= 8 else '#f56565'};">{row['Runway']:.1f}mo</div>
                    <div style="font-size: 0.9rem; color: #a0aec0;">${row['Pro_Forma_Cash']:.1f}M cash</div>
                </div>
                <div>
                    <div style="color: #a0aec0; font-size: 0.9rem;">7-Day Trend</div>
                    <div style="font-size: 1.5rem; font-weight: bold; color: {'#48bb78' if row['Return_7d'] > 0 else '#f56565'};">{row['Return_7d']:+.1f}%</div>
                    <div style="font-size: 0.9rem; color: #a0aec0;">30d: {row['Return_30d']:+.1f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Factor breakdown
        with st.expander(f"🔍 See 10-Factor Analysis Breakdown for {row['Symbol']}", expanded=False):
            st.markdown('<div class="breakdown-grid">', unsafe_allow_html=True)
            
            factors = [
                ('Cash Runway', row['F1_Cash_Pass'], row['F1_Cash_Score'], row['F1_Cash_Reason'], '35%'),
                ('Performance', row['F3_Perf_Pass'], row['F3_Perf_Score'], row['F3_Perf_Reason'], '20%'),
                ('Momentum (7d)', row['F4_Mom_Pass'], row['F4_Mom_Score'], row['F4_Mom_Reason'], '15%'),
                ('Position Size', row['F2_Size_Pass'], row['F2_Size_Score'], row['F2_Size_Reason'], '10%'),
                ('Jurisdiction', row['F5_Juris_Pass'], row['F5_Juris_Score'], row['F5_Juris_Reason'], '8%'),
                ('Stage', row['F6_Stage_Pass'], row['F6_Stage_Score'], row['F6_Stage_Reason'], '5%'),
                ('Volatility', row['F7_Vol_Pass'], row['F7_Vol_Score'], row['F7_Vol_Reason'], '3%'),
                ('Trend (30d)', row['F9_Trend_Pass'], row['F9_Trend_Score'], row['F9_Trend_Reason'], '2%'),
                ('Liquidity', row['F8_Liq_Pass'], row['F8_Liq_Score'], row['F8_Liq_Reason'], '1%'),
                ('Metal Sentiment', row['F10_Metal_Pass'], row['F10_Metal_Score'], row['F10_Metal_Reason'], '1%'),
            ]
            
            for name, passed, score, reason, weight in factors:
                factor_class = 'factor-pass' if passed else ('factor-warning' if score > 40 else 'factor-fail')
                tick_class = 'tick-pass' if passed else ('tick-warning' if score > 40 else 'tick-fail')
                tick_symbol = '✓' if passed else ('⚠' if score > 40 else '✗')
                
                st.markdown(f"""
                <div class="factor-card {factor_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <div>
                            <span class="{tick_class}">{tick_symbol}</span>
                            <strong style="margin-left: 0.5rem;">{name}</strong>
                        </div>
                        <div style="font-size: 0.9rem; color: #a0aec0;">Weight: {weight}</div>
                    </div>
                    <div style="font-size: 1.2rem; margin: 0.5rem 0;">{reason}</div>
                    <div style="font-size: 1rem; color: #a0aec0;">Score: {score:.0f}/100</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Export
    st.markdown("---")
    st.download_button(
        "📥 Download Complete Analysis",
        df.to_csv(index=False),
        f"analysis_{datetime.date.today()}.csv",
        "text/csv",
        use_container_width=True
    )
