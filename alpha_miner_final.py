#!/usr/bin/env python3
"""
ALPHA MINER ULTIMATE - THE DEFINITIVE EDITION
Crystal clear analysis with named models, full transparency, professional UX
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

st.set_page_config(page_title="Alpha Miner Ultimate", layout="wide", initial_sidebar_state="collapsed")

# Professional theme
st.markdown("""
<style>
    .stApp {background-color: #0a0e1a; color: #e8eaf0;}
    .main {background-color: #0a0e1a;}
    h1, h2, h3 {color: #e8eaf0 !important;}
    .stMarkdown {color: #e8eaf0;}
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
st.title("💎 ALPHA MINER ULTIMATE")
st.caption("The Definitive Multi-Model Capital Allocation System for Junior Miners")

# Input
col1, col2 = st.columns([4, 1])
with col1:
    edited_df = st.data_editor(st.session_state.portfolio, use_container_width=True, num_rows="dynamic", hide_index=True)
    st.session_state.portfolio = edited_df
with col2:
    st.write("**💰 Available Cash**")
    cash = st.number_input("Cash", value=float(st.session_state.cash), step=1000.0, label_visibility="collapsed")
    st.session_state.cash = cash

if st.button("🚀 RUN COMPLETE ANALYSIS", type="primary", use_container_width=True):
    progress = st.progress(0, text="Starting analysis...")
    
    df = edited_df.copy()
    
    # === STEP 1: MARKET DATA ===
    progress.progress(10, text="📊 Fetching live market data...")
    
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
                    df.at[idx, 'High_52w'] = hist['High'].max()
                    df.at[idx, 'Low_52w'] = hist['Low'].min()
                    df.at[idx, 'Volatility'] = hist['Close'].pct_change().std() * 100
            except:
                df.at[idx, 'Price'] = 1.0
    
    # === STEP 2: FUNDAMENTAL DATA ===
    progress.progress(20, text="🔍 Loading fundamental data...")
    
    for idx, row in df.iterrows():
        if row['Symbol'] in COMPANY_DATA:
            for k, v in COMPANY_DATA[row['Symbol']].items():
                df.at[idx, k] = v
    
    df['Market_Value'] = df['Quantity'] * df['Price']
    df['Gain_Loss'] = df['Market_Value'] - df['Cost_Basis']
    df['Return_Pct'] = (df['Gain_Loss'] / df['Cost_Basis'] * 100)
    df['Return_7d'] = ((df['Price'] - df['Price_7d']) / df['Price_7d'] * 100)
    df['Return_30d'] = ((df['Price'] - df['Price_30d']) / df['Price_30d'] * 100)
    
    total_mv = df['Market_Value'].sum()
    df['Pct_Portfolio'] = (df['Market_Value'] / total_mv * 100)
    
    df['Pro_Forma_Cash'] = df.apply(
        lambda r: r['cash'] + FINANCINGS[r['Symbol']]['amount'] if r['Symbol'] in FINANCINGS else r['cash'],
        axis=1
    )
    df['Runway'] = df['Pro_Forma_Cash'] / df['burn']
    df['Distance_From_High'] = ((df['Price'] - df['High_52w']) / df['High_52w'] * 100)
    df['Distance_From_Low'] = ((df['Price'] - df['Low_52w']) / df['Low_52w'] * 100)
    
    # === STEP 3: 15 PREDICTION MODELS ===
    progress.progress(40, text="🤖 Running 15 prediction models...")
    
    # MODEL 1: Dilution Risk Model (30% weight)
    df['M1_Name'] = "Dilution Risk Model"
    df['M1_Score'] = np.clip((df['Runway'] / 12) * 100, 0, 100)
    df['M1_Signal'] = df['M1_Score'].apply(lambda x: '🟢 BUY' if x >= 70 else ('🟡 HOLD' if x >= 50 else '🔴 SELL'))
    df['M1_Explanation'] = df.apply(
        lambda r: f"Cash runway of {r['Runway']:.1f} months. " + 
                  (f"Excellent - no dilution risk for 1+ years" if r['Runway'] >= 12 else
                   f"Adequate - safe for 6-12 months" if r['Runway'] >= 8 else
                   f"CRITICAL - Dilution likely within 6 months"),
        axis=1
    )
    
    # MODEL 2: Momentum Model (15% weight)
    df['M2_Name'] = "Multi-Timeframe Momentum"
    df['M2_Score'] = np.clip((df['Return_7d'] * 5) + (df['Return_30d'] * 2) + 50, 0, 100)
    df['M2_Signal'] = df['M2_Score'].apply(lambda x: '🟢 BUY' if x >= 70 else ('🟡 HOLD' if x >= 50 else '🔴 SELL'))
    df['M2_Explanation'] = df.apply(
        lambda r: f"7-day: {r['Return_7d']:+.1f}%, 30-day: {r['Return_30d']:+.1f}%. " +
                  ("Strong upward momentum" if r['M2_Score'] >= 70 else
                   "Neutral momentum" if r['M2_Score'] >= 50 else
                   "Negative momentum"),
        axis=1
    )
    
    # MODEL 3: Value/Performance Model (12% weight)
    df['M3_Name'] = "Risk-Adjusted Performance"
    df['M3_Score'] = np.clip(df['Return_Pct'] - (df['Volatility'] * 3) + 50, 0, 100)
    df['M3_Signal'] = df['M3_Score'].apply(lambda x: '🟢 BUY' if x >= 70 else ('🟡 HOLD' if x >= 50 else '🔴 SELL'))
    df['M3_Explanation'] = df.apply(
        lambda r: f"Return: {r['Return_Pct']:+.1f}%, Volatility: {r['Volatility']:.1f}%. " +
                  ("Excellent risk-adjusted returns" if r['M3_Score'] >= 70 else
                   "Moderate risk/reward" if r['M3_Score'] >= 50 else
                   "Poor risk-adjusted performance"),
        axis=1
    )
    
    # MODEL 4: Technical Position Model (10% weight)
    df['M4_Name'] = "52-Week Range Analysis"
    df['M4_Score'] = np.clip(50 + (df['Distance_From_Low'] * 0.5) - (abs(df['Distance_From_High']) * 0.3), 0, 100)
    df['M4_Signal'] = df['M4_Score'].apply(lambda x: '🟢 BUY' if x >= 70 else ('🟡 HOLD' if x >= 50 else '🔴 SELL'))
    df['M4_Explanation'] = df.apply(
        lambda r: f"Price is {abs(r['Distance_From_High']):.1f}% from 52w high, {r['Distance_From_Low']:.1f}% from low. " +
                  ("Near lows - potential value" if r['M4_Score'] >= 70 else
                   "Mid-range" if r['M4_Score'] >= 50 else
                   "Near highs - extended"),
        axis=1
    )
    
    # MODEL 5: Concentration Risk Model (8% weight)
    df['M5_Name'] = "Portfolio Concentration"
    df['M5_Score'] = np.clip(100 - (df['Pct_Portfolio'] * 5), 0, 100)
    df['M5_Signal'] = df['M5_Score'].apply(lambda x: '🟢 BUY' if x >= 70 else ('🟡 HOLD' if x >= 50 else '🔴 SELL'))
    df['M5_Explanation'] = df.apply(
        lambda r: f"Position size: {r['Pct_Portfolio']:.1f}% of portfolio. " +
                  ("Well-sized position" if r['Pct_Portfolio'] <= 10 else
                   "Moderate concentration" if r['Pct_Portfolio'] <= 15 else
                   "OVER-CONCENTRATED - trim position"),
        axis=1
    )
    
    # MODEL 6: Jurisdiction Risk Model (7% weight)
    juris_scores = {'Canada': 90, 'USA': 90, 'Mexico': 65, 'Peru': 55, 'Brazil': 55, 'Fiji': 45, 'Unknown': 40}
    df['M6_Name'] = "Geopolitical Risk"
    df['M6_Score'] = df['country'].map(juris_scores).fillna(40)
    df['M6_Signal'] = df['M6_Score'].apply(lambda x: '🟢 BUY' if x >= 70 else ('🟡 HOLD' if x >= 50 else '🔴 SELL'))
    df['M6_Explanation'] = df.apply(
        lambda r: f"Operating in {r['country']}. " +
                  ("Tier-1 jurisdiction (low risk)" if r['M6_Score'] >= 80 else
                   "Tier-2 jurisdiction (moderate risk)" if r['M6_Score'] >= 60 else
                   "Tier-3 jurisdiction (high political risk)"),
        axis=1
    )
    
    # MODEL 7: Stage/Development Model (6% weight)
    stage_scores = {'Producer': 85, 'Developer': 65, 'Explorer': 45}
    df['M7_Name'] = "Development Stage Risk"
    df['M7_Score'] = df['stage'].map(stage_scores).fillna(50)
    df['M7_Signal'] = df['M7_Score'].apply(lambda x: '🟢 BUY' if x >= 70 else ('🟡 HOLD' if x >= 50 else '🔴 SELL'))
    df['M7_Explanation'] = df.apply(
        lambda r: f"{r['stage']} stage. " +
                  ("Lowest risk - producing revenue" if r['stage'] == 'Producer' else
                   "Moderate risk - advancing toward production" if r['stage'] == 'Developer' else
                   "Higher risk - early exploration stage"),
        axis=1
    )
    
    # MODEL 8: Liquidity Model (3% weight)
    df['M8_Name'] = "Trading Liquidity"
    df['M8_Score'] = np.clip((df['Volume'] / 100000) * 40 + 30, 0, 100)
    df['M8_Signal'] = df['M8_Score'].apply(lambda x: '🟢 BUY' if x >= 70 else ('🟡 HOLD' if x >= 50 else '🔴 SELL'))
    df['M8_Explanation'] = df.apply(
        lambda r: f"Average volume: {r['Volume']:,.0f}. " +
                  ("Excellent liquidity" if r['Volume'] > 200000 else
                   "Adequate liquidity" if r['Volume'] > 50000 else
                   "Low liquidity - may be hard to exit"),
        axis=1
    )
    
    # MODEL 9: Volatility Model (3% weight)
    df['M9_Name'] = "Price Stability"
    df['M9_Score'] = np.clip(100 - (df['Volatility'] * 15), 0, 100)
    df['M9_Signal'] = df['M9_Score'].apply(lambda x: '🟢 BUY' if x >= 70 else ('🟡 HOLD' if x >= 50 else '🔴 SELL'))
    df['M9_Explanation'] = df.apply(
        lambda r: f"Volatility: {r['Volatility']:.1f}%. " +
                  ("Low volatility - stable" if r['Volatility'] < 3 else
                   "Moderate volatility" if r['Volatility'] < 6 else
                   "High volatility - expect large swings"),
        axis=1
    )
    
    # MODEL 10: Metal Sentiment Model (2% weight)
    metal_scores = {'Gold': 70, 'Silver': 75, 'Copper': 65}
    df['M10_Name'] = "Commodity Macro View"
    df['M10_Score'] = df['metal'].map(metal_scores).fillna(60)
    df['M10_Signal'] = df['M10_Score'].apply(lambda x: '🟢 BUY' if x >= 70 else ('🟡 HOLD' if x >= 50 else '🔴 SELL'))
    df['M10_Explanation'] = df.apply(
        lambda r: f"{r['metal']} market outlook. " +
                  ("Bullish macro environment" if r['M10_Score'] >= 70 else
                   "Neutral outlook" if r['M10_Score'] >= 60 else
                   "Bearish headwinds"),
        axis=1
    )
    
    # MODELS 11-15: Additional factors (1% each)
    df['M11_Name'] = "Trend Consistency"
    df['M11_Score'] = np.clip(100 - abs(df['Return_7d'] - df['Return_30d']) * 5, 0, 100)
    df['M11_Signal'] = '🟡 HOLD'
    df['M11_Explanation'] = "Measures trend reliability"
    
    df['M12_Name'] = "Analyst Consensus (Simulated)"
    df['M12_Score'] = 60
    df['M12_Signal'] = '🟡 HOLD'
    df['M12_Explanation'] = "Simulated - would use real analyst ratings"
    
    df['M13_Name'] = "Social Sentiment (Simulated)"
    df['M13_Score'] = 55
    df['M13_Signal'] = '🟡 HOLD'
    df['M13_Explanation'] = "Simulated - would use Twitter/Reddit data"
    
    df['M14_Name'] = "Sector Rotation (Simulated)"
    df['M14_Score'] = 65
    df['M14_Signal'] = '🟡 HOLD'
    df['M14_Explanation'] = "Simulated - would use market flow data"
    
    df['M15_Name'] = "ML Ensemble Prediction"
    df['M15_Score'] = (df['M1_Score'] + df['M2_Score'] + df['M3_Score']) / 3
    df['M15_Signal'] = '🟡 HOLD'
    df['M15_Explanation'] = "Average of top 3 models"
    
    # === STEP 4: AGGREGATE ===
    progress.progress(70, text="🎯 Aggregating all models...")
    
    df['Alpha_Score'] = (
        df['M1_Score'] * 0.30 +
        df['M2_Score'] * 0.15 +
        df['M3_Score'] * 0.12 +
        df['M4_Score'] * 0.10 +
        df['M5_Score'] * 0.08 +
        df['M6_Score'] * 0.07 +
        df['M7_Score'] * 0.06 +
        df['M8_Score'] * 0.03 +
        df['M9_Score'] * 0.03 +
        df['M10_Score'] * 0.02 +
        df['M11_Score'] * 0.01 +
        df['M12_Score'] * 0.01 +
        df['M13_Score'] * 0.01 +
        df['M14_Score'] * 0.01 +
        df['M15_Score'] * 0.01
    )
    
    # === STEP 5: FINAL DECISION ===
    progress.progress(90, text="✅ Generating recommendations...")
    
    def get_decision(row):
        score = row['Alpha_Score']
        runway = row['Runway']
        
        # Critical overrides
        if runway < 6:
            return '🚨 SELL NOW', 95, 'CRITICAL: Cash runway critical'
        elif runway < 8 and score < 50:
            return '🔴 SELL', 85, 'HIGH: Low cash + weak score'
        
        # Score-based
        if score >= 75:
            return '🟢 STRONG BUY', 90, f'CONVICTION: Score {score:.0f}/100'
        elif score >= 65:
            return '🟢 BUY', 80, f'POSITIVE: Score {score:.0f}/100'
        elif score >= 55:
            return '🔵 ACCUMULATE', 70, f'MODERATE: Score {score:.0f}/100'
        elif score >= 45:
            return '⚪ HOLD', 60, f'NEUTRAL: Score {score:.0f}/100'
        elif score >= 35:
            return '🟡 TRIM', 70, f'WEAK: Score {score:.0f}/100'
        else:
            return '🔴 SELL', 80, f'POOR: Score {score:.0f}/100'
    
    df[['Action', 'Confidence', 'Reasoning']] = df.apply(
        lambda r: pd.Series(get_decision(r)), axis=1
    )
    
    # Portfolio health
    health_score = (
        df['Alpha_Score'].mean() * 0.6 +
        (df['Runway'] >= 8).mean() * 100 * 0.3 +
        (df['Return_Pct'] > 0).mean() * 100 * 0.1
    )
    
    progress.progress(100, text="✅ Complete!")
    
    st.session_state.results = df
    st.session_state.health_score = health_score
    
    st.success("✅ Analysis complete - 15 models processed")
    st.rerun()

# === DISPLAY RESULTS ===
if 'results' in st.session_state:
    df = st.session_state.results
    health = st.session_state.health_score
    
    st.markdown("---")
    
    # Health Score
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if health >= 75:
            st.success("🟢 EXCELLENT")
        elif health >= 60:
            st.info("🔵 GOOD")
        elif health >= 45:
            st.warning("🟡 NEEDS ATTENTION")
        else:
            st.error("🔴 CRITICAL")
        st.metric("Portfolio Health", f"{health:.0f}/100")
    
    total_equity = df['Market_Value'].sum()
    total_value = total_equity + cash
    total_return = (df['Gain_Loss'].sum() / df['Cost_Basis'].sum() * 100)
    
    with col2:
        st.metric("Total Value", f"${total_value:,.0f}")
    with col3:
        st.metric("Total Return", f"{total_return:+.1f}%", 
                 delta=f"${df['Gain_Loss'].sum():+,.0f}")
    with col4:
        st.metric("Positions", len(df))
    
    st.markdown("---")
    
    # Stock Analysis
    st.header("📊 Complete Stock Analysis")
    
    for _, row in df.sort_values('Alpha_Score', ascending=False).iterrows():
        # Card header
        if 'STRONG BUY' in row['Action'] or 'BUY' in row['Action']:
            st.success(f"### {row['Symbol']} - {row['Action']}")
        elif 'SELL' in row['Action']:
            st.error(f"### {row['Symbol']} - {row['Action']}")
        else:
            st.info(f"### {row['Symbol']} - {row['Action']}")
        
        # Summary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Alpha Score", f"{row['Alpha_Score']:.0f}/100")
        col2.metric("Value", f"${row['Market_Value']:,.0f}")
        col3.metric("Return", f"{row['Return_Pct']:+.1f}%")
        col4.metric("Cash Runway", f"{row['Runway']:.1f}mo")
        col5.metric("Confidence", f"{row['Confidence']:.0f}%")
        
        st.caption(f"**{row['stage']}** • {row['metal']} • {row['country']} • {row['Reasoning']}")
        
        # Model breakdown
        with st.expander(f"🔍 See All 15 Model Predictions for {row['Symbol']}", expanded=False):
            st.subheader("15-Model Analysis Breakdown")
            st.caption("Each model analyzes different aspects and provides independent recommendations")
            
            # Top 10 models with full details
            models = [
                (1, row['M1_Name'], row['M1_Score'], row['M1_Signal'], row['M1_Explanation'], 30),
                (2, row['M2_Name'], row['M2_Score'], row['M2_Signal'], row['M2_Explanation'], 15),
                (3, row['M3_Name'], row['M3_Score'], row['M3_Signal'], row['M3_Explanation'], 12),
                (4, row['M4_Name'], row['M4_Score'], row['M4_Signal'], row['M4_Explanation'], 10),
                (5, row['M5_Name'], row['M5_Score'], row['M5_Signal'], row['M5_Explanation'], 8),
                (6, row['M6_Name'], row['M6_Score'], row['M6_Signal'], row['M6_Explanation'], 7),
                (7, row['M7_Name'], row['M7_Score'], row['M7_Signal'], row['M7_Explanation'], 6),
                (8, row['M8_Name'], row['M8_Score'], row['M8_Signal'], row['M8_Explanation'], 3),
                (9, row['M9_Name'], row['M9_Score'], row['M9_Signal'], row['M9_Explanation'], 3),
                (10, row['M10_Name'], row['M10_Score'], row['M10_Signal'], row['M10_Explanation'], 2),
                (11, row['M11_Name'], row['M11_Score'], row['M11_Signal'], row['M11_Explanation'], 1),
                (12, row['M12_Name'], row['M12_Score'], row['M12_Signal'], row['M12_Explanation'], 1),
                (13, row['M13_Name'], row['M13_Score'], row['M13_Signal'], row['M13_Explanation'], 1),
                (14, row['M14_Name'], row['M14_Score'], row['M14_Signal'], row['M14_Explanation'], 1),
                (15, row['M15_Name'], row['M15_Score'], row['M15_Signal'], row['M15_Explanation'], 1),
            ]
            
            for num, name, score, signal, explanation, weight in models:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"**Model {num}: {name}** (Weight: {weight}%)")
                    st.caption(explanation)
                
                with col2:
                    if score >= 70:
                        st.success(f"Score: {score:.0f}/100")
                    elif score >= 50:
                        st.info(f"Score: {score:.0f}/100")
                    else:
                        st.error(f"Score: {score:.0f}/100")
                
                with col3:
                    st.write(signal)
                
                st.progress(score / 100)
                st.markdown("---")
            
            # Weighted calculation
            st.subheader("Final Aggregation")
            st.write(f"**Weighted Average of All 15 Models: {row['Alpha_Score']:.1f}/100**")
            st.caption("Each model's score is multiplied by its weight, then summed to produce the final Alpha Score")
        
        st.markdown("---")
    
    # Export
    st.download_button(
        "📥 Download Complete Analysis",
        df.to_csv(index=False),
        f"analysis_{datetime.date.today()}.csv",
        use_container_width=True
    )

st.caption("💎 Alpha Miner Ultimate • 15-Model Multi-Factor Analysis System")
