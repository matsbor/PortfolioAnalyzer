#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import datetime

try:
    import yfinance as yf
    YFINANCE = True
except:
    YFINANCE = False

st.set_page_config(page_title="Alpha Miner", layout="wide")
st.title("💎 Alpha Miner - Complete System")

# Initialize
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame({
        'Symbol': ['JZRIF', 'SMAGF', 'ITRG', 'LOMLF', 'TSKFF', 'AGXPF', 'GSVRF', 
                   'EXNRF', 'WRLGF', 'JAGGF', 'BORMF', 'AG', 'LUCMF', 'SMDRF'],
        'Quantity': [19841, 8335, 2072, 24557, 13027, 32342, 9049, 
                     11749, 25929, 2965, 5172, 638, 7079, 7072],
        'Cost Basis': [5959.25, 9928.39, 7838.13, 5006.76, 13857.41, 24015.26, 2415.31,
                       3242.97, 18833.21, 14558.14, 5540.99, 7594.99, 8550.49, 6939.18]
    })

# Company data
COMPANY_DATA = {
    'ITRG': {'cash': 12, 'burn': 0.8}, 'SMAGF': {'cash': 8, 'burn': 0.7},
    'WRLGF': {'cash': 10, 'burn': 1.0}, 'EXNRF': {'cash': 20, 'burn': 0.6},
    'LOMLF': {'cash': 12, 'burn': 0.9}, 'GSVRF': {'cash': 20, 'burn': 1.5},
    'AGXPF': {'cash': 18, 'burn': 1.5}, 'TSKFF': {'cash': 15, 'burn': 1.2},
    'BORMF': {'cash': 7, 'burn': 0.6}, 'JAGGF': {'cash': 25, 'burn': 2.0},
    'LUCMF': {'cash': 6, 'burn': 0.8}, 'AG': {'cash': 50, 'burn': 3.0},
    'SMDRF': {'cash': 12, 'burn': 1.0}, 'JZRIF': {'cash': 10, 'burn': 0.5}
}

st.header("📊 Your Portfolio")
edited_df = st.data_editor(st.session_state.portfolio, use_container_width=True, num_rows="dynamic")
st.session_state.portfolio = edited_df

if st.button("🚀 Analyze Portfolio", type="primary", use_container_width=True):
    with st.spinner("Fetching prices..."):
        df = edited_df.copy()
        
        # Get prices
        if YFINANCE:
            for idx, row in df.iterrows():
                try:
                    ticker = yf.Ticker(row['Symbol'])
                    hist = ticker.history(period="2d")
                    if not hist.empty:
                        df.at[idx, 'Price'] = hist['Close'].iloc[-1]
                except:
                    df.at[idx, 'Price'] = 0
        
        # Calculate
        df['Market Value'] = df['Quantity'] * df['Price']
        df['Gain/Loss'] = df['Market Value'] - df['Cost Basis']
        df['Return %'] = (df['Gain/Loss'] / df['Cost Basis'] * 100)
        
        total_mv = df['Market Value'].sum()
        df['% Portfolio'] = (df['Market Value'] / total_mv * 100)
        
        # Cash runway
        df['Cash (M)'] = df['Symbol'].map(lambda x: COMPANY_DATA.get(x, {}).get('cash', 10))
        df['Burn (M)'] = df['Symbol'].map(lambda x: COMPANY_DATA.get(x, {}).get('burn', 1))
        df['Runway'] = df['Cash (M)'] / df['Burn (M)']
        
        # Alerts
        df['Alert'] = ''
        df.loc[df['Runway'] < 6, 'Alert'] = '🚨 LOW CASH'
        df.loc[df['Return %'] < -50, 'Alert'] = '🚨 MAJOR LOSS'
        df.loc[(df['Return %'] > 50) & (df['Runway'] > 8), 'Alert'] = '🟢 WINNER'
        
        st.session_state.results = df
    
    st.success("✅ Analysis complete!")

# Results
if 'results' in st.session_state:
    df = st.session_state.results
    
    st.markdown("---")
    
    total_value = df['Market Value'].sum()
    total_gain = df['Gain/Loss'].sum()
    total_return = (total_gain / df['Cost Basis'].sum() * 100)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Value", f"${total_value:,.0f}")
    col2.metric("Total Return", f"${total_gain:+,.0f}")
    col3.metric("Return %", f"{total_return:+.1f}%")
    col4.metric("Positions", len(df))
    
    # Alerts
    critical = df[df['Alert'].str.contains('🚨')]
    if len(critical) > 0:
        st.markdown("### 🚨 Critical Alerts")
        for _, row in critical.iterrows():
            st.error(f"{row['Symbol']}: {row['Alert']} - Runway: {row['Runway']:.1f} months, Return: {row['Return %']:+.0f}%")
    
    winners = df[df['Alert'].str.contains('🟢')]
    if len(winners) > 0:
        st.markdown("### 🟢 Winners")
        for _, row in winners.iterrows():
            st.success(f"{row['Symbol']}: {row['Return %']:+.1f}% (${row['Gain/Loss']:+,.0f})")
    
    st.markdown("### 📊 Complete Portfolio")
    st.dataframe(df[['Symbol', 'Quantity', 'Price', 'Market Value', 'Return %', 'Runway', 'Alert']]
                 .style.format({
                     'Price': '${:.4f}',
                     'Market Value': '${:,.0f}',
                     'Return %': '{:+.1f}%',
                     'Runway': '{:.1f}'
                 }), use_container_width=True, height=500)
