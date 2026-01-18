#!/usr/bin/env python3
"""
ALPHA MINER WEB INTERFACE
Simple web dashboard for portfolio analysis

INSTALL:
    pip install streamlit --break-system-packages

RUN:
    streamlit run alpha_miner_web.py

Then open browser to: http://localhost:8501
"""

import streamlit as st
import pandas as pd
import datetime
from io import StringIO

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Alpha Miner Ultimate",
    page_icon="💎",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:20px !important;
    font-weight: bold;
}
.critical-alert {
    background-color: #ff4444;
    color: white;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}
.buy-signal {
    background-color: #00cc66;
    color: white;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}
.hold-signal {
    background-color: #666666;
    color: white;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("💎 Alpha Miner Ultimate")
st.subheader("Complete Capital Allocation System for Junior Miners")
st.caption("Philosophy: SURVIVE FIRST, ALPHA SECOND")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.markdown("### Survival Gate Thresholds")
    min_cash_runway = st.slider("Min Cash Runway (months)", 4, 12, 8)
    max_position_size = st.slider("Max Position Size (%)", 10, 20, 15)
    
    st.markdown("### Recent Financings")
    st.caption("Add recent financings to adjust cash runway")
    
    if 'financings' not in st.session_state:
        st.session_state.financings = {
            'LOMLF': {'amount': 8.0, 'date': 'Dec 2025/Jan 2026', 'source': 'Arete Capital'},
            'EXNRF': {'amount': 15.0, 'date': 'Q4 2025', 'source': 'Credit facility'}
        }
    
    new_symbol = st.text_input("Stock Symbol")
    new_amount = st.number_input("Amount Raised ($M)", 0.0, 100.0, 5.0)
    new_date = st.text_input("Date", "Jan 2026")
    new_source = st.text_input("Source", "Private placement")
    
    if st.button("Add Financing"):
        if new_symbol:
            st.session_state.financings[new_symbol] = {
                'amount': new_amount,
                'date': new_date,
                'source': new_source
            }
            st.success(f"Added financing for {new_symbol}")
    
    if st.session_state.financings:
        st.markdown("### Current Financings:")
        for sym, data in st.session_state.financings.items():
            st.caption(f"**{sym}**: ${data['amount']}M ({data['date']})")

# Main area tabs
tab1, tab2, tab3 = st.tabs(["📊 Portfolio Analysis", "📈 Quick Update", "📝 Manual Entry"])

with tab1:
    st.header("Upload Portfolio")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload your portfolio CSV",
        type=['csv'],
        help="CSV should have: Symbol, Quantity, CostBasis, Stage, Metal, Country, Cash, MonthlyBurn"
    )
    
    # Sample CSV download
    sample_csv = """Symbol,Quantity,CostBasis,Stage,Metal,Country,Cash,MonthlyBurn
ITRG,2072,7838.13,Explorer,Gold,Canada,12,0.8
AGXPF,32342,24015.26,Explorer,Silver,Peru,18,1.5
WRLGF,25929,18833.21,Explorer,Gold,Canada,10,1.0"""
    
    st.download_button(
        label="📥 Download Sample CSV",
        data=sample_csv,
        file_name="portfolio_sample.csv",
        mime="text/csv"
    )
    
    if uploaded_file is not None:
        # Load CSV
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.lower()
        
        # Handle column names
        mappings = {
            'costbasis': 'cost_basis',
            'qty': 'quantity',
            'ticker': 'symbol'
        }
        for old, new in mappings.items():
            if old in df.columns and new not in df.columns:
                df.rename(columns={old: new}, inplace=True)
        
        st.success(f"✅ Loaded {len(df)} positions")
        
        # Show portfolio
        with st.expander("📋 View Portfolio Data"):
            st.dataframe(df)
        
        # Analyze button
        if st.button("🚀 Run Analysis", type="primary"):
            with st.spinner("Fetching prices and analyzing..."):
                # Get prices
                if YFINANCE_AVAILABLE:
                    for idx, row in df.iterrows():
                        try:
                            ticker = yf.Ticker(row['symbol'])
                            hist = ticker.history(period="5d")
                            if not hist.empty:
                                df.at[idx, 'current_price'] = hist['Close'].iloc[-1]
                                prev = hist['Close'].iloc[-2] if len(hist) > 1 else hist['Close'].iloc[-1]
                                df.at[idx, 'day_change_pct'] = ((hist['Close'].iloc[-1] - prev) / prev * 100)
                        except:
                            df.at[idx, 'current_price'] = 0
                            df.at[idx, 'day_change_pct'] = 0
                else:
                    st.warning("⚠️ yfinance not installed. Using dummy prices.")
                    df['current_price'] = 1.0
                    df['day_change_pct'] = 0.0
                
                # Calculate values
                df['market_value'] = df['quantity'] * df['current_price']
                df['gain_loss'] = df['market_value'] - df['cost_basis']
                df['gain_loss_pct'] = (df['gain_loss'] / df['cost_basis'] * 100).fillna(0)
                total_mv = df['market_value'].sum()
                df['pct_portfolio'] = (df['market_value'] / total_mv * 100).fillna(0)
                
                # Calculate cash runway with pro-forma adjustments
                df['pro_forma_cash'] = df['cash']
                df['financing_note'] = ""
                
                for idx, row in df.iterrows():
                    symbol = row['symbol']
                    if symbol in st.session_state.financings:
                        financing = st.session_state.financings[symbol]
                        df.at[idx, 'pro_forma_cash'] = row['cash'] + financing['amount']
                        df.at[idx, 'financing_note'] = f"+${financing['amount']}M ({financing['date']})"
                
                df['runway_months'] = df['pro_forma_cash'] / df['monthly_burn']
                
                # Survival gates
                df['cash_gate'] = df['runway_months'] >= min_cash_runway
                df['size_gate'] = df['pct_portfolio'] <= max_position_size
                df['survival_passed'] = df['cash_gate'] & df['size_gate']
                
                # Alpha scoring
                df['alpha_score'] = 50.0
                df.loc[df['gain_loss_pct'] > 100, 'alpha_score'] += 15
                df.loc[df['gain_loss_pct'] > 50, 'alpha_score'] += 10
                df.loc[df['gain_loss_pct'] < -20, 'alpha_score'] -= 15
                df.loc[df['day_change_pct'] > 5, 'alpha_score'] += 10
                df.loc[df['day_change_pct'] < -5, 'alpha_score'] -= 10
                
                # Recommendations
                def get_recommendation(row):
                    if not row['survival_passed']:
                        if not row['cash_gate']:
                            return '🚨 SELL', 'CRITICAL - Cash runway'
                        else:
                            return '🟡 TRIM', 'Over-concentrated'
                    elif row['alpha_score'] >= 75:
                        return '🟢 STRONG BUY', 'Excellent setup'
                    elif row['alpha_score'] >= 60:
                        return '🔵 BUY', 'Good opportunity'
                    elif row['alpha_score'] >= 45:
                        return '⚪ HOLD', 'Neutral'
                    else:
                        return '🟡 REDUCE', 'Weakening'
                
                df[['recommendation', 'reason']] = df.apply(
                    lambda row: pd.Series(get_recommendation(row)), axis=1
                )
                
                # Store in session
                st.session_state.results = df
                st.success("✅ Analysis complete!")
        
        # Display results
        if 'results' in st.session_state:
            df = st.session_state.results
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_value = df['market_value'].sum()
            total_gain = df['gain_loss'].sum()
            total_return_pct = (total_gain / df['cost_basis'].sum() * 100)
            critical_issues = len(df[~df['cash_gate']])
            
            with col1:
                st.metric("Total Value", f"${total_value:,.0f}")
            with col2:
                st.metric("Total Return", f"${total_gain:+,.0f}", f"{total_return_pct:+.1f}%")
            with col3:
                st.metric("Positions", len(df))
            with col4:
                st.metric("Critical Alerts", critical_issues, delta_color="inverse")
            
            st.markdown("---")
            
            # Critical alerts
            critical = df[~df['cash_gate']]
            if len(critical) > 0:
                st.markdown("## 🚨 CRITICAL ALERTS - Immediate Action Required")
                
                for _, row in critical.iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div class="critical-alert">
                        <strong>{row['symbol']}</strong> - {row['recommendation']} | 
                        Cash Runway: {row['runway_months']:.1f} months 
                        {row['financing_note']}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Top opportunities
            buys = df[df['recommendation'].str.contains('BUY')].sort_values('alpha_score', ascending=False)
            if len(buys) > 0:
                st.markdown("## 🟢 Top Opportunities")
                
                for _, row in buys.head(5).iterrows():
                    with st.container():
                        st.markdown(f"""
                        <div class="buy-signal">
                        <strong>{row['symbol']}</strong> - {row['recommendation']} | 
                        Alpha Score: {row['alpha_score']:.0f}/100 | 
                        Return: {row['gain_loss_pct']:+.1f}% | 
                        Runway: {row['runway_months']:.1f} months
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Complete holdings table
            st.markdown("## 📊 Complete Holdings")
            
            display_df = df[[
                'symbol', 'recommendation', 'alpha_score', 'market_value', 
                'gain_loss_pct', 'day_change_pct', 'runway_months', 
                'pct_portfolio'
            ]].copy()
            
            display_df.columns = [
                'Symbol', 'Action', 'Alpha', 'Value', 'Return %', 
                'Day %', 'Runway', '% Port'
            ]
            
            st.dataframe(
                display_df.style.format({
                    'Value': '${:,.0f}',
                    'Return %': '{:+.1f}%',
                    'Day %': '{:+.2f}%',
                    'Alpha': '{:.0f}',
                    'Runway': '{:.1f}',
                    '% Port': '{:.1f}%'
                }),
                use_container_width=True
            )
            
            # Export button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Report (CSV)",
                data=csv,
                file_name=f"alpha_report_{datetime.date.today()}.csv",
                mime="text/csv"
            )

with tab2:
    st.header("📈 Quick Price Update")
    st.caption("Quickly update prices without re-uploading CSV")
    
    if 'results' in st.session_state:
        df = st.session_state.results
        
        col1, col2 = st.columns(2)
        
        with col1:
            symbol = st.selectbox("Select Stock", df['symbol'].tolist())
        
        with col2:
            if st.button("🔄 Refresh Price"):
                if YFINANCE_AVAILABLE:
                    try:
                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(period="1d")
                        if not hist.empty:
                            new_price = hist['Close'].iloc[-1]
                            st.success(f"Current price: ${new_price:.4f}")
                    except Exception as e:
                        st.error(f"Error: {e}")
                else:
                    st.warning("yfinance not installed")
    else:
        st.info("Upload and analyze a portfolio first")

with tab3:
    st.header("📝 Manual Data Entry")
    st.caption("Update cash runway and financing data")
    
    if 'results' in st.session_state:
        df = st.session_state.results
        
        symbol = st.selectbox("Select Stock to Update", df['symbol'].tolist(), key='manual_symbol')
        
        current_data = df[df['symbol'] == symbol].iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Current Data")
            st.write(f"**Cash:** ${current_data['cash']:.1f}M")
            st.write(f"**Monthly Burn:** ${current_data['monthly_burn']:.1f}M")
            st.write(f"**Runway:** {current_data['runway_months']:.1f} months")
        
        with col2:
            st.markdown("### Update")
            new_cash = st.number_input("Cash ($M)", 0.0, 100.0, float(current_data['cash']))
            new_burn = st.number_input("Monthly Burn ($M)", 0.1, 10.0, float(current_data['monthly_burn']))
            
            if st.button("Update Data"):
                df.loc[df['symbol'] == symbol, 'cash'] = new_cash
                df.loc[df['symbol'] == symbol, 'monthly_burn'] = new_burn
                st.session_state.results = df
                st.success("✅ Updated! Re-run analysis to see changes.")
    else:
        st.info("Upload and analyze a portfolio first")

# Footer
st.markdown("---")
st.caption("💎 Alpha Miner Ultimate | Philosophy: Survive First, Alpha Second | Data: 15-min delayed (free)")
