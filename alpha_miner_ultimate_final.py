#!/usr/bin/env python3
"""
ALPHA MINER ULTIMATE - ITERATION 2
Improvements: Visual indicators, urgency grouping, portfolio health score, capital allocation
"""
import streamlit as st
import pandas as pd
import datetime

try:
    import yfinance as yf
    YFINANCE = True
except:
    YFINANCE = False

st.set_page_config(page_title="Alpha Miner", layout="wide", initial_sidebar_state="collapsed")

# Enhanced CSS
st.markdown("""
<style>
    .health-excellent {background: linear-gradient(135deg, #00cc66 0%, #00994d 100%); color: white; padding: 2rem; border-radius: 15px; text-align: center;}
    .health-good {background: linear-gradient(135deg, #66cc00 0%, #4d9900 100%); color: white; padding: 2rem; border-radius: 15px; text-align: center;}
    .health-warning {background: linear-gradient(135deg, #ff9900 0%, #cc7700 100%); color: white; padding: 2rem; border-radius: 15px; text-align: center;}
    .health-critical {background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%); color: white; padding: 2rem; border-radius: 15px; text-align: center;}
    .urgency-today {border-left: 6px solid #ff4444; padding: 1.5rem; margin: 1rem 0; background: #fff5f5;}
    .urgency-week {border-left: 6px solid #00cc66; padding: 1.5rem; margin: 1rem 0; background: #f0fff4;}
    .urgency-monitor {border-left: 6px solid #666; padding: 1.5rem; margin: 1rem 0; background: #f8f9fa;}
    .traffic-light {display: inline-block; width: 20px; height: 20px; border-radius: 50%; margin: 0 5px;}
    .light-green {background: #00cc66;}
    .light-yellow {background: #ffcc00;}
    .light-red {background: #ff4444;}
    .capital-box {background: #e3f2fd; padding: 1.5rem; border-radius: 10px; border: 2px solid #2196f3;}
    .metric-large {font-size: 3rem; font-weight: bold; margin: 0;}
    .metric-label {font-size: 1rem; color: #666; margin: 0;}
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

FINANCINGS = {
    'LOMLF': {'amount': 8.0, 'date': 'Jan 2026'},
    'EXNRF': {'amount': 15.0, 'date': 'Q4 2025'}
}

# HEADER
st.title("💎 Alpha Miner Ultimate")
st.caption("Complete Capital Allocation System • Survive First, Alpha Second")

# Portfolio input
st.subheader("📊 Your Portfolio")
col1, col2 = st.columns([4, 1])

with col1:
    edited_df = st.data_editor(
        st.session_state.portfolio,
        use_container_width=True,
        num_rows="dynamic",
        hide_index=True
    )
    st.session_state.portfolio = edited_df

with col2:
    st.markdown("**💰 Cash**")
    cash = st.number_input("Available", value=float(st.session_state.cash), step=1000.0, label_visibility="collapsed")
    st.session_state.cash = cash

# Analyze button
if st.button("🚀 ANALYZE PORTFOLIO", type="primary", use_container_width=True):
    # Progress indicator
    progress = st.progress(0)
    status = st.empty()
    
    status.text("⏳ Fetching live prices...")
    progress.progress(20)
    
    df = edited_df.copy()
    
    # Get prices
    for idx, row in df.iterrows():
        if YFINANCE:
            try:
                ticker = yf.Ticker(row['Symbol'])
                hist = ticker.history(period="5d")
                if not hist.empty:
                    df.at[idx, 'Price'] = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-2] if len(hist) > 1 else hist['Close'].iloc[-1]
                    df.at[idx, 'DayChange'] = ((hist['Close'].iloc[-1] - prev) / prev * 100)
            except:
                df.at[idx, 'Price'] = 0
                df.at[idx, 'DayChange'] = 0
    
    status.text("🔍 Running survival gates...")
    progress.progress(40)
    
    # Add data
    for idx, row in df.iterrows():
        if row['Symbol'] in COMPANY_DATA:
            for k, v in COMPANY_DATA[row['Symbol']].items():
                df.at[idx, k] = v
    
    # Calculate
    df['Market_Value'] = df['Quantity'] * df['Price']
    df['Gain_Loss'] = df['Market_Value'] - df['Cost_Basis']
    df['Return_Pct'] = (df['Gain_Loss'] / df['Cost_Basis'] * 100)
    
    total_mv = df['Market_Value'].sum()
    df['Pct_Portfolio'] = (df['Market_Value'] / total_mv * 100)
    
    status.text("💰 Calculating cash runways...")
    progress.progress(60)
    
    # Cash runway
    df['Pro_Forma_Cash'] = df.apply(
        lambda r: r['cash'] + FINANCINGS[r['Symbol']]['amount'] if r['Symbol'] in FINANCINGS else r['cash'],
        axis=1
    )
    df['Runway'] = df['Pro_Forma_Cash'] / df['burn']
    
    # Gates
    df['Gate_Cash'] = df['Runway'] >= 8
    df['Gate_Size'] = df['Pct_Portfolio'] <= 15
    df['Gate_Perf'] = df['Return_Pct'] > -40
    JURIS = {'Canada': 10, 'USA': 10, 'Mexico': 40, 'Peru': 45, 'Brazil': 45, 'Fiji': 50, 'Unknown': 50}
    df['Juris_Risk'] = df['country'].map(JURIS)
    df['Gate_Juris'] = df['Juris_Risk'] <= 60
    df['Survival'] = df['Gate_Cash'] & df['Gate_Size'] & df['Gate_Perf'] & df['Gate_Juris']
    
    status.text("📊 Calculating alpha scores...")
    progress.progress(80)
    
    # Narrative & Alpha
    def get_narrative(ret):
        if ret < -40: return "⚫ DEAD"
        elif ret < 10: return "⚪ IGNORED"
        elif ret < 50: return "🟢 EMERGING"
        elif ret < 150: return "🔵 VALIDATION"
        else: return "🟡 CROWDED"
    
    df['Narrative'] = df['Return_Pct'].apply(get_narrative)
    
    df['Alpha'] = 50.0
    df.loc[df['Return_Pct'] > 100, 'Alpha'] += 15
    df.loc[df['Return_Pct'] > 50, 'Alpha'] += 10
    df.loc[df['Return_Pct'] < -20, 'Alpha'] -= 15
    df.loc[df['Narrative'] == '🟢 EMERGING', 'Alpha'] += 20
    df.loc[df['Narrative'] == '🔵 VALIDATION', 'Alpha'] += 10
    
    # Decision
    def get_decision(row):
        if not row['Survival']:
            if not row['Gate_Cash']:
                return '🚨 SELL NOW', 1.0, 'TODAY', f"Cash: {row['Runway']:.1f}mo"
            else:
                return '🟡 REDUCE', 0.8, 'THIS_WEEK', 'Survival concerns'
        elif row['Alpha'] >= 75:
            return '🟢 STRONG BUY', 0.85, 'THIS_WEEK', f"{row['Narrative']}"
        elif row['Alpha'] >= 60:
            return '🔵 BUY', 0.75, 'THIS_WEEK', f"{row['Narrative']}"
        elif row['Alpha'] >= 45:
            return '⚪ HOLD', 0.60, 'MONITOR', 'Neutral'
        else:
            return '🟡 TRIM', 0.70, 'MONITOR', 'Weakening'
    
    df[['Action', 'Confidence', 'Urgency', 'Reason']] = df.apply(
        lambda r: pd.Series(get_decision(r)), axis=1
    )
    
    # Portfolio health score
    health_score = (
        (df['Survival'].sum() / len(df)) * 40 +  # 40% weight on survival
        (df['Alpha'].mean() / 100) * 30 +  # 30% weight on alpha
        ((df['Return_Pct'] > 0).sum() / len(df)) * 30  # 30% weight on winners
    ) * 100
    
    status.text("✅ Analysis complete!")
    progress.progress(100)
    
    st.session_state.results = df
    st.session_state.health_score = health_score
    st.session_state.analysis_time = datetime.datetime.now()
    
    st.success("✅ Analysis complete!")
    st.rerun()

# RESULTS
if 'results' in st.session_state:
    df = st.session_state.results
    health_score = st.session_state.health_score
    
    st.markdown("---")
    
    # PORTFOLIO HEALTH SCORE
    if health_score >= 80:
        health_class = "health-excellent"
        health_status = "🟢 EXCELLENT"
        health_msg = "Your portfolio is in great shape!"
    elif health_score >= 60:
        health_class = "health-good"
        health_status = "🟡 GOOD"
        health_msg = "Portfolio is healthy with some areas to watch"
    elif health_score >= 40:
        health_class = "health-warning"
        health_status = "🟠 WARNING"
        health_msg = "Several concerns need attention"
    else:
        health_class = "health-critical"
        health_status = "🔴 CRITICAL"
        health_msg = "Immediate action required"
    
    st.markdown(f"""
    <div class="{health_class}">
        <div class="metric-label">PORTFOLIO HEALTH SCORE</div>
        <div class="metric-large">{health_score:.0f}/100</div>
        <div style="font-size: 1.5rem; margin-top: 1rem;">{health_status}</div>
        <div style="margin-top: 0.5rem;">{health_msg}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Quick metrics
    total_equity = df['Market_Value'].sum()
    total_value = total_equity + cash
    total_gain = df['Gain_Loss'].sum()
    total_return = (total_gain / df['Cost_Basis'].sum() * 100)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Value", f"${total_value:,.0f}")
    col2.metric("Equity", f"${total_equity:,.0f}", f"{total_return:+.1f}%")
    col3.metric("Cash", f"${cash:,.0f}", f"{cash/total_value*100:.1f}%")
    col4.metric("Positions", len(df))
    
    # Traffic lights
    sells = df[df['Urgency'] == 'TODAY']
    buys = df[df['Action'].str.contains('BUY')]
    issues = df[~df['Survival']]
    
    st.markdown(f"""
    <div style="padding: 1rem; background: #f8f9fa; border-radius: 10px; margin: 1rem 0;">
        <strong>Quick Status:</strong>
        <span class="traffic-light {'light-red' if len(sells) > 0 else 'light-green'}"></span>
        <span>{len(sells)} Critical Actions</span>
        <span class="traffic-light {'light-green' if len(buys) > 0 else 'light-yellow'}"></span>
        <span>{len(buys)} Buy Opportunities</span>
        <span class="traffic-light {'light-red' if len(issues) > 0 else 'light-green'}"></span>
        <span>{len(issues)} Survival Concerns</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # TODAY (Critical)
    today_actions = df[df['Urgency'] == 'TODAY']
    if len(today_actions) > 0:
        st.markdown("## 🚨 DO TODAY")
        st.error("Critical issues requiring immediate action")
        
        for _, row in today_actions.iterrows():
            st.markdown(f"""
            <div class="urgency-today">
                <strong style="font-size: 1.3rem;">{row['Symbol']} - {row['Action']}</strong><br>
                <strong>Value:</strong> ${row['Market_Value']:,.0f} • <strong>Return:</strong> {row['Return_Pct']:+.1f}%<br>
                <strong>Issue:</strong> {row['Reason']}<br>
                <strong>Confidence:</strong> {row['Confidence']:.0%}
            </div>
            """, unsafe_allow_html=True)
    
    # THIS WEEK (Opportunities)
    week_actions = df[df['Urgency'] == 'THIS_WEEK']
    if len(week_actions) > 0:
        st.markdown("## 🎯 THIS WEEK")
        
        # Capital allocation suggestions
        if cash > 10000:
            st.markdown(f"""
            <div class="capital-box">
                <strong style="font-size: 1.2rem;">💰 Capital Allocation Plan</strong><br>
                Available: ${cash:,.0f} • Suggested deployment: ${min(cash * 0.6, 30000):,.0f} (keep {cash * 0.4 / cash * 100:.0f}% reserve)
            </div>
            """, unsafe_allow_html=True)
        
        buys_week = week_actions[week_actions['Action'].str.contains('BUY')].sort_values('Alpha', ascending=False)
        
        for idx, row in buys_week.iterrows():
            suggested = min(cash * 0.15, 10000)
            shares = int(suggested / row['Price'])
            
            st.markdown(f"""
            <div class="urgency-week">
                <strong style="font-size: 1.3rem;">{row['Symbol']} - {row['Action']}</strong> • Alpha: {row['Alpha']:.0f}/100<br>
                <strong>Current:</strong> ${row['Market_Value']:,.0f} ({row['Pct_Portfolio']:.1f}%)<br>
                <strong>Return:</strong> {row['Return_Pct']:+.1f}% • <strong>Phase:</strong> {row['Narrative']}<br>
                <strong>💡 Suggested:</strong> Add ${suggested:,.0f} ({shares:,} shares @ ${row['Price']:.4f})<br>
                <strong>Runway:</strong> {row['Runway']:.1f} months ✅
            </div>
            """, unsafe_allow_html=True)
    
    # MONITOR
    monitor = df[df['Urgency'] == 'MONITOR']
    if len(monitor) > 0:
        with st.expander(f"📊 MONITOR ({len(monitor)} positions)"):
            for _, row in monitor.iterrows():
                st.write(f"**{row['Symbol']}** - {row['Action']} • {row['Reason']}")
    
    # Full table
    st.markdown("---")
    st.subheader("📊 Complete Portfolio")
    
    st.dataframe(
        df[['Symbol', 'Action', 'Alpha', 'Market_Value', 'Return_Pct', 'Runway', 'Urgency']].style.format({
            'Alpha': '{:.0f}',
            'Market_Value': '${:,.0f}',
            'Return_Pct': '{:+.1f}%',
            'Runway': '{:.1f}'
        }).background_gradient(subset=['Alpha'], cmap='RdYlGn'),
        use_container_width=True
    )
