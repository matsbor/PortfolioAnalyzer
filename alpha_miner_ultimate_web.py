#!/usr/bin/env python3
"""
ALPHA MINER ULTIMATE - COMPLETE WEB INTERFACE
All features: Survival gates, truth arbitration, narrative detection, detailed analysis
"""
import streamlit as st
import pandas as pd
import datetime

try:
    import yfinance as yf
    YFINANCE = True
except:
    YFINANCE = False
    st.warning("⚠️ Install yfinance: pip install yfinance")

st.set_page_config(page_title="Alpha Miner Ultimate", layout="wide")
st.title("💎 Alpha Miner Ultimate - Complete System")
st.caption("Philosophy: SURVIVE FIRST, ALPHA SECOND")

# Initialize with REAL Schwab data
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = pd.DataFrame({
        'Symbol': ['JZRIF', 'ITRG', 'SMAGF', 'LOMLF', 'TSKFF', 'AGXPF', 'GSVRF', 
                   'EXNRF', 'WRLGF', 'JAGGF', 'BORMF', 'AG', 'LUCMF', 'SMDRF'],
        'Quantity': [19841, 2072, 8335, 24557, 13027, 32342, 9049, 
                     11749, 25929, 2965, 5172, 638, 7079, 7072],
        'Cost Basis': [5959.25, 7838.13, 9928.39, 5006.76, 13857.41, 24015.26, 2415.31,
                       3242.97, 18833.21, 14558.14, 5540.99, 7594.99, 8550.49, 6939.18]
    })

# Company fundamentals
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

# Recent financings (Pro-forma adjustments)
FINANCINGS = {
    'LOMLF': {'amount': 8.0, 'date': 'Jan 2026', 'source': 'Arete Capital'},
    'EXNRF': {'amount': 15.0, 'date': 'Q4 2025', 'source': 'Credit facility'}
}

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Analysis Settings")
    
    min_cash_runway = st.slider("Min Cash Runway (months)", 4, 12, 8)
    max_position_size = st.slider("Max Position Size (%)", 10, 20, 15)
    
    st.markdown("---")
    st.markdown("### 💰 Recent Financings")
    st.caption("Pro-forma cash adjustments")
    for sym, fin in FINANCINGS.items():
        st.caption(f"**{sym}**: +${fin['amount']}M ({fin['date']})")
    
    st.markdown("---")
    if st.button("📊 Update Company Data"):
        st.info("Feature coming: Edit cash/burn rates")

# Portfolio editor
st.header("📊 Step 1: Your Portfolio")
st.caption("Edit quantities/cost basis from your latest Schwab data")

edited_df = st.data_editor(
    st.session_state.portfolio,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "Symbol": st.column_config.TextColumn("Symbol", width="small"),
        "Quantity": st.column_config.NumberColumn("Quantity", format="%d"),
        "Cost Basis": st.column_config.NumberColumn("Cost Basis $", format="%.2f"),
    }
)

st.session_state.portfolio = edited_df

# Analysis button
if st.button("🚀 RUN COMPLETE ANALYSIS", type="primary", use_container_width=True):
    with st.spinner("Running complete analysis with survival gates..."):
        df = edited_df.copy()
        
        # Get live prices
        for idx, row in df.iterrows():
            symbol = row['Symbol']
            if YFINANCE:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="5d")
                    if not hist.empty:
                        df.at[idx, 'Price'] = hist['Close'].iloc[-1]
                        prev = hist['Close'].iloc[-2] if len(hist) > 1 else hist['Close'].iloc[-1]
                        df.at[idx, 'DayChange%'] = ((hist['Close'].iloc[-1] - prev) / prev * 100)
                    else:
                        df.at[idx, 'Price'] = 0
                        df.at[idx, 'DayChange%'] = 0
                except:
                    df.at[idx, 'Price'] = 0
                    df.at[idx, 'DayChange%'] = 0
        
        # Add company data
        for idx, row in df.iterrows():
            symbol = row['Symbol']
            if symbol in COMPANY_DATA:
                for key, val in COMPANY_DATA[symbol].items():
                    df.at[idx, key] = val
        
        # Calculate values
        df['MarketValue'] = df['Quantity'] * df['Price']
        df['GainLoss'] = df['MarketValue'] - df['Cost Basis']
        df['Return%'] = (df['GainLoss'] / df['Cost Basis'] * 100)
        
        total_mv = df['MarketValue'].sum()
        df['%Portfolio'] = (df['MarketValue'] / total_mv * 100)
        
        # SURVIVAL GATE 1: Cash Runway with Pro-forma
        df['ProFormaCash'] = df.apply(
            lambda r: r['cash'] + FINANCINGS[r['Symbol']]['amount'] 
            if r['Symbol'] in FINANCINGS else r['cash'], axis=1
        )
        df['Runway'] = df['ProFormaCash'] / df['burn']
        df['Gate1_Cash'] = df['Runway'] >= min_cash_runway
        df['Gate1_Reason'] = df.apply(
            lambda r: f"✅ {r['Runway']:.1f} months" if r['Gate1_Cash'] 
            else f"🚨 CRITICAL: Only {r['Runway']:.1f} months cash", axis=1
        )
        
        # SURVIVAL GATE 2: Position Size
        df['Gate2_Size'] = df['%Portfolio'] <= max_position_size
        df['Gate2_Reason'] = df.apply(
            lambda r: f"✅ {r['%Portfolio']:.1f}%" if r['Gate2_Size']
            else f"⚠️ OVER: {r['%Portfolio']:.1f}% (max {max_position_size}%)", axis=1
        )
        
        # SURVIVAL GATE 3: Jurisdiction Risk
        JURISDICTION_SCORES = {
            'Canada': 10, 'USA': 10, 'Mexico': 40, 'Peru': 45, 
            'Brazil': 45, 'Fiji': 50, 'Unknown': 50
        }
        df['JurisdictionRisk'] = df['country'].map(JURISDICTION_SCORES)
        df['Gate3_Jurisdiction'] = df['JurisdictionRisk'] <= 60
        df['Gate3_Reason'] = df.apply(
            lambda r: f"✅ {r['country']}: {r['JurisdictionRisk']}/100" if r['Gate3_Jurisdiction']
            else f"⚠️ HIGH RISK: {r['country']}: {r['JurisdictionRisk']}/100", axis=1
        )
        
        # SURVIVAL GATE 4: Performance
        df['Gate4_Performance'] = df['Return%'] > -40
        df['Gate4_Reason'] = df.apply(
            lambda r: f"✅ {r['Return%']:+.1f}%" if r['Gate4_Performance']
            else f"🚨 MAJOR LOSS: {r['Return%']:+.1f}%", axis=1
        )
        
        # SURVIVAL GATE 5: Check for financing note
        df['FinancingNote'] = df['Symbol'].map(
            lambda s: f" (+${FINANCINGS[s]['amount']}M {FINANCINGS[s]['date']})" 
            if s in FINANCINGS else ""
        )
        
        # Overall survival
        df['SurvivalPassed'] = (df['Gate1_Cash'] & df['Gate2_Size'] & 
                                df['Gate3_Jurisdiction'] & df['Gate4_Performance'])
        
        # NARRATIVE PHASE DETECTION
        def detect_narrative(return_pct, stage):
            if return_pct < -40: return "⚫ DEAD"
            elif return_pct < 10: return "⚪ IGNORED"
            elif 10 <= return_pct < 50: return "🟢 EMERGING (BEST)"
            elif 50 <= return_pct < 150: return "🔵 VALIDATION (GOOD)"
            else: return "🟡 CROWDED (CAUTION)"
        
        df['Narrative'] = df.apply(lambda r: detect_narrative(r['Return%'], r['stage']), axis=1)
        
        # ALPHA SCORING
        df['AlphaScore'] = 50.0
        df.loc[df['Return%'] > 100, 'AlphaScore'] += 15
        df.loc[df['Return%'] > 50, 'AlphaScore'] += 10
        df.loc[df['Return%'] > 20, 'AlphaScore'] += 5
        df.loc[df['Return%'] < -20, 'AlphaScore'] -= 15
        df.loc[df['DayChange%'] > 5, 'AlphaScore'] += 10
        df.loc[df['DayChange%'] < -5, 'AlphaScore'] -= 10
        df.loc[df['Narrative'].str.contains('EMERGING'), 'AlphaScore'] += 20
        df.loc[df['Narrative'].str.contains('VALIDATION'), 'AlphaScore'] += 10
        df.loc[df['Narrative'].str.contains('CROWDED'), 'AlphaScore'] -= 15
        df.loc[df['stage'] == 'Producer', 'AlphaScore'] += 5
        
        # TRUTH ARBITRATION
        def arbitrate_decision(row):
            # RULE 0: Survival failure overrides
            if not row['SurvivalPassed']:
                if not row['Gate1_Cash']:
                    return '🚨 SELL / DO NOT BUY', 1.0, row['Gate1_Reason'], '🔒 SURVIVAL OVERRIDE'
                else:
                    return '🟡 REDUCE', 0.8, 'Multiple survival concerns', '⚠️ High risk'
            
            # Normal scoring
            if row['AlphaScore'] >= 75:
                return '🟢 STRONG BUY', 0.85, f"Excellent setup | {row['Narrative']}", None
            elif row['AlphaScore'] >= 60:
                return '🔵 BUY', 0.75, f"Good opportunity | {row['Narrative']}", None
            elif row['AlphaScore'] >= 45:
                return '⚪ HOLD', 0.60, f"Neutral | {row['Narrative']}", None
            elif row['AlphaScore'] >= 30:
                return '🟡 TRIM', 0.70, f"Weakening | {row['Narrative']}", None
            else:
                return '🔴 SELL', 0.80, f"Multiple red flags | {row['Narrative']}", None
        
        df[['Recommendation', 'Confidence', 'Reasoning', 'Override']] = df.apply(
            lambda r: pd.Series(arbitrate_decision(r)), axis=1
        )
        
        st.session_state.results = df
        st.session_state.analysis_time = datetime.datetime.now()
    
    st.success("✅ Complete analysis finished!")
    st.rerun()

# DISPLAY RESULTS
if 'results' in st.session_state:
    df = st.session_state.results
    
    st.markdown("---")
    st.header(f"📊 Analysis Results - {st.session_state.analysis_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Summary metrics
    total_value = df['MarketValue'].sum()
    total_gain = df['GainLoss'].sum()
    total_cb = df['Cost Basis'].sum()
    total_return = (total_gain / total_cb * 100)
    critical_count = len(df[~df['Gate1_Cash']])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Value", f"${total_value:,.2f}")
    col2.metric("Total Return", f"${total_gain:+,.2f}", f"{total_return:+.1f}%")
    col3.metric("Positions", len(df))
    col4.metric("🚨 Critical", critical_count, delta_color="inverse")
    
    # Critical alerts
    critical = df[~df['SurvivalPassed']].sort_values('Runway')
    if len(critical) > 0:
        st.markdown("## 🚨 CRITICAL ALERTS - Immediate Action")
        for _, row in critical.iterrows():
            with st.expander(f"🚨 {row['Symbol']} - {row['Recommendation']}", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Position Value", f"${row['MarketValue']:,.0f}")
                    st.metric("Return", f"{row['Return%']:+.1f}%", f"${row['GainLoss']:+,.0f}")
                with col2:
                    st.metric("Cash Runway", f"{row['Runway']:.1f} months")
                    st.metric("Confidence", f"{row['Confidence']:.0%}")
                
                st.markdown("**Reasoning:**")
                st.error(row['Reasoning'])
                if row['Override']:
                    st.warning(f"**Override:** {row['Override']}")
                
                st.markdown("**Survival Gate Results:**")
                st.write(f"• Gate 1 (Cash): {row['Gate1_Reason']}{row['FinancingNote']}")
                st.write(f"• Gate 2 (Size): {row['Gate2_Reason']}")
                st.write(f"• Gate 3 (Jurisdiction): {row['Gate3_Reason']}")
                st.write(f"• Gate 4 (Performance): {row['Gate4_Reason']}")
    
    # Top opportunities
    opportunities = df[df['Recommendation'].str.contains('BUY')].sort_values('AlphaScore', ascending=False)
    if len(opportunities) > 0:
        st.markdown("## 🟢 Top Opportunities")
        for _, row in opportunities.head(5).iterrows():
            with st.expander(f"🟢 {row['Symbol']} - {row['Recommendation']} (Alpha: {row['AlphaScore']:.0f}/100)"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Value", f"${row['MarketValue']:,.0f}")
                    st.metric("Return", f"{row['Return%']:+.1f}%")
                with col2:
                    st.metric("Alpha Score", f"{row['AlphaScore']:.0f}/100")
                    st.metric("Narrative", row['Narrative'])
                with col3:
                    st.metric("Runway", f"{row['Runway']:.1f} months")
                    st.metric("Confidence", f"{row['Confidence']:.0%}")
                
                st.markdown("**Analysis:**")
                st.success(row['Reasoning'])
                
                st.markdown("**Survival Gates:**")
                st.write(f"• Cash: {row['Gate1_Reason']}{row['FinancingNote']}")
                st.write(f"• Size: {row['Gate2_Reason']}")
                st.write(f"• Jurisdiction: {row['Gate3_Reason']}")
                st.write(f"• Performance: {row['Gate4_Reason']}")
    
    # Complete breakdown
    st.markdown("## 📋 Complete Position Breakdown")
    
    for _, row in df.sort_values('AlphaScore', ascending=False).iterrows():
        with st.expander(f"{row['Symbol']} - {row['Recommendation']} | Alpha: {row['AlphaScore']:.0f}/100 | {row['Narrative']}"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("**Position**")
                st.write(f"Quantity: {row['Quantity']:,.0f}")
                st.write(f"Price: ${row['Price']:.4f}")
                st.write(f"Value: ${row['MarketValue']:,.2f}")
                st.write(f"% Portfolio: {row['%Portfolio']:.1f}%")
            
            with col2:
                st.markdown("**Performance**")
                st.write(f"Cost Basis: ${row['Cost Basis']:,.2f}")
                st.write(f"Gain/Loss: ${row['GainLoss']:+,.2f}")
                st.write(f"Return: {row['Return%']:+.1f}%")
                st.write(f"Day Change: {row['DayChange%']:+.2f}%")
            
            with col3:
                st.markdown("**Fundamentals**")
                st.write(f"Stage: {row['stage']}")
                st.write(f"Metal: {row['metal']}")
                st.write(f"Country: {row['country']}")
                st.write(f"Runway: {row['Runway']:.1f} months")
            
            with col4:
                st.markdown("**Decision**")
                st.write(f"Alpha: {row['AlphaScore']:.0f}/100")
                st.write(f"Narrative: {row['Narrative']}")
                st.write(f"Action: {row['Recommendation']}")
                st.write(f"Confidence: {row['Confidence']:.0%}")
            
            st.markdown("---")
            st.markdown("**Detailed Analysis:**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Survival Gates:**")
                gate_status = "✅ PASSED" if row['SurvivalPassed'] else "🚨 FAILED"
                st.write(f"Overall: {gate_status}")
                st.write(f"• {row['Gate1_Reason']}{row['FinancingNote']}")
                st.write(f"• {row['Gate2_Reason']}")
                st.write(f"• {row['Gate3_Reason']}")
                st.write(f"• {row['Gate4_Reason']}")
            
            with col2:
                st.markdown("**Recommendation:**")
                st.write(f"Action: {row['Recommendation']}")
                st.write(f"Reasoning: {row['Reasoning']}")
                st.write(f"Confidence: {row['Confidence']:.0%}")
                if row['Override']:
                    st.warning(f"Override: {row['Override']}")
    
    # Export
    st.markdown("---")
    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download Complete Analysis (CSV)",
        csv,
        f"alpha_analysis_{datetime.date.today()}.csv",
        "text/csv",
        use_container_width=True
    )

st.markdown("---")
st.caption("💎 Alpha Miner Ultimate | Complete Capital Allocation System | SURVIVE FIRST, ALPHA SECOND")
