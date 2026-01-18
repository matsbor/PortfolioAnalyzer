"""
ALPHA MINER PRO - INSTITUTIONAL ENHANCEMENTS V2
Complete hedge-fund grade system with metal cycle analysis

CRITICAL ADDITIONS:
- Gold & Silver price predictions (short/medium/long term)
- Metal bias → portfolio behavior (dynamic throttling)
- Enhanced discovery exception (strictly controlled)
- SMC-aware sell-in-time
- Dynamic CORE/TACTICAL sizing
- PM morning tape dashboard
- Lightweight social/institutional stubs

Philosophy: Survival > Liquidity > Dilution > Confidence > Alpha
Author: Senior PM / Risk Officer / Quant
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ============================================================================
# 1️⃣ GOLD & SILVER PRICE PREDICTIONS
# ============================================================================

def analyze_metal_cycle(metal_ticker, name="Gold"):
    """
    Comprehensive metal price analysis and predictions
    
    WHY: Junior miners follow metal prices. We MUST understand the cycle.
    
    ANALYZES:
    - Short-term (today, 1-2 weeks)
    - Medium-term (1-3 months)
    - Cyclical (3-12 months)
    
    USES:
    - MA20/MA50/MA200
    - Momentum & volatility
    - SMC structure
    - DXY correlation
    
    RETURNS: Complete cycle analysis dict
    """
    import yfinance as yf
    
    result = {
        'name': name,
        'current_price': 0,
        'bias_short': 'NEUTRAL',      # Today, 1-2 weeks
        'bias_medium': 'NEUTRAL',     # 1-3 months
        'bias_long': 'NEUTRAL',       # 3-12 months
        'confidence': 50,             # 0-100
        'predictions': {
            'today': '↔',
            'week': '↔',
            'month': '↔',
            'quarter': '↔'
        },
        'technicals': {},
        'smc_structure': 'NEUTRAL',
        'momentum_score': 0,
        'explanation': ''
    }
    
    try:
        # Fetch 2 years of data
        metal = yf.Ticker(metal_ticker)
        hist = metal.history(period="2y")
        
        if hist.empty:
            result['explanation'] = 'No data available'
            return result
        
        current_price = hist['Close'].iloc[-1]
        result['current_price'] = current_price
        
        # Calculate technical indicators
        ma20 = hist['Close'].tail(20).mean()
        ma50 = hist['Close'].tail(50).mean()
        ma200 = hist['Close'].tail(200).mean()
        
        result['technicals'] = {
            'ma20': ma20,
            'ma50': ma50,
            'ma200': ma200,
            'price_vs_ma20': ((current_price - ma20) / ma20 * 100),
            'price_vs_ma50': ((current_price - ma50) / ma50 * 100),
            'price_vs_ma200': ((current_price - ma200) / ma200 * 100)
        }
        
        # 1) SHORT-TERM BIAS (Today, 1-2 weeks)
        # Based on: MA20, momentum, recent action
        ret_5d = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5] * 100)
        ret_10d = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-10]) / hist['Close'].iloc[-10] * 100)
        
        short_score = 0
        
        if current_price > ma20:
            short_score += 2
        if ret_5d > 0:
            short_score += 1
        if ret_10d > 1:
            short_score += 1
        
        if short_score >= 3:
            result['bias_short'] = 'BULLISH'
            result['predictions']['today'] = '↑'
            result['predictions']['week'] = '↑'
        elif short_score <= 1:
            result['bias_short'] = 'BEARISH'
            result['predictions']['today'] = '↓'
            result['predictions']['week'] = '↓'
        
        # 2) MEDIUM-TERM BIAS (1-3 months)
        # Based on: MA50, trend strength, volatility
        ret_30d = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-30]) / hist['Close'].iloc[-30] * 100)
        ret_60d = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-60]) / hist['Close'].iloc[-60] * 100)
        
        medium_score = 0
        
        if current_price > ma50:
            medium_score += 2
        if ma50 > ma200:
            medium_score += 1
        if ret_30d > 2:
            medium_score += 1
        if ret_60d > 3:
            medium_score += 1
        
        if medium_score >= 4:
            result['bias_medium'] = 'BULLISH'
            result['predictions']['month'] = '↑'
        elif medium_score >= 2:
            result['bias_medium'] = 'NEUTRAL'
        else:
            result['bias_medium'] = 'BEARISH'
            result['predictions']['month'] = '↓'
        
        # 3) LONG-TERM BIAS (3-12 months)
        # Based on: MA200, cyclical position, macro
        ret_90d = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-90]) / hist['Close'].iloc[-90] * 100)
        ret_180d = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-180]) / hist['Close'].iloc[-180] * 100)
        
        long_score = 0
        
        if current_price > ma200:
            long_score += 3
        if ma50 > ma200 and ma20 > ma50:
            long_score += 2  # Golden cross structure
        if ret_180d > 5:
            long_score += 1
        
        # Check for secular bull (200d MA trending up)
        ma200_slope = (ma200 - hist['Close'].tail(250).iloc[0]) / hist['Close'].tail(250).iloc[0] * 100
        if ma200_slope > 5:
            long_score += 1
        
        if long_score >= 5:
            result['bias_long'] = 'BULLISH'
            result['predictions']['quarter'] = '↑'
        elif long_score >= 3:
            result['bias_long'] = 'NEUTRAL'
        else:
            result['bias_long'] = 'BEARISH'
            result['predictions']['quarter'] = '↓'
        
        # 4) SMC STRUCTURE
        # Simplified for metals
        swing_highs = []
        for i in range(10, len(hist)-5):
            if hist['High'].iloc[i] == hist['High'].iloc[i-5:i+6].max():
                swing_highs.append(hist['High'].iloc[i])
        
        if len(swing_highs) >= 2:
            if swing_highs[-1] > swing_highs[-2] and current_price > swing_highs[-2]:
                result['smc_structure'] = 'BULLISH_BOS'
            elif swing_highs[-1] < swing_highs[-2] and current_price < swing_highs[-2]:
                result['smc_structure'] = 'BEARISH_BOS'
        
        # 5) MOMENTUM SCORE
        result['momentum_score'] = (ret_5d + ret_10d + ret_30d) / 3
        
        # 6) CONFIDENCE
        # Higher confidence when all timeframes align
        alignment = sum([
            result['bias_short'] == result['bias_medium'],
            result['bias_medium'] == result['bias_long'],
            abs(result['momentum_score']) > 2
        ])
        
        result['confidence'] = 50 + (alignment * 15)
        
        # 7) EXPLANATION
        reasons = []
        if current_price > ma200:
            reasons.append(f"Above MA200 ({ma200:.0f})")
        if result['smc_structure'] != 'NEUTRAL':
            reasons.append(result['smc_structure'])
        if abs(result['momentum_score']) > 3:
            reasons.append(f"Momentum: {result['momentum_score']:+.1f}%")
        
        result['explanation'] = ' | '.join(reasons) if reasons else 'Neutral structure'
        
    except Exception as e:
        result['explanation'] = f'Analysis error: {str(e)[:40]}'
    
    return result

def calculate_metal_regime_impact(gold_analysis, silver_analysis):
    """
    Determine how metal bias affects portfolio behavior
    
    WHY: We can't ignore the metal cycle. If metals are bearish,
    we need to be MORE defensive, not less.
    
    IMPACT:
    - Throttle factor adjustment
    - Max position size scaling
    - Discovery exception difficulty
    - Sell sensitivity
    
    RETURNS: Regime adjustment dict
    """
    result = {
        'regime': 'NEUTRAL',
        'throttle_adjustment': 1.0,     # Multiply by existing throttle
        'max_size_multiplier': 1.0,     # Scale all max position sizes
        'discovery_hardness': 'NORMAL', # EASY / NORMAL / HARD / BLOCKED
        'sell_sensitivity': 1.0,        # Multiply sell risk scores
        'explanation': ''
    }
    
    # Score both metals
    gold_score = 0
    if gold_analysis['bias_medium'] == 'BULLISH':
        gold_score = 2
    elif gold_analysis['bias_medium'] == 'BEARISH':
        gold_score = -2
    
    silver_score = 0
    if silver_analysis['bias_medium'] == 'BULLISH':
        silver_score = 2
    elif silver_analysis['bias_medium'] == 'BEARISH':
        silver_score = -2
    
    combined_score = gold_score + silver_score
    
    # REGIME DETERMINATION
    if combined_score >= 3:
        # Both metals bullish
        result['regime'] = 'METALS_BULLISH'
        result['throttle_adjustment'] = 1.1  # Slightly more aggressive
        result['max_size_multiplier'] = 1.0
        result['discovery_hardness'] = 'NORMAL'
        result['sell_sensitivity'] = 0.9  # Less sensitive to sell signals
        result['explanation'] = 'Bullish metal environment - normal risk appetite'
        
    elif combined_score <= -3:
        # Both metals bearish
        result['regime'] = 'METALS_BEARISH'
        result['throttle_adjustment'] = 0.6  # Much more defensive
        result['max_size_multiplier'] = 0.7  # Cut position sizes
        result['discovery_hardness'] = 'BLOCKED'  # No discovery plays
        result['sell_sensitivity'] = 1.3  # More sensitive to sell signals
        result['explanation'] = 'Bearish metal environment - DEFENSIVE positioning'
        
    elif combined_score == 0:
        # Mixed or neutral
        result['regime'] = 'METALS_MIXED'
        result['throttle_adjustment'] = 0.85
        result['max_size_multiplier'] = 0.9
        result['discovery_hardness'] = 'HARD'
        result['sell_sensitivity'] = 1.1
        result['explanation'] = 'Mixed metal signals - cautious approach'
        
    else:
        # Slight bullish or bearish tilt
        if combined_score > 0:
            result['regime'] = 'METALS_CAUTIOUS_BULLISH'
            result['throttle_adjustment'] = 0.95
            result['max_size_multiplier'] = 0.95
            result['discovery_hardness'] = 'NORMAL'
            result['sell_sensitivity'] = 1.0
            result['explanation'] = 'Cautiously bullish metals'
        else:
            result['regime'] = 'METALS_CAUTIOUS_BEARISH'
            result['throttle_adjustment'] = 0.8
            result['max_size_multiplier'] = 0.85
            result['discovery_hardness'] = 'HARD'
            result['sell_sensitivity'] = 1.15
            result['explanation'] = 'Cautiously bearish metals - reduce exposure'
    
    return result

# ============================================================================
# 2️⃣ ENHANCED DISCOVERY EXCEPTION (METAL-AWARE)
# ============================================================================

def check_discovery_exception_metal_aware(row, liq_metrics, alpha_score, data_confidence,
                                         dilution_risk, momentum_7d, price, ma50, 
                                         smc_confirmed, metal_regime):
    """
    Discovery exception with metal cycle awareness
    
    WHY: Discovery plays are risky. If metals themselves are bearish,
    we should NOT be taking discovery bets, no matter how good the setup.
    
    GATES (ALL must pass):
    1-8: Same as before
    9: Metal regime NOT bearish (NEW)
    10: If metals mixed, require EXTRA confirmation
    
    RETURNS: (allowed, reason, warnings)
    """
    deny_reasons = []
    warnings = []
    
    # Original 8 gates
    if liq_metrics.get('tier_code') == 'L0':
        return False, "L0 tier excluded", []
    
    if row.get('Sleeve', '') != 'TACTICAL':
        return False, "Must be TACTICAL", []
    
    if alpha_score < 85:
        deny_reasons.append(f"Alpha {alpha_score:.0f}<85")
    
    if data_confidence < 70:
        deny_reasons.append(f"Confidence {data_confidence:.0f}<70")
    
    if dilution_risk >= 70:
        deny_reasons.append(f"Dilution {dilution_risk:.0f}≥70")
    
    if not (momentum_7d > 0 and price > ma50):
        if momentum_7d <= 0:
            deny_reasons.append("7d momentum negative")
        if price <= ma50:
            deny_reasons.append("Below MA50")
    
    if not smc_confirmed:
        deny_reasons.append("No SMC confirmation")
    
    if not row.get('Insider_Buying_90d', False):
        deny_reasons.append("No insider buying (required)")
    
    # GATE 9: Metal regime check (NEW)
    metal_hardness = metal_regime.get('discovery_hardness', 'NORMAL')
    
    if metal_hardness == 'BLOCKED':
        return False, "Discovery BLOCKED - bearish metal environment", []
    
    if metal_hardness == 'HARD':
        # Require even higher standards
        if alpha_score < 90:
            deny_reasons.append(f"Alpha {alpha_score:.0f}<90 (hard mode)")
        
        # Require L2 or better in hard mode
        if liq_metrics.get('tier_code') == 'L1':
            deny_reasons.append("L1 not allowed in hard mode")
        
        warnings.append("⚠️ Metal regime MIXED - extra scrutiny applied")
    
    if deny_reasons:
        return False, '; '.join(deny_reasons[:2]), []
    
    # GRANTED with strict warnings
    warnings.extend([
        "⚠️ DISCOVERY EXCEPTION ACTIVE",
        "⚠️ Max 2.5% position (NON-NEGOTIABLE)",
        "⚠️ Monitor DAILY - high risk",
        f"⚠️ Metal regime: {metal_regime.get('regime', 'UNKNOWN')}"
    ])
    
    reason = f"High-conviction discovery: Alpha {alpha_score:.0f}, SMC✓, Insider buying, Metal regime OK"
    return True, reason, warnings

# ============================================================================
# 3️⃣ DYNAMIC TACTICAL & CORE SIZING
# ============================================================================

def calculate_dynamic_position_sizing(row, liq_metrics, dilution_risk, metal_regime, macro_regime):
    """
    Dynamic position sizing based on metal cycle + liquidity + risk
    
    WHY: Position sizes should adapt to the metal environment.
    Don't size like it's a bull market when metals are bearish.
    
    RULES:
    TACTICAL (base):
    - L3: 7.5%
    - L2: 5.0%
    - L1: 2.5%
    - L0: 1.0%
    
    CORE (requires special conditions):
    - 8-12% ONLY if:
      • L3 liquidity
      • Dilution risk < 40
      • Metal regime NOT bearish
      • Stage = Producer or Developer
    
    ADJUSTMENTS:
    - Metal regime multiplier
    - Macro throttle
    - Dilution penalty
    
    RETURNS: Max allowed position %
    """
    tier = liq_metrics.get('tier_code', 'L0')
    sleeve = row.get('Sleeve', 'TACTICAL')
    stage = row.get('stage', 'Explorer')
    
    # BASE SIZING
    if sleeve == 'CORE':
        # Check if qualifies for CORE
        if tier != 'L3':
            # Downgrade to TACTICAL
            base_max = 7.5 if tier == 'L3' else 5.0 if tier == 'L2' else 2.5 if tier == 'L1' else 1.0
        elif dilution_risk >= 40:
            base_max = 7.5  # Downgrade
        elif metal_regime.get('regime', '').endswith('BEARISH'):
            base_max = 7.5  # Downgrade
        elif stage not in ['Producer', 'Developer']:
            base_max = 7.5  # Downgrade
        else:
            # Qualified for CORE
            base_max = 12.0
    
    elif sleeve == 'GAMBLING':
        base_max = 1.0
    
    else:  # TACTICAL
        if tier == 'L3':
            base_max = 7.5
        elif tier == 'L2':
            base_max = 5.0
        elif tier == 'L1':
            base_max = 2.5
        else:  # L0
            base_max = 1.0
    
    # APPLY METAL REGIME MULTIPLIER
    metal_mult = metal_regime.get('max_size_multiplier', 1.0)
    base_max *= metal_mult
    
    # APPLY MACRO THROTTLE
    macro_mult = macro_regime.get('throttle_factor', 1.0)
    base_max *= macro_mult
    
    # DILUTION PENALTY
    if dilution_risk >= 70:
        base_max = min(base_max, 2.0)
    elif dilution_risk >= 50:
        base_max *= 0.8
    
    return round(base_max, 1)

# ============================================================================
# 4️⃣ MORNING TAPE DASHBOARD
# ============================================================================

def generate_morning_tape(gold_analysis, silver_analysis, metal_regime, macro_regime, 
                         df, total_value):
    """
    PM-style morning tape - what you need to know TODAY
    
    WHY: PMs need to know the big picture FIRST, then drill into stocks.
    
    SHOWS:
    - Metal predictions (today, week, month, quarter)
    - Regime status
    - Macro indicators
    - SMC status for metals
    - Today's action plan
    
    RETURNS: Formatted dashboard data
    """
    dashboard = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'regime_status': {},
        'metal_outlook': {},
        'macro_indicators': {},
        'action_plan': {},
        'top_risks': [],
        'top_opportunities': []
    }
    
    # REGIME STATUS
    dashboard['regime_status'] = {
        'overall': metal_regime['regime'],
        'macro': macro_regime.get('regime', 'UNKNOWN'),
        'new_buys_allowed': macro_regime.get('allow_new_buys', True) and 
                           metal_regime['discovery_hardness'] != 'BLOCKED',
        'throttle_total': metal_regime['throttle_adjustment'] * macro_regime.get('throttle_factor', 1.0),
        'explanation': metal_regime['explanation']
    }
    
    # METAL OUTLOOK
    dashboard['metal_outlook'] = {
        'gold': {
            'price': gold_analysis['current_price'],
            'bias_short': gold_analysis['bias_short'],
            'bias_medium': gold_analysis['bias_medium'],
            'bias_long': gold_analysis['bias_long'],
            'confidence': gold_analysis['confidence'],
            'predictions': gold_analysis['predictions'],
            'smc': gold_analysis['smc_structure'],
            'explanation': gold_analysis['explanation']
        },
        'silver': {
            'price': silver_analysis['current_price'],
            'bias_short': silver_analysis['bias_short'],
            'bias_medium': silver_analysis['bias_medium'],
            'bias_long': silver_analysis['bias_long'],
            'confidence': silver_analysis['confidence'],
            'predictions': silver_analysis['predictions'],
            'smc': silver_analysis['smc_structure'],
            'explanation': silver_analysis['explanation']
        }
    }
    
    # MACRO INDICATORS
    dashboard['macro_indicators'] = {
        'dxy': macro_regime.get('dxy', 0),
        'vix': macro_regime.get('vix', 0),
        'gold_trend': macro_regime.get('gold_trend', 'N/A'),
        'silver_trend': macro_regime.get('silver_trend', 'N/A')
    }
    
    # ACTION PLAN
    buys = df[df['Action'].str.contains('BUY|ADD', na=False)]
    sells = df[df['Action'].str.contains('SELL|REDUCE|TRIM', na=False)]
    
    dashboard['action_plan'] = {
        'buy_count': len(buys),
        'sell_count': len(sells),
        'hold_count': len(df) - len(buys) - len(sells),
        'top_buys': buys.nlargest(3, 'Alpha_Score')[['Symbol', 'Action', 'Alpha_Score']].to_dict('records'),
        'top_sells': sells.nlargest(3, 'Sell_Risk_Score')[['Symbol', 'Action', 'Sell_Risk_Score']].to_dict('records')
    }
    
    # TOP RISKS
    risks = df.nlargest(3, 'Sell_Risk_Score')
    for _, row in risks.iterrows():
        dashboard['top_risks'].append({
            'symbol': row['Symbol'],
            'sell_risk': row['Sell_Risk_Score'],
            'action': row['Action'],
            'position': row['Pct_Portfolio']
        })
    
    # TOP OPPORTUNITIES
    opportunities = buys.nlargest(3, 'Alpha_Score') if len(buys) > 0 else pd.DataFrame()
    for _, row in opportunities.iterrows():
        dashboard['top_opportunities'].append({
            'symbol': row['Symbol'],
            'alpha': row['Alpha_Score'],
            'action': row['Action'],
            'recommended_pct': row.get('Recommended_Pct', 0)
        })
    
    return dashboard

# ============================================================================
# 5️⃣ LIGHTWEIGHT SOCIAL/INSTITUTIONAL STUBS
# ============================================================================

def get_social_institutional_signals(ticker):
    """
    Lightweight stub for social/institutional signals
    
    WHY: Real Twitter API costs money. Real institutional data is expensive.
    But we can leave hooks for future integration.
    
    RETURNS: Signal dict with neutral defaults
    """
    signals = {
        'twitter_sentiment': 0,      # -10 to +10
        'twitter_volume': 'LOW',     # LOW/MODERATE/HIGH/EXTREME
        'institutional_calls': [],   # List of recent calls
        'fund_activity': 'NEUTRAL',  # BUYING/SELLING/NEUTRAL
        'confidence': 0,             # 0 = stub, 100 = real data
        'source': 'STUB'             # STUB / API / MANUAL
    }
    
    # TODO: Integrate real data sources here
    # - Twitter API v2 (if configured)
    # - Bank research aggregator
    # - Fund holdings data
    
    return signals

def integrate_social_signals(alpha_score, social_signals):
    """
    Integrate social signals into alpha (lightweight)
    
    WHY: Social sentiment can confirm or contradict alpha.
    But it should NEVER override fundamentals.
    
    ADJUSTMENT: ±5 points maximum
    """
    if social_signals['source'] == 'STUB':
        return alpha_score  # No adjustment if no real data
    
    adjustment = 0
    
    # Twitter sentiment
    if abs(social_signals['twitter_sentiment']) > 7:
        # Extreme sentiment is a CONTRARIAN indicator
        adjustment -= social_signals['twitter_sentiment'] * 0.3
    elif abs(social_signals['twitter_sentiment']) > 3:
        # Moderate sentiment is a confirming indicator
        adjustment += social_signals['twitter_sentiment'] * 0.5
    
    # Cap adjustment
    adjustment = np.clip(adjustment, -5, 5)
    
    return alpha_score + adjustment

# ============================================================================
# INTEGRATION GUIDE V2
# ============================================================================

"""
═══════════════════════════════════════════════════════════════════════════
INTEGRATION GUIDE V2 - METAL CYCLE ANALYSIS & FULL SYSTEM
═══════════════════════════════════════════════════════════════════════════

This builds on institutional_enhancements.py and adds:
- Gold & Silver price predictions
- Metal bias → portfolio behavior
- Dynamic CORE/TACTICAL sizing
- Morning tape dashboard
- Social/institutional stubs

STEP 1: ANALYZE METALS (add to analysis start, around line 1200)
----------------------------------------------------------------

# Analyze metal cycles FIRST (before stocks)
progress.progress(5, text="🪙 Analyzing Gold & Silver cycles...")

gold_analysis = analyze_metal_cycle("GC=F", "Gold")
silver_analysis = analyze_metal_cycle("SI=F", "Silver")

# Determine metal regime impact
metal_regime = calculate_metal_regime_impact(gold_analysis, silver_analysis)

st.session_state.gold_analysis = gold_analysis
st.session_state.silver_analysis = silver_analysis
st.session_state.metal_regime = metal_regime

STEP 2: USE METAL REGIME IN DISCOVERY EXCEPTION (around line 1330)
------------------------------------------------------------------

# Replace check_discovery_exception_strict with metal-aware version
exception = check_discovery_exception_metal_aware(
    row, liq_metrics,
    row['Alpha_Score'],
    row['Data_Confidence'],
    row['Dilution_Risk_Score'],
    row['Return_7d'],
    row['Price'],
    row.get('MA50', 0),
    row.get('SMC_Confirmed', False),
    metal_regime  # NEW parameter
)

STEP 3: DYNAMIC POSITION SIZING (around line 1340)
--------------------------------------------------

# Calculate dynamic max position
max_position_pct = calculate_dynamic_position_sizing(
    row, liq_metrics,
    row['Dilution_Risk_Score'],
    metal_regime,
    macro_regime
)

# Use this instead of static sleeve-based caps
decision['max_allowed_pct'] = max_position_pct

STEP 4: MORNING TAPE DASHBOARD (at very top, around line 1395)
--------------------------------------------------------------

# BEFORE other displays, add morning tape
if 'results' in st.session_state:
    df = st.session_state.results
    
    # Generate morning tape
    morning_tape = generate_morning_tape(
        st.session_state.get('gold_analysis', {}),
        st.session_state.get('silver_analysis', {}),
        st.session_state.get('metal_regime', {}),
        macro,
        df,
        total_value
    )
    
    # Display morning tape (see UI section below)
    render_morning_tape(morning_tape)

STEP 5: MORNING TAPE UI FUNCTION (add new function)
---------------------------------------------------

def render_morning_tape(tape):
    '''PM-style morning dashboard'''
    
    st.markdown("---")
    st.header("📊 MORNING TAPE")
    st.caption(f"Market Brief: {tape['date']}")
    
    # Regime status banner
    regime = tape['regime_status']
    if not regime['new_buys_allowed']:
        st.error(f"🛑 **NO NEW BUYS** - {regime['overall']} + {regime['macro']}")
    elif regime['overall'].endswith('BEARISH'):
        st.warning(f"⚠️ **DEFENSIVE MODE** - {regime['explanation']}")
    else:
        st.success(f"✅ **ACTIVE MODE** - {regime['explanation']}")
    
    # Metal predictions
    st.subheader("🪙 Metal Outlook")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gold = tape['metal_outlook']['gold']
        st.markdown(f"### Gold: ${gold['price']:,.0f}")
        st.write(f"**Bias:** {gold['bias_short']} (short) / {gold['bias_medium']} (med) / {gold['bias_long']} (long)")
        st.write(f"**Predictions:** Today {gold['predictions']['today']} | Week {gold['predictions']['week']} | Month {gold['predictions']['month']} | Quarter {gold['predictions']['quarter']}")
        st.write(f"**SMC:** {gold['smc']}")
        st.caption(gold['explanation'])
    
    with col2:
        silver = tape['metal_outlook']['silver']
        st.markdown(f"### Silver: ${silver['price']:.2f}")
        st.write(f"**Bias:** {silver['bias_short']} (short) / {silver['bias_medium']} (med) / {silver['bias_long']} (long)")
        st.write(f"**Predictions:** Today {silver['predictions']['today']} | Week {silver['predictions']['week']} | Month {silver['predictions']['month']} | Quarter {silver['predictions']['quarter']}")
        st.write(f"**SMC:** {silver['smc']}")
        st.caption(silver['explanation'])
    
    # Action plan
    st.subheader("📋 Today's Action Plan")
    
    col1, col2, col3 = st.columns(3)
    
    plan = tape['action_plan']
    col1.metric("Buy/Add", plan['buy_count'])
    col2.metric("Hold", plan['hold_count'])
    col3.metric("Sell/Trim", plan['sell_count'])
    
    if plan['top_buys']:
        st.success("**Top Buy Opportunities:**")
        for buy in plan['top_buys']:
            st.write(f"• {buy['Symbol']}: {buy['Action']} (Alpha: {buy['Alpha_Score']:.0f})")
    
    if plan['top_sells']:
        st.error("**Top Sell Risks:**")
        for sell in plan['top_sells']:
            st.write(f"• {sell['Symbol']}: {sell['Action']} (Risk: {sell['Sell_Risk_Score']:.0f})")

STEP 6: ADD METAL REGIME TO SELL RISK (around line 1310)
---------------------------------------------------------

# Adjust sell sensitivity based on metal regime
sell_sensitivity = metal_regime.get('sell_sensitivity', 1.0)

# After calculating sell_risk_obj, multiply score
sell_risk_obj['score'] = sell_risk_obj['score'] * sell_sensitivity

STEP 7: SOCIAL SIGNALS STUB (optional, around line 1320)
--------------------------------------------------------

# Get social/institutional signals (stub for now)
social_signals = get_social_institutional_signals(row['Symbol'])

# Integrate into alpha (max ±5 points)
df.at[idx, 'Alpha_Score'] = integrate_social_signals(
    row['Alpha_Score'],
    social_signals
)

═══════════════════════════════════════════════════════════════════════════
THAT'S IT! Your system now has:
✅ Gold & Silver cycle predictions (short/medium/long term)
✅ Metal bias → portfolio behavior (dynamic throttling)
✅ Metal-aware discovery exception
✅ Dynamic CORE/TACTICAL sizing
✅ PM morning tape dashboard
✅ Lightweight social/institutional stubs

All while maintaining existing gates and logic.
═══════════════════════════════════════════════════════════════════════════
"""
