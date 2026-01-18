"""
ALPHA MINER PRO - INSTITUTIONAL ENHANCEMENTS
Surgical additions to existing system - DO NOT REPLACE CORE LOGIC

These enhancements add institutional-grade decision intelligence WITHOUT
rewriting your existing system. Each function is designed to PLUG IN to
existing gates and scoring logic.

PHILOSOPHY:
- Survival > Alpha (always)
- Liquidity > Hype (always)  
- Discipline > Emotion (always)
- Multiple independent confirmations required for high-risk plays

Author: Senior PM / Risk Officer
Date: 2026-01-16
"""

import numpy as np
import pandas as pd
from datetime import datetime

# ============================================================================
# 1️⃣ ENHANCED SMC - INSTITUTIONAL PRICE ACTION
# ============================================================================

def calculate_smc_institutional(hist_data, current_price):
    """
    Enhanced Smart Money Concepts for institutional decision-making
    
    WHY:
    Institutions leave footprints in price action that retail misses.
    This detects accumulation/distribution BEFORE it's obvious.
    
    WHAT IT ADDS:
    - Premium/Discount zones (buy discount, sell premium)
    - Impulse quality (strong = institutional, weak = retail hype)
    - Failed structure breaks (distribution warning)
    - Liquidity engineering (equal highs/lows = stop hunts)
    
    INTEGRATION:
    Call after existing calculate_smc_signals
    Use for CONFIRMATION, not as primary signal
    """
    result = {
        'smc_confirmed': False,
        'zone': 'NEUTRAL',  # PREMIUM / DISCOUNT / EQUILIBRIUM
        'impulse_quality': 0,  # -10 to +10
        'failed_breaks': [],
        'liquidity_engineered': False,
        'explanation': ''
    }
    
    if hist_data.empty or len(hist_data) < 100:
        result['explanation'] = 'Insufficient data'
        return result
    
    try:
        df = hist_data.tail(200).copy().reset_index(drop=True)
        
        # Calculate 100-bar range
        range_df = df.tail(100)
        range_high = range_df['High'].max()
        range_low = range_df['Low'].min()
        range_mid = (range_high + range_low) / 2
        range_size = range_high - range_low
        
        # 1) PREMIUM/DISCOUNT ZONES
        # WHY: Institutions accumulate in discount, distribute in premium
        if current_price > range_mid + range_size * 0.25:
            result['zone'] = 'PREMIUM'
        elif current_price < range_mid - range_size * 0.25:
            result['zone'] = 'DISCOUNT'
        else:
            result['zone'] = 'EQUILIBRIUM'
        
        # 2) IMPULSE QUALITY
        # WHY: Strong impulse + high volume = institutional
        #      Weak impulse + low volume = retail hope
        for i in range(len(df)-1, max(0, len(df)-50), -1):
            if i < 5:
                break
            
            move_pct = abs((df['Close'].iloc[i] - df['Close'].iloc[i-5]) / df['Close'].iloc[i-5] * 100)
            
            if move_pct > 5:  # Significant move
                avg_vol_impulse = df['Volume'].iloc[i-5:i].mean()
                avg_vol_baseline = df['Volume'].iloc[max(0,i-25):i-5].mean()
                vol_ratio = avg_vol_impulse / avg_vol_baseline if avg_vol_baseline > 0 else 1
                
                if vol_ratio > 1.5:
                    result['impulse_quality'] += 5
                elif vol_ratio < 0.8:
                    result['impulse_quality'] -= 3
                
                # Clean bodies = institutional
                body_ratios = []
                for j in range(i-5, i):
                    hl_range = df['High'].iloc[j] - df['Low'].iloc[j]
                    if hl_range > 0:
                        body = abs(df['Close'].iloc[j] - df['Open'].iloc[j])
                        body_ratios.append(body / hl_range)
                
                if body_ratios and np.mean(body_ratios) > 0.6:
                    result['impulse_quality'] += 3
                break
        
        # 3) FAILED BREAKS (distribution signal)
        # WHY: False breakouts = smart money selling into retail FOMO
        swing_highs = []
        for i in range(10, len(df)-5):
            if df['High'].iloc[i] == df['High'].iloc[i-5:i+6].max():
                swing_highs.append((i, df['High'].iloc[i]))
        
        recent_bars = df.tail(20)
        for idx, swing_high in swing_highs[-5:]:
            for bar in recent_bars.itertuples():
                if bar.High > swing_high * 1.001 and bar.Close < swing_high * 0.999:
                    result['failed_breaks'].append('FAILED_HIGH')
                    result['impulse_quality'] -= 5
                    break
        
        # 4) LIQUIDITY ENGINEERING
        # WHY: Equal highs/lows = stops being hunted before real move
        recent_highs = df.tail(50)['High'].values
        equal_high_count = sum(
            1 for i in range(len(recent_highs)-1)
            for j in range(i+1, len(recent_highs))
            if abs(recent_highs[i] - recent_highs[j]) / recent_highs[i] < 0.005
        )
        
        if equal_high_count >= 3:
            result['liquidity_engineered'] = True
        
        # 5) CONFIRMATION LOGIC
        confirmation_score = 0
        reasons = []
        
        if result['zone'] == 'DISCOUNT' and result['impulse_quality'] > 3:
            confirmation_score += 1
            reasons.append("Accumulation zone + strong impulse")
        
        if result['zone'] == 'DISCOUNT' and not result['failed_breaks']:
            confirmation_score += 1
            reasons.append("Clean structure in discount")
        
        if result['impulse_quality'] > 5:
            confirmation_score += 1
            reasons.append("Institutional-grade impulse")
        
        if result['zone'] == 'PREMIUM' and result['failed_breaks']:
            confirmation_score -= 2
            reasons.append("Distribution pattern")
        
        if result['liquidity_engineered'] and result['zone'] == 'PREMIUM':
            confirmation_score -= 1
            reasons.append("Liquidity grab in premium")
        
        result['smc_confirmed'] = confirmation_score >= 2
        result['explanation'] = ' | '.join(reasons) if reasons else 'No clear pattern'
        
    except Exception as e:
        result['explanation'] = f'Error: {str(e)[:40]}'
    
    return result

# ============================================================================
# 2️⃣ STRICT DISCOVERY EXCEPTION
# ============================================================================

def check_discovery_exception_strict(row, liq_metrics, alpha_score, data_confidence, 
                                     dilution_risk, momentum_7d, price, ma50, smc_confirmed):
    """
    STRICT discovery exception - institutional standards
    
    WHY:
    Discovery plays = high risk. We need MULTIPLE independent confirmations.
    Insider buying is now REQUIRED (they know something we don't).
    
    GATES (ALL must pass):
    1. NOT L0 tier
    2. TACTICAL sleeve only
    3. Alpha ≥ 85 (very high)
    4. Data confidence ≥ 70 (not blind)
    5. Dilution risk < 70 (manageable)
    6. Momentum: 7d > 0 AND price > MA50
    7. SMC confirmed (institutional validation)
    8. Insider buying in last 90d (REQUIRED)
    
    RETURNS: (allowed, reason, warnings)
    """
    deny_reasons = []
    
    # GATE 1: Tier
    if liq_metrics.get('tier_code') == 'L0':
        return False, "L0 tier excluded", []
    
    # GATE 2: Sleeve
    if row.get('Sleeve', '') != 'TACTICAL':
        return False, "Must be TACTICAL", []
    
    # GATE 3: Alpha
    if alpha_score < 85:
        deny_reasons.append(f"Alpha {alpha_score:.0f}<85")
    
    # GATE 4: Confidence
    if data_confidence < 70:
        deny_reasons.append(f"Confidence {data_confidence:.0f}<70")
    
    # GATE 5: Dilution
    if dilution_risk >= 70:
        deny_reasons.append(f"Dilution {dilution_risk:.0f}≥70")
    
    # GATE 6: Momentum
    if not (momentum_7d > 0 and price > ma50):
        if momentum_7d <= 0:
            deny_reasons.append("7d momentum negative")
        if price <= ma50:
            deny_reasons.append("Below MA50")
    
    # GATE 7: SMC
    if not smc_confirmed:
        deny_reasons.append("No SMC confirmation")
    
    # GATE 8: Insider (REQUIRED)
    if not row.get('Insider_Buying_90d', False):
        deny_reasons.append("No insider buying (required)")
    
    if deny_reasons:
        return False, '; '.join(deny_reasons[:2]), []
    
    # GRANTED with strict warnings
    warnings = [
        "⚠️ DISCOVERY EXCEPTION ACTIVE",
        "⚠️ Max 2.5% position (non-negotiable)",
        "⚠️ Monitor DAILY - high risk"
    ]
    
    reason = f"High-conviction discovery: Alpha {alpha_score:.0f}, SMC✓, Insider buying"
    return True, reason, warnings

# ============================================================================
# 3️⃣ PRECISE NEWS CLASSIFICATION
# ============================================================================

def classify_financing_precision(news_items):
    """
    Precise financing classification to prevent false signals
    
    WHY:
    "Announces financing" vs "Closes financing" are OPPOSITE signals.
    The former increases dilution risk, latter reduces it.
    
    CLASSIFICATION:
    - CLOSED_PP: Financing closed (reduces risk -15)
    - STRATEGIC: Strategic financing closed (reduces risk -20)
    - OPEN_PP: Announced but not closed (increases risk +15)
    - ATM_ACTIVE: Active ATM program (increases risk +25)
    - SHELF_FILED: Shelf registration (increases risk +20)
    
    RETURNS: Financing analysis dict
    """
    result = {
        'type': None,
        'status': None,
        'dilution_adj': 0,  # Adjustment to dilution risk score
        'confidence': 100,
        'headline': None,
        'explanation': ''
    }
    
    close_words = ['closes', 'closed', 'closing', 'completes', 'completed']
    open_words = ['announces', 'proposes', 'seeks', 'plans', 'intends']
    atm_words = ['atm', 'at-the-market', 'at the market']
    shelf_words = ['shelf', 'prospectus', 'registration']
    strategic_words = ['strategic', 'cornerstone', 'lead investor']
    
    for item in news_items:
        title = item.get('title', '').lower()
        
        if not any(w in title for w in ['financing', 'placement', 'offering', 'capital', 'atm', 'shelf', 'prospectus']):
            continue
        
        # CLOSED (good news)
        if any(w in title for w in close_words):
            result['status'] = 'CLOSED'
            result['dilution_adj'] = -15
            result['headline'] = item.get('title')
            result['explanation'] = 'Financing closed - runway extended'
            
            if any(w in title for w in strategic_words):
                result['type'] = 'STRATEGIC'
                result['dilution_adj'] = -20
                result['explanation'] = 'Strategic financing - institutional validation'
            else:
                result['type'] = 'CLOSED_PP'
            break
        
        # ANNOUNCED (bad news)
        elif any(w in title for w in open_words):
            result['status'] = 'ANNOUNCED'
            result['dilution_adj'] = +15
            result['headline'] = item.get('title')
            
            if any(w in title for w in atm_words):
                result['type'] = 'ATM_ACTIVE'
                result['dilution_adj'] = +25
                result['explanation'] = 'Active ATM - ongoing dilution'
            elif any(w in title for w in shelf_words):
                result['type'] = 'SHELF_FILED'
                result['dilution_adj'] = +20
                result['explanation'] = 'Shelf filed - dilution imminent'
            else:
                result['type'] = 'OPEN_PP'
                result['explanation'] = 'Financing announced - execution risk'
        
        # ATM/SHELF without closing
        elif any(w in title for w in atm_words + shelf_words):
            if result['status'] is None:
                result['status'] = 'ACTIVE'
                result['type'] = 'ATM_ACTIVE' if 'atm' in title else 'SHELF_FILED'
                result['dilution_adj'] = +25
                result['headline'] = item.get('title')
                result['explanation'] = 'Active dilution mechanism'
    
    # Reduce confidence if vague
    if result['headline']:
        hl = result['headline'].lower()
        if any(w in hl for w in ['may', 'could', 'potential', 'considers']):
            result['confidence'] = 60
        elif any(w in hl for w in ['confirms', 'completes', 'announces']):
            result['confidence'] = 90
    
    return result

# ============================================================================
# 4️⃣ SOCIAL SENTIMENT PROXY (NO API REQUIRED)
# ============================================================================

def calculate_social_proxy(ticker, news_items, price_7d, volume_spike):
    """
    Social sentiment proxy WITHOUT requiring Twitter API
    
    WHY:
    We can infer buzz from:
    - News velocity (more news = more attention)
    - Price/news divergence (big move + news = hype)
    - Volume spikes (retail FOMO)
    
    PHILOSOPHY:
    This is a NEGATIVE signal (hype warning), not a buy trigger.
    High score = exercise caution.
    
    RETURNS: Sentiment dict with warnings
    """
    result = {
        'score': 0,  # -20 to +20
        'buzz': 'LOW',  # LOW/MODERATE/HIGH/EXTREME
        'hype_warning': False,
        'explanation': ''
    }
    
    # 1) News velocity
    news_7d = len([n for n in news_items if n.get('timestamp', 0) > 0])
    
    if news_7d >= 10:
        result['score'] += 10
        result['buzz'] = 'EXTREME'
    elif news_7d >= 5:
        result['score'] += 5
        result['buzz'] = 'HIGH'
    elif news_7d >= 2:
        result['score'] += 2
        result['buzz'] = 'MODERATE'
    
    # 2) Price/news divergence (hype detector)
    if abs(price_7d) > 15 and news_7d >= 3:
        if price_7d > 15:
            result['score'] += 8
            result['hype_warning'] = True
            result['explanation'] = 'Extreme bullish hype - retail FOMO likely'
        else:
            result['score'] -= 8
            result['explanation'] = 'Negative buzz - retail panic'
    
    # 3) Volume spike
    if volume_spike > 3.0:
        result['score'] += 5
        if result['hype_warning']:
            result['explanation'] += ' | Volume confirms retail activity'
    
    # 4) Headline sentiment
    pos_words = ['breakthrough', 'major', 'significant', 'exceptional']
    neg_words = ['delays', 'issues', 'problems', 'disappoints']
    
    sentiment = 0
    for item in news_items[:5]:
        title = item.get('title', '').lower()
        sentiment += sum(1 for w in pos_words if w in title)
        sentiment -= sum(1 for w in neg_words if w in title)
    
    result['score'] += sentiment * 2
    result['score'] = np.clip(result['score'], -20, 20)
    
    # Final warnings
    if result['score'] > 15:
        result['hype_warning'] = True
        if not result['explanation']:
            result['explanation'] = 'Extreme bullish sentiment - exercise caution'
    elif result['score'] < -15:
        result['explanation'] = 'Extreme bearish sentiment - capitulation opportunity?'
    
    return result

# ============================================================================
# 5️⃣ ENHANCED SELL-IN-TIME
# ============================================================================

def add_institutional_sell_triggers(sell_score, triggers_hard, triggers_soft, 
                                    row, hist_data, ma50, ma200, smc_data, news_items):
    """
    Add institutional distribution signals to existing sell logic
    
    WHY:
    Most losses come from holding through distribution.
    These signals detect it EARLY.
    
    NEW TRIGGERS:
    - Failed MA reclaim (trying to break MA and failing)
    - SMC distribution (failed breaks in premium)
    - News/price divergence (bad news + rally = distribution)
    - Liquidity grab + weakness
    
    MODIFIES IN PLACE: sell_score, triggers
    """
    ret_7d = row.get('Return_7d', 0)
    
    # 1) Failed MA reclaim
    if len(hist_data) >= 10 and ma50 > 0:
        recent = hist_data.tail(10)
        touched = False
        failed = False
        
        for bar in recent.itertuples():
            if bar.High >= ma50 * 0.99:
                touched = True
            if touched and bar.Close < ma50 * 0.95:
                failed = True
        
        if failed:
            sell_score += 15
            triggers_soft.append("⚠️ Failed MA50 reclaim")
    
    # 2) SMC distribution
    if smc_data.get('failed_breaks') and smc_data.get('zone') == 'PREMIUM':
        sell_score += 20
        triggers_hard.append("💀 SMC distribution (failed breaks in premium)")
    
    # 3) News/price divergence
    negative_news = any(
        w in item.get('title', '').lower()
        for item in news_items
        for w in ['disappoints', 'delays', 'miss', 'below', 'weak']
    )
    
    if negative_news and ret_7d > 0:
        sell_score += 10
        triggers_soft.append("⚠️ Bad news + price holding (distribution?)")
    
    # 4) Liquidity engineering + weakness
    if smc_data.get('liquidity_engineered') and ret_7d < -3:
        sell_score += 12
        triggers_soft.append("⚠️ Liquidity grab → weakness")
    
    return sell_score

# ============================================================================
# 6️⃣ PORTFOLIO INTELLIGENCE
# ============================================================================

def calculate_portfolio_risk_intelligence(df, total_value):
    """
    Portfolio-level risk analytics
    
    WHY:
    Individual risk is one thing. Portfolio concentration is another.
    This prevents overexposure to any single risk factor.
    
    METRICS:
    - Illiquid exposure % (L0/L1)
    - Weighted dilution risk
    - Average days to exit
    - Stage breakdown (Explorer/Developer/Producer)
    - Capital at risk in 90 days
    
    RETURNS: Risk intelligence dict with warnings
    """
    result = {
        'illiquid_pct': 0,
        'weighted_dilution': 0,
        'avg_exit_days': 0,
        'stage_breakdown': {},
        'capital_at_risk_90d': 0,
        'warnings': [],
        'health': 'GOOD'
    }
    
    try:
        # 1) Illiquid exposure
        illiquid = df[df['Liq_tier_code'].isin(['L0', 'L1'])]
        result['illiquid_pct'] = illiquid['Pct_Portfolio'].sum()
        
        if result['illiquid_pct'] > 25:
            result['warnings'].append(
                f"⚠️ {result['illiquid_pct']:.0f}% in illiquid (L0/L1)"
            )
        
        # 2) Weighted dilution
        df['wt_dil'] = df['Dilution_Risk_Score'] * (df['Market_Value'] / total_value)
        result['weighted_dilution'] = df['wt_dil'].sum()
        
        if result['weighted_dilution'] > 50:
            result['warnings'].append(
                f"⚠️ Weighted dilution {result['weighted_dilution']:.0f}/100"
            )
        
        # 3) Avg exit days
        result['avg_exit_days'] = df['Liq_days_to_exit'].mean()
        
        if result['avg_exit_days'] > 10:
            result['warnings'].append(
                f"⚠️ Avg {result['avg_exit_days']:.1f}d to exit portfolio"
            )
        
        # 4) Stage breakdown
        stage_pct = df.groupby('stage')['Pct_Portfolio'].sum().to_dict()
        result['stage_breakdown'] = stage_pct
        
        explorer_pct = stage_pct.get('Explorer', 0)
        if explorer_pct > 40:
            result['warnings'].append(
                f"⚠️ {explorer_pct:.0f}% in Explorers (high risk)"
            )
        
        # 5) Capital at risk (runway < 9mo)
        at_risk = df[df['Runway'] < 9]
        result['capital_at_risk_90d'] = at_risk['Market_Value'].sum()
        risk_pct = (result['capital_at_risk_90d'] / total_value * 100) if total_value > 0 else 0
        
        if risk_pct > 30:
            result['warnings'].append(
                f"⚠️ ${result['capital_at_risk_90d']:,.0f} ({risk_pct:.0f}%) needs financing <9mo"
            )
        
        # Overall health
        warning_count = len(result['warnings'])
        if warning_count == 0:
            result['health'] = 'EXCELLENT'
        elif warning_count <= 2:
            result['health'] = 'GOOD'
        elif warning_count <= 4:
            result['health'] = 'CAUTION'
        else:
            result['health'] = 'HIGH RISK'
    
    except Exception as e:
        result['health'] = 'ERROR'
        result['warnings'].append(f"Analysis error: {str(e)[:40]}")
    
    return result

# ============================================================================
# INTEGRATION INSTRUCTIONS
# ============================================================================

"""
═══════════════════════════════════════════════════════════════════════════
INTEGRATION GUIDE - HOW TO ADD THESE TO YOUR EXISTING SYSTEM
═══════════════════════════════════════════════════════════════════════════

STEP 1: ADD THIS FILE
---------------------
Save this file as: institutional_enhancements.py
Place it in the same directory as alpha_miner_enhanced.py

STEP 2: IMPORT IN MAIN FILE
---------------------------
At the top of alpha_miner_enhanced.py (around line 12), add:

    from institutional_enhancements import (
        calculate_smc_institutional,
        check_discovery_exception_strict,
        classify_financing_precision,
        calculate_social_proxy,
        add_institutional_sell_triggers,
        calculate_portfolio_risk_intelligence
    )

STEP 3: IN ANALYSIS LOOP (around line 1300)
-------------------------------------------
After calculating existing SMC, add:

    # Enhanced institutional SMC
    smc_inst = calculate_smc_institutional(hist, row['Price'])
    df.at[idx, 'SMC_Confirmed'] = smc_inst['smc_confirmed']
    df.at[idx, 'SMC_Zone'] = smc_inst['zone']
    df.at[idx, 'SMC_Explanation'] = smc_inst['explanation']
    
    # Financing classification
    financing = classify_financing_precision(news_cache.get(row['Symbol'], []))
    df.at[idx, 'Financing_Status'] = financing['status']
    df.at[idx, 'Financing_Type'] = financing['type']
    
    # Adjust dilution risk
    df.at[idx, 'Dilution_Risk_Score'] = max(0, 
        row['Dilution_Risk_Score'] + financing['dilution_adj']
    )
    
    # Social sentiment
    vol_spike = 1.0
    if len(hist) >= 20:
        recent_vol = hist['Volume'].tail(5).mean()
        base_vol = hist['Volume'].tail(20).mean()
        vol_spike = recent_vol / base_vol if base_vol > 0 else 1.0
    
    social = calculate_social_proxy(
        row['Symbol'],
        news_cache.get(row['Symbol'], []),
        row['Return_7d'],
        vol_spike
    )
    df.at[idx, 'Social_Score'] = social['score']
    df.at[idx, 'Hype_Warning'] = social['hype_warning']

STEP 4: REPLACE DISCOVERY EXCEPTION (around line 1330)
------------------------------------------------------
Change check_discovery_exception call to:

    exception = check_discovery_exception_strict(
        row, liq_metrics,
        row['Alpha_Score'],
        row['Data_Confidence'],
        row['Dilution_Risk_Score'],
        row['Return_7d'],
        row['Price'],
        row.get('MA50', 0),
        row.get('SMC_Confirmed', False)  # NEW
    )

STEP 5: ENHANCE SELL RISK (around line 1310)
--------------------------------------------
After existing calculate_sell_risk, add:

    # Add institutional triggers
    smc_data_inst = {
        'failed_breaks': smc_inst.get('failed_breaks', []),
        'zone': smc_inst.get('zone', 'NEUTRAL'),
        'liquidity_engineered': smc_inst.get('liquidity_engineered', False)
    }
    
    sell_risk_obj['score'] = add_institutional_sell_triggers(
        sell_risk_obj['score'],
        sell_risk_obj['hard_triggers'],
        sell_risk_obj['soft_triggers'],
        row, hist, row.get('MA50', 0), row.get('MA200', 0),
        smc_data_inst, news_cache.get(row['Symbol'], [])
    )

STEP 6: ADD PORTFOLIO INTELLIGENCE TO COMMAND CENTER (around line 1420)
-----------------------------------------------------------------------
After existing command center metrics, add:

    # Portfolio Risk Intelligence
    portfolio_intel = calculate_portfolio_risk_intelligence(df, total_mv + cash)
    
    st.markdown("### 🏦 Portfolio Health Check")
    col1, col2, col3, col4 = st.columns(4)
    
    health_color = {
        'EXCELLENT': '🟢',
        'GOOD': '🔵',
        'CAUTION': '🟡',
        'HIGH RISK': '🔴'
    }
    
    col1.metric("Overall Health", 
                f"{health_color.get(portfolio_intel['health'], '⚪')} {portfolio_intel['health']}")
    col2.metric("Illiquid Exposure", f"{portfolio_intel['illiquid_pct']:.1f}%")
    col3.metric("Weighted Dilution", f"{portfolio_intel['weighted_dilution']:.0f}/100")
    col4.metric("Avg Exit Days", f"{portfolio_intel['avg_exit_days']:.1f}d")
    
    if portfolio_intel['warnings']:
        st.warning("**Portfolio Concentration Warnings:**")
        for warning in portfolio_intel['warnings']:
            st.write(warning)
    
    # Stage breakdown
    if portfolio_intel['stage_breakdown']:
        st.markdown("**Exposure by Stage:**")
        for stage, pct in portfolio_intel['stage_breakdown'].items():
            st.caption(f"{stage}: {pct:.1f}%")

STEP 7: ADD NEW BADGES IN DISPLAY (around line 1500)
----------------------------------------------------
After existing badges, add:

    # SMC institutional confirmation
    if row.get('SMC_Confirmed', False):
        badge_html += '<span class="badge-l3">SMC ✓ Institutional</span> '
    
    # SMC zone
    zone = row.get('SMC_Zone', 'NEUTRAL')
    if zone == 'DISCOUNT':
        badge_html += '<span class="badge-l3">DISCOUNT ZONE</span> '
    elif zone == 'PREMIUM':
        badge_html += '<span class="badge-l1">PREMIUM ZONE</span> '
    
    # Hype warning
    if row.get('Hype_Warning', False):
        badge_html += '<span class="badge-gambling">⚠️ HYPE</span> '
    
    # Financing status
    fin_status = row.get('Financing_Status')
    if fin_status == 'CLOSED':
        badge_html += '<span class="badge-l3">PP CLOSED ✓</span> '
    elif fin_status == 'ANNOUNCED':
        badge_html += '<span class="badge-l1">PP OPEN ⚠️</span> '
    elif fin_status == 'ACTIVE':
        badge_html += '<span class="badge-gambling">ATM ACTIVE ⚠️</span> '

STEP 8: ADD TO EXPANDABLE DETAILS (around line 1600)
----------------------------------------------------
In the detailed ticker view, add:

    # Institutional SMC Analysis
    st.markdown("---")
    st.subheader("🏦 Institutional Price Action")
    st.write(f"**Zone:** {row.get('SMC_Zone', 'N/A')}")
    st.write(f"**Confirmed:** {'✅ Yes' if row.get('SMC_Confirmed', False) else '❌ No'}")
    st.write(f"**Explanation:** {row.get('SMC_Explanation', 'No analysis')}")
    
    # Social sentiment
    st.markdown("---")
    st.subheader("📱 Market Buzz Analysis")
    social_score = row.get('Social_Score', 0)
    st.write(f"**Sentiment Score:** {social_score:+.0f}/20")
    if row.get('Hype_Warning', False):
        st.error("⚠️ HYPE WARNING: Extreme bullish sentiment detected")
    
    # Financing detail
    if row.get('Financing_Type'):
        st.markdown("---")
        st.subheader("💰 Financing Analysis")
        st.write(f"**Type:** {row.get('Financing_Type')}")
        st.write(f"**Status:** {row.get('Financing_Status')}")
        
        dilution_adj = financing.get('dilution_adj', 0)
        if dilution_adj < 0:
            st.success(f"Dilution risk reduced by {abs(dilution_adj)} points")
        elif dilution_adj > 0:
            st.error(f"Dilution risk increased by {dilution_adj} points")

═══════════════════════════════════════════════════════════════════════════
THAT'S IT! These are SURGICAL additions that enhance without breaking.
═══════════════════════════════════════════════════════════════════════════

Your system will now have:
✅ Institutional SMC validation
✅ Strict discovery exception (insider buying required)
✅ Precise financing classification (closed vs open)
✅ Social sentiment proxy (no API needed)
✅ Enhanced sell triggers (distribution detection)
✅ Portfolio-level risk intelligence

All while maintaining your existing gates and philosophy.
"""
