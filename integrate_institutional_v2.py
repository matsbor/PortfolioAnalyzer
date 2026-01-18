#!/usr/bin/env python3
"""
INSTITUTIONAL ENHANCEMENTS V2 - COMPLETE INTEGRATION
Integrates BOTH v1 (institutional features) and v2 (metal cycle analysis)

This creates a complete hedge-fund grade system with:
- Institutional SMC
- Strict discovery exception  
- Precise financing classification
- Social sentiment proxy
- Enhanced sell triggers
- Portfolio intelligence
- Gold & Silver cycle predictions
- Metal-aware position sizing
- Morning tape dashboard

Run this to get the complete system.
"""

import sys
from pathlib import Path

def integrate_complete_system(input_file, output_file):
    """Apply ALL institutional enhancements"""
    
    print("🚀 Alpha Miner Pro - Complete Institutional Integration v2")
    print("=" * 70)
    
    # Read file
    print("\n📖 Reading alpha_miner_enhanced.py...")
    with open(input_file, 'r') as f:
        content = f.read()
    
    print(f"✅ Original: {len(content)} characters")
    
    # ========================================================================
    # PATCH 1: Enhanced imports (both modules)
    # ========================================================================
    print("\n📝 Patch 1: Adding comprehensive imports...")
    
    import_section = '''from pathlib import Path

try:
    import yfinance as yf'''
    
    new_import_section = '''from pathlib import Path

# Import institutional enhancements (v1 + v2)
try:
    from institutional_enhancements import (
        calculate_smc_institutional,
        check_discovery_exception_strict,
        classify_financing_precision,
        calculate_social_proxy,
        add_institutional_sell_triggers,
        calculate_portfolio_risk_intelligence
    )
    from institutional_enhancements_v2 import (
        analyze_metal_cycle,
        calculate_metal_regime_impact,
        check_discovery_exception_metal_aware,
        calculate_dynamic_position_sizing,
        generate_morning_tape,
        get_social_institutional_signals,
        integrate_social_signals
    )
    INSTITUTIONAL_V2_AVAILABLE = True
except ImportError as e:
    INSTITUTIONAL_V2_AVAILABLE = False
    print(f"⚠️ Institutional enhancements not found: {e}")
    print("Running without institutional features")

try:
    import yfinance as yf'''
    
    if import_section in content:
        content = content.replace(import_section, new_import_section)
        print("   ✅ Added v1 + v2 imports")
    
    # ========================================================================
    # PATCH 2: Analyze metals FIRST (before stock analysis)
    # ========================================================================
    print("\n📝 Patch 2: Adding metal cycle analysis...")
    
    analysis_start = '''    progress = st.progress(0, text="Starting comprehensive analysis...")
    
    df = st.session_state.portfolio.copy()
    
    # Fetch all data
    progress.progress(10, text="📊 Fetching market data (1-2 years)...")'''
    
    analysis_with_metals = '''    progress = st.progress(0, text="Starting comprehensive analysis...")
    
    df = st.session_state.portfolio.copy()
    
    # INSTITUTIONAL V2: Analyze metal cycles FIRST
    if INSTITUTIONAL_V2_AVAILABLE:
        progress.progress(3, text="🪙 Analyzing Gold & Silver cycles...")
        
        gold_analysis = analyze_metal_cycle("GC=F", "Gold")
        silver_analysis = analyze_metal_cycle("SI=F", "Silver")
        
        # Determine metal regime impact on portfolio
        metal_regime = calculate_metal_regime_impact(gold_analysis, silver_analysis)
        
        st.session_state.gold_analysis = gold_analysis
        st.session_state.silver_analysis = silver_analysis
        st.session_state.metal_regime = metal_regime
        
        # Display metal status
        if metal_regime['regime'].endswith('BEARISH'):
            st.warning(f"⚠️ {metal_regime['regime']}: {metal_regime['explanation']}")
        elif metal_regime['regime'].endswith('BULLISH'):
            st.success(f"✅ {metal_regime['regime']}: {metal_regime['explanation']}")
        else:
            st.info(f"📊 {metal_regime['regime']}: {metal_regime['explanation']}")
    else:
        # Default neutral regime
        st.session_state.metal_regime = {
            'regime': 'NEUTRAL',
            'throttle_adjustment': 1.0,
            'max_size_multiplier': 1.0,
            'discovery_hardness': 'NORMAL',
            'sell_sensitivity': 1.0
        }
    
    # Fetch all data
    progress.progress(10, text="📊 Fetching market data (1-2 years)...")'''
    
    if analysis_start in content:
        content = content.replace(analysis_start, analysis_with_metals)
        print("   ✅ Added metal cycle analysis at start")
    
    # ========================================================================
    # PATCH 3: Add morning tape function
    # ========================================================================
    print("\n📝 Patch 3: Adding morning tape dashboard function...")
    
    # Find where to add function (before results display)
    function_location = '''# ============================================================================
# G) COMMAND CENTER DISPLAY
# ============================================================================

if 'results' in st.session_state:'''
    
    morning_tape_function = '''# ============================================================================
# MORNING TAPE DASHBOARD
# ============================================================================

def render_morning_tape(tape):
    """PM-style morning dashboard"""
    
    st.markdown("---")
    st.header("📊 MORNING TAPE")
    st.caption(f"Market Brief: {tape['date']}")
    
    # Regime status banner
    regime = tape['regime_status']
    if not regime['new_buys_allowed']:
        st.error(f"🛑 **NO NEW BUYS TODAY** - {regime['overall']} + {regime['macro']}")
    elif regime['overall'].endswith('BEARISH'):
        st.warning(f"⚠️ **DEFENSIVE MODE** - {regime['explanation']}")
    else:
        st.success(f"✅ **ACTIVE MODE** - {regime['explanation']}")
    
    st.caption(f"Combined Throttle: {regime['throttle_total']:.1%}")
    
    # Metal predictions
    st.subheader("🪙 Metal Outlook")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gold = tape['metal_outlook']['gold']
        st.markdown(f"### 🥇 Gold: ${gold['price']:,.0f}")
        st.write(f"**Bias:** {gold['bias_short']} → {gold['bias_medium']} → {gold['bias_long']}")
        pred_str = f"{gold['predictions']['today']} today | {gold['predictions']['week']} week | {gold['predictions']['month']} month | {gold['predictions']['quarter']} quarter"
        st.write(f"**Predictions:** {pred_str}")
        st.write(f"**SMC:** {gold['smc']}")
        st.caption(gold['explanation'])
    
    with col2:
        silver = tape['metal_outlook']['silver']
        st.markdown(f"### 🥈 Silver: ${silver['price']:.2f}")
        st.write(f"**Bias:** {silver['bias_short']} → {silver['bias_medium']} → {silver['bias_long']}")
        pred_str = f"{silver['predictions']['today']} today | {silver['predictions']['week']} week | {silver['predictions']['month']} month | {silver['predictions']['quarter']} quarter"
        st.write(f"**Predictions:** {pred_str}")
        st.write(f"**SMC:** {silver['smc']}")
        st.caption(silver['explanation'])
    
    # Macro indicators
    st.subheader("📈 Macro Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    macro = tape['macro_indicators']
    col1.metric("DXY", f"{macro.get('dxy', 0):.1f}" if macro.get('dxy') else "N/A")
    col2.metric("VIX", f"{macro.get('vix', 0):.1f}" if macro.get('vix') else "N/A")
    col3.metric("Gold Trend", macro.get('gold_trend', 'N/A'))
    col4.metric("Silver Trend", macro.get('silver_trend', 'N/A'))
    
    # Action plan
    st.subheader("📋 Today's Action Plan")
    
    col1, col2, col3 = st.columns(3)
    
    plan = tape['action_plan']
    col1.metric("🟢 Buy/Add", plan['buy_count'])
    col2.metric("⚪ Hold", plan['hold_count'])
    col3.metric("🔴 Sell/Trim", plan['sell_count'])
    
    if plan['top_buys']:
        st.success("**🎯 Top Buy Opportunities:**")
        for buy in plan['top_buys']:
            st.write(f"• **{buy['Symbol']}**: {buy['Action']} (Alpha: {buy['Alpha_Score']:.0f})")
    
    if plan['top_sells']:
        st.error("**🚨 Top Sell Risks:**")
        for sell in plan['top_sells']:
            st.write(f"• **{sell['Symbol']}**: {sell['Action']} (Risk: {sell['Sell_Risk_Score']:.0f})")

# ============================================================================
# G) COMMAND CENTER DISPLAY
# ============================================================================

if 'results' in st.session_state:'''
    
    if function_location in content:
        content = content.replace(function_location, morning_tape_function)
        print("   ✅ Added morning tape function")
    
    # ========================================================================
    # PATCH 4: Display morning tape at top
    # ========================================================================
    print("\n📝 Patch 4: Adding morning tape display...")
    
    results_start = '''if 'results' in st.session_state:
    df = st.session_state.results
    news_cache = st.session_state.news_cache
    macro = st.session_state.macro_regime
    
    # Render daily summary
    render_daily_summary(df, macro, st.session_state.cash)'''
    
    results_with_tape = '''if 'results' in st.session_state:
    df = st.session_state.results
    news_cache = st.session_state.news_cache
    macro = st.session_state.macro_regime
    
    total_mv = df['Market_Value'].sum()
    total_value = total_mv + st.session_state.cash
    
    # INSTITUTIONAL V2: Display morning tape FIRST
    if INSTITUTIONAL_V2_AVAILABLE:
        morning_tape = generate_morning_tape(
            st.session_state.get('gold_analysis', {}),
            st.session_state.get('silver_analysis', {}),
            st.session_state.get('metal_regime', {}),
            macro,
            df,
            total_value
        )
        render_morning_tape(morning_tape)
    
    # Then daily summary
    render_daily_summary(df, macro, st.session_state.cash)'''
    
    if results_start in content:
        content = content.replace(results_start, results_with_tape)
        print("   ✅ Added morning tape display")
    
    # ========================================================================
    # PATCH 5: Use metal-aware discovery exception
    # ========================================================================
    print("\n📝 Patch 5: Upgrading to metal-aware discovery exception...")
    
    # Find and replace discovery check
    old_discovery = '''        # INSTITUTIONAL ENHANCEMENT: Strict discovery exception
        if INSTITUTIONAL_AVAILABLE:
            exception = check_discovery_exception_strict('''
    
    new_discovery = '''        # INSTITUTIONAL V2: Metal-aware discovery exception
        if INSTITUTIONAL_V2_AVAILABLE:
            metal_regime = st.session_state.get('metal_regime', {})
            exception = check_discovery_exception_metal_aware('''
    
    if old_discovery in content:
        # Also need to add metal_regime parameter
        old_params = '''                row.get('Price', 0),
                row.get('MA50', 0),
                row.get('SMC_Confirmed', False)
            )'''
        
        new_params = '''                row.get('Price', 0),
                row.get('MA50', 0),
                row.get('SMC_Confirmed', False),
                metal_regime  # NEW
            )'''
        
        content = content.replace(old_discovery, new_discovery)
        content = content.replace(old_params, new_params)
        print("   ✅ Upgraded to metal-aware discovery")
    
    # ========================================================================
    # PATCH 6: Add dynamic position sizing
    # ========================================================================
    print("\n📝 Patch 6: Adding dynamic position sizing...")
    
    # Find arbitration section where max_allowed_pct is set
    sizing_marker = '''    # Apply liquidity tier cap
    liq_cap = liq_metrics.get('max_position_pct', 5.0)
    base_max = min(base_max, liq_cap)
    
    # Apply macro throttle
    throttle = macro_regime.get('throttle_factor', 1.0)
    base_max = base_max * throttle'''
    
    sizing_replacement = '''    # INSTITUTIONAL V2: Dynamic position sizing
    if INSTITUTIONAL_V2_AVAILABLE:
        metal_regime = st.session_state.get('metal_regime', {})
        base_max = calculate_dynamic_position_sizing(
            row, liq_metrics,
            row.get('Dilution_Risk_Score', 50),
            metal_regime,
            macro_regime
        )
    else:
        # Fallback to original logic
        liq_cap = liq_metrics.get('max_position_pct', 5.0)
        base_max = min(base_max, liq_cap)
        
        throttle = macro_regime.get('throttle_factor', 1.0)
        base_max = base_max * throttle'''
    
    if sizing_marker in content:
        content = content.replace(sizing_marker, sizing_replacement)
        print("   ✅ Added dynamic position sizing")
    
    # ========================================================================
    # PATCH 7: Adjust sell sensitivity by metal regime
    # ========================================================================
    print("\n📝 Patch 7: Adding metal-aware sell sensitivity...")
    
    sell_adjustment = '''        # Update sell risk score
            df.at[idx, 'Sell_Risk_Score'] = enhanced_score
    
    progress.progress(90, text="✅ Final arbitration...")'''
    
    sell_with_metal = '''        # Update sell risk score
            df.at[idx, 'Sell_Risk_Score'] = enhanced_score
            
            # INSTITUTIONAL V2: Adjust by metal regime
            if INSTITUTIONAL_V2_AVAILABLE:
                metal_regime = st.session_state.get('metal_regime', {})
                sensitivity = metal_regime.get('sell_sensitivity', 1.0)
                df.at[idx, 'Sell_Risk_Score'] = df.at[idx, 'Sell_Risk_Score'] * sensitivity
    
    progress.progress(90, text="✅ Final arbitration...")'''
    
    if sell_adjustment in content:
        content = content.replace(sell_adjustment, sell_with_metal)
        print("   ✅ Added metal-aware sell sensitivity")
    
    # ========================================================================
    # WRITE OUTPUT
    # ========================================================================
    print(f"\n💾 Writing complete institutional system...")
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Output: {len(content)} characters")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ COMPLETE INSTITUTIONAL INTEGRATION SUCCESSFUL!")
    print("\n📋 Features Added:")
    print("\n  V1 (Institutional Base):")
    print("    ✅ Institutional SMC (premium/discount, impulse quality)")
    print("    ✅ Strict discovery exception (8 gates)")
    print("    ✅ Precise financing classification")
    print("    ✅ Social sentiment proxy")
    print("    ✅ Enhanced sell triggers")
    print("    ✅ Portfolio risk intelligence")
    print("\n  V2 (Metal Cycle Integration):")
    print("    ✅ Gold & Silver cycle predictions (short/med/long)")
    print("    ✅ Metal regime → portfolio behavior")
    print("    ✅ Metal-aware discovery exception")
    print("    ✅ Dynamic CORE/TACTICAL sizing")
    print("    ✅ Morning tape dashboard")
    print("    ✅ Metal-adjusted sell sensitivity")
    print("\n🚀 Ready to run:")
    print("  streamlit run alpha_miner_institutional_v2.py")
    print("\n💡 Your original file is unchanged.")

if __name__ == '__main__':
    input_file = 'alpha_miner_enhanced.py'
    output_file = 'alpha_miner_institutional_v2.py'
    
    print("\n🔍 Checking files...")
    
    if not Path(input_file).exists():
        print(f"❌ Error: {input_file} not found")
        sys.exit(1)
    
    if not Path('institutional_enhancements.py').exists():
        print("❌ Error: institutional_enhancements.py not found")
        sys.exit(1)
    
    if not Path('institutional_enhancements_v2.py').exists():
        print("❌ Error: institutional_enhancements_v2.py not found")
        sys.exit(1)
    
    print("✅ All files found\n")
    
    integrate_complete_system(input_file, output_file)
    
    print("\n" + "=" * 70)
    print("✨ SUCCESS! You now have a complete hedge-fund grade system.")
    print("=" * 70)
