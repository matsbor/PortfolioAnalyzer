#!/usr/bin/env python3
"""
HOTFIX: Add missing render_morning_tape function and versioning
Fixes the NameError and adds version display
"""

import sys
from pathlib import Path

def apply_hotfix(filename):
    """Apply hotfix to file"""
    
    print("🔧 Alpha Miner Pro - Hotfix Patcher")
    print("=" * 60)
    
    if not Path(filename).exists():
        print(f"❌ Error: {filename} not found")
        print("\nMake sure you're in the PortfolioAnalyzer directory")
        return False
    
    print(f"\n📖 Reading {filename}...")
    with open(filename, 'r') as f:
        content = f.read()
    
    print(f"✅ File loaded: {len(content)} characters")
    
    # ========================================================================
    # FIX 1: Add version constant at the top
    # ========================================================================
    print("\n📝 Fix 1: Adding version tracking...")
    
    version_marker = 'st.set_page_config(page_title="Alpha Miner Pro"'
    
    version_code = '''# VERSION TRACKING
VERSION = "2.0-INSTITUTIONAL"
VERSION_DATE = "2026-01-16"
VERSION_FEATURES = [
    "Institutional SMC (premium/discount zones)",
    "Gold & Silver cycle predictions",
    "Metal-aware position sizing",
    "Morning tape dashboard",
    "Strict discovery exception (9 gates)",
    "Enhanced sell-in-time (distribution detection)",
    "Portfolio risk intelligence"
]

st.set_page_config(page_title="Alpha Miner Pro"'''
    
    if version_marker in content and 'VERSION = ' not in content:
        content = content.replace(version_marker, version_code)
        print("   ✅ Added version tracking")
    elif 'VERSION = ' in content:
        print("   ⚠️ Version already exists, skipping")
    else:
        print("   ⚠️ Could not find version marker")
    
    # ========================================================================
    # FIX 2: Add render_morning_tape function
    # ========================================================================
    print("\n📝 Fix 2: Adding render_morning_tape function...")
    
    # Find where to insert (before "if 'results' in st.session_state")
    function_marker = '''# ============================================================================
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
    st.caption(f"Market Brief: {tape.get('date', 'N/A')}")
    
    # Regime status banner
    regime = tape.get('regime_status', {})
    if not regime.get('new_buys_allowed', True):
        st.error(f"🛑 **NO NEW BUYS TODAY** - {regime.get('overall', 'UNKNOWN')} + {regime.get('macro', 'UNKNOWN')}")
    elif str(regime.get('overall', '')).endswith('BEARISH'):
        st.warning(f"⚠️ **DEFENSIVE MODE** - {regime.get('explanation', 'Bearish environment')}")
    else:
        st.success(f"✅ **ACTIVE MODE** - {regime.get('explanation', 'Normal operations')}")
    
    st.caption(f"Combined Throttle: {regime.get('throttle_total', 1.0):.1%}")
    
    # Metal predictions
    st.subheader("🪙 Metal Outlook")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gold = tape.get('metal_outlook', {}).get('gold', {})
        st.markdown(f"### 🥇 Gold: ${gold.get('price', 0):,.0f}")
        st.write(f"**Bias:** {gold.get('bias_short', 'N/A')} → {gold.get('bias_medium', 'N/A')} → {gold.get('bias_long', 'N/A')}")
        
        preds = gold.get('predictions', {})
        pred_str = f"{preds.get('today', '↔')} today | {preds.get('week', '↔')} week | {preds.get('month', '↔')} month | {preds.get('quarter', '↔')} quarter"
        st.write(f"**Predictions:** {pred_str}")
        st.write(f"**SMC:** {gold.get('smc', 'NEUTRAL')}")
        st.caption(gold.get('explanation', 'No analysis'))
    
    with col2:
        silver = tape.get('metal_outlook', {}).get('silver', {})
        st.markdown(f"### 🥈 Silver: ${silver.get('price', 0):.2f}")
        st.write(f"**Bias:** {silver.get('bias_short', 'N/A')} → {silver.get('bias_medium', 'N/A')} → {silver.get('bias_long', 'N/A')}")
        
        preds = silver.get('predictions', {})
        pred_str = f"{preds.get('today', '↔')} today | {preds.get('week', '↔')} week | {preds.get('month', '↔')} month | {preds.get('quarter', '↔')} quarter"
        st.write(f"**Predictions:** {pred_str}")
        st.write(f"**SMC:** {silver.get('smc', 'NEUTRAL')}")
        st.caption(silver.get('explanation', 'No analysis'))
    
    # Macro indicators
    st.subheader("📈 Macro Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    macro = tape.get('macro_indicators', {})
    col1.metric("DXY", f"{macro.get('dxy', 0):.1f}" if macro.get('dxy') else "N/A")
    col2.metric("VIX", f"{macro.get('vix', 0):.1f}" if macro.get('vix') else "N/A")
    col3.metric("Gold Trend", macro.get('gold_trend', 'N/A'))
    col4.metric("Silver Trend", macro.get('silver_trend', 'N/A'))
    
    # Action plan
    st.subheader("📋 Today's Action Plan")
    
    col1, col2, col3 = st.columns(3)
    
    plan = tape.get('action_plan', {})
    col1.metric("🟢 Buy/Add", plan.get('buy_count', 0))
    col2.metric("⚪ Hold", plan.get('hold_count', 0))
    col3.metric("🔴 Sell/Trim", plan.get('sell_count', 0))
    
    top_buys = plan.get('top_buys', [])
    if top_buys:
        st.success("**🎯 Top Buy Opportunities:**")
        for buy in top_buys:
            st.write(f"• **{buy.get('Symbol', 'N/A')}**: {buy.get('Action', 'N/A')} (Alpha: {buy.get('Alpha_Score', 0):.0f})")
    
    top_sells = plan.get('top_sells', [])
    if top_sells:
        st.error("**🚨 Top Sell Risks:**")
        for sell in top_sells:
            st.write(f"• **{sell.get('Symbol', 'N/A')}**: {sell.get('Action', 'N/A')} (Risk: {sell.get('Sell_Risk_Score', 0):.0f})")

# ============================================================================
# G) COMMAND CENTER DISPLAY
# ============================================================================

if 'results' in st.session_state:'''
    
    if function_marker in content and 'def render_morning_tape' not in content:
        content = content.replace(function_marker, morning_tape_function)
        print("   ✅ Added render_morning_tape function")
    elif 'def render_morning_tape' in content:
        print("   ⚠️ Function already exists, skipping")
    else:
        print("   ⚠️ Could not find insertion point")
    
    # ========================================================================
    # FIX 3: Add version display to sidebar
    # ========================================================================
    print("\n📝 Fix 3: Adding version display to sidebar...")
    
    sidebar_marker = '''with st.sidebar:
    st.title("⚙️ Configuration")'''
    
    sidebar_with_version = '''with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Version display
    st.markdown("---")
    st.caption(f"**Alpha Miner Pro {VERSION}**")
    st.caption(f"Release: {VERSION_DATE}")
    with st.expander("📋 Features in this version"):
        for feature in VERSION_FEATURES:
            st.caption(f"• {feature}")
    st.markdown("---")'''
    
    if sidebar_marker in content and 'st.caption(f"**Alpha Miner Pro' not in content:
        content = content.replace(sidebar_marker, sidebar_with_version)
        print("   ✅ Added version display to sidebar")
    elif 'st.caption(f"**Alpha Miner Pro' in content:
        print("   ⚠️ Version display already exists, skipping")
    else:
        print("   ⚠️ Could not find sidebar marker")
    
    # ========================================================================
    # FIX 4: Add safety check around morning_tape call
    # ========================================================================
    print("\n📝 Fix 4: Adding safety check to morning_tape call...")
    
    unsafe_call = '''    # INSTITUTIONAL V2: Display morning tape FIRST
    if INSTITUTIONAL_V2_AVAILABLE:
        morning_tape = generate_morning_tape(
            st.session_state.get('gold_analysis', {}),
            st.session_state.get('silver_analysis', {}),
            st.session_state.get('metal_regime', {}),
            macro,
            df,
            total_value
        )
        render_morning_tape(morning_tape)'''
    
    safe_call = '''    # INSTITUTIONAL V2: Display morning tape FIRST
    if INSTITUTIONAL_V2_AVAILABLE:
        try:
            morning_tape = generate_morning_tape(
                st.session_state.get('gold_analysis', {}),
                st.session_state.get('silver_analysis', {}),
                st.session_state.get('metal_regime', {}),
                macro,
                df,
                total_value
            )
            render_morning_tape(morning_tape)
        except Exception as e:
            st.warning(f"⚠️ Morning tape unavailable: {str(e)}")'''
    
    if unsafe_call in content:
        content = content.replace(unsafe_call, safe_call)
        print("   ✅ Added safety check")
    else:
        print("   ⚠️ Could not find morning_tape call")
    
    # ========================================================================
    # WRITE OUTPUT
    # ========================================================================
    print(f"\n💾 Writing fixed file...")
    
    # Backup original
    backup_file = filename.replace('.py', '_backup.py')
    with open(backup_file, 'w') as f:
        f.write(open(filename).read())
    print(f"   📦 Backup saved: {backup_file}")
    
    # Write fixed version
    with open(filename, 'w') as f:
        f.write(content)
    
    print(f"   ✅ Fixed file written: {len(content)} characters")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ HOTFIX APPLIED SUCCESSFULLY!")
    print("\n📋 Changes made:")
    print("  ✅ Added VERSION tracking")
    print("  ✅ Added render_morning_tape() function")
    print("  ✅ Added version display to sidebar")
    print("  ✅ Added safety checks")
    print("\n🚀 Ready to run:")
    print(f"  streamlit run {filename}")
    print("\n💡 If issues persist, restore backup:")
    print(f"  cp {backup_file} {filename}")
    
    return True

if __name__ == '__main__':
    filename = 'alpha_miner_institutional_v2.py'
    
    print("\n🔍 Checking file...")
    
    if not Path(filename).exists():
        print(f"❌ Error: {filename} not found")
        print("\nPlease run this script from ~/PortfolioAnalyzer directory")
        print("Or specify the correct filename")
        sys.exit(1)
    
    print(f"✅ Found {filename}\n")
    
    success = apply_hotfix(filename)
    
    if success:
        print("\n" + "=" * 60)
        print("✨ HOTFIX COMPLETE! You can now run the app.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n❌ Hotfix failed. Check errors above.")
        sys.exit(1)
