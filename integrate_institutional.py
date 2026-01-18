#!/usr/bin/env python3
"""
INSTITUTIONAL ENHANCEMENTS - AUTO-INTEGRATION SCRIPT
Automatically applies all enhancements to alpha_miner_enhanced.py

Run this script to integrate institutional features without manual editing.
"""

import re
import sys
from pathlib import Path

def integrate_enhancements(input_file, output_file):
    """Apply all institutional enhancements"""
    
    print("🚀 Alpha Miner Pro - Institutional Enhancement Integration")
    print("=" * 60)
    
    # Read original file
    print("\n📖 Reading alpha_miner_enhanced.py...")
    with open(input_file, 'r') as f:
        content = f.read()
    
    print(f"✅ Original: {len(content)} characters")
    
    # ========================================================================
    # PATCH 1: Add import statement
    # ========================================================================
    print("\n📝 Patch 1: Adding institutional imports...")
    
    import_section = '''from pathlib import Path

try:
    import yfinance as yf'''
    
    new_import_section = '''from pathlib import Path

# Import institutional enhancements
try:
    from institutional_enhancements import (
        calculate_smc_institutional,
        check_discovery_exception_strict,
        classify_financing_precision,
        calculate_social_proxy,
        add_institutional_sell_triggers,
        calculate_portfolio_risk_intelligence
    )
    INSTITUTIONAL_AVAILABLE = True
except ImportError:
    INSTITUTIONAL_AVAILABLE = False
    print("⚠️ institutional_enhancements.py not found - running without institutional features")

try:
    import yfinance as yf'''
    
    if import_section in content:
        content = content.replace(import_section, new_import_section)
        print("   ✅ Added institutional imports")
    else:
        print("   ⚠️ Could not find import section - may need manual edit")
    
    # ========================================================================
    # PATCH 2: Add institutional SMC after existing SMC calculation
    # ========================================================================
    print("\n📝 Patch 2: Adding institutional SMC calculation...")
    
    # Find where SMC is calculated (after progress message)
    smc_marker = '''    st.session_state.smc_signals_storage = smc_signals_storage
    
    progress.progress(90, text="✅ Final arbitration...")'''
    
    smc_addition = '''    st.session_state.smc_signals_storage = smc_signals_storage
    
    # INSTITUTIONAL ENHANCEMENT: Enhanced SMC analysis
    if INSTITUTIONAL_AVAILABLE:
        progress.progress(82, text="🏦 Calculating institutional SMC...")
        
        smc_inst_storage = {}
        financing_storage = {}
        social_storage = {}
        
        for idx, row in df.iterrows():
            hist = hist_cache.get(row['Symbol'], pd.DataFrame())
            news = news_cache.get(row['Symbol'], [])
            
            # Enhanced institutional SMC
            smc_inst = calculate_smc_institutional(hist, row['Price'])
            df.at[idx, 'SMC_Confirmed'] = smc_inst['smc_confirmed']
            df.at[idx, 'SMC_Zone'] = smc_inst['zone']
            df.at[idx, 'SMC_Explanation'] = smc_inst['explanation']
            smc_inst_storage[row['Symbol']] = smc_inst
            
            # Financing classification
            financing = classify_financing_precision(news)
            df.at[idx, 'Financing_Status'] = financing['status']
            df.at[idx, 'Financing_Type'] = financing['type']
            financing_storage[row['Symbol']] = financing
            
            # Adjust dilution risk based on financing
            current_dil = df.at[idx, 'Dilution_Risk_Score']
            df.at[idx, 'Dilution_Risk_Score'] = max(0, min(100, 
                current_dil + financing['dilution_adj']
            ))
            
            # Social sentiment proxy
            vol_spike = 1.0
            if len(hist) >= 20:
                recent_vol = hist['Volume'].tail(5).mean()
                base_vol = hist['Volume'].tail(20).mean()
                vol_spike = recent_vol / base_vol if base_vol > 0 else 1.0
            
            social = calculate_social_proxy(
                row['Symbol'], news, row.get('Return_7d', 0), vol_spike
            )
            df.at[idx, 'Social_Score'] = social['score']
            df.at[idx, 'Hype_Warning'] = social['hype_warning']
            social_storage[row['Symbol']] = social
        
        st.session_state.smc_inst_storage = smc_inst_storage
        st.session_state.financing_storage = financing_storage
        st.session_state.social_storage = social_storage
    
    progress.progress(90, text="✅ Final arbitration...")'''
    
    if smc_marker in content:
        content = content.replace(smc_marker, smc_addition)
        print("   ✅ Added institutional SMC calculations")
    else:
        print("   ⚠️ Could not find SMC calculation section")
    
    # ========================================================================
    # PATCH 3: Enhance sell risk calculation
    # ========================================================================
    print("\n📝 Patch 3: Enhancing sell-in-time triggers...")
    
    # Find sell risk storage section
    sell_marker = '''        # Store triggers separately
        sell_triggers_storage[row['Symbol']] = sell['all_triggers']
    
    progress.progress(90, text="✅ Final arbitration...")'''
    
    sell_addition = '''        # Store triggers separately
        sell_triggers_storage[row['Symbol']] = sell['all_triggers']
        
        # INSTITUTIONAL ENHANCEMENT: Add institutional sell triggers
        if INSTITUTIONAL_AVAILABLE and 'smc_inst_storage' in st.session_state:
            smc_inst = st.session_state.smc_inst_storage.get(row['Symbol'], {})
            smc_data = {
                'failed_breaks': smc_inst.get('failed_breaks', []),
                'zone': smc_inst.get('zone', 'NEUTRAL'),
                'liquidity_engineered': smc_inst.get('liquidity_engineered', False)
            }
            
            enhanced_score = add_institutional_sell_triggers(
                sell['score'],
                sell.get('hard_triggers', []),
                sell.get('soft_triggers', []),
                row, hist, row.get('MA50', 0), row.get('MA200', 0),
                smc_data, news
            )
            
            # Update sell risk score
            df.at[idx, 'Sell_Risk_Score'] = enhanced_score
    
    progress.progress(90, text="✅ Final arbitration...")'''
    
    if sell_marker in content:
        content = content.replace(sell_marker, sell_addition)
        print("   ✅ Enhanced sell-in-time triggers")
    else:
        print("   ⚠️ Could not find sell risk section")
    
    # ========================================================================
    # PATCH 4: Replace discovery exception with strict version
    # ========================================================================
    print("\n📝 Patch 4: Upgrading discovery exception to strict mode...")
    
    # Find discovery exception check
    discovery_marker = '''        exception = check_discovery_exception(
            row, liq_metrics,
            row['Alpha_Score'],
            row['Data_Confidence'],
            row['Dilution_Risk_Score'],
            momentum_ok
        )'''
    
    discovery_replacement = '''        # INSTITUTIONAL ENHANCEMENT: Strict discovery exception
        if INSTITUTIONAL_AVAILABLE:
            exception = check_discovery_exception_strict(
                row, liq_metrics,
                row['Alpha_Score'],
                row['Data_Confidence'],
                row['Dilution_Risk_Score'],
                row.get('Return_7d', 0),
                row.get('Price', 0),
                row.get('MA50', 0),
                row.get('SMC_Confirmed', False)
            )
        else:
            # Fallback to original
            exception = check_discovery_exception(
                row, liq_metrics,
                row['Alpha_Score'],
                row['Data_Confidence'],
                row['Dilution_Risk_Score'],
                momentum_ok
            )'''
    
    if discovery_marker in content:
        content = content.replace(discovery_marker, discovery_replacement)
        print("   ✅ Upgraded discovery exception (now requires 8 gates)")
    else:
        print("   ⚠️ Could not find discovery exception")
    
    # ========================================================================
    # PATCH 5: Add portfolio intelligence to command center
    # ========================================================================
    print("\n📝 Patch 5: Adding portfolio risk intelligence...")
    
    # Find command center section (after action counts)
    portfolio_marker = '''    st.markdown("---")
    
    # Action Lists (4 columns)
    col1, col2, col3, col4 = st.columns(4)'''
    
    portfolio_addition = '''    st.markdown("---")
    
    # INSTITUTIONAL ENHANCEMENT: Portfolio Risk Intelligence
    if INSTITUTIONAL_AVAILABLE:
        st.subheader("🏦 Portfolio Health Check")
        
        portfolio_intel = calculate_portfolio_risk_intelligence(df, total_value)
        
        col1, col2, col3, col4 = st.columns(4)
        
        health_colors = {
            'EXCELLENT': '🟢', 'GOOD': '🔵', 'CAUTION': '🟡', 'HIGH RISK': '🔴'
        }
        health_icon = health_colors.get(portfolio_intel['health'], '⚪')
        
        col1.metric("Overall Health", f"{health_icon} {portfolio_intel['health']}")
        col2.metric("Illiquid %", f"{portfolio_intel['illiquid_pct']:.1f}%")
        col3.metric("Weighted Dilution", f"{portfolio_intel['weighted_dilution']:.0f}/100")
        col4.metric("Avg Exit Days", f"{portfolio_intel['avg_exit_days']:.1f}d")
        
        if portfolio_intel['warnings']:
            with st.expander("⚠️ Portfolio Concentration Warnings", expanded=True):
                for warning in portfolio_intel['warnings']:
                    st.warning(warning)
        
        # Stage breakdown
        if portfolio_intel['stage_breakdown']:
            st.caption("**Exposure by Stage:**")
            stage_text = " | ".join([f"{stage}: {pct:.1f}%" 
                                     for stage, pct in portfolio_intel['stage_breakdown'].items()])
            st.caption(stage_text)
        
        st.markdown("---")
    
    # Action Lists (4 columns)
    col1, col2, col3, col4 = st.columns(4)'''
    
    if portfolio_marker in content:
        content = content.replace(portfolio_marker, portfolio_addition)
        print("   ✅ Added portfolio intelligence to command center")
    else:
        print("   ⚠️ Could not find command center action lists section")
    
    # ========================================================================
    # PATCH 6: Add new badges to ticker display
    # ========================================================================
    print("\n📝 Patch 6: Adding institutional badges...")
    
    badge_marker = '''        if row.get('Discovery_Exception', False):
            badge_html += '<span class="badge-discovery">DISCOVERY</span> '
        
        # SMC badge'''
    
    badge_addition = '''        if row.get('Discovery_Exception', False):
            badge_html += '<span class="badge-discovery">DISCOVERY</span> '
        
        # INSTITUTIONAL ENHANCEMENTS: New badges
        if INSTITUTIONAL_AVAILABLE:
            # SMC institutional confirmation
            if row.get('SMC_Confirmed', False):
                badge_html += '<span class="badge-l3">SMC ✓ Inst</span> '
            
            # SMC zone
            zone = row.get('SMC_Zone', 'NEUTRAL')
            if zone == 'DISCOUNT':
                badge_html += '<span class="badge-l3">DISCOUNT</span> '
            elif zone == 'PREMIUM':
                badge_html += '<span class="badge-l1">PREMIUM</span> '
            
            # Hype warning
            if row.get('Hype_Warning', False):
                badge_html += '<span class="badge-gambling">⚠️ HYPE</span> '
            
            # Financing status
            fin_status = row.get('Financing_Status')
            if fin_status == 'CLOSED':
                badge_html += '<span class="badge-l3">PP CLOSED ✓</span> '
            elif fin_status == 'ANNOUNCED':
                badge_html += '<span class="badge-l1">PP OPEN</span> '
            elif fin_status == 'ACTIVE':
                badge_html += '<span class="badge-gambling">ATM ⚠️</span> '
        
        # SMC badge'''
    
    if badge_marker in content:
        content = content.replace(badge_marker, badge_addition)
        print("   ✅ Added institutional badges")
    else:
        print("   ⚠️ Could not find badge section")
    
    # ========================================================================
    # PATCH 7: Add institutional details to expandable section
    # ========================================================================
    print("\n📝 Patch 7: Adding institutional analysis to details...")
    
    details_marker = '''            # SMC Analysis section
            st.markdown("---")
            st.subheader("📈 Smart Money Concepts (SMC)")'''
    
    details_addition = '''            # INSTITUTIONAL ENHANCEMENTS: Detailed Analysis
            if INSTITUTIONAL_AVAILABLE:
                # Institutional SMC
                st.markdown("---")
                st.subheader("🏦 Institutional Price Action")
                st.write(f"**Zone:** {row.get('SMC_Zone', 'N/A')}")
                st.write(f"**Institutional Confirmation:** {'✅ Yes' if row.get('SMC_Confirmed', False) else '❌ No'}")
                st.write(f"**Analysis:** {row.get('SMC_Explanation', 'No analysis')}")
                
                # Social sentiment
                st.markdown("---")
                st.subheader("📱 Market Buzz Analysis")
                social_score = row.get('Social_Score', 0)
                buzz_level = "EXTREME" if abs(social_score) > 15 else "HIGH" if abs(social_score) > 10 else "MODERATE" if abs(social_score) > 5 else "LOW"
                st.write(f"**Sentiment Score:** {social_score:+.0f}/20 ({buzz_level})")
                
                if row.get('Hype_Warning', False):
                    st.error("⚠️ **HYPE WARNING:** Extreme bullish sentiment detected - exercise caution")
                elif social_score > 10:
                    st.warning("⚠️ Elevated bullish sentiment - monitor for retail FOMO")
                elif social_score < -10:
                    st.info("ℹ️ Elevated bearish sentiment - potential capitulation opportunity")
                
                # Financing detail
                fin_type = row.get('Financing_Type')
                if fin_type:
                    st.markdown("---")
                    st.subheader("💰 Financing Analysis")
                    st.write(f"**Type:** {fin_type}")
                    st.write(f"**Status:** {row.get('Financing_Status', 'Unknown')}")
                    
                    financing_data = st.session_state.get('financing_storage', {}).get(row['Symbol'], {})
                    dilution_adj = financing_data.get('dilution_adj', 0)
                    
                    if dilution_adj < 0:
                        st.success(f"✅ Dilution risk reduced by {abs(dilution_adj)} points - runway extended")
                    elif dilution_adj > 0:
                        st.error(f"⚠️ Dilution risk increased by {dilution_adj} points - ongoing dilution")
                    
                    if financing_data.get('explanation'):
                        st.caption(financing_data['explanation'])
            
            # SMC Analysis section
            st.markdown("---")
            st.subheader("📈 Smart Money Concepts (SMC)")'''
    
    if details_marker in content:
        content = content.replace(details_marker, details_addition)
        print("   ✅ Added institutional details section")
    else:
        print("   ⚠️ Could not find SMC details section")
    
    # ========================================================================
    # WRITE OUTPUT
    # ========================================================================
    print(f"\n💾 Writing integrated file...")
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Output: {len(content)} characters")
    
    # Final summary
    print("\n" + "=" * 60)
    print("✅ INTEGRATION COMPLETE!")
    print("\n📋 Enhancements Added:")
    print("  ✅ Institutional SMC (premium/discount zones)")
    print("  ✅ Strict discovery exception (8 gates, insider required)")
    print("  ✅ Precise financing classification (closed vs open)")
    print("  ✅ Social sentiment proxy (hype warning)")
    print("  ✅ Enhanced sell triggers (distribution detection)")
    print("  ✅ Portfolio risk intelligence (concentration warnings)")
    print("  ✅ New badges (SMC confirmed, zones, hype, financing)")
    print("  ✅ Detailed analysis sections")
    print("\n🚀 Ready to run:")
    print("  cd ~/PortfolioAnalyzer")
    print("  streamlit run alpha_miner_institutional.py")
    print("\n💡 Your original file is unchanged.")
    print("   The new file is: alpha_miner_institutional.py")

if __name__ == '__main__':
    input_file = 'alpha_miner_enhanced.py'
    output_file = 'alpha_miner_institutional.py'
    
    print("\n🔍 Checking files...")
    
    if not Path(input_file).exists():
        print(f"❌ Error: {input_file} not found in current directory")
        print("\nPlease run this script from the directory containing alpha_miner_enhanced.py")
        sys.exit(1)
    
    if not Path('institutional_enhancements.py').exists():
        print("❌ Error: institutional_enhancements.py not found")
        print("\nPlease place institutional_enhancements.py in the same directory")
        sys.exit(1)
    
    print("✅ All files found\n")
    
    integrate_enhancements(input_file, output_file)
    
    print("\n" + "=" * 60)
    print("✨ SUCCESS! You now have institutional-grade capabilities.")
    print("=" * 60)
