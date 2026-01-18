#!/usr/bin/env python3
import streamlit as st
import pandas as pd
import re

st.set_page_config(page_title="Alpha Miner - Schwab", layout="wide")
st.title("💎 Alpha Miner - Schwab Copy/Paste")

paste_text = st.text_area("Paste Schwab data here:", height=400)

if st.button("🚀 Analyze", type="primary") and paste_text:
    lines = [l.strip() for l in paste_text.split('\n') if l.strip()]
    
    positions = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Look for symbol (all caps, alone on line)
        if re.match(r'^[A-Z]{3,6}$', line):
            symbol = line
            
            # Look ahead for the data line with Quantity, Price, etc
            for j in range(i+1, min(i+10, len(lines))):
                data_line = lines[j]
                
                # Look for pattern: Quantity###Price$###...Market Value$###...Cost Basis$###
                qty_match = re.search(r'Quantity([\d,]+)', data_line)
                price_match = re.search(r'Price\$([\d.]+)', data_line)
                mv_match = re.search(r'Market Value\$([\d,]+\.?\d*)', data_line)
                cb_match = re.search(r'Cost Basis\$([\d,]+\.?\d*)', data_line)
                
                if qty_match and mv_match and cb_match:
                    qty = float(qty_match.group(1).replace(',', ''))
                    price = float(price_match.group(1)) if price_match else 0
                    mv = float(mv_match.group(1).replace(',', ''))
                    cb = float(cb_match.group(1).replace(',', ''))
                    
                    positions.append({
                        'Symbol': symbol,
                        'Quantity': qty,
                        'Price': price,
                        'Market Value': mv,
                        'Cost Basis': cb,
                        'Gain/Loss': mv - cb,
                        'Return %': ((mv - cb) / cb * 100) if cb > 0 else 0
                    })
                    break
            
            i = j + 1
        else:
            i += 1
    
    if positions:
        df = pd.DataFrame(positions)
        st.success(f"✅ Found {len(df)} positions")
        
        total_value = df['Market Value'].sum()
        total_gain = df['Gain/Loss'].sum()
        total_cb = df['Cost Basis'].sum()
        total_return = (total_gain / total_cb * 100) if total_cb > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Value", f"${total_value:,.0f}")
        col2.metric("Total Return", f"${total_gain:+,.0f}", f"{total_return:+.1f}%")
        col3.metric("Positions", len(df))
        
        st.markdown("---")
        st.dataframe(df.style.format({
            'Price': '${:.4f}',
            'Market Value': '${:,.0f}',
            'Cost Basis': '${:,.0f}',
            'Gain/Loss': '${:+,.0f}',
            'Return %': '{:+.1f}%'
        }), use_container_width=True)
        
        st.markdown("### 🟢 Winners")
        for _, row in df.nlargest(3, 'Return %').iterrows():
            st.success(f"**{row['Symbol']}**: {row['Return %']:+.1f}% (${row['Gain/Loss']:+,.0f})")
        
        st.markdown("### 🔴 Losers")  
        for _, row in df.nsmallest(3, 'Return %').iterrows():
            st.error(f"**{row['Symbol']}**: {row['Return %']:+.1f}% (${row['Gain/Loss']:+,.0f})")
        
        # Download
        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "portfolio.csv", "text/csv")
    else:
        st.error("❌ Could not parse. Please paste the ENTIRE Schwab positions page (Cmd+A, Cmd+C)")
        st.info("Debug: Paste a sample here in chat and I'll fix the parser")
