#!/bin/bash
# ALPHA MINER PRO - STARTUP CHECKLIST & COMMANDS
# Simple script to check everything and start the app

echo "🔍 ALPHA MINER PRO - STARTUP CHECK"
echo "=================================="
echo ""

# Check directory
echo "📁 Current directory:"
pwd
echo ""

# Check for all required files
echo "📋 Checking required files..."
echo ""

files_ok=true

if [ -f "alpha_miner_enhanced.py" ]; then
    echo "✅ alpha_miner_enhanced.py (original base)"
else
    echo "❌ alpha_miner_enhanced.py - MISSING"
    files_ok=false
fi

if [ -f "institutional_enhancements.py" ]; then
    echo "✅ institutional_enhancements.py (v1 - institutional base)"
else
    echo "❌ institutional_enhancements.py - MISSING"
    files_ok=false
fi

if [ -f "institutional_enhancements_v2.py" ]; then
    echo "✅ institutional_enhancements_v2.py (v2 - metal cycle)"
else
    echo "❌ institutional_enhancements_v2.py - MISSING"
    files_ok=false
fi

if [ -f "integrate_institutional_v2.py" ]; then
    echo "✅ integrate_institutional_v2.py (integration script)"
else
    echo "❌ integrate_institutional_v2.py - MISSING"
    files_ok=false
fi

echo ""

if [ "$files_ok" = false ]; then
    echo "❌ MISSING FILES!"
    echo ""
    echo "You need to download these files from Claude:"
    echo "  1. institutional_enhancements.py"
    echo "  2. institutional_enhancements_v2.py"
    echo "  3. integrate_institutional_v2.py"
    echo ""
    echo "Make sure they're all in: ~/PortfolioAnalyzer/"
    exit 1
fi

echo "✅ All files present!"
echo ""

# Check if integrated version exists
if [ -f "alpha_miner_institutional_v2.py" ]; then
    echo "✅ Integrated version already exists!"
    echo ""
    echo "🚀 Ready to run:"
    echo "   streamlit run alpha_miner_institutional_v2.py"
    echo ""
    read -p "Run it now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        streamlit run alpha_miner_institutional_v2.py
    fi
else
    echo "⚙️  Integrated version not found. Need to integrate first."
    echo ""
    echo "🔧 Running integration now..."
    echo ""
    python3 integrate_institutional_v2.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Integration successful!"
        echo ""
        echo "🚀 Ready to run:"
        echo "   streamlit run alpha_miner_institutional_v2.py"
        echo ""
        read -p "Run it now? (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            streamlit run alpha_miner_institutional_v2.py
        fi
    else
        echo ""
        echo "❌ Integration failed. Check errors above."
        exit 1
    fi
fi
