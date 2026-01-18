#!/usr/bin/env python3
"""
ALPHA MINER COMPLETE - DAILY USE VERSION
Upload your portfolio as CSV, get instant analysis with free real-time data

HOW TO USE:
1. Export portfolio from broker as CSV
2. Place CSV file in same folder as this script
3. Run: python3 alpha_miner_complete_daily.py portfolio.csv
4. Get instant analysis with live prices

DATA SOURCES (ALL FREE):
- yfinance: 15-minute delayed prices (FREE, no API key)
- Alpha Vantage: Daily data (FREE, requires API key)
- Portfolio data: YOUR CSV file
"""

import sys
import datetime
import pandas as pd
import numpy as np
from pathlib import Path

# Try to import yfinance (free, no API key needed)
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️  yfinance not installed. Install with: pip install yfinance")

# Configuration
CACHE_PRICES_MINUTES = 15  # Cache prices for 15 minutes
price_cache = {}
cache_timestamp = {}

# ============================================================================
# CSV PORTFOLIO LOADER
# ============================================================================

def load_portfolio_from_csv(filepath: str) -> pd.DataFrame:
    """
    Load portfolio from CSV
    
    Expected CSV format (can be in any order):
    Symbol, Quantity, CostBasis
    
    OR if your broker provides more:
    Symbol, Quantity, Price, MarketValue, CostBasis, DayChange%
    
    Minimum required: Symbol, Quantity, CostBasis
    """
    try:
        df = pd.read_csv(filepath)
        
        # Clean column names (remove spaces, lowercase)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Handle different naming conventions
        # costbasis -> cost_basis, etc
        column_mappings = {
            'costbasis': 'cost_basis',
            'cost': 'cost_basis',
            'basis': 'cost_basis',
            'qty': 'quantity',
            'shares': 'quantity',
            'ticker': 'symbol',
            'stock': 'symbol'
        }
        
        for old_name, new_name in column_mappings.items():
            if old_name in df.columns and new_name not in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        # Check required columns
        required = ['symbol', 'quantity', 'cost_basis']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            print(f"❌ ERROR: Missing required columns: {missing}")
            print(f"📋 Available columns: {list(df.columns)}")
            print(f"\n💡 CSV must have at least: Symbol, Quantity, CostBasis")
            sys.exit(1)
        
        print(f"✅ Loaded {len(df)} positions from {filepath}")
        return df
    
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {filepath}")
        print(f"\n💡 Make sure your CSV file is in the same folder as this script")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ ERROR loading CSV: {e}")
        sys.exit(1)


def create_sample_csv():
    """
    Create a sample CSV file for user reference
    """
    sample_data = """Symbol,Quantity,CostBasis,Stage,Metal,Country
AG,638,7594.99,Producer,Silver,Mexico
AGXPF,32342,24015.26,Explorer,Silver,Peru
WRLGF,25929,18833.21,Explorer,Gold,Canada
JAGGF,2965,14558.14,Producer,Gold,Brazil
TSKFF,13027,13857.41,Explorer,Gold,Canada
SMDRF,7072,6939.18,Developer,Silver,Mexico
GSVRF,9049,2415.31,Producer,Silver,Mexico
ITRG,2072,7838.13,Explorer,Gold,Canada
SMAGF,8335,9928.39,Developer,Gold,Unknown
LOMLF,24557,5006.76,Explorer,Gold,Fiji
LUCMF,7079,8550.49,Explorer,Gold,Canada
BORMF,5172,5540.99,Explorer,Gold,Canada
EXNRF,11749,3242.97,Developer,Silver,Mexico
JZRIF,19841,5959.25,Explorer,Gold,Canada"""
    
    with open('portfolio_sample.csv', 'w') as f:
        f.write(sample_data)
    
    print("✅ Created portfolio_sample.csv")
    print("   Edit this file with your actual positions")


# ============================================================================
# FREE PRICE DATA (15-minute delay)
# ============================================================================

def get_live_price(symbol: str) -> dict:
    """
    Get current price using yfinance (FREE, 15-min delay)
    
    Returns: {
        'price': float,
        'day_change_pct': float,
        'volume': int,
        'last_update': datetime
    }
    """
    # Check cache (refresh every 15 minutes)
    now = datetime.datetime.now()
    if symbol in price_cache:
        cache_age = (now - cache_timestamp[symbol]).seconds / 60
        if cache_age < CACHE_PRICES_MINUTES:
            return price_cache[symbol]
    
    if not YFINANCE_AVAILABLE:
        return {
            'price': 0,
            'day_change_pct': 0,
            'volume': 0,
            'last_update': now,
            'error': 'yfinance not installed'
        }
    
    try:
        ticker = yf.Ticker(symbol)
        
        # Get current data
        info = ticker.info
        hist = ticker.history(period="5d")
        
        if hist.empty:
            print(f"⚠️  No data for {symbol}")
            return {
                'price': 0,
                'day_change_pct': 0,
                'volume': 0,
                'last_update': now,
                'error': 'No data available'
            }
        
        current_price = hist['Close'].iloc[-1]
        prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        day_change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close > 0 else 0
        volume = hist['Volume'].iloc[-1]
        
        result = {
            'price': float(current_price),
            'day_change_pct': float(day_change_pct),
            'volume': int(volume),
            'last_update': now
        }
        
        # Cache it
        price_cache[symbol] = result
        cache_timestamp[symbol] = now
        
        return result
    
    except Exception as e:
        print(f"⚠️  Error fetching {symbol}: {e}")
        return {
            'price': 0,
            'day_change_pct': 0,
            'volume': 0,
            'last_update': now,
            'error': str(e)
        }


def get_all_prices(symbols: list) -> dict:
    """
    Get prices for all symbols
    Shows progress bar
    """
    print(f"\n📊 Fetching live prices for {len(symbols)} stocks...")
    prices = {}
    
    for i, symbol in enumerate(symbols, 1):
        print(f"   [{i}/{len(symbols)}] {symbol}...", end='\r')
        prices[symbol] = get_live_price(symbol)
    
    print(f"✅ Prices updated (15-min delayed, free)        ")
    return prices


# ============================================================================
# SIMPLE ANALYZER
# ============================================================================

class QuickAnalyzer:
    """
    Simplified analyzer for multiple daily runs
    Focuses on most critical checks
    """
    
    def __init__(self, portfolio_df: pd.DataFrame, prices: dict, cash: float = 0):
        self.df = portfolio_df
        self.prices = prices
        self.cash = cash
        self.date = datetime.datetime.now()
        
        # Calculate current values
        self.calculate_current_values()
    
    def calculate_current_values(self):
        """Update portfolio with current prices"""
        self.df['current_price'] = self.df['symbol'].map(lambda s: self.prices.get(s, {}).get('price', 0))
        self.df['day_change_pct'] = self.df['symbol'].map(lambda s: self.prices.get(s, {}).get('day_change_pct', 0))
        self.df['market_value'] = self.df['quantity'] * self.df['current_price']
        self.df['gain_loss'] = self.df['market_value'] - self.df['cost_basis']
        self.df['gain_loss_pct'] = (self.df['gain_loss'] / self.df['cost_basis'] * 100).fillna(0)
        
        total_mv = self.df['market_value'].sum()
        self.df['pct_portfolio'] = (self.df['market_value'] / total_mv * 100).fillna(0)
        
        self.total_equity = total_mv
        self.total_value = total_mv + self.cash
    
    def quick_analysis(self) -> str:
        """
        Quick analysis report - optimized for multiple daily checks
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"ALPHA MINER - QUICK ANALYSIS")
        lines.append(f"{self.date.strftime('%Y-%m-%d %H:%M:%S')} (Data: 15-min delayed)")
        lines.append("=" * 80)
        lines.append("")
        
        # Portfolio summary
        total_gain = self.df['gain_loss'].sum()
        total_cb = self.df['cost_basis'].sum()
        total_return_pct = (total_gain / total_cb * 100) if total_cb > 0 else 0
        
        lines.append("PORTFOLIO SNAPSHOT")
        lines.append("-" * 80)
        lines.append(f"Total Value:    ${self.total_value:>12,.2f}")
        lines.append(f"Total Return:   ${total_gain:>+12,.2f}  ({total_return_pct:>+6.2f}%)")
        lines.append(f"Positions:      {len(self.df):>12}")
        
        # Today's movers
        lines.append("")
        lines.append("📈 TODAY'S BIGGEST MOVERS")
        lines.append("-" * 80)
        
        # Sort by day change
        sorted_df = self.df.sort_values('day_change_pct', ascending=False)
        
        # Top 3 gainers
        lines.append("\n🟢 Top Gainers:")
        for _, row in sorted_df.head(3).iterrows():
            lines.append(f"   {row['symbol']:<10} {row['day_change_pct']:>+6.2f}%   ${row['market_value']:>10,.0f}")
        
        # Top 3 losers
        lines.append("\n🔴 Top Losers:")
        for _, row in sorted_df.tail(3).iterrows():
            lines.append(f"   {row['symbol']:<10} {row['day_change_pct']:>+6.2f}%   ${row['market_value']:>10,.0f}")
        
        # Alerts
        lines.append("")
        lines.append("⚠️  ALERTS")
        lines.append("-" * 80)
        
        alerts = []
        
        # Big moves
        for _, row in self.df.iterrows():
            if abs(row['day_change_pct']) > 5:
                alerts.append(f"   {row['symbol']}: {row['day_change_pct']:+.1f}% move - Check for news")
        
        # Big positions
        for _, row in self.df.iterrows():
            if row['pct_portfolio'] > 15:
                alerts.append(f"   {row['symbol']}: {row['pct_portfolio']:.1f}% of portfolio - Over-concentrated")
        
        # Big losers
        for _, row in self.df.iterrows():
            if row['gain_loss_pct'] < -30:
                alerts.append(f"   {row['symbol']}: {row['gain_loss_pct']:.1f}% loss - Review thesis")
        
        if alerts:
            lines.extend(alerts)
        else:
            lines.append("   ✅ No critical alerts")
        
        # Full holdings
        lines.append("")
        lines.append("")
        lines.append("COMPLETE HOLDINGS")
        lines.append("=" * 80)
        lines.append(f"{'Symbol':<10} {'Price':>10} {'Day %':>8} {'Value':>12} {'Return':>10} {'% Port':>8}")
        lines.append("-" * 80)
        
        # Sort by portfolio weight
        sorted_df = self.df.sort_values('pct_portfolio', ascending=False)
        
        for _, row in sorted_df.iterrows():
            lines.append(
                f"{row['symbol']:<10} "
                f"${row['current_price']:>9.4f} "
                f"{row['day_change_pct']:>+7.2f}% "
                f"${row['market_value']:>11,.0f} "
                f"{row['gain_loss_pct']:>+9.1f}% "
                f"{row['pct_portfolio']:>7.1f}%"
            )
        
        lines.append("")
        lines.append("=" * 80)
        lines.append("💡 TIP: Run this multiple times per day to track changes")
        lines.append("📊 Data updates: Every 15 minutes (free tier)")
        lines.append("")
        
        return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution"""
    
    print("=" * 80)
    print("ALPHA MINER - MULTIPLE DAILY USE VERSION")
    print("Free 15-minute delayed data (good enough for juniors)")
    print("=" * 80)
    print("")
    
    # Check for CSV file
    if len(sys.argv) < 2:
        print("❌ ERROR: No CSV file provided")
        print("")
        print("📋 USAGE:")
        print("   python3 alpha_miner_complete_daily.py your_portfolio.csv")
        print("")
        print("📄 CSV FORMAT:")
        print("   Minimum columns: Symbol, Quantity, CostBasis")
        print("   Optional: Stage, Metal, Country")
        print("")
        print("💡 Creating sample CSV for you...")
        create_sample_csv()
        print("")
        print("📝 Edit portfolio_sample.csv with your data, then run:")
        print("   python3 alpha_miner_complete_daily.py portfolio_sample.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    # Check if yfinance is installed
    if not YFINANCE_AVAILABLE:
        print("❌ ERROR: yfinance not installed")
        print("")
        print("📦 Install it with:")
        print("   pip install yfinance --break-system-packages")
        print("")
        print("   OR")
        print("")
        print("   pip3 install yfinance")
        sys.exit(1)
    
    # Load portfolio
    df = load_portfolio_from_csv(csv_file)
    
    # Get live prices
    symbols = df['symbol'].tolist()
    prices = get_all_prices(symbols)
    
    # Get cash position (if user wants to add it)
    cash = 0
    if 'cash' in df.columns:
        cash = df['cash'].iloc[0] if len(df) > 0 else 0
    
    # Run analysis
    analyzer = QuickAnalyzer(df, prices, cash)
    report = analyzer.quick_analysis()
    
    # Display
    print(report)
    
    # Save
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analysis_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write(report)
    
    print(f"💾 Report saved: {filename}")
    print("")
    print("🔄 Run again anytime - prices update every 15 minutes")


if __name__ == "__main__":
    main()
