#!/usr/bin/env python3
"""
Backtest Runner for Alpha Miner Pro
Conservative, auditable 6-month walk-forward backtest for verification and governance.

Usage:
    python backtest_runner.py --start 2024-01-01 --end 2024-06-30 --portfolio_csv portfolio.csv --initial_cash 200000
"""

import argparse
import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple, Optional

# Import decision logic from core module (import-safe, no Streamlit)
try:
    from alpha_miner_core import (
        calculate_alpha_models,
        calculate_sell_risk,
        calculate_liquidity_metrics,
        calculate_financing_overhang,
        calculate_dilution_risk,
        calculate_data_confidence,
        arbitrate_final_decision,
        get_benchmark_data,
        MODEL_ROLES,
        calculate_macro_regime,
        calculate_tape_gate
    )
except ImportError as e:
    print(f"Error importing from alpha_miner_core: {e}")
    print("Make sure alpha_miner_core.py is in the same directory.")
    sys.exit(1)

# Liquidity constraints for backtesting
LIQUIDITY_BUY_LIMITS = {
    'L0': 0.0,      # No buys
    'L1': 0.25,    # 0.25% of portfolio per day
    'L2': 0.5,     # 0.5% of portfolio per day
    'L3': 1.0      # 1.0% of portfolio per day
}

def parse_args():
    parser = argparse.ArgumentParser(description='Backtest Alpha Miner Pro recommendations')
    parser.add_argument('--start', type=str, required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', type=str, required=True, help='End date YYYY-MM-DD')
    parser.add_argument('--portfolio_csv', type=str, required=True, help='Path to portfolio CSV')
    parser.add_argument('--initial_cash', type=float, required=True, help='Initial cash amount')
    parser.add_argument('--rebalance', action='store_true', default=True, help='Enable rebalancing (default: True)')
    parser.add_argument('--allow_leverage', action='store_true', help='Allow leverage (default: False)')
    parser.add_argument('--slippage_bps', type=int, default=20, help='Slippage in basis points (default: 20)')
    parser.add_argument('--max_position_pct', type=float, default=10.0, help='Max position percent (default: 10.0)')
    parser.add_argument('--data_dir', type=str, default='./.backtest_cache', help='Data cache directory')
    parser.add_argument('--offline', action='store_true', help='Offline mode (no network calls)')
    parser.add_argument('--strict_mode', action='store_true', default=True, help='Strict mode (default: True)')
    return parser.parse_args()

def get_trading_days(start_date: str, end_date: str, data_dir: Path, offline: bool) -> List[str]:
    """Get list of trading days between start and end dates"""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    # Try to load from cache first
    cache_file = data_dir / f"SPY_calendar_{start_date}_{end_date}.csv"
    if cache_file.exists():
        try:
            cal = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return [d.strftime('%Y-%m-%d') for d in cal.index]
        except:
            pass
    
    if offline:
        # Fallback: exclude weekends only
        dates = pd.date_range(start, end, freq='B')
        return [d.strftime('%Y-%m-%d') for d in dates]
    
    # Get SPY calendar for trading days
    try:
        cal = yf.Ticker("SPY").history(start=start, end=end)
        if not cal.empty:
            trading_days = [d.strftime('%Y-%m-%d') for d in cal.index]
            # Cache it
            data_dir.mkdir(parents=True, exist_ok=True)
            cal.to_csv(cache_file)
            return trading_days
    except:
        pass
    
    # Fallback: exclude weekends
    dates = pd.date_range(start, end, freq='B')
    return [d.strftime('%Y-%m-%d') for d in dates]

def load_or_fetch_price_data(symbol: str, start: str, end: str, data_dir: Path, offline: bool) -> pd.DataFrame:
    """Load price data from cache or fetch from yfinance"""
    cache_file = data_dir / f"{symbol}_{start}_{end}.csv"
    
    if cache_file.exists():
        try:
            hist = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return hist
        except Exception as e:
            print(f"Warning: Could not load cached data for {symbol}: {e}")
            if offline:
                raise FileNotFoundError(f"Offline mode: Corrupted cache for {symbol}. Cache file: {cache_file}")
    
    if offline:
        raise FileNotFoundError(f"Offline mode: Missing cached data for {symbol}. Cache file: {cache_file}")
    
    # Fetch from yfinance
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start, end=end)
        if not hist.empty:
            # Save to cache
            data_dir.mkdir(parents=True, exist_ok=True)
            hist.to_csv(cache_file)
            return hist
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"Warning: Could not fetch {symbol}: {e}")
        return pd.DataFrame()

def simulate_day(
    date: str,
    portfolio: pd.DataFrame,
    cash: float,
    hist_cache: Dict[str, pd.DataFrame],
    news_cache: Dict[str, List],
    info_cache: Dict[str, Dict],
    strict_mode: bool,
    allow_leverage: bool,
    slippage_bps: int,
    max_position_pct: float
) -> Tuple[pd.DataFrame, float, List[Dict], Dict]:
    """
    Simulate one trading day and return updated portfolio, cash, trades, and daily stats
    """
    total_value = portfolio['Market_Value'].sum() + cash
    trades = []
    num_buys = 0
    num_avoids = 0
    
    # Calculate macro regime and tape gate (simplified for backtest - use defaults)
    # In backtest, we use simplified regime to avoid Streamlit dependencies
    macro_regime = {
        'regime': 'NEUTRAL',
        'allow_new_buys': True,
        'throttle_factor': 1.0,
        'dxy': 0,
        'vix': 0,
        'factors': ['Backtest mode - neutral']
    }
    
    tape_gate = {
        'new_buys_allowed': True,
        'throttle': 1.0,
        'reasons': ['Backtest mode']
    }
    
    # Process each position
    for idx, row in portfolio.iterrows():
        symbol = row['Symbol']
        hist = hist_cache.get(symbol, pd.DataFrame())
        news = news_cache.get(symbol, [])
        info = info_cache.get(symbol, {})
        
        if hist.empty:
            continue
        
        # Get current price (use closest available date if exact date not found)
        if date in hist.index:
            current_price = hist.loc[date, 'Close']
        else:
            # Find closest date before or on target date
            hist_before = hist[hist.index <= date]
            if not hist_before.empty:
                current_price = hist_before['Close'].iloc[-1]
            else:
                current_price = row.get('Price', 0)
        
        if current_price == 0 or pd.isna(current_price):
            continue
        
        # Prepare row data for calculations (convert Series to dict)
        row_dict = row.to_dict()
        row_dict['Price'] = current_price
        row_dict['Market_Value'] = row_dict.get('Market_Value', 0) or (row.get('Quantity', 0) * current_price)
        
        # Calculate metrics (simplified - using cached data)
        # Use historical slice up to current date
        hist_slice = hist[hist.index <= date] if date in hist.index else hist
        if hist_slice.empty:
            hist_slice = hist
        
        liq = calculate_liquidity_metrics(symbol, hist_slice, current_price, row_dict['Market_Value'], total_value)
        liq_metrics = {'tier_code': liq.get('tier_code', 'L0'), 'max_position_pct': liq.get('max_position_pct', 1.0)}
        
        # Data confidence
        fund_dict = info.get('fundamentals', {})
        inferred = info.get('inferred_flags', {})
        data_conf = calculate_data_confidence(fund_dict, info.get('info_dict', {}), inferred)
        
        # Calculate drawdown for dilution risk
        if len(hist) >= 90:
            hist_slice = hist[hist.index <= date] if date in hist.index else hist
            if len(hist_slice) >= 90:
                high_90d = hist_slice['High'].tail(90).max()
                drawdown_90d = ((current_price - high_90d) / high_90d * 100) if high_90d > 0 else 0
            else:
                drawdown_90d = 0
        else:
            drawdown_90d = 0
        
        # Dilution risk
        dilution = calculate_dilution_risk(
            row_dict.get('Runway', 12.0),
            row_dict.get('stage', 'Explorer'),
            abs(drawdown_90d),
            news,
            row_dict.get('cash', 10.0) == 10.0,
            row_dict.get('burn_source', 'default') == 'default',
            row_dict.get('Insider_Buying_90d', False)
        )
        
        # Sell risk (use hist_slice already calculated above)
        ma50 = hist_slice['Close'].tail(50).mean() if len(hist_slice) >= 50 else current_price
        ma200 = hist_slice['Close'].tail(200).mean() if len(hist_slice) >= 200 else current_price
        sell_risk = calculate_sell_risk(row_dict, hist_slice, ma50, ma200, news, macro_regime)
        
        # Alpha
        benchmark = get_benchmark_data(row_dict.get('metal', 'Gold'))
        alpha_result = calculate_alpha_models(row_dict, hist_slice, benchmark)
        alpha_score = alpha_result.get('alpha_score', 50)
        
        # Financing overhang (pass False for institutional_v3_available in backtest)
        overhang = calculate_financing_overhang(news, symbol, row_dict.get('Runway', 12.0), institutional_v3_available=False)
        row_dict['Financing_Overhang_Score'] = overhang['score']
        
        # Final decision
        discovery = (False, '')
        decision = arbitrate_final_decision(
            row_dict, liq_metrics, data_conf, dilution, sell_risk,
            alpha_score, macro_regime, discovery, tape_gate,
            strict_mode=strict_mode
        )
        
        # Strict Provenance Mode check (simplified - assume data is available in backtest)
        if strict_mode and decision['action'] == 'Buy':
            # Check if key inputs are unknown/inferred (simplified check)
            if hist.empty or not news:
                decision['action'] = 'HOLD'
                decision['confidence'] = 'Low'
                decision['primary_gating_reason'] = "Insufficient verified inputs (strict provenance mode)"
        
        # Apply tape gate
        if not tape_gate.get('new_buys_allowed', True) and decision['action'] == 'Buy':
            decision['action'] = 'HOLD'
            decision['warnings'].append("Tape gate blocked new buy")
        
        # Calculate target position
        current_value = row_dict.get('Market_Value', 0) or (row.get('Quantity', 0) * current_price)
        current_pct = (current_value / total_value * 100) if total_value > 0 else 0
        target_pct = decision.get('recommended_pct', current_pct)
        target_value = (target_pct / 100.0) * total_value
        trade_dollars = target_value - current_value
        
        # Liquidity constraint
        liq_tier = liq.get('tier_code', 'L0')
        daily_buy_limit_pct = LIQUIDITY_BUY_LIMITS.get(liq_tier, 0.0)
        
        if trade_dollars > 0:  # Buy
            if liq_tier == 'L0':
                trade_dollars = 0  # No buys for L0
            else:
                max_buy_dollars = (daily_buy_limit_pct / 100.0) * total_value
                trade_dollars = min(trade_dollars, max_buy_dollars)
            
            # Cash constraint
            if not allow_leverage and trade_dollars > cash:
                trade_dollars = cash
            
            if trade_dollars > 0:
                # Apply slippage
                slippage = (slippage_bps / 10000.0) * trade_dollars
                actual_trade = trade_dollars - slippage
                shares = actual_trade / current_price
                
                trades.append({
                    'date': date,
                    'symbol': symbol,
                    'side': 'BUY',
                    'dollars': trade_dollars,
                    'price': current_price,
                    'slippage': slippage,
                    'reason': decision.get('primary_gating_reason', 'Alpha signal')
                })
                
                cash -= trade_dollars
                portfolio.at[idx, 'Quantity'] += shares
                portfolio.at[idx, 'Cost_Basis'] += actual_trade
                num_buys += 1
        
        elif trade_dollars < 0:  # Sell
            trade_dollars = abs(trade_dollars)
            shares = trade_dollars / current_price
            current_quantity = row.get('Quantity', 0)
            actual_shares = min(shares, current_quantity)
            actual_trade = actual_shares * current_price
            
            # Apply slippage
            slippage = (slippage_bps / 10000.0) * actual_trade
            net_proceeds = actual_trade - slippage
            
            trades.append({
                'date': date,
                'symbol': symbol,
                'side': 'SELL',
                'dollars': actual_trade,
                'price': current_price,
                'slippage': slippage,
                'reason': decision.get('primary_gating_reason', 'Risk signal')
            })
            
            cash += net_proceeds
            portfolio.at[idx, 'Quantity'] -= actual_shares
            current_cost_basis = row.get('Cost_Basis', 0)
            current_mv = row_dict.get('Market_Value', 0)
            if current_mv > 0:
                cost_reduction = actual_trade * (current_cost_basis / current_mv)
                portfolio.at[idx, 'Cost_Basis'] = max(0, current_cost_basis - cost_reduction)
            num_avoids += 1 if decision['action'] == 'Avoid' else 0
        
        # Update market value
        portfolio.at[idx, 'Price'] = current_price
        portfolio.at[idx, 'Market_Value'] = portfolio.at[idx, 'Quantity'] * current_price
    
    # Daily stats
    equity = portfolio['Market_Value'].sum()
    total_value = equity + cash
    daily_stats = {
        'date': date,
        'equity': equity,
        'cash': cash,
        'total_value': total_value,
        'drawdown': 0.0,  # Will calculate from peak
        'num_buys': num_buys,
        'num_avoids': num_avoids
    }
    
    return portfolio, cash, trades, daily_stats

def run_backtest(args):
    """Main backtest execution"""
    # Parse dates
    start_date = args.start
    end_date = args.end
    
    # Setup
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    trading_days = get_trading_days(start_date, end_date, data_dir, args.offline)
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path('./reports')
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Load portfolio
    portfolio = pd.read_csv(args.portfolio_csv)
    
    # Ensure required columns exist
    if 'Quantity' not in portfolio.columns:
        portfolio['Quantity'] = 0.0
    if 'Cost_Basis' not in portfolio.columns:
        portfolio['Cost_Basis'] = 0.0
    if 'Price' not in portfolio.columns:
        portfolio['Price'] = 0.0
    if 'Market_Value' not in portfolio.columns:
        portfolio['Market_Value'] = 0.0
    if 'Runway' not in portfolio.columns:
        portfolio['Runway'] = 12.0
    if 'stage' not in portfolio.columns:
        portfolio['stage'] = 'Explorer'
    if 'metal' not in portfolio.columns:
        portfolio['metal'] = 'Gold'
    if 'cash' not in portfolio.columns:
        portfolio['cash'] = 10.0
    if 'burn_source' not in portfolio.columns:
        portfolio['burn_source'] = 'default'
    if 'Insider_Buying_90d' not in portfolio.columns:
        portfolio['Insider_Buying_90d'] = False
    if 'Pct_Portfolio' not in portfolio.columns:
        portfolio['Pct_Portfolio'] = 0.0
    
    cash = args.initial_cash
    
    # Initialize caches
    hist_cache = {}
    news_cache = {}
    info_cache = {}
    
    # Load historical data for all symbols
    print(f"Loading price data for {len(portfolio)} symbols...")
    for symbol in portfolio['Symbol'].unique():
        hist = load_or_fetch_price_data(symbol, start_date, end_date, data_dir, args.offline)
        if not hist.empty:
            hist_cache[symbol] = hist
        else:
            print(f"Warning: No data for {symbol}")
        
        # Load news and fundamentals (simplified - use empty/defaults for backtest)
        news_cache[symbol] = []
        info_cache[symbol] = {
            'fundamentals': {},
            'info_dict': {},
            'inferred_flags': {}
        }
    
    # Initialize portfolio prices on first day
    if trading_days:
        first_day = trading_days[0]
        for idx, row in portfolio.iterrows():
            symbol = row['Symbol']
            if symbol in hist_cache:
                hist = hist_cache[symbol]
                # Find closest date to first_day
                hist_before = hist[hist.index <= first_day]
                if not hist_before.empty:
                    first_price = hist_before['Close'].iloc[-1]
                    portfolio.at[idx, 'Price'] = first_price
                    portfolio.at[idx, 'Market_Value'] = portfolio.at[idx, 'Quantity'] * first_price
                elif not hist.empty:
                    # Use first available price
                    portfolio.at[idx, 'Price'] = hist['Close'].iloc[0]
                    portfolio.at[idx, 'Market_Value'] = portfolio.at[idx, 'Quantity'] * portfolio.at[idx, 'Price']
    
    # Run simulation day by day
    all_trades = []
    daily_stats_list = []
    peak_value = args.initial_cash + portfolio['Market_Value'].sum()
    
    print(f"Running backtest from {start_date} to {end_date} ({len(trading_days)} trading days)...")
    
    for i, date in enumerate(trading_days):
        if i % 10 == 0:
            print(f"  Processing {date} ({i+1}/{len(trading_days)})...")
        
        portfolio, cash, trades, daily_stats = simulate_day(
            date, portfolio.copy(), cash, hist_cache, news_cache, info_cache,
            args.strict_mode, args.allow_leverage, args.slippage_bps, args.max_position_pct
        )
        
        # Enforce cash constraint
        if not args.allow_leverage and cash < 0:
            print(f"Warning: Cash went negative on {date}: ${cash:.2f}. Adjusting...")
            cash = 0.0
        
        all_trades.extend(trades)
        daily_stats['drawdown'] = ((peak_value - daily_stats['total_value']) / peak_value * 100) if peak_value > 0 else 0
        if daily_stats['total_value'] > peak_value:
            peak_value = daily_stats['total_value']
        daily_stats_list.append(daily_stats)
    
    # Generate reports
    print("Generating reports...")
    
    # Trades CSV
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv(reports_dir / 'backtest_trades.csv', index=False)
    
    # Daily stats CSV
    daily_df = pd.DataFrame(daily_stats_list)
    daily_df.to_csv(reports_dir / 'backtest_daily.csv', index=False)
    
    # Summary JSON
    if len(daily_stats_list) > 0:
        initial_value = daily_stats_list[0]['total_value']
        final_value = daily_stats_list[-1]['total_value']
        days = len(daily_stats_list)
        years = days / 252.0
        
        cagr = ((final_value / initial_value) ** (1.0 / years) - 1) * 100 if years > 0 and initial_value > 0 else 0
        
        values = [d['total_value'] for d in daily_stats_list]
        returns = pd.Series(values).pct_change().dropna()
        vol = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0
        
        max_dd = max([d['drawdown'] for d in daily_stats_list]) if daily_stats_list else 0
        
        total_trade_value = sum([abs(t['dollars']) for t in all_trades])
        turnover = (total_trade_value / initial_value) if initial_value > 0 else 0
        
        summary = {
            'start_date': start_date,
            'end_date': end_date,
            'initial_value': initial_value,
            'final_value': final_value,
            'cagr_pct': round(cagr, 2),
            'max_drawdown_pct': round(max_dd, 2),
            'volatility_pct': round(vol, 2),
            'turnover': round(turnover, 2),
            'total_trades': len(all_trades),
            'trading_days': days
        }
        
        with open(reports_dir / 'backtest_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nBacktest complete!")
        print(f"  Initial: ${initial_value:,.0f}")
        print(f"  Final: ${final_value:,.0f}")
        print(f"  CAGR: {cagr:.2f}%")
        print(f"  Max Drawdown: {max_dd:.2f}%")
        print(f"  Reports saved to ./reports/")
    else:
        print("Warning: No daily stats generated")

if __name__ == '__main__':
    args = parse_args()
    run_backtest(args)
