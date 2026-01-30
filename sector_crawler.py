#!/usr/bin/env python3
"""
V5.0: SOVEREIGN MINER ENGINE - Autonomous Sector Discovery
Automated ticker discovery for TSX Venture (.V) and OTC (.QB/.QX) mining stocks.
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

# Optional yfinance import
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

# Load hunting settings
HUNTING_SETTINGS_PATH = Path('./v5_hunting_settings.json')
if HUNTING_SETTINGS_PATH.exists():
    with open(HUNTING_SETTINGS_PATH, 'r') as f:
        HUNTING_SETTINGS = json.load(f)
else:
    # Default settings
    HUNTING_SETTINGS = {
        "max_stock_price": 5.00,
        "max_market_cap_millions": 500.0,
        "min_cash_balance": 500000,
        "max_aisc_threshold": 1400.0,
        "preferred_jurisdictions": ["Canada", "USA", "Australia", "Mexico"],
        "high_beta_mode": True
    }


def load_hunting_settings() -> Dict:
    """Load V5 hunting settings from JSON file."""
    if HUNTING_SETTINGS_PATH.exists():
        try:
            with open(HUNTING_SETTINGS_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load hunting settings: {e}")
            return HUNTING_SETTINGS
    return HUNTING_SETTINGS


def filter_by_hunting_settings(symbol: str, price: float, market_cap: float = None) -> Tuple[bool, str]:
    """
    V5.0: Filter ticker based on hunting settings.
    
    Primary Filter:
    - Reject if Price > $5.00
    - Reject if Market Cap > $500M
    
    Args:
        symbol: Ticker symbol
        price: Current stock price
        market_cap: Market capitalization in millions (optional)
    
    Returns:
        (passed: bool, reason: str)
    """
    settings = load_hunting_settings()
    max_price = settings.get('max_stock_price', 5.00)
    max_mcap = settings.get('max_market_cap_millions', 500.0)
    
    if price > max_price:
        return False, f"Price ${price:.2f} exceeds ${max_price:.2f} limit"
    
    if market_cap is not None and market_cap > max_mcap:
        return False, f"Market Cap ${market_cap:.1f}M exceeds ${max_mcap:.1f}M limit"
    
    return True, "Passed hunting filters"


def discover_tsxv_tickers(sector: str = "mining", max_results: int = 50) -> List[str]:
    """
    V5.0: Discover TSX Venture (.V) tickers.
    
    Note: In production, this would query TSX-V exchange data or use a ticker database.
    For now, this is a placeholder that returns an empty list.
    Actual implementation would require:
    - TSX-V exchange API access, OR
    - A ticker database/CSV file with TSX-V symbols
    
    Args:
        sector: Sector filter (e.g., "mining", "energy")
        max_results: Maximum number of tickers to return
    
    Returns:
        List of ticker symbols (e.g., ["ABRA.V", "BORMF.V"])
    """
    # PLACEHOLDER: In production, this would query TSX-V exchange or ticker database
    # For now, return empty list - users can provide symbols via CSV
    return []


def discover_otc_tickers(sector: str = "mining", max_results: int = 50) -> List[str]:
    """
    V5.0: Discover OTC (.QB/.QX) tickers.
    
    Note: In production, this would query OTC Markets data.
    For now, this is a placeholder that returns an empty list.
    Actual implementation would require:
    - OTC Markets API access, OR
    - A ticker database/CSV file with OTC symbols
    
    Args:
        sector: Sector filter (e.g., "mining", "energy")
        max_results: Maximum number of tickers to return
    
    Returns:
        List of ticker symbols (e.g., ["ABRA", "BORMF"])
    """
    # PLACEHOLDER: In production, this would query OTC Markets or ticker database
    # For now, return empty list - users can provide symbols via CSV
    return []


def validate_ticker_against_settings(symbol: str, hist_data: pd.DataFrame = None) -> Dict:
    """
    V5.0: Validate a ticker against hunting settings using price data.
    
    Args:
        symbol: Ticker symbol
        hist_data: Historical price data (optional, will fetch if not provided)
    
    Returns:
        Dict with 'passed' (bool), 'reason' (str), 'price' (float), 'market_cap' (float)
    """
    result = {
        'passed': False,
        'reason': '',
        'price': 0.0,
        'market_cap': None
    }
    
    if not YFINANCE_AVAILABLE:
        result['reason'] = 'yfinance not available'
        return result
    
    try:
        # Fetch current price
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Get current price
        price = info.get('currentPrice', 0) or info.get('regularMarketPrice', 0)
        if price == 0:
            # Try to get from history
            if hist_data is not None and not hist_data.empty:
                price = hist_data['Close'].iloc[-1]
            else:
                hist = ticker.history(period="1d")
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
        
        if price == 0:
            result['reason'] = 'Could not fetch price'
            return result
        
        result['price'] = float(price)
        
        # Get market cap (optional)
        market_cap = info.get('marketCap', None)
        if market_cap:
            market_cap_millions = market_cap / 1_000_000
            result['market_cap'] = market_cap_millions
        else:
            # Estimate from shares outstanding and price
            shares_outstanding = info.get('sharesOutstanding', None)
            if shares_outstanding and price > 0:
                market_cap_millions = (shares_outstanding * price) / 1_000_000
                result['market_cap'] = market_cap_millions
        
        # Apply hunting filters
        passed, reason = filter_by_hunting_settings(symbol, price, result.get('market_cap'))
        result['passed'] = passed
        result['reason'] = reason
        
    except Exception as e:
        result['reason'] = f"Error validating {symbol}: {str(e)}"
    
    return result


def crawl_sector(symbols: List[str] = None, max_results: int = 50) -> pd.DataFrame:
    """
    V5.0: Main sector crawling function.
    
    If symbols are provided, validates them against hunting settings.
    If not provided, attempts to discover new tickers (placeholder for now).
    
    Args:
        symbols: Optional list of symbols to validate (if None, attempts discovery)
        max_results: Maximum number of results to return
    
    Returns:
        DataFrame with columns: Symbol, Price, Market_Cap_M, Passed, Reason
    """
    results = []
    
    if symbols:
        # Validate provided symbols
        for symbol in symbols[:max_results]:
            validation = validate_ticker_against_settings(symbol)
            results.append({
                'Symbol': symbol,
                'Price': validation['price'],
                'Market_Cap_M': validation.get('market_cap'),
                'Passed': validation['passed'],
                'Reason': validation['reason']
            })
    else:
        # Attempt discovery (placeholder - returns empty for now)
        # In production, this would call discover_tsxv_tickers() and discover_otc_tickers()
        pass
    
    if not results:
        return pd.DataFrame(columns=['Symbol', 'Price', 'Market_Cap_M', 'Passed', 'Reason'])
    
    df = pd.DataFrame(results)
    # Filter to only passed symbols
    df_passed = df[df['Passed'] == True].copy()
    
    return df_passed.head(max_results)
