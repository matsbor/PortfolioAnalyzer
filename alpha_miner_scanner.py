#!/usr/bin/env python3
"""
ALPHA MINER SCANNER - Mining Discovery Scanner Module
Batch processes symbols through alpha models to identify discovery candidates.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

# Import core computation functions
from alpha_miner_core import (
    calculate_alpha_models,
    calculate_liquidity_metrics,
    calculate_data_confidence,
    calculate_dilution_risk,
    calculate_sell_risk,
    get_benchmark_data,
    calculate_macro_regime
)

# Import batch data loading from backtest_runner
try:
    import sys
    from pathlib import Path
    # Add parent directory to path to import backtest_runner
    parent_dir = Path(__file__).parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    from backtest_runner import load_or_fetch_price_data_batch
except ImportError:
    # Fallback if backtest_runner not available
    def load_or_fetch_price_data_batch(*args, **kwargs):
        raise ImportError("backtest_runner module not available for batch loading")


def scan_symbols(
    symbols: List[str],
    data_dir: Path = Path('./.scanner_cache'),
    days_lookback: int = 252,
    macro_regime: Optional[Dict] = None,
    min_alpha_score: float = 40.0,
    max_results: int = 5
) -> pd.DataFrame:
    """
    Scan a list of symbols through alpha models to identify discovery candidates.
    
    Args:
        symbols: List of symbols to scan (e.g., TSX-V miners)
        data_dir: Cache directory for price data
        days_lookback: Number of days of historical data to fetch
        macro_regime: Current macro regime dict (if None, will be calculated)
        min_alpha_score: Minimum alpha score to include in results
        max_results: Maximum number of results to return
    
    Returns:
        DataFrame with columns: Symbol, Alpha_Score, Survival_Score, Stage, 
        Liquidity_Tier, Action, Confidence, etc.
    """
    if not symbols:
        return pd.DataFrame()
    
    # Calculate date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_lookback)).strftime('%Y-%m-%d')
    
    # Load or fetch price data in batch (avoids rate limits)
    # Create cache directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or fetch price data in batch (avoids rate limits)
    try:
        hist_cache, missing_symbols = load_or_fetch_price_data_batch(
            symbols,
            start_date,
            end_date,
            data_dir,
            offline=False,  # Scanner needs network access
            allow_partial_cache=True,  # Continue with partial results
            optional_symbols=None
        )
        
        # Note: missing_symbols are silently ignored (scanner is exploratory)
    except Exception as e:
        # If batch loading fails, return empty results
        # In Streamlit context, errors will be caught and displayed in UI
        raise  # Re-raise to be caught by UI error handler
    
    # Calculate macro regime if not provided
    if macro_regime is None:
        macro_regime = calculate_macro_regime()
    
    # Process each symbol
    results = []
    if not hist_cache:
        return pd.DataFrame()
    
    regime = macro_regime.get('regime', 'NEUTRAL')
    is_defensive = (regime == 'DEFENSIVE')
    
    for symbol in symbols:
        if symbol not in hist_cache:
            continue
        
        hist = hist_cache[symbol]
        if hist.empty or len(hist) < 20:  # Need at least 20 days of data
            continue
        
        # Get latest price and create row dict
        latest_price = hist['Close'].iloc[-1]
        hist_slice = hist.tail(252)  # Use available data
        
        # Create minimal row dict for alpha calculation
        # Calculate returns safely
        return_7d = 0
        if len(hist) >= 7:
            price_7d_ago = hist['Close'].iloc[-7]
            return_7d = ((latest_price - price_7d_ago) / price_7d_ago * 100) if price_7d_ago > 0 else 0
        
        return_30d = 0
        if len(hist) >= 30:
            price_30d_ago = hist['Close'].iloc[-30]
            return_30d = ((latest_price - price_30d_ago) / price_30d_ago * 100) if price_30d_ago > 0 else 0
        
        return_90d = 0
        if len(hist) >= 90:
            price_90d_ago = hist['Close'].iloc[-90]
            return_90d = ((latest_price - price_90d_ago) / price_90d_ago * 100) if price_90d_ago > 0 else 0
        
        row_dict = {
            'Symbol': symbol,
            'Price': latest_price,
            'Market_Value': 0,  # Not needed for scanning
            'Return_7d': return_7d,
            'Return_30d': return_30d,
            'Return_90d': return_90d,
            'Pct_From_52w_High': 0,  # Will be calculated if needed
            'Pct_From_52w_Low': 0,
            'Volatility_60d': hist['Close'].pct_change(fill_method=None).tail(60).std() * 100 if len(hist) >= 60 else 0,
            'MA50': hist['Close'].tail(50).mean() if len(hist) >= 50 else latest_price,
            'MA200': hist['Close'].tail(200).mean() if len(hist) >= 200 else latest_price,
            'Drawdown_90d': 0,
            'Runway': 12.0,  # Default assumption
            'stage': 'Explorer',  # Default assumption (will be filtered if DEFENSIVE)
            'metal': 'Gold',  # Default assumption
            'cash': 10.0,
            'burn_source': 'default',
            'Insider_Buying_90d': False,
            'Data_Confidence': 50
        }
        
        # Calculate 52-week positioning
        if len(hist) >= 252:
            high_52w = hist['High'].tail(252).max()
            low_52w = hist['Low'].tail(252).min()
            row_dict['Pct_From_52w_High'] = ((latest_price - high_52w) / high_52w * 100) if high_52w > 0 else 0
            row_dict['Pct_From_52w_Low'] = ((latest_price - low_52w) / low_52w * 100) if low_52w > 0 else 0
        
        # Calculate alpha models
        benchmark = get_benchmark_data(row_dict.get('metal', 'Gold'))
        alpha_result = calculate_alpha_models(row_dict, hist_slice, benchmark)
        alpha_score = alpha_result.get('alpha_score', 0)
        
        # Calculate survival score (from M3 model logic)
        runway = row_dict.get('Runway', 12)
        data_conf = row_dict.get('Data_Confidence', 50)
        survival_score = 50
        if runway >= 18:
            survival_score = 80
        elif runway >= 12:
            survival_score = 65
        elif runway < 6:
            survival_score = 20
        else:
            survival_score = 40
        
        # Adjust survival based on data confidence
        if data_conf < 40:
            survival_score = max(20, survival_score - 20)
        
        # Calculate liquidity metrics
        liq = calculate_liquidity_metrics(
            symbol, hist_slice, latest_price, 0, 1000000  # Use dummy values for scanning
        )
        liq_tier = liq.get('tier_code', 'UNKNOWN')
        
        # MACRO FILTER: If DEFENSIVE regime, only show Producers with high Survival
        # Note: In a real implementation, stage would come from fundamental data
        # For now, we'll use a heuristic: if survival_score >= 60, assume Producer-like quality
        stage = row_dict.get('stage', 'Explorer')
        if is_defensive:
            # In DEFENSIVE mode, only show Producers with high Survival (≥60)
            # Since we don't have actual stage data, use survival_score as proxy
            if survival_score < 60:
                continue  # Skip low-survival in DEFENSIVE regime
            # Mark as Producer for display
            stage = 'Producer'
        
        # Filter by minimum alpha score
        if alpha_score < min_alpha_score:
            continue
        
        # Calculate additional metrics
        dilution = calculate_dilution_risk(
            runway, stage, 0, [], data_conf < 40, False, False
        )
        
        # Determine action based on alpha score
        if alpha_score >= 75:
            action = 'Buy'
            confidence = 'High'
        elif alpha_score >= 60:
            action = 'Buy'
            confidence = 'Medium'
        elif alpha_score >= 50:
            action = 'HOLD'
            confidence = 'Medium'
        else:
            action = 'HOLD'
            confidence = 'Low'
        
        results.append({
            'Symbol': symbol,
            'Alpha_Score': round(alpha_score, 1),
            'Survival_Score': survival_score,
            'Stage': stage,
            'Liquidity_Tier': liq_tier,
            'Action': action,
            'Confidence': confidence,
            'Dilution_Risk': dilution.get('score', 50),
            'Return_7d': round(row_dict['Return_7d'], 1),
            'Return_30d': round(row_dict['Return_30d'], 1),
            'Volatility_60d': round(row_dict['Volatility_60d'], 1),
            'Price': round(latest_price, 2),
            'MA50': round(row_dict['MA50'], 2),
            'MA200': round(row_dict['MA200'], 2)
        })
    
    # Convert to DataFrame and sort by Alpha Score
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results)
    df = df.sort_values('Alpha_Score', ascending=False)
    
    # Return top N results
    return df.head(max_results)


def load_symbols_from_csv(csv_path: Path) -> List[str]:
    """
    Load symbols from a CSV file.
    Expected format: CSV with a 'Symbol' column.
    
    Note: For Streamlit file uploaders, use pd.read_csv(uploaded_file) directly
    instead of this function, as Streamlit provides BytesIO objects, not file paths.
    """
    try:
        df = pd.read_csv(csv_path)
        if 'Symbol' in df.columns:
            symbols = df['Symbol'].dropna().astype(str).str.strip().str.upper().tolist()
            return symbols
        else:
            raise ValueError(f"CSV file must have a 'Symbol' column. Found columns: {df.columns.tolist()}")
    except Exception as e:
        raise ValueError(f"Error loading symbols from CSV: {e}")


def load_master_discovery_list(csv_path: Optional[Path] = None, max_symbols: int = 200) -> List[str]:
    """
    V4.0: Load symbols from master_discovery_list.csv for bulk processing.
    
    Supports up to 200 symbols for batch scanning. The CSV can have:
    - A 'Symbol' column (required)
    - Optional metadata columns (Stage, Metal, Jurisdiction, etc.) for filtering
    
    Args:
        csv_path: Path to master_discovery_list.csv (default: ./master_discovery_list.csv)
        max_symbols: Maximum number of symbols to load (default: 200)
    
    Returns:
        List of symbol strings (uppercase, deduplicated)
    
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV format is invalid
    """
    if csv_path is None:
        csv_path = Path('./master_discovery_list.csv')
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Master discovery list not found: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        
        # Handle various column name formats
        symbol_column = None
        for col in df.columns:
            if col.strip().lower() in ['symbol', 'ticker', 'stock', 'sym']:
                symbol_column = col
                break
        
        if symbol_column is None:
            raise ValueError(f"CSV must have a 'Symbol' column. Found columns: {df.columns.tolist()}")
        
        # Extract symbols, deduplicate, and limit to max_symbols
        symbols = df[symbol_column].dropna().astype(str).str.strip().str.upper().unique().tolist()
        
        # Limit to max_symbols
        if len(symbols) > max_symbols:
            symbols = symbols[:max_symbols]
            print(f"⚠️  Warning: Loaded {max_symbols} symbols (limit). Total available: {len(df)}")
        else:
            print(f"✓ Loaded {len(symbols)} symbols from {csv_path.name}")
        
        return symbols
    
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: {csv_path}")
    except Exception as e:
        raise ValueError(f"Error loading master discovery list: {e}")
