#!/usr/bin/env python3
"""
Backtest Runner for Alpha Miner Pro
Conservative, auditable 6-month walk-forward backtest for verification and governance.

Usage:
    python backtest_runner.py --start 2024-01-01 --end 2024-06-30 --portfolio_csv portfolio_enhanced.csv --initial_cash 200000

Timezone Convention:
    All price series indices are normalized to tz-naive UTC for deterministic comparisons.
    This ensures compatibility between yfinance (which may return tz-aware indices) and
    backtest timestamps (which are tz-naive).
"""

import sys  # Must be first import after shebang for global access
import argparse
import pandas as pd
import numpy as np
import json
import time
import hashlib
import random
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional
import yfinance as yf
import os
from typing import Dict, List, Tuple, Optional

# Tiingo: primary data source (yfinance is fallback)
TIINGO_AVAILABLE = False
TiingoClient = None
try:
    from tiingo import TiingoClient as _TiingoClient
    TiingoClient = _TiingoClient
    TIINGO_AVAILABLE = True
except ImportError:
    pass

try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent / "hey.env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path)
except ImportError:
    pass

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
        calculate_tape_gate,
        calculate_gs_ratio_bias,
        fetch_gold_silver_prices  # V4.0 Phase 3: Automated GSR fetching
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
    'L3': 1.0,     # 1.0% of portfolio per day
    'UNKNOWN': 0.0  # No buys unless allow_leverage (capital protection)
}


def _normalize_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize price DataFrame to tz-naive DatetimeIndex for deterministic comparisons.
    
    All price series indices are normalized to tz-naive UTC for deterministic comparisons.
    This ensures compatibility between yfinance (which may return tz-aware indices) and
    backtest timestamps (which are tz-naive).
    
    Args:
        df: DataFrame with datetime index (may be tz-aware or tz-naive)
    
    Returns:
        DataFrame with tz-naive DatetimeIndex, sorted ascending
    """
    if df.empty:
        return df
    
    # Convert index to DatetimeIndex with timezone safety
    idx = pd.to_datetime(df.index, errors="coerce", utc=True)
    
    # Drop rows where index conversion failed (NaT)
    valid_mask = ~idx.isna()
    if not valid_mask.all():
        df = df.loc[valid_mask].copy()
        idx = idx[valid_mask]
    
    # Handle timezone: convert to UTC if tz-aware, then strip to tz-naive
    # Force timezone safety: use utc=True then tz_localize(None)
    if idx.tz is not None:
        # Convert tz-aware to UTC, then strip timezone
        idx = idx.tz_convert('UTC').tz_localize(None)
    else:
        # If already tz-naive, ensure it's treated as UTC-naive
        idx = pd.DatetimeIndex(idx)
    
    # Set normalized index
    df.index = idx
    
    # Sort index ascending
    df = df.sort_index()
    
    return df

def parse_args():
    parser = argparse.ArgumentParser(description='Backtest Alpha Miner Pro recommendations')
    parser.add_argument('--start', type=str, required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', type=str, required=True, help='End date YYYY-MM-DD')
    parser.add_argument('--portfolio_csv', type=str, required=False, help='Path to portfolio CSV (optional for --build_cache_only and --verify_cache)')
    parser.add_argument('--initial_cash', type=float, required=False, help='Initial cash amount (optional for --build_cache_only and --verify_cache)')
    parser.add_argument('--rebalance', action='store_true', default=True, help='Enable rebalancing (default: True)')
    parser.add_argument('--allow_leverage', action='store_true', help='Allow leverage (default: False)')
    parser.add_argument('--max_position_pct', type=float, default=10.0, help='Max position percent (default: 10.0)')
    parser.add_argument('--monte_carlo', type=int, default=None, 
                       help='Run Monte Carlo analysis with N iterations (adds +/-15%% random execution variance to slippage). Example: --monte_carlo 50')
    parser.add_argument('--data_dir', type=str, default='./.backtest_cache', help='Data cache directory')
    parser.add_argument('--offline', action='store_true', help='Offline mode (no network calls)')
    parser.add_argument('--strict_mode', action='store_true', default=True, help='Strict mode (default: True)')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols to process (default: all symbols in portfolio)')
    parser.add_argument('--retry_missing_cache', action='store_true', help='Retry fetching only missing symbols from manifest, then exit (no simulation)')
    parser.add_argument('--allow_partial_cache', action='store_true', help='Allow proceeding with partial cache if some symbols fail to fetch (default: False)')
    parser.add_argument('--build_cache_only', action='store_true', help='Fetch and cache price data only, then exit (no simulation)')
    parser.add_argument('--verify_cache', action='store_true', help='Verify cache exists and is loadable for all required symbols, then exit')
    parser.add_argument('--skip_missing_symbols', action='store_true', help='Continue with symbols that have data, skip missing ones (reports skipped in summary)')
    parser.add_argument('--sell_policy', type=str, default='veto_only', choices=['veto_only', 'triggers', 'rebalance'], 
                       help='Sell policy: veto_only (sell only on hard veto - default, conservative), triggers (sell on active sell triggers), rebalance (sell/trim to hit targets). Default: veto_only')
    parser.add_argument('--risk_mode', type=str, default='BALANCED', choices=['CONSERVATIVE', 'BALANCED', 'AGGRESSIVE'],
                       help='Risk mode: CONSERVATIVE (strict thresholds), BALANCED (default), AGGRESSIVE (lower thresholds, higher position caps). Default: BALANCED')
    parser.add_argument('--warmup_days', type=int, default=20,
                       help='Number of warmup days to prime moving averages before first trade (default: 20)')
    parser.add_argument('--trailing_stop_pct', type=float, default=None,
                       help='Trailing stop percentage (default: 15%% for AGGRESSIVE, 20%% for BALANCED, 25%% for CONSERVATIVE). Triggers SELL if price drops this %% from high water mark.')
    return parser.parse_args()

def get_trading_days(start_date: str, end_date: str, data_dir: Path, offline: bool) -> List[str]:
    """
    Get list of trading days between start and end dates.
    Returns list of date strings (YYYY-MM-DD) for tz-naive comparisons.
    """
    # Parse to tz-naive Timestamps
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    # Try to load from cache first
    cache_file = data_dir / f"SPY_calendar_{start_date}_{end_date}.csv"
    if cache_file.exists():
        try:
            cal = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            # Normalize calendar index to tz-naive for consistency
            cal.index = pd.to_datetime(cal.index, utc=True).tz_localize(None)
            if cal.index.tz is not None:
                cal.index = cal.index.tz_convert('UTC').tz_localize(None)
            return [d.strftime('%Y-%m-%d') for d in cal.index]
        except:
            pass
    
    if offline:
        # Fallback: exclude weekends only (tz-naive dates)
        dates = pd.date_range(start, end, freq='B')
        return [d.strftime('%Y-%m-%d') for d in dates]
    
    # Get SPY calendar for trading days
    try:
        cal = yf.Ticker("SPY").history(start=start, end=end)
        if not cal.empty:
            # Normalize to tz-naive before caching
            cal = _normalize_price_df(cal)
            trading_days = [d.strftime('%Y-%m-%d') for d in cal.index]
            # Cache it (normalized, tz-naive)
            data_dir.mkdir(parents=True, exist_ok=True)
            cal.to_csv(cache_file)
            return trading_days
    except:
        pass
    
    # Fallback: exclude weekends (tz-naive dates)
    dates = pd.date_range(start, end, freq='B')
    return [d.strftime('%Y-%m-%d') for d in dates]

def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def _load_manifest(data_dir: Path) -> dict:
    """Load cache manifest.json"""
    manifest_file = data_dir / 'manifest.json'
    if manifest_file.exists():
        try:
            with open(manifest_file, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}


def _save_manifest(data_dir: Path, manifest: dict):
    """Save cache manifest.json"""
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = data_dir / 'manifest.json'
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)


def _tiingo_records_to_df(records) -> pd.DataFrame:
    """Convert Tiingo EOD list-of-dicts to yfinance-compatible DataFrame."""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if 'date' not in df.columns:
        return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    renames = {'open': 'Open', 'high': 'High', 'low': 'Low',
               'close': 'Close', 'volume': 'Volume',
               'adjOpen': 'Open', 'adjHigh': 'High', 'adjLow': 'Low',
               'adjClose': 'Close', 'adjVolume': 'Volume'}
    for k, v in renames.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]
        elif k in df.columns and v in df.columns:
            pass  # Don't overwrite if already mapped
    for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if c not in df.columns:
            df[c] = np.nan
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()


def _get_tiingo_client():
    """Initialize a Tiingo client from environment. Returns None if unavailable."""
    if not TIINGO_AVAILABLE or TiingoClient is None:
        return None
    api_key = (os.getenv("TIINGO_API_KEY") or "").strip()
    if not api_key:
        return None
    try:
        return TiingoClient({"api_key": api_key})
    except Exception:
        return None


def _fetch_tiingo_single(client, symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch a single symbol from Tiingo EOD API.
    Tries geography-first variants for Canadian tickers.
    Returns yfinance-compatible DataFrame or empty DataFrame.
    """
    base = symbol.upper().strip()
    # Geography-first variants (same order as main app)
    if base.endswith('.TO'):
        base_clean = base[:-3]
        variants = [base_clean, f"TSX:{base_clean}", f"{base_clean}.TO", f"{base_clean}F"]
    elif base.endswith('.V'):
        base_clean = base[:-2]
        variants = [base_clean, f"TSXV:{base_clean}", f"{base_clean}.V", f"{base_clean}F"]
    else:
        variants = [base]

    for variant in variants:
        try:
            data = client.get_ticker_price(
                variant, startDate=start, endDate=end,
                frequency="daily", fmt="json"
            )
            if data and isinstance(data, list) and len(data) > 0:
                hist = _tiingo_records_to_df(data)
                if not hist.empty and len(hist) >= 5:
                    return hist
        except Exception:
            continue
    return pd.DataFrame()


def _fetch_with_retry(symbols: List[str], start: str, end: str, max_retries: int = 3, skip_missing: bool = False) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Fetch price data for multiple symbols.
    Primary: Tiingo API (per-symbol).  Fallback: yfinance batch download.

    Returns:
        (results dict, failed_symbols list)
    """
    results = {}
    failed_symbols = []

    # ── Phase 1: Try Tiingo first (primary source) ──
    tiingo_client = _get_tiingo_client()
    tiingo_remaining = list(symbols)

    if tiingo_client is not None:
        print(f"  [Tiingo] Fetching {len(symbols)} symbol(s) ...")
        tiingo_failed = []
        for symbol in symbols:
            try:
                hist = _fetch_tiingo_single(tiingo_client, symbol, start, end)
                if not hist.empty:
                    hist = _normalize_price_df(hist)
                    if 'Volume' not in hist.columns:
                        hist['Volume'] = 0.0
                    results[symbol] = hist
                else:
                    tiingo_failed.append(symbol)
            except Exception:
                tiingo_failed.append(symbol)
            # Tiingo rate-limit courtesy (50 req/hr on free tier, generous on paid)
            time.sleep(0.15 + random.uniform(0, 0.1))

        tiingo_remaining = tiingo_failed
        if tiingo_failed:
            print(f"  [Tiingo] {len(symbols) - len(tiingo_failed)}/{len(symbols)} succeeded, "
                  f"{len(tiingo_failed)} falling back to yfinance: {tiingo_failed}")
        else:
            print(f"  [Tiingo] All {len(symbols)} symbols fetched successfully")
    else:
        print("  [Tiingo] Not available — using yfinance for all symbols")

    # ── Phase 2: yfinance fallback for anything Tiingo missed ──
    if tiingo_remaining:
        per_symbol_sleep = 0.2 + random.uniform(0, 0.3)

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(per_symbol_sleep)

                print(f"  [yfinance] Fetching {len(tiingo_remaining)} symbol(s) (attempt {attempt + 1}/{max_retries}) ...")
                hist_dict = yf.download(
                    tiingo_remaining,
                    start=start,
                    end=end,
                    group_by='ticker',
                    progress=False
                )

                # Parse yfinance response
                if len(tiingo_remaining) == 1:
                    symbol = tiingo_remaining[0]
                    if not hist_dict.empty:
                        hist_dict = {symbol: hist_dict}
                    else:
                        hist_dict = {}
                else:
                    if isinstance(hist_dict.columns, pd.MultiIndex):
                        available_symbols = hist_dict.columns.get_level_values(0).unique()
                        hist_dict_parsed = {}
                        for sym in tiingo_remaining:
                            if sym in available_symbols:
                                try:
                                    sym_df = hist_dict.xs(sym, level=0, axis=1)
                                    if not sym_df.empty:
                                        hist_dict_parsed[sym] = sym_df
                                except (KeyError, ValueError):
                                    pass
                        hist_dict = hist_dict_parsed
                    else:
                        hist_dict = {}

                # Process results
                batch_failed = []
                for symbol in tiingo_remaining:
                    if symbol in hist_dict and not hist_dict[symbol].empty:
                        hist = _normalize_price_df(hist_dict[symbol])
                        if 'Volume' not in hist.columns:
                            hist['Volume'] = 0.0
                        results[symbol] = hist
                    else:
                        if symbol not in results:
                            batch_failed.append(symbol)

                if not batch_failed or skip_missing:
                    failed_symbols = batch_failed
                    break

                if batch_failed and attempt < max_retries - 1:
                    tiingo_remaining = batch_failed
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"  [yfinance] {len(batch_failed)} failed. Retrying in {wait_time:.1f}s ...")
                    time.sleep(wait_time)
                    continue

                failed_symbols = batch_failed
                break

            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = '429' in error_str or 'too many requests' in error_str or 'rate limit' in error_str

                if (is_rate_limit or attempt < max_retries - 1):
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    if is_rate_limit:
                        print(f"  [yfinance] Rate limit hit. Retrying in {wait_time:.1f}s ...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  [yfinance] Error after {max_retries} attempts: {e}")
                    for symbol in tiingo_remaining:
                        if symbol not in results:
                            results[symbol] = pd.DataFrame()
                            if symbol not in failed_symbols:
                                failed_symbols.append(symbol)
                    break

    return results, failed_symbols


def load_or_fetch_price_data(symbol: str, start: str, end: str, data_dir: Path, offline: bool) -> pd.DataFrame:
    """
    Load price data from cache or fetch from yfinance.
    Returns DataFrame with tz-naive DatetimeIndex (normalized).
    
    Note: For batch fetching, use load_or_fetch_price_data_batch() instead.
    """
    cache_file = data_dir / f"{symbol}_{start}_{end}.csv"
    
    if cache_file.exists():
        try:
            hist = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            # Normalize to tz-naive (handles both tz-aware and tz-naive cached data)
            hist = _normalize_price_df(hist)
            # Check if normalization resulted in empty DataFrame (corrupted data)
            if hist.empty:
                if offline:
                    raise FileNotFoundError(f"Offline mode: Corrupted cache for {symbol} (empty after normalization). Cache file: {cache_file}")
                return pd.DataFrame()
            # Ensure Volume column exists (yfinance typically includes it)
            if 'Volume' not in hist.columns:
                hist['Volume'] = 0.0
            return hist
        except FileNotFoundError:
            # Re-raise FileNotFoundError from normalization check
            raise
        except Exception as e:
            print(f"Warning: Could not load cached data for {symbol}: {e}")
            if offline:
                raise FileNotFoundError(f"Offline mode: Corrupted cache for {symbol}. Cache file: {cache_file}")
    
    if offline:
        raise FileNotFoundError(f"Offline mode: Missing cached data for {symbol}. Cache file: {cache_file}")

    # Fetch from Tiingo first (primary), then yfinance (fallback)
    hist = pd.DataFrame()

    tiingo_client = _get_tiingo_client()
    if tiingo_client is not None:
        try:
            hist = _fetch_tiingo_single(tiingo_client, symbol, start, end)
            if not hist.empty:
                hist = _normalize_price_df(hist)
                if 'Volume' not in hist.columns:
                    hist['Volume'] = 0.0
        except Exception:
            hist = pd.DataFrame()

    # Fallback to yfinance if Tiingo returned nothing
    if hist.empty:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start, end=end)
            if not hist.empty:
                hist = _normalize_price_df(hist)
                if 'Volume' not in hist.columns:
                    hist['Volume'] = 0.0
        except Exception as e:
            print(f"Warning: Could not fetch {symbol}: {e}")
            return pd.DataFrame()

    if not hist.empty:
        data_dir.mkdir(parents=True, exist_ok=True)
        hist.to_csv(cache_file)
        return hist
    return pd.DataFrame()


def load_or_fetch_price_data_batch(
    symbols: List[str], 
    start: str, 
    end: str, 
    data_dir: Path, 
    offline: bool,
    allow_partial_cache: bool = False,
    optional_symbols: Optional[List[str]] = None
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    Load or fetch price data for multiple symbols in batch.
    Uses manifest.json to track cache status.
    
    Args:
        optional_symbols: List of symbols that are optional (non-fatal if missing, e.g., benchmarks)
    
    Returns:
        (hist_cache dict, missing_symbols list)
    """
    if optional_symbols is None:
        optional_symbols = []
    
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(data_dir)
    
    # Initialize manifest if needed
    cache_key = f"{start}_{end}"
    if cache_key not in manifest:
        manifest[cache_key] = {
            'start': start,
            'end': end,
            'created_at': datetime.now().isoformat(),
            'symbols': {}
        }
    
    hist_cache = {}
    missing_symbols = []
    symbols_to_fetch = []
    
    # Check cache for each symbol
    for symbol in symbols:
        cache_file = data_dir / f"{symbol}_{start}_{end}.csv"
        symbol_status = manifest[cache_key]['symbols'].get(symbol, {})
        
        if cache_file.exists():
            # Verify cache file matches manifest (if manifest has hash)
            if 'sha256' in symbol_status:
                try:
                    current_hash = _compute_file_hash(cache_file)
                    if current_hash == symbol_status['sha256']:
                        # Cache is valid, load it
                        try:
                            hist = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                            hist = _normalize_price_df(hist)
                            if not hist.empty:
                                if 'Volume' not in hist.columns:
                                    hist['Volume'] = 0.0
                                hist_cache[symbol] = hist
                                continue  # Skip fetching
                        except Exception as e:
                            print(f"Warning: Could not load cached data for {symbol}: {e}")
                            if offline:
                                missing_symbols.append(symbol)
                                continue
                    # Hash mismatch - fall through to fetch
                except Exception as e:
                    print(f"Warning: Could not compute hash for {symbol}: {e}")
                    # Fall through to try loading anyway
            
            # No hash in manifest or hash mismatch - try to load anyway
            try:
                hist = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                hist = _normalize_price_df(hist)
                if not hist.empty:
                    if 'Volume' not in hist.columns:
                        hist['Volume'] = 0.0
                    hist_cache[symbol] = hist
                    # Update manifest with hash
                    symbol_status['sha256'] = _compute_file_hash(cache_file)
                    symbol_status['rows'] = len(hist)
                    symbol_status['status'] = 'ok'
                    manifest[cache_key]['symbols'][symbol] = symbol_status
                    # Save manifest after updating
                    _save_manifest(data_dir, manifest)
                    continue
            except Exception as e:
                print(f"Warning: Could not load cached data for {symbol}: {e}")
                if offline:
                    # Check if symbol is optional (benchmarks are non-fatal)
                    if optional_symbols is not None and symbol in optional_symbols:
                        continue  # Skip optional symbols silently in offline mode
                    missing_symbols.append(symbol)
                    continue
        
        # Cache missing or invalid
        if offline:
            # Check if symbol is optional (benchmarks are non-fatal)
            if optional_symbols is not None and symbol in optional_symbols:
                continue  # Skip optional symbols silently in offline mode
            missing_symbols.append(symbol)
        else:
            symbols_to_fetch.append(symbol)
    
    # Fetch missing symbols in batch
    if symbols_to_fetch and not offline:
        print(f"Fetching price data for {len(symbols_to_fetch)} symbols...")
        fetched_results, failed_fetch = _fetch_with_retry(symbols_to_fetch, start, end, skip_missing=allow_partial_cache)
        
        # Save fetched data and update manifest
        fetch_failed_symbols = []
        for symbol in symbols_to_fetch:
            hist = fetched_results.get(symbol, pd.DataFrame())
            cache_file = data_dir / f"{symbol}_{start}_{end}.csv"
            
            if not hist.empty:
                # Ensure normalized before saving (tz-naive UTC)
                hist = _normalize_price_df(hist)
                # Save to cache
                hist.to_csv(cache_file)
                # Update manifest
                symbol_status = {
                    'status': 'ok',
                    'rows': len(hist),
                    'sha256': _compute_file_hash(cache_file)
                }
                hist_cache[symbol] = hist
            else:
                # Fetch failed
                symbol_status = {
                    'status': 'missing',
                    'rows': 0,
                    'sha256': None
                }
                if not allow_partial_cache:
                    missing_symbols.append(symbol)
                    fetch_failed_symbols.append(symbol)
                else:
                    print(f"Warning: Failed to fetch {symbol}, proceeding with partial cache")
            
            manifest[cache_key]['symbols'][symbol] = symbol_status
        
        # Save updated manifest
        _save_manifest(data_dir, manifest)
        
        # Update cache manifest with normalization policy
        cache_manifest_file = data_dir / 'cache_manifest.json'
        cache_manifest = {
            'date_range': f"{start}_{end}",
            'symbols_cached': sorted([s for s in symbols if s in hist_cache]),
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'normalization_policy': 'UTC-naive (tz-naive DatetimeIndex in UTC)',
            'cache_format': 'CSV with DatetimeIndex'
        }
        with open(cache_manifest_file, 'w') as f:
            json.dump(cache_manifest, f, indent=2)
        
        # Raise clear error for failed fetches (if not allowing partial cache)
        if fetch_failed_symbols and not allow_partial_cache:
            error_msg = f"Failed to fetch data for {len(fetch_failed_symbols)} symbol(s) after retries:\n"
            for symbol in sorted(fetch_failed_symbols):
                cache_file = data_dir / f"{symbol}_{start}_{end}.csv"
                error_msg += f"\n  - {symbol}\n"
                error_msg += f"    Date range: {start} to {end}\n"
                error_msg += f"    Expected cache: {cache_file}"
            error_msg += f"\n\nSuggested command to retry:\n"
            error_msg += f"  python3 backtest_runner.py --start {start} --end {end} --symbols {','.join(sorted(fetch_failed_symbols))} --build_cache_only --data_dir {data_dir}"
            raise RuntimeError(error_msg)
    
    # Validate offline mode (exclude optional symbols from error)
    if optional_symbols is None:
        optional_symbols = []
    missing_non_optional = [s for s in missing_symbols if s not in optional_symbols]
    if offline and missing_non_optional:
        if not allow_partial_cache:
            error_msg = f"Offline mode: Missing cached data for {len(missing_non_optional)} symbol(s):\n"
            error_msg += "\n".join(f"  - {sym}" for sym in sorted(missing_non_optional))
            error_msg += f"\n\nTo rebuild cache, run:\n"
            error_msg += f"  python3 backtest_runner.py --start {start} --end {end} --portfolio_csv <csv> --initial_cash <cash> --data_dir {data_dir} --build_cache_only\n"
            error_msg += f"\nOr to retry only missing symbols:\n"
            error_msg += f"  python3 backtest_runner.py --start {start} --end {end} --symbols {','.join(sorted(missing_non_optional))} --retry_missing_cache --data_dir {data_dir}"
            raise FileNotFoundError(error_msg)
        # If allow_partial_cache, missing_symbols will be reported in summary
    
    # Return only non-optional missing symbols
    missing_symbols = missing_non_optional
    
    return hist_cache, missing_symbols

def get_liquidity_adjusted_slippage(liquidity_tier: str, risk_mode: str = 'BALANCED', 
                                     execution_variance: float = 0.0) -> float:
    """
    Get liquidity-adjusted slippage based on institutional standards.
    
    Tiered Slippage Mapping (Institutional Standard):
    - L3 (Producers/Liquid): 10 bps
    - L2 (Developers/Moderate): 50 bps
    - L1 (Explorers/Thin): 250 bps
    - L0 (Danger Zone): 750 bps (simulating wide bid-ask spread of illiquid miners)
    - UNKNOWN: 100 bps (conservative default)
    
    Args:
        liquidity_tier: Liquidity tier code (L3, L2, L1, L0, UNKNOWN)
        risk_mode: Risk mode (CONSERVATIVE, BALANCED, AGGRESSIVE)
        execution_variance: Random execution variance (e.g., +/- 0.15 for Monte Carlo)
    
    Returns:
        Slippage as a decimal multiplier (e.g., 0.0010 for 10 bps)
    """
    base_slippage_bps = {
        'L3': 10,     # Producers/Liquid
        'L2': 50,     # Developers/Moderate
        'L1': 250,    # Explorers/Thin
        'L0': 750,    # Danger Zone (illiquid miners)
        'UNKNOWN': 100  # Conservative default
    }
    
    slippage_bps = base_slippage_bps.get(liquidity_tier, 100)  # Default to 100 if tier not found
    
    # Adjust based on risk_mode
    if risk_mode == 'AGGRESSIVE':
        # AGGRESSIVE mode: Accept 10% higher slippage for L1/L0 (more aggressive execution)
        if liquidity_tier in ['L1', 'L0']:
            slippage_bps = int(slippage_bps * 1.1)
    elif risk_mode == 'CONSERVATIVE':
        # CONSERVATIVE mode: Add 20% buffer for safety
        slippage_bps = int(slippage_bps * 1.2)
    
    # Note: Execution variance is now applied AFTER this function returns (in simulate_day)
    # This ensures base slippage is calculated first, then variance is applied (Monte Carlo handshake)
    # This function only returns the base slippage multiplier
    
    # Convert bps to decimal multiplier
    slippage_decimal = slippage_bps / 10000.0
    
    return slippage_decimal


def calculate_daily_indicators(hist_slice: pd.DataFrame, current_price: float) -> Dict:
    """
    Calculate daily indicators from hist_slice (trailing window).
    Ensures indicators are recalculated fresh each day based on that day's data window.
    """
    indicators = {
        'Return_7d': 0.0,
        'Return_30d': 0.0,
        'Return_90d': 0.0,
        'Pct_From_52w_High': 0.0,
        'Pct_From_52w_Low': 0.0,
        'Volatility_60d': 0.0,
        'MA50': current_price,
        'MA200': current_price,
        'Drawdown_90d': 0.0
    }
    
    if hist_slice.empty:
        return indicators
    
    # Returns (using trailing window relative to last available price)
    if len(hist_slice) >= 7:
        price_7d_ago = hist_slice['Close'].iloc[-7] if len(hist_slice) >= 7 else current_price
        indicators['Return_7d'] = ((current_price - price_7d_ago) / price_7d_ago * 100) if price_7d_ago > 0 else 0
    
    if len(hist_slice) >= 30:
        price_30d_ago = hist_slice['Close'].iloc[-30] if len(hist_slice) >= 30 else current_price
        indicators['Return_30d'] = ((current_price - price_30d_ago) / price_30d_ago * 100) if price_30d_ago > 0 else 0
    
    if len(hist_slice) >= 90:
        price_90d_ago = hist_slice['Close'].iloc[-90] if len(hist_slice) >= 90 else current_price
        indicators['Return_90d'] = ((current_price - price_90d_ago) / price_90d_ago * 100) if price_90d_ago > 0 else 0
    
    # 52-week positioning (using available window, up to 252 days)
    lookback_window = min(252, len(hist_slice))
    if lookback_window > 0:
        high_window = hist_slice['High'].tail(lookback_window).max()
        low_window = hist_slice['Low'].tail(lookback_window).min()
        indicators['Pct_From_52w_High'] = ((current_price - high_window) / high_window * 100) if high_window > 0 else 0
        indicators['Pct_From_52w_Low'] = ((current_price - low_window) / low_window * 100) if low_window > 0 else 0
    
    # Volatility (60-day)
    if len(hist_slice) >= 60:
        returns = hist_slice['Close'].pct_change(fill_method=None).tail(60)
        indicators['Volatility_60d'] = returns.std() * 100 if not returns.empty else 0.0
    
    # Moving averages
    if len(hist_slice) >= 50:
        indicators['MA50'] = hist_slice['Close'].tail(50).mean()
    if len(hist_slice) >= 200:
        indicators['MA200'] = hist_slice['Close'].tail(200).mean()
    
    # Drawdown (90-day)
    if len(hist_slice) >= 90:
        high_90d = hist_slice['High'].tail(90).max()
        indicators['Drawdown_90d'] = ((current_price - high_90d) / high_90d * 100) if high_90d > 0 else 0
    
    return indicators


def check_rebalancing_triggers(
    portfolio: pd.DataFrame,
    symbol: str,
    current_value: float,
    total_value: float,
    cost_basis: float
) -> Dict:
    """
    V4.0: Check rebalancing triggers for a position.
    
    Trigger A (Size): If position exceeds 20% of total value, generate REBALANCE_SELL to trim to 10%
    Trigger B (Profit): If unrealized gain exceeds 100% (2-bagger), sell initial principal, leave house money
    
    Args:
        portfolio: Portfolio DataFrame
        symbol: Symbol to check
        current_value: Current market value of position
        total_value: Total portfolio value (cash + equity)
        cost_basis: Original cost basis of position
    
    Returns:
        Dict with 'triggered' (bool), 'trigger_type' (str), 'target_pct' (float), 'reason' (str)
    """
    result = {
        'triggered': False,
        'trigger_type': None,
        'target_pct': None,
        'reason': ''
    }
    
    if total_value <= 0 or current_value <= 0:
        return result
    
    # Trigger A: Size-based rebalancing (position > 20% of total value)
    position_pct = (current_value / total_value * 100) if total_value > 0 else 0.0
    if position_pct > 20.0:
        result['triggered'] = True
        result['trigger_type'] = 'REBALANCE_SELL'
        result['target_pct'] = 10.0  # Trim to 10%
        result['reason'] = f'Position size {position_pct:.1f}% exceeds 20% limit - rebalancing to 10%'
        return result
    
    # Trigger B: Profit-based rebalancing (2-bagger: >100% gain)
    if cost_basis > 0:
        unrealized_gain_pct = ((current_value - cost_basis) / cost_basis * 100) if cost_basis > 0 else 0.0
        if unrealized_gain_pct > 100.0:
            # Sell initial principal, leave house money to run
            # Calculate how much to sell: sell enough to recover cost_basis
            result['triggered'] = True
            result['trigger_type'] = 'REBALANCE_SELL'
            # Target: sell enough to recover cost_basis (principal protection)
            # Keep the profit portion (house money) running
            sell_value = cost_basis  # Sell principal, keep profit
            result['target_pct'] = (sell_value / total_value * 100) if total_value > 0 else 0.0
            result['reason'] = f'2-bagger detected ({unrealized_gain_pct:.1f}% gain) - harvesting principal, leaving house money'
            return result
    
    return result


def validate_trade_integrity(
    portfolio: pd.DataFrame,
    total_value_before: float,
    total_value_after: float,
    tolerance_pct: float = 0.01
) -> Dict:
    """
    V5.0: Pre-flight guardrails for trade integrity validation.
    
    Check 1: Ensure New_Weight (Recommended_Pct) across all holdings equals 100.00% (allow 0.01% float error).
    Check 2: Verify Total_Value remains constant (within 0.1%) before and after a rebalance simulation.
    
    Args:
        portfolio: Portfolio DataFrame with Recommended_Pct column
        total_value_before: Total portfolio value before rebalance
        total_value_after: Total portfolio value after rebalance simulation
        tolerance_pct: Tolerance for weight sum (default 0.01%)
    
    Returns:
        Dict with 'passed' (bool), 'checks' (list of check results), 'errors' (list of error messages)
    """
    result = {
        'passed': True,
        'checks': [],
        'errors': []
    }
    
    # Check 1: New_Weight (Recommended_Pct) sum equals 100.00%
    if 'Recommended_Pct' in portfolio.columns:
        weight_sum = portfolio['Recommended_Pct'].sum()
        target_sum = 100.0
        deviation = abs(weight_sum - target_sum)
        
        check1_passed = deviation <= tolerance_pct
        result['checks'].append({
            'name': 'Weight Sum Validation',
            'passed': check1_passed,
            'expected': f'{target_sum:.2f}%',
            'actual': f'{weight_sum:.2f}%',
            'deviation': f'{deviation:.4f}%',
            'tolerance': f'{tolerance_pct:.2f}%'
        })
        
        if not check1_passed:
            result['passed'] = False
            result['errors'].append(
                f"Weight sum validation failed: Sum of Recommended_Pct = {weight_sum:.4f}% "
                f"(expected 100.00%, deviation {deviation:.4f}% > tolerance {tolerance_pct:.2f}%)"
            )
    else:
        result['checks'].append({
            'name': 'Weight Sum Validation',
            'passed': False,
            'error': 'Recommended_Pct column not found in portfolio'
        })
        result['passed'] = False
        result['errors'].append('Recommended_Pct column not found in portfolio')
    
    # Check 2: Total_Value remains constant (within 0.1%)
    if total_value_before > 0:
        value_change_pct = abs((total_value_after - total_value_before) / total_value_before * 100)
        value_tolerance = 0.1  # 0.1% tolerance for total value
        
        check2_passed = value_change_pct <= value_tolerance
        result['checks'].append({
            'name': 'Total Value Conservation',
            'passed': check2_passed,
            'expected': f'${total_value_before:,.2f}',
            'actual': f'${total_value_after:,.2f}',
            'change_pct': f'{value_change_pct:.4f}%',
            'tolerance': f'{value_tolerance:.2f}%'
        })
        
        if not check2_passed:
            result['passed'] = False
            result['errors'].append(
                f"Total value conservation failed: Change = {value_change_pct:.4f}% "
                f"(tolerance {value_tolerance:.2f}%). Before: ${total_value_before:,.2f}, After: ${total_value_after:,.2f}"
            )
    else:
        result['checks'].append({
            'name': 'Total Value Conservation',
            'passed': False,
            'error': 'total_value_before is zero or negative'
        })
        result['passed'] = False
        result['errors'].append('total_value_before is zero or negative')
    
    return result


def check_threshold_rebalancing(
    current_pct: float,
    target_pct: float,
    tolerance_pct: float = 5.0
) -> Dict:
    """
    V5.0 Sovereign Hunter: Check threshold rebalancing with 5% Relative Tolerance Band (Goldilocks Rule).
    
    Logic:
    - If current weight moves 5%+ above target (e.g., 10% → 15%), trim to target
    - If current weight falls 5%+ below target (e.g., 10% → 5%), add to target
    - If current weight stays within ±2% of target (e.g., 8-12% for 10% target), do nothing (save fees)
    
    Example for target = 10%:
    - Current = 15% (5% above) → TRIM to 10%
    - Current = 5% (5% below) → ADD to 10%
    - Current = 8-12% (within ±2%) → DO NOTHING
    
    Args:
        current_pct: Current position weight percentage
        target_pct: Target position weight percentage
        tolerance_pct: Relative tolerance percentage (default 5.0%)
    
    Returns:
        Dict with 'triggered' (bool), 'action' (str: 'TRIM'/'ADD'/'HOLD'), 'target_pct' (float), 'reason' (str)
    """
    result = {
        'triggered': False,
        'action': 'HOLD',
        'target_pct': target_pct,
        'reason': ''
    }
    
    if target_pct <= 0:
        return result
    
    # Calculate deviation from target
    deviation = current_pct - target_pct
    
    # Tolerance band: ±2% absolute (do nothing zone)
    tolerance_absolute = 2.0
    lower_bound = target_pct - tolerance_absolute
    upper_bound = target_pct + tolerance_absolute
    
    # Rebalancing thresholds: ±5% relative
    trim_threshold = target_pct + tolerance_pct  # e.g., 10% + 5% = 15%
    add_threshold = target_pct - tolerance_pct  # e.g., 10% - 5% = 5%
    
    # Check if within "do nothing" zone (8-12% for 10% target)
    if lower_bound <= current_pct <= upper_bound:
        result['triggered'] = False
        result['action'] = 'HOLD'
        result['reason'] = f'Weight {current_pct:.1f}% within tolerance band ({lower_bound:.1f}%-{upper_bound:.1f}%) - do nothing (save fees)'
        return result
    
    # Check if above trim threshold (15% for 10% target)
    if current_pct >= trim_threshold:
        result['triggered'] = True
        result['action'] = 'TRIM'
        result['target_pct'] = target_pct
        result['reason'] = f'Weight {current_pct:.1f}% exceeds trim threshold {trim_threshold:.1f}% (target {target_pct:.1f}% + {tolerance_pct:.1f}%) - trim to {target_pct:.1f}%'
        return result
    
    # Check if below add threshold (5% for 10% target)
    if current_pct <= add_threshold:
        result['triggered'] = True
        result['action'] = 'ADD'
        result['target_pct'] = target_pct
        result['reason'] = f'Weight {current_pct:.1f}% below add threshold {add_threshold:.1f}% (target {target_pct:.1f}% - {tolerance_pct:.1f}%) - add to {target_pct:.1f}%'
        return result
    
    # Between tolerance band and thresholds (e.g., 12-15% or 5-8% for 10% target)
    # Still do nothing to avoid excessive trading
    result['triggered'] = False
    result['action'] = 'HOLD'
    result['reason'] = f'Weight {current_pct:.1f}% between tolerance band and threshold - hold to avoid fees'
    
    return result


def check_diversification_veto(
    portfolio: pd.DataFrame,
    symbol: str,
    symbol_metadata: Dict,
    total_value: float
) -> Dict:
    """
    V4.0: Check diversification constraints before allowing a BUY.
    
    Veto A: If total exposure to a Jurisdiction > 35%, veto new buys in that jurisdiction
    Veto B: If total exposure to a Metal_Type > 50%, veto new buys in that metal
    
    Args:
        portfolio: Current portfolio DataFrame
        symbol: Symbol being considered for purchase
        symbol_metadata: Dict with 'Jurisdiction' and 'Metal_Type' for the symbol
        total_value: Total portfolio value (cash + equity)
    
    Returns:
        Dict with 'veto_applied' (bool), 'veto_reason' (str)
    """
    result = {
        'veto_applied': False,
        'veto_reason': ''
    }
    
    if total_value <= 0:
        return result
    
    # Get symbol's jurisdiction and metal type
    jurisdiction = symbol_metadata.get('Jurisdiction', 'Unknown')
    metal_type = symbol_metadata.get('Metal_Type', 'Gold')
    
    # Calculate current exposure by jurisdiction
    if 'Jurisdiction' in portfolio.columns and 'Market_Value' in portfolio.columns:
        jurisdiction_exposure = portfolio[portfolio['Jurisdiction'] == jurisdiction]['Market_Value'].sum()
        jurisdiction_pct = (jurisdiction_exposure / total_value * 100) if total_value > 0 else 0.0
        
        if jurisdiction_pct > 35.0:
            result['veto_applied'] = True
            result['veto_reason'] = f'Diversification Veto: {jurisdiction} exposure ({jurisdiction_pct:.1f}%) exceeds 35% limit'
            return result
    
    # Calculate current exposure by metal type
    if 'Metal_Type' in portfolio.columns and 'Market_Value' in portfolio.columns:
        metal_exposure = portfolio[portfolio['Metal_Type'] == metal_type]['Market_Value'].sum()
        metal_pct = (metal_exposure / total_value * 100) if total_value > 0 else 0.0
        
        if metal_pct > 50.0:
            result['veto_applied'] = True
            result['veto_reason'] = f'Diversification Veto: {metal_type} exposure ({metal_pct:.1f}%) exceeds 50% limit'
            return result
    
    return result


def simulate_day(
    date: str,
    portfolio: pd.DataFrame,
    cash: float,
    hist_cache: Dict[str, pd.DataFrame],
    news_cache: Dict[str, List],
    info_cache: Dict[str, Dict],
    strict_mode: bool,
    allow_leverage: bool,
    max_position_pct: float,
    sell_policy: str = 'veto_only',
    day_number: int = 0,  # For decision logging
    risk_mode: str = 'BALANCED',  # CONSERVATIVE, BALANCED, AGGRESSIVE
    warmup_days: int = 20,  # Days to skip before first trade
    force_deployment: bool = False,  # Force deployment on Day 21 (post-warmup)
    trailing_stop_pct: float = None,  # Trailing stop percentage (defaults based on risk_mode)
    execution_variance: float = 0.0,  # Random execution variance for Monte Carlo (e.g., 0.15 for +/-15%)
    decision_cache: Optional[Dict] = None,  # Cached decisions for Monte Carlo optimization
    save_decisions: bool = False  # If True, save decisions to cache
) -> Tuple[pd.DataFrame, float, List[Dict], Dict]:
    """
    Simulate one trading day with strict temporal isolation.
    
    Execution Order:
    - Step A: Mark-to-Market existing positions using Yesterday's Close
    - Step B: Run arbitrate_final_decision using data through Yesterday (no look-ahead)
    - Step C: Execute trades at Today's Close price (with slippage)
    - Step D: Update Portfolio State
    
    Args:
        date: Trading date (YYYY-MM-DD)
        day_number: Day number in backtest (0-indexed, for logging)
    """
    # Convert date to tz-naive Timestamp with timezone safety
    date_ts = pd.to_datetime(date, utc=True).tz_localize(None)
    
    # Calculate yesterday (previous trading day) for decision-making
    # Use date - 1 day, but we'll use hist data up to (but not including) today
    yesterday_ts = date_ts - pd.Timedelta(days=1)
    
    trades = []
    num_buys = 0
    num_avoids = 0
    intended_trades = 0  # Track intended trades (decisions made)
    executed_trades = 0  # Track executed trades (passed all gates)
    
    # V4.0 Phase 3: Automatically fetch GC=F and SI=F for GSR calculation
    # For backtest, use historical prices from hist_cache (strict temporal isolation)
    gsr_bias = None
    try:
        # Try to get gold/silver prices from hist_cache first (for backtest consistency)
        gold_hist = hist_cache.get('GC=F', pd.DataFrame())
        silver_hist = hist_cache.get('SI=F', pd.DataFrame())
        
        if not gold_hist.empty and not silver_hist.empty:
            # Use historical prices up to yesterday (strict temporal isolation)
            gold_slice = gold_hist[gold_hist.index < date_ts]
            silver_slice = silver_hist[silver_hist.index < date_ts]
            
            if not gold_slice.empty and not silver_slice.empty:
                gold_price = gold_slice['Close'].iloc[-1]
                silver_price = silver_slice['Close'].iloc[-1]
                gsr_bias = calculate_gs_ratio_bias(gold_price, silver_price)
    except Exception:
        # Silently fail - GSR bonus is optional (won't break backtest if GC=F/SI=F not cached)
        gsr_bias = None
    
    # DYNAMIC MACRO INTEGRATION: Calculate macro regime daily using fresh benchmark data
    # Use GDX (gold miners ETF) or GLD (gold ETF) as sector benchmark
    benchmark_symbol = 'GDX'  # Gold miners ETF as sector proxy
    benchmark_hist = hist_cache.get(benchmark_symbol, pd.DataFrame())
    
    # Get benchmark hist_slice up to yesterday (strict temporal isolation)
    benchmark_hist_slice = pd.DataFrame()
    if not benchmark_hist.empty:
        benchmark_hist_slice = benchmark_hist[benchmark_hist.index < date_ts]
    
    # Calculate macro regime dynamically using benchmark data
    if not benchmark_hist_slice.empty:
        macro_regime = calculate_macro_regime(hist_slice=benchmark_hist_slice, date_ts=date_ts)
    else:
        # Fallback: use simplified regime if benchmark not available
        macro_regime = {
            'regime': 'NEUTRAL',
            'allow_new_buys': True,
            'throttle_factor': 1.0,
            'dxy': 0,
            'vix': 0,
            'factors': ['Backtest mode - neutral (benchmark unavailable)']
        }
    
    # Set trailing stop percentage based on risk_mode if not explicitly provided
    # Junior miners are inherently volatile (20-30% swings are normal).
    # Stops that are too tight liquidate good positions during normal volatility.
    if trailing_stop_pct is None:
        if risk_mode == 'AGGRESSIVE':
            trailing_stop_pct = 25.0  # 25% for AGGRESSIVE (wider for juniors)
        elif risk_mode == 'BALANCED':
            trailing_stop_pct = 30.0  # 30% for BALANCED (junior-miner appropriate)
        elif risk_mode == 'CONSERVATIVE':
            trailing_stop_pct = 35.0  # 35% for CONSERVATIVE (widest, let positions breathe)
        else:
            trailing_stop_pct = 30.0  # Default
    
    # Apply risk_mode settings to macro_regime
    if risk_mode == 'AGGRESSIVE':
        macro_regime['throttle_factor'] = 1.2  # 20% throttle boost
        macro_regime['regime'] = 'BULL'  # Treat as bull for aggressive deployment
    elif risk_mode == 'CONSERVATIVE':
        macro_regime['throttle_factor'] = 0.8  # 20% throttle reduction
        macro_regime['regime'] = 'DEFENSIVE'  # More defensive posture
    
    tape_gate = calculate_tape_gate(macro_regime, gold_analysis=None, silver_analysis=None)
    
    # Initialize High_Water_Mark column if it doesn't exist
    if 'High_Water_Mark' not in portfolio.columns:
        portfolio['High_Water_Mark'] = 0.0
    
    # Initialize Trailing_Stop_Cooldown column for 5-day cooldown after trailing stop
    if 'Trailing_Stop_Cooldown' not in portfolio.columns:
        portfolio['Trailing_Stop_Cooldown'] = 0  # Days remaining in cooldown (0 = no cooldown)
    
    # ============================================================================
    # STEP A: Mark-to-Market existing positions using Yesterday's Close
    # ============================================================================
    for idx, row in portfolio.iterrows():
        symbol = row['Symbol']
        hist = hist_cache.get(symbol, pd.DataFrame())
        
        if hist.empty:
            # Keep existing price if no data
            current_price = row.get('Price', 0)
            if current_price > 0:
                portfolio.at[idx, 'Price'] = current_price
                portfolio.at[idx, 'Market_Value'] = row.get('Quantity', 0) * current_price
            continue
        
        # Get yesterday's close for mark-to-market (strict temporal isolation)
        # Use data strictly BEFORE today (yesterday or earlier)
        hist_before_today = hist[hist.index < date_ts]
        if not hist_before_today.empty:
            yesterday_close = hist_before_today['Close'].iloc[-1]
            portfolio.at[idx, 'Price'] = yesterday_close
            portfolio.at[idx, 'Market_Value'] = row.get('Quantity', 0) * yesterday_close
            
            # DYNAMIC TRAILING VETO: Check for trailing stop trigger
            # Get current high water mark (init to yesterday_close if 0)
            current_hwm = portfolio.at[idx, 'High_Water_Mark']
            if current_hwm == 0 or current_hwm < yesterday_close:
                # Initialize or update high water mark
                portfolio.at[idx, 'High_Water_Mark'] = yesterday_close
                current_hwm = yesterday_close
            
            # Check if price has dropped by trailing_stop_pct from high water mark
            if current_hwm > 0 and row.get('Quantity', 0) > 0:
                drop_pct = ((current_hwm - yesterday_close) / current_hwm * 100) if current_hwm > 0 else 0
                if drop_pct >= trailing_stop_pct:
                    # Trailing stop triggered - force SELL action
                    # Store in portfolio for use in decision logic
                    portfolio.at[idx, '_trailing_stop_triggered'] = True
                    portfolio.at[idx, '_trailing_stop_drop_pct'] = np.float64(drop_pct)
                    portfolio.at[idx, '_trailing_stop_hwm'] = np.float64(current_hwm)
                    # Start 5-day cooldown period (blacklist from buying)
                    portfolio.at[idx, 'Trailing_Stop_Cooldown'] = 5
                else:
                    portfolio.at[idx, '_trailing_stop_triggered'] = False
                    
                    # Decrement cooldown if active
                    current_cooldown = int(portfolio.at[idx, 'Trailing_Stop_Cooldown'])
                    if current_cooldown > 0:
                        portfolio.at[idx, 'Trailing_Stop_Cooldown'] = current_cooldown - 1
        else:
            # No historical data before today, keep existing price
            current_price = row.get('Price', 0)
            if current_price > 0:
                portfolio.at[idx, 'Price'] = current_price
                portfolio.at[idx, 'Market_Value'] = row.get('Quantity', 0) * current_price
            portfolio.at[idx, '_trailing_stop_triggered'] = False
    
    # Recalculate total_value after mark-to-market
    total_value = portfolio['Market_Value'].sum() + cash
    
    # Skip trading during warmup period (prime moving averages)
    if day_number < warmup_days:
        # During warmup, only mark-to-market but don't make trade decisions
        # Return empty trades list but updated portfolio state
        equity = portfolio['Market_Value'].sum()
        daily_stats = {
            'date': date,
            'equity': equity,
            'cash': cash,
            'total_value': total_value,
            'drawdown': 0.0,
            'num_buys': 0,
            'num_avoids': 0
        }
        return portfolio, cash, [], daily_stats
    
    # Decision logging: Track first 5 symbols on first 3 days
    decision_log = []
    log_decisions = (day_number < 3)
    symbols_logged = 0
    max_symbols_to_log = 5
    
    # Check if we're in a bull regime for cash deployment
    regime = macro_regime.get('regime', 'NEUTRAL')
    is_bull_regime = regime in ['BULL', 'EXPANSION', 'RISK-ON']
    
    # Collect all decisions first for cash deployment logic
    symbol_decisions = {}  # Store decisions for cash deployment
    
    # ============================================================================
    # STEP B: Run arbitrate_final_decision using data through Yesterday
    # STEP C: Execute trades at Today's Close price
    # STEP D: Update Portfolio State
    # ============================================================================
    for idx, row in portfolio.iterrows():
        symbol = row['Symbol']
        hist = hist_cache.get(symbol, pd.DataFrame())
        news = news_cache.get(symbol, [])
        info = info_cache.get(symbol, {})
        
        if hist.empty:
            continue
        
        # STRICT TEMPORAL ISOLATION: Use data strictly BEFORE today for decision-making
        # hist_slice contains only data up to (but not including) today
        hist_slice = hist[hist.index < date_ts]
        if hist_slice.empty:
            # No historical data before today, skip this symbol
            continue
        
        # Get yesterday's close for decision-making (last available price before today)
        yesterday_close = hist_slice['Close'].iloc[-1]
        
        # Get today's close for execution (with timezone safety)
        if date_ts in hist.index:
            today_close = hist.loc[date_ts, 'Close']
        else:
            # Today's price not available, use yesterday's close for execution too
            today_close = yesterday_close
        
        if pd.isna(yesterday_close) or pd.isna(today_close) or yesterday_close == 0 or today_close == 0:
            continue
        
        # Prepare row data for calculations (use yesterday's price for decision context)
        row_dict = row.to_dict()
        row_dict['Price'] = yesterday_close  # Use yesterday's price for decision context
        row_dict['Market_Value'] = row.get('Quantity', 0) * yesterday_close
        
        # DECISION CACHING: Check if we should use cached decisions (Monte Carlo optimization)
        cache_key = f"{date}_{symbol}"
        if decision_cache and cache_key in decision_cache:
            # Use cached decision data (skip expensive alpha/decision calculations)
            cached = decision_cache[cache_key]
            daily_indicators = cached.get('daily_indicators', {})
            liq_tier = cached.get('liq_tier', 'UNKNOWN')
            liq_reason = cached.get('liq_reason', 'Unknown')
            alpha_score = cached.get('alpha_score', 50)
            decision = cached.get('decision', {'action': 'HOLD', 'confidence': 'Low'})
            # Update row_dict with cached indicators
            for key in ['Return_7d', 'Return_30d', 'Return_90d', 'Pct_From_52w_High', 'Pct_From_52w_Low',
                       'Volatility_60d', 'MA50', 'MA200', 'Drawdown_90d']:
                if key in daily_indicators:
                    row_dict[key] = daily_indicators[key]
        else:
            # FIX STATIC ALPHA: Recalculate indicators from hist_slice daily (fresh calculation each day)
            daily_indicators = calculate_daily_indicators(hist_slice, yesterday_close)
            # Update row_dict with fresh indicators
            row_dict['Return_7d'] = daily_indicators['Return_7d']
            row_dict['Return_30d'] = daily_indicators['Return_30d']
            row_dict['Return_90d'] = daily_indicators['Return_90d']
            row_dict['Pct_From_52w_High'] = daily_indicators['Pct_From_52w_High']
            row_dict['Pct_From_52w_Low'] = daily_indicators['Pct_From_52w_Low']
            row_dict['Volatility_60d'] = daily_indicators['Volatility_60d']
            row_dict['MA50'] = daily_indicators['MA50']
            row_dict['MA200'] = daily_indicators['MA200']
            row_dict['Drawdown_90d'] = daily_indicators['Drawdown_90d']
            
            # Calculate metrics using hist_slice (data through yesterday, no look-ahead)
            liq = calculate_liquidity_metrics(symbol, hist_slice, yesterday_close, row_dict['Market_Value'], total_value)
            liq_tier = liq.get('tier_code', 'UNKNOWN')
            liq_reason = liq.get('liquidity_reason', 'Unknown')
            liq_metrics = {'tier_code': liq_tier, 'max_position_pct': liq.get('max_position_pct', 0.0)}
        
            # Data confidence
            fund_dict = info.get('fundamentals', {})
            inferred = info.get('inferred_flags', {})
            data_conf = calculate_data_confidence(fund_dict, info.get('info_dict', {}), inferred)
            
            # Use fresh drawdown from indicators
            drawdown_90d = daily_indicators['Drawdown_90d']
            
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
            
            # Sell risk (use hist_slice - data through yesterday)
            ma50 = hist_slice['Close'].tail(50).mean() if len(hist_slice) >= 50 else yesterday_close
            ma200 = hist_slice['Close'].tail(200).mean() if len(hist_slice) >= 200 else yesterday_close
            sell_risk = calculate_sell_risk(row_dict, hist_slice, ma50, ma200, news, macro_regime)
            
            # Alpha (using hist_slice - no look-ahead)
            benchmark = get_benchmark_data(row_dict.get('metal', 'Gold'))
            alpha_result = calculate_alpha_models(row_dict, hist_slice, benchmark)
            alpha_score = alpha_result.get('alpha_score', 50)
            
            # Financing overhang
            overhang = calculate_financing_overhang(news, symbol, row_dict.get('Runway', 12.0), institutional_v3_available=False)
            row_dict['Financing_Overhang_Score'] = overhang['score']
            
            # Final decision (using data through yesterday, no look-ahead)
            discovery = (False, '')
            # Pass risk_mode and gsr_bias to arbitrate_final_decision for High-Torque Mode and V4.0 Phase 3
            decision = arbitrate_final_decision(
                row_dict, liq_metrics, data_conf, dilution, sell_risk,
                alpha_score, macro_regime, discovery, tape_gate,
                strict_mode=strict_mode,
                risk_mode=risk_mode,
                gsr_bias=gsr_bias  # V4.0 Phase 3: Automated GSR bonus application
            )
            
            # Save decision to cache if requested (for Monte Carlo optimization)
            # FORCE-ENABLE: Always save if save_decisions is True (ensures cache is populated)
            if save_decisions:
                # CRITICAL: Ensure decision_cache exists and is mutable (passed by reference)
                # If None, initialize it (should not happen, but safety check)
                if decision_cache is None:
                    decision_cache = {}
                # Store immutable copy of decision data (cache is modified in-place)
                decision_cache[cache_key] = {
                    'daily_indicators': daily_indicators.copy() if isinstance(daily_indicators, dict) else daily_indicators,
                    'liq_tier': liq_tier,
                    'liq_reason': liq_reason,
                    'alpha_score': alpha_score,
                    'decision': decision.copy() if isinstance(decision, dict) else decision  # Store a copy to avoid mutation
                }
        
        # Apply risk_mode adjustments after decision
        # AGGRESSIVE mode: Lower thresholds, raise position caps, disable non-critical vetoes
        if risk_mode == 'AGGRESSIVE':
            # Raise max position cap if AGGRESSIVE
            if decision.get('max_allowed_pct', 5.0) < 15.0:
                decision['max_allowed_pct'] = min(15.0, decision.get('max_allowed_pct', 5.0) * 1.5)
            
            # Disable non-critical vetoes (only keep Liquidity L0 and Dilution >= 80)
            # Non-critical vetoes: Data Confidence < 40, Sell Risk soft triggers
            if decision.get('veto_applied', False):
                veto_model = decision.get('veto_model', '')
                if veto_model == 'Risk' and 'Data confidence' not in str(decision.get('primary_gating_reason', '')):
                    # Disable sell risk soft vetoes in AGGRESSIVE mode
                    decision['veto_applied'] = False
                    decision['veto_model'] = None
                    decision['action'] = 'HOLD'  # Downgrade to HOLD instead of Avoid
                    decision['warnings'].append("⚠️ AGGRESSIVE mode: Non-critical veto disabled")
        
        # Strict Provenance Mode check (must happen before decision logging)
        if strict_mode and decision['action'] == 'Buy':
            # Check if key inputs are unknown/inferred (simplified check)
            if hist_slice.empty or not news:
                decision['action'] = 'HOLD'
                decision['confidence'] = 'Low'
                decision['primary_gating_reason'] = "Insufficient verified inputs (strict provenance mode)"
        
        # HARDENED TRAILING VETO: Check for 5-day cooldown before allowing buys
        trailing_stop_cooldown = row.get('Trailing_Stop_Cooldown', 0)
        if trailing_stop_cooldown > 0 and decision['action'] == 'Buy':
            # Symbol is in cooldown period - blacklist from buying
            decision['action'] = 'HOLD'
            decision['confidence'] = 'Low'
            decision['primary_gating_reason'] = f"Trailing stop cooldown: {trailing_stop_cooldown} days remaining (avoid catching falling knife)"
            decision['warnings'].append(f"⏳ Trailing stop cooldown: {trailing_stop_cooldown} days remaining - blacklisted from buying")
        
        # Apply tape gate
        if not tape_gate.get('new_buys_allowed', True) and decision['action'] == 'Buy':
            decision['action'] = 'HOLD'
            decision['warnings'].append("Tape gate blocked new buy")
        
        # Decision logging: First 5 symbols on first 3 days (after all decision modifications)
        if log_decisions and symbols_logged < max_symbols_to_log:
            decision_log.append({
                'symbol': symbol,
                'date': date,
                'alpha_score': alpha_score,
                'action': decision.get('action', 'HOLD'),
                'primary_gating_reason': decision.get('primary_gating_reason', ''),
                'veto_applied': decision.get('veto_applied', False),
                'veto_model': decision.get('veto_model', '')
            })
            symbols_logged += 1
        
        # Calculate target position (using yesterday's price for current value)
        current_value = row_dict.get('Market_Value', 0) or (row.get('Quantity', 0) * yesterday_close)
        current_pct = (current_value / total_value * 100) if total_value > 0 else 0
        target_pct = decision.get('recommended_pct', current_pct)
        target_value = (target_pct / 100.0) * total_value
        trade_dollars = target_value - current_value
        
        # HARDENED TRAILING VETO: Check for 5-day cooldown period
        trailing_stop_cooldown = row.get('Trailing_Stop_Cooldown', 0)
        if trailing_stop_cooldown > 0:
            # Symbol is in cooldown period - blacklist from buying
            if decision.get('action') == 'Buy':
                decision['action'] = 'HOLD'
                decision['confidence'] = 'Low'
                decision['primary_gating_reason'] = f"Trailing stop cooldown: {trailing_stop_cooldown} days remaining (avoid catching falling knife)"
                decision['warnings'].append(f"⏳ Trailing stop cooldown: {trailing_stop_cooldown} days remaining - blacklisted from buying")
        
        # DYNAMIC TRAILING VETO: Override decision if trailing stop triggered
        trailing_stop_triggered = row.get('_trailing_stop_triggered', False)
        if trailing_stop_triggered:
            # Trailing stop overrides all other signals - force SELL regardless of sell_policy
            decision['action'] = 'Sell'
            decision['confidence'] = 'High'
            decision['veto_applied'] = True
            decision['veto_model'] = 'Trailing Stop'
            drop_pct = row.get('_trailing_stop_drop_pct', 0)
            hwm = row.get('_trailing_stop_hwm', 0)
            decision['primary_gating_reason'] = f"Trailing stop triggered: Price dropped {drop_pct:.1f}% from high water mark ${hwm:.2f}"
            decision['reasoning'].append(f"Profit protection: {drop_pct:.1f}% drop from peak ${hwm:.2f}")
            decision['warnings'].append(f"🛑 Trailing stop: {drop_pct:.1f}% drop from ${hwm:.2f}")
        
        # Apply sell policy: only sell if policy allows it (unless trailing stop triggered)
        action = decision.get('action', 'HOLD')
        veto_applied = decision.get('veto_applied', False)
        veto_model = decision.get('veto_model', '')
        
        # Trailing stop bypasses sell_policy - always allow SELL
        if trailing_stop_triggered:
            should_sell = True  # Force sell on trailing stop
        else:
            should_sell = False  # Will be set below based on sell_policy
        
        # Store decision data for cash deployment logic (after veto_applied/veto_model are set)
        symbol_decisions[symbol] = {
            'decision': decision,
            'alpha_score': alpha_score,
            'current_pct': current_pct,
            'current_value': current_value,
            'target_pct': target_pct,
            'target_value': target_value,
            'trade_dollars': trade_dollars,
            'row_dict': row_dict,
            'idx': idx,
            'yesterday_close': yesterday_close,
            'today_close': today_close,
            'liq_tier': liq_tier,
            'liq_reason': liq_reason,
            'action': action,
            'veto_applied': veto_applied,
            'veto_model': veto_model,
            'trailing_stop_triggered': trailing_stop_triggered
        }
        
        # V5.0 Sovereign Hunter: Check threshold rebalancing for ADD action (before BUY/SELL logic)
        # Get target weight from decision
        decision_target_pct = decision.get('recommended_pct', row.get('Pct_Portfolio', 0))
        if decision_target_pct <= 0:
            decision_target_pct = current_pct
        
        threshold_rebalance = check_threshold_rebalancing(current_pct, decision_target_pct, tolerance_pct=5.0)
        
        # If threshold rebalancing says ADD, override decision to Buy
        if threshold_rebalance['triggered'] and threshold_rebalance['action'] == 'ADD':
            if action == 'HOLD' or action == 'Avoid':
                # Override to Buy for threshold rebalancing
                decision['action'] = 'Buy'
                action = 'Buy'
                decision['primary_gating_reason'] = threshold_rebalance['reason']
                decision['warnings'].append(f"⚖️ V5.0 Threshold Rebalance: {threshold_rebalance['reason']}")
                # Calculate target trade size
                target_value = (threshold_rebalance['target_pct'] / 100.0) * total_value
                buy_value = target_value - current_value
                if buy_value > 0:
                    trade_dollars = buy_value  # Positive = buy
        
        # CRITICAL: HOLD action explicitly prevents SELL orders (unless sell_policy == 'rebalance' OR trailing stop triggered)
        should_sell = False
        if trade_dollars < 0:  # Would be a sell
            # Trailing stop bypasses all sell_policy restrictions - force SELL
            # Also force full liquidation on trailing stop
            if trailing_stop_triggered:
                should_sell = True  # Force sell on trailing stop regardless of action/sell_policy
                # Override decision to ensure SELL action and full liquidation
                decision['action'] = 'Sell'
                action = 'Sell'
                # Force full position sale (set trade_dollars to current_value)
                trade_dollars = -current_value  # Negative = full sell
                decision['recommended_pct'] = 0.0  # Full liquidation
                decision['warnings'].append(f"🛑 TRAILING STOP: Full position liquidation")
            # FIRST: Check if action is HOLD - if so, block sell unless rebalance policy (or trailing stop)
            elif action == 'HOLD':
                if sell_policy == 'rebalance':
                    # Rebalance mode: allow sells even on HOLD if position exceeds target
                    # But only if it's a significant over-weight
                    if abs(trade_dollars) / total_value < 0.01:  # Less than 1% deviation
                        should_sell = False  # Don't sell on HOLD for tiny deviations
                    else:
                        should_sell = True  # Allow rebalance-driven sell even on HOLD
                else:
                    # Veto-only or triggers: HOLD explicitly means NO SELL
                    should_sell = False
                    trade_dollars = 0  # Block sell trade
            elif action == 'Avoid':
                # Avoid action: only sell if sell_policy allows it
                if sell_policy == 'veto_only':
                    # Veto-only: Avoid doesn't trigger sell unless there's a hard veto
                    should_sell = (veto_applied and 
                                  (veto_model in ['Liquidity', 'Capital Structure'] or 
                                   'L0' in str(decision.get('primary_gating_reason', ''))))
                elif sell_policy == 'triggers':
                    # Triggers: Avoid action triggers sell
                    should_sell = True
                elif sell_policy == 'rebalance':
                    # Rebalance: Allow sell
                    should_sell = True
            elif action == 'Sell':
                # Explicit Sell action: allow if policy permits
                if sell_policy == 'veto_only':
                    # Veto-only: Only if hard veto
                    should_sell = (veto_applied and 
                                  (veto_model in ['Liquidity', 'Capital Structure'] or 
                                   'L0' in str(decision.get('primary_gating_reason', ''))))
                else:
                    # Triggers or rebalance: Allow explicit Sell
                    should_sell = True
            else:
                # Other actions (Buy, etc.): Check sell_policy
                if sell_policy == 'veto_only':
                    # Only sell on hard veto
                    should_sell = (veto_applied and 
                                  (veto_model in ['Liquidity', 'Capital Structure'] or 
                                   'L0' in str(decision.get('primary_gating_reason', ''))))
                elif sell_policy == 'triggers':
                    # Sell on active sell triggers
                    should_sell = (veto_applied and veto_model in ['Sell Risk', 'Dilution Risk'])
                elif sell_policy == 'rebalance':
                    # Always rebalance to target
                    should_sell = True
        
        # Liquidity constraint
        daily_buy_limit_pct = LIQUIDITY_BUY_LIMITS.get(liq_tier, 0.0)
        
        if trade_dollars > 0:  # Buy
            if liq_tier == 'L0':
                trade_dollars = 0  # No buys for L0
            elif liq_tier == 'UNKNOWN':
                # Block new buys for UNKNOWN unless allow_leverage (capital protection)
                if not allow_leverage:
                    trade_dollars = 0
                else:
                    # Allow small buys with leverage enabled (conservative)
                    max_buy_dollars = (0.1 / 100.0) * total_value  # 0.1% max for UNKNOWN
                    trade_dollars = min(trade_dollars, max_buy_dollars)
            else:
                max_buy_dollars = (daily_buy_limit_pct / 100.0) * total_value
                trade_dollars = min(trade_dollars, max_buy_dollars)
            
            # Cash constraint
            if not allow_leverage and trade_dollars > cash:
                trade_dollars = cash
            
            if trade_dollars > 0:
                # V4.0: Check diversification veto BEFORE executing BUY
                symbol_metadata = {
                    'Jurisdiction': row_dict.get('Jurisdiction', 'Unknown'),
                    'Metal_Type': row_dict.get('Metal_Type', row_dict.get('metal', 'Gold'))
                }
                div_veto = check_diversification_veto(portfolio, symbol, symbol_metadata, total_value)
                if div_veto['veto_applied']:
                    # Diversification veto triggered - block the buy
                    decision['veto_applied'] = True
                    decision['veto_model'] = 'Diversification'
                    decision['primary_gating_reason'] = div_veto['veto_reason']
                    decision['warnings'].append(f"🚫 {div_veto['veto_reason']}")
                    trade_dollars = 0  # Block the trade
                    continue  # Skip to next symbol
                
                # Track intended trade
                intended_trades += 1
                # STEP C: Execute trade at Today's Close price (with liquidity-adjusted slippage)
                # Use today_close for execution (not yesterday_close used for decision)
                # Define slippage before any BUY logic (as requested)
                
                # Get base slippage without execution variance first
                base_slippage_multiplier = get_liquidity_adjusted_slippage(
                    liq_tier, risk_mode, execution_variance=0.0
                )
                # Apply execution variance after base slippage is calculated (Monte Carlo handshake)
                if execution_variance != 0.0:
                    import random
                    variance_multiplier = 1.0 + random.uniform(-execution_variance, execution_variance)
                    slippage_multiplier = base_slippage_multiplier * variance_multiplier
                else:
                    slippage_multiplier = base_slippage_multiplier
                
                # Calculate tiered_slippage_bps for logging (convert multiplier to bps)
                tiered_slippage_bps = int(slippage_multiplier * 10000)
                
                # Fallback: If UNKNOWN liquidity tier, default to 20 bps (standard default)
                if liq_tier == 'UNKNOWN' and tiered_slippage_bps == 100:
                    tiered_slippage_bps = 20  # Fallback to standard default
                
                # MARKET IMPACT TRACKING: Calculate market impact for capacity analysis
                # Market_Impact = Trade_Dollars / Daily_Volume_Dollars
                # Use 20-day average daily volume for capacity constraint
                avg_20d_volume_dollars = 0.0
                daily_volume_dollars = 0.0
                if 'Volume' in hist.columns:
                    # Calculate 20-day average volume (using data up to yesterday for decision)
                    hist_slice_for_volume = hist[hist.index < date_ts]
                    if len(hist_slice_for_volume) >= 20 and 'Volume' in hist_slice_for_volume.columns:
                        # Get last 20 days of volume data
                        volume_20d = hist_slice_for_volume['Volume'].tail(20)
                        close_20d = hist_slice_for_volume['Close'].tail(20)
                        # Calculate average daily volume in dollars
                        volume_dollars_20d = volume_20d * close_20d
                        avg_20d_volume_dollars = volume_dollars_20d.mean() if len(volume_dollars_20d) > 0 else 0.0
                    
                    # Also get today's volume for market impact calculation
                    if date_ts in hist.index:
                        daily_volume = hist.loc[date_ts, 'Volume']
                        daily_volume_dollars = daily_volume * today_close if daily_volume > 0 else 0.0
                
                # Use daily volume for market impact reporting
                market_impact = (trade_dollars / daily_volume_dollars * 100) if daily_volume_dollars > 0 else 0.0
                
                # PROFESSIONAL LIQUIDITY GATE: 1.5% Impact Gate - Cancel trade if impact exceeds 1.5%
                # A professional fund would never intentionally take a 7% hit on entry. Better to stay in cash.
                if market_impact > 1.5:
                    # Trade canceled due to excessive market impact - skip to next symbol
                    continue  # Skip to next symbol
                
                # HARD CAPACITY CONSTRAINT: A single trade cannot exceed 10% of DAILY dollar volume
                # This prevents "Death Spiral" by limiting position size based on reality
                # ENFORCE: Calculate max_allowed_dollars and use min() to cap trade size BEFORE slippage
                max_allowed_dollars = daily_volume_dollars * 0.10 if daily_volume_dollars > 0 else float('inf')
                target_trade_dollars = trade_dollars  # Store original target
                actual_trade_dollars = min(target_trade_dollars, max_allowed_dollars)  # Hard constraint
                
                # Calculate partial fill metrics
                partial_fill_pct = 1.0  # Default: full fill
                idle_cash_from_cap = 0.0  # Track cash left idle due to capacity constraint
                if actual_trade_dollars < target_trade_dollars:
                    # Trade was capped - calculate partial fill
                    partial_fill_pct = actual_trade_dollars / target_trade_dollars
                    idle_cash_from_cap = target_trade_dollars - actual_trade_dollars  # Remaining cash stays idle
                    # Remaining funds stay in cash bucket (not executed)
                
                # Update trade_dollars to the capped value (BEFORE calculating slippage)
                trade_dollars = actual_trade_dollars
                
                # CRITICAL: Stop trade if slippage exceeds 5% (500bps) - prevents "Death Spiral"
                if slippage_multiplier > 0.05:  # 5% = 500bps
                    # Slippage too high - skip this trade to protect capital
                    continue  # Skip to next symbol
                
                # Subtract slippage from Close price directly
                execution_price = today_close * (1.0 - slippage_multiplier)
                
                # Validate execution price is positive (prevent $0.00 trades)
                if execution_price <= 0:
                    # Invalid execution price - skip this trade
                    continue  # Skip to next symbol
                
                # Calculate slippage dollars for reporting
                slippage = trade_dollars * slippage_multiplier
                actual_trade = trade_dollars - slippage
                shares = actual_trade / execution_price
                
                # Round shares to match portfolio precision (prevent dtype warnings)
                shares = round(shares, 6)  # 6 decimal places for fractional shares
                shares = float(shares)  # Ensure it's a Python float (matches float64)
                
                trades.append({
                    'date': date,
                    'symbol': symbol,
                    'side': 'BUY',
                    'dollars': trade_dollars,
                    'price': execution_price,  # Today's close for execution
                    'slippage': slippage,
                    'slippage_bps': tiered_slippage_bps,  # Use calculated tiered slippage
                    'action': decision.get('action', 'Buy'),
                    'confidence': decision.get('confidence', 'Medium'),
                    'veto_applied': decision.get('veto_applied', False),
                    'veto_model': decision.get('veto_model', ''),
                    'veto_reason': decision.get('primary_gating_reason', '') if decision.get('veto_applied', False) else '',
                    'tape_gate_allowed': tape_gate.get('new_buys_allowed', True),
                    'liquidity_tier': liq_tier,
                    'liquidity_reason': liq_reason,
                    'reason': decision.get('primary_gating_reason', 'Alpha signal'),
                    'market_impact_pct': round(market_impact, 2),  # Market impact percentage
                    'partial_fill_pct': round(partial_fill_pct * 100, 1) if partial_fill_pct < 1.0 else 100.0  # Partial fill percentage
                })
                
                # Track executed trade (passed all gates)
                executed_trades += 1
                
                # STEP D: Update Portfolio State
                cash = np.float64(cash - trade_dollars)  # Maintain np.float64 dtype
                # Ensure Quantity and Cost_Basis operations maintain np.float64 dtype
                portfolio.at[idx, 'Quantity'] = np.float64(portfolio.at[idx, 'Quantity']) + np.float64(shares)
                portfolio.at[idx, 'Cost_Basis'] = np.float64(portfolio.at[idx, 'Cost_Basis']) + np.float64(actual_trade)
                
                # Initialize High_Water_Mark for new position (set to purchase price)
                if portfolio.at[idx, 'High_Water_Mark'] == 0:
                    portfolio.at[idx, 'High_Water_Mark'] = np.float64(execution_price)
                
                num_buys += 1
        
        elif trade_dollars < 0 and should_sell:  # Sell (only if sell_policy allows OR trailing stop)
            # V4.0: Check rebalancing triggers BEFORE executing SELL
            rebalance_check = check_rebalancing_triggers(
                portfolio, symbol, current_value, total_value, row.get('Cost_Basis', 0)
            )
            
            # V5.0 Sovereign Hunter: Check threshold rebalancing (5% Relative Tolerance Band)
            # Get target weight from decision (Recommended_Pct)
            target_pct = decision.get('recommended_pct', row.get('Pct_Portfolio', 0))
            if target_pct <= 0:
                # If no target, use current weight as target (no rebalancing needed)
                target_pct = current_pct
            
            threshold_rebalance = check_threshold_rebalancing(current_pct, target_pct, tolerance_pct=5.0)
            
            # Track if this is a profit harvest (2-bagger) - initialize outside if block for scope
            is_profit_harvest = False
            house_money_value = 0.0
            
            # V5.0: If threshold rebalancing triggered (TRIM), override trade logic
            if threshold_rebalance['triggered'] and threshold_rebalance['action'] == 'TRIM':
                # Trim position: sell down to target
                target_value = (threshold_rebalance['target_pct'] / 100.0) * total_value
                sell_value = current_value - target_value
                if sell_value > 0:
                    # Check market impact before executing
                    hist = hist_cache.get(symbol, pd.DataFrame())
                    daily_volume_dollars = 0.0
                    if not hist.empty and date_ts in hist.index and 'Volume' in hist.columns:
                        daily_volume = hist.loc[date_ts, 'Volume']
                        daily_volume_dollars = daily_volume * today_close if daily_volume > 0 else 0.0
                    
                    threshold_market_impact = (sell_value / daily_volume_dollars * 100) if daily_volume_dollars > 0 else 0.0
                    
                    # Respect 1.5% Impact Gate
                    if threshold_market_impact > 1.5:
                        decision['warnings'].append(f"⏸️ V5.0 Threshold rebalance postponed: Market impact {threshold_market_impact:.2f}% exceeds 1.5% gate")
                        trade_dollars = 0
                        should_sell = False
                    else:
                        trade_dollars = -sell_value  # Negative = sell
                        decision['action'] = 'Sell'
                        decision['primary_gating_reason'] = threshold_rebalance['reason']
                        decision['warnings'].append(f"⚖️ V5.0 Threshold Rebalance: {threshold_rebalance['reason']}")
                else:
                    trade_dollars = 0
                    should_sell = False
            # V4.0: If V4.0 rebalancing trigger fired, override trade_dollars to match target
            elif rebalance_check['triggered']:
                if rebalance_check['trigger_type'] == 'REBALANCE_SELL':
                    # Check if this is profit harvesting (2-bagger)
                    cost_basis = row.get('Cost_Basis', 0)
                    if cost_basis > 0:
                        unrealized_gain_pct = ((current_value - cost_basis) / cost_basis * 100) if cost_basis > 0 else 0.0
                        if unrealized_gain_pct > 100.0:
                            is_profit_harvest = True
                            # Calculate house money (profit portion that stays)
                            house_money_value = current_value - cost_basis
                    
                    # Calculate target value based on target_pct
                    target_value = (rebalance_check['target_pct'] / 100.0) * total_value
                    # Calculate how much to sell to reach target
                    sell_value = current_value - target_value
                    if sell_value > 0:
                        # V4.0 Safety Check: Check market impact BEFORE executing rebalance sell
                        # Get daily volume for market impact calculation
                        hist = hist_cache.get(symbol, pd.DataFrame())
                        daily_volume_dollars = 0.0
                        if not hist.empty and date_ts in hist.index and 'Volume' in hist.columns:
                            daily_volume = hist.loc[date_ts, 'Volume']
                            daily_volume_dollars = daily_volume * today_close if daily_volume > 0 else 0.0
                        
                        # Calculate market impact for the proposed rebalance sell
                        rebalance_market_impact = (sell_value / daily_volume_dollars * 100) if daily_volume_dollars > 0 else 0.0
                        
                        # V4.0: If market impact > 1.5%, postpone rebalance sell until next day
                        if rebalance_market_impact > 1.5:
                            # Postpone rebalance - skip this trade
                            decision['warnings'].append(f"⏸️ Rebalance postponed: Market impact {rebalance_market_impact:.2f}% exceeds 1.5% gate (will retry tomorrow)")
                            trade_dollars = 0
                            should_sell = False
                            continue  # Skip to next symbol
                        
                        # Market impact is acceptable - proceed with rebalance
                        trade_dollars = -sell_value  # Negative = sell
                        decision['action'] = 'Sell'
                        decision['primary_gating_reason'] = rebalance_check['reason']
                        if is_profit_harvest:
                            decision['warnings'].append(f"💰 {rebalance_check['reason']} | House Money: ${house_money_value:,.0f} remains")
                        else:
                            decision['warnings'].append(f"⚖️ {rebalance_check['reason']}")
                    else:
                        # Already at or below target, no sell needed
                        trade_dollars = 0
                        should_sell = False
            
            # Track intended trade
            intended_trades += 1
            # STEP C: Execute trade at Today's Close price (with slippage)
            # Use today_close for execution (not yesterday_close used for decision)
            execution_price = today_close
            
            trade_dollars = abs(trade_dollars)
            current_quantity = np.float64(row.get('Quantity', 0))  # Ensure np.float64 type
            
            # Trailing stop: Full position liquidation
            if trailing_stop_triggered:
                actual_shares = current_quantity  # Sell all shares
                trade_dollars = actual_shares * execution_price  # Recalculate trade dollars
            else:
                # Normal sell: Calculate shares from trade_dollars
                shares = trade_dollars / execution_price
                actual_shares = min(shares, current_quantity)
            
            # Round actual_shares to match portfolio precision (prevent dtype warnings)
            actual_shares = round(actual_shares, 6)  # 6 decimal places for fractional shares
            actual_shares = float(actual_shares)  # Ensure it's a Python float (matches float64)
            
            actual_trade = actual_shares * execution_price
            
            # Calculate liquidity-adjusted slippage
            # Get base slippage without execution variance first
            base_slippage_multiplier = get_liquidity_adjusted_slippage(
                liq_tier, risk_mode, execution_variance=0.0
            )
            # Apply execution variance after base slippage is calculated (Monte Carlo handshake)
            if execution_variance != 0.0:
                import random
                variance_multiplier = 1.0 + random.uniform(-execution_variance, execution_variance)
                slippage_multiplier = base_slippage_multiplier * variance_multiplier
            else:
                slippage_multiplier = base_slippage_multiplier
            
            # Calculate tiered_slippage_bps for logging (convert multiplier to bps)
            tiered_slippage_bps = int(slippage_multiplier * 10000)
            
            # Fallback: If UNKNOWN liquidity tier, default to 20 bps (standard default)
            if liq_tier == 'UNKNOWN' and tiered_slippage_bps == 100:
                tiered_slippage_bps = 20  # Fallback to standard default
            
            # MARKET IMPACT TRACKING: Calculate market impact for capacity analysis
            # Use 20-day average daily volume for capacity constraint
            avg_20d_volume_dollars = 0.0
            daily_volume_dollars = 0.0
            if 'Volume' in hist.columns:
                # Calculate 20-day average volume (using data up to yesterday for decision)
                hist_slice_for_volume = hist[hist.index < date_ts]
                if len(hist_slice_for_volume) >= 20 and 'Volume' in hist_slice_for_volume.columns:
                    # Get last 20 days of volume data
                    volume_20d = hist_slice_for_volume['Volume'].tail(20)
                    close_20d = hist_slice_for_volume['Close'].tail(20)
                    # Calculate average daily volume in dollars
                    volume_dollars_20d = volume_20d * close_20d
                    avg_20d_volume_dollars = volume_dollars_20d.mean() if len(volume_dollars_20d) > 0 else 0.0
                
                # Also get today's volume for market impact calculation
                if date_ts in hist.index:
                    daily_volume = hist.loc[date_ts, 'Volume']
                    daily_volume_dollars = daily_volume * today_close if daily_volume > 0 else 0.0
            
            # Use daily volume for market impact reporting
            market_impact = (actual_trade / daily_volume_dollars * 100) if daily_volume_dollars > 0 else 0.0
            
            # PROFESSIONAL LIQUIDITY GATE: 1.5% Impact Gate - Cancel trade if impact exceeds 1.5%
            # A professional fund would never intentionally take a 7% hit on exit. Better to hold position.
            if market_impact > 1.5:
                # Trade canceled due to excessive market impact - skip to next symbol
                continue  # Skip to next symbol
            
            # HARD CAPACITY CONSTRAINT: A single trade cannot exceed 10% of DAILY dollar volume
            # ENFORCE: Calculate max_allowed_dollars and use min() to cap trade size BEFORE slippage
            max_allowed_dollars = daily_volume_dollars * 0.10 if daily_volume_dollars > 0 else float('inf')
            target_trade_dollars = actual_trade  # Store original target
            actual_trade_dollars = min(target_trade_dollars, max_allowed_dollars)  # Hard constraint
            
            # Calculate partial fill metrics
            partial_fill_pct = 1.0  # Default: full fill
            idle_cash_from_cap = 0.0  # Track cash left idle due to capacity constraint
            if actual_trade_dollars < target_trade_dollars:
                # Trade was capped - calculate partial fill
                partial_fill_pct = actual_trade_dollars / target_trade_dollars
                idle_cash_from_cap = target_trade_dollars - actual_trade_dollars  # Remaining cash stays idle
                actual_shares = actual_shares * partial_fill_pct  # Reduce shares proportionally
                # Remaining funds stay in cash bucket (not executed)
            
            # Update actual_trade to the capped value (BEFORE calculating slippage)
            actual_trade = actual_trade_dollars
            
            # CRITICAL: Stop trade if slippage exceeds 5% (500bps) - prevents "Death Spiral"
            if slippage_multiplier > 0.05:  # 5% = 500bps
                # Slippage too high - skip this trade to protect capital
                continue  # Skip to next symbol
            
            execution_price = today_close * (1.0 - slippage_multiplier)  # Subtract slippage from Close
            
            # Validate execution price is positive (prevent $0.00 trades)
            if execution_price <= 0:
                # Invalid execution price - skip this trade
                continue  # Skip to next symbol
            
            # Recalculate actual_trade after partial fill adjustment and slippage
            actual_trade = actual_shares * execution_price
            
            # Calculate slippage dollars for reporting
            slippage = actual_trade * slippage_multiplier
            net_proceeds = actual_trade - slippage
            
            # Validation: action must match side (HOLD should never produce SELL, except for trailing stop)
            trade_action = decision.get('action', 'Sell')
            if trade_action == 'HOLD' and sell_policy != 'rebalance' and not trailing_stop_triggered:
                # This should never happen with fixed veto-only logic (trailing stop is exception)
                raise ValueError(f"Invalid trade: action=HOLD but side=SELL for {symbol} on {date}. This indicates a bug in sell_policy logic.")
            
            # V4.0: Add house money info to trade log if profit harvesting
            trade_reason = decision.get('primary_gating_reason', 'Risk signal')
            if is_profit_harvest and house_money_value > 0:
                trade_reason = f"{trade_reason} | House Money: ${house_money_value:,.0f}"
            
            trades.append({
                'date': date,
                'symbol': symbol,
                'side': 'SELL',
                'dollars': actual_trade,
                'price': execution_price,  # Today's close for execution
                'slippage': slippage,
                'slippage_bps': tiered_slippage_bps,  # Use tiered slippage
                'action': trade_action,
                'confidence': decision.get('confidence', 'Medium'),
                'veto_applied': decision.get('veto_applied', False),
                'veto_model': decision.get('veto_model', ''),
                'veto_reason': decision.get('primary_gating_reason', '') if decision.get('veto_applied', False) else '',
                'tape_gate_allowed': tape_gate.get('new_buys_allowed', True),
                'liquidity_tier': liq_tier,
                'liquidity_reason': liq_reason,
                'reason': trade_reason,  # Includes house money info if profit harvesting
                'market_impact_pct': round(market_impact, 2),  # Market impact percentage
                'partial_fill_pct': round(partial_fill_pct * 100, 1) if partial_fill_pct < 1.0 else 100.0,  # Partial fill percentage
                'house_money_value': round(house_money_value, 2) if is_profit_harvest else 0.0  # V4.0: Track house money
            })
            
            # Track executed trade (passed all gates)
            executed_trades += 1
            
            # STEP D: Update Portfolio State
            cash = np.float64(cash + net_proceeds)  # Maintain np.float64 dtype
            # Ensure Quantity and Cost_Basis operations maintain np.float64 dtype
            portfolio.at[idx, 'Quantity'] = np.float64(portfolio.at[idx, 'Quantity']) - np.float64(actual_shares)
            
            # If trailing stop triggered, ensure full liquidation (Quantity already set to 0 above)
            if trailing_stop_triggered:
                portfolio.at[idx, 'Quantity'] = np.float64(0.0)
            
            current_cost_basis = np.float64(row.get('Cost_Basis', 0))  # Ensure np.float64 type
            current_mv = np.float64(row_dict.get('Market_Value', 0))  # Ensure np.float64 type
            if current_mv > 0:
                cost_reduction = actual_trade * (current_cost_basis / current_mv)
                portfolio.at[idx, 'Cost_Basis'] = np.float64(max(0, current_cost_basis - cost_reduction))
            
            # Reset High_Water_Mark if position is fully liquidated
            if portfolio.at[idx, 'Quantity'] == 0:
                portfolio.at[idx, 'High_Water_Mark'] = np.float64(0.0)
            
            num_avoids += 1 if decision['action'] == 'Avoid' else 0
        
        # STEP D (continued): Update market value using today's close after trades
        portfolio.at[idx, 'Price'] = today_close  # Use today's close for final mark-to-market
        portfolio.at[idx, 'Market_Value'] = portfolio.at[idx, 'Quantity'] * today_close
        
        # Reset High Water Mark: Update if today's close is a new high
        current_hwm = np.float64(portfolio.at[idx, 'High_Water_Mark'])
        if today_close > current_hwm and portfolio.at[idx, 'Quantity'] > 0:  # Only update if still holding
            # New high reached - reset high water mark
            portfolio.at[idx, 'High_Water_Mark'] = np.float64(today_close)
            # Clear trailing stop trigger flag when new high is reached
            portfolio.at[idx, '_trailing_stop_triggered'] = False
            # Clear cooldown when new high is reached (position recovering)
            if portfolio.at[idx, 'Trailing_Stop_Cooldown'] > 0:
                portfolio.at[idx, 'Trailing_Stop_Cooldown'] = 0
    
    # ============================================================================
    # CASH DEPLOYMENT LOGIC: Eliminate Cash Drag in Bull Regimes
    # Includes deploy_idle_cash function for High-Torque Mode
    # ============================================================================
    # Best Effort Deployment on Day 21 (first day after warmup)
    # Note: Trades are subject to 1.5% Impact Gate and 10% Volume Cap - no forced bad entries
    if force_deployment and cash > 0 and total_value > 0:
        cash_pct = (cash / total_value) * 100
        if cash_pct > 10.0:  # Force deployment if cash > 10%
            # Collect all symbols with alpha >= 40 (relaxed threshold for initial deployment)
            buy_candidates = []
            for sym, dec_data in symbol_decisions.items():
                if (dec_data['alpha_score'] >= 40 and
                    not dec_data['veto_applied'] and
                    dec_data['liq_tier'] != 'L0'):
                    buy_candidates.append(dec_data)
            
            # Sort by alpha score (descending)
            buy_candidates.sort(key=lambda x: x['alpha_score'], reverse=True)
            
            # Deploy all available cash to top Alpha-ranked symbols
            target_deployment = cash * 0.95  # Deploy 95% of cash
            deployed_cash = 0.0
            max_position_size = max_position_pct
            if risk_mode == 'AGGRESSIVE':
                max_position_size = 15.0
            
            for candidate in buy_candidates[:10]:  # Top 10 symbols
                if deployed_cash >= target_deployment:
                    break
                
                symbol = candidate['row_dict'].get('Symbol', '')
                idx = candidate['idx']
                current_value = candidate['current_value']
                today_close = candidate['today_close']
                
                current_pct = (current_value / total_value * 100) if total_value > 0 else 0
                max_position_value = (max_position_size / 100.0) * total_value
                remaining_capacity = max(0, max_position_value - current_value)
                
                if remaining_capacity > 0:
                    deploy_amount = min(remaining_capacity, target_deployment - deployed_cash, cash - deployed_cash)
                    if deploy_amount > 100:
                        # Track intended trade
                        intended_trades += 1
                        
                        # Calculate market impact BEFORE execution (1.5% Impact Gate)
                        # Get daily volume for market impact calculation
                        hist = hist_cache.get(symbol, pd.DataFrame())
                        daily_volume_dollars = 0.0
                        if not hist.empty and date_ts in hist.index and 'Volume' in hist.columns:
                            daily_volume = hist.loc[date_ts, 'Volume']
                            daily_volume_dollars = daily_volume * today_close if daily_volume > 0 else 0.0
                        
                        market_impact = (deploy_amount / daily_volume_dollars * 100) if daily_volume_dollars > 0 else 0.0
                        
                        # PROFESSIONAL LIQUIDITY GATE: 1.5% Impact Gate - Skip if impact exceeds 1.5%
                        if market_impact > 1.5:
                            # Trade canceled due to excessive market impact - skip to next candidate
                            continue  # Skip to next candidate
                        
                        # Define slippage before trade execution
                        base_slippage_multiplier = get_liquidity_adjusted_slippage(
                            candidate['liq_tier'], risk_mode, execution_variance=0.0
                        )
                        # Apply execution variance after base slippage (Monte Carlo handshake)
                        if execution_variance != 0.0:
                            import random
                            variance_multiplier = 1.0 + random.uniform(-execution_variance, execution_variance)
                            slippage_multiplier = base_slippage_multiplier * variance_multiplier
                        else:
                            slippage_multiplier = base_slippage_multiplier
                        
                        tiered_slippage_bps = int(slippage_multiplier * 10000)
                        # Fallback for UNKNOWN tier
                        if candidate['liq_tier'] == 'UNKNOWN' and tiered_slippage_bps == 100:
                            tiered_slippage_bps = 20
                        
                        execution_price = today_close * (1.0 - slippage_multiplier)  # Subtract slippage from Close
                        slippage = deploy_amount * slippage_multiplier
                        actual_trade = deploy_amount - slippage
                        shares = actual_trade / execution_price
                        shares = round(shares, 6)
                        shares = float(shares)
                        
                        trades.append({
                            'date': date,
                            'symbol': symbol,
                            'side': 'BUY',
                            'dollars': deploy_amount,
                            'price': execution_price,
                            'slippage': slippage,
                            'slippage_bps': tiered_slippage_bps,  # Use calculated tiered slippage
                            'action': 'Buy',
                            'confidence': 'High',
                            'veto_applied': False,
                            'veto_model': '',
                            'veto_reason': '',
                            'tape_gate_allowed': True,
                            'liquidity_tier': candidate['liq_tier'],
                            'liquidity_reason': candidate.get('liq_reason', ''),
                            'reason': f'Day 21 Best Effort Deployment: Alpha {candidate["alpha_score"]:.0f}/100',
                            'market_impact_pct': round(market_impact, 2)  # Market impact percentage
                        })
                        
                        # Track executed trade (passed all gates)
                        executed_trades += 1
                        
                        cash = np.float64(cash - deploy_amount)  # Maintain np.float64 dtype
                        portfolio.at[idx, 'Quantity'] = np.float64(portfolio.at[idx, 'Quantity']) + np.float64(shares)
                        portfolio.at[idx, 'Cost_Basis'] = np.float64(portfolio.at[idx, 'Cost_Basis']) + np.float64(actual_trade)
                        portfolio.at[idx, 'Price'] = today_close
                        portfolio.at[idx, 'Market_Value'] = portfolio.at[idx, 'Quantity'] * today_close
                        deployed_cash += deploy_amount
                        num_buys += 1
            
            if deployed_cash > 0:
                total_value = portfolio['Market_Value'].sum() + cash
                print(f"    ✓ Best effort deployed ${deployed_cash:,.0f} to {num_buys} positions")
        
        # Note: High_Water_Mark initialization for Day 21 deployment handled above
    
    # Aggressive Deployment Logic: Minimum Invested floor
    # If in bull regime and cash > 20% of total_value, deploy aggressively
    if is_bull_regime and cash > 0 and total_value > 0:
        cash_pct = (cash / total_value) * 100
        # Aggressive Deployment Logic: Minimum Invested floor
        # If risk_mode is AGGRESSIVE or BALANCED and cash > 15%, deploy aggressively
        # (Lowered from 20% to 15% to reduce cash drag after trailing stop sells)
        if risk_mode in ['AGGRESSIVE', 'BALANCED'] and cash_pct > 15.0:
            # Find buy candidates (alpha >= 40, not vetoed, Buy action or HOLD with positive alpha)
            # In High-Torque Mode, we deploy even with alpha as low as 40
            buy_candidates = []
            for sym, dec_data in symbol_decisions.items():
                if (dec_data['alpha_score'] >= 40 and  # Allow alpha as low as 40 for aggressive deployment
                    not dec_data['veto_applied'] and
                    dec_data['liq_tier'] != 'L0'):
                    # Include Buy actions or HOLD with positive alpha (will force Buy deployment)
                    if dec_data['action'] in ['Buy'] or (dec_data['action'] == 'HOLD' and dec_data['alpha_score'] >= 40):
                        buy_candidates.append(dec_data)
            
            # Sort by alpha score (descending) - prioritize top Alpha-ranked symbols
            buy_candidates.sort(key=lambda x: x['alpha_score'], reverse=True)
            
            # Deploy to top 3 Alpha-ranked symbols
            top_candidates = buy_candidates[:3]
            # Calculate minimum deployment: ensure cash doesn't exceed 15% after deployment
            min_deployment = max(0, cash - (total_value * 0.15))  # Bring cash down to 15% max
            target_deployment = min(min_deployment, cash * 0.95)  # Deploy up to 95% of available cash
            deployed_cash = 0.0
            max_position_size = max_position_pct  # Use CLI parameter (default 10%, can be 15% in AGGRESSIVE)
            
            if risk_mode == 'AGGRESSIVE':
                max_position_size = 15.0  # Higher cap in aggressive mode
            
            for candidate in buy_candidates:
                if deployed_cash >= target_deployment:
                    break
                
                symbol = candidate['row_dict'].get('Symbol', '')
                idx = candidate['idx']
                current_value = candidate['current_value']
                yesterday_close = candidate['yesterday_close']
                today_close = candidate['today_close']
                liq_tier = candidate['liq_tier']
                
                # Calculate how much we can deploy to this symbol
                current_pct = (current_value / total_value * 100) if total_value > 0 else 0
                max_position_value = (max_position_size / 100.0) * total_value
                remaining_capacity = max(0, max_position_value - current_value)
                
                if remaining_capacity > 0:
                    # Deploy up to remaining capacity or remaining target
                    deploy_amount = min(remaining_capacity, target_deployment - deployed_cash, cash - deployed_cash)
                    
                    if deploy_amount > 100:  # Only deploy if meaningful (> $100)
                        # Define slippage before trade execution
                        base_slippage_multiplier = get_liquidity_adjusted_slippage(
                            liq_tier, risk_mode, execution_variance=0.0
                        )
                        # Apply execution variance after base slippage (Monte Carlo handshake)
                        if execution_variance != 0.0:
                            import random
                            variance_multiplier = 1.0 + random.uniform(-execution_variance, execution_variance)
                            slippage_multiplier = base_slippage_multiplier * variance_multiplier
                        else:
                            slippage_multiplier = base_slippage_multiplier
                        
                        tiered_slippage_bps = int(slippage_multiplier * 10000)
                        # Fallback for UNKNOWN tier
                        if liq_tier == 'UNKNOWN' and tiered_slippage_bps == 100:
                            tiered_slippage_bps = 20
                        
                        # Subtract slippage from Close price directly
                        execution_price = today_close * (1.0 - slippage_multiplier)
                        slippage = deploy_amount * slippage_multiplier
                        actual_trade = deploy_amount - slippage
                        shares = actual_trade / execution_price
                        shares = round(shares, 6)
                        shares = float(shares)
                        
                        trades.append({
                            'date': date,
                            'symbol': symbol,
                            'side': 'BUY',
                            'dollars': deploy_amount,
                            'price': execution_price,
                            'slippage': slippage,
                            'slippage_bps': tiered_slippage_bps,  # Use calculated tiered slippage
                            'action': 'Buy',
                            'confidence': 'High',
                            'veto_applied': False,
                            'veto_model': '',
                            'veto_reason': '',
                            'tape_gate_allowed': tape_gate.get('new_buys_allowed', True),
                            'liquidity_tier': liq_tier,
                            'liquidity_reason': candidate.get('liq_reason', ''),
                            'reason': f'Cash deployment: Alpha {candidate["alpha_score"]:.0f}/100 (Bull regime)'
                        })
                        
                        # Update portfolio
                        cash = np.float64(cash - deploy_amount)  # Maintain np.float64 dtype
                        portfolio.at[idx, 'Quantity'] = np.float64(portfolio.at[idx, 'Quantity']) + np.float64(shares)
                        portfolio.at[idx, 'Cost_Basis'] = np.float64(portfolio.at[idx, 'Cost_Basis']) + np.float64(actual_trade)
                        portfolio.at[idx, 'Price'] = today_close
                        portfolio.at[idx, 'Market_Value'] = portfolio.at[idx, 'Quantity'] * today_close
                        deployed_cash += deploy_amount
                        num_buys += 1
            
            if deployed_cash > 0:
                # Update total_value after cash deployment
                total_value = portfolio['Market_Value'].sum() + cash
    
    # Decision logging: Print for first 5 symbols on first 3 days
    if decision_log:
        print(f"\n[Decision Log - Day {day_number + 1}, {date}]")
        for log_entry in decision_log:
            print(f"  {log_entry['symbol']}: action={log_entry['action']}, alpha={log_entry['alpha_score']:.1f}, "
                  f"gating={log_entry['primary_gating_reason'][:50]}, veto={log_entry['veto_applied']}")
    
    # Daily stats (using today's prices for final mark-to-market)
    equity = portfolio['Market_Value'].sum()
    total_value = equity + cash
    daily_stats = {
        'date': date,
        'equity': equity,
        'cash': cash,
        'total_value': total_value,
        'drawdown': 0.0,  # Will calculate from peak
        'num_buys': num_buys,
        'num_avoids': num_avoids,
        'regime': regime,  # Track regime for reporting
        'intended_trades': intended_trades,  # Track intended trades (decisions made)
        'executed_trades': executed_trades  # Track executed trades (passed all gates)
    }
    
    return portfolio, cash, trades, daily_stats

def verify_cache(portfolio_csv: Optional[str], start_date: str, end_date: str, data_dir: Path, symbols: Optional[List[str]] = None) -> Tuple[bool, List[str]]:
    """
    Verify cache exists and is loadable for all required symbols.
    
    Args:
        portfolio_csv: Path to portfolio CSV (optional if symbols provided)
        start_date: Start date
        end_date: End date
        data_dir: Cache directory
        symbols: Optional list of symbols to verify (if portfolio_csv is None)
    
    Returns:
        (is_valid, missing_or_broken list)
    """
    if portfolio_csv:
        portfolio = pd.read_csv(portfolio_csv)
        required_symbols = portfolio['Symbol'].unique().tolist()
        
        if symbols:
            requested_symbols = [s.strip().upper() for s in symbols] if isinstance(symbols, list) else [s.strip().upper() for s in symbols.split(',')]
            required_symbols = [s for s in required_symbols if s in requested_symbols]
    elif symbols:
        # Use symbols directly
        if isinstance(symbols, list):
            required_symbols = [s.strip().upper() if isinstance(s, str) else s for s in symbols]
        else:
            required_symbols = [s.strip().upper() for s in symbols.split(',')]
    else:
        raise ValueError("verify_cache requires either portfolio_csv or symbols")
    
    # Sort for deterministic order
    required_symbols = sorted(required_symbols)
    
    missing_or_broken = []
    
    for symbol in required_symbols:
        cache_file = data_dir / f"{symbol}_{start_date}_{end_date}.csv"
        
        if not cache_file.exists():
            missing_or_broken.append(symbol)
            continue
        
        try:
            hist = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            hist = _normalize_price_df(hist)
            if hist.empty:
                missing_or_broken.append(symbol)
                continue
            # Cache is valid
        except Exception as e:
            missing_or_broken.append(symbol)
            continue
    
    return len(missing_or_broken) == 0, missing_or_broken

def build_cache_only(portfolio_csv: str, start_date: str, end_date: str, data_dir: Path, 
                     symbols: Optional[List[str]] = None, skip_missing: bool = False) -> Tuple[bool, List[str]]:
    """
    Build cache only (fetch and cache price data, no simulation).
    
    Returns:
        (success, failed_symbols list)
    """
    portfolio = pd.read_csv(portfolio_csv)
    symbols_to_fetch = portfolio['Symbol'].unique().tolist()
    
    if symbols:
        requested_symbols = [s.strip().upper() for s in symbols.split(',')]
        symbols_to_fetch = [s for s in symbols_to_fetch if s in requested_symbols]
    
    # Sort for deterministic order
    symbols_to_fetch = sorted(symbols_to_fetch)
    
    print(f"Building cache for {len(symbols_to_fetch)} symbol(s)...")
    
    # Use batch fetch
    hist_cache, missing_symbols = load_or_fetch_price_data_batch(
        symbols_to_fetch,
        start_date,
        end_date,
        data_dir,
        offline=False,
        allow_partial_cache=skip_missing
    )
    
    if missing_symbols and not skip_missing:
        print(f"\nError: Failed to cache {len(missing_symbols)} symbol(s):")
        for sym in sorted(missing_symbols):
            cache_file = data_dir / f"{sym}_{start_date}_{end_date}.csv"
            print(f"  - {sym}")
            print(f"    Expected cache: {cache_file}")
            print(f"    Date range: {start_date} to {end_date}")
        print(f"\nSuggested command to retry:")
        print(f"  python3 backtest_runner.py --start {start_date} --end {end_date} --symbols {','.join(sorted(missing_symbols))} --build_cache_only --data_dir {data_dir}")
        return False, missing_symbols
    
    if missing_symbols:
        print(f"\nWarning: Skipped {len(missing_symbols)} symbol(s) (--skip_missing_symbols enabled):")
        for sym in sorted(missing_symbols):
            print(f"  - {sym}")
    
    print(f"\n✓ Cache build complete: {len(hist_cache)} symbol(s) cached successfully")
    return True, missing_symbols


def calculate_drawdown_stats(equity_curve: pd.Series) -> Dict:
    """
    Calculate comprehensive drawdown statistics from an equity curve.

    Args:
        equity_curve: pd.Series with DatetimeIndex and float values representing
                      portfolio value over time.

    Returns:
        Dict with keys:
            max_drawdown_pct, max_drawdown_start, max_drawdown_end,
            max_drawdown_recovery, max_drawdown_duration_days,
            calmar_ratio, current_drawdown_pct, time_underwater_pct,
            avg_drawdown_pct
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return {
            'max_drawdown_pct': 0.0,
            'max_drawdown_start': None,
            'max_drawdown_end': None,
            'max_drawdown_recovery': None,
            'max_drawdown_duration_days': 0,
            'calmar_ratio': 0.0,
            'current_drawdown_pct': 0.0,
            'time_underwater_pct': 0.0,
            'avg_drawdown_pct': 0.0,
        }

    equity = equity_curve.sort_index().astype(float)
    running_max = equity.cummax()

    # Drawdown series as negative percentages (0.0 means at peak)
    drawdown_pct = (equity - running_max) / running_max * 100.0

    # --- Max drawdown identification ---
    max_dd_idx = drawdown_pct.idxmin()
    max_dd_pct = float(drawdown_pct.loc[max_dd_idx])  # negative value

    # Find the peak that precedes the trough
    peak_value = running_max.loc[max_dd_idx]
    # The peak date is the last date where equity equalled the running max
    # before (or at) the trough date
    pre_trough = equity.loc[:max_dd_idx]
    peak_dates = pre_trough[pre_trough >= peak_value].index
    max_dd_start = peak_dates[-1] if len(peak_dates) > 0 else equity.index[0]

    max_dd_end = max_dd_idx

    # Duration from peak to trough in calendar days
    max_dd_duration = (max_dd_end - max_dd_start).days

    # Recovery: first date after trough where equity >= peak_value
    post_trough = equity.loc[max_dd_end:]
    recovered = post_trough[post_trough >= peak_value]
    max_dd_recovery = recovered.index[0] if len(recovered) > 0 else None
    # If recovery date equals the trough itself and there are more dates after,
    # that means the trough *is* back at peak (dd ~ 0), which is fine.

    # --- Calmar ratio ---
    # Annualized return: CAGR over the full curve period
    total_days = (equity.index[-1] - equity.index[0]).days
    if total_days > 0 and equity.iloc[0] > 0:
        total_return = equity.iloc[-1] / equity.iloc[0]
        annualized_return = (total_return ** (365.0 / total_days) - 1.0) * 100.0
    else:
        annualized_return = 0.0

    if max_dd_pct != 0.0:
        calmar = annualized_return / abs(max_dd_pct)
    else:
        calmar = 0.0

    # --- Current drawdown ---
    current_dd = float(drawdown_pct.iloc[-1])

    # --- Time underwater ---
    underwater_days = int((drawdown_pct < 0.0).sum())
    time_underwater = (underwater_days / len(drawdown_pct)) * 100.0

    # --- Average drawdown ---
    # Average of all drawdown values (only the periods that are negative)
    negative_dd = drawdown_pct[drawdown_pct < 0.0]
    avg_dd = float(negative_dd.mean()) if len(negative_dd) > 0 else 0.0

    return {
        'max_drawdown_pct': max_dd_pct,
        'max_drawdown_start': max_dd_start,
        'max_drawdown_end': max_dd_end,
        'max_drawdown_recovery': max_dd_recovery,
        'max_drawdown_duration_days': max_dd_duration,
        'calmar_ratio': calmar,
        'current_drawdown_pct': current_dd,
        'time_underwater_pct': time_underwater,
        'avg_drawdown_pct': avg_dd,
    }


def calculate_transaction_costs(price: float, shares: float, side: str,
                                spread_bps: float = 50,
                                commission: float = 0.0) -> Dict:
    """
    Calculate transaction cost for a single trade, modelling bid-ask spread impact.

    For junior miners the default spread is 50 bps (0.50%).  The spread is
    split symmetrically around the mid-price:
        ask = price * (1 + spread_bps / 20000)
        bid = price * (1 - spread_bps / 20000)

    Args:
        price:      Mid-price of the security.
        shares:     Number of shares traded (positive).
        side:       'BUY' or 'SELL' (case-insensitive).
        spread_bps: Bid-ask spread in basis points (default 50 = 0.50%).
        commission: Flat commission per trade in dollars (default 0.0).

    Returns:
        Dict with keys:
            execution_price: The effective fill price after spread.
            cost:            Total dollar cost of the trade (spread + commission).
            cost_pct:        Total cost as a percentage of notional value.
            spread_impact:   Dollar cost attributable to the spread alone.
    """
    side_upper = side.strip().upper()
    half_spread = spread_bps / 20000.0

    if side_upper == 'BUY':
        execution_price = price * (1.0 + half_spread)
    elif side_upper == 'SELL':
        execution_price = price * (1.0 - half_spread)
    else:
        raise ValueError(f"side must be 'BUY' or 'SELL', got '{side}'")

    spread_impact = abs(execution_price - price) * shares
    total_cost = spread_impact + commission

    notional = price * shares
    cost_pct = (total_cost / notional * 100.0) if notional > 0.0 else 0.0

    return {
        'execution_price': execution_price,
        'cost': total_cost,
        'cost_pct': cost_pct,
        'spread_impact': spread_impact,
    }


def segment_by_regime(daily_df: pd.DataFrame,
                      gold_hist: Optional[pd.DataFrame] = None) -> Dict:
    """
    Segment backtest results by gold-market regime and compute per-regime
    performance statistics.

    Regime classification (requires gold price history):
        bull  : gold Close > 200-day MA  AND  50-day MA > 200-day MA
        bear  : gold Close < 200-day MA  AND  50-day MA < 200-day MA
        choppy: everything else

    If *gold_hist* is not provided, all days are classified as 'unknown'.

    Args:
        daily_df:  DataFrame with columns [date, equity, cash, total_value]
                   produced by the backtest simulation loop.
        gold_hist: Optional DataFrame with a 'Close' column indexed by date
                   (e.g. GLD ETF or GC=F futures data).

    Returns:
        Dict with keys 'bull', 'bear', 'choppy' (or 'unknown') each mapping
        to a stats dict:
            { days, annualized_return, sharpe, max_drawdown, win_rate,
              avg_daily_return }
        Plus a 'regime_series' key holding a pd.Series of regime labels
        aligned with the daily_df dates.
    """

    df = daily_df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
    else:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

    # Build the regime series
    if gold_hist is not None and not gold_hist.empty:
        gold = gold_hist.copy()
        if 'Close' not in gold.columns:
            # Attempt common alternatives
            for alt in ['Adj Close', 'close', 'adj_close']:
                if alt in gold.columns:
                    gold = gold.rename(columns={alt: 'Close'})
                    break
        if 'Close' not in gold.columns:
            # Cannot classify -- fall back to unknown
            regime_labels = pd.Series('unknown', index=df.index)
        else:
            gold.index = pd.to_datetime(gold.index)
            gold = gold.sort_index()
            ma200 = gold['Close'].rolling(window=200, min_periods=200).mean()
            ma50 = gold['Close'].rolling(window=50, min_periods=50).mean()

            regime_labels = pd.Series('choppy', index=gold.index)
            bull_mask = (gold['Close'] > ma200) & (ma50 > ma200)
            bear_mask = (gold['Close'] < ma200) & (ma50 < ma200)
            regime_labels[bull_mask] = 'bull'
            regime_labels[bear_mask] = 'bear'

            # Reindex to match daily_df dates, forward-fill for any
            # backtest dates missing from the gold series
            regime_labels = regime_labels.reindex(df.index, method='ffill')
            # Any remaining NaN (before first gold data) -> 'unknown'
            regime_labels = regime_labels.fillna('unknown')
    else:
        regime_labels = pd.Series('unknown', index=df.index)

    # Helper: compute stats for a subset of the equity curve
    def _regime_stats(sub_df: pd.DataFrame) -> Dict:
        n = len(sub_df)
        if n < 2:
            return {
                'days': n,
                'annualized_return': 0.0,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'avg_daily_return': 0.0,
            }

        tv = sub_df['total_value'].astype(float)
        daily_returns = tv.pct_change().dropna()

        # Annualized return
        total_return = tv.iloc[-1] / tv.iloc[0] if tv.iloc[0] > 0 else 1.0
        calendar_days = (tv.index[-1] - tv.index[0]).days
        if calendar_days > 0 and total_return > 0:
            ann_return = (total_return ** (365.0 / calendar_days) - 1.0) * 100.0
        else:
            ann_return = 0.0

        # Sharpe (annualized, assuming 252 trading days, risk-free = 0)
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        running_max = tv.cummax()
        dd = (tv - running_max) / running_max * 100.0
        max_dd = float(dd.min())

        # Win rate: fraction of days with positive returns
        if len(daily_returns) > 0:
            win_rate = float((daily_returns > 0).sum()) / len(daily_returns) * 100.0
        else:
            win_rate = 0.0

        avg_daily = float(daily_returns.mean() * 100.0) if len(daily_returns) > 0 else 0.0

        return {
            'days': n,
            'annualized_return': ann_return,
            'sharpe': float(sharpe),
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'avg_daily_return': avg_daily,
        }

    # Compute stats for each regime present
    result: Dict = {}
    unique_regimes = regime_labels.unique()
    for regime in unique_regimes:
        mask = regime_labels == regime
        sub = df.loc[mask]
        if 'total_value' not in sub.columns:
            # If the column is named differently, try equity + cash
            if 'equity' in sub.columns and 'cash' in sub.columns:
                sub = sub.copy()
                sub['total_value'] = sub['equity'].astype(float) + sub['cash'].astype(float)
            else:
                result[regime] = _regime_stats(pd.DataFrame())
                continue
        result[regime] = _regime_stats(sub)

    # Ensure the canonical regimes always appear in the output
    for canonical in ('bull', 'bear', 'choppy'):
        if canonical not in result:
            result[canonical] = {
                'days': 0,
                'annualized_return': 0.0,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'avg_daily_return': 0.0,
            }

    result['regime_series'] = regime_labels
    return result


def run_backtest(args):
    """Main backtest execution"""
    # Initialize execution variance for Monte Carlo (0.0 = no variance, 0.15 = +/-15%)
    execution_variance = 0.15 if args.monte_carlo else 0.0
    
    # Monte Carlo analysis: run N iterations with 10x speed optimization
    if args.monte_carlo and args.monte_carlo > 1:
        print(f"\n=== MONTE CARLO ANALYSIS: {args.monte_carlo} iterations ===")
        print(f"Execution variance: +/-15% to slippage on every trade")
        print(f"🚀 OPTIMIZED MODE: Running base strategy once, then iterating execution only\n")
        
        # STEP 1: Run base strategy ONCE (no execution variance) to cache all decisions
        print("Step 1: Running base strategy (caching decisions)...")
        base_result = _run_single_backtest(args, execution_variance=0.0, cache_decisions=True)
        decision_cache = base_result.get('decision_cache', {})
        
        # FORCE-ENABLE: Ensure cache is always available (should never be empty after iteration 0)
        if not decision_cache:
            print("⚠️  Warning: Decision cache is empty after base run. This should not happen.")
            print("   Falling back to standard Monte Carlo method (slower)...")
            decision_cache = {}  # Initialize empty cache
        
        # STEP 2: Execute-only iterations (10x faster) - ALWAYS use cache if available
        print(f"Step 2: Running {args.monte_carlo} execution-only iterations (using cached decisions)...")
        mc_results = []
        mc_start_time = time.time()
        
        for mc_iteration in range(args.monte_carlo):
            iter_start = time.time()
            # Execute from cached decisions with random variance
            result = _run_single_backtest(args, execution_variance=execution_variance, 
                                         use_decision_cache=decision_cache if decision_cache else None)
            mc_results.append(result)
            
            # Professional progress indicator
            elapsed_total = time.time() - mc_start_time
            avg_time_per_iter = elapsed_total / (mc_iteration + 1)
            remaining_iters = args.monte_carlo - (mc_iteration + 1)
            est_remaining_sec = avg_time_per_iter * remaining_iters
            
            # Format time remaining
            if est_remaining_sec < 60:
                time_str = f"{est_remaining_sec:.0f}s"
            elif est_remaining_sec < 3600:
                minutes = int(est_remaining_sec // 60)
                seconds = int(est_remaining_sec % 60)
                time_str = f"{minutes}m {seconds}s"
            else:
                hours = int(est_remaining_sec // 3600)
                minutes = int((est_remaining_sec % 3600) // 60)
                time_str = f"{hours}h {minutes}m"
            
            # Print progress (flush to ensure it appears immediately and updates properly)
            print(f"Iteration {mc_iteration + 1}/{args.monte_carlo} | Est. Remaining: {time_str} | CAGR: {result['cagr']:.2f}%", flush=True)
        
        # Calculate Monte Carlo statistics
        cagrs = [r['cagr'] for r in mc_results]
        cagrs_sorted = sorted(cagrs)
        mean_cagr = sum(cagrs) / len(cagrs)
        best_cagr = max(cagrs)
        worst_cagr = min(cagrs)
        safe_cagr = cagrs_sorted[int(len(cagrs_sorted) * 0.05)]  # 5th percentile
        
        luck_factor = best_cagr - mean_cagr
        
        print(f"\n=== MONTE CARLO RESULTS ===")
        print(f"Mean CAGR: {mean_cagr:.2f}%")
        print(f"Best CAGR: {best_cagr:.2f}%")
        print(f"Worst CAGR: {worst_cagr:.2f}%")
        print(f"Safe CAGR (5th percentile): {safe_cagr:.2f}%")
        print(f"Luck Factor (Best - Mean): {luck_factor:.2f}%")
        
        # Save Monte Carlo summary with institutional KPIs
        reports_dir = Path('./reports')
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate institutional metrics from ALL results (mean aggregation)
        # Fix: Calculate mean capacity limit estimate, not sum or best result
        capacity_estimates = []
        market_impacts = []
        execution_efficiencies = []
        
        for result in mc_results:
            inst_summary = result.get('institutional_summary', {})
            if inst_summary:
                cap_est = inst_summary.get('capacity_limit_estimate', 0)
                if cap_est > 0:  # Only include valid estimates
                    capacity_estimates.append(cap_est)
                market_impact = inst_summary.get('avg_market_impact_pct', 0)
                if market_impact > 0:
                    market_impacts.append(market_impact)
                exec_eff = inst_summary.get('execution_efficiency_pct', 100)
                execution_efficiencies.append(exec_eff)
        
        # Calculate mean values (not sum!)
        mean_capacity_limit = sum(capacity_estimates) / len(capacity_estimates) if capacity_estimates else 0.0
        avg_market_impact = sum(market_impacts) / len(market_impacts) if market_impacts else 0.0
        mean_execution_efficiency = sum(execution_efficiencies) / len(execution_efficiencies) if execution_efficiencies else 100.0
        
        mc_summary = {
            'iterations': args.monte_carlo,
            'execution_variance_pct': 15.0,
            'mean_cagr': mean_cagr,
            'best_cagr': best_cagr,
            'worst_cagr': worst_cagr,
            'safe_cagr': safe_cagr,
            'luck_factor': luck_factor,
            'all_cagrs': cagrs,
            'institutional_summary': {
                'safe_cagr_pct': round(safe_cagr, 2),
                'capacity_limit_estimate_m': round(mean_capacity_limit / 1_000_000, 2),  # Mean, not sum!
                'avg_market_impact_pct': round(avg_market_impact, 2),
                'execution_efficiency_pct': round(mean_execution_efficiency, 2)
            }
        }
        with open(reports_dir / 'monte_carlo_summary.json', 'w') as f:
            json.dump(mc_summary, f, indent=2)
        
        print(f"\nMonte Carlo summary saved to ./reports/monte_carlo_summary.json")
        print(f"\n=== INSTITUTIONAL SUMMARY ===")
        print(f"Safe CAGR (5th percentile): {safe_cagr:.2f}%")
        # Hard Rule: If mean CAGR is negative, show "$0 (Illiquid List)"
        if mean_cagr < 0:
            print(f"Capacity Limit Estimate: $0 (Illiquid List)")
        else:
            print(f"Capacity Limit Estimate: ${mean_capacity_limit/1_000_000:.2f}M")
        print(f"Avg Market Impact: {avg_market_impact:.2f}%")
        print(f"Execution Efficiency: {mean_execution_efficiency:.1f}%")
        return
    
    # Standard single backtest run
    _run_single_backtest(args, execution_variance=execution_variance)

def _run_single_backtest(args, execution_variance: float = 0.0, 
                         cache_decisions: bool = False, 
                         use_decision_cache: Optional[Dict] = None):
    """
    Run a single backtest iteration.
    
    Args:
        cache_decisions: If True, cache all decisions (alpha scores, vetoes, target sizes) for Monte Carlo optimization
        use_decision_cache: If provided, use cached decisions and only vary execution (Monte Carlo optimization)
    """
    # Parse dates (moved here so it's available in both paths)
    start_date = args.start
    end_date = args.end
    
    # Setup
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    trading_days = get_trading_days(start_date, end_date, data_dir, args.offline)
    # Sort trading days for deterministic order
    trading_days = sorted(trading_days)
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path('./reports')
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # For cache-only modes, portfolio_csv is optional
    if args.build_cache_only or args.verify_cache:
        # These modes can work with --symbols only
        if not args.portfolio_csv and not args.symbols:
            print("Error: --build_cache_only and --verify_cache require either --portfolio_csv or --symbols")
            print("\nExample commands:")
            print("  python3 backtest_runner.py --start 2024-01-01 --end 2024-06-30 --symbols AAPL,MSFT --build_cache_only --data_dir ./.backtest_cache/")
            print("  python3 backtest_runner.py --start 2024-01-01 --end 2024-06-30 --symbols AAPL,MSFT --verify_cache --data_dir ./.backtest_cache/")
            sys.exit(1)
    else:
        # Full backtest requires portfolio_csv and initial_cash
        # Default to portfolio_enhanced.csv if not specified
        if not args.portfolio_csv:
            default_portfolio = 'portfolio_enhanced.csv'
            # Check if default exists, otherwise require explicit specification
            if Path(default_portfolio).exists():
                args.portfolio_csv = default_portfolio
                print(f"Using default portfolio file: {default_portfolio}")
            else:
                print("Error: --portfolio_csv is required for full backtest")
                print("\nExample command:")
                print("  python3 backtest_runner.py --start 2024-01-01 --end 2024-06-30 --portfolio_csv portfolio_enhanced.csv --initial_cash 100000 --data_dir ./.backtest_cache/")
                sys.exit(1)
        if args.initial_cash is None:
            print("Error: --initial_cash is required for full backtest")
            print("\nExample command:")
            print("  python3 backtest_runner.py --start 2024-01-01 --end 2024-06-30 --portfolio_csv portfolio_enhanced.csv --initial_cash 100000 --data_dir ./.backtest_cache/")
            sys.exit(1)
    
    # Load portfolio (if provided)
    if args.portfolio_csv:
        portfolio = pd.read_csv(args.portfolio_csv)
    else:
        # For cache-only modes without portfolio_csv, create empty portfolio
        portfolio = pd.DataFrame(columns=['Symbol', 'Quantity', 'Price', 'Market_Value', 'Cost_Basis'])
    
    # Normalize column names (handle variations like CostBasis vs Cost_Basis)
    if 'CostBasis' in portfolio.columns and 'Cost_Basis' not in portfolio.columns:
        portfolio['Cost_Basis'] = portfolio['CostBasis']
    
    # Ensure required columns exist and cast to proper dtypes
    if 'Quantity' not in portfolio.columns:
        portfolio['Quantity'] = 0.0
    else:
        # Explicitly cast Quantity to np.float64 for precision over long backtests
        portfolio['Quantity'] = portfolio['Quantity'].astype(np.float64)
    
    if 'Cost_Basis' not in portfolio.columns:
        portfolio['Cost_Basis'] = 0.0
    else:
        # Explicitly cast Cost_Basis to np.float64 for precision over long backtests
        portfolio['Cost_Basis'] = portfolio['Cost_Basis'].astype(np.float64)
    
    # Initialize High_Water_Mark column for trailing stop tracking
    if 'High_Water_Mark' not in portfolio.columns:
        portfolio['High_Water_Mark'] = 0.0
    else:
        portfolio['High_Water_Mark'] = portfolio['High_Water_Mark'].astype(np.float64)
    
    # Initialize Trailing_Stop_Cooldown column for 5-day cooldown
    if 'Trailing_Stop_Cooldown' not in portfolio.columns:
        portfolio['Trailing_Stop_Cooldown'] = 0
    else:
        portfolio['Trailing_Stop_Cooldown'] = portfolio['Trailing_Stop_Cooldown'].astype('int64')
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
    # V4.0: Add Jurisdiction and Metal_Type columns (derived from Country and Metal if not present)
    if 'Jurisdiction' not in portfolio.columns:
        # Derive from Country column if available
        if 'Country' in portfolio.columns:
            portfolio['Jurisdiction'] = portfolio['Country']
        else:
            portfolio['Jurisdiction'] = 'Unknown'
    if 'Metal_Type' not in portfolio.columns:
        # Derive from Metal column if available
        if 'metal' in portfolio.columns:
            portfolio['Metal_Type'] = portfolio['metal']
        elif 'Metal' in portfolio.columns:
            portfolio['Metal_Type'] = portfolio['Metal']
        else:
            portfolio['Metal_Type'] = 'Gold'
    if 'cash' not in portfolio.columns:
        portfolio['cash'] = 10.0
    if 'burn_source' not in portfolio.columns:
        portfolio['burn_source'] = 'default'
    if 'Insider_Buying_90d' not in portfolio.columns:
        portfolio['Insider_Buying_90d'] = False
    if 'Pct_Portfolio' not in portfolio.columns:
        portfolio['Pct_Portfolio'] = 0.0
    
    # Ensure cash is float64 for dtype safety (explicit cast to np.float64)
    cash = np.float64(args.initial_cash) if args.initial_cash is not None else np.float64(0.0)
    
    # Determine symbols to process
    # PRIORITY: CLI symbols (--symbols) are the master list for utility modes (verify_cache/build_cache_only)
    # Skip portfolio validation entirely for utility flags
    if args.verify_cache or args.build_cache_only:
        # Utility modes: Prioritize --symbols flag as master list
        if args.symbols:
            symbols_to_process = [s.strip().upper() for s in args.symbols.split(',')]
            if args.verify_cache:
                print(f"Verifying cache for {len(symbols_to_process)} symbol(s) from --symbols flag: {', '.join(symbols_to_process)}")
            elif args.build_cache_only:
                print(f"Building cache for {len(symbols_to_process)} symbol(s) from --symbols flag: {', '.join(symbols_to_process)}")
        else:
            # Fallback to portfolio symbols if --symbols not provided (for utility modes)
            if len(portfolio) > 0 and 'Symbol' in portfolio.columns:
                symbols_to_process = portfolio['Symbol'].unique().tolist()
            else:
                symbols_to_process = []
            if symbols_to_process:
                if args.verify_cache:
                    print(f"Verifying cache for {len(symbols_to_process)} symbol(s) from portfolio: {', '.join(symbols_to_process)}")
                elif args.build_cache_only:
                    print(f"Building cache for {len(symbols_to_process)} symbol(s) from portfolio: {', '.join(symbols_to_process)}")
    else:
        # Normal backtest mode: Get symbols from portfolio
        if len(portfolio) > 0 and 'Symbol' in portfolio.columns:
            symbols_to_process = portfolio['Symbol'].unique().tolist()
        else:
            symbols_to_process = []
        
        # Filter symbols if --symbols flag provided (normal mode)
        if args.symbols:
            requested_symbols = [s.strip().upper() for s in args.symbols.split(',')]
            symbols_to_process = [s for s in symbols_to_process if s in requested_symbols]
            # Portfolio validation: ensure requested symbols exist in portfolio (only for normal backtest mode)
            if not symbols_to_process:
                print(f"Error: None of the requested symbols {requested_symbols} found in portfolio")
                sys.exit(1)
            # Filter portfolio to only include requested symbols
            if len(portfolio) > 0:
                portfolio = portfolio[portfolio['Symbol'].isin(symbols_to_process)].copy()
            if symbols_to_process:
                print(f"Processing {len(symbols_to_process)} symbol(s): {', '.join(symbols_to_process)}")
    
    # Handle verify_cache mode
    if args.verify_cache:
        # symbols_to_process is already set above (prioritizes --symbols flag)
        if not symbols_to_process:
            print("Error: --verify_cache requires either --symbols or --portfolio_csv with symbols")
            print("\nExample commands:")
            print("  python3 backtest_runner.py --start 2024-01-01 --end 2024-06-30 --symbols AAPL,MSFT --verify_cache --data_dir ./.backtest_cache/")
            print("  python3 backtest_runner.py --start 2024-01-01 --end 2024-06-30 --portfolio_csv portfolio.csv --verify_cache --data_dir ./.backtest_cache/")
            sys.exit(1)
        
        # Use symbols_to_process directly (master list from --symbols or portfolio)
        is_valid, missing = verify_cache(
            None,  # No portfolio CSV needed - use symbols directly
            start_date,
            end_date,
            data_dir,
            symbols=symbols_to_process
        )
        
        if is_valid:
            print("✓ Cache verification passed: All required symbols are cached and loadable")
            sys.exit(0)
        else:
            print(f"✗ Cache verification failed: {len(missing)} symbol(s) missing or broken:")
            for sym in sorted(missing):
                cache_file = data_dir / f"{sym}_{start_date}_{end_date}.csv"
                print(f"  - {sym} (expected: {cache_file})")
            print(f"\nTo build cache, run:")
            print(f"  python3 backtest_runner.py --start {start_date} --end {end_date} --symbols {','.join(sorted(missing))} --build_cache_only --data_dir {data_dir}")
            sys.exit(2)
    
    # Handle build_cache_only mode
    if args.build_cache_only:
        # symbols_to_process is already set above (prioritizes --symbols flag)
        if not symbols_to_process:
            print("Error: --build_cache_only requires either --symbols or --portfolio_csv with symbols")
            print("\nExample commands:")
            print("  python3 backtest_runner.py --start 2024-01-01 --end 2024-06-30 --symbols AAPL,MSFT --build_cache_only --data_dir ./.backtest_cache/")
            print("  python3 backtest_runner.py --start 2024-01-01 --end 2024-06-30 --portfolio_csv portfolio.csv --build_cache_only --data_dir ./.backtest_cache/")
            sys.exit(1)
        
        # Create a minimal portfolio CSV in memory for build_cache_only function
        import tempfile
        temp_portfolio = pd.DataFrame({'Symbol': symbols_to_process, 'Quantity': [0] * len(symbols_to_process)})
        temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_portfolio.to_csv(temp_csv.name, index=False)
        temp_csv.close()
        success, failed = build_cache_only(
            temp_csv.name,
            start_date,
            end_date,
            data_dir,
            symbols=None,  # Already filtered in temp CSV (symbols_to_process)
            skip_missing=args.skip_missing_symbols
        )
        import os
        os.unlink(temp_csv.name)
        sys.exit(0 if success else 1)
    
    # Handle retry_missing_cache mode (check manifest first)
    if args.retry_missing_cache:
        manifest = _load_manifest(data_dir)
        cache_key = f"{start_date}_{end_date}"
        if cache_key not in manifest:
            print(f"Error: No manifest found for date range {start_date} to {end_date}")
            print(f"Run a normal backtest first to create the cache manifest.")
            sys.exit(1)
        
        # Find missing symbols from manifest
        manifest_symbols = manifest[cache_key].get('symbols', {})
        missing_from_manifest = []
        for symbol in symbols_to_process:
            symbol_status = manifest_symbols.get(symbol, {})
            if symbol_status.get('status') != 'ok':
                missing_from_manifest.append(symbol)
        
        if not missing_from_manifest:
            print("✓ All symbols are cached (no missing symbols to retry)")
            sys.exit(0)
        
        # Use only missing symbols for retry
        symbols_to_process = missing_from_manifest
        print(f"Found {len(missing_from_manifest)} missing symbol(s) in manifest: {', '.join(missing_from_manifest)}")
    
    # Initialize caches
    hist_cache = {}
    news_cache = {}
    info_cache = {}
    
    # Load historical data for all symbols (batch mode)
    print(f"Loading price data for {len(symbols_to_process)} symbols...")
    
    # Sort symbols for deterministic order
    symbols_to_process = sorted(symbols_to_process)
    
    # Add benchmark symbols for dynamic macro regime calculation
    # GDX (gold miners ETF) and GLD/SLV for macro regime
    # GC=F and SI=F for V4.0 Phase 3: Automated GSR calculation
    benchmark_symbols = ['GDX', 'GLD', 'SLV', 'GC=F', 'SI=F']  # Sector benchmarks + GSR futures
    symbols_with_benchmarks = list(set(symbols_to_process + benchmark_symbols))
    
    # Use batch fetching for better reliability (includes benchmarks)
    # Note: In offline mode, if benchmarks are missing, use fallback macro regime
    hist_cache, missing_symbols = load_or_fetch_price_data_batch(
        symbols_with_benchmarks, 
        start_date, 
        end_date, 
        data_dir, 
        args.offline,
        allow_partial_cache=True,  # Allow partial cache for benchmarks (they're optional)
        optional_symbols=benchmark_symbols  # Benchmarks are optional (non-fatal if missing)
    )
    
    # Filter missing_symbols to exclude benchmarks (benchmarks are optional - non-fatal if missing)
    missing_non_benchmark = [s for s in missing_symbols if s not in benchmark_symbols]
    missing_benchmarks = [s for s in missing_symbols if s in benchmark_symbols]
    
    # Update missing_symbols to exclude benchmarks for error reporting
    missing_symbols = missing_non_benchmark
    
    # Track skipped symbols for reporting
    skipped_symbols = missing_symbols.copy() if (args.skip_missing_symbols or args.allow_partial_cache) else []
    
    # Check for missing symbols (excluding benchmarks which are optional)
    missing_non_benchmark = [s for s in missing_symbols if s not in benchmark_symbols]
    if missing_non_benchmark:
        if args.retry_missing_cache:
            # Retry mode: just fetch missing and exit (already handled above, but double-check)
            print(f"Retrying fetch for {len(missing_non_benchmark)} missing symbol(s)...")
            retry_cache, still_missing = load_or_fetch_price_data_batch(
                sorted(missing_non_benchmark),
                start_date,
                end_date,
                data_dir,
                offline=False,
                allow_partial_cache=False
            )
            if still_missing:
                print(f"Error: Still missing {len(still_missing)} symbol(s): {', '.join(sorted(still_missing))}")
                sys.exit(1)
            else:
                print("✓ All missing symbols cached successfully")
                sys.exit(0)
        else:
            # Normal mode: fail if missing symbols (unless skip_missing_symbols or allow_partial_cache)
            if args.skip_missing_symbols or args.allow_partial_cache:
                print(f"Warning: Missing price data for {len(missing_non_benchmark)} symbol(s): {', '.join(sorted(missing_non_benchmark))}")
                print("Proceeding with partial cache as requested...")
            else:
                error_msg = f"Missing price data for {len(missing_non_benchmark)} symbol(s):\n"
                error_msg += "\n".join(f"  - {sym}" for sym in sorted(missing_non_benchmark))
                error_msg += f"\n\nTo build cache, run:\n"
                error_msg += f"  python3 backtest_runner.py --start {start_date} --end {end_date} --portfolio_csv {args.portfolio_csv} --initial_cash {args.initial_cash} --data_dir {data_dir} --build_cache_only\n"
                error_msg += f"\nOr to retry only missing symbols:\n"
                error_msg += f"  python3 backtest_runner.py --start {start_date} --end {end_date} --symbols {','.join(sorted(missing_non_benchmark))} --retry_missing_cache --data_dir {data_dir}"
                print(f"Error: {error_msg}")
                sys.exit(1)
    
    # Note: Missing benchmarks (GDX, GLD, SLV) are non-fatal - macro regime will use fallback
    missing_benchmarks = [s for s in missing_symbols if s in benchmark_symbols]
    if missing_benchmarks and not args.offline:
        print(f"Warning: Benchmark symbols not cached: {', '.join(missing_benchmarks)}. Using simplified macro regime.")
    
    # Initialize news and fundamentals cache (simplified - use empty/defaults for backtest)
    for symbol in symbols_to_process:
        news_cache[symbol] = []
        info_cache[symbol] = {
            'fundamentals': {},
            'info_dict': {},
            'inferred_flags': {}
        }
    
    # Initialize portfolio prices on first day (for initial_value calculation)
    initial_equity = 0.0
    if trading_days:
        first_day = trading_days[0]
        first_day_ts = pd.Timestamp(first_day)
        for idx, row in portfolio.iterrows():
            symbol = row['Symbol']
            if symbol in hist_cache:
                hist = hist_cache[symbol]
                # Hist should already be normalized (tz-naive) from load_or_fetch_price_data
                # first_day_ts is tz-naive, so comparisons are safe
                # Find closest date to first_day
                hist_before = hist[hist.index <= first_day_ts]
                if not hist_before.empty:
                    first_price = hist_before['Close'].iloc[-1]
                    portfolio.at[idx, 'Price'] = first_price
                    portfolio.at[idx, 'Market_Value'] = portfolio.at[idx, 'Quantity'] * first_price
                    initial_equity += portfolio.at[idx, 'Market_Value']
                elif not hist.empty:
                    # Use first available price
                    portfolio.at[idx, 'Price'] = hist['Close'].iloc[0]
                    portfolio.at[idx, 'Market_Value'] = portfolio.at[idx, 'Quantity'] * portfolio.at[idx, 'Price']
                    initial_equity += portfolio.at[idx, 'Market_Value']
    
    # Initial value = initial cash + initial equity (positions valued at first trading day)
    initial_value_calc = cash + initial_equity
    
    # Count initial positions (symbols with Quantity > 0)
    num_positions_start = sum(1 for idx, row in portfolio.iterrows() if row.get('Quantity', 0) > 0)
    
    # Initialize decision cache for Monte Carlo optimization
    # FORCE-ENABLE: Always initialize cache to ensure it's available
    if use_decision_cache is not None:
        # Use provided cache (from iteration 0) - immutable copy
        decision_cache = use_decision_cache.copy() if use_decision_cache else {}
    elif cache_decisions:
        # Create new cache for iteration 0
        decision_cache = {}
    else:
        # Default: empty cache (will be populated if save_decisions is True)
        decision_cache = {}
    
    # Run simulation day by day
    all_trades = []
    daily_stats_list = []
    # Use calculated initial_value (cash + initial equity)
    peak_value = initial_value_calc if initial_value_calc > 0 else (args.initial_cash + portfolio['Market_Value'].sum())
    
    # Track buy-and-hold baseline: hold initial quantities throughout, mark-to-market daily
    buy_and_hold_portfolio = portfolio.copy()
    buy_and_hold_cash = cash
    buy_and_hold_initial_value = initial_value_calc
    
    # Cash-only baseline: stay in cash from day 1
    cash_only_initial_value = initial_value_calc
    cash_only_final_value = cash_only_initial_value  # Will be updated if cash changes (slippage, etc.)
    
    # Track regime for reporting (regime heatmap)
    regime_days = {'BULL': 0, 'EXPANSION': 0, 'RISK-ON': 0, 'NEUTRAL': 0, 'DEFENSIVE': 0}
    
    # Track regime profit for Regime Capture analysis
    regime_profit = {'BULL': 0.0, 'EXPANSION': 0.0, 'RISK-ON': 0.0, 'NEUTRAL': 0.0, 'DEFENSIVE': 0.0}
    prev_total_value = initial_value_calc  # Initialize for profit calculation
    
    print(f"Running backtest from {start_date} to {end_date} ({len(trading_days)} trading days)...")
    print(f"Warmup period: {args.warmup_days} days (no trades during warmup)")
    print(f"Risk mode: {args.risk_mode}")
    
    # Professional progress tracking
    start_time = time.time()
    for i, date in enumerate(trading_days):
        # Progress indicator: show every 10 days or on last day
        if i % 10 == 0 or i == len(trading_days) - 1:
            elapsed = time.time() - start_time
            days_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
            remaining_days = len(trading_days) - (i + 1)
            est_remaining_sec = remaining_days / days_per_sec if days_per_sec > 0 else 0
            
            # Format time remaining professionally
            if est_remaining_sec < 60:
                time_str = f"{est_remaining_sec:.0f}s"
            elif est_remaining_sec < 3600:
                minutes = int(est_remaining_sec // 60)
                seconds = int(est_remaining_sec % 60)
                time_str = f"{minutes}m {seconds}s"
            else:
                hours = int(est_remaining_sec // 3600)
                minutes = int((est_remaining_sec % 3600) // 60)
                time_str = f"{hours}h {minutes}m"
            
            print(f"  [{i+1}/{len(trading_days)}] {date} | Days/sec: {days_per_sec:.1f} | Est. Remaining: {time_str}")
        
        # DYNAMIC MACRO INTEGRATION: Calculate macro regime daily using fresh benchmark data
        # Use GDX (gold miners ETF) as sector benchmark
        benchmark_symbol = 'GDX'
        benchmark_hist = hist_cache.get(benchmark_symbol, pd.DataFrame())
        
        # Get benchmark hist_slice up to today (for daily regime calculation)
        benchmark_hist_slice = pd.DataFrame()
        date_ts = pd.to_datetime(date, utc=True).tz_localize(None)
        if not benchmark_hist.empty:
            benchmark_hist_slice = benchmark_hist[benchmark_hist.index <= date_ts]
        
        # Calculate macro regime dynamically using benchmark data
        # If sector crashes >10% in 2 days, switch to DEFENSIVE within 48 hours
        if not benchmark_hist_slice.empty and len(benchmark_hist_slice) >= 2:
            macro_regime = calculate_macro_regime(hist_slice=benchmark_hist_slice, date_ts=date_ts)
        else:
            # Fallback: use simplified regime if benchmark not available
            macro_regime = {
                'regime': 'NEUTRAL',
                'allow_new_buys': True,
                'throttle_factor': 1.0,
                'dxy': 0,
                'vix': 0,
                'factors': ['Backtest mode - neutral (benchmark unavailable)']
            }
        
        # Apply risk_mode settings to macro_regime (may override calculated regime)
        if args.risk_mode == 'AGGRESSIVE':
            if macro_regime['regime'] == 'NEUTRAL':  # Only override if neutral
                macro_regime['throttle_factor'] = 1.2  # 20% throttle boost
                macro_regime['regime'] = 'BULL'  # Treat as bull for aggressive deployment
            else:
                macro_regime['throttle_factor'] = max(macro_regime.get('throttle_factor', 1.0), 1.2)
        elif args.risk_mode == 'CONSERVATIVE':
            # Conservative mode: Use calculated regime or default to DEFENSIVE
            if macro_regime['regime'] == 'NEUTRAL':
                macro_regime['throttle_factor'] = 0.8  # 20% throttle reduction
                macro_regime['regime'] = 'DEFENSIVE'  # More defensive posture
        
        # Track regime for reporting
        regime = macro_regime.get('regime', 'NEUTRAL')
        if regime in regime_days:
            regime_days[regime] += 1
        
        # Warmup days: Skip trades but still mark-to-market and track stats
        skip_trades = (i < args.warmup_days)
        
        # Best Effort "Day 21" Deployment: Attempt to deploy idle cash on first day after warmup
        # Note: Trades subject to 1.5% Impact Gate and 10% Volume Cap - no forced bad entries
        is_first_trading_day = (i == args.warmup_days)
        force_deployment = False
        if is_first_trading_day and cash > 0:
            # Calculate available cash percentage
            total_value_before = portfolio['Market_Value'].sum() + cash
            cash_pct_before = (cash / total_value_before * 100) if total_value_before > 0 else 0
            if cash_pct_before > 10.0:  # If more than 10% cash after warmup, attempt best effort deployment
                force_deployment = True
                print(f"  🚀 Day {i+1} (Post-Warmup): Best effort deploying {cash_pct_before:.1f}% idle cash...")
        
        portfolio, cash, trades, daily_stats = simulate_day(
            date, portfolio.copy(), cash, hist_cache, news_cache, info_cache,
            args.strict_mode, args.allow_leverage, args.max_position_pct,
            sell_policy=args.sell_policy,
            day_number=i,  # Pass day number for decision logging
            risk_mode=args.risk_mode,
            warmup_days=args.warmup_days,
            force_deployment=force_deployment,  # Pass force_deployment flag
            trailing_stop_pct=args.trailing_stop_pct,  # Pass trailing stop percentage
            execution_variance=execution_variance,  # Pass execution variance for Monte Carlo
            decision_cache=decision_cache,  # Pass decision cache for Monte Carlo optimization
            save_decisions=cache_decisions  # Save decisions if caching enabled
        )
        
        # Trades already cleared during warmup in simulate_day
        # No need to revert here
        
        # Track regime profit for Regime Capture analysis
        # Store regime and daily profit for each day
        if 'regime_profit' not in locals():
            regime_profit = {'BULL': 0.0, 'EXPANSION': 0.0, 'RISK-ON': 0.0, 'NEUTRAL': 0.0, 'DEFENSIVE': 0.0}
        
        # Calculate daily profit (change in total_value)
        if i > 0 and len(daily_stats_list) > 0:
            prev_total_value = daily_stats_list[-1]['total_value']
            current_total_value = daily_stats['total_value']
            daily_profit = current_total_value - prev_total_value
            regime_profit[regime] = regime_profit.get(regime, 0.0) + daily_profit
        elif i == 0:
            # First day: no profit yet
            pass
        
        # Track regime from daily_stats (if available)
        if 'regime' in daily_stats:
            regime = daily_stats['regime']
            if regime in regime_days:
                regime_days[regime] += 1
        
        # Track regime profit for Regime Capture analysis
        current_total_value = daily_stats['total_value']
        daily_profit = current_total_value - prev_total_value
        regime_profit[regime] = regime_profit.get(regime, 0.0) + daily_profit
        prev_total_value = current_total_value  # Update for next iteration
        
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
    
    # Trades CSV - sort for deterministic order (by date, then symbol)
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        # Ensure all required fields exist
        required_trade_fields = ['date', 'symbol', 'side', 'dollars', 'price', 'slippage_bps', 
                                 'action', 'confidence', 'veto_applied', 'veto_model', 'veto_reason',
                                 'tape_gate_allowed', 'liquidity_tier', 'liquidity_reason']
        for field in required_trade_fields:
            if field not in trades_df.columns:
                trades_df[field] = ''
        
        # Validation: Check for action/side mismatches (HOLD should never produce SELL with default policy)
        if args.sell_policy == 'veto_only':
            mismatches = trades_df[(trades_df['action'] == 'HOLD') & (trades_df['side'] == 'SELL')]
            if not mismatches.empty:
                print(f"WARNING: Found {len(mismatches)} trade(s) with action=HOLD but side=SELL:")
                for _, row in mismatches.iterrows():
                    print(f"  - {row['symbol']} on {row['date']}: action={row['action']}, side={row['side']}")
                print("This should not happen with sell_policy=veto_only. This may indicate a bug.")
        
        # Sort for deterministic output
        trades_df = trades_df.sort_values(['date', 'symbol'])
        trades_df.to_csv(reports_dir / 'backtest_trades.csv', index=False)
    
    # Daily stats CSV
    daily_df = pd.DataFrame(daily_stats_list)
    daily_df.to_csv(reports_dir / 'backtest_daily.csv', index=False)
    
    # Summary JSON
    if len(daily_stats_list) > 0:
        # Use calculated initial_value (cash + initial equity at first trading day)
        initial_value = initial_value_calc if initial_value_calc > 0 else daily_stats_list[0]['total_value']
        # Final value = final equity (last trading day prices) + final cash
        final_equity = daily_stats_list[-1]['equity']
        final_cash = daily_stats_list[-1]['cash']
        final_value = final_equity + final_cash
        days = len(daily_stats_list)
        years = days / 252.0
        
        cagr = ((final_value / initial_value) ** (1.0 / years) - 1) * 100 if years > 0 and initial_value > 0 else 0
        
        values = [d['total_value'] for d in daily_stats_list]
        returns = pd.Series(values).pct_change(fill_method=None).dropna()
        vol = returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0
        
        max_dd = max([d['drawdown'] for d in daily_stats_list]) if daily_stats_list else 0
        
        total_trade_value = sum([abs(t['dollars']) for t in all_trades])
        turnover = (total_trade_value / initial_value) if initial_value > 0 else 0
        
        # Calculate additional metrics
        # Average exposure (equity / total_value)
        exposure_pcts = [d['equity'] / d['total_value'] * 100 if d['total_value'] > 0 else 0 for d in daily_stats_list]
        exposure_pct_avg = sum(exposure_pcts) / len(exposure_pcts) if exposure_pcts else 0
        
        # Days in cash (equity == 0)
        num_days_in_cash = sum(1 for d in daily_stats_list if d['equity'] == 0)
        
        # Percentage of time in cash
        pct_time_in_cash = (num_days_in_cash / days * 100) if days > 0 else 0
        
        # First and last trade dates
        first_trade_date = all_trades[0]['date'] if all_trades else None
        last_trade_date = all_trades[-1]['date'] if all_trades else None
        
        # Number of positions at end (count symbols with Quantity > 0 in final portfolio)
        num_positions_end = sum(1 for idx, row in portfolio.iterrows() if row.get('Quantity', 0) > 0)
        
        # Get symbols info
        symbols_requested = sorted(symbols_to_process)
        symbols_loaded = sorted([s for s in symbols_to_process if s in hist_cache])
        symbols_skipped = sorted(skipped_symbols) if skipped_symbols else []
        
        # Calculate baselines
        # Buy-and-hold: hold initial portfolio positions, no trades, mark-to-market daily
        buy_and_hold_initial = buy_and_hold_initial_value
        
        # Calculate buy-and-hold final value: mark-to-market initial positions at last trading day
        if len(daily_stats_list) > 0 and trading_days:
            last_day = trading_days[-1]
            last_day_ts = pd.Timestamp(last_day)
            buy_and_hold_final_equity = 0.0
            for idx, row in buy_and_hold_portfolio.iterrows():
                symbol = row['Symbol']
                initial_qty = row.get('Quantity', 0)
                if initial_qty > 0 and symbol in hist_cache:
                    hist = hist_cache[symbol]
                    if last_day_ts in hist.index:
                        final_price = hist.loc[last_day_ts, 'Close']
                    else:
                        hist_before = hist[hist.index <= last_day_ts]
                        if not hist_before.empty:
                            final_price = hist_before['Close'].iloc[-1]
                        else:
                            final_price = row.get('Price', 0)
                    if final_price > 0:
                        buy_and_hold_final_equity += initial_qty * final_price
            # Buy-and-hold: final = final equity (mark-to-market) + initial cash (no trades)
            buy_and_hold_final_value = buy_and_hold_final_equity + buy_and_hold_cash
            buy_and_hold_return_pct = ((buy_and_hold_final_value / buy_and_hold_initial) - 1) * 100 if buy_and_hold_initial > 0 else 0
        else:
            buy_and_hold_final_value = buy_and_hold_initial
            buy_and_hold_return_pct = 0.0
        
        # Cash return (0% baseline)
        cash_return_pct = 0.0
        
        # Optional benchmark (SPY or GLD) if available
        benchmark_return_pct = None
        benchmark_symbol = None
        if not args.offline:
            # Try SPY first, then GLD
            for bench_sym in ['SPY', 'GLD']:
                bench_cache_file = data_dir / f"{bench_sym}_{start_date}_{end_date}.csv"
                if bench_cache_file.exists():
                    try:
                        bench_hist = pd.read_csv(bench_cache_file, index_col=0, parse_dates=True)
                        bench_hist = _normalize_price_df(bench_hist)
                        if not bench_hist.empty and 'Close' in bench_hist.columns:
                            bench_start_price = bench_hist['Close'].iloc[0]
                            bench_end_price = bench_hist['Close'].iloc[-1]
                            if bench_start_price > 0:
                                benchmark_return_pct = ((bench_end_price / bench_start_price) - 1) * 100
                                benchmark_symbol = bench_sym
                                break
                    except:
                        pass
        
        # Calculate Regime Heatmap (percentage of days in each regime)
        total_regime_days = sum(regime_days.values())
        regime_heatmap = {}
        for regime, count in regime_days.items():
            if total_regime_days > 0:
                pct = (count / total_regime_days) * 100
                regime_heatmap[regime] = round(pct, 2)
            else:
                regime_heatmap[regime] = 0.0
        
        # Calculate Opportunity Cost (ROI lost by sitting in cash)
        # Compare actual performance vs buy-and-hold
        opportunity_cost_pct = 0.0
        opportunity_cost_dollars = 0.0
        if buy_and_hold_return_pct > 0 and final_value < buy_and_hold_final_value:
            # Lost ROI vs buy-and-hold
            opportunity_cost_pct = buy_and_hold_return_pct - ((final_value / initial_value - 1) * 100)
            opportunity_cost_dollars = buy_and_hold_final_value - final_value
        
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
            'trading_days': days,
            'symbols_requested': symbols_requested,
            'symbols_loaded': symbols_loaded,
            'symbols_skipped': symbols_skipped,
            'cache_dir': str(data_dir),
            'regime_heatmap': regime_heatmap,  # Regime Heatmap: percentage of days in each regime
            'opportunity_cost_pct': round(opportunity_cost_pct, 2),  # ROI lost vs buy-and-hold
            'opportunity_cost_dollars': round(opportunity_cost_dollars, 2),  # Dollar cost of cash drag
            'offline_used': args.offline,
            'run_timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'buy_and_hold_return_pct': round(buy_and_hold_return_pct, 2),
            'buy_and_hold_final_value': round(buy_and_hold_final_value, 2),
            'cash_return_pct': round(cash_return_pct, 2),
            'cash_only_final_value': round(cash_only_final_value, 2),
            'exposure_pct_avg': round(exposure_pct_avg, 2),
            'num_days_in_cash': num_days_in_cash,
            'pct_time_in_cash': round(pct_time_in_cash, 2),
            'num_positions_start': num_positions_start,
            'num_positions_end': num_positions_end,
            'first_trade_date': first_trade_date,
            'last_trade_date': last_trade_date
        }
        
        if benchmark_return_pct is not None:
            summary['benchmark_return_pct'] = round(benchmark_return_pct, 2)
            summary['benchmark_symbol'] = benchmark_symbol
        
        # INSTITUTIONAL SUMMARY: Calculate additional KPIs (moved here to access benchmark_return_pct)
        # 1. Safe CAGR (5th percentile) - already calculated in Monte Carlo, use actual CAGR if not MC
        safe_cagr = cagr  # Default to actual CAGR if not Monte Carlo
        
        # 2. Calculate average market impact across all trades
        market_impacts = [t.get('market_impact_pct', 0) for t in all_trades if t.get('market_impact_pct', 0) > 0]
        avg_market_impact = sum(market_impacts) / len(market_impacts) if market_impacts else 0.0
        
        # 3. Execution Efficiency: Percentage of intended trades that passed all gates
        total_intended_trades = sum(d.get('intended_trades', 0) for d in daily_stats_list)
        total_executed_trades = sum(d.get('executed_trades', 0) for d in daily_stats_list)
        execution_efficiency = (total_executed_trades / total_intended_trades * 100) if total_intended_trades > 0 else 100.0
        
        # 4. Capacity Limit: Calculate using formula: Capacity = Initial_Cash * (1.0 + Benchmark_Return) / (1.0 + Avg_Market_Impact)
        # Get benchmark return (use variable directly, not from summary)
        benchmark_return_pct_val = benchmark_return_pct if benchmark_return_pct is not None else 0.0
        benchmark_return = benchmark_return_pct_val / 100.0  # Convert to decimal
        
        # Calculate capacity using formula: Capacity = Initial_Cash * (1.0 + Benchmark_Return) / (1.0 + Avg_Market_Impact)
        # Convert avg_market_impact from percentage to decimal
        avg_market_impact_decimal = avg_market_impact / 100.0 if avg_market_impact > 0 else 0.0
        
        if avg_market_impact_decimal > 0:
            # Use formula: Capacity = Initial_Cash * (1.0 + Benchmark_Return) / (1.0 + Avg_Market_Impact)
            capacity_limit_estimate = initial_value * (1.0 + benchmark_return) / (1.0 + avg_market_impact_decimal)
        else:
            # Fallback if no market impact data
            capacity_limit_estimate = initial_value * (1.0 + benchmark_return) if benchmark_return > 0 else initial_value
        
        # Hard Rule: If CAGR is negative, show "$0 (Illiquid List)"
        if cagr < 0:
            capacity_limit_estimate = 0.0
        
        # Warn if initial_cash significantly exceeds capacity limit (only if CAGR is positive)
        if cagr >= 0 and initial_value > capacity_limit_estimate * 1.2:  # 20% over capacity
            excess_pct = ((initial_value - capacity_limit_estimate) / capacity_limit_estimate) * 100
            print(f"\n⚠️  WARNING: Initial capital (${initial_value:,.0f}) exceeds estimated capacity limit (${capacity_limit_estimate:,.0f}) by {excess_pct:.1f}%")
            print(f"   This may result in excessive market impact and reduced returns.")
            print(f"   Recommended maximum AUM: ${capacity_limit_estimate:,.0f}")
        
        # 5. Regime Capture: Percentage of profit generated in BULL vs NEUTRAL regimes
        total_profit = final_value - initial_value
        regime_capture = {}
        if total_profit != 0:
            for regime_name, profit in regime_profit.items():
                pct = (profit / total_profit * 100) if total_profit != 0 else 0.0
                regime_capture[regime_name] = round(pct, 2)
        else:
            # Fallback: use regime_heatmap as proxy
            for regime_name, pct_days in regime_heatmap.items():
                regime_capture[regime_name] = pct_days  # Use days as proxy for profit
        
        # Add institutional summary to summary dict
        summary['institutional_summary'] = {
            'safe_cagr_pct': round(safe_cagr, 2),
            'capacity_limit_estimate': round(capacity_limit_estimate, 0),
            'capacity_limit_estimate_m': round(capacity_limit_estimate / 1_000_000, 2),  # In millions
            'avg_market_impact_pct': round(avg_market_impact, 2),
            'execution_efficiency_pct': round(execution_efficiency, 2),
            'regime_capture': regime_capture,
            'bull_regime_profit_pct': regime_capture.get('BULL', 0.0) + regime_capture.get('EXPANSION', 0.0) + regime_capture.get('RISK-ON', 0.0),
            'neutral_regime_profit_pct': regime_capture.get('NEUTRAL', 0.0),
            'defensive_regime_profit_pct': regime_capture.get('DEFENSIVE', 0.0)
        }
        
        with open(reports_dir / 'backtest_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create run manifest with provenance
        import platform
        import subprocess
        # sys is already imported at the top of the file - no need to re-import
        
        # Get git commit hash if available
        git_hash = None
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                git_hash = result.stdout.strip()
        except:
            pass
        
        # Calculate SHA256 hashes of reports
        report_hashes = {}
        for report_file in ['backtest_summary.json', 'backtest_trades.csv', 'backtest_daily.csv']:
            report_path = reports_dir / report_file
            if report_path.exists():
                report_hashes[report_file] = _compute_file_hash(report_path)
        
        run_manifest = {
            'git_commit_hash': git_hash,
            'python_version': sys.version,
            'platform': platform.platform(),
            'run_timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'cache_dir': str(data_dir),
            'symbols_loaded': symbols_loaded,
            'symbols_skipped': symbols_skipped,
            'report_hashes': report_hashes,
            'sell_policy': args.sell_policy,
            'offline_mode': args.offline
        }
        
        with open(reports_dir / 'backtest_run_manifest.json', 'w') as f:
            json.dump(run_manifest, f, indent=2)
        
        # Calculate Realism Score: percentage of portfolio allocated to L0/L1 vs L3
        # Higher L0/L1 allocation = higher risk = higher realism score
        total_trades = len(all_trades)
        if total_trades > 0:
            l0_l1_trades = sum(1 for t in all_trades if t.get('liquidity_tier') in ['L0', 'L1'])
            l3_trades = sum(1 for t in all_trades if t.get('liquidity_tier') == 'L3')
            realism_score = (l0_l1_trades / total_trades * 100) if total_trades > 0 else 0.0
        else:
            realism_score = 0.0
        
        run_manifest['realism_score'] = realism_score
        run_manifest['l0_l1_trades'] = l0_l1_trades if total_trades > 0 else 0
        run_manifest['l3_trades'] = l3_trades if total_trades > 0 else 0
        run_manifest['total_trades'] = total_trades
        
        # Note: Institutional summary is already calculated and added to summary dict above (before JSON write)
        # No need to recalculate here - use the existing summary['institutional_summary']
        
        print(f"\nBacktest complete!")
        print(f"  Initial: ${initial_value:,.0f}")
        print(f"  Final: ${final_value:,.0f}")
        print(f"  CAGR: {cagr:.2f}%")
        print(f"  Max Drawdown: {max_dd:.2f}%")
        print(f"  Realism Score: {realism_score:.1f}% (L0/L1 allocation)")
        
        # Print Institutional Summary
        if 'institutional_summary' in summary:
            inst_summary = summary['institutional_summary']
            print(f"\n=== INSTITUTIONAL SUMMARY ===")
            print(f"  Safe CAGR (5th percentile): {inst_summary.get('safe_cagr_pct', cagr):.2f}%")
            # Hard Rule: If CAGR is negative, show "$0 (Illiquid List)"
            if cagr < 0:
                print(f"  Capacity Limit Estimate: $0 (Illiquid List)")
            else:
                print(f"  Capacity Limit Estimate: ${inst_summary.get('capacity_limit_estimate_m', 0):.2f}M")
            print(f"  Avg Market Impact: {inst_summary.get('avg_market_impact_pct', 0):.2f}%")
            print(f"  Execution Efficiency: {inst_summary.get('execution_efficiency_pct', 100):.1f}%")
            print(f"  Regime Capture:")
            print(f"    BULL/EXPANSION/RISK-ON: {inst_summary.get('bull_regime_profit_pct', 0):.1f}%")
            print(f"    NEUTRAL: {inst_summary.get('neutral_regime_profit_pct', 0):.1f}%")
            print(f"    DEFENSIVE: {inst_summary.get('defensive_regime_profit_pct', 0):.1f}%")
        
        print(f"  Reports saved to ./reports/")
        
        # Return results for Monte Carlo analysis
        result = {
            'cagr': cagr,
            'final_value': final_value,
            'initial_value': initial_value,
            'max_drawdown': max_dd,
            'realism_score': realism_score
        }
        
        # Include institutional summary in result for Monte Carlo aggregation
        if 'institutional_summary' in summary:
            result['institutional_summary'] = summary['institutional_summary']
        
        # If caching decisions, return decision cache (populated during simulation)
        if cache_decisions:
            # Return the populated decision_cache from the simulation loop
            # This cache was populated in simulate_day when save_decisions=True
            result['decision_cache'] = decision_cache if decision_cache else {}
            if not decision_cache:
                print("⚠️  Warning: Decision cache is empty at end of simulation. Check save_decisions logic.")
        else:
            # Even if not caching, return empty cache structure for consistency
            result['decision_cache'] = {}
        
        return result
    else:
        print("Warning: No daily stats generated")
        return {
            'cagr': 0.0,
            'final_value': 0.0,
            'initial_value': 0.0,
            'max_drawdown': 0.0,
            'realism_score': 0.0
        }

if __name__ == '__main__':
    args = parse_args()
    run_backtest(args)
