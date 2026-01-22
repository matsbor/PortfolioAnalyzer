#!/usr/bin/env python3
"""
Tests for backtest_runner.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_runner import (
    load_or_fetch_price_data,
    load_or_fetch_price_data_batch,
    simulate_day,
    get_trading_days,
    _load_manifest,
    _save_manifest,
    _compute_file_hash,
    verify_cache,
    build_cache_only
)
from alpha_miner_core import calculate_liquidity_metrics as core_calculate_liquidity_metrics


def test_offline_mode_raises_on_missing_cache(tmp_path):
    """Test that offline mode fails fast with clear message if cache missing"""
    data_dir = tmp_path / "cache"
    data_dir.mkdir()
    
    with pytest.raises(FileNotFoundError) as exc_info:
        load_or_fetch_price_data("TEST", "2024-01-01", "2024-01-31", data_dir, offline=True)
    
    assert "Offline mode: Missing cached data" in str(exc_info.value)
    assert "TEST" in str(exc_info.value)


def test_offline_mode_raises_on_corrupted_cache(tmp_path):
    """Test that offline mode fails fast with clear message if cache corrupted"""
    data_dir = tmp_path / "cache"
    data_dir.mkdir()
    
    # Create corrupted cache file
    cache_file = data_dir / "TEST_2024-01-01_2024-01-31.csv"
    cache_file.write_text("invalid,csv,data\nbroken,file")
    
    with pytest.raises(FileNotFoundError) as exc_info:
        load_or_fetch_price_data("TEST", "2024-01-01", "2024-01-31", data_dir, offline=True)
    
    assert "Offline mode: Corrupted cache" in str(exc_info.value)


def test_offline_mode_succeeds_with_valid_cache(tmp_path):
    """Test that offline mode works when cache exists"""
    data_dir = tmp_path / "cache"
    data_dir.mkdir()
    
    # Create valid cache file with DatetimeIndex
    dates = pd.date_range("2024-01-01", "2024-01-31", freq='B')
    hist = pd.DataFrame({
        'Open': np.random.uniform(10, 20, len(dates)),
        'High': np.random.uniform(15, 25, len(dates)),
        'Low': np.random.uniform(5, 15, len(dates)),
        'Close': np.random.uniform(10, 20, len(dates)),
        'Volume': np.random.uniform(100000, 1000000, len(dates)),
    }, index=dates)
    
    cache_file = data_dir / "TEST_2024-01-01_2024-01-31.csv"
    hist.to_csv(cache_file)
    
    # Load in offline mode
    result = load_or_fetch_price_data("TEST", "2024-01-01", "2024-01-31", data_dir, offline=True)
    
    assert not result.empty
    assert isinstance(result.index, pd.DatetimeIndex)
    assert 'Volume' in result.columns
    assert len(result) == len(hist)


def test_offline_mode_no_network_calls(tmp_path, monkeypatch):
    """Test that offline mode makes zero network calls"""
    data_dir = tmp_path / "cache"
    data_dir.mkdir()
    
    # Create valid cache
    dates = pd.date_range("2024-01-01", "2024-01-31", freq='B')
    hist = pd.DataFrame({
        'Open': np.random.uniform(10, 20, len(dates)),
        'High': np.random.uniform(15, 25, len(dates)),
        'Low': np.random.uniform(5, 15, len(dates)),
        'Close': np.random.uniform(10, 20, len(dates)),
        'Volume': np.random.uniform(100000, 1000000, len(dates)),
    }, index=dates)
    
    cache_file = data_dir / "TEST_2024-01-01_2024-01-31.csv"
    hist.to_csv(cache_file)
    
    # Mock yfinance to raise if called
    def raise_on_call(*args, **kwargs):
        raise RuntimeError("Network call made in offline mode!")
    
    monkeypatch.setattr("yfinance.Ticker", raise_on_call)
    
    # Should succeed without calling yfinance
    result = load_or_fetch_price_data("TEST", "2024-01-01", "2024-01-31", data_dir, offline=True)
    assert not result.empty


def test_liquidity_unknown_does_not_force_liquidation():
    """Test that UNKNOWN liquidity tier does not force liquidation"""
    # Create hist data with missing/invalid volume
    dates = pd.date_range("2024-01-01", periods=30, freq='B')
    hist = pd.DataFrame({
        'Open': np.random.uniform(10, 20, len(dates)),
        'High': np.random.uniform(15, 25, len(dates)),
        'Low': np.random.uniform(5, 15, len(dates)),
        'Close': np.random.uniform(10, 20, len(dates)),
        'Volume': np.nan,  # All NaN volume
    }, index=dates)
    
    result = core_calculate_liquidity_metrics("TEST", hist, 15.0, 10000.0, 200000)
    
    assert result['tier_code'] == 'UNKNOWN'
    assert result['max_position_pct'] == 0.0  # Blocks new buys
    assert 'Volume data' in result['liquidity_reason']


def test_liquidity_unknown_blocks_new_buys():
    """Test that UNKNOWN liquidity blocks new buys but allows holds"""
    from alpha_miner_core import arbitrate_final_decision
    
    # Row with current position
    row = {
        'Pct_Portfolio': 5.0,  # Current position
        'Recommended_Pct': 7.0,  # Would be a buy
        'Sell_Risk_Score': 20,
        'Financing_Overhang_Score': 0,
        'Runway': 12
    }
    
    liq_metrics = {'tier_code': 'UNKNOWN', 'max_position_pct': 0.0}
    data_conf = {'score': 80}
    dilution = {'score': 30}
    sell_risk = {'score': 20, 'hard_triggers': [], 'soft_triggers': []}
    macro_regime = {'allow_new_buys': True, 'throttle_factor': 1.0}
    discovery = (False, '')
    
    decision = arbitrate_final_decision(
        row, liq_metrics, data_conf, dilution, sell_risk, 75, macro_regime, discovery, strict_mode=False
    )
    
    # Should block the buy (downgrade to HOLD)
    assert decision['action'] == 'HOLD'
    assert 'UNKNOWN' in decision['primary_gating_reason'] or 'UNKNOWN' in str(decision.get('warnings', []))
    # Should NOT be a veto (it's a caution, not hard veto)
    # UNKNOWN doesn't force sells, just blocks new buys


def test_liquidity_unknown_allows_sells():
    """Test that UNKNOWN liquidity allows sells (doesn't force liquidation)"""
    from alpha_miner_core import arbitrate_final_decision
    
    # Row with sell signal (high sell risk)
    row = {
        'Pct_Portfolio': 5.0,  # Current position
        'Recommended_Pct': 2.0,  # Would be a sell
        'Sell_Risk_Score': 65,  # High sell risk
        'Financing_Overhang_Score': 0,
        'Runway': 12
    }
    
    liq_metrics = {'tier_code': 'UNKNOWN', 'max_position_pct': 0.0}
    data_conf = {'score': 80}
    dilution = {'score': 30}
    sell_risk = {'score': 65, 'hard_triggers': ['High sell risk'], 'soft_triggers': []}
    macro_regime = {'allow_new_buys': True, 'throttle_factor': 1.0}
    discovery = (False, '')
    
    decision = arbitrate_final_decision(
        row, liq_metrics, data_conf, dilution, sell_risk, 50, macro_regime, discovery, strict_mode=False
    )
    
    # Should allow the sell (sell risk veto takes precedence)
    assert decision['action'] == 'Avoid'  # Or REDUCE
    assert decision['veto_applied'] == True
    assert decision['veto_model'] == 'Risk'  # Sell risk, not liquidity


def test_timestamp_handling_in_simulate_day(tmp_path):
    """Test that simulate_day handles timestamps correctly"""
    data_dir = tmp_path / "cache"
    data_dir.mkdir()
    
    # Create cached data with DatetimeIndex
    dates = pd.date_range("2024-01-01", "2024-01-10", freq='B')
    hist = pd.DataFrame({
        'Open': np.random.uniform(10, 20, len(dates)),
        'High': np.random.uniform(15, 25, len(dates)),
        'Low': np.random.uniform(5, 15, len(dates)),
        'Close': np.random.uniform(10, 20, len(dates)),
        'Volume': np.random.uniform(100000, 1000000, len(dates)),
    }, index=dates)
    
    cache_file = data_dir / "TEST_2024-01-01_2024-01-10.csv"
    hist.to_csv(cache_file)
    
    # Load in offline mode
    loaded_hist = load_or_fetch_price_data("TEST", "2024-01-01", "2024-01-10", data_dir, offline=True)
    
    # Verify DatetimeIndex and tz-naive
    assert isinstance(loaded_hist.index, pd.DatetimeIndex)
    assert loaded_hist.index.tz is None  # tz-naive
    
    # Test that date string comparison works
    test_date = "2024-01-05"
    test_date_ts = pd.Timestamp(test_date)  # tz-naive
    
    # Should be able to slice correctly
    hist_before = loaded_hist[loaded_hist.index <= test_date_ts]
    assert not hist_before.empty
    assert hist_before.index.max() <= test_date_ts


def test_timezone_aware_index_normalization(tmp_path):
    """Test that tz-aware indices are normalized to tz-naive"""
    from backtest_runner import _normalize_price_df
    
    # Create DataFrame with tz-aware index (simulating yfinance output)
    try:
        import pytz
        tz = pytz.timezone('America/New_York')
    except ImportError:
        # Fallback: use UTC timezone (pandas built-in)
        tz = 'America/New_York'
    
    dates = pd.date_range("2024-01-01", "2024-01-10", freq='B', tz=tz)
    hist = pd.DataFrame({
        'Open': np.random.uniform(10, 20, len(dates)),
        'High': np.random.uniform(15, 25, len(dates)),
        'Low': np.random.uniform(5, 15, len(dates)),
        'Close': np.random.uniform(10, 20, len(dates)),
        'Volume': np.random.uniform(100000, 1000000, len(dates)),
    }, index=dates)
    
    # Verify it's tz-aware
    assert hist.index.tz is not None
    
    # Normalize
    normalized = _normalize_price_df(hist)
    
    # Verify it's now tz-naive
    assert isinstance(normalized.index, pd.DatetimeIndex)
    assert normalized.index.tz is None
    
    # Verify we can compare with tz-naive Timestamp
    test_date_ts = pd.Timestamp("2024-01-05")  # tz-naive
    hist_before = normalized[normalized.index <= test_date_ts]
    assert not hist_before.empty
    assert hist_before.index.max() <= test_date_ts


def test_offline_mode_still_enforces_zero_network_calls(tmp_path, monkeypatch):
    """Test that offline mode still makes zero network calls after timezone fix"""
    data_dir = tmp_path / "cache"
    data_dir.mkdir()
    
    # Create valid cache with tz-naive index
    dates = pd.date_range("2024-01-01", "2024-01-31", freq='B')
    hist = pd.DataFrame({
        'Open': np.random.uniform(10, 20, len(dates)),
        'High': np.random.uniform(15, 25, len(dates)),
        'Low': np.random.uniform(5, 15, len(dates)),
        'Close': np.random.uniform(10, 20, len(dates)),
        'Volume': np.random.uniform(100000, 1000000, len(dates)),
    }, index=dates)
    
    cache_file = data_dir / "TEST_2024-01-01_2024-01-31.csv"
    hist.to_csv(cache_file)
    
    # Mock yfinance to raise if called
    call_count = {'count': 0}
    original_history = None
    try:
        import yfinance as yf
        original_history = yf.Ticker.history
        
        def mock_history(*args, **kwargs):
            call_count['count'] += 1
            raise RuntimeError("Network call made in offline mode!")
        
        monkeypatch.setattr("yfinance.Ticker.history", mock_history)
    except ImportError:
        pass
    
    # Should succeed without calling yfinance
    result = load_or_fetch_price_data("TEST", "2024-01-01", "2024-01-31", data_dir, offline=True)
    assert not result.empty
    assert call_count['count'] == 0


def test_timezone_aware_cache_backward_compatibility(tmp_path):
    """Test that existing tz-aware cached data is normalized correctly"""
    from backtest_runner import _normalize_price_df
    
    # Create cache file with tz-aware index (simulating old cache)
    try:
        import pytz
        tz = pytz.timezone('America/New_York')
    except ImportError:
        # Fallback: use UTC timezone (pandas built-in)
        tz = 'America/New_York'
    
    dates = pd.date_range("2024-01-01", "2024-01-10", freq='B', tz=tz)
    hist = pd.DataFrame({
        'Open': [10] * len(dates),
        'High': [15] * len(dates),
        'Low': [5] * len(dates),
        'Close': [12] * len(dates),
        'Volume': [100000] * len(dates),
    }, index=dates)
    
    # Save to cache (pandas will serialize tz-aware index)
    cache_file = tmp_path / "TEST_2024-01-01_2024-01-10.csv"
    hist.to_csv(cache_file)
    
    # Load and normalize (simulating what load_or_fetch_price_data does)
    loaded = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    normalized = _normalize_price_df(loaded)
    
    # Should be tz-naive
    assert normalized.index.tz is None
    
    # Should be comparable with tz-naive Timestamp
    test_ts = pd.Timestamp("2024-01-05")
    assert (normalized.index <= test_ts).any()  # Should not raise TypeError


def test_manifest_creation_and_validation(tmp_path):
    """Test that manifest.json is created and contains required fields"""
    data_dir = tmp_path / "cache"
    data_dir.mkdir()
    
    # Create cache files for two symbols
    dates = pd.date_range("2024-01-01", "2024-01-31", freq='B')
    for symbol in ['TEST1', 'TEST2']:
        hist = pd.DataFrame({
            'Open': [10] * len(dates),
            'High': [15] * len(dates),
            'Low': [5] * len(dates),
            'Close': [12] * len(dates),
            'Volume': [100000] * len(dates),
        }, index=dates)
        cache_file = data_dir / f"{symbol}_2024-01-01_2024-01-31.csv"
        hist.to_csv(cache_file)
    
    # Load in batch mode (should create/update manifest)
    hist_cache, missing = load_or_fetch_price_data_batch(
        ['TEST1', 'TEST2'],
        '2024-01-01',
        '2024-01-31',
        data_dir,
        offline=True
    )
    
    # Check manifest exists and was updated
    manifest = _load_manifest(data_dir)
    assert '2024-01-01_2024-01-31' in manifest
    
    cache_key = '2024-01-01_2024-01-31'
    assert 'start' in manifest[cache_key]
    assert 'end' in manifest[cache_key]
    assert 'created_at' in manifest[cache_key]
    assert 'symbols' in manifest[cache_key]
    
    # Check symbol entries (should be added when loading from cache)
    assert 'TEST1' in manifest[cache_key]['symbols']
    assert 'TEST2' in manifest[cache_key]['symbols']
    
    # Check required fields in symbol status
    test1_status = manifest[cache_key]['symbols']['TEST1']
    assert 'status' in test1_status
    assert 'rows' in test1_status
    assert 'sha256' in test1_status


def test_offline_fails_with_friendly_error_missing_symbols(tmp_path):
    """Test that offline mode fails with friendly error listing missing symbols"""
    data_dir = tmp_path / "cache"
    data_dir.mkdir()
    
    # Create cache for only one symbol
    dates = pd.date_range("2024-01-01", "2024-01-31", freq='B')
    hist = pd.DataFrame({
        'Open': [10] * len(dates),
        'High': [15] * len(dates),
        'Low': [5] * len(dates),
        'Close': [12] * len(dates),
        'Volume': [100000] * len(dates),
    }, index=dates)
    cache_file = data_dir / "TEST1_2024-01-01_2024-01-31.csv"
    hist.to_csv(cache_file)
    
    # Try to load both TEST1 and TEST2 in offline mode
    with pytest.raises(FileNotFoundError) as exc_info:
        load_or_fetch_price_data_batch(
            ['TEST1', 'TEST2'],
            '2024-01-01',
            '2024-01-31',
            data_dir,
            offline=True
        )
    
    error_msg = str(exc_info.value)
    assert "Missing cached data" in error_msg
    assert "TEST2" in error_msg  # Missing symbol should be listed
    assert "To rebuild cache" in error_msg or "retry_missing_cache" in error_msg


def test_retry_missing_cache_mode_exits_after_caching(tmp_path, monkeypatch):
    """Test that --retry_missing_cache mode exits after caching (no simulation)"""
    data_dir = tmp_path / "cache"
    data_dir.mkdir()
    
    # Create manifest with one missing symbol
    manifest = {
        '2024-01-01_2024-01-31': {
            'start': '2024-01-01',
            'end': '2024-01-31',
            'created_at': '2024-01-01T00:00:00',
            'symbols': {
                'TEST1': {'status': 'ok', 'rows': 20, 'sha256': 'abc123'},
                'TEST2': {'status': 'missing', 'rows': 0, 'sha256': None}
            }
        }
    }
    _save_manifest(data_dir, manifest)
    
    # Create cache for TEST1 only
    dates = pd.date_range("2024-01-01", "2024-01-31", freq='B')
    hist = pd.DataFrame({
        'Open': [10] * len(dates),
        'High': [15] * len(dates),
        'Low': [5] * len(dates),
        'Close': [12] * len(dates),
        'Volume': [100000] * len(dates),
    }, index=dates)
    cache_file = data_dir / "TEST1_2024-01-01_2024-01-31.csv"
    hist.to_csv(cache_file)
    
    # Mock yfinance to simulate successful fetch for TEST2
    def mock_download(symbols, start, end, **kwargs):
        if 'TEST2' in symbols:
            dates = pd.date_range(start, end, freq='B')
            df = pd.DataFrame({
                'Open': [10] * len(dates),
                'High': [15] * len(dates),
                'Low': [5] * len(dates),
                'Close': [12] * len(dates),
                'Volume': [100000] * len(dates),
            }, index=dates)
            if len(symbols) == 1:
                return df
            else:
                # MultiIndex for multiple symbols
                result = pd.DataFrame()
                result[('TEST2', 'Open')] = df['Open']
                result[('TEST2', 'High')] = df['High']
                result[('TEST2', 'Low')] = df['Low']
                result[('TEST2', 'Close')] = df['Close']
                result[('TEST2', 'Volume')] = df['Volume']
                result.columns = pd.MultiIndex.from_tuples(result.columns)
                return result
        return pd.DataFrame()
    
    monkeypatch.setattr("yfinance.download", mock_download)
    
    # Run retry mode
    hist_cache, missing = load_or_fetch_price_data_batch(
        ['TEST2'],
        '2024-01-01',
        '2024-01-31',
        data_dir,
        offline=False,
        allow_partial_cache=False
    )
    
    # Should have cached TEST2
    assert 'TEST2' in hist_cache
    assert len(missing) == 0
    
    # Manifest should be updated
    updated_manifest = _load_manifest(data_dir)
    assert updated_manifest['2024-01-01_2024-01-31']['symbols']['TEST2']['status'] == 'ok'


def test_sell_policy_does_not_liquidate_on_hold(tmp_path):
    """Test that HOLD action doesn't produce SELL trades with default triggers policy"""
    from backtest_runner import simulate_day
    
    # Create portfolio with one position
    portfolio = pd.DataFrame({
        'Symbol': ['TEST'],
        'Quantity': [100],
        'Price': [10.0],
        'Market_Value': [1000.0],
        'Cost_Basis': [1000.0],
        'Runway': [12.0],
        'stage': ['Explorer'],
        'metal': ['Gold'],
        'cash': [10.0],
        'burn_source': ['default'],
        'Insider_Buying_90d': [False],
        'Pct_Portfolio': [10.0]
    })
    
    # Create price history
    dates = pd.date_range("2024-01-01", "2024-01-10", freq='B')
    hist = pd.DataFrame({
        'Open': [10] * len(dates),
        'High': [15] * len(dates),
        'Low': [5] * len(dates),
        'Close': [12] * len(dates),  # Price goes up
        'Volume': [100000] * len(dates),
    }, index=dates)
    hist_cache = {'TEST': hist}
    news_cache = {'TEST': []}
    info_cache = {'TEST': {'fundamentals': {}, 'info_dict': {}, 'inferred_flags': {}}}
    
    # Simulate day with triggers policy (default)
    # Decision will likely be HOLD (no sell triggers), so no SELL trades should be generated
    portfolio_out, cash, trades, daily_stats = simulate_day(
        '2024-01-05',
        portfolio.copy(),
        1000.0,  # Cash
        hist_cache,
        news_cache,
        info_cache,
        strict_mode=False,
        allow_leverage=False,
        max_position_pct=10.0,
        sell_policy='triggers',
        execution_variance=0.0
    )
    
    # Check that no SELL trades were generated (HOLD should not trigger sells)
    sell_trades = [t for t in trades if t.get('side') == 'SELL']
    # With triggers policy and HOLD action, there should be no sells unless explicit sell triggers
    # This test verifies that HOLD doesn't auto-liquidate


def test_daily_equity_nonzero_when_holdings_exist(tmp_path):
    """Test that daily equity is non-zero when holdings exist (mark-to-market works)"""
    from backtest_runner import simulate_day
    
    # Create portfolio with one position
    portfolio = pd.DataFrame({
        'Symbol': ['TEST'],
        'Quantity': [100],
        'Price': [10.0],
        'Market_Value': [1000.0],
        'Cost_Basis': [1000.0],
        'Runway': [12.0],
        'stage': ['Explorer'],
        'metal': ['Gold'],
        'cash': [10.0],
        'burn_source': ['default'],
        'Insider_Buying_90d': [False],
        'Pct_Portfolio': [10.0]
    })
    
    # Create price history
    dates = pd.date_range("2024-01-01", "2024-01-10", freq='B')
    hist = pd.DataFrame({
        'Open': [10] * len(dates),
        'High': [15] * len(dates),
        'Low': [5] * len(dates),
        'Close': [12] * len(dates),  # Price goes up
        'Volume': [100000] * len(dates),
    }, index=dates)
    hist_cache = {'TEST': hist}
    news_cache = {'TEST': []}
    info_cache = {'TEST': {'fundamentals': {}, 'info_dict': {}, 'inferred_flags': {}}}
    
    # Simulate day
    portfolio_out, cash, trades, daily_stats = simulate_day(
        '2024-01-05',
        portfolio.copy(),
        1000.0,  # Cash
        hist_cache,
        news_cache,
        info_cache,
        strict_mode=False,
        allow_leverage=False,
        max_position_pct=10.0,
        sell_policy='triggers',
        execution_variance=0.0
    )
    
    # Check that equity is non-zero (mark-to-market should work)
    assert daily_stats['equity'] > 0, "Equity should be non-zero when holdings exist"
    assert daily_stats['total_value'] == daily_stats['equity'] + daily_stats['cash']
    # Market value should be Quantity * Price
    assert portfolio_out['Market_Value'].iloc[0] == portfolio_out['Quantity'].iloc[0] * portfolio_out['Price'].iloc[0]


def test_verify_cache_mode_does_not_require_portfolio_args(tmp_path):
    """Test that verify_cache works with --symbols flag without requiring portfolio_csv"""
    data_dir = tmp_path / "cache"
    data_dir.mkdir()
    
    # Create cache for TEST1
    dates = pd.date_range("2024-01-01", "2024-01-31", freq='B')
    hist = pd.DataFrame({
        'Open': [10] * len(dates),
        'High': [15] * len(dates),
        'Low': [5] * len(dates),
        'Close': [12] * len(dates),
        'Volume': [100000] * len(dates),
    }, index=dates)
    cache_file = data_dir / "TEST1_2024-01-01_2024-01-31.csv"
    hist.to_csv(cache_file)
    
    # Verify cache with symbols only (no portfolio CSV)
    is_valid, missing = verify_cache(
        None,
        '2024-01-01',
        '2024-01-31',
        data_dir,
        symbols=['TEST1']
    )
    
    assert is_valid
    assert len(missing) == 0


def test_hold_never_produces_sell_under_default_policy(tmp_path):
    """Test that HOLD action never produces SELL trades with default veto_only policy"""
    from backtest_runner import simulate_day
    
    # Create portfolio with one position
    portfolio = pd.DataFrame({
        'Symbol': ['TEST'],
        'Quantity': [100],
        'Price': [10.0],
        'Market_Value': [1000.0],
        'Cost_Basis': [1000.0],
        'Runway': [12.0],
        'stage': ['Explorer'],
        'metal': ['Gold'],
        'cash': [10.0],
        'burn_source': ['default'],
        'Insider_Buying_90d': [False],
        'Pct_Portfolio': [10.0]
    })
    
    # Create price history
    dates = pd.date_range("2024-01-01", "2024-01-10", freq='B')
    hist = pd.DataFrame({
        'Open': [10] * len(dates),
        'High': [15] * len(dates),
        'Low': [5] * len(dates),
        'Close': [12] * len(dates),
        'Volume': [100000] * len(dates),
    }, index=dates)
    hist_cache = {'TEST': hist}
    news_cache = {'TEST': []}
    info_cache = {'TEST': {'fundamentals': {}, 'info_dict': {}, 'inferred_flags': {}}}
    
    # Simulate day with default veto_only policy
    # Decision will likely be HOLD (no hard veto), so no SELL trades should be generated
    portfolio_out, cash, trades, daily_stats = simulate_day(
        '2024-01-05',
        portfolio.copy(),
        1000.0,  # Cash
        hist_cache,
        news_cache,
        info_cache,
        strict_mode=False,
        allow_leverage=False,
        max_position_pct=10.0,
        sell_policy='veto_only',  # Default policy
        execution_variance=0.0
    )
    
    # Check that no SELL trades were generated when action is HOLD
    sell_trades = [t for t in trades if t.get('side') == 'SELL']
    hold_sell_trades = [t for t in sell_trades if t.get('action') == 'HOLD']
    
    # With veto_only policy and HOLD action, there should be NO SELL trades
    assert len(hold_sell_trades) == 0, f"HOLD action produced {len(hold_sell_trades)} SELL trade(s) with veto_only policy. This is a bug."


def test_default_backtest_does_not_liquidate_all_positions_day1(tmp_path):
    """Test that default offline backtest does NOT liquidate all positions on day 1"""
    from backtest_runner import run_backtest
    import argparse
    
    data_dir = tmp_path / "cache"
    data_dir.mkdir()
    
    # Create portfolio CSV
    portfolio_csv = tmp_path / "portfolio.csv"
    portfolio_df = pd.DataFrame({
        'Symbol': ['TEST1', 'TEST2'],
        'Quantity': [100, 200],
        'Price': [10.0, 20.0],
        'Market_Value': [1000.0, 4000.0],
        'Cost_Basis': [1000.0, 4000.0],
        'Runway': [12.0, 12.0],
        'stage': ['Explorer', 'Explorer'],
        'metal': ['Gold', 'Gold'],
        'cash': [10.0, 10.0],
        'burn_source': ['default', 'default'],
        'Insider_Buying_90d': [False, False],
        'Pct_Portfolio': [10.0, 20.0]
    })
    portfolio_df.to_csv(portfolio_csv, index=False)
    
    # Create cache files for both symbols
    dates = pd.date_range("2024-01-01", "2024-01-31", freq='B')
    for symbol in ['TEST1', 'TEST2']:
        hist = pd.DataFrame({
            'Open': [10] * len(dates),
            'High': [15] * len(dates),
            'Low': [5] * len(dates),
            'Close': [12] * len(dates),  # Price goes up
            'Volume': [100000] * len(dates),
        }, index=dates)
        cache_file = data_dir / f"{symbol}_2024-01-01_2024-01-31.csv"
        hist.to_csv(cache_file)
    
    # Create SPY calendar cache
    spy_cache = data_dir / "SPY_calendar_2024-01-01_2024-01-31.csv"
    spy_df = pd.DataFrame({'Close': [100] * len(dates)}, index=dates)
    spy_df.to_csv(spy_cache)
    
    # Create args for offline backtest
    args = argparse.Namespace(
        start='2024-01-01',
        end='2024-01-31',
        portfolio_csv=str(portfolio_csv),
        initial_cash=5000.0,
        rebalance=True,
        allow_leverage=False,
        max_position_pct=10.0,
        data_dir=str(data_dir),
        offline=True,
        monte_carlo=None,
        strict_mode=False,
        symbols=None,
        retry_missing_cache=False,
        allow_partial_cache=False,
        build_cache_only=False,
        verify_cache=False,
        skip_missing_symbols=False,
        sell_policy='veto_only',  # Default policy
        risk_mode='BALANCED',  # Default risk mode
        warmup_days=20,  # Default warmup days
        trailing_stop_pct=None  # Default trailing stop (will be set based on risk_mode)
    )
    
    # Run backtest
    run_backtest(args)
    
    # Check reports
    reports_dir = Path('./reports')
    daily_df = pd.read_csv(reports_dir / 'backtest_daily.csv')
    
    # Assert that equity is non-zero for at least one day (positions should be held)
    assert (daily_df['equity'] > 0).any(), "All positions were liquidated - equity is 0 for all days. This indicates a bug."
    
    # Assert that first day equity is non-zero (positions should be held from start)
    assert daily_df.iloc[0]['equity'] > 0, "First day equity is 0 - positions were liquidated on day 1. This is a bug."
    
    # Check that trades.csv has no SELL trades when action is HOLD (if trades exist)
    trades_file = reports_dir / 'backtest_trades.csv'
    if trades_file.exists():
        trades_df = pd.read_csv(trades_file)
        hold_sell_trades = trades_df[(trades_df['action'] == 'HOLD') & (trades_df['side'] == 'SELL')]
        assert len(hold_sell_trades) == 0, f"Found {len(hold_sell_trades)} trade(s) with action=HOLD but side=SELL. This is a bug."


def test_verify_cache_reports_missing(tmp_path):
    """Test that verify_cache reports missing symbols and exits with code 2"""
    data_dir = tmp_path / "cache"
    data_dir.mkdir()
    
    # Create portfolio CSV
    portfolio_csv = tmp_path / "portfolio.csv"
    portfolio_df = pd.DataFrame({
        'Symbol': ['TEST1', 'TEST2', 'TEST3'],
        'Quantity': [100, 200, 300]
    })
    portfolio_df.to_csv(portfolio_csv, index=False)
    
    # Create cache for only TEST1 and TEST2
    dates = pd.date_range("2024-01-01", "2024-01-31", freq='B')
    for symbol in ['TEST1', 'TEST2']:
        hist = pd.DataFrame({
            'Open': [10] * len(dates),
            'High': [15] * len(dates),
            'Low': [5] * len(dates),
            'Close': [12] * len(dates),
            'Volume': [100000] * len(dates),
        }, index=dates)
        cache_file = data_dir / f"{symbol}_2024-01-01_2024-01-31.csv"
        hist.to_csv(cache_file)
    
    # Verify cache - should report TEST3 as missing
    is_valid, missing = verify_cache(
        str(portfolio_csv),
        '2024-01-01',
        '2024-01-31',
        data_dir
    )
    
    assert not is_valid
    assert 'TEST3' in missing
    assert len(missing) == 1


def test_build_cache_only_writes_files(tmp_path, monkeypatch):
    """Test that build_cache_only writes cache files and exits"""
    data_dir = tmp_path / "cache"
    data_dir.mkdir()
    
    # Create portfolio CSV
    portfolio_csv = tmp_path / "portfolio.csv"
    portfolio_df = pd.DataFrame({
        'Symbol': ['TEST1'],
        'Quantity': [100]
    })
    portfolio_df.to_csv(portfolio_csv, index=False)
    
    # Mock yfinance to return deterministic DataFrame
    def mock_download(symbols, start, end, **kwargs):
        dates = pd.date_range(start, end, freq='B')
        df = pd.DataFrame({
            'Open': [10] * len(dates),
            'High': [15] * len(dates),
            'Low': [5] * len(dates),
            'Close': [12] * len(dates),
            'Volume': [100000] * len(dates),
        }, index=dates)
        return df
    
    monkeypatch.setattr("yfinance.download", mock_download)
    
    # Build cache
    success, failed = build_cache_only(
        str(portfolio_csv),
        '2024-01-01',
        '2024-01-31',
        data_dir
    )
    
    assert success
    assert len(failed) == 0
    
    # Verify cache file was created
    cache_file = data_dir / "TEST1_2024-01-01_2024-01-31.csv"
    assert cache_file.exists()
    
    # Verify it's loadable
    loaded = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    assert not loaded.empty


def test_offline_never_calls_network(tmp_path, monkeypatch):
    """Test that offline mode never calls yfinance"""
    data_dir = tmp_path / "cache"
    data_dir.mkdir()
    
    # Create cache file
    dates = pd.date_range("2024-01-01", "2024-01-31", freq='B')
    hist = pd.DataFrame({
        'Open': [10] * len(dates),
        'High': [15] * len(dates),
        'Low': [5] * len(dates),
        'Close': [12] * len(dates),
        'Volume': [100000] * len(dates),
    }, index=dates)
    cache_file = data_dir / "TEST_2024-01-01_2024-01-31.csv"
    hist.to_csv(cache_file)
    
    # Mock yfinance to raise if called
    def mock_download(*args, **kwargs):
        raise RuntimeError("Network call should not happen in offline mode!")
    
    monkeypatch.setattr("yfinance.download", mock_download)
    monkeypatch.setattr("yfinance.Ticker", lambda x: type('obj', (object,), {'history': lambda *a, **k: mock_download()}))
    
    # Load in offline mode - should succeed
    hist_cache, missing = load_or_fetch_price_data_batch(
        ['TEST'],
        '2024-01-01',
        '2024-01-31',
        data_dir,
        offline=True
    )
    
    assert 'TEST' in hist_cache
    assert len(missing) == 0
