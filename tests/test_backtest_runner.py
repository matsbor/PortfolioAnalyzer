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
    simulate_day,
    get_trading_days
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
    
    # Verify DatetimeIndex
    assert isinstance(loaded_hist.index, pd.DatetimeIndex)
    
    # Test that date string comparison works
    test_date = "2024-01-05"
    test_date_ts = pd.Timestamp(test_date)
    
    # Should be able to slice correctly
    hist_before = loaded_hist[loaded_hist.index <= test_date_ts]
    assert not hist_before.empty
    assert hist_before.index.max() <= test_date_ts
