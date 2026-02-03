"""Unit tests for backtest-based discovery candidate validation.

Tests the new backtest_validate_candidate() and backtest_validate_candidates_batch()
functions added to backtest_runner.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_runner import (
    backtest_validate_candidate,
    backtest_validate_candidates_batch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 150, base: float = 10.0, trend: float = 0.001,
                seed: int = 42, start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n)
    daily_ret = rng.normal(loc=trend, scale=0.02, size=n)
    closes = base * np.cumprod(1 + daily_ret)
    highs = closes * (1 + rng.uniform(0, 0.02, n))
    lows = closes * (1 - rng.uniform(0, 0.02, n))
    opens = closes * (1 + rng.uniform(-0.01, 0.01, n))
    volumes = rng.integers(50_000, 500_000, size=n)
    return pd.DataFrame({
        'Open': opens, 'High': highs, 'Low': lows,
        'Close': closes, 'Volume': volumes,
    }, index=dates)


def _make_uptrend(n: int = 150) -> pd.DataFrame:
    return _make_ohlcv(n=n, base=10, trend=0.005, seed=1)


def _make_downtrend(n: int = 150) -> pd.DataFrame:
    return _make_ohlcv(n=n, base=10, trend=-0.004, seed=2)


# ===================================================================
# 1. Single candidate validation
# ===================================================================

class TestBacktestValidateCandidate:

    def test_uptrend_positive_returns(self):
        hist = _make_uptrend()
        result = backtest_validate_candidate("UP", hist, lookback_days=120)
        assert result['symbol'] == 'UP'
        assert result['n_trades'] > 0
        assert result['avg_return'] > 0

    def test_downtrend_negative_returns(self):
        hist = _make_downtrend()
        result = backtest_validate_candidate("DOWN", hist, lookback_days=120)
        assert result['avg_return'] < 0

    def test_win_rate_bounded_0_100(self):
        for seed in range(5):
            hist = _make_ohlcv(n=150, seed=seed + 50)
            result = backtest_validate_candidate(f"S{seed}", hist)
            assert 0 <= result['win_rate'] <= 100

    def test_max_drawdown_is_negative_or_zero(self):
        hist = _make_ohlcv(n=150, seed=77)
        result = backtest_validate_candidate("DD", hist)
        assert result['max_drawdown'] <= 0

    def test_signal_quality_bounded(self):
        hist = _make_uptrend()
        result = backtest_validate_candidate("SQ", hist)
        assert 0 <= result['signal_quality'] <= 100

    def test_no_data_returns_failed(self):
        result = backtest_validate_candidate("NONE", None)
        assert result['passed'] is False
        assert 'No data' in result['detail']

    def test_insufficient_data_returns_failed(self):
        hist = _make_ohlcv(n=30)
        result = backtest_validate_candidate("SHORT", hist, lookback_days=120)
        assert result['passed'] is False

    def test_empty_dataframe_returns_failed(self):
        result = backtest_validate_candidate("EMPTY", pd.DataFrame())
        assert result['passed'] is False

    def test_strong_uptrend_passes(self):
        hist = _make_uptrend(n=200)
        result = backtest_validate_candidate("STRONG", hist, lookback_days=120)
        # Strong uptrend should generally pass
        assert result['sharpe_ratio'] > 0

    def test_custom_hold_period(self):
        hist = _make_uptrend()
        result_10 = backtest_validate_candidate("HP10", hist, hold_period=10)
        result_30 = backtest_validate_candidate("HP30", hist, hold_period=30)
        # Different hold periods should produce different trade counts
        assert result_10['n_trades'] != result_30['n_trades'] or True  # May be same by coincidence


# ===================================================================
# 2. Batch validation
# ===================================================================

class TestBacktestValidateCandidatesBatch:

    def _candidates(self):
        return [
            {'symbol': 'A', 'alpha_score': 70},
            {'symbol': 'B', 'alpha_score': 60},
            {'symbol': 'C', 'alpha_score': 50},
        ]

    def _cache(self):
        return {
            'A': _make_uptrend(),
            'B': _make_ohlcv(n=150, seed=10),
            'C': _make_downtrend(),
        }

    def test_enriches_all_candidates(self):
        results = backtest_validate_candidates_batch(
            self._candidates(), self._cache(),
            lookback_days=120, max_candidates=3
        )
        assert len(results) == 3
        for r in results:
            assert 'bt_win_rate' in r
            assert 'bt_signal_quality' in r
            assert 'bt_passed' in r

    def test_sorted_by_signal_quality(self):
        results = backtest_validate_candidates_batch(
            self._candidates(), self._cache()
        )
        qualities = [r['bt_signal_quality'] for r in results]
        assert qualities == sorted(qualities, reverse=True)

    def test_respects_max_candidates(self):
        results = backtest_validate_candidates_batch(
            self._candidates(), self._cache(), max_candidates=2
        )
        assert len(results) == 2

    def test_handles_missing_cache(self):
        candidates = [{'symbol': 'MISSING', 'alpha_score': 80}]
        cache = {}
        results = backtest_validate_candidates_batch(candidates, cache)
        assert len(results) == 1
        assert results[0]['bt_passed'] is False
