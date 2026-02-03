"""
Comprehensive unit tests for technical_analysis.py.

Covers: RSI, MACD, Bollinger Bands, OBV, ADX, and the composite
calculate_all_ta function.  Every test builds synthetic OHLCV DataFrames
so no network access or external data is required.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is importable (mirrors conftest.py).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from technical_analysis import (
    calculate_adx,
    calculate_all_ta,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_obv,
    calculate_rsi,
)


# ---------------------------------------------------------------------------
# Helpers for building synthetic OHLCV DataFrames
# ---------------------------------------------------------------------------

def _make_ohlcv(closes, *, volumes=None, spread=0.5, start="2025-01-01"):
    """Build an OHLCV DataFrame from a sequence of closing prices.

    Parameters
    ----------
    closes : list or np.ndarray
        Sequence of closing prices.
    volumes : list or np.ndarray or None
        Matching volume data.  Defaults to constant 1 000 000.
    spread : float
        Half the distance between High and Low around Close.
    start : str
        Start date for the DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns Open, High, Low, Close, Volume and a
        DatetimeIndex.
    """
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    if volumes is None:
        volumes = np.full(n, 1_000_000.0)
    else:
        volumes = np.asarray(volumes, dtype=float)
    dates = pd.date_range(start=start, periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": closes - spread * 0.3,
            "High": closes + spread,
            "Low": closes - spread,
            "Close": closes,
            "Volume": volumes,
        },
        index=dates,
    )


def _make_empty_ohlcv():
    """Return an empty OHLCV DataFrame with the correct columns."""
    return pd.DataFrame(
        columns=["Open", "High", "Low", "Close", "Volume"],
        dtype=float,
    )


# ===================================================================
# 1. RSI Tests
# ===================================================================


class TestRSI:
    """Tests for calculate_rsi."""

    def test_rsi_known_values(self):
        """Alternating up-down pattern should yield RSI near 50."""
        # 40 bars alternating +1 / -1 around 100
        pattern = [100 + (1 if i % 2 == 0 else -1) for i in range(40)]
        hist = _make_ohlcv(pattern)
        rsi = calculate_rsi(hist)

        assert isinstance(rsi, pd.Series)
        last_rsi = rsi.iloc[-1]
        # Alternating gains and losses of equal magnitude -> RSI ~ 50
        assert 35.0 <= last_rsi <= 65.0, (
            f"Expected RSI near 50 for alternating pattern, got {last_rsi}"
        )

    def test_rsi_all_up(self):
        """Monotonically increasing prices should push RSI toward 100."""
        closes = np.linspace(50, 150, 50)  # steady climb
        hist = _make_ohlcv(closes)
        rsi = calculate_rsi(hist)

        last_rsi = rsi.iloc[-1]
        assert last_rsi >= 90.0, (
            f"Expected RSI near 100 for all-up data, got {last_rsi}"
        )

    def test_rsi_all_down(self):
        """Monotonically decreasing prices should push RSI toward 0."""
        closes = np.linspace(150, 50, 50)  # steady decline
        hist = _make_ohlcv(closes)
        rsi = calculate_rsi(hist)

        last_rsi = rsi.iloc[-1]
        assert last_rsi <= 10.0, (
            f"Expected RSI near 0 for all-down data, got {last_rsi}"
        )

    def test_rsi_insufficient_data(self):
        """A single-row DataFrame should return a NaN-filled Series."""
        hist = _make_ohlcv([100.0])
        rsi = calculate_rsi(hist)

        assert isinstance(rsi, pd.Series)
        # With only 1 row the function cannot compute diffs -> all NaN
        assert rsi.isna().all(), "Expected all NaN for single-row input"

    def test_rsi_empty_dataframe(self):
        """Empty DataFrame should return an empty Series without crashing."""
        hist = _make_empty_ohlcv()
        rsi = calculate_rsi(hist)

        assert isinstance(rsi, pd.Series)
        assert len(rsi) == 0


# ===================================================================
# 2. MACD Tests
# ===================================================================


class TestMACD:
    """Tests for calculate_macd."""

    def test_macd_bullish_crossover(self):
        """A sustained downtrend followed by a single sharp spike should
        produce a bullish crossover on the last bar.

        During the decline, the fast EMA stays below the slow EMA so
        MACD is negative and below the signal line.  The large final
        spike lifts the fast EMA much more than the slow EMA, causing
        MACD to jump above the signal line in one bar.
        """
        # 58-bar steady decline keeps MACD well below signal.
        decline = np.linspace(100, 42, 58).tolist()
        # One flat bar (second-to-last) then a massive spike (last bar).
        spike = [42.0, 120.0]
        closes = decline + spike  # 60 bars total
        hist = _make_ohlcv(closes)
        result = calculate_macd(hist)

        assert result["crossover"] == "bullish", (
            f"Expected bullish crossover, got '{result['crossover']}'"
        )

    def test_macd_flat_data(self):
        """Constant price should yield MACD, signal, and histogram ~ 0."""
        closes = [100.0] * 60
        hist = _make_ohlcv(closes)
        result = calculate_macd(hist)

        assert abs(result["macd_line"]) < 1e-6, (
            f"Expected MACD line ~0, got {result['macd_line']}"
        )
        assert abs(result["signal_line"]) < 1e-6, (
            f"Expected signal line ~0, got {result['signal_line']}"
        )
        assert abs(result["histogram"]) < 1e-6, (
            f"Expected histogram ~0, got {result['histogram']}"
        )

    def test_macd_returns_all_keys(self):
        """Result dictionary must contain every documented key."""
        hist = _make_ohlcv(np.linspace(90, 110, 50))
        result = calculate_macd(hist)

        expected_keys = {
            "macd_line",
            "signal_line",
            "histogram",
            "crossover",
            "macd_series",
            "signal_series",
            "histogram_series",
        }
        assert set(result.keys()) == expected_keys


# ===================================================================
# 3. Bollinger Bands Tests
# ===================================================================


class TestBollingerBands:
    """Tests for calculate_bollinger_bands."""

    def test_bollinger_squeeze(self):
        """Very low-volatility data should trigger a squeeze
        (bandwidth < 0.05)."""
        # 30 bars hovering tightly around 100 (noise < 0.1)
        rng = np.random.default_rng(42)
        closes = 100.0 + rng.normal(scale=0.01, size=30)
        hist = _make_ohlcv(closes, spread=0.01)
        result = calculate_bollinger_bands(hist)

        assert result["squeeze"] is True, (
            f"Expected squeeze=True for low-vol data, "
            f"bandwidth={result['bandwidth']}"
        )

    def test_bollinger_pct_b_range(self):
        """For moderate-volatility data, %B should be between 0 and 1."""
        rng = np.random.default_rng(7)
        closes = 100.0 + np.cumsum(rng.normal(scale=0.5, size=40))
        hist = _make_ohlcv(closes)
        result = calculate_bollinger_bands(hist)

        pct_b = result["pct_b"]
        assert not np.isnan(pct_b), "Expected finite %B"
        # For a price that stays within +-2 sigma, %B is typically 0-1.
        # Allow small overshoot since it *can* exceed the bands.
        assert -0.5 <= pct_b <= 1.5, (
            f"Expected %B in reasonable range, got {pct_b}"
        )

    def test_bollinger_returns_series(self):
        """upper_series, middle_series, lower_series must be pd.Series."""
        hist = _make_ohlcv(np.linspace(95, 105, 30))
        result = calculate_bollinger_bands(hist)

        for key in ("upper_series", "middle_series", "lower_series"):
            assert isinstance(result[key], pd.Series), (
                f"Expected pd.Series for '{key}', got {type(result[key])}"
            )


# ===================================================================
# 4. OBV Tests
# ===================================================================


class TestOBV:
    """Tests for calculate_obv."""

    def test_obv_rising_volume(self):
        """Consistently rising price with volume should produce rising OBV."""
        closes = np.linspace(100, 120, 20)
        volumes = np.full(20, 1_000_000.0)
        hist = _make_ohlcv(closes, volumes=volumes)
        result = calculate_obv(hist)

        assert result["obv_trend"] == "rising", (
            f"Expected OBV trend 'rising', got '{result['obv_trend']}'"
        )
        # OBV series should be monotonically increasing (after the first bar)
        obv_s = result["obv_series"]
        diffs = obv_s.diff().iloc[2:]  # skip first NaN + initial bar
        assert (diffs >= 0).all(), "OBV should be non-decreasing for rising prices"

    def test_obv_divergence_detection(self):
        """Price falling while OBV rises should flag bullish divergence.

        Construction: over the last 6 bars the *close* declines but most
        of the intra-bar moves are upward with heavy volume, then a
        small drop on light volume at the end pulls the close below the
        starting point.
        """
        # Prefix: 10 bars of neutral movement (enough history)
        prefix_closes = [100.0] * 10
        prefix_volumes = [500_000.0] * 10

        # Divergence window (6 bars):
        # Bars that rise on big volume, then one bar that drops on tiny volume.
        div_closes = [100.0, 102.0, 104.0, 106.0, 108.0, 99.0]
        div_volumes = [2_000_000, 2_000_000, 2_000_000, 2_000_000,
                       2_000_000, 100]

        closes = prefix_closes + div_closes
        volumes = prefix_volumes + div_volumes
        hist = _make_ohlcv(closes, volumes=volumes)
        result = calculate_obv(hist)

        # Price fell over the 5-bar lookback (99 < 108), but accumulated
        # OBV should have risen thanks to the heavy up-volume bars.
        assert result["obv_divergence"] == "bullish", (
            f"Expected bullish divergence, got '{result['obv_divergence']}'"
        )


# ===================================================================
# 5. ADX Tests
# ===================================================================


class TestADX:
    """Tests for calculate_adx."""

    def test_adx_strong_trend(self):
        """Steady uptrend with expanding highs/lows should give ADX > 25."""
        n = 60
        base = np.linspace(100, 200, n)
        hist = _make_ohlcv(base, spread=1.0)
        result = calculate_adx(hist)

        assert result["adx"] > 25.0, (
            f"Expected ADX > 25 for strong trend, got {result['adx']}"
        )
        assert result["trend_strength"] == "strong"

    def test_adx_weak_trend(self):
        """Choppy sideways data should produce ADX < 20."""
        rng = np.random.default_rng(99)
        # Oscillate tightly around 100 with no net direction
        n = 80
        noise = rng.normal(scale=0.5, size=n)
        closes = 100.0 + noise
        hist = _make_ohlcv(closes, spread=0.3)
        result = calculate_adx(hist)

        assert result["adx"] < 20.0, (
            f"Expected ADX < 20 for sideways data, got {result['adx']}"
        )
        assert result["trend_strength"] == "weak"


# ===================================================================
# 6. calculate_all_ta Tests
# ===================================================================


class TestCalculateAllTA:
    """Tests for calculate_all_ta."""

    @staticmethod
    def _large_hist():
        """Return a 100-bar trending DataFrame suitable for all indicators."""
        rng = np.random.default_rng(123)
        closes = 100.0 + np.cumsum(rng.normal(loc=0.1, scale=1.0, size=100))
        return _make_ohlcv(closes)

    def test_all_ta_returns_complete_dict(self):
        """Result must contain every documented top-level key."""
        hist = self._large_hist()
        result = calculate_all_ta(hist)

        expected_keys = {
            "rsi",
            "macd",
            "bollinger",
            "obv",
            "adx",
            "fibonacci",
            "stochastic",
            "cci",
            "atr",
            "market_structure",
            "ta_signal",
            "ta_score",
            "ta_reasons",
        }
        assert set(result.keys()) == expected_keys

    def test_all_ta_score_range(self):
        """ta_score must be an integer in [0, 100]."""
        hist = self._large_hist()
        result = calculate_all_ta(hist)

        assert isinstance(result["ta_score"], int)
        assert 0 <= result["ta_score"] <= 100, (
            f"ta_score out of range: {result['ta_score']}"
        )

    def test_all_ta_signal_values(self):
        """ta_signal must be one of the five documented labels."""
        hist = self._large_hist()
        result = calculate_all_ta(hist)

        valid_signals = {"STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"}
        assert result["ta_signal"] in valid_signals, (
            f"Unexpected signal: {result['ta_signal']}"
        )

    def test_all_ta_empty_data(self):
        """Empty DataFrame should return neutral defaults without crashing."""
        hist = _make_empty_ohlcv()
        result = calculate_all_ta(hist)

        assert isinstance(result, dict)
        # Score should fall at the neutral baseline (50) since no
        # adjustments can be computed.
        assert result["ta_score"] == 50
        assert result["ta_signal"] == "NEUTRAL"

    def test_all_ta_insufficient_data(self):
        """Short DataFrame (< 35 rows) should still return a valid dict
        with neutral defaults for indicators that require more data."""
        hist = _make_ohlcv(np.linspace(100, 105, 10))
        result = calculate_all_ta(hist)

        assert isinstance(result, dict)
        assert 0 <= result["ta_score"] <= 100
        assert result["ta_signal"] in {
            "STRONG_BUY", "BUY", "NEUTRAL", "SELL", "STRONG_SELL"
        }
        # MACD requires slow+signal=35 rows -> should be nan
        assert np.isnan(result["macd"]["macd_line"])
        # Bollinger requires 20 rows -> should be nan
        assert np.isnan(result["bollinger"]["upper"])
