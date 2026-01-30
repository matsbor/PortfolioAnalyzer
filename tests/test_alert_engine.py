"""
Tests for alert_engine.py -- covers price, volume, TA, financing, portfolio,
check_all_alerts orchestration, alert summary, and persistence round-trip.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alert_engine import (
    check_price_alerts,
    check_volume_alerts,
    check_ta_alerts,
    check_financing_alerts,
    check_portfolio_alerts,
    check_all_alerts,
    save_alerts,
    load_alerts,
    get_alert_summary,
)

# ---------------------------------------------------------------------------
# Helpers -- synthetic data builders
# ---------------------------------------------------------------------------


def _make_ohlcv(
    closes,
    highs=None,
    lows=None,
    volumes=None,
    start_date="2025-01-01",
):
    """Build a minimal OHLCV DataFrame from lists."""
    n = len(closes)
    dates = pd.date_range(start=start_date, periods=n, freq="B")
    if highs is None:
        highs = [c + 0.5 for c in closes]
    if lows is None:
        lows = [c - 0.5 for c in closes]
    if volumes is None:
        volumes = [100_000] * n
    return pd.DataFrame(
        {"Close": closes, "High": highs, "Low": lows, "Volume": volumes},
        index=dates,
    )


# ===================================================================
# 1. Price Alerts
# ===================================================================


class TestPriceAlerts:
    """Tests for check_price_alerts."""

    def test_price_breakout_detected(self):
        """Close > 20-day high should produce a PRICE_BREAKOUT alert.

        Build 22 bars where the first 21 have Close / High at 10.0 and the
        final bar jumps to 11.0 so that latest_close > resistance AND
        prev_close <= resistance (the crossover condition).
        """
        closes = [10.0] * 21 + [11.0]
        highs = [10.0] * 21 + [11.5]
        lows = [9.5] * 21 + [10.5]
        hist = _make_ohlcv(closes, highs=highs, lows=lows)

        alerts = check_price_alerts("TST.V", hist)

        assert len(alerts) == 1
        alert = alerts[0]
        assert alert["alert_type"] == "PRICE_BREAKOUT"
        assert alert["symbol"] == "TST.V"
        assert alert["severity"] == "info"
        assert alert["value"] == 11.0

    def test_price_breakdown_detected(self):
        """Close < 20-day low should produce a PRICE_BREAKDOWN alert.

        Build 22 bars where the first 21 have Close / Low at 10.0 and the
        final bar drops to 9.0 so that latest_close < support AND
        prev_close >= support.
        """
        closes = [10.0] * 21 + [9.0]
        highs = [10.5] * 21 + [10.0]
        lows = [10.0] * 21 + [8.5]
        hist = _make_ohlcv(closes, highs=highs, lows=lows)

        alerts = check_price_alerts("TST.V", hist)

        assert len(alerts) == 1
        alert = alerts[0]
        assert alert["alert_type"] == "PRICE_BREAKDOWN"
        assert alert["symbol"] == "TST.V"
        assert alert["severity"] == "warning"
        assert alert["value"] == 9.0

    def test_no_price_alert_normal(self):
        """A close that stays within the 20-day range triggers nothing."""
        closes = [10.0] * 22
        highs = [10.5] * 22
        lows = [9.5] * 22
        hist = _make_ohlcv(closes, highs=highs, lows=lows)

        alerts = check_price_alerts("TST.V", hist)
        assert alerts == []


# ===================================================================
# 2. Volume Alerts
# ===================================================================


class TestVolumeAlerts:
    """Tests for check_volume_alerts."""

    def test_volume_spike_detected(self):
        """Volume > 2x 20-day average should fire VOLUME_SPIKE."""
        volumes = [100_000] * 21 + [250_000]
        closes = [10.0] * 22
        hist = _make_ohlcv(closes, volumes=volumes)

        alerts = check_volume_alerts("TST.V", hist)

        assert len(alerts) == 1
        alert = alerts[0]
        assert alert["alert_type"] == "VOLUME_SPIKE"
        assert alert["symbol"] == "TST.V"
        assert alert["severity"] == "info"
        assert alert["value"] == 2.5

    def test_no_volume_alert_normal(self):
        """Volume at or below the average should produce no alerts."""
        volumes = [100_000] * 22
        closes = [10.0] * 22
        hist = _make_ohlcv(closes, volumes=volumes)

        alerts = check_volume_alerts("TST.V", hist)
        assert alerts == []


# ===================================================================
# 3. TA Alerts
# ===================================================================


class TestTAAlerts:
    """Tests for check_ta_alerts."""

    def test_rsi_oversold_alert(self):
        """RSI < 30 should produce RSI_OVERSOLD."""
        ta = {"rsi": 25.0}
        alerts = check_ta_alerts("TST.V", ta)

        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "RSI_OVERSOLD"
        assert alerts[0]["symbol"] == "TST.V"
        assert alerts[0]["severity"] == "info"
        assert alerts[0]["value"] == 25.0

    def test_rsi_overbought_alert(self):
        """RSI > 70 should produce RSI_OVERBOUGHT."""
        ta = {"rsi": 78.5}
        alerts = check_ta_alerts("TST.V", ta)

        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "RSI_OVERBOUGHT"
        assert alerts[0]["symbol"] == "TST.V"
        assert alerts[0]["severity"] == "warning"
        assert alerts[0]["value"] == 78.5

    def test_macd_crossover_alert(self):
        """Bullish crossover (prev_histogram <= 0, current > 0) should fire
        MACD_BULLISH_CROSS."""
        ta = {
            "macd_histogram": 0.05,
            "prev_macd_histogram": -0.02,
        }
        alerts = check_ta_alerts("TST.V", ta)

        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "MACD_BULLISH_CROSS"
        assert alerts[0]["symbol"] == "TST.V"
        assert alerts[0]["severity"] == "info"
        assert alerts[0]["value"] == 0.05


# ===================================================================
# 4. Financing Alerts
# ===================================================================


class TestFinancingAlerts:
    """Tests for check_financing_alerts."""

    def test_financing_detected(self):
        """A news headline containing 'financing' should produce
        FINANCING_DETECTED."""
        news = [{"title": "Company announces $10M financing round"}]
        alerts = check_financing_alerts("TST.V", news)

        assert len(alerts) == 1
        assert alerts[0]["alert_type"] == "FINANCING_DETECTED"
        assert alerts[0]["symbol"] == "TST.V"
        assert alerts[0]["severity"] == "critical"

    def test_no_financing_normal(self):
        """News without financing keywords should produce no alert."""
        news = [
            {"title": "Company reports strong Q3 results"},
            {"title": "New drill results exceed expectations"},
        ]
        alerts = check_financing_alerts("TST.V", news)
        assert alerts == []


# ===================================================================
# 5. Portfolio Alerts
# ===================================================================


class TestPortfolioAlerts:
    """Tests for check_portfolio_alerts."""

    def test_principal_harvest_detected(self):
        """Market_Value >= 2x Cost_Basis should fire PRINCIPAL_HARVEST."""
        df = pd.DataFrame(
            {
                "Symbol": ["GOLD.TO"],
                "Market_Value": [50_000.0],
                "Cost_Basis": [20_000.0],
            }
        )
        alerts = check_portfolio_alerts(df)

        assert len(alerts) == 1
        alert = alerts[0]
        assert alert["alert_type"] == "PRINCIPAL_HARVEST"
        assert alert["symbol"] == "GOLD.TO"
        assert alert["severity"] == "info"
        assert alert["value"] == 2.5


# ===================================================================
# 6. check_all_alerts orchestration
# ===================================================================


class TestCheckAllAlerts:
    """Tests for check_all_alerts -- deduplication, sorting, return type."""

    @staticmethod
    def _build_scenario():
        """Construct caches that will trigger multiple alert types for the
        same symbol so we can verify dedup and sort behaviour."""
        # Portfolio with two symbols that qualify for harvest
        portfolio_df = pd.DataFrame(
            {
                "Symbol": ["AAA", "AAA"],
                "Market_Value": [50_000.0, 50_000.0],
                "Cost_Basis": [20_000.0, 20_000.0],
            }
        )

        # History that triggers a volume spike for AAA
        volumes = [100_000] * 21 + [300_000]
        closes = [10.0] * 22
        hist_df = _make_ohlcv(closes, volumes=volumes)
        hist_cache = {"AAA": hist_df}

        # News that triggers a financing alert for AAA
        news_cache = {"AAA": [{"title": "AAA closes bought deal financing"}]}

        # TA that triggers RSI oversold for AAA
        ta_cache = {"AAA": {"rsi": 20.0}}

        spot_prices: dict = {}

        return portfolio_df, hist_cache, news_cache, spot_prices, ta_cache

    def test_all_alerts_deduplication(self):
        """No duplicate (symbol, alert_type) pairs should survive."""
        portfolio_df, hist_cache, news_cache, spot_prices, ta_cache = (
            self._build_scenario()
        )
        alerts = check_all_alerts(
            portfolio_df, hist_cache, news_cache, spot_prices, ta_cache
        )

        keys = [(a["symbol"], a["alert_type"]) for a in alerts]
        assert len(keys) == len(set(keys)), "Duplicate (symbol, alert_type) found"

    def test_all_alerts_sorted_by_severity(self):
        """Alerts must be ordered critical -> warning -> info."""
        portfolio_df, hist_cache, news_cache, spot_prices, ta_cache = (
            self._build_scenario()
        )
        alerts = check_all_alerts(
            portfolio_df, hist_cache, news_cache, spot_prices, ta_cache
        )

        severity_order = {"critical": 0, "warning": 1, "info": 2}
        severity_values = [severity_order.get(a["severity"], 99) for a in alerts]
        assert severity_values == sorted(severity_values), (
            "Alerts are not sorted by severity"
        )

    def test_all_alerts_returns_list(self):
        """check_all_alerts must always return a list even with empty inputs."""
        empty_df = pd.DataFrame(columns=["Symbol"])
        result = check_all_alerts(empty_df, {}, {}, {})

        assert isinstance(result, list)


# ===================================================================
# 7. Alert Summary
# ===================================================================


class TestAlertSummary:
    """Tests for get_alert_summary."""

    def test_summary_counts(self):
        """Severity counts should match the input alerts."""
        alerts = [
            {"symbol": "A", "alert_type": "X", "severity": "critical", "message": "", "value": None, "timestamp": ""},
            {"symbol": "A", "alert_type": "Y", "severity": "warning", "message": "", "value": None, "timestamp": ""},
            {"symbol": "B", "alert_type": "Z", "severity": "info", "message": "", "value": None, "timestamp": ""},
            {"symbol": "B", "alert_type": "W", "severity": "info", "message": "", "value": None, "timestamp": ""},
        ]
        summary = get_alert_summary(alerts)

        assert summary["total"] == 4
        assert summary["critical"] == 1
        assert summary["warning"] == 1
        assert summary["info"] == 2
        assert summary["by_symbol"]["A"] == 2
        assert summary["by_symbol"]["B"] == 2

    def test_summary_empty(self):
        """An empty alert list should give all-zero counts."""
        summary = get_alert_summary([])

        assert summary["total"] == 0
        assert summary["critical"] == 0
        assert summary["warning"] == 0
        assert summary["info"] == 0
        assert summary["by_symbol"] == {}


# ===================================================================
# 8. Persistence
# ===================================================================


class TestPersistence:
    """Tests for save_alerts / load_alerts round-trip."""

    def test_save_and_load_roundtrip(self, tmp_path):
        """Alerts saved to disk should be loadable and match the originals."""
        filepath = str(tmp_path / "test_alerts.json")

        original_alerts = [
            {
                "symbol": "TST.V",
                "alert_type": "VOLUME_SPIKE",
                "severity": "info",
                "message": "TST.V volume spike",
                "value": 2.5,
                "timestamp": pd.Timestamp.utcnow().isoformat(),
            },
            {
                "symbol": "GOLD.TO",
                "alert_type": "FINANCING_DETECTED",
                "severity": "critical",
                "message": "GOLD.TO financing detected",
                "value": None,
                "timestamp": pd.Timestamp.utcnow().isoformat(),
            },
        ]

        save_alerts(original_alerts, filepath=filepath)

        # Verify the file was written
        assert Path(filepath).exists()

        loaded = load_alerts(filepath=filepath, max_age_hours=1)

        assert len(loaded) == len(original_alerts)
        for orig, got in zip(original_alerts, loaded):
            assert got["symbol"] == orig["symbol"]
            assert got["alert_type"] == orig["alert_type"]
            assert got["severity"] == orig["severity"]
            assert got["message"] == orig["message"]
            assert got["value"] == orig["value"]
