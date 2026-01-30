#!/usr/bin/env python3
"""
Sovereign Trust Engine (V6.0 Alpha) - Price Projection Models
Deep-Hunt logic: Mean Reversion, Momentum/RS, Commodity Proxy.
Output: Projected 30d window (high / low / expected).
"""
from __future__ import annotations

import datetime
import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Optional Tiingo
try:
    from tiingo import TiingoClient
    _TIINGO_AVAILABLE = True
except ImportError:
    TiingoClient = None
    _TIINGO_AVAILABLE = False


def _ensure_client():
    key = (os.getenv("TIINGO_API_KEY") or "").strip()
    if not key or not _TIINGO_AVAILABLE or TiingoClient is None:
        return None
    try:
        return TiingoClient({"api_key": key})
    except Exception:
        return None


def _fetch_history(client, symbol: str, years: int = 15) -> Optional[pd.DataFrame]:
    if client is None:
        return None
    end_d = datetime.date.today()
    start_d = end_d - datetime.timedelta(days=years * 365)
    start_s = start_d.strftime("%Y-%m-%d")
    end_s = end_d.strftime("%Y-%m-%d")
    try:
        data = client.get_ticker_price(
            symbol, startDate=start_s, endDate=end_s, frequency="daily", fmt="json"
        )
        if not data or not isinstance(data, list) or len(data) == 0:
            return None
        df = pd.DataFrame(data)
        if "date" not in df.columns:
            return None
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        for old, new in [("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close"), ("volume", "Volume")]:
            if old in df.columns:
                df[new] = df[old]
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c not in df.columns:
                df[c] = np.nan
        return df[["Open", "High", "Low", "Close", "Volume"]].copy()
    except Exception:
        return None


# --- Model A: Mean Reversion (15y oversold levels) ---
def model_a_mean_reversion(hist: pd.DataFrame) -> Dict[str, float]:
    """Use 15y history to find oversold levels and 30d mean-reversion targets."""
    out = {"low_30d": np.nan, "high_30d": np.nan, "expected_30d": np.nan, "oversold_pct": np.nan}
    if hist is None or hist.empty or "Close" not in hist.columns:
        return out
    close = hist["Close"].dropna()
    if len(close) < 60:
        return out
    current = float(close.iloc[-1])
    if current <= 0:
        return out
    # Percentiles over full history
    p05 = float(np.percentile(close, 5))
    p25 = float(np.percentile(close, 25))
    p50 = float(np.percentile(close, 50))
    p75 = float(np.percentile(close, 75))
    p95 = float(np.percentile(close, 95))
    # Oversold: % below median
    oversold_pct = (p50 - current) / p50 * 100 if p50 > 0 else 0
    out["oversold_pct"] = oversold_pct
    # 30d window: revert toward median; low ~ p25, high ~ p75, expected ~ p50
    out["low_30d"] = min(current, p25)
    out["high_30d"] = max(current, p75)
    out["expected_30d"] = p50
    return out


# --- Model B: Momentum / RS vs GDX ---
def model_b_momentum_rs(
    hist: pd.DataFrame,
    gdx: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Alpha DNA velocity vs GDX; 30d momentum-extrapolated targets."""
    out = {"low_30d": np.nan, "high_30d": np.nan, "expected_30d": np.nan, "rs_velocity": np.nan}
    if hist is None or hist.empty or "Close" not in hist.columns:
        return out
    close = hist["Close"].dropna()
    if len(close) < 30:
        return out
    current = float(close.iloc[-1])
    if current <= 0:
        return out
    ret_30 = (close.iloc[-1] / close.iloc[-30] - 1.0) * 100 if len(close) >= 30 else 0.0
    ret_90 = (close.iloc[-1] / close.iloc[-90] - 1.0) * 100 if len(close) >= 90 else ret_30
    # Simple velocity: 30d return as % (use for extrapolation)
    rs_velocity = float(ret_30)
    out["rs_velocity"] = rs_velocity
    # Extrapolate 30d: conservative low, momentum high
    monthly = ret_30 / 30.0 if ret_30 else 0.0
    low_pct = min(0, monthly) * 30
    high_pct = max(0, monthly) * 30
    out["low_30d"] = current * (1 + low_pct / 100)
    out["high_30d"] = current * (1 + high_pct / 100)
    out["expected_30d"] = current * (1 + monthly * 15 / 100)
    if gdx is not None and not gdx.empty and "Close" in gdx.columns:
        g = gdx["Close"].dropna()
        if len(g) >= 30:
            g_ret = (g.iloc[-1] / g.iloc[-30] - 1.0) * 100
            # Relative strength: stock vs GDX
            out["rs_velocity"] = float(ret_30 - g_ret)
    return out


# --- Model C: Commodity Proxy (correlate with Gold/Silver) ---
def model_c_commodity_proxy(
    hist: pd.DataFrame,
    gold: Optional[pd.DataFrame] = None,
    silver: Optional[pd.DataFrame] = None,
    metal: str = "Gold",
) -> Dict[str, float]:
    """Correlate stock with raw Gold/Silver; 30d proxy-based targets."""
    out = {"low_30d": np.nan, "high_30d": np.nan, "expected_30d": np.nan, "correlation": np.nan}
    if hist is None or hist.empty or "Close" not in hist.columns:
        return out
    close = hist["Close"].dropna()
    if len(close) < 30:
        return out
    current = float(close.iloc[-1])
    if current <= 0:
        return out
    proxy = gold if metal and str(metal).lower().startswith("g") else silver
    if proxy is None or proxy.empty or "Close" not in proxy.columns:
        proxy = gold if gold is not None and not gold.empty else silver
    corr = np.nan
    if proxy is not None and not proxy.empty and "Close" in proxy.columns:
        px = proxy["Close"].dropna()
        # Align by index
        common = close.index.intersection(px.index)
        if len(common) >= 20:
            a = close.reindex(common).ffill().bfill()
            b = px.reindex(common).ffill().bfill()
            valid = ~(a.isna() | b.isna())
            if valid.sum() >= 20:
                c = np.corrcoef(a[valid].values, b[valid].values)[0, 1]
                corr = float(c) if not np.isnan(c) else np.nan
    out["correlation"] = corr
    # Use proxy return for 30d if available
    ret_stock = (close.iloc[-1] / close.iloc[-30] - 1.0) * 100 if len(close) >= 30 else 0.0
    out["expected_30d"] = current * (1 + ret_stock / 100)
    out["low_30d"] = current * 0.95
    out["high_30d"] = current * 1.10
    return out


def project_30d_window(
    symbol: str,
    metal: str = "Gold",
    hist: Optional[pd.DataFrame] = None,
    gdx: Optional[pd.DataFrame] = None,
    gold: Optional[pd.DataFrame] = None,
    silver: Optional[pd.DataFrame] = None,
    tiingo_client: Any = None,
) -> Dict[str, Any]:
    """
    Run all three models and aggregate into a single 30d projected window.
    Returns dict: low_30d, high_30d, expected_30d, model_a, model_b, model_c.
    """
    client = tiingo_client or _ensure_client()
    if hist is None and client:
        hist = _fetch_history(client, symbol)
    if gdx is None and client:
        gdx = _fetch_history(client, "GDX")
    if gold is None and client:
        gold = _fetch_history(client, "GLD")
    if silver is None and client:
        silver = _fetch_history(client, "SLV")

    a = model_a_mean_reversion(hist)
    b = model_b_momentum_rs(hist, gdx)
    c = model_c_commodity_proxy(hist, gold, silver, metal)

    lows = [x["low_30d"] for x in (a, b, c) if not np.isnan(x["low_30d"])]
    highs = [x["high_30d"] for x in (a, b, c) if not np.isnan(x["high_30d"])]
    expected = [x["expected_30d"] for x in (a, b, c) if not np.isnan(x["expected_30d"])]

    low_30d = float(np.nanmin(lows)) if lows else np.nan
    high_30d = float(np.nanmax(highs)) if highs else np.nan
    expected_30d = float(np.nanmean(expected)) if expected else np.nan

    return {
        "low_30d": low_30d,
        "high_30d": high_30d,
        "expected_30d": expected_30d,
        "model_a": a,
        "model_b": b,
        "model_c": c,
    }
