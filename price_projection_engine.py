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


# --- Model D: Bollinger Band Regression (TA-driven targets) ---
def model_d_bollinger_regression(hist: pd.DataFrame) -> Dict[str, float]:
    """Use Bollinger Bands to project 30d price targets.

    Logic:
    - Squeeze (bandwidth < 5%) → breakout imminent, widen expected range
    - %B < 0.2 (near lower band) → mean-reversion upward expected
    - %B > 0.8 (near upper band) → mean-reversion downward expected
    - Band-walk (price riding upper/lower band for 3+ bars) → trend target
    """
    out = {"low_30d": np.nan, "high_30d": np.nan, "expected_30d": np.nan,
           "squeeze": False, "pct_b": np.nan, "band_walk": "none"}
    if hist is None or hist.empty or "Close" not in hist.columns:
        return out
    close = hist["Close"].dropna()
    if len(close) < 25:
        return out
    current = float(close.iloc[-1])
    if current <= 0:
        return out

    # Bollinger Band calculation (20-period, 2 std dev)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2.0 * std20
    lower = sma20 - 2.0 * std20

    upper_val = float(upper.iloc[-1])
    lower_val = float(lower.iloc[-1])
    middle_val = float(sma20.iloc[-1])
    band_range = upper_val - lower_val

    if band_range <= 0 or np.isnan(band_range):
        return out

    pct_b = (current - lower_val) / band_range
    bandwidth = band_range / middle_val if middle_val > 0 else 0
    squeeze = bandwidth < 0.05
    out["squeeze"] = squeeze
    out["pct_b"] = pct_b

    # Band-walk detection: price near upper or lower band for 3+ consecutive bars
    band_walk = "none"
    if len(upper) >= 5:
        recent_pct_b = []
        for i in range(-5, 0):
            u = float(upper.iloc[i])
            l = float(lower.iloc[i])
            r = u - l
            if r > 0:
                recent_pct_b.append((float(close.iloc[i]) - l) / r)
        if len(recent_pct_b) >= 3:
            last3 = recent_pct_b[-3:]
            if all(b > 0.8 for b in last3):
                band_walk = "upper"
            elif all(b < 0.2 for b in last3):
                band_walk = "lower"
    out["band_walk"] = band_walk

    # Projection logic
    if squeeze:
        # Squeeze → expect expanded range (use 1.5x band width as target range)
        out["low_30d"] = current - band_range * 0.75
        out["high_30d"] = current + band_range * 0.75
        out["expected_30d"] = middle_val  # revert to mean during squeeze
    elif band_walk == "upper":
        # Trending up → target upper band + half-width continuation
        out["low_30d"] = middle_val
        out["high_30d"] = upper_val + band_range * 0.25
        out["expected_30d"] = upper_val
    elif band_walk == "lower":
        # Trending down → target lower band - half-width continuation
        out["low_30d"] = lower_val - band_range * 0.25
        out["high_30d"] = middle_val
        out["expected_30d"] = lower_val
    elif pct_b < 0.2:
        # Near lower band → expect mean reversion up
        out["low_30d"] = lower_val
        out["high_30d"] = middle_val + band_range * 0.25
        out["expected_30d"] = middle_val
    elif pct_b > 0.8:
        # Near upper band → expect mean reversion down
        out["low_30d"] = middle_val - band_range * 0.25
        out["high_30d"] = upper_val
        out["expected_30d"] = middle_val
    else:
        # Mid-band → use band boundaries as range
        out["low_30d"] = lower_val
        out["high_30d"] = upper_val
        out["expected_30d"] = middle_val

    return out


# --- Model E: BOS / Market Structure Projection ---
def model_e_structure_projection(hist: pd.DataFrame) -> Dict[str, float]:
    """Use Break of Structure (BOS) and swing points to project 30d targets.

    Logic:
    - Bullish BOS → target next swing high, support at last swing low
    - Bearish BOS → target next swing low, resistance at last swing high
    - CHoCH → use opposite structure targets (reversal expected)
    - Ranging → use swing range as bounds
    """
    out = {"low_30d": np.nan, "high_30d": np.nan, "expected_30d": np.nan,
           "structure": "RANGING", "event": "NONE"}
    if hist is None or hist.empty or len(hist) < 50:
        return out
    close = hist["Close"].dropna()
    current = float(close.iloc[-1])
    if current <= 0:
        return out

    df = hist.tail(100).copy().reset_index(drop=True)

    # Identify swing points (5-bar pivots)
    swing_highs = []
    swing_lows = []
    for i in range(5, len(df) - 5):
        if df["High"].iloc[i] == df["High"].iloc[i - 5:i + 6].max():
            swing_highs.append(float(df["High"].iloc[i]))
        if df["Low"].iloc[i] == df["Low"].iloc[i - 5:i + 6].min():
            swing_lows.append(float(df["Low"].iloc[i]))

    if len(swing_highs) < 3 or len(swing_lows) < 3:
        return out

    # Market structure
    rh = swing_highs[-3:]
    rl = swing_lows[-3:]
    hh = rh[2] > rh[1] > rh[0]
    hl = rl[2] > rl[1] > rl[0]
    lh = rh[2] < rh[1] < rh[0]
    ll = rl[2] < rl[1] < rl[0]

    last_sh = swing_highs[-1]
    last_sl = swing_lows[-1]
    swing_range = last_sh - last_sl if last_sh > last_sl else 0

    # ATR-based buffer for BOS detection
    if len(df) >= 15 and current > 0:
        _tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr14 = float(_tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
        bos_buf = max(0.01, min(0.04, 0.5 * atr14 / current))
    else:
        bos_buf = 0.015

    bullish = hh and hl
    bearish = lh and ll

    if bullish:
        out["structure"] = "BULLISH"
        if current > last_sh * (1.0 + bos_buf):
            out["event"] = "BOS_UP"
            # BOS up → project continuation: target next swing range above
            out["low_30d"] = last_sl
            out["high_30d"] = last_sh + swing_range
            out["expected_30d"] = last_sh + swing_range * 0.5
        else:
            out["low_30d"] = last_sl
            out["high_30d"] = last_sh * (1.0 + bos_buf * 2)
            out["expected_30d"] = (last_sh + current) / 2
    elif bearish:
        out["structure"] = "BEARISH"
        if current < last_sl * (1.0 - bos_buf):
            out["event"] = "BOS_DOWN"
            # BOS down → project continuation: target next swing range below
            out["low_30d"] = last_sl - swing_range
            out["high_30d"] = last_sh
            out["expected_30d"] = last_sl - swing_range * 0.5
        else:
            out["low_30d"] = last_sl * (1.0 - bos_buf * 2)
            out["high_30d"] = last_sh
            out["expected_30d"] = (last_sl + current) / 2
    else:
        # Ranging → use swing boundaries
        out["structure"] = "RANGING"
        out["low_30d"] = last_sl
        out["high_30d"] = last_sh
        out["expected_30d"] = (last_sh + last_sl) / 2

    # CHoCH detection (reversal)
    if bullish and current < last_sl * (1.0 - bos_buf):
        out["event"] = "CHOCH_DOWN"
        out["expected_30d"] = last_sl - swing_range * 0.3
    elif bearish and current > last_sh * (1.0 + bos_buf):
        out["event"] = "CHOCH_UP"
        out["expected_30d"] = last_sh + swing_range * 0.3

    return out


# --- Model F: Linear Regression Channel ---
def model_f_regression_channel(hist: pd.DataFrame, lookback: int = 60) -> Dict[str, float]:
    """Use linear regression channel to project 30d targets.

    Fits OLS to recent price action and projects the trend line forward,
    with channel width based on standard error of the residuals.
    """
    out = {"low_30d": np.nan, "high_30d": np.nan, "expected_30d": np.nan,
           "slope_pct": np.nan, "r_squared": np.nan}
    if hist is None or hist.empty or "Close" not in hist.columns:
        return out
    close = hist["Close"].dropna().tail(lookback)
    if len(close) < 30:
        return out
    current = float(close.iloc[-1])
    if current <= 0:
        return out

    # OLS regression: price = a + b * x
    n = len(close)
    x = np.arange(n, dtype=float)
    y = close.values.astype(float)
    x_mean = x.mean()
    y_mean = y.mean()
    ss_xx = ((x - x_mean) ** 2).sum()
    ss_xy = ((x - x_mean) * (y - y_mean)).sum()

    if ss_xx == 0:
        return out

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    # R-squared
    y_pred = intercept + slope * x
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    out["r_squared"] = float(r_sq)

    # Standard error of residuals (channel width)
    std_err = np.sqrt(ss_res / max(1, n - 2))

    # Slope as daily % change
    slope_pct = (slope / current * 100) if current > 0 else 0
    out["slope_pct"] = float(slope_pct)

    # Project forward 30 days
    x_future = float(n - 1 + 30)  # 30 trading days ahead
    projected_center = intercept + slope * x_future

    # Weight by R-squared: high R² → tighter channel, low R² → wider
    channel_mult = 2.0 if r_sq > 0.7 else 2.5 if r_sq > 0.4 else 3.0
    out["expected_30d"] = float(projected_center)
    out["high_30d"] = float(projected_center + channel_mult * std_err)
    out["low_30d"] = float(projected_center - channel_mult * std_err)

    # Sanity: don't project negative prices
    out["low_30d"] = max(0.001, out["low_30d"])

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
    d = model_d_bollinger_regression(hist)
    e = model_e_structure_projection(hist)
    f = model_f_regression_channel(hist)

    all_models = [a, b, c, d, e, f]
    lows = [x["low_30d"] for x in all_models if not np.isnan(x.get("low_30d", np.nan))]
    highs = [x["high_30d"] for x in all_models if not np.isnan(x.get("high_30d", np.nan))]
    expected = [x["expected_30d"] for x in all_models if not np.isnan(x.get("expected_30d", np.nan))]

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
        "model_d_bollinger": d,
        "model_e_structure": e,
        "model_f_regression": f,
    }
