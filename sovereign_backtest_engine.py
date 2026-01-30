#!/usr/bin/env python3
"""
V6.5 Deep-Trust: Multi-Model Backtest Engine (Sovereign Win Rate)
Runs 5y and 15y buy-and-hold backtests for the 6-Symbol Strike List.
Outputs: Sharpe Ratio, Max Drawdown, Sovereign Win Rate (% of positive periods).
"""
from __future__ import annotations

import datetime
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from tiingo import TiingoClient
    _TIINGO_AVAILABLE = True
except ImportError:
    TiingoClient = None
    _TIINGO_AVAILABLE = False

# 6-Symbol Strike List (matches fetch_ticker TICKER_STRIKE_MAP)
STRIKE_SYMBOLS = ["SKE.TO", "DSVSF", "NXE", "MAG", "PAAS", "AG"]


def _client():
    key = (os.getenv("TIINGO_API_KEY") or "").strip()
    if not key or not _TIINGO_AVAILABLE or TiingoClient is None:
        return None
    try:
        return TiingoClient({"api_key": key})
    except Exception:
        return None


def _fetch_hist(client, symbol: str, years: int) -> Optional[pd.DataFrame]:
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
        for o, n in [("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close"), ("volume", "Volume")]:
            if o in df.columns:
                df[n] = df[o]
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c not in df.columns:
                df[c] = np.nan
        return df[["Open", "High", "Low", "Close", "Volume"]].copy()
    except Exception:
        return None


def _returns(close: pd.Series) -> pd.Series:
    return close.pct_change(fill_method=None).dropna()


def _sharpe(close: pd.Series, annualize: bool = True) -> float:
    r = _returns(close)
    if r.empty or r.std() == 0:
        return np.nan
    mu = r.mean()
    sig = r.std()
    ann = np.sqrt(252) if annualize else 1.0
    return float((mu / sig) * ann)


def _max_drawdown(close: pd.Series) -> float:
    if close.empty or close.min() <= 0:
        return np.nan
    runmax = close.cummax()
    dd = (close - runmax) / runmax
    return float(dd.min())


def _win_rate(close: pd.Series, window: int = 252) -> float:
    """Sovereign Win Rate: % of rolling windows with positive return."""
    if len(close) < window + 1:
        return np.nan
    ret = close.pct_change(window, fill_method=None).dropna()
    if ret.empty:
        return np.nan
    return float((ret > 0).mean() * 100)


def run_strike_backtest(
    symbols: Optional[List[str]] = None,
    years_list: Optional[List[int]] = None,
    client: Any = None,
) -> Dict[str, Dict[str, float]]:
    """
    Run 5y and 15y buy-and-hold backtests per symbol.
    Returns: { symbol: { sharpe_5y, sharpe_15y, max_dd_5y, max_dd_15y, win_rate_5y, win_rate_15y } }
    """
    symbols = symbols or STRIKE_SYMBOLS
    years_list = years_list or [5, 15]
    c = client or _client()
    out: Dict[str, Dict[str, float]] = {}
    for sym in symbols:
        res = {
            "sharpe_5y": np.nan,
            "sharpe_15y": np.nan,
            "max_dd_5y": np.nan,
            "max_dd_15y": np.nan,
            "win_rate_5y": np.nan,
            "win_rate_15y": np.nan,
        }
        for y in years_list:
            h = _fetch_hist(c, sym, y)
            if h is None or h.empty or "Close" not in h.columns:
                continue
            close = h["Close"].dropna()
            if len(close) < 60:
                continue
            key_s = f"sharpe_{y}y"
            key_d = f"max_dd_{y}y"
            key_w = f"win_rate_{y}y"
            res[key_s] = _sharpe(close)
            res[key_d] = _max_drawdown(close)
            res[key_w] = _win_rate(close)
        out[sym] = res
    return out
