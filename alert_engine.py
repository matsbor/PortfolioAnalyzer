"""
Alert Engine -- Real-time monitoring and notification system for mining stock portfolios.

Monitors portfolio positions and market conditions, generating alerts when
configurable thresholds are breached. Supports price breakouts/breakdowns,
volume spikes, technical-analysis signals, financing detection, portfolio-level
checks (principal harvest, liquidity downgrades), and metal regime changes.

Each alert is a dict with keys:
    symbol, alert_type, severity, message, value, timestamp
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Alert type registry
# ---------------------------------------------------------------------------

ALERT_TYPES: Dict[str, Dict[str, str]] = {
    "PRICE_BREAKOUT": {"severity": "info", "desc": "Price broke above resistance"},
    "PRICE_BREAKDOWN": {"severity": "warning", "desc": "Price broke below support"},
    "VOLUME_SPIKE": {"severity": "info", "desc": "Volume exceeds 2x 20-day average"},
    "RSI_OVERSOLD": {"severity": "info", "desc": "RSI dropped below 30"},
    "RSI_OVERBOUGHT": {"severity": "warning", "desc": "RSI rose above 70"},
    "MACD_BULLISH_CROSS": {"severity": "info", "desc": "MACD bullish crossover"},
    "MACD_BEARISH_CROSS": {"severity": "warning", "desc": "MACD bearish crossover"},
    "FINANCING_DETECTED": {"severity": "critical", "desc": "Financing announcement detected"},
    "INSIDER_BUYING": {"severity": "info", "desc": "Insider purchase detected"},
    "LIQUIDITY_DOWNGRADE": {"severity": "warning", "desc": "Liquidity tier downgraded"},
    "METAL_REGIME_CHANGE": {"severity": "warning", "desc": "Metal crossed key moving average"},
    "BOLLINGER_SQUEEZE": {"severity": "info", "desc": "Bollinger Band squeeze detected"},
    "PRINCIPAL_HARVEST": {"severity": "info", "desc": "Position reached 2x cost basis"},
    "RUNWAY_CRISIS": {"severity": "critical", "desc": "Cash runway below 6 months"},
    "DEATH_CROSS": {"severity": "warning", "desc": "50-day MA crossed below 200-day MA"},
    "GAP_DOWN": {"severity": "warning", "desc": "Price gapped down more than 10%"},
    "DILUTION_HIGH": {"severity": "warning", "desc": "Dilution risk score above 70"},
    "INSIDER_SELLING": {"severity": "warning", "desc": "Insider sale detected"},
}

_SEVERITY_ORDER: Dict[str, int] = {"critical": 0, "warning": 1, "info": 2}

_DEFAULT_ALERTS_PATH: Path = Path.home() / ".alpha_miner_alerts.json"

_FINANCING_KEYWORDS: List[str] = [
    "financing",
    "placement",
    "offering",
    "capital raise",
    "bought deal",
    "shelf",
    "atm",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_alert(
    symbol: str,
    alert_type: str,
    message: str,
    value: Optional[float] = None,
) -> Dict[str, Any]:
    """Build a single alert dict with consistent shape."""
    severity = ALERT_TYPES.get(alert_type, {}).get("severity", "info")
    return {
        "symbol": symbol,
        "alert_type": alert_type,
        "severity": severity,
        "message": message,
        "value": value,
        "timestamp": datetime.utcnow().isoformat(),
    }


def _safe_series(series: pd.Series) -> pd.Series:
    """Return a copy with NaN/Inf values dropped."""
    return series.replace([np.inf, -np.inf], np.nan).dropna()


# ---------------------------------------------------------------------------
# 1. Price alerts
# ---------------------------------------------------------------------------


def check_price_alerts(
    symbol: str,
    hist: pd.DataFrame,
    support_level: Optional[float] = None,
    resistance_level: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Detect price breakouts and breakdowns.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    hist : pd.DataFrame
        OHLCV DataFrame with a ``Close`` column (and optionally ``High``/``Low``).
    support_level : float, optional
        Explicit support price.  When *None* the 20-day rolling low is used.
    resistance_level : float, optional
        Explicit resistance price.  When *None* the 20-day rolling high is used.

    Returns
    -------
    list[dict]
        Zero or more alert dicts.
    """
    alerts: List[Dict[str, Any]] = []

    if hist is None or hist.empty or len(hist) < 2:
        return alerts

    close_col = "Close" if "Close" in hist.columns else "Adj Close" if "Adj Close" in hist.columns else None
    if close_col is None:
        return alerts

    close = _safe_series(hist[close_col])
    if close.empty:
        return alerts

    latest_close = float(close.iloc[-1])
    lookback = min(20, len(close) - 1)

    if lookback < 1:
        return alerts

    # Determine resistance / support
    if resistance_level is not None:
        resistance = float(resistance_level)
    else:
        high_col = "High" if "High" in hist.columns else close_col
        resistance = float(hist[high_col].iloc[-lookback - 1 : -1].max())

    if support_level is not None:
        support = float(support_level)
    else:
        low_col = "Low" if "Low" in hist.columns else close_col
        support = float(hist[low_col].iloc[-lookback - 1 : -1].min())

    # Previous close for crossover detection
    prev_close = float(close.iloc[-2])

    if latest_close > resistance and prev_close <= resistance:
        alerts.append(
            _make_alert(
                symbol,
                "PRICE_BREAKOUT",
                f"{symbol} broke above resistance {resistance:.2f} (close={latest_close:.2f})",
                value=latest_close,
            )
        )

    if latest_close < support and prev_close >= support:
        alerts.append(
            _make_alert(
                symbol,
                "PRICE_BREAKDOWN",
                f"{symbol} broke below support {support:.2f} (close={latest_close:.2f})",
                value=latest_close,
            )
        )

    return alerts


# ---------------------------------------------------------------------------
# 2. Volume alerts
# ---------------------------------------------------------------------------


def check_volume_alerts(
    symbol: str,
    hist: pd.DataFrame,
    threshold: float = 2.0,
) -> List[Dict[str, Any]]:
    """Alert when the latest volume exceeds *threshold* x the 20-day average.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    hist : pd.DataFrame
        OHLCV DataFrame with a ``Volume`` column.
    threshold : float
        Multiplier above the 20-day mean volume to trigger the alert.

    Returns
    -------
    list[dict]
        Zero or one alert dict.
    """
    alerts: List[Dict[str, Any]] = []

    if hist is None or hist.empty or "Volume" not in hist.columns:
        return alerts

    vol = _safe_series(hist["Volume"])
    if len(vol) < 2:
        return alerts

    latest_vol = float(vol.iloc[-1])
    lookback = min(20, len(vol) - 1)
    avg_vol = float(vol.iloc[-lookback - 1 : -1].mean())

    if avg_vol <= 0:
        return alerts

    ratio = latest_vol / avg_vol

    if ratio >= threshold:
        alerts.append(
            _make_alert(
                symbol,
                "VOLUME_SPIKE",
                f"{symbol} volume spike: {latest_vol:,.0f} = {ratio:.1f}x 20-day avg ({avg_vol:,.0f})",
                value=round(ratio, 2),
            )
        )

    return alerts


# ---------------------------------------------------------------------------
# 3. Technical-analysis alerts
# ---------------------------------------------------------------------------


def check_ta_alerts(
    symbol: str,
    ta_result: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Check RSI, MACD crossover, and Bollinger squeeze from a TA result dict.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    ta_result : dict
        Output from ``technical_analysis.calculate_all_ta()`` (or compatible
        structure).  Expected keys: ``rsi``, ``macd``, ``macd_signal``,
        ``macd_histogram``, ``bb_bandwidth`` (or ``bb_upper``/``bb_lower``/``bb_mid``).

    Returns
    -------
    list[dict]
        Alert dicts for any triggered TA conditions.
    """
    alerts: List[Dict[str, Any]] = []

    if not ta_result or not isinstance(ta_result, dict):
        return alerts

    # --- RSI -----------------------------------------------------------------
    rsi = ta_result.get("rsi")
    if rsi is not None and np.isfinite(rsi):
        rsi_val = float(rsi)
        if rsi_val < 30:
            alerts.append(
                _make_alert(
                    symbol,
                    "RSI_OVERSOLD",
                    f"{symbol} RSI oversold at {rsi_val:.1f}",
                    value=round(rsi_val, 2),
                )
            )
        elif rsi_val > 70:
            alerts.append(
                _make_alert(
                    symbol,
                    "RSI_OVERBOUGHT",
                    f"{symbol} RSI overbought at {rsi_val:.1f}",
                    value=round(rsi_val, 2),
                )
            )

    # --- MACD crossover ------------------------------------------------------
    macd_val = ta_result.get("macd")
    macd_sig = ta_result.get("macd_signal")
    macd_hist = ta_result.get("macd_histogram")
    prev_hist = ta_result.get("prev_macd_histogram")

    if macd_hist is not None and prev_hist is not None:
        if np.isfinite(macd_hist) and np.isfinite(prev_hist):
            if prev_hist <= 0 < macd_hist:
                alerts.append(
                    _make_alert(
                        symbol,
                        "MACD_BULLISH_CROSS",
                        f"{symbol} MACD bullish crossover (histogram {macd_hist:+.4f})",
                        value=round(float(macd_hist), 4),
                    )
                )
            elif prev_hist >= 0 > macd_hist:
                alerts.append(
                    _make_alert(
                        symbol,
                        "MACD_BEARISH_CROSS",
                        f"{symbol} MACD bearish crossover (histogram {macd_hist:+.4f})",
                        value=round(float(macd_hist), 4),
                    )
                )
    elif macd_val is not None and macd_sig is not None:
        # Fallback: simple level comparison (less reliable without previous bar)
        if np.isfinite(macd_val) and np.isfinite(macd_sig):
            cross_val = float(macd_val) - float(macd_sig)
            prev_cross = ta_result.get("prev_macd_diff")
            if prev_cross is not None and np.isfinite(prev_cross):
                if prev_cross <= 0 < cross_val:
                    alerts.append(
                        _make_alert(
                            symbol,
                            "MACD_BULLISH_CROSS",
                            f"{symbol} MACD bullish crossover (MACD={float(macd_val):.4f})",
                            value=round(cross_val, 4),
                        )
                    )
                elif prev_cross >= 0 > cross_val:
                    alerts.append(
                        _make_alert(
                            symbol,
                            "MACD_BEARISH_CROSS",
                            f"{symbol} MACD bearish crossover (MACD={float(macd_val):.4f})",
                            value=round(cross_val, 4),
                        )
                    )

    # --- Bollinger squeeze ---------------------------------------------------
    bb_bandwidth = ta_result.get("bb_bandwidth")
    if bb_bandwidth is None:
        # Try to compute from upper/lower/mid
        bb_upper = ta_result.get("bb_upper")
        bb_lower = ta_result.get("bb_lower")
        bb_mid = ta_result.get("bb_mid")
        if (
            bb_upper is not None
            and bb_lower is not None
            and bb_mid is not None
            and np.isfinite(bb_upper)
            and np.isfinite(bb_lower)
            and np.isfinite(bb_mid)
            and float(bb_mid) != 0
        ):
            bb_bandwidth = (float(bb_upper) - float(bb_lower)) / float(bb_mid)

    if bb_bandwidth is not None and np.isfinite(bb_bandwidth):
        bb_bandwidth = float(bb_bandwidth)
        # A bandwidth below 0.10 (10 %) is a classic squeeze threshold
        if bb_bandwidth < 0.10:
            alerts.append(
                _make_alert(
                    symbol,
                    "BOLLINGER_SQUEEZE",
                    f"{symbol} Bollinger squeeze detected (bandwidth={bb_bandwidth:.4f})",
                    value=round(bb_bandwidth, 4),
                )
            )

    return alerts


# ---------------------------------------------------------------------------
# 4. Financing alerts
# ---------------------------------------------------------------------------


def check_financing_alerts(
    symbol: str,
    news_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Scan news headlines for financing-related keywords.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    news_items : list[dict]
        Each item should have a ``title`` (str) key.

    Returns
    -------
    list[dict]
        Zero or one ``FINANCING_DETECTED`` alert.
    """
    alerts: List[Dict[str, Any]] = []

    if not news_items:
        return alerts

    for item in news_items:
        title = str(item.get("title", "")).lower()
        for keyword in _FINANCING_KEYWORDS:
            if keyword in title:
                alerts.append(
                    _make_alert(
                        symbol,
                        "FINANCING_DETECTED",
                        f"{symbol} financing detected: \"{item.get('title', '')}\"",
                        value=None,
                    )
                )
                # One alert per symbol is sufficient
                return alerts

    return alerts


# ---------------------------------------------------------------------------
# 5. Portfolio-level alerts
# ---------------------------------------------------------------------------


def check_portfolio_alerts(
    portfolio_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Check portfolio positions for harvest targets and liquidity downgrades.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Portfolio DataFrame.  Expected columns include ``Symbol``,
        ``Market_Value``, ``Cost_Basis``, and optionally ``Liq_Tier_Code``
        (or ``Liq_Tier``) and ``Prev_Liq_Tier_Code`` for downgrade detection.

    Returns
    -------
    list[dict]
        Alert dicts for principal-harvest and liquidity-downgrade conditions.
    """
    alerts: List[Dict[str, Any]] = []

    if portfolio_df is None or portfolio_df.empty:
        return alerts

    has_mv = "Market_Value" in portfolio_df.columns
    has_cb = "Cost_Basis" in portfolio_df.columns
    has_sym = "Symbol" in portfolio_df.columns

    if not has_sym:
        return alerts

    # Liquidity tier column detection
    liq_col = None
    for candidate in ("Liq_Tier_Code", "Liq_Tier", "liq_tier"):
        if candidate in portfolio_df.columns:
            liq_col = candidate
            break

    prev_liq_col = None
    for candidate in ("Prev_Liq_Tier_Code", "Prev_Liq_Tier", "prev_liq_tier"):
        if candidate in portfolio_df.columns:
            prev_liq_col = candidate
            break

    # Tier ordering: higher number = better liquidity
    tier_rank: Dict[str, int] = {"L0": 0, "L1": 1, "L2": 2, "L3": 3}

    for _, row in portfolio_df.iterrows():
        sym = str(row["Symbol"])

        # Principal harvest: Market_Value >= 2 * Cost_Basis
        if has_mv and has_cb:
            try:
                mv = float(row["Market_Value"])
                cb = float(row["Cost_Basis"])
                if cb > 0 and mv >= 2.0 * cb:
                    ratio = mv / cb
                    alerts.append(
                        _make_alert(
                            sym,
                            "PRINCIPAL_HARVEST",
                            f"{sym} reached {ratio:.1f}x cost basis (MV=${mv:,.0f} vs Cost=${cb:,.0f})",
                            value=round(ratio, 2),
                        )
                    )
            except (TypeError, ValueError):
                pass

        # Liquidity tier downgrade
        if liq_col is not None and prev_liq_col is not None:
            try:
                current_tier = str(row.get(liq_col, "")).strip().upper()
                prev_tier = str(row.get(prev_liq_col, "")).strip().upper()
                cur_rank = tier_rank.get(current_tier)
                prev_rank = tier_rank.get(prev_tier)
                if cur_rank is not None and prev_rank is not None and cur_rank < prev_rank:
                    alerts.append(
                        _make_alert(
                            sym,
                            "LIQUIDITY_DOWNGRADE",
                            f"{sym} liquidity downgraded from {prev_tier} to {current_tier}",
                            value=float(cur_rank),
                        )
                    )
            except (TypeError, ValueError):
                pass

    return alerts


# ---------------------------------------------------------------------------
# 6. Metal regime alerts
# ---------------------------------------------------------------------------


def check_metal_regime_alerts(
    spot_prices: Dict[str, Any],
    hist_cache: Dict[str, pd.DataFrame],
) -> List[Dict[str, Any]]:
    """Detect gold/silver crossing their 200-day moving average.

    Parameters
    ----------
    spot_prices : dict
        Must contain ``gold_live`` and/or ``silver_live`` float values.
    hist_cache : dict
        Keyed by ticker (``GLD``, ``SLV``); values are OHLCV DataFrames.

    Returns
    -------
    list[dict]
        Zero or more ``METAL_REGIME_CHANGE`` alerts.
    """
    alerts: List[Dict[str, Any]] = []

    if not spot_prices or not hist_cache:
        return alerts

    # Metal-to-ETF mapping: configurable via spot_prices dict.
    # Callers can pass 'gold_etf_ticker' / 'silver_etf_ticker' keys
    # to override the defaults. This avoids hardcoding specific ETFs.
    gold_etf = spot_prices.get('gold_etf_ticker', 'GLD')
    silver_etf = spot_prices.get('silver_etf_ticker', 'SLV')
    metal_map = [
        ("gold_live", gold_etf, "Gold"),
        ("silver_live", silver_etf, "Silver"),
    ]

    for price_key, etf_ticker, metal_name in metal_map:
        live_price = spot_prices.get(price_key)
        if live_price is None or not np.isfinite(live_price):
            continue

        etf_hist = hist_cache.get(etf_ticker)
        if etf_hist is None or etf_hist.empty:
            continue

        close_col = "Close" if "Close" in etf_hist.columns else None
        if close_col is None:
            continue

        close = _safe_series(etf_hist[close_col])
        if len(close) < 200:
            continue

        ma200 = float(close.iloc[-200:].mean())
        prev_close = float(close.iloc[-1])
        current_price = float(live_price)

        if ma200 <= 0:
            continue

        # Crossed above
        if prev_close <= ma200 < current_price:
            alerts.append(
                _make_alert(
                    etf_ticker,
                    "METAL_REGIME_CHANGE",
                    f"{metal_name} crossed above 200-day MA ({ma200:.2f}); "
                    f"live={current_price:.2f}",
                    value=round(current_price, 2),
                )
            )
        # Crossed below
        elif prev_close >= ma200 > current_price:
            alerts.append(
                _make_alert(
                    etf_ticker,
                    "METAL_REGIME_CHANGE",
                    f"{metal_name} crossed below 200-day MA ({ma200:.2f}); "
                    f"live={current_price:.2f}",
                    value=round(current_price, 2),
                )
            )

    return alerts


# ---------------------------------------------------------------------------
# 7. Master check -- run ALL alerts
# ---------------------------------------------------------------------------


def check_survival_alerts(
    symbol: str,
    row: Dict[str, Any],
    hist: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    """Check for survival-critical conditions: runway crisis, dilution, death cross, gap down.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    row : dict
        Portfolio row data (may contain Runway_Months, Dilution_Risk_Score, etc.).
    hist : pd.DataFrame, optional
        OHLCV history for death cross and gap-down detection.

    Returns
    -------
    list[dict]
        Alerts found.
    """
    alerts: List[Dict[str, Any]] = []

    # Runway crisis: cash runway < 6 months
    runway = row.get('Runway_Months')
    if runway is not None:
        try:
            runway_val = float(runway)
            if runway_val < 6:
                alerts.append(_make_alert(
                    symbol, "RUNWAY_CRISIS",
                    f"Cash runway {runway_val:.1f} months (< 6 months) — financing likely imminent",
                    value=runway_val,
                ))
        except (TypeError, ValueError):
            pass

    # Dilution risk critical
    dilution = row.get('Dilution_Risk_Score')
    if dilution is not None:
        try:
            dil_val = float(dilution)
            if dil_val >= 70:
                alerts.append(_make_alert(
                    symbol, "DILUTION_HIGH",
                    f"Dilution risk {dil_val:.0f}/100 — elevated share dilution risk",
                    value=dil_val,
                ))
        except (TypeError, ValueError):
            pass

    if hist is not None and not hist.empty and 'Close' in hist.columns:
        close = hist['Close'].dropna()

        # Death cross: MA50 < MA200
        if len(close) >= 200:
            ma50 = close.rolling(50).mean()
            ma200 = close.rolling(200).mean()
            if (len(ma50) >= 2 and len(ma200) >= 2
                    and not (np.isnan(ma50.iloc[-1]) or np.isnan(ma200.iloc[-1])
                             or np.isnan(ma50.iloc[-2]) or np.isnan(ma200.iloc[-2]))):
                prev_above = ma50.iloc[-2] >= ma200.iloc[-2]
                curr_below = ma50.iloc[-1] < ma200.iloc[-1]
                if prev_above and curr_below:
                    alerts.append(_make_alert(
                        symbol, "DEATH_CROSS",
                        f"50-day MA ({ma50.iloc[-1]:.2f}) crossed below 200-day MA ({ma200.iloc[-1]:.2f})",
                        value=float(ma50.iloc[-1]),
                    ))

        # Gap down: > 10% single-day drop
        if len(close) >= 2:
            prev_close = float(close.iloc[-2])
            curr_close = float(close.iloc[-1])
            if prev_close > 0:
                pct_change = (curr_close - prev_close) / prev_close * 100
                if pct_change <= -10.0:
                    alerts.append(_make_alert(
                        symbol, "GAP_DOWN",
                        f"Price dropped {pct_change:.1f}% in one day (${prev_close:.2f} -> ${curr_close:.2f})",
                        value=pct_change,
                    ))

    return alerts


def check_insider_selling_alerts(
    symbol: str,
    news_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Check news for insider selling keywords.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    news_items : list[dict]
        News items to scan.

    Returns
    -------
    list[dict]
        Alerts found.
    """
    alerts: List[Dict[str, Any]] = []
    _SELL_KEYWORDS = ["insider sell", "insider sale", "disposition", "sold shares",
                      "executive sell", "director sell", "officer sell"]
    for item in (news_items or []):
        title = (item.get("title") or "").lower()
        for kw in _SELL_KEYWORDS:
            if kw in title:
                alerts.append(_make_alert(
                    symbol, "INSIDER_SELLING",
                    f"Insider selling detected: {item.get('title', 'Unknown')}",
                ))
                return alerts  # One alert per symbol
    return alerts


def check_all_alerts(
    portfolio_df: pd.DataFrame,
    hist_cache: Dict[str, pd.DataFrame],
    news_cache: Dict[str, List[Dict[str, Any]]],
    spot_prices: Dict[str, Any],
    ta_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Run every alert check and return a deduplicated, severity-sorted list.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Portfolio positions DataFrame.
    hist_cache : dict
        ``{symbol: pd.DataFrame}`` of OHLCV history.
    news_cache : dict
        ``{symbol: [news_item, ...]}`` from news sources.
    spot_prices : dict
        Live metal spot prices (``gold_live``, ``silver_live``, ...).
    ta_cache : dict, optional
        ``{symbol: ta_result_dict}`` from ``calculate_all_ta()``.

    Returns
    -------
    list[dict]
        Deduplicated alerts sorted by severity (critical > warning > info),
        then alphabetically by symbol.
    """
    all_alerts: List[Dict[str, Any]] = []

    # Determine symbols from portfolio
    symbols: List[str] = []
    if portfolio_df is not None and not portfolio_df.empty and "Symbol" in portfolio_df.columns:
        symbols = portfolio_df["Symbol"].dropna().unique().tolist()

    # Per-symbol checks
    for sym in symbols:
        hist = hist_cache.get(sym) if hist_cache else None

        # Price alerts
        if hist is not None and not hist.empty:
            all_alerts.extend(check_price_alerts(sym, hist))

        # Volume alerts
        if hist is not None and not hist.empty:
            all_alerts.extend(check_volume_alerts(sym, hist))

        # TA alerts
        if ta_cache and sym in ta_cache:
            all_alerts.extend(check_ta_alerts(sym, ta_cache[sym]))

        # Financing / news alerts
        news_items = news_cache.get(sym) if news_cache else None
        if news_items:
            all_alerts.extend(check_financing_alerts(sym, news_items))
            all_alerts.extend(check_insider_selling_alerts(sym, news_items))

        # Survival alerts (runway, dilution, death cross, gap down)
        row_data = {}
        if portfolio_df is not None and not portfolio_df.empty and "Symbol" in portfolio_df.columns:
            sym_rows = portfolio_df[portfolio_df["Symbol"] == sym]
            if not sym_rows.empty:
                row_data = sym_rows.iloc[0].to_dict()
        all_alerts.extend(check_survival_alerts(sym, row_data, hist))

    # Portfolio-level alerts
    if portfolio_df is not None and not portfolio_df.empty:
        all_alerts.extend(check_portfolio_alerts(portfolio_df))

    # Metal regime alerts
    if spot_prices and hist_cache:
        all_alerts.extend(check_metal_regime_alerts(spot_prices, hist_cache))

    # Deduplicate by (symbol, alert_type) -- keep first occurrence
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for alert in all_alerts:
        key = (alert["symbol"], alert["alert_type"])
        if key not in seen:
            seen.add(key)
            deduped.append(alert)

    # Sort: severity (critical first), then symbol alphabetically
    deduped.sort(key=lambda a: (_SEVERITY_ORDER.get(a["severity"], 99), a["symbol"]))

    return deduped


# ---------------------------------------------------------------------------
# 8. Persistence -- save
# ---------------------------------------------------------------------------


def save_alerts(
    alerts: List[Dict[str, Any]],
    filepath: Optional[str] = None,
) -> None:
    """Append *alerts* to a JSON file, keeping the most recent 500 entries.

    Parameters
    ----------
    alerts : list[dict]
        Alert dicts to persist.
    filepath : str, optional
        Target JSON file.  Defaults to ``~/.alpha_miner_alerts.json``.
    """
    path = Path(filepath) if filepath else _DEFAULT_ALERTS_PATH

    existing: List[Dict[str, Any]] = []
    if path.exists():
        try:
            raw = path.read_text(encoding="utf-8")
            existing = json.loads(raw) if raw.strip() else []
        except (json.JSONDecodeError, OSError):
            existing = []

    combined = existing + list(alerts)

    # Keep only the last 500
    combined = combined[-500:]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(combined, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# 9. Persistence -- load
# ---------------------------------------------------------------------------


def load_alerts(
    filepath: Optional[str] = None,
    max_age_hours: float = 24,
) -> List[Dict[str, Any]]:
    """Load alerts from JSON file, filtering to the last *max_age_hours*.

    Parameters
    ----------
    filepath : str, optional
        Source JSON file.  Defaults to ``~/.alpha_miner_alerts.json``.
    max_age_hours : float
        Only return alerts whose ``timestamp`` is within this many hours of
        the current UTC time.

    Returns
    -------
    list[dict]
        Filtered alert dicts.
    """
    path = Path(filepath) if filepath else _DEFAULT_ALERTS_PATH

    if not path.exists():
        return []

    try:
        raw = path.read_text(encoding="utf-8")
        data: List[Dict[str, Any]] = json.loads(raw) if raw.strip() else []
    except (json.JSONDecodeError, OSError):
        return []

    cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)

    filtered: List[Dict[str, Any]] = []
    for alert in data:
        ts_str = alert.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str)
            if ts >= cutoff:
                filtered.append(alert)
        except (ValueError, TypeError):
            # If we cannot parse the timestamp, include it conservatively
            filtered.append(alert)

    return filtered


# ---------------------------------------------------------------------------
# 10. Summary
# ---------------------------------------------------------------------------


def get_alert_summary(alerts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Produce a summary of alert counts by severity and symbol.

    Parameters
    ----------
    alerts : list[dict]
        Alert dicts (as produced by any ``check_*`` function).

    Returns
    -------
    dict
        ``{'total': int, 'critical': int, 'warning': int, 'info': int,
           'by_symbol': {symbol: count, ...}}``
    """
    summary: Dict[str, Any] = {
        "total": 0,
        "critical": 0,
        "warning": 0,
        "info": 0,
        "by_symbol": {},
    }

    if not alerts:
        return summary

    summary["total"] = len(alerts)

    for alert in alerts:
        sev = alert.get("severity", "info")
        if sev in summary:
            summary[sev] += 1

        sym = alert.get("symbol", "UNKNOWN")
        summary["by_symbol"][sym] = summary["by_symbol"].get(sym, 0) + 1

    return summary
