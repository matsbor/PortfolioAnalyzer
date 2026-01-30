"""
Technical Analysis Module for Mining Stock Portfolio Analyzer.

Provides functions for computing standard technical indicators on OHLCV
price data, including RSI, MACD, Bollinger Bands, OBV, ADX, and Fibonacci
retracement levels.  All functions operate on pandas DataFrames with
columns: Open, High, Low, Close, Volume and a DatetimeIndex.

Dependencies: pandas, numpy (no other imports).
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_hist(hist: pd.DataFrame, min_rows: int = 1,
                   required_cols: list = None) -> bool:
    """Validate that the input DataFrame is suitable for analysis.

    Args:
        hist: OHLCV DataFrame to validate.
        min_rows: Minimum number of non-trivial rows required.
        required_cols: Column names that must be present and not all-NaN.
                       Defaults to ``['Close']`` when *None*.

    Returns:
        True if *hist* passes every check, False otherwise.
    """
    if hist is None or not isinstance(hist, pd.DataFrame) or hist.empty:
        return False
    if len(hist) < min_rows:
        return False
    if required_cols is None:
        required_cols = ['Close']
    for col in required_cols:
        if col not in hist.columns:
            return False
        if hist[col].isna().all():
            return False
    return True


def _nan_series(hist: pd.DataFrame, name: str = '') -> pd.Series:
    """Return a NaN-filled Series that matches *hist*'s index (or empty)."""
    if hist is not None and isinstance(hist, pd.DataFrame) and not hist.empty:
        return pd.Series(np.nan, index=hist.index, name=name)
    return pd.Series(dtype=float, name=name)


# ---------------------------------------------------------------------------
# 1. Relative Strength Index (RSI)
# ---------------------------------------------------------------------------

def calculate_rsi(hist: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index using Wilder's smoothing.

    Wilder's smoothing is an exponential moving average with
    ``alpha = 1 / period``, which is equivalent to a span of
    ``2 * period - 1``.

    Args:
        hist: OHLCV DataFrame with at least a ``Close`` column.
        period: RSI look-back period (default 14).

    Returns:
        A :class:`pandas.Series` of RSI values in the range 0--100.
        Returns a NaN-filled Series when the input is empty or has
        fewer than 2 rows.
    """
    if not _validate_hist(hist, min_rows=2, required_cols=['Close']):
        return _nan_series(hist, name='RSI')

    delta = hist['Close'].diff()
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    alpha = 1.0 / period
    avg_gain = gains.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = losses.ewm(alpha=alpha, adjust=False).mean()

    # Avoid division by zero: where avg_loss is 0 RS is undefined -> RSI = 100
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Where avg_loss was 0 (all gains), RSI should be 100
    rsi = rsi.fillna(100.0)

    # Where avg_gain was also 0 (no movement), RSI is conventionally 50
    no_movement = (avg_gain == 0) & (avg_loss == 0)
    rsi[no_movement] = 50.0

    rsi.name = 'RSI'
    return rsi


# ---------------------------------------------------------------------------
# 2. Moving Average Convergence Divergence (MACD)
# ---------------------------------------------------------------------------

def calculate_macd(hist: pd.DataFrame, fast: int = 12, slow: int = 26,
                   signal: int = 9) -> dict:
    """Calculate the MACD indicator with crossover detection.

    Uses standard exponential moving averages (``span`` parameter) for
    the fast / slow lines and the signal smoother.

    Args:
        hist: OHLCV DataFrame with at least a ``Close`` column.
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).
        signal: Signal-line EMA period (default 9).

    Returns:
        A dictionary with keys:

        * ``macd_line`` -- latest MACD line value (float).
        * ``signal_line`` -- latest signal line value (float).
        * ``histogram`` -- latest histogram value (float).
        * ``crossover`` -- ``'bullish'``, ``'bearish'``, or ``'none'``.
        * ``macd_series`` -- full MACD line (:class:`pandas.Series`).
        * ``signal_series`` -- full signal line (:class:`pandas.Series`).
        * ``histogram_series`` -- full histogram (:class:`pandas.Series`).
    """
    if not _validate_hist(hist, min_rows=slow + signal,
                          required_cols=['Close']):
        empty = _nan_series(hist)
        return {
            'macd_line': np.nan,
            'signal_line': np.nan,
            'histogram': np.nan,
            'crossover': 'none',
            'macd_series': empty,
            'signal_series': empty,
            'histogram_series': empty,
        }

    close = hist['Close']
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    # Crossover detection (compare last two bars)
    crossover = 'none'
    if len(macd_line) >= 2:
        curr_above = macd_line.iloc[-1] > signal_line.iloc[-1]
        prev_above = macd_line.iloc[-2] > signal_line.iloc[-2]
        if curr_above and not prev_above:
            crossover = 'bullish'
        elif not curr_above and prev_above:
            crossover = 'bearish'

    return {
        'macd_line': float(macd_line.iloc[-1]),
        'signal_line': float(signal_line.iloc[-1]),
        'histogram': float(histogram.iloc[-1]),
        'crossover': crossover,
        'macd_series': macd_line,
        'signal_series': signal_line,
        'histogram_series': histogram,
    }


# ---------------------------------------------------------------------------
# 3. Bollinger Bands
# ---------------------------------------------------------------------------

def calculate_bollinger_bands(hist: pd.DataFrame, period: int = 20,
                              std_dev: float = 2.0) -> dict:
    """Calculate Bollinger Bands with Percent-B and bandwidth.

    Args:
        hist: OHLCV DataFrame with at least a ``Close`` column.
        period: SMA / rolling-std window length (default 20).
        std_dev: Number of standard deviations for the outer bands
                 (default 2.0).

    Returns:
        A dictionary with keys:

        * ``upper``, ``middle``, ``lower`` -- latest band values (float).
        * ``pct_b`` -- current price position within the bands
          (0 = lower band, 1 = upper band; can exceed 0--1).
        * ``bandwidth`` -- normalised band width
          ``(upper - lower) / middle``.
        * ``squeeze`` -- *True* when ``bandwidth < 0.05``, hinting at a
          potential breakout.
        * ``upper_series``, ``middle_series``, ``lower_series`` --
          full :class:`pandas.Series` for plotting.
    """
    if not _validate_hist(hist, min_rows=period, required_cols=['Close']):
        empty = _nan_series(hist)
        return {
            'upper': np.nan,
            'middle': np.nan,
            'lower': np.nan,
            'pct_b': np.nan,
            'bandwidth': np.nan,
            'squeeze': False,
            'upper_series': empty,
            'middle_series': empty,
            'lower_series': empty,
        }

    close = hist['Close']
    middle = close.rolling(window=period).mean()
    rolling_std = close.rolling(window=period).std()
    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std

    current_price = float(close.iloc[-1])
    upper_val = float(upper.iloc[-1])
    middle_val = float(middle.iloc[-1])
    lower_val = float(lower.iloc[-1])

    band_range = upper_val - lower_val
    if band_range > 0.0 and not np.isnan(band_range):
        pct_b = (current_price - lower_val) / band_range
    else:
        pct_b = np.nan

    if middle_val != 0.0 and not np.isnan(middle_val):
        bandwidth = band_range / middle_val
    else:
        bandwidth = np.nan

    squeeze = bool(bandwidth < 0.05) if not np.isnan(bandwidth) else False

    return {
        'upper': upper_val,
        'middle': middle_val,
        'lower': lower_val,
        'pct_b': float(pct_b),
        'bandwidth': float(bandwidth),
        'squeeze': squeeze,
        'upper_series': upper,
        'middle_series': middle,
        'lower_series': lower,
    }


# ---------------------------------------------------------------------------
# 4. On-Balance Volume (OBV)
# ---------------------------------------------------------------------------

def calculate_obv(hist: pd.DataFrame) -> dict:
    """Calculate On-Balance Volume with trend and divergence detection.

    Divergence is determined by comparing the 5-day directional change
    in closing price versus the 5-day directional change in OBV:

    * **Bullish divergence** -- price fell while OBV rose (hidden buying).
    * **Bearish divergence** -- price rose while OBV fell (hidden selling).

    Args:
        hist: OHLCV DataFrame with ``Close`` and ``Volume`` columns.

    Returns:
        A dictionary with keys:

        * ``obv_current`` -- latest OBV reading (float).
        * ``obv_trend`` -- ``'rising'`` or ``'falling'``.
        * ``obv_divergence`` -- ``'bullish'``, ``'bearish'``, or
          ``'none'``.
        * ``obv_series`` -- full OBV :class:`pandas.Series`.
    """
    if not _validate_hist(hist, min_rows=2,
                          required_cols=['Close', 'Volume']):
        return {
            'obv_current': np.nan,
            'obv_trend': 'falling',
            'obv_divergence': 'none',
            'obv_series': _nan_series(hist, name='OBV'),
        }

    close = hist['Close']
    volume = hist['Volume']

    # +volume when close rises, -volume when close falls, 0 when flat
    direction = np.sign(close.diff())
    obv = (direction * volume).fillna(0.0).cumsum()
    obv.name = 'OBV'

    obv_current = float(obv.iloc[-1])

    # Compare the most recent 5 trading days (or fewer if not available)
    lookback = min(5, len(obv) - 1)
    if lookback > 0:
        obv_change = float(obv.iloc[-1] - obv.iloc[-1 - lookback])
        price_change = float(close.iloc[-1] - close.iloc[-1 - lookback])

        obv_trend = 'rising' if obv_change > 0 else 'falling'

        price_rising = price_change > 0
        obv_rising = obv_change > 0

        if not price_rising and obv_rising:
            divergence = 'bullish'
        elif price_rising and not obv_rising:
            divergence = 'bearish'
        else:
            divergence = 'none'
    else:
        obv_trend = 'falling'
        divergence = 'none'

    return {
        'obv_current': obv_current,
        'obv_trend': obv_trend,
        'obv_divergence': divergence,
        'obv_series': obv,
    }


# ---------------------------------------------------------------------------
# 5. Average Directional Index (ADX)
# ---------------------------------------------------------------------------

def calculate_adx(hist: pd.DataFrame, period: int = 14) -> dict:
    """Calculate the Average Directional Index.

    Uses Wilder's smoothing (``alpha = 1/period``) for True Range and
    Directional Movement values.

    Trend strength classification:

    * **strong** -- ADX > 25
    * **moderate** -- 20 <= ADX <= 25
    * **weak** -- ADX < 20

    Args:
        hist: OHLCV DataFrame with ``High``, ``Low``, and ``Close``
              columns.
        period: ADX look-back period (default 14).

    Returns:
        A dictionary with keys:

        * ``adx`` -- ADX value 0--100 (float).
        * ``plus_di`` -- +DI value (float).
        * ``minus_di`` -- -DI value (float).
        * ``trend_strength`` -- ``'strong'``, ``'moderate'``, or
          ``'weak'``.
    """
    default = {
        'adx': np.nan,
        'plus_di': np.nan,
        'minus_di': np.nan,
        'trend_strength': 'weak',
    }

    if not _validate_hist(hist, min_rows=period + 1,
                          required_cols=['High', 'Low', 'Close']):
        return default

    high = hist['High']
    low = hist['Low']
    close = hist['Close']
    prev_close = close.shift(1)

    # True Range
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=hist.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=hist.index,
    )

    # Wilder's smoothing
    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    smooth_plus = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    smooth_minus = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    # Directional Indicators
    atr_safe = atr.replace(0.0, np.nan)
    plus_di = 100.0 * smooth_plus / atr_safe
    minus_di = 100.0 * smooth_minus / atr_safe

    # DX -> ADX
    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = 100.0 * di_diff / di_sum.replace(0.0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    adx_val = float(adx.iloc[-1])
    plus_di_val = float(plus_di.iloc[-1])
    minus_di_val = float(minus_di.iloc[-1])

    if np.isnan(adx_val):
        trend_strength = 'weak'
    elif adx_val > 25:
        trend_strength = 'strong'
    elif adx_val >= 20:
        trend_strength = 'moderate'
    else:
        trend_strength = 'weak'

    return {
        'adx': adx_val,
        'plus_di': plus_di_val,
        'minus_di': minus_di_val,
        'trend_strength': trend_strength,
    }


# ---------------------------------------------------------------------------
# 6. Fibonacci Retracement Levels
# ---------------------------------------------------------------------------

def calculate_fibonacci_levels(hist: pd.DataFrame,
                               lookback: int = 60) -> dict:
    """Calculate Fibonacci retracement levels over a look-back window.

    Identifies the highest high and lowest low over the most recent
    *lookback* trading days and computes standard Fibonacci retracement
    price levels.

    Args:
        hist: OHLCV DataFrame.  Uses ``High`` / ``Low`` when available,
              otherwise falls back to ``Close``.
        lookback: Number of recent trading days to consider (default 60).

    Returns:
        A dictionary with keys:

        * ``high`` -- period high price (float).
        * ``low`` -- period low price (float).
        * ``levels`` -- dict mapping level names (``'0.0'`` through
          ``'1.0'``) to price values.
        * ``nearest_support`` -- closest Fibonacci level **below** the
          current price (float).
        * ``nearest_resistance`` -- closest Fibonacci level **above** the
          current price (float).
    """
    if not _validate_hist(hist, min_rows=2, required_cols=['Close']):
        return {
            'high': np.nan,
            'low': np.nan,
            'levels': {},
            'nearest_support': np.nan,
            'nearest_resistance': np.nan,
        }

    data = hist.tail(lookback)

    # Prefer High/Low columns; fall back to Close
    if 'High' in data.columns and not data['High'].isna().all():
        period_high = float(data['High'].max())
    else:
        period_high = float(data['Close'].max())

    if 'Low' in data.columns and not data['Low'].isna().all():
        period_low = float(data['Low'].min())
    else:
        period_low = float(data['Close'].min())

    diff = period_high - period_low

    ratios = [
        ('0.0', 0.0),
        ('0.236', 0.236),
        ('0.382', 0.382),
        ('0.5', 0.5),
        ('0.618', 0.618),
        ('0.786', 0.786),
        ('1.0', 1.0),
    ]
    levels = {name: period_low + diff * ratio for name, ratio in ratios}

    current_price = float(hist['Close'].iloc[-1])

    supports = [v for v in levels.values() if v < current_price]
    nearest_support = float(max(supports)) if supports else float(period_low)

    resistances = [v for v in levels.values() if v > current_price]
    nearest_resistance = (float(min(resistances)) if resistances
                          else float(period_high))

    return {
        'high': period_high,
        'low': period_low,
        'levels': levels,
        'nearest_support': nearest_support,
        'nearest_resistance': nearest_resistance,
    }


# ---------------------------------------------------------------------------
# 7. Composite Analysis
# ---------------------------------------------------------------------------

def calculate_all_ta(hist: pd.DataFrame) -> dict:
    """Run every technical indicator and produce an aggregate signal.

    Combines RSI, MACD, Bollinger Bands, OBV, ADX, and Fibonacci levels
    into a single result dictionary.  An overall technical-analysis score
    (0--100, where 50 is neutral) and a categorical signal string are
    derived from the individual readings.

    **Scoring weights**

    ==========  ==========================================================
    Indicator   Contribution
    ==========  ==========================================================
    RSI         +10 oversold (<30), -10 overbought (>70),
                +5 if 30--40, -5 if 60--70
    MACD        +10 bullish crossover, -10 bearish crossover,
                +5 if histogram positive *and* rising
    Bollinger   +5 if %B < 0.2, -5 if %B > 0.8, +10 if squeeze
    OBV         +5 bullish divergence, -5 bearish divergence,
                +3 if OBV trend rising
    ADX         Net adjustment multiplied by 1.2 (strong, ADX > 25) or
                0.8 (weak, ADX < 20)
    ==========  ==========================================================

    **Signal mapping**

    ============  ============
    Score range   Signal
    ============  ============
    >= 70         STRONG_BUY
    >= 58         BUY
    43 -- 57      NEUTRAL
    <= 42         SELL
    <= 30         STRONG_SELL
    ============  ============

    Args:
        hist: OHLCV DataFrame.

    Returns:
        A dictionary with keys ``rsi``, ``macd``, ``bollinger``, ``obv``,
        ``adx``, ``fibonacci``, ``ta_signal``, ``ta_score``, and
        ``ta_reasons``.
    """
    # --- Compute each indicator ----------------------------------------
    rsi_series = calculate_rsi(hist)
    macd_result = calculate_macd(hist)
    bollinger_result = calculate_bollinger_bands(hist)
    obv_result = calculate_obv(hist)
    adx_result = calculate_adx(hist)
    fib_result = calculate_fibonacci_levels(hist)

    # Package RSI into a dict with both scalar and series
    rsi_current = np.nan
    if len(rsi_series) > 0:
        last_val = rsi_series.iloc[-1]
        if not np.isnan(last_val):
            rsi_current = float(last_val)

    rsi_dict = {
        'current': rsi_current,
        'series': rsi_series,
    }

    # --- Scoring -------------------------------------------------------
    adjustment = 0.0
    reasons: list = []

    # RSI contribution
    if not np.isnan(rsi_current):
        if rsi_current < 30:
            adjustment += 10
            reasons.append(
                f"RSI oversold ({rsi_current:.1f}): +10"
            )
        elif rsi_current > 70:
            adjustment -= 10
            reasons.append(
                f"RSI overbought ({rsi_current:.1f}): -10"
            )
        elif 30 <= rsi_current <= 40:
            adjustment += 5
            reasons.append(
                f"RSI approaching oversold ({rsi_current:.1f}): +5"
            )
        elif 60 <= rsi_current <= 70:
            adjustment -= 5
            reasons.append(
                f"RSI approaching overbought ({rsi_current:.1f}): -5"
            )

    # MACD crossover contribution
    if macd_result['crossover'] == 'bullish':
        adjustment += 10
        reasons.append("MACD bullish crossover: +10")
    elif macd_result['crossover'] == 'bearish':
        adjustment -= 10
        reasons.append("MACD bearish crossover: -10")

    # MACD histogram momentum
    if not np.isnan(macd_result['histogram']):
        hist_s = macd_result['histogram_series']
        if isinstance(hist_s, pd.Series) and len(hist_s) >= 2:
            curr_h = float(hist_s.iloc[-1])
            prev_h = float(hist_s.iloc[-2])
            if curr_h > 0 and curr_h > prev_h:
                adjustment += 5
                reasons.append(
                    "MACD histogram positive and rising: +5"
                )

    # Bollinger Bands contribution
    pct_b = bollinger_result['pct_b']
    if not np.isnan(pct_b):
        if pct_b < 0.2:
            adjustment += 5
            reasons.append(
                f"Price near lower Bollinger Band (%B={pct_b:.2f}): +5"
            )
        elif pct_b > 0.8:
            adjustment -= 5
            reasons.append(
                f"Price near upper Bollinger Band (%B={pct_b:.2f}): -5"
            )

    if bollinger_result['squeeze']:
        adjustment += 10
        reasons.append(
            "Bollinger Band squeeze detected (potential breakout): +10"
        )

    # OBV contribution
    if obv_result['obv_divergence'] == 'bullish':
        adjustment += 5
        reasons.append("Bullish OBV divergence: +5")
    elif obv_result['obv_divergence'] == 'bearish':
        adjustment -= 5
        reasons.append("Bearish OBV divergence: -5")

    if obv_result['obv_trend'] == 'rising':
        adjustment += 3
        reasons.append("OBV trend rising: +3")

    # ADX multiplier (applied to the aggregate adjustment)
    adx_val = adx_result['adx']
    if not np.isnan(adx_val):
        if adx_val > 25:
            adjustment *= 1.2
            reasons.append(
                f"Strong trend (ADX={adx_val:.1f}): "
                f"signals multiplied by 1.2"
            )
        elif adx_val < 20:
            adjustment *= 0.8
            reasons.append(
                f"Weak trend (ADX={adx_val:.1f}): "
                f"signals multiplied by 0.8"
            )

    # Final score, clamped to [0, 100]
    ta_score = int(round(50 + adjustment))
    ta_score = max(0, min(100, ta_score))

    # Signal mapping
    if ta_score >= 70:
        ta_signal = 'STRONG_BUY'
    elif ta_score >= 58:
        ta_signal = 'BUY'
    elif ta_score <= 30:
        ta_signal = 'STRONG_SELL'
    elif ta_score <= 42:
        ta_signal = 'SELL'
    else:
        ta_signal = 'NEUTRAL'

    return {
        'rsi': rsi_dict,
        'macd': macd_result,
        'bollinger': bollinger_result,
        'obv': obv_result,
        'adx': adx_result,
        'fibonacci': fib_result,
        'ta_signal': ta_signal,
        'ta_score': ta_score,
        'ta_reasons': reasons,
    }
