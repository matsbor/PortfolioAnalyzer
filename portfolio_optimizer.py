#!/usr/bin/env python3
"""
Portfolio Optimizer - Mean-Variance, Risk Parity, and Concentration Analysis
for mining stock portfolios.

Provides correlation analysis, metal-beta calculation, Markowitz optimization,
risk-parity weighting, Kelly sizing, concentration risk detection, and
rebalance trade generation.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# scipy is optional -- degrade gracefully when missing
try:
    from scipy.optimize import minimize as _scipy_minimize
    SCIPY_AVAILABLE = True
except ImportError:
    _scipy_minimize = None  # type: ignore[assignment]
    SCIPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# 1. Correlation matrix
# ---------------------------------------------------------------------------

def calculate_correlation_matrix(
    hist_cache: Dict[str, pd.DataFrame],
    min_overlap: int = 30,
) -> pd.DataFrame:
    """Compute pairwise Pearson correlation of close prices across symbols.

    Parameters
    ----------
    hist_cache : Dict[str, pd.DataFrame]
        Mapping of ticker symbol to a DataFrame that contains at least a
        ``Close`` column indexed (or indexable) by date.
    min_overlap : int, optional
        Minimum number of common trading days required for a pair to receive
        a correlation value.  Pairs below this threshold are set to ``NaN``.

    Returns
    -------
    pd.DataFrame
        Square correlation matrix with symbol names as both index and columns.
    """
    close_series: Dict[str, pd.Series] = {}
    for symbol, df in hist_cache.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        s = df["Close"].dropna()
        if s.empty:
            continue
        s.index = pd.to_datetime(s.index)
        close_series[symbol] = s

    symbols = sorted(close_series.keys())
    if not symbols:
        return pd.DataFrame()

    # Build an aligned DataFrame of close prices (outer join keeps all dates)
    aligned = pd.DataFrame({sym: close_series[sym] for sym in symbols})
    aligned.sort_index(inplace=True)

    n = len(symbols)
    corr_data = np.full((n, n), np.nan)

    for i in range(n):
        corr_data[i, i] = 1.0
        for j in range(i + 1, n):
            pair = aligned[[symbols[i], symbols[j]]].dropna()
            if len(pair) >= min_overlap:
                r = pair[symbols[i]].corr(pair[symbols[j]])
                corr_data[i, j] = r
                corr_data[j, i] = r

    return pd.DataFrame(corr_data, index=symbols, columns=symbols)


# ---------------------------------------------------------------------------
# 2. Rolling correlation
# ---------------------------------------------------------------------------

def calculate_rolling_correlation(
    hist_cache: Dict[str, pd.DataFrame],
    window: int = 60,
) -> Dict[str, pd.Series]:
    """Compute rolling pairwise correlation for every unique symbol pair.

    Parameters
    ----------
    hist_cache : Dict[str, pd.DataFrame]
        Same format as :func:`calculate_correlation_matrix`.
    window : int, optional
        Rolling window size in trading days.

    Returns
    -------
    Dict[str, pd.Series]
        Keys are ``"SYM1_SYM2"`` (alphabetically ordered), values are
        ``pd.Series`` of rolling correlations indexed by date.
    """
    close_series: Dict[str, pd.Series] = {}
    for symbol, df in hist_cache.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        s = df["Close"].dropna()
        if s.empty:
            continue
        s.index = pd.to_datetime(s.index)
        close_series[symbol] = s

    symbols = sorted(close_series.keys())
    result: Dict[str, pd.Series] = {}

    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            sym_a, sym_b = symbols[i], symbols[j]
            pair = pd.DataFrame({sym_a: close_series[sym_a],
                                 sym_b: close_series[sym_b]}).dropna()
            if len(pair) < window:
                continue
            rolling_corr = pair[sym_a].rolling(window).corr(pair[sym_b])
            rolling_corr = rolling_corr.dropna()
            if not rolling_corr.empty:
                key = f"{sym_a}_{sym_b}"
                result[key] = rolling_corr

    return result


# ---------------------------------------------------------------------------
# 3. Metal beta (OLS regression vs benchmark)
# ---------------------------------------------------------------------------

def calculate_metal_beta(
    hist: pd.DataFrame,
    benchmark_hist: pd.DataFrame,
) -> dict:
    """Calculate beta of a stock relative to a commodity benchmark ETF.

    Uses ordinary least-squares regression of daily returns:
        ``R_stock = alpha + beta * R_benchmark + epsilon``

    Parameters
    ----------
    hist : pd.DataFrame
        Stock price history with a ``Close`` column.
    benchmark_hist : pd.DataFrame
        Benchmark ETF (e.g. GLD, SLV, URA) price history with ``Close``.

    Returns
    -------
    dict
        ``{'beta': float, 'r_squared': float, 'alpha': float}``
        where *alpha* is annualised Jensen's alpha.
    """
    empty = {"beta": np.nan, "r_squared": np.nan, "alpha": np.nan}

    if hist is None or benchmark_hist is None:
        return empty
    if hist.empty or benchmark_hist.empty:
        return empty
    if "Close" not in hist.columns or "Close" not in benchmark_hist.columns:
        return empty

    stock_close = hist["Close"].dropna()
    bench_close = benchmark_hist["Close"].dropna()
    stock_close.index = pd.to_datetime(stock_close.index)
    bench_close.index = pd.to_datetime(bench_close.index)

    # Align on common dates
    combined = pd.DataFrame({"stock": stock_close, "bench": bench_close}).dropna()
    if len(combined) < 30:
        return empty

    stock_ret = combined["stock"].pct_change().dropna()
    bench_ret = combined["bench"].pct_change().dropna()

    # Re-align after pct_change
    aligned = pd.DataFrame({"y": stock_ret, "x": bench_ret}).dropna()
    if len(aligned) < 20:
        return empty

    y = aligned["y"].values
    x = aligned["x"].values

    # OLS via normal equations: y = alpha + beta * x
    x_with_const = np.column_stack([np.ones(len(x)), x])
    try:
        # (X'X)^-1 X'y
        coeffs = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return empty

    alpha_daily = float(coeffs[0])
    beta = float(coeffs[1])

    # R-squared
    y_hat = x_with_const @ coeffs
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot != 0.0 else np.nan

    # Annualise Jensen's alpha (252 trading days)
    alpha_annual = float((1.0 + alpha_daily) ** 252 - 1.0)

    return {"beta": beta, "r_squared": r_squared, "alpha": alpha_annual}


# ---------------------------------------------------------------------------
# 4. Mean-variance (Markowitz) optimisation
# ---------------------------------------------------------------------------

def _annualised_return_and_cov(
    hist_cache: Dict[str, pd.DataFrame],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract annualised mean returns and covariance from aligned close data."""
    close_map: Dict[str, pd.Series] = {}
    for sym, df in hist_cache.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        s = df["Close"].dropna()
        if len(s) < 30:
            continue
        s.index = pd.to_datetime(s.index)
        close_map[sym] = s

    symbols = sorted(close_map.keys())
    if len(symbols) < 2:
        return np.array([]), np.array([[]]), symbols

    aligned = pd.DataFrame({s: close_map[s] for s in symbols}).dropna()
    if len(aligned) < 30:
        return np.array([]), np.array([[]]), symbols

    daily_returns = aligned.pct_change().dropna()
    mean_daily = daily_returns.mean().values
    cov_daily = daily_returns.cov().values

    # Annualise
    mu = mean_daily * 252
    cov = cov_daily * 252

    return mu, cov, symbols


def _portfolio_stats(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float,
) -> Tuple[float, float, float]:
    """Return (expected return, volatility, Sharpe) for given weights."""
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def optimize_mean_variance(
    hist_cache: Dict[str, pd.DataFrame],
    risk_free_rate: float = 0.05,
    target_return: Optional[float] = None,
    max_weight: float = 0.15,
) -> dict:
    """Markowitz mean-variance portfolio optimisation.

    Parameters
    ----------
    hist_cache : Dict[str, pd.DataFrame]
        Ticker -> DataFrame with ``Close`` column.
    risk_free_rate : float, optional
        Annualised risk-free rate for Sharpe calculation.
    target_return : float or None, optional
        If given, find the minimum-variance portfolio that achieves this
        annualised return.  Otherwise maximise the Sharpe ratio.
    max_weight : float, optional
        Upper bound on any single position weight.

    Returns
    -------
    dict
        ``{'weights', 'expected_return', 'expected_volatility',
           'sharpe_ratio', 'efficient_frontier'}``
    """
    empty_result = {
        "weights": {},
        "expected_return": np.nan,
        "expected_volatility": np.nan,
        "sharpe_ratio": np.nan,
        "efficient_frontier": [],
    }

    if not SCIPY_AVAILABLE:
        warnings.warn(
            "scipy is not installed -- mean-variance optimisation unavailable.",
            stacklevel=2,
        )
        return empty_result

    mu, cov, symbols = _annualised_return_and_cov(hist_cache)
    n = len(symbols)
    if n < 2 or mu.size == 0:
        return empty_result

    bounds = tuple((0.0, max_weight) for _ in range(n))
    # Weights must sum to 1
    constraints: list = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    if target_return is not None:
        # Minimise variance subject to target return
        constraints.append(
            {"type": "eq", "fun": lambda w, _mu=mu, _tr=target_return: w @ _mu - _tr}
        )

        def objective(w: np.ndarray) -> float:
            return float(w @ cov @ w)
    else:
        # Maximise Sharpe ratio  ==  minimise negative Sharpe
        def objective(w: np.ndarray) -> float:
            ret = w @ mu
            vol = np.sqrt(w @ cov @ w)
            if vol < 1e-12:
                return 1e6
            return -(ret - risk_free_rate) / vol

    w0 = np.full(n, 1.0 / n)
    result = _scipy_minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        # Fall back to equal weight
        opt_w = w0
    else:
        opt_w = result.x

    # Clip tiny negatives from numerical noise and re-normalise
    opt_w = np.maximum(opt_w, 0.0)
    wsum = opt_w.sum()
    if wsum > 0:
        opt_w = opt_w / wsum

    ret_opt, vol_opt, sharpe_opt = _portfolio_stats(opt_w, mu, cov, risk_free_rate)

    # --- Efficient frontier (20 points) ---
    frontier: List[Tuple[float, float]] = []
    min_ret = float(mu.min())
    max_ret = float(mu.max())
    if min_ret < max_ret:
        for tr in np.linspace(min_ret, max_ret, 20):
            cons_ef = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "eq", "fun": lambda w, _tr=tr: w @ mu - _tr},
            ]

            def obj_ef(w: np.ndarray) -> float:
                return float(w @ cov @ w)

            r_ef = _scipy_minimize(
                obj_ef,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=cons_ef,
                options={"maxiter": 500, "ftol": 1e-10},
            )
            if r_ef.success:
                v = float(np.sqrt(r_ef.x @ cov @ r_ef.x))
                frontier.append((v, float(tr)))

    weights_dict = {symbols[i]: float(opt_w[i]) for i in range(n)}

    return {
        "weights": weights_dict,
        "expected_return": ret_opt,
        "expected_volatility": vol_opt,
        "sharpe_ratio": sharpe_opt,
        "efficient_frontier": frontier,
    }


# ---------------------------------------------------------------------------
# 5. Risk-parity optimisation
# ---------------------------------------------------------------------------

def optimize_risk_parity(
    hist_cache: Dict[str, pd.DataFrame],
    max_weight: float = 0.15,
) -> dict:
    """Equal risk contribution (risk-parity) portfolio.

    Each position contributes equally to total portfolio variance.

    Parameters
    ----------
    hist_cache : Dict[str, pd.DataFrame]
        Ticker -> DataFrame with ``Close`` column.
    max_weight : float, optional
        Maximum weight for any single position.

    Returns
    -------
    dict
        ``{'weights', 'risk_contributions', 'portfolio_volatility'}``
    """
    empty_result: dict = {
        "weights": {},
        "risk_contributions": {},
        "portfolio_volatility": np.nan,
    }

    if not SCIPY_AVAILABLE:
        warnings.warn(
            "scipy is not installed -- risk-parity optimisation unavailable.",
            stacklevel=2,
        )
        return empty_result

    mu, cov, symbols = _annualised_return_and_cov(hist_cache)
    n = len(symbols)
    if n < 2 or mu.size == 0:
        return empty_result

    target_rc = 1.0 / n  # each asset's target share of total risk

    def _risk_contrib(w: np.ndarray) -> np.ndarray:
        """Marginal risk contribution of each asset."""
        port_var = w @ cov @ w
        if port_var < 1e-16:
            return np.zeros(n)
        port_vol = np.sqrt(port_var)
        marginal = cov @ w
        rc = w * marginal / port_vol
        return rc

    def objective(w: np.ndarray) -> float:
        rc = _risk_contrib(w)
        total_rc = rc.sum()
        if total_rc < 1e-16:
            return 1e6
        rc_pct = rc / total_rc
        # Sum of squared deviations from equal contribution
        return float(np.sum((rc_pct - target_rc) ** 2))

    bounds = tuple((1e-6, max_weight) for _ in range(n))
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    w0 = np.full(n, 1.0 / n)
    result = _scipy_minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-14},
    )

    if not result.success:
        opt_w = w0
    else:
        opt_w = result.x

    opt_w = np.maximum(opt_w, 0.0)
    wsum = opt_w.sum()
    if wsum > 0:
        opt_w = opt_w / wsum

    port_vol = float(np.sqrt(opt_w @ cov @ opt_w))
    rc = _risk_contrib(opt_w)
    total_rc = rc.sum()
    rc_pct = rc / total_rc if total_rc > 0 else np.zeros(n)

    weights_dict = {symbols[i]: float(opt_w[i]) for i in range(n)}
    rc_dict = {symbols[i]: float(rc_pct[i]) for i in range(n)}

    return {
        "weights": weights_dict,
        "risk_contributions": rc_dict,
        "portfolio_volatility": port_vol,
    }


# ---------------------------------------------------------------------------
# 6. Kelly criterion position sizing
# ---------------------------------------------------------------------------

def kelly_position_size(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    max_fraction: float = 0.25,
) -> float:
    """Kelly Criterion bet sizing, capped for safety.

    .. math::
        f^* = \\frac{p \\cdot b - q}{b}

    where *p* = win_rate, *q* = 1-p, *b* = avg_win / avg_loss.

    Parameters
    ----------
    win_rate : float
        Historical probability of a winning trade (0..1).
    avg_win : float
        Average profit on a winning trade (absolute value).
    avg_loss : float
        Average loss on a losing trade (absolute value, positive number).
    max_fraction : float, optional
        Hard cap on the Kelly fraction (quarter-Kelly is common practice).

    Returns
    -------
    float
        Recommended fraction of capital to allocate (0.0 if Kelly is
        non-positive, meaning the edge is absent or negative).
    """
    if avg_loss <= 0.0:
        # Cannot compute odds ratio -- return zero (no bet)
        return 0.0

    p = max(0.0, min(1.0, win_rate))
    q = 1.0 - p
    b = abs(avg_win) / abs(avg_loss)

    if b <= 0.0:
        return 0.0

    kelly = (p * b - q) / b

    if kelly <= 0.0:
        return 0.0

    return min(kelly, max_fraction)


# ---------------------------------------------------------------------------
# 7. Concentration risk detection
# ---------------------------------------------------------------------------

def detect_concentration_risk(
    portfolio_df: pd.DataFrame,
    metadata: Optional[Dict[str, dict]] = None,
) -> dict:
    """Analyse portfolio for excessive concentration.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Must contain columns ``Symbol`` and ``Market_Value``.
        May also contain ``metal``, ``country``, ``stage``; if missing these
        are looked up from *metadata* (keyed by symbol).
    metadata : dict or None, optional
        ``{symbol: {'metal': ..., 'country': ..., 'stage': ...}}``.

    Returns
    -------
    dict
        ``{'warnings': list[str],
           'concentration_score': float (0=diversified, 100=concentrated),
           'by_metal': dict, 'by_country': dict, 'by_stage': dict}``
    """
    result: dict = {
        "warnings": [],
        "concentration_score": 0.0,
        "by_metal": {},
        "by_country": {},
        "by_stage": {},
    }

    if portfolio_df is None or portfolio_df.empty:
        return result
    if "Symbol" not in portfolio_df.columns or "Market_Value" not in portfolio_df.columns:
        return result

    df = portfolio_df.copy()
    total_value = df["Market_Value"].sum()
    if total_value <= 0:
        return result

    metadata = metadata or {}

    # Ensure classification columns exist; fill from metadata when needed
    for col in ("metal", "country", "stage"):
        if col not in df.columns:
            df[col] = df["Symbol"].map(
                lambda s, _c=col: metadata.get(s, {}).get(_c, "Unknown")
            )
        else:
            df[col] = df[col].fillna(
                df["Symbol"].map(
                    lambda s, _c=col: metadata.get(s, {}).get(_c, "Unknown")
                )
            )

    warnings_list: List[str] = []
    penalty_points = 0.0  # accumulate toward concentration_score

    # --- Metal concentration ---
    by_metal: Dict[str, float] = {}
    for metal, group in df.groupby("metal"):
        pct = group["Market_Value"].sum() / total_value * 100.0
        by_metal[str(metal)] = round(pct, 2)
        if pct > 40.0:
            warnings_list.append(
                f"Metal concentration: {metal} is {pct:.1f}% of portfolio (threshold 40%)"
            )
            penalty_points += min((pct - 40.0) * 1.5, 30.0)

    # --- Country concentration ---
    by_country: Dict[str, float] = {}
    for country, group in df.groupby("country"):
        pct = group["Market_Value"].sum() / total_value * 100.0
        by_country[str(country)] = round(pct, 2)
        if pct > 50.0:
            warnings_list.append(
                f"Country concentration: {country} is {pct:.1f}% of portfolio (threshold 50%)"
            )
            penalty_points += min((pct - 50.0) * 1.5, 25.0)

    # --- Stage concentration ---
    by_stage: Dict[str, float] = {}
    for stage, group in df.groupby("stage"):
        pct = group["Market_Value"].sum() / total_value * 100.0
        by_stage[str(stage)] = round(pct, 2)
        if pct > 60.0:
            warnings_list.append(
                f"Stage concentration: {stage} is {pct:.1f}% of portfolio (threshold 60%)"
            )
            penalty_points += min((pct - 60.0) * 1.2, 20.0)

    # --- Correlated pair concentration ---
    # Build a quick weight map for pair checks
    weight_map: Dict[str, float] = {}
    for _, row in df.iterrows():
        sym = row["Symbol"]
        w = row["Market_Value"] / total_value
        weight_map[sym] = weight_map.get(sym, 0.0) + w

    symbols = sorted(weight_map.keys())
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            combined = (weight_map[symbols[i]] + weight_map[symbols[j]]) * 100.0
            if combined > 20.0:
                # Note: correlation data not directly available here; flag
                # any pair whose combined weight breaches threshold.
                # Callers can pair this with calculate_correlation_matrix().
                warnings_list.append(
                    f"Pair weight: {symbols[i]} + {symbols[j]} = {combined:.1f}% "
                    f"(threshold 20% combined when correlation > 0.8)"
                )
                penalty_points += min((combined - 20.0) * 0.5, 15.0)

    # Clamp score between 0 and 100
    concentration_score = min(max(penalty_points, 0.0), 100.0)

    result["warnings"] = warnings_list
    result["concentration_score"] = round(concentration_score, 1)
    result["by_metal"] = by_metal
    result["by_country"] = by_country
    result["by_stage"] = by_stage

    return result


# ---------------------------------------------------------------------------
# 8. Rebalance trade suggestions
# ---------------------------------------------------------------------------

def suggest_rebalance_trades(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    portfolio_value: float,
    min_trade_pct: float = 0.5,
) -> List[dict]:
    """Generate a list of trades to move from current to target weights.

    Parameters
    ----------
    current_weights : Dict[str, float]
        ``{symbol: weight}`` summing to ~1.0.
    target_weights : Dict[str, float]
        Desired ``{symbol: weight}`` summing to ~1.0.
    portfolio_value : float
        Total portfolio value in dollars.
    min_trade_pct : float, optional
        Minimum trade size as a percentage of portfolio to include in output.
        Smaller trades are filtered out to avoid excessive transaction costs.

    Returns
    -------
    List[dict]
        Each entry: ``{'symbol', 'action', 'current_weight', 'target_weight',
        'trade_value', 'trade_shares'}``
        (``trade_shares`` is an estimate assuming the most recent close price
        is not available here, so it is set based on a $10 placeholder or
        the caller can refine).
    """
    all_symbols = sorted(set(list(current_weights.keys()) + list(target_weights.keys())))
    trades: List[dict] = []

    for sym in all_symbols:
        cw = current_weights.get(sym, 0.0)
        tw = target_weights.get(sym, 0.0)
        diff = tw - cw
        diff_pct = abs(diff) * 100.0

        if diff_pct < min_trade_pct:
            continue

        trade_value = abs(diff) * portfolio_value

        # Estimate shares (placeholder at $10/share if no price available)
        estimated_shares = max(1, int(round(trade_value / 10.0)))

        trades.append({
            "symbol": sym,
            "action": "BUY" if diff > 0 else "SELL",
            "current_weight": round(cw, 6),
            "target_weight": round(tw, 6),
            "trade_value": round(trade_value, 2),
            "trade_shares": estimated_shares,
        })

    # Sort: largest trades first
    trades.sort(key=lambda t: t["trade_value"], reverse=True)
    return trades
