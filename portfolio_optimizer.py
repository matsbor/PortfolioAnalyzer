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
        # Fall back to equal weight — log the failure for transparency
        opt_w = w0
        import warnings
        warnings.warn(
            f"Mean-variance optimization failed ({result.message}); "
            f"falling back to equal-weight allocation.",
            stacklevel=2,
        )
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
    fractional: float = 0.5,
    num_positions: int = 1,
) -> float:
    """Kelly Criterion bet sizing with fractional Kelly and portfolio awareness.

    Uses fractional Kelly (default half-Kelly) to reduce risk of ruin from
    estimation error in win_rate and payoff ratio.  Also scales down when
    many positions are held simultaneously to prevent over-leverage.

    .. math::
        f^* = \\text{fractional} \\times \\frac{p \\cdot b - q}{b}

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
    fractional : float, optional
        Fraction of full Kelly to use (0.5 = half-Kelly, reduces ruin risk).
    num_positions : int, optional
        Number of concurrent positions.  Kelly is further scaled by
        ``1 / max(num_positions, 1)`` to prevent combined leverage > 100%.

    Returns
    -------
    float
        Recommended fraction of capital to allocate (0.0 if Kelly is
        non-positive, meaning the edge is absent or negative).
    """
    if avg_loss <= 0.0:
        return 0.0

    p = max(0.0, min(1.0, win_rate))
    q = 1.0 - p
    b = abs(avg_win) / abs(avg_loss)

    if b <= 0.0:
        return 0.0

    kelly = (p * b - q) / b

    if kelly <= 0.0:
        return 0.0

    # Apply fractional Kelly and portfolio scaling
    kelly *= fractional
    kelly /= max(num_positions, 1)

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
    prices: Optional[Dict[str, float]] = None,
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

        # Estimate shares from actual price if available, else $10/share fallback
        share_price = (prices or {}).get(sym, 10.0) or 10.0
        estimated_shares = max(1, int(round(trade_value / share_price)))

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


# ---------------------------------------------------------------------------
# 9. Discovery-aware rebalancing
# ---------------------------------------------------------------------------

def calculate_portfolio_health_score(
    hist_cache: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
) -> dict:
    """Calculate a composite portfolio health score.

    Evaluates diversification, risk concentration, and return efficiency.

    Parameters
    ----------
    hist_cache : Dict[str, pd.DataFrame]
        Ticker -> DataFrame with ``Close`` column.
    weights : Dict[str, float]
        Current portfolio weights.

    Returns
    -------
    dict
        ``{'health_score': float (0-100), 'diversification': float,
           'return_efficiency': float, 'risk_score': float, 'details': list[str]}``
    """
    result = {
        'health_score': 50.0,
        'diversification': 50.0,
        'return_efficiency': 50.0,
        'risk_score': 50.0,
        'details': [],
    }

    if not hist_cache or not weights:
        return result

    # Diversification: based on number of positions and weight distribution
    n_positions = len([w for w in weights.values() if w > 0.01])
    max_weight = max(weights.values()) if weights else 0
    hhi = sum(w ** 2 for w in weights.values())  # Herfindahl-Hirschman Index

    # HHI of 1/n = perfectly diversified; HHI of 1.0 = single stock
    diversification = max(0, min(100, (1 - hhi) * 100))
    if n_positions < 5:
        diversification *= 0.7
    if max_weight > 0.20:
        diversification *= 0.85

    result['diversification'] = round(diversification, 1)

    # Return efficiency: portfolio Sharpe-like metric
    close_map = {}
    for sym, df in hist_cache.items():
        if df is not None and not df.empty and 'Close' in df.columns and sym in weights:
            s = df['Close'].dropna()
            if len(s) >= 30:
                s.index = pd.to_datetime(s.index)
                close_map[sym] = s

    if len(close_map) >= 2:
        symbols = sorted(close_map.keys())
        aligned = pd.DataFrame({s: close_map[s] for s in symbols}).dropna()
        if len(aligned) >= 30:
            daily_ret = aligned.pct_change().dropna()
            w_arr = np.array([weights.get(s, 0) for s in symbols])
            w_sum = w_arr.sum()
            if w_sum > 0:
                w_arr = w_arr / w_sum
                port_ret = daily_ret.values @ w_arr
                mean_r = float(np.mean(port_ret)) * 252
                std_r = float(np.std(port_ret)) * np.sqrt(252)
                if std_r > 0:
                    sharpe = mean_r / std_r
                    result['return_efficiency'] = round(
                        max(0, min(100, 50 + sharpe * 20)), 1
                    )

    # Risk score: drawdown-based
    risk_score = 70.0  # Default moderate
    if close_map:
        max_dds = []
        for sym, s in close_map.items():
            if len(s) >= 30:
                window = s.tail(90)
                cummax = window.cummax()
                dd = ((window - cummax) / cummax).min()
                max_dds.append(abs(float(dd)) * 100)
        if max_dds:
            avg_dd = np.mean(max_dds)
            if avg_dd > 40:
                risk_score = 30.0
            elif avg_dd > 25:
                risk_score = 50.0
            elif avg_dd > 15:
                risk_score = 65.0
            else:
                risk_score = 85.0

    result['risk_score'] = round(risk_score, 1)

    # Composite
    result['health_score'] = round(
        diversification * 0.3 + result['return_efficiency'] * 0.4 + risk_score * 0.3,
        1,
    )

    result['details'] = [
        f"Diversification: {diversification:.0f}/100 (HHI={hhi:.3f}, {n_positions} positions)",
        f"Return Efficiency: {result['return_efficiency']:.0f}/100",
        f"Risk Score: {risk_score:.0f}/100",
    ]

    return result


def integrate_discovery_into_rebalance(
    current_weights: Dict[str, float],
    swap_recommendations: List[dict],
    portfolio_value: float,
    prices: Optional[Dict[str, float]] = None,
    max_new_position_pct: float = 0.05,
) -> List[dict]:
    """Generate trade list that incorporates discovery swap recommendations.

    Takes the swap recommendations from the discovery engine and converts
    them into concrete rebalance trades (sell existing + buy new).

    Parameters
    ----------
    current_weights : Dict[str, float]
        Current portfolio weights.
    swap_recommendations : List[dict]
        Output of discovery_engine.generate_swap_recommendations().
    portfolio_value : float
        Total portfolio value.
    prices : Dict[str, float], optional
        Current prices for share estimation.
    max_new_position_pct : float
        Maximum weight for any new position.

    Returns
    -------
    List[dict]
        Trades in the same format as suggest_rebalance_trades().
    """
    target_weights = dict(current_weights)
    prices = prices or {}

    for swap in swap_recommendations:
        sell_sym = swap.get('sell_symbol', '')
        buy_sym = swap.get('buy_symbol', '')

        if not sell_sym or not buy_sym:
            continue

        # Free up weight from the sell
        sell_weight = target_weights.get(sell_sym, 0)
        freed_weight = sell_weight * 0.5  # Sell half the position

        # Confidence-scaled allocation
        confidence = swap.get('confidence', 'Low')
        if confidence == 'High':
            alloc_factor = 1.0
        elif confidence == 'Medium':
            alloc_factor = 0.7
        else:
            alloc_factor = 0.4

        new_weight = min(freed_weight * alloc_factor, max_new_position_pct)

        target_weights[sell_sym] = sell_weight - freed_weight
        target_weights[buy_sym] = target_weights.get(buy_sym, 0) + new_weight

    # Normalize if weights exceed 1.0
    total = sum(target_weights.values())
    if total > 1.0:
        target_weights = {k: v / total for k, v in target_weights.items()}

    return suggest_rebalance_trades(
        current_weights, target_weights, portfolio_value,
        min_trade_pct=0.3, prices=prices,
    )
