#!/usr/bin/env python3
"""
Discovery Engine - Find, score, and rank candidate stocks against your portfolio.

Scans the mining universe beyond your current holdings, scores each candidate
using the same 11-model alpha system, and produces clear swap recommendations:
  "Sell X (weakest), Buy Y (strongest candidate) — expected +N alpha improvement"

The engine also runs a quick historical validation (lookback backtest) on each
candidate to verify its signal quality before recommending a swap.

Usage (standalone):
    from discovery_engine import run_discovery
    report = run_discovery(portfolio_df, hist_cache, risk_mode='BALANCED')

Usage (integrated with Streamlit / backtest):
    from discovery_engine import (
        scan_universe,
        score_candidates,
        rank_candidates_vs_portfolio,
        generate_swap_recommendations,
        build_discovery_report,
    )
"""
from __future__ import annotations

import datetime
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Import from project modules (import-safe, no Streamlit)
try:
    from alpha_miner_core import (
        calculate_alpha_models,
        calculate_sell_risk,
        calculate_liquidity_metrics,
        calculate_data_confidence,
        calculate_dilution_risk,
        calculate_financing_overhang,
        arbitrate_final_decision,
        calculate_macro_regime,
        calculate_tape_gate,
        get_benchmark_data,
        RISK_PROFILES,
    )
    _CORE_AVAILABLE = True
except ImportError:
    _CORE_AVAILABLE = False

try:
    from mining_tickers import get_all_mining_tickers
    _TICKERS_AVAILABLE = True
except ImportError:
    _TICKERS_AVAILABLE = False

try:
    from technical_analysis import (
        calculate_rsi,
        calculate_macd,
        calculate_bollinger_bands,
        calculate_obv,
        calculate_adx,
    )
    _TA_AVAILABLE = True
except ImportError:
    _TA_AVAILABLE = False

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False
    yf = None


# ---------------------------------------------------------------------------
# 1. Universe scanning
# ---------------------------------------------------------------------------

def scan_universe(
    portfolio_symbols: List[str],
    max_candidates: int = 80,
    include_portfolio: bool = False,
) -> List[str]:
    """Return candidate symbols from the mining universe that are NOT in portfolio.

    Combines curated miner list + Tiingo supplement, then removes symbols
    already held in the portfolio (unless include_portfolio=True).

    Parameters
    ----------
    portfolio_symbols : List[str]
        Symbols currently in the portfolio.
    max_candidates : int
        Maximum number of candidates to return.
    include_portfolio : bool
        If True, include portfolio symbols in the scan (for re-scoring).

    Returns
    -------
    List[str]
        Candidate ticker symbols to evaluate.
    """
    if not _TICKERS_AVAILABLE:
        return []

    all_tickers = get_all_mining_tickers(max_symbols=200)

    held = {s.upper().strip() for s in portfolio_symbols}
    if include_portfolio:
        candidates = all_tickers
    else:
        candidates = [t for t in all_tickers if t.upper().strip() not in held]

    return candidates[:max_candidates]


# ---------------------------------------------------------------------------
# 2. Fetch candidate data
# ---------------------------------------------------------------------------

def fetch_candidate_data(
    symbols: List[str],
    period: str = "6mo",
    existing_cache: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV history for candidate symbols.

    Uses existing cache when available, otherwise fetches via yfinance.

    Parameters
    ----------
    symbols : List[str]
        Candidate symbols to fetch.
    period : str
        History period (default "6mo").
    existing_cache : dict, optional
        Pre-fetched price data to reuse.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Symbol -> OHLCV DataFrame.
    """
    cache = dict(existing_cache or {})
    missing = [s for s in symbols if s not in cache or cache[s] is None or cache[s].empty]

    if missing and _YF_AVAILABLE:
        for sym in missing:
            try:
                hist = yf.Ticker(sym).history(period=period)
                if hist is not None and not hist.empty and len(hist) >= 20:
                    cache[sym] = hist
            except Exception:
                pass

    return cache


# ---------------------------------------------------------------------------
# 3. Score candidates using the alpha model
# ---------------------------------------------------------------------------

def _build_candidate_row(symbol: str, hist: pd.DataFrame) -> dict:
    """Build a minimal row dict for calculate_alpha_models from price history."""
    if hist is None or hist.empty or len(hist) < 20:
        return {}

    close = hist['Close'].dropna()
    if close.empty:
        return {}

    price = float(close.iloc[-1])
    high_52w = float(close.max())
    pct_from_high = ((price - high_52w) / high_52w * 100) if high_52w > 0 else 0

    ret_30d = 0.0
    ret_90d = 0.0
    ret_7d = 0.0
    if len(close) >= 30:
        ret_30d = (price / float(close.iloc[-30]) - 1) * 100
    if len(close) >= 90:
        ret_90d = (price / float(close.iloc[-90]) - 1) * 100
    if len(close) >= 7:
        ret_7d = (price / float(close.iloc[-7]) - 1) * 100

    # MA calculations
    ma50 = float(close.tail(50).mean()) if len(close) >= 50 else price
    ma200 = float(close.tail(200).mean()) if len(close) >= 200 else ma50

    # Volatility (60d standard deviation of returns)
    vol_60d = 0.0
    if len(close) >= 60:
        daily_ret = close.tail(60).pct_change().dropna()
        vol_60d = float(daily_ret.std() * np.sqrt(252) * 100)

    # Average daily volume
    avg_vol = 0
    if 'Volume' in hist.columns:
        avg_vol = int(hist['Volume'].tail(20).mean())

    # Liquidity tier from volume
    if avg_vol >= 500_000:
        liq_tier = 'L3'
    elif avg_vol >= 100_000:
        liq_tier = 'L2'
    elif avg_vol >= 10_000:
        liq_tier = 'L1'
    else:
        liq_tier = 'L0'

    # Drawdown from 90d high
    drawdown_90d = 0.0
    if len(close) >= 90:
        high_90 = float(close.tail(90).max())
        if high_90 > 0:
            drawdown_90d = (price - high_90) / high_90 * 100

    row = {
        'Symbol': symbol,
        'Price': price,
        'Pct_From_52w_High': pct_from_high,
        'Return_7d': ret_7d,
        'Return_30d': ret_30d,
        'Return_90d': ret_90d,
        'MA50': ma50,
        'MA200': ma200,
        'Volatility_60d': vol_60d,
        'Volume_Avg': avg_vol,
        'Liq_tier_code': liq_tier,
        'Runway': 24,  # Default — no balance sheet data for new candidates
        'Dilution_Risk_Score': 30,  # Default moderate
        'Drawdown_90d': drawdown_90d,
        'Pct_Portfolio': 0,  # Not in portfolio
        'stage': 'Explorer',  # Default
        'metal': 'Gold',  # Default
    }

    return row


def score_candidates(
    candidate_symbols: List[str],
    hist_cache: Dict[str, pd.DataFrame],
    benchmark_data: Optional[pd.DataFrame] = None,
    macro_regime: Optional[dict] = None,
    risk_mode: str = 'BALANCED',
) -> List[dict]:
    """Score each candidate through the alpha model pipeline.

    Parameters
    ----------
    candidate_symbols : List[str]
        Symbols to score.
    hist_cache : Dict[str, pd.DataFrame]
        Price histories.
    benchmark_data : pd.DataFrame, optional
        Benchmark (GDXJ/SILJ) for relative strength.
    macro_regime : dict, optional
        Current macro regime dict.
    risk_mode : str
        Risk profile name.

    Returns
    -------
    List[dict]
        Scored candidates sorted by alpha_score descending. Each dict contains:
        symbol, alpha_score, models (M1-M11 breakdown), action, confidence,
        momentum_30d, momentum_90d, rsi, volatility, liq_tier, trend_quality, ...
    """
    if not _CORE_AVAILABLE:
        return []

    if macro_regime is None:
        macro_regime = {'regime': 'NEUTRAL', 'throttle_factor': 1.0, 'allow_new_buys': True}

    scored = []
    for sym in candidate_symbols:
        hist = hist_cache.get(sym)
        if hist is None or hist.empty or len(hist) < 20:
            continue

        row = _build_candidate_row(sym, hist)
        if not row:
            continue

        # Calculate alpha models
        try:
            alpha_result = calculate_alpha_models(row, hist, benchmark_data)
        except Exception:
            continue

        alpha_score = alpha_result.get('alpha_score', 0)
        models = alpha_result.get('models', {})

        # Calculate trend quality score (proprietary composite)
        trend_quality = _calculate_trend_quality(hist)

        # Calculate signal freshness (recency of momentum breakout)
        signal_freshness = _calculate_signal_freshness(hist)

        # Calculate risk-adjusted score
        vol = row.get('Volatility_60d', 30)
        risk_adjusted_alpha = alpha_score / max(vol / 30, 0.5) if vol > 0 else alpha_score

        scored.append({
            'symbol': sym,
            'alpha_score': round(alpha_score, 1),
            'risk_adjusted_alpha': round(risk_adjusted_alpha, 1),
            'models': models,
            'breakdown': alpha_result.get('breakdown', []),
            'price': row['Price'],
            'momentum_30d': round(row['Return_30d'], 1),
            'momentum_90d': round(row['Return_90d'], 1),
            'volatility': round(vol, 1),
            'liq_tier': row['Liq_tier_code'],
            'pct_from_52w_high': round(row['Pct_From_52w_High'], 1),
            'avg_volume': row['Volume_Avg'],
            'trend_quality': round(trend_quality, 1),
            'signal_freshness': round(signal_freshness, 1),
            'drawdown_90d': round(row.get('Drawdown_90d', 0), 1),
        })

    # Sort by alpha score descending
    scored.sort(key=lambda x: x['alpha_score'], reverse=True)
    return scored


def _calculate_trend_quality(hist: pd.DataFrame) -> float:
    """Calculate trend quality: consistency of price direction.

    Measures what fraction of rolling 5-day windows show positive returns,
    weighted toward recent periods. Range: 0 (persistent downtrend) to
    100 (persistent uptrend).
    """
    if hist is None or hist.empty or 'Close' not in hist.columns:
        return 50.0

    close = hist['Close'].dropna()
    if len(close) < 20:
        return 50.0

    # Use last 60 trading days (or all available)
    window = close.tail(min(60, len(close)))
    n = len(window)

    # Rolling 5-day returns
    returns_5d = window.pct_change(5).dropna()
    if returns_5d.empty:
        return 50.0

    # Weight recent periods more heavily (exponential decay)
    weights = np.exp(np.linspace(-1, 0, len(returns_5d)))
    weights /= weights.sum()

    positive = (returns_5d > 0).astype(float).values
    weighted_positive_rate = float(np.dot(positive, weights))

    # Price relative to moving averages (trend alignment)
    ma20 = float(close.tail(20).mean()) if len(close) >= 20 else float(close.mean())
    ma50 = float(close.tail(50).mean()) if len(close) >= 50 else ma20
    cur = float(close.iloc[-1])

    alignment_bonus = 0.0
    if cur > ma20 > ma50:
        alignment_bonus = 15.0  # Perfect bullish alignment
    elif cur > ma20:
        alignment_bonus = 8.0
    elif cur < ma20 < ma50:
        alignment_bonus = -15.0  # Perfect bearish alignment
    elif cur < ma20:
        alignment_bonus = -8.0

    # Higher-highs / higher-lows check
    hh_hl_bonus = 0.0
    if len(close) >= 40:
        mid = len(close) // 2
        first_half = close.iloc[-40:-20]
        second_half = close.iloc[-20:]
        if not first_half.empty and not second_half.empty:
            if second_half.max() > first_half.max() and second_half.min() > first_half.min():
                hh_hl_bonus = 10.0
            elif second_half.max() < first_half.max() and second_half.min() < first_half.min():
                hh_hl_bonus = -10.0

    score = weighted_positive_rate * 70 + alignment_bonus + hh_hl_bonus + 15
    return max(0, min(100, score))


def _calculate_signal_freshness(hist: pd.DataFrame) -> float:
    """Measure recency of a bullish momentum signal.

    Checks how recently the stock crossed above its 20-day MA, and
    whether volume confirmed the move. Range: 0 (stale/no signal) to
    100 (fresh breakout confirmed by volume).
    """
    if hist is None or hist.empty or 'Close' not in hist.columns:
        return 0.0

    close = hist['Close'].dropna()
    if len(close) < 25:
        return 0.0

    ma20 = close.rolling(20).mean()
    above_ma = close > ma20

    # Find the most recent crossover from below to above
    crossovers = above_ma.astype(int).diff()
    bullish_cross = crossovers[crossovers == 1]

    if bullish_cross.empty:
        # No recent bullish crossover — check if currently above MA
        if close.iloc[-1] > ma20.iloc[-1]:
            return 30.0  # Sustained above MA, but stale signal
        return 0.0

    # Days since last bullish crossover
    last_cross_idx = bullish_cross.index[-1]
    try:
        days_since = (close.index[-1] - last_cross_idx).days
    except (TypeError, AttributeError):
        # If index isn't datetime, count positions
        last_pos = close.index.get_loc(last_cross_idx)
        days_since = len(close) - 1 - last_pos

    # Freshness decays exponentially
    freshness = 100 * np.exp(-days_since / 15)

    # Volume confirmation bonus
    vol_bonus = 0.0
    if 'Volume' in hist.columns and len(hist) >= 5:
        recent_vol = hist['Volume'].tail(5).mean()
        avg_vol = hist['Volume'].tail(20).mean()
        if avg_vol > 0 and recent_vol > avg_vol * 1.3:
            vol_bonus = 15.0  # Volume surge confirms the move

    return max(0, min(100, freshness + vol_bonus))


# ---------------------------------------------------------------------------
# 4. Lookback validation (mini-backtest)
# ---------------------------------------------------------------------------

def validate_candidate_lookback(
    symbol: str,
    hist: pd.DataFrame,
    lookback_days: int = 90,
) -> dict:
    """Run a quick lookback validation on a candidate.

    Simulates: "If we had bought this stock N days ago, what would
    the result be?" Also measures win-rate on rolling 20-day holds.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    hist : pd.DataFrame
        OHLCV price history (at least lookback_days rows).
    lookback_days : int
        How far back to validate.

    Returns
    -------
    dict
        {symbol, lookback_return, win_rate_20d, max_drawdown,
         sharpe_ratio, consistency_score, passed_validation}
    """
    result = {
        'symbol': symbol,
        'lookback_return': 0.0,
        'win_rate_20d': 0.0,
        'max_drawdown': 0.0,
        'sharpe_ratio': 0.0,
        'consistency_score': 0.0,
        'passed_validation': False,
        'validation_detail': '',
    }

    if hist is None or hist.empty or 'Close' not in hist.columns:
        result['validation_detail'] = 'No price data'
        return result

    close = hist['Close'].dropna()
    if len(close) < lookback_days:
        result['validation_detail'] = f'Insufficient data ({len(close)} < {lookback_days} days)'
        return result

    window = close.tail(lookback_days)

    # Lookback return
    lookback_ret = (float(window.iloc[-1]) / float(window.iloc[0]) - 1) * 100
    result['lookback_return'] = round(lookback_ret, 2)

    # Rolling 20-day hold win rate
    daily_ret = window.pct_change().dropna()
    if len(daily_ret) >= 20:
        rolling_20d = daily_ret.rolling(20).sum()
        valid_rolls = rolling_20d.dropna()
        if len(valid_rolls) > 0:
            win_rate = float((valid_rolls > 0).sum()) / len(valid_rolls) * 100
            result['win_rate_20d'] = round(win_rate, 1)

    # Max drawdown
    cummax = window.cummax()
    drawdown = (window - cummax) / cummax * 100
    result['max_drawdown'] = round(float(drawdown.min()), 2)

    # Sharpe ratio (annualized)
    if len(daily_ret) >= 20:
        mean_ret = float(daily_ret.mean())
        std_ret = float(daily_ret.std())
        if std_ret > 0:
            result['sharpe_ratio'] = round(mean_ret / std_ret * np.sqrt(252), 2)

    # Consistency score: measures how smooth the returns are
    # Penalizes choppy / whipsaw behavior
    if len(daily_ret) >= 20:
        # Autocorrelation of returns (positive = trending, negative = mean-reverting)
        autocorr = float(daily_ret.autocorr(lag=1)) if len(daily_ret) > 1 else 0
        # Fraction of days in the direction of overall trend
        trend_dir = 1 if lookback_ret > 0 else -1
        aligned_days = float(((daily_ret * trend_dir) > 0).sum()) / len(daily_ret)
        result['consistency_score'] = round(
            max(0, min(100, aligned_days * 70 + max(autocorr, 0) * 30)), 1
        )

    # Validation pass/fail
    passes = []
    fails = []

    if result['sharpe_ratio'] >= 0.3:
        passes.append(f"Sharpe {result['sharpe_ratio']:.2f} >= 0.3")
    else:
        fails.append(f"Sharpe {result['sharpe_ratio']:.2f} < 0.3")

    if result['max_drawdown'] >= -40:
        passes.append(f"MaxDD {result['max_drawdown']:.1f}% >= -40%")
    else:
        fails.append(f"MaxDD {result['max_drawdown']:.1f}% < -40%")

    if result['win_rate_20d'] >= 40:
        passes.append(f"WinRate {result['win_rate_20d']:.0f}% >= 40%")
    else:
        fails.append(f"WinRate {result['win_rate_20d']:.0f}% < 40%")

    result['passed_validation'] = len(fails) == 0
    detail_parts = []
    if passes:
        detail_parts.append("PASS: " + ", ".join(passes))
    if fails:
        detail_parts.append("FAIL: " + ", ".join(fails))
    result['validation_detail'] = " | ".join(detail_parts)

    return result


# ---------------------------------------------------------------------------
# 5. Rank candidates vs portfolio positions
# ---------------------------------------------------------------------------

def rank_candidates_vs_portfolio(
    candidates: List[dict],
    portfolio_scores: List[dict],
) -> List[dict]:
    """Compare scored candidates against the weakest portfolio positions.

    For each candidate, calculates:
    - alpha_improvement: candidate alpha - weakest holding alpha
    - risk_adjusted_improvement: risk-adj alpha comparison
    - upgrade_grade: A/B/C/D letter grade for the potential improvement

    Parameters
    ----------
    candidates : List[dict]
        Scored candidates (output of score_candidates).
    portfolio_scores : List[dict]
        Scored current holdings (same format as candidates).

    Returns
    -------
    List[dict]
        Candidates enriched with comparison metrics, sorted by improvement.
    """
    if not portfolio_scores:
        # No portfolio to compare against — return candidates as-is
        for c in candidates:
            c['alpha_improvement'] = 0
            c['weakest_holding'] = 'N/A'
            c['upgrade_grade'] = 'N/A'
        return candidates

    # Sort portfolio by alpha ascending (weakest first)
    portfolio_sorted = sorted(portfolio_scores, key=lambda x: x.get('alpha_score', 0))

    # Take bottom 25% of portfolio as potential sell candidates
    n_weak = max(1, len(portfolio_sorted) // 4)
    weak_holdings = portfolio_sorted[:n_weak]
    avg_weak_alpha = np.mean([h.get('alpha_score', 0) for h in weak_holdings])

    for c in candidates:
        c_alpha = c.get('alpha_score', 0)
        improvement = c_alpha - avg_weak_alpha
        c['alpha_improvement'] = round(improvement, 1)
        c['weakest_holding'] = weak_holdings[0].get('symbol', 'N/A')
        c['weakest_holding_alpha'] = round(weak_holdings[0].get('alpha_score', 0), 1)

        # Grade the improvement
        if improvement >= 20:
            c['upgrade_grade'] = 'A'
        elif improvement >= 10:
            c['upgrade_grade'] = 'B'
        elif improvement >= 5:
            c['upgrade_grade'] = 'C'
        else:
            c['upgrade_grade'] = 'D'

    # Sort by improvement descending
    candidates.sort(key=lambda x: x.get('alpha_improvement', 0), reverse=True)
    return candidates


# ---------------------------------------------------------------------------
# 6. Generate swap recommendations
# ---------------------------------------------------------------------------

def generate_swap_recommendations(
    ranked_candidates: List[dict],
    portfolio_scores: List[dict],
    max_swaps: int = 5,
    min_improvement: float = 5.0,
    require_validation: bool = True,
) -> List[dict]:
    """Generate specific "Sell X, Buy Y" swap recommendations.

    Parameters
    ----------
    ranked_candidates : List[dict]
        Candidates ranked vs portfolio (output of rank_candidates_vs_portfolio).
    portfolio_scores : List[dict]
        Scored current holdings.
    max_swaps : int
        Maximum number of swaps to recommend.
    min_improvement : float
        Minimum alpha improvement to recommend a swap.
    require_validation : bool
        If True, only recommend candidates that passed lookback validation.

    Returns
    -------
    List[dict]
        Each swap: {sell_symbol, sell_alpha, buy_symbol, buy_alpha,
        alpha_improvement, upgrade_grade, reasoning, confidence}
    """
    if not portfolio_scores or not ranked_candidates:
        return []

    # Sort portfolio by alpha ascending (weakest first)
    portfolio_sorted = sorted(portfolio_scores, key=lambda x: x.get('alpha_score', 0))

    # Track which portfolio positions have been "swapped out"
    used_sells = set()
    swaps = []

    for candidate in ranked_candidates:
        if len(swaps) >= max_swaps:
            break

        if candidate.get('alpha_improvement', 0) < min_improvement:
            continue

        if require_validation and not candidate.get('passed_validation', False):
            continue

        # Skip illiquid candidates
        if candidate.get('liq_tier') == 'L0':
            continue

        # Find the weakest portfolio position not yet used
        sell_target = None
        for p in portfolio_sorted:
            p_sym = p.get('symbol', '')
            if p_sym not in used_sells and p.get('alpha_score', 0) < candidate.get('alpha_score', 0):
                sell_target = p
                break

        if sell_target is None:
            continue

        sell_sym = sell_target.get('symbol', '')
        used_sells.add(sell_sym)

        improvement = candidate['alpha_score'] - sell_target.get('alpha_score', 0)

        # Confidence based on multiple factors
        confidence_factors = 0
        if candidate.get('passed_validation', False):
            confidence_factors += 1
        if candidate.get('trend_quality', 0) >= 60:
            confidence_factors += 1
        if candidate.get('signal_freshness', 0) >= 30:
            confidence_factors += 1
        if candidate.get('liq_tier') in ('L3', 'L2'):
            confidence_factors += 1
        if improvement >= 15:
            confidence_factors += 1

        if confidence_factors >= 4:
            confidence = 'High'
        elif confidence_factors >= 2:
            confidence = 'Medium'
        else:
            confidence = 'Low'

        # Build reasoning
        reasoning = []
        reasoning.append(
            f"Alpha improvement: {candidate['alpha_score']:.0f} vs {sell_target.get('alpha_score', 0):.0f} "
            f"(+{improvement:.0f} points)"
        )
        if candidate.get('momentum_30d', 0) > 5:
            reasoning.append(f"Strong 30d momentum: +{candidate['momentum_30d']:.1f}%")
        if candidate.get('trend_quality', 0) >= 65:
            reasoning.append(f"High trend quality: {candidate['trend_quality']:.0f}/100")
        if candidate.get('signal_freshness', 0) >= 50:
            reasoning.append(f"Fresh breakout signal: {candidate['signal_freshness']:.0f}/100")
        if candidate.get('sharpe_ratio', 0) >= 0.5:
            reasoning.append(f"Good risk-adjusted returns: Sharpe {candidate.get('sharpe_ratio', 0):.2f}")
        if sell_target.get('alpha_score', 0) < 40:
            reasoning.append(f"Sell target weak: alpha only {sell_target.get('alpha_score', 0):.0f}/100")

        swaps.append({
            'sell_symbol': sell_sym,
            'sell_alpha': round(sell_target.get('alpha_score', 0), 1),
            'sell_price': sell_target.get('price', 0),
            'buy_symbol': candidate['symbol'],
            'buy_alpha': round(candidate['alpha_score'], 1),
            'buy_price': candidate.get('price', 0),
            'alpha_improvement': round(improvement, 1),
            'risk_adjusted_improvement': round(
                candidate.get('risk_adjusted_alpha', 0) - sell_target.get('risk_adjusted_alpha', 0), 1
            ),
            'upgrade_grade': candidate.get('upgrade_grade', 'D'),
            'confidence': confidence,
            'reasoning': reasoning,
            'candidate_detail': {
                'momentum_30d': candidate.get('momentum_30d', 0),
                'momentum_90d': candidate.get('momentum_90d', 0),
                'volatility': candidate.get('volatility', 0),
                'trend_quality': candidate.get('trend_quality', 0),
                'signal_freshness': candidate.get('signal_freshness', 0),
                'lookback_return': candidate.get('lookback_return', 0),
                'win_rate_20d': candidate.get('win_rate_20d', 0),
                'max_drawdown': candidate.get('max_drawdown', 0),
                'sharpe_ratio': candidate.get('sharpe_ratio', 0),
                'liq_tier': candidate.get('liq_tier', 'UNKNOWN'),
            },
        })

    return swaps


# ---------------------------------------------------------------------------
# 7. Build discovery report
# ---------------------------------------------------------------------------

def build_discovery_report(
    candidates: List[dict],
    swaps: List[dict],
    portfolio_scores: List[dict],
    macro_regime: Optional[dict] = None,
) -> dict:
    """Build a comprehensive discovery report.

    Parameters
    ----------
    candidates : List[dict]
        All scored candidates.
    swaps : List[dict]
        Swap recommendations.
    portfolio_scores : List[dict]
        Current portfolio scores.
    macro_regime : dict, optional
        Current macro regime.

    Returns
    -------
    dict
        Complete report with summary, top_candidates, swap_recommendations,
        portfolio_weaknesses, and statistics.
    """
    regime = (macro_regime or {}).get('regime', 'NEUTRAL')
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Portfolio statistics
    portfolio_alphas = [p.get('alpha_score', 0) for p in portfolio_scores]
    portfolio_avg_alpha = float(np.mean(portfolio_alphas)) if portfolio_alphas else 0
    portfolio_min_alpha = float(np.min(portfolio_alphas)) if portfolio_alphas else 0

    # Candidate statistics
    candidate_alphas = [c.get('alpha_score', 0) for c in candidates]
    candidates_above_avg = [c for c in candidates if c.get('alpha_score', 0) > portfolio_avg_alpha]
    validated = [c for c in candidates if c.get('passed_validation', False)]

    # Identify portfolio weaknesses
    weak_positions = sorted(portfolio_scores, key=lambda x: x.get('alpha_score', 0))[:5]
    weaknesses = []
    for w in weak_positions:
        weaknesses.append({
            'symbol': w.get('symbol', ''),
            'alpha_score': round(w.get('alpha_score', 0), 1),
            'issue': _diagnose_weakness(w),
        })

    # Summary verdict
    if len(swaps) >= 3 and swaps[0].get('alpha_improvement', 0) >= 15:
        verdict = "REBALANCE_RECOMMENDED"
        verdict_detail = (
            f"Found {len(swaps)} high-quality swap opportunities. "
            f"Top swap improves alpha by +{swaps[0]['alpha_improvement']:.0f} points."
        )
    elif len(swaps) >= 1:
        verdict = "MINOR_ADJUSTMENTS"
        verdict_detail = (
            f"Found {len(swaps)} swap opportunity(s) for incremental improvement."
        )
    else:
        verdict = "PORTFOLIO_OPTIMAL"
        verdict_detail = "No compelling swap opportunities found. Current portfolio holds up well."

    return {
        'timestamp': timestamp,
        'macro_regime': regime,
        'verdict': verdict,
        'verdict_detail': verdict_detail,
        'summary': {
            'candidates_scanned': len(candidates),
            'candidates_above_portfolio_avg': len(candidates_above_avg),
            'candidates_validated': len(validated),
            'swap_recommendations': len(swaps),
            'portfolio_avg_alpha': round(portfolio_avg_alpha, 1),
            'portfolio_min_alpha': round(portfolio_min_alpha, 1),
            'best_candidate_alpha': round(candidates[0]['alpha_score'], 1) if candidates else 0,
        },
        'top_candidates': candidates[:10],
        'swap_recommendations': swaps,
        'portfolio_weaknesses': weaknesses,
    }


def _diagnose_weakness(position: dict) -> str:
    """Diagnose why a portfolio position is weak."""
    issues = []
    alpha = position.get('alpha_score', 0)
    if alpha < 30:
        issues.append("very low alpha")
    elif alpha < 50:
        issues.append("below-average alpha")

    if position.get('momentum_30d', 0) < -10:
        issues.append(f"negative momentum ({position.get('momentum_30d', 0):.0f}% 30d)")
    if position.get('trend_quality', 50) < 35:
        issues.append("poor trend quality")
    if position.get('volatility', 0) > 60:
        issues.append(f"high volatility ({position.get('volatility', 0):.0f}%)")
    if position.get('liq_tier') == 'L0':
        issues.append("illiquid (L0)")

    return "; ".join(issues) if issues else "marginal metrics"


# ---------------------------------------------------------------------------
# 8. Main entrypoint
# ---------------------------------------------------------------------------

def run_discovery(
    portfolio_df: pd.DataFrame,
    hist_cache: Dict[str, pd.DataFrame],
    risk_mode: str = 'BALANCED',
    benchmark_data: Optional[pd.DataFrame] = None,
    macro_regime: Optional[dict] = None,
    max_candidates: int = 60,
    max_swaps: int = 5,
    validate_lookback: bool = True,
    lookback_days: int = 90,
) -> dict:
    """Run the full discovery pipeline: scan -> score -> rank -> swap -> report.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Current portfolio with at least a 'Symbol' column.
    hist_cache : Dict[str, pd.DataFrame]
        Existing price data cache (portfolio + any pre-fetched).
    risk_mode : str
        Risk profile name.
    benchmark_data : pd.DataFrame, optional
        Benchmark ETF data.
    macro_regime : dict, optional
        Current macro regime.
    max_candidates : int
        Max candidates to scan.
    max_swaps : int
        Max swap recommendations.
    validate_lookback : bool
        Whether to run lookback validation on candidates.
    lookback_days : int
        Lookback period for validation.

    Returns
    -------
    dict
        Full discovery report (see build_discovery_report).
    """
    # Extract portfolio symbols
    if portfolio_df is not None and not portfolio_df.empty and 'Symbol' in portfolio_df.columns:
        portfolio_symbols = portfolio_df['Symbol'].tolist()
    else:
        portfolio_symbols = list(hist_cache.keys())

    # Step 1: Scan universe
    candidate_symbols = scan_universe(portfolio_symbols, max_candidates=max_candidates)

    # Step 2: Fetch candidate data
    candidate_cache = fetch_candidate_data(candidate_symbols, existing_cache=hist_cache)

    # Step 3: Score candidates
    candidates = score_candidates(
        candidate_symbols, candidate_cache, benchmark_data, macro_regime, risk_mode
    )

    # Step 4: Score current portfolio (same pipeline for apples-to-apples comparison)
    portfolio_scored = score_candidates(
        portfolio_symbols, hist_cache, benchmark_data, macro_regime, risk_mode
    )

    # Step 5: Lookback validation on top candidates
    if validate_lookback:
        for c in candidates[:20]:
            sym = c['symbol']
            hist = candidate_cache.get(sym)
            if hist is not None:
                validation = validate_candidate_lookback(sym, hist, lookback_days)
                c.update(validation)

    # Step 6: Rank candidates vs portfolio
    candidates = rank_candidates_vs_portfolio(candidates, portfolio_scored)

    # Step 7: Generate swap recommendations
    swaps = generate_swap_recommendations(
        candidates, portfolio_scored,
        max_swaps=max_swaps,
        min_improvement=5.0,
        require_validation=validate_lookback,
    )

    # Step 8: Build report
    report = build_discovery_report(candidates, swaps, portfolio_scored, macro_regime)

    return report


def format_discovery_report_text(report: dict) -> str:
    """Format a discovery report as human-readable text for console / logs.

    Parameters
    ----------
    report : dict
        Output of run_discovery() or build_discovery_report().

    Returns
    -------
    str
        Formatted multi-line report.
    """
    lines = []
    lines.append("=" * 72)
    lines.append("  DISCOVERY ENGINE REPORT")
    lines.append("=" * 72)
    lines.append(f"  Timestamp  : {report.get('timestamp', 'N/A')}")
    lines.append(f"  Regime     : {report.get('macro_regime', 'N/A')}")
    lines.append(f"  Verdict    : {report.get('verdict', 'N/A')}")
    lines.append(f"  Detail     : {report.get('verdict_detail', '')}")
    lines.append("")

    summary = report.get('summary', {})
    lines.append("--- SCAN SUMMARY ---")
    lines.append(f"  Candidates scanned          : {summary.get('candidates_scanned', 0)}")
    lines.append(f"  Above portfolio avg alpha   : {summary.get('candidates_above_portfolio_avg', 0)}")
    lines.append(f"  Passed lookback validation  : {summary.get('candidates_validated', 0)}")
    lines.append(f"  Swap recommendations        : {summary.get('swap_recommendations', 0)}")
    lines.append(f"  Portfolio avg alpha          : {summary.get('portfolio_avg_alpha', 0):.1f}")
    lines.append(f"  Best candidate alpha         : {summary.get('best_candidate_alpha', 0):.1f}")
    lines.append("")

    # Swap recommendations
    swaps = report.get('swap_recommendations', [])
    if swaps:
        lines.append("--- SWAP RECOMMENDATIONS ---")
        for i, s in enumerate(swaps, 1):
            lines.append(f"  #{i}  SELL {s['sell_symbol']:8s} (alpha {s['sell_alpha']:5.1f})"
                         f"  ->  BUY {s['buy_symbol']:8s} (alpha {s['buy_alpha']:5.1f})"
                         f"  | Improvement: +{s['alpha_improvement']:.0f}  Grade: {s['upgrade_grade']}"
                         f"  Confidence: {s['confidence']}")
            detail = s.get('candidate_detail', {})
            lines.append(f"       Momentum: 30d={detail.get('momentum_30d', 0):+.1f}%  "
                         f"90d={detail.get('momentum_90d', 0):+.1f}%  "
                         f"Trend: {detail.get('trend_quality', 0):.0f}/100  "
                         f"Signal: {detail.get('signal_freshness', 0):.0f}/100  "
                         f"Sharpe: {detail.get('sharpe_ratio', 0):.2f}")
            for reason in s.get('reasoning', []):
                lines.append(f"       - {reason}")
            lines.append("")
    else:
        lines.append("--- No swap recommendations (portfolio is well-positioned) ---")
        lines.append("")

    # Top candidates
    top = report.get('top_candidates', [])[:10]
    if top:
        lines.append("--- TOP 10 CANDIDATES ---")
        lines.append(f"  {'#':>3s}  {'Symbol':8s}  {'Alpha':>6s}  {'RiskAdj':>7s}  "
                     f"{'30d%':>6s}  {'90d%':>6s}  {'Trend':>5s}  {'Signal':>6s}  "
                     f"{'Liq':>3s}  {'Grade':>5s}  {'Validated':>9s}")
        lines.append("  " + "-" * 82)
        for i, c in enumerate(top, 1):
            val_str = "YES" if c.get('passed_validation') else "no"
            lines.append(
                f"  {i:3d}  {c['symbol']:8s}  {c['alpha_score']:6.1f}  "
                f"{c.get('risk_adjusted_alpha', 0):7.1f}  "
                f"{c.get('momentum_30d', 0):+6.1f}  "
                f"{c.get('momentum_90d', 0):+6.1f}  "
                f"{c.get('trend_quality', 0):5.0f}  "
                f"{c.get('signal_freshness', 0):6.0f}  "
                f"{c.get('liq_tier', '?'):>3s}  "
                f"{c.get('upgrade_grade', '-'):>5s}  "
                f"{val_str:>9s}"
            )
        lines.append("")

    # Portfolio weaknesses
    weak = report.get('portfolio_weaknesses', [])
    if weak:
        lines.append("--- PORTFOLIO WEAKNESSES (Bottom 5) ---")
        for w in weak:
            lines.append(f"  {w['symbol']:8s}  alpha={w['alpha_score']:5.1f}  -> {w['issue']}")
        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)
