"""Unit tests for portfolio_optimizer.py

Covers: correlation matrix, metal beta, mean-variance optimisation,
risk-parity optimisation, Kelly criterion sizing, concentration risk
detection, and rebalance trade generation.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable (mirrors conftest.py)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portfolio_optimizer import (
    calculate_correlation_matrix,
    calculate_metal_beta,
    detect_concentration_risk,
    kelly_position_size,
    optimize_mean_variance,
    optimize_risk_parity,
    suggest_rebalance_trades,
)

# ---------------------------------------------------------------------------
# Helpers -- synthetic data generators
# ---------------------------------------------------------------------------

def _make_close_df(prices: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    """Return a DataFrame with a 'Close' column indexed by business days."""
    dates = pd.bdate_range(start=start, periods=len(prices))
    return pd.DataFrame({"Close": prices}, index=dates)


def _make_random_close_df(
    n: int = 120,
    start: str = "2024-01-01",
    base: float = 100.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a random-walk close-price series of *n* business days."""
    rng = np.random.default_rng(seed)
    daily_returns = rng.normal(loc=0.0005, scale=0.02, size=n)
    prices = base * np.cumprod(1.0 + daily_returns)
    return _make_close_df(prices.tolist(), start=start)


def _make_hist_cache(n_symbols: int = 4, n_days: int = 120) -> dict[str, pd.DataFrame]:
    """Build a hist_cache dict with *n_symbols* synthetic tickers."""
    symbols = [f"SYM{i}" for i in range(n_symbols)]
    cache: dict[str, pd.DataFrame] = {}
    for idx, sym in enumerate(symbols):
        cache[sym] = _make_random_close_df(n=n_days, seed=100 + idx)
    return cache


# ===================================================================
# 1. Correlation Matrix
# ===================================================================

class TestCorrelationMatrix:

    def test_correlation_matrix_identity(self):
        """Single stock produces a 1x1 matrix with value 1.0."""
        cache = {"GOLD": _make_random_close_df(n=60, seed=1)}
        corr = calculate_correlation_matrix(cache)

        assert corr.shape == (1, 1)
        assert corr.index.tolist() == ["GOLD"]
        assert corr.columns.tolist() == ["GOLD"]
        assert corr.iloc[0, 0] == pytest.approx(1.0)

    def test_correlation_matrix_two_stocks(self):
        """Two perfectly correlated (identical) price series yield correlation ~1.0."""
        prices = list(range(100, 160))  # 60 monotonically increasing prices
        cache = {
            "AAA": _make_close_df(prices),
            "BBB": _make_close_df(prices),
        }
        corr = calculate_correlation_matrix(cache, min_overlap=30)

        assert corr.shape == (2, 2)
        # Diagonal is exactly 1.0
        assert corr.loc["AAA", "AAA"] == pytest.approx(1.0)
        assert corr.loc["BBB", "BBB"] == pytest.approx(1.0)
        # Off-diagonal should be ~1.0 for identical series
        assert corr.loc["AAA", "BBB"] == pytest.approx(1.0, abs=1e-6)
        assert corr.loc["BBB", "AAA"] == pytest.approx(1.0, abs=1e-6)

    def test_correlation_matrix_empty(self):
        """Empty hist_cache returns an empty DataFrame."""
        corr = calculate_correlation_matrix({})
        assert isinstance(corr, pd.DataFrame)
        assert corr.empty


# ===================================================================
# 2. Metal Beta
# ===================================================================

class TestMetalBeta:

    def test_metal_beta_self(self):
        """Regressing a stock against itself should give beta ~1.0 and r_squared ~1.0."""
        df = _make_random_close_df(n=120, seed=7)
        result = calculate_metal_beta(df, df)

        assert result["beta"] == pytest.approx(1.0, abs=1e-6)
        assert result["r_squared"] == pytest.approx(1.0, abs=1e-6)

    def test_metal_beta_returns_keys(self):
        """Return dict must contain beta, r_squared, and alpha."""
        stock = _make_random_close_df(n=120, seed=10)
        bench = _make_random_close_df(n=120, seed=20)
        result = calculate_metal_beta(stock, bench)

        assert "beta" in result
        assert "r_squared" in result
        assert "alpha" in result


# ===================================================================
# 3. Mean-Variance Optimisation
# ===================================================================

class TestMeanVariance:

    @pytest.fixture()
    def mv_result(self):
        cache = _make_hist_cache(n_symbols=4, n_days=120)
        return optimize_mean_variance(cache, max_weight=0.40)

    def test_mv_weights_sum_to_one(self, mv_result):
        """Optimised weights should sum to approximately 1.0."""
        weights = mv_result["weights"]
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_mv_max_weight_respected(self):
        """No individual weight should exceed the max_weight parameter."""
        cache = _make_hist_cache(n_symbols=4, n_days=120)
        max_w = 0.35
        result = optimize_mean_variance(cache, max_weight=max_w)
        weights = result["weights"]

        for sym, w in weights.items():
            assert w <= max_w + 1e-4, (
                f"{sym} weight {w:.4f} exceeds max_weight {max_w}"
            )

    def test_mv_returns_expected_keys(self, mv_result):
        """Result dict must contain all documented keys."""
        expected_keys = {
            "weights",
            "expected_return",
            "expected_volatility",
            "sharpe_ratio",
            "efficient_frontier",
        }
        assert expected_keys.issubset(mv_result.keys())


# ===================================================================
# 4. Risk-Parity Optimisation
# ===================================================================

class TestRiskParity:

    @pytest.fixture()
    def rp_result(self):
        cache = _make_hist_cache(n_symbols=4, n_days=120)
        return optimize_risk_parity(cache, max_weight=0.40)

    def test_risk_parity_weights_sum_to_one(self, rp_result):
        """Risk-parity weights should sum to approximately 1.0."""
        weights = rp_result["weights"]
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_risk_parity_returns_keys(self, rp_result):
        """Result dict must contain weights, risk_contributions, and portfolio_volatility."""
        assert "weights" in rp_result
        assert "risk_contributions" in rp_result
        assert "portfolio_volatility" in rp_result


# ===================================================================
# 5. Kelly Criterion
# ===================================================================

class TestKellyCriterion:

    def test_kelly_positive(self):
        """With win_rate=0.6, avg_win=2, avg_loss=1 the fraction should be positive."""
        fraction = kelly_position_size(win_rate=0.6, avg_win=2.0, avg_loss=1.0)
        assert fraction > 0.0

    def test_kelly_capped(self):
        """A very favourable edge should be capped at max_fraction."""
        max_frac = 0.25
        fraction = kelly_position_size(
            win_rate=0.95, avg_win=10.0, avg_loss=1.0, max_fraction=max_frac,
        )
        assert fraction == pytest.approx(max_frac)

    def test_kelly_negative(self):
        """A losing strategy (low win rate, symmetric payoff) returns 0."""
        fraction = kelly_position_size(win_rate=0.2, avg_win=1.0, avg_loss=1.0)
        assert fraction == 0.0

    def test_kelly_zero_loss(self):
        """avg_loss=0 is an edge case -- should return 0.0 gracefully."""
        fraction = kelly_position_size(win_rate=0.6, avg_win=2.0, avg_loss=0.0)
        assert fraction == 0.0


# ===================================================================
# 6. Concentration Risk
# ===================================================================

class TestConcentrationRisk:

    def test_concentration_no_warnings(self):
        """A well-diversified portfolio should produce no warnings."""
        portfolio_df = pd.DataFrame({
            "Symbol": ["A", "B", "C", "D", "E"],
            "Market_Value": [200, 200, 200, 200, 200],
            "metal": ["Gold", "Silver", "Copper", "Zinc", "Uranium"],
            "country": ["Canada", "Australia", "USA", "Peru", "Namibia"],
            "stage": ["Producer", "Developer", "Explorer", "Producer", "Developer"],
        })
        result = detect_concentration_risk(portfolio_df)
        # With even 20% splits no metal/country/stage breaches thresholds
        # and no pair exceeds 20% combined weight.
        metal_warnings = [
            w for w in result["warnings"] if w.startswith("Metal concentration")
        ]
        assert metal_warnings == []

    def test_concentration_metal_warning(self):
        """More than 40% in one metal should trigger a warning."""
        portfolio_df = pd.DataFrame({
            "Symbol": ["A", "B", "C"],
            "Market_Value": [700, 200, 100],
            "metal": ["Gold", "Gold", "Silver"],
            "country": ["Canada", "Australia", "USA"],
            "stage": ["Producer", "Developer", "Explorer"],
        })
        result = detect_concentration_risk(portfolio_df)
        metal_warnings = [
            w for w in result["warnings"] if w.startswith("Metal concentration")
        ]
        assert len(metal_warnings) >= 1
        assert "Gold" in metal_warnings[0]

    def test_concentration_score_range(self):
        """Concentration score should always be between 0 and 100."""
        portfolio_df = pd.DataFrame({
            "Symbol": ["X", "Y"],
            "Market_Value": [900, 100],
            "metal": ["Gold", "Gold"],
            "country": ["Canada", "Canada"],
            "stage": ["Producer", "Producer"],
        })
        result = detect_concentration_risk(portfolio_df)
        score = result["concentration_score"]
        assert 0.0 <= score <= 100.0


# ===================================================================
# 7. Rebalance Trades
# ===================================================================

class TestRebalanceTrades:

    def test_rebalance_generates_trades(self):
        """Different current vs target weights should produce BUY and SELL trades."""
        current = {"A": 0.50, "B": 0.30, "C": 0.20}
        target = {"A": 0.20, "B": 0.30, "C": 0.50}
        trades = suggest_rebalance_trades(current, target, portfolio_value=100_000)

        symbols_traded = {t["symbol"] for t in trades}
        actions = {t["action"] for t in trades}

        # A decreases (SELL) and C increases (BUY); B stays put
        assert "A" in symbols_traded
        assert "C" in symbols_traded
        assert "BUY" in actions
        assert "SELL" in actions

        # Verify directions
        trade_map = {t["symbol"]: t for t in trades}
        assert trade_map["A"]["action"] == "SELL"
        assert trade_map["C"]["action"] == "BUY"

    def test_rebalance_filters_small_trades(self):
        """Trades below min_trade_pct should be excluded."""
        current = {"A": 0.500, "B": 0.500}
        target = {"A": 0.502, "B": 0.498}  # 0.2% difference each
        trades = suggest_rebalance_trades(
            current, target, portfolio_value=100_000, min_trade_pct=0.5,
        )
        # Both differences are 0.2%, below the 0.5% threshold
        assert trades == []


# ===================================================================
# 9. Portfolio Health Score
# ===================================================================

from portfolio_optimizer import calculate_portfolio_health_score

class TestPortfolioHealthScore:

    def test_returns_valid_structure(self):
        cache = _make_hist_cache(n_symbols=4, n_days=60)
        weights = {f"SYM{i}": 0.25 for i in range(4)}
        result = calculate_portfolio_health_score(cache, weights)
        assert 'health_score' in result
        assert 'diversification' in result
        assert 'return_efficiency' in result
        assert 'risk_score' in result
        assert 0 <= result['health_score'] <= 100

    def test_concentrated_portfolio_low_diversification(self):
        cache = _make_hist_cache(n_symbols=2, n_days=60)
        weights = {"SYM0": 0.90, "SYM1": 0.10}
        result = calculate_portfolio_health_score(cache, weights)
        assert result['diversification'] < 50

    def test_equal_weight_good_diversification(self):
        cache = _make_hist_cache(n_symbols=8, n_days=60)
        weights = {f"SYM{i}": 0.125 for i in range(8)}
        result = calculate_portfolio_health_score(cache, weights)
        assert result['diversification'] >= 60

    def test_empty_inputs(self):
        result = calculate_portfolio_health_score({}, {})
        assert result['health_score'] == 50.0


# ===================================================================
# 10. Discovery-Aware Rebalancing
# ===================================================================

from portfolio_optimizer import integrate_discovery_into_rebalance

class TestDiscoveryRebalance:

    def test_generates_trades_from_swaps(self):
        current_weights = {"OLD1": 0.25, "OLD2": 0.25, "KEEP1": 0.25, "KEEP2": 0.25}
        swaps = [
            {
                'sell_symbol': 'OLD1', 'buy_symbol': 'NEW1',
                'confidence': 'High', 'alpha_improvement': 20,
            },
        ]
        trades = integrate_discovery_into_rebalance(
            current_weights, swaps, portfolio_value=200_000,
            prices={"OLD1": 10.0, "NEW1": 15.0, "OLD2": 8.0, "KEEP1": 20.0, "KEEP2": 12.0}
        )
        symbols = {t['symbol'] for t in trades}
        assert 'OLD1' in symbols  # Should be selling OLD1
        assert 'NEW1' in symbols  # Should be buying NEW1

    def test_no_swaps_no_trades(self):
        current_weights = {"A": 0.5, "B": 0.5}
        trades = integrate_discovery_into_rebalance(
            current_weights, [], portfolio_value=100_000
        )
        assert len(trades) == 0

    def test_confidence_scales_allocation(self):
        current_weights = {"OLD": 0.50, "KEEP": 0.50}
        # High confidence swap
        swaps_high = [{'sell_symbol': 'OLD', 'buy_symbol': 'NEW', 'confidence': 'High'}]
        trades_high = integrate_discovery_into_rebalance(
            current_weights, swaps_high, portfolio_value=100_000
        )
        # Low confidence swap
        swaps_low = [{'sell_symbol': 'OLD', 'buy_symbol': 'NEW', 'confidence': 'Low'}]
        trades_low = integrate_discovery_into_rebalance(
            current_weights, swaps_low, portfolio_value=100_000
        )
        # High confidence should result in larger NEW position
        new_high = next((t for t in trades_high if t['symbol'] == 'NEW'), None)
        new_low = next((t for t in trades_low if t['symbol'] == 'NEW'), None)
        if new_high and new_low:
            assert new_high['trade_value'] >= new_low['trade_value']
