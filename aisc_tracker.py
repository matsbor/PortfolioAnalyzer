#!/usr/bin/env python3
"""
AISC Tracker — All-In Sustaining Cost tracking for mining producers.
Key metric for evaluating mining company profitability and margin of safety.

Data resolution order (API-first):
1. yfinance financial statements (operating expenses, cost of revenue, gross margins)
2. Tiingo fundamentals API (if available)
3. User-editable aisc_data.json file (community-maintained overrides)
4. Industry-average fallback (last resort, clearly labeled)

NO hardcoded per-company AISC values in this Python source file.
All known values live in aisc_data.json which users can update.
"""

import json
import os
import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Optional imports
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent
_AISC_JSON_PATH = _DATA_DIR / 'aisc_data.json'
_AISC_CACHE_PATH = _DATA_DIR / '.aisc_cache.json'
_CACHE_TTL_HOURS = 24  # Re-fetch financial data after this many hours


# ---------------------------------------------------------------------------
# Industry-average AISC fallback ranges (last resort only)
# These are approximate industry medians. They are ONLY used when:
#   - yfinance financials are unavailable
#   - Tiingo fundamentals are unavailable
#   - aisc_data.json has no entry for the symbol
# ---------------------------------------------------------------------------

_INDUSTRY_FALLBACK: Dict[str, Dict[str, Any]] = {
    'Gold':    {'average': 1350, 'high': 1600, 'low': 950, 'unit': '$/oz'},
    'Silver':  {'average': 19.50, 'high': 25.00, 'low': 14.00, 'unit': '$/oz'},
    'Uranium': {'average': 45.00, 'high': 65.00, 'low': 30.00, 'unit': '$/lb'},
    'Copper':  {'average': 2.60, 'high': 3.50, 'low': 1.90, 'unit': '$/lb'},
    'Lithium': {'average': 12000, 'high': 20000, 'low': 8000, 'unit': '$/t'},
}


# ---------------------------------------------------------------------------
# JSON data file helpers
# ---------------------------------------------------------------------------

def load_aisc_data(filepath: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Load user-editable AISC data from aisc_data.json.

    This file is the community-maintained source of known AISC values.
    Users should update this file with the latest reported AISC from
    company quarterly/annual filings.

    Returns empty dict if file doesn't exist (no hardcoded fallback).
    """
    path = Path(filepath) if filepath else _AISC_JSON_PATH
    if not path.exists():
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        return {k.upper(): v for k, v in data.items() if isinstance(v, dict)}
    except (json.JSONDecodeError, OSError):
        return {}


def save_aisc_data(data: Dict[str, Dict[str, Any]],
                   filepath: Optional[str] = None) -> None:
    """Persist AISC data to aisc_data.json."""
    path = Path(filepath) if filepath else _AISC_JSON_PATH
    normalised = {k.upper(): v for k, v in data.items()}
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(normalised, fh, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Cache helpers (API results are cached to avoid repeated fetches)
# ---------------------------------------------------------------------------

def _load_cache() -> Dict[str, Any]:
    """Load the AISC estimation cache."""
    if not _AISC_CACHE_PATH.exists():
        return {}
    try:
        with open(_AISC_CACHE_PATH, 'r', encoding='utf-8') as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    """Save the AISC estimation cache."""
    try:
        with open(_AISC_CACHE_PATH, 'w', encoding='utf-8') as fh:
            json.dump(cache, fh, indent=2)
    except OSError:
        pass


def _cache_is_fresh(entry: Dict[str, Any]) -> bool:
    """Check if a cache entry is still fresh."""
    ts = entry.get('cached_at', '')
    if not ts:
        return False
    try:
        cached_time = datetime.datetime.fromisoformat(ts)
        age = datetime.datetime.now(datetime.timezone.utc) - cached_time
        return age.total_seconds() < _CACHE_TTL_HOURS * 3600
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# API-first AISC estimation
# ---------------------------------------------------------------------------

def estimate_aisc_from_yfinance(symbol: str, metal: str) -> Optional[Dict[str, Any]]:
    """Estimate AISC from live yfinance financial data.

    Tries multiple estimation methods using real financial statements:
    1. Operating expense ratio (highest confidence)
    2. Cost of revenue with sustaining capex multiplier
    3. Gross margin inversion

    Returns None if yfinance is unavailable or data is insufficient.
    """
    if not YFINANCE_AVAILABLE:
        return None

    industry = _INDUSTRY_FALLBACK.get(metal)
    if not industry:
        return None

    avg_aisc = industry['average']

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        if not info:
            return None
    except Exception:
        return None

    # Method 1: Operating expense ratio (best signal)
    total_opex = info.get('totalOperatingExpenses')
    total_revenue = info.get('totalRevenue')
    if total_opex and total_revenue and total_revenue > 0:
        cost_ratio = total_opex / total_revenue
        estimated = avg_aisc * cost_ratio
        return {
            'aisc_estimate': round(estimated, 2),
            'confidence': 'medium',
            'method': 'yfinance_opex_ratio',
            'source': 'yfinance',
            'raw_opex': total_opex,
            'raw_revenue': total_revenue,
        }

    # Method 2: Cost of revenue ratio
    cost_of_revenue = info.get('costOfRevenue')
    if cost_of_revenue and total_revenue and total_revenue > 0:
        cost_ratio = cost_of_revenue / total_revenue
        # 1.15x multiplier for sustaining capex (industry convention)
        estimated = avg_aisc * cost_ratio * 1.15
        return {
            'aisc_estimate': round(estimated, 2),
            'confidence': 'low',
            'method': 'yfinance_cost_of_revenue',
            'source': 'yfinance',
            'raw_cor': cost_of_revenue,
            'raw_revenue': total_revenue,
        }

    # Method 3: Gross margin inversion
    gross_margins = info.get('grossMargins')
    if gross_margins is not None and 0 < gross_margins < 1.0:
        cost_fraction = 1.0 - gross_margins
        estimated = avg_aisc * cost_fraction * 1.15
        return {
            'aisc_estimate': round(estimated, 2),
            'confidence': 'low',
            'method': 'yfinance_gross_margin',
            'source': 'yfinance',
            'raw_margin': gross_margins,
        }

    return None


def estimate_aisc_from_tiingo(symbol: str, metal: str) -> Optional[Dict[str, Any]]:
    """Estimate AISC from Tiingo fundamentals API.

    Uses Tiingo's fundamental data endpoints for operating metrics.
    Returns None if Tiingo API key is not set or data is unavailable.
    """
    if not REQUESTS_AVAILABLE:
        return None

    tiingo_key = os.getenv('TIINGO_API_KEY', '').strip()
    if not tiingo_key:
        return None

    industry = _INDUSTRY_FALLBACK.get(metal)
    if not industry:
        return None

    avg_aisc = industry['average']

    try:
        # Tiingo fundamentals endpoint
        url = f"https://api.tiingo.com/tiingo/fundamentals/{symbol}/statements"
        headers = {'Authorization': f'Token {tiingo_key}', 'Content-Type': 'application/json'}
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code != 200:
            return None

        statements = resp.json()
        if not statements:
            return None

        # Get most recent statement
        latest = statements[0] if isinstance(statements, list) else statements
        statement_data = latest.get('statementData', {})
        income = statement_data.get('incomeStatement', [])

        revenue = None
        cost_of_rev = None
        for item in income:
            if item.get('dataCode') == 'revenue':
                revenue = item.get('value')
            elif item.get('dataCode') == 'costRev':
                cost_of_rev = item.get('value')

        if revenue and cost_of_rev and revenue > 0:
            cost_ratio = cost_of_rev / revenue
            estimated = avg_aisc * cost_ratio * 1.15
            return {
                'aisc_estimate': round(estimated, 2),
                'confidence': 'medium',
                'method': 'tiingo_fundamentals',
                'source': 'tiingo',
                'raw_revenue': revenue,
                'raw_cost': cost_of_rev,
            }
    except Exception:
        pass

    return None


def estimate_aisc_from_info(info: Dict[str, Any], metal: str) -> Optional[Dict[str, Any]]:
    """Estimate AISC from a pre-fetched yfinance .info dict.

    Same logic as estimate_aisc_from_yfinance but works with an already-fetched
    info dict (avoids redundant API call if caller already has it).
    """
    industry = _INDUSTRY_FALLBACK.get(metal)
    if not industry:
        return None

    avg_aisc = industry['average']

    # Method 1: Operating expense ratio
    total_opex = info.get('totalOperatingExpenses')
    total_revenue = info.get('totalRevenue')
    if total_opex and total_revenue and total_revenue > 0:
        cost_ratio = total_opex / total_revenue
        estimated = avg_aisc * cost_ratio
        return {
            'aisc_estimate': round(estimated, 2),
            'confidence': 'medium',
            'method': 'info_opex_ratio',
            'source': 'yfinance_info',
        }

    # Method 2: Cost of revenue
    cost_of_revenue = info.get('costOfRevenue')
    if cost_of_revenue and total_revenue and total_revenue > 0:
        cost_ratio = cost_of_revenue / total_revenue
        estimated = avg_aisc * cost_ratio * 1.15
        return {
            'aisc_estimate': round(estimated, 2),
            'confidence': 'low',
            'method': 'info_cost_of_revenue',
            'source': 'yfinance_info',
        }

    # Method 3: Gross margin inversion
    gross_margins = info.get('grossMargins')
    if gross_margins is not None and 0 < gross_margins < 1.0:
        cost_fraction = 1.0 - gross_margins
        estimated = avg_aisc * cost_fraction * 1.15
        return {
            'aisc_estimate': round(estimated, 2),
            'confidence': 'low',
            'method': 'info_gross_margin',
            'source': 'yfinance_info',
        }

    return None


# ---------------------------------------------------------------------------
# Margin calculation
# ---------------------------------------------------------------------------

def calculate_aisc_margin(aisc: float, spot_price: float,
                          metal: str) -> Dict[str, Any]:
    """Calculate the margin between the spot price and AISC.

    Args:
        aisc: All-In Sustaining Cost per unit.
        spot_price: Current spot price per unit (same unit as aisc).
        metal: Metal name (used only for informational output).

    Returns:
        Dict with margin_pct, margin_absolute, assessment, metal.
    """
    if spot_price <= 0:
        return {
            'margin_pct': 0.0,
            'margin_absolute': 0.0,
            'assessment': 'unknown',
            'metal': metal,
        }

    margin_abs = spot_price - aisc
    margin_pct = (margin_abs / spot_price) * 100.0

    if margin_pct > 30.0:
        assessment = 'excellent'
    elif margin_pct > 20.0:
        assessment = 'good'
    elif margin_pct > 10.0:
        assessment = 'adequate'
    elif margin_pct > 0.0:
        assessment = 'thin'
    else:
        assessment = 'negative'

    return {
        'margin_pct': round(margin_pct, 2),
        'margin_absolute': round(margin_abs, 2),
        'assessment': assessment,
        'metal': metal,
    }


# ---------------------------------------------------------------------------
# Master scoring function (API-first resolution)
# ---------------------------------------------------------------------------

def get_aisc_score(symbol: str, metal: str, spot_price: float,
                   info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a comprehensive AISC score for a mining stock.

    Data resolution order (API-first, no hardcoded per-company values):
    1. Check 24h cache for a previous API result
    2. If caller provided yfinance .info dict, estimate from that
    3. Fetch live from yfinance financial statements
    4. Fetch from Tiingo fundamentals API
    5. Look up in user-editable aisc_data.json
    6. Fall back to industry average (clearly labeled)

    Args:
        symbol: Ticker symbol (case-insensitive).
        metal: Primary metal (e.g. 'Gold', 'Silver', 'Uranium').
        spot_price: Current spot price per unit.
        info: Optional pre-fetched yfinance .info dict.

    Returns:
        Dict with aisc_estimate, aisc_source, margin_pct, score, unit, assessment.
    """
    sym = symbol.upper()
    unit = _INDUSTRY_FALLBACK.get(metal, {}).get('unit', '$/oz')
    aisc_value = 0.0
    aisc_source = 'industry_average'

    # --- Step 1: Check cache -------------------------------------------------
    cache = _load_cache()
    cached = cache.get(sym)
    if cached and _cache_is_fresh(cached):
        aisc_value = cached.get('aisc_estimate', 0.0)
        aisc_source = cached.get('source', 'cached')
        if aisc_value > 0:
            margin = calculate_aisc_margin(aisc_value, spot_price, metal)
            score = int(max(0.0, min(100.0, margin['margin_pct'] * 2.0)))
            return {
                'aisc_estimate': round(aisc_value, 2),
                'aisc_source': f"{aisc_source} (cached)",
                'margin_pct': margin['margin_pct'],
                'score': score,
                'unit': unit,
                'assessment': margin['assessment'],
            }

    # --- Step 2: Estimate from caller-provided info dict ---------------------
    if info:
        result = estimate_aisc_from_info(info, metal)
        if result and result.get('aisc_estimate', 0) > 0:
            aisc_value = result['aisc_estimate']
            aisc_source = result.get('source', 'yfinance_info')
            # Cache it
            cache[sym] = {
                'aisc_estimate': aisc_value,
                'source': aisc_source,
                'method': result.get('method', ''),
                'confidence': result.get('confidence', 'low'),
                'cached_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            _save_cache(cache)
            margin = calculate_aisc_margin(aisc_value, spot_price, metal)
            score = int(max(0.0, min(100.0, margin['margin_pct'] * 2.0)))
            return {
                'aisc_estimate': round(aisc_value, 2),
                'aisc_source': aisc_source,
                'margin_pct': margin['margin_pct'],
                'score': score,
                'unit': unit,
                'assessment': margin['assessment'],
            }

    # --- Step 3: Live yfinance fetch -----------------------------------------
    yf_result = estimate_aisc_from_yfinance(sym, metal)
    if yf_result and yf_result.get('aisc_estimate', 0) > 0:
        aisc_value = yf_result['aisc_estimate']
        aisc_source = 'yfinance'
        cache[sym] = {
            'aisc_estimate': aisc_value,
            'source': 'yfinance',
            'method': yf_result.get('method', ''),
            'confidence': yf_result.get('confidence', 'low'),
            'cached_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        _save_cache(cache)
        margin = calculate_aisc_margin(aisc_value, spot_price, metal)
        score = int(max(0.0, min(100.0, margin['margin_pct'] * 2.0)))
        return {
            'aisc_estimate': round(aisc_value, 2),
            'aisc_source': aisc_source,
            'margin_pct': margin['margin_pct'],
            'score': score,
            'unit': unit,
            'assessment': margin['assessment'],
        }

    # --- Step 4: Tiingo fundamentals -----------------------------------------
    tiingo_result = estimate_aisc_from_tiingo(sym, metal)
    if tiingo_result and tiingo_result.get('aisc_estimate', 0) > 0:
        aisc_value = tiingo_result['aisc_estimate']
        aisc_source = 'tiingo'
        cache[sym] = {
            'aisc_estimate': aisc_value,
            'source': 'tiingo',
            'method': tiingo_result.get('method', ''),
            'confidence': tiingo_result.get('confidence', 'low'),
            'cached_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        _save_cache(cache)
        margin = calculate_aisc_margin(aisc_value, spot_price, metal)
        score = int(max(0.0, min(100.0, margin['margin_pct'] * 2.0)))
        return {
            'aisc_estimate': round(aisc_value, 2),
            'aisc_source': aisc_source,
            'margin_pct': margin['margin_pct'],
            'score': score,
            'unit': unit,
            'assessment': margin['assessment'],
        }

    # --- Step 5: User-editable JSON file -------------------------------------
    json_data = load_aisc_data()
    if sym in json_data:
        record = json_data[sym]
        aisc_value = float(record.get('aisc', 0))
        aisc_source = 'aisc_data.json'
        unit = record.get('unit', unit)

        # Royalty/streaming companies — not comparable
        if aisc_value == 0 and record.get('unit') == 'N/A':
            return {
                'aisc_estimate': 0.0,
                'aisc_source': 'aisc_data.json',
                'margin_pct': 0.0,
                'score': 0,
                'unit': 'N/A',
                'assessment': 'royalty_streaming',
            }

        if aisc_value > 0:
            margin = calculate_aisc_margin(aisc_value, spot_price, metal)
            score = int(max(0.0, min(100.0, margin['margin_pct'] * 2.0)))
            return {
                'aisc_estimate': round(aisc_value, 2),
                'aisc_source': aisc_source,
                'margin_pct': margin['margin_pct'],
                'score': score,
                'unit': unit,
                'assessment': margin['assessment'],
            }

    # --- Step 6: Industry average fallback (last resort) ---------------------
    industry = _INDUSTRY_FALLBACK.get(metal)
    if industry:
        aisc_value = industry['average']
        unit = industry['unit']
    aisc_source = 'industry_average_fallback'

    margin = calculate_aisc_margin(aisc_value, spot_price, metal)
    score = int(max(0.0, min(100.0, margin['margin_pct'] * 2.0)))

    return {
        'aisc_estimate': round(aisc_value, 2),
        'aisc_source': aisc_source,
        'margin_pct': margin['margin_pct'],
        'score': score,
        'unit': unit,
        'assessment': margin['assessment'],
    }


# ---------------------------------------------------------------------------
# P/NAV estimation
# ---------------------------------------------------------------------------

def estimate_pnav(market_cap_m: float,
                  reserves_oz: float,
                  spot_price: float,
                  aisc: float,
                  discount_rate: float = 0.05,
                  mine_life_years: int = 10,
                  annual_production_oz: Optional[float] = None) -> Dict[str, Any]:
    """Estimate Price-to-NAV using a simple DCF model.

    NAV = sum of discounted annual cash flow = production * (spot - AISC).

    Args:
        market_cap_m: Current market cap in millions of dollars.
        reserves_oz: Total proven + probable reserves in ounces.
        spot_price: Current spot price per unit.
        aisc: All-In Sustaining Cost per unit.
        discount_rate: Annual discount rate (default 5%).
        mine_life_years: Assumed remaining mine life in years (default 10).
        annual_production_oz: Annual production. If None, estimated from reserves.

    Returns:
        Dict with nav_m, pnav, assessment.
    """
    if annual_production_oz is None:
        if mine_life_years > 0:
            annual_production_oz = reserves_oz / mine_life_years
        else:
            annual_production_oz = 0.0

    annual_cashflow = annual_production_oz * (spot_price - aisc)

    nav = 0.0
    for t in range(1, mine_life_years + 1):
        nav += annual_cashflow / ((1.0 + discount_rate) ** t)

    nav_m = nav / 1_000_000.0

    if nav_m > 0:
        pnav = market_cap_m / nav_m
    else:
        pnav = float('inf')

    if pnav < 0.5:
        assessment = 'deep_value'
    elif pnav < 0.8:
        assessment = 'undervalued'
    elif pnav <= 1.2:
        assessment = 'fair'
    else:
        assessment = 'premium'

    return {
        'nav_m': round(nav_m, 2),
        'pnav': round(pnav, 2) if pnav != float('inf') else float('inf'),
        'assessment': assessment,
    }
