#!/usr/bin/env python3
"""
AISC Tracker — All-In Sustaining Cost tracking for mining producers.
Key metric for evaluating mining company profitability and margin of safety.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Industry-average AISC ranges by primary metal (2024-2025 estimates)
# ---------------------------------------------------------------------------

AISC_INDUSTRY_AVERAGES: Dict[str, Dict[str, Any]] = {
    'Gold': {'average': 1350, 'high': 1600, 'low': 950, 'unit': '$/oz'},
    'Silver': {'average': 19.50, 'high': 25.00, 'low': 14.00, 'unit': '$/oz'},
    'Uranium': {'average': 45.00, 'high': 65.00, 'low': 30.00, 'unit': '$/lb'},
    'Copper': {'average': 2.60, 'high': 3.50, 'low': 1.90, 'unit': '$/lb'},
    'Lithium': {'average': 12000, 'high': 20000, 'low': 8000, 'unit': '$/t'},
}

# ---------------------------------------------------------------------------
# Hardcoded AISC values for well-known producers (approximate 2024-2025)
# ---------------------------------------------------------------------------

KNOWN_AISC: Dict[str, Dict[str, Any]] = {
    # Gold producers
    'GOLD': {'aisc': 1280, 'unit': '$/oz', 'metal': 'Gold', 'source': 'known'},
    'NEM':  {'aisc': 1400, 'unit': '$/oz', 'metal': 'Gold', 'source': 'known'},
    'AEM':  {'aisc': 1200, 'unit': '$/oz', 'metal': 'Gold', 'source': 'known'},
    'KGC':  {'aisc': 1350, 'unit': '$/oz', 'metal': 'Gold', 'source': 'known'},
    'BTG':  {'aisc': 1300, 'unit': '$/oz', 'metal': 'Gold', 'source': 'known'},
    'EGO':  {'aisc': 1250, 'unit': '$/oz', 'metal': 'Gold', 'source': 'known'},
    'IAG':  {'aisc': 1500, 'unit': '$/oz', 'metal': 'Gold', 'source': 'known'},
    # Silver producers
    'AG':   {'aisc': 22.50, 'unit': '$/oz', 'metal': 'Silver', 'source': 'known'},
    'PAAS': {'aisc': 18.00, 'unit': '$/oz', 'metal': 'Silver', 'source': 'known'},
    'HL':   {'aisc': 12.50, 'unit': '$/oz', 'metal': 'Silver', 'source': 'known'},
    # Uranium producers
    'CCJ':  {'aisc': 35.00, 'unit': '$/lb', 'metal': 'Uranium', 'source': 'known'},
    # Copper producers
    'FCX':  {'aisc': 2.45, 'unit': '$/lb', 'metal': 'Copper', 'source': 'known'},
    'SCCO': {'aisc': 2.10, 'unit': '$/lb', 'metal': 'Copper', 'source': 'known'},
    # Royalty / streaming companies — AISC not applicable
    'FNV':  {'aisc': 0, 'unit': 'N/A', 'metal': 'Gold', 'source': 'known'},
    'WPM':  {'aisc': 0, 'unit': 'N/A', 'metal': 'Gold', 'source': 'known'},
    'RGLD': {'aisc': 0, 'unit': 'N/A', 'metal': 'Gold', 'source': 'known'},
}

_DEFAULT_JSON_PATH = Path(__file__).resolve().parent / 'aisc_data.json'


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def load_aisc_data(filepath: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Load AISC data from the JSON file (if it exists) and merge with KNOWN_AISC.

    The JSON file takes precedence over the hardcoded data so that manual
    overrides are respected.

    Args:
        filepath: Path to the JSON data file.  Defaults to ``aisc_data.json``
                  in the same directory as this module.

    Returns:
        A dict mapping ticker symbol to an AISC record::

            { 'GOLD': {'aisc': 1280, 'unit': '$/oz', 'metal': 'Gold', 'source': 'known'}, ... }
    """
    merged: Dict[str, Dict[str, Any]] = {}
    merged.update(KNOWN_AISC)

    path = Path(filepath) if filepath else _DEFAULT_JSON_PATH

    if path.exists():
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                file_data: Dict[str, Any] = json.load(fh)
            for symbol, record in file_data.items():
                symbol_upper = symbol.upper()
                if isinstance(record, dict) and 'aisc' in record:
                    record.setdefault('source', 'json')
                    merged[symbol_upper] = record
        except (json.JSONDecodeError, OSError):
            pass  # Gracefully fall back to hardcoded data

    return merged


def save_aisc_data(data: Dict[str, Dict[str, Any]],
                   filepath: Optional[str] = None) -> None:
    """Persist user-provided AISC data to ``aisc_data.json``.

    Args:
        data: Dict mapping ticker symbol to AISC record.  Each record should
              contain at minimum ``aisc``, ``unit``, and ``metal`` keys.
        filepath: Destination path.  Defaults to ``aisc_data.json`` beside
                  this module.
    """
    path = Path(filepath) if filepath else _DEFAULT_JSON_PATH

    normalised: Dict[str, Dict[str, Any]] = {}
    for symbol, record in data.items():
        normalised[symbol.upper()] = record

    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(normalised, fh, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Estimation from financial data
# ---------------------------------------------------------------------------

def estimate_aisc_from_financials(info: Dict[str, Any],
                                  metal: str) -> Dict[str, Any]:
    """Estimate AISC from a yfinance ``.info`` dict.

    This is inherently approximate.  The method attempts to derive an
    all-in cost figure from operating expenses and revenue data, then scales
    it to the conventional per-unit cost using the industry-average AISC as
    a reference.

    The estimation proceeds through increasingly coarse methods:

    1. **Operating-expense ratio** — ``totalOperatingExpenses / totalRevenue``
       scaled against the industry average for *metal*.  Confidence: *medium*.
    2. **Cost-of-revenue ratio** — ``costOfRevenue / totalRevenue`` with a
       1.15x multiplier (to approximate sustaining capex).  Confidence: *low*.
    3. **Gross-margin inversion** — derives cost fraction from gross margins.
       Confidence: *low*.

    If none of the above can be computed, returns the industry average with
    confidence *low*.

    Args:
        info: A dict of financial data as returned by ``yfinance.Ticker.info``.
        metal: Primary metal produced (e.g. ``'Gold'``, ``'Silver'``).

    Returns:
        A dict with keys ``aisc_estimate``, ``confidence``, and ``method``.
    """
    industry = AISC_INDUSTRY_AVERAGES.get(metal)
    if industry is None:
        return {
            'aisc_estimate': 0.0,
            'confidence': 'low',
            'method': 'unknown_metal',
        }

    avg_aisc: float = industry['average']

    # --- Method 1: operating-expense ratio -----------------------------------
    total_opex = info.get('totalOperatingExpenses')
    total_revenue = info.get('totalRevenue')

    if total_opex and total_revenue and total_revenue > 0:
        cost_ratio = total_opex / total_revenue
        estimated = avg_aisc * cost_ratio
        return {
            'aisc_estimate': round(estimated, 2),
            'confidence': 'medium',
            'method': 'opex_ratio',
        }

    # --- Method 2: cost-of-revenue ratio with sustaining-capex multiplier ----
    cost_of_revenue = info.get('costOfRevenue')

    if cost_of_revenue and total_revenue and total_revenue > 0:
        cost_ratio = cost_of_revenue / total_revenue
        sustaining_multiplier = 1.15  # approximate sustaining capex add-on
        estimated = avg_aisc * cost_ratio * sustaining_multiplier
        return {
            'aisc_estimate': round(estimated, 2),
            'confidence': 'low',
            'method': 'cost_of_revenue_ratio',
        }

    # --- Method 3: gross-margin inversion ------------------------------------
    gross_margins = info.get('grossMargins')

    if gross_margins is not None and gross_margins < 1.0:
        cost_fraction = 1.0 - gross_margins
        sustaining_multiplier = 1.15
        estimated = avg_aisc * cost_fraction * sustaining_multiplier
        return {
            'aisc_estimate': round(estimated, 2),
            'confidence': 'low',
            'method': 'gross_margin_inversion',
        }

    # --- Fallback: industry average ------------------------------------------
    return {
        'aisc_estimate': avg_aisc,
        'confidence': 'low',
        'method': 'industry_average_fallback',
    }


# ---------------------------------------------------------------------------
# Margin calculation
# ---------------------------------------------------------------------------

def calculate_aisc_margin(aisc: float, spot_price: float,
                          metal: str) -> Dict[str, Any]:
    """Calculate the margin between the spot price and AISC.

    Args:
        aisc: All-In Sustaining Cost per unit.
        spot_price: Current spot price per unit (same unit as *aisc*).
        metal: Metal name (used only for informational output).

    Returns:
        A dict containing:

        - ``margin_pct``       — margin as a percentage of the spot price.
        - ``margin_absolute``  — absolute difference (spot - AISC).
        - ``assessment``       — qualitative label.
        - ``metal``            — echo of the *metal* argument.
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
# Master scoring function
# ---------------------------------------------------------------------------

def get_aisc_score(symbol: str, metal: str, spot_price: float,
                   info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a comprehensive AISC score for a mining stock.

    Resolution order:

    1. Look up *symbol* in the merged known + JSON data.
    2. If not found and *info* is provided, estimate from financials.
    3. Fall back to the industry average for *metal*.

    The 0-100 score maps the AISC margin percentage to a bounded integer:

    - 100 corresponds to a margin of 50 % or more.
    -   0 corresponds to a margin of 0 % or less.

    Args:
        symbol: Ticker symbol (case-insensitive).
        metal: Primary metal (e.g. ``'Gold'``).
        spot_price: Current spot price in the same unit as the AISC.
        info: Optional yfinance ``.info`` dict for financial estimation.

    Returns:
        A dict with keys ``aisc_estimate``, ``aisc_source``, ``margin_pct``,
        ``score``, ``unit``, and ``assessment``.
    """
    sym = symbol.upper()
    all_data = load_aisc_data()

    aisc_value: float = 0.0
    aisc_source: str = 'industry_average'
    unit: str = AISC_INDUSTRY_AVERAGES.get(metal, {}).get('unit', '$/oz')

    # --- Step 1: known / JSON lookup -----------------------------------------
    if sym in all_data:
        record = all_data[sym]
        aisc_value = float(record['aisc'])
        aisc_source = record.get('source', 'known')
        unit = record.get('unit', unit)

        # Royalty / streaming companies — not directly comparable
        if aisc_value == 0 and record.get('unit') == 'N/A':
            return {
                'aisc_estimate': 0.0,
                'aisc_source': 'known',
                'margin_pct': 0.0,
                'score': 0,
                'unit': 'N/A',
                'assessment': 'royalty_streaming',
            }

    # --- Step 2: estimate from financials ------------------------------------
    elif info is not None:
        estimation = estimate_aisc_from_financials(info, metal)
        aisc_value = estimation['aisc_estimate']
        aisc_source = 'estimated'

    # --- Step 3: industry average fallback -----------------------------------
    else:
        industry = AISC_INDUSTRY_AVERAGES.get(metal)
        if industry:
            aisc_value = industry['average']
            unit = industry['unit']
        aisc_source = 'industry_average'

    # --- Margin & score ------------------------------------------------------
    margin = calculate_aisc_margin(aisc_value, spot_price, metal)
    margin_pct: float = margin['margin_pct']

    # Linear mapping: 0 % margin → score 0, 50 % margin → score 100
    score = int(max(0.0, min(100.0, margin_pct * 2.0)))

    return {
        'aisc_estimate': round(aisc_value, 2),
        'aisc_source': aisc_source,
        'margin_pct': margin_pct,
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

    The Net Asset Value (NAV) is computed as the discounted sum of future
    free-cash-flow from mining, where annual cash flow equals
    ``annual_production * (spot_price - aisc)``.

    If *annual_production_oz* is not provided it is estimated as
    ``reserves_oz / mine_life_years``.

    Args:
        market_cap_m: Current market capitalisation in **millions** of dollars.
        reserves_oz: Total proven + probable reserves in ounces (or the
                     relevant unit matching *spot_price* and *aisc*).
        spot_price: Current spot price per unit.
        aisc: All-In Sustaining Cost per unit.
        discount_rate: Annual discount rate (default 5 %).
        mine_life_years: Assumed remaining mine life in years (default 10).
        annual_production_oz: Annual production.  If ``None``, estimated from
                              *reserves_oz* / *mine_life_years*.

    Returns:
        A dict with keys:

        - ``nav_m``      — estimated NAV in millions of dollars.
        - ``pnav``       — Price / NAV ratio.
        - ``assessment`` — qualitative label (``'deep_value'``,
          ``'undervalued'``, ``'fair'``, or ``'premium'``).
    """
    if annual_production_oz is None:
        if mine_life_years > 0:
            annual_production_oz = reserves_oz / mine_life_years
        else:
            annual_production_oz = 0.0

    annual_cashflow = annual_production_oz * (spot_price - aisc)

    # Discounted cash flow sum
    nav = 0.0
    for t in range(1, mine_life_years + 1):
        nav += annual_cashflow / ((1.0 + discount_rate) ** t)

    # Convert to millions
    nav_m = nav / 1_000_000.0

    # P/NAV
    if nav_m > 0:
        pnav = market_cap_m / nav_m
    else:
        pnav = float('inf')

    # Assessment
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
