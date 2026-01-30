#!/usr/bin/env python3
"""
Insider Transaction Tracker — Detect insider buying/selling signals.
Uses yfinance insider_transactions data when available.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Transaction type classification
# ---------------------------------------------------------------------------

_PURCHASE_KEYWORDS = ['purchase', 'buy', 'acquisition']
_SALE_KEYWORDS = ['sale', 'sell', 'disposition']
_OPTION_KEYWORDS = ['option', 'exercise', 'conversion']


def _classify_transaction(text: str) -> str:
    """Classify a transaction description into Purchase, Sale, or Option Exercise."""
    lower = text.lower() if text else ''
    for kw in _PURCHASE_KEYWORDS:
        if kw in lower:
            return 'Purchase'
    for kw in _SALE_KEYWORDS:
        if kw in lower:
            return 'Sale'
    for kw in _OPTION_KEYWORDS:
        if kw in lower:
            return 'Option Exercise'
    return 'Unknown'


def _safe_float(value, default: float = 0.0) -> float:
    """Convert a value to float, returning *default* on failure."""
    try:
        result = float(value)
        if pd.isna(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    """Convert a value to int, returning *default* on failure."""
    try:
        result = int(float(value))
        return result
    except (TypeError, ValueError):
        return default


def _safe_str(value, default: str = '') -> str:
    """Convert a value to str, returning *default* for NaN / None."""
    if value is None:
        return default
    if isinstance(value, float) and pd.isna(value):
        return default
    return str(value)


# ---------------------------------------------------------------------------
# 1. fetch_insider_transactions
# ---------------------------------------------------------------------------

def fetch_insider_transactions(symbol: str, days: int = 90) -> dict:
    """Fetch insider transactions for *symbol* over the last *days* days.

    Attempts to pull data from yfinance.  When the library is missing or the
    data is empty / unavailable the function still returns a well-formed dict
    with ``source`` set to ``'unavailable'``.

    Returns
    -------
    dict
        Keys documented in the module docstring.
    """
    unavailable_result = {
        'has_insider_buying': None,
        'has_insider_selling': None,
        'net_shares': 0,
        'buy_count': 0,
        'sell_count': 0,
        'total_buy_value': 0.0,
        'total_sell_value': 0.0,
        'net_value': 0.0,
        'largest_purchase': 0.0,
        'largest_sale': 0.0,
        'insider_names': [],
        'transactions': [],
        'source': 'unavailable',
        'last_updated': datetime.utcnow().isoformat(),
    }

    if not YFINANCE_AVAILABLE:
        return unavailable_result

    try:
        ticker = yf.Ticker(symbol)
        txns = ticker.insider_transactions
    except Exception:
        return unavailable_result

    # yfinance may return None, an empty DataFrame, or something unexpected.
    if txns is None or (isinstance(txns, pd.DataFrame) and txns.empty):
        return unavailable_result
    if not isinstance(txns, pd.DataFrame):
        return unavailable_result

    # ----- Determine the date column -----
    date_col = None
    for candidate in ['Start Date', 'Date', 'startDate', 'date']:
        if candidate in txns.columns:
            date_col = candidate
            break

    if date_col is None:
        # No recognisable date column – return unavailable.
        return unavailable_result

    # Convert to datetime and filter to the requested window.
    txns = txns.copy()
    txns[date_col] = pd.to_datetime(txns[date_col], errors='coerce')
    cutoff = datetime.utcnow() - timedelta(days=days)
    txns = txns.dropna(subset=[date_col])
    txns = txns[txns[date_col] >= cutoff]

    if txns.empty:
        return {
            **unavailable_result,
            'has_insider_buying': False,
            'has_insider_selling': False,
            'source': 'yfinance',
        }

    # ----- Identify column names flexibly -----
    def _find_col(candidates, default=None):
        for c in candidates:
            if c in txns.columns:
                return c
        return default

    text_col = _find_col(['Text', 'Transaction', 'transaction', 'text'])
    shares_col = _find_col(['Shares', 'shares', 'Share'])
    value_col = _find_col(['Value', 'value', 'Cost', 'cost'])
    insider_col = _find_col(
        ['Insider', 'insider', 'Insider Trading', 'Name', 'name',
         'Insider Name', 'insiderName']
    )

    # ----- Walk through rows and accumulate stats -----
    buy_count = 0
    sell_count = 0
    total_buy_value = 0.0
    total_sell_value = 0.0
    total_buy_shares = 0
    total_sell_shares = 0
    largest_purchase = 0.0
    largest_sale = 0.0
    insider_names: List[str] = []
    records: List[dict] = []

    for _, row in txns.iterrows():
        raw_text = _safe_str(row.get(text_col)) if text_col else ''
        category = _classify_transaction(raw_text)

        shares = _safe_int(row.get(shares_col) if shares_col else 0)
        value = _safe_float(row.get(value_col) if value_col else 0.0)
        name = _safe_str(row.get(insider_col) if insider_col else '')

        if name and name not in insider_names:
            insider_names.append(name)

        record = {
            'date': str(row.get(date_col, '')),
            'insider': name,
            'type': category,
            'shares': shares,
            'value': abs(value),
            'text': raw_text,
        }
        records.append(record)

        abs_value = abs(value)
        abs_shares = abs(shares)

        if category == 'Purchase':
            buy_count += 1
            total_buy_value += abs_value
            total_buy_shares += abs_shares
            if abs_value > largest_purchase:
                largest_purchase = abs_value
        elif category == 'Sale':
            sell_count += 1
            total_sell_value += abs_value
            total_sell_shares += abs_shares
            if abs_value > largest_sale:
                largest_sale = abs_value
        # Option Exercise transactions are recorded but do not shift counts.

    net_shares = total_buy_shares - total_sell_shares
    net_value = total_buy_value - total_sell_value

    return {
        'has_insider_buying': buy_count > 0,
        'has_insider_selling': sell_count > 0,
        'net_shares': net_shares,
        'buy_count': buy_count,
        'sell_count': sell_count,
        'total_buy_value': round(total_buy_value, 2),
        'total_sell_value': round(total_sell_value, 2),
        'net_value': round(net_value, 2),
        'largest_purchase': round(largest_purchase, 2),
        'largest_sale': round(largest_sale, 2),
        'insider_names': insider_names,
        'transactions': records,
        'source': 'yfinance',
        'last_updated': datetime.utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# 2. fetch_insider_transactions_batch
# ---------------------------------------------------------------------------

def fetch_insider_transactions_batch(
    symbols: List[str], days: int = 90
) -> Dict[str, dict]:
    """Fetch insider transactions for every symbol in *symbols*.

    Processing is sequential (no threading needed for typical portfolio sizes).

    Returns
    -------
    dict
        Mapping of symbol -> fetch_insider_transactions result.
    """
    results: Dict[str, dict] = {}
    for sym in symbols:
        results[sym] = fetch_insider_transactions(sym, days=days)
    return results


# ---------------------------------------------------------------------------
# 3. calculate_insider_signal
# ---------------------------------------------------------------------------

def calculate_insider_signal(insider_data: dict) -> dict:
    """Compute an insider-activity signal from the output of
    ``fetch_insider_transactions``.

    Score ranges
    ------------
    90-100  Strong Buy  — 3+ purchases, net buying > $100k
    70-89   Buy         — 1-2 purchases, net buying > $10k
    50      Neutral     — no activity or data unavailable
    20-49   Sell        — net selling > $50k
    0-19    Strong Sell — 3+ sales, net selling > $200k

    Returns
    -------
    dict
        ``{'score': int, 'signal': str, 'reasons': list[str]}``
    """
    reasons: List[str] = []

    if insider_data.get('source') == 'unavailable':
        reasons.append('Insider data unavailable')
        return {'score': 50, 'signal': 'Neutral', 'reasons': reasons}

    buy_count = insider_data.get('buy_count', 0)
    sell_count = insider_data.get('sell_count', 0)
    net_value = insider_data.get('net_value', 0.0)
    total_buy = insider_data.get('total_buy_value', 0.0)
    total_sell = insider_data.get('total_sell_value', 0.0)

    # No transactions at all → Neutral
    if buy_count == 0 and sell_count == 0:
        reasons.append('No insider transactions in period')
        return {'score': 50, 'signal': 'Neutral', 'reasons': reasons}

    # ----- Strong Buy -----
    if buy_count >= 3 and net_value > 100_000:
        score = min(100, 90 + min(buy_count - 3, 10))
        reasons.append(f'{buy_count} insider purchases')
        reasons.append(f'Net buying ${net_value:,.0f}')
        return {'score': score, 'signal': 'Strong Buy', 'reasons': reasons}

    # ----- Buy -----
    if buy_count >= 1 and net_value > 10_000:
        # Scale 70-89 based on value magnitude
        value_factor = min((net_value - 10_000) / 90_000, 1.0)
        score = 70 + int(value_factor * 19)
        reasons.append(f'{buy_count} insider purchase(s)')
        reasons.append(f'Net buying ${net_value:,.0f}')
        return {'score': score, 'signal': 'Buy', 'reasons': reasons}

    # ----- Strong Sell -----
    if sell_count >= 3 and (-net_value) > 200_000:
        score = max(0, 19 - min(sell_count - 3, 19))
        reasons.append(f'{sell_count} insider sales')
        reasons.append(f'Net selling ${abs(net_value):,.0f}')
        return {'score': score, 'signal': 'Strong Sell', 'reasons': reasons}

    # ----- Sell -----
    if (-net_value) > 50_000:
        value_factor = min(((-net_value) - 50_000) / 150_000, 1.0)
        score = 49 - int(value_factor * 29)
        reasons.append(f'{sell_count} insider sale(s)')
        reasons.append(f'Net selling ${abs(net_value):,.0f}')
        return {'score': score, 'signal': 'Sell', 'reasons': reasons}

    # ----- Mild buying (below $10k threshold) -----
    if net_value > 0:
        reasons.append(f'{buy_count} small purchase(s), ${total_buy:,.0f} total')
        return {'score': 55, 'signal': 'Neutral', 'reasons': reasons}

    # ----- Mild selling (below $50k threshold) -----
    if net_value < 0:
        reasons.append(f'{sell_count} small sale(s), ${total_sell:,.0f} total')
        return {'score': 45, 'signal': 'Neutral', 'reasons': reasons}

    # Exact zero net value (buys and sells balanced)
    reasons.append('Balanced insider activity')
    return {'score': 50, 'signal': 'Neutral', 'reasons': reasons}


# ---------------------------------------------------------------------------
# 4. get_insider_summary
# ---------------------------------------------------------------------------

def get_insider_summary(insider_data: dict) -> str:
    """Return a concise one-line summary of insider activity.

    Examples
    --------
    - ``"3 buys ($45K net) by CEO, CFO"``
    - ``"No insider activity"``
    - ``"Data unavailable"``
    """
    if insider_data.get('source') == 'unavailable':
        return 'Data unavailable'

    buy_count = insider_data.get('buy_count', 0)
    sell_count = insider_data.get('sell_count', 0)

    if buy_count == 0 and sell_count == 0:
        return 'No insider activity'

    parts: List[str] = []

    # Describe buys
    if buy_count > 0:
        parts.append(f'{buy_count} buy{"s" if buy_count != 1 else ""}')

    # Describe sells
    if sell_count > 0:
        parts.append(f'{sell_count} sale{"s" if sell_count != 1 else ""}')

    # Net value
    net_value = insider_data.get('net_value', 0.0)
    abs_net = abs(net_value)
    if abs_net >= 1_000_000:
        value_str = f'${abs_net / 1_000_000:.1f}M'
    elif abs_net >= 1_000:
        value_str = f'${abs_net / 1_000:.0f}K'
    else:
        value_str = f'${abs_net:,.0f}'

    direction = 'net' if net_value >= 0 else 'net sold'
    parts_str = ', '.join(parts)
    summary = f'{parts_str} ({value_str} {direction})'

    # Append insider names (truncated to 3)
    names = insider_data.get('insider_names', [])
    if names:
        display_names = names[:3]
        name_str = ', '.join(display_names)
        if len(names) > 3:
            name_str += f' +{len(names) - 3} more'
        summary += f' by {name_str}'

    return summary


# ---------------------------------------------------------------------------
# 5. update_portfolio_insider_flags
# ---------------------------------------------------------------------------

def update_portfolio_insider_flags(
    portfolio_df: pd.DataFrame, days: int = 90
) -> pd.DataFrame:
    """Replace hardcoded Insider_Buying_90d flags with live data.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        Must contain a ``Symbol`` column.  An ``Insider_Buying_90d`` column
        will be created or overwritten.
    days : int
        Look-back window in calendar days.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with updated columns:
        - ``Insider_Buying_90d`` (bool)
        - ``Insider_Score`` (int, 0-100)
        - ``Insider_Summary`` (str)
    """
    df = portfolio_df.copy()

    if 'Symbol' not in df.columns:
        # Cannot proceed without symbols – add empty columns and return.
        df['Insider_Buying_90d'] = False
        df['Insider_Score'] = 50
        df['Insider_Summary'] = 'No symbol column'
        return df

    symbols = df['Symbol'].dropna().unique().tolist()
    batch = fetch_insider_transactions_batch(symbols, days=days)

    buying_flags: List[bool] = []
    scores: List[int] = []
    summaries: List[str] = []

    for _, row in df.iterrows():
        sym = row.get('Symbol')
        data = batch.get(sym)
        if data is None:
            buying_flags.append(False)
            scores.append(50)
            summaries.append('Data unavailable')
            continue

        has_buying = data.get('has_insider_buying')
        buying_flags.append(bool(has_buying) if has_buying is not None else False)

        signal = calculate_insider_signal(data)
        scores.append(signal['score'])

        summaries.append(get_insider_summary(data))

    df['Insider_Buying_90d'] = buying_flags
    df['Insider_Score'] = scores
    df['Insider_Summary'] = summaries

    return df
