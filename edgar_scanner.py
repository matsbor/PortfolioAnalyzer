#!/usr/bin/env python3
"""
SEC EDGAR Scanner — Discover mining companies from public filings.
Uses the free EDGAR Full-Text Search API (no key required).
Endpoint: https://efts.sec.gov/LATEST/search-index
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

try:
    import requests

    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EFTS_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
USER_AGENT = "Alpha Miner Pro research@example.com"
REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
}

DEFAULT_KEYWORDS: List[str] = [
    "mining",
    "gold mining",
    "silver mining",
    "uranium mining",
    "mineral exploration",
    "junior miner",
]

# Module-level cache for the company-tickers JSON mapping.
_tickers_cache: Optional[Dict] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_requests() -> None:
    """Raise a clear error when the ``requests`` library is not installed."""
    if not _HAS_REQUESTS:
        raise ImportError(
            "The 'requests' package is required but not installed. "
            "Install it with: pip install requests"
        )


def _rate_limit_pause() -> None:
    """Sleep briefly to stay within EDGAR's 10-requests-per-second policy."""
    time.sleep(0.11)


def _fetch_tickers_map() -> Dict:
    """Download and cache the SEC company-tickers JSON.

    Returns a dict keyed by CIK (as string, zero-padded to 10 digits)
    with values ``{'ticker': str, 'title': str}``.
    """
    global _tickers_cache
    if _tickers_cache is not None:
        return _tickers_cache

    _ensure_requests()
    try:
        resp = requests.get(
            COMPANY_TICKERS_URL, headers=REQUEST_HEADERS, timeout=15
        )
        resp.raise_for_status()
        raw: Dict = resp.json()
    except Exception:
        _tickers_cache = {}
        return _tickers_cache

    mapped: Dict[str, Dict[str, str]] = {}
    for _index, entry in raw.items():
        cik_str = str(entry.get("cik_str", "")).zfill(10)
        ticker = entry.get("ticker", "")
        title = entry.get("title", "")
        mapped[cik_str] = {"ticker": ticker, "title": title}

    _tickers_cache = mapped
    return _tickers_cache


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_mining_filings(
    keywords: Optional[List[str]] = None,
    days_back: int = 90,
    max_results: int = 50,
) -> List[Dict[str, Optional[str]]]:
    """Search SEC EDGAR filings for mining-related companies.

    Parameters
    ----------
    keywords : list of str, optional
        Search terms to query.  Defaults to a curated list of mining terms.
    days_back : int
        How many days into the past to search (default 90).
    max_results : int
        Maximum number of filing records to return (default 50).

    Returns
    -------
    list of dict
        Each dict contains:
        ``company``, ``cik``, ``ticker`` (or None), ``filing_type``,
        ``filed_date``, ``description``.
    """
    _ensure_requests()

    if keywords is None:
        keywords = DEFAULT_KEYWORDS

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    filings: List[Dict[str, Optional[str]]] = []
    seen_keys: set = set()

    for keyword in keywords:
        if len(filings) >= max_results:
            break

        params = {
            "q": keyword,
            "dateRange": "custom",
            "startdt": start_str,
            "enddt": end_str,
            "forms": "10-K,10-Q,8-K,20-F",
        }

        _rate_limit_pause()
        try:
            resp = requests.get(
                EFTS_SEARCH_URL,
                params=params,
                headers=REQUEST_HEADERS,
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            continue

        hits = data.get("hits", data.get("filings", []))
        if isinstance(hits, dict):
            hits = hits.get("hits", [])

        for hit in hits:
            if len(filings) >= max_results:
                break

            source = hit.get("_source", hit)

            company = (
                source.get("entity_name")
                or source.get("display_names", [""])[0]
                if isinstance(source.get("display_names"), list)
                else source.get("entity_name", "Unknown")
            )
            if not company:
                company = "Unknown"

            cik_raw = source.get("entity_id") or source.get("cik") or ""
            cik = str(cik_raw).zfill(10)

            filing_type = (
                source.get("form_type")
                or source.get("file_type")
                or source.get("forms", "")
            )
            filed_date = (
                source.get("file_date")
                or source.get("period_of_report")
                or source.get("date_filed", "")
            )
            description = (
                source.get("file_description")
                or source.get("description")
                or ""
            )

            # Deduplicate on CIK + filing type + date
            dedup_key = f"{cik}|{filing_type}|{filed_date}"
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            ticker = cik_to_ticker(cik, company_name=str(company))

            filings.append(
                {
                    "company": str(company),
                    "cik": cik,
                    "ticker": ticker,
                    "filing_type": str(filing_type),
                    "filed_date": str(filed_date),
                    "description": str(description),
                }
            )

    return filings


def cik_to_ticker(cik: str, company_name: str = "") -> Optional[str]:
    """Attempt to map a CIK number to a ticker symbol.

    Parameters
    ----------
    cik : str
        The Central Index Key for the company (will be zero-padded to 10
        digits internally).
    company_name : str, optional
        Company name used as a fallback matching hint.

    Returns
    -------
    str or None
        The ticker symbol if a match is found, otherwise ``None``.
    """
    cik_padded = str(cik).zfill(10)

    try:
        tickers_map = _fetch_tickers_map()
    except Exception:
        return None

    if not tickers_map:
        return None

    # Direct CIK lookup
    entry = tickers_map.get(cik_padded)
    if entry:
        return entry.get("ticker") or None

    # Fallback: try matching by company name (case-insensitive)
    if company_name:
        company_upper = company_name.upper()
        for _cik_key, info in tickers_map.items():
            if info.get("title", "").upper() == company_upper:
                return info.get("ticker") or None

    return None


def extract_mining_tickers_from_filings(
    filings: List[Dict[str, Optional[str]]],
) -> List[str]:
    """Extract unique, sorted ticker symbols from filing results.

    Parameters
    ----------
    filings : list of dict
        Filing records as returned by :func:`search_mining_filings`.

    Returns
    -------
    list of str
        Deduplicated, uppercase, sorted list of tickers (Nones removed).
    """
    tickers: set = set()
    for filing in filings:
        raw = filing.get("ticker")
        if raw:
            tickers.add(str(raw).upper())
    return sorted(tickers)


def discover_mining_companies(
    days_back: int = 90,
    max_results: int = 100,
) -> Dict:
    """Master discovery function: search EDGAR, resolve tickers, summarise.

    Parameters
    ----------
    days_back : int
        How many days into the past to search (default 90).
    max_results : int
        Maximum filing records to retrieve (default 100).

    Returns
    -------
    dict
        ``tickers``  – list of discovered ticker strings.
        ``companies`` – full list of filing record dicts.
        ``total_filings`` – count of filings returned.
        ``search_date_range`` – human-readable date range string.
        ``source`` – always ``'SEC EDGAR EFTS'``.
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)
    date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

    filings = search_mining_filings(days_back=days_back, max_results=max_results)
    tickers = extract_mining_tickers_from_filings(filings)

    return {
        "tickers": tickers,
        "companies": filings,
        "total_filings": len(filings),
        "search_date_range": date_range,
        "source": "SEC EDGAR EFTS",
    }


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("SEC EDGAR Mining Scanner")
    print("=" * 50)
    result = discover_mining_companies(days_back=30, max_results=20)
    print(f"Date range : {result['search_date_range']}")
    print(f"Source     : {result['source']}")
    print(f"Filings    : {result['total_filings']}")
    print(f"Tickers    : {', '.join(result['tickers']) or '(none resolved)'}")
    print()
    for company in result["companies"][:10]:
        tkr = company["ticker"] or "N/A"
        print(
            f"  [{tkr:>6}]  {company['company'][:40]:<40}  "
            f"{company['filing_type']:<6}  {company['filed_date']}"
        )
