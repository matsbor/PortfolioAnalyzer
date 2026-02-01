#!/usr/bin/env python3
"""
V7.0: Sovereign Global Scan – Live mining ticker discovery.
get_all_mining_tickers() returns North American mining symbols without CSV dependency.
Uses Tiingo Search API + curated fallback.
"""
from __future__ import annotations

import os
import re
from typing import List, Set

# Optional Tiingo / requests
try:
    from tiingo import TiingoClient
    _TIINGO_AVAILABLE = True
except ImportError:
    TiingoClient = None
    _TIINGO_AVAILABLE = False

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    _REQUESTS_AVAILABLE = False

# Focused search keywords — multi-word phrases yield more relevant results
# and avoid matching non-mining companies (e.g. "gold" alone matches Gold's Gym)
_SEARCH_KEYWORDS = [
    "gold mining", "silver mining", "uranium mining", "copper mining",
    "lithium mining", "gold miner", "silver miner", "mining exploration",
    "precious metals", "mineral resources",
]

# Curated US & Canadian miners — verified as actively trading (Jan 2025).
# fetch_ticker_with_fallback() already tries .TO/.V variants for every symbol,
# so we only include the PRIMARY listing (US ticker or .TO if TSX-only).
# Removed: MAG (acquired by Newmont 2024), AUY (acquired by PAAS 2023),
#   GSV (acquired by Orla 2022), AXU (acquired by First Majestic 2022),
#   FCU (acquired by Paladin 2024), LTHM (merged into Arcadium 2024),
#   MMX.TO (acquired by Triple Flag 2023), TRQ.TO (acquired by Rio Tinto 2022),
#   CMMC.TO (acquired by Hudbay 2023), GORO/SVBL/TAO/DSV/GXU/ISO/FUU/NAC
#   (delisted/non-mining/no US data), FMC (agricultural chemicals, not mining),
#   DSVSF (OTC foreign, unreliable data), PLL (redomiciled).
# Removed duplicates: FM/LUN/IVN/CS/HBM/ERO have .TO entries already.
_TOP_MINERS: List[str] = [
    # Tier 1: Major Gold/Silver Producers (US-listed)
    "GOLD", "NEM", "AEM", "FNV", "WPM", "RGLD", "AG", "PAAS", "HL",
    "KGC", "BVN", "EGO", "IAG", "OR", "SSRM", "MUX", "GFI", "CDE",
    "EXK", "FSM", "SVM", "BTG",
    # Tier 2: Uranium (US-listed)
    "CCJ", "NXE", "DNN", "UUUU", "URG", "UEC", "EU", "URA",
    # Tier 3: Base Metals & Lithium (US-listed)
    "FCX", "SCCO", "TECK", "HBM", "LAC", "ALB", "SQM", "LIT",
    # Tier 4: Canadian TSX — only tickers with NO US listing
    "SKE.TO", "EQX.TO", "LUG.TO", "K.TO", "EDV.TO", "NGD.TO",
    "TXG.TO", "LUN.TO", "FM.TO", "HBM.TO", "IVN.TO", "CS.TO",
    "ERO.TO", "LAC.TO", "OSK.TO", "BTO.TO", "DPM.TO",
    # Tier 5: Global diversified (US ADRs/listings)
    "BHP", "RIO", "VALE", "GLNCY",
]

# Legacy alias
_FALLBACK_TICKERS: List[str] = _TOP_MINERS.copy()
_TOP_100_MINERS = _TOP_MINERS  # backward compat


def _expand_fallback() -> List[str]:
    """Return deduplicated base tickers only (no .TO/.V expansion)."""
    out: Set[str] = set()
    for t in _FALLBACK_TICKERS:
        t = str(t).strip().upper()
        if not t or t in out:
            continue
        out.add(t)
    return sorted(out)


# OTC suffixes that indicate unreliable pink-sheet / foreign / delinquent stocks
_OTC_REJECT_SUFFIXES = frozenset("FDQE")


def _is_otc_junk(ticker: str) -> bool:
    """Detect OTC pink-sheet / foreign / delinquent symbols.

    5-letter all-alpha tickers ending in F (foreign), D (delinquent),
    Q (bankruptcy), or E (delinquent SEC filing) almost never have
    reliable price data and should be excluded from the scan.
    """
    t = ticker.upper().strip()
    if "." in t:
        return False  # Has exchange suffix (.TO, .V) — not OTC
    if len(t) == 5 and t.isalpha() and t[-1] in _OTC_REJECT_SUFFIXES:
        return True
    return False


def _tiingo_search(api_key: str, query: str, limit: int = 100) -> List[dict]:
    """Call Tiingo Search API. Returns list of {ticker, name, assetType, isActive}."""
    if not _REQUESTS_AVAILABLE or not api_key:
        return []
    url = f"https://api.tiingo.com/tiingo/utilities/search/{query}"
    try:
        r = requests.get(url, params={"token": api_key}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data[:limit]
        return []
    except Exception:
        return []


def _is_north_american(ticker: str, name: str = "") -> bool:
    """Heuristic: US (no suffix), Canada (.TO, .V). Rejects OTC junk."""
    t = (ticker or "").upper()
    n = (name or "").lower()
    # Reject OTC pink-sheet symbols early
    if _is_otc_junk(t):
        return False
    if t.endswith(".TO") or t.endswith(".V"):
        return True
    if re.match(r"^[A-Z]{1,5}$", t) and not t.endswith(".L"):
        return True
    if "TSX" in n or "NYSE" in n or "NASDAQ" in n or "TORONTO" in n:
        return True
    return False


def get_all_mining_tickers(max_symbols: int = 500, use_tiingo: bool = True) -> List[str]:
    """
    North American mining symbols via Tiingo Search API + curated fallback.

    Returns deduplicated, filtered tickers. OTC pink-sheet symbols are
    excluded. The curated list is always appended (deduped) so that core
    miners are never missed even when Tiingo search is down.
    """
    api_key = (os.getenv("TIINGO_API_KEY") or "").strip()
    seen: Set[str] = set()
    result: List[str] = []

    if use_tiingo and _REQUESTS_AVAILABLE and api_key:
        try:
            for kw in _SEARCH_KEYWORDS:
                items = _tiingo_search(api_key, kw, limit=100)
                for it in items:
                    ticker = (it.get("ticker") or "").strip()
                    if not ticker:
                        continue
                    asset = (it.get("assetType") or "").lower()
                    if "stock" not in asset and asset != "stock":
                        continue
                    if it.get("isActive") is False:
                        continue
                    if _is_otc_junk(ticker):
                        continue
                    name = it.get("name") or ""
                    if not _is_north_american(ticker, name):
                        continue
                    key = ticker.upper()
                    if key in seen:
                        continue
                    seen.add(key)
                    result.append(ticker)
                    if len(result) >= max_symbols:
                        break
                if len(result) >= max_symbols:
                    break
        except Exception:
            pass

    # Always merge the curated list (dedup against Tiingo results)
    for t in _TOP_MINERS:
        base = re.sub(r"\.(TO|V|A|B)$", "", t.upper())
        if t.upper() in seen or base in seen:
            continue
        seen.add(t.upper())
        seen.add(base)
        result.append(t)
        if len(result) >= max_symbols:
            break

    # Safety: if everything failed, use curated list directly
    if not result:
        for t in _TOP_MINERS:
            result.append(t)
            if len(result) >= max_symbols:
                break

    return result[:max_symbols]
