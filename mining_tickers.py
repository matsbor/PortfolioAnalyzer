#!/usr/bin/env python3
"""
V7.0: Sovereign Global Scan – Live mining ticker discovery.
get_all_mining_tickers() returns 1,000+ North American mining symbols without CSV dependency.
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

# North American mining keywords for Tiingo Search
_SEARCH_KEYWORDS = [
    "gold", "silver", "uranium", "mining", "mineral", "metals", "copper",
    "lithium", "cobalt", "ore", "exploration", "miner", "resources",
]

# V7.3: Top 100 US & Canadian Miners (Hard-Mapped Fallback)
# Priority order: Major producers first, then developers, then explorers
_TOP_100_MINERS: List[str] = [
    # Tier 1: Major Gold/Silver Producers (US & Canada)
    "GOLD", "NEM", "AEM", "FNV", "WPM", "RGLD", "AG", "PAAS", "HL", "MAG",
    "KGC", "AUY", "BVN", "EGO", "IAG", "OR", "SSRM", "MUX", "GFI", "CDE",
    "EXK", "FSM", "GSV", "GORO", "SVBL", "TAO", "DSV", "SVM", "AXU", "BTG",
    # Tier 2: Uranium Producers (CCJ, NXE prioritized)
    "CCJ", "NXE", "DNN", "UUUU", "URG", "UEC", "EU", "PEN", "NAC", "GXU",
    "ISO", "LAM", "FCU", "FUU", "GLO", "FMC", "U", "URA",
    # Tier 3: Base Metals & Diversified
    "FCX", "SCCO", "TECK", "FM", "HBM", "LUN", "ERO", "IVN", "CS",
    "LAC", "LTHM", "ALB", "SQM", "LITM", "PLL", "LIT",
    # Tier 4: Canadian TSX Listings
    "SKE.TO", "EQX.TO", "LUG.TO", "OR.TO", "WPM.TO", "FNV.TO", "AEM.TO",
    "K.TO", "EDV.TO", "G.TO", "SA.TO", "NGD.TO", "B2G.TO", "AR.TO",
    "MMX.TO", "TXG.TO", "LGD.TO", "ORR.TO", "LUN.TO", "FM.TO", "HBM.TO",
    "TECK.A", "TECK.B", "TRQ.TO", "IVN.TO", "CS.TO", "ERO.TO", "LAC.TO",
    "CMMC.TO",
    # Tier 5: OTC & Additional (diversified global miners)
    "DSVSF", "BHP", "RIO", "VALE", "GLNCY",
]

# Legacy fallback (kept for backward compatibility)
_FALLBACK_TICKERS: List[str] = _TOP_100_MINERS.copy()
# NOTE: We no longer expand with .TO/.V variants because
# fetch_ticker_with_fallback() already tries geography-first variants
# (plain, TSX:, .TO, OTC) for every symbol. Adding .TO/.V duplicates
# just doubles API calls and inflates skip counts.
def _expand_fallback() -> List[str]:
    """Return deduplicated base tickers only (no .TO/.V expansion)."""
    out: Set[str] = set()
    for t in _FALLBACK_TICKERS:
        t = str(t).strip().upper()
        if not t or t in out:
            continue
        out.add(t)
    return sorted(out)


def _tiingo_search(api_key: str, query: str, limit: int = 200) -> List[dict]:
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
    """Heuristic: US (no suffix), Canada (.TO, .V), or OTC-style."""
    t = (ticker or "").upper()
    n = (name or "").lower()
    if t.endswith(".TO") or t.endswith(".V"):
        return True
    if re.match(r"^[A-Z]{1,5}$", t) and not t.endswith(".L"):
        return True
    if "TSX" in n or "NYSE" in n or "NASDAQ" in n or "TORONTO" in n:
        return True
    return False


def get_all_mining_tickers(max_symbols: int = 2000, use_tiingo: bool = True) -> List[str]:
    """
    V7.3: Live list of 1,000+ North American mining symbols. No CSV required.
    
    V7.3 Update: NOT restricted - always returns Top 100 hard-mapped list if Tiingo fails.
    Uses Tiingo Search for keywords (gold, silver, mining, uranium, etc.), then
    Top 100 US & Canadian Miners fallback (AEM, GOLD, FCX, CCJ, NXE, PAAS, etc.).
    
    Returns deduplicated, North America–filtered tickers, prioritized by tier.
    """
    api_key = (os.getenv("TIINGO_API_KEY") or "").strip()
    seen: Set[str] = set()
    result: List[str] = []

    # V7.3: Try Tiingo first (if available and enabled)
    if use_tiingo and _REQUESTS_AVAILABLE and api_key:
        try:
            for kw in _SEARCH_KEYWORDS:
                items = _tiingo_search(api_key, kw, limit=250)
                for it in items:
                    ticker = (it.get("ticker") or "").strip()
                    if not ticker:
                        continue
                    asset = (it.get("assetType") or "").lower()
                    if "stock" not in asset and asset != "stock":
                        continue
                    if it.get("isActive") is False:
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
                        return result[:max_symbols]
        except Exception:
            # V7.3: Tiingo failed - fall through to Top 100 fallback
            pass

    # Add Top 100 hard-mapped list (dedup against Tiingo search results)
    # For each ticker, also skip if its base form is already present
    # (e.g., skip "AEM.TO" if "AEM" was already found by Tiingo search)
    for t in _TOP_100_MINERS:
        base = re.sub(r"\.(TO|V|A|B)$", "", t.upper())
        if t.upper() in seen or base in seen:
            continue
        seen.add(t.upper())
        seen.add(base)  # Mark base as seen to prevent .TO/.V duplicates later
        result.append(t)
        if len(result) >= max_symbols:
            break

    # V7.4: Safety check - always return at least Top 100 if result is empty
    if len(result) == 0:
        # Fallback to Top 100 if everything failed
        for t in _TOP_100_MINERS:
            if len(result) >= max_symbols:
                break
            result.append(t)
    
    return result[:max_symbols]
