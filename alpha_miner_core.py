#!/usr/bin/env python3
"""
ALPHA MINER CORE - Import-safe computation functions
Shared logic for both Streamlit app and backtest runner.
No Streamlit UI execution at import time.
"""
import pandas as pd
import numpy as np
import datetime
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Optional yfinance import (graceful fallback)
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

# MODEL GOVERNANCE - Define model roles and caps
# V8.0: Aligned with actual model implementations in calculate_alpha_models()
MODEL_ROLES = {
    'Alpha': {
        'models': [
            'M1_Momentum', 'M2_Value', 'M3_Survival', 'M4_Dilution',
            'M5_Liquidity', 'M6_RelStrength', 'M7_SMC', 'M8_StageFit',
            'M9_VolMomentum', 'M10_TA', 'M11_FA',
        ],
        'role': 'recommend',
        'weight_cap': 100.0
    },
    'Risk': {
        'models': ['Sell_Risk', 'Data_Confidence'],
        'role': 'veto',
        'threshold': 60.0
    },
    'Capital_Structure': {
        'models': ['Dilution_Risk', 'Financing_Overhang'],
        'role': 'veto',
        'threshold': 80.0
    },
    'Liquidity': {
        'models': ['Liquidity_Tier', 'Days_To_Exit'],
        'role': 'veto',
        'threshold': 'L0'
    },
    'Regime': {
        'models': ['Tape_Gate', 'Metal_Regime'],
        'role': 'throttle',
        'threshold': None
    }
}

# RISK PROFILES
RISK_PROFILES = {
    "Aggressive": {
        "max_pos_pct": {"L3": 12.0, "L2": 9.0, "L1": 6.0, "L0": 1.0},
        "min_data_confidence_for_buys": 55,
        "strict_downgrade_confidence": 50,
        "sell_risk_floor": 15,
    },
    "Balanced": {
        "max_pos_pct": {"L3": 10.0, "L2": 7.5, "L1": 5.0, "L0": 1.0},
        "min_data_confidence_for_buys": 65,
        "strict_downgrade_confidence": 60,
        "sell_risk_floor": 20,
    },
    "Defensive": {
        "max_pos_pct": {"L3": 7.5, "L2": 6.0, "L1": 4.0, "L0": 1.0},
        "min_data_confidence_for_buys": 75,
        "strict_downgrade_confidence": 70,
        "sell_risk_floor": 25,
    },
}

PORTFOLIO_SIZE = 200000  # $200k portfolio

# Evidence pack directory
EVIDENCE_DIR = Path.home() / '.alpha_miner_evidence_packs'
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)


def get_risk_profile_preset(name: str) -> dict:
    """Return risk profile preset dict with safe default."""
    preset = RISK_PROFILES.get(name) or RISK_PROFILES.get("Balanced")
    # copy so callers can modify without mutating global
    out = dict(preset)
    out["name"] = name if name in RISK_PROFILES else "Balanced"
    return out


def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def validate_data_invariants(df: pd.DataFrame, arg2=None, arg3=None):
    """
    Backwards compatible:
      - validate_data_invariants(df, news_cache)
      - validate_data_invariants(df, alpha_models_storage, news_cache)

    Returns: dict with keys: ok(bool), errors(list[str]), warnings(list[str]), per_symbol(dict)
    """

    # If only 2 args were provided: (df, news_cache)
    if arg3 is None:
        alpha_models_storage = {}
        news_cache = arg2 or {}
    else:
        # 3 args: (df, alpha_models_storage, news_cache)
        alpha_models_storage = arg2 or {}
        news_cache = arg3 or {}

    errors, warnings = [], []
    per_symbol = {}

    # Portfolio-level checks
    required_cols = [
        'Symbol','Price','Market_Value','Pct_Portfolio','Alpha_Score','Sell_Risk_Score',
        'Data_Confidence','Dilution_Risk_Score','Liq_tier_code'
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    # Alpha weights sum check (models that exist)
    for sym, models in (alpha_models_storage or {}).items():
        try:
            total_w = 0.0
            for _, v in models.items():
                # stored as weighted contributions already
                total_w += float(v)
            if not (0 <= total_w <= 100):
                warnings.append(f"{sym}: alpha contribution sum out of range: {total_w:.1f}")
        except Exception:
            warnings.append(f"{sym}: could not validate alpha model contributions")

    # Row-level checks
    for _, r in df.iterrows():
        sym = str(r.get('Symbol','')).strip()
        pe = []
        pw = []
        try:
            price = float(r.get('Price', 0) or 0)
            if price <= 0:
                pw.append('price<=0')
            mv = float(r.get('Market_Value', 0) or 0)
            if mv < 0:
                pe.append('market_value<0')
            pct = float(r.get('Pct_Portfolio', 0) or 0)
            if pct < 0 or pct > 100:
                pe.append('pct_portfolio_out_of_range')
            burn = float(r.get('burn', 1) or 1)
            cash = float(r.get('cash', 0) or 0)
            if burn <= 0:
                pw.append('burn<=0 (runway invalid)')
            if cash < 0:
                pe.append('cash<0')

            for col in ['Sell_Risk_Score','Dilution_Risk_Score','Data_Confidence','SMC_Score']:
                if col in df.columns:
                    v = float(r.get(col, 0) or 0)
                    if v < 0 or v > 100:
                        pw.append(f"{col}_out_of_range")

            # News timestamps sanity
            items = (news_cache or {}).get(sym, [])
            if items:
                bad_ts = 0
                for it in items:
                    ts = it.get('timestamp', 0) or 0
                    if ts and (ts < 946684800 or ts > 1893456000):
                        bad_ts += 1
                if bad_ts:
                    pw.append(f"{bad_ts} news items have invalid timestamps")
        except Exception as e:
            pw.append(f"row_validation_exception:{e}")

        if pe or pw:
            per_symbol[sym] = {'errors': pe, 'warnings': pw}

    ok = (len(errors) == 0)
    return {'ok': ok, 'errors': errors, 'warnings': warnings, 'per_symbol': per_symbol}


def enforce_strict_mode(df: pd.DataFrame, validation_results: dict, risk_profile: str, strict_mode: bool = True):
    """Downgrade actions when inputs/data quality aren't strong enough."""
    if not strict_mode:
        return df, []

    profile = RISK_PROFILES.get(risk_profile, RISK_PROFILES['Balanced'])
    min_conf = profile['min_data_confidence_for_buys']
    downgrades = []

    per_symbol = (validation_results or {}).get('per_symbol', {})

    def _is_buy_action(a: str) -> bool:
        a = (a or '').upper()
        return ('BUY' in a) or ('ADD' in a) or ('ACCUMULATE' in a)

    df2 = df.copy()
    for i, r in df2.iterrows():
        sym = r.get('Symbol','')
        action = r.get('Action','')
        conf = float(r.get('Data_Confidence', 0) or 0)
        issues = per_symbol.get(sym, {})
        has_errors = bool(issues.get('errors'))
        has_warnings = bool(issues.get('warnings'))

        if _is_buy_action(action) and (conf < min_conf or has_errors):
            df2.at[i, 'Action'] = '⚪ HOLD'
            df2.at[i, 'Confidence'] = min(float(r.get('Confidence', 60) or 60), profile['strict_downgrade_confidence'])
            rs = list(r.get('Reasoning') or [])
            rs = rs if isinstance(rs, list) else [str(rs)]
            rs.insert(0, f"STRICT MODE: downgraded due to data confidence ({conf:.0f}) or validation errors")
            df2.at[i, 'Reasoning'] = rs
            downgrades.append(sym)
        elif _is_buy_action(action) and has_warnings and conf < (min_conf + 10):
            # soften but don't fully block
            df2.at[i, 'Action'] = '🔵 ADD ⚠️'
            rs = list(r.get('Reasoning') or [])
            rs = rs if isinstance(rs, list) else [str(rs)]
            rs.insert(0, "STRICT MODE: caution due to validation warnings")
            df2.at[i, 'Reasoning'] = rs

    return df2, downgrades


def create_evidence_pack(
    df: pd.DataFrame,
    portfolio_input: pd.DataFrame,
    cash: float,
    macro_regime: dict,
    news_cache: dict,
    alpha_breakdown_storage: dict,
    sell_triggers_storage: dict,
    dilution_factors_storage: dict,
    conf_breakdown_storage: dict,
    meta: dict | None = None,
    tape_gate: dict | None = None,
    version: str = "2.2-PRODUCTION",
    version_date: str = "2026-01-16",
):
    """Create a replayable, self-contained evidence pack (zero network calls needed to render)."""
    pack = {
        'evidence_pack_id': f"ep_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'created_at_utc': _now_iso(),
        'app_version': version,
        'app_version_date': version_date,
        'meta': meta or {},
        'inputs': {
            'portfolio': portfolio_input.to_dict(orient='records'),
            'cash': float(cash),
            'freeze_time': meta.get('freeze_time', False) if meta else False,
            'disable_sector_fallback_news': meta.get('disable_sector_fallback_news', False) if meta else False,
            'disable_inferred_fundamentals': meta.get('disable_inferred_fundamentals', False) if meta else False,
        },
        'macro_regime': macro_regime or {},
        'tape_gate': tape_gate or {},
        'results': df.to_dict(orient='records'),
        'caches': {
            'news_cache': news_cache or {},
            'alpha_breakdown_storage': alpha_breakdown_storage or {},
            'sell_triggers_storage': sell_triggers_storage or {},
            'dilution_factors_storage': dilution_factors_storage or {},
            'conf_breakdown_storage': conf_breakdown_storage or {},
        },
    }
    return pack


def save_evidence_pack(pack: dict) -> Path:
    ep_id = pack.get('evidence_pack_id', f"ep_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    path = EVIDENCE_DIR / f"{ep_id}.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(pack, f, indent=2)
    return path


def list_evidence_packs():
    return sorted(EVIDENCE_DIR.glob('ep_*.json'), key=lambda x: x.stat().st_mtime, reverse=True)


def load_evidence_pack(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_liquidity_metrics(ticker, hist_data, current_price, current_position_value, portfolio_size=PORTFOLIO_SIZE):
    """
    Calculate liquidity metrics and tier classification.
    Returns UNKNOWN tier if volume data is missing or invalid (NaN/zeros).
    """
    result = {
        'tier_code': 'UNKNOWN',
        'tier_name': 'Unknown Liquidity',
        'dollar_vol_20d': 0,
        'avg_vol_20d': 0,
        'max_position_pct': 0.0,  # Block new buys by default for UNKNOWN
        'days_to_exit': 99,
        'exit_flag': '❓ UNKNOWN',
        'volume_valid': False,
        'liquidity_reason': 'Volume data missing or invalid'
    }
    
    if hist_data.empty or len(hist_data) < 20:
        result['liquidity_reason'] = 'Insufficient historical data (< 20 days)'
        return result
    
    # Check if Volume column exists and has valid data
    if 'Volume' not in hist_data.columns:
        result['liquidity_reason'] = 'Volume column missing from historical data'
        return result
    
    try:
        recent = hist_data.tail(20)
        # Check for NaN or all-zero volume
        vol_series = recent['Volume']
        if vol_series.isna().all() or (vol_series == 0).all():
            result['liquidity_reason'] = 'Volume data is all NaN or zero'
            return result
        
        # Calculate average volume, ignoring NaN and zeros
        valid_vol = vol_series[vol_series > 0].dropna()
        if len(valid_vol) == 0:
            result['liquidity_reason'] = 'No valid (non-zero) volume data'
            return result
        
        avg_vol = valid_vol.mean()
        result['avg_vol_20d'] = avg_vol
        result['volume_valid'] = True
        
        # Dollar volume
        dollar_vol = avg_vol * current_price
        result['dollar_vol_20d'] = dollar_vol
        
        # Days to exit (assume 10% daily volume limit)
        if dollar_vol > 0:
            days_to_exit = current_position_value / (dollar_vol * 0.10)
            result['days_to_exit'] = min(days_to_exit, 99)
        
        # Tier classification (only if we have valid volume)
        if dollar_vol >= 500000:
            result['tier_code'] = 'L3'
            result['tier_name'] = 'Highly Liquid'
            result['max_position_pct'] = 10.0
            result['exit_flag'] = '✅ L3'
            result['liquidity_reason'] = f'Dollar volume ${dollar_vol:,.0f}/day (L3)'
        elif dollar_vol >= 200000:
            result['tier_code'] = 'L2'
            result['tier_name'] = 'Liquid'
            result['max_position_pct'] = 7.5
            result['exit_flag'] = '🟢 L2'
            result['liquidity_reason'] = f'Dollar volume ${dollar_vol:,.0f}/day (L2)'
        elif dollar_vol >= 50000:
            result['tier_code'] = 'L1'
            result['tier_name'] = 'Moderate'
            result['max_position_pct'] = 5.0
            result['exit_flag'] = '🟡 L1'
            result['liquidity_reason'] = f'Dollar volume ${dollar_vol:,.0f}/day (L1)'
        else:
            result['tier_code'] = 'L0'
            result['tier_name'] = 'Illiquid'
            result['max_position_pct'] = 1.0
            result['exit_flag'] = '⚠️ L0'
            result['liquidity_reason'] = f'Dollar volume ${dollar_vol:,.0f}/day (L0)'
    
    except Exception as e:
        result['liquidity_reason'] = f'Error calculating liquidity: {str(e)}'
    
    return result


def calculate_data_confidence(fund_dict, info_dict, inferred_flags):
    """Calculate confidence in our data"""
    score = 100
    breakdown = []
    
    # Penalize defaults
    if fund_dict.get('burn_source') == 'default':
        score -= 30
        breakdown.append("⚠️ Using default burn rate (-30)")
    elif fund_dict.get('burn_source') == 'netincome':
        score -= 10
        breakdown.append("⚠️ Burn from net income, not cashflow (-10)")
    
    # Penalize inferred data
    if inferred_flags.get('stage_inferred', True):
        score -= 15
        breakdown.append("⚠️ Stage inferred from assets (-15)")
    
    if inferred_flags.get('metal_inferred', True):
        score -= 10
        breakdown.append("⚠️ Metal type inferred (-10)")
    
    # Reward real data
    if info_dict.get('totalRevenue'):
        score += 10
        breakdown.append("✅ Has revenue data (+10)")
    
    if info_dict.get('totalCash'):
        score += 10
        breakdown.append("✅ Has cash data (+10)")
    
    score = max(0, min(100, score))
    
    if score >= 80:
        verdict = "HIGH"
    elif score >= 60:
        verdict = "MEDIUM"
    else:
        verdict = "LOW"
    
    return {'score': score, 'verdict': verdict, 'breakdown': breakdown}


def calculate_dilution_risk(runway, stage, drawdown_90d, news_items, 
                           cash_missing, burn_missing, insider_buying):
    """Calculate dilution risk"""
    score = 0
    factors = []
    
    # Runway
    if runway < 6:
        score += 40
        factors.append(f"💀 Runway {runway:.1f}mo < 6mo CRITICAL (+40)")
    elif runway < 12:
        score += 20
        factors.append(f"⚠️ Runway {runway:.1f}mo < 12mo (+20)")
    else:
        factors.append(f"✅ Runway {runway:.1f}mo adequate")
    
    # Stage
    if stage == 'Explorer':
        score += 15
        factors.append("⚠️ Explorer stage (+15)")
    
    # Drawdown
    if drawdown_90d > 40:
        score += 15
        factors.append(f"⚠️ Drawdown {drawdown_90d:.0f}% > 40% (+15)")
    
    # News indicators
    low_cash_news = any('low cash' in item.get('title', '').lower() or 
                       'needs cash' in item.get('title', '').lower() 
                       for item in news_items)
    
    financing_news = any('financing' in item.get('title', '').lower() or 
                        'placement' in item.get('title', '').lower() or
                        'offering' in item.get('title', '').lower()
                        for item in news_items)
    
    if low_cash_news:
        score += 20
        factors.append("💀 'Low cash' in news (+20)")
    
    if financing_news:
        score += 15
        factors.append("⚠️ Financing mentioned in news (+15)")
    
    # Data quality
    if cash_missing:
        score += 10
        factors.append("⚠️ Cash data missing (+10)")
    
    if burn_missing:
        score += 10
        factors.append("⚠️ Burn rate uncertain (+10)")
    
    # Insider buying reduces risk
    if insider_buying:
        score = max(0, score - 15)
        factors.append("✅ Insider buying (-15)")
    
    score = min(100, score)
    
    if score >= 70:
        verdict = "CRITICAL"
    elif score >= 50:
        verdict = "HIGH"
    elif score >= 30:
        verdict = "MODERATE"
    else:
        verdict = "LOW"
    
    return {'score': score, 'verdict': verdict, 'factors': factors}


def normalize_timestamp(ts):
    """Normalize timestamp to valid Unix timestamp"""
    if ts is None or ts <= 0:
        return None
    
    # Handle milliseconds
    if ts > 1e12:
        ts = ts / 1000
    
    # Validate range (2000-2030)
    if ts < 946684800 or ts > 1893456000:
        return None
    
    return ts


def tag_news(news_items):
    """Tag news items based on content"""
    for item in news_items:
        title_lower = item.get('title', '').lower()
        tags = []
        
        if any(word in title_lower for word in ['financing', 'placement', 'offering', 'capital raise']):
            tags.append('💰')
        if any(word in title_lower for word in ['drill', 'exploration', 'discovers', 'intercepts']):
            tags.append('🔍')
        if any(word in title_lower for word in ['production', 'produces', 'mining']):
            tags.append('⚙️')
        if any(word in title_lower for word in ['acquisition', 'acquires', 'merger']):
            tags.append('🤝')
        
        item['tag_string'] = ' '.join(tags) if tags else ''
    
    return news_items


def calculate_alpha_models(row, hist_data, benchmark_data):
    """
    V8.0: 11-model alpha scoring system.
    Weights normalized to exactly 100%.

    Model architecture:
      M1  Momentum           15%  - Multi-timeframe momentum (30d + 90d + RSI)
      M2  Value              10%  - Mean-reversion from 52w high + P/B signal
      M3  Survival           15%  - Runway quality (no confidence multiplier)
      M4  Dilution           10%  - Inverse of dilution risk
      M5  Liquidity           5%  - Tier-based (L0–L3)
      M6  RelStrength         7%  - 90d stock vs benchmark outperformance
      M7  SMC                 8%  - Smart Money Concepts (set later)
      M8  StageFit            5%  - Stage + metal regime alignment
      M9  VolMomentum         7%  - Volatility/RSI trend assessment
      M10 TA                  8%  - Technical Analysis composite (from calculate_all_ta)
      M11 FA                 10%  - Fundamental Analysis score
                            ----
                            100%

    NOTE: M7 (SMC) placeholder is replaced after SMC calculation.
    M10 (TA) and M11 (FA) use pre-computed scores from the row when available.
    """
    models = {}
    breakdown = []

    # === M1: Momentum (15%) — multi-timeframe + RSI ===========================
    ret_30d = row.get('Return_30d', 0)
    ret_90d = row.get('Return_90d', 0)
    rsi_val = row.get('RSI', 50)

    # Continuous 30d scoring (no step-function dead zones)
    if ret_30d >= 20:
        m1_30d = 90
    elif ret_30d >= 10:
        m1_30d = 70 + (ret_30d - 10) * 2  # 70-90
    elif ret_30d >= 0:
        m1_30d = 50 + ret_30d * 2           # 50-70
    elif ret_30d >= -10:
        m1_30d = 50 + ret_30d * 2           # 30-50
    elif ret_30d >= -20:
        m1_30d = 30 + (ret_30d + 10) * 2   # 10-30
    else:
        m1_30d = 10

    # 90d trend confirmation (+/- adjustment)
    if ret_90d > 15:
        m1_90d_adj = 10
    elif ret_90d > 5:
        m1_90d_adj = 5
    elif ret_90d < -15:
        m1_90d_adj = -10
    elif ret_90d < -5:
        m1_90d_adj = -5
    else:
        m1_90d_adj = 0

    # RSI adjustment
    if rsi_val < 30:
        m1_rsi_adj = 8   # Oversold bounce opportunity
    elif rsi_val > 70:
        m1_rsi_adj = -5  # Overbought caution
    elif rsi_val > 50:
        m1_rsi_adj = 3   # Bullish momentum
    else:
        m1_rsi_adj = 0

    momentum_score = max(0, min(100, m1_30d + m1_90d_adj + m1_rsi_adj))
    models['M1_Momentum'] = momentum_score * 0.15
    breakdown.append(f"M1 Momentum: {momentum_score}/100 x 15% = {models['M1_Momentum']:.1f} (30d:{ret_30d:+.1f}%, 90d:{ret_90d:+.1f}%, RSI:{rsi_val:.0f})")

    # === M2: Value Positioning (10%) — distance from high + P/B ===============
    pct_from_high = row.get('Pct_From_52w_High', 0)

    # Continuous scoring (no dead zones)
    if pct_from_high <= -50:
        m2_price = 90
    elif pct_from_high <= -25:
        m2_price = 60 + (-pct_from_high - 25) * 1.2  # 60-90
    elif pct_from_high <= -10:
        m2_price = 45 + (-pct_from_high - 10) * 1.0  # 45-60
    elif pct_from_high <= -5:
        m2_price = 40 + (-pct_from_high - 5) * 1.0   # 40-45
    else:
        m2_price = max(15, 40 + pct_from_high * 5)    # 15-40

    # P/B bonus (from FA data if available)
    pb = row.get('P_B', None)
    m2_pb_adj = 0
    if pb is not None and isinstance(pb, (int, float)) and pb > 0:
        if pb < 1.0:
            m2_pb_adj = 10   # Trading below book value
        elif pb < 2.0:
            m2_pb_adj = 5
        elif pb > 5.0:
            m2_pb_adj = -5

    value_score = max(0, min(100, m2_price + m2_pb_adj))
    models['M2_Value'] = value_score * 0.10
    breakdown.append(f"M2 Value: {value_score}/100 x 10% = {models['M2_Value']:.1f} (from_high:{pct_from_high:+.1f}%)")

    # === M3: Survival Quality (15%) — runway only, NO confidence multiplier ===
    runway = row.get('Runway', 12)

    # Continuous scoring
    if runway >= 24:
        survival_score = 90
    elif runway >= 18:
        survival_score = 75 + (runway - 18) * 2.5    # 75-90
    elif runway >= 12:
        survival_score = 60 + (runway - 12) * 2.5    # 60-75
    elif runway >= 6:
        survival_score = 30 + (runway - 6) * 5.0     # 30-60
    else:
        survival_score = max(5, runway * 5)            # 0-30

    models['M3_Survival'] = survival_score * 0.15
    breakdown.append(f"M3 Survival: {survival_score:.0f}/100 x 15% = {models['M3_Survival']:.1f} (Runway:{runway:.1f}mo)")

    # === M4: Dilution Penalty (10%) ===========================================
    dil_risk = row.get('Dilution_Risk_Score', 50)
    dilution_score = max(0, min(100, 100 - dil_risk))

    models['M4_Dilution'] = dilution_score * 0.10
    breakdown.append(f"M4 Dilution: {dilution_score:.0f}/100 x 10% = {models['M4_Dilution']:.1f}")

    # === M5: Liquidity (5%) ==================================================
    tier = row.get('Liq_tier_code', 'L0')
    liq_score = {'L3': 85, 'L2': 65, 'L1': 45, 'L0': 15, 'UNKNOWN': 30}.get(tier, 30)

    models['M5_Liquidity'] = liq_score * 0.05
    breakdown.append(f"M5 Liquidity: {liq_score}/100 x 5% = {models['M5_Liquidity']:.1f} ({tier})")

    # === M6: Relative Strength (7%) ==========================================
    rel_score = 50
    if benchmark_data is not None and not hist_data.empty:
        try:
            n = min(90, len(hist_data) - 1, len(benchmark_data) - 1)
            if n >= 20:
                stock_ret = (hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[-n] - 1) * 100
                bench_ret = (benchmark_data['Close'].iloc[-1] / benchmark_data['Close'].iloc[-n] - 1) * 100
                outperformance = stock_ret - bench_ret
                # Continuous scoring
                rel_score = max(10, min(90, 50 + outperformance * 2))
        except Exception:
            pass

    models['M6_RelStrength'] = rel_score * 0.07
    breakdown.append(f"M6 RelStrength: {rel_score:.0f}/100 x 7% = {models['M6_RelStrength']:.1f}")

    # === M7: SMC (8%) — placeholder, replaced after SMC calculation ===========
    models['M7_SMC'] = 50 * 0.08
    breakdown.append(f"M7 SMC: 50/100 x 8% = {models['M7_SMC']:.1f} (recalculated later)")

    # === M8: Stage/Metal Fit (5%) — stage + metal regime alignment ============
    stage = row.get('stage', 'Explorer')
    metal = row.get('metal', 'Gold')
    metal_regime = row.get('Metal_Regime', 'neutral')

    stage_base = {'Producer': 65, 'Developer': 55}.get(stage, 50)
    # Metal regime alignment: if the stock's metal is in a bull regime, boost
    metal_adj = 0
    if isinstance(metal_regime, str):
        if 'bull' in metal_regime.lower():
            metal_adj = 15
        elif 'bear' in metal_regime.lower():
            metal_adj = -10
    # Explorer bonus in bull regime (high torque)
    if stage == 'Explorer' and metal_adj > 0:
        metal_adj += 10

    stage_score = max(0, min(100, stage_base + metal_adj))
    models['M8_StageFit'] = stage_score * 0.05
    breakdown.append(f"M8 StageFit: {stage_score}/100 x 5% = {models['M8_StageFit']:.1f} ({stage}/{metal})")

    # === M9: Vol/Momentum Risk Assessment (7%) ================================
    volatility = row.get('Volatility_60d', 0)
    current_price = row.get('Price', 0)
    sma200 = row.get('MA200', 0)

    vol_momentum_score = 50
    if volatility > 0 and sma200 > 0:
        trend_down = current_price < sma200
        if trend_down:
            if volatility > 50:
                vol_momentum_score = 20
            elif volatility > 30:
                vol_momentum_score = 35
            else:
                vol_momentum_score = 45
        else:
            if rsi_val > 50 and volatility > 30:
                vol_momentum_score = 80
            elif rsi_val > 50:
                vol_momentum_score = 70
            elif volatility > 50:
                vol_momentum_score = 40
            else:
                vol_momentum_score = 55

    models['M9_VolMomentum'] = vol_momentum_score * 0.07
    breakdown.append(f"M9 VolMomentum: {vol_momentum_score}/100 x 7% = {models['M9_VolMomentum']:.1f}")

    # === M10: Technical Analysis Composite (8%) ===============================
    # Uses pre-computed TA_Score from technical_analysis.py (0-100, 50=neutral)
    ta_score_raw = row.get('TA_Score', 50)
    if not isinstance(ta_score_raw, (int, float)):
        ta_score_raw = 50
    ta_score = max(0, min(100, float(ta_score_raw)))

    models['M10_TA'] = ta_score * 0.08
    breakdown.append(f"M10 TA: {ta_score:.0f}/100 x 8% = {models['M10_TA']:.1f} (RSI/MACD/BB/OBV/ADX)")

    # === M11: Fundamental Analysis Score (10%) ================================
    # Uses pre-computed FA_Score from calculate_fundamental_score()
    fa_score_raw = row.get('FA_Score', 50)
    if not isinstance(fa_score_raw, (int, float)):
        fa_score_raw = 50
    fa_score = max(0, min(100, float(fa_score_raw)))

    models['M11_FA'] = fa_score * 0.10
    breakdown.append(f"M11 FA: {fa_score:.0f}/100 x 10% = {models['M11_FA']:.1f} (Piotroski/AISC/P-B)")

    # === Total ================================================================
    alpha_score = sum(models.values())

    return {
        'alpha_score': alpha_score,
        'models': models,
        'breakdown': breakdown
    }


def calculate_fundamental_score(row: Dict, sector_data: Optional[List[Dict]] = None) -> Dict:
    """
    V7.5: Calculate Fundamental Alpha (FA) Score based on mining fundamentals.
    
    For stocks with Market Cap < $500M:
    - IGNORE P/E ratio and Dividend Yield (juniors invest heavily in drilling)
    - Score based on:
      * Price to Book (P/B): Lower is better (undervalued assets)
      * Cash vs Debt: High points if Cash > Debt (Runway)
      * Insider Ownership: Boost if Insiders > 10%
      * Current Ratio: Must be > 1.5 (Liquidity to survive)
    
    For stocks with Market Cap >= $500M:
    - Use traditional metrics (AISC, Cash/Debt, MCAP/OZ)
    
    Args:
        row: Row dict with symbol data
        sector_data: Optional list of sector peers for MCAP/OZ percentile calculation
    
    Returns:
        Dict with 'fa_score' (float), 'components' (dict), 'reasoning' (list), 'signal' (str: 'BUY'/'SELL'/'NEUTRAL')
    """
    result = {
        'fa_score': 0.0,
        'components': {},
        'reasoning': [],
        'signal': 'NEUTRAL'
    }
    
    market_cap = row.get('Market_Cap', row.get('market_cap', 0))
    info_dict = row.get('info_dict', {})
    
    # V7.5: For stocks < $500M, use Junior/Mid-Tier scoring (ignore P/E and Dividend Yield)
    if market_cap > 0 and market_cap < 500:
        # Component 1: Price to Book (P/B) - Score ∝ 1/P/B (Lower is better)
        price_to_book = info_dict.get('priceToBook', info_dict.get('priceToBookTrailing12Months', None))
        if price_to_book is not None and price_to_book > 0:
            # V7.5: Score proportional to 1/P/B (inverse relationship)
            # Lower P/B = higher score
            pb_score = (1.0 / price_to_book) * 20  # Scale factor of 20 for reasonable range
            pb_score = min(20.0, pb_score)  # Cap at +20 Alpha
            
            if price_to_book < 1.0:
                result['fa_score'] += pb_score
                result['components']['pb_bonus'] = pb_score
                result['reasoning'].append(f"✅ P/B {price_to_book:.2f} < 1.0 (undervalued): +{pb_score:.1f} Alpha (1/P/B scoring)")
            elif price_to_book < 2.0:
                result['fa_score'] += pb_score
                result['components']['pb_bonus'] = pb_score
                result['reasoning'].append(f"✅ P/B {price_to_book:.2f} < 2.0 (good value): +{pb_score:.1f} Alpha (1/P/B scoring)")
            elif price_to_book > 5.0:
                result['fa_score'] -= 10.0
                result['components']['pb_penalty'] = -10.0
                result['reasoning'].append(f"⚠️ P/B {price_to_book:.2f} > 5.0 (overvalued): -10 Alpha")
            else:
                result['fa_score'] += pb_score * 0.5  # Half score for middle range
                result['components']['pb_bonus'] = pb_score * 0.5
                result['reasoning'].append(f"P/B {price_to_book:.2f} (neutral): +{pb_score*0.5:.1f} Alpha (1/P/B scoring)")
        else:
            result['components']['pb_bonus'] = 0.0
            result['reasoning'].append("P/B data unavailable")
        
        # Component 2: Cash vs Debt (Runway Factor) - Boost score if Cash > Total Debt
        cash = row.get('Cash', row.get('cash', info_dict.get('totalCash', 0)))
        if cash == 0:
            cash = info_dict.get('totalCash', 0) / 1_000_000 if info_dict.get('totalCash') else 0
        debt = row.get('Debt', row.get('debt', info_dict.get('totalDebt', 0)))
        if debt == 0:
            debt = info_dict.get('totalDebt', 0) / 1_000_000 if info_dict.get('totalDebt') else 0
        
        if cash > 0 and debt >= 0:
            if cash > debt:
                # V7.5: Boost score (not just fixed amount) - proportional to cash/debt ratio
                cash_debt_ratio = cash / max(debt, 1.0)  # Avoid division by zero
                bonus = min(20.0, cash_debt_ratio * 5.0)  # Scale: 2x cash = +10, 4x cash = +20
                result['fa_score'] += bonus
                result['components']['cash_debt_bonus'] = bonus
                result['reasoning'].append(f"✅ Cash ${cash:,.0f}M > Debt ${debt:,.0f}M (Runway Factor, {cash_debt_ratio:.1f}x): +{bonus:.1f} Alpha")
                if result['signal'] != 'SELL':
                    result['signal'] = 'BUY'
            else:
                result['components']['cash_debt_bonus'] = 0.0
                result['reasoning'].append(f"Cash ${cash:,.0f}M ≤ Debt ${debt:,.0f}M (no bonus)")
        else:
            result['components']['cash_debt_bonus'] = 0.0
            result['reasoning'].append("Cash/Debt data unavailable")
        
        # Component 3: Insider Ownership (> 10% = +20% bonus to score)
        insider_ownership = info_dict.get('heldPercentInsiders', None)
        if insider_ownership is not None:
            if insider_ownership > 10:
                # V7.5: +20% bonus to FA score (not just +12 Alpha)
                base_score = result['fa_score']
                bonus_pct = 0.20
                bonus_amount = base_score * bonus_pct
                result['fa_score'] += bonus_amount
                result['components']['insider_bonus'] = bonus_amount
                result['reasoning'].append(f"✅ Insider Ownership {insider_ownership:.1f}% > 10%: +{bonus_pct*100:.0f}% bonus ({bonus_amount:.1f} Alpha)")
                if result['signal'] != 'SELL':
                    result['signal'] = 'BUY'
            elif insider_ownership > 5:
                result['fa_score'] += 6.0
                result['components']['insider_bonus'] = 6.0
                result['reasoning'].append(f"Insider Ownership {insider_ownership:.1f}% > 5%: +6 Alpha")
            else:
                result['components']['insider_bonus'] = 0.0
                result['reasoning'].append(f"Insider Ownership {insider_ownership:.1f}% (no bonus)")
        else:
            result['components']['insider_bonus'] = 0.0
            result['reasoning'].append("Insider Ownership data unavailable")
        
        # Component 4: Current Ratio (Must be > 1.5 for liquidity)
        current_ratio = info_dict.get('currentRatio', None)
        if current_ratio is not None:
            if current_ratio > 1.5:
                result['fa_score'] += 10.0
                result['components']['current_ratio_bonus'] = 10.0
                result['reasoning'].append(f"✅ Current Ratio {current_ratio:.2f} > 1.5 (liquidity): +10 Alpha")
            elif current_ratio < 1.0:
                result['fa_score'] -= 15.0
                result['components']['current_ratio_penalty'] = -15.0
                result['reasoning'].append(f"🔴 Current Ratio {current_ratio:.2f} < 1.0 (illiquid): -15 Alpha")
                result['signal'] = 'SELL'
            else:
                result['components']['current_ratio_bonus'] = 0.0
                result['reasoning'].append(f"Current Ratio {current_ratio:.2f} (marginal)")
        else:
            result['components']['current_ratio_bonus'] = 0.0
            result['reasoning'].append("Current Ratio data unavailable")
        
        # Note: P/E and Dividend Yield are IGNORED for juniors (as per requirements)
        result['reasoning'].append("ℹ️ P/E and Dividend Yield ignored for juniors (< $500M)")
    
    else:
        # V7.5: For stocks >= $500M, use traditional scoring
        # Component 1: AISC Penalty (AISC > $1,400 = Sell/Avoid)
        aisc = row.get('AISC', row.get('aisc', None))
        if aisc is not None and aisc > 0:
            if aisc > 1400:
                result['fa_score'] -= 15.0
                result['components']['aisc_penalty'] = -15.0
                result['reasoning'].append(f"🔴 AISC ${aisc:.0f}/oz > $1,400: -15 Alpha (Sell/Avoid)")
                result['signal'] = 'SELL'
            elif aisc < 1100:
                # Bonus for low AISC
                result['fa_score'] += 10.0
                result['components']['aisc_bonus'] = 10.0
                result['reasoning'].append(f"✅ AISC ${aisc:.0f}/oz < $1,100: +10 Alpha")
            else:
                result['components']['aisc_penalty'] = 0.0
                result['reasoning'].append(f"AISC ${aisc:.0f}/oz (neutral)")
        else:
            result['components']['aisc_penalty'] = 0.0
            result['reasoning'].append("AISC data unavailable")
        
        # Component 2: Cash > Debt Reward (Cash > Debt = Buy)
        cash = row.get('Cash', row.get('cash', 0))
        debt = row.get('Debt', row.get('debt', 0))
        if cash > 0 and debt >= 0:
            if cash > debt:
                result['fa_score'] += 10.0
                result['components']['cash_debt_bonus'] = 10.0
                result['reasoning'].append(f"✅ Cash ${cash:,.0f} > Debt ${debt:,.0f}: +10 Alpha (Buy)")
                if result['signal'] != 'SELL':
                    result['signal'] = 'BUY'
            else:
                result['components']['cash_debt_bonus'] = 0.0
                result['reasoning'].append(f"Cash ${cash:,.0f} ≤ Debt ${debt:,.0f} (no bonus)")
        else:
            result['components']['cash_debt_bonus'] = 0.0
            result['reasoning'].append("Cash/Debt data unavailable")
    
    # Component 3: Low MCAP/OZ Reward (Low MCAP/OZ = Buy/Green)
    market_cap = row.get('Market_Cap', row.get('market_cap', 0))
    ounces_reserve = row.get('Ounces_Reserve', row.get('ounces_reserve', row.get('Reserve_Oz', 0)))
    
    if market_cap > 0 and ounces_reserve > 0 and sector_data:
        # Calculate this symbol's MCAP/OZ
        mcap_per_oz = market_cap / ounces_reserve
        
        # Calculate MCAP/OZ for all sector peers
        sector_mcap_per_oz = []
        for peer in sector_data:
            peer_mcap = peer.get('Market_Cap', peer.get('market_cap', 0))
            peer_oz = peer.get('Ounces_Reserve', peer.get('ounces_reserve', peer.get('Reserve_Oz', 0)))
            if peer_mcap > 0 and peer_oz > 0:
                sector_mcap_per_oz.append(peer_mcap / peer_oz)
        
        if sector_mcap_per_oz:
            # Calculate 20th percentile (bottom 20% = low MCAP/OZ = Buy/Green)
            percentile_20 = np.percentile(sector_mcap_per_oz, 20)
            
            if mcap_per_oz <= percentile_20:
                result['fa_score'] += 15.0
                result['components']['mcap_oz_bonus'] = 15.0
                result['reasoning'].append(f"✅ Low MCAP/OZ ${mcap_per_oz:.2f} (bottom 20%, ≤${percentile_20:.2f}): +15 Alpha (Buy/Green)")
                if result['signal'] != 'SELL':
                    result['signal'] = 'BUY'
            else:
                # Check if high MCAP/OZ (top 20% = overvalued)
                percentile_80 = np.percentile(sector_mcap_per_oz, 80)
                if mcap_per_oz >= percentile_80:
                    result['fa_score'] -= 10.0
                    result['components']['mcap_oz_penalty'] = -10.0
                    result['reasoning'].append(f"⚠️ High MCAP/OZ ${mcap_per_oz:.2f} (top 20%, ≥${percentile_80:.2f}): -10 Alpha")
                else:
                    result['components']['mcap_oz_bonus'] = 0.0
                    result['reasoning'].append(f"MCAP/OZ ${mcap_per_oz:.2f} (middle range)")
        else:
            result['components']['mcap_oz_bonus'] = 0.0
            result['reasoning'].append("Insufficient sector data for MCAP/OZ comparison")
    else:
        result['components']['mcap_oz_bonus'] = 0.0
        if not sector_data:
            result['reasoning'].append("Sector data unavailable for MCAP/OZ comparison")
        else:
            result['reasoning'].append("MCAP/OZ data unavailable")
    
    return result


def detect_market_buzz(hist_data: pd.DataFrame, threshold_multiplier: float = 3.0) -> Dict:
    """
    V5.0: Detect Market Buzz by identifying volume spikes.
    
    Logic: Volume spike = current volume > (threshold_multiplier × 20-day average)
    Default threshold: +300% above 20-day average (threshold_multiplier = 3.0)
    
    Args:
        hist_data: Historical price/volume data (must have 'Volume' column)
        threshold_multiplier: Multiplier for volume spike detection (default 3.0 = 300% above average)
    
    Returns:
        Dict with 'buzz_detected' (bool), 'volume_spike_pct' (float), 'current_volume' (float), 'avg_20d_volume' (float)
    """
    result = {
        'buzz_detected': False,
        'volume_spike_pct': 0.0,
        'current_volume': 0.0,
        'avg_20d_volume': 0.0,
        'reason': ''
    }
    
    if hist_data is None or hist_data.empty or 'Volume' not in hist_data.columns:
        result['reason'] = 'Insufficient volume data'
        return result
    
    if len(hist_data) < 20:
        result['reason'] = 'Need at least 20 days of data'
        return result
    
    # Get current volume (most recent day)
    current_volume = float(hist_data['Volume'].iloc[-1])
    result['current_volume'] = current_volume
    
    # Calculate 20-day average volume
    avg_20d_volume = float(hist_data['Volume'].tail(20).mean())
    result['avg_20d_volume'] = avg_20d_volume
    
    if avg_20d_volume <= 0:
        result['reason'] = 'Invalid average volume'
        return result
    
    # Calculate volume spike percentage
    volume_spike_pct = ((current_volume - avg_20d_volume) / avg_20d_volume * 100) if avg_20d_volume > 0 else 0.0
    result['volume_spike_pct'] = volume_spike_pct
    
    # Check if spike exceeds threshold (default: 300% = 3.0x multiplier)
    threshold_pct = (threshold_multiplier - 1.0) * 100  # Convert multiplier to percentage
    if volume_spike_pct >= threshold_pct:
        result['buzz_detected'] = True
        result['reason'] = f"Volume spike: {volume_spike_pct:.1f}% above 20-day average (threshold: {threshold_pct:.1f}%)"
    else:
        result['reason'] = f"Volume {volume_spike_pct:.1f}% above average (below {threshold_pct:.1f}% threshold)"
    
    return result


def calculate_sell_risk(row, hist_data, ma50, ma200, news_items, macro_regime):
    """
    Calculate sell risk with comprehensive triggers.

    Hard triggers (high-conviction sell signals):
    - Runway < 6 months (cash crisis imminent)
    - Below MA200 in defensive macro
    - Death cross (MA50 < MA200) — institutional distribution signal
    - Gap-down > 10% (catastrophic event)

    Soft triggers (warning signals that accumulate):
    - Drawdown > 50%, 30d return < -20%
    - >10% below MA50, RSI overbought (unless SMC bullish)
    - Volume distribution (declining volume on up days)
    - Consecutive decline (5+ days)
    - Negative news (expanded keyword list)
    """
    score = 0
    hard_triggers = []
    soft_triggers = []

    runway = row.get('Runway', 12)
    price = row.get('Price', 0)
    ret_7d = row.get('Return_7d', 0)
    ret_30d = row.get('Return_30d', 0)
    drawdown = abs(row.get('Drawdown_90d', 0))

    # ── Hard triggers ──
    if runway < 6:
        score += 50
        hard_triggers.append(f"💀 Runway {runway:.1f}mo < 6mo CRITICAL")

    if ma200 > 0 and price < ma200:
        if macro_regime.get('regime') == 'DEFENSIVE':
            score += 30
            hard_triggers.append("💀 Below MA200 + Defensive macro")
        else:
            score += 15
            soft_triggers.append("⚠️ Below MA200")

    # Death cross: MA50 crosses below MA200 — institutional exit signal
    if ma50 > 0 and ma200 > 0 and ma50 < ma200:
        score += 20
        hard_triggers.append("💀 Death cross (MA50 < MA200)")

    # Gap-down detection: catastrophic single-day drop
    if hist_data is not None and len(hist_data) >= 2:
        try:
            prev_close = hist_data['Close'].iloc[-2]
            curr_open = hist_data['Open'].iloc[-1] if 'Open' in hist_data.columns else price
            if prev_close > 0:
                gap_pct = (prev_close - curr_open) / prev_close * 100
                if gap_pct > 10:
                    score += 25
                    hard_triggers.append(f"💀 Gap-down {gap_pct:.1f}% (catastrophic)")
                elif gap_pct > 5:
                    score += 10
                    soft_triggers.append(f"⚠️ Gap-down {gap_pct:.1f}%")
        except (IndexError, KeyError):
            pass

    # ── Soft triggers ──
    if drawdown > 50:
        score += 15
        soft_triggers.append(f"⚠️ Drawdown {drawdown:.0f}% > 50%")

    if ret_30d < -20:
        score += 10
        soft_triggers.append(f"⚠️ 30d return {ret_30d:.0f}% < -20%")

    if ma50 > 0 and price < ma50 * 0.90:
        score += 10
        soft_triggers.append("⚠️ >10% below MA50")

    # Consecutive decline: 5+ red days in a row = distribution
    if hist_data is not None and len(hist_data) >= 5:
        try:
            recent_closes = hist_data['Close'].tail(6).values
            consecutive_down = 0
            for k in range(1, len(recent_closes)):
                if recent_closes[k] < recent_closes[k - 1]:
                    consecutive_down += 1
                else:
                    consecutive_down = 0
            if consecutive_down >= 5:
                score += 15
                soft_triggers.append(f"⚠️ {consecutive_down} consecutive down days")
            elif consecutive_down >= 3:
                score += 5
                soft_triggers.append(f"⚠️ {consecutive_down} consecutive down days")
        except (IndexError, KeyError):
            pass

    # Volume distribution: declining volume on up days vs rising volume on down days
    if hist_data is not None and len(hist_data) >= 20 and 'Volume' in hist_data.columns:
        try:
            recent = hist_data.tail(20)
            changes = recent['Close'].diff()
            up_days = recent[changes > 0]
            down_days = recent[changes < 0]
            if len(up_days) > 0 and len(down_days) > 0:
                avg_up_vol = up_days['Volume'].mean()
                avg_down_vol = down_days['Volume'].mean()
                if avg_up_vol > 0 and avg_down_vol / avg_up_vol > 1.5:
                    score += 10
                    soft_triggers.append("⚠️ Volume distribution (heavy selling)")
        except (IndexError, KeyError):
            pass

    # Volatility Harvesting: Ignore RSI overbought signals if SMC_Bias is strongly BULLISH
    rsi = row.get('RSI', 50)
    smc_bias = row.get('SMC_Bias', 'Neutral')
    is_strongly_bullish_smc = 'BULLISH' in str(smc_bias).upper() and 'STRONG' in str(smc_bias).upper()

    if rsi > 70 and not is_strongly_bullish_smc:
        score += 5
        soft_triggers.append(f"⚠️ RSI {rsi:.0f} > 70 (overbought)")
    elif rsi > 70 and is_strongly_bullish_smc:
        soft_triggers.append(f"✅ RSI {rsi:.0f} > 70 but SMC strongly BULLISH - ignoring overbought")

    # News triggers — expanded keyword set for mining sector
    _neg_news_keywords = [
        'low cash', 'needs financing', 'suspends', 'lawsuit', 'permit denied',
        'management departure', 'ceo resigns', 'cfo resigns', 'accident', 'spill',
        'regulatory', 'delisted', 'halt', 'investigation', 'fraud', 'bankruptcy',
        'dilution', 'write-down', 'impairment', 'downgrade', 'default',
    ]
    for item in news_items:
        title_lower = item.get('title', '').lower()
        if any(kw in title_lower for kw in _neg_news_keywords):
            score += 15
            soft_triggers.append(f"⚠️ Negative news: {item['title'][:50]}")
            break

    score = min(100, score)

    if score >= 60:
        verdict = "SELL NOW"
    elif score >= 40:
        verdict = "CONSIDER SELLING"
    elif score >= 20:
        verdict = "WATCH"
    else:
        verdict = "NORMAL"

    all_triggers = hard_triggers + soft_triggers

    return {
        'score': score,
        'verdict': verdict,
        'hard_triggers': hard_triggers,
        'soft_triggers': soft_triggers,
        'all_triggers': all_triggers
    }


def calculate_tape_gate(macro_regime, gold_analysis=None, silver_analysis=None):
    """
    Calculate tape/regime gate decision helper.
    Returns: {regime_label, new_buys_allowed, throttle, reasons[]}
    """
    gate = {
        'regime_label': 'NEUTRAL',
        'new_buys_allowed': True,
        'throttle': 1.0,
        'reasons': []
    }
    
    # Use macro_regime inputs
    dxy = macro_regime.get('dxy', 0)
    vix = macro_regime.get('vix', 0)
    allow_new_buys = macro_regime.get('allow_new_buys', True)
    throttle_factor = macro_regime.get('throttle_factor', 1.0)
    regime = macro_regime.get('regime', 'NEUTRAL')
    
    # DXY trend
    dxy_ma = macro_regime.get('dxy_ma', dxy) if 'dxy_ma' in macro_regime else dxy
    if dxy > 0 and dxy_ma > 0:
        if dxy > dxy_ma * 1.05:
            gate['reasons'].append('DXY: Strong (bearish for gold)')
            gate['throttle'] *= 0.8
        elif dxy < dxy_ma * 0.95:
            gate['reasons'].append('DXY: Weak (bullish for gold)')
        else:
            gate['reasons'].append('DXY: Neutral')
    else:
        gate['reasons'].append('DXY: Unknown')
    
    # VIX regime
    if vix > 0:
        if vix > 25:
            gate['regime_label'] = 'DEFENSIVE'
            gate['new_buys_allowed'] = False
            gate['throttle'] = 0.5
            gate['reasons'].append(f'VIX: {vix:.1f} (defensive)')
        elif vix < 15:
            gate['regime_label'] = 'RISK-ON'
            gate['reasons'].append(f'VIX: {vix:.1f} (risk-on)')
        else:
            gate['reasons'].append(f'VIX: {vix:.1f} (neutral)')
    else:
        gate['reasons'].append('VIX: Unknown')
    
    # Metal outlook (if available)
    # Mining-Specific Regimes: Check Gold/Silver correlation for AGGRESSIVE mode
    is_aggressive_tape = False
    if gold_analysis and silver_analysis:
        gold_bias = gold_analysis.get('bias_short', 'NEUTRAL')
        silver_bias = silver_analysis.get('bias_short', 'NEUTRAL')
        
        # Check if gold and silver correlation is rising alongside prices
        # In production, this would check 30-day correlation trends
        # For now, check if both are bullish (indicates sector strength)
        if 'BULLISH' in str(gold_bias) and 'BULLISH' in str(silver_bias):
            # Rising correlation + rising prices = AGGRESSIVE tape
            is_aggressive_tape = True
            gate['throttle'] = 1.5  # Boost throttle for aggressive deployment
            gate['mode'] = 'Aggressive'
            gate['reasons'].append('Metals: Bullish + Rising Correlation (AGGRESSIVE)')
        elif 'BEARISH' in str(gold_bias) or 'BEARISH' in str(silver_bias):
            gate['throttle'] *= 0.9
            gate['reasons'].append('Metals: Bearish')
        elif 'BULLISH' in str(gold_bias) or 'BULLISH' in str(silver_bias):
            gate['reasons'].append('Metals: Bullish')
        else:
            gate['reasons'].append('Metals: Neutral')
    else:
        gate['reasons'].append('Metals: Unknown')
    
    # Override with macro_regime settings
    gate['new_buys_allowed'] = allow_new_buys
    gate['throttle'] = min(gate['throttle'], throttle_factor)
    if regime == 'DEFENSIVE':
        gate['regime_label'] = 'DEFENSIVE'
        gate['new_buys_allowed'] = False
    
    # Store aggressive flag for use in arbitrate_final_decision
    gate['is_aggressive'] = is_aggressive_tape
    
    return gate


def fetch_gold_silver_prices() -> Dict:
    """
    V4.0 Phase 3: Automatically fetch GC=F (Gold) and SI=F (Silver) prices via yfinance.
    Called at the start of any run to enable automated macro-rotation.
    
    Returns:
        Dict with 'gold_price', 'silver_price', 'gs_ratio', 'success', 'error'
    """
    result = {
        'gold_price': 0.0,
        'silver_price': 0.0,
        'gs_ratio': 0.0,
        'success': False,
        'error': None
    }
    
    if not YFINANCE_AVAILABLE:
        result['error'] = 'yfinance not available'
        return result
    
    try:
        # Fetch Gold futures (GC=F)
        gold_ticker = yf.Ticker("GC=F")
        gold_hist = gold_ticker.history(period="1d")
        
        if not gold_hist.empty and 'Close' in gold_hist.columns:
            result['gold_price'] = float(gold_hist['Close'].iloc[-1])
        else:
            result['error'] = 'Could not fetch GC=F price'
            return result
        
        # Fetch Silver futures (SI=F)
        silver_ticker = yf.Ticker("SI=F")
        silver_hist = silver_ticker.history(period="1d")
        
        if not silver_hist.empty and 'Close' in silver_hist.columns:
            result['silver_price'] = float(silver_hist['Close'].iloc[-1])
        else:
            result['error'] = 'Could not fetch SI=F price'
            return result
        
        # Calculate GSR
        if result['silver_price'] > 0:
            result['gs_ratio'] = result['gold_price'] / result['silver_price']
            result['success'] = True
        else:
            result['error'] = 'Invalid silver price'
        
    except Exception as e:
        result['error'] = f"Error fetching prices: {str(e)}"
    
    return result


def calculate_gs_ratio_bias(gold_price: float, silver_price: float) -> Dict:
    """
    Calculate Gold/Silver Ratio (GSR) bias for institutional rotation.
    
    Logic:
    - If GSR > 85 (Silver historically cheap): +10 Alpha Bonus to Silver explorers
    - If GSR < 65 (Gold historically cheap): +10 Alpha Bonus to Gold producers
    
    Args:
        gold_price: Current gold price (e.g., from GC=F or GLD)
        silver_price: Current silver price (e.g., from SI=F or SLV)
    
    Returns:
        Dict with 'gs_ratio', 'silver_bonus', 'gold_bonus', 'reason'
    """
    if gold_price <= 0 or silver_price <= 0:
        return {
            'gs_ratio': 0.0,
            'silver_bonus': 0,
            'gold_bonus': 0,
            'reason': 'Invalid prices'
        }
    
    gs_ratio = gold_price / silver_price
    
    silver_bonus = 0
    gold_bonus = 0
    reason = f"GSR: {gs_ratio:.1f}"
    
    if gs_ratio > 80:
        # Silver historically cheap - favor Silver explorers (V4.0: threshold lowered from 85 to 80)
        silver_bonus = 10
        reason += " (Silver cheap: +10 Torque Bonus to Silver explorers)"
    elif gs_ratio < 60:
        # Gold historically cheap - shift bias to Gold producers
        gold_bonus = 10
        reason += " (Gold cheap: Shift bias to Gold producers)"
    else:
        reason += " (Neutral)"
    
    return {
        'gs_ratio': gs_ratio,
        'silver_bonus': silver_bonus,
        'gold_bonus': gold_bonus,
        'reason': reason
    }


def calculate_macro_regime(hist_slice: pd.DataFrame = None, date_ts: pd.Timestamp = None):
    """
    Calculate macro regime dynamically using fresh benchmark data.
    
    If hist_slice is provided (for backtest), use it to calculate regime from benchmark.
    Otherwise, fetch live data (for Streamlit UI).
    
    Args:
        hist_slice: Historical price data for benchmark (GDX/GLD) up to date_ts
        date_ts: Current date timestamp for backtest (tz-naive)
    
    Returns:
        Dict with regime, throttle_factor, allow_new_buys, factors, dxy, vix
    """
    regime = {
        'regime': 'NEUTRAL',
        'factors': [],
        'allow_new_buys': True,
        'throttle_factor': 1.0,
        'dxy': 0,
        'vix': 0
    }
    
    # For backtest mode: use hist_slice if provided
    if hist_slice is not None and not hist_slice.empty:
        # Use benchmark data (GDX/GLD) to detect regime changes
        # Check for sector crash: if benchmark drops >10% in last 2 days, switch to DEFENSIVE
        if len(hist_slice) >= 2:
            current_price = hist_slice['Close'].iloc[-1]
            price_2d_ago = hist_slice['Close'].iloc[-2] if len(hist_slice) >= 2 else current_price
            
            drop_pct = ((price_2d_ago - current_price) / price_2d_ago * 100) if price_2d_ago > 0 else 0
            
            if drop_pct > 10.0:  # Sector crash detected
                regime['regime'] = 'DEFENSIVE'
                regime['allow_new_buys'] = False
                regime['throttle_factor'] = 0.5
                regime['factors'].append(f"Sector crash: -{drop_pct:.1f}% in 2 days")
            elif drop_pct > 5.0:  # Significant drop
                regime['regime'] = 'NEUTRAL'
                regime['throttle_factor'] = 0.8
                regime['factors'].append(f"Sector weakness: -{drop_pct:.1f}% in 2 days")
        
        # Check trend using 20-day MA
        if len(hist_slice) >= 20:
            ma20 = hist_slice['Close'].tail(20).mean()
            current_price = hist_slice['Close'].iloc[-1]
            
            if current_price > ma20 * 1.05:
                regime['regime'] = 'BULL'
                regime['throttle_factor'] = 1.2
                regime['factors'].append("Sector >5% above MA20 (bullish)")
            elif current_price < ma20 * 0.95:
                regime['regime'] = 'DEFENSIVE'
                regime['allow_new_buys'] = False
                regime['throttle_factor'] = 0.5
                regime['factors'].append("Sector >5% below MA20 (bearish)")
    
    # For Streamlit UI mode: fetch live data
    if not YFINANCE_AVAILABLE:
        return regime
    
    try:
        # DXY
        dxy = yf.Ticker("DX-Y.NYB")
        dxy_hist = dxy.history(period="6mo")
        if not dxy_hist.empty:
            dxy_price = dxy_hist['Close'].iloc[-1]
            dxy_ma50 = dxy_hist['Close'].tail(50).mean()
            regime['dxy'] = dxy_price
            
            if dxy_price > dxy_ma50 * 1.05:
                regime['factors'].append("DXY strong (bearish for gold)")
                regime['throttle_factor'] = 0.8
            elif dxy_price < dxy_ma50 * 0.95:
                regime['factors'].append("DXY weak (bullish for gold)")
        
        # VIX
        vix = yf.Ticker("^VIX")
        vix_hist = vix.history(period="3mo")
        if not vix_hist.empty:
            vix_price = vix_hist['Close'].iloc[-1]
            regime['vix'] = vix_price
            
            if vix_price > 25:
                regime['regime'] = 'DEFENSIVE'
                regime['factors'].append(f"VIX {vix_price:.1f} > 25 (defensive)")
                regime['allow_new_buys'] = False
                regime['throttle_factor'] = 0.5
            elif vix_price < 15:
                regime['regime'] = 'RISK-ON'
                regime['factors'].append(f"VIX {vix_price:.1f} < 15 (risk-on)")
        
        # Gold trend — check both MA50 and MA200
        gold = yf.Ticker("GC=F")
        gold_hist = gold.history(period="1y")
        if not gold_hist.empty and len(gold_hist) >= 50:
            gold_ma50 = gold_hist['Close'].tail(50).mean()
            gold_price = gold_hist['Close'].iloc[-1]

            if gold_price > gold_ma50 * 1.05:
                regime['factors'].append("Gold above MA50 (bullish)")
            elif gold_price < gold_ma50 * 0.95:
                regime['factors'].append("Gold below MA50 (bearish)")
                regime['throttle_factor'] *= 0.9

            # Gold MA200 — long-term trend (critical for miners)
            if len(gold_hist) >= 200:
                gold_ma200 = gold_hist['Close'].tail(200).mean()
                regime['gold_ma200'] = gold_ma200
                if gold_price > gold_ma200:
                    regime['factors'].append("Gold above MA200 (long-term bullish)")
                else:
                    regime['factors'].append("Gold below MA200 (long-term bearish)")
                    regime['throttle_factor'] *= 0.85

                # Gold death cross (MA50 < MA200) — sector-wide sell signal
                if gold_ma50 < gold_ma200:
                    regime['factors'].append("⚠️ Gold death cross (MA50 < MA200)")
                    regime['throttle_factor'] *= 0.8
                    if regime['regime'] not in ('DEFENSIVE',):
                        regime['regime'] = 'CAUTIOUS'

            # ETF flow proxy: GLD volume trend (20d avg vs 50d avg)
            try:
                gld = yf.Ticker("GLD")
                gld_hist = gld.history(period="3mo")
                if not gld_hist.empty and len(gld_hist) >= 50:
                    gld_vol_20 = gld_hist['Volume'].tail(20).mean()
                    gld_vol_50 = gld_hist['Volume'].tail(50).mean()
                    if gld_vol_50 > 0:
                        vol_ratio = gld_vol_20 / gld_vol_50
                        if vol_ratio > 1.3:
                            regime['factors'].append(f"GLD volume surge ({vol_ratio:.1f}x avg — institutional interest)")
                        elif vol_ratio < 0.7:
                            regime['factors'].append(f"GLD volume drought ({vol_ratio:.1f}x avg — low interest)")
                            regime['throttle_factor'] *= 0.95
            except Exception:
                pass

        # Real rates proxy: TIP (TIPS ETF) trend
        try:
            tip = yf.Ticker("TIP")
            tip_hist = tip.history(period="3mo")
            if not tip_hist.empty and len(tip_hist) >= 20:
                tip_price = tip_hist['Close'].iloc[-1]
                tip_ma20 = tip_hist['Close'].tail(20).mean()
                if tip_price < tip_ma20 * 0.98:
                    regime['factors'].append("Real rates rising (TIP falling — headwind for gold)")
                    regime['throttle_factor'] *= 0.9
                elif tip_price > tip_ma20 * 1.02:
                    regime['factors'].append("Real rates falling (TIP rising — tailwind for gold)")
        except Exception:
            pass

    except Exception:
        pass

    if not regime['factors']:
        regime['factors'] = ['Neutral market conditions']
    
    return regime


def calculate_financing_overhang(news_items, ticker, runway_months, institutional_v3_available=False):
    """
    Calculate financing overhang score (0-100).
    Integrates with analyze_news_intelligence if v3 available, otherwise lightweight fallback.
    
    Returns: dict with 'score' (0-100) and 'reasons' (list of 2 short strings)
    """
    result = {
        'score': 0.0,
        'reasons': []
    }
    
    if not news_items:
        result['reasons'] = ['No news available']
        return result
    
    # Try v3 integration first (if available)
    if institutional_v3_available:
        try:
            from institutional_enhancements_v3 import analyze_news_intelligence
            news_intel = analyze_news_intelligence(news_items, ticker)
            status = news_intel.get('financing_status')
            fin_type = news_intel.get('financing_type')
            impact = news_intel.get('financing_impact', 0)
            
            # Find most recent financing event days_ago from news items
            days_ago = None
            for item in news_items:
                ts = item.get('timestamp', 0)
                if ts > 0:
                    try:
                        if ts > 1e12:
                            ts = ts / 1000
                        news_date = datetime.datetime.fromtimestamp(ts)
                        days = (datetime.datetime.now() - news_date).days
                        if days_ago is None or days < days_ago:
                            days_ago = days
                    except (ValueError, TypeError):
                        pass
            
            if status == 'CLOSED':
                if days_ago is not None and days_ago <= 7:
                    # PP_CLOSED <=7d: overhang drops materially
                    result['score'] = max(20, 40 + impact)
                    result['reasons'] = [f'Financing closed {days_ago}d ago', 'Runway extended']
                elif days_ago is not None and days_ago <= 30:
                    result['score'] = max(20, 35 + impact)
                    result['reasons'] = [f'Financing closed {days_ago}d ago', 'Recent close']
                else:
                    result['score'] = max(0, 30 + impact)
                    result['reasons'] = ['Financing closed', 'Older event']
            elif status == 'ANNOUNCED' or status == 'PRICED':
                # ANNOUNCED not closed: 60-85
                if fin_type == 'ATM':
                    result['score'] = min(95, 85 + (impact if impact > 0 else 10))
                    result['reasons'] = ['Active ATM', 'Ongoing dilution risk']
                elif fin_type == 'SHELF':
                    result['score'] = min(95, 80 + (impact if impact > 0 else 10))
                    result['reasons'] = ['Shelf filed', 'Dilution imminent']
                else:
                    recency_factor = max(0, 30 - (days_ago or 90)) / 30.0
                    result['score'] = 60 + (25 * recency_factor)
                    result['reasons'] = ['Financing announced', 'Not yet closed']
            elif status == 'NONE':
                result['score'] = 0.0
                result['reasons'] = ['No financing events']
            else:
                # Unknown status
                result['score'] = 10.0
                result['reasons'] = ['Unknown financing status']
            
            return result
        except Exception as e:
            # Fall through to lightweight fallback
            pass
    
    # Lightweight fallback: keyword matching
    financing_keywords = {
        'shelf': ['shelf', 'prospectus', 'registration statement'],
        'atm': ['atm', 'at-the-market', 'at the market'],
        'closed': ['closes', 'closed', 'completes', 'completed', 'closing of'],
        'announced': ['announces', 'proposes', 'intends to', 'plans to', 'seeks']
    }
    
    most_recent_event = None
    most_recent_days = None
    
    for item in news_items:
        title_lower = (item.get('title', '') or '').lower()
        ts = item.get('timestamp', 0)
        
        # Check if financing-related
        is_financing = any(word in title_lower for word in 
                          ['financing', 'placement', 'offering', 'capital raise', 'bought deal'])
        if not is_financing:
            continue
        
        # Determine stage and type
        stage = None
        fin_type = None
        
        if any(word in title_lower for word in financing_keywords['closed']):
            stage = 'CLOSED'
        elif any(word in title_lower for word in financing_keywords['announced']):
            stage = 'ANNOUNCED'
        
        if any(word in title_lower for word in financing_keywords['shelf']):
            fin_type = 'SHELF'
        elif any(word in title_lower for word in financing_keywords['atm']):
            fin_type = 'ATM'
        
        if stage:
            # Calculate days ago
            days_ago = None
            if ts > 0:
                try:
                    if ts > 1e12:
                        ts = ts / 1000
                    news_date = datetime.datetime.fromtimestamp(ts)
                    days_ago = (datetime.datetime.now() - news_date).days
                except (ValueError, TypeError):
                    pass
            
            if most_recent_days is None or (days_ago is not None and days_ago < most_recent_days):
                most_recent_event = {'stage': stage, 'type': fin_type or 'PP', 'days_ago': days_ago}
                most_recent_days = days_ago
    
    # Score based on most recent event
    if most_recent_event:
        stage = most_recent_event['stage']
        fin_type = most_recent_event['type']
        days_ago = most_recent_event.get('days_ago')
        
        if stage == 'CLOSED':
            if days_ago is not None and days_ago <= 7:
                result['score'] = 30.0
                result['reasons'] = [f'Financing closed {days_ago}d ago', 'Runway extended']
            else:
                result['score'] = 20.0
                result['reasons'] = ['Financing closed', 'Older event']
        elif stage == 'ANNOUNCED':
            if fin_type == 'ATM':
                result['score'] = 85.0
                result['reasons'] = ['Active ATM', 'Ongoing dilution']
            elif fin_type == 'SHELF':
                result['score'] = 80.0
                result['reasons'] = ['Shelf filed', 'Dilution imminent']
            else:
                recency_factor = max(0, 30 - (days_ago or 90)) / 30.0 if days_ago is not None else 0.5
                result['score'] = 60.0 + (25.0 * recency_factor)
                result['reasons'] = ['Financing announced', 'Not yet closed']
    else:
        result['score'] = 0.0
        result['reasons'] = ['No financing events detected']
    
    return result


def arbitrate_final_decision(row, liq_metrics, data_conf, dilution, sell_risk, 
                             alpha_score, macro_regime, discovery, tape_gate=None, strict_mode=False, risk_mode='BALANCED', gsr_bias=None):
    """
    Final decision arbitration with model hierarchy and veto logic.
    Model roles: Alpha (recommends), Risk/Liquidity/Overhang (may veto).
    
    Args:
        gsr_bias: Optional dict from calculate_gs_ratio_bias() for V4.0 Phase 3 automated macro-rotation
    """
    decision = {
        'action': 'HOLD',
        'confidence': 'Low',
        'recommended_pct': row.get('Pct_Portfolio', 0),
        'max_allowed_pct': 5.0,
        'primary_gating_reason': '',
        'reasoning': [],
        'gates_passed': [],
        'gates_failed': [],
        'warnings': [],
        'veto_applied': False,
        'veto_model': None
    }
    
    # V4.0 Phase 3: Apply GSR Torque Bonus to Silver symbols if GSR > 80
    metal = row.get('metal', row.get('Metal', 'Gold'))
    metal_type = row.get('Metal_Type', metal)
    if gsr_bias and metal_type and 'Silver' in str(metal_type):
        silver_bonus = gsr_bias.get('silver_bonus', 0)
        if silver_bonus > 0:
            alpha_score += silver_bonus  # Apply +10 Torque Bonus
            decision['reasoning'].append(f"💰 GSR Torque Bonus: +{silver_bonus} Alpha (GSR {gsr_bias.get('gs_ratio', 0):.1f} > 80)")
            decision['gates_passed'].append(f"GSR Silver Bonus: +{silver_bonus}")
    
    # Gate checks
    liq_tier = liq_metrics.get('tier_code', 'L0')
    conf_score = data_conf['score']
    dil_score = dilution['score']
    sell_score = sell_risk['score']
    
    # Tape gate enforcement (if provided)
    if tape_gate:
        if not tape_gate.get('new_buys_allowed', True):
            # Check if this is a buy action
            current_pct = row.get('Pct_Portfolio', 0)
            recommended_pct = row.get('Recommended_Pct', current_pct)
            is_buy_action = recommended_pct > current_pct
            
            if is_buy_action:
                decision['gates_failed'].append("🛑 Tape gate: New buys not allowed")
                # Downgrade to HOLD or REDUCE based on strict mode
                if strict_mode and sell_score >= 30:
                    decision['action'] = '🔴 REDUCE'
                    decision['recommended_pct'] = current_pct * 0.5
                else:
                    decision['action'] = '⚪ HOLD'
                    decision['recommended_pct'] = current_pct
                decision['warnings'].append("Tape gate blocked new buy")
                return decision
        
        # Apply throttle to positive deltas
        throttle = tape_gate.get('throttle', 1.0)
        if throttle < 1.0:
            current_pct = row.get('Pct_Portfolio', 0)
            if decision['recommended_pct'] > current_pct:
                delta = decision['recommended_pct'] - current_pct
                decision['recommended_pct'] = current_pct + (delta * throttle)
                decision['warnings'].append(f"Tape throttle: {throttle:.2f}x applied")
    
    # Hard gates
    if not macro_regime.get('allow_new_buys', True):
        decision['gates_failed'].append("🛑 Defensive macro - no new buys")
        if sell_score >= 30:
            decision['action'] = '🔴 REDUCE'
            decision['recommended_pct'] = row.get('Pct_Portfolio', 0) * 0.5
        return decision
    
    # Model hierarchy: Risk models can VETO
    if sell_score >= 60:
        decision['action'] = 'Avoid'
        decision['confidence'] = 'High'
        decision['recommended_pct'] = 0
        decision['reasoning'].extend(sell_risk['hard_triggers'])
        decision['gates_failed'].append(f"🔴 Sell risk {sell_score}/100 CRITICAL")
        decision['primary_gating_reason'] = f"Risk model veto: Sell risk {sell_score}/100 exceeds critical threshold"
        decision['veto_applied'] = True
        decision['veto_model'] = 'Risk'
        return decision
    
    if conf_score < 40:
        decision['gates_failed'].append(f"⚠️ Data confidence {conf_score}/100 too low")
        decision['action'] = 'Avoid'
        decision['confidence'] = 'Low'
        decision['primary_gating_reason'] = f"Data confidence {conf_score}/100 too low for reliable analysis"
        decision['veto_applied'] = True
        decision['veto_model'] = 'Risk'
        return decision
    
    # Size caps
    tier_caps = {'L3': 10.0, 'L2': 7.5, 'L1': 5.0, 'L0': 1.0}
    base_max = tier_caps.get(liq_tier, 1.0)
    
    # Apply discovery exception if granted
    if discovery[0]:
        base_max = min(base_max, 2.5)
        decision['warnings'].append("⚠️ Discovery exception: max 2.5%")
        decision['action'] = '🔵 ADD ⚠️'
    
    # Apply macro throttle
    base_max *= macro_regime.get('throttle_factor', 1.0)
    
    decision['max_allowed_pct'] = base_max
    
    # Decision logic
    current_pct = row.get('Pct_Portfolio', 0)
    
    # Additional risk veto checks (Dilution, Liquidity)
    dilution_score = dilution.get('score', 50)
    financing_overhang = row.get('Financing_Overhang_Score', 0)
    
    # Liquidity veto
    # Only force veto for L0 (confirmed illiquid)
    # UNKNOWN blocks new buys but doesn't force sells (capital protection)
    if liq_tier == 'L0':
        decision['action'] = 'Avoid'
        decision['confidence'] = 'Medium'
        decision['primary_gating_reason'] = "Liquidity model veto: L0 tier (illiquid)"
        decision['veto_applied'] = True
        decision['veto_model'] = 'Liquidity'
        decision['reasoning'].append(f"Liquidity veto: {liq_tier} tier")
        return decision
    elif liq_tier == 'UNKNOWN':
        # UNKNOWN liquidity: block new buys but don't force sells
        current_pct = row.get('Pct_Portfolio', 0)
        recommended_pct = row.get('Recommended_Pct', current_pct)
        is_buy_action = recommended_pct > current_pct
        
        if is_buy_action:
            decision['action'] = 'HOLD'
            decision['confidence'] = 'Low'
            decision['recommended_pct'] = current_pct
            decision['primary_gating_reason'] = "Liquidity UNKNOWN: Blocking new buys (capital protection)"
            decision['warnings'].append("Liquidity tier UNKNOWN - volume data missing/invalid")
            # Don't set veto_applied=True for UNKNOWN - it's a caution, not a hard veto
            return decision
        # For sells/reduces, allow them to proceed (don't block based on UNKNOWN)
    
    # Dilution/Financing veto
    if dilution_score >= 80 or financing_overhang >= 80:
        decision['action'] = 'Avoid'
        decision['confidence'] = 'High'
        veto_reason = f"Capital structure veto: "
        if dilution_score >= 80:
            veto_reason += f"Dilution risk {dilution_score}/100"
        if financing_overhang >= 80:
            if dilution_score >= 80:
                veto_reason += f" + Financing overhang {financing_overhang}/100"
            else:
                veto_reason += f"Financing overhang {financing_overhang}/100"
        decision['primary_gating_reason'] = veto_reason
        decision['veto_applied'] = True
        decision['veto_model'] = 'Capital Structure'
        decision['reasoning'].append(veto_reason)
        return decision
    
    # Regime-aware alpha thresholds
    regime = macro_regime.get('regime', 'NEUTRAL')
    # Check for bull/expansion regimes (RISK-ON is also treated as bullish)
    is_bull_regime = regime in ['BULL', 'EXPANSION', 'RISK-ON']
    # Also check if throttle_factor indicates bullish conditions (> 1.0)
    throttle_factor = macro_regime.get('throttle_factor', 1.0)
    if throttle_factor > 1.0:
        is_bull_regime = True
    
    # High-Torque Mode: Increase Explorer position size in BULL regime (after is_bull_regime is defined)
    stage = row.get('stage', '').strip() if isinstance(row.get('stage'), str) else ''
    if stage == 'Explorer' and is_bull_regime:
        # Allow up to 15% allocation for Explorer stocks in bull regimes
        if base_max < 15.0:
            base_max = 15.0
            decision['max_allowed_pct'] = 15.0
            decision['warnings'].append(f"⚡ High-Torque: Explorer + BULL regime = 15% max position size")
    
    # Base alpha thresholds
    buy_threshold = 60  # Default Buy threshold
    strong_buy_threshold = 75  # Default Strong Buy threshold
    sell_exit_threshold = 40  # Default exit threshold
    
    # AGGRESSIVE mode: Adjusted thresholds for mining sector beta
    if risk_mode == 'AGGRESSIVE':
        buy_threshold = 35  # Capture early breakouts
        strong_buy_threshold = 50  # Strong buy at 50
        sell_exit_threshold = 30  # Let winners run longer
        decision['warnings'].append(f"🔥 AGGRESSIVE mode: Buy={buy_threshold}, Strong={strong_buy_threshold}, Exit={sell_exit_threshold}")
    
    # Regime Sensitivity: Lower Buy threshold in BULL/EXPANSION regimes
    if is_bull_regime:
        if risk_mode != 'AGGRESSIVE':  # Don't override AGGRESSIVE thresholds
            buy_threshold = 50  # Lower threshold from 60 to 50 in bull markets
        # High-Torque Mode: Further lower threshold for BALANCED in bull regimes
        if risk_mode == 'BALANCED':
            buy_threshold = 40  # Front-run moves: Lower threshold to 40 for aggressive alpha deployment
            decision['warnings'].append(f"🚀 High-Torque Mode ({risk_mode}): Bull regime threshold lowered to {buy_threshold}")
        elif risk_mode != 'AGGRESSIVE':
            decision['warnings'].append(f"🐂 Bull regime detected ({regime}): Lowered Buy threshold to {buy_threshold}")
    
    # Mining Torque Factor: Add bonus for Explorer stage with AGGRESSIVE tape gate
    stage = row.get('stage', '').strip()
    # Check if tape gate is aggressive (throttle > 1.0, mode='Aggressive', or is_aggressive flag)
    tape_gate_throttle = tape_gate.get('throttle', 1.0) if tape_gate else 1.0
    tape_gate_allowed = tape_gate.get('new_buys_allowed', True) if tape_gate else True
    tape_gate_mode = tape_gate.get('mode', 'Neutral') if tape_gate else 'Neutral'
    tape_gate_is_aggressive = tape_gate.get('is_aggressive', False) if tape_gate else False
    is_aggressive_tape = (tape_gate_throttle > 1.0 or 
                         tape_gate_mode == 'Aggressive' or 
                         tape_gate_is_aggressive or
                         (tape_gate and 'aggressive' in str(tape_gate.get('reasons', [])).lower()))
    
    alpha_score_adjusted = alpha_score
    
    if stage == 'Explorer' and is_aggressive_tape and tape_gate_allowed:
        torque_bonus = 15 if risk_mode == 'AGGRESSIVE' else 10  # +15 in AGGRESSIVE mode, +10 otherwise
        alpha_score_adjusted = alpha_score + torque_bonus
        decision['warnings'].append(f"⚡ Mining Torque: Explorer + Aggressive tape gate (+{torque_bonus} alpha boost)")
        decision['reasoning'].append(f"Mining torque applied: {alpha_score:.0f} → {alpha_score_adjusted:.0f}")
    
    # AGGRESSIVE mode adjustments (when tape gate is AGGRESSIVE)
    if is_aggressive_tape:
        # Lower Buy threshold to 45 in AGGRESSIVE mode
        if buy_threshold > 45:
            buy_threshold = 45
            decision['warnings'].append(f"🔥 AGGRESSIVE mode: Buy threshold lowered to {buy_threshold}")
        
        # Raise Max Position Size from 10% to 15%
        base_max = decision.get('max_allowed_pct', 5.0)
        if base_max < 15.0:
            decision['max_allowed_pct'] = 15.0
            decision['warnings'].append(f"🔥 AGGRESSIVE mode: Max position size raised to 15%")
    
    alpha_score_for_decision = alpha_score_adjusted
    
    # Discovery Exception: One-time High Confidence buy even if alpha < threshold
    discovery_active = discovery[0] if isinstance(discovery, (tuple, list)) and len(discovery) > 0 else False
    discovery_reason = discovery[1] if isinstance(discovery, (tuple, list)) and len(discovery) > 1 else ''
    
    # Check if Discovery tag is in news intelligence (passed via discovery tuple)
    if discovery_active and discovery_reason:
        # Allow buy even below threshold with discovery exception
        if alpha_score_for_decision >= 45:  # Still need some minimum alpha
            decision['action'] = 'Buy'
            decision['confidence'] = 'High'
            decision['recommended_pct'] = min(base_max * 0.8, current_pct + 1.5)
            decision['primary_gating_reason'] = f"Discovery exception: {discovery_reason} (Alpha: {alpha_score:.0f}/100)"
            decision['warnings'].append(f"🔍 Discovery exception: High Confidence buy despite alpha {alpha_score:.0f} < {buy_threshold}")
            decision['reasoning'].append(f"Discovery tag detected: {discovery_reason}")
            decision['gates_passed'].append("✅ Discovery exception granted")
            # Don't return yet - continue to add alpha reasoning
    
    # Exit logic: Make Alpha exit less sensitive during bull regimes
    # Note: sell_exit_threshold is already set above based on risk_mode (30 for AGGRESSIVE, 40 default)
    if is_bull_regime and risk_mode != 'AGGRESSIVE':  # Don't override AGGRESSIVE exit threshold
        sell_exit_threshold = 50  # Higher threshold during bull = less sensitive exits
        decision['warnings'].append(f"🐂 Bull regime: Exit threshold raised to {sell_exit_threshold} (reduced shakeout risk)")
    
    # V8.0: Fixed arbitration logic.
    # Priority order:
    #   1. Discovery exception (if active and alpha >= 45) — sticky, not overridable
    #   2. Hard sell signals (sell_score >= sell_exit_threshold) — veto everything
    #   3. Moderate sell signals (sell_score >= soft_caution_threshold) — reduce, don't avoid
    #   4. Alpha-based buy (alpha >= strong_buy or buy_threshold)
    #   5. Default HOLD

    # Discovery exception is STICKY — not overridden by moderate sell risk
    if discovery_active and discovery_reason and alpha_score_for_decision >= 45:
        # Already set above, keep it. Only a hard sell veto can override.
        if sell_score >= sell_exit_threshold:
            # Hard sell overrides even discovery
            decision['action'] = 'Avoid'
            decision['confidence'] = 'High'
            decision['recommended_pct'] = current_pct * 0.5
            decision['reasoning'].extend(sell_risk.get('soft_triggers', [])[:2])
            decision['primary_gating_reason'] = f"Hard sell overrides discovery: sell risk {sell_score}/100 >= {sell_exit_threshold}"
        # else: keep the Buy from discovery exception above

    elif sell_score >= sell_exit_threshold:
        # Hard sell signal — veto
        decision['action'] = 'Avoid'
        decision['confidence'] = 'High'
        decision['recommended_pct'] = current_pct * 0.5
        decision['reasoning'].extend(sell_risk.get('soft_triggers', [])[:2])
        decision['primary_gating_reason'] = f"Risk signals: Sell risk {sell_score}/100 >= threshold {sell_exit_threshold}"

    elif sell_score >= 40:
        # Moderate sell risk — REDUCE, don't avoid (was: sell_score >= 20 → Avoid, blocking all buys)
        decision['action'] = 'REDUCE'
        decision['confidence'] = 'Medium'
        decision['recommended_pct'] = current_pct * 0.85
        decision['primary_gating_reason'] = f"Moderate sell risk: {sell_score}/100 (reducing exposure)"

    elif alpha_score_for_decision >= strong_buy_threshold and current_pct < base_max:
        if alpha_score_for_decision >= 80:
            decision['action'] = 'Buy'
            decision['confidence'] = 'High'
        else:
            decision['action'] = 'Buy'
            decision['confidence'] = 'Medium'
        decision['recommended_pct'] = min(base_max, current_pct + 2.0)
        decision['primary_gating_reason'] = f"Alpha model: {alpha_score_for_decision:.0f}/100"

    elif alpha_score_for_decision >= buy_threshold and current_pct < base_max * 0.8:
        decision['action'] = 'Buy'
        decision['confidence'] = 'Medium'
        decision['recommended_pct'] = min(base_max * 0.8, current_pct + 1.0)
        decision['primary_gating_reason'] = f"Alpha model: {alpha_score_for_decision:.0f}/100"

    else:
        decision['action'] = 'HOLD'
        decision['confidence'] = 'Low'
        decision['recommended_pct'] = current_pct
        decision['primary_gating_reason'] = f"Insufficient alpha: {alpha_score_for_decision:.0f}/100 (threshold: {buy_threshold})"
    
    decision['reasoning'].append(f"Alpha: {alpha_score:.0f}/100")
    decision['gates_passed'].append(f"✅ Liquidity: {liq_tier}")
    decision['gates_passed'].append(f"✅ Confidence: {conf_score}/100")
    
    return decision


def get_benchmark_data(metal):
    """Fetch benchmark"""
    if not YFINANCE_AVAILABLE:
        return None
    
    try:
        ticker = "SILJ" if metal == 'Silver' else "GDXJ"
        bench = yf.Ticker(ticker)
        return bench.history(period="6mo")
    except Exception:
        return None
