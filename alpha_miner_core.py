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
MODEL_ROLES = {
    'Alpha': {
        'models': ['M1_Momentum', 'M2_Value', 'M3_Relative', 'M4_Volatility', 'M5_Benchmark', 'M6_Discovery', 'M7_SMC'],
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
        breakdown.append("✅ Has cash data")
    
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
    Calculate 6-model alpha score
    NOTE: Model 7 (SMC) will be added to this score AFTER SMC calculation
    """
    models = {}
    breakdown = []
    
    # M1: Momentum (20%)
    ret_30d = row.get('Return_30d', 0)
    ret_90d = row.get('Return_90d', 0)
    
    momentum_score = 50
    if ret_30d > 10:
        momentum_score = 75
    elif ret_30d > 5:
        momentum_score = 65
    elif ret_30d < -10:
        momentum_score = 25
    elif ret_30d < -5:
        momentum_score = 35
    
    models['M1_Momentum'] = momentum_score * 0.20
    breakdown.append(f"M1 Momentum: {momentum_score}/100 × 20% = {models['M1_Momentum']:.1f}")
    
    # M2: Value Positioning (15%)
    pct_from_high = row.get('Pct_From_52w_High', 0)
    
    value_score = 50
    if pct_from_high < -40:
        value_score = 80
    elif pct_from_high < -25:
        value_score = 65
    elif pct_from_high > -5:
        value_score = 30
    
    models['M2_Value'] = value_score * 0.15
    breakdown.append(f"M2 Value: {value_score}/100 × 15% = {models['M2_Value']:.1f}")
    
    # M3: Survival Quality (20%)
    runway = row.get('Runway', 12)
    data_conf = row.get('Data_Confidence', 50)
    
    survival_score = 50
    if runway >= 18:
        survival_score = 80
    elif runway >= 12:
        survival_score = 65
    elif runway < 6:
        survival_score = 20
    
    # Adjust by data confidence
    survival_score = survival_score * (data_conf / 100)
    
    models['M3_Survival'] = survival_score * 0.20
    breakdown.append(f"M3 Survival: {survival_score:.0f}/100 × 20% = {models['M3_Survival']:.1f}")
    
    # M4: Dilution Penalty (13%)
    dil_risk = row.get('Dilution_Risk_Score', 50)
    dilution_score = 100 - dil_risk
    
    models['M4_Dilution'] = dilution_score * 0.13
    breakdown.append(f"M4 Dilution: {dilution_score:.0f}/100 × 13% = {models['M4_Dilution']:.1f}")
    
    # M5: Liquidity (8%)
    tier = row.get('Liq_tier_code', 'L0')
    liq_score = {'L3': 90, 'L2': 70, 'L1': 50, 'L0': 20}.get(tier, 50)
    
    models['M5_Liquidity'] = liq_score * 0.08
    breakdown.append(f"M5 Liquidity: {liq_score}/100 × 8% = {models['M5_Liquidity']:.1f}")
    
    # M6: Relative Strength (8%)
    rel_score = 50
    if benchmark_data is not None and not hist_data.empty:
        try:
            stock_ret = ((hist_data['Close'].iloc[-1] - hist_data['Close'].iloc[-90]) / 
                        hist_data['Close'].iloc[-90] * 100) if len(hist_data) >= 90 else 0
            bench_ret = ((benchmark_data['Close'].iloc[-1] - benchmark_data['Close'].iloc[-90]) / 
                        benchmark_data['Close'].iloc[-90] * 100) if len(benchmark_data) >= 90 else 0
            
            outperformance = stock_ret - bench_ret
            if outperformance > 10:
                rel_score = 80
            elif outperformance > 0:
                rel_score = 60
            elif outperformance < -10:
                rel_score = 30
        except:
            pass
    
    models['M6_RelStrength'] = rel_score * 0.08
    breakdown.append(f"M6 RelStrength: {rel_score}/100 × 8% = {models['M6_RelStrength']:.1f}")
    
    # M7: SMC (8%) - will be added later after SMC calculation
    # For now, use neutral 50
    models['M7_SMC'] = 50 * 0.08
    breakdown.append(f"M7 SMC: 50/100 × 8% = {models['M7_SMC']:.1f} (calculated later)")
    
    # M8: Stage/Metal Fit (8%)
    stage = row.get('stage', 'Explorer')
    metal = row.get('metal', 'Gold')
    
    stage_score = 50
    if stage == 'Producer':
        stage_score = 70
    elif stage == 'Developer':
        stage_score = 60
    
    models['M8_StageFit'] = stage_score * 0.08
    breakdown.append(f"M8 StageFit: {stage_score}/100 × 8% = {models['M8_StageFit']:.1f}")
    
    # Calculate total (before SMC adjustment)
    alpha_score = sum(models.values())
    
    return {
        'alpha_score': alpha_score,
        'models': models,
        'breakdown': breakdown
    }


def calculate_sell_risk(row, hist_data, ma50, ma200, news_items, macro_regime):
    """Calculate sell risk with triggers"""
    score = 0
    hard_triggers = []
    soft_triggers = []
    
    runway = row.get('Runway', 12)
    price = row.get('Price', 0)
    ret_7d = row.get('Return_7d', 0)
    ret_30d = row.get('Return_30d', 0)
    drawdown = abs(row.get('Drawdown_90d', 0))
    
    # Hard triggers
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
    
    # Soft triggers
    if drawdown > 50:
        score += 15
        soft_triggers.append(f"⚠️ Drawdown {drawdown:.0f}% > 50%")
    
    if ret_30d < -20:
        score += 10
        soft_triggers.append(f"⚠️ 30d return {ret_30d:.0f}% < -20%")
    
    if ma50 > 0 and price < ma50 * 0.90:
        score += 10
        soft_triggers.append("⚠️ >10% below MA50")
    
    # News triggers
    for item in news_items:
        title_lower = item.get('title', '').lower()
        if any(word in title_lower for word in ['low cash', 'needs financing', 'suspends']):
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
    if gold_analysis and silver_analysis:
        gold_bias = gold_analysis.get('bias_short', 'NEUTRAL')
        silver_bias = silver_analysis.get('bias_short', 'NEUTRAL')
        if 'BEARISH' in str(gold_bias) or 'BEARISH' in str(silver_bias):
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
    
    return gate


def calculate_macro_regime():
    """Calculate macro regime"""
    regime = {
        'regime': 'NEUTRAL',
        'factors': [],
        'allow_new_buys': True,
        'throttle_factor': 1.0,
        'dxy': 0,
        'vix': 0
    }
    
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
        
        # Gold trend
        gold = yf.Ticker("GC=F")
        gold_hist = gold.history(period="6mo")
        if not gold_hist.empty and len(gold_hist) >= 50:
            gold_ma50 = gold_hist['Close'].tail(50).mean()
            gold_price = gold_hist['Close'].iloc[-1]
            
            if gold_price > gold_ma50 * 1.05:
                regime['factors'].append("Gold above MA50 (bullish)")
            elif gold_price < gold_ma50 * 0.95:
                regime['factors'].append("Gold below MA50 (bearish)")
                regime['throttle_factor'] *= 0.9
    
    except:
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
                    except:
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
                except:
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
                             alpha_score, macro_regime, discovery, tape_gate=None, strict_mode=False):
    """
    Final decision arbitration with model hierarchy and veto logic.
    Model roles: Alpha (recommends), Risk/Liquidity/Overhang (may veto).
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
    
    # Alpha model recommendations (only if not vetoed)
    if sell_score >= 40:
        decision['action'] = 'Avoid'
        decision['confidence'] = 'High'
        decision['recommended_pct'] = current_pct * 0.5
        decision['reasoning'].extend(sell_risk['soft_triggers'][:2])
        decision['primary_gating_reason'] = f"Risk signals: Sell risk {sell_score}/100"
    
    elif sell_score >= 20:
        decision['action'] = 'Avoid'
        decision['confidence'] = 'Medium'
        decision['recommended_pct'] = current_pct * 0.8
        decision['primary_gating_reason'] = f"Elevated sell risk: {sell_score}/100"
    
    elif alpha_score >= 75 and current_pct < base_max:
        if alpha_score >= 85:
            decision['action'] = 'Buy'
            decision['confidence'] = 'High'
        else:
            decision['action'] = 'Buy'
            decision['confidence'] = 'Medium'
        
        decision['recommended_pct'] = min(base_max, current_pct + 2.0)
        decision['primary_gating_reason'] = f"Alpha model: {alpha_score:.0f}/100"
    
    elif alpha_score >= 60 and current_pct < base_max * 0.8:
        decision['action'] = 'Buy'
        decision['confidence'] = 'Medium'
        decision['recommended_pct'] = min(base_max * 0.8, current_pct + 1.0)
        decision['primary_gating_reason'] = f"Alpha model: {alpha_score:.0f}/100"
    
    else:
        decision['action'] = 'HOLD'
        decision['confidence'] = 'Low'
        decision['recommended_pct'] = current_pct
        decision['primary_gating_reason'] = f"Insufficient alpha signal: {alpha_score:.0f}/100"
    
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
    except:
        return None
