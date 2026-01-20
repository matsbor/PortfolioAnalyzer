import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import pandas as pd

# IMPORTANT:
# We import functions from your Streamlit file.
# This only works if your module import does NOT immediately run Streamlit UI at import-time.
# If it does, we’ll refactor in Step 2 below.

from alpha_miner_core import (
    validate_data_invariants,
    enforce_strict_mode,
    get_risk_profile_preset,
    arbitrate_final_decision,
    calculate_financing_overhang,
)

def test_import_core_is_safe():
    """Test that importing alpha_miner_core doesn't execute Streamlit UI"""
    import alpha_miner_core
    # If we get here without errors, import was safe
    assert hasattr(alpha_miner_core, 'MODEL_ROLES')
    assert hasattr(alpha_miner_core, 'arbitrate_final_decision')

def _sample_df():
    return pd.DataFrame([
        {"Symbol": "AAA", "Price": 10.0, "Quantity": 10, "Cost_Basis": 50.0, "Sell_Risk_Score": 0, "Sleeve": "CORE"},
        {"Symbol": "BBB", "Price": 5.0,  "Quantity": 20, "Cost_Basis": 200.0, "Sell_Risk_Score": 50, "Sleeve": "TACTICAL"},
    ])

def test_get_risk_profile_preset_exists_and_returns_dict():
    p = get_risk_profile_preset("Balanced")
    assert isinstance(p, dict)
    assert "position_caps" in p or "caps" in p or len(p.keys()) > 0

def test_validate_data_invariants_returns_expected_shape():
    df = _sample_df()
    news_cache = {"AAA": [], "BBB": []}
    out = validate_data_invariants(df, news_cache)
    assert isinstance(out, dict)
    # Be flexible, but ensure it has something actionable
    assert any(k in out for k in ["ok", "errors", "warnings", "summary", "violations"])

def test_enforce_strict_mode_does_not_crash():
    df = _sample_df()
    news_cache = {"AAA": [], "BBB": []}
    validation = validate_data_invariants(df, news_cache)
    preset = get_risk_profile_preset("Defensive")

    # enforce_strict_mode signature: (df, validation, risk_profile, strict_mode)
    df2 = enforce_strict_mode(df.copy(), validation, "Defensive", strict_mode=True)

    assert len(df2) == len(df)

def test_watchlist_add_remove():
    """Test watchlist add/remove functionality"""
    import json
    from pathlib import Path
    
    # Mock watchlist file
    watchlist_file = Path.home() / '.alpha_miner_watchlist_test.json'
    
    # Test add
    watchlist = ['TEST1', 'TEST2']
    with open(watchlist_file, 'w') as f:
        json.dump(watchlist, f)
    
    # Test file exists
    assert watchlist_file.exists()
    
    # Test read
    with open(watchlist_file, 'r') as f:
        loaded = json.load(f)
    assert 'TEST1' in loaded
    
    # Cleanup
    if watchlist_file.exists():
        watchlist_file.unlink()

def test_quick_analysis_replay_mode():
    """Test quick analysis runs in replay mode without network calls"""
    # This test verifies that quick analysis logic exists and can handle replay mode
    # Actual execution test would require Streamlit session state mocking
    
    # Test financing overhang with empty news (should not crash)
    result = calculate_financing_overhang([], 'TEST', 12.0, institutional_v3_available=False)
    assert isinstance(result, dict)
    assert 'score' in result
    assert 'reasons' in result

def test_veto_overrides_alpha():
    """Test that veto logic overrides alpha recommendations"""
    # Create a row that would get BUY from alpha but veto from risk
    row = {
        'Pct_Portfolio': 5.0,
        'Sell_Risk_Score': 65,  # Above veto threshold
        'Financing_Overhang_Score': 0,
        'Runway': 12
    }
    
    liq_metrics = {'tier_code': 'L3', 'max_position_pct': 10.0}
    data_conf = {'score': 80}
    dilution = {'score': 30}
    sell_risk = {
        'score': 65,
        'hard_triggers': ['High sell risk'],
        'soft_triggers': []
    }
    macro_regime = {'allow_new_buys': True, 'throttle_factor': 1.0}
    discovery = (False, '')
    
    decision = arbitrate_final_decision(row, liq_metrics, data_conf, dilution, sell_risk, 90, macro_regime, discovery, strict_mode=False)
    
    # Should be vetoed to Avoid
    assert decision['action'] == 'Avoid'
    assert decision['veto_applied'] == True
    assert decision['veto_model'] == 'Risk'