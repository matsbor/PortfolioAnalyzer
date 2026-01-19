import pandas as pd

# IMPORTANT:
# We import functions from your Streamlit file.
# This only works if your module import does NOT immediately run Streamlit UI at import-time.
# If it does, we’ll refactor in Step 2 below.

from alpha_miner_institutional_v2 import (
    validate_data_invariants,
    enforce_strict_mode,
    get_risk_profile_preset,
)

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

    # Some versions name this param "risk_profile" or expect the preset
    try:
        df2 = enforce_strict_mode(df.copy(), validation, preset, "Defensive")
    except TypeError:
        df2 = enforce_strict_mode(df.copy(), validation, preset)

    assert len(df2) == len(df)