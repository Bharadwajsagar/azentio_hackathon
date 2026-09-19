import pandas as pd
from services.feature_service import RISK_FEATURE_COLUMNS
from services.labeling_service import generate_pseudo_labels


def _df_with_risk_flags(flags: dict) -> pd.DataFrame:
    row = {c: False for c in RISK_FEATURE_COLUMNS}
    row.update(flags)
    row["transaction_id"] = "T1"
    return pd.DataFrame([row])


def test_below_threshold_is_not_pseudo_fraud():
    df = _df_with_risk_flags({RISK_FEATURE_COLUMNS[0]: True, RISK_FEATURE_COLUMNS[1]: True})
    out = generate_pseudo_labels(df, threshold=3)
    assert out.loc[0, "_weak_risk_score"] == 2
    assert bool(out.loc[0, "_pseudo_is_fraud"]) is False


def test_at_threshold_is_pseudo_fraud():
    flags = {c: True for c in RISK_FEATURE_COLUMNS[:3]}
    df = _df_with_risk_flags(flags)
    out = generate_pseudo_labels(df, threshold=3)
    assert out.loc[0, "_weak_risk_score"] == 3
    assert bool(out.loc[0, "_pseudo_is_fraud"]) is True


def test_columns_are_explicitly_marked_as_weak_supervision():
    # naming convention itself is part of the "don't call these real labels" contract
    df = _df_with_risk_flags({})
    out = generate_pseudo_labels(df)
    assert "_pseudo_is_fraud" in out.columns
    assert "_weak_risk_score" in out.columns
