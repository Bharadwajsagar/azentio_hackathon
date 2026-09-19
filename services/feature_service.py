"""Structured, deterministic fraud-signal features computed from cleaned data.

These are the "evidence" fields handed to the SLM -- the model classifies
based on this structured evidence plus the sanitized note, rather than being
asked to reason over a raw merged row from scratch.
"""

import pandas as pd
from utils.logging import get_logger

logger = get_logger(__name__)

RISK_FEATURE_COLUMNS = [
    "is_high_amount_ratio",
    "is_odd_hour",
    "is_high_velocity",
    "is_far_from_home",
    "is_new_device_flag",
    "is_high_risk_customer",
    "is_pep",
    "account_reference_broken",
]


def compute_fraud_features(merged: pd.DataFrame) -> pd.DataFrame:
    df = merged.copy()

    df["is_high_amount_ratio"] = df["amount_to_account_avg_ratio"] > 5
    df["is_odd_hour"] = df["transaction_hour"].apply(
        lambda h: (h <= 5 or h >= 23) if pd.notna(h) else False
    )
    df["is_high_velocity"] = df["txn_count_last_24h"].fillna(0) >= 4
    df["is_far_from_home"] = df["distance_from_home_km"].fillna(0) > 500
    df["is_new_device_flag"] = (
        df["is_new_device"].fillna(0).astype(str).isin(["1", "1.0", "True", "true"])
    )
    df["is_high_risk_customer"] = df.get("risk_rating", pd.Series(index=df.index)).eq("HIGH")
    df["is_pep"] = (
        df.get("is_politically_exposed", pd.Series(index=df.index))
        .fillna(0)
        .astype(str)
        .isin(["1", "1.0"])
    )
    # account_reference_broken is already set by merge_service; keep as-is if present
    if "account_reference_broken" not in df.columns:
        df["account_reference_broken"] = False

    for c in RISK_FEATURE_COLUMNS:
        df[c] = df[c].fillna(False).astype(bool)

    logger.info(
        "compute_fraud_features: added %d structured risk features", len(RISK_FEATURE_COLUMNS)
    )
    return df
