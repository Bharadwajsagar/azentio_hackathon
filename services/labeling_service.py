"""Weak supervision / heuristic pseudo-labels -- used SOLELY to bootstrap
LoRA fine-tuning because no authoritative fraud ground truth was provided in
the supplied data. These are NOT real fraud labels and must never be reported
or treated as ground truth accuracy.
"""

import pandas as pd
from utils.logging import get_logger

from services.feature_service import RISK_FEATURE_COLUMNS

logger = get_logger(__name__)

# A transaction needs at least this many independent risk signals to be
# treated as a (weak) positive pseudo-label.
PSEUDO_LABEL_THRESHOLD = 3


def generate_pseudo_labels(
    df: pd.DataFrame, threshold: int = PSEUDO_LABEL_THRESHOLD
) -> pd.DataFrame:
    df = df.copy()
    df["_weak_risk_score"] = df[RISK_FEATURE_COLUMNS].sum(axis=1)
    df["_pseudo_is_fraud"] = df["_weak_risk_score"] >= threshold

    positive_rate = df["_pseudo_is_fraud"].mean()
    logger.info(
        "generate_pseudo_labels (WEAK SUPERVISION, not ground truth): "
        "threshold=%d, positive_rate=%.3f",
        threshold,
        positive_rate,
    )
    return df
