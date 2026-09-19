"""Orchestrates: load -> clean -> merge -> synthesize notes -> sanitize ->
features -> weak labels, and assembles the data-quality report."""

from services.cleaning_service import clean_accounts, clean_customers, clean_transactions
from services.data_loader import load_all
from services.feature_service import compute_fraud_features
from services.labeling_service import generate_pseudo_labels
from services.merge_service import merge_transactions
from services.notes_service import synthesize_transaction_notes
from services.sanitization_service import sanitize_dataframe
from utils.logging import get_logger

logger = get_logger(__name__)


def run_data_pipeline():
    tx, acc, cust = load_all()

    tx_clean = clean_transactions(tx)
    acc_clean = clean_accounts(acc)
    cust_clean = clean_customers(cust)

    merged, merge_stats = merge_transactions(tx_clean, acc_clean, cust_clean)
    merged = synthesize_transaction_notes(merged)
    merged, audit_records = sanitize_dataframe(merged)
    merged = compute_fraud_features(merged)
    merged = generate_pseudo_labels(merged)

    detector_stats = _evaluate_detector(merged)

    quality_report = {
        "rows_loaded": {"transactions": len(tx), "accounts": len(acc), "customers": len(cust)},
        "amount_parse_failures": int(merged["amount_clean"].isna().sum()),
        "timestamp_suspect_count": int(merged["is_timestamp_suspect"].sum()),
        "merge": merge_stats,
        "synthetic_notes_injection_rate": float(merged["_gt_is_injection_attempt"].mean()),
        "injection_detector": detector_stats,
        "weak_pseudo_label_positive_rate": float(merged["_pseudo_is_fraud"].mean()),
    }
    logger.info("run_data_pipeline: quality_report=%s", quality_report)
    return merged, audit_records, quality_report


def _evaluate_detector(df):
    gt = df["_gt_is_injection_attempt"]
    det = df["injection_detected"]
    tp = int(((gt) & (det)).sum())
    fp = int(((~gt) & (det)).sum())
    fn = int(((gt) & (~det)).sum())
    tn = int(((~gt) & (~det)).sum())
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall}
