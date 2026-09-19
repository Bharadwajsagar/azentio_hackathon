"""Orchestrates prompt building -> batched SLM inference -> validation, and
the honest baseline-vs-fine-tuned comparison (never against real fraud
ground truth, since none was supplied -- only against the heuristic
pseudo-labels used for weak supervision, clearly labeled as such)."""

from models.audit import AuditRecord
from models.transaction import TransactionRecord
from services.inference_service import run_inference
from utils.logging import get_logger

logger = get_logger(__name__)


def build_records(df):
    return [TransactionRecord.from_row(row) for _, row in df.iterrows()]


def run_predictions(model, tokenizer, df, batch_size=16):
    records = build_records(df)
    predictions, audit_meta = run_inference(model, tokenizer, records, batch_size=batch_size)
    return predictions, audit_meta


def merge_audit(base_audit_records: list, inference_audit_meta: list, df):
    """Combine the sanitization-stage audit (injection detection) with the
    inference-stage audit (JSON validity / fallback) into one AuditRecord per
    transaction, keyed by row order (both lists are built from the same df)."""
    merged = []
    for base, inf_meta, (_, row) in zip(base_audit_records, inference_audit_meta, df.iterrows()):
        merged.append(
            AuditRecord(
                transaction_id=base.transaction_id,
                injection_detected=base.injection_detected,
                matched_pattern=base.matched_pattern,
                ground_truth_is_injection=base.ground_truth_is_injection,
                json_valid=inf_meta["json_valid"],
                validation_error=inf_meta["validation_error"],
                fallback_used=inf_meta["fallback_used"],
                weak_risk_score=int(row["_weak_risk_score"]),
                pseudo_is_fraud=bool(row["_pseudo_is_fraud"]),
            )
        )
    return merged


def compare_against_pseudo_labels(predictions, df, label: str):
    """Agreement rate between model predictions and the WEAK heuristic
    pseudo-labels. This is NOT accuracy against real fraud -- it only tells us
    whether the model's judgment aligns with the rule-based heuristic it was
    (for the fine-tuned case) trained to approximate."""
    pseudo = df["_pseudo_is_fraud"].tolist()
    agree = sum(1 for p, gt in zip(predictions, pseudo) if p.is_fraud == gt)
    rate = agree / len(predictions) if predictions else float("nan")
    stats = {
        "label": label,
        "n": len(predictions),
        "agreement_with_weak_pseudo_labels": rate,
        "predicted_positive_rate": sum(1 for p in predictions if p.is_fraud) / len(predictions),
    }
    logger.info("compare_against_pseudo_labels[%s]: %s", label, stats)
    return stats
