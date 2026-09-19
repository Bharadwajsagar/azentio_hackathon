"""Pre-LLM injection detector + sanitizer.

Design decision (explicit, not incidental): the SLM is never responsible for
detecting the attack. Untrusted note text is regex-scanned and replaced BEFORE
it is placed anywhere near the model's context. This is a defense-in-depth
first layer -- validated recall is ~64% against paraphrased/novel phrasings
(100% against exact known templates), which is why services/prompt_service.py
adds a second layer (structural fencing of untrusted text + strict output
schema validation) rather than relying on regex alone.
"""

import re

import pandas as pd
from models.audit import AuditRecord
from utils.logging import get_logger

logger = get_logger(__name__)

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above)\s+instructions",
    r"ignore\s+the\s+system\s+prompt",
    r"disregard\s+(all\s+)?(prior|previous)\s+instructions",
    r"you\s+are\s+now\s+",
    r"^\s*system\s*:",
    r"^\s*developer\s*:",
    r"^\s*assistant\s*:",
    r"classify\s+this.*as\s+safe",
    r"output\s+is_fraud\s*=\s*false",
    r"do\s+not\s+flag",
    r"new\s+instructions?\s*:",
    r"override\s+(the\s+)?fraud\s+check",
    r"\bnew\s+task\b",
    r'"is_fraud"\s*:\s*false',
]
_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

SUSPICIOUS_NOTE_PLACEHOLDER = "[SUSPICIOUS_NOTE_REMOVED]"


def sanitize_note(note: str) -> tuple[str, bool, str | None]:
    """Returns (sanitized_text, injection_detected, matched_pattern)."""
    if not isinstance(note, str) or note.strip() == "":
        return "", False, None
    for pat in _COMPILED_PATTERNS:
        if pat.search(note):
            return SUSPICIOUS_NOTE_PLACEHOLDER, True, pat.pattern
    return note.strip(), False, None


def sanitize_dataframe(tx: pd.DataFrame, note_col: str = "transaction_notes__SYNTHETIC"):
    tx = tx.copy()
    sanitized, detected, matched = [], [], []
    for note in tx[note_col]:
        s, d, p = sanitize_note(note)
        sanitized.append(s)
        detected.append(d)
        matched.append(p)

    tx["transaction_notes_sanitized"] = sanitized
    tx["injection_detected"] = detected
    tx["_matched_pattern"] = matched

    audit_records = [
        AuditRecord(
            transaction_id=row["transaction_id"],
            injection_detected=row["injection_detected"],
            matched_pattern=row["_matched_pattern"],
            ground_truth_is_injection=row.get("_gt_is_injection_attempt"),
            json_valid=False,  # filled in later by validation_service
            validation_error=None,
            fallback_used=False,
            weak_risk_score=0,  # filled in later by feature/labeling service
            pseudo_is_fraud=False,
        )
        for _, row in tx.iterrows()
    ]

    if "_gt_is_injection_attempt" in tx.columns:
        gt = tx["_gt_is_injection_attempt"]
        det = tx["injection_detected"]
        tp = int(((gt) & (det)).sum())
        fp = int(((~gt) & (det)).sum())
        fn = int(((gt) & (~det)).sum())
        tn = int(((~gt) & (~det)).sum())
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        logger.info(
            "sanitize_dataframe detector eval vs planted ground truth: "
            "TP=%d FP=%d FN=%d TN=%d precision=%.3f recall=%.3f",
            tp,
            fp,
            fn,
            tn,
            precision,
            recall,
        )

    return tx, audit_records
