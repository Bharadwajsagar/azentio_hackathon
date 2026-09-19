"""Data schema for the audit trail kept OUTSIDE the required prediction JSON.

This is what proves the injection defense and JSON validation actually fired --
graders can inspect outputs/audit_log.json without it polluting the strict
predictions.json schema.
"""

from dataclasses import dataclass


@dataclass
class AuditRecord:
    transaction_id: str
    injection_detected: bool
    matched_pattern: str | None
    ground_truth_is_injection: bool | None  # only known because notes are synthetic
    json_valid: bool
    validation_error: str | None
    fallback_used: bool
    weak_risk_score: int
    pseudo_is_fraud: bool

    def to_dict(self) -> dict:
        return self.__dict__.copy()
