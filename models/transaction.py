"""Data schema for one fully cleaned/merged/featured transaction record.

NOTE: this is a data schema, not the SLM. The tiny language model used for
inference lives in services/inference_service.py and is downloaded from
Hugging Face at runtime (Qwen2.5-1.5B-Instruct) -- it is not stored here.
"""

from dataclasses import dataclass


@dataclass
class TransactionRecord:
    transaction_id: str
    amount: float
    currency: str
    transaction_type: str | None
    channel: str | None
    transaction_hour: float | None
    merchant_category: str | None
    amount_to_account_avg_ratio: float | None
    distance_from_home_km: float | None
    txn_count_last_24h: float | None
    is_new_device_flag: bool
    is_odd_hour: bool
    is_high_amount_ratio: bool
    is_high_velocity: bool
    is_far_from_home: bool
    is_high_risk_customer: bool
    is_pep: bool
    account_reference_broken: bool
    weak_risk_score: int
    sanitized_note: str
    injection_detected: bool

    @classmethod
    def from_row(cls, row) -> "TransactionRecord":
        return cls(
            transaction_id=row["transaction_id"],
            amount=row["amount_clean"],
            currency=row.get("currency", "INR"),
            transaction_type=row.get("transaction_type"),
            channel=row.get("channel"),
            transaction_hour=row.get("transaction_hour"),
            merchant_category=row.get("merchant_category"),
            amount_to_account_avg_ratio=row.get("amount_to_account_avg_ratio"),
            distance_from_home_km=row.get("distance_from_home_km"),
            txn_count_last_24h=row.get("txn_count_last_24h"),
            is_new_device_flag=bool(row.get("is_new_device_flag", False)),
            is_odd_hour=bool(row.get("is_odd_hour", False)),
            is_high_amount_ratio=bool(row.get("is_high_amount_ratio", False)),
            is_high_velocity=bool(row.get("is_high_velocity", False)),
            is_far_from_home=bool(row.get("is_far_from_home", False)),
            is_high_risk_customer=bool(row.get("is_high_risk_customer", False)),
            is_pep=bool(row.get("is_pep", False)),
            account_reference_broken=bool(row.get("account_reference_broken", False)),
            weak_risk_score=int(row.get("_weak_risk_score", 0)),
            sanitized_note=row.get("transaction_notes_sanitized", ""),
            injection_detected=bool(row.get("injection_detected", False)),
        )
