from models.audit import AuditRecord
from models.prediction import PredictionOutput
from models.transaction import TransactionRecord


def test_prediction_output_to_dict_has_exactly_required_keys():
    pred = PredictionOutput(
        transaction_id="TXN_1", is_fraud=True, confidence=0.9123456, justification="reason"
    )
    d = pred.to_dict()
    assert set(d.keys()) == {"transaction_id", "is_fraud", "confidence", "justification"}
    assert d["confidence"] == 0.9123  # rounded
    assert isinstance(d["is_fraud"], bool)


def test_audit_record_to_dict_roundtrip():
    rec = AuditRecord(
        transaction_id="TXN_1",
        injection_detected=True,
        matched_pattern="ignore.*instructions",
        ground_truth_is_injection=True,
        json_valid=True,
        validation_error=None,
        fallback_used=False,
        weak_risk_score=4,
        pseudo_is_fraud=True,
    )
    d = rec.to_dict()
    assert d["transaction_id"] == "TXN_1"
    assert d["weak_risk_score"] == 4


def test_transaction_record_from_row(merged_df):
    row = merged_df.iloc[0]
    # from_row expects the post-feature-engineering columns; fill in the minimal set here
    row = row.copy()
    for col, default in [
        ("is_new_device_flag", False),
        ("is_odd_hour", False),
        ("is_high_amount_ratio", False),
        ("is_high_velocity", False),
        ("is_far_from_home", False),
        ("is_high_risk_customer", False),
        ("is_pep", False),
        ("_weak_risk_score", 0),
        ("transaction_notes_sanitized", ""),
        ("injection_detected", False),
    ]:
        if col not in row or row[col] is None:
            row[col] = default

    record = TransactionRecord.from_row(row)
    assert record.transaction_id == row["transaction_id"]
    assert isinstance(record.account_reference_broken, bool)
