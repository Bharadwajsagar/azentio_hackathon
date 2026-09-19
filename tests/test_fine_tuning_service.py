import json

from services.feature_service import RISK_FEATURE_COLUMNS
from services.fine_tuning_service import _weak_label_to_completion


def _row(risk_flags: dict, is_fraud: bool, score: int, transaction_id: str = "TXN_1"):
    row = {c: False for c in RISK_FEATURE_COLUMNS}
    row.update(risk_flags)
    row["transaction_id"] = transaction_id
    row["_pseudo_is_fraud"] = is_fraud
    row["_weak_risk_score"] = score
    return row


def test_completion_is_valid_json_with_required_keys():
    row = _row({"is_odd_hour": True, "is_high_amount_ratio": True}, is_fraud=False, score=2)
    completion = json.loads(_weak_label_to_completion(row))
    assert set(completion.keys()) == {"transaction_id", "is_fraud", "confidence", "justification"}


def test_justification_mentions_actual_triggered_reasons():
    row = _row(
        {"is_high_amount_ratio": True, "is_pep": True, "is_odd_hour": True},
        is_fraud=True,
        score=3,
    )
    completion = json.loads(_weak_label_to_completion(row))
    assert "amount far above the account's average" in completion["justification"]
    assert "politically exposed" in completion["justification"]
    # a reason that was NOT triggered should not appear
    assert "new device" not in completion["justification"]


def test_non_fraud_with_no_signals_has_generic_justification():
    row = _row({}, is_fraud=False, score=0)
    completion = json.loads(_weak_label_to_completion(row))
    assert completion["is_fraud"] is False
    assert "No significant risk signals" in completion["justification"]


def test_confidence_always_in_valid_range_across_scores():
    for score in range(0, 9):
        row = _row({}, is_fraud=score >= 3, score=score)
        completion = json.loads(_weak_label_to_completion(row))
        assert 0.0 <= completion["confidence"] <= 1.0
