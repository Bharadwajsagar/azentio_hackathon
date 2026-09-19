"""Strict JSON schema extraction/validation for SLM output, plus a rule-based
fallback for when the model's output can't be trusted -- a tiny (<3B) model
asked for raw JSON will sometimes wrap it in prose or drift out of schema,
and an injection that partially succeeds should never produce a silently
wrong verdict.
"""

import json
import re

from models.prediction import PredictionOutput

REQUIRED_KEYS = {"transaction_id", "is_fraud", "confidence", "justification"}

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(raw_text: str) -> tuple[dict | None, str | None]:
    raw_text = _CODE_FENCE_RE.sub("", raw_text.strip())
    match = _JSON_OBJECT_RE.search(raw_text)
    if not match:
        return None, "no_json_object_found"
    try:
        return json.loads(match.group(0)), None
    except json.JSONDecodeError as e:
        return None, f"json_decode_error: {e}"


def validate_schema(obj, expected_transaction_id: str) -> tuple[dict | None, str | None]:
    if not isinstance(obj, dict):
        return None, "not_a_dict"
    missing = REQUIRED_KEYS - set(obj.keys())
    if missing:
        return None, f"missing_keys: {missing}"
    if not isinstance(obj["is_fraud"], bool):
        return None, "is_fraud_not_bool"
    try:
        conf = float(obj["confidence"])
    except (TypeError, ValueError):
        return None, "confidence_not_numeric"
    if not (0.0 <= conf <= 1.0):
        return None, "confidence_out_of_range"
    if not isinstance(obj["justification"], str) or not obj["justification"].strip():
        return None, "justification_invalid"

    clean = {
        # never trust the model to echo the id back correctly
        "transaction_id": expected_transaction_id,
        "is_fraud": bool(obj["is_fraud"]),
        "confidence": round(conf, 4),
        "justification": obj["justification"].strip(),
    }
    return clean, None


def parse_model_output(
    raw_text: str, expected_transaction_id: str
) -> tuple[PredictionOutput | None, str | None]:
    obj, err = extract_json(raw_text)
    if err:
        return None, err
    clean, err = validate_schema(obj, expected_transaction_id)
    if err:
        return None, err
    return PredictionOutput(**clean), None


def fallback_prediction(
    transaction_id: str, weak_risk_score: int, max_score: int = 8
) -> PredictionOutput:
    """Used only when the model's output fails validation (even after retry).
    Deterministic, explainable, and clearly distinguishable in the audit log
    via fallback_used=True -- never silently mixed in as a 'real' model call.
    """
    is_fraud = weak_risk_score >= 3
    confidence = (
        min(0.5 + 0.5 * (weak_risk_score / max_score), 0.99)
        if is_fraud
        else max(0.5 - 0.5 * (weak_risk_score / max_score), 0.01)
    )
    return PredictionOutput(
        transaction_id=transaction_id,
        is_fraud=is_fraud,
        confidence=round(confidence, 4),
        justification=(
            "Model output failed strict JSON validation after retry; fell back to the "
            "rule-based risk score computed from structured transaction evidence."
        ),
    )
