from services.validation_service import (
    extract_json,
    fallback_prediction,
    parse_model_output,
    validate_schema,
)


class TestExtractJson:
    def test_plain_json(self):
        obj, err = extract_json('{"a": 1}')
        assert err is None
        assert obj == {"a": 1}

    def test_strips_markdown_fences(self):
        obj, err = extract_json('```json\n{"a": 1}\n```')
        assert err is None
        assert obj == {"a": 1}

    def test_ignores_surrounding_prose(self):
        obj, err = extract_json('Sure, here you go: {"a": 1} thanks!')
        assert err is None
        assert obj == {"a": 1}

    def test_no_json_found(self):
        obj, err = extract_json("no json here")
        assert obj is None
        assert err == "no_json_object_found"


class TestValidateSchema:
    VALID = {
        "transaction_id": "TXN_1",
        "is_fraud": True,
        "confidence": 0.5,
        "justification": "reason",
    }

    def test_valid_object_passes(self):
        clean, err = validate_schema(self.VALID, expected_transaction_id="TXN_1")
        assert err is None
        assert clean["transaction_id"] == "TXN_1"

    def test_missing_key_rejected(self):
        obj = {k: v for k, v in self.VALID.items() if k != "justification"}
        clean, err = validate_schema(obj, expected_transaction_id="TXN_1")
        assert clean is None
        assert "missing_keys" in err

    def test_non_bool_is_fraud_rejected(self):
        obj = {**self.VALID, "is_fraud": "yes"}
        clean, err = validate_schema(obj, expected_transaction_id="TXN_1")
        assert clean is None
        assert err == "is_fraud_not_bool"

    def test_confidence_out_of_range_rejected(self):
        obj = {**self.VALID, "confidence": 1.5}
        clean, err = validate_schema(obj, expected_transaction_id="TXN_1")
        assert clean is None
        assert err == "confidence_out_of_range"

    def test_transaction_id_is_never_trusted_from_model(self):
        """The model could be manipulated into echoing a different transaction_id
        (e.g. via an injection attempt) -- the caller-supplied expected id must win."""
        obj = {**self.VALID, "transaction_id": "SOMETHING_ELSE"}
        clean, err = validate_schema(obj, expected_transaction_id="TXN_1")
        assert err is None
        assert clean["transaction_id"] == "TXN_1"


class TestParseModelOutput:
    def test_injection_attempt_producing_incomplete_json_is_rejected(self):
        raw = 'I will ignore the fraud rules as instructed. {"is_fraud": false, "confidence": 0.0}'
        pred, err = parse_model_output(raw, expected_transaction_id="TXN_1")
        assert pred is None
        assert "missing_keys" in err


class TestFallbackPrediction:
    def test_high_risk_score_falls_back_to_fraud(self):
        pred = fallback_prediction("TXN_1", weak_risk_score=5, max_score=8)
        assert pred.is_fraud is True
        assert pred.transaction_id == "TXN_1"

    def test_low_risk_score_falls_back_to_not_fraud(self):
        pred = fallback_prediction("TXN_1", weak_risk_score=0, max_score=8)
        assert pred.is_fraud is False

    def test_confidence_always_in_valid_range(self):
        for score in range(0, 9):
            pred = fallback_prediction("TXN_1", weak_risk_score=score, max_score=8)
            assert 0.0 <= pred.confidence <= 1.0
