from services.feature_service import RISK_FEATURE_COLUMNS, compute_fraud_features


def test_all_risk_columns_present_and_boolean(merged_df):
    out = compute_fraud_features(merged_df)
    for col in RISK_FEATURE_COLUMNS:
        assert col in out.columns
        assert out[col].dtype == bool


def test_account_reference_broken_preserved_from_merge(merged_df):
    out = compute_fraud_features(merged_df)
    row_d = out.set_index("transaction_id").loc["TXN_D"]
    assert bool(row_d["account_reference_broken"]) is True


def test_high_risk_customer_flag_from_customer_data(merged_df):
    out = compute_fraud_features(merged_df)
    by_id = out.set_index("transaction_id")
    # TXN_C/TXN_D resolve to CUST_2 (risk_rating=HIGH) via ACC_2; TXN_A/TXN_B -> CUST_1 (LOW)
    assert bool(by_id.loc["TXN_A", "is_high_risk_customer"]) is False
    assert bool(by_id.loc["TXN_C", "is_high_risk_customer"]) is True


def test_odd_hour_flag(merged_df):
    out = compute_fraud_features(merged_df)
    by_id = out.set_index("transaction_id")
    assert bool(by_id.loc["TXN_C", "is_odd_hour"]) is True  # hour=3
    assert bool(by_id.loc["TXN_A", "is_odd_hour"]) is False  # hour=10
